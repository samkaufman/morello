use crate::datadeps::SpecKey;
use crate::db::{DbKey, TableKey};
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::rtree::RTreeDyn;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::utils::diagonals_shifted;
use itertools::{izip, Itertools};
use once_cell::unsync::OnceCell;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::iter;
use std::rc::Rc;

/// Wraps an [RTreeDyn] to restruct mutations to those which are relatively independent of the
/// database's underlying space.
///
/// **Note:** This API is unstable and experimental.
#[derive(Clone)]
pub struct SpecGeometry<Tgt: Target>(
    pub(crate) HashMap<TableKey, RTreeDyn<()>>,
    Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
);

// TODO: Can we take more of these fields by reference? This requires a lot of cloning.
#[derive(Clone)]
pub struct SpecGeometryRect<Tgt: Target> {
    key: TableKey,
    bottom: Vec<BimapInt>,
    top: Vec<BimapInt>,
    cached_specs: OnceCell<(Spec<Tgt>, Spec<Tgt>)>,
    bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
}

impl<Tgt: Target> SpecGeometry<Tgt> {
    pub fn new(bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>) -> Self {
        Self(HashMap::new(), bimap)
    }

    pub fn single(
        spec: &Spec<Tgt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        let mut sg = Self::new(bimap);
        sg.insert_spec(spec);
        sg
    }

    pub fn bimap(&self) -> &Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>> {
        &self.1
    }

    pub fn insert_spec(&mut self, spec: &Spec<Tgt>) {
        debug_assert!(spec.is_canonical());
        let (key, pt) = self.1.apply(spec);
        let pt_i64 = pt.iter().map(|v| BimapSInt::from(*v)).collect::<Vec<_>>();
        self.0
            .entry(key)
            .or_insert_with(|| RTreeDyn::empty(pt.len()))
            .merge_insert(&pt_i64, &pt_i64, (), true);
    }

    pub fn insert_rect(&mut self, rect: SpecGeometryRect<Tgt>) {
        // TODO: Assert BiMaps are the same.
        let SpecGeometryRect {
            key,
            bottom,
            top,
            cached_specs: _,
            bimap: _,
        } = rect;
        let bottom_i64 = bottom.iter().map(|&v| v.into()).collect::<Vec<_>>();
        let top_i64 = top.iter().map(|&v| v.into()).collect::<Vec<_>>();
        self.0
            .entry(key)
            .or_insert_with(|| RTreeDyn::empty(bottom.len()))
            .merge_insert(&bottom_i64, &top_i64, (), true);
    }

    pub fn extend(&mut self, source: impl Iterator<Item = SpecGeometryRect<Tgt>>) {
        source.for_each(|rect| {
            self.insert_rect(rect);
        });
    }

    pub fn accums(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.iter().flat_map(|rect| rect.accums())
    }

    /// Yields `Fill` Specs for the outputs of every Spec in the [SpecGeometry].
    pub fn outputs_fills(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.iter().flat_map(|rect| rect.outputs_fills())
    }

    pub fn is_empty(&self) -> bool {
        self.0.values().all(|tree| tree.is_empty())
    }

    /// Iterates over rectangles' bottom and top [Spec]s.
    pub fn iter(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.0.iter().flat_map(move |(key, rtree)| {
            rtree.iter().map(move |rect| SpecGeometryRect {
                key: key.clone(),
                bottom: rect.0.iter().map(|v| (*v).try_into().unwrap()).collect(),
                top: rect.1.iter().map(|v| (*v).try_into().unwrap()).collect(),
                cached_specs: OnceCell::new(),
                bimap: Rc::clone(&self.1),
            })
        })
    }
}

impl<Tgt: Target> From<SpecGeometryRect<Tgt>> for SpecGeometry<Tgt> {
    fn from(rect: SpecGeometryRect<Tgt>) -> Self {
        let SpecGeometryRect {
            key,
            bottom,
            top,
            cached_specs: _,
            bimap,
        } = rect;
        let bottom_i64 = bottom.iter().map(|v| (*v).into()).collect::<Vec<_>>();
        let top_i64 = top.iter().map(|v| (*v).into()).collect::<Vec<_>>();
        let mut rtree = RTreeDyn::empty(bottom.len());
        rtree.insert(&bottom_i64, &top_i64, ());
        SpecGeometry(std::iter::once((key, rtree)).collect(), bimap)
    }
}

impl<Tgt: Target> Debug for SpecGeometry<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpecGeometry")
            .field(&self.0)
            .finish_non_exhaustive()
    }
}

impl<Tgt: Target> SpecGeometryRect<Tgt> {
    pub(crate) fn new(
        key: TableKey,
        bottom: Vec<BimapInt>,
        top: Vec<BimapInt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        assert_eq!(bottom.len(), top.len());
        assert!(bottom.iter().zip(&top).all(|(b, t)| b <= t));
        #[cfg(debug_assertions)]
        {
            let mut key = (key.clone(), bottom.clone());
            for v in bottom
                .iter()
                .zip(&top)
                .map(|(l, h)| (*l..=*h))
                .multi_cartesian_product()
            {
                key.1 = v;
                let spec = bimap.apply_inverse(&key);
                if !spec.is_canonical() {
                    panic!("Rect contains non-canonical Spec: {spec}");
                }
            }
        }
        Self {
            key,
            bottom,
            top,
            cached_specs: OnceCell::new(),
            bimap,
        }
    }

    /// Creates a new [SpecGeometryRect] which contains a single [Spec].
    ///
    /// The caller must ensure that the [Spec] is canonical. Behavior is undefined otherwise.
    pub(crate) fn single(
        spec: &Spec<Tgt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        debug_assert!(spec.is_canonical());
        let (key, pt) = bimap.apply(spec);
        Self {
            key,
            bottom: pt.clone(),
            top: pt,
            cached_specs: OnceCell::with_value((spec.clone(), spec.clone())),
            bimap,
        }
    }

    // TODO: Remove. Only exists for wrapping RTree rect in [synthesize_block].
    pub(crate) fn bimap(&self) -> Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>> {
        Rc::clone(&self.bimap)
    }

    pub fn table_key(&self) -> &TableKey {
        &self.key
    }

    pub fn spec_key(&self) -> &SpecKey {
        &self.key.0
    }

    pub fn bottom(&self) -> &Spec<Tgt> {
        &self.specs().0
    }

    pub fn top(&self) -> &Spec<Tgt> {
        &self.specs().1
    }

    pub fn bottom_point(&self) -> &[BimapInt] {
        &self.bottom
    }

    pub fn top_point(&self) -> &[BimapInt] {
        &self.top
    }

    pub(crate) fn intersects(&self, other: &SpecGeometryRect<Tgt>) -> bool {
        for (self_bottom, self_top, other_bottom, other_top) in
            izip!(&self.bottom, &self.top, &other.bottom, &other.top)
        {
            if self_top < other_bottom || self_bottom > other_top {
                return false;
            }
        }
        true
    }

    pub fn accums(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> {
        let key @ (spec_key, _) = self.table_key();

        let mut result: Option<SpecGeometryRect<Tgt>> = None;
        // If it has an accumulating variant, it's in the first dimension.
        // TODO: Abstract this `match` away somehow.
        let has_accum = match spec_key {
            SpecKey::Matmul { .. }
            | SpecKey::Conv { .. }
            | SpecKey::SoftmaxDenominatorAndMax { .. }
            | SpecKey::Max { .. }
            | SpecKey::SoftmaxDenominator { .. }
            | SpecKey::SoftmaxDenominatorAndUnscaled { .. }
            | SpecKey::SoftmaxDenominatorAndUnscaledFromMax { .. } => true,
            SpecKey::Softmax { .. }
            | SpecKey::SoftmaxComplete { .. }
            | SpecKey::Move { .. }
            | SpecKey::Fill { .. }
            | SpecKey::OnePrefix { .. }
            | SpecKey::Broadcast { .. }
            | SpecKey::DivideVec { .. }
            | SpecKey::DivideVecScalar { .. }
            | SpecKey::Compose { .. } => false,
        };
        if has_accum && self.bottom_point()[0] > 0 {
            let mut bottom = self.bottom_point().to_vec();
            let mut top = self.top_point().to_vec();
            bottom[0] = 0;
            top[0] = 0;
            result = Some(SpecGeometryRect::new(
                key.clone(),
                bottom,
                top,
                self.bimap(),
            ));
        }
        result.into_iter()
    }

    /// Yields `Fill` Specs for the outputs of every Spec in the [SpecGeometryRect].
    pub fn outputs_fills(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> {
        let bimap = self.bimap();

        let top_output_idx = self.top().0.unique_output_index().unwrap();
        let bottom_output_idx = self.bottom().0.unique_output_index().unwrap();
        let TensorSpec {
            shape: top_output_shape,
            dtype: top_output_dtype,
            aux: top_output_aux,
        } = self.top().0.parameter(top_output_idx);
        let TensorSpec {
            shape: bottom_output_shape,
            dtype: bottom_output_dtype,
            aux: bottom_output_aux,
        } = self.bottom().0.parameter(bottom_output_idx);

        let fill_value = self
            .top()
            .0
            .initial_accumulating_value_for_output(top_output_idx)
            .expect("rect should have Spec kinds supporting accumulating");
        debug_assert_eq!(
            fill_value,
            self.bottom()
                .0
                .initial_accumulating_value_for_output(bottom_output_idx)
                .unwrap()
        );

        let mut fill_top = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill { value: fill_value },
                    spec_shape: top_output_shape,
                    dtypes: vec![top_output_dtype],
                },
                vec![top_output_aux],
                self.top().0.serial_only(),
            ),
            self.top().1.clone(),
        );
        let mut fill_bottom = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill { value: fill_value },
                    spec_shape: bottom_output_shape,
                    dtypes: vec![bottom_output_dtype],
                },
                vec![bottom_output_aux],
                self.bottom().0.serial_only(),
            ),
            self.bottom().1.clone(),
        );
        fill_top.canonicalize().unwrap();
        fill_bottom.canonicalize().unwrap();

        let (fill_key, fill_top_pt) = bimap.apply(&fill_top);
        let (fill_bottom_key, fill_bottom_pt) = bimap.apply(&fill_bottom);
        debug_assert_eq!(fill_key, fill_bottom_key);

        iter::once(SpecGeometryRect::<Tgt>::new(
            fill_key,
            fill_bottom_pt,
            fill_top_pt,
            bimap,
        ))
    }

    /// Returns an iterator over the [Spec]s in this rectangle.
    pub fn iter_specs(&self) -> impl Iterator<Item = Spec<Tgt>> + '_ {
        let low_pt = self.bottom_point();
        let high_pt = self.top_point();
        diagonals_shifted(low_pt, high_pt).flatten().map(|pt| {
            let composed_key = (self.key.clone(), pt.to_vec());
            let mut spec = self.bimap.apply_inverse(&composed_key);
            spec.canonicalize().unwrap();
            spec
        })
    }

    /// Returns the bottom and top [Spec]s for [bottom_point] and [top_point].
    ///
    /// These are lazily cached.
    fn specs(&self) -> &(Spec<Tgt>, Spec<Tgt>) {
        self.cached_specs
            .get_or_try_init(|| -> Result<_, ()> {
                let mut projection = (
                    self.key.clone(),
                    self.bottom_point()
                        .iter()
                        .map(|v| BimapInt::try_from(*v).unwrap())
                        .collect::<Vec<_>>(),
                );
                let bottom = self.bimap.apply_inverse(&projection);
                projection.1 = self
                    .top_point()
                    .iter()
                    .map(|v| BimapInt::try_from(*v).unwrap())
                    .collect::<Vec<_>>();
                let top = self.bimap.apply_inverse(&projection);
                Ok((bottom, top))
            })
            .unwrap()
    }
}

impl<Tgt: Target> fmt::Debug for SpecGeometryRect<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpecGeometryRect")
            .field("key", &self.key)
            .field("bottom", &self.bottom)
            .field("top", &self.top)
            .field("cached_specs", &self.cached_specs)
            .finish_non_exhaustive()
    }
}
