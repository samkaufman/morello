use itertools::iproduct;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::fmt::Display;
use std::num::NonZeroU32;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::{row_major, Layout};
use crate::target::{MemoryLevel, Target};
use crate::utils::join_into_string;

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct TensorSpec<Tgt: Target> {
    pub shape: Shape, // TODO: Rename to shape
    pub dtype: Dtype,
    pub aux: TensorSpecAux<Tgt>,
}

// TODO: This probably shouldn't be public.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TensorSpecAux<Tgt: Target> {
    pub contig: Contig,
    pub aligned: bool,
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_size: Option<DimSize>,
}

pub struct TensorSpecAuxSurMap<Tgt: Target> {
    tensor_shape: SmallVec<[DimSize; 3]>, // TODO: Make into &'a [DimSize]
    tensor_dtype: Dtype,
    phantom: std::marker::PhantomData<Tgt>,
}

#[derive(Default)]
pub struct TensorSpecAuxNonDepBimap<Tgt: Target> {
    phantom: std::marker::PhantomData<Tgt>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct TensorSpecArbMaxShape(pub Shape);

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new_canon(
        shape: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> Self {
        let mut r = Self::new_noncanon(
            shape,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_size,
        );
        r.canonicalize();
        r
    }

    pub fn new_noncanon(
        shape: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> Self {
        Self::new_noncanon_with_aux(
            shape,
            dtype,
            TensorSpecAux {
                contig: contiguous_abs,
                aligned,
                level,
                layout,
                vector_size,
            },
        )
    }

    pub fn new_noncanon_with_aux(shape: Shape, dtype: Dtype, aux: TensorSpecAux<Tgt>) -> Self {
        if shape.is_empty() || shape.iter().any(|&d| d < 1) {
            panic!("Invalid shape: {:?}", shape);
        }

        if aux.vector_size.is_some() != aux.level.vector_rf() {
            panic!(
                "vector_size must be specified if and only if the bank ({:?}) is a vector register file", aux.level
            )
        }

        TensorSpec { shape, dtype, aux }
    }

    pub fn layout(&self) -> Layout {
        self.aux.layout.clone()
    }

    pub fn set_layout(&mut self, new_layout: Layout) {
        self.aux.layout = new_layout;
    }

    pub fn set_contiguous_abs(&mut self, contiguous_abs: Contig) {
        self.aux.contig = contiguous_abs;
    }

    pub fn is_contiguous(&self) -> bool {
        self.aux.contig == self.aux.layout.contiguous_full()
    }

    /// Returns true if this TensorSpec can be tiled to the given shape.
    pub fn is_valid_tile_shape(&self, shape: &[DimSize]) -> bool {
        debug_assert_eq!(shape.len(), self.shape.len());
        debug_assert!(shape.iter().zip(self.shape.iter()).all(|(i, o)| i <= o));
        let all_ones = shape.iter().all(|d| *d == 1);
        all_ones || self.aux.layout.applies_to_shape(shape)
    }

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size()) * self.shape.iter().copied().map(u64::from).product::<u64>()
    }

    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn contiguous_abs(&self) -> Contig {
        self.aux.contig
    }

    pub fn aligned(&self) -> bool {
        self.aux.aligned
    }

    pub fn level(&self) -> <Tgt as Target>::Level {
        self.aux.level
    }

    pub fn vector_size(&self) -> Option<DimSize> {
        self.aux.vector_size
    }

    pub fn set_level(&mut self, level: Tgt::Level, vector_size: Option<DimSize>) {
        assert_eq!(
            level.vector_rf(),
            vector_size.is_some(),
            "Cannot set level to {:?} with vector shape {:?}",
            level,
            vector_size
        );
        self.aux.level = level;
        self.aux.vector_size = vector_size;
    }

    /// Returns a new TensorSpec with the given shape and alignment.
    ///
    /// The result's layout and contiguousness abstraction will have been
    /// canoncialized for the given shape.
    pub fn shrink(&mut self, shape: &[DimSize], aligned: bool) {
        // This implementation is similar to `canonicalize`, but the tile
        // contiguousness is computed from both old and new shapes.
        self.aux.contig = self
            .layout()
            .tile_contiguity(shape, &self.shape, self.aux.contig);
        self.shape = Shape::from(shape);
        self.aux.layout = self.aux.layout.canonicalize_for_shape(&self.shape);
        self.aux.aligned = aligned;
    }

    pub fn canonicalize(&mut self) {
        self.aux.canonicalize(&self.shape, self.aux.aligned);
    }

    /// Returns a TensorSpec with given size-one dimensions dropped.
    ///
    /// The given dimension indices must be sorted in ascending order.
    ///
    /// The result will be canoncialized. If any given dimension index is not
    /// size one, this method panics.
    pub fn squeeze_dims(&self, dropped_dims: &[u8]) -> TensorSpec<Tgt> {
        // Make sure dropped_dims is sorted.
        debug_assert!(dropped_dims.windows(2).all(|w| w[0] < w[1]));

        let mut new_shape = Shape::from(self.shape());
        for &dim in dropped_dims.iter().rev() {
            assert_eq!(
                new_shape[usize::from(dim)],
                1,
                "Cannot drop non-degenerate dimension {} of shape {:?}",
                dim,
                new_shape
            );
            new_shape.remove(dim.into());
        }

        let (new_layout, new_contig) = if new_shape.iter().all(|&d| d == 1) {
            let new_layout = row_major(new_shape.len().try_into().unwrap());
            (new_layout.clone(), new_layout.contiguous_full())
        } else {
            let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
            self.layout()
                .dim_drop(&dropped_dims_set, self.contiguous_abs())
        };

        TensorSpec::new_canon(
            new_shape,
            self.dtype(),
            new_contig,
            self.aligned(),
            self.level(),
            new_layout,
            self.vector_size(),
        )
    }

    // TODO: Shouldn't need this method. Should be implicit in Spec validity.
    pub fn can_move_to(&self, dest_layout: &Layout, dest_level: &Tgt::Level) -> bool {
        // If the destination is in VRF, then the operand volume must be a multiple of at least one
        // of the vector sizes.
        let vector_bytes = dest_level.vector_bytes();
        if !vector_bytes.is_empty() {
            let vol: DimSize = self.shape().iter().product();
            let bytes = vol * DimSize::from(self.dtype.size());
            if vector_bytes.iter().all(|&vb| bytes % vb != 0) {
                return false;
            }
        }
        true
    }
}

impl<Tgt: Target> Display for TensorSpec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims_part = self
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let aux_part = tensorspec_aux_str(&self.aux);

        write!(f, "({}, {}", dims_part, self.dtype)?;
        if aux_part.is_empty() {
            write!(f, ")")
        } else {
            write!(f, ", {})", aux_part)
        }
    }
}

/// Implements [`proptest::arbitrary::Arbitrary`] to yield valid [TensorSpec]s.
///
/// This generates [TensorSpec]s of varying shapes, dtypes, levels and, alignments.
/// The [Layout], vector shape, and contiguousness abstraction are constrained to be valid together
/// and for the generated shape.
///
/// The maximum shape and rank of the [TensorSpec] can be controlled by providing a
/// [TensorSpecArbMaxShape].
#[cfg(test)]
impl<Tgt: Target> proptest::arbitrary::Arbitrary for TensorSpec<Tgt> {
    type Parameters = TensorSpecArbMaxShape;
    type Strategy = proptest::strategy::BoxedStrategy<TensorSpec<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        args.0
            .into_iter()
            .map(|m| 1..=m)
            .collect::<Vec<_>>()
            .prop_flat_map(|shp| {
                let shp1 = TensorSpecArbMaxShape(Shape::from(shp));
                let shp2 = shp1.clone();
                let aux_strategy = TensorSpecAux::arbitrary_with(shp1);
                let dtype_strategy = any::<Dtype>();
                (Just(shp2), dtype_strategy, aux_strategy)
            })
            .prop_map(|(shp, dtype, aux)| {
                let mut tensor_spec = TensorSpec::new_noncanon_with_aux(shp.0, dtype, aux);
                tensor_spec.canonicalize();
                tensor_spec
            })
            .boxed()
    }
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub fn canonicalize(&mut self, shape: &Shape, aligned: bool) {
        self.contig = self.layout.tile_contiguity(shape, shape, self.contig);
        self.layout = self.layout.canonicalize_for_shape(shape);
        self.aligned = aligned;
    }
}

impl<Tgt: Target> Display for TensorSpecAux<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aux_part = tensorspec_aux_str(self);
        if aux_part.is_empty() {
            write!(f, "(_)")
        } else {
            write!(f, "({})", aux_part)
        }
    }
}

#[cfg(test)]
impl<Tgt: Target> proptest::arbitrary::Arbitrary for TensorSpecAux<Tgt> {
    type Parameters = TensorSpecArbMaxShape;
    type Strategy = proptest::strategy::BoxedStrategy<TensorSpecAux<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let shape = args.0;
        any::<Dtype>()
            .prop_flat_map(move |d| arb_tensorspecaux(&shape, d))
            .boxed()
    }
}

#[cfg(test)]
impl Default for TensorSpecArbMaxShape {
    fn default() -> Self {
        use smallvec::smallvec;
        Self(smallvec![8, 8])
    }
}

impl<Tgt: Target> TensorSpecAuxSurMap<Tgt> {
    pub fn new(tensor_shape: &[DimSize], tensor_dtype: Dtype) -> Self {
        debug_assert!(!tensor_shape.is_empty());
        Self {
            tensor_shape: tensor_shape.into(),
            tensor_dtype,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<Tgt> SurMap for TensorSpecAuxSurMap<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((), [BimapInt; 1]);
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + 'static>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        ((), [BiMap::apply(&Tgt::Level::bimap(), &t.level).into()])
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let ((), [level_int]) = i;
        let level = BiMap::apply_inverse(&Tgt::Level::bimap(), &(*level_int).try_into().unwrap());
        let dtype_bytes = u32::from(self.tensor_dtype.size());
        let mut vector_options = level
            .vector_bytes()
            .iter()
            .map(|&vb| Some(vb / dtype_bytes))
            .collect::<Vec<_>>();
        if vector_options.is_empty() {
            vector_options.push(None);
        }
        Box::new(
            iproduct!(
                [true, false],
                Tgt::all_layouts_for_shape(&self.tensor_shape),
                vector_options
            )
            .flat_map(move |(aligned, layout, vector_size)| {
                layout
                    .all_contiguous_abs()
                    .map(move |contig| TensorSpecAux {
                        contig,
                        aligned,
                        layout: layout.clone(),
                        vector_size,
                        level,
                    })
            }),
        )
    }
}

impl<Tgt> BiMap for TensorSpecAuxNonDepBimap<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((Layout, u8, Option<NonZeroU32>), [BimapInt; 2]);

    fn apply(&self, aux: &TensorSpecAux<Tgt>) -> Self::Codomain {
        (
            (
                aux.layout.clone(),
                BiMap::apply(&Tgt::Level::bimap(), &aux.level),
                aux.vector_size.map(|v| NonZeroU32::try_from(v).unwrap()),
            ),
            [aux.contig.into(), aux.aligned as _],
        )
    }

    fn apply_inverse(&self, v: &Self::Codomain) -> Self::Domain {
        let ((layout, level_val, vector_size), [contig, aligned_val]) = v;

        // `unwrap_or_else` rather than `unwrap` to avoid needing a Debug bound
        let level = BiMap::apply_inverse(&Tgt::Level::bimap(), level_val);
        TensorSpecAux {
            layout: layout.clone(),
            contig: (*contig).try_into().unwrap(),
            aligned: *aligned_val != 0,
            level,
            vector_size: vector_size.map(|v| v.try_into().unwrap()),
        }
    }
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(5);

    if aux.level != Tgt::default_level() {
        parts.push(aux.level.to_string());
    }

    if !aux.layout.is_row_major() {
        parts.push(aux.layout.to_string());
    }

    if aux.contig != aux.layout.contiguous_full() {
        parts.push(format!("c{}", aux.contig));
    }

    if !aux.aligned {
        parts.push(String::from("ua"));
    }

    if let Some(vector_size) = aux.vector_size {
        parts.push(vector_size.to_string());
    }

    join_into_string(&parts, ", ")
}

#[cfg(test)]
fn arb_tensorspecaux<Tgt: Target>(
    max_shape: &[DimSize],
    dtype: Dtype,
) -> impl proptest::strategy::Strategy<Value = TensorSpecAux<Tgt>> {
    use crate::spec::gen_vector_sizes_opt;
    use proptest::prelude::*;
    use proptest::sample::select;

    let max_shape = Shape::from(max_shape);

    (
        select(Tgt::all_layouts_for_shape(&max_shape)),
        select(Tgt::levels().to_vec()),
    )
        .prop_flat_map(move |(layout, level)| {
            let contiguous_abs = layout.all_contiguous_abs().collect::<Vec<_>>();
            (
                Just(layout),
                Just(level),
                select(contiguous_abs),
                any::<bool>(),
                select(
                    gen_vector_sizes_opt(Some(&max_shape), dtype, level.vector_bytes())
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .prop_map(
            |(layout, level, contig, aligned, vector_size)| TensorSpecAux {
                contig,
                aligned,
                level,
                layout,
                vector_size,
            },
        )
        .boxed()
}
