use itertools::{iproduct, Itertools};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;
use std::fmt::Display;
use std::iter::once;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::{row_major, Layout, LayoutBuilder, LayoutError, PhysDim};
use crate::target::{MemoryLevel, Target};
use crate::utils::join_into_string;

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct TensorSpec<Tgt: Target> {
    pub shape: Shape,
    pub dtype: Dtype,
    pub aux: TensorSpecAux<Tgt>,
}

// TODO: This probably shouldn't be public.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TensorSpecAux<Tgt: Target> {
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_size: Option<DimSize>, // # number of values in a vector register
}

#[derive(Clone)]
pub struct TensorSpecAuxSurMap<Tgt: Target> {
    tensor_shape: Vec<DimSize>, // TODO: Make into &'a [DimSize]
    tensor_dtype: Dtype,
    phantom: std::marker::PhantomData<Tgt>,
}

#[derive(Clone)]
pub struct TensorSpecAuxNonDepBimap<Tgt: Target> {
    pub dtype: Dtype,
    pub phantom: std::marker::PhantomData<Tgt>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct TensorSpecArbMaxShape(pub Shape);

#[derive(thiserror::Error, Debug)]
pub enum CanonicalizeError {
    #[error("Layout does not apply: {0}")]
    LayoutError(#[from] LayoutError),
    #[error("Target does not support the specified vector size for this data type")]
    VectorSizeInvalid,
    #[error("Tensor volume is not compatible with the specified vector size")]
    VectorSizeVolumeIncompatible,
}

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new_canon<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        level: Tgt::Level,
        layout: L,
        vector_size: Option<DimSize>,
    ) -> Self {
        Self::new_canon_checked(shape, dtype, level, layout, vector_size).unwrap()
    }

    pub fn new_canon_checked<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        level: Tgt::Level,
        layout: L,
        vector_size: Option<DimSize>,
    ) -> Result<Self, CanonicalizeError> {
        let mut r = Self::new_noncanon(shape, dtype, level, layout, vector_size);
        r.canonicalize()?;
        Ok(r)
    }

    pub fn new_noncanon<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        level: Tgt::Level,
        layout: L,
        vector_size: Option<DimSize>,
    ) -> Self {
        let layout = layout.build(&shape);
        Self::new_noncanon_with_aux(
            shape,
            dtype,
            TensorSpecAux {
                level,
                layout,
                vector_size,
            },
        )
    }

    pub fn new_noncanon_with_aux(shape: Shape, dtype: Dtype, aux: TensorSpecAux<Tgt>) -> Self {
        if shape.is_empty() {
            panic!("Invalid shape: {shape:?}");
        }
        match (aux.vector_size, aux.level.vector_bytes()) {
            (None, []) => {}
            (None, [_, ..]) | (Some(_), []) => {
                panic!(
                    "vector_size must be specified if and only if the bank ({:?}) is a vector register file", aux.level
                );
            }
            (Some(vector_size), v) => {
                let vector_bytes = vector_size.get() * u32::from(dtype.size());
                if !v.contains(&vector_bytes) {
                    // TODO: Remove vector_bytes from panic message
                    panic!(
                        "Invalid vector size {:?} (bytes: {}) for dtype {:?} and level {:?} (poss.: {:?})",
                        vector_size, vector_bytes, dtype, aux.level, v
                    );
                }
            }
        };
        TensorSpec { shape, dtype, aux }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.aux.layout
    }

    #[inline]
    pub fn set_layout(&mut self, new_layout: Layout) {
        self.aux.layout = new_layout;
    }

    /// Returns true if this TensorSpec can be tiled to the given shape.
    pub fn is_valid_tile_shape(&self, shape: &[DimSize], parallel_loop: bool) -> bool {
        let original_shape = self.shape();
        let level = self.level();

        debug_assert_eq!(original_shape.len(), shape.len());
        debug_assert!(shape.iter().zip(original_shape).all(|(i, o)| i <= o));

        if parallel_loop && !level.can_parallel_tile() && original_shape != shape {
            return false;
        }
        if shape.iter().all(|d| d.get() == 1) {
            return true;
        }
        !level.has_layout() || self.aux.layout.applies_to_shape(shape)
    }

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size()) * u64::from(self.volume().get())
    }

    /// Returns the memory units consumed by this tensor based on its level's memory model.
    ///
    /// For register-counting levels, returns register counts, possibly divided by vector size.
    /// For other levels, returns an estimate of cache lines used.
    pub fn memory_units(&self) -> u64 {
        if self.level().counts_registers() {
            if let Some(vector_size) = self.vector_size() {
                u64::from(self.volume().get().div_ceil(vector_size.get()))
            } else {
                u64::from(self.volume().get())
            }
        } else {
            self.layout()
                .estimate_cache_lines::<Tgt>(self.shape(), self.dtype())
                .into()
        }
    }

    #[inline]
    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn volume(&self) -> DimSize {
        DimSize::new(self.shape.iter().map(|d| d.get()).product()).unwrap()
    }

    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    // TODO: Remove. Should not be public.
    #[inline]
    pub(crate) fn contiguous_abs(&self) -> Contig {
        self.aux.layout.contig()
    }

    // TODO: Remove. This is just sugar for Layout's is_fully_contiguous().
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.aux.layout.is_fully_contiguous()
    }

    #[inline]
    pub const fn level(&self) -> <Tgt as Target>::Level {
        self.aux.level
    }

    #[inline]
    pub fn vector_size(&self) -> Option<DimSize> {
        self.aux.vector_size
    }

    pub fn set_level(&mut self, level: Tgt::Level, vector_size: Option<DimSize>) {
        assert_eq!(
            level.vector_rf(),
            vector_size.is_some(),
            "Cannot set level to {level:?} with vector shape {vector_size:?}"
        );
        self.aux.level = level;
        self.aux.vector_size = vector_size;
    }

    /// Returns a new TensorSpec with the given shape.
    ///
    /// The result's layout and contiguousness abstraction will have been
    /// canonicalized for the given shape.
    pub fn shrink(&mut self, shape: &[DimSize]) -> Result<(), LayoutError> {
        if self.aux.level.has_layout() {
            self.aux.layout = self.aux.layout.update_for_tiling(self.shape(), shape)?;
        } else {
            assert!(self.aux.layout.is_empty());
        }
        self.shape = Shape::from(shape);
        Ok(())
    }

    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        let vector_size = self.aux.vector_size;
        check_tensor_vector_size::<Tgt>(self.shape(), self.dtype, &self.level(), vector_size)?;
        self.aux.canonicalize(&self.shape)
    }

    /// Returns a TensorSpec with given size-one dimensions dropped.
    ///
    /// The given dimension indices must be sorted in ascending order.
    ///
    /// The result will be canonicalized. If any given dimension index is not
    /// size one, this method panics.
    pub fn squeeze_dims(&self, dropped_dims: &[u8]) -> TensorSpec<Tgt> {
        // Make sure dropped_dims is sorted.
        debug_assert!(dropped_dims.windows(2).all(|w| w[0] < w[1]));

        let mut new_shape = Shape::from(self.shape());
        for &dim in dropped_dims.iter().rev() {
            assert_eq!(
                new_shape[usize::from(dim)].get(),
                1,
                "Cannot drop non-degenerate dimension {dim} of shape {new_shape:?}"
            );
            new_shape.remove(dim.into());
        }

        let new_layout = if new_shape.iter().all(|&d| d.get() == 1) {
            row_major(&new_shape)
        } else {
            let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
            self.layout().dim_drop(&dropped_dims_set)
        };

        Self::new_canon(
            new_shape,
            self.dtype(),
            self.level(),
            new_layout,
            self.vector_size(),
        )
    }

    pub(crate) fn one_prefix(&self) -> TensorSpec<Tgt> {
        let mut new_shape = Shape::with_capacity(self.shape().len() + 1);
        new_shape.push(nz!(1u32));
        new_shape.extend_from_slice(self.shape());

        let mut new_layout = self.layout().clone();
        let was_initially_contiguous = self.layout().is_fully_contiguous();

        for (logical_dim, _) in new_layout.dims.iter_mut() {
            *logical_dim += 1;
        }
        new_layout.dims.insert(0, (0, PhysDim::Dynamic));

        let mut new_contig = self.contiguous_abs();
        if was_initially_contiguous {
            new_contig = new_layout.contiguous_full();
        }
        new_layout.set_contig(new_contig);

        Self::new_canon(
            new_shape,
            self.dtype(),
            self.level(),
            new_layout,
            self.vector_size(),
        )
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
            write!(f, ", {aux_part})")
        }
    }
}

/// Implements [`proptest::arbitrary::Arbitrary`] to yield canonical [TensorSpec]s.
///
/// This generates [TensorSpec]s of varying shapes, dtypes, and levels.
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

        arb_noncanon_tensorspec(&args.0)
            .prop_filter_map("TensorSpec was not canonical", |mut tensor_spec| {
                let canon_result = tensor_spec.canonicalize();
                canon_result.ok().map(|_| tensor_spec)
            })
            .boxed()
    }
}

#[cfg(test)]
fn arb_noncanon_tensorspec<Tgt: Target>(
    max_shape: &[DimSize],
) -> impl proptest::strategy::Strategy<Value = TensorSpec<Tgt>> {
    use proptest::prelude::*;

    max_shape
        .iter()
        .map(|m| 1..=m.get())
        .collect::<Vec<_>>()
        .prop_flat_map(|shp| (Just(shp), any::<Dtype>()))
        .prop_flat_map(|(shp, dtype)| {
            let shp = Shape::from(
                shp.iter()
                    .map(|&x| DimSize::new(x).unwrap())
                    .collect::<Vec<_>>(),
            );
            let aux_strategy = arb_tensorspecaux(&shp, dtype);
            (Just(shp), Just(dtype), aux_strategy)
        })
        .prop_map(|(shp, dtype, aux)| TensorSpec::new_noncanon_with_aux(shp, dtype, aux))
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub(crate) fn canonicalize(&mut self, shape: &[DimSize]) -> Result<(), CanonicalizeError> {
        if !self.level.has_layout() {
            self.layout = Layout::empty();
        } else {
            self.layout = self.layout.update_for_tiling(shape, shape)?;
        }
        Ok(())
    }

    pub fn is_canonical(&self, shape: &[DimSize]) -> bool {
        if !self.level.has_layout() {
            return self.layout.is_empty();
        }
        match self.layout.update_for_tiling(shape, shape) {
            Ok(new_layout) => new_layout == self.layout,
            Err(_) => false, // If update_for_tiling fails, the layout is not canonical
        }
    }
}

impl<Tgt: Target> Display for TensorSpecAux<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aux_part = tensorspec_aux_str(self);
        if aux_part.is_empty() {
            write!(f, "(_)")
        } else {
            write!(f, "({aux_part})")
        }
    }
}

#[cfg(test)]
impl<Tgt: Target> proptest::arbitrary::Arbitrary for TensorSpecAux<Tgt> {
    type Parameters = (TensorSpecArbMaxShape, Option<Dtype>);
    type Strategy = proptest::strategy::BoxedStrategy<TensorSpecAux<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let (shape, dtype) = args;
        if let Some(dtype) = dtype {
            arb_tensorspecaux(&shape.0, dtype).boxed()
        } else {
            any::<Dtype>()
                .prop_flat_map(move |d| arb_tensorspecaux(&shape.0, d))
                .boxed()
        }
    }
}

#[cfg(test)]
impl Default for TensorSpecArbMaxShape {
    fn default() -> Self {
        use crate::shape;
        Self(shape![8, 8])
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
        let vector_bytes = level.vector_bytes();
        let mut vector_options = vector_bytes
            .iter()
            .map(|&vb| Some(vb / dtype_bytes))
            .collect::<Vec<_>>();
        if vector_options.is_empty() {
            vector_options.push(None);
        }
        let layouts = if level.has_layout() {
            Tgt::all_layouts_for_shape(&self.tensor_shape, self.tensor_dtype)
        } else {
            vec![Layout::empty()]
        };

        Box::new(iproduct!(layouts.into_iter(), vector_options).flat_map(
            move |(layout, vector_size)| {
                layout.all_contiguous_abs().map(move |contig| {
                    let mut layout_with_contig = layout.clone();
                    layout_with_contig.set_contig(contig);
                    TensorSpecAux {
                        layout: layout_with_contig,
                        vector_size: vector_size.map(|v| DimSize::new(v).unwrap()),
                        level,
                    }
                })
            },
        ))
    }
}

impl<Tgt: Target> TensorSpecAuxNonDepBimap<Tgt> {
    pub fn new(dtype: Dtype) -> Self {
        Self {
            dtype,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<Tgt> BiMap for TensorSpecAuxNonDepBimap<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((Layout,), [BimapInt; 3]);

    fn apply(&self, aux: &TensorSpecAux<Tgt>) -> Self::Codomain {
        let level_int = BiMap::apply(&Tgt::Level::bimap(), &aux.level);
        let tgt_vector_bytes = Tgt::Level::vector_bytes(&aux.level);
        debug_assert!(
            tgt_vector_bytes.iter().tuple_windows().all(|(a, b)| a <= b),
            "target's vector_bytes must be sorted"
        );
        let vector_size_idx = aux
            .vector_size
            .map(|v| {
                let vector_bytes = v.get() * u32::from(self.dtype.size());
                tgt_vector_bytes.binary_search(&vector_bytes).unwrap()
            })
            .unwrap_or(0);
        (
            (aux.layout.clone(),),
            [
                level_int.into(),
                aux.layout.contig().into(),
                vector_size_idx.try_into().unwrap(),
            ],
        )
    }

    fn apply_inverse(&self, v: &Self::Codomain) -> Self::Domain {
        let ((ref layout,), [level_val, contig, vector_size_idx]) = *v;

        // `unwrap_or_else` rather than `unwrap` to avoid needing a Debug bound
        let level = BiMap::apply_inverse(&Tgt::Level::bimap(), &level_val.try_into().unwrap());

        let tgt_vector_bytes = Tgt::Level::vector_bytes(&level);
        let vector_size = if tgt_vector_bytes.is_empty() {
            assert_eq!(vector_size_idx, 0);
            None
        } else {
            let b = tgt_vector_bytes[usize::try_from(vector_size_idx).unwrap()]
                / u32::from(self.dtype.size());
            Some(DimSize::new(b).unwrap())
        };

        let mut layout = layout.clone();
        layout.set_contig(contig.try_into().unwrap());
        TensorSpecAux {
            layout,
            level,
            vector_size,
        }
    }
}

pub(crate) fn gen_vector_sizes(
    dtype: Dtype,
    vector_bytes: &[u32],
) -> impl Iterator<Item = DimSize> + '_ {
    assert!(!vector_bytes.is_empty());
    assert!(
        vector_bytes
            .iter()
            .all(|&vb| vb % u32::from(dtype.size()) == 0),
        "vector_bytes must be a multiple of dtype size"
    );
    vector_bytes.iter().map(move |&vb| {
        let value_cnt = vb / u32::from(dtype.size());
        DimSize::new(value_cnt).unwrap()
    })
}

pub(crate) fn gen_vector_sizes_opt(
    dtype: Dtype,
    vector_bytes: &[u32],
) -> impl Iterator<Item = Option<DimSize>> + '_ {
    let mut iter_a = None;
    let mut iter_b = None;
    if vector_bytes.is_empty() {
        iter_a = Some(once(None));
    } else {
        iter_b = Some(gen_vector_sizes(dtype, vector_bytes).map(Some));
    }
    iter_a
        .into_iter()
        .flatten()
        .chain(iter_b.into_iter().flatten())
}

/// Checks if an in-VRF tensor's shape, dtype, and chosen vector size are valid for a given level.
///
/// This checks both that the vector exists for that target and that the vector size is a multiple
/// of the shape and dtype.
///
/// Returns `Ok`, a `CanonicalizeError::VectorSizeInvalid`, or
/// `CanonicalizeError::VectorSizeVolumeIncompatible`.
pub(crate) fn check_tensor_vector_size<Tgt: Target>(
    shape: &[DimSize],
    dtype: Dtype,
    level: &Tgt::Level,
    vector_size: Option<DimSize>,
) -> Result<(), CanonicalizeError> {
    let vector_bytes_allowed = level.vector_bytes();
    if vector_bytes_allowed.is_empty() {
        debug_assert!(vector_size.is_none());
        return Ok(());
    }
    let vector_size = vector_size.expect("vector_size must be Some for vector level");
    let vector_size_bytes = vector_size.get() * u32::from(dtype.size());
    if !vector_bytes_allowed.contains(&vector_size_bytes) {
        return Err(CanonicalizeError::VectorSizeInvalid);
    }
    let volume = DimSize::new(shape.iter().map(|d| d.get()).product()).unwrap();
    let bytes = volume.get() * u32::from(dtype.size());

    if !vector_bytes_allowed
        .iter()
        .any(|&vb| bytes.is_multiple_of(vb))
    {
        return Err(CanonicalizeError::VectorSizeInvalid);
    }
    if !volume.get().is_multiple_of(vector_size.get()) {
        return Err(CanonicalizeError::VectorSizeVolumeIncompatible);
    }
    Ok(())
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(5);
    parts.push(aux.level.to_string());

    if !aux.layout.is_row_major() || !aux.layout.is_fully_contiguous() {
        parts.push(aux.layout.to_string());
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
    use crate::layout::LayoutArbRankBounds;
    use proptest::prelude::*;
    use proptest::sample::select;

    (
        any_with::<Layout>(LayoutArbRankBounds::for_shape(max_shape)),
        select(Tgt::levels().to_vec()),
    )
        .prop_flat_map(move |(layout, level)| {
            let contiguous_abs = layout.all_contiguous_abs().collect::<Vec<_>>();
            (
                Just(layout),
                Just(level),
                select(contiguous_abs),
                select(gen_vector_sizes_opt(dtype, level.vector_bytes()).collect::<Vec<_>>()),
            )
        })
        .prop_map(|(mut layout, level, contig, vector_size)| {
            layout.set_contig(contig);
            TensorSpecAux {
                level,
                layout,
                vector_size,
            }
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::layout::{row_major, Layout};
    use crate::target::{ArmTarget, Avx2Target, CpuMemoryLevel, MemoryLevel, Target};
    use crate::tensorspec::{arb_noncanon_tensorspec, TensorSpec, TensorSpecArbMaxShape};
    use crate::{layout, shape};
    use proptest::prelude::*;
    use proptest::proptest;

    proptest! {
        // TODO: Make an ARM variant
        #[test]
        fn test_canonicalize_errors_if_vector_not_a_multiple_avx2(
            tspec in arb_noncanon_tensorspec::<Avx2Target>(&[nz!(16u32), nz!(16u32)])
                .prop_filter("TensorSpec is not in VRF", |t| t.level().vector_rf())
        ) {
            let volume = tspec.volume().get();
            let dtype = tspec.dtype();

            let mut can_move = true;
            // If the destination is in VRF, then the operand volume must be a multiple of at least one
            // of the vector sizes.
            let vector_bytes = tspec.level().vector_bytes();
            if !vector_bytes.is_empty() {
                let bytes = volume * u32::from(dtype.size());
                if vector_bytes.iter().all(|&vb| bytes % vb != 0) {
                    can_move = false;
                }
            }

            if !can_move {
                let mut tspec = tspec;
                prop_assert!(tspec.canonicalize().is_err());
            }
        }

        // TODO: Modify `any::<TensorSpec<_>>` to generate multiple ranks and dtypes.
        #[test]
        fn test_tensorspec_canonicalize_should_be_idempodent_avx2(tspec in any::<TensorSpec<Avx2Target>>()) {
            shared_tensorspec_canonicalize_should_be_idempodent(tspec)
        }

        // TODO: Modify `any::<TensorSpec<_>>` to generate multiple ranks and dtypes.
        #[test]
        fn test_tensorspec_canonicalize_should_be_idempodent_arm(tspec in any::<TensorSpec<ArmTarget>>()) {
            shared_tensorspec_canonicalize_should_be_idempodent(tspec)
        }

        #[test]
        fn test_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change_avx2(
            tspec in any_with::<TensorSpec<Avx2Target>>(TensorSpecArbMaxShape(shape![4, 4, 4, 4]))
        ) {
            shared_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change(tspec)
        }

        #[test]
        fn test_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change_arm(
            tspec in any_with::<TensorSpec<ArmTarget>>(TensorSpecArbMaxShape(shape![4, 4, 4, 4]))
        ) {
            shared_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change(tspec)
        }

        // TODO: Add ARM variant
        #[test]
        fn test_tensorspecaux_canonicalize_is_noop_if_already_canonical_avx2(
            shape in [1..=16u32, 1..=16u32]
                .prop_map(|v| v.map(|x| x.try_into().unwrap())),
            aux in any::<TensorSpecAux<Avx2Target>>()
        ) {
            let mut canonicalized_aux = aux.clone();
            if canonicalized_aux.canonicalize(&shape).is_ok() {
                prop_assert_eq!(
                    aux == canonicalized_aux, aux.is_canonical(&shape),
                    "canonicalized aux {:?}; shape {:?}",
                    canonicalized_aux, shape
                );
            }
        }

        #[test]
        fn test_tensorspecauxnondepbimap_inverts(
            (dtype, aux) in any::<Dtype>()
                .prop_flat_map(|d| {
                    (Just(d), any_with::<TensorSpecAux<Avx2Target>>((Default::default(), Some(d))))
                })
        ) {
            let bimap = TensorSpecAuxNonDepBimap::<Avx2Target> {
                dtype,
                phantom: std::marker::PhantomData,
            };
            let output = BiMap::apply(&bimap, &aux);
            assert_eq!(aux, BiMap::apply_inverse(&bimap, &output));
        }
    }

    #[test]
    fn test_tensorspec_canonicalize_drops_unused_dynamic_dimensions() {
        let layout = layout![1, 0, 1 p(8)];
        let tensorspec = TensorSpec::<Avx2Target>::new_canon(
            shape![32, 8],
            Dtype::Uint8,
            CpuMemoryLevel::GL,
            layout,
            None,
        );
        assert_eq!(tensorspec.layout(), &row_major(tensorspec.shape()));
        assert_eq!(
            tensorspec.contiguous_abs(),
            row_major(tensorspec.shape()).contiguous_full()
        );
    }

    // TODO: Rename
    #[test]
    fn test_1() {
        let mut tspec = TensorSpec::<Avx2Target> {
            shape: shape![5, 2, 8, 4],
            dtype: crate::common::Dtype::Uint8,
            aux: {
                let mut layout = layout![0, 2, 3, 1];
                layout.set_contig(3);
                crate::tensorspec::TensorSpecAux {
                    level: CpuMemoryLevel::GL,
                    layout,
                    vector_size: None,
                }
            },
        };
        tspec.canonicalize().unwrap();
        assert_eq!(tspec.aux.layout.contig(), 3);
    }

    #[test]
    #[should_panic]
    fn test_cannot_build_tensorspec_with_invalid_vector_size_canon() {
        let mut l = layout![0 p(4)];
        l.set_contiguous_none();
        TensorSpec::<Avx2Target>::new_canon(
            shape![1, 1, 1],
            Dtype::Uint32,
            CpuMemoryLevel::VRF,
            l,
            Some(nz!(16u32)),
        );
    }

    #[test]
    #[should_panic]
    fn test_cannot_build_tensorspec_with_invalid_vector_size_noncanon() {
        let mut l = layout![0 p(4)];
        l.set_contiguous_none();
        TensorSpec::<Avx2Target>::new_noncanon(
            shape![1, 1, 1],
            Dtype::Uint32,
            CpuMemoryLevel::VRF,
            l,
            Some(nz!(16u32)),
        );
    }

    #[test]
    #[should_panic]
    fn test_cannot_build_tensorspec_with_invalid_vector_size_noncanon_with_aux() {
        let mut layout = layout![0 p(4)];
        layout.set_contiguous_none();
        TensorSpec::<Avx2Target>::new_noncanon_with_aux(
            shape![1, 1, 1],
            Dtype::Uint32,
            TensorSpecAux {
                level: CpuMemoryLevel::VRF,
                layout,
                vector_size: Some(nz!(16u32)),
            },
        );
    }

    #[test]
    fn test_canonicalize_rejects_unsupported_vector_size_even_when_volume_is_multiple() {
        let mut spec = TensorSpec::<Avx2Target> {
            shape: shape![1, 1, 16],
            dtype: Dtype::Uint8,
            aux: TensorSpecAux {
                level: CpuMemoryLevel::VRF,
                layout: layout![0, 1, 2],
                vector_size: Some(nz!(8u32)),
            },
        };
        assert!(matches!(
            spec.canonicalize(),
            Err(CanonicalizeError::VectorSizeInvalid)
        ));
    }

    fn shared_tensorspec_canonicalize_should_be_idempodent<Tgt: Target>(
        mut tspec: TensorSpec<Tgt>,
    ) {
        tspec.canonicalize().unwrap();
        let mut second = tspec.clone();
        second.canonicalize().unwrap();
        assert_eq!(tspec, second);
    }

    fn shared_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change<Tgt: Target>(
        tspec: TensorSpec<Tgt>,
    ) {
        let mut second = tspec.clone();
        second.canonicalize().unwrap();

        let Layout { dims: dims_a, .. } = tspec.layout();
        let Layout { dims: dims_b, .. } = second.layout();
        assert!(
            dims_a != dims_b || tspec.layout().contig() == second.layout().contig(),
            "Dims were unchanged, but contig. changed from {:?} to {:?}",
            tspec.layout().contig(),
            second.layout().contig()
        );
    }
}
