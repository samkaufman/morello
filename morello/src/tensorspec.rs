use itertools::iproduct;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;
use std::fmt::Display;
use std::iter::once;
use std::num::NonZeroU32;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::{row_major, Layout, LayoutError, PhysDim};
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
    pub contig: Contig,
    pub aligned: bool,
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

#[derive(Default, Clone)]
pub struct TensorSpecAuxNonDepBimap<Tgt: Target> {
    phantom: std::marker::PhantomData<Tgt>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct TensorSpecArbMaxShape(pub Shape);

#[derive(thiserror::Error, Debug)]
pub enum CanonicalizeError {
    #[error("Layout does not apply: {0}")]
    LayoutError(#[from] LayoutError),
    #[error("Volume is not a multiple of vector size")]
    VectorSizeInvalid,
}

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
        Self::new_canon_checked(
            shape,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_size,
        )
        .unwrap()
    }

    pub fn new_canon_checked(
        shape: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> Result<Self, CanonicalizeError> {
        let mut r = Self::new_noncanon(
            shape,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_size,
        );
        r.canonicalize()?;
        Ok(r)
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
        if shape.is_empty() {
            panic!("Invalid shape: {:?}", shape);
        }
        if aux.vector_size.is_some() != aux.level.vector_rf() {
            panic!(
                "vector_size must be specified if and only if the bank ({:?}) is a vector register file", aux.level
            )
        }
        TensorSpec { shape, dtype, aux }
    }

    pub fn layout(&self) -> &Layout {
        &self.aux.layout
    }

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
        self.aux.layout.applies_to_shape(shape)
    }

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size()) * u64::from(self.volume().get())
    }

    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn volume(&self) -> DimSize {
        DimSize::new(self.shape.iter().map(|d| d.get()).product()).unwrap()
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn contiguous_abs(&self) -> Contig {
        self.aux.contig
    }

    pub fn set_contiguous_abs(&mut self, contiguous_abs: Contig) {
        self.aux.contig = contiguous_abs;
    }

    pub fn is_contiguous(&self) -> bool {
        self.aux.contig == self.aux.layout.contiguous_full()
    }

    pub fn aligned(&self) -> bool {
        self.aux.aligned
    }

    pub fn set_aligned(&mut self, aligned: bool) {
        self.aux.aligned = aligned;
    }

    pub const fn level(&self) -> <Tgt as Target>::Level {
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
    /// canonicalized for the given shape.
    pub fn shrink(&mut self, shape: &[DimSize], aligned: bool) -> Result<(), LayoutError> {
        let (new_layout, new_contig) =
            self.aux
                .layout
                .update_for_tiling(self.shape(), shape, self.aux.contig)?;
        self.shape = Shape::from(shape);
        self.aux.layout = new_layout;
        self.aux.contig = new_contig;
        self.aux.aligned = aligned;
        Ok(())
    }

    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        let vector_size = self.aux.vector_size;
        if !check_tensor_vector_size::<Tgt>(self.shape(), self.dtype, &self.level(), vector_size) {
            return Err(CanonicalizeError::VectorSizeInvalid);
        }
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
                "Cannot drop non-degenerate dimension {} of shape {:?}",
                dim,
                new_shape
            );
            new_shape.remove(dim.into());
        }

        let (new_layout, new_contig) = if new_shape.iter().all(|&d| d.get() == 1) {
            let new_layout = row_major(new_shape.len().try_into().unwrap());
            let new_contig = new_layout.contiguous_full();
            (new_layout, new_contig)
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

    pub(crate) fn one_prefix(&self) -> TensorSpec<Tgt> {
        let mut new_shape = Vec::with_capacity(self.shape().len() + 1);
        new_shape.push(nz!(1u32));
        new_shape.extend_from_slice(self.shape());

        let mut new_layout = self.layout().clone();
        let was_initially_contiguous = new_layout.contiguous_full() == self.contiguous_abs();

        for (logical_dim, _) in new_layout.0.iter_mut() {
            *logical_dim += 1;
        }
        new_layout.0.insert(0, (0, PhysDim::Dynamic));

        let mut new_contig = self.contiguous_abs();
        if was_initially_contiguous {
            new_contig = new_layout.contiguous_full();
        }

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
    pub fn can_move_to(&self, _dest_layout: &Layout, dest_level: &Tgt::Level) -> bool {
        // If the destination is in VRF, then the operand volume must be a multiple of at least one
        // of the vector sizes.
        let vector_bytes = dest_level.vector_bytes();
        if !vector_bytes.is_empty() {
            let bytes = self.volume().get() * u32::from(self.dtype.size());
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

/// Implements [`proptest::arbitrary::Arbitrary`] to yield canonical [TensorSpec]s.
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
        .into_iter()
        .map(|m| 1..=m.get())
        .collect::<Vec<_>>()
        .prop_flat_map(|shp| {
            let shp = TensorSpecArbMaxShape(Shape::from(
                shp.iter()
                    .map(|&x| DimSize::new(x).unwrap())
                    .collect::<Vec<_>>(),
            ));
            let aux_strategy = TensorSpecAux::arbitrary_with(shp.clone());
            let dtype_strategy = any::<Dtype>();
            (Just(shp), dtype_strategy, aux_strategy)
        })
        .prop_map(|(shp, dtype, aux)| TensorSpec::new_noncanon_with_aux(shp.0, dtype, aux))
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub(crate) fn canonicalize(&mut self, shape: &[DimSize]) -> Result<(), CanonicalizeError> {
        let (new_layout, new_contig) = self.layout.update_for_tiling(shape, shape, self.contig)?;
        self.layout = new_layout;
        self.contig = new_contig;
        Ok(())
    }

    pub fn is_canonical(&self, shape: &[DimSize]) -> bool {
        if !self.layout.is_row_major() && shape.iter().all(|d| d.get() == 1) {
            false
        } else {
            let Layout(dims) = &self.layout;

            // Count the number of packings applied to each logical dimension.
            // As a special case, `packings` is empty if there are no packed dims.
            // (This avoids potentially spilling onto the heap for unpacked
            // layouts.)
            let mut packings = Vec::new();
            for (logical_dim, s) in dims.as_slice() {
                if matches!(s, PhysDim::Packed(_)) {
                    if packings.is_empty() {
                        packings.resize(dims.len(), 0);
                    }
                    packings[usize::from(*logical_dim)] += 1;
                }
            }

            if !packings.is_empty() {
                for idx in (0..dims.len()).rev() {
                    let (logical_dim, s) = dims[idx];
                    let logical_dim_usize = usize::from(logical_dim);
                    if packings[logical_dim_usize] == 1
                        && s == PhysDim::Packed(shape[logical_dim_usize])
                    {
                        return false;
                    }
                }
            }

            if self.layout.has_noncanon_size_one_dynamic_dimensions(shape) {
                return false;
            }

            let physical_rank = dims.len();
            let first_contig_idx = u8::try_from(physical_rank).unwrap() - self.contig;
            if first_contig_idx > 0 {
                let ps = self
                    .layout
                    .physical_size(first_contig_idx - 1, shape)
                    .unwrap()
                    .get();
                if ps == 1 {
                    return false;
                }
            }
            true
        }
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
        Box::new(
            iproduct!(
                [true, false],
                Tgt::all_layouts_for_shape(&self.tensor_shape, self.tensor_dtype),
                vector_options
            )
            .flat_map(move |(aligned, layout, vector_size)| {
                layout
                    .all_contiguous_abs()
                    .map(move |contig| TensorSpecAux {
                        contig,
                        aligned,
                        layout: layout.clone(),
                        vector_size: vector_size.map(|v| DimSize::new(v).unwrap()),
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
    type Codomain = ((Layout, u8, u32), [BimapInt; 2]);

    fn apply(&self, aux: &TensorSpecAux<Tgt>) -> Self::Codomain {
        (
            (
                aux.layout.clone(),
                BiMap::apply(&Tgt::Level::bimap(), &aux.level),
                aux.vector_size.map(|v| v.get()).unwrap_or(0),
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
            vector_size: NonZeroU32::new(*vector_size).map(Some).unwrap_or(None),
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
/// Returns `true` if the level is not a VRF.
pub(crate) fn check_tensor_vector_size<Tgt: Target>(
    shape: &[DimSize],
    dtype: Dtype,
    level: &Tgt::Level,
    vector_size: Option<DimSize>,
) -> bool {
    let vector_bytes = level.vector_bytes();
    if vector_bytes.is_empty() {
        debug_assert!(vector_size.is_none());
        return true;
    }
    let vector_size = vector_size.unwrap();
    let volume = DimSize::new(shape.iter().map(|d| d.get()).product()).unwrap();
    let bytes = volume.get() * u32::from(dtype.size());
    volume.get() % vector_size.get() == 0 && vector_bytes.iter().any(|&vb| bytes % vb == 0)
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(5);
    parts.push(aux.level.to_string());

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
    use proptest::prelude::*;
    use proptest::sample::select;

    let max_shape = Shape::from(max_shape);
    (
        select(Tgt::all_layouts_for_shape(&max_shape, dtype)),
        select(Tgt::levels().to_vec()),
    )
        .prop_flat_map(move |(layout, level)| {
            let contiguous_abs = layout.all_contiguous_abs().collect::<Vec<_>>();
            (
                Just(layout),
                Just(level),
                select(contiguous_abs),
                any::<bool>(),
                select(gen_vector_sizes_opt(dtype, level.vector_bytes()).collect::<Vec<_>>()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::layout::{row_major, Layout, PhysDim};
    use crate::target::{ArmTarget, CpuMemoryLevel, MemoryLevel, Target, X86Target};
    use crate::tensorspec::{arb_noncanon_tensorspec, TensorSpec, TensorSpecArbMaxShape};
    use crate::{layout, shape};
    use proptest::prelude::*;
    use proptest::proptest;

    proptest! {
        // TODO: Make an ARM variant
        #[test]
        fn test_canonicalize_errors_if_vector_not_a_multiple_x86(
            tspec in arb_noncanon_tensorspec::<X86Target>(&[nz!(16u32), nz!(16u32)])
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
        fn tensorspec_canonicalize_should_be_idempodent_x86(tspec in any::<TensorSpec<X86Target>>()) {
            shared_tensorspec_canonicalize_should_be_idempodent(tspec)
        }

        // TODO: Modify `any::<TensorSpec<_>>` to generate multiple ranks and dtypes.
        #[test]
        fn tensorspec_canonicalize_should_be_idempodent_arm(tspec in any::<TensorSpec<ArmTarget>>()) {
            shared_tensorspec_canonicalize_should_be_idempodent(tspec)
        }

        #[test]
        fn tensorspec_canonicalize_only_changes_contig_if_layout_dims_change_x86(
            tspec in any_with::<TensorSpec<X86Target>>(TensorSpecArbMaxShape(shape![4, 4, 4, 4]))
        ) {
            shared_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change(tspec)
        }

        #[test]
        fn tensorspec_canonicalize_only_changes_contig_if_layout_dims_change_arm(
            tspec in any_with::<TensorSpec<ArmTarget>>(TensorSpecArbMaxShape(shape![4, 4, 4, 4]))
        ) {
            shared_tensorspec_canonicalize_only_changes_contig_if_layout_dims_change(tspec)
        }

        // TODO: Add ARM variant
        #[test]
        fn test_tensorspecaux_canonicalize_is_noop_if_already_canonical_x86(
            shape in [1..=16u32, 1..=16u32]
                .prop_map(|v| v.map(|x| x.try_into().unwrap())),
            aux in any::<TensorSpecAux<X86Target>>()
        ) {
            let mut canonicalized_aux = aux.clone();
            if canonicalized_aux.canonicalize(&shape).is_ok() {
                if aux == canonicalized_aux {
                    prop_assert!(aux.is_canonical(&shape));
                } else {
                    prop_assert!(!aux.is_canonical(&shape));
                }
            }
        }
    }

    #[test]
    fn test_tensorspec_canonicalize_drops_unused_dynamic_dimensions() {
        let layout = Layout::new(vec![
            (1, PhysDim::Dynamic),
            (0, PhysDim::Dynamic),
            (1, PhysDim::Packed(nz!(8u32))),
        ]);
        let tensorspec = TensorSpec::<X86Target>::new_canon(
            shape![32, 8],
            Dtype::Uint8,
            layout.contiguous_full(),
            true,
            CpuMemoryLevel::GL,
            layout,
            None,
        );
        assert_eq!(tensorspec.layout(), &row_major(2));
        assert_eq!(tensorspec.contiguous_abs(), row_major(2).contiguous_full());
    }

    // TODO: Rename
    #[test]
    fn test_1() {
        let mut tspec = TensorSpec::<X86Target> {
            shape: shape![5, 2, 8, 4],
            dtype: crate::common::Dtype::Uint8,
            aux: crate::tensorspec::TensorSpecAux {
                contig: 3,
                aligned: false,
                level: CpuMemoryLevel::GL,
                layout: layout![
                    (0, PhysDim::Dynamic),
                    (2, PhysDim::Dynamic),
                    (3, PhysDim::Dynamic),
                    (1, PhysDim::Dynamic),
                    (2, PhysDim::Packed(4))
                ],
                vector_size: None,
            },
        };
        tspec.canonicalize().unwrap();
        assert_eq!(tspec.aux.contig, 3);
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

        let Layout(dims_a) = tspec.layout();
        let Layout(dims_b) = second.layout();
        assert!(
            dims_a != dims_b || tspec.aux.contig == second.aux.contig,
            "Dims were unchanged, but contig. changed from {:?} to {:?}",
            tspec.aux.contig,
            second.aux.contig
        );
    }
}
