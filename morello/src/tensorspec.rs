use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;
use std::fmt::Display;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::{Layout, LayoutBimap, LayoutBuilder, LayoutError, PhysDim};
use crate::target::{Memory, Target};
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
    pub memory: Tgt::Memory,
    pub layout: Layout,
}

#[derive(Clone)]
pub struct TensorSpecAuxSurMap<Tgt: Target> {
    tensor_shape: Vec<DimSize>, // TODO: Make into &'a [DimSize]
    tensor_dtype: Dtype,
    phantom: std::marker::PhantomData<Tgt>,
}

#[derive(Clone)]
pub struct TensorSpecAuxNonDepBimap<Tgt: Target> {
    layout_bimap: LayoutBimap,
    pub phantom: std::marker::PhantomData<Tgt>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct TensorSpecArbMaxShape(pub Shape);

#[derive(thiserror::Error, Debug)]
pub enum CanonicalizeError {
    #[error("Layout does not apply: {0}")]
    LayoutError(#[from] LayoutError),
    #[error("No vector size for the tensor's dtype divides its volume")]
    NoVectorSizeForVolume,
}

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new_canon<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        memory: Tgt::Memory,
        layout: L,
    ) -> Self {
        Self::new_canon_checked(shape, dtype, memory, layout).unwrap()
    }

    pub fn new_canon_checked<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        memory: Tgt::Memory,
        layout: L,
    ) -> Result<Self, CanonicalizeError> {
        let mut r = Self::new_noncanon(shape, dtype, memory, layout);
        r.canonicalize()?;
        Ok(r)
    }

    pub fn new_noncanon<L: LayoutBuilder>(
        shape: Shape,
        dtype: Dtype,
        memory: Tgt::Memory,
        layout: L,
    ) -> Self {
        let layout = layout.build(&shape);
        Self::new_noncanon_with_aux(shape, dtype, TensorSpecAux { memory, layout })
    }

    pub fn new_noncanon_with_aux(shape: Shape, dtype: Dtype, aux: TensorSpecAux<Tgt>) -> Self {
        if shape.is_empty() {
            panic!("Invalid shape: {shape:?}");
        }
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
        let memory = self.memory();

        debug_assert_eq!(original_shape.len(), shape.len());
        debug_assert!(shape.iter().zip(original_shape).all(|(i, o)| i <= o));

        if parallel_loop && !memory.can_parallel_tile() && original_shape != shape {
            return false;
        }
        if !self.tile_keeps_vector_size(shape) {
            return false;
        }
        !memory.has_layout() || self.aux.layout.applies_to_shape(shape)
    }

    /// Returns true if a tile of the given shape has the same vector size as its parent.
    pub(crate) fn tile_keeps_vector_size(&self, shape: &[DimSize]) -> bool {
        let Some(parent_vector_size) = self.vector_size() else {
            return true;
        };
        let tile_volume = DimSize::new(shape.iter().map(|d| d.get()).product()).unwrap();
        Tgt::vector_size_for(self.dtype(), tile_volume) == Some(parent_vector_size)
    }

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size()) * u64::from(self.volume().get())
    }

    /// Returns the memory units consumed by this tensor based on that memory's model.
    ///
    /// For register-counting memories, returns register counts, possibly divided by vector size.
    /// For other memories, returns an estimate of cache lines used.
    pub fn memory_units(&self) -> u64 {
        if self.memory().counts_registers() {
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
    pub const fn memory(&self) -> <Tgt as Target>::Memory {
        self.aux.memory
    }

    // TODO: Rename and document `vector_size` "accessor"
    #[inline]
    pub fn vector_size(&self) -> Option<DimSize> {
        if !self.memory().vector_rf() {
            return None;
        }
        Tgt::vector_size_for(self.dtype(), self.volume())
    }

    /// Returns a new TensorSpec with the given shape.
    ///
    /// The result's layout and contiguousness abstraction will have been
    /// canonicalized for the given shape.
    pub fn shrink(&mut self, shape: &[DimSize]) -> Result<(), LayoutError> {
        if self.aux.memory.has_layout() {
            self.aux.layout = self.aux.layout.update_for_tiling(self.shape(), shape)?;
        } else {
            assert!(self.aux.layout.is_empty());
        }
        self.shape = Shape::from(shape);
        Ok(())
    }

    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        check_tensor_vector_size::<Tgt>(self.shape(), self.dtype, &self.memory())?;
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

        let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
        let new_layout = self.layout().dim_drop(&dropped_dims_set);

        Self::new_canon(new_shape, self.dtype(), self.memory(), new_layout)
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

        Self::new_canon(new_shape, self.dtype(), self.memory(), new_layout)
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
/// This generates [TensorSpec]s of varying shapes, dtypes, and memories.
/// The [Layout] and contiguousness abstraction are constrained to be valid together and for the
/// generated shape.
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
            let aux_strategy = any_with::<TensorSpecAux<Tgt>>(TensorSpecArbMaxShape(shp.clone()));
            (Just(shp), Just(dtype), aux_strategy)
        })
        .prop_map(|(shp, dtype, aux)| TensorSpec::new_noncanon_with_aux(shp, dtype, aux))
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub(crate) fn canonicalize(&mut self, shape: &[DimSize]) -> Result<(), CanonicalizeError> {
        if !self.memory.has_layout() {
            self.layout = Layout::empty();
        } else {
            self.layout = self.layout.update_for_tiling(shape, shape)?;
        }
        Ok(())
    }

    pub fn is_canonical(&self, shape: &[DimSize]) -> bool {
        if !self.memory.has_layout() {
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
    type Parameters = TensorSpecArbMaxShape;
    type Strategy = proptest::strategy::BoxedStrategy<TensorSpecAux<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use crate::layout::LayoutArbParams;
        use proptest::prelude::*;
        use proptest::sample::select;

        (
            any_with::<Layout>(LayoutArbParams::valid_for_shape(&args.0)),
            select(Tgt::memories().to_vec()),
        )
            .prop_flat_map(move |(layout, memory)| {
                let contiguous_abs = layout.all_contiguous_abs().collect::<Vec<_>>();
                (Just(layout), Just(memory), select(contiguous_abs))
            })
            .prop_map(|(mut layout, memory, contig)| {
                layout.set_contig(contig);
                TensorSpecAux { memory, layout }
            })
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
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Memory, Codomain = u8>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((), [BimapInt; 1]);
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + 'static>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        ((), [BiMap::apply(&Tgt::Memory::bimap(), &t.memory).into()])
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let ((), [level_int]) = i;
        let memory = BiMap::apply_inverse(&Tgt::Memory::bimap(), &(*level_int).try_into().unwrap());
        let layouts = if memory.has_layout() {
            Tgt::all_layouts_for_shape(&self.tensor_shape, self.tensor_dtype)
        } else {
            vec![Layout::empty()]
        };

        Box::new(layouts.into_iter().flat_map(move |layout| {
            layout.all_contiguous_abs().map(move |contig| {
                let mut layout_with_contig = layout.clone();
                layout_with_contig.set_contig(contig);
                TensorSpecAux {
                    layout: layout_with_contig,
                    memory,
                }
            })
        }))
    }
}

impl<Tgt: Target> TensorSpecAuxNonDepBimap<Tgt> {
    pub fn new(tensor_shape: &[DimSize]) -> Self {
        Self {
            layout_bimap: LayoutBimap {
                tensor_shape: tensor_shape.into(),
            },
            phantom: std::marker::PhantomData,
        }
    }
}

impl<Tgt> BiMap for TensorSpecAuxNonDepBimap<Tgt>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Memory, Codomain = u8>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((), [BimapInt; 4]);

    fn apply(&self, aux: &TensorSpecAux<Tgt>) -> Self::Codomain {
        debug_assert!(BiMap::defined_for(self, aux));

        let level_int = BiMap::apply(&Tgt::Memory::bimap(), &aux.memory);
        let [physdim_value, pack_size_value] = BiMap::apply(&self.layout_bimap, &aux.layout);
        (
            (),
            [
                level_int.into(),
                physdim_value,
                pack_size_value,
                (aux.layout.contiguous_full() - aux.layout.contig()).into(),
            ],
        )
    }

    fn apply_inverse(&self, v: &Self::Codomain) -> Self::Domain {
        let ((), [level_val, physdim_val, pack_size_val, contig]) = *v;

        let memory = BiMap::apply_inverse(&Tgt::Memory::bimap(), &level_val.try_into().unwrap());
        let mut layout = BiMap::apply_inverse(&self.layout_bimap, &[physdim_val, pack_size_val]);
        let contig: Contig = contig.try_into().unwrap();
        layout.set_contig(layout.contiguous_full() - contig);
        TensorSpecAux { layout, memory }
    }

    fn defined_for(&self, aux: &TensorSpecAux<Tgt>) -> bool {
        BiMap::defined_for(&self.layout_bimap, &aux.layout)
    }
}

/// Returns an error if `memory` is VRF and no vector size is available that divides tensor volume.
pub(crate) fn check_tensor_vector_size<Tgt: Target>(
    shape: &[DimSize],
    dtype: Dtype,
    memory: &Tgt::Memory,
) -> Result<(), CanonicalizeError> {
    if !memory.vector_rf() {
        return Ok(());
    }
    let volume = DimSize::new(shape.iter().map(|d| d.get()).product()).unwrap();
    if Tgt::vector_size_for(dtype, volume).is_none() {
        return Err(CanonicalizeError::NoVectorSizeForVolume);
    }
    Ok(())
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(2);
    parts.push(aux.memory.to_string());

    if !aux.layout.is_row_major() || !aux.layout.is_fully_contiguous() {
        parts.push(aux.layout.to_string());
    }

    join_into_string(&parts, ", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Dtype, Shape};
    use crate::layout::{row_major, Layout};
    use crate::target::{ArmTarget, Avx2Target, Avx512Target, CpuMemory, Memory, Target};
    use crate::tensorspec::{arb_noncanon_tensorspec, TensorSpec, TensorSpecArbMaxShape};
    use crate::{layout, shape};
    use proptest::prelude::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_vrf_canonicalize_reports_no_vector_size_iff_none_divides_volume_avx2(
            mut tspec in arb_noncanon_tensorspec::<Avx2Target>(&[nz!(16u32), nz!(16u32)])
                .prop_filter("TensorSpec is not in VRF", |t| t.memory().vector_rf())
        ) {
            let has_vector_size = Avx2Target::vector_size_for(tspec.dtype(), tspec.volume()).is_some();
            let vector_size_error = matches!(
                tspec.canonicalize(),
                Err(CanonicalizeError::NoVectorSizeForVolume)
            );
            prop_assert_eq!(vector_size_error, !has_vector_size);
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
            (shape, aux) in [1..=8u32, 1..=8u32]
                .prop_map(|shape| Shape::from(
                    shape.map(|d| d.try_into().unwrap()).as_slice()
                ))
                .prop_flat_map(|shape| {
                    (
                        Just(shape.clone()),
                        any_with::<TensorSpecAux<Avx2Target>>(TensorSpecArbMaxShape(shape)),
                    )
                })
        ) {
            let bimap = TensorSpecAuxNonDepBimap::<Avx2Target>::new(&shape);
            prop_assume!(BiMap::defined_for(&bimap, &aux));
            let output = BiMap::apply(&bimap, &aux);
            assert_eq!(aux, BiMap::apply_inverse(&bimap, &output));
        }

        #[test]
        fn test_vrf_tile_valid_iff_vector_size_unchanged_avx2(
            (tspec, tile_shape) in arb_vrf_tensorspec_and_tile_shape::<Avx2Target>()
        ) {
            shared_test_vrf_tile_valid_iff_vector_size_unchanged(&tspec, &tile_shape)?;
        }

        #[test]
        fn test_vrf_tile_valid_iff_vector_size_unchanged_avx512(
            (tspec, tile_shape) in arb_vrf_tensorspec_and_tile_shape::<Avx512Target>()
        ) {
            shared_test_vrf_tile_valid_iff_vector_size_unchanged(&tspec, &tile_shape)?;
        }

        #[test]
        fn test_vrf_tile_valid_iff_vector_size_unchanged_arm(
            (tspec, tile_shape) in arb_vrf_tensorspec_and_tile_shape::<ArmTarget>()
        ) {
            shared_test_vrf_tile_valid_iff_vector_size_unchanged(&tspec, &tile_shape)?;
        }

        /// If a VRF tensor can be tiled to `tile_a`, then tiling it to a smaller `tile_b` directly
        /// must be valid exactly when tiling `tile_a` to `tile_b` is.
        ///
        /// This is dominated by `scheduling::tiling::tests::test_tile_out_transitivity`, which
        /// checks the same property end-to-end over arbitrary Specs. We have this test because it
        /// targets VRF tensors (which arbitrary Specs rarely produce) and isolates
        /// [TensorSpec::is_valid_tile_shape].
        #[test]
        fn test_vrf_tile_validity_is_transitive_avx2(
            (tspec, tile_a, tile_b) in arb_vrf_tensorspec_and_tile_shape::<Avx2Target>()
                .prop_flat_map(|(tspec, tile_a)| {
                    let tile_b = tile_a
                        .iter()
                        .map(|d| (1..=d.get()).prop_map(|d| DimSize::new(d).unwrap()))
                        .collect::<Vec<_>>();
                    (Just(tspec), Just(tile_a), tile_b)
                })
        ) {
            prop_assume!(tspec.is_valid_tile_shape(&tile_a, false));
            let mut tspec_a = tspec.clone();
            tspec_a.shrink(&tile_a).unwrap();
            prop_assert_eq!(
                tspec.is_valid_tile_shape(&tile_b, false),
                tspec_a.is_valid_tile_shape(&tile_b, false),
                "{} -> {:?} vs. {} -> {:?}",
                tspec,
                tile_b,
                tspec_a,
                tile_b
            );
        }
    }

    /// Test that the contig. coordinate doesn't change when tiling sets a dimeension to one.
    ///
    /// This is useful for improving the compression of the dependency R*-Tree when it contains all
    /// tilings of a Spec.
    #[test]
    fn test_tensorspecauxnondepbimap_contig_code_unchanged_when_tiling_dim_to_one() {
        let mut layout = layout![0, 1];
        layout.set_contig(1);
        let mut spec = TensorSpec::<Avx2Target>::new_noncanon(
            shape![4, 4],
            Dtype::Float32,
            CpuMemory::GL,
            layout,
        );

        let before_bimap = TensorSpecAuxNonDepBimap::<Avx2Target>::new(spec.shape());
        let ((), before_encoding) = BiMap::apply(&before_bimap, &spec.aux);

        spec.shrink(&shape![4, 1]).unwrap();

        let after_bimap = TensorSpecAuxNonDepBimap::<Avx2Target>::new(spec.shape());
        let ((), after_encoding) = BiMap::apply(&after_bimap, &spec.aux);

        assert_eq!(before_encoding[3], after_encoding[3]);
    }

    #[test]
    fn test_tensorspec_canonicalize_drops_unused_dynamic_dimensions() {
        let layout = layout![1, 0, 1 p(8)];
        let tensorspec =
            TensorSpec::<Avx2Target>::new_canon(shape![32, 8], Dtype::Uint8, CpuMemory::GL, layout);
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
                    memory: CpuMemory::GL,
                    layout,
                }
            },
        };
        tspec.canonicalize().unwrap();
        assert_eq!(tspec.aux.layout.contig(), 3);
    }

    #[test]
    fn test_vrf_tiling_rejects_narrower_vector_size() {
        // A 24-element f32 VRF tensor on AVX2 uses 8-lane vectors. A 12-element tile evenly divides
        // it but would use 4-lane vectors. Allowing it breaks tiling transitivity: its own
        // 8-element tile (a partial tile, since 8 does not divide 12) would derive 8-lane vectors
        // again.
        let spec = TensorSpec::<Avx2Target>::new_canon(
            shape![24],
            Dtype::Float32,
            CpuMemory::VRF,
            layout![0],
        );
        assert_eq!(spec.vector_size(), Some(nz!(8u32)));
        assert!(!spec.is_valid_tile_shape(&shape![12], false));
    }

    #[test]
    fn test_vrf_tile_accepts_same_submax_vector_size() {
        // A 3x4 f32 VRF tensor on AVX2 uses narrower-than-maximum 4-lane vectors. As a result, it
        // is valid to tile to 1x4, which will address exactly one whole vector register.
        let spec = TensorSpec::<Avx2Target>::new_canon(
            shape![3, 4],
            Dtype::Float32,
            CpuMemory::VRF,
            layout![0, 1],
        );
        assert!(spec.is_valid_tile_shape(&shape![1, 4], false));
    }

    #[test]
    fn test_canonicalize_rejects_volume_not_divisible_by_any_vector() {
        let mut spec = TensorSpec::<Avx2Target> {
            shape: shape![3, 2],
            dtype: Dtype::Float32,
            aux: TensorSpecAux {
                memory: CpuMemory::VRF,
                layout: layout![0, 1],
            },
        };
        assert!(matches!(
            spec.canonicalize(),
            Err(CanonicalizeError::NoVectorSizeForVolume)
        ));
    }

    #[test]
    fn test_tiling_packed_suffix_not_fully_contiguous() {
        let mut spec = TensorSpec::<Avx2Target>::new_canon(
            shape![1, 8, 8],
            Dtype::Float32,
            CpuMemory::L1,
            layout![1, 2, 1 p(2)],
        );
        spec.shrink(&shape![1, 8, 1]).unwrap();
        assert!(
            !spec.is_contiguous(),
            "Tiling to 1x8x1 should break contiguousness; contig={:?} layout={:?}",
            spec.layout().contig(),
            spec.layout(),
        );
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

    /// Checks that a VRF tensor can be tiled to `tile_shape` exactly when the tile keeps the
    /// tensor's vector size (and its layout applies to the tile).
    fn shared_test_vrf_tile_valid_iff_vector_size_unchanged<Tgt: Target>(
        tspec: &TensorSpec<Tgt>,
        tile_shape: &[DimSize],
    ) -> Result<(), TestCaseError> {
        let tile_volume = DimSize::new(tile_shape.iter().map(|d| d.get()).product()).unwrap();
        let tile_vector_size = Tgt::vector_size_for(tspec.dtype(), tile_volume);

        let vector_sizes_match = tile_vector_size == tspec.vector_size();
        let layout_applies =
            !tspec.memory().has_layout() || tspec.layout().applies_to_shape(tile_shape);
        let should_be_able_to_tile = vector_sizes_match && layout_applies;

        prop_assert_eq!(
            tspec.is_valid_tile_shape(tile_shape, false),
            should_be_able_to_tile,
        );
        Ok(())
    }

    /// Generates a canonical, in-VRF [TensorSpec] paired with a shape to tile it to.
    fn arb_vrf_tensorspec_and_tile_shape<Tgt: Target>(
    ) -> impl Strategy<Value = (TensorSpec<Tgt>, Vec<DimSize>)> {
        any_with::<TensorSpec<Tgt>>(TensorSpecArbMaxShape(shape![16, 16, 16]))
            .prop_filter("TensorSpec is not in VRF", |t| t.memory().vector_rf())
            .prop_flat_map(|tspec| {
                let tile_shape = tspec
                    .shape()
                    .iter()
                    .map(|d| (1..=d.get()).prop_map(|d| DimSize::new(d).unwrap()))
                    .collect::<Vec<_>>();
                (Just(tspec), tile_shape)
            })
    }
}
