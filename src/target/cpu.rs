use crate::codegen::c_utils::VecType;
use crate::common::{DimSize, Dtype};
use crate::cost::MainCost;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::kernels::KernelType;
use crate::layout::{col_major, nhwc, row_major, Layout};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::{dim_range, LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use crate::target::{MemoryLevel, Target, TargetId, LEVEL_COUNT};
use crate::tensorspec::TensorSpec;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter;

pub(super) trait CpuTarget:
    Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static
{
    fn target_id() -> TargetId;
    fn vec_types() -> &'static [VecType; 4];
}

#[allow(clippy::upper_case_acronyms)]
#[derive(
    Eq, PartialEq, Debug, Copy, Clone, Hash, Deserialize, Serialize, enum_iterator::Sequence,
)]
pub enum CpuMemoryLevel {
    RF,
    VRF,
    L1,
    GL,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuMemoryLevelBimap;

impl<T: CpuTarget> Target for T {
    type Level = CpuMemoryLevel;

    fn line_size() -> u32 {
        32
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new([64, 1024, 32_768, 1_073_741_824]))
    }

    fn processors() -> u8 {
        32
    }

    fn default_level() -> Self::Level {
        CpuMemoryLevel::GL
    }

    fn levels() -> [Self::Level; LEVEL_COUNT] {
        [
            CpuMemoryLevel::RF,
            CpuMemoryLevel::VRF,
            CpuMemoryLevel::L1,
            CpuMemoryLevel::GL,
        ]
    }

    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => vec![slower],
            CpuMemoryLevel::L1 => vec![slower, CpuMemoryLevel::RF, CpuMemoryLevel::VRF],
            CpuMemoryLevel::GL => vec![slower, CpuMemoryLevel::L1],
        }
    }

    fn all_layouts_for_shape(rank: u8, dtype: Dtype) -> Vec<Layout> {
        let all_target_vector_bytes = Self::levels()
            .into_iter()
            .flat_map(|lvl| lvl.vector_bytes().iter().copied())
            .collect::<Vec<_>>();

        // The following could be faster. It keeps two copies of the non-packed layouts
        // (`base` and the first few values in `result`) and it doesn't compute the size
        // of the `result` ahead-of-time, potentially causing some Vec resizes.
        let base = match rank {
            2 => vec![row_major(2), col_major(2)],
            4 => vec![row_major(4), nhwc()],
            _ => vec![row_major(rank)],
        };

        // Extend with all possible packings.
        let all_packing_sizes = pack_sizes(None, dtype, &all_target_vector_bytes);
        let mut result = base.clone();
        result.extend(base.iter().flat_map(|original_layout| {
            let Layout::New(dims) = original_layout;
            (0..dims.len()).cartesian_product(&all_packing_sizes).map(
                |(packing_dim, &packing_size)| {
                    Layout::New(
                        dims.iter()
                            .cloned()
                            .chain(iter::once((
                                packing_dim.try_into().unwrap(),
                                Some(packing_size),
                            )))
                            .collect(),
                    )
                },
            )
        }));
        result
    }

    fn move_destination_layouts(shape: &[DimSize], dtype: Dtype) -> Vec<Layout> {
        let all_target_vector_bytes = Self::levels()
            .into_iter()
            .flat_map(|lvl| lvl.vector_bytes().iter().copied())
            .collect::<Vec<_>>();

        // The following could be faster. It keeps two copies of the non-packed layouts
        // (`base` and the first few values in `result`) and it doesn't compute the size
        // of the `result` ahead-of-time, potentially causing some Vec resizes.
        let base = match shape.len() {
            2 => vec![row_major(2), col_major(2)],
            4 => vec![row_major(4), nhwc()],
            r => vec![row_major(r.try_into().unwrap())],
        };
        let mut result = base.clone();
        result.extend(base.iter().flat_map(|original_layout| {
            packed_layouts_for_standard_layout(
                original_layout,
                shape,
                dtype,
                &all_target_vector_bytes,
            )
        }));
        debug_assert!(result.iter().all(|r| r.applies_to_shape(shape)));
        result
    }

    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>> {
        match spec {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
                PrimitiveSpecType::Matmul { accum } => {
                    if *accum {
                        let mut microkernels = vec![];
                        if mult_applies_to_operands(&spec.parameters()) {
                            microkernels.push(Action::Place(KernelType::Mult));
                        }
                        if broadcastvecmult_applies_to_operands(&spec.parameters()) {
                            microkernels.push(Action::Place(KernelType::BroadcastVecMult));
                        }
                        Box::new(microkernels.into_iter())
                    } else {
                        Box::new(iter::empty())
                    }
                }
                PrimitiveSpecType::Conv { .. } => Box::new(iter::empty()),
                PrimitiveSpecType::Move { .. } => {
                    let mut microkernels = vec![];
                    if valueassign_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::ValueAssign));
                    }
                    if vectorassign_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::VectorAssign));
                    }
                    Box::new(microkernels.into_iter())
                }
                PrimitiveSpecType::Zero { .. } => {
                    let mut microkernels = vec![];
                    if memsetzero_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::MemsetZero));
                    }
                    if vectorzero_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::VectorZero));
                    }
                    Box::new(microkernels.into_iter())
                }
            },
            LogicalSpec::Compose { .. } => Box::new(iter::empty()),
        }
    }

    fn target_id() -> TargetId {
        <Self as CpuTarget>::target_id()
    }

    fn vec_types() -> &'static [VecType; 4] {
        <Self as CpuTarget>::vec_types()
    }
}

impl MemoryLevel for CpuMemoryLevel {
    fn is_addressed(&self) -> bool {
        match &self {
            CpuMemoryLevel::RF => true,
            CpuMemoryLevel::VRF => true,
            CpuMemoryLevel::L1 => false,
            CpuMemoryLevel::GL => true,
        }
    }

    fn cache_hit_cost(&self) -> MainCost {
        match &self {
            CpuMemoryLevel::RF => 0,
            CpuMemoryLevel::VRF => 0,
            CpuMemoryLevel::L1 => 10,
            CpuMemoryLevel::GL => 100,
        }
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match &self {
            // CpuMemoryLevel::VRF => &[16, 32],
            CpuMemoryLevel::VRF => &[32],
            _ => &[],
        }
    }
}

impl PartialOrd for CpuMemoryLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (CpuMemoryLevel::RF, CpuMemoryLevel::VRF) => None,
            (CpuMemoryLevel::VRF, CpuMemoryLevel::RF) => None,
            (CpuMemoryLevel::RF, _) => Some(Ordering::Less),
            (CpuMemoryLevel::VRF, _) => Some(Ordering::Less),
            (_, CpuMemoryLevel::RF) => Some(Ordering::Greater),
            (_, CpuMemoryLevel::VRF) => Some(Ordering::Greater),
            (CpuMemoryLevel::L1, CpuMemoryLevel::GL) => Some(Ordering::Less),
            (CpuMemoryLevel::GL, CpuMemoryLevel::L1) => Some(Ordering::Greater),
            (CpuMemoryLevel::L1, CpuMemoryLevel::L1) => unreachable!(),
            (CpuMemoryLevel::GL, CpuMemoryLevel::GL) => unreachable!(),
        }
    }
}

impl Display for CpuMemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                CpuMemoryLevel::RF => "RF",
                CpuMemoryLevel::VRF => "VRF",
                CpuMemoryLevel::L1 => "L1",
                CpuMemoryLevel::GL => "GL",
            }
        )
    }
}

impl BiMap for CpuMemoryLevelBimap {
    type Domain = CpuMemoryLevel;
    type Codomain = u8;

    fn apply(&self, level: &CpuMemoryLevel) -> u8 {
        match level {
            CpuMemoryLevel::RF => 0,
            CpuMemoryLevel::VRF => 1,
            CpuMemoryLevel::L1 => 2,
            CpuMemoryLevel::GL => 3,
        }
    }

    fn apply_inverse(&self, i: &u8) -> CpuMemoryLevel {
        match *i {
            0 => CpuMemoryLevel::RF,
            1 => CpuMemoryLevel::VRF,
            2 => CpuMemoryLevel::L1,
            3 => CpuMemoryLevel::GL,
            _ => panic!("Invalid index: {}", i),
        }
    }
}

impl CanonicalBimap for CpuMemoryLevel {
    type Bimap = CpuMemoryLevelBimap;

    fn bimap() -> Self::Bimap {
        CpuMemoryLevelBimap
    }
}

pub fn valueassign_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    debug_assert_eq!(operands.len(), 2);

    if operands.iter().flat_map(|o| o.shape()).any(|&d| d != 1) {
        return false;
    }

    for o in &operands[1..] {
        if (o.dtype(), o.layout()) != (operands[0].dtype(), operands[0].layout()) {
            return false;
        }
    }

    operands.iter().any(|o| o.level() == CpuMemoryLevel::RF)
        && operands
            .iter()
            .all(|o| o.level() == CpuMemoryLevel::RF || o.level() == CpuMemoryLevel::L1)
}

pub fn vectorassign_applies_to_operands<Tgt: Target>(operands: &[TensorSpec<Tgt>]) -> bool {
    if operands.iter().any(|o| !o.is_contiguous()) {
        return false;
    }
    if operands[0].dtype() != operands[1].dtype() {
        return false;
    }
    if operands[0].shape() != operands[1].shape() {
        return false;
    }
    if operands[0].layout() != operands[1].layout() {
        return false;
    }

    let mut has_vrf = false;
    for o in operands {
        if o.level().vector_rf() {
            has_vrf = true;
            match o.vector_size() {
                Some(vector_size) => {
                    let volume = o.shape().iter().product::<DimSize>();
                    if vector_size != volume {
                        return false;
                    }
                }
                None => {
                    panic!("No vector_size on operand in level {:?}", o.level());
                }
            }
        }
    }
    has_vrf
}

pub fn memsetzero_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != CpuMemoryLevel::RF {
        return false;
    }
    true
}

pub fn vectorzero_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != CpuMemoryLevel::VRF {
        return false;
    }
    let volume = operands[0].shape().iter().product::<DimSize>();
    match operands[0].vector_size() {
        Some(vector_size) if vector_size != volume => {
            return false;
        }
        None => return false,
        _ => (),
    };
    true
}

pub fn broadcastvecmult_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if operands[0].level() != CpuMemoryLevel::RF {
        return false;
    }
    for i in 1..3 {
        if operands[i].level() != CpuMemoryLevel::VRF {
            return false;
        }
        let volume = operands[i].shape().iter().product::<DimSize>();
        if volume != operands[i].vector_size().unwrap() {
            return false;
        }
        if !operands[i].aligned() || !operands[i].is_contiguous() {
            return false;
        }
        if operands[0].dtype() != operands[i].dtype() {
            return false;
        }
    }
    if operands[0].shape().iter().any(|d| *d != 1) {
        return false;
    }
    if operands[1].shape().len() != 2 || operands[1].shape()[0] != 1 {
        return false;
    }
    if operands[2].shape().to_vec() != vec![1, operands[1].shape()[1]] {
        return false;
    }
    true
}

pub fn mult_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    operands
        .iter()
        .all(|o| o.level() == CpuMemoryLevel::RF && o.shape().iter().all(|&d| d == 1))
}

/// Yields versions of the given [Layout] with packed dimensions added.
///
/// The specific sizes of the inner/packed dimension depend on the given layout, tensor
/// shape, and tensor [Dtype].
fn packed_layouts_for_standard_layout<'a>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
) -> impl Iterator<Item = Layout> + 'a {
    let Layout::New(dims) = &original_layout;
    debug_assert!(dims.iter().all(|(_, s)| s.is_none()));

    let final_nonone_dim = {
        let mut d = dims.len() - 1;
        while d > 0 && shape[d] == 1 {
            d -= 1;
        }
        d
    };

    dims[..final_nonone_dim].iter().flat_map(move |&(dim, _)| {
        let dims = dims.clone();
        let mut it = None;
        if shape[usize::from(dim)] != 1 {
            it = Some(
                pack_sizes(
                    Some(shape[usize::from(dim)]),
                    dtype,
                    all_target_vector_bytes,
                )
                .into_iter()
                .map(move |strip_size| {
                    Layout::new(
                        dims.iter()
                            .cloned()
                            .chain(iter::once((dim, Some(strip_size))))
                            .collect(),
                    )
                }),
            );
        }
        it.into_iter().flatten()
    })
}

fn pack_sizes(
    max_size: Option<DimSize>,
    dtype: Dtype,
    all_target_vector_bytes: &[u32],
) -> SmallVec<[DimSize; 2]> {
    all_target_vector_bytes
        .iter()
        .filter_map(|&bytes| {
            let s = bytes / DimSize::from(dtype.size());
            if max_size.map(|m| s < m).unwrap_or(true) {
                Some(s)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{
        common::{DimSize, Dtype},
        expr::{NonAffineExpr, Substitute},
        layout::BufferVar,
        layout::Layout,
        opaque_symbol::OpaqueSymbol,
        target::{Target, X86Target},
    };
    use itertools::Itertools;
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        #[test]
        fn test_all_layouts_for_shape_are_unique(
            rank in 1..=5u8, dtype in any::<Dtype>(),
        ) {
            assert_unique_layouts(&X86Target::all_layouts_for_shape(rank, dtype));
        }

        #[test]
        fn test_all_layouts_for_shape_contains_move_destination_layouts(
            shape in proptest::collection::vec(1..8u32, 1..=5),
            dtype in any::<Dtype>(),
        ) {
            let rank = shape.len().try_into().unwrap();
            let superset: HashSet<_> = X86Target::all_layouts_for_shape(rank, dtype)
                .into_iter()
                .collect();
            let subset: HashSet<_> = X86Target::move_destination_layouts(&shape, dtype)
                .into_iter()
                .collect();
            assert!(superset.is_superset(&subset));
        }

        #[test]
        fn test_move_destination_layouts_have_unique_index_expressions(
            shape in proptest::collection::vec(1..8u32, 1..=5),
            dtype in any::<Dtype>(),
        ) {
            let expr_id = OpaqueSymbol::new();
            let mut index_exprs: Vec<(Vec<i32>, Layout)> = Vec::new();
            for layout in X86Target::move_destination_layouts(&shape, dtype) {
                let ie = layout.buffer_indexing_expr(&expr_id, &shape);
                let evaluated_pts = eval_all_index_expr_points(&ie, &shape);
                for (prev_pts, prev_layout) in index_exprs.iter() {
                    if prev_pts == &evaluated_pts {
                        panic!(
                            "Layouts had identical indexing expressions {} \
                            for shape {:?}: {:?} and {:?}", ie, shape, prev_layout, layout
                        );
                    }
                }
                index_exprs.push((evaluated_pts, layout));
            }
        }

        #[test]
        #[should_panic]
        fn test_packed_layout_with_strip_size_one_panics(
            example in arb_test_packed_layout_with_strip_size_one_is_row_major()
        ) {
            let (shape, strip_dim) = example;
            Layout::new_packed(shape.len().try_into().unwrap(), strip_dim, 1);
        }
    }

    prop_compose! {
        fn arb_test_packed_layout_with_strip_size_one_is_row_major()
            (shape in proptest::collection::vec(1..8u32, 1..=5))
            (strip_dim in 0..shape.len(), shape in Just(shape))
            -> (Vec<DimSize>, u8)
        {
            (shape, strip_dim.try_into().unwrap())
        }
    }

    fn assert_unique_layouts(layouts: &[Layout]) {
        let layouts_set = layouts.iter().collect::<HashSet<_>>();
        assert_eq!(layouts.len(), layouts_set.len());
    }

    fn eval_all_index_expr_points(expr: &NonAffineExpr<BufferVar>, shape: &[DimSize]) -> Vec<i32> {
        let mut results = vec![];
        for pt in shape.iter().map(|&d| 0..d).multi_cartesian_product() {
            let evaluated: NonAffineExpr<&str> = expr.clone().map_vars(&mut |var| match var {
                BufferVar::TileIdx(_, _) => panic!("TileIdx in index expression"),
                BufferVar::Pt(dim, _) => NonAffineExpr::constant(pt[dim as usize] as i32),
            });
            assert!(
                evaluated.0.is_empty(),
                "Non-constant index expression: {:?}",
                evaluated
            );
            results.push(evaluated.1);
        }
        results
    }
}
