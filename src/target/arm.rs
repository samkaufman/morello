use crate::common::{DimSize, Dtype};
use crate::imp::kernels::KernelType;
use crate::layout::{nhwc, row_major, Layout};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
// TODO: Use X86MemoryLevel for now for simplicity,
//       but this should be changed to ArmMemoryLevel eventually.
use crate::codegen::c_utils::VecType;
use crate::target::{MemoryLevel, Target, Targets, X86MemoryLevel};
use crate::tensorspec::TensorSpec;

use serde::Serialize;
use smallvec::smallvec;
use std::fmt::Debug;
use std::iter;

const ARM_VEC_TYPES: [VecType; 4] = [
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "uint32x4x2_t",
        load_fn: "vld2q_u32",
        store_fn: "vst2q_u32",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "uint32x4_t",
        load_fn: "vld1q_u32",
        store_fn: "vst1q_u32",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 32,
        name: "vub32",
        native_type_name: "uint8x16x2_t",
        load_fn: "vld2q_u8",
        store_fn: "vst2q_u8",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 16,
        name: "vub16",
        native_type_name: "uint8x16_t",
        load_fn: "vld1q_u8",
        store_fn: "vst1q_u8",
    },
];

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct ArmTarget;

impl Target for ArmTarget {
    type Level = X86MemoryLevel;

    fn line_size() -> u32 {
        32
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new(smallvec![64, 1024, 32_768, 1_073_741_824]))
    }

    fn processors() -> u8 {
        32
    }

    fn default_level() -> X86MemoryLevel {
        X86MemoryLevel::GL
    }

    fn levels() -> Vec<Self::Level> {
        enum_iterator::all::<Self::Level>().collect()
    }

    fn faster_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower {
            X86MemoryLevel::RF | X86MemoryLevel::VRF => vec![],
            X86MemoryLevel::L1 => vec![X86MemoryLevel::RF, X86MemoryLevel::VRF],
            X86MemoryLevel::GL => vec![X86MemoryLevel::L1],
        }
    }

    fn all_layouts_for_shape(shape: &[DimSize]) -> Vec<Layout> {
        // warn!("NHWC and packed layouts are unimplemented.");

        let rm_iter = iter::once(row_major(shape.len().try_into().unwrap()));
        if shape.iter().all(|d| *d == 1) {
            rm_iter.collect()
        } else if shape.len() == 2 {
            rm_iter
                .chain(iter::once(Layout::Standard {
                    dim_order: smallvec![1, 0],
                }))
                .collect()
        } else if shape.len() == 4 {
            rm_iter.chain(iter::once(nhwc())).collect()
        } else {
            rm_iter.collect()
        }
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

    fn by_enum() -> Targets {
        Targets::Arm
    }

    fn get_vec_types() -> &'static [VecType; 4] {
        &ARM_VEC_TYPES
    }
}

pub fn valueassign_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    debug_assert_eq!(operands.len(), 2);

    if operands.iter().flat_map(|o| o.dim_sizes()).any(|&d| d != 1) {
        return false;
    }

    for o in &operands[1..] {
        if (o.dtype(), o.layout()) != (operands[0].dtype(), operands[0].layout()) {
            return false;
        }
    }

    operands.iter().any(|o| o.level() == X86MemoryLevel::RF)
        && operands
            .iter()
            .all(|o| o.level() == X86MemoryLevel::RF || o.level() == X86MemoryLevel::L1)
}

pub fn vectorassign_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    if operands.iter().any(|o| !o.is_contiguous()) {
        return false;
    }
    if operands[0].dtype() != operands[1].dtype() {
        return false;
    }
    if operands[0].dim_sizes() != operands[1].dim_sizes() {
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
                    let volume = o.dim_sizes().iter().product::<DimSize>();
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

pub fn cacheaccess_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    return false;

    if operands.iter().all(|o| o.level().is_addressed()) {
        return false;
    }
    if operands.iter().any(|o| !o.is_contiguous()) {
        return false;
    }
    if operands[0].dtype() != operands[1].dtype() {
        return false;
    }
    if operands[0].dim_sizes() != operands[1].dim_sizes() {
        return false;
    }
    if operands[0].layout() != operands[1].layout() {
        return false;
    }
    true
}

pub fn memsetzero_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    true
}

pub fn vectorzero_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::VRF {
        return false;
    }
    let volume = operands[0].dim_sizes().iter().product::<DimSize>();
    match operands[0].vector_size() {
        Some(vector_size) if vector_size != volume => {
            return false;
        }
        None => return false,
        _ => (),
    };
    true
}

pub fn broadcastvecmult_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    for i in 1..3 {
        if operands[i].level() != X86MemoryLevel::VRF {
            return false;
        }
        let volume = operands[i].dim_sizes().iter().product::<DimSize>();
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
    if operands[0].dim_sizes().iter().any(|d| *d != 1) {
        return false;
    }
    if operands[1].dim_sizes().len() != 2 || operands[1].dim_sizes()[0] != 1 {
        return false;
    }
    if operands[2].dim_sizes().to_vec() != vec![1, operands[1].dim_sizes()[1]] {
        return false;
    }
    true
}

pub fn mult_applies_to_operands(operands: &[TensorSpec<ArmTarget>]) -> bool {
    operands
        .iter()
        .all(|o| o.level() == X86MemoryLevel::RF && o.dim_sizes().iter().all(|&d| d == 1))
}
