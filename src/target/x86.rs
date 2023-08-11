use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{
    broadcastvecmult_applies_to_operands, memsetzero_applies_to_operands, mult_applies_to_operands,
    valueassign_applies_to_operands, vectorassign_applies_to_operands,
    vectorzero_applies_to_operands, CpuMemoryLevel, Target, TargetId,
};

use crate::imp::kernels::KernelType;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use serde::Serialize;
use std::fmt::Debug;
use std::iter;

const X86_VEC_TYPES: [VecType; 4] = [
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 32,
        name: "vub32",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 16,
        name: "vub16",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
];

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct X86Target;

impl Target for X86Target {
    type Level = CpuMemoryLevel;

    fn default_level() -> CpuMemoryLevel {
        CpuMemoryLevel::GL
    }

    fn levels() -> Vec<Self::Level> {
        enum_iterator::all::<Self::Level>().collect()
    }

    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => vec![slower],
            CpuMemoryLevel::L1 => vec![slower, CpuMemoryLevel::RF, CpuMemoryLevel::VRF],
            CpuMemoryLevel::GL => vec![slower, CpuMemoryLevel::L1],
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

    fn target_id() -> TargetId {
        TargetId::X86
    }

    fn vec_types() -> &'static [VecType; 4] {
        &X86_VEC_TYPES
    }
}
