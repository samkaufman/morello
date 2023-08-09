use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::target::{
    broadcastvecmult_applies_to_operands, memsetzero_applies_to_operands, mult_applies_to_operands,
    valueassign_applies_to_operands, vectorassign_applies_to_operands,
    vectorzero_applies_to_operands, MemoryLevel, Target, TargetId,
};

use crate::imp::kernels::KernelType;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
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
    type Level = X86MemoryLevel;

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

#[allow(clippy::upper_case_acronyms)]
#[derive(
    Eq, PartialEq, Debug, Copy, Clone, Hash, Deserialize, Serialize, enum_iterator::Sequence,
)]
pub enum X86MemoryLevel {
    RF,
    VRF,
    L1,
    GL,
}

impl MemoryLevel for X86MemoryLevel {
    fn is_addressed(&self) -> bool {
        match &self {
            X86MemoryLevel::RF => true,
            X86MemoryLevel::VRF => true,
            X86MemoryLevel::L1 => false,
            X86MemoryLevel::GL => true,
        }
    }

    fn cache_hit_cost(&self) -> MainCost {
        match &self {
            X86MemoryLevel::RF => 0,
            X86MemoryLevel::VRF => 0,
            X86MemoryLevel::L1 => 10,
            X86MemoryLevel::GL => 100,
        }
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match &self {
            X86MemoryLevel::VRF => &[16, 32],
            _ => &[],
        }
    }
}

impl PartialOrd for X86MemoryLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (X86MemoryLevel::RF, X86MemoryLevel::VRF) => None,
            (X86MemoryLevel::VRF, X86MemoryLevel::RF) => None,
            (X86MemoryLevel::RF, _) => Some(Ordering::Less),
            (X86MemoryLevel::VRF, _) => Some(Ordering::Less),
            (_, X86MemoryLevel::RF) => Some(Ordering::Greater),
            (_, X86MemoryLevel::VRF) => Some(Ordering::Greater),
            (X86MemoryLevel::L1, X86MemoryLevel::GL) => Some(Ordering::Less),
            (X86MemoryLevel::GL, X86MemoryLevel::L1) => Some(Ordering::Greater),
            (X86MemoryLevel::L1, X86MemoryLevel::L1) => unreachable!(),
            (X86MemoryLevel::GL, X86MemoryLevel::GL) => unreachable!(),
        }
    }
}

impl Display for X86MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                X86MemoryLevel::RF => "RF",
                X86MemoryLevel::VRF => "VRF",
                X86MemoryLevel::L1 => "L1",
                X86MemoryLevel::GL => "GL",
            }
        )
    }
}
