use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{
    broadcastvecmult_applies_to_operands, memsetzero_applies_to_operands, mult_applies_to_operands,
    valueassign_applies_to_operands, vectorassign_applies_to_operands,
    vectorzero_applies_to_operands, Target, Targets, X86MemoryLevel,
};

use crate::imp::kernels::KernelType;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use serde::Serialize;
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
    // TODO: Use X86MemoryLevel for now for simplicity,
    //       but this should be changed to ArmMemoryLevel eventually.
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
        {
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
    }

    fn by_enum() -> Targets {
        Targets::Arm
    }

    fn vec_types() -> &'static [VecType; 4] {
        &ARM_VEC_TYPES
    }
}
