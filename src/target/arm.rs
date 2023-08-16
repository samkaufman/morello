use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{cpu::CpuTarget, TargetId};

use serde::Serialize;
use std::fmt::Debug;

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

impl CpuTarget for ArmTarget {
    fn target_id() -> TargetId {
        TargetId::Arm
    }

    fn vec_types() -> &'static [VecType; 4] {
        &ARM_VEC_TYPES
    }
}
