use super::{CpuKernel, CpuTarget, Kernel, TargetId};
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::memorylimits::MemoryAllocation;
use crate::spec::LogicalSpec;
use crate::{codegen::c_utils::VecType, views::View};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

const ARM_VEC_TYPES: [VecType; 16] = [
    VecType {
        dtype: Dtype::Bfloat16,
        value_cnt: 16,
        name: "vbf16_16",
        native_type_name: "bfloat16x8x2_t",
        load_fn: "vld2q_u16",
        load_fn_arg0: "const uint16_t",
        store_fn: "vst2q_u16",
        store_fn_arg0: "uint16_t",
    },
    VecType {
        dtype: Dtype::Bfloat16,
        value_cnt: 8,
        name: "vbf16_8",
        native_type_name: "bfloat16x8_t",
        load_fn: "vld1q_u16",
        load_fn_arg0: "const uint16_t",
        store_fn: "vst1q_u16",
        store_fn_arg0: "uint16_t",
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 8,
        name: "vf8",
        native_type_name: "float32x4x2_t",
        load_fn: "vld2q_f32",
        load_fn_arg0: "const float32_t",
        store_fn: "vst2q_f32",
        store_fn_arg0: "float32_t",
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 4,
        name: "vf4",
        native_type_name: "float32x4_t",
        load_fn: "vld1q_f32",
        load_fn_arg0: "const float32_t",
        store_fn: "vst1q_f32",
        store_fn_arg0: "float32_t",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 8,
        name: "vsi8",
        native_type_name: "int32x4x2_t",
        load_fn: "vld2q_s32",
        load_fn_arg0: "const int32_t",
        store_fn: "vst2q_s32",
        store_fn_arg0: "int32_t",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 4,
        name: "vsi4",
        native_type_name: "int32x4_t",
        load_fn: "vld1q_s32",
        load_fn_arg0: "const int32_t",
        store_fn: "vst1q_s32",
        store_fn_arg0: "int32_t",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "uint32x4x2_t",
        load_fn: "vld2q_u32",
        load_fn_arg0: "const uint32_t",
        store_fn: "vst2q_u32",
        store_fn_arg0: "uint32_t",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "uint32x4_t",
        load_fn: "vld1q_u32",
        load_fn_arg0: "const uint32_t",
        store_fn: "vst1q_u32",
        store_fn_arg0: "uint32_t",
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 8,
        name: "vsi8",
        native_type_name: "int16x4x2_t",
        load_fn: "vld2q_s16",
        load_fn_arg0: "const int16_t",
        store_fn: "vst2q_s16",
        store_fn_arg0: "int16_t",
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 4,
        name: "vsi4",
        native_type_name: "int16x4_t",
        load_fn: "vld1q_s16",
        load_fn_arg0: "const int16_t",
        store_fn: "vst1q_s16",
        store_fn_arg0: "int16_t",
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "uint16x4x2_t",
        load_fn: "vld2q_u16",
        load_fn_arg0: "const uint16_t",
        store_fn: "vst2q_u16",
        store_fn_arg0: "uint16_t",
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "uint16x4_t",
        load_fn: "vld1q_u16",
        load_fn_arg0: "const uint16_t",
        store_fn: "vst1q_u16",
        store_fn_arg0: "uint16_t",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 32,
        name: "vsb32",
        native_type_name: "int8x16x2_t",
        load_fn: "vld2q_s8",
        load_fn_arg0: "const int8_t",
        store_fn: "vst2q_s8",
        store_fn_arg0: "int8_t",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 16,
        name: "vsb16",
        native_type_name: "int8x16_t",
        load_fn: "vld1q_s8",
        load_fn_arg0: "const int8_t",
        store_fn: "vst1q_s8",
        store_fn_arg0: "int8_t",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 32,
        name: "vub32",
        native_type_name: "uint8x16x2_t",
        load_fn: "vld2q_u8",
        load_fn_arg0: "const uint8_t",
        store_fn: "vst2q_u8",
        store_fn_arg0: "uint8_t",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 16,
        name: "vub16",
        native_type_name: "uint8x16_t",
        load_fn: "vld1q_u8",
        load_fn_arg0: "const uint8_t",
        store_fn: "vst1q_u8",
        store_fn_arg0: "uint8_t",
    },
];

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct ArmTarget;

#[derive(Clone, Copy, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct ArmKernel(CpuKernel);

impl CpuTarget for ArmTarget {
    type Kernel = ArmKernel;

    fn target_id() -> TargetId {
        TargetId::Arm
    }

    fn vec_types() -> &'static [VecType; 16] {
        &ARM_VEC_TYPES
    }
}

impl Kernel for ArmKernel {
    type Tgt = ArmTarget;

    fn argument_count(&self) -> u8 {
        self.0.argument_count()
    }

    fn applies_to_logical_spec(&self, logical_spec: &LogicalSpec<Self::Tgt>) -> bool {
        self.0.applies_to_logical_spec(logical_spec)
    }

    fn memory_allocated<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MemoryAllocation {
        self.0.memory_allocated(parameters)
    }

    fn main_cost<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MainCost {
        self.0.main_cost(parameters)
    }

    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn into_cpu_kernel(self) -> Option<CpuKernel> {
        Some(self.0)
    }
}

impl From<CpuKernel> for ArmKernel {
    fn from(kernel: CpuKernel) -> Self {
        Self(kernel)
    }
}
