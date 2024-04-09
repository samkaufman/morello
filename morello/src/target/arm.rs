use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{cpu::CpuTarget, TargetId};

use serde::Serialize;
use std::fmt::Debug;

const ARM_VEC_TYPES: [VecType; 12] = [
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

impl CpuTarget for ArmTarget {
    fn target_id() -> TargetId {
        TargetId::Arm
    }

    fn vec_types() -> &'static [VecType; 12] {
        &ARM_VEC_TYPES
    }
}

#[cfg(test)]
mod tests {
    use super::ArmTarget;
    use crate::target::Target;

    #[test]
    fn test_arm_levels_equals_all_enum_cases() {
        let enum_levels = enum_iterator::all::<<ArmTarget as Target>::Level>().collect::<Vec<_>>();
        let listed_levels = ArmTarget::levels();
        assert_eq!(&enum_levels, &listed_levels);
    }
}
