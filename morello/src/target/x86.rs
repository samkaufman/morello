use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{cpu::CpuTarget, TargetId};

use serde::Serialize;
use std::fmt::Debug;

const X86_VEC_TYPES: [VecType; 16] = [
    VecType {
        dtype: Dtype::Bfloat16,
        value_cnt: 16,
        name: "vbf16_16",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Bfloat16,
        value_cnt: 8,
        name: "vbf16_8",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 8,
        name: "vf8",
        native_type_name: "__m256",
        load_fn: "_mm256_loadu_ps",
        load_fn_arg0: "float const",
        store_fn: "_mm256_storeu_ps",
        store_fn_arg0: "float",
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 4,
        name: "vf4",
        native_type_name: "__m128",
        load_fn: "_mm_loadu_ps",
        load_fn_arg0: "float",
        store_fn: "_mm_storeu_ps",
        store_fn_arg0: "float",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 8,
        name: "vsi8",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 4,
        name: "vsi4",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 16,
        name: "vsi16",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 8,
        name: "vsi8",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 16,
        name: "vui16",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 32,
        name: "vsb32",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 16,
        name: "vsb16",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 32,
        name: "vub32",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        load_fn_arg0: "__m256i",
        store_fn: "_mm256_storeu_si256",
        store_fn_arg0: "__m256i",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 16,
        name: "vub16",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        load_fn_arg0: "__m128i",
        store_fn: "_mm_storeu_si128",
        store_fn_arg0: "__m128i",
    },
];

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct X86Target;

impl CpuTarget for X86Target {
    fn target_id() -> TargetId {
        TargetId::X86
    }

    fn vec_types() -> &'static [VecType; 16] {
        &X86_VEC_TYPES
    }
}

#[cfg(test)]
mod tests {
    use super::X86Target;
    use crate::target::Target;

    #[test]
    fn test_x86_levels_equals_all_enum_cases() {
        let enum_levels = enum_iterator::all::<<X86Target as Target>::Level>().collect::<Vec<_>>();
        let listed_levels = X86Target::levels();
        assert_eq!(&enum_levels, &listed_levels);
    }
}
