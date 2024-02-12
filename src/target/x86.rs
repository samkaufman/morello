use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{cpu::CpuTarget, TargetId};

use serde::Serialize;
use std::fmt::Debug;

const X86_VEC_TYPES: [VecType; 8] = [
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 8,
        name: "vsi8",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 4,
        name: "vsi4",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
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
        dtype: Dtype::Sint8,
        value_cnt: 32,
        name: "vsb32",
        native_type_name: "__m256i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 16,
        name: "vsb16",
        native_type_name: "__m128i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
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

impl CpuTarget for X86Target {
    fn target_id() -> TargetId {
        TargetId::X86
    }

    fn vec_types() -> &'static [VecType; 8] {
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
