use crate::codegen::c_utils::VecType;
use crate::common::Dtype;
use crate::target::{cpu::CpuTarget, TargetId};

use serde::Serialize;
use std::fmt::Debug;

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

impl CpuTarget for X86Target {
    fn target_id() -> TargetId {
        TargetId::X86
    }

    fn vec_types() -> &'static [VecType; 4] {
        &X86_VEC_TYPES
    }
}
