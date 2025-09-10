use super::{cpu::CpuTarget, CpuKernel, Kernel, TargetId};
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::memorylimits::MemoryAllocation;
use crate::spec::LogicalSpec;
use crate::{codegen::c_utils::VecType, views::View};

use serde::{Deserialize, Serialize};
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
pub struct Avx2Target;

#[derive(Clone, Copy, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct X86Kernel(CpuKernel);

impl CpuTarget for Avx2Target {
    type Kernel = X86Kernel;

    fn target_id() -> TargetId {
        TargetId::Avx2
    }

    fn vec_types() -> &'static [VecType; 16] {
        &X86_VEC_TYPES
    }
}

impl Kernel for X86Kernel {
    type Tgt = Avx2Target;

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

impl From<CpuKernel> for X86Kernel {
    fn from(kernel: CpuKernel) -> Self {
        Self(kernel)
    }
}
