use super::cpu::CpuMemoryLevelBimap;
use super::{cpu::CpuTarget, CpuKernel, Kernel, TargetId};
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::memorylimits::{MemVec, MemoryAllocation, MemoryLimits};
use crate::spec::LogicalSpec;
use crate::target::{CpuMemoryLevel, MemoryLevel};
use crate::{codegen::c_utils::VecType, views::View};

use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

crate::target::x86::define_x86_vec_types!(
    X86_AVX512_VEC_TYPES,
    24,
    VecType {
        dtype: Dtype::Bfloat16,
        value_cnt: 32,
        name: "vbf16_32",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 16,
        name: "vf16",
        native_type_name: "__m512",
        load_fn: "_mm512_loadu_ps",
        load_fn_arg0: "float const",
        store_fn: "_mm512_storeu_ps",
        store_fn_arg0: "float"
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 16,
        name: "vsi16",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 16,
        name: "vui16",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 32,
        name: "vsi32",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 32,
        name: "vui32",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 64,
        name: "vsb64",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 64,
        name: "vub64",
        native_type_name: "__m512i",
        load_fn: "_mm512_loadu_si512",
        load_fn_arg0: "__m512i",
        store_fn: "_mm512_storeu_si512",
        store_fn_arg0: "__m512i"
    },
);

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct Avx512Target;

#[derive(Clone, Copy, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Avx512Kernel(CpuKernel);

#[derive(PartialEq, Eq, PartialOrd, Debug, Copy, Clone, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Avx512MemoryLevel(pub CpuMemoryLevel);

impl CpuTarget for Avx512Target {
    type Kernel = Avx512Kernel;
    type Level = Avx512MemoryLevel;

    fn target_id() -> TargetId {
        TargetId::Avx512
    }

    fn vec_types() -> &'static [VecType] {
        &X86_AVX512_VEC_TYPES
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new_mixed(
            [16, 32, 1_024, 33_554_432],
            [true, true, false, false],
        ))
    }
}

impl Kernel for Avx512Kernel {
    type Tgt = Avx512Target;

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

impl From<CpuKernel> for Avx512Kernel {
    fn from(kernel: CpuKernel) -> Self {
        Self(kernel)
    }
}

impl MemoryLevel for Avx512MemoryLevel {
    fn is_addressed(&self) -> bool {
        self.0.is_addressed()
    }

    fn can_parallel_tile(&self) -> bool {
        self.0.can_parallel_tile()
    }

    fn cache_hit_cost(&self) -> MainCost {
        self.0.cache_hit_cost()
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match self.0 {
            CpuMemoryLevel::VRF => {
                debug_assert_eq!(self.0.vector_bytes(), &[16, 32]);
                &[16, 32, 64]
            }
            _ => {
                debug_assert_eq!(self.0.vector_bytes(), &[]);
                &[]
            }
        }
    }

    fn counts_registers(&self) -> bool {
        self.0.counts_registers()
    }

    fn has_layout(&self) -> bool {
        self.0.has_layout()
    }

    fn vector_rf(&self) -> bool {
        self.0.vector_rf()
    }
}

impl PartialEq<CpuMemoryLevel> for Avx512MemoryLevel {
    fn eq(&self, other: &CpuMemoryLevel) -> bool {
        self.0 == *other
    }
}

impl Display for Avx512MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CpuMemoryLevel> for Avx512MemoryLevel {
    fn from(level: CpuMemoryLevel) -> Self {
        Self(level)
    }
}

impl From<Avx512MemoryLevel> for CpuMemoryLevel {
    fn from(val: Avx512MemoryLevel) -> Self {
        val.0
    }
}

pub struct Avx512MemoryLevelBimap;

impl BiMap for Avx512MemoryLevelBimap {
    type Domain = Avx512MemoryLevel;
    type Codomain = u8;

    fn apply(&self, level: &Avx512MemoryLevel) -> u8 {
        <CpuMemoryLevelBimap as BiMap>::apply(&CpuMemoryLevelBimap, &level.0)
    }

    fn apply_inverse(&self, i: &u8) -> Avx512MemoryLevel {
        Avx512MemoryLevel(<CpuMemoryLevelBimap as BiMap>::apply_inverse(
            &CpuMemoryLevelBimap,
            i,
        ))
    }
}

impl CanonicalBimap for Avx512MemoryLevel {
    type Bimap = Avx512MemoryLevelBimap;

    fn bimap() -> Self::Bimap {
        Avx512MemoryLevelBimap
    }
}
