use super::{cpu::CpuTarget, CpuKernel, Kernel, TargetId};
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::memorylimits::{MemVec, MemoryAllocation, MemoryLimits};
use crate::spec::LogicalSpec;
use crate::target::CpuMemory;
use crate::{codegen::c_utils::VecType, views::View};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

super::x86::define_x86_vec_types!(X86_AVX2_VEC_TYPES, 16);

#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct Avx2Target;

#[derive(Clone, Copy, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Avx2Kernel(CpuKernel);

impl CpuTarget for Avx2Target {
    type Kernel = Avx2Kernel;
    type Memory = CpuMemory;

    fn target_id() -> TargetId {
        TargetId::Avx2
    }

    fn vec_types() -> &'static [VecType] {
        &X86_AVX2_VEC_TYPES
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new_mixed(
            [16, 16, 1_024, 33_554_432],
            [true, true, false, false],
        ))
    }
}

impl Kernel for Avx2Kernel {
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

impl From<CpuKernel> for Avx2Kernel {
    fn from(kernel: CpuKernel) -> Self {
        Self(kernel)
    }
}
