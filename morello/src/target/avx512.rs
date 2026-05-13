use super::cpu::{vector_accum_count, CpuMemoryBimap, VECTOR_ACCUM_COUNT};
use super::{cpu::CpuTarget, CpuKernel, Kernel, TargetId};
use crate::common::Dtype;
use crate::cost::MainCost;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::memorylimits::{MemVec, MemoryAllocation, MemoryLimits};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use crate::target::{CpuMemory, Memory};
use crate::{codegen::c_utils::VecType, views::View};

use nonzero::nonzero as nz;
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
    },
    VecType {
        dtype: Dtype::Float32,
        value_cnt: 16,
        name: "vf16",
        native_type_name: "__m512",
    },
    VecType {
        dtype: Dtype::Sint32,
        value_cnt: 16,
        name: "vsi16",
        native_type_name: "__m512i",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 16,
        name: "vui16",
        native_type_name: "__m512i",
    },
    VecType {
        dtype: Dtype::Sint16,
        value_cnt: 32,
        name: "vsi32",
        native_type_name: "__m512i",
    },
    VecType {
        dtype: Dtype::Uint16,
        value_cnt: 32,
        name: "vui32",
        native_type_name: "__m512i",
    },
    VecType {
        dtype: Dtype::Sint8,
        value_cnt: 64,
        name: "vsb64",
        native_type_name: "__m512i",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 64,
        name: "vub64",
        native_type_name: "__m512i",
    },
);

/// x86 AVX-512 CPU target with AVX512F, AVX512BW, AVX512_BF16, and FMA.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Default, Debug, Serialize)]
pub struct Avx512Target;

#[derive(Clone, Copy, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum Avx512Kernel {
    Cpu(CpuKernel),
    DotProductLoopVdpbf16ps,
}

#[derive(PartialEq, Eq, PartialOrd, Debug, Copy, Clone, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Avx512Memory(pub CpuMemory);

impl CpuTarget for Avx512Target {
    type Kernel = Avx512Kernel;
    type Memory = Avx512Memory;

    fn target_id() -> TargetId {
        TargetId::Avx512
    }

    fn vec_types() -> &'static [VecType] {
        &X86_AVX512_VEC_TYPES
    }

    fn target_specific_kernels(spec: &LogicalSpec<Self>) -> Vec<Self::Kernel> {
        if matches!(
            spec,
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: true },
                    ..
                },
                _,
                _,
            )
        ) {
            vec![Avx512Kernel::DotProductLoopVdpbf16ps]
        } else {
            Vec::new()
        }
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
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => cpu_kernel.argument_count(),
            Avx512Kernel::DotProductLoopVdpbf16ps => 3,
        }
    }

    fn applies_to_logical_spec(&self, logical_spec: &LogicalSpec<Self::Tgt>) -> bool {
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => cpu_kernel.applies_to_logical_spec(logical_spec),
            Avx512Kernel::DotProductLoopVdpbf16ps => {
                vdpbf16ps_applies_to_logical_spec(logical_spec)
            }
        }
    }

    fn memory_allocated<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MemoryAllocation {
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => cpu_kernel.memory_allocated(parameters),
            Avx512Kernel::DotProductLoopVdpbf16ps => {
                let strip = 32;
                let accum_count = parameters
                    .first()
                    .and_then(|parameter| parameter.spec().shape().get(2).copied())
                    .and_then(|k| vector_accum_count(k.get(), strip))
                    .unwrap_or(VECTOR_ACCUM_COUNT);
                MemoryAllocation::Simple([0, u64::from(accum_count) + 2, 0, 0])
            }
        }
    }

    fn main_cost<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MainCost {
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => cpu_kernel.main_cost(parameters),
            Avx512Kernel::DotProductLoopVdpbf16ps => {
                let Some(lhs_spec) = parameters.first().map(|parameter| parameter.spec()) else {
                    return 0;
                };
                let Some(k) = lhs_spec.shape().get(2).map(|d| d.get()) else {
                    return 0;
                };
                let instr_count = k / 32;
                let accum_count = vector_accum_count(k, 32).unwrap_or(VECTOR_ACCUM_COUNT);

                // VDPBF16PS ZMM,..,M512 has RThroughput of 0.50 cycles on Zen 5
                const REDUCTION_COST: MainCost = 4;
                instr_count + REDUCTION_COST * accum_count
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => cpu_kernel.name(),
            Avx512Kernel::DotProductLoopVdpbf16ps => "DotProductLoopVdpbf16ps",
        }
    }

    fn into_cpu_kernel(self) -> Option<CpuKernel> {
        match self {
            Avx512Kernel::Cpu(cpu_kernel) => Some(cpu_kernel),
            Avx512Kernel::DotProductLoopVdpbf16ps => None,
        }
    }

    fn into_avx512_kernel(self) -> Option<Avx512Kernel> {
        Some(self)
    }
}

impl From<CpuKernel> for Avx512Kernel {
    fn from(kernel: CpuKernel) -> Self {
        Self::Cpu(kernel)
    }
}

fn vdpbf16ps_applies_to_logical_spec(logical_spec: &LogicalSpec<Avx512Target>) -> bool {
    let LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: true },
            ..
        },
        _,
        _,
    ) = logical_spec
    else {
        return false;
    };

    let operands = logical_spec.parameters();
    let [lhs, rhs, out] = &operands[..] else {
        return false;
    };
    if lhs.dtype() != Dtype::Bfloat16
        || rhs.dtype() != Dtype::Bfloat16
        || out.dtype() != Dtype::Float32
    {
        return false;
    }

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = out.shape();
    if lhs_shape.len() != 3 || rhs_shape.len() != 3 || out_shape.len() != 3 {
        return false;
    }
    if lhs_shape[0] != nz!(1u32)
        || lhs_shape[1] != nz!(1u32)
        || rhs_shape[0] != nz!(1u32)
        || rhs_shape[1] != lhs_shape[2]
        || rhs_shape[2] != nz!(1u32)
        || out_shape.iter().any(|d| d.get() != 1)
    {
        return false;
    }

    if vector_accum_count(lhs_shape[2].get(), 32).is_none() {
        return false;
    }

    let Ok(expected_rhs_layout) = crate::layout![0, 2, 1].canonicalize(rhs_shape) else {
        return false;
    };
    lhs.layout().is_row_major()
        && rhs.layout() == &expected_rhs_layout
        && lhs.is_contiguous()
        && rhs.is_contiguous()
        && lhs.memory() == CpuMemory::L1
        && rhs.memory() == CpuMemory::L1
        && out.memory() == CpuMemory::RF
}

impl Memory for Avx512Memory {
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
            CpuMemory::VRF => {
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

impl PartialEq<CpuMemory> for Avx512Memory {
    fn eq(&self, other: &CpuMemory) -> bool {
        self.0 == *other
    }
}

impl Display for Avx512Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<CpuMemory> for Avx512Memory {
    fn from(memory: CpuMemory) -> Self {
        Self(memory)
    }
}

impl From<Avx512Memory> for CpuMemory {
    fn from(val: Avx512Memory) -> Self {
        val.0
    }
}

pub struct Avx512MemoryBimap;

impl BiMap for Avx512MemoryBimap {
    type Domain = Avx512Memory;
    type Codomain = u8;

    fn apply(&self, memory: &Avx512Memory) -> u8 {
        <CpuMemoryBimap as BiMap>::apply(&CpuMemoryBimap, &memory.0)
    }

    fn apply_inverse(&self, i: &u8) -> Avx512Memory {
        Avx512Memory(<CpuMemoryBimap as BiMap>::apply_inverse(&CpuMemoryBimap, i))
    }
}

impl CanonicalBimap for Avx512Memory {
    type Bimap = Avx512MemoryBimap;

    fn bimap() -> Self::Bimap {
        Avx512MemoryBimap
    }
}

#[cfg(test)]
mod tests {
    use super::{Avx512Kernel, Avx512Target};
    use crate::codegen::CodeGen;
    use crate::layout::row_major;
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::spec::{LogicalSpec, Spec};
    use crate::target::CpuMemory::{L1, RF};
    use crate::target::{Kernel, Target};
    use crate::{layout, lspec};

    #[test]
    fn test_vdpbf16ps_kernel_applies_to_bf16_dot_product_tile() {
        let mut logical_spec: LogicalSpec<Avx512Target> = lspec!(MatmulAccum(
            [1, 1, 128, 1],
            (bf16, L1, row_major),
            (bf16, L1, layout![0, 2, 1]),
            (f32, RF, row_major),
            serial
        ));
        logical_spec.canonicalize().unwrap();
        assert!(Avx512Kernel::DotProductLoopVdpbf16ps.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_vdpbf16ps_kernel_rejects_short_k() {
        let mut logical_spec: LogicalSpec<Avx512Target> = lspec!(MatmulAccum(
            [1, 1, 16, 1],
            (bf16, L1, row_major),
            (bf16, L1, layout![0, 2, 1]),
            (f32, RF, row_major),
            serial
        ));
        logical_spec.canonicalize().unwrap();
        assert!(!Avx512Kernel::DotProductLoopVdpbf16ps.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_vdpbf16ps_kernel_codegen_emits_intrinsic() {
        let mut logical_spec: LogicalSpec<Avx512Target> = lspec!(MatmulAccum(
            [1, 1, 128, 1],
            (bf16, L1, row_major),
            (bf16, L1, layout![0, 2, 1]),
            (f32, RF, row_major),
            serial
        ));
        logical_spec.canonicalize().unwrap();
        let spec = Spec(logical_spec, Avx512Target::max_mem());
        let implementation = spec.select(Avx512Kernel::DotProductLoopVdpbf16ps);

        let mut c = String::new();
        implementation.emit(true, None, &mut c).unwrap();
        assert!(c.contains("_mm512_dpbf16_ps"));
    }
}
