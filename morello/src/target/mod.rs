mod arm;
mod avx2;
mod avx512;
pub(crate) mod common_actions;
pub(crate) mod cpu;
mod x86;

pub use arm::ArmTarget;
pub use avx2::Avx2Target;
pub use avx512::{Avx512Kernel, Avx512Target};
pub use cpu::{CpuKernel, CpuMemory, CpuSolver, CpuTarget};

use crate::common::DimSize;
use crate::cost::MainCost;
use crate::layout::Layout;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::Action;
use crate::search::ImplReducer;
use crate::spatial_action_solver::SpatialSolver;
use crate::spec::{LogicalSpec, Spec};
use crate::views::View;
use crate::{codegen::c_utils::VecType, common::Dtype};

use serde::de::DeserializeOwned;
use serde::Serialize;

use std::fmt::{Debug, Display};
use std::hash::Hash;

// TODO: This should be generic per Target. Right now, all targets must have 4 (or 3) memories!
#[cfg(not(feature = "drop-rf"))]
pub const MEMORY_COUNT: usize = 4;
#[cfg(feature = "drop-rf")]
pub const MEMORY_COUNT: usize = 3;

// TODO: Do we need so many trait bounds, here or in [CpuTarget]?
pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Memory: Memory;
    type Kernel: Kernel<Tgt = Self>;
    type ActionsIter<'a>: Iterator<Item = Action<Self>> + 'a;
    type SpatialSolver: SpatialSolver<Self>;

    // TODO: Rename fn and return type once there's only one kind of solver.
    fn spatial_solvers<'s>(
        spec: &'s Spec<Self>,
        reducer: &mut ImplReducer,
    ) -> impl Iterator<Item = Self::SpatialSolver> + 's;

    fn line_size() -> u32;
    fn cache_hit_cost(memory: Self::Memory) -> MainCost;
    fn max_mem() -> MemoryLimits;
    fn processors() -> u8;
    fn default_memory() -> Self::Memory;
    fn memories() -> [Self::Memory; MEMORY_COUNT];
    fn possible_destination_memories(slower: Self::Memory) -> Vec<Self::Memory>;

    /// Returns possible layouts for a tensor of given shape and data type.
    fn all_layouts_for_shape(shape: &[DimSize], dtype: Dtype) -> Vec<Layout>;

    /// Returns destination layouts for a move from a given tensor shape and [Dtype].
    ///
    /// All returned layouts are applicable to the given shape
    /// ([Layout::applies_to_shape] returns `true` for each [Layout]).
    /// The returned layouts are a non-strict subset of those returned by
    /// [Self::all_layouts_for_shape].
    fn move_destination_layouts(shape: &[DimSize], dtype: Dtype) -> Vec<Layout> {
        Self::all_layouts_for_shape(shape, dtype)
    }

    /// Yield target-specific actions which apply to a given [LogicalSpec].
    fn actions(spec: &LogicalSpec<Self>) -> Self::ActionsIter<'_>;

    /// Get corresponding [TargetId] enum
    fn target_id() -> TargetId;

    /// Get corresponding vector types
    fn vec_types() -> &'static [VecType];

    /// Returns the widest vector size available for `dtype` that evenly divides `volume`, or `None`
    /// if no vector type for `dtype` divides it.
    fn vector_size_for(dtype: Dtype, volume: DimSize) -> Option<DimSize> {
        Self::vec_types()
            .iter()
            .filter(|vec_type| vec_type.dtype == dtype)
            .map(|vec_type| u32::from(vec_type.value_cnt))
            .filter(|&value_cnt| volume.get().is_multiple_of(value_cnt))
            .max()
            .map(|value_cnt| DimSize::new(value_cnt).unwrap())
    }
}

pub trait Memory:
    Send + PartialOrd + Eq + Display + Debug + std::hash::Hash + Copy + DeserializeOwned + Serialize
{
    fn is_addressed(&self) -> bool;
    fn can_parallel_tile(&self) -> bool;
    fn vector_rf(&self) -> bool;
    fn counts_registers(&self) -> bool;
    /// Returns true if this memory has (meaningful) layouts associated with it.
    /// Some memories (e.g., RF) have no meaningful relationship beween values.
    fn has_layout(&self) -> bool;
}

pub trait Kernel: PartialEq + Eq + Copy + Clone + Hash + Debug {
    type Tgt: Target;

    fn argument_count(&self) -> u8;

    fn applies_to_logical_spec(&self, logical_spec: &LogicalSpec<Self::Tgt>) -> bool;

    fn memory_allocated<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MemoryAllocation;

    fn main_cost<P: View<Tgt = Self::Tgt>>(&self, parameters: &[P]) -> MainCost;

    fn name(&self) -> &'static str;

    // TODO: Remove after composing kernel code generators.
    fn into_cpu_kernel(self) -> Option<CpuKernel> {
        None
    }

    // TODO: Remove after composing kernel code generators.
    fn into_avx512_kernel(self) -> Option<Avx512Kernel> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum TargetId {
    Avx2,
    Avx512,
    Arm,
}

impl Default for TargetId {
    fn default() -> Self {
        match std::env::consts::ARCH {
            "x86" | "x86_64" => TargetId::Avx2,
            "arm" | "aarch64" => TargetId::Arm,
            arch => unimplemented!("Architecture {} not supported", arch),
        }
    }
}

impl Display for TargetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetId::Avx2 => write!(f, "Avx2"),
            TargetId::Avx512 => write!(f, "Avx512"),
            TargetId::Arm => write!(f, "Arm"),
        }
    }
}

impl std::str::FromStr for TargetId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Avx2" => Ok(TargetId::Avx2),
            "Avx512" => Ok(TargetId::Avx512),
            "Arm" => Ok(TargetId::Arm),
            _ => Err(format!("Unknown target: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;
    use strum::VariantArray;

    /// Asserts that no two [VecType]s for a target share a `name`.
    ///
    /// Each distinct [VecType] a target emits becomes its own C typedef, so two sharing a `name`
    /// would emit conflicting typedefs for the same identifier.
    fn assert_vector_type_names_unique<Tgt: Target>() {
        let mut by_name = HashMap::new();
        for vec_type in Tgt::vec_types() {
            if let Some(previous) = by_name.insert(vec_type.name, vec_type) {
                panic!(
                    "{:?} gives the name {} to both {:?}x{} and {:?}x{}",
                    Tgt::target_id(),
                    vec_type.name,
                    previous.dtype,
                    previous.value_cnt,
                    vec_type.dtype,
                    vec_type.value_cnt,
                );
            }
        }
    }

    #[test]
    fn test_vector_type_names_are_unique() {
        assert_vector_type_names_unique::<ArmTarget>();
        assert_vector_type_names_unique::<Avx2Target>();
        assert_vector_type_names_unique::<Avx512Target>();
    }

    #[test]
    fn test_widest_vector_sizes_match_target_widths() {
        assert_widest_vector_bytes::<ArmTarget>(32);
        assert_widest_vector_bytes::<Avx2Target>(32);
        assert_widest_vector_bytes::<Avx512Target>(64);
    }

    #[test]
    fn test_vector_size_for_picks_widest_divisor_of_volume() {
        let size_for = |dtype, volume: u32| {
            Avx512Target::vector_size_for(dtype, volume.try_into().unwrap()).map(|v| v.get())
        };
        assert_eq!(size_for(Dtype::Float32, 64), Some(16));
        assert_eq!(size_for(Dtype::Float32, 24), Some(8));
        assert_eq!(size_for(Dtype::Float32, 12), Some(4));
        assert_eq!(size_for(Dtype::Float32, 6), None);
        assert_eq!(size_for(Dtype::Uint8, 48), Some(16));
        assert_eq!(
            Avx2Target::vector_size_for(Dtype::Float32, 16.try_into().unwrap()),
            Some(8.try_into().unwrap())
        );
    }

    proptest! {
        #[test]
        fn test_target_id_display_fromstr_roundtrip(target_id: TargetId) {
            let display_str = target_id.to_string();
            let parsed = display_str.parse::<TargetId>().unwrap();
            prop_assert_eq!(parsed, target_id);
        }
    }

    fn assert_widest_vector_bytes<Tgt: Target>(expected_bytes: u32) {
        for &dtype in Dtype::VARIANTS {
            let widest = Tgt::vec_types()
                .iter()
                .filter(|vec_type| vec_type.dtype == dtype)
                .map(|vec_type| u32::from(vec_type.value_cnt))
                .max()
                .unwrap_or_else(|| panic!("Target has no vector type for {dtype:?}"));
            assert_eq!(
                widest * u32::from(dtype.size()),
                expected_bytes,
                "unexpected widest vector width for {dtype:?}"
            );
        }
    }
}
