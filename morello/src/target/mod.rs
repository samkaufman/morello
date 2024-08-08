mod arm;
pub(crate) mod common_actions; // TODO: Drop pub(crate) once MoveSolver doesn't need this.
pub(crate) mod cpu;
mod x86;

pub use arm::ArmTarget;
pub use cpu::{CpuKernel, CpuMemoryLevel, CpuTarget};
pub use x86::X86Target;

use crate::common::DimSize;
use crate::cost::MainCost;
use crate::layout::Layout;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::LogicalSpec;
use crate::views::View;
use crate::{codegen::c_utils::VecType, common::Dtype};

use serde::de::DeserializeOwned;
use serde::Serialize;

use std::fmt::{Debug, Display};
use std::hash::Hash;

// TODO: This should be generic per Target. Right now, all targets must have 4 levels!
pub const LEVEL_COUNT: usize = 4;

// TODO: Do we need so many trait bounds, here or in [CpuTarget]?
pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Level: MemoryLevel;
    type Kernel: Kernel<Tgt = Self>;
    type ActionsIter<'a>: Iterator<Item = Action<Self>> + 'a;

    fn line_size() -> u32;
    fn max_mem() -> MemoryLimits;
    fn processors() -> u8;
    fn default_level() -> Self::Level;
    fn levels() -> [Self::Level; LEVEL_COUNT];
    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level>;

    /// Returns possible layouts for a tensor of given shape and data type.
    fn all_layouts_for_shape(shape: &[DimSize], dtype: Dtype) -> Vec<Layout>;

    /// Returns destination layouts for a move from a given tensor shape and [Dtype].
    ///
    /// All returned layouts are applicable to the given shape
    /// ([Layout::applies_to_shape] returns `true` for each [Layout]).
    /// The returned layouts are a subset of those returned by [Self::all_layouts_for_shape].
    fn move_destination_layouts(shape: &[DimSize], dtype: Dtype) -> Vec<Layout>;

    /// Yield target-specific actions which apply to a given [LogicalSpec].
    fn actions(spec: &LogicalSpec<Self>) -> Self::ActionsIter<'_>;

    /// Get corresponding [TargetId] enum
    fn target_id() -> TargetId;

    /// Get corresponding vector types
    fn vec_types() -> &'static [VecType; 16];
}

pub trait MemoryLevel:
    Send + PartialOrd + Eq + Display + Debug + std::hash::Hash + Copy + DeserializeOwned + Serialize
{
    fn is_addressed(&self) -> bool;
    fn can_parallel_tile(&self) -> bool;
    fn cache_hit_cost(&self) -> MainCost;
    fn vector_bytes(&self) -> &'static [u32];
    fn vector_rf(&self) -> bool {
        !self.vector_bytes().is_empty()
    }
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
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum TargetId {
    X86,
    Arm,
}

impl Default for TargetId {
    fn default() -> Self {
        match std::env::consts::ARCH {
            "x86" | "x86_64" => TargetId::X86,
            "arm" | "aarch64" => TargetId::Arm,
            arch => unimplemented!("Architecture {} not supported", arch),
        }
    }
}
