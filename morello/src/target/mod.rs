mod arm;
mod cpu;
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
use crate::tensorspec::TensorSpec;
use crate::views::Param;
use crate::{codegen::c_utils::VecType, common::Dtype};

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};

// TODO: This should be generic per Target. Right now, all targets must have 4 levels!
pub const LEVEL_COUNT: usize = 4;

// TODO: Do we need so many trait bounds, here or in [CpuTarget]?
pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Level: MemoryLevel;
    type Kernel: Kernel;

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
    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>> + '_>;

    /// Get corresponding [TargetId] enum
    fn target_id() -> TargetId;

    /// Get corresponding vector types
    fn vec_types() -> &'static [VecType; 16];
}

pub trait MemoryLevel:
    Send + PartialOrd + Eq + Display + Debug + std::hash::Hash + Copy + DeserializeOwned + Serialize
{
    fn is_addressed(&self) -> bool;
    fn cache_hit_cost(&self) -> MainCost;
    fn vector_bytes(&self) -> &'static [u32];
    fn vector_rf(&self) -> bool {
        !self.vector_bytes().is_empty()
    }
}

pub trait Kernel: PartialEq + Eq + Copy + Clone + Debug {
    fn argument_count(&self) -> u8;

    // TODO: Make into `applies_to_spec`
    fn applies_to_parameters<Tgt: CpuTarget>(&self, parameters: &[TensorSpec<Tgt>]) -> bool;

    // TODO: Take something more generic than Param.
    fn memory_allocated<Tgt: Target>(&self, parameters: &[Param<Tgt>]) -> MemoryAllocation;
    fn main_cost<Tgt: Target>(&self, parameters: &[Param<Tgt>]) -> MainCost;

    fn name(&self) -> &'static str;

    fn all_kernels() -> &'static [Self];
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum TargetId {
    X86,
    Arm,
}
