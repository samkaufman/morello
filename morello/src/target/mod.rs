mod arm;
mod cpu;
mod x86;

pub use arm::ArmTarget;
pub use cpu::CpuMemoryLevel;
pub use x86::X86Target;

use crate::common::DimSize;
use crate::cost::MainCost;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::Action;
use crate::spec::LogicalSpec;
use crate::{codegen::c_utils::VecType, common::Dtype};

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};

// TODO: This should be generic per Target. Right now, all targets must have 4 levels!
pub const LEVEL_COUNT: usize = 4;

// TODO: Do we need so many trait bounds, here or in [CpuTarget]?
pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Level: MemoryLevel;

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
    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>>;

    /// Get corresponding [TargetId] enum
    fn target_id() -> TargetId;

    /// Get corresponding vector types
    fn vec_types() -> &'static [VecType; 12];
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

#[derive(Clone, Copy)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum TargetId {
    X86,
    Arm,
}
