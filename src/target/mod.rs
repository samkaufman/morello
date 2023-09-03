mod arm;
mod cpu;
mod x86;

pub use arm::ArmTarget;
pub use cpu::CpuMemoryLevel;
pub use x86::X86Target;

use crate::codegen::c_utils::VecType;
use crate::common::DimSize;
use crate::cost::MainCost;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::Action;
use crate::spec::LogicalSpec;

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::{Debug, Display};

pub const MAX_LEVEL_COUNT: usize = 4;

// TODO: Do we need so many trait bounds, here or in [CpuTarget]?
pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Level: MemoryLevel;

    fn line_size() -> u32;
    fn max_mem() -> MemoryLimits;
    fn processors() -> u8;
    fn default_level() -> Self::Level;
    fn levels() -> Vec<Self::Level>;
    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level>;

    // TODO: Canonicalization should unify the following two fns:
    fn all_layouts_for_shape(shape: &[DimSize]) -> Vec<Layout>;
    fn move_destination_layouts(shape: &[DimSize]) -> Vec<Layout>;

    /// Yield target-specific actions which apply to a given [LogicalSpec].
    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>>;

    /// Get corresponding [TargetId] enum
    fn target_id() -> TargetId;

    /// Get corresponding vector types
    fn vec_types() -> &'static [VecType; 4];
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
pub enum TargetId {
    X86,
    Arm,
}
