mod arm;
mod x86;

pub use arm::ArmTarget;
pub use x86::{X86MemoryLevel, X86Target};

use crate::codegen::c_utils::VecType;
use crate::common::DimSize;
use crate::cost::MainCost;
use crate::layout::{nhwc, row_major, Layout};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::LogicalSpec;
use crate::tensorspec::TensorSpec;

use clap::ValueEnum;
use serde::de::DeserializeOwned;
use serde::Serialize;
use smallvec::smallvec;
use std::fmt::{Debug, Display};
use std::iter;

pub const MAX_LEVEL_COUNT: usize = 4;

pub trait Target: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Level: MemoryLevel;

    fn line_size() -> u32 {
        32
    }
    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new(smallvec![64, 1024, 32_768, 1_073_741_824]))
    }
    fn processors() -> u8 {
        32
    }
    fn default_level() -> Self::Level;
    fn levels() -> Vec<Self::Level>;
    fn faster_destination_levels(slower: Self::Level) -> Vec<Self::Level>;

    fn all_layouts_for_shape(shape: &[DimSize]) -> Vec<Layout> {
        // warn!("NHWC and packed layouts are unimplemented.");

        let rm_iter = iter::once(row_major(shape.len().try_into().unwrap()));
        if shape.iter().all(|d| *d == 1) {
            rm_iter.collect()
        } else if shape.len() == 2 {
            rm_iter
                .chain(iter::once(Layout::Standard {
                    dim_order: smallvec![1, 0],
                }))
                .collect()
        } else if shape.len() == 4 {
            rm_iter.chain(iter::once(nhwc())).collect()
        } else {
            rm_iter.collect()
        }
    }

    /// Yield target-specific actions which apply to a given [LogicalSpec].
    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>>;

    /// Get corresponding [Targets] enum
    fn by_enum() -> Targets;

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

#[derive(Clone, ValueEnum)]
pub enum Targets {
    X86,
    Arm,
}

pub fn valueassign_applies_to_operands<Tgt: Target<Level = X86MemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    debug_assert_eq!(operands.len(), 2);

    if operands.iter().flat_map(|o| o.dim_sizes()).any(|&d| d != 1) {
        return false;
    }

    for o in &operands[1..] {
        if (o.dtype(), o.layout()) != (operands[0].dtype(), operands[0].layout()) {
            return false;
        }
    }

    operands.iter().any(|o| o.level() == X86MemoryLevel::RF)
        && operands
            .iter()
            .all(|o| o.level() == X86MemoryLevel::RF || o.level() == X86MemoryLevel::L1)
}

pub fn vectorassign_applies_to_operands<Tgt: Target>(operands: &[TensorSpec<Tgt>]) -> bool {
    if operands.iter().any(|o| !o.is_contiguous()) {
        return false;
    }
    if operands[0].dtype() != operands[1].dtype() {
        return false;
    }
    if operands[0].dim_sizes() != operands[1].dim_sizes() {
        return false;
    }
    if operands[0].layout() != operands[1].layout() {
        return false;
    }

    let mut has_vrf = false;
    for o in operands {
        if o.level().vector_rf() {
            has_vrf = true;
            match o.vector_size() {
                Some(vector_size) => {
                    let volume = o.dim_sizes().iter().product::<DimSize>();
                    if vector_size != volume {
                        return false;
                    }
                }
                None => {
                    panic!("No vector_size on operand in level {:?}", o.level());
                }
            }
        }
    }
    has_vrf
}

pub fn cacheaccess_applies_to_operands<Tgt: Target>(_operands: &[TensorSpec<Tgt>]) -> bool {
    return false;

    // if operands.iter().all(|o| o.level().is_addressed()) {
    //     return false;
    // }
    // if operands.iter().any(|o| !o.is_contiguous()) {
    //     return false;
    // }
    // if operands[0].dtype() != operands[1].dtype() {
    //     return false;
    // }
    // if operands[0].dim_sizes() != operands[1].dim_sizes() {
    //     return false;
    // }
    // if operands[0].layout() != operands[1].layout() {
    //     return false;
    // }
    // true
}

pub fn memsetzero_applies_to_operands<Tgt: Target<Level = X86MemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    true
}

pub fn vectorzero_applies_to_operands<Tgt: Target<Level = X86MemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::VRF {
        return false;
    }
    let volume = operands[0].dim_sizes().iter().product::<DimSize>();
    match operands[0].vector_size() {
        Some(vector_size) if vector_size != volume => {
            return false;
        }
        None => return false,
        _ => (),
    };
    true
}

pub fn broadcastvecmult_applies_to_operands<Tgt: Target<Level = X86MemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    for i in 1..3 {
        if operands[i].level() != X86MemoryLevel::VRF {
            return false;
        }
        let volume = operands[i].dim_sizes().iter().product::<DimSize>();
        if volume != operands[i].vector_size().unwrap() {
            return false;
        }
        if !operands[i].aligned() || !operands[i].is_contiguous() {
            return false;
        }
        if operands[0].dtype() != operands[i].dtype() {
            return false;
        }
    }
    if operands[0].dim_sizes().iter().any(|d| *d != 1) {
        return false;
    }
    if operands[1].dim_sizes().len() != 2 || operands[1].dim_sizes()[0] != 1 {
        return false;
    }
    if operands[2].dim_sizes().to_vec() != vec![1, operands[1].dim_sizes()[1]] {
        return false;
    }
    true
}

pub fn mult_applies_to_operands<Tgt: Target<Level = X86MemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    operands
        .iter()
        .all(|o| o.level() == X86MemoryLevel::RF && o.dim_sizes().iter().all(|&d| d == 1))
}
