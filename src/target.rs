use crate::common::DimSize;
use crate::cost::MainCost;
use crate::imp::kernels::KernelType;
use crate::layout::{nhwc, row_major, Layout};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::{
    broadcastvecmult_applies_to_operands, memsetzero_applies_to_operands, mult_applies_to_operands,
    valueassign_applies_to_operands, vectorassign_applies_to_operands,
    vectorzero_applies_to_operands, Action,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter;

pub const MAX_LEVEL_COUNT: usize = 4;

pub trait Target: Clone + Copy + std::hash::Hash + Eq + Debug + 'static {
    type Level: MemoryLevel;

    fn line_size() -> u32;
    fn max_mem() -> MemoryLimits;
    fn processors() -> u8;
    fn default_level() -> Self::Level;
    fn levels() -> Vec<Self::Level>;
    fn faster_destination_levels(slower: Self::Level) -> Vec<Self::Level>;

    fn all_layouts_for_shape(shape: &[DimSize]) -> Vec<Layout>;

    /// Yield target-specific expansions of given Spec.
    fn expansions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>>;
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

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Serialize)]
pub struct X86Target;

impl Target for X86Target {
    type Level = X86MemoryLevel;

    fn line_size() -> u32 {
        32
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new(smallvec![64, 1024, 32_768, 1_073_741_824]))
    }

    fn processors() -> u8 {
        32
    }

    fn default_level() -> X86MemoryLevel {
        X86MemoryLevel::GL
    }

    fn levels() -> Vec<Self::Level> {
        enum_iterator::all::<Self::Level>().collect()
    }

    fn faster_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower {
            X86MemoryLevel::RF | X86MemoryLevel::VRF => vec![],
            X86MemoryLevel::L1 => vec![X86MemoryLevel::RF, X86MemoryLevel::VRF],
            X86MemoryLevel::GL => vec![X86MemoryLevel::L1],
        }
    }

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

    fn expansions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>> {
        match spec {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
                PrimitiveSpecType::Matmul { accum } => {
                    if *accum {
                        let mut microkernels = vec![];
                        if mult_applies_to_operands(&spec.parameters()) {
                            microkernels.push(Action::Place(KernelType::Mult));
                        }
                        if broadcastvecmult_applies_to_operands(&spec.parameters()) {
                            microkernels.push(Action::Place(KernelType::BroadcastVecMult));
                        }
                        Box::new(microkernels.into_iter())
                    } else {
                        Box::new(iter::empty())
                    }
                }
                PrimitiveSpecType::Conv { .. } => Box::new(iter::empty()),
                PrimitiveSpecType::Move { .. } => {
                    let mut microkernels = vec![];
                    if valueassign_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::ValueAssign));
                    }
                    if vectorassign_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::VectorAssign));
                    }
                    Box::new(microkernels.into_iter())
                }
                PrimitiveSpecType::Zero { .. } => {
                    let mut microkernels = vec![];
                    if memsetzero_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::MemsetZero));
                    }
                    if vectorzero_applies_to_operands(&spec.parameters()) {
                        microkernels.push(Action::Place(KernelType::VectorZero));
                    }
                    Box::new(microkernels.into_iter())
                }
            },
            LogicalSpec::Compose { .. } => Box::new(iter::empty()),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(
    Eq, PartialEq, Debug, Copy, Clone, Hash, Deserialize, Serialize, enum_iterator::Sequence,
)]
pub enum X86MemoryLevel {
    RF,
    VRF,
    L1,
    GL,
}

impl MemoryLevel for X86MemoryLevel {
    fn is_addressed(&self) -> bool {
        match &self {
            X86MemoryLevel::RF => true,
            X86MemoryLevel::VRF => true,
            X86MemoryLevel::L1 => false,
            X86MemoryLevel::GL => true,
        }
    }

    fn cache_hit_cost(&self) -> MainCost {
        match &self {
            X86MemoryLevel::RF => 0,
            X86MemoryLevel::VRF => 0,
            X86MemoryLevel::L1 => 10,
            X86MemoryLevel::GL => 100,
        }
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match &self {
            X86MemoryLevel::VRF => &[16, 32],
            _ => &[],
        }
    }
}

impl PartialOrd for X86MemoryLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (X86MemoryLevel::RF, X86MemoryLevel::VRF) => None,
            (X86MemoryLevel::VRF, X86MemoryLevel::RF) => None,
            (X86MemoryLevel::RF, _) => Some(Ordering::Less),
            (X86MemoryLevel::VRF, _) => Some(Ordering::Less),
            (_, X86MemoryLevel::RF) => Some(Ordering::Greater),
            (_, X86MemoryLevel::VRF) => Some(Ordering::Greater),
            (X86MemoryLevel::L1, X86MemoryLevel::GL) => Some(Ordering::Less),
            (X86MemoryLevel::GL, X86MemoryLevel::L1) => Some(Ordering::Greater),
            (X86MemoryLevel::L1, X86MemoryLevel::L1) => unreachable!(),
            (X86MemoryLevel::GL, X86MemoryLevel::GL) => unreachable!(),
        }
    }
}

impl Display for X86MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                X86MemoryLevel::RF => "RF",
                X86MemoryLevel::VRF => "VRF",
                X86MemoryLevel::L1 => "L1",
                X86MemoryLevel::GL => "GL",
            }
        )
    }
}