use crate::common::DimSize;
use crate::cost::MainCost;
use crate::imp::kernels::KernelType;
use crate::layout::{nhwc, row_major, Layout};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};

use crate::tensorspec::TensorSpec;
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

    /// Yield target-specific actions which apply to a given [LogicalSpec].
    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>>;
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

    fn actions(spec: &LogicalSpec<Self>) -> Box<dyn Iterator<Item = Action<Self>>> {
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

pub fn valueassign_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
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

pub fn vectorassign_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
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

pub fn cacheaccess_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    return false;

    if operands.iter().all(|o| o.level().is_addressed()) {
        return false;
    }
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
    true
}

pub fn memsetzero_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    true
}

pub fn vectorzero_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
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

pub fn broadcastvecmult_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
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

pub fn mult_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    operands
        .iter()
        .all(|o| o.level() == X86MemoryLevel::RF && o.dim_sizes().iter().all(|&d| d == 1))
}
