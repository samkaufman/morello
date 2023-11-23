use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::utils::{bit_length, bit_length_inverse};
use crate::{
    target::{Target, LEVEL_COUNT},
    utils::prev_power_of_two,
};

use itertools::Itertools;
use log::warn;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::{iter, ops::Sub};

/// MemoryLimits are bounds on available memory for each level of a target.
///
/// There are two variants. `MemoryLimits::Standard` simply counts the number of bytes
/// remaining at each level, and no Spec satisfying the MemoryLimits should have a peak
/// exceeding those. `MemoryLimits::Pipeline` counts separately the memory used by
/// intermediate tensors just before or after a stage. This expands the set of
/// `ImplNode::Pipeline`s which might satisfy a Spec, because it is valid for
/// that `Pipeline` to assume that those bytes have been freed after its own
/// first and last stages complete.
///
/// By convention, MemoryLimits are always discretized to powers of two. It is
/// responsibility of the constructor to call `discretize`.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub enum MemoryLimits {
    // TODO: Implement Pipeline as described above.
    Standard(MemVec),
}

/// The memory allocated by a single [Impl] node.
///
/// These are *not* snapped to zero or powers of two.
///
/// Put another way: this is a description of the memory live during execution of a single node,
/// ignoring children.
pub enum MemoryAllocation {
    Simple([u64; LEVEL_COUNT]),
    Inner(Vec<[u64; LEVEL_COUNT]>),
    Pipeline {
        intermediate_consumption: Vec<[u64; LEVEL_COUNT]>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct MemVec([u8; LEVEL_COUNT]);

#[derive(Default)]
pub struct MemoryLimitsBimap<Tgt: Target> {
    phantom: std::marker::PhantomData<Tgt>,
}

impl MemoryLimits {
    /// Convert to a `MemoryLimits::Standard`.
    ///
    /// This is always safe, but conservative.
    pub fn into_standard(self) -> Self {
        match self {
            MemoryLimits::Standard(_) => self,
        }
    }

    pub fn discretize(&mut self) {
        match self {
            MemoryLimits::Standard(mem_vec) => {
                for i in 0..mem_vec.len() {
                    mem_vec.set_unscaled(i, prev_power_of_two(mem_vec.get_unscaled(i)));
                }
            }
        }
    }

    /// Produce new MemoryLimits for each child of a node after some allocation.
    /// Returns `None` if the given memory allocation exceeds this limit.
    ///
    /// Not that this ignores base memory allocations at the leaves. It is intended to
    /// be used to prune actions which consume too much memory without traversing.
    pub fn transition<Tgt: Target>(
        &self,
        allocated: &MemoryAllocation,
    ) -> Option<Vec<MemoryLimits>> {
        warn!("Not transitioning to pipeline MemoryLimits yet");

        let per_child_diffs: Box<dyn Iterator<Item = _>> = match allocated {
            MemoryAllocation::Simple(v) => Box::new(iter::repeat(v)),
            MemoryAllocation::Inner(during_children) => Box::new(during_children.iter()),
            MemoryAllocation::Pipeline {
                intermediate_consumption: _,
            } => todo!(),
        };

        match self {
            MemoryLimits::Standard(cur_limit) => {
                let mut result = Vec::with_capacity(cur_limit.len());
                for child_allocation in per_child_diffs {
                    debug_assert_eq!(child_allocation.len(), cur_limit.len());
                    let Some(to_push) = cur_limit.clone().checked_sub_snap_down(child_allocation)
                    else {
                        return None;
                    };
                    let mut to_push = MemoryLimits::Standard(to_push);
                    to_push.discretize();
                    result.push(to_push);
                }
                Some(result)
            }
        }
    }
}

impl PartialOrd for MemoryLimits {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MemoryLimits::Standard(limits_vec), MemoryLimits::Standard(other_limits_vec)) => {
                limits_vec.partial_cmp(other_limits_vec)
            }
        }
    }

    fn ge(&self, other: &Self) -> bool {
        match (self, other) {
            (MemoryLimits::Standard(limits_vec), MemoryLimits::Standard(other_limits_vec)) => {
                limits_vec.ge(other_limits_vec)
            }
        }
    }

    fn le(&self, other: &Self) -> bool {
        match (self, other) {
            (MemoryLimits::Standard(limits_vec), MemoryLimits::Standard(other_limits_vec)) => {
                limits_vec.le(other_limits_vec)
            }
        }
    }
}

impl Display for MemoryLimits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLimits::Standard(mem_vec) => mem_vec.fmt(f),
        }
    }
}
impl MemoryAllocation {
    pub fn none<Tgt: Target>() -> Self {
        MemoryAllocation::Simple([0; LEVEL_COUNT])
    }
}

impl MemVec {
    pub fn new(contents: [u64; LEVEL_COUNT]) -> Self {
        MemVec::new_from_binary_scaled(contents.map(|v| bit_length(v).try_into().unwrap()))
    }

    pub fn new_from_binary_scaled(contents: [u8; LEVEL_COUNT]) -> Self {
        MemVec(contents)
    }

    pub fn zero<Tgt: Target>() -> Self {
        debug_assert_eq!(Tgt::levels().len(), LEVEL_COUNT);
        MemVec([0; LEVEL_COUNT])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get_binary_scaled(&self, idx: usize) -> u8 {
        self.0[idx]
    }

    pub fn get_unscaled(&self, idx: usize) -> u64 {
        bit_length_inverse(self.0[idx].into())
    }

    pub fn set_unscaled(&mut self, idx: usize, value: u64) {
        self.0[idx] = bit_length(value).try_into().unwrap();
    }

    pub fn checked_sub_snap_down(self, rhs: &[u64; LEVEL_COUNT]) -> Option<MemVec> {
        let mut result = self;
        for idx in 0..result.len() {
            let cur = bit_length_inverse(result.0[idx].into());
            if cur < rhs[idx] {
                return None;
            }
            result.0[idx] = bit_length(prev_power_of_two(cur - rhs[idx]))
                .try_into()
                .unwrap();
        }
        Some(result)
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.iter().map(|&v| bit_length_inverse(v.into()))
    }

    pub fn iter_binary_scaled(&self) -> impl Iterator<Item = u8> + '_ {
        self.0.iter().copied()
    }

    pub fn map<F>(mut self, mut f: F) -> MemVec
    where
        F: FnMut(u64) -> u64,
    {
        for idx in 0..self.len() {
            self.0[idx] = bit_length(f(bit_length_inverse(self.0[idx].into())))
                .try_into()
                .unwrap();
        }
        self
    }

    /// Returns an [Iterator] over smaller power-of-two [MemVec]s.
    ///
    /// ```
    /// # use smallvec::smallvec;
    /// # use morello::memorylimits::MemVec;
    /// # use morello::target::X86Target;
    /// let it = MemVec::new([2, 1, 0, 0]).iter_down_by_powers_of_two::<X86Target>();
    /// assert_eq!(it.collect::<Vec<_>>(), vec![
    ///     MemVec::new([2, 1, 0, 0]),
    ///     MemVec::new([2, 0, 0, 0]),
    ///     MemVec::new([1, 1, 0, 0]),
    ///     MemVec::new([1, 0, 0, 0]),
    ///     MemVec::new([0, 1, 0, 0]),
    ///     MemVec::new([0, 0, 0, 0]),
    /// ]);
    /// ```
    pub fn iter_down_by_powers_of_two<T: Target>(&self) -> impl Iterator<Item = MemVec> {
        self.iter_binary_scaled()
            .map(|t| (0..=t).rev())
            .multi_cartesian_product()
            .map(|prod| MemVec::new_from_binary_scaled(prod.try_into().unwrap()))
    }
}

impl PartialOrd for MemVec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        debug_assert_eq!(self.0.len(), other.0.len());
        let first_cmp = self.0[0].partial_cmp(&other.0[0]);
        debug_assert!(first_cmp.is_some());
        for idx in 1..self.0.len() {
            if self.0[idx].partial_cmp(&other.0[idx]) != first_cmp {
                return None;
            }
        }
        first_cmp
    }

    fn le(&self, other: &Self) -> bool {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0.iter().zip(&other.0).all(|(a, b)| a <= b)
    }

    fn ge(&self, other: &Self) -> bool {
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0.iter().zip(&other.0).all(|(a, b)| a >= b)
    }
}

impl Sub for MemVec {
    type Output = Self;

    // TODO: Implement this without converting to linear space. (And add test!)
    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Sub<MemVec> for &MemVec {
    type Output = MemVec;

    fn sub(self, rhs: MemVec) -> Self::Output {
        self.clone() - rhs
    }
}

impl Display for MemVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.iter().join(", "))
    }
}

impl<Tgt: Target> BiMap for MemoryLimitsBimap<Tgt> {
    type Domain = MemoryLimits;
    type Codomain = SmallVec<[BimapInt; 4]>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        match t {
            MemoryLimits::Standard(limits_vec) => {
                debug_assert_eq!(limits_vec.len(), Tgt::levels().len());
                limits_vec
                    .iter_binary_scaled()
                    .map(|v| BimapInt::try_from(v).unwrap())
                    .collect()
            }
        }
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        // Convert array from BimapInt to u8.
        let a = std::array::from_fn(|idx| i[idx].try_into().unwrap());
        MemoryLimits::Standard(MemVec::new_from_binary_scaled(a))
    }
}

#[cfg(test)]
pub fn arb_memorylimits<Tgt: Target>(
    maximum_memory: &MemVec,
) -> impl proptest::strategy::Strategy<Value = MemoryLimits> {
    arb_memorylimits_ext(&MemVec::zero::<Tgt>(), maximum_memory)
}

#[cfg(test)]
pub fn arb_memorylimits_ext(
    minimum_memory: &MemVec,
    maximum_memory: &MemVec,
) -> impl proptest::strategy::Strategy<Value = MemoryLimits> {
    use proptest::prelude::*;

    let component_ranges = minimum_memory
        .iter_binary_scaled()
        .zip(maximum_memory.iter_binary_scaled())
        .map(|(b, t)| b..=t)
        .collect::<Vec<_>>();
    component_ranges.prop_map(|v| {
        let linear_v = v
            .into_iter()
            .map(|v| bit_length_inverse(v.into()))
            .collect::<Vec<_>>();
        MemoryLimits::Standard(MemVec::new(linear_v.try_into().unwrap()))
    })
}
