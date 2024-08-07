use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::utils::{bit_length, bit_length_inverse, next_binary_power};
use crate::{
    target::{Target, LEVEL_COUNT},
    utils::prev_power_of_two,
};

use itertools::{Either, Itertools};
use log::warn;
use serde::{Deserialize, Serialize};

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
#[derive(Debug)]
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

    pub fn zero_levels_slower_than_all<Tgt>(&mut self, levels_bounds: &[Tgt::Level])
    where
        Tgt: Target,
    {
        // This is slow; it visits the product rather than computing a lub when that's correct.
        let target_levels = Tgt::levels();
        match self {
            MemoryLimits::Standard(mem_vec) => {
                for (limit_idx, cur) in target_levels.iter().enumerate() {
                    if levels_bounds.iter().all(|bound| bound < cur) {
                        mem_vec.set_unscaled(limit_idx, 0);
                    }
                }
            }
        }
    }

    pub fn any_nonzero_levels_slower_than<Tgt>(&self, levels_bounds: &[Tgt::Level]) -> bool
    where
        Tgt: Target,
    {
        // This is slow; it visits the product rather than computing a lub when that's correct.
        let target_levels = Tgt::levels();
        match self {
            MemoryLimits::Standard(mem_vec) => {
                for (limit_idx, cur) in target_levels.iter().enumerate() {
                    if mem_vec.get_unscaled(limit_idx) != 0
                        && levels_bounds.iter().all(|bound| bound < cur)
                    {
                        return true;
                    }
                }
            }
        }
        false
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

        let per_child_diffs = match allocated {
            MemoryAllocation::Simple(v) => Either::Left(iter::repeat(v)),
            MemoryAllocation::Inner(during_children) => Either::Right(during_children.iter()),
            MemoryAllocation::Pipeline {
                intermediate_consumption: _,
            } => todo!(),
        };

        match self {
            MemoryLimits::Standard(cur_limit) => {
                let mut result = Vec::with_capacity(cur_limit.len());
                for child_allocation in per_child_diffs {
                    debug_assert_eq!(child_allocation.len(), cur_limit.len());
                    let to_push = cur_limit.clone().checked_sub_snap_down(child_allocation)?;
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

    fn lt(&self, other: &Self) -> bool {
        match (self, other) {
            (MemoryLimits::Standard(limits_vec), MemoryLimits::Standard(other_limits_vec)) => {
                limits_vec.lt(other_limits_vec)
            }
        }
    }

    fn gt(&self, other: &Self) -> bool {
        match (self, other) {
            (MemoryLimits::Standard(limits_vec), MemoryLimits::Standard(other_limits_vec)) => {
                limits_vec.gt(other_limits_vec)
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
    pub fn none() -> Self {
        MemoryAllocation::Simple([0; LEVEL_COUNT])
    }

    // TODO: Document.
    pub fn peak_memory_from_child_peaks<Tgt: Target>(&self, child_peaks: &[MemVec]) -> MemVec {
        let mut peak = MemVec::zero::<Tgt>();
        match self {
            MemoryAllocation::Simple(own) => {
                for child_peak in child_peaks {
                    for (i, o) in own.iter().enumerate() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(
                                peak.get_unscaled(i).max(o + child_peak.get_unscaled(i)),
                            ),
                        );
                    }
                }
            }
            MemoryAllocation::Inner(child_adds) => {
                debug_assert_eq!(child_peaks.len(), child_adds.len());
                for (child_peak, own_child_alloc) in child_peaks.iter().zip(child_adds) {
                    for (i, o) in own_child_alloc.iter().enumerate() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(
                                peak.get_unscaled(i).max(child_peak.get_unscaled(i) + *o),
                            ),
                        )
                    }
                }
            }
            MemoryAllocation::Pipeline {
                intermediate_consumption,
            } => {
                debug_assert_eq!(child_peaks.len() + 1, intermediate_consumption.len());
                let z = [0; LEVEL_COUNT];
                let mut preceding_consumption = &z;
                let mut following_consumption = &intermediate_consumption[0];
                for (child_idx, child_peak) in child_peaks.iter().enumerate() {
                    for i in 0..peak.len() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(peak.get_unscaled(i).max(
                                preceding_consumption[i]
                                    + child_peak.get_unscaled(i)
                                    + following_consumption[i],
                            )),
                        );
                    }
                    preceding_consumption = following_consumption;
                    following_consumption =
                        intermediate_consumption.get(child_idx + 1).unwrap_or(&z);
                }
            }
        }
        peak
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
        for (result_entry, &r) in result.0.iter_mut().zip(rhs) {
            let cur = bit_length_inverse((*result_entry).into());
            if cur < r {
                return None;
            }
            *result_entry = bit_length(prev_power_of_two(cur - r)).try_into().unwrap();
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

    fn lt(&self, other: &Self) -> bool {
        debug_assert_eq!(self.0.len(), other.0.len());
        let mut found_diff = false;
        for (l, r) in self.0.iter().zip(&other.0) {
            if l > r {
                return false;
            }
            if l < r {
                found_diff = true;
            }
        }
        found_diff
    }

    fn gt(&self, other: &Self) -> bool {
        debug_assert_eq!(self.0.len(), other.0.len());
        let mut found_diff = false;
        for (l, r) in self.0.iter().zip(&other.0) {
            if l < r {
                return false;
            }
            if l > r {
                found_diff = true;
            }
        }
        found_diff
    }
}

impl Sub for MemVec {
    type Output = Self;

    // TODO: Implement this without converting to linear space. (And add test!)
    fn sub(self, _rhs: Self) -> Self::Output {
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
    type Codomain = Vec<BimapInt>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        match t {
            MemoryLimits::Standard(limits_vec) => {
                debug_assert_eq!(limits_vec.len(), Tgt::levels().len());
                limits_vec.iter_binary_scaled().map_into().collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{ArmTarget, CpuMemoryLevel, X86Target};
    use proptest::prelude::*;

    #[test]
    fn test_zero_levels_slower_than_all_x86() {
        shared_test_zero_levels_slower_than_all::<X86Target>();
    }

    #[test]
    fn test_zero_levels_slower_than_all_arm() {
        shared_test_zero_levels_slower_than_all::<ArmTarget>();
    }

    #[test]
    fn test_memorylimits_standard_partialord() {
        let a = MemoryLimits::Standard(MemVec::new_from_binary_scaled([1, 2, 3, 4]));
        let b = MemoryLimits::Standard(MemVec::new_from_binary_scaled([1, 2, 3, 4]));
        assert!(a >= b);
        assert!(a <= b);
        assert!(!a.lt(&b));
        assert!(!a.gt(&b));

        let b = MemoryLimits::Standard(MemVec::new_from_binary_scaled([1, 2, 3, 5]));
        assert!(a < b);
        assert!(a <= b);
        assert!(!a.gt(&b));
        assert!(!a.ge(&b));

        let b = MemoryLimits::Standard(MemVec::new_from_binary_scaled([1, 2, 1, 5]));
        assert!(!a.lt(&b));
        assert!(!a.le(&b));
        assert!(!a.gt(&b));
        assert!(!a.ge(&b));
        assert!(a.partial_cmp(&b).is_none())
    }

    proptest! {
        #[test]
        fn test_zero_levels_slow_than_all_consistent_with_any_nonzero_x86(
            limits in arb_memorylimits::<X86Target>(&MemVec::new([1; LEVEL_COUNT])),
            bounds in prop::collection::vec(any::<CpuMemoryLevel>(), 0..=3)
        ) {
            shared_test_zero_levels_slow_than_all_consistent_with_any_nonzero::<X86Target>(
                limits, &bounds
            );
        }

        #[test]
        fn test_zero_levels_slow_than_all_consistent_with_any_nonzero_arm(
            limits in arb_memorylimits::<ArmTarget>(&MemVec::new([1; LEVEL_COUNT])),
            bounds in prop::collection::vec(any::<CpuMemoryLevel>(), 0..=3)
        ) {
            shared_test_zero_levels_slow_than_all_consistent_with_any_nonzero::<ArmTarget>(
                limits, &bounds
            );
        }
    }

    fn shared_test_zero_levels_slower_than_all<Tgt>()
    where
        Tgt: Target<Level = CpuMemoryLevel>,
    {
        let mut levels = MemoryLimits::Standard(MemVec::new([8, 8, 8, 8]));
        levels.zero_levels_slower_than_all::<Tgt>(&[CpuMemoryLevel::GL]);
        assert_eq!(levels, MemoryLimits::Standard(MemVec::new([8, 8, 8, 8])));

        levels = MemoryLimits::Standard(MemVec::new([8, 8, 8, 8]));
        levels.zero_levels_slower_than_all::<Tgt>(&[CpuMemoryLevel::L1]);
        assert_eq!(levels, MemoryLimits::Standard(MemVec::new([8, 8, 8, 0])));

        // Notice that VRF is *not* slwoer than RF. This maps to an assumption that we can move from
        // RF to VRF.
        levels = MemoryLimits::Standard(MemVec::new([8, 8, 8, 8]));
        levels.zero_levels_slower_than_all::<Tgt>(&[CpuMemoryLevel::RF]);
        assert_eq!(levels, MemoryLimits::Standard(MemVec::new([8, 8, 0, 0])));
    }

    fn shared_test_zero_levels_slow_than_all_consistent_with_any_nonzero<Tgt: Target>(
        limits: MemoryLimits,
        bounds: &[Tgt::Level],
    ) {
        let mut zeroed_limits = limits.clone();
        zeroed_limits.zero_levels_slower_than_all::<Tgt>(bounds);
        assert_eq!(
            zeroed_limits == limits,
            !limits.any_nonzero_levels_slower_than::<Tgt>(bounds)
        );
    }
}
