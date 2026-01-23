use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::utils::{bit_length, bit_length_inverse, next_binary_power};
use crate::{
    target::{MemoryLevel, Target, LEVEL_COUNT},
    utils::prev_power_of_two,
};

use itertools::{izip, Either, Itertools};
use log::warn;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::{iter, ops::Sub};

/// MemoryLimits are bounds on available memory for each level of a target.
///
/// There are two variants. `MemoryLimits::Standard` counts the number of registers for
/// register-counting levels and the number of cache lines (with size
/// [Target::line_size]) for all other levels. No Spec satisfying the MemoryLimits
/// should have a peak exceeding those. `MemoryLimits::Pipeline` counts separately the
/// memory used by intermediate tensors just before or after a stage. This expands the
/// set of `ImplNode::Pipeline`s which might satisfy a Spec, because it is valid for
/// that `Pipeline` to assume that those bytes have been freed after its own first and
/// last stages complete.
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

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct MemVec([u8; LEVEL_COUNT]);

pub struct MemVecIter<'a> {
    mem_vec: &'a MemVec,
    index: usize,
}

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

    pub fn discretize<Tgt: Target>(&mut self) {
        match self {
            MemoryLimits::Standard(mem_vec) => {
                Tgt::levels()
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, level)| {
                        if !level.counts_registers() {
                            mem_vec.set_bit_length(i, prev_power_of_two(mem_vec.get_unscaled(i)));
                        }
                    });
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
                        mem_vec.set_bit_length(limit_idx, 0);
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
                    let to_push = cur_limit
                        .clone()
                        .checked_sub_snap_down(child_allocation)
                        .ok()?;
                    let mut to_push = MemoryLimits::Standard(to_push);
                    to_push.discretize::<Tgt>();
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
        let levels = Tgt::levels();
        let mut peak = MemVec::zero::<Tgt>();
        match self {
            MemoryAllocation::Simple(own) => {
                for child_peak in child_peaks {
                    for (i, o) in own.iter().enumerate() {
                        let mut computed_peak =
                            peak.get_unscaled(i).max(o + child_peak.get_unscaled(i));
                        if !levels[i].counts_registers() {
                            computed_peak = next_binary_power(computed_peak);
                        }
                        peak.set_snap_up(i, computed_peak);
                    }
                }
            }
            MemoryAllocation::Inner(child_adds) => {
                debug_assert_eq!(child_peaks.len(), child_adds.len());
                for (child_peak, own_child_alloc) in child_peaks.iter().zip(child_adds) {
                    for (i, o) in own_child_alloc.iter().enumerate() {
                        let mut computed_peak =
                            peak.get_unscaled(i).max(child_peak.get_unscaled(i) + *o);
                        if !levels[i].counts_registers() {
                            computed_peak = next_binary_power(computed_peak);
                        }
                        peak.set_snap_up(i, computed_peak);
                    }
                }
            }
            MemoryAllocation::Pipeline {
                intermediate_consumption,
            } => {
                debug_assert_eq!(child_peaks.len(), intermediate_consumption.len() + 1);
                let z = [0; LEVEL_COUNT];
                let mut preceding_consumption = &z;
                let mut following_consumption = &intermediate_consumption[0];
                for (child_idx, child_peak) in child_peaks.iter().enumerate() {
                    for i in 0..peak.len() {
                        let mut raw_peak = peak.get_unscaled(i).max(
                            preceding_consumption[i]
                                + child_peak.get_unscaled(i)
                                + following_consumption[i],
                        );
                        if !levels[i].counts_registers() {
                            raw_peak = next_binary_power(raw_peak);
                        }
                        peak.set_snap_up(i, raw_peak);
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
        let mut result = MemVec([0; LEVEL_COUNT]);
        for (i, &value) in contents.iter().enumerate() {
            result.set(i, value);
        }
        result
    }

    pub fn new_for_target<Tgt: Target>(limits_arr: [u64; LEVEL_COUNT]) -> Self {
        let levels = Tgt::levels();
        let mut corrected_limits = limits_arr;

        // For levels that don't count registers (use bit-length encoding),
        // we need to ensure values are powers of two
        for (i, level) in levels.iter().enumerate() {
            if !level.counts_registers() {
                corrected_limits[i] = next_binary_power(corrected_limits[i]);
            }
        }

        Self::new_mixed(
            corrected_limits,
            levels.map(|level| level.counts_registers()),
        )
    }

    /// Creates a new MemVec with provided encoding types.
    ///
    /// `values`: The actual values to store
    /// `use_raw_encoding`: true for raw encoding, false for bit-length encoding
    pub fn new_mixed(values: [u64; LEVEL_COUNT], use_raw_encoding: [bool; LEVEL_COUNT]) -> Self {
        let mut result = [0u8; LEVEL_COUNT];
        for i in 0..LEVEL_COUNT {
            if use_raw_encoding[i] {
                result[i] = Self::encode_raw(values[i]);
            } else {
                result[i] = Self::encode_bit_length(values[i]);
            }
        }
        MemVec(result)
    }

    pub fn zero<Tgt: Target>() -> Self {
        debug_assert_eq!(Tgt::levels().len(), LEVEL_COUNT);
        MemVec([0; LEVEL_COUNT])
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get_unscaled(&self, idx: usize) -> u64 {
        let value = self.0[idx];
        if value & 0x80 == 0 {
            // Leading bit is 0: bit length encoding
            bit_length_inverse(value.into())
        } else {
            // Leading bit is 1: raw value encoding
            u64::from(value & 0x7F)
        }
    }

    pub fn get_binary_scaled(&self, idx: usize) -> u8 {
        let value = self.0[idx];
        if value & 0x80 == 0 {
            // Already bit length encoded
            value
        } else {
            // Raw encoded, convert to bit length
            let raw_value = u64::from(value & 0x7F);
            if raw_value == 0 || raw_value.is_power_of_two() {
                bit_length(raw_value).try_into().unwrap()
            } else {
                panic!("value cannot be converted to bit length")
            }
        }
    }

    pub fn set(&mut self, idx: usize, value: u64) {
        if value <= 127 && (value == 0 || !value.is_power_of_two()) {
            self.set_raw(idx, value);
        } else if value == 0 || value.is_power_of_two() {
            self.set_bit_length(idx, value);
        } else {
            panic!("Value {value} too large for raw encoding and not bit-length-encoding");
        }
    }

    /// Sets a value at the specified index, snapping up to the next encodable value if necessary.
    ///
    /// Values > 127 that are not powers of two are snapped up to the next power of two.
    pub fn set_snap_up(&mut self, idx: usize, value: u64) {
        if value < 128 {
            self.set_raw(idx, value);
        } else {
            let snapped_value = if value.is_power_of_two() {
                value
            } else {
                value.next_power_of_two()
            };
            self.set_bit_length(idx, snapped_value);
        }
    }

    /// Sets a raw-encoded value at the specified index.
    ///
    /// Panics if `value > 127`.
    fn set_raw(&mut self, idx: usize, value: u64) {
        self.0[idx] = Self::encode_raw(value);
    }

    /// Encodes a u64 value as raw lines (leading bit 1).
    ///
    /// Panics if `value > 127`.
    fn encode_raw(value: u64) -> u8 {
        if value > 127 {
            panic!("Raw encoded value {value} exceeds maximum of 127");
        }
        u8::try_from(value).unwrap() | 0x80
    }

    /// Encodes a u64 value as bit length (leading bit 0).
    ///
    /// Panics if `bit_length(value) > 127`.
    fn encode_bit_length(value: u64) -> u8 {
        let bit_len = bit_length(value);
        if bit_len > 127 {
            panic!("Bit length {bit_len} exceeds maximum of 127");
        }
        bit_len.try_into().unwrap()
    }

    /// Decodes a value according to its most significant bit.
    fn decode_value(encoded: u8) -> u64 {
        if encoded & 0x80 == 0 {
            bit_length_inverse(encoded.into())
        } else {
            u64::from(encoded & 0x7F)
        }
    }

    /// Sets a bit-length encoded value at the specified index.
    ///
    /// Panics if `bit_length(value) > 127`.
    fn set_bit_length(&mut self, idx: usize, value: u64) {
        self.0[idx] = Self::encode_bit_length(value);
    }

    pub fn checked_sub_snap_down(self, rhs: &[u64; LEVEL_COUNT]) -> Result<MemVec, usize> {
        let mut result = self;
        for (i, (result_entry, &r)) in result.0.iter_mut().zip(rhs).enumerate() {
            let value = *result_entry;
            let cur = if value & 0x80 == 0 {
                bit_length_inverse(value.into())
            } else {
                u64::from(value & 0x7F)
            };

            if cur < r {
                return Err(i);
            }

            let new_value = prev_power_of_two(cur - r);
            if value & 0x80 == 0 {
                // Was bit length encoded, encode result as bit length
                *result_entry = Self::encode_bit_length(new_value);
            } else {
                // Was raw encoded, encode result as raw
                *result_entry = Self::encode_raw(new_value);
            }
        }
        Ok(result)
    }

    pub fn iter(&self) -> MemVecIter<'_> {
        MemVecIter {
            mem_vec: self,
            index: 0,
        }
    }

    pub fn map<F>(mut self, mut f: F) -> MemVec
    where
        F: FnMut(u64) -> u64,
    {
        for idx in 0..self.len() {
            self.set(idx, f(self.get_unscaled(idx)));
        }
        self
    }

    /// Snaps memory values up to the next power of two.
    ///
    /// Only applies to non-register-counting levels.
    pub fn snap_up_for_target<Tgt: Target>(self) -> MemVec {
        let levels = Tgt::levels();
        let mut result = self;
        for (i, level) in levels.iter().enumerate() {
            if !level.counts_registers() {
                let current = result.get_unscaled(i);
                if current != 0 {
                    result.set_bit_length(i, current.next_power_of_two());
                }
            }
        }
        result
    }

    /// Returns a new MemVec with the elementwise maximum of self and other.
    pub fn max(self, other: &Self) -> Self {
        self.elementwise_compare(other, |a, b| a.max(b), |a, b| a.max(b))
    }

    /// Returns a new MemVec with the elementwise minimum of self and other.
    pub fn min(self, other: &Self) -> Self {
        self.elementwise_compare(other, |a, b| a.min(b), |a, b| a.min(b))
    }

    fn elementwise_compare<F, G>(self, other: &Self, compare_fn_u64: F, compare_fn_u8: G) -> Self
    where
        F: Fn(u64, u64) -> u64,
        G: Fn(u8, u8) -> u8,
    {
        let mut result = [0u8; LEVEL_COUNT];
        for (&self_encoded, &other_encoded, r) in izip!(&self.0, &other.0, &mut result) {
            if (self_encoded & 0x80) == (other_encoded & 0x80) {
                // Same encoding, can compare directly. Leading bit won't affect result.
                *r = compare_fn_u8(self_encoded, other_encoded);
            } else {
                let self_value = Self::decode_value(self_encoded);
                let other_value = Self::decode_value(other_encoded);
                let comparison_result = compare_fn_u64(self_value, other_value);
                if comparison_result == self_value {
                    *r = self_encoded;
                } else {
                    *r = other_encoded;
                }
            }
        }
        MemVec(result)
    }
}

impl<'a> IntoIterator for &'a MemVec {
    type Item = u64;
    type IntoIter = MemVecIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        MemVecIter {
            mem_vec: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a mut MemVec {
    type Item = u64;
    type IntoIter = MemVecIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        MemVecIter {
            mem_vec: self,
            index: 0,
        }
    }
}

impl PartialEq for MemVec {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0.iter().zip(&other.0).all(|(a, b)| {
            if (a & 0x80) == (b & 0x80) {
                a == b
            } else {
                Self::decode_value(*a) == Self::decode_value(*b)
            }
        })
    }
}

impl Eq for MemVec {}

impl std::hash::Hash for MemVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for &encoded in &self.0 {
            Self::decode_value(encoded).hash(state);
        }
    }
}

impl PartialOrd for MemVec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.0.len() != other.0.len() {
            return None;
        }

        let compare_encoded = |a: u8, b: u8| -> Ordering {
            if (a & 0x80) == (b & 0x80) {
                a.cmp(&b)
            } else {
                Self::decode_value(a).cmp(&Self::decode_value(b))
            }
        };

        let mut has_less = false;
        let mut has_greater = false;

        for (a, b) in self.0.iter().zip(&other.0) {
            match compare_encoded(*a, *b) {
                Ordering::Less => has_less = true,
                Ordering::Greater => has_greater = true,
                Ordering::Equal => {}
            }

            // If we have both less and greater, no consistent ordering
            if has_less && has_greater {
                return None;
            }
        }

        if has_less {
            Some(Ordering::Less)
        } else if has_greater {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }

    fn le(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0.iter().zip(&other.0).all(|(a, b)| {
            if (a & 0x80) == (b & 0x80) {
                a <= b
            } else {
                Self::decode_value(*a) <= Self::decode_value(*b)
            }
        })
    }

    fn ge(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0.iter().zip(&other.0).all(|(a, b)| {
            if (a & 0x80) == (b & 0x80) {
                a >= b
            } else {
                Self::decode_value(*a) >= Self::decode_value(*b)
            }
        })
    }

    fn lt(&self, other: &Self) -> bool {
        matches!(self.partial_cmp(other), Some(std::cmp::Ordering::Less))
    }

    fn gt(&self, other: &Self) -> bool {
        matches!(self.partial_cmp(other), Some(std::cmp::Ordering::Greater))
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

impl fmt::Debug for MemVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemVec([")?;
        for (i, &byte) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let decoded_value = Self::decode_value(byte);
            if byte & 0x80 == 0 {
                write!(f, "{decoded_value}*")?;
            } else {
                write!(f, "{decoded_value}")?;
            }
        }
        write!(f, "])")
    }
}

impl Display for MemVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.iter().join(", "))
    }
}

impl Iterator for MemVecIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.mem_vec.len() {
            let value = self.mem_vec.get_unscaled(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.mem_vec.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MemVecIter<'_> {}

impl<Tgt: Target> BiMap for MemoryLimitsBimap<Tgt> {
    type Domain = MemoryLimits;
    type Codomain = Vec<BimapInt>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        match t {
            MemoryLimits::Standard(limits_vec) => Tgt::levels()
                .into_iter()
                .enumerate()
                .map(|(i, level)| {
                    if level.counts_registers() {
                        BimapInt::try_from(limits_vec.get_unscaled(i)).unwrap()
                    } else {
                        BimapInt::from(limits_vec.get_binary_scaled(i))
                    }
                })
                .collect(),
        }
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        let levels = Tgt::levels();
        let mut values = [0u64; LEVEL_COUNT];
        for idx in 0..LEVEL_COUNT {
            if levels[idx].counts_registers() {
                values[idx] = i[idx].into();
            } else {
                values[idx] = bit_length_inverse(i[idx]);
            }
        }
        MemoryLimits::Standard(MemVec::new_for_target::<Tgt>(values))
    }
}

#[cfg(test)]
pub fn arb_memorylimits<Tgt: Target>(
    maximum_memory: &MemVec,
) -> impl proptest::strategy::Strategy<Value = MemoryLimits> {
    arb_memorylimits_ext::<Tgt>(&MemVec::zero::<Tgt>(), maximum_memory)
}

#[cfg(test)]
pub fn arb_memorylimits_ext<Tgt: Target>(
    minimum_memory: &MemVec,
    maximum_memory: &MemVec,
) -> impl proptest::strategy::Strategy<Value = MemoryLimits> {
    use proptest::prelude::*;
    use proptest::strategy::BoxedStrategy;

    let levels = Tgt::levels();
    let component_ranges = (0..LEVEL_COUNT)
        .map(|i| -> BoxedStrategy<u64> {
            if levels[i].counts_registers() {
                // For register-counting levels, use raw integer ranges
                (minimum_memory.get_unscaled(i)..=maximum_memory.get_unscaled(i)).boxed()
            } else {
                // For non-register levels, use binary scaling (power-of-two ranges)
                (minimum_memory.get_binary_scaled(i)..=maximum_memory.get_binary_scaled(i))
                    .prop_map(|bit_len| bit_length_inverse(bit_len.into()))
                    .boxed()
            }
        })
        .collect::<Vec<_>>();

    component_ranges.prop_map(move |v| {
        let mut values = [0u64; LEVEL_COUNT];
        let mut use_raw_encoding = [false; LEVEL_COUNT];
        for i in 0..LEVEL_COUNT {
            values[i] = v[i];
            use_raw_encoding[i] = levels[i].counts_registers();
        }
        MemoryLimits::Standard(MemVec::new_mixed(values, use_raw_encoding))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{ArmTarget, Avx2Target, CpuMemoryLevel};
    use proptest::{array::uniform4, prelude::*};

    #[test]
    fn test_zero_levels_slower_than_all_avx2() {
        shared_test_zero_levels_slower_than_all::<Avx2Target>();
    }

    #[test]
    fn test_zero_levels_slower_than_all_arm() {
        shared_test_zero_levels_slower_than_all::<ArmTarget>();
    }

    #[test]
    fn test_memorylimits_standard_partialord() {
        let a = MemoryLimits::Standard(MemVec::new([1, 2, 3, 4]));
        let b = MemoryLimits::Standard(MemVec::new([1, 2, 3, 4]));
        assert!(a >= b);
        assert!(a <= b);
        assert!(!a.lt(&b));
        assert!(!a.gt(&b));
        assert!(matches!(a.partial_cmp(&b), Some(Ordering::Equal)));

        let b = MemoryLimits::Standard(MemVec::new([1, 2, 3, 5]));
        assert!(a < b);
        assert!(a <= b);
        assert!(!a.gt(&b));
        assert!(!a.ge(&b));
        assert!(matches!(a.partial_cmp(&b), Some(Ordering::Less)));

        let b = MemoryLimits::Standard(MemVec::new([1, 2, 1, 5]));
        assert!(!a.lt(&b));
        assert!(!a.le(&b));
        assert!(!a.gt(&b));
        assert!(!a.ge(&b));
        assert!(a.partial_cmp(&b).is_none())
    }

    #[test]
    fn test_memvec_equality_compares_decoded_values() {
        let memvec1 = MemVec::new([1, 2, 3, 4]);
        let memvec2 = MemVec::new([1, 2, 3, 4]);
        assert_eq!(memvec1, memvec2);

        // Test with different encodings that decode to same value
        // 129 (128+1) has MSB set and decodes to 1
        // 1 has MSB not set and bit_length_inverse(1) = 2^(1-1) = 1
        let mut test_memvec1 = MemVec::new([1, 1, 1, 1]);
        let mut test_memvec2 = MemVec::new([1, 1, 1, 1]);
        test_memvec1.0[0] = 129; // MSB set, decodes to 1
        test_memvec2.0[0] = 1; // MSB not set, also decodes to 1
        assert_eq!(test_memvec1, test_memvec2);
        assert!(!test_memvec1.lt(&test_memvec2));
        assert!(!test_memvec1.gt(&test_memvec2));
        assert!(test_memvec1.le(&test_memvec2));
        assert!(test_memvec1.ge(&test_memvec2));
    }

    #[test]
    fn test_memvec_map() {
        // Test with basic values using default encoding
        let original = MemVec::new([1, 2, 4, 8]);
        let doubled = original.clone().map(|x| x * 2);

        assert_eq!(doubled.get_unscaled(0), 2);
        assert_eq!(doubled.get_unscaled(1), 4);
        assert_eq!(doubled.get_unscaled(2), 8);
        assert_eq!(doubled.get_unscaled(3), 16);

        // Check that values are transformed correctly
        let mixed = MemVec::new([1, 2, 3, 4]);
        let mapped = mixed.map(|x| x + 10);
        assert_eq!(mapped.get_unscaled(0), 11);
        assert_eq!(mapped.get_unscaled(1), 12);
        assert_eq!(mapped.get_unscaled(2), 13);
        assert_eq!(mapped.get_unscaled(3), 14);
    }

    #[test]
    #[should_panic(expected = "Value 129 too large for raw encoding and not bit-length-encoding")]
    fn test_memvec_map_panics_when_mapping_to_non_power_of_two_above_128() {
        let memvec = MemVec::new([1, 2, 4, 8]);
        memvec.map(|x| if x == 1 { 129 } else { x });
    }

    #[test]
    #[should_panic(expected = "Value 200 too large for raw encoding and not bit-length-encoding")]
    fn test_set_panics_on_non_power_of_two_above_127() {
        let mut memvec = MemVec::zero::<Avx2Target>();
        memvec.set(0, 200); // This should panic
    }

    #[test]
    fn test_memvec_set_snap_up() {
        let mut memvec = MemVec::zero::<Avx2Target>();
        for i in 0..=128 {
            memvec.set_snap_up(0, i);
            assert_eq!(memvec.get_unscaled(0), i);
        }
        memvec.set_snap_up(1, 256);
        assert_eq!(memvec.get_unscaled(1), 256);
        memvec.set_snap_up(2, 129);
        assert_eq!(memvec.get_unscaled(2), 256); // next power of 2 after 129
        memvec.set_snap_up(2, 200);
        assert_eq!(memvec.get_unscaled(2), 256); // next power of 2 after 200
        memvec.set_snap_up(2, 257);
        assert_eq!(memvec.get_unscaled(2), 512); // next power of 2 after 257
    }

    proptest! {
        // TODO: Add test of larger powers of two
        #[test]
        fn test_memvec_new_returns_initial_non_bit_length_scaled_values(
            initials in uniform4(0..=16u64)
        ) {
            let memvec = MemVec::new(initials);
            prop_assert_eq!(memvec.get_unscaled(0), initials[0]);
            prop_assert_eq!(memvec.get_unscaled(1), initials[1]);
            prop_assert_eq!(memvec.get_unscaled(2), initials[2]);
            prop_assert_eq!(memvec.get_unscaled(3), initials[3]);
        }

        #[test]
        fn test_zero_levels_slow_than_all_consistent_with_any_nonzero_avx2(
            limits in arb_memorylimits::<Avx2Target>(&MemVec::new([1; LEVEL_COUNT])),
            bounds in prop::collection::vec(any::<CpuMemoryLevel>(), 0..=3)
        ) {
            shared_test_zero_levels_slow_than_all_consistent_with_any_nonzero::<Avx2Target>(
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
