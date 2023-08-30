use itertools::Itertools;
use log::warn;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt::{Display, Formatter};
use std::{
    iter,
    ops::{Index, IndexMut, Sub},
};

use crate::utils::iter_powers_of_two;
use crate::{
    target::{Target, MAX_LEVEL_COUNT},
    utils::prev_power_of_two,
};

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
/// Put another way: this is a description of the memory live during execution of a single node,
/// ignoring children.
pub enum MemoryAllocation {
    Simple(MemVec),
    Inner(Vec<MemVec>),
    Pipeline {
        intermediate_consumption: Vec<MemVec>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Deserialize, Serialize)]
pub struct MemVec(SmallVec<[u64; MAX_LEVEL_COUNT]>);

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
                    mem_vec[i] = prev_power_of_two(mem_vec[i]);
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

        let per_child_diffs: Box<dyn Iterator<Item = &MemVec>> = match allocated {
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
                    let Some(to_push) = cur_limit
                        .clone()
                        .checked_sub(child_allocation) else
                    {
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

impl MemoryAllocation {
    pub fn none<Tgt: Target>() -> Self {
        MemoryAllocation::Simple(MemVec::zero::<Tgt>())
    }
}

impl Display for MemoryLimits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLimits::Standard(mem_vec) => mem_vec.fmt(f),
        }
    }
}

impl MemVec {
    pub fn new(contents: SmallVec<[u64; MAX_LEVEL_COUNT]>) -> Self {
        assert!(contents.iter().all(|&v| v == 0 || v.is_power_of_two()));
        MemVec(contents)
    }

    pub fn zero<Tgt: Target>() -> Self {
        MemVec(SmallVec::from_elem(0, Tgt::levels().len()))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn checked_sub(self, rhs: &MemVec) -> Option<MemVec> {
        assert_eq!(self.len(), rhs.len());
        let mut result = self;
        for idx in 0..result.len() {
            if result[idx] < rhs[idx] {
                return None;
            }
            result[idx] -= rhs[idx];
        }
        Some(result)
    }

    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.0.iter()
    }

    /// Returns an [Iterator] over smaller power-of-two [MemVec]s.
    ///
    /// ```
    /// # use smallvec::smallvec;
    /// # use morello::memorylimits::MemVec;
    /// # use morello::target::X86Target;
    /// let it = MemVec::new(smallvec![2, 1]).iter_down_by_powers_of_two::<X86Target>();
    /// assert_eq!(it.collect::<Vec<_>>(), vec![
    ///     MemVec::new(smallvec![2, 1]),
    ///     MemVec::new(smallvec![2, 0]),
    ///     MemVec::new(smallvec![1, 1]),
    ///     MemVec::new(smallvec![1, 0]),
    ///     MemVec::new(smallvec![0, 1]),
    ///     MemVec::new(smallvec![0, 0]),
    /// ]);
    /// ```
    pub fn iter_down_by_powers_of_two<T: Target>(&self) -> impl Iterator<Item = MemVec> {
        self.into_iter()
            .map(|l| iter_powers_of_two(l, true).rev())
            .multi_cartesian_product()
            .map(move |prod| MemVec::new(prod.into_iter().collect()))
    }
}

impl Sub for MemVec {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let mut result = self;
        for idx in 0..result.len() {
            result[idx] -= rhs[idx];
        }
        result
    }
}

impl Sub<MemVec> for &MemVec {
    type Output = MemVec;

    fn sub(self, rhs: MemVec) -> Self::Output {
        self.clone() - rhs
    }
}

impl Index<usize> for MemVec {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for MemVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for MemVec {
    type Item = u64;

    type IntoIter = smallvec::IntoIter<[u64; MAX_LEVEL_COUNT]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// .into_iter() on a MemVec will just return values since we know they're small.
impl<'a> IntoIterator for &'a MemVec {
    type Item = u64;

    type IntoIter = iter::Cloned<std::slice::Iter<'a, u64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().cloned()
    }
}

impl FromIterator<u64> for MemVec {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        MemVec(iter.into_iter().collect())
    }
}

impl Display for MemVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0.iter().join(", "))
    }
}
