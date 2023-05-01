use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

use smallvec::SmallVec;

use crate::{
    imp::ImplNode,
    spec::Spec,
    target::{Target, MAX_LEVEL_COUNT},
    utils::prev_power_of_two,
};

#[derive(Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub enum MemoryLimits {
    Standard(MemVec),
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Deserialize, Serialize)]
pub struct MemVec(SmallVec<[u64; MAX_LEVEL_COUNT]>);

impl MemoryLimits {
    /// Produce new MemoryLimits for each child of a node. Returns `None` if the
    /// the memory allocation exceeds this limit.
    ///
    /// Not that this ignores base memory allocations at the leaves. It is intended to
    /// be used to prune expansions which consume too much memory without traversing.
    pub fn transition<Tgt: Target>(
        &self,
        node_spec: &Spec<Tgt>,
        node: &ImplNode<Tgt>,
    ) -> Option<Vec<MemoryLimits>> {
        let allocated = node.memory_allocated(node_spec);
        let mut result = Vec::with_capacity(allocated.during_children.len());
        match &self {
            MemoryLimits::Standard(cur_limit) => {
                for child_allocation in allocated.during_children {
                    assert_eq!(child_allocation.len(), cur_limit.len());
                    let mut new_bound = cur_limit.clone();
                    for idx in 0..new_bound.len() {
                        if new_bound[idx] < child_allocation[idx] {
                            return None;
                        }
                        new_bound[idx] -= child_allocation[idx];
                        // Round down to power of two. This underapproximates
                        // the amount of available memory.
                        new_bound[idx] = prev_power_of_two(new_bound[idx]);
                    }

                    result.push(MemoryLimits::Standard(new_bound));
                }
            }
        }
        Some(result)
    }
}

impl MemVec {
    pub fn new(contents: SmallVec<[u64; MAX_LEVEL_COUNT]>) -> Self {
        assert!(contents.iter().all(|&v| v == 0 || v.is_power_of_two()));
        MemVec(contents)
    }

    pub fn zero<Tgt: Target>() -> Self {
        (0..Tgt::levels().len()).map(|_| 0).collect()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.0.iter()
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

    type IntoIter = smallvec::IntoIter<[u64; MAX_LEVEL_COUNT]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.clone().into_iter()
    }
}

impl FromIterator<u64> for MemVec {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        MemVec(iter.into_iter().collect())
    }
}
