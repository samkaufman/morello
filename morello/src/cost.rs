use std::cmp::Ordering;

use crate::imp::{Impl, ImplExt};
use crate::memorylimits::MemVec;
use crate::target::Target;
use crate::utils::snap_memvec_up;

use serde::{Deserialize, Serialize};

#[derive(Clone, Eq, PartialEq, Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Cost {
    pub main: MainCost,
    pub peaks: MemVec,
    pub depth: u8,
}

pub type MainCost = u32;

impl Cost {
    /// Compute the cost an [Impl].
    ///
    /// This will traverse the entire [Impl] and have a runtime proportion to the number of nodes.
    pub fn from_impl<Tgt, I>(imp: &I) -> Cost
    where
        Tgt: Target,
        I: Impl<Tgt>,
    {
        let child_costs = imp
            .children()
            .iter()
            .map(|k| Cost::from_impl(k))
            .collect::<Vec<_>>();
        Cost::from_child_costs(imp, &child_costs)
    }

    /// Compute the cost of an [Impl], given the costs of its children.
    ///
    /// Unlike [Cost::from_impl], this has constant time.
    pub fn from_child_costs<Tgt, I>(imp: &I, child_costs: &[Cost]) -> Cost
    where
        Tgt: Target,
        I: Impl<Tgt>,
    {
        let child_main_costs = child_costs.iter().map(|k| k.main).collect::<Vec<_>>();
        let child_peaks = child_costs
            .iter()
            .map(|k| k.peaks.clone())
            .collect::<Vec<_>>();
        let main_cost: MainCost = imp.compute_main_cost(&child_main_costs);
        // TODO: Handle other kinds of memory, not just standard/TinyMap peaks.
        let raised_peaks = snap_memvec_up(imp.peak_memory_from_child_peaks(&child_peaks), false);
        Cost {
            main: main_cost,
            peaks: raised_peaks,
            depth: child_costs
                .iter()
                .map(|k| k.depth)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        }
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        let main_cmp = self.main.cmp(&other.main);
        if main_cmp != Ordering::Equal {
            return main_cmp;
        }

        // Define memory consumption ordering lexicographically, just so we have some total
        // order for Costs.
        debug_assert_eq!(self.peaks.len(), other.peaks.len());
        for i in 0..self.peaks.len() {
            let peaks_cmp = self
                .peaks
                .get_binary_scaled(i)
                .cmp(&other.peaks.get_binary_scaled(i));
            if peaks_cmp != Ordering::Equal {
                return peaks_cmp;
            }
        }

        self.depth.cmp(&other.depth)
    }
}
