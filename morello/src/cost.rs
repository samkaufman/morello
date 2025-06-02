use std::cmp::Ordering;

use crate::common::DimSize;
use crate::imp::Impl;
use crate::memorylimits::MemVec;
use crate::target::Target;

use num_rational::Ratio;
use serde::{Deserialize, Serialize};

#[derive(Clone, Eq, PartialEq, Hash, Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Cost {
    pub main: MainCost,
    pub peaks: MemVec,
    pub depth: u8,
}

#[derive(Clone, Eq, PartialEq, Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub(crate) struct NormalizedCost {
    pub intensity: CostIntensity,
    pub peaks: MemVec,
    pub depth: u8,
}

#[derive(Default, Hash, Clone, Copy, Eq, PartialEq, Debug, Deserialize, Serialize)]
pub(crate) struct CostIntensity(Ratio<u32>);

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
        Cost::from_node_and_child_costs(imp, &child_costs)
    }

    /// Compute the cost of an [Impl], given the costs of its children.
    ///
    /// Unlike [Cost::from_impl], this has constant time.
    pub fn from_node_and_child_costs<Tgt, I>(imp: &I, child_costs: &[Cost]) -> Cost
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
        let raised_peaks = imp
            .memory_allocated()
            .peak_memory_from_child_peaks::<Tgt>(&child_peaks)
            .snap_up_for_target::<Tgt>(false);
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
            let peaks_cmp = self.peaks.get_unscaled(i).cmp(&other.peaks.get_unscaled(i));
            if peaks_cmp != Ordering::Equal {
                return peaks_cmp;
            }
        }

        self.depth.cmp(&other.depth)
    }
}

impl NormalizedCost {
    pub fn new(cost: Cost, volume: DimSize) -> Self {
        NormalizedCost {
            intensity: CostIntensity::new(cost.main, volume),
            peaks: cost.peaks,
            depth: cost.depth,
        }
    }
}

impl CostIntensity {
    pub fn new(cost: MainCost, volume: DimSize) -> Self {
        Self(Ratio::new(cost, volume.get()))
    }

    pub fn into_main_cost_for_volume(mut self, volume: DimSize) -> MainCost {
        self.0 *= volume.get();
        assert!(self.0.is_integer());
        self.0.to_integer()
    }
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for CostIntensity {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let original_main_cost = any::<MainCost>();
        let volume = 1u32..;
        (original_main_cost, volume)
            .prop_map(|(main_cost, volume)| CostIntensity(Ratio::new(main_cost, volume)))
            .boxed()
    }
}
