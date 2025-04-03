use std::cmp::Ordering;
use std::num::NonZeroU64;

use crate::memorylimits::MemVec;
use crate::target::Target;
use crate::{common::DimSize, imp::Impl};

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
pub struct NormalizedCost {
    pub intensity: CostIntensity,
    pub peaks: MemVec,
    pub depth: u8,
}

#[derive(Default, Hash, Clone, Copy, Eq, PartialEq, Debug, Deserialize, Serialize)]
pub struct CostIntensity(Ratio<u32>);

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
        cost_tail_cmp(&self.peaks, self.depth, &other.peaks, other.depth)
    }
}

impl NormalizedCost {
    pub fn new(cost: Cost, volume: NonZeroU64) -> Self {
        NormalizedCost {
            intensity: CostIntensity::new(cost.main, volume),
            peaks: cost.peaks,
            depth: cost.depth,
        }
    }

    pub fn into_main_cost_for_volume(self, volume: NonZeroU64) -> Cost {
        Cost {
            main: self.intensity.into_main_cost_for_volume(volume),
            peaks: self.peaks,
            depth: self.depth,
        }
    }
}

impl PartialOrd for NormalizedCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NormalizedCost {
    fn cmp(&self, other: &Self) -> Ordering {
        let intensity_cmp = self.intensity.0.cmp(&other.intensity.0);
        if intensity_cmp != Ordering::Equal {
            return intensity_cmp;
        }
        cost_tail_cmp(&self.peaks, self.depth, &other.peaks, other.depth)
    }
}

impl CostIntensity {
    pub fn new(cost: MainCost, volume: NonZeroU64) -> Self {
        // Build a Ratio<u64> to reduce, then convert to lower-precision Ratio.
        let r64 = Ratio::<u64>::new(cost.into(), volume.get());
        let num_r = *r64.numer();
        let den_r = *r64.denom();
        Self(Ratio::new_raw(
            num_r.try_into().expect("numerator should fit in u32"),
            den_r.try_into().expect("denominator should fit in u32"),
        ))
    }

    pub fn into_main_cost_for_volume(self, volume: NonZeroU64) -> MainCost {
        let mut wider = Ratio::<u64>::new_raw((*self.0.numer()).into(), (*self.0.denom()).into());
        wider *= volume.get();
        assert!(wider.is_integer());
        wider
            .to_integer()
            .try_into()
            .expect("cost should fit in u32")
    }
}

fn cost_tail_cmp(lhs_peaks: &MemVec, lhs_depth: u8, rhs_peaks: &MemVec, rhs_depth: u8) -> Ordering {
    // Define memory consumption ordering lexicographically, just so we have some total
    // order for Costs.
    debug_assert_eq!(lhs_peaks.len(), rhs_peaks.len());
    for i in 0..lhs_peaks.len() {
        let peaks_cmp = lhs_peaks
            .get_binary_scaled(i)
            .cmp(&rhs_peaks.get_binary_scaled(i));
        if peaks_cmp != Ordering::Equal {
            return peaks_cmp;
        }
    }

    lhs_depth.cmp(&rhs_depth)
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for CostIntensity {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let original_main_cost = any::<MainCost>();
        let volume = 1u64..;
        (original_main_cost, volume)
            .prop_map(|(main_cost, volume)| {
                CostIntensity::new(main_cost, NonZeroU64::new(volume).unwrap())
            })
            .boxed()
    }
}
