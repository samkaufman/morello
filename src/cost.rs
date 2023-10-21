use crate::imp::{Impl, ImplExt};
use crate::memorylimits::MemVec;
use crate::target::Target;
use crate::utils::snap_memvec_up;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Cost {
    pub main: MainCost,
    pub peaks: MemVec,
    pub depth: u32,
}

pub type MainCost = u64;

impl Cost {
    pub fn from_child_costs<Tgt, Aux, I>(imp: &I, child_costs: &[Cost]) -> Cost
    where
        Tgt: Target,
        Aux: Clone,
        I: Impl<Tgt, Aux>,
    {
        let child_main_costs = child_costs
            .iter()
            .map(|k| k.main)
            .collect::<SmallVec<[_; 3]>>();
        let child_peaks = child_costs
            .iter()
            .map(|k| k.peaks.clone())
            .collect::<SmallVec<[_; 3]>>();
        let main_cost: MainCost = imp.compute_main_cost(&child_main_costs);
        // TODO: Handle other kinds of memory, not just standard/TinyMap peaks.
        let raised_peaks = snap_memvec_up(imp.peak_memory_from_child_peaks(&child_peaks), false);
        Cost {
            main: main_cost,
            peaks: raised_peaks,
            depth: 1 + child_costs.iter().map(|k| k.depth).max().unwrap_or(0),
        }
    }
}
