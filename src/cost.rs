use crate::imp::{movelet_inner_tensorspec, ImplNode};
use crate::memorylimits::MemVec;
use crate::spec::Spec;
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::TensorSpec;
use crate::utils::snap_availables_up_memvec;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp;

const INST_COST: MainCost = 1000;
const ASSIGN_INST_COST: MainCost = 10;
const MAX_COST: MainCost = u64::MAX;

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Cost {
    pub main: MainCost,
    pub peaks: MemVec,
    pub depth: u32,
}

pub type MainCost = u64;

impl Cost {
    pub fn from_child_costs<Tgt: Target>(
        spec: &Spec<Tgt>,
        imp: &ImplNode<Tgt>,
        child_costs: &[Cost],
    ) -> Cost {
        let child_main_costs = child_costs
            .iter()
            .map(|k| k.main)
            .collect::<SmallVec<[_; 3]>>();
        let child_peaks = child_costs
            .iter()
            .map(|k| k.peaks.clone())
            .collect::<SmallVec<[_; 3]>>();
        debug_assert_eq!(child_main_costs.len(), imp.child_count(spec));
        let main_cost: MainCost = compute_cost_node(spec, imp, &child_main_costs);
        // TODO: Handle other kinds of memory, not just standard/TinyMap peaks.
        let raised_peaks =
            snap_availables_up_memvec(imp.peak_memory_from_child_peaks(spec, &child_peaks), false);
        Cost {
            main: main_cost,
            peaks: raised_peaks,
            depth: 1 + child_costs.iter().map(|k| k.depth).max().unwrap_or(0),
        }
    }
}

pub fn move_cost<Tgt: Target>(
    src: &TensorSpec<Tgt>,
    dest: &TensorSpec<Tgt>,
    prefetching: bool,
) -> MainCost {
    let src_hit_cost = src.level().cache_hit_cost();
    let dest_hit_cost = dest.level().cache_hit_cost();

    let src_cache_lines = MainCost::from(src.layout().estimate_cache_lines::<Tgt>(
        src.dim_sizes(),
        src.dtype(),
        src.is_contiguous(),
    ));
    let dest_cache_lines = MainCost::from(dest.layout().estimate_cache_lines::<Tgt>(
        dest.dim_sizes(),
        dest.dtype(),
        dest.is_contiguous(),
    ));

    let src_cost = 10 * (src_hit_cost * src_cache_lines);
    let dest_cost = 10 * (dest_hit_cost * dest_cache_lines);

    let mut cost: MainCost = src_cost + dest_cost;
    if prefetching {
        cost /= 2;
    }
    if !src.is_contiguous() || src.layout() != dest.layout() {
        cost *= 2;
    }
    cost
}

fn compute_cost_node<Tgt: Target>(
    spec: &Spec<Tgt>,
    op: &ImplNode<Tgt>,
    child_costs: &[MainCost],
) -> MainCost {
    match op {
        ImplNode::Pipeline { .. } | ImplNode::AccumBlock | ImplNode::SpatialSplit => {
            child_costs.iter().sum()
        }
        ImplNode::Loop { parallel, .. } => {
            let factor = if *parallel {
                let processors = Tgt::processors() as u32;
                let steps = op.steps(spec);
                let main_steps = op.full_steps(spec);
                ((main_steps + processors - 1) / processors) + (steps - main_steps)
            } else {
                op.steps(spec)
            };
            cmp::min(child_costs[0] * MainCost::from(factor), MAX_COST)
        }
        ImplNode::MoveLet {
            source_idx,
            destination_level,
            destination_layout,
            destination_vector_shape,
            prefetch,
        } => {
            let op_operands = spec.operands();
            let src = &op_operands[usize::from(*source_idx)];
            let mcost = move_cost(
                src,
                &movelet_inner_tensorspec(
                    src,
                    destination_level,
                    destination_layout,
                    destination_vector_shape.as_ref().map(|v| v.as_slice()),
                ),
                *prefetch,
            );
            child_costs.iter().sum::<MainCost>() + mcost
        }
        ImplNode::Mult | ImplNode::BroadcastVecMult => INST_COST,
        ImplNode::ValueAssign
        | ImplNode::VectorAssign
        | ImplNode::MemsetZero
        | ImplNode::VectorZero => ASSIGN_INST_COST,
    }
}
