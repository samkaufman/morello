use smallvec::SmallVec;
use std::sync::RwLock;

use crate::common::Spec;
use crate::cost::Cost;
use crate::imp::{visit_leaves, Impl, ImplNode};
use crate::memorylimits::MemoryLimits;
use crate::scheduling::Action;
use crate::table::Database;
use crate::target::Target;

struct ImplReducer<Tgt: Target> {
    results: Vec<(Action<Tgt>, Cost)>,
    top_k: usize,
}

#[derive(Clone)]
struct ParentSummary<'a, Tgt: Target> {
    seen: &'a Spec<Tgt>,
    tail: Option<&'a ParentSummary<'a, Tgt>>,
}

enum ParentSummaryTransitionResult<'a, Tgt: Target> {
    PruneAction,
    NewSummary(ParentSummary<'a, Tgt>),
}

/// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<'d, Tgt: Target, D: Database<Tgt> + 'd>(
    db: &RwLock<D>,
    goal: &Spec<Tgt>,
    top_k: usize,
) -> (SmallVec<[(Action<Tgt>, Cost); 1]>, u64, u64) {
    top_down_inner(db, goal, top_k, 0, &ParentSummary::new(goal))
}

fn top_down_inner<'d, Tgt: Target, D: Database<Tgt> + 'd>(
    db: &RwLock<D>,
    goal: &Spec<Tgt>,
    top_k: usize,
    depth: usize,
    parent_summary: &ParentSummary<Tgt>,
) -> (SmallVec<[(Action<Tgt>, Cost); 1]>, u64, u64) {
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // First, check if the Spec is already in the database.
    if let Some(stored) = db.read().unwrap().get(goal) {
        return (stored.clone(), 1, 0);
    }

    let mut hits = 0u64;
    let mut misses = 1u64;

    // Enumerate action applications, computing their costs from their childrens' costs.
    let mut reducer = ImplReducer::new(top_k);

    for action in goal.0.actions() {
        // TODO: Integrate transition logic into `apply`.
        let Some(partial_impl) = action.apply(goal) else {
            continue;
        };

        // Compute all nested Specs, accumulating their costs into nested_spec_costs.
        let mut unsat = false;
        let mut nested_spec_costs = Vec::new();
        visit_leaves(&partial_impl, &mut |leaf| {
            let ImplNode::SpecApp(spec_app) = leaf else {
                return true;
            };
            let nested_spec = &spec_app.0;

            assert_ne!(
                goal, nested_spec,
                "Immediate recursion on ({}, {:?}) after applying {:?}",
                goal.0, goal.1, action
            );

            let summary_to_forward = match parent_summary.transition(&action, nested_spec) {
                ParentSummaryTransitionResult::PruneAction => return false,
                ParentSummaryTransitionResult::NewSummary(new_summary) => new_summary,
            };
            let (child_results, subhits, submisses) =
                top_down_inner(db, nested_spec, top_k, depth + 1, &summary_to_forward);
            hits += subhits;
            misses += submisses;
            if child_results.is_empty() {
                unsat = true;
                return false;
            }
            nested_spec_costs.push(child_results[0].1.clone()); // TODO: Should move
            true
        });
        if unsat {
            continue;
        }

        let cost = Cost::from_child_costs(&partial_impl, &nested_spec_costs);
        if let MemoryLimits::Standard(g) = &goal.1 {
            debug_assert!(
                cost.peaks.iter().zip(g).all(|(a, b)| *a <= b),
                "While synthesizing {:?}, action yielded memory \
                bound-violating {:?} with peak memory {:?}",
                goal,
                partial_impl,
                cost.peaks
            );
        }
        reducer.insert(action, cost);
    }

    // Save to memo. table and return.
    let final_result = reducer.finalize();
    db.write().unwrap().put(goal.clone(), final_result.clone());
    (final_result, hits, misses)
}

impl<'a, Tgt: Target> ParentSummary<'a, Tgt> {
    fn new(initial_spec: &'a Spec<Tgt>) -> Self {
        ParentSummary {
            seen: initial_spec,
            tail: None,
        }
    }

    fn transition(
        &'a self,
        _via_action: &Action<Tgt>,
        nested_spec: &'a Spec<Tgt>,
    ) -> ParentSummaryTransitionResult<'a, Tgt> {
        let mut cur_option = Some(self);
        while let Some(cur) = cur_option {
            if nested_spec == cur.seen {
                return ParentSummaryTransitionResult::PruneAction;
            }
            cur_option = cur.tail;
        }

        ParentSummaryTransitionResult::NewSummary(ParentSummary {
            seen: nested_spec,
            tail: Some(self),
        })
    }
}

impl<Tgt: Target> ImplReducer<Tgt> {
    fn new(top_k: usize) -> Self {
        ImplReducer {
            results: vec![],
            top_k,
        }
    }

    fn insert(&mut self, new_impl: Action<Tgt>, cost: Cost) {
        self.results.push((new_impl, cost));
    }

    fn finalize(&self) -> SmallVec<[(Action<Tgt>, Cost); 1]> {
        // Using sorted here for stability.
        let mut sorted_results = self.results.clone();
        sorted_results.sort_by_key(|x| x.1.clone());
        sorted_results.truncate(self.top_k);
        sorted_results.into()
    }
}
