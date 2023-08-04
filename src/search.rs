use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use std::sync::RwLock;

use crate::common::Spec;
use crate::cost::Cost;
use crate::imp::{visit_leaves, Impl, ImplNode};
use crate::memorylimits::MemoryLimits;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use crate::table::Database;
use crate::target::Target;

struct ImplReducer<Tgt: Target> {
    results: SmallVec<[(Action<Tgt>, Cost); 1]>,
    top_k: usize,
}

/// A summary of a sequence of Specs to determine whether a sub-Spec should be pruned during search.
#[derive(Clone)]
struct ParentSummary<'a, Tgt: Target> {
    seen: &'a LogicalSpec<Tgt>,
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

            let summary_to_forward = match parent_summary.transition(nested_spec) {
                ParentSummaryTransitionResult::PruneAction => {
                    unsat = true;
                    return false;
                }
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
        let MemoryLimits::Standard(goal_vec) = &goal.1;
        debug_assert!(
            cost.peaks.iter().zip(goal_vec).all(|(a, b)| *a <= b),
            "While synthesizing {:?}, action yielded memory \
            bound-violating {:?} with peak memory {:?}",
            goal,
            partial_impl,
            cost.peaks
        );
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
            seen: &initial_spec.0,
            tail: None,
        }
    }

    /// Determine whether the given sub-Spec should be pruned during search.
    ///
    /// If this method returns [ParentSummaryTransitionResult::PruneAction], the caller should
    /// prune the given sub-Spec. If [ParentSummaryTransitionResult::NewSummary] is returned,
    /// the wrapped [ParentSummary] should be used for nested children.
    fn transition(&'a self, nested_spec: &'a Spec<Tgt>) -> ParentSummaryTransitionResult<'a, Tgt> {
        // Prune any cycles. These can result from within-level cycles between layouts, not just in
        // terms of introduced Move Specs, but the non-Move/moved "body" Specs as well.
        let mut cur_option = Some(self);
        while let Some(cur) = cur_option {
            if cur.seen == &nested_spec.0 {
                return ParentSummaryTransitionResult::PruneAction;
            }
            cur_option = cur.tail;
        }

        ParentSummaryTransitionResult::NewSummary(ParentSummary {
            seen: &nested_spec.0,
            tail: Some(self),
        })
    }
}

impl<Tgt: Target> ImplReducer<Tgt> {
    fn new(top_k: usize) -> Self {
        ImplReducer {
            results: smallvec![],
            top_k,
        }
    }

    fn insert(&mut self, new_impl: Action<Tgt>, cost: Cost) {
        match self.results.binary_search_by_key(&&cost, |imp| &imp.1) {
            Ok(idx) | Err(idx) => {
                if idx < self.top_k {
                    if self.results.len() == self.top_k {
                        self.results.pop();
                    }
                    self.results.insert(idx, (new_impl, cost));
                }
            }
        }
        debug_assert!(self.results.iter().tuple_windows().all(|(a, b)| a.1 < b.1));
        debug_assert!(self.results.len() <= self.top_k);
    }

    fn finalize(self) -> SmallVec<[(Action<Tgt>, Cost); 1]> {
        self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::layout::{row_major, Layout};
    use crate::spec::{LogicalSpec, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{X86MemoryLevel, X86Target};
    use crate::tensorspec::TensorSpecAux;
    use smallvec::smallvec;

    #[test]
    fn test_parentsummary_doesnt_prune_rm_to_cm_relayout() {
        let cm = Layout::Standard {
            dim_order: smallvec![1, 0],
        };
        let spec1 = example_zero_spec(X86MemoryLevel::GL, row_major(2));
        let spec2 = example_zero_spec(X86MemoryLevel::GL, cm);
        let s1 = ParentSummary::new(&spec1);
        assert!(matches!(
            s1.transition(&spec2),
            ParentSummaryTransitionResult::NewSummary(_)
        ));
    }

    #[test]
    fn test_parentsummary_prunes_immediate_move_cycles() {
        let cm = Layout::Standard {
            dim_order: smallvec![1, 0],
        };
        let initial_spec =
            example_move_spec(X86MemoryLevel::GL, row_major(2), X86MemoryLevel::GL, cm);
        let s1 = ParentSummary::new(&initial_spec);
        assert!(matches!(
            s1.transition(&initial_spec),
            ParentSummaryTransitionResult::PruneAction
        ));
    }

    #[test]
    fn test_parentsummary_prunes_one_step_cycle() {
        let cm = Layout::Standard {
            dim_order: smallvec![1, 0],
        };
        let spec1 = example_move_spec(
            X86MemoryLevel::GL,
            row_major(2),
            X86MemoryLevel::GL,
            cm.clone(),
        );
        let spec2 = example_move_spec(X86MemoryLevel::GL, cm, X86MemoryLevel::GL, row_major(2));
        let s1 = ParentSummary::new(&spec1);
        let ParentSummaryTransitionResult::NewSummary(s2) = s1.transition(&spec2) else {
            panic!();
        };
        assert!(matches!(
            s2.transition(&spec1),
            ParentSummaryTransitionResult::PruneAction
        ));
    }
    fn example_move_spec(
        from_level: X86MemoryLevel,
        from_layout: Layout,
        to_level: X86MemoryLevel,
        to_layout: Layout,
    ) -> Spec<X86Target> {
        Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    spec_shape: smallvec![4, 4],
                    dtype: Dtype::Uint8,
                },
                PrimitiveAux(vec![
                    TensorSpecAux {
                        contig: from_layout.contiguous_full(),
                        aligned: false,
                        level: from_level,
                        layout: from_layout,
                        vector_size: None,
                    },
                    TensorSpecAux {
                        contig: to_layout.contiguous_full(),
                        aligned: false,
                        level: to_level,
                        layout: to_layout,
                        vector_size: None,
                    },
                ]),
                false,
            ),
            X86Target::max_mem(),
        )
    }

    fn example_zero_spec(level: X86MemoryLevel, layout: Layout) -> Spec<X86Target> {
        Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    spec_shape: smallvec![4, 4],
                    dtype: Dtype::Uint8,
                },
                PrimitiveAux(vec![TensorSpecAux {
                    contig: layout.contiguous_full(),
                    aligned: false,
                    level,
                    layout,
                    vector_size: None,
                }]),
                false,
            ),
            X86Target::max_mem(),
        )
    }
}
