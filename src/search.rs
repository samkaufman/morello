use itertools::Itertools;
use smallvec::{smallvec, SmallVec};

use crate::cost::Cost;
use crate::db::{ActionIdx, Database};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::imp::{visit_leaves, ImplNode};
use crate::memorylimits::MemoryLimits;
use crate::spec::{LogicalSpec, Spec};
use crate::target::Target;

struct ImplReducer {
    results: SmallVec<[(ActionIdx, Cost); 1]>,
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
pub fn top_down<'d, Tgt, D>(
    db: &'d D,
    goal: &Spec<Tgt>,
    top_k: usize,
) -> (SmallVec<[(ActionIdx, Cost); 1]>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
    D: Database<'d>,
{
    let (actions, hits, misses) = top_down_inner(db, goal, top_k, 0, &ParentSummary::new(goal));
    (actions.as_ref().clone(), hits, misses)
}

fn top_down_inner<'d, Tgt, D>(
    db: &'d D,
    goal: &Spec<Tgt>,
    top_k: usize,
    depth: usize,
    parent_summary: &ParentSummary<Tgt>,
) -> (D::ValueRef, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
    D: Database<'d>,
{
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // First, check if the Spec is already in the database.
    if let Some(stored) = db.get(goal) {
        return (stored, 1, 0);
    }

    let mut hits = 0u64;
    let mut misses = 1u64;

    // Enumerate action applications, computing their costs from their childrens' costs.
    let mut reducer = ImplReducer::new(top_k);

    for (action_idx, action) in goal.0.actions().into_iter().enumerate() {
        let Ok(partial_impl) = action.apply(goal) else {
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
            if child_results.as_ref().is_empty() {
                unsat = true;
                return false;
            }
            nested_spec_costs.push(child_results.as_ref()[0].1.clone()); // TODO: Move, don't clone
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
        reducer.insert(action_idx.try_into().unwrap(), cost);
    }

    // Save to memo. table and return.
    let final_result = reducer.finalize();
    let final_result_ref = db.put(goal.clone(), final_result);
    (final_result_ref, hits, misses)
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

impl ImplReducer {
    fn new(top_k: usize) -> Self {
        ImplReducer {
            results: smallvec![],
            top_k,
        }
    }

    fn insert(&mut self, new_impl: ActionIdx, cost: Cost) {
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

    fn finalize(self) -> SmallVec<[(ActionIdx, Cost); 1]> {
        self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::layout::{row_major, Layout};
    use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{CpuMemoryLevel, X86Target};
    use crate::tensorspec::TensorSpecAux;
    use smallvec::smallvec;

    #[test]
    fn test_parentsummary_doesnt_prune_rm_to_cm_relayout() {
        let cm = Layout::Standard {
            dim_order: smallvec![1, 0],
        };
        let spec1 = example_zero_spec(CpuMemoryLevel::GL, row_major(2));
        let spec2 = example_zero_spec(CpuMemoryLevel::GL, cm);
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
            example_move_spec(CpuMemoryLevel::GL, row_major(2), CpuMemoryLevel::GL, cm);
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
            CpuMemoryLevel::GL,
            row_major(2),
            CpuMemoryLevel::GL,
            cm.clone(),
        );
        let spec2 = example_move_spec(CpuMemoryLevel::GL, cm, CpuMemoryLevel::GL, row_major(2));
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
        from_level: CpuMemoryLevel,
        from_layout: Layout,
        to_level: CpuMemoryLevel,
        to_layout: Layout,
    ) -> Spec<X86Target> {
        Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    spec_shape: smallvec![4, 4],
                    dtype: Dtype::Uint8,
                },
                vec![
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
                ],
                false,
            ),
            X86Target::max_mem(),
        )
    }

    fn example_zero_spec(level: CpuMemoryLevel, layout: Layout) -> Spec<X86Target> {
        Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    spec_shape: smallvec![4, 4],
                    dtype: Dtype::Uint8,
                },
                vec![TensorSpecAux {
                    contig: layout.contiguous_full(),
                    aligned: false,
                    level,
                    layout,
                    vector_size: None,
                }],
                false,
            ),
            X86Target::max_mem(),
        )
    }
}
