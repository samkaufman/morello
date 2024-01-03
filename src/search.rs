use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use std::panic::{self, AssertUnwindSafe};

use crate::cost::Cost;
use crate::db::{ActionIdx, Database};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryLimits;
use crate::spec::Spec;
use crate::target::Target;

struct ImplReducer {
    results: SmallVec<[(ActionIdx, Cost); 1]>,
    top_k: usize,
}

/// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<'d, Tgt, D>(
    db: &'d D,
    goal: &Spec<Tgt>,
    top_k: usize,
    parallel: bool,
) -> (SmallVec<[(ActionIdx, Cost); 1]>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'d> + Send + Sync,
{
    assert!(db.max_k().map_or(true, |k| k >= top_k));

    if !parallel {
        return top_down_spec(db, goal, top_k, 0, 1);
    }

    let thread_count = rayon::current_num_threads();
    let tasks = (0..thread_count)
        .zip(std::iter::repeat(goal.clone()))
        .collect::<Vec<_>>();
    // Collect all and take the result from the first call so that we get
    // deterministic results.
    tasks
        .into_par_iter()
        .map(|(i, g)| {
            // Exit immediately if any thread panics.  Since the type `D` (db)
            // in top_down_spec contains interior mutability, we need to wrap
            // it in AssertUnwindSafe to ensure that it is safe to transfer
            // across threads.  Because we are exiting immediately on panic,
            // we can safely use AssertUnwindSafe.
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                top_down_spec(db, &g, top_k, i, thread_count)
            }));
            match result {
                Ok(val) => val,
                Err(_) => std::process::exit(1),
            }
        })
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}

fn top_down_spec<'d, Tgt, D>(
    db: &'d D,
    goal: &Spec<Tgt>,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
) -> (SmallVec<[(ActionIdx, Cost); 1]>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'d>,
{
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // First, check if the Spec is already in the database.
    if let Some(stored) = db.get(goal) {
        return (stored.0, 1, 0);
    }

    let mut hits = 0u64;
    let mut misses = 1u64;

    // Enumerate action applications, computing their costs from their childrens' costs.
    let mut reducer = ImplReducer::new(top_k);

    let all_actions = goal.0.actions().into_iter().collect::<Vec<_>>();

    let initial_skip = thread_idx * all_actions.len() / thread_count;
    for action_idx in (initial_skip..all_actions.len()).chain(0..initial_skip) {
        let action = &all_actions[action_idx];
        let Ok(partial_impl) = action.apply(goal) else {
            continue;
        };

        // Let top_down_impl compute the final cost of completing this Impl.
        let (costs, action_impl_hits, action_impl_misses) =
            top_down_impl(db, &partial_impl, top_k, thread_idx, thread_count);
        hits += action_impl_hits;
        misses += action_impl_misses;
        if costs.is_empty() {
            continue;
        }

        let MemoryLimits::Standard(goal_vec) = &goal.1;
        debug_assert!(
            costs[0]
                .peaks
                .iter()
                .zip(goal_vec.iter())
                .all(|(a, b)| a <= b),
            "While synthesizing {:?}, action yielded memory \
            bound-violating {:?} with peak memory {:?}",
            goal,
            partial_impl,
            costs[0].peaks
        );
        reducer.insert(
            action_idx.try_into().unwrap(),
            costs.into_iter().exactly_one().unwrap(),
        );
    }

    // Copy into the memo. table and return.
    let final_result = reducer.finalize();
    db.put(goal.clone(), final_result.clone());
    (final_result, hits, misses)
}

fn top_down_impl<'d, Tgt, D, Aux>(
    db: &'d D,
    partial_impl: &ImplNode<Tgt, Aux>,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
) -> (SmallVec<[Cost; 1]>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'d>,
    Aux: Clone,
{
    if let ImplNode::SpecApp(spec_app) = partial_impl {
        let (action_costs, hits, misses) =
            top_down_spec(db, &spec_app.0, top_k, thread_idx, thread_count);
        (
            action_costs.into_iter().map(|(_, c)| c).collect(),
            hits,
            misses,
        )
    } else {
        let mut hits = 0;
        let mut misses = 0;
        let mut child_costs: SmallVec<[Cost; 3]> =
            SmallVec::with_capacity(partial_impl.children().len());
        for child_node in partial_impl.children() {
            let (mut child_results, subhits, submisses) =
                top_down_impl(db, child_node, top_k, thread_idx, thread_count);
            hits += subhits;
            misses += submisses;
            if child_results.is_empty() {
                // Unsatisfiable.
                return (smallvec![], hits, misses);
            } else if child_results.len() == 1 {
                child_costs.append(&mut child_results);
            } else {
                todo!("support k > 1");
            }
        }
        let partial_impl_cost = Cost::from_child_costs(partial_impl, &child_costs);
        (smallvec![partial_impl_cost], hits, misses)
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
    use crate::common::{DimSize, Dtype};
    use crate::db::DashmapDiskDatabase;
    use crate::layout::{row_major, Layout};
    use crate::memorylimits::MemVec;
    use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{CpuMemoryLevel, X86Target};
    use crate::tensorspec::TensorSpecAux;
    use crate::utils::{bit_length, bit_length_inverse};
    use proptest::prelude::*;
    use proptest::sample::select;
    use smallvec::smallvec;
    use std::rc::Rc;

    const TEST_SMALL_SIZE: DimSize = 2;
    const TEST_SMALL_MEM: u64 = 2048;

    proptest! {
        #[test]
        fn test_can_synthesize_any_spec(
            spec in any_with::<Spec<X86Target>>((Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM)))
        ) {
            let db = DashmapDiskDatabase::new(None, false, 1);
            top_down(&db, &spec, 1, false);
        }

        #[test]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_spec::<X86Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = DashmapDiskDatabase::new(None, false, 1);

            // Solve the first, lower Spec.
            let (lower_result_vec, _, _) = top_down(&db, &spec, 1, false);

            // If the lower spec can't be solved, then there is no way for the raised Spec to have
            // a worse solution, so we can return here.
            if let Some((_, lower_cost)) = lower_result_vec.first() {
                // Check that the raised result has no lower cost and does not move from being
                // possible to impossible.
                let (raised_result, _, _) = top_down(&db, &raised_spec, 1, false);
                let (_, raised_cost) = raised_result
                    .first()
                    .expect("raised result should be possible");
                assert!(raised_cost <= lower_cost);
            }
        }

        #[test]
        fn test_synthesis_at_peak_memory_yields_same_decision(
            spec in any_with::<Spec<X86Target>>((Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM)))
        ) {
            let db = DashmapDiskDatabase::new(None, false, 1);
            let (first_solutions, _, _) = top_down(&db, &spec, 1, false);
            let first_peak = if let Some(first_sol) = first_solutions.first() {
                first_sol.1.peaks.clone()
            } else {
                MemVec::zero::<X86Target>()
            };
            let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
            let (lower_solutions, _, _) = top_down(&db, &lower_spec, 1, false);
            assert_eq!(first_solutions, lower_solutions);
        }
    }

    #[test]
    fn test_synthesis_at_peak_memory_yields_same_decision_1() {
        let spec = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    spec_shape: smallvec![2, 2, 2, 2],
                    dtype: Dtype::Uint8,
                },
                vec![TensorSpecAux::<X86Target> {
                    contig: 0,
                    aligned: false,
                    level: CpuMemoryLevel::GL,
                    layout: row_major(4),
                    vector_size: None,
                }],
                false,
            ),
            MemoryLimits::Standard(MemVec::new_from_binary_scaled([0, 5, 7, 6])),
        );

        let db = DashmapDiskDatabase::new(None, false, 1);
        let (first_solutions, _, _) = top_down(&db, &spec, 1, false);
        let first_peak = if let Some(first_sol) = first_solutions.first() {
            first_sol.1.peaks.clone()
        } else {
            MemVec::zero::<X86Target>()
        };
        let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
        let (lower_solutions, _, _) = top_down(&db, &lower_spec, 1, false);
        assert_eq!(first_solutions, lower_solutions);
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

    fn lower_and_higher_spec<Tgt: Target>() -> impl Strategy<Value = (Spec<Tgt>, Spec<Tgt>)> {
        let MemoryLimits::Standard(mut top_memvec) = X86Target::max_mem();
        top_memvec = top_memvec.map(|v| v.min(TEST_SMALL_MEM));

        let top_memory_a = Rc::new(MemoryLimits::Standard(top_memvec));
        let top_memory_b = Rc::clone(&top_memory_a);
        let top_memory_c = Rc::clone(&top_memory_a);

        any_with::<Spec<Tgt>>((Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM)))
            .prop_filter("limits should not be max", move |s| &s.1 != &*top_memory_a)
            .prop_flat_map(move |spec| {
                let MemoryLimits::Standard(top_memvec) = top_memory_b.as_ref();
                let MemoryLimits::Standard(raised_memory) = &spec.1;
                let non_top_levels = (0..raised_memory.len())
                    .filter(|&idx| raised_memory.get_unscaled(idx) < top_memvec.get_unscaled(idx))
                    .collect::<Vec<_>>();
                (Just(spec), select(non_top_levels))
            })
            .prop_flat_map(move |(spec, dim_idx_to_raise)| {
                let MemoryLimits::Standard(top_memvec) = top_memory_c.as_ref();
                let MemoryLimits::Standard(spec_memvec) = &spec.1;

                let low = bit_length(spec_memvec.get_unscaled(dim_idx_to_raise));
                let high = bit_length(top_memvec.get_unscaled(dim_idx_to_raise));
                (Just(spec), Just(dim_idx_to_raise), (low + 1)..=high)
            })
            .prop_map(|(spec, dim_idx_to_raise, raise_bits)| {
                let raise_amount = bit_length_inverse(raise_bits);
                let mut raised_memory = spec.1.clone();
                let MemoryLimits::Standard(ref mut raised_memvec) = raised_memory;
                raised_memvec.set_unscaled(dim_idx_to_raise, raise_amount);
                let raised_spec = Spec(spec.0.clone(), raised_memory);
                (spec, raised_spec)
            })
    }
}
