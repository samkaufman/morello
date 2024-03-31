use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use std::num::NonZeroUsize;

use crate::cost::Cost;
use crate::db::{ActionIdx, Database, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryLimits;
use crate::scheduling::ApplyError;
use crate::spec::Spec;
use crate::target::Target;

struct ImplReducer<'a> {
    results: SmallVec<[(ActionIdx, Cost); 1]>,
    top_k: usize,
    preferences: &'a [ActionIdx],
}

/// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<'d, Tgt, D>(
    db: &'d D,
    goal: &Spec<Tgt>,
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> (SmallVec<[(ActionIdx, Cost); 1]>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'d> + Send + Sync,
{
    assert!(db.max_k().map_or(true, |k| k >= top_k));

    let mut canonical_goal = goal.clone();
    canonical_goal
        .canonicalize()
        .expect("should be possible to canonicalize goal Spec");

    let thread_count = jobs
        .map(|j| j.get())
        .unwrap_or_else(rayon::current_num_threads);
    if thread_count == 1 {
        return top_down_spec(db, &canonical_goal, top_k, 0, 1);
    }

    let tasks = (0..thread_count)
        .zip(std::iter::repeat(canonical_goal.clone()))
        .collect::<Vec<_>>();
    // Collect all and take the result from the first call so that we get
    // deterministic results.
    tasks
        .into_par_iter()
        .map(|(i, g)| top_down_spec(db, &g, top_k, i, thread_count))
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
    let get_result = db.get_with_preference(goal);
    let mut preferences: &[_] = &[];
    if let GetPreference::Hit(v) = get_result {
        return (v.0, 1, 0);
    }
    if let GetPreference::Miss(Some(p)) = &get_result {
        preferences = p;
    }

    let mut hits = 0u64;
    let mut misses = 1u64;

    // Enumerate action applications, computing their costs from their childrens' costs.
    let mut reducer = ImplReducer::new(top_k, preferences);

    let all_actions = goal.0.actions().into_iter().collect::<Vec<_>>();

    let initial_skip = thread_idx * all_actions.len() / thread_count;
    for action_idx in (initial_skip..all_actions.len()).chain(0..initial_skip) {
        let action = &all_actions[action_idx];
        let partial_impl = match action.apply(goal) {
            Ok(imp) => imp,
            Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => continue,
            Err(ApplyError::SpecNotCanonical) => panic!("Goal was not canonical: {goal}"),
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

impl<'a> ImplReducer<'a> {
    fn new(top_k: usize, preferences: &'a [ActionIdx]) -> Self {
        ImplReducer {
            results: smallvec![],
            top_k,
            preferences,
        }
    }

    fn insert(&mut self, new_action_idx: ActionIdx, cost: Cost) {
        match self.results.binary_search_by_key(&&cost, |imp| &imp.1) {
            Ok(idx) => {
                debug_assert!(idx < self.top_k);
                // Replace something if it improves preference count, and do
                //   nothing if not.
                let mut to_set = None;
                for i in self.iter_surrounding_matching_cost_indices(idx, &cost) {
                    // TODO: Instead of filtering here, just push down the length.
                    if i >= self.preferences.len() {
                        continue;
                    }

                    if new_action_idx == self.preferences[i] {
                        to_set = Some(i);
                        break;
                    }
                }
                if let Some(i) = to_set {
                    self.results[i].0 = new_action_idx;
                }

                debug_assert!(self.results.len() <= self.top_k);
            }
            Err(idx) if idx < self.top_k => {
                if self.results.len() == self.top_k {
                    self.results.pop();
                }
                self.results.insert(idx, (new_action_idx, cost));
            }
            Err(_) => {}
        }
        debug_assert!(self.results.iter().tuple_windows().all(|(a, b)| a.1 < b.1));
        debug_assert!(self.results.len() <= self.top_k);
        debug_assert!(self.results.iter().map(|(a, _)| a).all_unique());
    }

    fn finalize(self) -> SmallVec<[(ActionIdx, Cost); 1]> {
        self.results
    }

    fn iter_surrounding_matching_cost_indices<'s>(
        &'s self,
        initial_idx: usize,
        cost: &'s Cost,
    ) -> impl Iterator<Item = usize> + 's {
        debug_assert_eq!(&self.results[initial_idx].1, cost);
        std::iter::once(initial_idx)
            .chain(
                ((initial_idx + 1)..self.results.len())
                    .take_while(move |idx| &self.results[*idx].1 == cost),
            )
            .chain(
                (0..initial_idx)
                    .rev()
                    .take_while(move |idx| &self.results[*idx].1 == cost),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{DimSize, Dtype};
    use crate::db::DashmapDiskDatabase;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::memorylimits::MemVec;
    use crate::spec::{arb_canonical_spec, LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{CpuMemoryLevel::GL, X86Target};
    use crate::tensorspec::TensorSpecAux;
    use crate::utils::{bit_length, bit_length_inverse};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use proptest::sample::select;
    use std::rc::Rc;

    const TEST_SMALL_SIZE: DimSize = 2;
    const TEST_SMALL_MEM: u64 = 2048;

    proptest! {
        // TODO: Add an ARM variant!
        // TODO: Remove restriction to canonical Specs. Should synth. any Spec.
        #[test]
        #[ignore]
        fn test_can_synthesize_any_canonical_spec(
            spec in arb_canonical_spec::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = DashmapDiskDatabase::try_new(None, false, 1).unwrap();
            top_down(&db, &spec, 1, Some(nz!(1usize)));
        }

        #[test]
        #[ignore]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_canonical_specs::<X86Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = DashmapDiskDatabase::try_new(None, false, 1).unwrap();

            // Solve the first, lower Spec.
            let (lower_result_vec, _, _) = top_down(&db, &spec, 1, Some(nz!(1usize)));

            // If the lower spec can't be solved, then there is no way for the raised Spec to have
            // a worse solution, so we can return here.
            if let Some((_, lower_cost)) = lower_result_vec.first() {
                // Check that the raised result has no lower cost and does not move from being
                // possible to impossible.
                let (raised_result, _, _) = top_down(&db, &raised_spec, 1, Some(nz!(1usize)));
                let (_, raised_cost) = raised_result
                    .first()
                    .expect("raised result should be possible");
                assert!(raised_cost <= lower_cost);
            }
        }

        #[test]
        #[ignore]
        fn test_synthesis_at_peak_memory_yields_same_decision(
            spec in arb_canonical_spec::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = DashmapDiskDatabase::try_new(None, false, 1).unwrap();
            let (first_solutions, _, _) = top_down(&db, &spec, 1, Some(nz!(1usize)));
            let first_peak = if let Some(first_sol) = first_solutions.first() {
                first_sol.1.peaks.clone()
            } else {
                MemVec::zero::<X86Target>()
            };
            let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
            let (lower_solutions, _, _) = top_down(&db, &lower_spec, 1, Some(nz!(1usize)));
            assert_eq!(first_solutions, lower_solutions);
        }
    }

    #[test]
    fn test_synthesis_at_peak_memory_yields_same_decision_1() {
        let spec = Spec::<X86Target>(
            lspec!(Zero([2, 2, 2, 2], (u8, GL, row_major(4), c0, ua))),
            MemoryLimits::Standard(MemVec::new_from_binary_scaled([0, 5, 7, 6])),
        );

        let db = DashmapDiskDatabase::try_new(None, false, 1).unwrap();
        let (first_solutions, _, _) = top_down(&db, &spec, 1, Some(nz!(1usize)));
        let first_peak = if let Some(first_sol) = first_solutions.first() {
            first_sol.1.peaks.clone()
        } else {
            MemVec::zero::<X86Target>()
        };
        let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
        let (lower_solutions, _, _) = top_down(&db, &lower_spec, 1, Some(nz!(1usize)));
        assert_eq!(first_solutions, lower_solutions);
    }

    fn lower_and_higher_canonical_specs<Tgt: Target>(
    ) -> impl Strategy<Value = (Spec<Tgt>, Spec<Tgt>)> {
        let MemoryLimits::Standard(mut top_memvec) = X86Target::max_mem();
        top_memvec = top_memvec.map(|v| v.min(TEST_SMALL_MEM));

        let top_memory_a = Rc::new(MemoryLimits::Standard(top_memvec));
        let top_memory_b = Rc::clone(&top_memory_a);
        let top_memory_c = Rc::clone(&top_memory_a);

        arb_canonical_spec::<Tgt>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
            .prop_filter("limits should not be max", move |s| s.1 != *top_memory_a)
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
