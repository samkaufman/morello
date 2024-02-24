use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use std::num::NonZeroUsize;

use crate::cost::Cost;
use crate::db::{ActionIdx, Database, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::{Impl, ImplNode};
use crate::scheduling::ApplyError;
use crate::spec::Spec;
use crate::target::Target;

struct TopDownSearch<'d, D> {
    db: &'d D,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
    hits: u64,
    misses: u64,
}

/// On-going synthesis of a [Spec].
///
/// Synthesizing a [Spec] is a two-step process. First, the [Spec] expands all actions into [Impl]s
/// and stores the incomplete [Impl]s. Seconds, the caller collects those dependencies from the
/// database or recursively synthesizes them, providing them to this [SpecTask]. This struct stores
/// state between those two steps.
#[derive(Debug)]
struct SpecTask<Tgt: Target> {
    goal: Spec<Tgt>, // TODO: Do we really need to store the `goal`?
    reducer: ImplReducer,
    partial_impls: Vec<WorkingPartialImpl<Tgt>>,
    partial_impls_remaining: usize,
}

#[derive(Debug)]
enum WorkingPartialImpl<Tgt: Target> {
    Constructing {
        partial_impl: ImplNode<Tgt, ()>,
        subspec_costs: Vec<Option<Cost>>, // empty = unsat
        producing_action_idx: ActionIdx,
    },
    Unsat,
}

#[derive(Debug, Clone)] // TODO: Remove this Clone derive if not needed
struct ImplReducer {
    results: SmallVec<[(ActionIdx, Cost); 1]>,
    top_k: usize, // TODO: Shared between ImplReducers. Pull out?
    preferences: SmallVec<[ActionIdx; 1]>,
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
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    let mut canonical_goal = goal.clone();
    canonical_goal
        .canonicalize()
        .expect("should be possible to canonicalize goal Spec");

    let thread_count = jobs
        .map(|j| j.get())
        .unwrap_or_else(rayon::current_num_threads);
    if thread_count == 1 {
        let mut search = TopDownSearch::<'d, D> {
            db,
            top_k,
            thread_idx: 0,
            thread_count: 1,
            hits: 0,
            misses: 1,
        };
        return (search.run_spec(&canonical_goal), search.hits, search.misses);
    }

    let tasks = (0..thread_count)
        .zip(std::iter::repeat(canonical_goal.clone()))
        .collect::<Vec<_>>();
    // Collect all and take the result from the first call so that we get
    // deterministic results.
    tasks
        .into_par_iter()
        .map(|(i, g)| {
            let mut search = TopDownSearch::<'d, D> {
                db,
                top_k,
                thread_idx: i,
                thread_count,
                hits: 0,
                misses: 1,
            };
            (search.run_spec(&g), search.hits, search.misses)
        })
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}

impl<'d, D> TopDownSearch<'d, D>
where
    D: Database<'d> + Send + Sync,
{
    fn run_spec<Tgt>(&mut self, goal: &Spec<Tgt>) -> SmallVec<[(ActionIdx, Cost); 1]>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        // First, check if the initial goal Spec is already in the database.
        let preferences = match self.db.get_with_preference(goal) {
            GetPreference::Hit(v) => {
                self.hits += 1;
                return v.0;
            }
            GetPreference::Miss(m) => {
                self.misses += 1;
                m.unwrap_or_else(SmallVec::new)
            }
        };

        let mut spec_task = SpecTask::new(goal.clone(), preferences, self.top_k);
        let requests = spec_task.start(self).collect::<Vec<_>>();
        for (requested_spec, request_extra) in requests {
            let subresult = self.run_spec(&requested_spec);
            let to_provide = subresult.into_iter().next().map(|(_, c)| c);
            spec_task.resolve_request(self, request_extra, to_provide);
        }
        spec_task.complete(self)
    }
}

impl<Tgt: Target> SpecTask<Tgt> {
    fn new(goal: Spec<Tgt>, preferences: SmallVec<[ActionIdx; 1]>, top_k: usize) -> Self {
        SpecTask {
            goal,
            reducer: ImplReducer::new(top_k, preferences),
            partial_impls: Vec::new(),
            partial_impls_remaining: 0,
        }
    }

    /// Build partial [Impl]s for all actions, returning [Spec]s needed to compute this [Spec].
    ///
    /// The yielded [Spec]s are paired with additional data which must be provided when the [Spec]
    /// is resolved. If no [Spec]s are yielded, the task is immediately ready to be [complete]d.
    ///
    /// The caller should fully consume the returned iterator before calling [resolve_requests].
    fn start<D>(
        &mut self,
        search: &TopDownSearch<D>,
    ) -> impl Iterator<Item = (Spec<Tgt>, (usize, usize))> + '_ {
        // TODO: Make sure that calls to this function (`start`) happen only
        //   after the db is checked for the Spec

        // TODO: Define behavior for and document returning duplicates from this function.

        // TODO: How should the caller behave if there are no requirements?

        let all_actions = self.goal.0.actions().into_iter().collect::<Vec<_>>();
        let initial_skip = search.thread_idx * all_actions.len() / search.thread_count;
        (initial_skip..all_actions.len())
            .chain(0..initial_skip)
            .flat_map(move |action_idx| {
                let action = &all_actions[action_idx];
                let partial_impl_idx = self.partial_impls.len();
                let mut nested_specs = Vec::new();
                match action.apply(&self.goal) {
                    Ok(partial_impl) => {
                        // TODO: Add partial Impl to a list for reduction, and yield its nested Specs.
                        collect_nested_specs(&partial_impl, &mut nested_specs);

                        // If the resulting Impl is already complete, update the reducer. If there
                        // are nested sub-Specs, then store the partial Impl for resolution by the
                        // caller.
                        if nested_specs.is_empty() {
                            // TODO: Avoid walking the Impl twice; once to collect nested Specs and
                            //   once compute cost.
                            self.reducer.insert(
                                u16::try_from(action_idx).unwrap(),
                                Cost::from_impl(&partial_impl),
                            );
                        } else {
                            self.partial_impls.push(WorkingPartialImpl::Constructing {
                                partial_impl,
                                subspec_costs: vec![None; nested_specs.len()],
                                producing_action_idx: action_idx.try_into().unwrap(),
                            });
                            self.partial_impls_remaining += 1;

                            // Ensure that the Specs we return per partial_impl_idx are unique, which is
                            // promised by the contract of this function. This should be fine: right
                            // now, all actions expand to Impls with unique sub-Specs.
                            // TODO: Future-proof by de-duplicating here and in broadcasting in
                            //   `resolve_request`.
                            debug_assert!(nested_specs.iter().all_unique());
                        }
                    }
                    Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => {}
                    Err(ApplyError::SpecNotCanonical) => panic!("Goal was not canonical: {}", self.goal),
                };
                nested_specs
                    .into_iter()
                    .enumerate()
                    .map(move |(i, spec)| (spec, (partial_impl_idx, i)))
            })
    }

    // TODO: Do we actually need the `spec` at all, or can we just use the `extra` data?
    fn resolve_request<'d, D>(
        &mut self,
        search: &TopDownSearch<'d, D>,
        extra: (usize, usize),
        cost: Option<Cost>, // `None` means that the Spec was unsat
    ) where
        D: Database<'d>,
    {
        if self.partial_impls_remaining == 0 {
            return;
        }

        let (working_impl_idx, child_idx) = extra;
        let mut became_unsat = false;
        let entry = self.partial_impls.get_mut(working_impl_idx).unwrap();
        match entry {
            WorkingPartialImpl::Constructing {
                partial_impl,
                subspec_costs,
                producing_action_idx,
            } => {
                if let Some(cost) = cost {
                    let entry = &mut subspec_costs[child_idx];
                    debug_assert!(entry.is_none(), "Spec was already resolved");
                    *entry = Some(cost);

                    if subspec_costs.iter().all(|c| c.is_some()) {
                        self.partial_impls_remaining -= 1;

                        // TODO: Move rather than clone the child_costs.
                        let final_cost = compute_impl_cost(
                            partial_impl,
                            &mut subspec_costs.iter().map(|c| c.as_ref().unwrap().clone()),
                        );
                        self.reducer.insert(*producing_action_idx, final_cost);
                    }
                } else {
                    self.partial_impls_remaining -= 1;
                    became_unsat = true;
                }
            }
            WorkingPartialImpl::Unsat => {}
        };

        if became_unsat {
            *entry = WorkingPartialImpl::Unsat;
        }
    }

    fn complete<'d, D>(self, search: &TopDownSearch<'d, D>) -> SmallVec<[(ActionIdx, Cost); 1]>
    where
        D: Database<'d>,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert_eq!(self.partial_impls_remaining, 0);

        // TODO: Avoid this clone
        let final_result = self.reducer.clone().finalize();
        // TODO: Check that the final costs are below `self.goal`'s peaks.
        search.db.put(self.goal.clone(), final_result.clone());
        final_result
    }
}

impl ImplReducer {
    fn new(top_k: usize, preferences: SmallVec<[ActionIdx; 1]>) -> Self {
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

/// Push all nested [Spec]s in an Impl into a given [Vec], left to right.
fn collect_nested_specs<Tgt: Target, A: Clone>(imp: &ImplNode<Tgt, A>, out: &mut Vec<Spec<Tgt>>) {
    match imp {
        ImplNode::SpecApp(spec_app) => {
            out.push(spec_app.0.clone());
        }
        _ => {
            for child in imp.children() {
                collect_nested_specs(child, out);
            }
        }
    }
}

fn compute_impl_cost<Tgt: Target, A: Clone, I>(imp: &ImplNode<Tgt, A>, costs: &mut I) -> Cost
where
    I: Iterator<Item = Cost>,
{
    match imp {
        ImplNode::SpecApp(_) => costs.next().unwrap(),
        _ => {
            let child_costs = imp
                .children()
                .iter()
                .map(|child| compute_impl_cost(child, costs))
                .collect::<SmallVec<[_; 1]>>();
            Cost::from_child_costs(imp, &child_costs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DimSize;
    use crate::db::DashmapDiskDatabase;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::memorylimits::{MemVec, MemoryLimits};
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
