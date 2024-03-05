use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::iter;
use std::num::NonZeroUsize;
use std::rc::Rc;

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

#[derive(Debug)]
enum SearchFinalResult {
    Computed(SmallVec<[(ActionIdx, Cost); 1]>),
    Empty {
        preferences: Option<SmallVec<[ActionIdx; 1]>>,
    },
}

/// On-going synthesis of a [Spec].
///
/// Synthesizing a [Spec] is a three-step process. First, the [Spec] expands all actions into
/// [Impl]s and stores the incomplete [Impl]s. Second, the caller collects those dependencies from
/// the database or recursively synthesizes them, providing them to this [SpecTask]. Third, the
/// Spec is "completed:" the optimal action and its costs are computed and stored to the database,
/// and this [SpecTask] is dropped.
#[derive(Debug)]
struct SpecTask<Tgt: Target> {
    goal: Spec<Tgt>, // TODO: Do we really need to store the `goal`?
    reducer: ImplReducer,
    partial_impls: Vec<WorkingPartialImpl<Tgt>>,
    partial_impls_incomplete: usize,
    request_batches_returned: usize,
    max_children: usize, // TODO: Combine with request_batches_returned
    last_batch_requests_remaining: usize, // TODO: Remove
}

#[derive(Debug)]
enum WorkingPartialImpl<Tgt: Target> {
    Constructing {
        partial_impl: ImplNode<Tgt, ()>,
        subspecs: Vec<Spec<Tgt>>,
        subspec_costs: Vec<Option<Cost>>, // empty = unsat; all Some = ready-to-complete
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
        let xx = search.synth_specs(&[canonical_goal]);
        return (xx.into_iter().next().unwrap(), search.hits, search.misses);
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
            let r = (search.synth_specs(&[g]), search.hits, search.misses);
            (r.0.into_iter().next().unwrap(), r.1, r.2)
        })
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}

impl<'d, D> TopDownSearch<'d, D>
where
    D: Database<'d> + Send + Sync,
{
    fn synth_specs<'a, Tgt, I>(&mut self, goals: I) -> Vec<SmallVec<[(ActionIdx, Cost); 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
        I: IntoIterator<Item = &'a Spec<Tgt>> + 'a,
    {
        let mut goals = goals.into_iter();
        // TODO: Add debug assert. that all goals are unique.

        let mut final_results = Vec::with_capacity(goals.size_hint().0);
        let mut working_set = HashMap::with_capacity(goals.size_hint().0);
        let mut subblocks = Vec::new();
        let mut request_map = HashMap::<Rc<Spec<Tgt>>, Vec<_>>::new();
        let mut ready_to_complete = Vec::new();

        let working_set_rep = goals.next().expect("goals must not be empty");

        // First, check if the initial goal Specs are already in the database. (And set up
        // final_results and working_set.)

        for (idx, goal) in iter::once(working_set_rep).chain(goals).enumerate() {
            // TODO: Pay attention to what happens if final_results[i] computes final_results[i + 1]
            //   ahead of time.
            debug_assert!(specs_share_page(working_set_rep, goal));
            final_results.push(self.search_within_working_block(
                goal,
                Some(idx),
                working_set_rep,
                &mut working_set,
                &mut subblocks,
                &mut request_map,
                &mut ready_to_complete,
            ));
        }

        while !working_set.is_empty() {
            // TODO: We want to ensure that, by the end of this loop iteration, all external
            //   requests have been resolved. This is required if we want sane behavior
            //   w.r.t. determining unsat partial Impls.

            // Recurse into every sub-block. When it returns, resolve requests according to
            // `request_map`.
            self.recurse_subblocks(&subblocks, &mut working_set, &mut request_map);

            self.complete_ready_tasks(
                ready_to_complete,
                &mut working_set,
                &mut request_map,
                &mut final_results,
            );

            subblocks = Vec::new();
            ready_to_complete = Vec::new();

            // Produce more sub-block requests.
            let mut next_batches = Vec::with_capacity(working_set.len());
            for (sp, (task, _)) in working_set.iter_mut() {
                let batch = task.next_request_batch();
                let req_iter = batch.map(|b| b.collect::<Vec<_>>().into_iter());
                next_batches.push((Rc::clone(sp), req_iter));
            }
            for (sp, req_batch) in next_batches {
                self.visit_dependencies(
                    req_batch,
                    working_set_rep,
                    &mut working_set,
                    &mut subblocks,
                    &mut request_map,
                    &mut ready_to_complete,
                    sp,
                );
            }
        }

        final_results
            .into_iter()
            .map(|r| match r {
                SearchFinalResult::Computed(v) => v,
                _ => unreachable!("final_results should be fully computed"),
            })
            .collect()
    }

    /// Recurse into sub-blocks and resolve the returned requests.
    fn recurse_subblocks<Tgt>(
        &mut self,
        subblocks: &[HashSet<Rc<Spec<Tgt>>>],
        working_set: &mut HashMap<Rc<Spec<Tgt>>, (SpecTask<Tgt>, Option<usize>)>,
        request_map: &mut HashMap<Rc<Spec<Tgt>>, Vec<(Rc<Spec<Tgt>>, (usize, usize))>>,
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        for subblock in subblocks {
            // TODO: Avoid the following clones
            let subblock_results = self.synth_specs(
                &subblock
                    .iter()
                    .map(|s| s.as_ref().clone())
                    .collect::<Vec<_>>(),
            );

            for (sp, r) in subblock.iter().zip(subblock_results) {
                let sp = sp.as_ref();
                // TODO: Remove following debug_assert!s.
                debug_assert!(working_set.get(sp).is_none());
                debug_assert!(request_map.get(sp).is_some());
                for (requesting_spec, request_id) in request_map.remove(sp).unwrap() {
                    let Some((requesting_task, _)) = working_set.get_mut(requesting_spec.as_ref())
                    else {
                        panic!(
                            "working_set did not have {}\nworking_set is {}",
                            requesting_spec.as_ref(),
                            working_set
                                .keys()
                                .map(|s| format!("{s}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    };
                    debug_assert_eq!(&requesting_task.goal, requesting_spec.as_ref());
                    requesting_task
                        .resolve_request(request_id, r.iter().next().map(|v| v.1.clone()));
                }
            }
        }
    }

    /// Transitively add a [Spec] and the dependencies from its first batch to the working set,
    /// unless that Spec is in the database.
    ///
    /// The caller must ensure `goal` shares the working set page.
    fn search_within_working_block<Tgt>(
        &mut self,
        goal: &Spec<Tgt>,
        final_result_idx: Option<usize>,
        working_set_representative: &Spec<Tgt>,
        working_set: &mut HashMap<Rc<Spec<Tgt>>, (SpecTask<Tgt>, Option<usize>)>,
        subblocks: &mut Vec<HashSet<Rc<Spec<Tgt>>>>,
        request_map: &mut HashMap<Rc<Spec<Tgt>>, Vec<(Rc<Spec<Tgt>>, (usize, usize))>>,
        ready_to_complete: &mut Vec<Rc<Spec<Tgt>>>,
    ) -> SearchFinalResult
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(
            specs_share_page(working_set_representative, goal),
            "called on Spec not in the working set: {goal}",
        );

        // Check the database and immediately return if present.
        let preferences = match self.db.get_with_preference(goal) {
            GetPreference::Hit(v) => {
                self.hits += 1;
                return SearchFinalResult::Computed(v.0);
            }
            GetPreference::Miss(preferences) => preferences,
        };
        self.misses += 1;

        // Add task to working_set.
        let task_goal = Rc::new(goal.clone());
        match working_set.entry(Rc::clone(&task_goal)) {
            Entry::Vacant(entry) => {
                let inserted = entry.insert((
                    SpecTask::start(
                        task_goal.as_ref().clone(),
                        preferences.clone().unwrap_or_else(SmallVec::new),
                        self,
                    ),
                    final_result_idx,
                ));

                // TODO: Avoid collecting batches into Vecs.
                // TODO: This means we'll be recursing on more Specs than needed

                let mut batches = Vec::new();
                loop {
                    let batch = inserted
                        .0
                        .next_request_batch()
                        .map(|b| b.collect::<Vec<_>>().into_iter());
                    let last_batch = batch.is_none();
                    batches.push(batch);
                    if last_batch {
                        break;
                    }
                }
                for batch in batches {
                    self.visit_dependencies(
                        batch,
                        working_set_representative,
                        working_set,
                        subblocks,
                        request_map,
                        ready_to_complete,
                        Rc::clone(&task_goal),
                    );
                }
            }
            Entry::Occupied(mut occupied_entry) => {
                let (_, fri) = occupied_entry.get_mut();
                debug_assert!(fri.is_none() || fri == &final_result_idx);
                *fri = final_result_idx;
            }
        }

        SearchFinalResult::Empty { preferences }
    }

    /// Visits the requests in the given next batch for the given task.
    ///
    /// If no requests remain, the Spec will be added to `ready_to_complete`. Otherwise, `subblocks`
    /// will be updated with new out-of-working-block Specs. [search_within_working_block] will be
    /// applied to working-block Specs which haven't already been added to `working_set`.
    fn visit_dependencies<Tgt, B>(
        &mut self,
        next_batch: Option<B>,
        working_set_representative: &Spec<Tgt>,
        working_set: &mut HashMap<Rc<Spec<Tgt>>, (SpecTask<Tgt>, Option<usize>)>,
        subblocks: &mut Vec<HashSet<Rc<Spec<Tgt>>>>,
        request_map: &mut HashMap<Rc<Spec<Tgt>>, Vec<(Rc<Spec<Tgt>>, (usize, usize))>>,
        ready_to_complete: &mut Vec<Rc<Spec<Tgt>>>,
        task_goal: Rc<Spec<Tgt>>,
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
        B: Iterator<Item = (Spec<Tgt>, (usize, usize))>,
    {
        let Some(request_batch) = next_batch else {
            ready_to_complete.push(task_goal);
            return;
        };

        for (subtask, request_id) in request_batch {
            let subtask_rc = Rc::new(subtask);

            // Push requested_spec into the working set or sub-block.
            if specs_share_page(working_set_representative, &subtask_rc) {
                // Skip anything we've visited already. Use the working set to track what's visited.
                if !working_set.contains_key(subtask_rc.as_ref()) {
                    self.search_within_working_block(
                        &subtask_rc,
                        None,
                        working_set_representative,
                        working_set,
                        subblocks,
                        request_map,
                        ready_to_complete,
                    );
                }
            } else if let Some(subblock) = subblocks
                .iter_mut()
                .find(|s| specs_share_page(s.iter().next().unwrap(), &subtask_rc))
            {
                subblock.insert(Rc::clone(&subtask_rc));
            } else {
                subblocks.push(HashSet::from([Rc::clone(&subtask_rc)]));
            }

            // Update request_map with an entry from dependency -> requester.
            request_map
                .entry(Rc::clone(&subtask_rc))
                .or_default()
                .push((Rc::clone(&task_goal), request_id));
            debug_assert!(request_map[subtask_rc.as_ref()].iter().all_unique());
        }
    }

    /// Transitively completes tasks in the working set, starting from those in the given Vec.
    ///
    /// When a task is completed, it is removed from the working set, its result is sent to its
    /// requesters, and final_results is updated if it was a goal. This procedure will also be
    /// applied to tasks which are completed as a result of this process.
    fn complete_ready_tasks<Tgt>(
        &mut self,
        ready_to_complete: Vec<Rc<Spec<Tgt>>>,
        working_set: &mut HashMap<Rc<Spec<Tgt>>, (SpecTask<Tgt>, Option<usize>)>,
        request_map: &mut HashMap<Rc<Spec<Tgt>>, Vec<(Rc<Spec<Tgt>>, (usize, usize))>>,
        final_results: &mut [SearchFinalResult],
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        // TODO: Avoid needing to unique-ify. (Is this possible?)
        let mut completed = HashSet::new();

        // Complete Specs which had no more requests.
        let mut accum_next = Vec::new();
        let mut target_set = ready_to_complete;
        while !target_set.is_empty() {
            // TODO: Avoid needing to unique-ify. (Is this possible?)
            for r in target_set {
                if !completed.insert(Rc::clone(&r)) {
                    continue;
                }

                let Some((task, final_result_idx)) = working_set.remove(r.as_ref()) else {
                    panic!(
                        "ready_to_complete contained a Spec not in the working set: {}",
                        r.as_ref()
                    );
                };
                let result = task.complete(self);
                for (requesting_spec, request_id) in
                    request_map.remove(r.as_ref()).into_iter().flatten()
                {
                    let (requesting_task, _) =
                        working_set.get_mut(requesting_spec.as_ref()).unwrap();
                    let rr = result.iter().next().map(|v| v.1.clone()); // TODO: Inline
                    requesting_task.resolve_request(request_id, rr);
                    if requesting_task.ready_to_complete() {
                        accum_next.push(requesting_spec);
                    }
                }

                if let Some(i) = final_result_idx {
                    debug_assert!(matches!(final_results[i], SearchFinalResult::Empty { .. }));
                    final_results[i] = SearchFinalResult::Computed(result);
                }
            }
            target_set = accum_next;
            accum_next = Vec::new();
        }
    }
}

impl<Tgt: Target> SpecTask<Tgt> {
    /// Begin computing the optimal implementation of a Spec.
    ///
    /// Internally, this will expand partial [Impl]s for all actions.
    fn start<D>(
        goal: Spec<Tgt>,
        preferences: SmallVec<[ActionIdx; 1]>,
        search: &TopDownSearch<D>,
    ) -> Self {
        let mut task = SpecTask {
            goal,
            reducer: ImplReducer::new(search.top_k, preferences),
            partial_impls: Vec::new(),
            partial_impls_incomplete: 0,
            last_batch_requests_remaining: 0,
            max_children: 0,
            request_batches_returned: 0,
        };

        let all_actions = task.goal.0.actions().into_iter().collect::<Vec<_>>();
        let initial_skip = search.thread_idx * all_actions.len() / search.thread_count;

        for action_idx in (initial_skip..all_actions.len()).chain(0..initial_skip) {
            let action = &all_actions[action_idx];
            match action.apply(&task.goal) {
                Ok(partial_impl) => {
                    let mut partial_impl_subspecs = Vec::new();
                    collect_nested_specs(&partial_impl, &mut partial_impl_subspecs);

                    let subspec_count = partial_impl_subspecs.len();
                    task.max_children = task.max_children.max(subspec_count);

                    // If the resulting Impl is already complete, update the reducer. If there
                    // are nested sub-Specs, then store the partial Impl for resolution by the
                    // caller.
                    if partial_impl_subspecs.is_empty() {
                        task.reducer.insert(
                            u16::try_from(action_idx).unwrap(),
                            Cost::from_impl(&partial_impl),
                        );
                    } else {
                        task.partial_impls.push(WorkingPartialImpl::Constructing {
                            partial_impl,
                            subspecs: partial_impl_subspecs,
                            subspec_costs: vec![None; subspec_count],
                            producing_action_idx: action_idx.try_into().unwrap(),
                        });
                        task.partial_impls_incomplete += 1;
                    }
                }
                Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => {}
                Err(ApplyError::SpecNotCanonical) => panic!(),
            };
        }

        task
    }

    /// Return an iterator over a set of [Spec]s needed to compute this task's goal.
    ///
    /// This will return `None` when all dependencies are resolved and the goal is computed.
    /// The caller should continue to call [next_request_batch] if an empty iterator is returned.
    fn next_request_batch(
        &mut self,
    ) -> Option<impl Iterator<Item = (Spec<Tgt>, (usize, usize))> + '_> {
        // debug_assert_eq!(
        //     self.last_batch_requests_remaining, 0,
        //     "Didn't resolve all requests from last batch for {}",
        //     self.goal
        // ); // TODO: Remove

        // TODO: Define behavior for and document returning duplicates from this function.

        if self.request_batches_returned == self.max_children {
            return None;
        }

        let subspec_idx = self.request_batches_returned;
        self.request_batches_returned += 1;
        let to_return: Vec<(Spec<Tgt>, (usize, usize))> = self
            .partial_impls
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let WorkingPartialImpl::Constructing { subspecs, .. } = p else {
                    return None;
                };
                subspecs
                    .get(subspec_idx)
                    .map(|s| (s.clone(), (i, subspec_idx)))
            })
            .collect::<Vec<_>>();
        self.last_batch_requests_remaining += to_return.len();

        // TODO: Assert/test that we return unique Specs
        Some(to_return.into_iter()) // TODO: Inline the iterator instead of collecting.
    }

    fn resolve_request(
        &mut self,
        id: (usize, usize),
        cost: Option<Cost>, // `None` means that the Spec was unsat
    ) {
        debug_assert_ne!(self.last_batch_requests_remaining, 0, "{}", self.goal);
        if self.partial_impls_incomplete == 0 {
            return;
        }

        self.last_batch_requests_remaining -= 1;

        let (working_impl_idx, child_idx) = id;
        let mut became_unsat = false;
        let entry = self.partial_impls.get_mut(working_impl_idx).unwrap();
        match entry {
            WorkingPartialImpl::Constructing {
                partial_impl,
                subspecs,
                subspec_costs,
                producing_action_idx,
            } => {
                if let Some(cost) = cost {
                    let entry = &mut subspec_costs[child_idx];
                    debug_assert!(entry.is_none(), "Spec was already resolved");
                    *entry = Some(cost);

                    if subspec_costs.iter().all(|c| c.is_some()) {
                        self.partial_impls_incomplete -= 1;

                        // TODO: Move rather than clone the child_costs.
                        let final_cost = compute_impl_cost(
                            partial_impl,
                            &mut subspec_costs.iter().map(|c| c.as_ref().unwrap().clone()),
                        );
                        self.reducer.insert(*producing_action_idx, final_cost);
                        // TODO: Does this signal ready-to-complete and prop.?
                    }
                } else {
                    became_unsat = true;
                }
            }
            WorkingPartialImpl::Unsat => {}
        };

        if became_unsat {
            self.partial_impls_incomplete -= 1;
            *entry = WorkingPartialImpl::Unsat;
        }
    }

    fn complete<'d, D>(self, search: &TopDownSearch<'d, D>) -> SmallVec<[(ActionIdx, Cost); 1]>
    where
        D: Database<'d>,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert_eq!(
            self.partial_impls_incomplete,
            0,
            "Some of {}'s partial Impls are unresolved:\n  {}",
            self.goal,
            self.partial_impls
                .iter()
                .filter(|pi| matches!(pi, WorkingPartialImpl::Constructing { .. }))
                .map(|pi| format!("{pi:?}"))
                .collect::<Vec<_>>()
                .join("\n  ")
        );

        // TODO: Avoid this clone
        let final_result = self.reducer.clone().finalize();
        // TODO: Check that the final costs are below `self.goal`'s peaks.
        search.db.put(self.goal.clone(), final_result.clone());
        final_result
    }

    fn ready_to_complete(&self) -> bool {
        self.partial_impls_incomplete == 0
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

// TODO: Can we replace this function with a more general `utils` crate fn. or something?
/// Push all nested [Spec]s in an Impl into a given [Vec], left to right.
fn collect_nested_specs<Tgt, A>(imp: &ImplNode<Tgt, A>, out: &mut Vec<Spec<Tgt>>)
where
    Tgt: Target,
    A: Clone + std::fmt::Debug,
{
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

// TODO: Move into Database.
fn specs_share_page<Tgt: Target>(lhs: &Spec<Tgt>, rhs: &Spec<Tgt>) -> bool {
    // TODO: This is one-Spec-per-page!
    match (&lhs.0, &rhs.0) {
        (
            crate::spec::LogicalSpec::Primitive(lhs_basics, _, _),
            crate::spec::LogicalSpec::Primitive(rhs_basics, _, _),
        ) => lhs_basics.spec_shape == rhs_basics.spec_shape,
        (crate::spec::LogicalSpec::Compose { .. }, crate::spec::LogicalSpec::Compose { .. }) => {
            todo!()
        }
        _ => false,
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
