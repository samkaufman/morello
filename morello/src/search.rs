use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use smallvec::{smallvec, SmallVec};
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::mem::{replace, take};
use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::cost::Cost;
use crate::db::{ActionCostVec, ActionIdx, Database, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::{Impl, ImplNode};
use crate::scheduling::ApplyError;
use crate::spec::Spec;
use crate::target::Target;

type RequestId = (usize, usize);

struct TopDownSearch<'d, D> {
    db: &'d D,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
    hits: u64,
    misses: u64,
}

struct BlockSearch<'a, 'd, D, Tgt: Target> {
    search: &'a TopDownSearch<'d, D>,
    working_set: HashMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
    // The following two fields map requested Specs (the keys) to the recipients
    // (Specs + RequestIds). The latter might be out-of-date by the time they are
    // resolved; for example, when another resolution removes that SpecTask from
    // `working_set` when a WorkingPartialImpl became Unsat.
    working_block_requests: HashMap<Spec<Tgt>, Vec<(Spec<Tgt>, RequestId)>>,
    subblock_requests: Vec<HashMap<Spec<Tgt>, Vec<(Spec<Tgt>, RequestId)>>>,
}

/// On-going synthesis of a [Spec]. (Essentially a coroutine.)
#[derive(Debug)]
enum SpecTask<Tgt: Target> {
    Running {
        reducer: ImplReducer,
        partial_impls: Vec<WorkingPartialImpl<Tgt>>,
        partial_impls_incomplete: usize,
        request_batches_returned: usize,
        max_children: usize, // TODO: Combine with request_batches_returned
    },
    Complete(ActionCostVec),
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
    Sat,
}

#[derive(Debug, Clone)] // TODO: Remove this Clone derive if not needed
struct ImplReducer {
    results: SmallVec<[(ActionIdx, Cost); 1]>,
    top_k: usize, // TODO: Shared between ImplReducers. Pull out?
    preferences: SmallVec<[ActionIdx; 1]>,
}

// Computes an optimal Impl for `goal` and stores it in `db`.
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
    // TODO: Just return the ActionCostVec directly
    let (r, h, m) = top_down_many(db, &[goal.clone()], top_k, jobs);
    (r.into_iter().next().unwrap().0, h, m)
}

pub fn top_down_many<'d, Tgt, D>(
    db: &'d D,
    goals: &[Spec<Tgt>],
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> (Vec<ActionCostVec>, u64, u64)
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

    let canonical_goals = goals
        .iter()
        .map(|g| {
            let mut g = g.clone();
            g.canonicalize()
                .expect("should be possible to canonicalize goal Spec");
            g
        })
        .collect::<Vec<_>>();

    let thread_count = jobs
        .map(|j| j.get())
        .unwrap_or_else(rayon::current_num_threads);
    if thread_count == 1 {
        let search = TopDownSearch::<'d, D> {
            db,
            top_k,
            thread_idx: 0,
            thread_count: 1,
            hits: 0,
            misses: 1,
        };
        let result = BlockSearch::synthesize(canonical_goals.iter(), &search, None);
        return (result, search.hits, search.misses);
    }

    let tasks = (0..thread_count)
        .zip(std::iter::repeat(canonical_goals.clone()))
        .collect::<Vec<_>>();
    // Collect all and take the result from the first call so that we get
    // deterministic results.
    tasks
        .into_par_iter()
        .map(|(i, gs)| {
            let search = TopDownSearch::<'d, D> {
                db,
                top_k,
                thread_idx: i,
                thread_count,
                hits: 0,
                misses: 1,
            };
            let r = BlockSearch::synthesize(gs.iter(), &search, None);
            (r, search.hits, search.misses)
        })
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}

impl<'d, D> TopDownSearch<'d, D> where D: Database<'d> + Send + Sync {}

impl<'a, 'd, D, Tgt> BlockSearch<'a, 'd, D, Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'d> + Send + Sync,
{
    fn synthesize<'i, I>(
        goals: I,
        search: &'a TopDownSearch<'d, D>,
        prefetch_after: Option<&Spec<Tgt>>,
    ) -> Vec<ActionCostVec>
    where
        I: IntoIterator<Item = &'i Spec<Tgt>> + 'i,
    {
        let goals = goals.into_iter().collect::<Vec<_>>();
        debug_assert!(goals.iter().all_unique());

        let mut block = BlockSearch {
            search,
            working_set: HashMap::with_capacity(goals.len()),
            working_block_requests: HashMap::new(),
            subblock_requests: Vec::new(),
        };
        let mut visited_in_stage = HashSet::new();
        let mut outbox = Vec::new();
        let mut final_result_tasks = Vec::with_capacity(goals.len());
        for g in &goals {
            final_result_tasks.push(block.visit_spec_wb(g, &mut visited_in_stage, &mut outbox));
        }

        loop {
            for (spec, completed_task_results) in outbox.drain(..) {
                block.resolve_wb_request(&spec, completed_task_results);
                block.working_set.remove(&spec).unwrap();
            }

            let mut subblock_reqs_iter = take(&mut block.subblock_requests).into_iter().peekable();
            while let Some(mut subblock) = subblock_reqs_iter.next() {
                // TODO: Move prefetch so that it happens after the get inside the recursive call.
                let mut prefetch_to_push_down: Option<&Spec<Tgt>> = None;
                match subblock_reqs_iter.peek() {
                    Some(next_subblock) => prefetch_to_push_down = next_subblock.keys().next(),
                    None => {
                        if let Some(prefetch_after) = prefetch_after.as_ref() {
                            search.db.prefetch(prefetch_after);
                        }
                    }
                }

                let subblock_goals = subblock.keys().cloned().collect::<Vec<_>>();
                let subblock_results =
                    Self::synthesize(&subblock_goals, search, prefetch_to_push_down);
                for (subspec, subspec_result) in subblock_goals.into_iter().zip(subblock_results) {
                    block.resolve_subblock_request(&mut subblock, &subspec, subspec_result);
                }
            }

            // TODO: Replace the following scan of the working set with an integer counter.
            let incomplete_specs = block
                .working_set
                .iter()
                .filter(|(_, v)| matches!(&*v.borrow(), SpecTask::Running { .. }))
                .map(|(k, _)| k.clone())
                .collect::<Vec<_>>();
            if incomplete_specs.is_empty() {
                break;
            }

            let ws_vec = block
                .working_set
                .iter()
                .map(|(k, v)| (k.clone(), Rc::clone(v)))
                .collect::<Vec<_>>();
            visited_in_stage.clear();
            for (spec, task_ref) in ws_vec {
                block.visit_next_request_batch(&spec, task_ref, &mut visited_in_stage, &mut outbox);
            }
        }
        debug_assert!(
            block.working_block_requests.is_empty(),
            "working_block_requests is not empty: {}",
            block
                .working_block_requests
                .keys()
                .map(|k| format!("{k}"))
                .join(", ")
        );
        debug_assert!(block.subblock_requests.is_empty());
        debug_assert!(
            block
                .working_set
                .values()
                .all(|v| matches!(*v.borrow(), SpecTask::Complete(_))),
            "working_set has incomplete members:\n{}",
            block
                .working_set
                .iter()
                .filter(|(_, v)| !matches!(*v.borrow(), SpecTask::Complete(_)))
                .map(|(k, v)| format!("{k}:\n  {:?}", v.borrow()))
                .join("\n")
        );

        // Extract final results from the completed tasks. (This leaves tasks in an invalid state,
        // but they'll be dropped immediately.)
        final_result_tasks
            .into_iter()
            .map(|task| {
                let SpecTask::Complete(task_result) = &mut *task.borrow_mut() else {
                    unreachable!("Expected goal to be complete.");
                };
                take(task_result)
            })
            .collect()
    }

    fn visit_spec_wb(
        &mut self,
        spec: &Spec<Tgt>,
        visited_in_stage: &mut HashSet<Spec<Tgt>>,
        outbox: &mut Vec<(Spec<Tgt>, ActionCostVec)>,
    ) -> Rc<RefCell<SpecTask<Tgt>>> {
        let task = self.get_wb_task(spec);
        if !visited_in_stage.contains(spec) {
            visited_in_stage.insert(spec.clone());
            self.visit_next_request_batch(spec, Rc::clone(&task), visited_in_stage, outbox);
        }
        task
    }

    /// Return or create from the working set a running task or return an immediately complete task.
    fn get_wb_task(&mut self, spec: &Spec<Tgt>) -> Rc<RefCell<SpecTask<Tgt>>> {
        match self.working_set.entry(spec.clone()) {
            Entry::Occupied(e) => Rc::clone(e.get()),
            Entry::Vacant(e) => {
                let task = SpecTask::start(spec.clone(), self.search);
                if matches!(&task, SpecTask::Running { .. }) {
                    let task_rc = Rc::new(RefCell::new(task));
                    e.insert(Rc::clone(&task_rc));
                    task_rc
                } else {
                    Rc::new(RefCell::new(task))
                }
            }
        }
    }

    fn visit_next_request_batch(
        &mut self,
        spec: &Spec<Tgt>,
        task_ref: Rc<RefCell<SpecTask<Tgt>>>,
        visited_in_stage: &mut HashSet<Spec<Tgt>>,
        outbox: &mut Vec<(Spec<Tgt>, ActionCostVec)>, // TODO: Make Option<Cost>
    ) {
        let task: &mut SpecTask<Tgt> = &mut task_ref.borrow_mut();
        if !matches!(task, SpecTask::Running { .. }) {
            return;
        }

        let next_batch_opt = task.next_request_batch();
        let b = next_batch_opt.map(|it| it.collect::<Vec<_>>()); // TODO: Need `collect`?
        if let Some(request_batch) = b {
            for (subspec, request_id) in request_batch {
                if self.search.db.specs_share_page(spec, &subspec) {
                    let subtask = self.visit_spec_wb(&subspec, visited_in_stage, outbox);
                    let subtask_ref = subtask.borrow();
                    match &*subtask_ref {
                        SpecTask::Running { .. } => {
                            drop(subtask_ref);
                            self.add_wb_request_mapping(spec, &subspec, request_id);
                        }
                        SpecTask::Complete(subtask_result) => {
                            let cost = subtask_result.iter().next().map(|v| v.1.clone());
                            task.resolve_request(request_id, cost, spec, self.search);
                            // At this point, the task_ref might have completed (be
                            // `SpecTask::Complete`). We want to propagate the completion to any
                            // tasks waiting within the working set, but we don't want to recurse
                            // into a Spec we're already borrowing lower on the current stack.
                            // That's overly complicated and will lead to a RefCell borrow panic.
                            // Instead, push thd completion into a queue (the `outbox`) we'll
                            // resolve later.
                            if let SpecTask::Complete(completed_task_results) = task {
                                outbox.push((spec.clone(), completed_task_results.clone()));
                            }
                        }
                    };
                } else {
                    self.add_subblock_request_mapping(spec, &subspec, request_id);
                }
            }
        }
    }

    fn resolve_wb_request(&mut self, subspec: &Spec<Tgt>, results: ActionCostVec) {
        Self::inner_resolve_subblock_request(
            &mut self.working_set,
            &mut self.working_block_requests,
            None,
            subspec,
            results,
            self.search,
        );
    }

    fn resolve_subblock_request(
        &mut self,
        subblock: &mut HashMap<Spec<Tgt>, Vec<(Spec<Tgt>, RequestId)>>,
        subspec: &Spec<Tgt>,
        results: ActionCostVec,
    ) {
        Self::inner_resolve_subblock_request(
            &mut self.working_set,
            subblock,
            Some(&mut self.working_block_requests),
            subspec,
            results,
            self.search,
        );
    }

    fn inner_resolve_subblock_request(
        working_set: &mut HashMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
        subblock: &mut HashMap<Spec<Tgt>, Vec<(Spec<Tgt>, RequestId)>>,
        next_subblock: Option<&mut HashMap<Spec<Tgt>, Vec<(Spec<Tgt>, RequestId)>>>,
        subspec: &Spec<Tgt>,
        results: ActionCostVec,
        search: &'a TopDownSearch<'d, D>,
    ) {
        let Some(rs) = subblock.remove(subspec) else {
            return;
        };

        let resolved_next_subblock = next_subblock.unwrap_or(subblock);

        let cost = results.0.into_iter().next().map(|v| v.1);
        for (wb_spec, request_id) in rs {
            // `wb_spec` should be in the working set unless its partial Impls became unsat.
            if let Some(requester_task) = working_set.get(&wb_spec) {
                let mut requester = requester_task.borrow_mut();
                if matches!(&*requester, SpecTask::Running { .. }) {
                    requester.resolve_request(request_id, cost.clone(), &wb_spec, search);
                    if let SpecTask::Complete(completed_requester_results) = &*requester {
                        // TODO: Avoid this clone by consuming the sub-block. (Do at the call site.)
                        let cloned_results = completed_requester_results.clone();
                        drop(requester);
                        Self::inner_resolve_subblock_request(
                            working_set,
                            resolved_next_subblock,
                            None,
                            &wb_spec,
                            cloned_results,
                            search,
                        );
                        working_set.remove(&wb_spec).unwrap();
                    }
                } else {
                    log::warn!(
                        "Requester {} was in working set but wasn't Running: {:?}",
                        wb_spec,
                        &*requester
                    )
                }
            }
        }
    }

    fn add_wb_request_mapping(
        &mut self,
        spec: &Spec<Tgt>,
        subspec: &Spec<Tgt>,
        request_id: RequestId,
    ) {
        debug_assert!(self.spec_in_working_set(spec));
        debug_assert!(self.spec_in_working_set(subspec));
        self.working_block_requests
            .entry(subspec.clone())
            .or_default()
            .push((spec.clone(), request_id));
    }

    fn add_subblock_request_mapping(
        &mut self,
        spec: &Spec<Tgt>,
        subspec: &Spec<Tgt>,
        request_id: RequestId,
    ) {
        debug_assert!(self.spec_in_working_set(spec));
        debug_assert!(!self.spec_in_working_set(subspec));
        let request_set = match self.subblock_requests.iter_mut().find(|s| {
            self.search
                .db
                .specs_share_page(s.keys().next().unwrap(), subspec)
        }) {
            Some(s) => s,
            None => {
                self.subblock_requests.push(HashMap::new());
                self.subblock_requests.last_mut().unwrap()
            }
        };
        request_set
            .entry(subspec.clone())
            .or_default()
            .push((spec.clone(), request_id));
    }

    fn spec_in_working_set(&self, spec: &Spec<Tgt>) -> bool {
        let ws_rep = self.working_set.keys().next().unwrap();
        self.search.db.specs_share_page(ws_rep, spec)
    }
}

impl<Tgt: Target> SpecTask<Tgt> {
    /// Begin computing the optimal implementation of a Spec.
    ///
    /// Internally, this will expand partial [Impl]s for all actions.
    fn start<'d, D>(goal: Spec<Tgt>, search: &TopDownSearch<'d, D>) -> Self
    where
        D: Database<'d> + Send + Sync,
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        // Check the database and immediately return if present.
        let preferences = match search.db.get_with_preference(&goal) {
            GetPreference::Hit(v) => {
                // TODO: Re-enable search hits and misses tracking
                // search.hits += 1;
                return SpecTask::Complete(v);
            }
            GetPreference::Miss(preferences) => preferences,
        };
        // search.misses += 1;

        let mut reducer = ImplReducer::new(search.top_k, preferences.unwrap_or_default());
        let mut max_children = 0;
        let mut partial_impls = Vec::new();
        let mut partial_impls_incomplete = 0;

        let all_actions = goal.0.actions().into_iter().collect::<Vec<_>>();
        let initial_skip = search.thread_idx * all_actions.len() / search.thread_count;

        for action_idx in (initial_skip..all_actions.len()).chain(0..initial_skip) {
            let action = &all_actions[action_idx];
            match action.apply(&goal) {
                Ok(partial_impl) => {
                    let mut partial_impl_subspecs = Vec::new();
                    collect_nested_specs(&partial_impl, &mut partial_impl_subspecs);

                    let subspec_count = partial_impl_subspecs.len();
                    max_children = max_children.max(subspec_count);

                    // If the resulting Impl is already complete, update the reducer. If there
                    // are nested sub-Specs, then store the partial Impl for resolution by the
                    // caller.
                    if partial_impl_subspecs.is_empty() {
                        reducer.insert(
                            u16::try_from(action_idx).unwrap(),
                            Cost::from_impl(&partial_impl),
                        );
                    } else {
                        partial_impls.push(WorkingPartialImpl::Constructing {
                            partial_impl,
                            subspecs: partial_impl_subspecs,
                            subspec_costs: vec![None; subspec_count],
                            producing_action_idx: action_idx.try_into().unwrap(),
                        });
                        partial_impls_incomplete += 1;
                    }
                }
                Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => {}
                Err(ApplyError::SpecNotCanonical) => panic!(),
            };
        }

        if partial_impls_incomplete == 0 {
            let action_costs = reducer.finalize();
            search.db.put(goal.clone(), action_costs.clone());
            SpecTask::Complete(ActionCostVec(action_costs))
        } else {
            SpecTask::Running {
                reducer,
                max_children,
                partial_impls,
                partial_impls_incomplete,
                request_batches_returned: 0,
            }
        }
    }

    /// Return an iterator over a set of [Spec]s needed to compute this task's goal.
    ///
    /// This will return `None` when all dependencies are resolved and the goal is computed.
    /// The caller should continue to call [next_request_batch] if an empty iterator is returned.
    fn next_request_batch(&mut self) -> Option<impl Iterator<Item = (Spec<Tgt>, RequestId)> + '_> {
        // TODO: Define behavior for and document returning duplicates from this function.

        let SpecTask::Running {
            partial_impls,
            request_batches_returned,
            max_children,
            ..
        } = self
        else {
            return None;
        };
        if request_batches_returned == max_children {
            return None;
        }

        let subspec_idx = *request_batches_returned;
        *request_batches_returned += 1;
        let to_return: Vec<(Spec<Tgt>, RequestId)> = partial_impls
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

        // TODO: Assert/test that we return unique Specs
        Some(to_return.into_iter()) // TODO: Inline the iterator instead of collecting.
    }

    fn resolve_request<'d, D>(
        &mut self,
        id: RequestId,
        cost: Option<Cost>, // `None` means that the Spec was unsat
        task_goal: &Spec<Tgt>,
        search: &TopDownSearch<'d, D>,
    ) where
        D: Database<'d> + Send + Sync,
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let SpecTask::Running {
            reducer,
            partial_impls,
            partial_impls_incomplete,
            request_batches_returned,
            max_children,
        } = self
        else {
            panic!("Task is not running");
        };

        if *partial_impls_incomplete == 0 {
            return;
        }

        let (working_impl_idx, child_idx) = id;
        let mut finished = false;
        let mut became_unsat = false;
        let partial_impls_cnt = partial_impls.len(); // TODO: remove
        let entry = partial_impls.get_mut(working_impl_idx).unwrap();
        match entry {
            WorkingPartialImpl::Constructing {
                partial_impl,
                subspecs,
                subspec_costs,
                producing_action_idx,
            } => {
                if let Some(cost) = cost {
                    let entry = &mut subspec_costs[child_idx];
                    debug_assert!(entry.is_none(), "Requested Spec was already resolved");
                    *entry = Some(cost);

                    // If all subspec costs for this partial Impl are completed, then reduce costs
                    // for the parent and transition this partial to a Sat state.
                    if subspec_costs.iter().all(|c| c.is_some()) {
                        finished = true;
                        reducer.insert(
                            *producing_action_idx,
                            compute_impl_cost(
                                partial_impl,
                                // TODO: Move rather than clone the child_costs.
                                &mut subspec_costs.iter().map(|c| c.as_ref().unwrap().clone()),
                            ),
                        );
                    }
                } else {
                    finished = true;
                    became_unsat = true;
                }
            }
            WorkingPartialImpl::Unsat => {}
            WorkingPartialImpl::Sat => {
                panic!("Resolved a request for an already-completed Spec");
            }
        };

        if finished {
            *partial_impls_incomplete -= 1;
            if became_unsat {
                *entry = WorkingPartialImpl::Unsat;
            } else {
                *entry = WorkingPartialImpl::Sat;
            }

            // If that was the last working partial Impl for this task, then the task is complete.
            if *partial_impls_incomplete == 0 {
                // TODO: Check that the final costs are below `task_goal`'s peaks.
                // TODO: Make sure completions prop. up the request DAG.
                let tmp_replacement = ImplReducer::new(0, SmallVec::new());
                let removed_reducer: ImplReducer = replace(reducer, tmp_replacement);
                let final_result = removed_reducer.finalize();
                search.db.put(task_goal.clone(), final_result.clone());
                *self = SpecTask::Complete(ActionCostVec(final_result));
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DimSize;
    use crate::db::RocksDatabase;
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
            let db = RocksDatabase::try_new(None, false, 1).unwrap();
            top_down(&db, &spec, 1, Some(nz!(1usize)));
        }

        #[test]
        #[ignore]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_canonical_specs::<X86Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = RocksDatabase::try_new(None, false, 1).unwrap();

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
            let db = RocksDatabase::try_new(None, false, 1).unwrap();
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

        let db = RocksDatabase::try_new(None, false, 1).unwrap();
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