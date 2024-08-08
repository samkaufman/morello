use indexmap::IndexMap;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::iter;
use std::mem::{replace, take};
use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::cost::Cost;
use crate::db::{ActionCostVec, ActionIdx, FilesDatabase, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::scheduling::{ActionSolver, ActionT as _, ApplyError};
use crate::spec::Spec;
use crate::target::Target;

type RequestId = (usize, usize);
type WorkingPartialImplHandle = (usize, RequestId);

struct TopDownSearch<'d> {
    db: &'d FilesDatabase,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
    hits: u64,
    misses: u64,
}

struct BlockSearch<'a, 'd, Tgt: Target> {
    search: &'a TopDownSearch<'d>,
    working_set: IndexMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
    working_set_running: usize,
    // The following two fields map requested Specs (the keys) to the recipients
    // (Specs + RequestIds). The latter might be out-of-date by the time they are
    // resolved; for example, when another resolution removes that SpecTask from
    // `working_set` when a WorkingPartialImpl became Unsat.
    working_block_requests: HashMap<usize, Vec<WorkingPartialImplHandle>>,
    subblock_requests: Vec<HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>>,
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
    // TODO: Shouldn't need this second bool to track if it's from the database
    Complete(ActionCostVec, bool),
}

#[derive(Debug)]
enum WorkingPartialImpl<Tgt: Target> {
    Constructing {
        solver: ActionSolver<Tgt>,
        subspecs: Vec<Spec<Tgt>>,
        subspec_costs: Vec<Option<Cost>>, // empty = unsat; all Some = ready-to-complete
        producing_action_idx: ActionIdx,
    },
    Unsat,
    Sat,
}

// TODO: Make this private once #[bench] gets stable.
#[doc(hidden)]
#[derive(Debug)]
pub struct ImplReducer {
    results: ImplReducerResults,
    top_k: usize,
    preferences: Vec<ActionIdx>,
}

#[derive(Debug)]
enum ImplReducerResults {
    One(Option<(Cost, ActionIdx)>),
    Many(BTreeSet<(Cost, ActionIdx)>),
}

enum RequestsMapRef<'a, Tgt: Target> {
    Internal(&'a mut HashMap<usize, Vec<WorkingPartialImplHandle>>),
    External(&'a mut HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>),
}

// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<Tgt>(
    db: &FilesDatabase,
    goal: &Spec<Tgt>,
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> (Vec<(ActionIdx, Cost)>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    // TODO: Just return the ActionCostVec directly
    let (r, h, m) = top_down_many(db, &[goal.clone()], top_k, jobs);
    (r.into_iter().next().unwrap().0, h, m)
}

pub fn top_down_many<'d, Tgt>(
    db: &'d FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> (Vec<ActionCostVec>, u64, u64)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    assert!(db.max_k().is_none_or(|k| k >= top_k));
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

    // Group goal Specs by database page.
    let mut grouped_canonical_goals = HashMap::<_, Vec<usize>>::new();
    for (idx, goal) in canonical_goals.iter().enumerate() {
        let page = db.page_id(goal);
        let key = (page.table_key, page.superblock_id);
        // TODO: prefetch here?
        grouped_canonical_goals.entry(key).or_default().push(idx);
    }

    let thread_count = jobs
        .map(|j| j.get())
        .unwrap_or_else(rayon::current_num_threads);

    let mut combined_results = vec![Default::default(); canonical_goals.len()];
    let mut combined_hits = 0;
    let mut combined_misses = 0;
    let mut goal_group = Vec::new();
    for page_group in grouped_canonical_goals.values() {
        goal_group.clear();
        goal_group.extend(page_group.iter().map(|&i| canonical_goals[i].clone()));

        let (result, hits, misses) = if thread_count == 1 {
            let search = TopDownSearch::<'d> {
                db,
                top_k,
                thread_idx: 0,
                thread_count: 1,
                hits: 0,
                misses: 1,
            };
            let r = BlockSearch::synthesize(&goal_group, &search, None);
            (r, search.hits, search.misses)
        } else {
            let tasks = (0..thread_count)
                .zip(std::iter::repeat(canonical_goals.clone()))
                .collect::<Vec<_>>();
            // Collect all and take the result from the first call so that we get
            // deterministic results.
            tasks
                .into_par_iter()
                .map(|(i, gs)| {
                    let search = TopDownSearch::<'d> {
                        db,
                        top_k,
                        thread_idx: i,
                        thread_count,
                        hits: 0,
                        misses: 1,
                    };
                    let r = BlockSearch::synthesize(&gs, &search, None);
                    (r, search.hits, search.misses)
                })
                .collect::<Vec<_>>()
                .pop()
                .unwrap()
        };

        for (r, i) in result.into_iter().zip(page_group) {
            combined_results[*i] = r;
        }
        combined_hits += hits;
        combined_misses += misses;
    }

    (combined_results, combined_hits, combined_misses)
}

impl<'a, 'd, Tgt> BlockSearch<'a, 'd, Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    fn synthesize(
        goals: &[Spec<Tgt>],
        search: &'a TopDownSearch<'d>,
        prefetch_after: Option<&Spec<Tgt>>,
    ) -> Vec<ActionCostVec> {
        debug_assert!(goals.iter().all_unique());

        let mut block = BlockSearch {
            search,
            working_set: IndexMap::with_capacity(goals.len()),
            working_set_running: 0,
            working_block_requests: HashMap::new(),
            subblock_requests: Vec::new(),
        };
        let mut visited_in_stage = HashSet::new();
        let mut outbox = Vec::new();
        for g in goals {
            let (spec_working_set_index, task) = block.get_task_internal(g);
            visited_in_stage.insert(g.clone());
            block.visit_next_request_batch(
                spec_working_set_index,
                g,
                Rc::clone(&task),
                &mut visited_in_stage,
                &mut outbox,
            );
        }

        loop {
            for (spec, completed_task_results) in outbox.drain(..) {
                block.resolve_request_internal(&spec, completed_task_results);
            }

            let new_vec = Vec::with_capacity(block.subblock_requests.len());
            let mut subblock_reqs_iter = replace(&mut block.subblock_requests, new_vec)
                .into_iter()
                .peekable();
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
                    block.resolve_request_external(&mut subblock, &subspec, subspec_result);
                }
            }

            debug_assert_eq!(
                block.working_set_running,
                block
                    .working_set
                    .values()
                    .filter(|v| matches!(&*v.borrow(), SpecTask::Running { .. }))
                    .count()
            );
            if block.working_set_running == 0 {
                break;
            }

            let ws_vec = block
                .working_set
                .iter()
                .enumerate()
                .filter(|(_, (_, task))| matches!(*task.borrow(), SpecTask::Running { .. }))
                .map(|(spec_idx, (spec, task))| (spec_idx, spec.clone(), Rc::clone(task)))
                .collect::<Vec<_>>();
            visited_in_stage.clear();
            for (spec_idx, spec, task_ref) in ws_vec {
                block.visit_next_request_batch(
                    spec_idx,
                    &spec,
                    task_ref,
                    &mut visited_in_stage,
                    &mut outbox,
                );
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

        // After this point, we'll be removing entries from working_set, so
        // WorkingPartialImplHandles will not be valid.

        // Gather all tasks requested by synthesize. This removes from the working set.
        let final_results = goals
            .iter()
            .map(|g| process_complete_task(search, g, block.working_set.swap_remove(g).unwrap()))
            .collect::<Vec<_>>();

        // Anything left in the working set is not a goal but should still be put
        for (spec, task) in block.working_set.drain(..) {
            process_complete_task(search, &spec, task);
        }

        final_results
    }

    /// Return a working set task and its index. If none exists for the [Spec], start one.
    fn get_task_internal(&mut self, spec: &Spec<Tgt>) -> (usize, Rc<RefCell<SpecTask<Tgt>>>) {
        match self.working_set.entry(spec.clone()) {
            indexmap::map::Entry::Occupied(e) => (e.index(), Rc::clone(e.get())),
            indexmap::map::Entry::Vacant(e) => {
                // Check the database and immediately return if present.
                let task = match self.search.db.get_with_preference(spec) {
                    GetPreference::Hit(v) => {
                        // TODO: Re-enable search hits and misses tracking
                        // search.hits += 1;
                        SpecTask::Complete(v, true)
                    }
                    GetPreference::Miss(preferences) => {
                        let started = SpecTask::start(spec.clone(), preferences, self.search);
                        if matches!(&started, SpecTask::Running { .. }) {
                            self.working_set_running += 1;
                        }
                        started
                    }
                };
                // search.misses += 1;
                let entry_index = e.index();
                let task_rc = Rc::new(RefCell::new(task));
                e.insert(Rc::clone(&task_rc));
                (entry_index, task_rc)
            }
        }
    }

    fn visit_next_request_batch(
        &mut self,
        working_set_spec_idx: usize,
        spec: &Spec<Tgt>,
        task_ref: Rc<RefCell<SpecTask<Tgt>>>,
        visited_in_stage: &mut HashSet<Spec<Tgt>>,
        outbox: &mut Vec<(Spec<Tgt>, ActionCostVec)>, // TODO: Make Option<Cost>
    ) {
        let mut task = task_ref.borrow_mut();
        if !matches!(&*task, SpecTask::Running { .. }) {
            return;
        }

        let page_id = self.search.db.page_id(spec);

        // collect to avoid keeping the borrow
        if let Some(next_batch) = task.next_request_batch().map(|v| v.collect::<Vec<_>>()) {
            for (subspec, request_id) in next_batch {
                if page_id.contains(&subspec) {
                    let (subspec_idx, subtask) = self.get_task_internal(&subspec);
                    if !visited_in_stage.contains(spec) {
                        visited_in_stage.insert(subspec.clone());
                        self.visit_next_request_batch(
                            subspec_idx,
                            &subspec,
                            Rc::clone(&subtask),
                            visited_in_stage,
                            outbox,
                        );
                    }

                    let subtask_ref = subtask.borrow();
                    match &*subtask_ref {
                        SpecTask::Running { .. } => {
                            drop(subtask_ref);
                            self.add_request_mapping_internal(
                                working_set_spec_idx,
                                subspec_idx,
                                request_id,
                            );
                        }
                        SpecTask::Complete(subtask_result, _) => {
                            let cost = subtask_result.iter().next().map(|v| v.1.clone());
                            task.resolve_request(request_id, cost);
                            // At this point, the task_ref might have completed (be
                            // `SpecTask::Complete`). We want to propagate the completion to any
                            // tasks waiting within the working set, but we don't want to recurse
                            // into a Spec we're already borrowing lower on the current stack.
                            // That's overly complicated and will lead to a RefCell borrow panic.
                            // Instead, push the completion into a queue (the `outbox`) we'll
                            // resolve later.
                            if let SpecTask::Complete(completed_task_results, _) = &*task {
                                self.working_set_running -= 1;
                                outbox.push((spec.clone(), completed_task_results.clone()));
                            }
                        }
                    };
                } else {
                    self.add_request_mapping_external(working_set_spec_idx, &subspec, request_id);
                }
            }
        }
    }

    fn resolve_request_internal(&mut self, subspec: &Spec<Tgt>, results: ActionCostVec) {
        Self::inner_resolve_request(
            &self.working_set,
            &mut self.working_set_running,
            RequestsMapRef::Internal(&mut self.working_block_requests),
            None,
            subspec,
            results,
        );
    }

    fn resolve_request_external(
        &mut self,
        subblock: &mut HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>,
        subspec: &Spec<Tgt>,
        results: ActionCostVec,
    ) {
        Self::inner_resolve_request(
            &self.working_set,
            &mut self.working_set_running,
            RequestsMapRef::External(subblock),
            Some(&mut self.working_block_requests),
            subspec,
            results,
        );
    }

    fn inner_resolve_request(
        working_set: &IndexMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
        working_set_running: &mut usize,
        mut subblock: RequestsMapRef<Tgt>,
        next_subblock: Option<&mut HashMap<usize, Vec<WorkingPartialImplHandle>>>,
        subspec: &Spec<Tgt>,
        results: ActionCostVec,
    ) {
        let Some(rs) = subblock.remove(subspec, working_set) else {
            return;
        };

        let mut resolved_next_subblock = next_subblock
            .map(RequestsMapRef::Internal)
            .unwrap_or(subblock);

        let cost = results.0.into_iter().next().map(|v| v.1);
        for (requester_wb_spec_idx, request_id) in rs {
            let (wb_spec, requester_task) = working_set.get_index(requester_wb_spec_idx).unwrap();
            let mut requester = requester_task.borrow_mut();
            // The SpecTask might already be Complete if it was unsat'ed by a prior resolution.
            if matches!(&*requester, SpecTask::Running { .. }) {
                requester.resolve_request(request_id, cost.clone());
                if let SpecTask::Complete(completed_requester_results, _) = &*requester {
                    // TODO: Avoid this clone by consuming the sub-block. (Do at the call site.)
                    *working_set_running -= 1;
                    let cloned_results = completed_requester_results.clone();
                    drop(requester);
                    Self::inner_resolve_request(
                        working_set,
                        working_set_running,
                        // TODO: Can we get rid of the following match? This is just a
                        match &mut resolved_next_subblock {
                            RequestsMapRef::Internal(m) => RequestsMapRef::Internal(m),
                            RequestsMapRef::External(m) => RequestsMapRef::External(m),
                        },
                        None,
                        wb_spec,
                        cloned_results,
                    );
                }
            }
        }
    }

    /// Update `working_block_requests` with a new request for an internal sub-Spec.
    ///
    /// Both Specs must be in the working set.
    fn add_request_mapping_internal(
        &mut self,
        requester_index: usize,
        requested_index: usize,
        request_id: RequestId,
    ) {
        self.working_block_requests
            .entry(requested_index)
            .or_default()
            .push((requester_index, request_id));
    }

    /// Update `subblock_requests` with a new request for the external `subspec` by a task in the
    /// working set.
    fn add_request_mapping_external(
        &mut self,
        working_set_spec_idx: usize,
        subspec: &Spec<Tgt>,
        request_id: RequestId,
    ) {
        let subspec_page = self.search.db.page_id(subspec);
        let request_set = match self
            .subblock_requests
            .iter_mut()
            .find(|s| subspec_page.contains(s.keys().next().unwrap()))
        {
            Some(s) => s,
            None => {
                if self.subblock_requests.is_empty() {
                    let requesting_spec =
                        self.working_set.get_index(working_set_spec_idx).unwrap().0;
                    self.search.db.prefetch(requesting_spec);
                }
                self.subblock_requests.push(HashMap::new());
                self.subblock_requests.last_mut().unwrap()
            }
        };
        request_set
            .entry(subspec.clone())
            .or_default()
            .push((working_set_spec_idx, request_id));
    }
}

impl<Tgt: Target> SpecTask<Tgt> {
    /// Begin computing the optimal implementation of a Spec.
    ///
    /// Internally, this will expand partial [Impl]s for all actions.
    fn start(
        goal: Spec<Tgt>,
        preferences: Option<Vec<ActionIdx>>,
        search: &TopDownSearch<'_>,
    ) -> Self
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let mut reducer = ImplReducer::new(search.top_k, preferences.unwrap_or_default());
        let mut max_children = 0;
        let mut partial_impls = Vec::new();
        let mut partial_impls_incomplete = 0;

        let all_actions = Tgt::actions(&goal.0).collect::<Vec<_>>();
        let initial_skip = search.thread_idx * all_actions.len() / search.thread_count;

        for action_idx in (initial_skip..all_actions.len()).chain(0..initial_skip) {
            let action = &all_actions[action_idx];
            match action.top_down_solver(&goal) {
                Ok(solver) => {
                    let partial_impl_subspecs = solver.subspecs().collect::<Vec<_>>();

                    let subspec_count = partial_impl_subspecs.len();
                    max_children = max_children.max(subspec_count);

                    // If the resulting Impl is already complete, update the reducer. If there
                    // are nested sub-Specs, then store the partial Impl for resolution by the
                    // caller.
                    if partial_impl_subspecs.is_empty() {
                        reducer.insert(
                            u16::try_from(action_idx).unwrap(),
                            solver.compute_cost(iter::empty()),
                        );
                    } else {
                        partial_impls.push(WorkingPartialImpl::Constructing {
                            solver,
                            subspecs: partial_impl_subspecs,
                            subspec_costs: vec![None; subspec_count],
                            producing_action_idx: action_idx.try_into().unwrap(),
                        });
                        partial_impls_incomplete += 1;
                    }
                }
                Err(ApplyError::NotApplicable(_)) => {}
                Err(ApplyError::SpecNotCanonical) => panic!(),
            };
        }

        if partial_impls_incomplete == 0 {
            SpecTask::Complete(ActionCostVec(reducer.finalize()), false)
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

        // TODO: Assert/test that we return unique Specs
        Some(partial_impls.iter().enumerate().filter_map(move |(i, p)| {
            let WorkingPartialImpl::Constructing { subspecs, .. } = p else {
                return None;
            };
            subspecs
                .get(subspec_idx)
                .map(|s| (s.clone(), (i, subspec_idx)))
        }))
    }

    fn resolve_request(
        &mut self,
        id: RequestId,
        cost: Option<Cost>, // `None` means that the Spec was unsat
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let SpecTask::Running {
            reducer,
            partial_impls,
            partial_impls_incomplete,
            request_batches_returned: _,
            max_children: _,
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
        let entry = partial_impls.get_mut(working_impl_idx).unwrap();
        match entry {
            WorkingPartialImpl::Constructing {
                solver,
                subspecs: _,
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
                            solver.compute_cost(
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
                let tmp_replacement = ImplReducer::new(0, Vec::new());
                let removed_reducer: ImplReducer = replace(reducer, tmp_replacement);
                let final_result = removed_reducer.finalize();
                *self = SpecTask::Complete(ActionCostVec(final_result), false);
            }
        }
    }
}

impl ImplReducer {
    pub fn new(top_k: usize, preferences: Vec<ActionIdx>) -> Self {
        debug_assert!(preferences.len() <= top_k);
        debug_assert!(
            preferences.iter().all_unique(),
            "Preferences should not contain duplicates"
        );

        ImplReducer {
            results: if top_k == 1 {
                ImplReducerResults::One(None)
            } else {
                ImplReducerResults::Many(BTreeSet::new())
            },
            top_k,
            preferences,
        }
    }

    pub fn insert(&mut self, new_action_idx: ActionIdx, new_cost: Cost) {
        let new_action = (new_cost, new_action_idx);
        match &mut self.results {
            ImplReducerResults::One(None) => {
                self.results = ImplReducerResults::One(Some(new_action));
            }
            ImplReducerResults::One(Some(action)) if *action > new_action => {
                self.results = ImplReducerResults::One(Some(new_action));
            }
            ImplReducerResults::Many(ref mut actions) => {
                if actions.len() < self.top_k {
                    // We have not yet filled the top_k, so just insert.
                    actions.insert(new_action);
                } else if actions.iter().any(|(cost, _)| *cost == new_action.0) {
                    debug_assert_eq!(actions.len(), self.top_k);

                    // We have filled the top_k and found the same cost in results, so
                    //   replace something if it improves preference count, and do
                    //   nothing if not.
                    if let Some((_, action)) = actions
                        .iter()
                        .enumerate()
                        // Since we know that results is sorted by Cost, this filter
                        //   only takes contiguous elements with the same cost.
                        .filter(|&(i, (cost, _))| {
                            i < self.preferences.len() && *cost == new_action.0
                        })
                        .find(|&(i, _)| self.preferences[i] == new_action.1)
                    {
                        actions.remove(&action.clone());
                        actions.insert(new_action);
                    }
                } else {
                    debug_assert_eq!(actions.len(), self.top_k);

                    // We have filled the top_k, but there is no same cost in results,
                    //   so replace the last element if it is worse than the new one.
                    actions.insert(new_action);
                    actions.pop_last();
                }

                debug_assert!(actions.iter().tuple_windows().all(|(a, b)| a.0 <= b.0));
                debug_assert!(actions.len() <= self.top_k);
                debug_assert!(actions.iter().map(|(_, a)| a).all_unique());
            }
            _ => {}
        }
    }

    fn finalize(self) -> Vec<(ActionIdx, Cost)> {
        match self.results {
            ImplReducerResults::One(None) => vec![],
            ImplReducerResults::One(Some((cost, action_idx))) => vec![(action_idx, cost)],
            ImplReducerResults::Many(actions) => actions
                .into_iter()
                .map(|(cost, action_idx)| (action_idx, cost))
                .collect(),
        }
    }
}

impl<Tgt: Target> RequestsMapRef<'_, Tgt> {
    fn remove(
        &mut self,
        key: &Spec<Tgt>,
        working_set: &IndexMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
    ) -> Option<Vec<WorkingPartialImplHandle>> {
        match self {
            RequestsMapRef::Internal(m) => m.remove(&working_set.get_index_of(key).unwrap()),
            RequestsMapRef::External(m) => m.remove(key),
        }
    }
}

fn process_complete_task<Tgt>(
    search: &TopDownSearch<'_>,
    spec: &Spec<Tgt>,
    task: Rc<RefCell<SpecTask<Tgt>>>,
) -> ActionCostVec
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let SpecTask::Complete(task_result, from_db) = &mut *task.borrow_mut() else {
        unreachable!("Expected goal to be complete.");
    };
    let action_costs = take(task_result);
    if !*from_db {
        search.db.put(spec.clone(), action_costs.0.clone());
    }
    action_costs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DimSize;
    use crate::db::FilesDatabase;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::memorylimits::{MemVec, MemoryLimits};
    use crate::spec::{arb_canonical_spec, LogicalSpec};
    use crate::target::{
        CpuMemoryLevel::{GL, L1, RF},
        X86Target,
    };
    use crate::utils::{bit_length, bit_length_inverse};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use proptest::sample::select;
    use std::rc::Rc;

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);
    const TEST_SMALL_MEM: u64 = 256;

    proptest! {
        // TODO: Add an ARM variant!
        // TODO: Remove restriction to canonical Specs. Should synth. any Spec.
        #[test]
        #[ignore]
        fn test_can_synthesize_any_canonical_spec(
            spec in arb_canonical_spec::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = FilesDatabase::new(None, false, 1, 128, 1);
            top_down(&db, &spec, 1, Some(nz!(1usize)));
        }

        #[test]
        #[ignore]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_canonical_specs::<X86Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = FilesDatabase::new(None, false, 1, 128, 1);

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
            let db = FilesDatabase::new(None, false, 1, 128, 1);
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

        #[test]
        fn test_implreducer_can_sort_any_top_k_actions(
            (top_k, mut action_costs) in arb_top_k_and_action_costs()
        ) {
            let preferences = vec![];
            let mut reducer = ImplReducer::new(top_k, preferences);

            for (cost, action_idx) in &action_costs {
                reducer.insert(*action_idx, cost.clone());
            }

            let finalized = reducer.finalize();
            action_costs.sort();
            assert_eq!(finalized.len(), action_costs.len());

            for (reduced, original) in finalized.into_iter().zip(action_costs.into_iter().map(|(action_idx, cost)| (cost, action_idx))) {
                assert_eq!(reduced, original);
            }
        }
    }

    fn arb_action_indices(top_k: usize) -> impl Strategy<Value = HashSet<ActionIdx>> {
        prop::collection::hash_set(any::<ActionIdx>(), 1..top_k)
    }

    fn arb_costs(top_k: usize) -> impl Strategy<Value = Vec<Cost>> {
        prop::collection::vec(any::<Cost>(), 1..top_k)
    }

    prop_compose! {
        fn arb_top_k_and_action_costs()(top_k in 2..128usize)
        (
            top_k in Just(top_k),
            action_indices in arb_action_indices(top_k),
            costs in arb_costs(top_k)
        ) -> (usize, Vec<(Cost, ActionIdx)>) {
            (top_k, costs.into_iter().zip(action_indices).collect())
        }
    }

    fn create_simple_cost(main: u32) -> Cost {
        Cost {
            main,
            peaks: MemVec::zero::<X86Target>(),
            depth: 0,
        }
    }

    #[test]
    fn test_implreducer_no_actions() {
        let top_k = 1;
        let preferences = vec![];
        let reducer = ImplReducer::new(top_k, preferences);

        let expected: Vec<_> = vec![];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_exactly_one_action() {
        let top_k = 1;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost1.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_cost() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);
        let cost3 = create_simple_cost(3);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost3.clone());
        reducer.insert(2, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1), (2, cost2), (1, cost3)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_action_idx() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost1.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (2, cost1.clone())];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_sort_by_cost_then_action_idx() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(1, cost1.clone());
        reducer.insert(0, cost2.clone());
        reducer.insert(2, cost1.clone());

        let expected: Vec<_> = vec![(1, cost1.clone()), (2, cost1.clone()), (0, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement() {
        let top_k = 3;
        let preferences = vec![0, 2, 3];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost1.clone());
        reducer.insert(2, cost1.clone());
        reducer.insert(3, cost1.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (3, cost1)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement_and_sort_by_cost() {
        let top_k = 3;
        let preferences = vec![0, 2, 3];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost2.clone());
        reducer.insert(1, cost2.clone());
        reducer.insert(2, cost2.clone());
        reducer.insert(3, cost1.clone());

        let expected: Vec<_> = vec![(3, cost1.clone()), (0, cost2.clone()), (1, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_preference_replacement_and_sort_by_cost_then_action_idx() {
        let top_k = 3;
        let preferences = vec![3, u16::MAX, 0];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost2.clone());
        reducer.insert(2, cost1.clone());
        // 0, 2, 1

        reducer.insert(3, cost1.clone());
        // 3, 2, 1 -> 2, 3, 1

        let expected: Vec<_> = vec![(2, cost1.clone()), (3, cost1.clone()), (1, cost2)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_cost_replacement() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);
        let cost3 = create_simple_cost(3);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost3.clone());
        reducer.insert(2, cost3.clone());
        reducer.insert(3, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1), (3, cost2), (1, cost3)];
        assert_eq!(reducer.finalize(), expected);
    }

    #[test]
    fn test_implreducer_no_cost_replacement() {
        let top_k = 3;
        let preferences = vec![];
        let mut reducer = ImplReducer::new(top_k, preferences);

        let cost1 = create_simple_cost(1);
        let cost2 = create_simple_cost(2);

        reducer.insert(0, cost1.clone());
        reducer.insert(1, cost1.clone());
        reducer.insert(2, cost1.clone());
        reducer.insert(3, cost2.clone());

        let expected: Vec<_> = vec![(0, cost1.clone()), (1, cost1.clone()), (2, cost1.clone())];
        assert_eq!(reducer.finalize(), expected, "no replacement should occur");
    }

    // TODO: Add a variant which checks that all Impls have their deps, not just the solution.
    #[test]
    fn test_synthesis_puts_all_dependencies_of_optimal_solution() {
        shared_test_synthesis_puts_all_dependencies_of_optimal_solution(lspec!(Move(
            [2, 2],
            (u8, L1, row_major, ua),
            (u8, RF, row_major, ua),
            serial
        )));
    }

    fn shared_test_synthesis_puts_all_dependencies_of_optimal_solution(
        logical_spec: LogicalSpec<X86Target>,
    ) {
        let spec = Spec::<X86Target>(
            logical_spec,
            MemoryLimits::Standard(MemVec::new_from_binary_scaled([1, 1, 1, 0])),
        );
        let db = FilesDatabase::new(None, false, 1, 128, 1);

        let (action_costs, _, _) = top_down(&db, &spec, 1, Some(nz!(1usize)));

        // Check that the synthesized Impl, include all sub-Impls are in the database. `get_impl`
        // requires all dependencies, so we use that.
        assert!(
            db.get_impl(&spec).is_some(),
            "No Impl stored for Spec: {spec}; top_down returned: {action_costs:?}"
        );
    }

    #[test]
    fn test_synthesis_at_peak_memory_yields_same_decision_1() {
        let spec = Spec::<X86Target>(
            lspec!(FillZero([2, 2, 2, 2], (u8, GL, row_major, c0, ua))),
            MemoryLimits::Standard(MemVec::new_from_binary_scaled([0, 5, 7, 6])),
        );

        let db = FilesDatabase::new(None, false, 1, 128, 1);
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
