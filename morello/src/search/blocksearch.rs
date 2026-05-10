use indexmap::IndexMap;
use itertools::Itertools;

use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::replace;
use std::rc::Rc;

use super::{utils::StageVisitSet, SpecTask, TopDownSearch, WorkingPartialImplHandle};
use crate::db::{ActionCostVec, ActionNum, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::search::RequestId;
use crate::spec::Spec;
use crate::target::Target;

type TaskHandle<Tgt> = (usize, Rc<RefCell<SpecTask<Tgt>>>);

/// State for one block of top-down synthesis.  This struct exists to support [synthesize], which
/// passes references to this [BlockSearch] into its many helpers.
///
/// The top-level search groups requested goals by database page (or a unique group for
/// non-memoizable [Spec]s). Each group is solved by a `BlockSearch`, which drives a stable
/// `working_set` of `SpecTask`s until every task in the block completes.
struct BlockSearch<'a, 'd, Tgt: Target> {
    search: &'a TopDownSearch<'d, Tgt>,
    /// Specs owned by this block and their running or completed [SpecTask]s.
    ///
    /// Entries are not removed while the `synthesize` solve loop is active, so indices are stable.
    /// The block is consumed before entries are drained, making later reindexing unobservable.
    ///
    /// Note that this is an [IndexMap], not a [HashMap], so that pending request handles can store
    /// indices, which is a simple way to avoid a bunch of hashing.
    working_set: IndexMap<Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>>,
    /// Number of [SpecTask]s in `working_set` that are still running.
    working_set_running: usize,
    /// Dependency edges within the work set waiting for a task to complete.
    ///
    /// Keys are indices into `working_set` for requested child Specs. Values identify parent
    /// partial implementations that requested that child result. A requester can become stale
    /// before the child resolves if another dependency already made that parent partial
    /// implementation unsatisfiable.
    working_block_requests: HashMap<usize, Vec<WorkingPartialImplHandle>>,
    /// External dependencies, grouped by subblock, that will be resolved by recursive [synthesize]
    /// calls.
    ///
    /// Note that a non-memoizable child requested by a non-memoizable parent stays in `working_set`
    /// instead.
    subblock_requests: Vec<HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>>,
}

/// Results from [BlockSearch::query_database_page_and_resolve_hits].
struct QueryDatabasePageResult<Tgt: Target> {
    goals: Vec<Spec<Tgt>>,
    requesters: Vec<Vec<WorkingPartialImplHandle>>,
    same_page_batch_misses: Option<Vec<Option<Vec<ActionNum>>>>,
}

impl<'a, 'd, Tgt> BlockSearch<'a, 'd, Tgt>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    /// Inserts the initial goals into the working set and visits their first request batches.
    fn seed_goals(
        &mut self,
        goals: &[Spec<Tgt>],
        same_page_batch_misses: Option<Vec<Option<Vec<ActionNum>>>>,
        visited_in_stage: &mut StageVisitSet,
        outbox: &mut Vec<(usize, ActionCostVec)>,
    ) {
        let mut same_page_batch_misses = same_page_batch_misses.map(|misses| misses.into_iter());
        for goal in goals {
            let (spec_working_set_index, task) = match &mut same_page_batch_misses {
                Some(misses) => {
                    self.get_task_from_same_page_batch_miss(goal, misses.next().unwrap())
                }
                None => self.get_task_internal(goal),
            };
            if !visited_in_stage.insert(spec_working_set_index) {
                continue;
            }
            self.visit_next_request_batch(
                spec_working_set_index,
                Rc::clone(&task),
                visited_in_stage,
                outbox,
            );
        }
    }

    /// Synthesizes all pending external child-Spec request groups and resolves their requesters.
    fn synthesize_subblocks<const SPATIAL: bool, F>(
        &mut self,
        prefetch_after: Option<&Spec<Tgt>>,
        complete_task: &mut F,
    ) where
        F: FnMut(&Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>) -> ActionCostVec,
    {
        let new_vec = Vec::with_capacity(self.subblock_requests.len());
        let mut subblock_reqs_iter = replace(&mut self.subblock_requests, new_vec)
            .into_iter()
            .peekable();
        while let Some(subblock) = subblock_reqs_iter.next() {
            let query_page_result = if SPATIAL {
                self.query_database_page_and_resolve_hits(subblock)
            } else {
                QueryDatabasePageResult::from_subblock(subblock)
            };
            if query_page_result.goals.is_empty() {
                continue;
            }

            let prefetch_to_push_down =
                self.prefetch_to_push_down(subblock_reqs_iter.peek(), prefetch_after);
            let synth_result = synthesize::<SPATIAL, _, _>(
                &query_page_result.goals,
                self.search,
                prefetch_to_push_down,
                query_page_result.same_page_batch_misses,
                complete_task,
            );
            for (requesters, subspec_result) in
                query_page_result.requesters.into_iter().zip(synth_result)
            {
                self.resolve_requesters(requesters, subspec_result);
            }
        }
    }

    /// Chooses the Spec that should be prefetched by the next recursive synthesis call.
    fn prefetch_to_push_down<'s>(
        &self,
        next_subblock: Option<&'s HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>>,
        prefetch_after: Option<&Spec<Tgt>>,
    ) -> Option<&'s Spec<Tgt>> {
        // TODO: Move prefetch so that it happens after the get inside the recursive call.
        if let Some(next_subblock) = next_subblock {
            return next_subblock.keys().next();
        }

        if let Some(prefetch_after) = prefetch_after.as_ref() {
            if self.search.db.can_memoize_efficiently(prefetch_after) {
                self.search.db.prefetch_canon(prefetch_after);
            }
        }
        None
    }

    /// Visits the next request batch for every task that is still running in this working set.
    fn visit_running_tasks(
        &mut self,
        visited_in_stage: &mut StageVisitSet,
        outbox: &mut Vec<(usize, ActionCostVec)>,
    ) {
        let ws_vec = self
            .working_set
            .iter()
            .enumerate()
            .filter(|(_, (_, task))| task.borrow().is_running())
            .map(|(spec_idx, (_, task))| (spec_idx, Rc::clone(task)))
            .collect::<Vec<_>>();
        visited_in_stage.reset_generation();
        for (spec_idx, task_ref) in ws_vec {
            self.visit_next_request_batch(spec_idx, task_ref, visited_in_stage, outbox);
        }
    }

    /// Removes completed tasks from the working set, calls `complete_task` for each one, and
    /// returns the completed results for `goals`.
    fn finish<F>(mut self, goals: &[Spec<Tgt>], complete_task: &mut F) -> Vec<ActionCostVec>
    where
        F: FnMut(&Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>) -> ActionCostVec,
    {
        // Gather all tasks requested by synthesize. This removes from the working set.
        //
        // After this point, we'll be removing entries from working_set, so
        // WorkingPartialImplHandles will not be valid.
        let final_results = goals
            .iter()
            .map(|g| complete_task(g, self.working_set.swap_remove(g).unwrap()))
            .collect::<Vec<_>>();

        // Anything left in the working set is not a goal but should still be put
        for (spec, task) in self.working_set.drain(..) {
            complete_task(&spec, task);
        }

        final_results
    }

    /// Return a working set task and its index. If none exists for the [Spec], start one.
    fn get_task_internal(&mut self, spec: &Spec<Tgt>) -> TaskHandle<Tgt> {
        let search = self.search;
        if let Some((idx, _, task)) = self.working_set.get_full(spec) {
            return (idx, Rc::clone(task));
        }

        let preferences = match search.db.get_with_preference_canon(spec) {
            GetPreference::Hit(result) => {
                return self.insert_task(spec, SpecTask::completed(result, true));
            }
            GetPreference::Miss(preferences) => preferences,
        };
        self.start_task(spec, preferences)
    }

    /// Return a working set task for a memoizable [Spec] that already missed in a same-page
    /// database batch query. This path never performs a database get.
    fn get_task_from_same_page_batch_miss(
        &mut self,
        spec: &Spec<Tgt>,
        batch_miss: Option<Vec<ActionNum>>,
    ) -> TaskHandle<Tgt> {
        debug_assert!(self.search.db.can_memoize_efficiently(spec));
        if let Some((idx, _, task)) = self.working_set.get_full(spec) {
            return (idx, Rc::clone(task));
        }

        self.start_task(spec, batch_miss)
    }

    /// Starts a task in a vacant working-set slot and updates the running-task count.
    fn start_task(
        &mut self,
        spec: &Spec<Tgt>,
        preferences: Option<Vec<ActionNum>>,
    ) -> TaskHandle<Tgt> {
        let task = SpecTask::start(spec.clone(), preferences, self.search);
        // search.misses += 1;
        self.insert_task(spec, task)
    }

    /// Inserts a task into a vacant working-set slot and returns its handle.
    fn insert_task(&mut self, spec: &Spec<Tgt>, task: SpecTask<Tgt>) -> TaskHandle<Tgt> {
        if task.is_running() {
            self.working_set_running += 1;
        }
        let task_ref = Rc::new(RefCell::new(task));
        let (entry_index, old_task) = self
            .working_set
            .insert_full(spec.clone(), Rc::clone(&task_ref));
        debug_assert!(old_task.is_none());
        (entry_index, task_ref)
    }

    /// Advances one running task by asking for and routing its next group of child [Spec]s.
    ///
    /// A [SpecTask] is tracking many candidate implementations for the same parent [Spec]. Each
    /// candidate can require several child [Spec]s before its cost can be computed. The task
    /// exposes those children one position at a time: first every candidate's first child, then
    /// every candidate's second child, and so on. This method processes one such group.
    ///
    /// Here "visit" means: route each requested child [Spec]. Same-page or same-non-memoizable
    /// children are kept in this block's `working_set`; this method recursively advances each such
    /// child at most once per stage, resolves the parent immediately if the child is already
    /// complete, or records a `working_block_requests` edge if it is still running. Children
    /// outside this block are appended to `subblock_requests` for later recursive synthesis.
    ///
    /// If resolving a same-block child completes the current task, its result is pushed to
    /// `outbox` instead of being propagated immediately. That avoids recursively borrowing tasks
    /// already on the current stack.
    fn visit_next_request_batch(
        &mut self,
        working_set_spec_idx: usize,
        task_ref: Rc<RefCell<SpecTask<Tgt>>>,
        visited_in_stage: &mut StageVisitSet,
        outbox: &mut Vec<(usize, ActionCostVec)>, // TODO: Make Option<Cost>
    ) {
        let db = self.search.db;
        let page_id_opt = {
            let spec = self.working_set.get_index(working_set_spec_idx).unwrap().0;
            db.can_memoize_efficiently(spec).then(|| db.page_id(spec))
        };
        let mut task = task_ref.borrow_mut();
        if !task.is_running() {
            return;
        }

        // collect to avoid keeping the borrow
        if let Some(next_batch) = task.next_request_batch().map(|v| v.collect::<Vec<_>>()) {
            for (subspec, request_id) in next_batch {
                let can_memoize_efficiently = db.can_memoize_efficiently(&subspec);
                let in_same_page = match (page_id_opt.as_ref(), can_memoize_efficiently) {
                    (Some(pid), true) => pid.contains(&subspec),
                    (None, false) => true, // Both non-spatial-memoizable
                    _ => false,
                };
                if in_same_page {
                    let (subspec_idx, subtask) = self.get_task_internal(&subspec);
                    if visited_in_stage.insert(subspec_idx) {
                        self.visit_next_request_batch(
                            subspec_idx,
                            Rc::clone(&subtask),
                            visited_in_stage,
                            outbox,
                        );
                    }

                    let subtask_ref = subtask.borrow();
                    if let Some(subtask_result) = subtask_ref.result() {
                        let cost = subtask_result.iter().next().map(|v| v.1.clone());
                        task.resolve_request(request_id, cost);
                        // At this point, the task_ref might have completed. We want to propagate
                        // the completion to any tasks waiting within the working set, but we don't
                        // want to recurse into a Spec we're already borrowing lower on the current
                        // stack. That's overly complicated and will lead to a RefCell borrow panic.
                        // Instead, push the completion into a queue (the `outbox`) we'll resolve
                        // later.
                        if let Some(completed_task_results) = task.result() {
                            self.working_set_running -= 1;
                            outbox.push((working_set_spec_idx, completed_task_results.clone()));
                        }
                    } else {
                        drop(subtask_ref);
                        self.working_block_requests
                            .entry(subspec_idx)
                            .or_default()
                            .push((working_set_spec_idx, request_id));
                    }
                } else {
                    self.add_request_mapping_external(working_set_spec_idx, &subspec, request_id);
                }
            }
        }
    }

    /// Looks up a set of same-page Specs and resolves memoized hits.
    ///
    /// Every Spec in `subblock` should belong to the same database page. This probes that page
    /// once, resolves any hits by updating the corresponding `WorkingPartialImplHandle`s, and
    /// returns the misses paired with their original requesters and miss preferences.
    fn query_database_page_and_resolve_hits(
        &mut self,
        subblock: HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>,
    ) -> QueryDatabasePageResult<Tgt> {
        let db = self.search.db;
        let entries = subblock.into_iter().collect::<Vec<_>>();

        let Some((first_spec, _)) = entries.first() else {
            return QueryDatabasePageResult::from_subblock(HashMap::new());
        };
        if !db.can_memoize_efficiently(first_spec) {
            return QueryDatabasePageResult::from_subblock(entries.into_iter().collect());
        }

        let memoized_results = {
            let memoizable_specs = entries.iter().map(|(spec, _)| spec).collect_vec();
            debug_assert!(memoizable_specs
                .iter()
                .all(|spec| db.can_memoize_efficiently(spec)));
            db.get_same_page_many_canon(&memoizable_specs)
        };
        let mut goals = Vec::with_capacity(entries.len());
        let mut requesters = Vec::with_capacity(entries.len());
        let mut same_page_batch_misses = Vec::with_capacity(entries.len());
        for ((spec, spec_requesters), lookup) in entries.into_iter().zip(memoized_results) {
            match lookup {
                GetPreference::Hit(result) => {
                    self.resolve_requesters(spec_requesters, result);
                }
                GetPreference::Miss(preferences) => {
                    goals.push(spec);
                    requesters.push(spec_requesters);
                    same_page_batch_misses.push(preferences);
                }
            }
        }

        QueryDatabasePageResult {
            goals,
            requesters,
            same_page_batch_misses: Some(same_page_batch_misses),
        }
    }

    fn inner_resolve_request(&mut self, subspec_idx: usize, results: ActionCostVec) {
        let Some(rs) = self.working_block_requests.remove(&subspec_idx) else {
            return;
        };

        self.resolve_requesters(rs, results);
    }

    fn resolve_requesters(&mut self, rs: Vec<WorkingPartialImplHandle>, results: ActionCostVec) {
        let cost = results.0.into_iter().next().map(|v| v.1);
        for (requester_wb_spec_idx, request_id) in rs {
            let requester_task =
                Rc::clone(self.working_set.get_index(requester_wb_spec_idx).unwrap().1);
            let mut requester = requester_task.borrow_mut();
            // The SpecTask might already be complete if it was unsat'ed by a prior resolution.
            if requester.is_running() {
                requester.resolve_request(request_id, cost.clone());
                if let Some(completed_requester_results) = requester.result().cloned() {
                    // TODO: Avoid this clone by threading completed requester results through
                    // the recursive resolution path.
                    self.working_set_running -= 1;
                    drop(requester);
                    self.inner_resolve_request(requester_wb_spec_idx, completed_requester_results);
                }
            }
        }
    }

    /// Update `subblock_requests` with a new request for the external `subspec` by a task in the
    /// working set.
    fn add_request_mapping_external(
        &mut self,
        working_set_spec_idx: usize,
        subspec: &Spec<Tgt>,
        request_id: RequestId,
    ) {
        let request_set = if self.search.db.can_memoize_efficiently(subspec) {
            let subspec_page = self.search.db.page_id(subspec);
            let matching_idx = self.subblock_requests.iter().position(|subblock| {
                let existing_spec = subblock.keys().next().unwrap();
                self.search.db.can_memoize_efficiently(existing_spec)
                    && subspec_page.contains(existing_spec)
            });
            if let Some(idx) = matching_idx {
                self.subblock_requests.get_mut(idx).unwrap()
            } else {
                if self.subblock_requests.is_empty() {
                    let requesting_spec =
                        self.working_set.get_index(working_set_spec_idx).unwrap().0;
                    if self.search.db.can_memoize_efficiently(requesting_spec) {
                        self.search.db.prefetch_canon(requesting_spec);
                    }
                }
                self.new_subblock_request_set()
            }
        } else {
            // Always create a fresh subblock for "non-spatial" specs to avoid
            // page computations.
            self.new_subblock_request_set()
        };
        request_set
            .entry(subspec.clone())
            .or_default()
            .push((working_set_spec_idx, request_id));
    }

    /// Allocate and return a request map for a fresh subblock (a set of sub-Specs which comprise a
    /// unit of work after recursion).
    fn new_subblock_request_set(
        &mut self,
    ) -> &mut HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>> {
        self.subblock_requests.push(HashMap::new());
        self.subblock_requests
            .last_mut()
            .expect("newly pushed subblock should exist")
    }

    /// Asserts `working_set_running` equals the number of `SpecTask::Running`s in `working_set`.
    fn assert_running_count_is_consistent(&self) {
        debug_assert_eq!(
            self.working_set_running,
            self.working_set
                .values()
                .filter(|v| v.borrow().is_running())
                .count()
        );
    }

    /// Asserts that no request mappings remain after all tasks have completed.
    fn assert_no_pending_requests(&self) {
        debug_assert!(
            self.working_block_requests.is_empty(),
            "working_block_requests is not empty: {}",
            self.working_block_requests
                .keys()
                .map(|k| format!("{k}"))
                .join(", ")
        );
        debug_assert!(self.subblock_requests.is_empty());
    }
}

impl<Tgt: Target> QueryDatabasePageResult<Tgt> {
    /// Converts a request map into recursive goals.
    fn from_subblock(subblock: HashMap<Spec<Tgt>, Vec<WorkingPartialImplHandle>>) -> Self {
        let (goals, requesters) = subblock.into_iter().unzip();
        Self {
            goals,
            requesters,
            same_page_batch_misses: None,
        }
    }
}

/// Synthesizes all `goals` in one working set until each has completed. All goals should have the
/// same database page (or, if not memoizable, should all be non-memoizable).
///
/// When `same_page_batch_misses` is present, it must be aligned with `goals` and is used to
/// initialize each goal from a database miss observed by a prior same-page batch lookup.
///
/// `complete_task` is called exactly once for every completed [SpecTask] that leaves this
/// block, including non-goal dependencies discovered while synthesizing `goals`. The callback
/// owns any side effects associated with completion, such as memoizing synthesized results, and
/// must return the task's [ActionCostVec].
pub fn synthesize<'a, 'd, const SPATIAL: bool, Tgt, F>(
    goals: &[Spec<Tgt>],
    search: &'a TopDownSearch<'d, Tgt>,
    prefetch_after: Option<&Spec<Tgt>>,
    same_page_batch_misses: Option<Vec<Option<Vec<ActionNum>>>>,
    complete_task: &mut F,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    F: FnMut(&Spec<Tgt>, Rc<RefCell<SpecTask<Tgt>>>) -> ActionCostVec,
{
    debug_assert!(goals.iter().all_unique());
    debug_assert!(same_page_batch_misses
        .as_ref()
        .is_none_or(|misses| misses.len() == goals.len()));

    let mut visited_in_stage = StageVisitSet::new(goals.len());
    let mut outbox = Vec::new();
    let mut block = BlockSearch {
        search,
        working_set: IndexMap::with_capacity(goals.len()),
        working_set_running: 0,
        working_block_requests: HashMap::new(),
        subblock_requests: Vec::new(),
    };
    block.seed_goals(
        goals,
        same_page_batch_misses,
        &mut visited_in_stage,
        &mut outbox,
    );

    loop {
        for (spec_idx, completed_task_results) in outbox.drain(..) {
            block.inner_resolve_request(spec_idx, completed_task_results);
        }
        block.synthesize_subblocks::<SPATIAL, F>(prefetch_after, complete_task);

        block.assert_running_count_is_consistent();
        if block.working_set_running == 0 {
            break;
        }

        block.visit_running_tasks(&mut visited_in_stage, &mut outbox);
    }
    block.assert_no_pending_requests();
    block.finish(goals, complete_task)
}
