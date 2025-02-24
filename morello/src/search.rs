use itertools::Itertools;

use std::borrow::Borrow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::rc::Rc;

use crate::cost::{Cost, NormalizedCost};
use crate::db::{ActionCostVec, ActionNum, FilesDatabase, TableKey};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::rtree::RTreeDyn;
use crate::scheduling::{
    Action, ActionT, BottomUpSolver, DependencyRequest, SpecGeometryRect, VisitUpdater,
};
use crate::spec::Spec;
use crate::target::Target;
use crate::utils::diagonals_shifted;

// TODO: Make this private once #[bench] gets stable.
#[doc(hidden)]
#[derive(Debug)]
pub struct ImplReducer {
    results: ImplReducerResults,
    top_k: usize,
    preferences: Vec<ActionNum>,
}

#[derive(Debug)]
enum ImplReducerResults {
    One(Option<(Cost, ActionNum)>),
    Many(BTreeSet<(Cost, ActionNum)>),
}

/// An [VisitUpdater] which logs completed Specs in a Vec.
struct TrackingUpdater<U, K> {
    inner_updater: U,
    goal_solvers_outstanding: HashMap<K, usize>,
}

// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<Tgt>(
    db: &FilesDatabase,
    goal: &Spec<Tgt>,
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> Vec<(ActionNum, Cost)>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    // TODO: Just return the ActionCostVec directly
    top_down_many(db, &[goal.clone()], top_k, jobs)
        .into_iter()
        .next()
        .unwrap()
        .0
}

// TODO: Adapt this to actually have some performance benefit w.r.t. the block-first algorithm.
pub fn top_down_many<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
    jobs: Option<NonZeroUsize>,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    assert!(db.max_k().is_none_or(|k| k >= top_k));
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // Group goals by database page.
    let mut grouped_canonical_goals = HashMap::<_, (Vec<_>, Vec<usize>)>::new();
    for (idx, goal) in goals.iter().enumerate() {
        let mut canonical_goal = goal.clone();
        canonical_goal
            .canonicalize()
            .expect("should be possible to canonicalize goal Spec");

        let page = db.page_id(&canonical_goal);
        let key = (page.table_key, page.page_id);
        let group_tuple = grouped_canonical_goals.entry(key).or_default();
        group_tuple.0.push(canonical_goal);
        group_tuple.1.push(idx);
    }

    // Synthesize each group with BlockSearch. Scatter results into combined_results.
    let mut combined_results = vec![ActionCostVec::default(); goals.len()];
    for (page_group, original_indices) in grouped_canonical_goals.values() {
        // TODO: Deduplicate Specs in `page_group`.
        for goal in page_group.iter() {
            synthesize_block(
                db,
                top_k,
                &SpecGeometryRect::single(goal, Rc::new(db.spec_bimap())),
            );
        }
        for (query, &original_index) in page_group.iter().zip(original_indices) {
            let result = db
                .get(query)
                .unwrap_or_else(|| panic!("db should contain goal after block synthesis: {query}"));
            combined_results[original_index] = result;
        }
    }
    combined_results
}

/// Synthesize all blocks between `low` and `high`, inclusive. Membership is determined by
/// the given `surmap` from [Spec]s to coordinates.
fn synthesize_block<Tgt>(db: &FilesDatabase, top_k: usize, block: &SpecGeometryRect<Tgt>)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    // TODO: Assert or otherwise enforce that the block's bimap equals the `db`'s bimap.

    // Assert that every Spec in the given block is canonical.
    debug_assert!(block.iter_specs().all(|g| g.is_canonical()));

    let mut solvers = Action::bottom_up_solvers().collect::<Vec<_>>();
    let mut requests = vec![]; // map solver indices to their requests
    requests.reserve_exact(solvers.len());

    let mut reducers = HashMap::new();
    let mut goal_solvers_outstanding = HashMap::new();
    block.iter_specs().for_each(|g| {
        reducers.insert(g.clone(), ImplReducer::new(top_k, vec![]));
        goal_solvers_outstanding.insert(g.clone(), solvers.len());
    });
    let mut tracking_updater = TrackingUpdater {
        inner_updater: &mut reducers,
        goal_solvers_outstanding,
    };

    // Build R-Trees of all goals' dependencies.
    // TODO: Use a Tgt-specific ActionT type.
    let mut deps_trees = HashMap::<TableKey, RTreeDyn<usize>>::new();
    for (solver_idx, solver) in solvers.iter_mut().enumerate() {
        // TODO: Call a ranged `apply_no_dependency_updates` equivalent instead of this loop.
        requests.push(solver.request(&block.clone().into()));
        block.iter_specs().for_each(|goal| {
            requests[solver_idx].apply_no_dependency_updates(&goal, &mut tracking_updater);
        });

        if let Some(query) = requests[solver_idx].queries() {
            // TODO: Review below comment.
            //
            // dep_low and dep_high are not guaranteed to be canonical. This can be useful in
            // constructing some ranges, such as those from 1x1 to mxn where the 1x1 point might
            // have different contiguousness.
            query.iter().for_each(|rect| {
                let (dep_pt_low, dep_pt_high) = (rect.bottom_point(), rect.top_point());
                let dep_pt_low_i64 = dep_pt_low
                    .iter()
                    .map(|&x| BimapSInt::from(x))
                    .collect::<Vec<_>>();
                let dep_pt_high_i64 = dep_pt_high
                    .iter()
                    .map(|&x| BimapSInt::from(x))
                    .collect::<Vec<_>>();
                if !deps_trees.contains_key(rect.table_key()) {
                    deps_trees.insert(rect.table_key().clone(), RTreeDyn::empty(dep_pt_low.len()));
                }
                let deps_tree_entry = deps_trees.get_mut(rect.table_key()).unwrap();
                deps_tree_entry.merge_insert(&dep_pt_low_i64, &dep_pt_high_i64, solver_idx);
            });
        }
    }

    // Start satisfying external (non-goal) dependencies. First, iterate over intersecting tiles,
    // then intersection-join with `deps_trees`, passing each to BottomUpSolver::visit_dependency.
    // TODO: Avoid recursion for dependencies which are *not* external!
    for (table_key, deps_tree) in &deps_trees {
        // Walk over every element of the dependency tree to collect all Specs which weren't in the
        // database and aren't current goals. Then recurse. Note that these Specs may span multiple
        // database pages.
        // TODO: Visiting every entry is probably very slow, since it walks over entries that are
        //       already in the database's R-Trees. Ideally, we preserve the geometry all the way
        //       through the solver calls.
        let mut missing_subspecs_rtree = deps_tree.clone();
        db.subtract_from(table_key, &mut missing_subspecs_rtree);

        // TODO: What do we do about the non-canonical points here? Or are they trimmed before put?
        // TODO: Recurse once, not once for each dependency tree.
        missing_subspecs_rtree
            .iter()
            .for_each(|(dep_bottom, dep_top, _)| {
                // TODO: Call synthesize_block once per block, not once per solver.
                let dep_bottom_u32 = dep_bottom
                    .iter()
                    .map(|&x| BimapInt::try_from(x).unwrap())
                    .collect::<Vec<_>>();
                let dep_top_u32 = dep_top
                    .iter()
                    .map(|&x| BimapInt::try_from(x).unwrap())
                    .collect::<Vec<_>>();
                synthesize_block(
                    db,
                    top_k,
                    // TODO: Don't build SpecGeometryRect. The point is preserving the BiMap by construction.
                    &SpecGeometryRect::new(
                        table_key.clone(),
                        dep_bottom_u32,
                        dep_top_u32,
                        block.bimap(),
                    ),
                );
            });

        // Spatial join on the database to collect all dependencies. At this point, after recursion,
        // all external dependencies for that table key should be in the database.
        // TODO: Assert that all dependencies are gathered.
        db.intersect(table_key, deps_tree).for_each(|intersection| {
            // TODO: Instead of resolving Specs individually, resolve ranges (possibly broken by
            //       dimensions/normalization group).
            let solver_idx = intersection.dep_meta;
            let ncosts = intersection
                .action_costs
                .0
                .iter()
                .map(|(_, c)| c.clone())
                .collect::<Vec<_>>();
            requests[solver_idx].visit_dependency(
                &SpecGeometryRect::new(
                    table_key.clone(),
                    intersection.bottom.iter().map(|&x| x as u32).collect(),
                    intersection.top.iter().map(|&x| x as u32).collect(),
                    block.bimap(),
                ),
                &ncosts,
                &mut tracking_updater,
            );

            // TODO: Remove the following
            #[cfg(debug_assertions)]
            diagonals_shifted(&intersection.bottom, &intersection.top)
                .flatten()
                .for_each(|pt| {
                    // For each point (canonical Spec) in the intersection, have each solver
                    // associated with that intersection visit. If that results in a completion of a
                    // Spec, push that onto a queue and loop.
                    let spec: Spec<Tgt> =
                        BiMap::apply_inverse(&db.spec_bimap(), &(table_key.clone(), pt.to_vec()));
                    assert!(spec.is_canonical(), "Spec should be canonical: {spec}");
                    if tracking_updater
                        .goal_solvers_outstanding
                        .contains_key(&spec)
                    {
                        // This can happen if the recursive call incidentally solves a goal as
                        // because it's a dependency (perhaps transitively) of another goal. This
                        // isn't a correctness issue, but it's wasted work which could probably be
                        // prevented.
                        log::warn!("Goal Spec showed up in external visit: {spec}");
                    }
                });
        });
    }

    // TODO: Start pushing results bottom-up within the goal block. Remember that
    //       we need links within coordinates of the space, but not between.
    // TODO: The following placeholder implementation sucks.
    // visit_queue maps Specs to decisions and requesting solver IDs. If there are no
    // solver IDs, then the Specs are just put into the database. If there are, then we
    // will track which Specs are updated as a consequence of a solver visit and enqueue
    // those.
    let mut visit_queue = HashMap::new();
    // Fill the visit_queue with the Specs completed with external dependencies only.
    assert!(
        tracking_updater
            .goal_solvers_outstanding
            .values()
            .any(|&x| x == 0),
        "No Specs were completed with external dependencies only, stalling algorithm: {:?}",
        tracking_updater.goal_solvers_outstanding,
    );

    process_visit_queue(block, &mut tracking_updater, &deps_trees, &mut visit_queue);

    // Repeatedly scan the visit_queue, putting into the database and enqueueing new Specs
    // affected by visits. (Pushes completion up the internal dep. lattice.)
    let mut visited = HashSet::<Spec<Tgt>>::new(); // TODO: Remove `visited`
    while !visit_queue.is_empty() {
        for (spec, (decisions, solver_ids)) in visit_queue.drain() {
            if !visited.insert(spec.clone()) {
                panic!("Spec should not be visited twice: {spec}");
            }

            // TODO: This converts to NormalizedCost, but do solvers just denormalize again?
            let normalized_costs = decisions
                .iter()
                .map(|x| NormalizedCost::new(x.1.clone(), spec.0.volume()))
                .collect::<Vec<_>>();
            for solver_id in solver_ids {
                // TODO: Instead of creating a SpecGeometryRect for each Spec, work at block level.
                requests[solver_id].visit_dependency(
                    &SpecGeometryRect::single(&spec, Rc::new(db.spec_bimap())),
                    &normalized_costs,
                    &mut tracking_updater,
                );
            }

            #[cfg(debug_assertions)]
            if db.get(&spec).is_some() {
                log::warn!("Re-putting {spec}");
            }
            db.put(spec, decisions);
        }

        // TODO: Repeatedly scanning all goals is an inefficient way to dispatch this queue.
        process_visit_queue(block, &mut tracking_updater, &deps_trees, &mut visit_queue);
    }

    #[cfg(debug_assertions)]
    {
        let incomplete_goals = tracking_updater
            .goal_solvers_outstanding
            .iter()
            .filter_map(|(g, &o)| {
                if o != usize::MAX {
                    Some(format!("{g}[{o}]"))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if !incomplete_goals.is_empty() {
            panic!(
                "Some goals were not completed: {}",
                incomplete_goals.join(", ")
            );
        }
    }
}

/// Scan the queue and process any Specs for which all solvers are complete.
fn process_visit_queue<Tgt>(
    block: &SpecGeometryRect<Tgt>,
    tracking_updater: &mut TrackingUpdater<&mut HashMap<Spec<Tgt>, ImplReducer>, Spec<Tgt>>,
    deps_trees: &HashMap<TableKey, RTreeDyn<usize>>,
    visit_queue: &mut HashMap<Spec<Tgt>, (Vec<(ActionNum, Cost)>, Vec<usize>)>,
) where
    Tgt: Target,
{
    block.iter_specs().for_each(|goal| {
        let incomplete_solvers = tracking_updater
            .goal_solvers_outstanding
            .get_mut(&goal)
            .unwrap();
        if *incomplete_solvers == 0 {
            let (spec_db_key, spec_pt) = block.bimap().apply(&goal);
            let spec_pt_i64 = spec_pt
                .iter()
                .map(|&x| BimapSInt::from(x))
                .collect::<Vec<_>>();
            let requesting_solver_idxs = deps_trees
                .get(&spec_db_key)
                .map(|t| t.locate_all_at_point(&spec_pt_i64).copied().collect())
                .unwrap_or_default();
            let reducer = tracking_updater.inner_updater.remove(&goal).unwrap();
            let decisions = reducer.finalize();
            visit_queue.insert(goal, (decisions, requesting_solver_idxs));
            *incomplete_solvers = usize::MAX;
        }
    });
}

impl ImplReducer {
    pub fn new(top_k: usize, preferences: Vec<ActionNum>) -> Self {
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

    pub fn insert(&mut self, new_action_num: ActionNum, new_cost: Cost) {
        let new_action = (new_cost, new_action_num);
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

    fn finalize(self) -> Vec<(ActionNum, Cost)> {
        match self.results {
            ImplReducerResults::One(None) => vec![],
            ImplReducerResults::One(Some((cost, action_num))) => vec![(action_num, cost)],
            ImplReducerResults::Many(actions) => actions
                .into_iter()
                .map(|(cost, action_num)| (action_num, cost))
                .collect(),
        }
    }
}

impl<Tgt, U, K> VisitUpdater<Tgt> for TrackingUpdater<U, K>
where
    Tgt: Target,
    U: VisitUpdater<Tgt>,
    K: Borrow<Spec<Tgt>> + Eq + Hash,
{
    fn complete_action(
        &mut self,
        spec: &Spec<Tgt>,
        action: ActionNum,
        normalized_cost: NormalizedCost,
    ) {
        self.inner_updater
            .complete_action(spec, action, normalized_cost.clone());
    }

    fn complete_spec(&mut self, spec: &Spec<Tgt>) {
        self.inner_updater.complete_spec(spec);
        let outstanding = self.goal_solvers_outstanding.get_mut(spec).unwrap();
        debug_assert_ne!(*outstanding, usize::MAX);
        *outstanding -= 1;
    }
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
    use std::collections::HashSet;
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
            let lower_result_vec = top_down(&db, &spec, 1, Some(nz!(1usize)));

            // If the lower spec can't be solved, then there is no way for the raised Spec to have
            // a worse solution, so we can return here.
            if let Some((_, lower_cost)) = lower_result_vec.first() {
                // Check that the raised result has no lower cost and does not move from being
                // possible to impossible.
                let raised_result = top_down(&db, &raised_spec, 1, Some(nz!(1usize)));
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
            let first_solutions = top_down(&db, &spec, 1, Some(nz!(1usize)));
            let first_peak = if let Some(first_sol) = first_solutions.first() {
                first_sol.1.peaks.clone()
            } else {
                MemVec::zero::<X86Target>()
            };
            let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
            let lower_solutions = top_down(&db, &lower_spec, 1, Some(nz!(1usize)));
            assert_eq!(first_solutions, lower_solutions);
        }

        #[test]
        fn test_implreducer_can_sort_any_top_k_actions(
            (top_k, mut action_costs) in arb_top_k_and_action_costs()
        ) {
            let preferences = vec![];
            let mut reducer = ImplReducer::new(top_k, preferences);

            for (cost, action_num) in &action_costs {
                reducer.insert(*action_num, cost.clone());
            }

            let finalized = reducer.finalize();
            action_costs.sort();
            assert_eq!(finalized.len(), action_costs.len());

            for (reduced, original) in finalized.into_iter().zip(action_costs.into_iter().map(|(action_num, cost)| (cost, action_num))) {
                assert_eq!(reduced, original);
            }
        }
    }

    fn arb_action_indices(top_k: usize) -> impl Strategy<Value = HashSet<ActionNum>> {
        prop::collection::hash_set(any::<ActionNum>(), 1..top_k)
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
        ) -> (usize, Vec<(Cost, ActionNum)>) {
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
    fn test_implreducer_sort_by_action_num() {
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
    fn test_implreducer_sort_by_cost_then_action_num() {
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
    fn test_implreducer_preference_replacement_and_sort_by_cost_then_action_num() {
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

        let action_costs = top_down(&db, &spec, 1, Some(nz!(1usize)));

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
        let first_solutions = top_down(&db, &spec, 1, Some(nz!(1usize)));
        let first_peak = if let Some(first_sol) = first_solutions.first() {
            first_sol.1.peaks.clone()
        } else {
            MemVec::zero::<X86Target>()
        };
        let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
        let lower_solutions = top_down(&db, &lower_spec, 1, Some(nz!(1usize)));
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
