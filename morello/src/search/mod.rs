use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::replace;
use std::rc::Rc;
use std::slice;

use crate::cost::Cost;
use crate::db::{ActionCostVec, ActionNum, FilesDatabase};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::ImplNode;
use crate::reconstruct::reconstruct_impls_from_actions;
use crate::spec::Spec;
use crate::target::Target;

pub use reducer::ImplReducer; // TODO: Ideally, ImplReducer isn't `pub`
use spectask::SpecTask;

mod blocksearch;
mod reducer;
mod spectask;
mod utils;

type RequestId = (usize, usize);
type WorkingPartialImplHandle = (usize, RequestId);

struct TopDownSearch<'d, Tgt: Target> {
    db: &'d FilesDatabase,
    top_k: usize,
    thread_idx: usize,
    thread_count: usize,
    phantom: std::marker::PhantomData<Tgt>,
}

// Computes an optimal Impl for `goal` and stores it in `db`.
pub fn top_down<Tgt>(db: &FilesDatabase, goal: &Spec<Tgt>, top_k: usize) -> Vec<(ActionNum, Cost)>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    top_down_many(db, slice::from_ref(goal), top_k)
        .into_iter()
        .next()
        .unwrap()
        .0
}

/// Synthesizes implementations of the given goals.
pub fn top_down_many<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    top_down_many_internal(db, goals, top_k)
}

/// Synthesizes implementations of the given goals using spatial database queries
/// to accelerate batched child lookups.
pub fn top_down_many_spatial<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    top_down_many_internal_with_spatial_queries(db, goals, top_k, true)
}

/// Returns optimal implementations of each goal [Spec].
///
/// In contrast to [top_down_many], this function returns fully materialized [ImplNode]s, not just
/// [ActionCostVec]s.
pub fn top_down_many_impls<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
) -> Vec<Vec<ImplNode<Tgt>>>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let action_costs = top_down_many_internal(db, goals, top_k);
    let lookup = move |spec: &Spec<Tgt>| db.get(spec);
    action_costs
        .into_iter()
        .zip(goals.iter())
        .map(|(costs, goal)| {
            if costs.0.is_empty() {
                Vec::new()
            } else {
                let mut canonical_goal = goal.clone();
                canonical_goal
                    .canonicalize()
                    .expect("should be possible to canonicalize goal Spec");
                reconstruct_impls_from_actions(&lookup, &canonical_goal, costs)
            }
        })
        .collect()
}

fn top_down_many_internal<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    top_down_many_internal_with_spatial_queries(db, goals, top_k, false)
}

/// Shared implementation for top-down synthesis with optional spatial database-page probing.
fn top_down_many_internal_with_spatial_queries<Tgt>(
    db: &FilesDatabase,
    goals: &[Spec<Tgt>],
    top_k: usize,
    spatial_queries: bool,
) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    assert!(db.max_k().is_none_or(|k| k >= top_k));
    if top_k > 1 {
        unimplemented!("Search for top_k > 1 not yet implemented.");
    }

    // Group goals by database page (or `None` if they use the non-spatial cache).
    let mut grouped_canonical_goals = HashMap::<_, (Vec<_>, Vec<usize>)>::new();
    for (idx, goal) in goals.iter().enumerate() {
        let mut canonical_goal = goal.clone();
        canonical_goal
            .canonicalize()
            .expect("should be possible to canonicalize goal Spec");

        let key = db.can_memoize_efficiently(&canonical_goal).then(|| {
            let page = db.page_id(&canonical_goal);
            (page.table_key, page.page_id)
        });
        let group_tuple = grouped_canonical_goals.entry(key).or_default();
        group_tuple.0.push(canonical_goal);
        group_tuple.1.push(idx);
    }

    // Synthesize each group with BlockSearch. Scatter results into combined_results.
    let mut combined_results = vec![Default::default(); goals.len()];
    for (group, original_indices) in grouped_canonical_goals.values() {
        let search = TopDownSearch {
            db,
            top_k,
            thread_idx: 0,
            thread_count: 1,
            phantom: std::marker::PhantomData,
        };
        let mut complete = |spec: &Spec<Tgt>, task: Rc<RefCell<SpecTask<Tgt>>>| {
            process_complete_task(&search, spec, task)
        };
        let result = if spatial_queries {
            blocksearch::synthesize::<true, _, _>(group, &search, None, None, &mut complete)
        } else {
            blocksearch::synthesize::<false, _, _>(group, &search, None, None, &mut complete)
        };

        for (r, &i) in result.into_iter().zip(original_indices) {
            combined_results[i] = r;
        }
    }

    combined_results
}

fn process_complete_task<Tgt>(
    search: &TopDownSearch<'_, Tgt>,
    spec: &Spec<Tgt>,
    task: Rc<RefCell<SpecTask<Tgt>>>,
) -> ActionCostVec
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let Some((task_result, from_db)) = replace(
        &mut *task.borrow_mut(),
        SpecTask::completed(ActionCostVec(Vec::new()), true),
    )
    .into_result() else {
        unreachable!("Expected goal to be complete.");
    };
    if !from_db {
        search.db.put_canon(spec, task_result.0.clone());
    }
    task_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DimSize;
    use crate::db::{FilesDatabase, TileScale};
    use crate::layout::row_major;
    use crate::lspec;
    use crate::memorylimits::{MemVec, MemoryLimits};
    use crate::spec::{arb_canonical_primitive_spec, arb_canonical_spec, LogicalSpec};
    use crate::target::Memory;
    use crate::target::{
        Avx2Target,
        CpuMemory::{GL, L1, RF},
        MEMORY_COUNT,
    };
    use crate::utils::{bit_length, bit_length_inverse};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use proptest::sample::select;
    use std::rc::Rc;

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);
    const TEST_SMALL_MEM: u64 = 64;

    proptest! {
        // applies to this `proptest!` block
        #![proptest_config(ProptestConfig::with_cases(8))]

        // TODO: Add an ARM variant!
        // TODO: Remove restriction to canonical Specs. Should synth. any Spec.
        #[test]
        #[ignore]
        fn test_can_synthesize_any_canonical_primitive_spec(
            spec in arb_canonical_primitive_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 2048, 1);
            top_down(&db, &spec, 1);
        }

        // TODO: Uncomment this test once search performance is back.
        // TODO: Add an ARM variant!
        // TODO: Remove restriction to canonical Specs. Should synth. any Spec.
        // #[test]
        // #[ignore]
        // fn test_can_synthesize_any_canonical_compose_spec(
        //     spec in arb_canonical_compose_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        // ) {
        //     let db = FilesDatabase::new(None, TileScale::Linear, 1, 2048, 1);
        //     top_down(&db, &spec, 1);
        // }

        // TODO: Uncomment this test once search performance is back.
        // TODO: Add an ARM variant!
        // TODO: Remove restriction to canonical Specs. Should synth. any Spec.
        // #[test]
        // #[ignore]
        // fn test_can_synthesize_any_canonical_compose_spec(
        //     spec in arb_canonical_compose_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        // ) {
        //     let db = FilesDatabase::new(None, TileScale::Linear, 1, 2048, 1);
        //     top_down(&db, &spec, 1);
        // }

        #[test]
        #[ignore]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_canonical_specs::<Avx2Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);

            // Solve the first, lower Spec.
            let lower_result_vec = top_down(&db, &spec, 1);

            // If the lower spec can't be solved, then there is no way for the raised Spec to have
            // a worse solution, so we can return here.
            if let Some((_, lower_cost)) = lower_result_vec.first() {
                // Check that the raised result has no lower cost and does not move from being
                // possible to impossible.
                let raised_result = top_down(&db, &raised_spec, 1);
                let (_, raised_cost) = raised_result
                    .first()
                    .expect("raised result should be possible");
                assert!(raised_cost <= lower_cost);
            }
        }

        #[test]
        #[ignore]
        fn test_synthesis_at_peak_memory_yields_same_decision(
            spec in arb_canonical_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);
            let first_solutions = top_down(&db, &spec, 1);
            let first_peak = if let Some(first_sol) = first_solutions.first() {
                first_sol.1.peaks.clone()
            } else {
                MemVec::zero::<Avx2Target>()
            };
            let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
            let lower_solutions = top_down(&db, &lower_spec, 1);
            assert_eq!(first_solutions, lower_solutions);
        }

    }

    // TODO: Add a variant which checks that all Impls have their deps, not just the solution.
    #[test]
    fn test_synthesis_puts_all_dependencies_of_optimal_solution() {
        shared_test_synthesis_puts_all_dependencies_of_optimal_solution(lspec!(Move(
            [2, 2],
            (u8, L1, row_major),
            (u8, RF, row_major),
            serial
        )));
    }

    fn shared_test_synthesis_puts_all_dependencies_of_optimal_solution(
        logical_spec: LogicalSpec<Avx2Target>,
    ) {
        let spec = Spec::<Avx2Target>(
            logical_spec,
            MemoryLimits::Standard(MemVec::new([1, 1, 1, 0])),
        );
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);

        let action_costs = top_down(&db, &spec, 1);

        // Check that the synthesized Impl, include all sub-Impls are in the database. `get_impl`
        // requires all dependencies, so we use that.
        assert!(
            db.get_impl(&spec).is_some(),
            "No Impl stored for Spec: {spec}; top_down returned: {action_costs:?}"
        );
    }

    #[test]
    fn test_synthesis_at_peak_memory_yields_same_decision_1() {
        let spec = Spec::<Avx2Target>(
            lspec!(FillZero([2, 2, 2, 2], (u8, GL, row_major, c0))),
            MemoryLimits::Standard(MemVec::new([0, 16, 64, 32])),
        );

        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);
        let first_solutions = top_down(&db, &spec, 1);
        let first_peak = if let Some(first_sol) = first_solutions.first() {
            first_sol.1.peaks.clone()
        } else {
            MemVec::zero::<Avx2Target>()
        };
        let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
        let lower_solutions = top_down(&db, &lower_spec, 1);
        assert_eq!(first_solutions, lower_solutions);
    }

    #[test]
    fn test_factorized_shape_db_memoizes_representable_subspecs_not_original() {
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::PowerOfTwo, 1, 128, 1);

        // Non-factorizable spec (should be stored in FilesDatabase's non-spatial cache)
        let spec_5 = Spec::<Avx2Target>(
            lspec!(Move([5], (u8, GL, row_major), (u8, RF, row_major))),
            MemoryLimits::Standard(MemVec::new([0, 64, 64, 32])),
        );
        let result = top_down(&db, &spec_5, 1);
        assert!(!result.is_empty(), "Should be able to synthesize Move([5])");
        assert!(
            db.get(&spec_5).is_some(),
            "Database should contain Move([5]) despite not being a factorized size"
        );

        // Factorizable subspecs should be memoized spatially.
        let spec_3 = Spec::<Avx2Target>(
            lspec!(Move([3], (u8, GL, row_major), (u8, RF, row_major))),
            MemoryLimits::Standard(MemVec::new([0, 64, 64, 32])),
        );
        assert!(
            db.get(&spec_3).is_some(),
            "Database should contain Move([3]) after synthesizing Move([5])"
        );
        let spec_2 = Spec::<Avx2Target>(
            lspec!(Move([2], (u8, GL, row_major), (u8, RF, row_major))),
            MemoryLimits::Standard(MemVec::new([0, 64, 64, 32])),
        );
        assert!(
            db.get(&spec_2).is_some(),
            "Database should contain Move([2]) after synthesizing Move([5])"
        );
        let spec_1 = Spec::<Avx2Target>(
            lspec!(Move([1], (u8, GL, row_major), (u8, RF, row_major))),
            MemoryLimits::Standard(MemVec::new([0, 64, 64, 32])),
        );
        assert!(
            db.get(&spec_1).is_some(),
            "Database should contain Move([1]) after synthesizing Move([5])"
        );
    }

    fn lower_and_higher_canonical_specs<Tgt: Target>(
    ) -> impl Strategy<Value = (Spec<Tgt>, Spec<Tgt>)> {
        let MemoryLimits::Standard(mut top_memvec) = Avx2Target::max_mem();
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

                let memories = Tgt::memories();
                let raise_strategy = if memories[dim_idx_to_raise].counts_registers() {
                    let low = spec_memvec.get_unscaled(dim_idx_to_raise);
                    let high = top_memvec.get_unscaled(dim_idx_to_raise);
                    ((low + 1)..=high).boxed()
                } else {
                    let low = bit_length(spec_memvec.get_unscaled(dim_idx_to_raise));
                    let high = bit_length(top_memvec.get_unscaled(dim_idx_to_raise));
                    ((low + 1)..=high).prop_map(bit_length_inverse).boxed()
                };
                (Just(spec), Just(dim_idx_to_raise), raise_strategy)
            })
            .prop_map(|(spec, dim_idx_to_raise, raise_amount)| {
                let MemoryLimits::Standard(base_memvec) = &spec.1;

                // Get current values
                let mut new_values: [u64; MEMORY_COUNT] =
                    base_memvec.iter().collect::<Vec<_>>().try_into().unwrap();

                // Update the specific memory
                new_values[dim_idx_to_raise] = raise_amount;

                // Create encoding flags based on whether each memory counts registers
                let memories = Tgt::memories();
                let encoding_flags: [bool; MEMORY_COUNT] = memories
                    .iter()
                    .map(|memory| memory.counts_registers())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let raised_memory =
                    MemoryLimits::Standard(MemVec::new_mixed(new_values, encoding_flags));
                let raised_spec = Spec(spec.0.clone(), raised_memory);
                (spec, raised_spec)
            })
    }
}
