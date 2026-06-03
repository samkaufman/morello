mod algo;
mod bindings;
mod reducer;
mod spectask;

use crate::cost::Cost;
use crate::db::{ActionCostVec, ActionNum, FilesDatabase};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::imp::ImplNode;
use crate::reconstruct::reconstruct_impls_from_actions;
use crate::spec::Spec;
use crate::target::Target;

use bindings::SpecProblem;
pub use reducer::ImplReducer; // TODO: Ideally, ImplReducer isn't `pub`
use spectask::SpecTask;

type RequestId = (usize, usize);

pub fn top_down_many<Tgt>(db: &FilesDatabase, goals: &[Spec<Tgt>]) -> Vec<ActionCostVec>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    assert!(db.max_k().is_none_or(|k| k >= 1));

    let problem = SpecProblem::<Tgt>::default();

    goals
        .iter()
        .map(|goal| {
            let mut canonical_goal = goal.clone();
            canonical_goal
                .canonicalize()
                .expect("should be possible to canonicalize goal Spec");
            debug_assert!(canonical_goal.is_canonical());
            algo::solve(problem.clone(), db, canonical_goal)
        })
        .collect()
}

pub fn top_down<Tgt>(db: &FilesDatabase, goal: &Spec<Tgt>) -> Vec<(ActionNum, Cost)>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    top_down_many(db, std::slice::from_ref(goal))
        .into_iter()
        .next()
        .unwrap()
        .0
}

/// Returns optimal implementations of each goal [Spec].
pub fn top_down_many_impls<Tgt>(db: &FilesDatabase, goals: &[Spec<Tgt>]) -> Vec<Vec<ImplNode<Tgt>>>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let canonical_goals = goals
        .iter()
        .map(|goal| {
            let mut canonical_goal = goal.clone();
            canonical_goal
                .canonicalize()
                .expect("should be possible to canonicalize goal Spec");
            canonical_goal
        })
        .collect::<Vec<_>>();
    let action_costs = top_down_many(db, &canonical_goals);
    let lookup = move |spec: &Spec<Tgt>| db.get(spec);
    action_costs
        .into_iter()
        .zip(canonical_goals.iter())
        .map(|(costs, goal)| {
            if costs.0.is_empty() {
                Vec::new()
            } else {
                reconstruct_impls_from_actions(&lookup, goal, costs)
            }
        })
        .collect()
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
        CpuMemory::{GL, L1},
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
            top_down_one(&db, &spec);
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
        //     top_down_one(&db, &spec);
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
        //     top_down_one(&db, &spec);
        // }

        #[test]
        #[ignore]
        fn test_more_memory_never_worsens_solution_with_shared_db(
            spec_pair in lower_and_higher_canonical_specs::<Avx2Target>()
        ) {
            let (spec, raised_spec) = spec_pair;
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);

            // Solve the first, lower Spec.
            let lower_result_vec = top_down_one(&db, &spec);

            // If the lower spec can't be solved, then there is no way for the raised Spec to have
            // a worse solution, so we can return here.
            if let Some((_, lower_cost)) = lower_result_vec.0.first() {
                // Check that the raised result has no lower cost and does not move from being
                // possible to impossible.
                let raised_result = top_down_one(&db, &raised_spec);
                let (_, raised_cost) = raised_result
                    .0
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
            println!("synth (new) test: {spec}");

            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);
            let first_solutions = top_down_many(&db, std::slice::from_ref(&spec));
            let first_peak = if let Some(first_sol) = first_solutions.first() {
                first_sol
                    .0
                    .first()
                    .map(|(_, cost)| cost.peaks.clone())
                    .unwrap_or_else(MemVec::zero::<Avx2Target>)
            } else {
                MemVec::zero::<Avx2Target>()
            };
            let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
            let lower_solutions = top_down_many(&db, std::slice::from_ref(&lower_spec));
            assert_eq!(first_solutions, lower_solutions);
        }
    }

    // TODO: Add a variant which checks that all Impls have their deps, not just the solution.
    #[test]
    fn test_synthesis_puts_all_dependencies_of_optimal_solution() {
        shared_test_synthesis_puts_all_dependencies_of_optimal_solution(lspec!(Move(
            [2, 2],
            (u8, GL, row_major),
            (u8, L1, row_major),
            serial
        )));
    }

    fn shared_test_synthesis_puts_all_dependencies_of_optimal_solution(
        logical_spec: LogicalSpec<Avx2Target>,
    ) {
        let spec = Spec::<Avx2Target>(
            logical_spec,
            MemoryLimits::Standard(MemVec::new([1; MEMORY_COUNT])),
        );
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);

        let action_costs = top_down_one(&db, &spec);

        // Check that the synthesized Impl, include all sub-Impls are in the database. `get_impl`
        // requires all dependencies, so we use that.
        assert!(
            db.get_impl(&spec).is_some(),
            "No Impl stored for Spec: {spec}; top_down_many returned: {action_costs:?}"
        );
    }

    #[test]
    fn test_synthesis_at_peak_memory_yields_same_decision_1() {
        let spec = Spec::<Avx2Target>(
            lspec!(FillZero([2, 2, 2, 2], (u8, GL, row_major, c0))),
            MemoryLimits::Standard(MemVec::new_for_cpu([0, 16, 64, 32])),
        );

        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);
        let first_solutions = top_down_many(&db, std::slice::from_ref(&spec));
        let first_peak = if let Some(first_sol) = first_solutions.first() {
            first_sol
                .0
                .first()
                .map(|(_, cost)| cost.peaks.clone())
                .unwrap_or_else(MemVec::zero::<Avx2Target>)
        } else {
            MemVec::zero::<Avx2Target>()
        };
        let lower_spec = Spec(spec.0, MemoryLimits::Standard(first_peak));
        let lower_solutions = top_down_many(&db, std::slice::from_ref(&lower_spec));
        assert_eq!(first_solutions, lower_solutions);
    }

    #[test]
    fn test_binary_scale_db_memoizes_non_factored_spec_pow2() {
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::PowerOfTwo, 1, 128, 1);

        // Non-power-of-two spec (should be stored in FilesDatabase's non-spatial cache)
        let spec_5 = Spec::<Avx2Target>(
            lspec!(Move([5], (u8, GL, row_major), (u8, L1, row_major))),
            MemoryLimits::Standard(MemVec::new_for_cpu([0, 64, 64, 32])),
        );

        let result = top_down(&db, &spec_5);
        assert!(!result.is_empty(), "Should be able to synthesize Move([5])");
        assert!(
            db.get(&spec_5).is_some(),
            "Database should contain Move([5]) despite not being a factorized size"
        );
    }

    fn top_down_one<Tgt>(db: &FilesDatabase, spec: &Spec<Tgt>) -> ActionCostVec
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        top_down_many(db, std::slice::from_ref(spec))
            .into_iter()
            .next()
            .unwrap()
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
