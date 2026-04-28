use crate::cost::NormalizedCost;
use crate::db::{ActionCostVec, FilesDatabase, GetPreference};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::linear::BimapSInt;
use crate::search::ImplReducer;
use crate::spatial_action_solver::SpatialSolver;
use crate::spec::Spec;
use crate::target::Target;
use std::collections::HashMap;

/// Recursive top-down search state for the spatial-query algorithm.
struct SpatialTopDownSearch<'d, Tgt: Target> {
    db: &'d FilesDatabase,
    top_k: usize,
    nonmemo_results: HashMap<Spec<Tgt>, ActionCostVec>,
}

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
    assert!(db.max_k().is_none_or(|k| k >= top_k));
    if top_k > 1 {
        todo!("Support top_k > 1");
    }

    let mut search = SpatialTopDownSearch {
        db,
        top_k,
        nonmemo_results: HashMap::new(),
    };
    // TODO: Batch goals instead of solving them independently.
    goals.iter().map(|goal| search.solve(goal)).collect()
}

impl<Tgt> SpatialTopDownSearch<'_, Tgt>
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    fn solve(&mut self, spec: &Spec<Tgt>) -> ActionCostVec {
        let mut spec = spec.clone();
        spec.canonicalize().unwrap();
        self.solve_canonical(&spec)
    }

    /// Like [Self::solve], but requires that the input `spec` is already canonical.
    /// Passing a non-canonical [Spec] is a logic error.
    fn solve_canonical(&mut self, spec: &Spec<Tgt>) -> ActionCostVec {
        debug_assert!(spec.is_canonical());

        // Return result from `db` or `nonmemo_cache` if available. Otherwise update
        // `preferences` and continue.
        let mut preferences = vec![];
        let memoizable = self.db.can_memoize(spec);
        if memoizable {
            preferences = match self.db.get_with_preference_canon(spec) {
                GetPreference::Hit(result) => return result,
                GetPreference::Miss(preferences) => preferences.unwrap_or_default(),
            }
        } else if let Some(result) = self.nonmemo_results.get(spec) {
            return result.clone();
        }

        let mut reducer = ImplReducer::new(self.top_k, preferences);
        let solvers = Tgt::spatial_solvers(spec, &mut reducer);
        let bimap = self.db.spec_bimap();

        solvers.for_each(|mut solver| {
            // TODO: Lift spatial_query out to merge work across solvers
            let query = solver.spatial_query(&bimap);
            let mut missing_query = query.clone();

            self.db
                .spatial_query(&query, |table_key, bottom, top, memoized_cost| {
                    solver.resolve(&bimap, table_key, bottom, top, memoized_cost.as_ref());
                    missing_query.subtract_rect(table_key, bottom, top);
                });

            missing_query.memoizable_specs(&bimap).for_each(|subspec| {
                assert!(subspec.is_canonical());
                let subspec_result = self.solve_canonical(&subspec);
                let (table_key, global_pt) = BiMap::apply(&bimap, &subspec);
                let global_pt = global_pt
                    .iter()
                    .map(|&coord| BimapSInt::from(coord))
                    .collect::<Vec<_>>();
                assert!(subspec_result.len() < 2);
                let normalized_cost = subspec_result
                    .first()
                    .map(|(_action, cost)| NormalizedCost::new(cost.clone(), subspec.0.volume()));
                solver.resolve(
                    &bimap,
                    &table_key,
                    &global_pt,
                    &global_pt,
                    normalized_cost.as_ref(),
                );
            });

            for subspec in query.unmemoizable_specs() {
                assert!(subspec.is_canonical());
                let subspec_result = self.solve_canonical(subspec);
                solver.resolve_unmemoizable_dependency(subspec, &subspec_result);
            }

            solver.finalize();
        });

        let result = ActionCostVec(reducer.finalize());
        if memoizable {
            self.db.put_canon(spec, result.0.clone());
        } else {
            self.nonmemo_results.insert(spec.clone(), result.clone());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::FilesDatabase;
    use crate::layout::row_major;
    use crate::lspec;
    use crate::memorylimits::{MemVec, MemoryLimits};
    use crate::search;
    use crate::target::{
        Avx2Target,
        CpuMemory::{GL, RF},
    };
    use std::slice;

    #[test]
    fn test_spatial_search_database_matches_top_down_for_small_move() {
        let spec = Spec::<Avx2Target>(
            lspec!(Move([2], (u8, GL, row_major), (u8, RF, row_major), serial)),
            MemoryLimits::Standard(MemVec::new([1, 1, 1, 0])),
        );
        let reference_db = FilesDatabase::new::<Avx2Target>(None, false, 1, 128, 1);
        let spatial_db = FilesDatabase::new::<Avx2Target>(None, false, 1, 128, 1);

        let reference = search::top_down_many(&reference_db, slice::from_ref(&spec), 1);
        let spatial = top_down_many(&spatial_db, slice::from_ref(&spec), 1);

        assert_eq!(spatial, reference);
        reference_db.assert_same_memoized_points_and_throughputs(&spatial_db);
    }
}
