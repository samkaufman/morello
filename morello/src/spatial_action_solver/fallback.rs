use crate::cost::{Cost, NormalizedCost};
use crate::db::{ActionCostVec, ActionNum};
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::scheduling::{Action, ActionSolver, ActionT as _, ApplyError};
use crate::search::ImplReducer;
use crate::spatial_action_solver::SpatialActionSolverT;
use crate::spatial_query::SpatialQuery;
use crate::spec::Spec;
use crate::target::Target;
use crate::utils::rect_contains_inclusive;
use std::hash::Hash;

pub struct FallbackSpatialActionSolver<'r, Tgt: Target> {
    reducer: &'r mut ImplReducer,
    candidates: Vec<ActionCandidate<Tgt>>,
}

struct ActionCandidate<Tgt: Target> {
    action_num: ActionNum,
    solver: ActionSolver<Tgt>,
    /// Sub-Specs produced by `solver`, in solver order.
    ///
    /// The order is critical because [ActionSolver::compute_cost] expects cost order to
    /// match [ActionSolver::subspecs].
    subspecs: Vec<Spec<Tgt>>,
    /// Per-child resolution state aligned by index with `subspecs`.  If `child_costs`
    /// is `None`, then this action is unsatisfiable or was already fed into the
    /// [ImplReducer].
    child_costs: Option<Vec<Option<Cost>>>,
}

impl<'r, Tgt: Target> FallbackSpatialActionSolver<'r, Tgt> {
    pub fn from_actions(
        reducer: &'r mut ImplReducer,
        goal: &Spec<Tgt>,
        actions: impl IntoIterator<Item = Action<Tgt>>,
    ) -> Self {
        let candidates = actions
            .into_iter()
            .enumerate()
            .filter_map(|(action_num, action)| match action.top_down_solver(goal) {
                Ok(solver) => Some(ActionCandidate::new(action_num.try_into().unwrap(), solver)),
                Err(ApplyError::NotApplicable(_)) => None,
                Err(ApplyError::SpecNotCanonical) => panic!(),
            })
            .collect();
        FallbackSpatialActionSolver::new(reducer, candidates)
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

impl<Tgt: Target> SpatialActionSolverT<Tgt> for FallbackSpatialActionSolver<'_, Tgt> {
    fn spatial_query<B, K>(&self, bimap: &B) -> SpatialQuery<Tgt, B, K>
    where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Eq + Hash,
    {
        SpatialQuery::from_subspecs(
            bimap,
            self.candidates.iter().flat_map(|x| {
                // TODO: unresolved_children can probably be replaced by assuming
                //       everything is unresolve
                x.unresolved_children().cloned()
            }),
        )
    }

    fn resolve<B, K>(
        &mut self,
        bimap: &B,
        table_key: &K,
        bottom: &[BimapSInt],
        top: &[BimapSInt],
        normalized_cost: Option<&NormalizedCost>,
    ) where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Eq + Hash,
    {
        // TODO: Iterating over all candidates is very inefficient.
        for candidate in &mut self.candidates {
            if let Some((action_num, cost)) =
                candidate.resolve(bimap, table_key, bottom, top, normalized_cost)
            {
                self.reducer.insert(action_num, cost);
            }
        }
    }

    fn resolve_unmemoizable_dependency(&mut self, spec: &Spec<Tgt>, result: &ActionCostVec) {
        assert!(result.len() < 2);
        let cost = result.iter().next().map(|(_, cost)| cost.clone());
        // TODO: Iterating over all candidates is very inefficient.
        for candidate in &mut self.candidates {
            if let Some((action_num, cost)) =
                candidate.resolve_unmemoizable_dependency(spec, cost.clone())
            {
                self.reducer.insert(action_num, cost);
            }
        }
    }

    fn finalize(self) {
        debug_assert!(self
            .candidates
            .iter()
            .all(|candidate| candidate.child_costs.is_none()));
    }
}

impl<'r, Tgt: Target> FallbackSpatialActionSolver<'r, Tgt> {
    fn new(reducer: &'r mut ImplReducer, candidates: Vec<ActionCandidate<Tgt>>) -> Self {
        let mut solver = FallbackSpatialActionSolver {
            reducer,
            candidates,
        };
        for candidate in &mut solver.candidates {
            if let Some((action_num, cost)) = candidate.try_complete() {
                solver.reducer.insert(action_num, cost);
            }
        }
        solver
    }
}

impl<Tgt: Target> ActionCandidate<Tgt> {
    fn new(action_num: ActionNum, solver: ActionSolver<Tgt>) -> Self {
        let subspecs = solver.subspecs().collect::<Vec<_>>();
        debug_assert!(subspecs.iter().all(|subspec| subspec.is_canonical()));
        let child_costs = Some(vec![None; subspecs.len()]);
        ActionCandidate {
            action_num,
            solver,
            subspecs,
            child_costs,
        }
    }

    fn unresolved_children(&self) -> impl Iterator<Item = &Spec<Tgt>> {
        let costs = self.child_costs.as_ref();
        self.subspecs
            .iter()
            .enumerate()
            .filter_map(move |(idx, subspec)| {
                costs
                    .is_some_and(|costs| costs[idx].is_none())
                    .then_some(subspec)
            })
    }

    fn resolve<B, K>(
        &mut self,
        bimap: &B,
        table_key: &K,
        bottom: &[BimapSInt],
        top: &[BimapSInt],
        normalized_cost: Option<&NormalizedCost>,
    ) -> Option<(ActionNum, Cost)>
    where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Eq + Hash,
    {
        let costs = self.child_costs.as_mut()?;
        for (idx, subspec) in self.subspecs.iter().enumerate() {
            debug_assert!(BiMap::defined_for(bimap, subspec));
            if costs[idx].is_some() {
                log::warn!(
                    "Candidate for action {} received multiple \
                    resolutions for child {idx}: ignoring subsequent resolution",
                    self.action_num
                );
                continue;
            }
            let (subspec_table_key, global_pt) = BiMap::apply(bimap, subspec);
            let global_pt_sint = global_pt
                .iter()
                .map(|&p| BimapSInt::from(p))
                .collect::<Vec<_>>();
            if subspec_table_key == *table_key
                && rect_contains_inclusive(top, bottom, &global_pt_sint)
            {
                let Some(normalized_cost) = normalized_cost else {
                    self.child_costs = None;
                    return None;
                };
                costs[idx] = Some(normalized_cost.clone().into_cost(subspec.0.volume()));
            }
        }
        self.try_complete()
    }

    fn resolve_unmemoizable_dependency(
        &mut self,
        spec: &Spec<Tgt>,
        cost: Option<Cost>,
    ) -> Option<(ActionNum, Cost)> {
        let costs = self.child_costs.as_mut()?;
        for (idx, subspec) in self.subspecs.iter().enumerate() {
            if costs[idx].is_some() || subspec != spec {
                continue;
            }
            let Some(cost) = &cost else {
                self.child_costs = None;
                return None;
            };
            costs[idx] = Some(cost.clone());
        }
        self.try_complete()
    }

    /// Computes this action's cost if all children are resolved, then marks it
    /// complete.  If any children are unresolved, or if the candidate was already
    /// rejected because a child had no implementation, returns `None` and leaves the
    /// candidate unchanged.
    fn try_complete(&mut self) -> Option<(ActionNum, Cost)> {
        let costs = self.child_costs.as_mut()?;
        if costs.iter().any(Option::is_none) {
            return None;
        }

        let cost = self
            .solver
            .compute_cost(costs.iter().map(|cost| cost.clone().unwrap()));
        self.child_costs = None;
        Some((self.action_num, cost))
    }
}
