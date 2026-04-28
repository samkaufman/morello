use crate::cost::{Cost, NormalizedCost};
use crate::db::{ActionCostVec, ActionNum};
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::scheduling::{Action, ActionSolver, ActionT as _, ApplyError};
use crate::search::ImplReducer;
use crate::spatial_action_solver::SpatialSolver;
use crate::spatial_query::SpatialQuery;
use crate::spec::Spec;
use crate::target::Target;
use itertools::Itertools;
use std::collections::HashMap;
use std::hash::Hash;

pub struct FallbackSpatialActionSolver<Tgt: Target> {
    candidates: Vec<ActionCandidate<Tgt>>,
    /// Pending candidate actions keyed by a sub-Spec they require.
    dependency_index: HashMap<Spec<Tgt>, Vec<DependencyHandle>>,
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

#[derive(Clone, Copy)]
struct DependencyHandle {
    candidate_idx: usize,
    child_idx: usize,
}

impl<Tgt: Target> FallbackSpatialActionSolver<Tgt> {
    pub fn from_actions(
        reducer: &mut ImplReducer,
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

impl<Tgt: Target> SpatialSolver<Tgt> for FallbackSpatialActionSolver<Tgt> {
    fn spatial_query<B, K>(&self, bimap: &B) -> SpatialQuery<Tgt, B, K>
    where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Eq + Hash,
    {
        SpatialQuery::from_subspecs(bimap, self.dependency_index.keys().cloned())
    }

    fn resolve<B, K>(
        &mut self,
        bimap: &B,
        table_key: &K,
        bottom: &[BimapSInt],
        top: &[BimapSInt],
        normalized_cost: Option<&NormalizedCost>,
        reducer: &mut ImplReducer,
    ) where
        B: BiMap<Domain = Spec<Tgt>, Codomain = (K, Vec<BimapInt>)>,
        K: Clone + Eq + Hash,
    {
        bottom
            .iter()
            .zip(top)
            .map(|(&bottom, &top)| {
                let bottom = BimapInt::try_from(bottom).unwrap();
                let top = BimapInt::try_from(top).unwrap();
                bottom..=top
            })
            .multi_cartesian_product()
            .for_each(|global_pt| {
                let spec = BiMap::apply_inverse(bimap, &(table_key.clone(), global_pt));
                // TODO: Do we need to run `contains_key`? resolve_spec is already going to access
                //       dependency_index.
                if self.dependency_index.contains_key(&spec) {
                    self.resolve_spec(
                        reducer,
                        &spec,
                        normalized_cost.map(|cost| cost.clone().into_cost(spec.0.volume())),
                    );
                }
            });
    }

    fn resolve_unmemoizable_dependency(
        &mut self,
        spec: &Spec<Tgt>,
        result: &ActionCostVec,
        reducer: &mut ImplReducer,
    ) {
        assert!(result.len() < 2);
        let cost = result.iter().next().map(|(_, cost)| cost.clone());
        self.resolve_spec(reducer, spec, cost);
    }

    fn finalize(self) {
        debug_assert!(self.dependency_index.is_empty());
        debug_assert!(self
            .candidates
            .iter()
            .all(|candidate| candidate.child_costs.is_none()));
    }
}

impl<Tgt: Target> FallbackSpatialActionSolver<Tgt> {
    fn new(reducer: &mut ImplReducer, candidates: Vec<ActionCandidate<Tgt>>) -> Self {
        let mut dependency_index = HashMap::<_, Vec<_>>::new();
        for (candidate_idx, candidate) in candidates.iter().enumerate() {
            for (child_idx, subspec) in candidate.unresolved_dependencies() {
                dependency_index
                    .entry(subspec.clone())
                    .or_default()
                    .push(DependencyHandle {
                        candidate_idx,
                        child_idx,
                    });
            }
        }

        let mut solver = FallbackSpatialActionSolver {
            candidates,
            dependency_index,
        };

        // Immediately complete any dependency-free candidates.
        for candidate in &mut solver.candidates {
            if let Some((action_num, cost)) = candidate.try_complete() {
                reducer.insert(action_num, cost);
            }
        }
        solver
    }

    /// Resolves the given dependency Spec, updating all depending candidates.
    ///
    /// This will remove the key from `dependency_index`. Any candidate for which this is the final
    /// outstanding dependency will be fed into the [ImplReducer]. If the cost is `None`, then the
    /// candidates depending on this Spec are rejected: they are not fed into the [ImplReducer], and
    /// their other dependencies are removed from `dependency_index`.
    fn resolve_spec(&mut self, reducer: &mut ImplReducer, spec: &Spec<Tgt>, cost: Option<Cost>) {
        let Some(handles) = self.dependency_index.remove(spec) else {
            return;
        };

        match cost {
            Some(cost) => {
                for handle in handles {
                    debug_assert_eq!(
                        &self.candidates[handle.candidate_idx].subspecs[handle.child_idx],
                        spec
                    );
                    if let Some((action_num, cost)) = self.candidates[handle.candidate_idx]
                        .resolve_child(handle.child_idx, cost.clone())
                    {
                        reducer.insert(action_num, cost);
                    }
                }
            }
            None => {
                for handle in handles {
                    self.reject_candidate(handle.candidate_idx);
                }
            }
        }
    }

    /// Like [ActionCandidate::reject], but also updates removes the candidate's dependencies from
    /// `dependency_index`.
    fn reject_candidate(&mut self, candidate_idx: usize) {
        for spec in self.candidates[candidate_idx].reject() {
            let Some(handles) = self.dependency_index.get_mut(&spec) else {
                return;
            };
            handles.retain(|handle| handle.candidate_idx != candidate_idx);
            if handles.is_empty() {
                self.dependency_index.remove(&spec);
            }
        }
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

    fn unresolved_dependencies(&self) -> impl Iterator<Item = (usize, &Spec<Tgt>)> {
        let costs = self.child_costs.as_ref();
        self.subspecs
            .iter()
            .enumerate()
            .filter_map(move |(idx, subspec)| {
                costs
                    .is_some_and(|costs| costs[idx].is_none())
                    .then_some((idx, subspec))
            })
    }

    fn resolve_child(&mut self, child_idx: usize, cost: Cost) -> Option<(ActionNum, Cost)> {
        let costs = self.child_costs.as_mut()?;
        assert!(
            costs[child_idx].is_none(),
            "Candidate for action {} received multiple resolutions for child {child_idx}",
            self.action_num
        );
        costs[child_idx] = Some(cost);
        self.try_complete()
    }

    /// Sets `child_costs` to `None` to indicate this candidate is unsatisfiable or already fed into
    /// the [ImplReducer], then returns the list of unresolved sub-Specs.
    fn reject(&mut self) -> Vec<Spec<Tgt>> {
        match self.child_costs.take() {
            Some(costs) => self
                .subspecs
                .iter()
                .zip(costs)
                .filter_map(|(subspec, cost)| cost.is_none().then_some(subspec.clone()))
                .collect(),
            None => vec![],
        }
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
