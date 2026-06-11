use super::{reducer::ImplReducer, RequestId};
use crate::{
    cost::Cost,
    db::{ActionCostVec, ActionNum},
    grid::{canon::CanonicalBimap, general::BiMap},
    scheduling::{ActionSolver, ActionT as _, ApplyError},
    spec::Spec,
    target::Target,
};
use std::{iter, mem::replace};

/// In-progress synthesis of a [Spec]. (Essentially a coroutine.)
#[derive(Debug)]
pub struct SpecTask<Tgt: Target>(SpecTaskState<Tgt>);

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum SpecTaskState<Tgt: Target> {
    Running {
        reducer: ImplReducer,
        partial_impls: Vec<WorkingPartialImpl<Tgt>>,
        partial_impls_incomplete: usize,
        request_batches_returned: usize,
        max_children: usize, // TODO: Combine with request_batches_returned
    },
    // TODO: Shouldn't need this second bool to track if it's from the database
    Complete {
        result: ActionCostVec,
        from_db: bool,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum WorkingPartialImpl<Tgt: Target> {
    Constructing {
        solver: ActionSolver<Tgt>,
        subspec_costs: Vec<Option<Cost>>, // empty = unsat; all Some = ready-to-complete
        producing_action_num: ActionNum,
    },
    Unsat,
    Sat,
}

impl<Tgt: Target> SpecTask<Tgt> {
    /// Creates a [SpecTask] for computing the optimal implementation of `goal`.
    ///
    /// This evaluates all actions that can be completed without child [Spec] dependencies
    /// immediately. If any applicable actions need child costs, they will be exposed through
    /// [Self::next_request_batch].
    pub fn start(goal: Spec<Tgt>) -> Self
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let mut reducer = ImplReducer::new(1, Vec::new());
        let mut max_children = 0;
        let mut partial_impls = Vec::new();
        let mut partial_impls_incomplete = 0;

        for (action_num, action) in Tgt::actions(&goal.0).enumerate() {
            match action.top_down_solver(&goal) {
                Ok(solver) => {
                    let subspec_count = solver.subspec_count();
                    max_children = max_children.max(subspec_count);

                    // If the resulting Impl is already complete, update the reducer. If there
                    // are nested sub-Specs, then store the partial Impl for resolution by the
                    // caller.
                    if subspec_count == 0 {
                        reducer.insert(
                            u16::try_from(action_num).unwrap(),
                            solver.compute_cost(iter::empty()),
                        );
                    } else {
                        partial_impls.push(WorkingPartialImpl::Constructing {
                            solver,
                            subspec_costs: vec![None; subspec_count],
                            producing_action_num: action_num.try_into().unwrap(),
                        });
                        partial_impls_incomplete += 1;
                    }
                }
                Err(ApplyError::NotApplicable(_)) => {}
                Err(ApplyError::SpecNotCanonical) => panic!(),
            };
        }

        if partial_impls_incomplete == 0 {
            Self::completed(ActionCostVec(reducer.finalize()), false)
        } else {
            Self(SpecTaskState::Running {
                reducer,
                max_children,
                partial_impls,
                partial_impls_incomplete,
                request_batches_returned: 0,
            })
        }
    }

    /// Returns an already-completed [SpecTask].
    pub fn completed(result: ActionCostVec, from_db: bool) -> Self {
        Self(SpecTaskState::Complete { result, from_db })
    }

    pub fn is_running(&self) -> bool {
        matches!(self.0, SpecTaskState::Running { .. })
    }

    /// Converts a completed task into its result. Returns `None` if is running.
    pub fn into_result(self) -> Option<(ActionCostVec, bool)> {
        match self.0 {
            SpecTaskState::Complete { result, from_db } => Some((result, from_db)),
            SpecTaskState::Running { .. } => None,
        }
    }

    /// Return an iterator over a set of [Spec]s needed to compute this task's goal.
    ///
    /// This will return `None` when all dependencies are resolved and the goal is computed.
    /// The caller should continue to call [next_request_batch] if an empty iterator is returned.
    pub fn next_request_batch(
        &mut self,
    ) -> Option<impl Iterator<Item = (Spec<Tgt>, RequestId)> + '_> {
        // TODO: Define behavior for and document returning duplicates from this function.

        let SpecTaskState::Running {
            partial_impls,
            request_batches_returned,
            max_children,
            ..
        } = &mut self.0
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
            let WorkingPartialImpl::Constructing { solver, .. } = p else {
                return None;
            };
            solver
                .subspec_at(subspec_idx)
                .map(|s| (s.clone(), (i, subspec_idx)))
        }))
    }

    pub fn resolve_request(
        &mut self,
        id: RequestId,
        cost: Option<Cost>, // `None` means that the Spec was unsat
    ) where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let SpecTaskState::Running {
            reducer,
            partial_impls,
            partial_impls_incomplete,
            request_batches_returned: _,
            max_children: _,
        } = &mut self.0
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
                subspec_costs,
                producing_action_num,
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
                            *producing_action_num,
                            // Safe to move the child costs out here because this partial Impl is
                            // about to leave `Constructing` and its `subspec_costs` will not be
                            // read again.
                            solver.compute_cost(
                                &mut subspec_costs.iter_mut().map(|c| c.take().unwrap()),
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
                self.0 = SpecTaskState::Complete {
                    result: ActionCostVec(final_result),
                    from_db: false,
                };
            }
        }
    }
}
