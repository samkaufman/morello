use crate::imp::kernels::KernelApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{
    Action, ActionT, ApplyError, NaiveBottomUpActionProvider, NaiveBottomUpSolver,
    NotApplicableReason,
};
use crate::spec::{LogicalSpec, Spec};
use crate::target::{Kernel, Target};
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};
use std::iter;

// TODO: Remove 'force' bool from Select
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Select<Tgt: Target>(pub Tgt::Kernel, pub bool);

#[derive(Default)]
pub struct SelectSolver<Tgt>(std::marker::PhantomData<Tgt>);

#[derive(Debug, Default)]
pub struct SelectActionProvider<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for Select<Tgt> {
    type BSolver = NaiveBottomUpSolver<Tgt, SelectActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let Select(k, force) = self;

        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        if !force && !k.applies_to_logical_spec(&spec.0) {
            // TODO: Use better error message-producing Error type.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Kernel does not apply to Spec",
            ))));
        }

        let arguments = operands
            .iter()
            .enumerate()
            .map(|(i, p)| ViewE::from(Param::new(i.try_into().unwrap(), p.clone())))
            .collect::<Vec<_>>();

        // Check that the kernel doesn't violate memory limits.
        match (force, k.memory_allocated(&arguments), &spec.1) {
            (true, _, _) => {}
            (false, MemoryAllocation::Inner(_) | MemoryAllocation::Pipeline { .. }, _) => {
                panic!("Kernel::memory_allocated returned non-Standard MemoryAllocation")
            }
            (false, MemoryAllocation::Simple(allocated), MemoryLimits::Standard(bounds)) => {
                for (i, (a, b)) in allocated.iter().zip(bounds.iter()).enumerate() {
                    if *a > b {
                        return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                            Tgt::levels()[i].to_string(),
                        )));
                    }
                }
            }
        };

        Ok(ImplNode::Kernel(KernelApp {
            kernel_type: *k,
            arguments,
            spec: Some(spec.clone()),
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for SelectActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        // TODO: Don't enumerate every Action to find the Action:Select.
        Tgt::actions(logical_spec)
            .filter(|a| matches!(a, Action::Select(_)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        emit_naivebottomupsolver_tests,
        target::{ArmTarget, X86Target},
    };

    emit_naivebottomupsolver_tests!(X86Target, SelectActionProvider<X86Target>, select_x86);
    emit_naivebottomupsolver_tests!(ArmTarget, SelectActionProvider<ArmTarget>, select_arm);
}
