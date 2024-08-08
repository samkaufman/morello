use crate::imp::kernels::KernelApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{ActionT, ApplyError, BottomUpSolver, NotApplicableReason};
use crate::spec::Spec;
use crate::target::{Kernel, Target};
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};

// TODO: Remove 'force' bool from Select
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Select<Tgt: Target>(pub Tgt::Kernel, pub bool);

#[derive(Default)]
pub struct SelectSolver<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for Select<Tgt> {
    type BSolver = SelectSolver<Tgt>;

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
}

impl<Tgt: Target> BottomUpSolver for SelectSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_spec(
        &self,
        spec: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)> {
        todo!()
    }

    fn dependencies_for_range(
        &self,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)> {
        todo!()
    }

    fn visit_dependency(&self, spec: &Spec<Self::Tgt>, cost: &crate::cost::Cost) {
        todo!()
    }
}
