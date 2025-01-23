use crate::imp::kernels::KernelApp;
use crate::imp::ImplNode;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::Spec;
use crate::target::{Kernel, Target};
use crate::views::{Param, ViewE};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Select<Tgt: Target>(pub Tgt::Kernel, pub bool);

impl<Tgt: Target> ActionT<Tgt> for Select<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let Select(k, force) = self;

        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        if !force && !k.applies_to_logical_spec(&spec.0) {
            // TODO: Use better error message-producing Error type.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(None)));
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
