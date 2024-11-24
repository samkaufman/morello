use itertools::Itertools;

use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::views::{Tensor, View, ViewE};

use crate::tensorspec::TensorSpec;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Pipeline<Tgt: Target> {
    pub intermediates: Vec<Rc<Tensor<Tgt>>>,
    pub stages: Vec<ImplNode<Tgt>>,
    // TODO: Should we compute the parameters from the children instead of storing?
    pub parameters: Vec<TensorSpec<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for Pipeline<Tgt> {
    type BindOut = Self;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.parameters.iter())
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.stages
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        debug_assert_eq!(self.intermediates.len(), self.stages.len() - 1);
        MemoryAllocation::Pipeline {
            intermediate_consumption: self
                .intermediates
                .iter()
                .map(|t| {
                    Tgt::levels().map(|l| {
                        if t.spec().level() == l {
                            t.spec().bytes_used()
                        } else {
                            0
                        }
                    })
                })
                .collect(),
        }
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        child_costs
            .iter()
            .copied()
            .reduce(|a, b| a.saturating_add(b))
            .expect("Pipeline should be given at least one child cost")
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        // TODO: This method could use some more precondition checks, esp. re: parameters.
        let new_impl = Pipeline {
            intermediates: self.intermediates.clone(),
            stages: new_children.collect(),
            parameters: self.parameters.clone(),
            spec: self.spec.clone(),
        };
        assert_eq!(new_impl.stages.len(), self.stages.len());
        assert_eq!(new_impl.intermediates.len(), self.intermediates.len());
        new_impl
    }

    fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut {
        debug_assert_eq!(self.stages.len(), self.intermediates.len() + 1);
        let new_intermediates = self
            .intermediates
            .iter()
            .map(|intermediate| {
                let ViewE::Tensor(tensor) = intermediate.bind(args) else {
                    unreachable!();
                };
                Rc::new(tensor)
            })
            .collect();
        Pipeline {
            intermediates: new_intermediates,
            stages: self.stages.iter().map(|stage| stage.clone().bind(args)).collect(),
            parameters: self.parameters,
            spec: self.spec,
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!(
            "pipeline ({})",
            self.intermediates
                .iter()
                .map(|i| format!("{}: {}", names.name(i), i.spec()))
                .join(", ")
        ))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}
