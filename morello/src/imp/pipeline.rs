use std::rc::Rc;

use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Tensor, View, ViewE};
use itertools::Itertools as _;

#[derive(Debug, Clone)]
pub struct Pipeline<Tgt: Target> {
    pub stages: Vec<ImplNode<Tgt>>, // all stages are [ImplNode::SpecApp]s
    pub wirings: Vec<StageWiring<Tgt>>,
    pub parameters: Vec<TensorSpec<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct StageWiring<Tgt: Target> {
    pub intermediate_tensors: Vec<Rc<Tensor<Tgt>>>, // TODO: Rename to intermediate_tensors
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
        MemoryAllocation::Pipeline {
            intermediate_consumption: self
                .wirings
                .iter()
                .map(|wiring| {
                    Tgt::levels().map(|l| {
                        let mut level_consumption = 0;
                        for t in &wiring.intermediate_tensors {
                            if t.spec().level() == l {
                                level_consumption += t.spec().bytes_used();
                            }
                        }
                        level_consumption
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
            stages: new_children.collect(),
            wirings: self.wirings.clone(),
            parameters: self.parameters.clone(),
            spec: self.spec.clone(),
        };
        assert_eq!(new_impl.stages.len(), self.stages.len());
        new_impl
    }

    fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut {
        debug_assert_eq!(self.stages.len(), self.wirings.len() + 1);
        let new_wirings = self
            .wirings
            .into_iter()
            .map(|stage_wiring| StageWiring {
                intermediate_tensors: stage_wiring
                    .intermediate_tensors
                    .into_iter()
                    .map(|tensor| {
                        let ViewE::Tensor(tensor) = Rc::unwrap_or_clone(tensor).bind(args) else {
                            unreachable!();
                        };
                        Rc::new(tensor)
                    })
                    .collect(),
            })
            .collect();
        Pipeline {
            stages: self
                .stages
                .iter()
                .map(|stage| stage.clone().bind(args))
                .collect(),
            wirings: new_wirings,
            parameters: self.parameters,
            spec: self.spec,
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!(
            "pipeline ({})",
            self.wirings
                .iter()
                .flat_map(|wiring| &wiring.intermediate_tensors)
                .map(|t| format!("{}: {}", names.name(&t), t.spec()))
                .join(", ")
        ))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}
