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
    pub passthrough_leading_input: bool,
    pub final_stage_output: u8,
    pub parameters: Vec<TensorSpec<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct StageWiring<Tgt: Target> {
    pub tensor_wirings: Vec<TensorWiring<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct TensorWiring<Tgt: Target> {
    pub tensor: Rc<Tensor<Tgt>>,
    pub producing_idx: u8,
    pub consuming_idx: u8,
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
                        for w in &wiring.tensor_wirings {
                            if w.tensor.spec().level() == l {
                                level_consumption += w.tensor.spec().bytes_used();
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
            final_stage_output: self.final_stage_output,
            passthrough_leading_input: self.passthrough_leading_input,
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
                tensor_wirings: stage_wiring
                    .tensor_wirings
                    .into_iter()
                    .map(|tensor_wiring| {
                        let TensorWiring {
                            tensor,
                            producing_idx,
                            consuming_idx,
                        } = tensor_wiring;
                        let ViewE::Tensor(tensor) = Rc::unwrap_or_clone(tensor).bind(args) else {
                            unreachable!();
                        };
                        TensorWiring {
                            tensor: Rc::new(tensor),
                            producing_idx,
                            consuming_idx,
                        }
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
            final_stage_output: self.final_stage_output,
            passthrough_leading_input: self.passthrough_leading_input,
            spec: self.spec,
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!(
            "pipeline ({})",
            self.wirings
                .iter()
                .flat_map(|wiring| &wiring.tensor_wirings)
                .map(|w| format!("{}: {}", names.name(&w.tensor), w.tensor.spec()))
                .join(", ")
        ))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}
