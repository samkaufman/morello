use crate::cost::MainCost;
use crate::imp::allocs::move_cost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Tensor, View, ViewE};
use itertools::Itertools as _;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Pipeline<Tgt: Target> {
    pub stages: Vec<ImplNode<Tgt>>, // all stages are [ImplNode::SpecApp]s
    pub wirings: Vec<StageWiring<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct StageWiring<Tgt: Target> {
    pub intermediate_tensors: Vec<Rc<Tensor<Tgt>>>, // TODO: Rename to intermediate_tensors
}

impl<Tgt: Target> Impl<Tgt> for Pipeline<Tgt> {
    type BindOut = Self;

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
                                level_consumption += t.spec().memory_units();
                            }
                        }
                        level_consumption
                    })
                })
                .collect(),
        }
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        debug_assert!(!child_costs.is_empty());
        let mut cost: MainCost = 0;
        for wiring in &self.wirings {
            for t in &wiring.intermediate_tensors {
                cost = cost.saturating_add(move_cost(t.spec(), false));
            }
        }
        for child_cost in child_costs {
            cost = cost.saturating_add(*child_cost);
        }
        cost
    }

    fn map_children<F, I>(self, f: F) -> Self
    where
        F: FnOnce(Vec<ImplNode<Tgt>>) -> I,
        I: Iterator<Item = ImplNode<Tgt>>,
    {
        let old_len = self.stages.len();
        let new_stages = f(self.stages).collect::<Vec<_>>();
        assert_eq!(new_stages.len(), old_len);
        Pipeline {
            stages: new_stages,
            ..self
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        debug_assert_eq!(self.stages.len(), self.wirings.len() + 1);
        let new_wirings = self
            .wirings
            .into_iter()
            .map(|stage_wiring| StageWiring {
                intermediate_tensors: stage_wiring
                    .intermediate_tensors
                    .into_iter()
                    .map(|tensor| {
                        let ViewE::Tensor(tensor) = Rc::unwrap_or_clone(tensor).bind(get_argument)
                        else {
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
                .map(|stage| stage.clone().bind(&mut |param_idx| get_argument(param_idx)))
                .collect(),
            wirings: new_wirings,
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

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>),
    {
        for child in self.children() {
            child.visit_params(visitor);
        }
    }
}
