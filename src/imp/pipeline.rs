use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::Target;
use crate::views::{Tensor, View};

use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Pipeline<Tgt: Target, Aux: Clone> {
    pub intermediates: Vec<Rc<Tensor<Tgt>>>,
    pub stages: Vec<ImplNode<Tgt, Aux>>,
    pub aux: Aux,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Pipeline<Tgt, Aux> {
    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &self.stages
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::Pipeline {
            intermediate_consumption: self
                .intermediates
                .iter()
                .map(|t| {
                    Tgt::levels()
                        .iter()
                        .map(|l| {
                            if &t.0.level() == l {
                                t.0.bytes_used()
                            } else {
                                0
                            }
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        child_costs.iter().sum()
    }

    fn line_strs<'a>(
        &'a self,
        _names: &mut NameEnv<'a, Tgt>,
        _args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        let intermeds = self
            .intermediates
            .iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<_>>()
            .join(", ");
        let main = format!("pipeline: ({:?})", intermeds);
        Some(main)
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        Pipeline {
            intermediates: self.intermediates.clone(),
            stages: new_children.collect(),
            aux: self.aux.clone(),
        }
    }
}
