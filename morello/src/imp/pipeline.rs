use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::views::{Param, Tensor, View};
use std::collections::HashMap;

use crate::tensorspec::TensorSpec;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Pipeline<Tgt: Target> {
    pub intermediates: Vec<Rc<Tensor<Tgt>>>,
    pub stages: Vec<ImplNode<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for Pipeline<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        todo!()
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.stages
    }

    fn memory_allocated(&self) -> MemoryAllocation {
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
        Pipeline {
            intermediates: self.intermediates.clone(),
            stages: new_children.collect(),
            spec: self.spec.clone(),
        }
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        _args: &[&'j dyn View<Tgt = Tgt>],
        _env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        todo!("Implement bind for Pipeline");
    }

    fn pprint_line<'a>(
        &'a self,
        _names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        _param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
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

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}
