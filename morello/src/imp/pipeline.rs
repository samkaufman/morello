use itertools::Itertools;

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
    // TODO: Should we compute the parameters from the children instead of storing?
    pub parameters: Vec<TensorSpec<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for Pipeline<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.parameters.iter())
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
        // TODO: This method could use some more precondition checks, esp. re: parameters.
        let new_impl = Pipeline {
            intermediates: self.intermediates.clone(),
            stages: new_children.collect(),
            parameters: self.parameters.clone(),
            spec: self.spec.clone(),
        };
        debug_assert_eq!(new_impl.intermediates.len(), self.intermediates.len());
        new_impl
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        debug_assert_eq!(self.stages.len(), self.intermediates.len() + 1);

        for intermediate in &self.intermediates {
            intermediate.bind(args, env);
        }

        // TODO: The following assumes that children (and self) put the output in the last position.

        // Bind the first stage
        let first_stage = self.stages.first().unwrap();
        let first_stage_parameter_count = usize::from(first_stage.parameter_count());
        debug_assert!(first_stage_parameter_count > 1);
        let mut subargs = vec![];
        subargs.reserve_exact(first_stage.parameter_count().into());
        let mut eaten = args.len() - first_stage_parameter_count;
        subargs.extend_from_slice(&args[eaten..args.len() - 1]);
        subargs.push(&self.intermediates[0]);
        debug_assert_eq!(subargs.len(), first_stage_parameter_count);
        first_stage.bind(&subargs, env);

        // Bind the middles stages
        for stage_idx in 1..self.stages.len() - 1 {
            let stage = &self.stages[stage_idx];
            let parameter_count = usize::from(stage.parameter_count());
            debug_assert!(parameter_count > 1);
            let nonhead_input_count = parameter_count - 2;
            subargs.clear();
            subargs.reserve_exact(parameter_count);
            subargs.push(self.intermediates[stage_idx - 1].as_ref());
            subargs.extend_from_slice(&args[(eaten - nonhead_input_count)..eaten]);
            subargs.push(self.intermediates[stage_idx].as_ref());
            eaten -= nonhead_input_count;
            debug_assert_eq!(subargs.len(), parameter_count);
            stage.bind(&subargs, env);
        }

        // Bind the last
        let last_stage = self.stages.last().unwrap();
        let last_stage_parameter_count = usize::from(last_stage.parameter_count());
        debug_assert!(last_stage_parameter_count > 1);
        subargs.clear();
        subargs.reserve_exact(last_stage_parameter_count);
        subargs.push(self.intermediates.last().unwrap().as_ref());
        subargs.extend_from_slice(&args[..eaten]);
        subargs.push(args[args.len() - 1]);
        debug_assert_eq!(subargs.len(), last_stage_parameter_count);
        last_stage.bind(&subargs, env);
    }

    fn pprint_line(
        &self,
        names: &mut NameEnv,
        _param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
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
