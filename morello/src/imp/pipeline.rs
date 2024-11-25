use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, Tensor, View};
use itertools::Itertools as _;
use std::collections::HashMap;
use std::rc::Rc;

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
        Pipeline {
            stages: new_children.collect(),
            wirings: self.wirings.clone(),
            parameters: self.parameters.clone(),
            final_stage_output: self.final_stage_output,
            passthrough_leading_input: self.passthrough_leading_input,
            spec: self.spec.clone(),
        }
    }

    fn bind<'i, 'a: 'i, 'j: 'i>(
        &'j self,
        args: &'a [&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        debug_assert_eq!(self.stages.len(), self.wirings.len() + 1);

        // Bind the first stage
        let mut subargs: Vec<&dyn View<Tgt = Tgt>> = vec![];
        subargs.reserve_exact(
            self.stages
                .iter()
                .map(|s| s.parameter_count())
                .max()
                .unwrap()
                .into(),
        );

        let mut remaining_pipeline_inputs = args[..args.len() - 1].to_vec();

        let first_stage = self.stages.first().unwrap();
        let first_stage_parameter_count = first_stage.parameter_count();
        debug_assert!(first_stage_parameter_count > 1);
        for i in (0..first_stage_parameter_count).rev() {
            if let Some(w) = self.wirings[0]
                .tensor_wirings
                .iter()
                .find(|w| w.producing_idx == i)
            {
                subargs.push(&w.tensor);
                continue;
            }
            if i == 0 && self.passthrough_leading_input {
                subargs.push(args[0]);
                continue;
            }
            subargs.push(remaining_pipeline_inputs.pop().unwrap());
        }
        subargs.reverse();
        first_stage.bind(&subargs, env);

        // Bind the middles stages
        // for stage_idx in 1..self.stages.len() - 1 {
        //     let stage = &self.stages[stage_idx];
        //     let parameter_count = usize::from(stage.parameter_count());
        //     debug_assert!(parameter_count > 1);
        //     let nonhead_input_count = parameter_count - 2;
        //     subargs.clear();
        //     //subargs.push(self.intermediates[stage_idx - 1].as_ref());
        //     subargs.extend_from_slice(&args[(eaten - nonhead_input_count)..eaten]);
        //     //subargs.push(self.intermediates[stage_idx].as_ref());
        //     // TODO: Slot in the wirings for the previous stage in specified positions
        //     // TODO: Slot in the wirings for the next stage in output positions
        //     eaten -= nonhead_input_count;
        //     debug_assert_eq!(subargs.len(), parameter_count);
        //     stage.bind(&subargs, env);
        // }
        if self.stages.len() > 2 {
            todo!("Implement binding the middle stages");
        }

        // Bind the last
        let last_stage = self.stages.last().unwrap();
        let last_stage_parameter_count = last_stage.parameter_count();
        subargs.clear();
        for i in (0..last_stage_parameter_count).rev() {
            if let Some(w) = self.wirings[self.wirings.len() - 1]
                .tensor_wirings
                .iter()
                .find(|w| w.consuming_idx == i)
            {
                subargs.push(&w.tensor);
                continue;
            }
            if i == self.final_stage_output {
                subargs.push(args[args.len() - 1]);
                continue;
            }

            if i == 0 && self.passthrough_leading_input {
                subargs.push(args[0]);
                continue;
            }

            // TODO: Just pop().unwrap() below
            // subargs.push(remaining_pipeline_inputs.pop().unwrap());
            if let Some(x) = remaining_pipeline_inputs.pop() {
                subargs.push(x);
            } else {
                subargs.push(subargs.last().cloned().unwrap());
            }
        }
        subargs.reverse();
        last_stage.bind(&subargs, env);
    }

    fn pprint_line(
        &self,
        names: &mut NameEnv,
        _param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
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
