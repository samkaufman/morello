use std::collections::HashMap;

use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, View};

#[derive(Debug, Clone)]
pub struct Block<Tgt: Target> {
    pub stages: Vec<ImplNode<Tgt>>,
    pub parameters: Vec<TensorSpec<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
    pub default_child: Option<usize>,
}

impl<Tgt: Target> Impl<Tgt> for Block<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.parameters.iter())
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.stages
    }

    fn default_child(&self) -> Option<usize> {
        self.default_child
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        child_costs
            .iter()
            .copied()
            .reduce(|a, b| a.saturating_add(b))
            .expect("Block should be given at least one child cost")
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        // TODO: Check that parameters are unchanged after children are replaced.
        // TODO: Flatten nested Blocks as well.
        let stages = new_children.collect::<Vec<_>>();
        debug_assert_eq!(stages.len(), self.stages.len());
        Self {
            stages,
            parameters: self.parameters.clone(),
            spec: self.spec.clone(),
            default_child: self.default_child,
        }
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        for stage in &self.stages {
            stage.bind(args, env);
        }
    }

    fn pprint_line(
        &self,
        _names: &mut NameEnv,
        _param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
        // TODO: Add an option to pprint Blocks.
        None
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}
