use super::{Impl, ImplNode};
use crate::{
    cost::MainCost,
    memorylimits::MemoryAllocation,
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
    tensorspec::TensorSpec,
    views::{View, ViewE},
};

/// A fused function and application. In short, `((..) => body)(parameters[0], .., parameters[n])`.
#[derive(Debug, Clone)]
pub struct FunctionApp<Tgt: Target> {
    pub body: Box<ImplNode<Tgt>>,
    pub parameters: Vec<ViewE<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for FunctionApp<Tgt> {
    type BindOut = ImplNode<Tgt>;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.parameters.iter().map(|p| p.spec()))
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        std::slice::from_ref(&self.body)
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        assert_eq!(child_costs.len(), 1);
        child_costs[0]
    }

    fn replace_children(&self, mut new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        let new_function = FunctionApp {
            body: Box::new(new_children.next().unwrap()),
            parameters: self.parameters.clone(),
            spec: self.spec.clone(),
        };
        debug_assert!(new_children.next().is_none());
        new_function
    }

    fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut {
        let body_args = self
            .parameters
            .iter()
            .map(|p| p.bind(args))
            .collect::<Vec<_>>();
        self.body.bind(&body_args)
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!("(..) => {}", self.body.pprint_line(names)?))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }

    fn default_child(&self) -> Option<usize> {
        Some(0)
    }
}
