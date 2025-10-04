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
    pub parameters: Vec<ViewE<Tgt>>, // TODO: Remove?
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for FunctionApp<Tgt> {
    type BindOut = ImplNode<Tgt>;

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

    fn map_children<F, I>(self, f: F) -> Self
    where
        F: FnOnce(Vec<ImplNode<Tgt>>) -> I,
        I: Iterator<Item = ImplNode<Tgt>>,
    {
        let mut new_children = f(vec![*self.body]);
        let new_body = new_children.next().expect("FunctionApp has one child");
        assert!(
            new_children.next().is_none(),
            "FunctionApp has only one child"
        );
        Self {
            body: Box::new(new_body),
            ..self
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        let body_args = self
            .parameters
            .iter()
            .map(|p| p.bind(get_argument))
            .collect::<Vec<_>>();
        self.body
            .bind(&mut |param_idx| body_args.get(usize::from(param_idx)).cloned())
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!("(..) => {}", self.body.pprint_line(names)?))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>),
    {
        for param in &self.parameters {
            param.visit_params(visitor);
        }
        self.body.visit_params(visitor);
    }

    fn default_child(&self) -> Option<usize> {
        Some(0)
    }
}
