use super::{Impl, ImplNode};
use crate::{
    cost::MainCost,
    memorylimits::MemoryAllocation,
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
    tensorspec::TensorSpec,
    views::{Param, View},
};
use std::{collections::HashMap, rc::Rc};

/// A fused function and application. In short, `((..) => body)(parameters[0], .., parameters[n])`.
#[derive(Debug, Clone)]
pub struct FunctionApp<Tgt: Target> {
    pub body: Box<ImplNode<Tgt>>,
    pub parameters: Vec<Rc<dyn View<Tgt = Tgt>>>,
    pub spec: Option<Spec<Tgt>>,
}
impl<Tgt: Target> FunctionApp<Tgt> {
    /// Create a [FunctionApp] with a [Param] for each index.
    pub fn default_app(body: ImplNode<Tgt>, spec: Spec<Tgt>) -> Self {
        Self {
            body: Box::new(body),
            parameters: spec
                .0
                .parameters()
                .into_iter()
                .enumerate()
                .map(|(i, tensorspec)| Rc::new(Param::new(i.try_into().unwrap(), tensorspec)) as _)
                .collect(),
            spec: Some(spec),
        }
    }
}

impl<Tgt: Target> Impl<Tgt> for FunctionApp<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        todo!()
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        std::slice::from_ref(&self.body)
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        self.body.memory_allocated()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        self.body.compute_main_cost(child_costs)
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

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        for a in &self.parameters {
            a.bind(args, env)
        }
        let new_args = self
            .parameters
            .iter()
            .map(|p| p.as_ref())
            .collect::<Vec<_>>();
        self.body.bind(&new_args, env)
    }

    fn pprint_line(
        &self,
        names: &mut NameEnv,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
        Some(format!(
            "(..) => {}",
            self.body.pprint_line(names, param_bindings)?
        ))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }

    fn default_child(&self) -> Option<usize> {
        Some(0)
    }
}
