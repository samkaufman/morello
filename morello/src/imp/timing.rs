use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::ViewE;
use std::slice;

#[derive(Debug, Clone)]
pub struct TimedRegion<Tgt: Target> {
    counter_name: String,
    child: Box<ImplNode<Tgt>>,
    spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> TimedRegion<Tgt> {
    pub fn new(counter_name: String, child: ImplNode<Tgt>) -> Self {
        let spec = child.spec().cloned();
        Self {
            counter_name,
            child: Box::new(child),
            spec,
        }
    }

    pub fn counter_name(&self) -> &str {
        &self.counter_name
    }

    pub fn child(&self) -> &ImplNode<Tgt> {
        self.child.as_ref()
    }
}

impl<Tgt: Target> Impl<Tgt> for TimedRegion<Tgt> {
    type BindOut = Self;

    fn children(&self) -> &[ImplNode<Tgt>] {
        slice::from_ref(self.child.as_ref())
    }

    fn default_child(&self) -> Option<usize> {
        Some(0)
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        debug_assert_eq!(child_costs.len(), 1);
        child_costs[0]
    }

    fn replace_children(&self, mut new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        let new_child = new_children
            .next()
            .expect("TimedRegion requires a single child");
        debug_assert!(
            new_children.next().is_none(),
            "TimedRegion can only have a single child"
        );
        let spec = new_child.spec().cloned();
        Self {
            counter_name: self.counter_name.clone(),
            child: Box::new(new_child),
            spec,
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        Self {
            counter_name: self.counter_name,
            child: Box::new(self.child.bind(get_argument)),
            spec: self.spec,
        }
    }

    fn pprint_line(&self, _names: &mut NameEnv) -> Option<String> {
        Some(format!("timed {}", self.counter_name))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>),
    {
        self.child.visit_params(visitor);
    }
}
