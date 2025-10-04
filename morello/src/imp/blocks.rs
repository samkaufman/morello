use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::ViewE;

#[derive(Debug, Clone)]
pub struct Block<Tgt: Target> {
    pub stages: Vec<ImplNode<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
    pub default_child: Option<usize>,
}

impl<Tgt: Target> Impl<Tgt> for Block<Tgt> {
    type BindOut = Self;

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

    fn map_children<F, I>(self, f: F) -> Self
    where
        F: FnOnce(Vec<ImplNode<Tgt>>) -> I,
        I: Iterator<Item = ImplNode<Tgt>>,
    {
        let old_len = self.stages.len();
        let new_stages = f(self.stages).collect::<Vec<_>>();
        assert_eq!(new_stages.len(), old_len);
        Self {
            stages: new_stages,
            spec: self.spec,
            default_child: self.default_child,
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        Self {
            stages: self
                .stages
                .into_iter()
                .map(|s| s.bind(get_argument))
                .collect(),
            ..self
        }
    }

    fn pprint_line(&self, _names: &mut NameEnv) -> Option<String> {
        Some("block:".to_string())
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
