use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::TensorSpec;
use crate::views::{View, ViewE};
use std::iter;

#[derive(Debug, Clone)]
pub struct Alloc<Tgt: Target> {
    pub parameter_idx: u8,
    // TODO: Needed if the body already has the new tensor?
    pub source_spec: TensorSpec<Tgt>,
    pub introduced: ViewE<Tgt>,
    pub has_prologue: bool,
    pub has_epilogue: bool,
    pub children: Vec<ImplNode<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Alloc<Tgt> {
    pub fn new(
        parameter_idx: u8,
        source_spec: TensorSpec<Tgt>,
        introduced: ViewE<Tgt>,
        prologue: Option<ImplNode<Tgt>>,
        main_stage: ImplNode<Tgt>,
        epilogue: Option<ImplNode<Tgt>>,
        spec: Option<Spec<Tgt>>,
    ) -> Self {
        let has_prologue = prologue.is_some();
        let has_epilogue = epilogue.is_some();
        let children = prologue
            .into_iter()
            .chain(iter::once(main_stage))
            .chain(epilogue)
            .collect();
        Self {
            parameter_idx,
            source_spec,
            introduced,
            has_prologue,
            has_epilogue,
            children,
            spec,
        }
    }

    pub fn prologue(&self) -> Option<&ImplNode<Tgt>> {
        if self.has_prologue {
            Some(&self.children[0])
        } else {
            None
        }
    }

    pub fn main_stage(&self) -> &ImplNode<Tgt> {
        if self.has_prologue {
            &self.children[1]
        } else {
            &self.children[0]
        }
    }

    pub fn epilogue(&self) -> Option<&ImplNode<Tgt>> {
        if self.has_epilogue {
            self.children.last()
        } else {
            None
        }
    }
}

impl<Tgt: Target> Impl<Tgt> for Alloc<Tgt> {
    type BindOut = Self;

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.children
    }

    fn default_child(&self) -> Option<usize> {
        Some(if self.has_prologue { 1 } else { 0 })
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        alloc_memory_allocation(self.introduced.spec())
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        let mut cost = move_cost(&self.source_spec);
        cost += move_cost(self.introduced.spec());
        child_costs.iter().fold(cost, |a, &b| a.saturating_add(b))
    }

    fn map_children<F, I>(self, f: F) -> Self
    where
        F: FnOnce(Vec<ImplNode<Tgt>>) -> I,
        I: Iterator<Item = ImplNode<Tgt>>,
    {
        let child_count = self.children.len();
        let new_children = f(self.children).collect::<Vec<_>>();
        assert_eq!(child_count, new_children.len());
        Self {
            children: new_children,
            ..self
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        let introduced = self.introduced.clone().bind(get_argument);
        let new_children = self
            .children
            .into_iter()
            .map(|child| child.bind(get_argument))
            .collect();
        Self {
            introduced,
            children: new_children,
            ..self
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        let cache_view_suffix = match &self.introduced {
            ViewE::Tensor(_) => String::from(""),
            ViewE::CacheView(cache_view) => {
                format!(" <- {}", names.get_name_or_display(&cache_view.source))
            }
            _ => unreachable!(),
        };
        let top = format!(
            "alloc {}: {}{}",
            names.name(&self.introduced),
            self.introduced.spec(),
            cache_view_suffix
        );
        Some(top)
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

pub(crate) fn move_cost<Tgt: Target>(accessed: &TensorSpec<Tgt>) -> MainCost {
    let hit_cost = accessed.level().cache_hit_cost();
    let mut cost = 0;
    if accessed.level().has_layout() {
        let src_cache_lines = MainCost::from(
            accessed
                .layout()
                .estimate_cache_lines::<Tgt>(accessed.shape(), accessed.dtype()),
        );
        cost = hit_cost * src_cache_lines;
    } else {
        assert_eq!(hit_cost, 0);
    }
    if !accessed.is_contiguous() {
        cost *= 2;
    }
    cost
}

pub(crate) fn alloc_memory_allocation<Tgt: Target>(
    introduced_spec: &TensorSpec<Tgt>,
) -> MemoryAllocation {
    MemoryAllocation::Simple(Tgt::levels().map(|level| {
        if introduced_spec.level() == level {
            introduced_spec.memory_units()
        } else {
            0u64
        }
    }))
}
