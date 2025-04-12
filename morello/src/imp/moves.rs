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
pub struct MoveLet<Tgt: Target> {
    pub parameter_idx: u8,
    // TODO: Needed if the body already has the new tensor?
    pub source_spec: TensorSpec<Tgt>,
    pub introduced: ViewE<Tgt>,
    pub has_prologue: bool,
    pub has_epilogue: bool,
    pub children: Vec<ImplNode<Tgt>>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> MoveLet<Tgt> {
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

impl<Tgt: Target> Impl<Tgt> for MoveLet<Tgt> {
    type BindOut = Self;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(
            self.main_stage()
                .parameters()
                .enumerate()
                .map(|(i, body_param)| {
                    if i == usize::from(self.parameter_idx) {
                        &self.source_spec
                    } else {
                        body_param
                    }
                }),
        )
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.children
    }

    fn default_child(&self) -> Option<usize> {
        Some(if self.has_prologue { 1 } else { 0 })
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        movelet_memory_allocation(self.introduced.spec())
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        let cost = move_cost(&self.source_spec, self.introduced.spec());
        child_costs.iter().fold(cost, |a, &b| a.saturating_add(b))
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        let new_children = new_children.collect::<Vec<_>>();
        debug_assert_eq!(self.children.len(), new_children.len());
        Self {
            parameter_idx: self.parameter_idx,
            source_spec: self.source_spec.clone(),
            introduced: self.introduced.clone(),
            has_prologue: self.has_prologue,
            has_epilogue: self.has_epilogue,
            children: new_children,
            spec: self.spec.clone(),
        }
    }

    fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut {
        let introduced = self.introduced.clone().bind(args);
        let new_children = self
            .children
            .into_iter()
            .map(|child| child.bind(args))
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
}

pub(crate) fn move_cost<Tgt: Target>(src: &TensorSpec<Tgt>, dest: &TensorSpec<Tgt>) -> MainCost {
    let src_hit_cost = src.level().cache_hit_cost();
    let dest_hit_cost = dest.level().cache_hit_cost();

    let src_cache_lines = MainCost::from(src.layout().estimate_cache_lines::<Tgt>(
        src.shape(),
        src.dtype(),
        src.contiguous_abs(),
    ));
    let dest_cache_lines = MainCost::from(dest.layout().estimate_cache_lines::<Tgt>(
        dest.shape(),
        dest.dtype(),
        dest.contiguous_abs(),
    ));

    let src_cost = src_hit_cost * src_cache_lines;
    let dest_cost = dest_hit_cost * dest_cache_lines;

    let mut cost: MainCost = src_cost + dest_cost;
    if !src.is_contiguous() {
        cost *= 2;
    }
    if !dest.is_contiguous() {
        cost *= 2;
    }
    cost
}

pub(crate) fn movelet_memory_allocation<Tgt: Target>(
    introduced_spec: &TensorSpec<Tgt>,
) -> MemoryAllocation {
    let bytes_consumed = introduced_spec.bytes_used();
    MemoryAllocation::Simple(Tgt::levels().map(|level| {
        if introduced_spec.level() == level {
            bytes_consumed
        } else {
            0u64
        }
    }))
}
