use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::TensorSpec;
use crate::views::{Tensor, View};
use std::iter;

use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct MoveLet<Tgt: Target, Aux: Clone> {
    pub parameter_idx: u8,
    // TODO: Needed if the body already has the new tensor?
    pub source_spec: TensorSpec<Tgt>,
    pub introduced: Rc<Tensor<Tgt>>,
    pub has_prologue: bool,
    pub has_epilogue: bool,
    children: Vec<ImplNode<Tgt, Aux>>,
    pub prefetch: bool,
    pub aux: Aux,
}

impl<Tgt: Target, Aux: Clone> MoveLet<Tgt, Aux> {
    pub fn new(
        parameter_idx: u8,
        source_spec: TensorSpec<Tgt>,
        introduced: Rc<Tensor<Tgt>>,
        prologue: Option<ImplNode<Tgt, Aux>>,
        main_stage: ImplNode<Tgt, Aux>,
        epilogue: Option<ImplNode<Tgt, Aux>>,
        prefetch: bool,
        aux: Aux,
    ) -> Self {
        let has_prologue = prologue.is_some();
        let has_epilogue = epilogue.is_some();
        let children = prologue
            .into_iter()
            .chain(iter::once(main_stage))
            .chain(epilogue.into_iter())
            .collect();
        Self {
            parameter_idx,
            source_spec,
            introduced,
            has_prologue,
            has_epilogue,
            children,
            prefetch,
            aux,
        }
    }
}

impl<Tgt: Target, Aux: Clone> MoveLet<Tgt, Aux> {
    pub fn prologue(&self) -> Option<&ImplNode<Tgt, Aux>> {
        if self.has_prologue {
            Some(&self.children[0])
        } else {
            None
        }
    }

    pub fn main_stage(&self) -> &ImplNode<Tgt, Aux> {
        if self.has_prologue {
            &self.children[1]
        } else {
            &self.children[0]
        }
    }

    pub fn epilogue(&self) -> Option<&ImplNode<Tgt, Aux>> {
        if self.has_epilogue {
            self.children.last()
        } else {
            None
        }
    }
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for MoveLet<Tgt, Aux> {
    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &self.children
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        let introduced_spec = self.introduced.spec();
        let mut bytes_consumed = introduced_spec.bytes_used();
        if self.prefetch {
            bytes_consumed *= 2;
        }
        MemoryAllocation::Simple(
            Tgt::levels()
                .iter()
                .map(|level| {
                    if introduced_spec.level() == *level {
                        bytes_consumed
                    } else {
                        0u64
                    }
                })
                .collect(),
        )
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        let cost = move_cost(&self.source_spec, self.introduced.spec(), self.prefetch);
        child_costs.iter().sum::<MainCost>() + cost
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        let new_children = new_children.collect::<Vec<_>>();
        debug_assert_eq!(self.children.len(), new_children.len());
        Self {
            parameter_idx: self.parameter_idx,
            source_spec: self.source_spec.clone(),
            introduced: self.introduced.clone(),
            has_prologue: self.has_prologue,
            has_epilogue: self.has_epilogue,
            children: new_children,
            prefetch: self.prefetch,
            aux: self.aux.clone(),
        }
    }

    fn line_strs<'a>(
        &'a self,
        names: &mut NameEnv<'a, Tgt>,
        _args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        // TODO: Include parameter_idx?
        // TODO: Include source_spec?
        let prefetch_str = if self.prefetch { "[p]" } else { "" };
        let top = format!("alloc{} {}", prefetch_str, names.name(&self.introduced));
        Some(top)
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }
}

pub fn move_cost<Tgt: Target>(
    src: &TensorSpec<Tgt>,
    dest: &TensorSpec<Tgt>,
    prefetching: bool,
) -> MainCost {
    let src_hit_cost = src.level().cache_hit_cost();
    let dest_hit_cost = dest.level().cache_hit_cost();

    let src_cache_lines = MainCost::from(src.layout().estimate_cache_lines::<Tgt>(
        src.dim_sizes(),
        src.dtype(),
        src.is_contiguous(),
    ));
    let dest_cache_lines = MainCost::from(dest.layout().estimate_cache_lines::<Tgt>(
        dest.dim_sizes(),
        dest.dtype(),
        dest.is_contiguous(),
    ));

    let src_cost = 10 * (src_hit_cost * src_cache_lines);
    let dest_cost = 10 * (dest_hit_cost * dest_cache_lines);

    let mut cost: MainCost = src_cost + dest_cost;
    if prefetching {
        cost /= 2;
    }
    if !src.is_contiguous() || src.layout() != dest.layout() {
        cost *= 2;
    }
    cost
}
