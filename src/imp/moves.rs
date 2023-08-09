use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::TensorSpec;
use crate::views::{CacheView, Param, Tensor, View};

use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct MoveLet<Tgt: Target, Aux: Clone> {
    pub parameter_idx: u8,
    // TODO: Needed if the body already has the new tensor?
    pub source_spec: TensorSpec<Tgt>,
    pub introduced: TensorOrCacheView<Param<Tgt>>,
    pub has_prologue: bool,
    pub has_epilogue: bool,
    pub children: Vec<ImplNode<Tgt, Aux>>,
    pub aux: Aux,
}

// TODO: Make private.
#[derive(Debug, Clone)]
pub enum TensorOrCacheView<V: View + 'static> {
    Tensor(Rc<Tensor<V::Tgt>>),
    CacheView(Rc<CacheView<V>>),
}

impl<Tgt: Target, Aux: Clone> MoveLet<Tgt, Aux> {
    pub fn new(
        parameter_idx: u8,
        source_spec: TensorSpec<Tgt>,
        introduced: TensorOrCacheView<Param<Tgt>>,
        prologue: Option<ImplNode<Tgt, Aux>>,
        main_stage: ImplNode<Tgt, Aux>,
        epilogue: Option<ImplNode<Tgt, Aux>>,
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
            aux,
        }
    }

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

    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &self.children
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        let introduced_spec = self.introduced.spec();
        let mut bytes_consumed = introduced_spec.bytes_used();
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
        let cost = move_cost(&self.source_spec, self.introduced.spec());
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
            aux: self.aux.clone(),
        }
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        self.introduced.inner_fat_ptr().bind(args, env);

        let param_idx = usize::from(self.parameter_idx);
        let mut moves_args = [args[param_idx], self.introduced.inner_fat_ptr()];

        if let Some(p) = self.prologue() {
            p.bind(&moves_args, env);
        }

        let mut inner_args = args.to_vec();
        inner_args[param_idx] = self.introduced.inner_fat_ptr();
        self.main_stage().bind(&inner_args, env);

        if let Some(e) = self.epilogue() {
            moves_args.swap(0, 1);
            e.bind(&moves_args, env);
        }
    }

    fn pprint_line<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
        let introduced_view = self.introduced.inner_fat_ptr();
        let cache_view_suffix = match &self.introduced {
            TensorOrCacheView::Tensor(_) => String::from(""),
            TensorOrCacheView::CacheView(cache_view) => {
                format!(
                    " <- {}",
                    names.get_name_or_display(param_bindings[&cache_view.source])
                )
            }
        };
        let top = format!(
            "alloc {}: {}{}",
            names.name(introduced_view),
            self.introduced.spec(),
            cache_view_suffix
        );
        Some(top)
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }
}

impl<V: View + 'static> TensorOrCacheView<V> {
    pub fn spec(&self) -> &TensorSpec<V::Tgt> {
        match self {
            TensorOrCacheView::Tensor(i) => i.spec(),
            TensorOrCacheView::CacheView(i) => i.spec(),
        }
    }

    pub fn inner_fat_ptr(&self) -> &(dyn View<Tgt = V::Tgt> + 'static) {
        match self {
            TensorOrCacheView::Tensor(i) => i,
            TensorOrCacheView::CacheView(i) => i,
        }
    }

    pub fn inner_rc(&self) -> Rc<dyn View<Tgt = V::Tgt>> {
        match self {
            TensorOrCacheView::Tensor(i) => Rc::clone(i) as Rc<_>,
            TensorOrCacheView::CacheView(i) => Rc::clone(i) as Rc<_>,
        }
    }
}

pub fn move_cost<Tgt: Target>(src: &TensorSpec<Tgt>, dest: &TensorSpec<Tgt>) -> MainCost {
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
    if !src.is_contiguous() || src.layout() != dest.layout() {
        cost *= 2;
    }
    cost
}
