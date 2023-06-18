use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::Target;
use crate::views::View;

use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub struct Block<Tgt: Target, Aux: Clone> {
    pub stages: Vec<ImplNode<Tgt, Aux>>,
    pub bindings: Vec<SmallVec<[u8; 3]>>,
    pub aux: Aux,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Block<Tgt, Aux> {
    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &self.stages
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none::<Tgt>()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        child_costs.iter().sum()
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }

    fn line_strs<'a>(
        &'a self,
        _names: &mut NameEnv<'a, Tgt>,
        _args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        None
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        // TODO: Flatten nested Blocks as well.
        Self {
            stages: new_children.collect(),
            bindings: self.bindings.clone(),
            aux: self.aux.clone(),
        }
    }
}
