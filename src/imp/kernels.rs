use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::Target;
use crate::views::View;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

const INST_COST: MainCost = 1000;
const ASSIGN_INST_COST: MainCost = 10;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Kernel<Aux>(pub KernelType, pub Aux);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum KernelType {
    Mult,
    BroadcastVecMult,
    ValueAssign,
    VectorAssign,
    MemsetZero,
    VectorZero,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Kernel<Aux> {
    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none::<Tgt>()
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        match self.0 {
            KernelType::Mult | KernelType::BroadcastVecMult => INST_COST,
            KernelType::ValueAssign
            | KernelType::VectorAssign
            | KernelType::MemsetZero
            | KernelType::VectorZero => ASSIGN_INST_COST,
        }
    }

    fn line_strs<'a>(
        &'a self,
        names: &mut NameEnv<'a, Tgt>,
        args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        let name = match self.0 {
            KernelType::Mult => "Mult",
            KernelType::BroadcastVecMult => "BroadcastVecMult",
            KernelType::ValueAssign => "ValueAssign",
            KernelType::VectorAssign => "VectorAssign",
            KernelType::MemsetZero => "MemsetZero",
            KernelType::VectorZero => "VectorZero",
        };
        let args_str = args
            .iter()
            .map(|&a| names.get_name_or_display(a))
            .join(", ");
        Some(format!("{}({})", name, args_str))
    }

    fn aux(&self) -> &Aux {
        &self.1
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        debug_assert_eq!(new_children.count(), 0);
        self.clone()
    }
}
