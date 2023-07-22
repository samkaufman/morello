use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, View};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

const INST_COST: MainCost = 1000;
const ASSIGN_INST_COST: MainCost = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel<Tgt: Target, Aux> {
    pub kernel_type: KernelType,
    pub arguments: SmallVec<[Param<Tgt>; 3]>,
    pub aux: Aux,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum KernelType {
    Mult,
    BroadcastVecMult,
    ValueAssign,
    VectorAssign,
    MemsetZero,
    VectorZero,
    CacheAccess,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Kernel<Tgt, Aux> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.arguments.iter().map(|param| param.spec()))
    }

    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none::<Tgt>()
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        match self.kernel_type {
            KernelType::Mult | KernelType::BroadcastVecMult => INST_COST,
            KernelType::ValueAssign
            | KernelType::VectorAssign
            | KernelType::MemsetZero
            | KernelType::VectorZero => ASSIGN_INST_COST,
            KernelType::CacheAccess => todo!("Add a CacheAccess cost model"),
        }
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        debug_assert_eq!(new_children.count(), 0);
        self.clone()
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        for a in &self.arguments {
            a.bind(args, env)
        }
    }

    fn line_strs<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        let name = match self.kernel_type {
            KernelType::Mult => "Mult",
            KernelType::BroadcastVecMult => "BroadcastVecMult",
            KernelType::ValueAssign => "ValueAssign",
            KernelType::VectorAssign => "VectorAssign",
            KernelType::MemsetZero => "MemsetZero",
            KernelType::VectorZero => "VectorZero",
            KernelType::CacheAccess => "CacheAccess",
        };
        let args_str = args
            .iter()
            .map(|&a| names.get_name_or_display(a))
            .join(", ");
        Some(format!("{name}({args_str})"))
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }
}
