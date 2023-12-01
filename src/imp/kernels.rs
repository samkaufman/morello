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
use std::fmt::Debug;

const INST_COST: MainCost = 100;
const ASSIGN_INST_COST: MainCost = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel<Tgt: Target, Aux> {
    pub kernel_type: KernelType,
    pub arguments: SmallVec<[Param<Tgt>; 3]>,
    pub aux: Aux,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, derive(Hash, proptest_derive::Arbitrary))]
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
        debug_assert_eq!(
            usize::from(self.kernel_type.argument_count()),
            self.arguments.len()
        );
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
        debug_assert_eq!(
            usize::from(self.kernel_type.argument_count()),
            self.arguments.len()
        );
        for a in &self.arguments {
            a.bind(args, env)
        }
    }

    fn pprint_line<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
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
        let args_str = self
            .arguments
            .iter()
            .map(|a| names.get_name_or_display(param_bindings[a]))
            .join(", ");
        Some(format!("{}({})", name, args_str))
    }

    fn aux(&self) -> &Aux {
        &self.aux
    }

    fn drop_aux(self) -> ImplNode<Tgt, ()> {
        ImplNode::Kernel(Kernel {
            kernel_type: self.kernel_type,
            arguments: self.arguments,
            aux: (),
        })
    }
}

impl KernelType {
    pub const fn argument_count(&self) -> u8 {
        match self {
            KernelType::Mult | KernelType::BroadcastVecMult => 3,
            KernelType::ValueAssign | KernelType::VectorAssign => 2,
            KernelType::MemsetZero | KernelType::VectorZero => 1,
            KernelType::CacheAccess => todo!(),
        }
    }
}

#[cfg(test)]
impl<Tgt: Target, Aux: Debug + proptest::arbitrary::Arbitrary> proptest::arbitrary::Arbitrary
    for Kernel<Tgt, Aux>
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        any::<KernelType>()
            .prop_flat_map(|kernel_type| {
                (
                    Just(kernel_type),
                    proptest::collection::vec(
                        any::<TensorSpec<Tgt>>(),
                        usize::from(kernel_type.argument_count()),
                    ),
                    any::<Aux>(),
                )
            })
            .prop_map(|(kernel_type, argument_specs, aux)| Kernel {
                kernel_type,
                arguments: argument_specs
                    .into_iter()
                    .enumerate()
                    .map(|(i, spec)| Param::new(i.try_into().unwrap(), spec))
                    .collect(),
                aux,
            })
            .boxed()
    }
}
