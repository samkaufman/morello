use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::target::{CpuMemoryLevel, MemoryLevel, Target};
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
    MultAdd,
    BroadcastVecMultAdd,
    TwoVecBroadcastVecMultAdd,
    PhysicalTransposeByte128,
    PhysicalTransposeByte256,
    ValueAssign,
    VectorAssign,
    MemsetZero,
    VectorZero,
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
        match self.kernel_type {
            KernelType::BroadcastVecMultAdd | KernelType::TwoVecBroadcastVecMultAdd => {
                let vec_tensor_spec = self.arguments[1].spec();
                let vb = u64::from(vec_tensor_spec.vector_size().unwrap())
                    * u64::from(vec_tensor_spec.dtype().size());
                MemoryAllocation::Simple(Tgt::levels().map(|level| {
                    if vec_tensor_spec.level() == level {
                        vb * 2
                    } else {
                        0
                    }
                }))
            }
            KernelType::PhysicalTransposeByte256 => MemoryAllocation::Simple(Tgt::levels().map(
                |level| {
                    if level.vector_rf() {
                        64
                    } else {
                        0
                    }
                },
            )),
            _ => MemoryAllocation::none::<Tgt>(),
        }
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        match self.kernel_type {
            KernelType::BroadcastVecMultAdd | KernelType::TwoVecBroadcastVecMultAdd => {
                let vector_size = self.arguments[1].spec().vector_size().unwrap();
                let volume = self.arguments[1].spec().volume();
                debug_assert_eq!(volume % vector_size, 0);
                let vector_count = volume / vector_size;
                let mut cost = INST_COST * ((vector_count * 2) + 1);

                // TwoVecBroadcastVecMultAdd takes an input from L1.
                if matches!(self.kernel_type, KernelType::TwoVecBroadcastVecMultAdd) {
                    // TODO: Instead, call `move_cost`. Requires specializing kernel to X86/ARM.
                    let mut l1_hit_cost = CpuMemoryLevel::L1.cache_hit_cost();
                    if !self.arguments[0].spec().is_contiguous() {
                        l1_hit_cost *= 2;
                    }
                    cost += l1_hit_cost;
                }

                cost
            }
            KernelType::PhysicalTransposeByte128 => ASSIGN_INST_COST * 2,
            KernelType::PhysicalTransposeByte256 => ASSIGN_INST_COST * 4,
            KernelType::MultAdd => INST_COST,
            KernelType::ValueAssign
            | KernelType::VectorAssign
            | KernelType::MemsetZero
            | KernelType::VectorZero => ASSIGN_INST_COST,
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
            KernelType::MultAdd => "MultAdd",
            KernelType::BroadcastVecMultAdd => "BroadcastVecMultAdd",
            KernelType::TwoVecBroadcastVecMultAdd => "TwoVecBroadcastVecMultAdd",
            KernelType::PhysicalTransposeByte128 => "PhysicalTransposeByte128",
            KernelType::PhysicalTransposeByte256 => "PhysicalTransposeByte256",
            KernelType::ValueAssign => "ValueAssign",
            KernelType::VectorAssign => "VectorAssign",
            KernelType::MemsetZero => "MemsetZero",
            KernelType::VectorZero => "VectorZero",
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
            KernelType::MultAdd
            | KernelType::BroadcastVecMultAdd
            | KernelType::TwoVecBroadcastVecMultAdd => 3,
            KernelType::PhysicalTransposeByte128
            | KernelType::PhysicalTransposeByte256
            | KernelType::ValueAssign
            | KernelType::VectorAssign => 2,
            KernelType::MemsetZero | KernelType::VectorZero => 1,
        }
    }
}

#[cfg(test)]
impl<Tgt: Target, Aux: Debug + proptest::arbitrary::Arbitrary> proptest::arbitrary::Arbitrary
    for Kernel<Tgt, Aux>
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
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
