use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::{Kernel, Target};
use crate::tensorspec::TensorSpec;
use crate::views::{Param, View};

use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelApp<Tgt: Target> {
    pub kernel_type: Tgt::Kernel,
    pub arguments: SmallVec<[Param<Tgt>; 3]>,
    pub spec: Option<Spec<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for KernelApp<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        debug_assert_eq!(
            usize::from(self.kernel_type.argument_count()),
            self.arguments.len()
        );
        Box::new(self.arguments.iter().map(|param| param.spec()))
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &[]
    }

    // TODO: Inline?
    fn memory_allocated(&self) -> MemoryAllocation {
        self.kernel_type.memory_allocated(&self.arguments)
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        self.kernel_type.main_cost(&self.arguments)
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
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
        let name = self.kernel_type.name();
        let args_str = self
            .arguments
            .iter()
            .map(|a| names.get_name_or_display(param_bindings[a]))
            .join(", ");
        Some(format!("{}({})", name, args_str))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }
}

#[cfg(test)]
impl<Tgt> proptest::arbitrary::Arbitrary for KernelApp<Tgt>
where
    Tgt: Target,
    Tgt::Kernel: Debug + Clone + proptest::arbitrary::Arbitrary,
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        any::<Tgt::Kernel>()
            .prop_flat_map(|kernel_type| {
                (
                    Just(kernel_type),
                    proptest::collection::vec(
                        any::<TensorSpec<Tgt>>(),
                        usize::from(kernel_type.argument_count()),
                    ),
                )
            })
            .prop_map(|(kernel_type, argument_specs)| KernelApp {
                kernel_type,
                arguments: argument_specs
                    .into_iter()
                    .enumerate()
                    .map(|(i, spec)| Param::new(i.try_into().unwrap(), spec))
                    .collect(),
                spec: None,
            })
            .boxed()
    }
}
