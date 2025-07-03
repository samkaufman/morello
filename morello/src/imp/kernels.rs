use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::{Kernel, Target};
use crate::tensorspec::TensorSpec;
use crate::views::{View, ViewE};
use itertools::Itertools;
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelApp<A: View> {
    pub kernel_type: <A::Tgt as Target>::Kernel,
    pub arguments: Vec<A>,
    pub spec: Option<Spec<A::Tgt>>,
}

impl<A: View> Impl<A::Tgt> for KernelApp<A> {
    type BindOut = KernelApp<ViewE<A::Tgt>>;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<A::Tgt>> + '_> {
        debug_assert_eq!(
            usize::from(self.kernel_type.argument_count()),
            self.arguments.len()
        );
        Box::new(self.arguments.iter().map(|param| param.spec()))
    }

    fn children(&self) -> &[ImplNode<A::Tgt>] {
        &[]
    }

    // TODO: Inline?
    fn memory_allocated(&self) -> MemoryAllocation {
        self.kernel_type.memory_allocated(&self.arguments)
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        self.kernel_type.main_cost(&self.arguments)
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<A::Tgt>>) -> Self {
        debug_assert_eq!(new_children.count(), 0);
        self.clone()
    }

    fn bind(self, args: &[ViewE<A::Tgt>]) -> Self::BindOut {
        debug_assert_eq!(
            usize::from(self.kernel_type.argument_count()),
            self.arguments.len()
        );
        KernelApp {
            arguments: self.arguments.into_iter().map(|a| a.bind(args)).collect(),
            kernel_type: self.kernel_type,
            spec: self.spec,
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        let name = self.kernel_type.name();
        let args_str = self
            .arguments
            .iter()
            .map(|a| names.get_name_or_display(a))
            .join(", ");
        Some(format!("{name}({args_str})"))
    }

    fn spec(&self) -> Option<&Spec<A::Tgt>> {
        self.spec.as_ref()
    }
}

#[cfg(test)]
impl<Tgt> proptest::arbitrary::Arbitrary for KernelApp<crate::views::Param<Tgt>>
where
    Tgt: crate::target::Target,
    Tgt::Kernel: Debug + Clone + proptest::arbitrary::Arbitrary,
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use crate::views::Param;
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
