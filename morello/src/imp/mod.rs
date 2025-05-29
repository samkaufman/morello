use crate::{
    cost::MainCost,
    imp::{
        allocs::Alloc, blocks::Block, functions::FunctionApp, kernels::KernelApp, loops::Loop,
        pipeline::Pipeline, subspecs::SpecApp,
    },
    memorylimits::MemoryAllocation,
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
    tensorspec::TensorSpec,
    views::ViewE,
};
use enum_dispatch::enum_dispatch;

pub mod allocs;
pub mod blocks;
pub mod functions;
pub mod kernels;
pub mod loops;
pub mod pipeline;
pub mod subspecs;

#[enum_dispatch]
pub trait Impl<Tgt: Target> {
    type BindOut: Impl<Tgt>;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_>;

    fn parameter_count(&self) -> u8 {
        self.parameters().count().try_into().unwrap()
    }

    fn children(&self) -> &[ImplNode<Tgt>];

    /// The index of the child which will be scheduled by pass-through scheduling operators when
    /// multiple children exist.
    fn default_child(&self) -> Option<usize> {
        None
    }

    /// Returns the amount of memory allocated by this Impl node alone.
    ///
    /// Spec applications allocate no memory.
    fn memory_allocated(&self) -> MemoryAllocation;

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost;

    #[must_use]
    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self;

    #[must_use]
    fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut;

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String>;

    /// The [Spec] this Impl satisfies, if any and known.
    ///
    /// This is not necessarily the only Spec the Impl satisifes. Instead, it is the concrete Spec
    /// goal used during scheduling.
    fn spec(&self) -> Option<&Spec<Tgt>>;
}

crate::impl_impl_for_enum!(
    ImplNode,
    Loop => Loop<Tgt>,
    Alloc => Alloc<Tgt>,
    Block => Block<Tgt>,
    Pipeline => Pipeline<Tgt>,
    Kernel => KernelApp<ViewE<Tgt>>,
    FunctionApp => FunctionApp<Tgt>,
    SpecApp => SpecApp<ViewE<Tgt>>,
);

// TODO: Move this mod down
pub mod macros {
    #[macro_export]
    macro_rules! impl_impl_for_enum {
        ($(#[$meta:meta])* $enum_name:ident $(, $variant:ident => $type:ty) *$(,)*) => {
            $(#[$meta])*
            #[derive(Debug, Clone)]
            pub enum $enum_name<Tgt: Target> {
                $(
                    $variant($type),
                )*
            }

            impl<Tgt: Target> Impl<Tgt> for $enum_name<Tgt> {
                type BindOut = Self;

                fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
                    match self {
                        $(Self::$variant(inner) => inner.parameters(),)*
                    }
                }

                fn parameter_count(&self) -> u8 {
                    match self {
                        $(Self::$variant(inner) => inner.parameter_count(),)*
                    }
                }

                fn children(&self) -> &[ImplNode<Tgt>] {
                    match self {
                        $(Self::$variant(inner) => inner.children(),)*
                    }
                }

                fn default_child(&self) -> Option<usize> {
                    match self {
                        $(Self::$variant(inner) => inner.default_child(),)*
                    }
                }

                fn memory_allocated(&self) -> MemoryAllocation {
                    match self {
                        $(Self::$variant(inner) => inner.memory_allocated(),)*
                    }
                }

                fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
                    match self {
                        $(Self::$variant(inner) => inner.compute_main_cost(child_costs),)*
                    }
                }

                fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
                    match self {
                        $(Self::$variant(inner) => Self::$variant(inner.replace_children(new_children)),)*
                    }
                }

                fn bind(self, args: &[ViewE<Tgt>]) -> Self::BindOut {
                    match self {
                        $(Self::$variant(inner) => inner.bind(args).into(),)*
                    }
                }

                fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
                    match self {
                        $(Self::$variant(inner) => inner.pprint_line(names),)*
                    }
                }

                fn spec(&self) -> Option<&Spec<Tgt>> {
                    match self {
                        $(Self::$variant(inner) => inner.spec(),)*
                    }
                }
            }

            $(
                impl<Tgt: Target> From<$type> for $enum_name<Tgt> {
                    fn from(inner: $type) -> Self {
                        Self::$variant(inner)
                    }
                }
            )*
        }
    }
}

#[cfg(test)]
impl<Tgt> proptest::arbitrary::Arbitrary for ImplNode<Tgt>
where
    Tgt: Target,
    Tgt::Kernel: proptest::arbitrary::Arbitrary,
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<ImplNode<Tgt>>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use crate::views::Param;
        use proptest::prelude::*;

        // TODO: Generate non-leaf Impls.
        let impl_leaf_strategy = prop_oneof![
            any::<KernelApp<Param<Tgt>>>().prop_map(|s| {
                ImplNode::Kernel(KernelApp {
                    kernel_type: s.kernel_type,
                    arguments: s.arguments.into_iter().map(ViewE::Param).collect(),
                    spec: s.spec,
                })
            }),
            any::<SpecApp<Param<Tgt>>>().prop_map(|s| {
                ImplNode::SpecApp(SpecApp::new(s.0, s.1.into_iter().map(ViewE::Param)))
            })
        ];
        impl_leaf_strategy.boxed()
    }
}

/// Calls the given function on all leaves of an Impl.
///
/// The given may return `false` to short-circuit, which will be propogated to the caller of this
/// function.
pub fn visit_leaves<Tgt, F>(imp: &ImplNode<Tgt>, f: &mut F) -> bool
where
    Tgt: Target,
    F: FnMut(&ImplNode<Tgt>) -> bool,
{
    let children = imp.children();
    if children.is_empty() {
        f(imp)
    } else {
        let c = imp.children();
        for child in c {
            let should_complete = visit_leaves(child, f);
            if !should_complete {
                return false;
            }
        }
        true
    }
}
