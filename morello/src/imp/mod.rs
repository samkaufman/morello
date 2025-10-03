use crate::{
    cost::MainCost,
    imp::{
        allocs::Alloc, blocks::Block, functions::FunctionApp, kernels::KernelApp, loops::Loop,
        pipeline::Pipeline, subspecs::SpecApp, timing::TimedRegion,
    },
    memorylimits::MemoryAllocation,
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
    tensorspec::TensorSpec,
    views::ViewE,
};
use enum_dispatch::enum_dispatch;
use std::collections::BTreeMap;

pub mod allocs;
pub mod blocks;
pub mod functions;
pub mod kernels;
pub mod loops;
pub mod pipeline;
pub mod subspecs;
pub mod timing;

#[enum_dispatch]
pub trait Impl<Tgt: Target> {
    type BindOut: Impl<Tgt>;

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

    // Replaces [Param] references within this Impl with concrete [ViewE] instances from
    // the provided getter function, recursively transforming the entire subtree to
    // eliminate parameterization.
    #[must_use]
    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut;

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String>;

    /// The [Spec] this Impl satisfies, if any and known.
    ///
    /// This is not necessarily the only Spec the Impl satisifes. Instead, it is the concrete Spec
    /// goal used during scheduling.
    fn spec(&self) -> Option<&Spec<Tgt>>;

    /// Visit parameters from this Impl using the provided visitor function.
    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>);
}

crate::impl_impl_for_enum!(
    ImplNode,
    Loop => Loop<Tgt>,
    Alloc => Alloc<Tgt>,
    Block => Block<Tgt>,
    Pipeline => Pipeline<Tgt>,
    TimedRegion => TimedRegion<Tgt>,
    Kernel => KernelApp<ViewE<Tgt>>,
    FunctionApp => FunctionApp<Tgt>,
    SpecApp => SpecApp<ViewE<Tgt>>,
);

// TODO: Make this a blanket impl for all `Impl`s
impl<Tgt: Target> ImplNode<Tgt> {
    /// Returns unbound parameters from Impl in parameter index order.
    pub fn collect_unbound_parameters(&self) -> Vec<TensorSpec<Tgt>> {
        let mut param_map = BTreeMap::new();
        self.visit_params(&mut |index, spec| {
            let previous_spec = param_map.insert(index, spec.clone());
            assert!(
                previous_spec.is_none() || previous_spec.as_ref().unwrap() == spec,
                "duplicate parameter index {index}; replacing {} with {spec}",
                previous_spec.unwrap()
            );
        });
        param_map.into_values().collect()
    }

    /// Calls the given function on all nested [ImplNode]s without children.
    ///
    /// The closure may return `false` to short-circuit, which will be propagated to the
    /// caller.
    pub fn visit_leaves<F>(&self, f: &mut F) -> bool
    where
        F: FnMut(&ImplNode<Tgt>) -> bool,
    {
        let children = self.children();
        if children.is_empty() {
            f(self)
        } else {
            for child in children {
                let should_continue = child.visit_leaves(f);
                if !should_continue {
                    return false;
                }
            }
            true
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

                fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
                    match self {
                        $(Self::$variant(inner) => inner.bind(get_argument).into(),)*
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

                fn visit_params<F>(&self, visitor: &mut F)
                where
                    F: FnMut(u8, &TensorSpec<Tgt>),
                {
                    match self {
                        $(Self::$variant(inner) => inner.visit_params(visitor),)*
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
