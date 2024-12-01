use enum_dispatch::enum_dispatch;

use std::collections::HashMap;
use std::fmt::Debug;

use crate::tensorspec::TensorSpec;
use crate::views::{Param, View};
use crate::{
    cost::MainCost,
    imp::{
        blocks::Block, functions::FunctionApp, kernels::KernelApp, loops::Loop, moves::MoveLet,
        pipeline::Pipeline, subspecs::SpecApp,
    },
    memorylimits::{MemVec, MemoryAllocation},
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
};

pub mod blocks;
pub mod functions;
pub mod kernels;
pub mod loops;
pub mod moves;
pub mod pipeline;
pub mod subspecs;

#[enum_dispatch]
pub trait Impl<Tgt: Target> {
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

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    );

    /// If the Impl node is a [Spec] application, returns that Spec.
    fn as_spec_app(&self) -> Option<&Spec<Tgt>> {
        None
    }

    fn pprint_line(
        &self,
        names: &mut NameEnv,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String>;

    /// The [Spec] this Impl satisfies, if any and known.
    ///
    /// This is not necessarily the only Spec the Impl satisifes. Instead, it is the concrete Spec
    /// goal used during scheduling.
    fn spec(&self) -> Option<&Spec<Tgt>>;
}

pub trait ImplExt<Tgt: Target>: Impl<Tgt> {
    /// Returns the peak memory for the Impl.
    ///
    /// This traverses the Impl tree. Spec applications are treated as allocating no
    /// memory.
    fn peak_memory(&self) -> MemVec;

    /// Call the given function on all nested [Spec]s.
    ///
    /// Traversal is short-circuited if the function returns `false`.
    fn visit_subspecs(&self, f: impl FnMut(&Spec<Tgt>) -> bool) -> bool;
}

/// A non-Spec node in an Impl program tree.
///
/// These usually result from applying an [Action](crate::scheduling::Action).
///
/// Unlike [Action](crate::scheduling::Action)s, parameters may be bound to "concrete" [Tensor]s and
/// other [View]s and stored in [Rc]s (rather than an explicit environment structure).
#[derive(Debug, Clone)]
#[enum_dispatch(Impl<Tgt>)]
pub enum ImplNode<Tgt: Target> {
    Loop(Loop<Tgt>),
    MoveLet(MoveLet<Tgt>),
    Block(Block<Tgt>),
    Pipeline(Pipeline<Tgt>),
    Kernel(KernelApp<Tgt>),
    FunctionApp(FunctionApp<Tgt>),
    SpecApp(SpecApp<Tgt, Spec<Tgt>>),
}

impl<Tgt: Target, T: Impl<Tgt>> ImplExt<Tgt> for T {
    fn peak_memory(&self) -> MemVec {
        let children = self.children();
        let mut child_peaks = Vec::with_capacity(children.len());
        for child in children {
            child_peaks.push(child.peak_memory());
        }
        self.memory_allocated()
            .peak_memory_from_child_peaks::<Tgt>(&child_peaks)
    }

    fn visit_subspecs(&self, mut f: impl FnMut(&Spec<Tgt>) -> bool) -> bool {
        visit_subspecs_erased(self, &mut f)
    }
}

fn visit_subspecs_erased<Tgt, T, F>(imp: &T, f: &mut F) -> bool
where
    Tgt: Target,
    T: Impl<Tgt>,
    F: FnMut(&Spec<Tgt>) -> bool,
{
    match imp.as_spec_app() {
        Some(applied_spec) => f(applied_spec),
        None => {
            for child in imp.children() {
                if !visit_subspecs_erased(child, f) {
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
        use proptest::prelude::*;

        // TODO: Generate non-leaf Impls.
        let impl_leaf_strategy = prop_oneof![
            any::<KernelApp<Tgt>>().prop_map(ImplNode::Kernel),
            any::<SpecApp<Tgt, Spec<Tgt>>>().prop_map(ImplNode::SpecApp)
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
