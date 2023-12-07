use enum_dispatch::enum_dispatch;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::target::LEVEL_COUNT;
use crate::tensorspec::TensorSpec;
use crate::utils::next_binary_power;
use crate::views::{Param, View};
use crate::{
    cost::MainCost,
    imp::{
        blocks::Block, kernels::Kernel, loops::Loop, moves::MoveLet, pipeline::Pipeline,
        subspecs::SpecApp,
    },
    memorylimits::{MemVec, MemoryAllocation},
    nameenv::NameEnv,
    spec::Spec,
    target::Target,
};

pub mod blocks;
pub mod kernels;
pub mod loops;
pub mod moves;
pub mod pipeline;
pub mod subspecs;

#[enum_dispatch]
pub trait Impl<Tgt: Target, Aux: Clone> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_>;

    fn parameter_count(&self) -> u8 {
        self.parameters().count().try_into().unwrap()
    }

    fn children(&self) -> &[ImplNode<Tgt, Aux>];

    /// Returns the amount of memory allocated by this Impl node alone.
    ///
    /// Spec applications allocate no memory.
    fn memory_allocated(&self) -> MemoryAllocation;

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost;

    #[must_use]
    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self;

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    );

    fn pprint_line<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String>;

    fn aux(&self) -> &Aux;

    fn drop_aux(self) -> ImplNode<Tgt, ()>;
}

pub trait ImplExt<Tgt: Target, Aux: Clone>: Impl<Tgt, Aux> {
    /// Returns the peak memory for the Impl.
    ///
    /// This traverses the Impl tree. Spec applications are treated as allocating no
    /// memory.
    fn peak_memory(&self) -> MemVec;

    fn peak_memory_from_child_peaks(&self, child_peaks: &[MemVec]) -> MemVec;
}

/// A non-Spec node in an Impl program tree.
///
/// These usually result from applying an [Action](crate::scheduling::Action).
///
/// Unlike [Action](crate::scheduling::Action)s, parameters may be bound to "concrete" [Tensor]s and
/// other [View]s and stored in [Rc]s (rather than an explicit environment structure).
#[derive(Debug, Clone)]
#[enum_dispatch(Impl<Tgt, Aux>)]
pub enum ImplNode<Tgt: Target, Aux: Clone> {
    Loop(Loop<Tgt, Aux>),
    MoveLet(MoveLet<Tgt, Aux>),
    Block(Block<Tgt, Aux>),
    Pipeline(Pipeline<Tgt, Aux>),
    Kernel(Kernel<Tgt, Aux>),
    SpecApp(SpecApp<Tgt, Spec<Tgt>, Aux>),
}

impl<Tgt: Target, Aux: Clone, T: Impl<Tgt, Aux>> ImplExt<Tgt, Aux> for T {
    fn peak_memory(&self) -> MemVec {
        let children = self.children();
        let mut child_peaks = SmallVec::<[MemVec; 1]>::with_capacity(children.len());
        for child in children {
            child_peaks.push(child.peak_memory());
        }
        self.peak_memory_from_child_peaks(&child_peaks)
    }

    fn peak_memory_from_child_peaks(&self, child_peaks: &[MemVec]) -> MemVec {
        let mut peak = MemVec::zero::<Tgt>();
        match self.memory_allocated() {
            MemoryAllocation::Simple(own) => {
                for child_peak in child_peaks {
                    for (i, o) in own.iter().enumerate() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(
                                peak.get_unscaled(i).max(o + child_peak.get_unscaled(i)),
                            ),
                        );
                    }
                }
            }
            MemoryAllocation::Inner(child_adds) => {
                debug_assert_eq!(child_peaks.len(), child_adds.len());
                for (child_peak, own_child_alloc) in child_peaks.iter().zip(&child_adds) {
                    for (i, o) in own_child_alloc.iter().enumerate() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(
                                peak.get_unscaled(i).max(child_peak.get_unscaled(i) + *o),
                            ),
                        )
                    }
                }
            }
            MemoryAllocation::Pipeline {
                intermediate_consumption,
            } => {
                debug_assert_eq!(child_peaks.len() + 1, intermediate_consumption.len());
                let z = [0; LEVEL_COUNT];
                let mut preceding_consumption = &z;
                let mut following_consumption = &intermediate_consumption[0];
                for (child_idx, child_peak) in child_peaks.iter().enumerate() {
                    for i in 0..peak.len() {
                        peak.set_unscaled(
                            i,
                            next_binary_power(peak.get_unscaled(i).max(
                                preceding_consumption[i]
                                    + child_peak.get_unscaled(i)
                                    + following_consumption[i],
                            )),
                        );
                    }
                    preceding_consumption = following_consumption;
                    following_consumption =
                        intermediate_consumption.get(child_idx + 1).unwrap_or(&z);
                }
            }
        }
        peak
    }
}

#[cfg(test)]
impl<Tgt, Aux> proptest::arbitrary::Arbitrary for ImplNode<Tgt, Aux>
where
    Tgt: Target,
    Aux: Debug + Clone + proptest::arbitrary::Arbitrary + 'static,
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<ImplNode<Tgt, Aux>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        // TODO: Generate non-leaf Impls.
        let impl_leaf_strategy = prop_oneof![
            any::<Kernel<Tgt, Aux>>().prop_map(ImplNode::Kernel),
            any::<SpecApp<Tgt, Spec<Tgt>, Aux>>().prop_map(ImplNode::SpecApp)
        ];
        impl_leaf_strategy.boxed()
    }
}

/// Calls the given function on all leaves of an Impl.
///
/// The given may return `false` to short-circuit, which will be propogated to the caller of this
/// function.
pub fn visit_leaves<Tgt, Aux: Clone, F>(imp: &ImplNode<Tgt, Aux>, f: &mut F) -> bool
where
    Tgt: Target,
    F: FnMut(&ImplNode<Tgt, Aux>) -> bool,
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
