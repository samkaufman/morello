use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::nameenv::NameEnv;
use crate::spec::{LogicalSpec, PrimitiveSpecType, Spec};
use crate::target::LEVEL_COUNT;
use crate::tensorspec::TensorSpec;
use crate::views::{View, ViewE};
use itertools::Itertools;
use std::borrow::Borrow;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct SpecApp<A: View>(pub Spec<A::Tgt>, pub Vec<A>);

impl<A: View> SpecApp<A> {
    pub fn new(spec: Spec<A::Tgt>, args: impl IntoIterator<Item = A>) -> Self {
        let a = args.into_iter().collect::<Vec<_>>();
        debug_assert_eq!(spec.0.operand_count(), a.len());
        Self(spec, a)
    }

    /// Create an application of a primitive [LogicalSpec].
    pub fn new_primitive_app(
        primitive_type: PrimitiveSpecType,
        args: impl IntoIterator<Item = A>,
        serial_only: bool,
        memory_limits: MemoryLimits,
    ) -> Self {
        let args_vec = args.into_iter().collect::<Vec<_>>();
        let primitive = LogicalSpec::primitive_from_parameters(
            primitive_type,
            args_vec.iter().map(|a| a.spec().clone()),
            serial_only,
        );
        debug_assert_eq!(primitive.operand_count(), args_vec.len());
        let mut spec = Spec(primitive, memory_limits);
        spec.canonicalize().unwrap();
        SpecApp::new(spec, args_vec)
    }
}

impl<A: View> Impl<A::Tgt> for SpecApp<A> {
    type BindOut = SpecApp<ViewE<A::Tgt>>;

    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<A::Tgt>> + '_> {
        Box::new(self.1.iter().map(|p| p.spec()))
    }

    fn children(&self) -> &[ImplNode<A::Tgt>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::Simple([0; LEVEL_COUNT])
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        log::warn!("Computed cost=0 for Spec");
        0
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<A::Tgt>>) -> Self {
        debug_assert_eq!(new_children.count(), 0);
        self.clone()
    }

    fn bind(self, args: &[ViewE<A::Tgt>]) -> Self::BindOut {
        debug_assert_eq!(self.0 .0.operand_count(), self.1.len());
        SpecApp(self.0, self.1.into_iter().map(|a| a.bind(args)).collect())
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        let args_str = self
            .1
            .iter()
            .map(|a| {
                a.to_param()
                    .map(|a| names.get_name_or_display(a))
                    .unwrap_or_else(|| names.get_name_or_display(a))
            })
            .join(", ");
        Some(format!("{}({})", self.0.borrow(), args_str))
    }

    fn spec(&self) -> Option<&Spec<A::Tgt>> {
        Some(self.0.borrow())
    }
}

#[cfg(test)]
impl<Tgt> proptest::arbitrary::Arbitrary for SpecApp<crate::views::Param<Tgt>>
where
    Tgt: crate::target::Target,
{
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        any::<Spec<Tgt>>()
            .prop_map(|spec| {
                let parameter_specs = spec.0.parameters();
                SpecApp(
                    spec,
                    parameter_specs
                        .into_iter()
                        .enumerate()
                        .map(|(idx, parameter_spec)| {
                            crate::views::Param::new(idx.try_into().unwrap(), parameter_spec)
                        })
                        .collect(),
                )
            })
            .boxed()
    }
}
