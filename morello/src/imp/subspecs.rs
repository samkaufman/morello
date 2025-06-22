use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::nameenv::NameEnv;
use crate::spec::{LogicalSpec, PrimitiveSpecType, Spec};
use crate::target::LEVEL_COUNT;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, View, ViewE};
use itertools::Itertools;
use std::borrow::Borrow;
use std::fmt::{self, Debug, Display};

#[derive(Debug, Clone)]
pub struct SpecApp<A: View>(pub Spec<A::Tgt>, pub Vec<A>);

impl<A: View> SpecApp<A> {
    pub fn new(spec: Spec<A::Tgt>, args: impl IntoIterator<Item = A>) -> Self {
        let a = args.into_iter().collect::<Vec<_>>();
        // Assert that the number of arguments matches. We *don't* assert that parameter and
        // argument Specs match because Moves may intentionally violate this, such as when a Move
        // canonicalizes both parameters to be row-major (i.e., erasing layouts).
        debug_assert_eq!(spec.0.operand_count(), a.len());
        Self(spec, a)
    }

    /// Create a [SpecApp] with [Param]s for each argument.
    pub fn new_with_default_params(spec: Spec<A::Tgt>) -> Self
    where
        A: From<Param<A::Tgt>>,
    {
        let arguments = spec
            .0
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, param)| A::from(Param::new(i.try_into().unwrap(), param)))
            .collect();
        Self(spec, arguments)
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

impl<A: View> Display for SpecApp<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.0)?;
        for (idx, arg) in self.1.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "_: {}", arg.spec())?;
        }
        write!(f, ")")
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
            .prop_map(SpecApp::new_with_default_params)
            .boxed()
    }
}
