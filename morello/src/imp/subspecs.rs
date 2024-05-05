use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpec;
use crate::views::{Param, View};

use itertools::Itertools;
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

// TODO: Do we still want to be generic over the specific Spec?
#[derive(Debug, Clone)]
pub struct SpecApp<Tgt, P>(
    pub P,
    pub SmallVec<[Rc<dyn View<Tgt = Tgt>>; 3]>,
    PhantomData<Tgt>,
)
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone;

impl<Tgt, P> SpecApp<Tgt, P>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone,
{
    pub fn new<ParamT, I>(spec: P, args: I) -> Self
    where
        ParamT: View<Tgt = Tgt> + 'static,
        I: IntoIterator<Item = ParamT>,
    {
        let cast_args = args
            .into_iter()
            .map(|v| Rc::new(v) as _)
            .collect::<SmallVec<_>>();
        Self(spec, cast_args, PhantomData)
    }
}

impl<Tgt, P> SpecApp<Tgt, P>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone,
{
    /// Returns a [Spec] application with [Param] operands.
    pub fn default_app(spec: P) -> Self {
        let operands = spec
            .borrow()
            .0
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, o)| Rc::new(Param::new(i.try_into().unwrap(), o)) as Rc<_>)
            .collect();
        SpecApp(spec, operands, PhantomData)
    }
}

impl<Tgt, P> Impl<Tgt> for SpecApp<Tgt, P>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone + Debug,
{
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.1.iter().map(|p| p.spec()))
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::Simple([0; LEVEL_COUNT])
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        log::warn!("Computed cost=0 for Spec");
        0
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
        for a in &self.1 {
            a.bind(args, env)
        }
    }

    fn pprint_line<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
        let args_str = self
            .1
            .iter()
            .map(|a| {
                names.get_name_or_display(a.to_param().map(|p| param_bindings[p]).unwrap_or(a))
            })
            .join(", ");
        Some(format!("{}({})", self.0.borrow(), args_str))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        Some(self.0.borrow())
    }
}

#[cfg(test)]
impl<Tgt> proptest::arbitrary::Arbitrary for SpecApp<Tgt, Spec<Tgt>>
where
    Tgt: Target,
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
                            Rc::new(Param::new(idx.try_into().unwrap(), parameter_spec)) as Rc<_>
                        })
                        .collect(),
                    PhantomData,
                )
            })
            .boxed()
    }
}
