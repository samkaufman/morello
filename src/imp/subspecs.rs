use crate::common::Spec;
use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::target::Target;
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
pub struct SpecApp<Tgt, P, Aux>(
    pub P,
    pub SmallVec<[Rc<dyn View<Tgt = Tgt>>; 3]>,
    pub Aux,
    PhantomData<Tgt>,
)
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone,
    Aux: Clone;

impl<Tgt, P, Aux> SpecApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone,
    Aux: Clone,
{
    pub fn new<ParamT, I>(spec: P, args: I) -> Self
    where
        Aux: Default,
        ParamT: View<Tgt = Tgt> + 'static,
        I: IntoIterator<Item = ParamT>,
    {
        let cast_args = args
            .into_iter()
            .map(|v| Rc::new(v) as _)
            .collect::<SmallVec<_>>();
        Self(spec, cast_args, Aux::default(), PhantomData)
    }
}

impl<Tgt, P, Aux> SpecApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone,
    Aux: Clone,
{
    pub fn default_app(spec: P) -> Self
    where
        Aux: Default,
    {
        let operands = spec
            .borrow()
            .0
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, o)| Rc::new(Param::new(i.try_into().unwrap(), o)) as Rc<_>)
            .collect();
        SpecApp(spec, operands, Aux::default(), PhantomData)
    }
}

impl<Tgt, P, Aux> Impl<Tgt, Aux> for SpecApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Spec<Tgt>> + Clone + Debug,
    Aux: Clone,
{
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        Box::new(self.1.iter().map(|p| p.spec()))
    }

    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        todo!()
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        todo!("What cost should we have for Spec applications?")
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
        for a in &self.1 {
            a.bind(args, env)
        }
    }

    fn line_strs<'a>(
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

    fn aux(&self) -> &Aux {
        &self.2
    }
}
