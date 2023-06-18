use crate::common::Problem;
use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::Target;
use crate::views::{Param, View};

use smallvec::SmallVec;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

// TODO: Do we still want to be generic over the specific Problem?
#[derive(Debug, Clone)]
pub struct ProblemApp<Tgt, P, Aux>(
    pub P,
    pub SmallVec<[Rc<dyn View<Tgt = Tgt>>; 3]>,
    pub Aux,
    PhantomData<Tgt>,
)
where
    Tgt: Target,
    P: Borrow<Problem<Tgt>> + Clone,
    Aux: Clone;

impl<Tgt, P, Aux> ProblemApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Problem<Tgt>> + Clone,
    Aux: Clone,
{
    pub fn new<ParamT: View<Tgt = Tgt> + 'static, I: IntoIterator<Item = ParamT>>(
        problem: P,
        args: I,
    ) -> Self
    where
        Aux: Default,
    {
        let cast_args: SmallVec<_> = args.into_iter().map(|v| Rc::new(v) as _).collect();
        Self(problem, cast_args, Aux::default(), PhantomData)
    }
}

impl<Tgt, P, Aux> ProblemApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Problem<Tgt>> + Clone,
    Aux: Clone,
{
    pub fn default_app(problem: P) -> Self
    where
        Aux: Default,
    {
        let operands = problem
            .borrow()
            .0
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, o)| Rc::new(Param(i.try_into().unwrap(), o)) as Rc<_>)
            .collect();
        ProblemApp(problem, operands, Aux::default(), PhantomData)
    }
}

impl<Tgt, P, Aux> Impl<Tgt, Aux> for ProblemApp<Tgt, P, Aux>
where
    Tgt: Target,
    P: Borrow<Problem<Tgt>> + Clone + Debug,
    Aux: Clone,
{
    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        &[]
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        todo!()
    }

    fn compute_main_cost(&self, _child_costs: &[MainCost]) -> MainCost {
        todo!("What cost should we have for Problem applications?")
    }

    fn aux(&self) -> &Aux {
        &self.2
    }

    fn line_strs<'a>(
        &'a self,
        _names: &mut NameEnv<'a, Tgt>,
        _args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        todo!()
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        debug_assert_eq!(new_children.count(), 0);
        self.clone()
    }
}
