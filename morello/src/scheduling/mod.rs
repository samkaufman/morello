use crate::alignment::aligned_approx;
use crate::common::{DimSize, Dtype};
use crate::cost::{Cost, MainCost, NormalizedCost};
use crate::db::{ActionNum, DbKey};
use crate::grid::general::BiMap;
use crate::imp::loops::compute_loop_main_cost;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::search::ImplReducer;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::utils::{snap_memvec_up, spec_diagonals_flat_shifted};
use crate::views::{Param, Tensor, TileError, View, ViewE};
use auto_impl::auto_impl;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::iter;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};

pub mod broadcast_first;
pub mod bufferize;
pub mod moves;
pub mod select;
pub mod spatial_split;
pub mod tiling;
pub mod to_accum;
pub mod to_max_and_denom;
pub mod to_max_and_unscaled;
pub mod to_softmax_parts;

pub trait ActionT<Tgt: Target> {
    type BSolver: BottomUpSolver<Tgt = Tgt>;
    type BSolverIter: Iterator<Item = Self::BSolver>;

    fn apply(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        if !spec.is_canonical() {
            return Err(ApplyError::SpecNotCanonical);
        }
        self.apply_unchecked_canon(spec)
    }

    /// Like [Action::apply], but does not check if the Spec is canonical. Passing a non-canonical
    /// Spec is a logic error.
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError>;

    /// Returns a value which produces sub-Spec requests and compute a [Cost].
    ///
    /// This is functionally equivalent to calling [Action::apply] to produce a partial Impl and
    /// then gathering its sub-Specs and computing a cost, but is usually faster.
    ///
    /// The caller must ensure that `spec` is in canonical form. Passing a non-canonical form is a
    /// logic error.
    fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionTopDownSolver<Tgt>, ApplyError> {
        self.apply_unchecked_canon(spec)
            .map(|applied| ActionTopDownSolver::Fallback(applied))
    }

    fn bottom_up_solvers() -> Self::BSolverIter;
}

/// Provides (some of) a [Spec]'s action dependencies and computes the cost of one or more of that
/// [Spec]'s actions from those dependencies.
pub trait BottomUpSolver {
    type Tgt: Target;

    fn dependencies_for_spec(
        &mut self,
        spec: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>;

    // TODO: Document specifically what's in the [low, high] set.
    fn dependencies_for_range<B>(
        &mut self,
        bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>;

    fn visit_dependency<U>(
        &mut self,
        spec: &Spec<Self::Tgt>,
        cost: &[NormalizedCost],
        updater: &mut U,
    ) where
        U: VisitUpdater<Self::Tgt>;

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>;
}

/// A trait for putting final Spec-action costs. [BottomUpSolver::visit_dependency] is expected to
/// call [VisitUpdater::complete] for satisfied solutions. (No call for unsat.)
#[auto_impl(&mut)]
pub trait VisitUpdater<Tgt: Target> {
    /// Store the cost-action for a particular [Spec]-action.
    fn complete_action(
        &mut self,
        spec: &Spec<Tgt>,
        action: ActionNum,
        normalized_cost: NormalizedCost,
    );

    /// Finalize a Spec after [VisitUpdater::complete_action] called for all possible decisions.
    fn complete_spec(&mut self, spec: &Spec<Tgt>);
}

/// An actions provider for [NaiveBottomUpSolver].
pub trait NaiveBottomUpActionProvider<Tgt: Target> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>>;

    // TODO: Remove
    fn debugging() -> Option<String> {
        None
    }
}

pub trait ActionEncodeDecode<Tgt: Target> {
    fn encode(&self, spec: &Spec<Tgt>) -> ActionNum;
    fn decode(spec: &Spec<Tgt>, encoding: ActionNum) -> Self;
}

macro_rules! action_dispatch {
    ( $(#[$meta:meta])* $name:ident, $solver:ident, $(($variant:tt, $innertype:ty)),*$(,)* ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
        #[serde(bound(
            deserialize = "Tgt::Kernel: Deserialize<'de>",
            serialize = "Tgt::Kernel: Serialize"
        ))]
        pub enum $name<Tgt: Target> {
            $( $variant($innertype) ),*
        }

        #[allow(non_snake_case)]
        pub enum $solver<Tgt: Target> {
            $( $variant(<$innertype as ActionT<Tgt>>::BSolver) ),*
        }

        impl<Tgt: Target> ActionT<Tgt> for $name<Tgt> {
            type BSolver = $solver<Tgt>;
            type BSolverIter = Box<dyn Iterator<Item = $solver<Tgt>> + 'static>;

            fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
                match self {
                    $( Self::$variant(a) => a.apply_unchecked_canon(spec) ),*
                }
            }

            fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionTopDownSolver<Tgt>, ApplyError> {
                match self {
                    $( Self::$variant(a) => a.top_down_solver(spec) ),*
                }
            }

            fn bottom_up_solvers() -> Self::BSolverIter {
                let it = std::iter::empty();
                $(
                    let sub = <$innertype as ActionT<Tgt>>::bottom_up_solvers();
                    let it = it.chain(sub.map(|i| $solver::<Tgt>::$variant(i)));
                )*
                Box::new(it)
            }
        }

        impl<Tgt: Target> BottomUpSolver for $solver<Tgt> {
            type Tgt = Tgt;

            fn dependencies_for_spec(
                &mut self,
                spec: &Spec<Tgt>,
            ) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
                match self {
                    $( Self::$variant(a) => a.dependencies_for_spec(spec) ),*
                }
            }

            fn dependencies_for_range<B>(
                &mut self,
                bimap: &B,
                low: &Spec<Self::Tgt>,
                high: &Spec<Self::Tgt>,
            ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
            where
                B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
            {
                match self {
                    $( Self::$variant(a) => a.dependencies_for_range(bimap, low, high) ),*
                }
            }

            fn visit_dependency<U>(
                &mut self,
                spec: &Spec<Tgt>,
                cost: &[NormalizedCost],
                updater: &mut U,
            )
            where
                U: VisitUpdater<Self::Tgt>
            {
                match self {
                    $( Self::$variant(a) => {
                        a.visit_dependency(spec, cost, updater)
                    } ),*
                }
            }

            fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
            where
                U: VisitUpdater<Self::Tgt>
            {
                match self {
                    $( Self::$variant(a) => a.apply_no_dependency_updates(spec, updater) ),*
                }
            }
        }

        $(
            impl<Tgt: Target> From<$innertype> for $name<Tgt> {
                fn from(value: $innertype) -> Self {
                    Self::$variant(value)
                }
            }
        )?
    };
}

action_dispatch! {
    /// A scheduling decision which can be applied to a Spec to produce an Impl.
    ///
    /// [Action]s contain the minimal amount of information needed to distinguish a one scheduling
    /// decision from another for a given Spec.
    Action,
    ActionBottomUpSolver,
    (TileOut, tiling::TileOut),
    (Split, tiling::Split),
    (Move, moves::Move::<Tgt>),
    (ToAccum, to_accum::ToAccum),
    (BroadcastFirst, broadcast_first::BroadcastFirst<Tgt>),
    (ToSoftmaxParts, to_softmax_parts::ToSoftmaxParts::<Tgt>),
    (ToSoftmaxPartsRecompute, to_softmax_parts::ToSoftmaxPartsRecompute::<Tgt>),
    (ToMaxAndDenominator, to_max_and_denom::ToMaxAndDenominator),
    (ToMaxAndUnscaled, to_max_and_unscaled::ToMaxAndUnscaled<Tgt>),
    (Bufferize, bufferize::Bufferize<Tgt>),
    (SpatialSplit, spatial_split::SpatialSplit),
    (Select, select::Select::<Tgt>),
}

#[derive(Debug)]
pub enum ActionTopDownSolver<Tgt: Target> {
    PrimitiveTileOut {
        outer_spec: Spec<Tgt>,
        body_spec: Spec<Tgt>,
    },
    Move {
        prologue: Option<Spec<Tgt>>,
        body: Spec<Tgt>,
        epilogue: Option<Spec<Tgt>>,
        base_main_cost: MainCost,
        allocation: MemoryAllocation,
    },
    Fallback(ImplNode<Tgt>),
}

#[derive(Debug, Default)]
pub struct NaiveBottomUpSolver<Tgt: Target, P> {
    requests_map: HashMap<Spec<Tgt>, Vec<(Rc<RefCell<NaiveWorkingImpl<Tgt>>>, usize)>>,
    _phantom: PhantomData<P>,
}

#[derive(Debug)]
struct NaiveWorkingImpl<Tgt: Target> {
    working_spec: Rc<RefCell<NaiveWorkingSpec<Tgt>>>,
    solver: ActionTopDownSolver<Tgt>,
    action_num: ActionNum, // TODO: Needed?
    subspec_costs: Vec<Option<Option<Cost>>>,
}

#[derive(Debug)]
pub struct NaiveWorkingSpec<Tgt: Target> {
    spec: Spec<Tgt>, // TODO: Needed? Basically just for debugging.
    impls: Vec<Weak<RefCell<NaiveWorkingImpl<Tgt>>>>,
    incomplete_impls: usize,
}

#[derive(thiserror::Error, Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum ApplyError {
    #[error("Cannot apply action to non-canonical Spec")]
    SpecNotCanonical,
    #[error("Action does not apply to this Spec: {0}")]
    NotApplicable(NotApplicableReason),
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum NotApplicableReason {
    OutOfMemory(String),
    TileShapeMatchesOriginal,
    TileShapeIsLarger,
    TileShapeInvalid,
    ParallelPrevented,
    LayoutIncompatible,
    SelfMove,
    VectorSizeInvalid(Dtype, DimSize),
    MultipleOutputs,
    Other(Option<&'static str>),
}

impl<Tgt, K> VisitUpdater<Tgt> for HashMap<K, ImplReducer>
where
    Tgt: Target,
    K: Borrow<Spec<Tgt>> + Eq + Hash,
{
    fn complete_action(
        &mut self,
        spec: &Spec<Tgt>,
        action: ActionNum,
        normalized_cost: NormalizedCost,
    ) {
        let reducer = self.get_mut(spec).unwrap();
        let cost = normalized_cost.into_main_cost_for_volume(spec.0.volume());
        reducer.insert(action, cost);
    }

    fn complete_spec(&mut self, _spec: &Spec<Tgt>) {}
}

// TODO: Encode without iterating actions.
impl<Tgt: Target> ActionEncodeDecode<Tgt> for Action<Tgt> {
    fn encode(&self, spec: &Spec<Tgt>) -> ActionNum {
        Tgt::actions(&spec.0)
            .position(|a| a == *self)
            .unwrap()
            .try_into()
            .unwrap()
    }

    fn decode(spec: &Spec<Tgt>, encoding: ActionNum) -> Self {
        Tgt::actions(&spec.0)
            .nth(encoding as usize)
            .unwrap()
            .clone()
    }
}

impl<Tgt: Target> ActionTopDownSolver<Tgt> {
    pub fn subspecs(&self) -> impl Iterator<Item = Spec<Tgt>> {
        match self {
            ActionTopDownSolver::PrimitiveTileOut {
                outer_spec: _,
                body_spec,
            } => {
                // TODO: Avoid this clone
                vec![body_spec.clone()].into_iter()
            }
            ActionTopDownSolver::Move {
                prologue,
                body,
                epilogue,
                base_main_cost: _,
                allocation: _,
            } => {
                // TODO: Avoid these clones. Return an iterator of references.
                let mut v: Vec<Spec<Tgt>> = Vec::with_capacity(3);
                v.extend(prologue.clone());
                v.push(body.clone());
                v.extend(epilogue.clone());
                v.into_iter()
            }
            ActionTopDownSolver::Fallback(partial_impl) => {
                let mut partial_impl_subspecs = Vec::new();
                collect_nested_specs(partial_impl, &mut partial_impl_subspecs);
                partial_impl_subspecs.into_iter()
            }
        }
    }

    pub fn compute_cost<I>(&self, mut child_costs: I) -> Cost
    where
        I: Iterator<Item = Cost>,
    {
        match self {
            ActionTopDownSolver::PrimitiveTileOut {
                outer_spec,
                body_spec,
            } => {
                let parallel = !outer_spec.0.serial_only() && body_spec.0.serial_only();
                match &outer_spec.0 {
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ:
                                PrimitiveSpecType::Matmul { .. }
                                | PrimitiveSpecType::Move
                                | PrimitiveSpecType::Fill { .. },
                            spec_shape,
                            ..
                        },
                        ..,
                    ) => {
                        let LogicalSpec::Primitive(
                            PrimitiveBasics {
                                spec_shape: body_shape,
                                ..
                            },
                            ..,
                        ) = &body_spec.0
                        else {
                            unreachable!();
                        };

                        let mut steps = 1;
                        let mut full_steps = 1;
                        for (o, t) in spec_shape.iter().zip(body_shape) {
                            steps *= o.get().div_ceil(t.get());
                            full_steps *= o.get() / t.get();
                        }
                        let mut cost = child_costs.next().unwrap();
                        cost.main =
                            compute_loop_main_cost::<Tgt>(steps, full_steps, parallel, cost.main);
                        cost.depth += 1;
                        cost
                    }
                    _ => unreachable!(),
                }
            }
            ActionTopDownSolver::Move {
                prologue: _,
                body: _,
                epilogue: _,
                base_main_cost,
                allocation,
            } => {
                let mut main = *base_main_cost;
                let mut child_peaks = vec![];
                let mut depth = 0;
                for child_cost in child_costs {
                    main = main.saturating_add(child_cost.main);
                    depth = depth.max(child_cost.depth);
                    child_peaks.push(child_cost.peaks);
                }
                depth += 1;
                // TODO: Is snap_memvec_up really needed or can we bake this into MemVec?
                let peaks = snap_memvec_up(
                    allocation.peak_memory_from_child_peaks::<Tgt>(&child_peaks),
                    false,
                );
                Cost { main, peaks, depth }
            }
            ActionTopDownSolver::Fallback(partial_impl) => {
                compute_impl_cost(partial_impl, &mut child_costs)
            }
        }
    }

    fn tiled_subspec_fast<B>(
        binds: B,
        original_spec: &Spec<Tgt>,
        tile_shape: &[DimSize],
        parallel: bool,
    ) -> Result<Spec<Tgt>, ApplyError>
    where
        B: Iterator<Item = (usize, usize)> + ExactSizeIterator,
    {
        let mut new_spec = original_spec.clone();
        match &mut new_spec.0 {
            LogicalSpec::Primitive(PrimitiveBasics { spec_shape, .. }, _, serial_only) => {
                for (o, t) in binds {
                    spec_shape[o] = tile_shape[t];
                }
                *serial_only = *serial_only || parallel;
            }
            _ => unreachable!(),
        }

        let outer_parameters = original_spec.0.parameters();
        let new_parameters = new_spec.0.parameters();
        let LogicalSpec::Primitive(_, new_auxes, _) = &mut new_spec.0 else {
            unreachable!();
        };

        // TODO: Should the following be optimized with `tile_shape_is_valid`?
        for ((outer, inner), new_aux) in outer_parameters
            .into_iter()
            .zip(new_parameters)
            .zip(new_auxes)
        {
            if outer.shape() == inner.shape() {
                continue;
            }

            // TODO: Need is_valid_tile_shape if we're calling update_for_tiling?
            if !outer.is_valid_tile_shape(inner.shape(), parallel) {
                return Err(ApplyError::NotApplicable(
                    NotApplicableReason::TileShapeInvalid,
                ));
            }
            let Ok((new_layout, new_contig)) = outer.layout().update_for_tiling(
                outer.shape(),
                inner.shape(),
                outer.contiguous_abs(),
            ) else {
                todo!();
            };
            new_aux.aligned = aligned_approx(inner.shape(), inner.shape(), &outer).unwrap();
            new_aux.layout = new_layout;
            new_aux.contig = new_contig;
        }

        if new_spec.canonicalize().is_err() {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeInvalid,
            ));
        }

        Ok(new_spec)
    }
}

impl Display for NotApplicableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotApplicableReason::OutOfMemory(lvl) => {
                write!(f, "Insufficient memory in {lvl}")
            }
            NotApplicableReason::TileShapeMatchesOriginal => {
                write!(f, "Tile shape matches original")
            }
            NotApplicableReason::TileShapeIsLarger => {
                write!(f, "Tile shape is larger than original")
            }
            NotApplicableReason::TileShapeInvalid => {
                write!(f, "Invalid tile shape")
            }
            NotApplicableReason::ParallelPrevented => {
                write!(f, "Cannot implement serial-only Spec with parallel tile")
            }
            NotApplicableReason::LayoutIncompatible => {
                write!(f, "Layout does not apply to tile size")
            }
            NotApplicableReason::SelfMove => {
                write!(
                    f,
                    "Source and destination TensorSpecs were equal after canonicalization"
                )
            }
            NotApplicableReason::VectorSizeInvalid(dtype, size) => {
                write!(
                    f,
                    "Target does not support {dtype} vectors with {size} values"
                )
            }
            NotApplicableReason::MultipleOutputs => {
                write!(f, "Spec has multiple outputs")
            }
            NotApplicableReason::Other(Some(reason_string)) => write!(f, "{}", reason_string),
            NotApplicableReason::Other(None) => write!(f, "Unknown reason"),
        }
    }
}

/// Returns applications of Zeroes corresponding to the outputs of the given Spec.
fn make_accum_inits_for_spec<Tgt: Target>(spec: &Spec<Tgt>) -> Vec<ImplNode<Tgt>> {
    (0..u8::try_from(spec.0.operand_count()).unwrap())
        .flat_map(|parameter_idx| {
            if !spec.0.parameter_is_output(parameter_idx.into()) {
                return None;
            }
            let accum_initial_value = spec
                .0
                .initial_accumulating_value_for_output(parameter_idx.into())?;
            let output = spec.0.parameter(parameter_idx.into());
            Some(ImplNode::from(SpecApp::new_primitive_app(
                PrimitiveSpecType::Fill {
                    value: accum_initial_value,
                },
                [ViewE::from(Param::new(parameter_idx, output))],
                spec.0.serial_only(),
                spec.1.clone(),
            )))
        })
        .collect()
}

/// Return an error if the tile shape is invalid for a given tensor.
///
/// This can return TileShapeMatchesOriginal, TileShapeIsLarger, TileShapeInvalid, or
/// Other.
fn check_tile_out_applies<Tgt: Target>(
    current_out_shape: &[DimSize],
    output_shape: &[DimSize],
    current_output: &TensorSpec<Tgt>,
    parallel: bool,
) -> Result<(), ApplyError> {
    if current_out_shape == output_shape {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeMatchesOriginal,
        ));
    }
    if output_shape
        .iter()
        .zip(current_out_shape)
        .any(|(out, cur)| out > cur)
    {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeIsLarger,
        ));
    }

    // Abort if it's invalid to tile the original output tensor
    // to the new shape (e.g., due to layout constraints).
    if !current_output.is_valid_tile_shape(output_shape, parallel) {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeInvalid,
        ));
    }

    if output_shape
        .iter()
        .enumerate()
        .any(|(dim, out_size)| current_out_shape[dim].get() % out_size.get() != 0)
    {
        return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
            "Original size is not a multiple of tile size",
        ))));
    }

    Ok(())
}

impl<Tgt, P> BottomUpSolver for NaiveBottomUpSolver<Tgt, P>
where
    Tgt: Target,
    P: NaiveBottomUpActionProvider<Tgt>,
{
    type Tgt = Tgt;

    fn dependencies_for_spec(
        &mut self,
        spec: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)> {
        // TODO: Test that this never returns and correctly visits duplicate sub-Specs.
        // TODO: Test that this can be called twice without messing up our internal state.

        let working_spec = Rc::new(RefCell::new(NaiveWorkingSpec {
            spec: spec.clone(),
            impls: vec![],
            incomplete_impls: 0,
        }));

        let mut out = vec![];
        for action in P::actions(&spec.0) {
            match action.top_down_solver(spec) {
                Ok(solver) => {
                    let subspecs = solver.subspecs().collect::<Vec<_>>();
                    if subspecs.is_empty() {
                        continue;
                    }
                    let working_impl = Rc::new(RefCell::new(NaiveWorkingImpl {
                        working_spec: Rc::clone(&working_spec),
                        solver,
                        action_num: action.encode(spec),
                        subspec_costs: vec![None; subspecs.len()],
                    }));
                    let mut working_spec_mut = RefCell::borrow_mut(&working_spec);
                    working_spec_mut.impls.push(Rc::downgrade(&working_impl));
                    if !subspecs.is_empty() {
                        working_spec_mut.incomplete_impls += 1;
                    }

                    for (subspec_idx, subspec) in subspecs.into_iter().enumerate() {
                        out.push((subspec.clone(), subspec.clone()));
                        self.requests_map
                            .entry(subspec)
                            .or_default()
                            .push((Rc::clone(&working_impl), subspec_idx));
                    }
                }
                Err(ApplyError::NotApplicable(_)) => {}
                Err(ApplyError::SpecNotCanonical) => panic!("given non-canon Spec: {spec}"),
            }
        }
        out
    }

    fn dependencies_for_range<B>(
        &mut self,
        bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
    {
        // TODO: Does this cover the right set of Specs? Mapping isn't defined!
        let low_projection = BiMap::apply(bimap, low);
        let high_projection = BiMap::apply(bimap, high);
        let table_key = &low_projection.0;
        debug_assert_eq!(table_key, &high_projection.0);
        spec_diagonals_flat_shifted(bimap, table_key, &low_projection.1, &high_projection.1)
            .flat_map(|spec| self.dependencies_for_spec(&spec))
            .collect()
    }

    fn visit_dependency<U>(&mut self, spec: &Spec<Tgt>, cost: &[NormalizedCost], updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        if cost.len() >= 2 {
            todo!("Support k > 1");
        }

        debug_assert!(spec.is_canonical());
        let Some(requests) = self.requests_map.get_mut(spec) else {
            panic!("spec never requested: {spec}");
        };

        for (working_impl_rc, request_subspec_idx) in requests {
            let mut working_impl = RefCell::borrow_mut(working_impl_rc);

            let new_slot_entry = Some(
                cost.first()
                    .map(|c| c.clone().into_main_cost_for_volume(spec.0.volume())),
            );
            let slot = &mut working_impl.subspec_costs[*request_subspec_idx];
            if slot.is_some() {
                if slot != &new_slot_entry {
                    panic!(
                        "subspec {spec} already set to a different value for parent {}",
                        RefCell::borrow(&working_impl.working_spec).spec
                    );
                }
                log::warn!("subspec {spec} already set");
                return;
            }
            *slot = new_slot_entry;

            if working_impl.subspec_costs.iter().all(|o| o.is_some()) {
                let solver = &working_impl.solver;
                let mut working_spec = RefCell::borrow_mut(&working_impl.working_spec);

                // Once all dependencies for a particular action are available and SAT, call
                // [VisitUpdater::complete_action].
                if working_impl
                    .subspec_costs
                    .iter()
                    .all(|o| matches!(o, Some(Some(_))))
                {
                    let cost = solver.compute_cost(
                        working_impl
                            .subspec_costs
                            .iter()
                            .map(|cost_option| cost_option.clone().unwrap().unwrap()),
                    );

                    let normalized_cost = NormalizedCost::new(cost, spec.0.volume());
                    updater.complete_action(
                        &working_spec.spec,
                        working_impl.action_num,
                        normalized_cost,
                    );
                }

                // Once all dependencies are available but not necessarily SAT, do some bookkeeping
                // around NaiveWorkingImpls.
                working_spec.incomplete_impls -= 1;
                if working_spec.incomplete_impls == 0 {
                    updater.complete_spec(&working_spec.spec);
                }
            }
        }
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        // TODO: Avoid the redundant (with dependencies_for_spec) scan of actions.
        let mut all_actions_complete = true;
        for action in P::actions(&spec.0) {
            match action.top_down_solver(spec) {
                Ok(solver) => {
                    let subspecs = solver.subspecs().collect::<Vec<_>>();
                    if subspecs.is_empty() {
                        let cost = solver.compute_cost(iter::empty());
                        let normalized_cost = NormalizedCost::new(cost, spec.0.volume());
                        updater.complete_action(spec, action.encode(spec), normalized_cost);
                    } else {
                        all_actions_complete = false;
                    }
                }
                Err(ApplyError::NotApplicable(_)) => {}
                Err(ApplyError::SpecNotCanonical) => panic!("given non-canon Spec: {spec}"),
            }
        }
        if all_actions_complete {
            updater.complete_spec(spec);
        }
    }
}

// TODO: Can we replace this function with a more general `utils` crate fn. or something?
/// Push all nested [Spec]s in an Impl into a given [Vec], left to right.
fn collect_nested_specs<Tgt: Target>(imp: &ImplNode<Tgt>, out: &mut Vec<Spec<Tgt>>) {
    match imp {
        ImplNode::SpecApp(spec_app) => {
            out.push(spec_app.0.clone());
        }
        _ => {
            for child in imp.children() {
                collect_nested_specs(child, out);
            }
        }
    }
}

fn compute_impl_cost<Tgt, I>(imp: &ImplNode<Tgt>, costs: &mut I) -> Cost
where
    Tgt: Target,
    I: Iterator<Item = Cost>,
{
    match imp {
        ImplNode::SpecApp(_) => costs.next().unwrap(),
        _ => {
            let child_costs = imp
                .children()
                .iter()
                .map(|child| compute_impl_cost(child, costs))
                .collect::<Vec<_>>();
            Cost::from_node_and_child_costs(imp, &child_costs)
        }
    }
}

/// Build the inner Compose (or Primitive) sub-Spec appliction of a Pipeline.
///
/// The inner sub-Spec is the sub-Spec which is executed first.
fn make_inner_compose<Tgt: Target>(
    index: usize,
    components: &[PrimitiveBasics],
    parent_operand_auxes: &[TensorSpecAux<Tgt>],
    serial_only: bool,
    intermediate_tensor: Tensor<Tgt>,
    new_limits: MemoryLimits,
) -> SpecApp<ViewE<Tgt>> {
    let inner_components = &components[(index + 1)..];
    let inner_input_count = 1 + inner_components
        .iter()
        .map(|c| c.typ.input_count() - 1)
        .sum::<usize>();

    // Collect parameter auxes from the parent Spec and add an aux for the intermediate/output.
    let offset = parent_operand_auxes.len() - inner_input_count - 1;
    let offset_u8 = u8::try_from(offset).unwrap();
    let passthrough_auxes = &parent_operand_auxes[offset..(parent_operand_auxes.len() - 1)];
    let mut auxes = vec![];
    auxes.reserve_exact(passthrough_auxes.len() + 1);
    auxes.extend_from_slice(passthrough_auxes);
    auxes.push(intermediate_tensor.spec().aux.clone());

    // Construct the Spec. (Next, we'll wrap this in a SpecApp.)
    let mut inner_compose = Spec(
        match inner_components {
            [] => unreachable!("should never be empty"),
            [single] => LogicalSpec::Primitive(single.clone(), auxes, serial_only),
            _ => LogicalSpec::Compose {
                components: inner_components.into(),
                operand_auxes: auxes,
                serial_only,
            },
        },
        new_limits,
    );
    inner_compose.canonicalize().unwrap();

    // Above, we inserted the output at the end. Check that's the real output position.
    debug_assert_eq!(
        inner_compose.0.unique_output_index(),
        Some(inner_compose.0.operand_count() - 1),
        "Inner Compose must have a single output for the intermediate to be in the correct position"
    );

    // Parameters
    let params = (0..u8::try_from(inner_compose.0.operand_count() - 1).unwrap())
        .map(|i| {
            ViewE::from(Param::new(
                i + offset_u8,
                inner_compose.0.parameter(i.into()),
            ))
        })
        .chain(std::iter::once(ViewE::from(intermediate_tensor)))
        .collect::<Vec<_>>();

    SpecApp::new(inner_compose, params)
}

/// Build the outer Compose (or Primitive) sub-Spec appliction of a Pipeline.
///
/// The outer sub-Spec is the sub-Spec which is executed second.
fn make_outer_compose<Tgt: Target>(
    index: usize,
    components: &[PrimitiveBasics],
    parent_operand_auxes: &[TensorSpecAux<Tgt>],
    serial_only: bool,
    intermediate_tensor: Tensor<Tgt>,
    new_limits: MemoryLimits,
) -> SpecApp<ViewE<Tgt>> {
    let outer_components = &components[..(index + 1)];
    let outer_input_count = outer_components
        .iter()
        .map(|c| c.typ.input_count() - 1)
        .sum::<usize>();
    let mut outer_operand_auxes = vec![];
    outer_operand_auxes.reserve_exact(outer_input_count + 2);
    outer_operand_auxes.extend_from_slice(&parent_operand_auxes[..outer_input_count]);
    let insertion_point =
        1 + outer_input_count - outer_components.last().unwrap().typ.input_count();
    outer_operand_auxes.insert(insertion_point, intermediate_tensor.0.aux.clone());
    outer_operand_auxes.push(parent_operand_auxes.last().unwrap().clone());
    let mut outer_compose = Spec(
        match outer_components {
            [] => unreachable!("should never be empty"),
            [single] => LogicalSpec::Primitive(single.clone(), outer_operand_auxes, serial_only),
            _ => LogicalSpec::Compose {
                components: outer_components.into(),
                operand_auxes: outer_operand_auxes,
                serial_only,
            },
        },
        new_limits.clone(),
    );
    outer_compose.canonicalize().unwrap();

    let mut args = vec![];
    args.reserve_exact(outer_input_count + 2);
    for (parent_idx, idx) in (0..insertion_point)
        .chain((insertion_point + 1)..(outer_input_count + 1))
        .enumerate()
    {
        args.push(ViewE::from(Param::new(
            parent_idx.try_into().unwrap(),
            outer_compose.0.parameter(idx),
        )));
    }
    args.insert(insertion_point, ViewE::from(intermediate_tensor));
    args.push(ViewE::from(Param::new(
        (parent_operand_auxes.len() - 1).try_into().unwrap(),
        outer_compose.0.unique_output().unwrap(),
    )));
    SpecApp::new(outer_compose, args)
}

/// Converts an internal [TileError] to an external [ApplyError].
fn tile_to_apply_err(err: TileError) -> ApplyError {
    match err {
        TileError::LayoutIncompatible(_) => {
            ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible)
        }
    }
}

// TODO: Move tests into their respective mods
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        imp::visit_leaves,
        memorylimits::MemVec,
        spec::arb_canonical_spec,
        target::{ArmTarget, X86Target},
    };
    use proptest::prelude::*;

    proptest! {
        // TODO: Add an ARM variant
        #[test]
        fn test_fast_path_is_equivalent_to_slow(spec in arb_canonical_spec::<X86Target>(None, None)) {
            for action in X86Target::actions(&spec.0) {
                match (action.top_down_solver(&spec), action.apply(&spec)) {
                    (Ok(solver), Ok(applied)) => {
                        let subspecs = solver.subspecs().collect::<Vec<_>>();
                        let mut applied_subspecs = Vec::new();
                        collect_nested_specs(&applied, &mut applied_subspecs);
                        prop_assert_eq!(&subspecs, &applied_subspecs);

                        // Generate some quick-n'-dirty sub-Spec costs and confirm that the fast
                        // and slow paths yield the same final cost.
                        let subspec_costs = (0..u8::try_from(subspecs.len()).unwrap())
                            .map(|subspec_idx| {
                                Cost {
                                    main: subspec_idx.into(),
                                    peaks: MemVec::zero::<X86Target>(),
                                    depth: subspec_idx,
                                }
                            })
                            .collect::<Vec<_>>();
                        let solver_cost = solver.compute_cost(subspec_costs.iter().cloned());
                        let applied_cost = compute_impl_cost(&applied, &mut subspec_costs.into_iter());
                        prop_assert_eq!(solver_cost, applied_cost);
                    },
                    (Err(a), Err(b)) => prop_assert_eq!(a, b),
                    (l, r) => {
                        prop_assert!(false, "solver returned {l:?} but apply returned {r:?}");
                    }
                }
            }
        }

        // TODO: Add an ARM variant
        #[test]
        fn test_action_encode_consistent_with_decode(spec in arb_canonical_spec::<X86Target>(None, None)) {
            for action in X86Target::actions(&spec.0) {
                let encoding = action.encode(&spec);
                let decoded = Action::decode(&spec, encoding);
                prop_assert_eq!(action, decoded);
            }
        }

        #[test]
        fn test_actions_introduce_subspec_arguments_with_matching_parameters_x86(
            spec in arb_canonical_spec::<X86Target>(None, None),
        ) {
            shared_test_actions_introduce_subspec_arguments_with_matching_parameters(spec)?;
        }

        #[test]
        fn test_actions_introduce_subspec_arguments_with_matching_parameters_arm(
            spec in arb_canonical_spec::<ArmTarget>(None, None),
        ) {
            shared_test_actions_introduce_subspec_arguments_with_matching_parameters(spec)?;
        }
    }

    // TODO: Add a solver version, if possible.
    fn shared_test_actions_introduce_subspec_arguments_with_matching_parameters<Tgt: Target>(
        spec: Spec<Tgt>,
    ) -> Result<(), proptest::prelude::TestCaseError> {
        for action in Tgt::actions(&spec.0) {
            // Skip Moves since they sometimes introduce Move Specs with mismatched
            // TensorSpecs.
            if matches!(action, Action::Move { .. }) {
                continue;
            }

            match action.apply(&spec) {
                Ok(rewritten) => {
                    let mut mismatch: Option<SpecApp<_>> = None;
                    visit_leaves(&rewritten, &mut |leaf| {
                        if let ImplNode::SpecApp(spec_app) = leaf {
                            // TODO: Do we really need this?
                            // Also skip Move sub-Specs, which can be introduced by actions other
                            // than Move.
                            if matches!(
                                spec_app.0 .0,
                                LogicalSpec::Primitive(
                                    PrimitiveBasics {
                                        typ: PrimitiveSpecType::Move,
                                        ..
                                    },
                                    ..
                                )
                            ) {
                                return true;
                            }

                            let spec_parameters = spec_app.0 .0.parameters();
                            let argument_tensorspecs = spec_app
                                .1
                                .iter()
                                .map(|v| v.spec().clone())
                                .collect::<Vec<_>>();
                            if spec_parameters != argument_tensorspecs {
                                mismatch = Some(spec_app.clone());
                                return false;
                            }
                        }
                        true
                    });
                    prop_assert!(
                        mismatch.is_none(),
                        "incorrect sub-Spec arguments after rewriting {} with {:?}: {}",
                        spec,
                        action,
                        mismatch.unwrap()
                    );
                }
                Err(ApplyError::NotApplicable(_)) => (),
                // TODO: Replace panic with a proptest-friendly result
                Err(e) => panic!("unexpected error: {:?}", e),
            };
        }
        Ok(())
    }
}
