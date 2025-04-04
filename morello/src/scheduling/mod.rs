use crate::common::DimSize;
use crate::cost::{Cost, MainCost, NormalizedCost};
use crate::datadeps::SpecKey;
use crate::db::{ActionNum, DbKey, TableKey};
use crate::grid::general::BiMap;
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::imp::loops::compute_loop_main_cost;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::rtree::RTreeDyn;
use crate::search::ImplReducer;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::utils::diagonals_shifted;
use crate::views::{Param, Tensor, TileError, View, ViewE};
use auto_impl::auto_impl;
use itertools::izip;
use once_cell::unsync::OnceCell;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::{self, Debug};
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
            .map(|applied| ActionTopDownSolver::Fallback(Box::new(applied)))
    }

    fn bottom_up_solvers() -> Self::BSolverIter;
}

/// Provides (some of) a [Spec]'s action dependencies and computes the cost of one or more of that
/// [Spec]'s actions from those dependencies.
pub trait BottomUpSolver {
    type Tgt: Target;
    type Request: DependencyRequest<Tgt = Self::Tgt>;

    fn request(&mut self, dependents: &SpecGeometry<Self::Tgt>) -> Self::Request;
}

/// A type, returned by [BottomUpSolver], which provides some dependencies of a [Spec]s' possible
/// implementations and then can be used to compute the costs of those possible implementations.
pub trait DependencyRequest {
    type Tgt: Target;

    fn queries(&self) -> Option<&SpecGeometry<Self::Tgt>>;

    fn visit_dependency<U>(
        &mut self,
        rectangle: &SpecGeometryRect<Self::Tgt>,
        cost: &[NormalizedCost],
        updater: &mut U,
    ) where
        U: VisitUpdater<Self::Tgt>;

    // TODO: Add a visit_dependency_range fn.

    // TODO: We need a rectangle-level version of this or do it during initialization.
    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>;
}

/// A trait for putting final Spec-action costs. [BottomUpSolver::visit_dependency] is expected to
/// call [VisitUpdater::complete] for satisfied solutions. (No call for unsat.)
#[auto_impl(&mut)]
pub trait VisitUpdater<Tgt: Target> {
    /// Store the cost and action number for a single implementation of `spec`.
    fn complete_action(
        &mut self,
        spec: &Spec<Tgt>,
        action: ActionNum,
        normalized_cost: NormalizedCost,
    );

    // TODO: Take `self` as a value to enforce lifetime.
    /// Finalize a Spec after [VisitUpdater::complete_action] called for all possible decisions.
    fn complete_spec(&mut self, spec: &Spec<Tgt>);
}

/// An actions provider for [NaiveBottomUpSolver].
pub trait NaiveBottomUpActionProvider<Tgt: Target> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>>;
}

/// Wraps an [RTreeDyn] to restruct mutations to those which are relatively independent of the
/// database's underlying space.
///
/// **Note:** This API is unstable and experimental.
#[derive(Clone)]
pub struct SpecGeometry<Tgt: Target>(
    pub(crate) HashMap<TableKey, RTreeDyn<()>>,
    Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
);

// TODO: Can we take more of these fields by reference? This requires a lot of cloning.
#[derive(Clone)]
pub struct SpecGeometryRect<Tgt: Target> {
    key: TableKey,
    bottom: Vec<BimapInt>,
    top: Vec<BimapInt>,
    cached_specs: OnceCell<(Spec<Tgt>, Spec<Tgt>)>,
    bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
}

pub trait ActionEncodeDecode<Tgt: Target> {
    fn encode(&self, spec: &Spec<Tgt>) -> ActionNum;
    fn decode(spec: &Spec<Tgt>, encoding: ActionNum) -> Self;
}

macro_rules! action_dispatch {
    (
        $(#[$meta:meta])* $name:ident, $solver:ident, $request:ident, $(($variant:tt, $innertype:ty)),*$(,)*
    ) => {
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

        #[allow(non_snake_case)]
        pub enum $request<Tgt: Target> {
            $( $variant(<<$innertype as ActionT<Tgt>>::BSolver as BottomUpSolver>::Request) ),*
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
            type Request = $request<Tgt>;

            fn request(&mut self, dependents: &SpecGeometry<Tgt>) -> Self::Request {
                match self {
                    $( Self::$variant(a) => $request::$variant(a.request(dependents)) ),*
                }
            }
        }

        impl<Tgt: Target> DependencyRequest for $request<Tgt> {
            type Tgt = Tgt;

            fn queries(&self) -> Option<&SpecGeometry<Tgt>> {
                match self {
                    $( Self::$variant(a) => a.queries() ),*
                }
            }

            fn visit_dependency<U>(
                &mut self,
                rectangle: &SpecGeometryRect<Tgt>,
                cost: &[NormalizedCost],
                updater: &mut U,
            )
            where
                U: VisitUpdater<Self::Tgt>
            {
                match self {
                    $( Self::$variant(a) => {
                        a.visit_dependency(rectangle, cost, updater)
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

        // $(
        //     impl<Tgt: Target> From<<<$innertype as ActionT<Tgt>>::BSolver as BottomUpSolver>::Request> for $request<Tgt> {
        //         fn from(value: <<$innertype as ActionT<Tgt>>::BSolver as BottomUpSolver>::Request) -> Self {
        //             $request::$variant(value)
        //         }
        //     }
        // )?
    };
}

action_dispatch! {
    /// A scheduling decision which can be applied to a Spec to produce an Impl.
    ///
    /// [Action]s contain the minimal amount of information needed to distinguish a one scheduling
    /// decision from another for a given Spec.
    Action,
    ActionBottomUpSolver,
    ActionBottomUpSolverRequest,
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
    PrimitiveTileOut(Box<PrimitiveTileOutSolver<Tgt>>),
    Move(Box<MoveActionSolver<Tgt>>),
    Fallback(Box<ImplNode<Tgt>>),
}

#[derive(Debug)]
pub struct PrimitiveTileOutSolver<Tgt: Target> {
    outer_spec: Spec<Tgt>,
    body_specs: Vec<Spec<Tgt>>,
}

#[derive(Debug)]
pub struct MoveActionSolver<Tgt: Target> {
    prologue: Option<Spec<Tgt>>,
    body: Spec<Tgt>,
    epilogue: Option<Spec<Tgt>>,
    base_main_cost: MainCost,
    allocation: MemoryAllocation,
}

#[derive(Debug, Default)]
pub struct NaiveBottomUpSolver<Tgt: Target, P> {
    _phantom: PhantomData<(Tgt, P)>,
}

type RequestsVec<Tgt> = Vec<(Rc<RefCell<NaiveWorkingImpl<Tgt>>>, usize)>;

#[derive(Debug)]
pub struct NaiveBottomUpSolverRequest<Tgt: Target, P> {
    requests: SpecGeometry<Tgt>, // TODO: Redundant with requests_maps' keys
    requests_map: HashMap<Spec<Tgt>, RequestsVec<Tgt>>,
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
    VectorSizeInvalid,
    VectorSizeVolumeIncompatible,
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
        reducer.insert(action, normalized_cost);
    }

    fn complete_spec(&mut self, _spec: &Spec<Tgt>) {}
}

impl<Tgt: Target> SpecGeometry<Tgt> {
    pub fn new(bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>) -> Self {
        Self(HashMap::new(), bimap)
    }

    pub fn single(
        spec: &Spec<Tgt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        let mut sg = Self::new(bimap);
        sg.insert_spec(spec);
        sg
    }

    pub fn bimap(&self) -> &Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>> {
        &self.1
    }

    pub fn insert_spec(&mut self, spec: &Spec<Tgt>) {
        let (key, pt) = self.1.apply(spec);
        let pt_i64 = pt.iter().map(|v| BimapSInt::from(*v)).collect::<Vec<_>>();
        self.0
            .entry(key)
            .or_insert_with(|| RTreeDyn::empty(pt.len()))
            .merge_insert(&pt_i64, &pt_i64, (), true);
    }

    pub fn insert_rect(&mut self, rect: SpecGeometryRect<Tgt>) {
        // TODO: Assert BiMaps are the same.
        let SpecGeometryRect {
            key,
            bottom,
            top,
            cached_specs: _,
            bimap: _,
        } = rect;
        let bottom_i64 = bottom.iter().map(|&v| v.into()).collect::<Vec<_>>();
        let top_i64 = top.iter().map(|&v| v.into()).collect::<Vec<_>>();
        self.0
            .entry(key)
            .or_insert_with(|| RTreeDyn::empty(bottom.len()))
            .merge_insert(&bottom_i64, &top_i64, (), true);
    }

    pub fn extend(&mut self, source: impl Iterator<Item = SpecGeometryRect<Tgt>>) {
        source.for_each(|rect| {
            self.insert_rect(rect);
        });
    }

    pub fn accums(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.iter().flat_map(|rect| rect.accums())
    }

    /// Yields `Fill` Specs for the outputs of every Spec in the [SpecGeometry].
    pub fn outputs_fills(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.iter().flat_map(|rect| rect.outputs_fills())
    }

    pub fn is_empty(&self) -> bool {
        self.0.values().all(|tree| tree.is_empty())
    }

    /// Iterates over rectangles' bottom and top [Spec]s.
    pub fn iter(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> + '_ {
        self.0.iter().flat_map(move |(key, rtree)| {
            rtree.iter().map(move |rect| SpecGeometryRect {
                key: key.clone(),
                bottom: rect.0.iter().map(|v| (*v).try_into().unwrap()).collect(),
                top: rect.1.iter().map(|v| (*v).try_into().unwrap()).collect(),
                cached_specs: OnceCell::new(),
                bimap: Rc::clone(&self.1),
            })
        })
    }
}

impl<Tgt: Target> From<SpecGeometryRect<Tgt>> for SpecGeometry<Tgt> {
    fn from(rect: SpecGeometryRect<Tgt>) -> Self {
        let SpecGeometryRect {
            key,
            bottom,
            top,
            cached_specs: _,
            bimap,
        } = rect;
        let bottom_i64 = bottom.iter().map(|v| (*v).into()).collect::<Vec<_>>();
        let top_i64 = top.iter().map(|v| (*v).into()).collect::<Vec<_>>();
        let mut rtree = RTreeDyn::empty(bottom.len());
        rtree.insert(&bottom_i64, &top_i64, ());
        SpecGeometry(std::iter::once((key, rtree)).collect(), bimap)
    }
}

impl<Tgt: Target> Debug for SpecGeometry<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpecGeometry")
            .field(&self.0)
            .finish_non_exhaustive()
    }
}

impl<Tgt: Target> SpecGeometryRect<Tgt> {
    // TODO: Remove. Only exists for wrapping RTree rect in [synthesize_block].
    pub(crate) fn new(
        key: TableKey,
        bottom: Vec<BimapInt>,
        top: Vec<BimapInt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        assert_eq!(bottom.len(), top.len());
        assert!(bottom.iter().zip(&top).all(|(b, t)| b <= t));
        Self {
            key,
            bottom,
            top,
            cached_specs: OnceCell::new(),
            bimap,
        }
    }

    // TODO: Remove. Only exists for wrapping RTree rect in [synthesize_block].
    pub(crate) fn single(
        spec: &Spec<Tgt>,
        bimap: Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>>,
    ) -> Self {
        let (key, pt) = bimap.apply(spec);
        Self {
            key,
            bottom: pt.clone(),
            top: pt,
            cached_specs: OnceCell::with_value((spec.clone(), spec.clone())),
            bimap,
        }
    }

    // TODO: Remove. Only exists for wrapping RTree rect in [synthesize_block].
    pub(crate) fn bimap(&self) -> Rc<dyn BiMap<Domain = Spec<Tgt>, Codomain = DbKey>> {
        Rc::clone(&self.bimap)
    }

    pub fn table_key(&self) -> &TableKey {
        &self.key
    }

    pub fn spec_key(&self) -> &SpecKey {
        &self.key.0
    }

    pub fn bottom(&self) -> &Spec<Tgt> {
        &self.specs().0
    }

    pub fn top(&self) -> &Spec<Tgt> {
        &self.specs().1
    }

    pub fn bottom_point(&self) -> &[BimapInt] {
        &self.bottom
    }

    pub fn top_point(&self) -> &[BimapInt] {
        &self.top
    }

    pub(crate) fn intersects(&self, other: &SpecGeometryRect<Tgt>) -> bool {
        for (self_bottom, self_top, other_bottom, other_top) in
            izip!(&self.bottom, &self.top, &other.bottom, &other.top)
        {
            if self_top < other_bottom || self_bottom > other_top {
                return false;
            }
        }
        true
    }

    pub fn accums(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> {
        let key @ (spec_key, _) = self.table_key();

        let mut result: Option<SpecGeometryRect<Tgt>> = None;
        // If it has an accumulating variant, it's in the first dimension.
        // TODO: Abstract this `match` away somehow.
        let has_accum = match spec_key {
            SpecKey::Matmul { .. }
            | SpecKey::Conv { .. }
            | SpecKey::SoftmaxDenominatorAndMax { .. }
            | SpecKey::Max { .. }
            | SpecKey::SoftmaxDenominator { .. }
            | SpecKey::SoftmaxDenominatorAndUnscaled { .. }
            | SpecKey::SoftmaxDenominatorAndUnscaledFromMax { .. } => true,
            SpecKey::Softmax { .. }
            | SpecKey::SoftmaxComplete { .. }
            | SpecKey::Move { .. }
            | SpecKey::Fill { .. }
            | SpecKey::OnePrefix { .. }
            | SpecKey::Broadcast { .. }
            | SpecKey::DivideVec { .. }
            | SpecKey::DivideVecScalar { .. }
            | SpecKey::Compose { .. } => false,
        };
        if has_accum && self.bottom_point()[0] > 0 {
            let mut bottom = self.bottom_point().to_vec();
            let mut top = self.top_point().to_vec();
            bottom[0] = 0;
            top[0] = 0;
            result = Some(SpecGeometryRect::new(
                key.clone(),
                bottom,
                top,
                self.bimap(),
            ));
        }
        result.into_iter()
    }

    /// Yields `Fill` Specs for the outputs of every Spec in the [SpecGeometryRect].
    pub fn outputs_fills(&self) -> impl Iterator<Item = SpecGeometryRect<Tgt>> {
        let bimap = self.bimap();

        let top_output_idx = self.top().0.unique_output_index().unwrap();
        let bottom_output_idx = self.bottom().0.unique_output_index().unwrap();
        let TensorSpec {
            shape: top_output_shape,
            dtype: top_output_dtype,
            aux: top_output_aux,
        } = self.top().0.parameter(top_output_idx);
        let TensorSpec {
            shape: bottom_output_shape,
            dtype: bottom_output_dtype,
            aux: bottom_output_aux,
        } = self.bottom().0.parameter(bottom_output_idx);

        let fill_value = self
            .top()
            .0
            .initial_accumulating_value_for_output(top_output_idx)
            .expect("rect should have Spec kinds supporting accumulating");
        debug_assert_eq!(
            fill_value,
            self.bottom()
                .0
                .initial_accumulating_value_for_output(bottom_output_idx)
                .unwrap()
        );

        let mut fill_top = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill { value: fill_value },
                    spec_shape: top_output_shape,
                    dtypes: vec![top_output_dtype],
                },
                vec![top_output_aux],
                self.top().0.serial_only(),
            ),
            self.top().1.clone(),
        );
        let mut fill_bottom = Spec(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill { value: fill_value },
                    spec_shape: bottom_output_shape,
                    dtypes: vec![bottom_output_dtype],
                },
                vec![bottom_output_aux],
                self.bottom().0.serial_only(),
            ),
            self.bottom().1.clone(),
        );
        fill_top.canonicalize().unwrap();
        fill_bottom.canonicalize().unwrap();

        let (fill_key, fill_top_pt) = bimap.apply(&fill_top);
        let (fill_bottom_key, fill_bottom_pt) = bimap.apply(&fill_bottom);
        debug_assert_eq!(fill_key, fill_bottom_key);

        iter::once(SpecGeometryRect::<Tgt>::new(
            fill_key,
            fill_bottom_pt,
            fill_top_pt,
            bimap,
        ))
    }

    /// Returns an iterator over the [Spec]s in this rectangle.
    pub fn iter_specs(&self) -> impl Iterator<Item = Spec<Tgt>> + '_ {
        let low_pt = self.bottom_point();
        let high_pt = self.top_point();
        diagonals_shifted(low_pt, high_pt).flatten().map(|pt| {
            let composed_key = (self.key.clone(), pt.to_vec());
            let mut spec = self.bimap.apply_inverse(&composed_key);
            spec.canonicalize().unwrap();
            spec
        })
    }

    /// Returns the bottom and top [Spec]s for [bottom_point] and [top_point].
    ///
    /// These are lazily cached.
    fn specs(&self) -> &(Spec<Tgt>, Spec<Tgt>) {
        self.cached_specs
            .get_or_try_init(|| -> Result<_, ()> {
                let mut projection = (
                    self.key.clone(),
                    self.bottom_point()
                        .iter()
                        .map(|v| BimapInt::try_from(*v).unwrap())
                        .collect::<Vec<_>>(),
                );
                let bottom = self.bimap.apply_inverse(&projection);
                projection.1 = self
                    .top_point()
                    .iter()
                    .map(|v| BimapInt::try_from(*v).unwrap())
                    .collect::<Vec<_>>();
                let top = self.bimap.apply_inverse(&projection);
                Ok((bottom, top))
            })
            .unwrap()
    }
}

impl<Tgt: Target> fmt::Debug for SpecGeometryRect<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpecGeometryRect")
            .field("key", &self.key)
            .field("bottom", &self.bottom)
            .field("top", &self.top)
            .field("cached_specs", &self.cached_specs)
            .finish_non_exhaustive()
    }
}

// TODO: Encode without iterating actions.
impl<Tgt: Target> ActionEncodeDecode<Tgt> for Action<Tgt> {
    fn encode(&self, spec: &Spec<Tgt>) -> ActionNum {
        Tgt::actions(&spec.0)
            .position(|a| a == *self)
            .unwrap_or_else(|| panic!("Action {self:?} does not apply to {}", spec.0))
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
            ActionTopDownSolver::PrimitiveTileOut(solver) => {
                // TODO: Avoid this clone
                solver.body_specs.clone().into_iter()
            }
            ActionTopDownSolver::Move(move_solver) => {
                // TODO: Avoid these clones. Return an iterator of references.
                let mut v: Vec<Spec<Tgt>> = Vec::with_capacity(3);
                v.extend(move_solver.prologue.clone());
                v.push(move_solver.body.clone());
                v.extend(move_solver.epilogue.clone());
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
            ActionTopDownSolver::PrimitiveTileOut(solver) => {
                if solver.body_specs.is_empty() {
                    unreachable!("PrimitiveTileOut should have at least one body spec");
                }

                let parallel =
                    !solver.outer_spec.0.serial_only() && solver.body_specs[0].0.serial_only();
                match &solver.outer_spec.0 {
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
                                spec_shape: main_body_shape,
                                ..
                            },
                            ..,
                        ) = &solver.body_specs[0].0
                        else {
                            unreachable!();
                        };

                        let mut child_main_costs = Vec::with_capacity(solver.body_specs.len());
                        let first_cost = child_costs.next().unwrap();
                        let mut max_depth = 0;
                        child_main_costs.push(first_cost.main);
                        for child_cost in child_costs {
                            child_main_costs.push(child_cost.main);
                            max_depth = max_depth.max(child_cost.depth);
                        }
                        debug_assert_eq!(child_main_costs.len(), solver.body_specs.len());

                        // per-axis (steps, full_steps) pairs for cost dispatch
                        let dims: Vec<(u32, u32)> = spec_shape
                            .iter()
                            .zip(main_body_shape)
                            .map(|(o, t)| (o.get().div_ceil(t.get()), o.get() / t.get()))
                            .collect();

                        let mut combined_cost = first_cost.clone();
                        combined_cost.main =
                            compute_loop_main_cost::<Tgt>(&dims, parallel, &child_main_costs);
                        combined_cost.depth = max_depth + 1;
                        combined_cost
                    }
                    _ => unreachable!(),
                }
            }
            ActionTopDownSolver::Move(move_solver) => {
                let mut main = move_solver.base_main_cost;
                let mut child_peaks = vec![];
                let mut depth = 0;
                for child_cost in child_costs {
                    main = main.saturating_add(child_cost.main);
                    depth = depth.max(child_cost.depth);
                    child_peaks.push(child_cost.peaks);
                }
                depth += 1;
                // TODO: Is snap_up really needed or can we bake this into MemVec?
                let peaks = move_solver
                    .allocation
                    .peak_memory_from_child_peaks::<Tgt>(&child_peaks)
                    .snap_up_for_target::<Tgt>(false);
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
            if outer.level().has_layout() {
                let Ok(new_layout) = outer
                    .layout()
                    .update_for_tiling(outer.shape(), inner.shape())
                else {
                    todo!();
                };
                new_aux.layout = new_layout;
            }
        }

        if new_spec.canonicalize().is_err() {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeInvalid,
            ));
        }

        Ok(new_spec)
    }
}

impl<Tgt: Target> From<MoveActionSolver<Tgt>> for ActionTopDownSolver<Tgt> {
    fn from(move_solver: MoveActionSolver<Tgt>) -> Self {
        ActionTopDownSolver::Move(Box::new(move_solver))
    }
}

impl<Tgt: Target> From<Box<MoveActionSolver<Tgt>>> for ActionTopDownSolver<Tgt> {
    fn from(move_solver: Box<MoveActionSolver<Tgt>>) -> Self {
        ActionTopDownSolver::Move(move_solver)
    }
}

impl<Tgt: Target> From<PrimitiveTileOutSolver<Tgt>> for ActionTopDownSolver<Tgt> {
    fn from(tile_out_solver: PrimitiveTileOutSolver<Tgt>) -> Self {
        ActionTopDownSolver::PrimitiveTileOut(Box::new(tile_out_solver))
    }
}

impl<Tgt: Target> From<Box<PrimitiveTileOutSolver<Tgt>>> for ActionTopDownSolver<Tgt> {
    fn from(tile_out_solver: Box<PrimitiveTileOutSolver<Tgt>>) -> Self {
        ActionTopDownSolver::PrimitiveTileOut(tile_out_solver)
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
            NotApplicableReason::VectorSizeInvalid => {
                write!(
                    f,
                    "Target does not support the specified vector size for this data type"
                )
            }
            NotApplicableReason::VectorSizeVolumeIncompatible => {
                write!(
                    f,
                    "Tensor volume is not compatible with the specified vector size"
                )
            }
            NotApplicableReason::MultipleOutputs => {
                write!(f, "Spec has multiple outputs")
            }
            NotApplicableReason::Other(Some(reason_string)) => write!(f, "{reason_string}"),
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

    Ok(())
}

impl<Tgt, P> BottomUpSolver for NaiveBottomUpSolver<Tgt, P>
where
    Tgt: Target,
    P: NaiveBottomUpActionProvider<Tgt>,
{
    type Tgt = Tgt;
    type Request = NaiveBottomUpSolverRequest<Tgt, P>;

    fn request(&mut self, dependents: &SpecGeometry<Tgt>) -> Self::Request {
        let mut requests_map = HashMap::<Spec<Tgt>, Vec<_>>::new();
        let mut requests = SpecGeometry::new(Rc::clone(&dependents.1));
        dependents.iter().for_each(|rect| {
            rect.iter_specs().for_each(|spec| {
                // TODO: Test that this never returns and correctly visits duplicate sub-Specs.
                // TODO: Test that this can be called twice without messing up our internal state.

                let working_spec = Rc::new(RefCell::new(NaiveWorkingSpec {
                    spec: spec.clone(),
                    impls: vec![],
                    incomplete_impls: 0,
                }));

                for action in P::actions(&spec.0) {
                    match action.top_down_solver(&spec) {
                        Ok(solver) => {
                            let subspecs = solver.subspecs().collect::<Vec<_>>();
                            if subspecs.is_empty() {
                                continue;
                            }
                            let working_impl = Rc::new(RefCell::new(NaiveWorkingImpl {
                                working_spec: Rc::clone(&working_spec),
                                solver,
                                action_num: action.encode(&spec),
                                subspec_costs: vec![None; subspecs.len()],
                            }));
                            let mut working_spec_mut = RefCell::borrow_mut(&working_spec);
                            working_spec_mut.impls.push(Rc::downgrade(&working_impl));
                            if !subspecs.is_empty() {
                                working_spec_mut.incomplete_impls += 1;
                            }

                            for (subspec_idx, subspec) in subspecs.into_iter().enumerate() {
                                requests.insert_spec(&subspec);
                                requests_map
                                    .entry(subspec)
                                    .or_default()
                                    .push((Rc::clone(&working_impl), subspec_idx));
                            }
                        }
                        Err(ApplyError::NotApplicable(_)) => {}
                        Err(ApplyError::SpecNotCanonical) => {
                            panic!("given non-canon Spec: {spec}")
                        }
                    }
                }
            })
        });
        NaiveBottomUpSolverRequest {
            requests,
            requests_map,
            _phantom: PhantomData,
        }
    }
}

impl<Tgt, P> DependencyRequest for NaiveBottomUpSolverRequest<Tgt, P>
where
    Tgt: Target,
    P: NaiveBottomUpActionProvider<Tgt>,
{
    type Tgt = Tgt;

    fn queries(&self) -> Option<&SpecGeometry<Tgt>> {
        Some(&self.requests)
    }

    fn visit_dependency<U>(
        &mut self,
        rectangle: &SpecGeometryRect<Tgt>,
        cost: &[NormalizedCost],
        updater: &mut U,
    ) where
        U: VisitUpdater<Self::Tgt>,
    {
        if cost.len() >= 2 {
            todo!("Support k > 1");
        }

        rectangle.iter_specs().for_each(|spec| {
            debug_assert!(spec.is_canonical());
            let Some(requests) = self.requests_map.get_mut(&spec) else {
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
                            "subspec {spec} already set to a different value: {:?} != {:?}",
                            slot.as_ref().unwrap(),
                            new_slot_entry.as_ref().unwrap()
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

                        let normalized_cost =
                            NormalizedCost::new(cost, working_spec.spec.0.volume());
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
        });
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        // TODO: Avoid the redundant (with dependencies) scan of actions.
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
        imp::ImplNode,
        memorylimits::MemVec,
        spec::arb_canonical_spec,
        target::{ArmTarget, X86Target},
    };
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_fast_path_is_equivalent_to_slow_x86(spec in arb_canonical_spec::<X86Target>(None, None)) {
            shared_test_fast_path_is_equivalent_to_slow(spec)?;
        }

        #[test]
        fn test_fast_path_is_equivalent_to_slow_arm(spec in arb_canonical_spec::<ArmTarget>(None, None)) {
            shared_test_fast_path_is_equivalent_to_slow(spec)?;
        }

        #[test]
        fn test_action_encode_consistent_with_decode_x86(spec in arb_canonical_spec::<X86Target>(None, None)) {
            shared_test_action_encode_consistent_with_decode(spec)?;
        }

        #[test]
        fn test_action_encode_consistent_with_decode_arm(spec in arb_canonical_spec::<ArmTarget>(None, None)) {
            shared_test_action_encode_consistent_with_decode(spec)?;
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

        #[test]
        fn test_actions_do_not_introduce_self_nested_subspec_x86(
            spec in arb_canonical_spec::<X86Target>(None, None),
        ) {
            shared_test_actions_do_not_introduce_self_nested_subspec(spec)?;
        }

        #[test]
        fn test_actions_do_not_introduce_self_nested_subspec_arm(
            spec in arb_canonical_spec::<ArmTarget>(None, None),
        ) {
            shared_test_actions_do_not_introduce_self_nested_subspec(spec)?;
        }
    }

    fn shared_test_action_encode_consistent_with_decode<Tgt: Target>(
        spec: Spec<Tgt>,
    ) -> Result<(), proptest::prelude::TestCaseError> {
        for action in Tgt::actions(&spec.0) {
            let encoding = action.encode(&spec);
            let decoded = Action::decode(&spec, encoding);
            prop_assert_eq!(action, decoded);
        }
        Ok(())
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
                    rewritten.visit_leaves(&mut |leaf| {
                        if let ImplNode::SpecApp(spec_app) = leaf {
                            // TODO: Do we really need this?
                            // Skip Move sub-Specs, which can be introduced by actions other
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
                Err(e) => panic!("unexpected error: {e:?}"),
            };
        }
        Ok(())
    }

    fn shared_test_actions_do_not_introduce_self_nested_subspec<Tgt: Target>(
        spec: Spec<Tgt>,
    ) -> Result<(), TestCaseError> {
        for action in Tgt::actions(&spec.0) {
            let Ok(rewritten) = action.apply(&spec) else {
                continue;
            };
            let mut found_self = false;
            rewritten.visit_leaves(&mut |leaf| {
                if let ImplNode::SpecApp(spec_app) = leaf {
                    if spec_app.0 == spec {
                        found_self = true;
                        return false;
                    }
                }
                true
            });
            prop_assert!(
                !found_self,
                "action {action:?} applied to {spec} yielded sub-Spec cycle",
            );
        }
        Ok(())
    }
    fn shared_test_fast_path_is_equivalent_to_slow<Tgt: Target>(
        spec: Spec<Tgt>,
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        for action in Tgt::actions(&spec.0) {
            match (action.top_down_solver(&spec), action.apply(&spec)) {
                (Ok(solver), Ok(applied)) => {
                    let subspecs = solver.subspecs().collect::<Vec<_>>();
                    let mut applied_subspecs = Vec::new();
                    collect_nested_specs(&applied, &mut applied_subspecs);
                    prop_assert_eq!(&subspecs, &applied_subspecs);

                    // Generate some quick-n'-dirty sub-Spec costs and confirm that the fast
                    // and slow paths yield the same final cost.
                    let subspec_costs = (0..u8::try_from(subspecs.len()).unwrap())
                        .map(|subspec_idx| Cost {
                            main: subspec_idx.into(),
                            peaks: MemVec::zero::<Tgt>(),
                            depth: subspec_idx,
                        })
                        .collect::<Vec<_>>();
                    let solver_cost = solver.compute_cost(subspec_costs.iter().cloned());
                    let applied_cost = compute_impl_cost(&applied, &mut subspec_costs.into_iter());
                    prop_assert_eq!(solver_cost, applied_cost);
                }
                (Err(a), Err(b)) => prop_assert_eq!(a, b),
                (l, r) => {
                    prop_assert!(false, "solver returned {l:?} but apply returned {r:?}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod shared_tests {
    pub mod internal {
        use crate::{
            cost::NormalizedCost, db::ActionNum, scheduling::VisitUpdater, search::ImplReducer,
            spec::Spec, target::Target,
        };

        pub struct MockVisitUpdater<Tgt: Target> {
            pub goal: Spec<Tgt>,
            pub reducer: ImplReducer,
            pub goal_completed: bool,
        }

        impl<Tgt: Target> VisitUpdater<Tgt> for MockVisitUpdater<Tgt> {
            fn complete_action(
                &mut self,
                spec: &Spec<Tgt>,
                action_num: ActionNum,
                normalized_cost: NormalizedCost,
            ) {
                if self.goal_completed {
                    panic!("complete_action called after complete_spec");
                }
                assert_eq!(&self.goal, spec);
                self.reducer.insert(action_num, normalized_cost);
            }

            fn complete_spec(&mut self, spec: &Spec<Tgt>) {
                if self.goal_completed {
                    panic!("complete_spec called twice");
                }
                assert_eq!(&self.goal, spec);
                self.goal_completed = true;
            }
        }
    }

    #[macro_export]
    macro_rules! emit_shared_naivebottomupactionprovider_tests {
        ($tgt:ty, $provider:ty, $suffix:ident) => {
            paste::paste! { proptest::prelude::proptest! {
                #[test]
                fn [< test_naivebottomupsolver_queries_for_single_canonical_spec_ $suffix >](
                    spec in $crate::spec::arb_canonical_spec::<$tgt>(None, None)
                ) {
                    use std::collections::HashSet;
                    use std::default::Default;
                    use std::rc::Rc;
                    use proptest::prelude::*;
                    use $crate::db::db_spec_bimap;
                    use $crate::scheduling::{
                        SpecGeometry, BottomUpSolver, NaiveBottomUpSolver, DependencyRequest
                    };

                    let mut solver = NaiveBottomUpSolver::<$tgt, $provider>::default();
                    let single_spec_geometry = SpecGeometry::single(
                        &spec, Rc::new(db_spec_bimap(false))
                    );
                    let request = solver.request(&single_spec_geometry);

                    let mut query_specs = HashSet::new();
                    if let Some(queries_geometry) = request.queries() {
                        queries_geometry.iter().for_each(|rect| {
                            query_specs.extend(rect.iter_specs());
                        });
                    }

                    let mut expansion_specs = HashSet::new();
                    let actions =
                        <$provider as NaiveBottomUpActionProvider<$tgt>>::actions(&spec.0);
                    for action in actions {
                        if let Ok(lowered_impl) = action.apply(&spec) {
                            lowered_impl.visit_leaves(&mut |leaf| {
                                if let ImplNode::SpecApp(spec_app) = leaf {
                                    expansion_specs.insert(spec_app.0.clone());
                                }
                                true
                            });
                        }
                    }

                    prop_assert_eq!(query_specs, expansion_specs);
                }

                #[test]
                fn [< test_naivebottomupsolver_computes_same_decision_for_single_canonical_spec_ $suffix >](
                    spec in $crate::spec::arb_canonical_spec::<$tgt>(None, None),
                    visit_deps_reverse in proptest::prelude::any::<bool>(),
                ) {
                    use std::collections::{HashMap, HashSet};
                    use std::default::Default;
                    use std::rc::Rc;
                    use proptest::prelude::*;
                    use nonzero::nonzero as nz;
                    use $crate::cost::{CostIntensity, NormalizedCost};
                    use $crate::db::db_spec_bimap;
                    use $crate::memorylimits::MemVec;
                    use $crate::scheduling::{
                        ActionEncodeDecode, Cost, SpecGeometry, SpecGeometryRect, BottomUpSolver,
                        NaiveBottomUpSolver, DependencyRequest, compute_impl_cost
                    };
                    use $crate::search::ImplReducer;
                    use $crate::scheduling::shared_tests::internal::MockVisitUpdater;
                    use $crate::target::LEVEL_COUNT;

                    let mut solver = NaiveBottomUpSolver::<$tgt, $provider>::default();
                    let single_spec_geometry = SpecGeometry::single(
                        &spec, Rc::new(db_spec_bimap(false))
                    );
                    let mut request = solver.request(&single_spec_geometry);

                    let mut query_specs = Vec::new();
                    if let Some(queries_geometry) = request.queries() {
                        queries_geometry.iter().for_each(|rect| {
                            query_specs.extend(rect.iter_specs());
                        });
                    }
                    if visit_deps_reverse {
                        query_specs.reverse();
                    }

                    let mut updater = MockVisitUpdater {
                        goal: spec.clone(),
                        reducer: ImplReducer::new(1, vec![]),
                        goal_completed: false,
                    };
                    request.apply_no_dependency_updates(&spec, &mut updater);

                    // Check that everything in query_specs is unique.
                    let mut unique_specs = HashSet::new();
                    for query_spec in &query_specs {
                        prop_assert!(unique_specs.insert(query_spec));
                    }

                    // TODO: Make a test variant which doesn't just push dependency unit rects.
                    // TODO: Make a test variant where we feed k>1 costs
                    let mut small_db = HashMap::<Spec<$tgt>, NormalizedCost>::new();
                    for (idx, dependency) in query_specs.iter().enumerate() {
                        let mut dep_costs_vec = vec![];
                        if let Some(dep_ncost) = small_db.get(dependency) {
                            dep_costs_vec.push(dep_ncost.clone());
                        } else {
                            let ncost = NormalizedCost {
                                intensity: CostIntensity::new(
                                    (idx % 8).try_into().unwrap(),
                                    nz!(1u64),
                                ),
                                peaks: MemVec::new(
                                    [(u64::try_from(idx).unwrap() % 2); LEVEL_COUNT]
                                ),
                                depth: (idx % 3).try_into().unwrap(),
                            };
                            small_db.insert(dependency.clone(), ncost.clone());
                            dep_costs_vec.push(ncost);
                        }
                        request.visit_dependency(
                            &SpecGeometryRect::single(
                                dependency, Rc::clone(single_spec_geometry.bimap())
                            ),
                            &dep_costs_vec,
                            &mut updater,
                        );
                    }
                    prop_assert!(updater.goal_completed);

                    // Compare to the result of a simpler solver.
                    let mut reducer = ImplReducer::new(1, vec![]);
                    let actions_iter =
                        <$provider as NaiveBottomUpActionProvider<$tgt>>::actions(&spec.0);
                    for action in actions_iter {
                        let Ok(lowered_impl) = action.apply(&spec) else {
                            continue;
                        };
                        let mut action_subcosts: Vec<Cost> = vec![];
                        lowered_impl.visit_leaves(&mut |leaf| {
                            if let ImplNode::SpecApp(spec_app) = leaf {
                                let normalized = small_db
                                    .get(&spec_app.0)
                                    .expect("subspec should be in small_db")
                                    .clone();
                                action_subcosts.push(normalized.into_main_cost_for_volume(
                                    spec_app.0.0.volume(),
                                ));
                            }
                            true
                        });
                        let computed_ncost = NormalizedCost::new(compute_impl_cost(
                            &lowered_impl,
                            &mut action_subcosts.into_iter(),
                        ), spec.0.volume());
                        reducer.insert(action.encode(&spec), computed_ncost);
                    }

                    prop_assert_eq!(reducer.finalize(), updater.reducer.finalize())
                }

                #[test]
                fn [< test_naivebottomupsolver_yields_same_subspecs_for_single_canonical_spec_ $suffix >](
                    spec in $crate::spec::arb_canonical_spec::<$tgt>(None, None),
                ) {
                    use std::collections::HashSet;
                    use std::default::Default;
                    use std::rc::Rc;
                    use proptest::prelude::*;
                    use $crate::db::db_spec_bimap;
                    use $crate::scheduling::{
                        SpecGeometry, BottomUpSolver, NaiveBottomUpSolver, DependencyRequest
                    };

                    let mut solver = NaiveBottomUpSolver::<$tgt, $provider>::default();
                    let single_spec_geometry = SpecGeometry::single(
                        &spec, Rc::new(db_spec_bimap(false))
                    );
                    let request = solver.request(&single_spec_geometry);

                    let mut query_subspecs = HashSet::new();
                    if let Some(queries_geometry) = request.queries() {
                        queries_geometry.iter().for_each(|rect| {
                            query_subspecs.extend(rect.iter_specs());
                        });
                    }

                    let mut action_subspecs = HashSet::new();
                    for action in <$provider as NaiveBottomUpActionProvider<$tgt>>::actions(&spec.0) {
                        if let Ok(lowered_impl) = action.apply(&spec) {
                            lowered_impl.visit_leaves(&mut |leaf| {
                                if let ImplNode::SpecApp(spec_app) = leaf {
                                    action_subspecs.insert(spec_app.0.clone());
                                }
                                true
                            });
                        }
                    }

                    prop_assert_eq!(query_subspecs, action_subspecs);
                }
            } }
        };
    }
}
