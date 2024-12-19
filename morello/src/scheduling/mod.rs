use crate::alignment::aligned_approx;
use crate::common::{DimSize, Dtype};
use crate::cost::{Cost, MainCost};
use crate::imp::loops::compute_loop_main_cost;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::utils::snap_memvec_up;
use crate::views::{Param, Tensor, TileError, View, ViewE};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use bufferize::Bufferize;
use moves::Move;
use select::Select;
use spatial_split::SpatialSplit;
use tiling::{Split, TileOut};
use to_accum::ToAccum;
use to_max_and_denom::ToMaxAndDenominator;
use to_softmax_parts::{ToSoftmaxParts, ToSoftmaxPartsRecompute};

pub mod bufferize;
pub mod moves;
pub mod select;
pub mod spatial_split;
pub mod tiling;
pub mod to_accum;
pub mod to_max_and_denom;
pub mod to_softmax_parts;

// TODO: Rename this. (`Action` should probably be X86-specific.)
#[enum_dispatch]
pub trait ActionT<Tgt: Target> {
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
    fn solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        self.apply_unchecked_canon(spec)
            .map(|applied| ActionSolver::Fallback(applied))
    }
}

/// A scheduling decision which can be applied to a Spec to produce an Impl.
///
/// [Action]s contain the minimal amount of information needed to distinguish a one scheduling
/// decision from another, which makes it appropriate for storing in a database so that the
/// corresponding Impl node can be computed given the Spec.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
#[serde(bound(
    deserialize = "Tgt::Kernel: Deserialize<'de>",
    serialize = "Tgt::Kernel: Serialize"
))]
#[enum_dispatch(ActionT<Tgt>)]
pub enum Action<Tgt: Target> {
    /// Tile the output tensor and its inputs to respect the updated inputs.
    TileOut(TileOut),
    Split(Split),
    /// Move a tensor to a different memory level, layout, and/or dtype.
    Move(Move<Tgt>),
    /// Allocate an output tensor, a Zero sub-Spec, and an accumulating variant of the receiver.
    ToAccum(ToAccum),
    ToSoftmaxParts(ToSoftmaxParts<Tgt>),
    ToSoftmaxPartsRecompute(ToSoftmaxPartsRecompute<Tgt>),
    /// Rewrites a SoftmaxDenominatorAndMax into a Max followed by SoftmaxDenominator.
    ToMaxAndDenominator(ToMaxAndDenominator),
    Bufferize(Bufferize<Tgt>),
    SpatialSplit(SpatialSplit),
    // TODO: Remove 'force' bool from Select
    #[serde(bound(
        deserialize = "Select<Tgt>: Deserialize<'de>",
        serialize = "Select<Tgt>: Serialize",
    ))]
    Select(Select<Tgt>),
}

#[derive(Debug)]
pub enum ActionSolver<Tgt: Target> {
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

impl<Tgt: Target> ActionSolver<Tgt> {
    pub fn subspecs(&self) -> impl Iterator<Item = Spec<Tgt>> {
        match self {
            ActionSolver::PrimitiveTileOut {
                outer_spec: _,
                body_spec,
            } => {
                // TODO: Avoid this clone
                vec![body_spec.clone()].into_iter()
            }
            ActionSolver::Move {
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
            ActionSolver::Fallback(partial_impl) => {
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
            ActionSolver::PrimitiveTileOut {
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
            ActionSolver::Move {
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
            ActionSolver::Fallback(partial_impl) => {
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
                .initial_accumulating_value_for_output(parameter_idx.into())
                .expect("output parameter should be compatible with ToAccum");
            let output = spec.0.parameter(parameter_idx.into());
            let subspec = LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Fill {
                        value: accum_initial_value,
                    },
                    spec_shape: output.shape.clone(),
                    dtypes: vec![output.dtype],
                },
                vec![output.aux.clone()],
                spec.0.serial_only(),
            );
            let mut spec = Spec(subspec, spec.1.clone());
            spec.canonicalize()
                .expect("ToAccum's introduced Zeroes should be canonicalizable");
            Some(SpecApp::new(spec, [ViewE::from(Param::new(parameter_idx, output))]).into())
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
        .enumerate()
        .any(|(dim, dim_size)| *dim_size > current_out_shape[dim])
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
        1 + outer_operand_auxes.len() - outer_components.last().unwrap().typ.input_count();
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

    // TODO: Wrap in SpecApp
    let mut params = vec![];
    params.reserve_exact(outer_input_count + 2);
    params.extend((0..outer_input_count).map(|i| {
        ViewE::from(Param::new(i.try_into().unwrap(), {
            outer_compose.0.parameter(i)
        }))
    }));
    params.insert(insertion_point, ViewE::from(intermediate_tensor));
    params.push(ViewE::from(Param::new(
        (parent_operand_auxes.len() - 1).try_into().unwrap(),
        outer_compose.0.unique_output().unwrap(),
    )));
    SpecApp::new(outer_compose, params)
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
    use crate::{memorylimits::MemVec, spec::arb_canonical_spec, target::X86Target};
    use proptest::prelude::*;

    proptest! {
        // TODO: Add an ARM variant
        #[test]
        fn test_fast_path_is_equivalent_to_slow(spec in arb_canonical_spec::<X86Target>(None, None)) {
            for action in X86Target::actions(&spec.0, None) {
                match (action.solver(&spec), action.apply(&spec)) {
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
    }
}
