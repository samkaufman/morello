use crate::common::{DimSize, Dtype, Shape};
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{
    Action, ActionT, ApplyError, NaiveBottomUpActionProvider, NaiveBottomUpSolver,
    NotApplicableReason,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::{self, TensorSpec, TensorSpecAux};
use crate::views::{Param, Tensor, View, ViewE};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::iter;
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToSoftmaxParts<Tgt: Target> {
    pub denominator_level: Tgt::Level,
    pub denominator_layout: Layout,
    pub denominator_vector_size: Option<DimSize>,
    pub exps_level: Tgt::Level,
    pub exps_layout: Layout,
    pub exps_vector_size: Option<DimSize>,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToSoftmaxPartsRecompute<Tgt: Target> {
    pub max_level: Tgt::Level,
    pub max_layout: Layout,
    pub max_vector_size: Option<DimSize>,
    pub denominator_level: Tgt::Level,
    pub denominator_layout: Layout,
    pub denominator_vector_size: Option<DimSize>,
}

#[derive(Default)]
pub struct ToSoftmaxPartsActionProvider<Tgt>(std::marker::PhantomData<Tgt>);

#[derive(Default)]
pub struct ToSoftmaxPartsRecomputeActionProvider<Tgt>(std::marker::PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for ToSoftmaxParts<Tgt> {
    type BSolver = NaiveBottomUpSolver<Tgt, ToSoftmaxPartsActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let LogicalSpec::Primitive(basics, _, _) = &spec.0 else {
            // TODO: Specialize NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Primitive",
            ))));
        };
        let PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim },
            spec_shape,
            dtypes,
        } = basics
        else {
            // TODO: Specialize NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Softmax",
            ))));
        };

        let operands = spec.0.parameters();

        let denominator_tensor = softmax_scalar_tensor::<Tgt>(
            *scan_dim,
            spec_shape,
            dtypes[0],
            self.denominator_level,
            self.denominator_layout.clone(),
            self.denominator_vector_size,
        )?;
        let mut exps_layout_contiguous = self.exps_layout.clone();
        exps_layout_contiguous.set_contiguous_full();
        let exps_tensor = Tensor::new(
            TensorSpec::<Tgt>::new_canon_checked(
                spec_shape.clone(),
                dtypes[0],
                self.exps_level,
                exps_layout_contiguous,
                self.exps_vector_size,
            )
            .map_err(|e| match e {
                tensorspec::CanonicalizeError::VectorSizeInvalid => {
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid)
                }
                _ => ApplyError::NotApplicable(NotApplicableReason::Other(None)),
            })?,
        );

        let lowered_limits =
            softmax_child_limits(&spec.1, [denominator_tensor.spec(), exps_tensor.spec()])?;

        let denom_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled {
                scan_dim: *scan_dim,
                accum: false,
            },
            [
                Param::new(0, operands[0].clone()).into(),
                denominator_tensor.clone().into(),
                exps_tensor.clone().into(),
            ],
            spec.0.serial_only(),
            lowered_limits.clone(),
        ));
        let scale_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::DivideVecScalar {
                scan_dim: *scan_dim,
            },
            [
                exps_tensor.clone().into(),
                denominator_tensor.clone().into(),
                Param::new(1, operands[1].clone()).into(),
            ],
            spec.0.serial_only(),
            lowered_limits.clone(),
        ));

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![denom_app, scale_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(denominator_tensor), Rc::new(exps_tensor)],
            }],
            spec: Some(spec.clone()),
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> ActionT<Tgt> for ToSoftmaxPartsRecompute<Tgt> {
    type BSolver = NaiveBottomUpSolver<Tgt, ToSoftmaxPartsRecomputeActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let LogicalSpec::Primitive(basics, _, _) = &spec.0 else {
            // TODO: Specialize NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Primitive",
            ))));
        };
        let PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim },
            spec_shape,
            dtypes,
        } = basics
        else {
            // TODO: Specialize NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Softmax",
            ))));
        };

        let operands = spec.0.parameters();

        // Make tensors for storing the maximum and denominator values.
        let max_tensor = softmax_scalar_tensor(
            *scan_dim,
            spec_shape,
            dtypes[0],
            self.max_level,
            self.max_layout.clone(),
            self.max_vector_size,
        )?;
        let denominator_tensor = softmax_scalar_tensor(
            *scan_dim,
            spec_shape,
            dtypes[0],
            self.denominator_level,
            self.denominator_layout.clone(),
            self.denominator_vector_size,
        )?;

        let lowered_limits =
            softmax_child_limits(&spec.1, [max_tensor.spec(), denominator_tensor.spec()])?;

        debug_assert_eq!(max_tensor.spec().dtype(), denominator_tensor.spec().dtype());
        let denom_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::SoftmaxDenominatorAndMax {
                scan_dim: *scan_dim,
            },
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(max_tensor.clone()),
                ViewE::from(denominator_tensor.clone()),
            ],
            spec.0.serial_only(),
            lowered_limits.clone(),
        ));
        let complete_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::SoftmaxComplete {
                scan_dim: *scan_dim,
            },
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(max_tensor.clone()),
                ViewE::from(denominator_tensor.clone()),
                ViewE::from(Param::new(1, operands[1].clone())),
            ],
            spec.0.serial_only(),
            lowered_limits,
        ));

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![denom_app, complete_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(max_tensor), Rc::new(denominator_tensor)],
            }],
            spec: Some(spec.clone()),
        }))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for ToSoftmaxPartsActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        // TODO: Return the actions!
        vec![]
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for ToSoftmaxPartsRecomputeActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        // TODO: Return the actions!
        vec![]
    }
}

fn softmax_scalar_tensor<Tgt: Target>(
    scan_dim: u8,
    spec_shape: &[DimSize],
    dtype: Dtype,
    max_level: Tgt::Level,
    max_layout: Layout,
    max_vector_size: Option<DimSize>,
) -> Result<Tensor<Tgt>, ApplyError> {
    debug_assert!(max_layout.is_fully_contiguous());
    let mut max_spec = TensorSpec {
        shape: Shape::from_slice(spec_shape),
        dtype,
        aux: TensorSpecAux {
            level: max_level,
            layout: max_layout,
            vector_size: max_vector_size,
        },
    };
    max_spec.shape[usize::from(scan_dim)] = nz!(1u32);
    max_spec.canonicalize().map_err(|e| match e {
        tensorspec::CanonicalizeError::VectorSizeInvalid => {
            ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid)
        }
        tensorspec::CanonicalizeError::VectorSizeVolumeIncompatible => {
            ApplyError::NotApplicable(NotApplicableReason::VectorSizeVolumeIncompatible)
        }
        tensorspec::CanonicalizeError::LayoutError(_) => {
            ApplyError::NotApplicable(NotApplicableReason::Other(None))
        }
    })?;
    Ok(Tensor::new(max_spec))
}

// TODO: Rename and use this elsewhere, such as in the Move planning.
fn softmax_child_limits<'a, Tgt: Target, I>(
    base_limits: &MemoryLimits,
    live_tensors: I,
) -> Result<MemoryLimits, ApplyError>
where
    I: IntoIterator<Item = &'a TensorSpec<Tgt>>,
{
    let mut new_buffer_consumption = [0u64; LEVEL_COUNT];
    for tensor_spec in live_tensors {
        let idx = Tgt::levels()
            .iter()
            .position(|l| l == &tensor_spec.level())
            .unwrap();
        new_buffer_consumption[idx] += tensor_spec.memory_units();
    }
    let mut lowered_limits = MemoryLimits::Standard(match base_limits.to_owned() {
        MemoryLimits::Standard(v) => v
            .clone()
            .checked_sub_snap_down(&new_buffer_consumption)
            .map_err(|oom_idx| {
                ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                    Tgt::levels()[oom_idx].to_string(),
                ))
            })?,
    });
    lowered_limits.discretize::<Tgt>();
    Ok(lowered_limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        emit_shared_naivebottomupactionprovider_tests,
        target::{ArmTarget, X86Target},
    };

    emit_shared_naivebottomupactionprovider_tests!(
        X86Target,
        ToSoftmaxPartsActionProvider<X86Target>,
        tosoftmaxparts_x86
    );
    emit_shared_naivebottomupactionprovider_tests!(
        ArmTarget,
        ToSoftmaxPartsActionProvider<ArmTarget>,
        tosoftmaxparts_arm
    );
    emit_shared_naivebottomupactionprovider_tests!(
        X86Target,
        ToSoftmaxPartsRecomputeActionProvider<X86Target>,
        tosoftmaxpartsrecompute_x86
    );
    emit_shared_naivebottomupactionprovider_tests!(
        ArmTarget,
        ToSoftmaxPartsRecomputeActionProvider<ArmTarget>,
        tosoftmaxpartsrecompute_arm
    );
}
