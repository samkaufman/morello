use crate::common::{DimSize, Dtype};
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::views::{Param, Tensor, View, ViewE};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToMaxAndUnscaled<Tgt: Target> {
    pub max_level: Tgt::Level,
    pub max_layout: Layout,
    pub max_vector_size: Option<DimSize>,
}

impl<Tgt: Target> ActionT<Tgt> for ToMaxAndUnscaled<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let LogicalSpec::Primitive(head, _, serial_only) = &spec.0 else {
            // TODO: Add a more specific NotApplicableReason
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndUnscaled only defined for Primitive",
            ))));
        };
        let PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, accum },
            spec_shape,
            dtypes,
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "ToMaxAndUnscaled is only defined for SoftmaxDenominatorAndUnscaled",
            ))));
        };
        if *accum {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Accumlating SoftmaxDenominatorAndUnscaled not supported",
            ))));
        }

        let operands = spec.0.parameters();

        // Make a Max. Parameter 0 is the input, and parameter 1 is the output.  The input,
        // of course, is provided externally, but the output is the intermediate max tensor.
        let max_tensor = scalar_tensor::<Tgt>(
            *scan_dim,
            spec_shape,
            dtypes[0],
            self.max_level,
            self.max_layout.clone(),
            self.max_vector_size,
        );
        let lowered_limits = child_limits(&spec.1, [max_tensor.spec()])?;

        let max_app = {
            let mut max_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Max {
                            dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: spec_shape.clone(),
                        dtypes: vec![dtypes[0]; 2],
                    },
                    vec![operands[0].aux.clone(), max_tensor.spec().aux.clone()],
                    *serial_only,
                ),
                lowered_limits.clone(),
            );
            max_spec.canonicalize().unwrap();
            let app_args: Vec<ViewE<Tgt>> = vec![
                Param::new(0, operands[0].clone()).into(),
                max_tensor.clone().into(),
            ];
            SpecApp::new(max_spec, app_args).into()
        };

        // Make SoftmaxDenominatorAndUnscaledFromMax.
        let complete_app = {
            let mut complete_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                            scan_dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: spec_shape.clone(),
                        dtypes: vec![dtypes[0], dtypes[0], dtypes[1], dtypes[2]],
                    },
                    vec![
                        operands[0].aux.clone(),
                        max_tensor.spec().aux.clone(),
                        operands[1].aux.clone(),
                        operands[2].aux.clone(),
                    ],
                    *serial_only,
                ),
                lowered_limits,
            );
            complete_spec.canonicalize().unwrap();
            let app_args: Vec<ViewE<Tgt>> = vec![
                Param::new(0, operands[0].clone()).into(),
                max_tensor.clone().into(),
                Param::new(1, operands[1].clone()).into(),
                Param::new(2, operands[2].clone()).into(),
            ];
            SpecApp::new(complete_spec, app_args).into()
        };

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![max_app, complete_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(max_tensor)],
            }],
            parameters: operands,
            spec: Some(spec.clone()),
        }))
    }
}

// TODO: This shares a lot of code with softmax_scalar_tensor.  Refactor to share code.
fn scalar_tensor<Tgt: Target>(
    scan_dim: u8,
    spec_shape: &[DimSize],
    dtype: Dtype,
    max_level: Tgt::Level,
    max_layout: Layout,
    max_vector_size: Option<DimSize>,
) -> Tensor<Tgt> {
    let mut max_spec = TensorSpec {
        shape: spec_shape.to_vec(),
        dtype,
        aux: TensorSpecAux {
            contig: max_layout.contiguous_full(),
            aligned: true,
            level: max_level,
            layout: max_layout,
            vector_size: max_vector_size,
        },
    };
    max_spec.shape[usize::from(scan_dim)] = nz!(1u32);
    Tensor::new(max_spec)
}

// TODO: Refactor to share code with softmax_child_limits.
fn child_limits<'a, Tgt: Target, I>(
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
        new_buffer_consumption[idx] +=
            u64::from(tensor_spec.volume().get()) * u64::from(tensor_spec.dtype.size());
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
    lowered_limits.discretize();
    Ok(lowered_limits)
}
