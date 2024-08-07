use crate::common::DimSize;
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::views::{Param, Tensor, View, ViewE};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct ToSoftmaxParts<Tgt: Target> {
    pub max_level: Tgt::Level,
    pub max_layout: Layout,
    pub max_vector_size: Option<DimSize>,
    pub denominator_level: Tgt::Level,
    pub denominator_layout: Layout,
    pub denominator_vector_size: Option<DimSize>,
}

impl<Tgt: Target> ActionT<Tgt> for ToSoftmaxParts<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

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

        // Make tensor for storing the maximum value.
        let mut max_spec = TensorSpec {
            shape: spec_shape.clone(),
            dtype: dtypes[0],
            aux: TensorSpecAux {
                contig: self.max_layout.contiguous_full(),
                aligned: true,
                level: self.max_level,
                layout: self.max_layout.clone(),
                vector_size: self.max_vector_size,
            },
        };
        max_spec.shape[usize::from(*scan_dim)] = nz!(1u32);
        let max_tensor = Tensor::new(max_spec);

        // Make tensor for storing the denominator
        let mut denominator_spec = TensorSpec {
            shape: spec_shape.clone(),
            dtype: dtypes[0],
            aux: TensorSpecAux {
                contig: self.denominator_layout.contiguous_full(),
                aligned: true,
                level: self.denominator_level,
                layout: self.denominator_layout.clone(),
                vector_size: self.denominator_vector_size,
            },
        };
        denominator_spec.shape[usize::from(*scan_dim)] = nz!(1u32);
        let denominator_tensor = Tensor::new(denominator_spec);

        let new_buffer_consumption = Tgt::levels().map(|l| {
            let mut r = 0;
            if self.max_level == l {
                let max_spec = max_tensor.spec();
                r += u64::from(max_spec.dtype.size()) * u64::from(max_spec.volume().get());
            }
            if self.denominator_level == l {
                let denominator_spec = denominator_tensor.spec();
                r += u64::from(denominator_spec.dtype.size())
                    * u64::from(denominator_spec.volume().get());
            }
            r
        });
        let mut lowered_limits = MemoryLimits::Standard(match &spec.1 {
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

        // Make the SoftmaxDenominatorAndMax sub-Spec
        let denom_app: ImplNode<Tgt> = {
            let mut denom_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::SoftmaxDenominatorAndMax {
                            scan_dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: basics.spec_shape.clone(),
                        dtypes: vec![dtypes[0]; 3],
                    },
                    vec![
                        operands[0].aux.clone(),
                        max_tensor.spec().aux.clone(),
                        denominator_tensor.spec().aux.clone(),
                    ],
                    spec.0.serial_only(),
                ),
                lowered_limits.clone(),
            );
            denom_spec.canonicalize().unwrap();
            let app_args = vec![
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(max_tensor.clone()),
                ViewE::from(denominator_tensor.clone()),
            ];
            SpecApp::new(denom_spec, app_args).into()
        };

        // Make the SoftmaxComplete sub-Spec
        let complete_app: ImplNode<Tgt> = {
            let mut complete_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::SoftmaxComplete {
                            scan_dim: *scan_dim,
                        },
                        spec_shape: basics.spec_shape.clone(),
                        dtypes: vec![dtypes[0], dtypes[0], dtypes[0], dtypes[1]],
                    },
                    vec![
                        operands[0].aux.clone(),
                        max_tensor.spec().aux.clone(),
                        denominator_tensor.spec().aux.clone(),
                        operands[1].aux.clone(),
                    ],
                    spec.0.serial_only(),
                ),
                lowered_limits,
            );
            complete_spec.canonicalize().unwrap();
            let app_args = vec![
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(max_tensor.clone()),
                ViewE::from(denominator_tensor.clone()),
                ViewE::from(Param::new(1, operands[1].clone())),
            ];
            SpecApp::new(complete_spec, app_args).into()
        };

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![denom_app, complete_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(max_tensor), Rc::new(denominator_tensor)],
            }],
            parameters: operands,
            spec: Some(spec.clone()),
        }))
    }
}
