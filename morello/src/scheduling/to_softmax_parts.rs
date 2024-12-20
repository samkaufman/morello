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

impl<Tgt: Target> ActionT<Tgt> for ToSoftmaxParts<Tgt> {
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
        );
        let exps_tensor = Tensor::new(TensorSpec::<Tgt> {
            shape: spec_shape.clone(),
            dtype: dtypes[0],
            aux: TensorSpecAux {
                contig: self.exps_layout.contiguous_full(),
                aligned: true,
                level: self.exps_level,
                layout: self.exps_layout.clone(),
                vector_size: self.exps_vector_size,
            },
        });

        let lowered_limits =
            softmax_child_limits(&spec.1, [denominator_tensor.spec(), exps_tensor.spec()])?;

        let denom_app: ImplNode<Tgt> = {
            let mut denom_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaled {
                            scan_dim: *scan_dim,
                            accum: false,
                        },
                        spec_shape: basics.spec_shape.clone(),
                        dtypes: vec![dtypes[0]; 3],
                    },
                    vec![
                        operands[0].aux.clone(),
                        denominator_tensor.spec().aux.clone(),
                        exps_tensor.spec().aux.clone(),
                    ],
                    spec.0.serial_only(),
                ),
                lowered_limits.clone(),
            );
            denom_spec.canonicalize().unwrap();
            let app_args = vec![
                Param::new(0, operands[0].clone()).into(),
                denominator_tensor.clone().into(),
                exps_tensor.clone().into(),
            ];
            SpecApp::new(denom_spec, app_args).into()
        };
        let scale_app: ImplNode<Tgt> = {
            let mut scale_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::DivideVecScalar {
                            scan_dim: *scan_dim,
                        },
                        spec_shape: basics.spec_shape.clone(),
                        dtypes: vec![dtypes[0]; 3],
                    },
                    vec![
                        exps_tensor.spec().aux.clone(),
                        denominator_tensor.spec().aux.clone(),
                    ],
                    spec.0.serial_only(),
                ),
                lowered_limits.clone(),
            );
            scale_spec.canonicalize().unwrap();
            let app_args = vec![
                denominator_tensor.clone().into(),
                exps_tensor.clone().into(),
                Param::new(1, operands[1].clone()).into(),
            ];
            SpecApp::new(scale_spec, app_args).into()
        };

        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![denom_app, scale_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(denominator_tensor), Rc::new(exps_tensor)],
            }],
            parameters: operands,
            spec: Some(spec.clone()),
        }))
    }
}

impl<Tgt: Target> ActionT<Tgt> for ToSoftmaxPartsRecompute<Tgt> {
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
        );
        let denominator_tensor = softmax_scalar_tensor(
            *scan_dim,
            spec_shape,
            dtypes[0],
            self.denominator_level,
            self.denominator_layout.clone(),
            self.denominator_vector_size,
        );

        let lowered_limits =
            softmax_child_limits(&spec.1, [max_tensor.spec(), denominator_tensor.spec()])?;

        let denom_app = softmax_denominatorandmax_subapp(
            *scan_dim,
            &basics.spec_shape,
            spec.0.serial_only(),
            lowered_limits.clone(),
            operands[0].clone(),
            &max_tensor,
            &denominator_tensor,
        );
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

fn softmax_scalar_tensor<Tgt: Target>(
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

fn softmax_denominatorandmax_subapp<Tgt: Target>(
    scan_dim: u8,
    spec_shape: &[DimSize],
    serial_only: bool,
    lowered_limits: MemoryLimits,
    input_tensorspec: TensorSpec<Tgt>,
    max_tensor: &Tensor<Tgt>,
    denominator_tensor: &Tensor<Tgt>,
) -> ImplNode<Tgt> {
    debug_assert_eq!(max_tensor.spec().dtype(), denominator_tensor.spec().dtype());
    let dtype = max_tensor.spec().dtype();
    let mut denom_spec = Spec(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim },
                spec_shape: spec_shape.to_vec(),
                dtypes: vec![dtype; 3],
            },
            vec![
                input_tensorspec.aux.clone(),
                max_tensor.spec().aux.clone(),
                denominator_tensor.spec().aux.clone(),
            ],
            serial_only,
        ),
        lowered_limits,
    );
    denom_spec.canonicalize().unwrap();
    let app_args = vec![
        ViewE::from(Param::new(0, input_tensorspec)),
        ViewE::from(max_tensor.clone()),
        ViewE::from(denominator_tensor.clone()),
    ];
    SpecApp::new(denom_spec, app_args).into()
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
