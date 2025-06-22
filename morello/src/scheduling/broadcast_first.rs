use crate::common::DimSize;
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::{CanonicalizeError, TensorSpec};
use crate::views::{Param, Tensor, View, ViewE};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct BroadcastFirst<Tgt: Target> {
    pub broadcast_level: Tgt::Level,
    pub broadcast_layout: Layout,
    pub broadcast_vector_size: Option<DimSize>,
}

impl<Tgt: Target> ActionT<Tgt> for BroadcastFirst<Tgt> {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let head = match &spec.0 {
            LogicalSpec::Primitive(basics, ..) => basics,
            LogicalSpec::Compose { components: _, .. } => todo!("Add support for Compose"),
        };

        let PrimitiveBasics {
            typ: PrimitiveSpecType::DivideVecScalar { scan_dim },
            dtypes,
            ..
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "BroadcastFirst only defined for DivideVecScalar",
            ))));
        };

        let operands = spec.0.parameters();

        let broadcast_destination = Tensor::new(
            TensorSpec::<Tgt>::new_canon_checked(
                head.spec_shape.clone(),
                dtypes[1],
                true,
                self.broadcast_level,
                self.broadcast_layout.clone(),
                self.broadcast_vector_size,
            )
            .map_err(|e| match e {
                CanonicalizeError::VectorSizeInvalid => {
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeInvalid)
                }
                CanonicalizeError::VectorSizeVolumeIncompatible => {
                    ApplyError::NotApplicable(NotApplicableReason::VectorSizeVolumeIncompatible)
                }
                _ => ApplyError::NotApplicable(NotApplicableReason::Other(None)),
            })?,
        );

        // Compute the memory limits for the new children.
        let new_limits = {
            let intermediate_mem_consumed = Tgt::levels().map(|l| {
                if self.broadcast_level == l {
                    broadcast_destination.spec().memory_units()
                } else {
                    0u64
                }
            });

            let mut m = MemoryLimits::Standard(match &spec.1 {
                MemoryLimits::Standard(v) => v
                    .clone()
                    .checked_sub_snap_down(&intermediate_mem_consumed)
                    .map_err(|oom_idx| {
                        ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                            Tgt::levels()[oom_idx].to_string(),
                        ))
                    })?,
            });
            m.discretize::<Tgt>();
            m
        };
        let broadcast_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::Broadcast { dim: *scan_dim },
            [
                ViewE::from(Param::new(1, operands[1].clone())),
                ViewE::from(broadcast_destination.clone()),
            ],
            spec.0.serial_only(),
            new_limits.clone(),
        ));
        let dividevec_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::DivideVec,
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(broadcast_destination.clone()),
                ViewE::from(Param::new(2, operands[2].clone())),
            ],
            spec.0.serial_only(),
            new_limits,
        ));
        Ok(ImplNode::Pipeline(Pipeline {
            stages: vec![broadcast_app, dividevec_app],
            wirings: vec![StageWiring {
                intermediate_tensors: vec![Rc::new(broadcast_destination)],
            }],
            parameters: operands,
            spec: Some(spec.clone()),
        }))
    }
}
