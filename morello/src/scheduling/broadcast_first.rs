use crate::common::DimSize;
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::scheduling::{ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, Tensor, ViewE};
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

        let broadcast_destination = Tensor::new(TensorSpec::<Tgt>::new_canon(
            head.spec_shape.to_vec(),
            dtypes[1],
            self.broadcast_layout.contiguous_full(),
            true,
            self.broadcast_level,
            self.broadcast_layout.clone(),
            self.broadcast_vector_size,
        ));
        let broadcast_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::Broadcast { dim: *scan_dim },
            [
                ViewE::from(Param::new(1, operands[1].clone())),
                ViewE::from(broadcast_destination.clone()),
            ],
            spec.0.serial_only(),
            spec.1.clone(),
        ));
        let dividevec_app = ImplNode::from(SpecApp::new_primitive_app(
            PrimitiveSpecType::DivideVec,
            [
                ViewE::from(Param::new(0, operands[0].clone())),
                ViewE::from(broadcast_destination.clone()),
                ViewE::from(Param::new(2, operands[2].clone())),
            ],
            spec.0.serial_only(),
            spec.1.clone(),
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
