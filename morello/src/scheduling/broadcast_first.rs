use crate::common::{DimSize, Shape};
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::scheduling::{tile_to_apply_err, ActionT, ApplyError, NotApplicableReason};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::views::{Param, Tile, View, ViewE, ViewExt};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::iter;

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
            ..
        } = head
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "BroadcastFirst only defined for DivideVecScalar",
            ))));
        };

        // TODO: Allocate a broadcast destination.
        // TODO: Emit a "Broadcast" Spec of some kind.
        // TODO: Emit a DivideVec for the elementwise division.
        todo!()
    }
}
