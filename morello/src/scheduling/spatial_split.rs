use crate::common::Shape;
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::scheduling::{
    ActionT,
    tile_to_apply_err, ApplyError, NotApplicableReason,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::views::{Param, Tile, View, ViewE, ViewExt};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::iter;

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct SpatialSplit {}

impl<Tgt: Target> ActionT<Tgt> for SpatialSplit {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Conv { accum: conv_accum },
                spec_shape: _,
                dtypes,
            },
            _,
            serial_only,
        ) = logical_spec
        else {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Not a Conv",
            ))));
        };
        if !*conv_accum {
            // TODO: Use a more specific NotApplicableReason.
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Conv is not accumulating",
            ))));
        }
        let rank: u8 = operands[0].shape.len().try_into().unwrap();

        // We're going to introduce a Loop which traverses all the pixels over all spatial
        // dimensions, vectorizing across the batch, channels, and filters dimmensions. This
        // means the loop body/the sub-Spec will have its spatial dims.  dropped, even
        // though they are larger than one.

        // Make Tiles over the inputs. These tiles will range over the spatial dimensions,
        // so this amounts to keeping the outer two dimensions of each (batch and channels,
        // then filters count and channels, respectively) and replacing the rest with ones.
        let [outer_image_tile, outer_filters_tile] = [0, 1].map(|idx| {
            let shape = operands[idx].shape()[..2]
                .iter()
                .chain(iter::repeat(&nz!(1u32)).take((rank - 2).into()))
                .copied()
                .collect::<Shape>();
            let step_sizes = shape.clone();
            Tile::new(
                shape,
                step_sizes,
                Param::new(idx.try_into().unwrap(), operands[idx].clone()),
            )
        });
        let [outer_image_tile, outer_filters_tile] = [
            outer_image_tile.map_err(tile_to_apply_err)?,
            outer_filters_tile.map_err(tile_to_apply_err)?,
        ];

        // Make views over the tiles we'll pass to the body of the loop. These are
        // tiles reshaped to drop the size-one dimensions and, in the case of the filters
        // argument, transposed. The result is matrix multiply-able tensor views!
        let inner_image_tile = outer_image_tile.clone().squeeze_dims(2..rank);
        let inner_filters_tile = outer_filters_tile.clone().squeeze_dims(2..rank).transpose();

        let inner_output_view = Param::new(2, operands[2].clone()).squeeze_dims(2..rank);

        let body_spec = LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: true },
                spec_shape: operands[0].shape()[..2]
                    .iter()
                    .copied()
                    .chain(iter::once(operands[1].shape()[0]))
                    .collect(),
                dtypes: dtypes.clone(),
            },
            vec![
                inner_image_tile.spec().aux.clone(),
                inner_filters_tile.spec().aux.clone(),
                inner_output_view.spec().aux.clone(),
            ],
            *serial_only,
        );

        Ok(ImplNode::Loop(Loop {
            tiles: vec![
                LoopTile {
                    parameter_index: outer_image_tile.view.0,
                    axes: vec![7, 8, 0, 1],
                    tile: outer_image_tile.boxed_viewe(),
                },
                LoopTile {
                    parameter_index: outer_filters_tile.view.0,
                    axes: vec![9, 8, 0, 1],
                    tile: outer_filters_tile.boxed_viewe(),
                },
            ],
            body: Box::new(
                SpecApp::new(
                    Spec(body_spec, spec.1.clone()),
                    [
                        ViewE::from(inner_image_tile),
                        ViewE::from(inner_filters_tile),
                        ViewE::from(inner_output_view),
                    ],
                )
                .into(),
            ),
            parallel: false,
            spec: Some(spec.clone()),
        }))
    }
}
