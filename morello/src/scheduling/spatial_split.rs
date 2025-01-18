use crate::common::Shape;
use crate::cost::NormalizedCost;
use crate::grid::general::BiMap;
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::scheduling::{
    tile_to_apply_err, ActionT, ApplyError, BottomUpSolver, DbKey, DependencyRequest,
    NotApplicableReason, VisitUpdater,
};
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::views::{Param, Tile, View, ViewE, ViewExt};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::iter;
use std::marker::PhantomData;

#[derive(Default, Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct SpatialSplit;

#[derive(Debug)]
pub struct SpatialSplitSolver<Tgt>(PhantomData<Tgt>);

#[derive(Debug)]
pub struct SpatialSplitSolverRequest<Tgt>(PhantomData<Tgt>);

impl<Tgt: Target> ActionT<Tgt> for SpatialSplit {
    type BSolver = SpatialSplitSolver<Tgt>;
    type BSolverIter = iter::Once<Self::BSolver>;

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
                .chain(iter::repeat_n(&nz!(1u32), (rank - 2).into()))
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
        let inner_image_tile = outer_image_tile.clone().squeeze_dims(2..rank).one_prefix();
        let inner_filters_tile = outer_filters_tile
            .clone()
            .squeeze_dims(2..rank)
            .transpose()
            .one_prefix();
        let inner_output_view = Param::new(2, operands[2].clone())
            .squeeze_dims(2..rank)
            .one_prefix();

        let body_spec = LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: true },
                spec_shape: vec![
                    nz!(1u32),
                    operands[0].shape()[0],
                    operands[0].shape()[1],
                    operands[1].shape()[0],
                ],
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

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(SpatialSplitSolver(PhantomData))
    }
}

impl<Tgt: Target> BottomUpSolver for SpatialSplitSolver<Tgt> {
    type Tgt = Tgt;
    type Request = SpatialSplitSolverRequest<Tgt>;

    fn dependencies_for_range<B>(
        &mut self,
        _bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Self::Request
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
    {
        if spec_is_conv(low) || spec_is_conv(high) {
            todo!()
        } else {
            SpatialSplitSolverRequest(PhantomData)
        }
    }
}

impl<Tgt: Target> DependencyRequest for SpatialSplitSolverRequest<Tgt> {
    type Tgt = Tgt;

    fn requested_ranges(&self) -> &[(Spec<Self::Tgt>, Spec<Self::Tgt>)] {
        &[]
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        if !spec_is_conv(spec) {
            updater.complete_spec(spec);
        } else {
            todo!("Handle any non-splittable Convs as well")
        }
    }

    fn visit_dependency<U>(&mut self, _spec: &Spec<Tgt>, _cost: &[NormalizedCost], _updater: &mut U)
    where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }
}

fn spec_is_conv<Tgt: Target>(spec: &Spec<Tgt>) -> bool {
    matches!(
        &spec.0,
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Conv { .. },
                ..
            },
            _,
            _
        )
    )
}
