use crate::alignment::aligned_approx;
use crate::common::{Contig, DimSize, Shape};
use crate::cost::NormalizedCost;
use crate::grid::linear::BimapSInt;
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::{Layout, PhysDim};
use crate::scheduling::{
    check_tile_out_applies, tile_to_apply_err, Action, ActionT, ActionTopDownSolver, ApplyError,
    BottomUpSolver, DependencyRequest, NaiveBottomUpActionProvider, NaiveBottomUpSolver,
    NotApplicableReason, SpecGeometry, SpecGeometryRect, VisitUpdater,
};
use crate::scheduling_sugar::SchedulingSugar;
use crate::spec::{
    self, FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec,
    DimAssociation,
};
use crate::target::common_actions::{split_actions, tile_out_actions};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::tiling::Tiling;
use crate::views::{Param, Tile, View, ViewE};
use itertools::{izip, Either, Itertools};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::iter;
use std::rc::Rc;

/// Whether `tile_out` actions should tile in all dimensions per Spec.
pub(crate) const MULTI_DIM_TILING: bool = true;

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum TileOut {
    SingleLoop {
        dim: u8,
        size: DimSize,
        parallel: bool,
    },
    MultiLoop {
        output_shape: Shape,
        parallel: bool,
    },
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct Split {
    pub k: DimSize,
}

#[derive(Default)]
pub struct TileOutSolver<Tgt>(std::marker::PhantomData<Tgt>);

#[derive(Default)]
pub struct SplitSolver<Tgt: Target> {
    requests: Vec<(Spec<Tgt>, Spec<Tgt>)>,
}

#[derive(Default)]
pub struct TileOutActionProvider<Tgt: Target>(std::marker::PhantomData<Tgt>);

#[derive(Default)]
pub struct SplitActionProvider<Tgt: Target>(std::marker::PhantomData<Tgt>);

#[derive(Debug)]
pub struct TileOutSolverRequest<Tgt: Target>(SpecGeometry<Tgt>);

#[derive(Debug)]
pub struct SplitSolverRequest<Tgt: Target>(SpecGeometry<Tgt>);

impl TileOut {
    fn tiled_output_shape(
        &self,
        untiled_output_shape: &[DimSize],
    ) -> Either<Vec<DimSize>, &[DimSize]> {
        match self {
            TileOut::SingleLoop {
                dim,
                size,
                parallel: _,
            } => {
                let mut output_shape_owned = untiled_output_shape.to_vec();
                output_shape_owned[*dim as usize] = *size;
                Either::Left(output_shape_owned)
            }
            TileOut::MultiLoop {
                output_shape,
                parallel: _,
            } => Either::Right(output_shape),
        }
    }

    fn parallel(&self) -> bool {
        match self {
            TileOut::SingleLoop { parallel, .. } | TileOut::MultiLoop { parallel, .. } => *parallel,
        }
    }
}

impl<Tgt: Target> ActionT<Tgt> for TileOut {
    // TODO: Instead use `type BSolver = TileOutSolver<Tgt>;`
    type BSolver = NaiveBottomUpSolver<Tgt, TileOutActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        // TODO: Replace SoftmaxDenominatorAndUnscaledFromMax case with more general tiling.
        if let Some(daufm_result) = tile_out_daufm(spec, self) {
            return daufm_result;
        };

        let Some(output_idx) = logical_spec.unique_output_index() else {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::MultipleOutputs,
            ));
        };

        let current_output = &operands[output_idx];
        let current_out_shape = current_output.shape();
        let rank = current_out_shape.len();

        let output_shape = self.tiled_output_shape(current_out_shape);
        let parallel = self.parallel();

        // TODO: Move assertions into solver() as well.
        if parallel && logical_spec.serial_only() {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::ParallelPrevented,
            ));
        }
        assert_eq!(
            output_shape.len(),
            current_out_shape.len(),
            "Expected {} dimensions; got {}",
            current_out_shape.len(),
            output_shape.len()
        );
        check_tile_out_applies(current_out_shape, &output_shape, current_output, parallel)?;

        // Tiling happens in three steps:
        // 1. Construct the simple tile corresponding to the new output shape.
        let output_idx_u8 = u8::try_from(output_idx).unwrap();
        let smaller_output_tiling = Tiling::new_simple(output_shape.either_into());
        let smaller_output = LoopTile {
            parameter_index: output_idx_u8,
            axes: (0..u8::try_from(rank).unwrap()).collect(),
            tile: smaller_output_tiling
                .apply(Param::new(output_idx_u8, current_output.clone()))
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        };

        // 2. Construct tilings which respect the data deps. of the new output tile.
        let Some(updated_input_tilings) =
            logical_spec.input_tilings_for_tile_out(&smaller_output_tiling)
        else {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Tiling doesn't apply to logical Spec",
            ))));
        };

        // 3. Reify the tilings into Tiles we'll store with this action. Tiles objects track
        // the index and shape of the Impl parameter being tiled.
        let mut next_fresh_loop_dim = u8::try_from(rank).unwrap();
        let mut new_tiles: Vec<LoopTile<Tgt>> = vec![];
        for (operand_idx, (original_input, (updated_input_tiling, updated_input_axes))) in
            operands.iter().zip(updated_input_tilings.0).enumerate()
        {
            if operand_idx == output_idx {
                continue;
            }

            let tiling_shape = updated_input_tiling.shape();
            if !original_input.is_valid_tile_shape(tiling_shape, parallel) {
                return Err(ApplyError::NotApplicable(
                    NotApplicableReason::TileShapeInvalid,
                ));
            }

            // Compute loop dimension names for the tile. Any axis which is None
            // is given a fresh integer identifier, otherwise is given the
            // identifier of the corresponding dimension in the output.
            let axes = updated_input_axes
                .iter()
                .map(|b| {
                    match *b {
                        Some(output_dim) => {
                            // It's correct to use the output dim. here because
                            // we earlier initialized the output axes to be
                            // simply their indices.
                            output_dim
                        }
                        None => {
                            let fresh_dim = next_fresh_loop_dim;
                            next_fresh_loop_dim += 1;
                            fresh_dim
                        }
                    }
                })
                .collect();

            if original_input.shape() != &tiling_shape[..] {
                let operand_idx_u8 = u8::try_from(operand_idx).unwrap();
                new_tiles.push(LoopTile {
                    parameter_index: operand_idx_u8,
                    axes,
                    tile: updated_input_tiling
                        .apply(Param::new(operand_idx_u8, original_input.clone()))
                        .map(|v| v.boxed_viewe())
                        .map_err(tile_to_apply_err)?,
                });
            }
        }
        new_tiles.push(smaller_output);

        tiling_apply_epilogue(spec, operands, new_tiles, parallel)
    }

    fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionTopDownSolver<Tgt>, ApplyError> {
        match &spec.0 {
            LogicalSpec::Primitive(basics, ..) => {
                // TODO: Replace SoftmaxDenominatorAndUnscaledFromMax case with more general tiling.
                if tile_out_daufm(spec, self).is_some() {
                    todo!("Implement solver for SoftmaxDenominatorAndUnscaledFromMax");
                };

                let Some(output_tensor) = spec.0.unique_output() else {
                    return Err(ApplyError::NotApplicable(
                        NotApplicableReason::MultipleOutputs,
                    ));
                };
                let untiled_output_shape = output_tensor.shape();
                let tile_shape = self.tiled_output_shape(untiled_output_shape);
                let parallel = self.parallel();

                check_tile_out_applies(
                    untiled_output_shape,
                    &tile_shape,
                    &output_tensor,
                    parallel,
                )?;

                match basics.typ {
                    PrimitiveSpecType::Matmul { .. } => {
                        return Ok(ActionTopDownSolver::PrimitiveTileOut {
                            outer_spec: spec.clone(),
                            body_spec: ActionTopDownSolver::tiled_subspec_fast(
                                [(0, 0), (1, 1), (3, 2)].into_iter(),
                                spec,
                                &tile_shape,
                                parallel,
                            )?,
                        });
                    }
                    PrimitiveSpecType::Move
                    | PrimitiveSpecType::Fill {
                        value: FillValue::Zero,
                    } => {
                        let rank = basics.spec_shape.len();
                        return Ok(ActionTopDownSolver::PrimitiveTileOut {
                            outer_spec: spec.clone(),
                            body_spec: ActionTopDownSolver::tiled_subspec_fast(
                                (0..rank).map(|i| (i, i)),
                                spec,
                                &tile_shape,
                                parallel,
                            )?,
                        });
                    }
                    _ => {}
                }
            }
            LogicalSpec::Compose { .. } => {}
        };

        self.apply_unchecked_canon(spec)
            .map(|applied| ActionTopDownSolver::Fallback(applied))
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

impl<Tgt: Target> ActionT<Tgt> for Split {
    type BSolver = NaiveBottomUpSolver<Tgt, SplitActionProvider<Tgt>>;
    type BSolverIter = iter::Once<Self::BSolver>;

    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        match logical_spec {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => {
                match typ {
                    PrimitiveSpecType::Matmul { accum: false } => {
                        // TODO: Should return an error instead?
                        panic!("Can only split an accumulating Matmul");
                    }
                    PrimitiveSpecType::Matmul { accum: true } => {
                        let [lhs, rhs] = &operands[..2] else {
                            panic!();
                        };
                        assert!(
                            self.k < lhs.shape()[2],
                            "Cannot split to k={} when inner dim. is not larger (it is {})",
                            self.k,
                            lhs.shape()[2]
                        );

                        if lhs.shape()[2].get() % self.k.get() != 0 {
                            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(
                                Some("Original size is not a multiple of split size"),
                            )));
                        }

                        let tiles = vec![
                            LoopTile {
                                parameter_index: 0,
                                axes: vec![0, 1, 2],
                                tile: Tile::new(
                                    vec![lhs.shape()[0], lhs.shape()[1], self.k],
                                    vec![lhs.shape()[0], lhs.shape()[1], self.k],
                                    Param::new(0, lhs.clone()),
                                )
                                .map(|v| v.boxed_viewe())
                                .map_err(tile_to_apply_err)?,
                            },
                            LoopTile {
                                parameter_index: 1,
                                axes: vec![0, 2, 3],
                                tile: Tile::new(
                                    vec![rhs.shape()[0], self.k, rhs.shape()[2]],
                                    vec![rhs.shape()[0], self.k, rhs.shape()[2]],
                                    Param::new(1, rhs.clone()),
                                )
                                .map(|v| v.boxed_viewe())
                                .map_err(tile_to_apply_err)?,
                            },
                        ];

                        tiling_apply_epilogue(spec, operands, tiles, false)
                    }
                    PrimitiveSpecType::Max { dim, accum: true }
                    | PrimitiveSpecType::SoftmaxDenominator {
                        scan_dim: dim,
                        accum: true,
                    } => {
                        let in_tensor_spec = &operands[0];
                        assert!(
                            self.k < in_tensor_spec.shape()[usize::from(*dim)],
                            "Cannot split to k={} when inner dim. is not larger (it is {})",
                            self.k,
                            in_tensor_spec.shape()[usize::from(*dim)]
                        );
                        if in_tensor_spec.shape()[usize::from(*dim)].get() % self.k.get() != 0 {
                            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(
                                Some("Original size is not a multiple of split size"),
                            )));
                        }

                        let mut split_shape = in_tensor_spec.shape().to_vec();
                        split_shape[usize::from(*dim)] = self.k;

                        let tiles = vec![LoopTile {
                            parameter_index: 0,
                            axes: (0..u8::try_from(in_tensor_spec.shape().len()).unwrap())
                                .collect(),
                            tile: Tile::new(
                                split_shape.clone(),
                                split_shape,
                                Param::new(0, in_tensor_spec.clone()),
                            )
                            .map(|t| t.boxed_viewe())
                            .map_err(tile_to_apply_err)?,
                        }];
                        tiling_apply_epilogue(spec, operands, tiles, false)
                    }
                    _ => unimplemented!("Split not implemented for {:?}", typ),
                }
            }
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only,
            } if matches!(
                components[0],
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: true },
                    ..
                }
            ) =>
            {
                let Some(output_idx) = logical_spec.unique_output_index() else {
                    panic!("Compose should have a unique output");
                };

                let [old_b, old_m, old_k, old_n] = &components[0].spec_shape[..] else {
                    todo!();
                };

                // Build a Loop out of a Compose with the head component removed.
                // TODO: Can we use a helper method to avoid dupe'ing with Bufferize?
                let compose_tail = Spec(
                    if components.len() == 2 {
                        LogicalSpec::Primitive(
                            components[1].clone(),
                            operand_auxes[1..].to_vec(),
                            *serial_only,
                        )
                    } else {
                        debug_assert!(components.len() > 2);
                        LogicalSpec::Compose {
                            components: components[1..].to_vec(),
                            operand_auxes: operand_auxes[1..].to_vec(),
                            serial_only: *serial_only,
                        }
                    },
                    spec.1.clone(),
                );
                debug_assert!(compose_tail.is_canonical());

                // Match the tail Compose's Loop, which we're about to mutate.
                let tail_compose_output_idx = compose_tail.0.unique_output_index().unwrap();
                let ImplNode::Loop(Loop {
                    mut tiles,
                    mut body,
                    parallel: _,
                    spec: _,
                }) = compose_tail.tile_out(&[old_b.get(), old_m.get(), self.k.get()])
                else {
                    unreachable!();
                };
                let ImplNode::SpecApp(SpecApp(app_spec, app_args)) = body.as_mut() else {
                    unreachable!();
                };

                // Remove the LoopTile for the output
                let tail_output_tile = tiles.remove(
                    tiles
                        .iter()
                        .position(|tile| {
                            usize::from(tile.parameter_index) == tail_compose_output_idx
                        })
                        .expect("tail Compose should have a loop over output"),
                );

                // Increment the Param indices for the LoopTiles in `tiles`.
                for tile in &mut tiles {
                    tile.parameter_index += 1;
                    if let ViewE::Param(p) = &mut *tile.tile.view {
                        p.0 += 1;
                    }
                }

                // Add a LoopTile for the new rhs argument on the head Compose.
                debug_assert_ne!(tail_output_tile.axes[0], 255);
                debug_assert_ne!(tail_output_tile.axes[2], 255);
                let new_head_rhs_looptile = LoopTile {
                    parameter_index: 0,
                    axes: vec![tail_output_tile.axes[0], tail_output_tile.axes[2], 255], // TODO: Replace 255
                    tile: Tiling::new_simple(vec![*old_b, self.k, *old_n])
                        .apply(Param::new(
                            0,
                            TensorSpec::new_noncanon_with_aux(
                                vec![*old_b, *old_k, *old_n],
                                components[0].dtypes[1],
                                operand_auxes[0].clone(),
                            ),
                        ))
                        .map(|v| v.boxed_viewe())
                        .map_err(tile_to_apply_err)?,
                };
                tiles.insert(0, new_head_rhs_looptile);
                let new_head_rhs_looptile = &tiles[0];

                // Add an argument for the new head rhs
                let mut new_app_args = Vec::<ViewE<Tgt>>::with_capacity(app_args.len() + 1);
                debug_assert!(!spec.0.parameter_is_output(0));
                new_app_args.push(Param::new(0, new_head_rhs_looptile.tile.spec().clone()).into());
                // Add Params with indices incremented to account for the
                // additional argument
                for (idx, a) in app_args.iter().enumerate() {
                    if compose_tail.0.parameter_is_output(idx) {
                        continue;
                    }
                    let Some(Param(param_idx, param_spec, ..)) = a.to_param() else {
                        unreachable!();
                    };
                    new_app_args.push(Param::new(param_idx + 1, param_spec.clone()).into());
                }
                // Add an output argument with the Spec's target output
                new_app_args.insert(
                    output_idx,
                    Param::new(
                        new_app_args.len().try_into().unwrap(),
                        logical_spec.parameter(output_idx).clone(),
                    )
                    .into(),
                );
                debug_assert_eq!(new_app_args.len(), spec.0.parameters().len());
                *app_args = new_app_args;

                // Replace the application Spec with a shrunken target
                let new_operands = app_args
                    .iter()
                    .map(|a| a.spec().clone())
                    .collect::<Vec<_>>();
                *app_spec = spec.clone();
                app_spec.0.replace_io(&new_operands);
                debug_assert!(app_spec.is_canonical());

                // TODO: Rather than returning a new struct, return the original Loop.
                debug_assert_eq!(body.parameters().cloned().collect::<Vec<_>>(), new_operands);
                Ok(ImplNode::Loop(Loop {
                    tiles,
                    body,
                    parallel: false,
                    spec: Some(spec.clone()),
                }))
            }
            LogicalSpec::Compose { .. } => todo!(),
        }
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
    }
}

fn tile_out_daufm<Tgt: Target>(
    spec: &Spec<Tgt>,
    tileout: &TileOut,
) -> Option<Result<ImplNode<Tgt>, ApplyError>> {
    let LogicalSpec::Primitive(
        PrimitiveBasics {
            typ:
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                    accum: true,
                    scan_dim,
                },
            ..
        },
        ..,
    ) = &spec.0
    else {
        return None;
    };

    let operands = spec.0.parameters();

    let current_output = &operands[3];
    let current_out_shape = current_output.shape();
    let rank = u8::try_from(current_out_shape.len()).unwrap();

    let new_output_shape = tileout.tiled_output_shape(current_out_shape);
    if tileout.parallel() {
        todo!("Support parallel loops over non-scan dimension");
    }

    assert_eq!(
        new_output_shape.len(),
        current_out_shape.len(),
        "Expected {} dimensions; got {}",
        current_out_shape.len(),
        new_output_shape.len()
    );
    if let Err(e) = check_tile_out_applies(
        current_out_shape,
        &new_output_shape,
        current_output,
        tileout.parallel(),
    ) {
        return Some(Err(e));
    }

    // Tiling happens in three steps:
    // 1. Construct the simple tile corresponding to the new output shape.
    let smaller_output_tiling = Tiling::new_simple(new_output_shape.clone().either_into());
    let new_smaller_output_tile =
        match smaller_output_tiling.apply(Param::new(3, current_output.clone())) {
            Ok(t) => t,
            Err(err) => return Some(Err(tile_to_apply_err(err))),
        };
    let smaller_output = LoopTile {
        parameter_index: 3,
        axes: (0..rank).collect(),
        tile: new_smaller_output_tile.boxed_viewe(),
    };

    // 2. Construct tilings which respect the data deps. of the new output tile. These tilings
    //    cover the first two parameters, which are inputs, as well as the third parameter,
    //    which is the denominator output.
    let mut onescan_shape: Shape = new_output_shape.clone().either_into();
    onescan_shape[usize::from(*scan_dim)] = nz!(1u32);
    let updated_input_tilings = [
        Tiling::new_simple(new_output_shape.either_into()), // input
        Tiling::new_simple(onescan_shape.clone()),          // max
        Tiling::new_simple(onescan_shape),                  // output: denominator
    ];

    // 3. Reify the tilings into Tiles we'll store with this action. Tiles objects track
    // the index and shape of the Impl parameter being tiled.
    let mut new_tiles: Vec<LoopTile<Tgt>> = vec![];
    for (operand_idx, (original_input, updated_input_tiling)) in
        operands.iter().zip(updated_input_tilings).enumerate()
    {
        let tiling_shape = updated_input_tiling.shape();
        if !original_input.is_valid_tile_shape(tiling_shape, false) {
            return Some(Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeInvalid,
            )));
        }

        if original_input.shape() != &tiling_shape[..] {
            let operand_idx_u8 = u8::try_from(operand_idx).unwrap();
            let tile = match updated_input_tiling
                .apply(Param::new(operand_idx_u8, original_input.clone()))
            {
                Ok(t) => t.boxed_viewe(),
                Err(err) => return Some(Err(tile_to_apply_err(err))),
            };
            new_tiles.push(LoopTile {
                parameter_index: operand_idx_u8,
                axes: (0..rank).collect(),
                tile,
            });
        }
    }
    new_tiles.push(smaller_output);

    Some(tiling_apply_epilogue(spec, operands, new_tiles, false))
}

impl<Tgt: Target> BottomUpSolver for TileOutSolver<Tgt> {
    type Tgt = Tgt;
    type Request = TileOutSolverRequest<Tgt>;

    fn request(&mut self, dependents: &SpecGeometry<Tgt>) -> Self::Request {
        // TODO: If top is parallel and bottom is serial-only, split into two two ranges.
        //       Right now, if they differ, we'll over-visit the serial-only children.

        // TODO: slice every rect into-same-except-shape rather than expecting exactly one output
        let dependencies = SpecGeometry::new(Rc::clone(&dependents.1));

        dependents.iter().for_each(|rect| {
            let parameter_shapes = rect.top().0.parameter_shapes();
            let high_output_shape = &parameter_shapes[rect.top().0.unique_output_index().unwrap()];
            let output_rank = high_output_shape.len();
            let output_rank_u8 = u8::try_from(output_rank).unwrap();

            if MULTI_DIM_TILING || !rect.bottom().0.serial_only() || !rect.top().0.serial_only() {
                todo!();
            } else {
                todo!();
            }
        });

        TileOutSolverRequest(dependencies)
    }
}

impl<Tgt: Target> DependencyRequest for TileOutSolverRequest<Tgt> {
    type Tgt = Tgt;

    fn queries(&self) -> Option<&SpecGeometry<Tgt>> {
        Some(&self.0)
    }

    fn visit_dependency<U>(
        &mut self,
        _rectangle: &SpecGeometryRect<Tgt>,
        _cost: &[NormalizedCost],
        _updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        // TODO: Have a faster check, especially for Compose.
        let output_idx = spec.0.unique_output_index().unwrap();
        let output_shape = spec.0.parameter_shape(output_idx);
        match &spec.0 {
            LogicalSpec::Primitive(..) => {
                if output_shape.iter().any(|d| d == &nz!(1u32)) {
                    updater.complete_spec(spec);
                }
            }
            LogicalSpec::Compose { .. } => {
                if MULTI_DIM_TILING || !spec.0.serial_only() {
                    let tileout = TileOut::MultiLoop {
                        output_shape: vec![nz!(1u32); output_shape.len()],
                        parallel: false,
                    };
                    if let Err(ApplyError::NotApplicable(
                        NotApplicableReason::TileShapeMatchesOriginal,
                    )) = tileout.apply_unchecked_canon(spec)
                    {
                        updater.complete_spec(spec);
                    }
                } else {
                    todo!()
                }
            }
        };
    }
}

impl<Tgt: Target> BottomUpSolver for SplitSolver<Tgt> {
    type Tgt = Tgt;
    type Request = SplitSolverRequest<Tgt>;

    fn request(&mut self, dependents: &SpecGeometry<Tgt>) -> Self::Request {
        todo!()
    }
}

impl<Tgt: Target> DependencyRequest for SplitSolverRequest<Tgt> {
    type Tgt = Tgt;

    fn queries(&self) -> Option<&SpecGeometry<Tgt>> {
        Some(&self.0)
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        let LogicalSpec::Primitive(ref basics, _, _) = spec.0 else {
            updater.complete_spec(spec);
            return;
        };
        if !matches!(basics.typ, PrimitiveSpecType::Matmul { accum: true }) {
            updater.complete_spec(spec);
        }
    }

    fn visit_dependency<U>(
        &mut self,
        _rectangle: &SpecGeometryRect<Tgt>,
        _cost: &[NormalizedCost],
        _updater: &mut U,
    ) where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for TileOutActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        tile_out_actions(logical_spec).collect()
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for SplitActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        split_actions(logical_spec).collect()
    }
}

/// Build a [Loop] over the given tiles with a shrunken [SpecApp] body.
fn tiling_apply_epilogue<Tgt: Target>(
    spec: &Spec<Tgt>,
    mut operands: Vec<TensorSpec<Tgt>>,
    tiles: Vec<LoopTile<Tgt>>,
    parallel: bool,
) -> Result<ImplNode<Tgt>, ApplyError> {
    // Modify `operands` TensorSpecs' shapes to accord with the tilings.
    for loop_tile in &tiles {
        let ref_op = &mut operands[usize::from(loop_tile.parameter_index)];
        let aligned =
            aligned_approx(loop_tile.tile.shape(), loop_tile.tile.step_sizes(), ref_op).unwrap();
        ref_op.shrink(loop_tile.tile.shape(), aligned).unwrap();
    }

    // Update the Spec with these new operands, as well as its serial_only flag.
    let mut inner_spec = spec.0.clone();
    inner_spec.replace_io(&operands);
    inner_spec.set_serial_only(inner_spec.serial_only() || parallel);
    inner_spec
        .canonicalize()
        .map_err(|canon_error| match canon_error {
            spec::CanonicalizeError::TensorSpecAuxCanonicalizeError(_) => {
                ApplyError::NotApplicable(NotApplicableReason::TileShapeInvalid)
            }
            spec::CanonicalizeError::SideEffectingComponent => {
                unreachable!("Compose-to-tile should not have side-effecting components")
            }
        })?;

    // Return a Loop containing the Spec as its body.
    let body = Box::new({
        let spec = Spec(inner_spec, spec.1.clone());
        let arguments = spec
            .0
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, o)| ViewE::from(Param::new(i.try_into().unwrap(), o)))
            .collect();
        SpecApp(spec, arguments).into()
    });
    Ok(ImplNode::Loop(Loop {
        tiles,
        body,
        parallel,
        spec: Some(spec.clone()),
    }))
}

// TODO: Shouldn't be pub
/// Compute the layout and contig for all arguments for all tile sizes of a [Spec].
///
/// This requires both `spec_shape` and tensor shapes because `spec_shape` is referenced
/// when computing cutoffs' offsets from a [DimAssociation].
pub(crate) fn compute_layout_contig_rects<'a>(
    spec_shape: &[DimSize],
    tensor_descriptions: &'a [(&[DimSize], &Layout, Contig, &[DimAssociation])],
    fixed_spec_dimensions: &[usize],
) -> impl Iterator<Item = (Vec<BimapSInt>, Vec<BimapSInt>, Vec<(Layout, Contig)>)> + 'a {
    let multidim_cutoffs =
        combine_contig_cutoffs(spec_shape, tensor_descriptions, fixed_spec_dimensions);
    compute_layout_contigs_for_cutoff_rects(tensor_descriptions, multidim_cutoffs)
}

/// Map the rectangles defined by `multidim_cutoffs` to the [Layout] and [Contig] for each argument
/// of all tiled [Spec]s in that rectangle.
fn compute_layout_contigs_for_cutoff_rects<'a>(
    tensor_descriptions: &'a [(&[DimSize], &Layout, Contig, &[DimAssociation])],
    multidim_cutoffs: Vec<Vec<u32>>,
) -> impl Iterator<Item = (Vec<BimapSInt>, Vec<BimapSInt>, Vec<(Layout, Contig)>)> + 'a {
    debug_assert!(multidim_cutoffs
        .iter()
        .all(|dim_cutoffs| dim_cutoffs.is_sorted()));

    multidim_cutoffs
        .into_iter()
        .map(|mc| mc.into_iter().tuple_windows::<(_, _)>())
        .multi_cartesian_product()
        .map(move |dims| {
            // `dims` is a Vec of axis-aligned ranges describing a single cutoff-bounded rectangle.
            let mut value = vec![];
            value.reserve_exact(tensor_descriptions.len());
            for (parent_shape, layout, contig, dim_associations) in tensor_descriptions {
                // Choose a representative point inside the tensor parameter.
                // TODO: Cache or reorganize the algorithm to visit each tensor rectangle just once.
                let representative_tile_shape = dim_associations
                    .iter()
                    .map(|association| match association {
                        DimAssociation::SpecDim(associated_dim) => {
                            DimSize::new(dims[*associated_dim].1 - 1).unwrap()
                        }
                        DimAssociation::SpecDimDifference(lhs_dim, rhs_dim, shift) => {
                            let lhs = dims[*lhs_dim].1 - 1;
                            let rhs = dims[*rhs_dim].1 - 1;
                            DimSize::new(shift.get() + lhs - rhs).unwrap()
                        }
                        DimAssociation::Constant(size) => *size,
                    })
                    .collect::<Vec<_>>();
                match layout.update_for_tiling(parent_shape, &representative_tile_shape, *contig) {
                    Ok((new_layout, new_contig)) => value.push((new_layout, new_contig)),
                    Err(_) => break,
                };
            }
            if value.len() != tensor_descriptions.len() {
                value.clear();
            }

            let bottom = dims
                .iter()
                .map(|(b, _)| BimapSInt::from(*b))
                .collect::<Vec<_>>();
            let top = dims
                .iter()
                .map(|(_, t)| BimapSInt::from(*t - 1))
                .collect::<Vec<_>>();
            (bottom, top, value)
        })
}

/// Compute cutoffs for all Spec dimensions.
///
/// fixed_spec_dimensions: Spec dimensions which are given no cutoffs and, for the purposes of
///   projecting [DimAssociation::SpecDimDifference], are treated as constant. The result will
///   contain no cutoffs for these dimensions (the [Vec]s will be empty).
///
/// Internally, this combines the results of [compute_contig_cutoffs_single_tensor] for each tensor.
/// Combination involves merging cutoffs from multiple tensor dimensions corresponding to the same
/// Spec dimension, as well as shifting them according to the coordination relation between Spec
/// and tensor dimensions (as defined by the [DimAssociation]s in `tensor_descriptions`).
fn combine_contig_cutoffs(
    spec_shape: &[DimSize],
    tensor_descriptions: &[(&[DimSize], &Layout, Contig, &[DimAssociation])],
    fixed_spec_dimensions: &[usize],
) -> Vec<Vec<u32>> {
    // TODO: Inline axis_grouped_cutoffs if possible

    // Compute cutoffs for each tensor and group them by Spec axis.
    let mut axis_grouped_cutoffs = vec![vec![]; spec_shape.len()];
    for (original_shape, layout, _, dim_associations) in tensor_descriptions {
        let dim_cutoffs_vec = compute_contig_cutoffs_single_tensor(original_shape, layout);
        assert_eq!(dim_cutoffs_vec.len(), dim_associations.len());
        for (i, association) in dim_associations.iter().enumerate() {
            for cutoff in &dim_cutoffs_vec[i] {
                match association {
                    DimAssociation::SpecDim(associated_dim) => {
                        if !fixed_spec_dimensions.contains(associated_dim) {
                            axis_grouped_cutoffs[*associated_dim].push(*cutoff);
                        }
                    }
                    DimAssociation::SpecDimDifference(lhs_dim, rhs_dim, shift) => {
                        debug_assert_ne!(lhs_dim, rhs_dim);
                        let mut lhs_fixed = false;
                        let mut rhs_fixed = false;
                        for dim in fixed_spec_dimensions {
                            if dim == lhs_dim {
                                lhs_fixed = true;
                            }
                            if dim == rhs_dim {
                                rhs_fixed = true;
                            }
                        }
                        if lhs_fixed && rhs_fixed {
                            continue;
                        }

                        if lhs_fixed {
                            let lhs = spec_shape[*lhs_dim].get();
                            let shifted_cutoff = *cutoff + lhs - shift.get();
                            axis_grouped_cutoffs[*rhs_dim].push(shifted_cutoff);
                        } else if rhs_fixed {
                            let rhs = spec_shape[*rhs_dim].get();
                            let shifted_cutoff = *cutoff + rhs - shift.get();
                            axis_grouped_cutoffs[*lhs_dim].push(shifted_cutoff);
                        } else {
                            panic!("one dimension must be fixed");
                        }
                    }
                    DimAssociation::Constant(_) => {}
                }
            }
        }
    }

    // TODO: Use `merge` instead to avoid a sort (dim_cutoffs should be sorted).
    for unsorted_cutoffs in &mut axis_grouped_cutoffs {
        unsorted_cutoffs.sort_unstable();
        unsorted_cutoffs.dedup();
    }

    debug_assert_eq!(axis_grouped_cutoffs.len(), spec_shape.len());
    axis_grouped_cutoffs
}

/// Computes the cutoffs for each dimension of a tensor in ascending order.
///
/// This is a helper function for [combine_contig_cutoffs].
fn compute_contig_cutoffs_single_tensor(
    original_shape: &[DimSize],
    layout: &Layout,
) -> Vec<Vec<u32>> {
    // We always want to break out index=0 (size=1), so we'll initialize with [1, 2].
    let mut result = vec![vec![1, 2]; original_shape.len()];

    // Find the smallest Packed/OddEven multiple for every logical dim. that has one.
    // We'll use this next to break out tile sizes which are multiples of that size.
    // This accomodates our restriction that we can't split Packed/OddEven physical
    // dimensions with tiling.
    let mut frequencies: Vec<Option<DimSize>> = vec![None; original_shape.len()];
    for (logical_dim, physical_dim) in layout.0.iter().rev() {
        match physical_dim {
            PhysDim::Packed(size) | PhysDim::OddEven(size) => {
                if let Some(existing_size) = frequencies[usize::from(*logical_dim)] {
                    let min_size = existing_size.min(*size);
                    debug_assert!(
                        existing_size.get() % min_size.get() == 0
                            || size.get() % min_size.get() == 0,
                        "One size should be a multiple of the other"
                    );
                    frequencies[usize::from(*logical_dim)] = Some(min_size);
                } else {
                    frequencies[usize::from(*logical_dim)] = Some(*size);
                }
            }
            PhysDim::Dynamic => {}
        }
    }

    for (dim_cutoffs, freq_opt, size) in izip!(result.iter_mut(), &frequencies, original_shape) {
        if let Some(freq) = freq_opt {
            let mut multiple = (*freq).max(nz!(2u32));
            while multiple < *size {
                dim_cutoffs.push(multiple.get());
                dim_cutoffs.push(multiple.get() + 1);
                multiple = DimSize::new(multiple.get() * 2).unwrap();
            }
        };
        if size.get() > 1 {
            dim_cutoffs.push(size.get());
            dim_cutoffs.push(size.get() + 1);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::db_spec_bimap;
    use crate::grid::canon::CanonicalBimap;
    use crate::grid::general::BiMap;
    use crate::imp::visit_leaves;
    use crate::layout::{arb_shape_and_same_rank_layout, row_major, Layout, PhysDim};
    use crate::scheduling::{Action, SpecGeometry};
    use crate::spec::arb_canonical_spec;
    use crate::target::{ArmTarget, CpuMemoryLevel, X86Target};

    use crate::{lspec, shape};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use proptest::{prop_assert_eq, proptest};
    use std::collections::HashSet;

    /// Test that a TileOut::SingleLoop with a non-multiple size fails to apply.
    ///
    /// These aren't implemented yet in Impl, so the action should fail.
    #[test]
    fn test_non_multiple_tile_out_single_returns_error() {
        shared_test_non_multiple_tiling_returns_error(Action::TileOut(TileOut::SingleLoop {
            dim: 0,
            size: nz!(3u32),
            parallel: false,
        }))
    }

    /// Test that a TileOut::MultiLoop with a non-multiple size fails to apply.
    ///
    /// These aren't implemented yet in Impl, so the action should fail.
    #[test]
    fn test_non_multiple_tile_out_multi_returns_error() {
        shared_test_non_multiple_tiling_returns_error(Action::TileOut(TileOut::MultiLoop {
            output_shape: shape![1, 4, 6],
            parallel: false,
        }))
    }

    /// Test that a TileOut::Split with a non-multiple size fails to apply.
    ///
    /// These aren't implemented yet in Impl, so the action should fail.
    #[test]
    fn test_non_multiple_split_returns_error() {
        shared_test_non_multiple_tiling_returns_error(Action::Split(Split { k: nz!(6u32) }))
    }

    #[test]
    fn test_compute_layout_contig_rects_1() {
        let rm = row_major(2);
        let spec_shape = shape![6, 6];
        let da = [DimAssociation::SpecDim(0), DimAssociation::SpecDim(1)];
        let tensor_descriptions = vec![(&spec_shape[..], &rm, rm.contiguous_full(), &da[..])];
        let fixed_spec_dimensions = &[];

        // TODO: Construct an R-Tree rather than testing for the specific output rectangles.
        let expected = HashSet::from([
            (
                (vec![1, 1]).clone(),
                (vec![1, 1]).clone(),
                vec![(rm.clone(), rm.contiguous_full())],
            ),
            (vec![2, 1], vec![5, 1], vec![(rm.clone(), 1)]),
            (
                vec![1, 2],
                vec![1, 5],
                vec![(rm.clone(), rm.contiguous_full())],
            ),
            (
                vec![2, 6],
                vec![5, 6],
                vec![(rm.clone(), rm.contiguous_full())],
            ),
            (vec![6, 1], vec![6, 1], vec![(rm.clone(), 1)]),
            (vec![6, 2], vec![6, 5], vec![(rm.clone(), 1)]),
            (vec![2, 2], vec![5, 5], vec![(rm.clone(), 1)]),
            (
                vec![1, 6],
                vec![1, 6],
                vec![(rm.clone(), rm.contiguous_full())],
            ),
            (
                vec![6, 6],
                vec![6, 6],
                vec![(rm.clone(), rm.contiguous_full())],
            ),
        ]);

        let result: HashSet<_> =
            compute_layout_contig_rects(&spec_shape, &tensor_descriptions, fixed_spec_dimensions)
                .collect();
        assert_eq!(result, expected);
    }

    proptest! {
        #[test]
        fn test_tileoutsolver_requests_match_apply_deps_x86(
            spec in arb_canonical_spec::<X86Target>(None, None),
        ) {
            shared_test_tileoutsolver_requests_match_apply_deps(spec)?;
        }

        #[test]
        fn test_tileoutsolver_requests_match_apply_deps_arm(
            spec in arb_canonical_spec::<ArmTarget>(None, None),
        ) {
            shared_test_tileoutsolver_requests_match_apply_deps(spec)?;
        }

        #[test]
        fn test_tiling_splitting_introduces_consistent_dependencies(
            spec in arb_canonical_spec::<X86Target>(None, None)
                // TODO: Remove this prop_filter
                .prop_filter("spec should be matmul, fill, etc.", |s| {
                    matches!(s.0, LogicalSpec::Primitive(PrimitiveBasics {
                        typ: PrimitiveSpecType::Matmul { .. } |
                             PrimitiveSpecType::Conv { .. } |
                             PrimitiveSpecType::Fill { .. } |
                             PrimitiveSpecType::SoftmaxComplete { .. } |
                             PrimitiveSpecType::SoftmaxDenominator { .. } |
                             PrimitiveSpecType::Max { .. },
                        .. }, ..))
                })
        ) {
            // TODO: Add some sort of test for parallel?

            let output_idx = spec.0.unique_output_index().unwrap();
            let tensor_shapes = spec.0.parameter_shapes();
            let output_shape = &tensor_shapes[output_idx];
            let parameter_count = spec.0.operand_count();
            let associations = (0..parameter_count)
                .map(|i| {
                    match &spec.0 {
                        LogicalSpec::Primitive(primitive_basics, ..) =>
                            primitive_basics.spec_dim_associations(i.try_into().unwrap()),
                        LogicalSpec::Compose { .. } => unreachable!(),
                    }
                })
                .collect::<Vec<_>>();
            let tensor_descriptions = (0..parameter_count).map(|i| {
                let layout = spec.0.parameter_layout(i);
                let contig = spec.0.parameter_contiguous_abs(i);
                (tensor_shapes[i].as_slice(), layout, contig, associations[i].as_slice())
            }).collect::<Vec<_>>();

            let LogicalSpec::Primitive(primitive_basics, _, _) = &spec.0 else {
                unreachable!();
            };

            let mut actions = vec![];
            for (dim, &output_size) in output_shape.iter().enumerate() {
                for size in 1..output_size.get() {
                    if output_size.get() % size != 0 {
                        continue;
                    }
                    actions.push(Action::TileOut(TileOut::SingleLoop {
                        dim: u8::try_from(dim).unwrap(),
                        size: DimSize::new(size).unwrap(),
                        parallel: false,
                    }));
                }
            }
            for sizes in output_shape.iter().map(|d| {
                (1..=d.get()).filter(|x| d.get() % x == 0)
            }).multi_cartesian_product() {
                if sizes.iter().zip(output_shape).all(|(s, d)| *s == d.get()) {
                    continue;
                }
                actions.push(Action::TileOut(TileOut::MultiLoop {
                    output_shape: sizes.iter().map(|&s| DimSize::new(s).unwrap()).collect(),
                    parallel: false,
                }));
            }
            if let LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) = &spec.0 {
                if matches!(typ, PrimitiveSpecType::Matmul { accum: true }) {
                    for k in 1..tensor_shapes[0][1].get() {
                        if tensor_shapes[0][1].get() % k != 0 {
                            continue;
                        }
                        actions.push(Action::Split(Split { k: DimSize::new(k).unwrap() }));
                    }
                }
            }

            let rects: Vec<_> =
                compute_layout_contig_rects(&primitive_basics.spec_shape, &tensor_descriptions, &[]).collect();

            for action in actions {
                match action.apply(&spec) {
                    Ok(applied_impl) => {
                        let mut subspec = None;
                        visit_leaves(&applied_impl, &mut |leaf| {
                            if let ImplNode::SpecApp(spec_app) = leaf {
                                assert!(subspec.is_none(), "Only one sub-Spec should be present");
                                subspec = Some(spec_app.0.clone());
                            }
                            true
                        });
                        let subspec = subspec.unwrap();

                        let spec_shape = match &subspec.0 {
                            LogicalSpec::Primitive(basics, _, _) => &basics.spec_shape,
                            LogicalSpec::Compose { .. } => todo!(),
                        };
                        let point = spec_shape.iter().map(|&d| BimapSInt::from(d.get())).collect::<Vec<_>>();
                        let located = locate_rect_containing_point(rects.iter(), &point);
                        prop_assert!(located.is_some(), "R-Tree should contain the point {:?}", point);
                        prop_assert_eq!(located.unwrap().len(), subspec.0.operand_count());
                        for (idx, (layout, contig)) in located.unwrap().iter().enumerate() {
                            let expected_layout = subspec.0.parameter_layout(idx);
                            let expected_contig = subspec.0.parameter_contiguous_abs(idx);
                            prop_assert_eq!(layout, expected_layout);
                            prop_assert_eq!(contig, &expected_contig);
                        }
                    },
                    Err(err) => {
                        let mut spec_shape = match &spec.0 {
                            LogicalSpec::Primitive(basics, _, _) => basics.spec_shape.to_vec(),
                            LogicalSpec::Compose { .. } => todo!(),
                        };
                        match &action {
                            Action::TileOut(TileOut::SingleLoop { dim, size, .. }) => {
                                match associations[output_idx][usize::from(*dim)] {
                                    DimAssociation::SpecDim(d) => {
                                        spec_shape[d] = *size;
                                    }
                                    DimAssociation::SpecDimDifference(..) => todo!(),
                                    DimAssociation::Constant(_) => {}
                                }
                            }
                            Action::TileOut(TileOut::MultiLoop { output_shape, .. }) => {
                                for (dim, &size) in output_shape.iter().enumerate() {
                                    match associations[output_idx][dim] {
                                        DimAssociation::SpecDim(d) => {
                                            spec_shape[d] = size;
                                        }
                                    DimAssociation::SpecDimDifference(..) => todo!(),
                                        DimAssociation::Constant(_) => {}
                                    }
                                }
                            }
                            Action::Split(Split { k }) => {
                                let DimAssociation::SpecDim(spec_dim) = associations[0][1] else {
                                    unreachable!();
                                };
                                spec_shape[spec_dim] = *k;
                            }
                            _ => {}
                        }

                        let point = spec_shape.iter().map(|d| BimapSInt::from(d.get())).collect::<Vec<_>>();
                        let located = locate_rect_containing_point(rects.iter(), &point);
                        prop_assert!(located.is_some(), "R-Tree should contain the point {:?}", point);
                        prop_assert!(
                            located.unwrap().is_empty(),
                            "Point {point:?} should have an empty value; action: {action:?} from {spec} produced {err:?}",
                        );
                    },
                };
            }
        }

        #[test]
        fn test_compute_contig_cutoffs_single_tensor_results_are_sorted(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let dim_cutoffs_vec = compute_contig_cutoffs_single_tensor(&shape, &layout);
            for dim_cutoffs in &dim_cutoffs_vec {
                prop_assert!(dim_cutoffs.is_sorted(), "{dim_cutoffs:?} is not sorted");
            }
        }

        #[test]
        fn test_compute_contig_cutoffs_single_tensor_breaks_out_first_index(
            (shape, layout) in arb_shape_and_same_rank_layout()
        ) {
            let dim_cutoffs_vec = compute_contig_cutoffs_single_tensor(&shape, &layout);
            for dim_cutoffs in &dim_cutoffs_vec {
                prop_assert_eq!(&dim_cutoffs[..2], &[1, 2]);
            }
        }
    }

    fn shared_test_non_multiple_tiling_returns_error(action: Action<X86Target>) {
        let bcm_layout = Layout::new(vec![
            (0, PhysDim::Dynamic),
            (2, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
        ]);
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [4, 8, 8, 8],
            (f32, CpuMemoryLevel::GL, bcm_layout),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let spec = Spec(logical_spec, X86Target::max_mem());
        let application = action.apply(&spec);
        assert!(
            matches!(
                application,
                Err(ApplyError::NotApplicable(NotApplicableReason::Other(_))),
            ),
            "expected NotApplicable(Other) but got {application:?}",
        );
    }

    /// Test that TileOutSolver's dependencies for single Specs match `apply`.
    ///
    /// Use the [FilesDatabase]'s [BiMap] to determine which Specs are members of the returned
    /// dependency request.
    fn shared_test_tileoutsolver_requests_match_apply_deps<Tgt>(
        spec: Spec<Tgt>,
    ) -> Result<(), proptest::test_runner::TestCaseError>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
        let mut solver = TileOutSolver::<Tgt>::default();
        let single_spec_set = SpecGeometry::new(Rc::new(db_spec_bimap(false)));
        let mut solver_dependencies = HashSet::new();
        if let Some(geometry) = solver
            .request(&SpecGeometry::single(
                &spec,
                Rc::clone(single_spec_set.bimap()),
            ))
            .queries()
        {
            geometry.iter().for_each(|rect| {
                rect.iter_specs().for_each(|s| {
                    solver_dependencies.insert(s);
                });
            });
        }
        let apply_dependencies: HashSet<Spec<Tgt>> = TileOutActionProvider::<Tgt>::actions(&spec.0)
            .into_iter()
            .flat_map(|action| {
                action
                    .apply(&spec)
                    .map(|expanded_impl| {
                        let mut subspecs = vec![];
                        visit_leaves(&expanded_impl, &mut |leaf| {
                            if let ImplNode::SpecApp(spec_app) = leaf {
                                subspecs.push(spec_app.0.clone());
                            }
                            true
                        });
                        subspecs
                    })
                    .unwrap_or_default()
            })
            .collect::<HashSet<_>>();
        prop_assert_eq!(apply_dependencies, solver_dependencies);
        Ok(())
    }

    fn locate_rect_containing_point<'a, R, V>(rects: R, point: &[BimapSInt]) -> Option<&'a V>
    where
        R: Iterator<Item = &'a (Vec<BimapSInt>, Vec<BimapSInt>, V)> + 'a,
        V: 'a,
    {
        for (bottom, top, value) in rects {
            if bottom
                .iter()
                .zip(top.iter())
                .zip(point.iter())
                .all(|((&b, &t), &p)| b <= p && p <= t)
            {
                return Some(value);
            }
        }
        None
    }
}
