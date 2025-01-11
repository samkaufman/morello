use crate::alignment::aligned_approx;
use crate::common::{DimSize, Shape};
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::scheduling::{
    check_tile_out_applies, tile_to_apply_err, ActionSolver, ActionT, ApplyError,
    NotApplicableReason,
};
use crate::scheduling_sugar::SchedulingSugar;
use crate::spec::{self, FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::tiling::Tiling;
use crate::views::{Param, Tile, View, ViewE};
use itertools::Either;
use serde::{Deserialize, Serialize};

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

impl TileOut {
    fn parallel(&self) -> bool {
        match self {
            TileOut::SingleLoop { parallel, .. } => *parallel,
            TileOut::MultiLoop { parallel, .. } => *parallel,
        }
    }
}

impl<Tgt: Target> ActionT<Tgt> for TileOut {
    fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

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

        loop_spec_with_shrunken_tiles(operands, new_tiles, logical_spec, parallel, spec)
    }

    fn solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        match &spec.0 {
            LogicalSpec::Primitive(basics, ..) => {
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
                        return Ok(ActionSolver::PrimitiveTileOut {
                            outer_spec: spec.clone(),
                            body_spec: ActionSolver::tiled_subspec_fast(
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
                        return Ok(ActionSolver::PrimitiveTileOut {
                            outer_spec: spec.clone(),
                            body_spec: ActionSolver::tiled_subspec_fast(
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
            .map(|applied| ActionSolver::Fallback(applied))
    }
}

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
}

impl<Tgt: Target> ActionT<Tgt> for Split {
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

                        loop_spec_with_shrunken_tiles(operands, tiles, logical_spec, false, spec)
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
                        loop_spec_with_shrunken_tiles(operands, tiles, logical_spec, false, spec)
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
}

fn loop_spec_with_shrunken_tiles<Tgt: Target>(
    mut operands: Vec<TensorSpec<Tgt>>,
    tiles: Vec<LoopTile<Tgt>>,
    logical_spec: &LogicalSpec<Tgt>,
    parallel: bool,
    spec: &Spec<Tgt>,
) -> Result<ImplNode<Tgt>, ApplyError> {
    // Modify `operands` TensorSpecs' shapes to accord with the tilings.
    for loop_tile in &tiles {
        let ref_op = &mut operands[usize::from(loop_tile.parameter_index)];
        let aligned =
            aligned_approx(loop_tile.tile.shape(), loop_tile.tile.step_sizes(), ref_op).unwrap();
        ref_op.shrink(loop_tile.tile.shape(), aligned).unwrap();
    }

    // Update the Spec with these new operands, as well as its serial_only flag.
    let mut inner_spec = logical_spec.clone();
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
    let body = Box::new(
        {
            let spec = Spec(inner_spec, spec.1.clone());
            let arguments = spec
                .0
                .parameters()
                .into_iter()
                .enumerate()
                .map(|(i, o)| ViewE::from(Param::new(i.try_into().unwrap(), o)))
                .collect();
            SpecApp(spec, arguments)
        }
        .into(),
    );
    Ok(ImplNode::Loop(Loop {
        tiles,
        body,
        parallel,
        spec: Some(spec.clone()),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{row_major, Layout, PhysDim};
    use crate::scheduling::Action;
    use crate::target::{CpuMemoryLevel, X86Target};
    use crate::{lspec, shape};
    use nonzero::nonzero as nz;

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
}
