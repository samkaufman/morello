use crate::alignment::aligned_approx;
use crate::common::Dtype;
use crate::common::{DimSize, Shape};
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::scheduling::{
    check_tile_out_applies, tile_to_apply_err, ActionSolver, ActionT, ApplyError,
    NotApplicableReason,
};
use crate::scheduling_sugar::SchedulingSugar;
use crate::spec::{
    self, FillValue, LogicalSpec, LogicalSpecInputTilingInference, PrimitiveBasics,
    PrimitiveSpecType, Spec,
};
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::views::{Param, View, ViewE};
use itertools::Either;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;

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
        let Some(LogicalSpecInputTilingInference {
            input_tilings,
            component_parameter_shapes: component_input_tilings,
        }) = logical_spec.input_tilings_for_tile_out(&smaller_output_tiling)
        else {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Tiling doesn't apply to logical Spec",
            ))));
        };

        // 3. Reify the tilings into Tiles we'll store with this action. Tiles objects track
        // the index and shape of the Impl parameter being tiled.
        let mut new_tiles =
            input_tiles_for_tile_out(&operands, input_tilings.0, rank, output_idx, parallel)?;
        new_tiles.push(smaller_output);
        loop_spec_with_shrunken_tiles(
            component_input_tilings,
            new_tiles,
            logical_spec,
            parallel,
            spec,
        )
    }

    fn top_down_solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
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

                        let lhs_tiling =
                            Tiling::new_simple(smallvec![lhs.shape()[0], lhs.shape()[1], self.k]);
                        let rhs_tiling =
                            Tiling::new_simple(smallvec![rhs.shape()[0], self.k, rhs.shape()[2]]);
                        let output_tiling = Tiling::new_simple(operands[2].shape().into());

                        let tiles = vec![
                            LoopTile {
                                parameter_index: 0,
                                axes: vec![0, 1, 2],
                                tile: lhs_tiling
                                    .apply(Param::new(0, lhs.clone()))
                                    .map(|v| v.boxed_viewe())
                                    .map_err(tile_to_apply_err)?,
                            },
                            LoopTile {
                                parameter_index: 1,
                                axes: vec![0, 2, 3],
                                tile: rhs_tiling
                                    .apply(Param::new(1, rhs.clone()))
                                    .map(|v| v.boxed_viewe())
                                    .map_err(tile_to_apply_err)?,
                            },
                        ];

                        loop_spec_with_shrunken_tiles(
                            vec![vec![
                                lhs_tiling.shape().clone(),
                                rhs_tiling.shape().clone(),
                                output_tiling.shape().clone(),
                            ]],
                            tiles,
                            logical_spec,
                            false,
                            spec,
                        )
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

                        let mut split_shape = Shape::from_slice(in_tensor_spec.shape());
                        split_shape[usize::from(*dim)] = self.k;
                        let split_tiling = Tiling::new_simple(split_shape.clone());
                        let tiles = vec![LoopTile {
                            parameter_index: 0,
                            axes: (0..u8::try_from(in_tensor_spec.shape().len()).unwrap())
                                .collect(),
                            tile: split_tiling
                                .apply(Param::new(0, in_tensor_spec.clone()))
                                .map(|v| v.boxed_viewe())
                                .map_err(tile_to_apply_err)?,
                        }];
                        let mut new_shapes = vec![split_shape]; // TODO: Reserve capacity
                        new_shapes.extend(operands.iter().skip(1).map(|o| o.shape().into()));
                        loop_spec_with_shrunken_tiles(
                            vec![new_shapes],
                            tiles,
                            logical_spec,
                            false,
                            spec,
                        )
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
                    tile: Tiling::new_simple(smallvec![*old_b, self.k, *old_n])
                        .apply(Param::new(
                            0,
                            TensorSpec::new_noncanon_with_aux(
                                smallvec![*old_b, *old_k, *old_n],
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
                debug_assert_eq!(new_app_args.len(), spec.0.operand_count());
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

/// A `TileOut` special case for [SoftmaxDenominatorAndUnscaledFromMax].
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
        Tiling::new_simple(new_output_shape.clone().either_into()), // input
        Tiling::new_simple(onescan_shape.clone()),                  // max
        Tiling::new_simple(onescan_shape.clone()),                  // output: denominator
    ];

    // 3. Reify the tilings into Tiles we'll store with this action.
    let tilings_and_bindings: Vec<(Tiling, Vec<Option<u8>>)> = updated_input_tilings
        .into_iter()
        .map(|tiling| (tiling, (0..rank).map(Some).collect()))
        .collect();
    match input_tiles_for_tile_out(&operands, tilings_and_bindings, usize::from(rank), 3, false) {
        Ok(mut new_tiles) => {
            new_tiles.push(smaller_output);
            Some(loop_spec_with_shrunken_tiles(
                vec![vec![
                    new_output_shape.clone().either_into(), // parameter 0: input
                    onescan_shape.clone(),                  // parameter 1: max
                    onescan_shape,                          // parameter 2: denominator output
                    new_output_shape.either_into(),         // parameter 3: unscaled output
                ]],
                new_tiles,
                &spec.0,
                false,
                spec,
            ))
        }
        Err(e) => Some(Err(e)),
    }
}

/// Returns a [Loop] with a smaller sub-Spec application.
fn loop_spec_with_shrunken_tiles<Tgt: Target>(
    component_parameter_shapes: Vec<Vec<Shape>>,
    tiles: Vec<LoopTile<Tgt>>,
    logical_spec: &LogicalSpec<Tgt>,
    parallel: bool,
    spec: &Spec<Tgt>,
) -> Result<ImplNode<Tgt>, ApplyError> {
    let mut inner_spec = logical_spec.clone();
    match &mut inner_spec {
        LogicalSpec::Primitive(prim_basics, primitive_aux, _) => {
            debug_assert_eq!(component_parameter_shapes.len(), 1);
            debug_assert_eq!(
                component_parameter_shapes[0].len(),
                prim_basics.dtypes.len()
            );

            let original_shapes = prim_basics.parameter_shapes();
            prim_basics.replace_io(
                &component_parameter_shapes[0]
                    .iter()
                    .zip(&prim_basics.dtypes)
                    .map(|(shape, &dtype)| (&shape[..], dtype))
                    .collect::<Vec<_>>(),
            );

            for (param_idx, ((original_shape, new_shape), aux)) in original_shapes
                .into_iter()
                .zip(prim_basics.parameter_shapes())
                .zip(primitive_aux.iter_mut())
                .enumerate()
            {
                update_aux_for_tiling(
                    aux,
                    &original_shape,
                    &new_shape,
                    prim_basics.dtypes[param_idx],
                );
            }
        }
        LogicalSpec::Compose {
            components,
            operand_auxes,
            ..
        } => {
            debug_assert!(!components.is_empty());
            debug_assert_eq!(components.len(), component_parameter_shapes.len());
            let original_component_shapes: Vec<Vec<Shape>> = components
                .iter()
                .map(|subspec| subspec.parameter_shapes())
                .collect();
            for (subspec, shapes) in components.iter_mut().zip(component_parameter_shapes) {
                debug_assert_eq!(shapes.len(), subspec.typ.operand_count());
                debug_assert_eq!(shapes.len(), subspec.dtypes.len());
                let new_operands: Vec<(&[DimSize], Dtype)> = shapes
                    .iter()
                    .zip(&subspec.dtypes)
                    .map(|(shape, &dtype)| (&shape[..], dtype))
                    .collect();
                subspec.replace_io(&new_operands);
            }
            update_compose_aux_for_tiling(components, &original_component_shapes, operand_auxes);
        }
    };
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

/// Create [LoopTile]s corresponding to a set of input [Tiling]s and bindings.
fn input_tiles_for_tile_out<Tgt: Target>(
    operands: &[TensorSpec<Tgt>],
    tilings_and_bindings: Vec<(Tiling, Vec<Option<u8>>)>,
    rank: usize,
    output_idx: usize,
    parallel: bool,
) -> Result<Vec<LoopTile<Tgt>>, ApplyError> {
    let mut next_fresh_loop_dim = u8::try_from(rank).unwrap();
    let mut new_tiles = Vec::new();
    for (operand_idx, (original_input, (tiling, axes_binding))) in
        operands.iter().zip(tilings_and_bindings).enumerate()
    {
        if operand_idx == output_idx {
            continue;
        }
        if !original_input.is_valid_tile_shape(tiling.shape(), parallel) {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeInvalid,
            ));
        }
        if original_input.shape() == &tiling.shape()[..] {
            continue;
        }

        let operand_idx_u8 = u8::try_from(operand_idx).unwrap();
        new_tiles.push(LoopTile {
            parameter_index: operand_idx_u8,
            axes: axes_binding
                .iter()
                .map(|binding| match *binding {
                    Some(output_dim) => output_dim,
                    None => {
                        // fresh axis IDs where none is bound
                        let fresh = next_fresh_loop_dim;
                        next_fresh_loop_dim += 1;
                        fresh
                    }
                })
                .collect(),
            tile: tiling
                .apply(Param::new(operand_idx_u8, original_input.clone()))
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        });
    }
    Ok(new_tiles)
}

// TODO: We really shouldn't need this helper function.
/// Updates the layout and alignment of `aux`. Alignment is not updated if the shape of the tensor
/// is unchanged.
fn update_aux_for_tiling<Tgt: Target>(
    aux: &mut crate::tensorspec::TensorSpecAux<Tgt>,
    original_shape: &[DimSize],
    new_shape: &[DimSize],
    dtype: Dtype,
) {
    let original_tensor_spec = TensorSpec {
        dtype,
        shape: original_shape.into(),
        aux: aux.clone(),
    };
    if let Ok(new_layout) = aux.layout.update_for_tiling(original_shape, new_shape) {
        aux.layout = new_layout;
    }
    if original_shape != new_shape {
        aux.aligned =
            aligned_approx(new_shape, new_shape, &original_tensor_spec).unwrap_or(aux.aligned);
    }
}

/// Runs [update_aux_for_tiling] on all Compose parameters' [TensorSpecAux]es.
fn update_compose_aux_for_tiling<Tgt: Target>(
    components: &[PrimitiveBasics],
    original_component_shapes: &[Vec<Shape>],
    operand_auxes: &mut [TensorSpecAux<Tgt>],
) {
    let mut aux_idx = 0;

    // First component's external parameters (skip output and first input)
    let c0_output_idx = components[0].typ.unique_output_index().unwrap();
    for parameter in 1..components[0].typ.operand_count() {
        if parameter != c0_output_idx {
            let original_shape = &original_component_shapes[0][parameter];
            let new_shape = components[0].parameter_shape(parameter);
            update_aux_for_tiling(
                &mut operand_auxes[aux_idx],
                original_shape,
                &new_shape,
                components[0].dtypes[parameter],
            );
            aux_idx += 1;
        }
    }

    // Middle components' external parameters (skip output and first input)
    for (component_idx, component) in components
        .iter()
        .enumerate()
        .take(components.len() - 1)
        .skip(1)
    {
        let output_idx = component.typ.unique_output_index().unwrap();
        for parameter in 1..component.typ.operand_count() {
            if parameter != output_idx {
                let original_shape = &original_component_shapes[component_idx][parameter];
                let new_shape = component.parameter_shape(parameter);
                update_aux_for_tiling(
                    &mut operand_auxes[aux_idx],
                    original_shape,
                    &new_shape,
                    component.dtypes[parameter],
                );
                aux_idx += 1;
            }
        }
    }

    // Last component's external parameters (skip output but include all inputs)
    let last_component_idx = components.len() - 1;
    let last_component = &components[last_component_idx];
    let cl_output_idx = last_component.typ.unique_output_index().unwrap();
    for parameter in 0..last_component.typ.operand_count() {
        if parameter != cl_output_idx {
            let original_shape = &original_component_shapes[last_component_idx][parameter];
            let new_shape = last_component.parameter_shape(parameter);
            update_aux_for_tiling(
                &mut operand_auxes[aux_idx],
                original_shape,
                &new_shape,
                last_component.dtypes[parameter],
            );
            aux_idx += 1;
        }
    }

    // Finally, handle the compose output (which is the first component's output)
    let original_shape = &original_component_shapes[0][c0_output_idx];
    let new_shape = components[0].parameter_shape(c0_output_idx);
    update_aux_for_tiling(
        &mut operand_auxes[aux_idx],
        original_shape,
        &new_shape,
        components[0].dtypes[c0_output_idx],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::loops::Loop;
    use crate::layout::{row_major, Layout, PhysDim};
    use crate::scheduling::Action;
    use crate::target::{CpuMemoryLevel, X86Target};
    use crate::tensorspec::{TensorSpec, TensorSpecArbMaxShape, TensorSpecAux};
    use crate::{shape, spec};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;

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
        let spec: Spec<X86Target> = spec!(MatmulAccum(
            [4, 8, 8, 8],
            (f32, CpuMemoryLevel::GL, bcm_layout),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let application = action.apply(&spec);
        assert!(
            matches!(
                application,
                Err(ApplyError::NotApplicable(NotApplicableReason::Other(_))),
            ),
            "expected NotApplicable(Other) but got {application:?}",
        );
    }

    #[test]
    fn test_tile_out_daufm() {
        // Create a SoftmaxDenominatorAndUnscaledFromMax Spec
        let spec = Spec::<X86Target>(
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                        scan_dim: 2,
                        accum: true,
                    },
                    spec_shape: smallvec![nz!(4u32), nz!(8u32), nz!(16u32)],
                    dtypes: vec![Dtype::Float32; 4],
                },
                vec![
                    TensorSpecAux {
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: row_major(3),
                        vector_size: None,
                    };
                    4
                ],
                false,
            ),
            X86Target::max_mem(),
        );

        let tile_action = TileOut::SingleLoop {
            dim: 2,
            size: nz!(8u32),
            parallel: false,
        };

        let ImplNode::Loop(Loop {
            tiles,
            body,
            parallel,
            spec: loop_spec,
        }) = tile_action
            .apply(&spec)
            .expect("TileOut should apply successfully")
        else {
            panic!("Expected ImplNode::Loop")
        };

        assert!(!parallel);
        assert_eq!(loop_spec, Some(spec));

        // Parameters 0 and 3 should be tiled
        let param_indices: Vec<u8> = tiles.iter().map(|t| t.parameter_index).collect();
        assert_eq!(param_indices, [0, 3]);

        // Check shapes of the body SpecApp
        let body_params: Vec<_> = body.parameters().collect();
        let expected_shapes = [
            &[nz!(4u32), nz!(8u32), nz!(8u32)][..], // Parameter 0: input (tiled)
            &[nz!(4u32), nz!(8u32), nz!(1u32)][..], // Parameter 1: max (reduced)
            &[nz!(4u32), nz!(8u32), nz!(1u32)][..], // Parameter 2: denominator (reduced)
            &[nz!(4u32), nz!(8u32), nz!(8u32)][..], // Parameter 3: output (tiled)
        ];
        assert_eq!(
            body_params.iter().map(|p| p.shape()).collect::<Vec<_>>(),
            expected_shapes
        );

        // Check that the tile shapes match the body parameter shapes
        assert!(tiles.iter().all(|tile| {
            let param_idx = usize::from(tile.parameter_index);
            tile.tile.shape() == body_params[param_idx].shape()
        }));
    }

    proptest! {
        /// Test that [update_aux_for_tiling] preserves auxiliary information when shapes are
        /// identical.
        #[test]
        fn test_update_aux_for_tiling_identical_shapes_preserves_aux(
            mut tspec in any_with::<TensorSpec<X86Target>>(TensorSpecArbMaxShape(shape![8, 8, 8]))
        ) {
            let shape = tspec.shape().to_vec();
            let dtype = tspec.dtype();
            let original_aux = tspec.aux.clone();
            update_aux_for_tiling(&mut tspec.aux, &shape, &shape, dtype);
            prop_assert_eq!(tspec.aux.layout, original_aux.layout);
            prop_assert_eq!(tspec.aux.aligned, original_aux.aligned);
            prop_assert_eq!(tspec.aux.level, original_aux.level);
            prop_assert_eq!(tspec.aux.vector_size, original_aux.vector_size);
        }
    }
}
