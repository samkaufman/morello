use itertools::Either;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::fmt::Display;
use std::rc::Rc;
use std::{iter, mem};

use crate::alignment::aligned_approx;
use crate::common::{DimSize, Dtype, Shape};
use crate::cost::{Cost, MainCost};
use crate::imp::blocks::Block;
use crate::imp::kernels::KernelApp;
use crate::imp::loops::{compute_loop_main_cost, Loop, LoopTile};
use crate::imp::moves::{move_cost, movelet_memory_allocation, MoveLet};
use crate::imp::pipeline::{Pipeline, StageWiring};
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling_sugar::SchedulingSugar;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{Kernel, MemoryLevel, Target};
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::{prev_power_of_two, snap_memvec_up};
use crate::views::{CacheView, Param, Tensor, Tile, TileError, View, ViewE, ViewExt};

/// A scheduling decision which can be applied to a Spec to produce an Impl.
///
/// [Action]s contain the minimal amount of information needed to distinguish a one scheduling
/// decision from another, which makes it appropriate for storing in a database so that the
/// corresponding Impl node can be computed given the Spec.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum Action<Tgt: Target> {
    /// Tile the output tensor and its inputs to respect the updated inputs.
    TileOut(TileOut),
    Split {
        k: DimSize,
    },
    /// Move a tensor to a different memory level, layout, and/or dtype.
    Move {
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    },
    /// Allocate an output tensor, a Zero sub-Spec, and an accumulating variant of the receiver.
    ToAccum,
    ToSoftmaxParts {
        max_level: Tgt::Level,
        max_layout: Layout,
        max_vector_size: Option<DimSize>,
        denominator_level: Tgt::Level,
        denominator_layout: Layout,
        denominator_vector_size: Option<DimSize>,
    },
    /// Rewrites a SoftmaxDenominatorAndMax into a Max followed by SoftmaxDenominator.
    ToMaxAndDenominator,
    Bufferize {
        index: usize,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    },
    SpatialSplit,
    // TODO: Remove 'force' bool from Place
    Place(Tgt::Kernel, bool),
}

#[derive(Debug)]
pub enum ActionSolver<Tgt: Target> {
    PrimitiveTileOut {
        outer_spec: Spec<Tgt>,
        body_spec: Spec<Tgt>,
    },
    Move {
        prologue: Option<Spec<Tgt>>,
        body: Spec<Tgt>,
        epilogue: Option<Spec<Tgt>>,
        base_main_cost: MainCost,
        allocation: MemoryAllocation,
    },
    Fallback(ImplNode<Tgt>),
}

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

/// Data useful to both a Move's [ActionSolver] or [ImplNode].
struct MoveLetPlan<'a, Tgt: Target> {
    outer_moved_operand_spec: &'a TensorSpec<Tgt>,
    new_spec: TensorSpec<Tgt>,
    prologue_spec: Option<Spec<Tgt>>,
    epilogue_spec: Option<Spec<Tgt>>,
    new_body_spec: Spec<Tgt>,
    new_operands: Vec<TensorSpec<Tgt>>,
    is_cache_miss: bool,
}

#[derive(thiserror::Error, Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum ApplyError {
    #[error("Cannot apply action to non-canonical Spec")]
    SpecNotCanonical,
    #[error("Action does not apply to this Spec: {0}")]
    NotApplicable(NotApplicableReason),
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum NotApplicableReason {
    OutOfMemory(String),
    TileShapeMatchesOriginal,
    TileShapeIsLarger,
    TileShapeInvalid,
    ParallelPrevented,
    LayoutIncompatible,
    SelfMove,
    VectorSizeInvalid(Dtype, DimSize),
    MultipleOutputs,
    Other(Option<&'static str>),
}

impl<Tgt: Target> Action<Tgt> {
    pub fn apply(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        if !spec.is_canonical() {
            return Err(ApplyError::SpecNotCanonical);
        }
        self.apply_unchecked_canon(spec)
    }

    /// Like [Action::apply], but does not check if the Spec is canonical. Passing a non-canonical
    /// Spec is a logic error.
    pub fn apply_unchecked_canon(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        match self {
            Action::TileOut(tileout) => {
                let Some(output_idx) = logical_spec.unique_output_index() else {
                    return Err(ApplyError::NotApplicable(
                        NotApplicableReason::MultipleOutputs,
                    ));
                };

                let current_output = &operands[output_idx];
                let current_out_shape = current_output.shape();
                let rank = current_out_shape.len();

                let output_shape = tileout.tiled_output_shape(current_out_shape);
                let parallel = tileout.parallel();

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
                    // is given a fresh integer identifier, otherwise is is given the
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

                self.loop_spec_with_shrunken_tiles(
                    operands,
                    new_tiles,
                    logical_spec,
                    parallel,
                    spec,
                )
            }
            Action::Split { k } => match logical_spec {
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
                                *k < lhs.shape()[1],
                                "Cannot split to k={k} when inner dim. is not larger (it is {})",
                                lhs.shape()[1]
                            );

                            if lhs.shape()[1].get() % k.get() != 0 {
                                return Err(ApplyError::NotApplicable(NotApplicableReason::Other(
                                    Some("Original size is not a multiple of split size"),
                                )));
                            }

                            let tiles = vec![
                                LoopTile {
                                    parameter_index: 0,
                                    axes: vec![0, 1],
                                    tile: Tile::new(
                                        vec![lhs.shape()[0], *k],
                                        vec![lhs.shape()[0], *k],
                                        Param::new(0, lhs.clone()),
                                    )
                                    .map(|v| v.boxed_viewe())
                                    .map_err(tile_to_apply_err)?,
                                },
                                LoopTile {
                                    parameter_index: 1,
                                    axes: vec![1, 2],
                                    tile: Tile::new(
                                        vec![*k, rhs.shape()[1]],
                                        vec![*k, rhs.shape()[1]],
                                        Param::new(1, rhs.clone()),
                                    )
                                    .map(|v| v.boxed_viewe())
                                    .map_err(tile_to_apply_err)?,
                                },
                            ];

                            self.loop_spec_with_shrunken_tiles(
                                operands,
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
                                *k < in_tensor_spec.shape()[usize::from(*dim)],
                                "Cannot split to k={k} when inner dim. is not larger (it is {})",
                                in_tensor_spec.shape()[usize::from(*dim)]
                            );
                            if in_tensor_spec.shape()[usize::from(*dim)].get() % k.get() != 0 {
                                return Err(ApplyError::NotApplicable(NotApplicableReason::Other(
                                    Some("Original size is not a multiple of split size"),
                                )));
                            }

                            let mut split_shape = in_tensor_spec.shape().to_vec();
                            split_shape[usize::from(*dim)] = *k;

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
                            self.loop_spec_with_shrunken_tiles(
                                operands,
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
                    let [old_m, old_k, old_n] = &components[0].spec_shape[..] else {
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
                    }) = compose_tail.tile_out(&[old_m.get(), k.get()])
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
                    let new_head_rhs_looptile = LoopTile {
                        parameter_index: 0,
                        axes: vec![tail_output_tile.axes[1], 255], // TODO: Replace 255
                        tile: Tiling::new_simple(vec![*k, *old_n])
                            .apply(Param::new(
                                0,
                                TensorSpec::new_noncanon_with_aux(
                                    vec![*old_k, *old_n],
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
                    new_app_args
                        .push(Param::new(0, new_head_rhs_looptile.tile.spec().clone()).into());
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
            },
            Action::Bufferize {
                index,
                level,
                layout,
                vector_size,
            } => {
                let LogicalSpec::Compose {
                    components,
                    operand_auxes,
                    serial_only,
                } = logical_spec
                else {
                    // TODO: Use a more specific NotApplicableReason.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "Not a Compose",
                    ))));
                };

                debug_assert!(*index < components.len() - 1);
                let consumer = &components[*index];
                let intermediate_tensor = Tensor::new(TensorSpec::new_canon(
                    consumer.input_shape(0),
                    consumer.input_dtype(0),
                    layout.contiguous_full(),
                    true,
                    *level,
                    layout.clone(),
                    *vector_size,
                ));

                // Compute the memory limits for the new children.
                let new_limits = {
                    // Compute the amount of memory consumed by the new, intermediate
                    // tensor.
                    // TODO: This shouldn't need to be both here and in `memory_allocated`.
                    let first_input_volume: u64 = consumer
                        .input_shape(0)
                        .into_iter()
                        .map(|v| u64::from(v.get()))
                        .product();
                    let intermediate_mem_consumed_nondiscrete = Tgt::levels().map(|l| {
                        if level == &l {
                            u64::from(consumer.dtypes[0].size()) * first_input_volume
                        } else {
                            0u64
                        }
                    });

                    // TODO: Use MemoryLimits::Pipeline where appropriate instead.
                    // (Already done by Pipeline::memory_allocated.)
                    let mut m = MemoryLimits::Standard(match &spec.1 {
                        MemoryLimits::Standard(v) => v
                            .clone()
                            .checked_sub_snap_down(&intermediate_mem_consumed_nondiscrete)
                            .map_err(|oom_idx| {
                                ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                                    Tgt::levels()[oom_idx].to_string(),
                                ))
                            })?,
                    });
                    m.discretize();
                    m
                };

                let inner_compose = ImplNode::from(make_inner_compose(
                    *index,
                    components,
                    operand_auxes,
                    *serial_only,
                    intermediate_tensor.clone(),
                    new_limits.clone(),
                ));
                let outer_compose = ImplNode::from(make_outer_compose(
                    *index,
                    components,
                    operand_auxes,
                    *serial_only,
                    intermediate_tensor.clone(),
                    new_limits,
                ));

                Ok(ImplNode::Pipeline(Pipeline {
                    stages: vec![inner_compose, outer_compose],
                    parameters: spec.0.parameters(),
                    wirings: vec![StageWiring {
                        intermediate_tensors: vec![Rc::new(intermediate_tensor)],
                    }],
                    spec: Some(spec.clone()),
                }))
            }
            Action::SpatialSplit => {
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
                let inner_filters_tile =
                    outer_filters_tile.clone().squeeze_dims(2..rank).transpose();

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
            Action::Move {
                source_idx,
                destination_dtype,
                destination_level,
                destination_layout,
                destination_vector_size,
            } => {
                let MoveLetPlan {
                    outer_moved_operand_spec,
                    new_spec,
                    prologue_spec,
                    epilogue_spec,
                    new_body_spec,
                    new_operands,
                    is_cache_miss,
                } = plan_movelet(
                    spec,
                    &operands,
                    *source_idx,
                    *destination_dtype,
                    *destination_level,
                    destination_layout,
                    *destination_vector_size,
                )?;

                let inner_moved_operand = if is_cache_miss {
                    let source = Param::new(*source_idx, outer_moved_operand_spec.clone());
                    ViewE::from(CacheView::new(source, new_spec))
                } else {
                    ViewE::from(Tensor::new(new_spec))
                };

                let source_param =
                    ViewE::from(Param::new(*source_idx, outer_moved_operand_spec.clone()));
                let prologue = prologue_spec.map(|s| {
                    ImplNode::from(SpecApp::new(
                        s.clone(),
                        [source_param.clone(), inner_moved_operand.clone()],
                    ))
                });
                let epilogue = epilogue_spec.map(|s| {
                    ImplNode::from(SpecApp::new(
                        s.clone(),
                        [inner_moved_operand.clone(), source_param],
                    ))
                });

                let main_stage = {
                    let inner_operands = new_operands.iter().enumerate().map(|(i, o)| {
                        if i == *source_idx as usize {
                            inner_moved_operand.clone()
                        } else {
                            ViewE::from(Param::new(u8::try_from(i).unwrap(), o.clone()))
                        }
                    });
                    ImplNode::from(SpecApp::new(new_body_spec, inner_operands))
                };

                Ok(ImplNode::MoveLet(MoveLet::new(
                    *source_idx,
                    outer_moved_operand_spec.clone(),
                    inner_moved_operand,
                    prologue,
                    main_stage,
                    epilogue,
                    Some(spec.clone()),
                )))
            }
            Action::ToAccum => {
                let head = match logical_spec {
                    LogicalSpec::Primitive(basics, ..) => basics,
                    LogicalSpec::Compose { components, .. } => &components[0],
                };

                let PrimitiveBasics {
                    typ:
                        PrimitiveSpecType::Matmul { accum }
                        | PrimitiveSpecType::Conv { accum }
                        | PrimitiveSpecType::Max { accum, .. }
                        | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                        | PrimitiveSpecType::SoftmaxDenominatorAndMax { accum, .. },
                    ..
                } = head
                else {
                    // TODO: Use a more specific NotApplicableReason.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "ToAccum is only defined for Matmul, Conv, Max, SoftmaxDenominator, and SoftmaxDenominatorAndMax",
                    ))));
                };
                if *accum {
                    // TODO: Use a more specific NotApplicableReason.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "Already accumulating",
                    ))));
                }

                let zero_apps = make_zeroes_for_spec(spec);
                let accum_app = {
                    let mut spec = Spec(logical_spec.clone_as_accum(), spec.1.clone());
                    spec.canonicalize()
                        .expect("ToAccum's introduced accumulating Spec should be canonicalizable");
                    let app_arguments = operands
                        .iter()
                        .enumerate()
                        .map(|(i, t)| ViewE::from(Param::new(i.try_into().unwrap(), t.clone())));
                    SpecApp::new(spec, app_arguments).into()
                };

                let mut stages = zero_apps;
                stages.push(accum_app);
                let default_child = Some(stages.len() - 1);
                Ok(ImplNode::Block(Block {
                    stages,
                    parameters: operands,
                    spec: Some(spec.clone()),
                    default_child,
                }))
            }
            Action::ToSoftmaxParts {
                max_level,
                max_layout,
                max_vector_size,
                denominator_level,
                denominator_layout,
                denominator_vector_size,
            } => {
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
                        contig: max_layout.contiguous_full(),
                        aligned: true,
                        level: *max_level,
                        layout: max_layout.clone(),
                        vector_size: *max_vector_size,
                    },
                };
                max_spec.shape[usize::from(*scan_dim)] = nz!(1u32);
                let max_tensor = Tensor::new(max_spec);

                // Make tensor for storing the denominator
                let mut denominator_spec = TensorSpec {
                    shape: spec_shape.clone(),
                    dtype: dtypes[0],
                    aux: TensorSpecAux {
                        contig: denominator_layout.contiguous_full(),
                        aligned: true,
                        level: *denominator_level,
                        layout: denominator_layout.clone(),
                        vector_size: *denominator_vector_size,
                    },
                };
                denominator_spec.shape[usize::from(*scan_dim)] = nz!(1u32);
                let denominator_tensor = Tensor::new(denominator_spec);

                let new_buffer_consumption = Tgt::levels().map(|l| {
                    let mut r = 0;
                    if max_level == &l {
                        let max_spec = max_tensor.spec();
                        r += u64::from(max_spec.dtype.size()) * u64::from(max_spec.volume().get());
                    }
                    if denominator_level == &l {
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
                        ViewE::from(Param::new(3, operands[1].clone())),
                    ];
                    SpecApp::new(complete_spec, app_args).into()
                };

                Ok(ImplNode::Pipeline(Pipeline {
                    stages: vec![denom_app, complete_app],
                    wirings: vec![StageWiring {
                        intermediate_tensors: vec![
                            Rc::new(max_tensor),
                            Rc::new(denominator_tensor),
                        ],
                    }],
                    parameters: operands,
                    spec: Some(spec.clone()),
                }))
            }
            Action::ToMaxAndDenominator => {
                // (x, maxes, denom) => {
                //   Max(x, maxes)
                //   SoftmaxDenom(x, maxes, denom)
                // }
                let LogicalSpec::Primitive(head, auxes, serial_only) = logical_spec else {
                    // TODO: Add a more specific NotApplicableReason
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "ToMaxAndDenominator only defined for Primitive",
                    ))));
                };
                let PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim, accum },
                    spec_shape,
                    dtypes,
                } = head
                else {
                    // TODO: Use a more specific NotApplicableReason.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "ToMaxAndDenominator is only defined for SoftmaxDenominatorAndMax",
                    ))));
                };
                if *accum {
                    // TODO: Use a more specific NotApplicableReason.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "Accumlating SoftmaxDenominatorAndMax not supported",
                    ))));
                }

                let max_app = {
                    let mut max_spec = Spec(
                        LogicalSpec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::Max {
                                    dim: *scan_dim,
                                    accum: false,
                                },
                                spec_shape: spec_shape.clone(),
                                dtypes: vec![dtypes[0], dtypes[1]],
                            },
                            vec![operands[0].aux.clone(), operands[1].aux.clone()],
                            *serial_only,
                        ),
                        spec.1.clone(),
                    );
                    max_spec.canonicalize().unwrap();
                    let app_args = vec![
                        ViewE::from(Param::new(0, operands[0].clone())),
                        ViewE::from(Param::new(1, operands[1].clone())),
                    ];
                    SpecApp::new(max_spec, app_args).into()
                };

                let denom_app = {
                    let mut denom_spec = Spec(
                        LogicalSpec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::SoftmaxDenominator {
                                    scan_dim: *scan_dim,
                                    accum: false,
                                },
                                spec_shape: spec_shape.clone(),
                                dtypes: dtypes.clone(),
                            },
                            auxes.clone(),
                            *serial_only,
                        ),
                        spec.1.clone(),
                    );
                    denom_spec.canonicalize().unwrap();
                    let app_args = vec![
                        ViewE::from(Param::new(0, operands[0].clone())),
                        ViewE::from(Param::new(1, operands[1].clone())),
                        ViewE::from(Param::new(2, operands[2].clone())),
                    ];
                    SpecApp::new(denom_spec, app_args).into()
                };

                Ok(ImplNode::Block(Block {
                    stages: vec![max_app, denom_app],
                    parameters: operands,
                    spec: Some(spec.clone()),
                    default_child: None,
                }))
            }
            Action::Place(k, force) => {
                if !force && !k.applies_to_logical_spec(&spec.0) {
                    // TODO: Use better error message-producing Error type.
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(None)));
                }

                let arguments = operands
                    .iter()
                    .enumerate()
                    .map(|(i, p)| ViewE::from(Param::new(i.try_into().unwrap(), p.clone())))
                    .collect::<Vec<_>>();

                // Check that the kernel doesn't violate memory limits.
                match (force, k.memory_allocated(&arguments), &spec.1) {
                    (true, _, _) => {}
                    (false, MemoryAllocation::Inner(_) | MemoryAllocation::Pipeline { .. }, _) => {
                        panic!("Kernel::memory_allocated returned non-Standard MemoryAllocation")
                    }
                    (
                        false,
                        MemoryAllocation::Simple(allocated),
                        MemoryLimits::Standard(bounds),
                    ) => {
                        for (i, (a, b)) in allocated.iter().zip(bounds.iter()).enumerate() {
                            if *a > b {
                                return Err(ApplyError::NotApplicable(
                                    NotApplicableReason::OutOfMemory(Tgt::levels()[i].to_string()),
                                ));
                            }
                        }
                    }
                };

                Ok(ImplNode::Kernel(KernelApp {
                    kernel_type: *k,
                    arguments,
                    spec: Some(spec.clone()),
                }))
            }
        }
    }

    fn loop_spec_with_shrunken_tiles(
        &self,
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
                aligned_approx(loop_tile.tile.shape(), loop_tile.tile.step_sizes(), ref_op)
                    .unwrap();
            ref_op.shrink(loop_tile.tile.shape(), aligned).unwrap();
        }

        // Update the Spec with these new operands, as well as its serial_only flag.
        let mut inner_spec = logical_spec.clone();
        inner_spec.replace_io(&operands);
        inner_spec.set_serial_only(inner_spec.serial_only() || parallel);
        inner_spec.canonicalize().unwrap();

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

    /// Returns a value which produces sub-Spec requests and compute a [Cost].
    ///
    /// This is functionally equivalent to calling [Action::apply] to produce a partial Impl and
    /// then gathering its sub-Specs and computing a cost, but is usually faster.
    ///
    /// The caller must ensure that `spec` is in canonical form. Passing a non-canonical form is a
    /// logic error.
    pub fn solver(&self, spec: &Spec<Tgt>) -> Result<ActionSolver<Tgt>, ApplyError> {
        match (self, &spec.0) {
            (Action::TileOut(tileout), LogicalSpec::Primitive(basics, ..)) => {
                let Some(output_tensor) = spec.0.unique_output() else {
                    return Err(ApplyError::NotApplicable(
                        NotApplicableReason::MultipleOutputs,
                    ));
                };
                let untiled_output_shape = output_tensor.shape();
                let tile_shape = tileout.tiled_output_shape(untiled_output_shape);
                let parallel = tileout.parallel();

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
                                [(0, 0), (2, 1)].into_iter(),
                                spec,
                                &tile_shape,
                                parallel,
                            )?,
                        });
                    }
                    PrimitiveSpecType::Zero | PrimitiveSpecType::Move => {
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
            (
                Action::Move {
                    source_idx,
                    destination_dtype,
                    destination_level,
                    destination_layout,
                    destination_vector_size,
                },
                _,
            ) => {
                let operands = spec.0.parameters();
                let plan = plan_movelet(
                    spec,
                    &operands,
                    *source_idx,
                    *destination_dtype,
                    *destination_level,
                    destination_layout,
                    *destination_vector_size,
                )?;
                let base_main_cost = move_cost(plan.outer_moved_operand_spec, &plan.new_spec);
                let allocation = movelet_memory_allocation(&plan.new_spec);
                return Ok(ActionSolver::Move {
                    prologue: plan.prologue_spec,
                    body: plan.new_body_spec,
                    epilogue: plan.epilogue_spec,
                    base_main_cost,
                    allocation,
                });
            }
            _ => {}
        };

        self.apply_unchecked_canon(spec)
            .map(|applied| ActionSolver::Fallback(applied))
    }
}

/// Returns applications of Zeroes corresponding to the outputs of the given Spec.
fn make_zeroes_for_spec<Tgt: Target>(spec: &Spec<Tgt>) -> Vec<ImplNode<Tgt>> {
    (0..u8::try_from(spec.0.operand_count()).unwrap())
        .flat_map(|parameter_idx| {
            if !spec.0.parameter_is_output(parameter_idx.into()) {
                return None;
            }
            let output = spec.0.parameter(parameter_idx.into());
            let subspec = LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    spec_shape: output.shape.clone(),
                    dtypes: vec![output.dtype],
                },
                vec![output.aux.clone()],
                spec.0.serial_only(),
            );
            let mut spec = Spec(subspec, spec.1.clone());
            spec.canonicalize()
                .expect("ToAccum's introduced Zeroes should be canonicalizable");
            Some(SpecApp::new(spec, [ViewE::from(Param::new(parameter_idx, output))]).into())
        })
        .collect()
}

impl<Tgt: Target> ActionSolver<Tgt> {
    pub fn subspecs(&self) -> impl Iterator<Item = Spec<Tgt>> {
        match self {
            ActionSolver::PrimitiveTileOut {
                outer_spec: _,
                body_spec,
            } => {
                // TODO: Avoid this clone
                vec![body_spec.clone()].into_iter()
            }
            ActionSolver::Move {
                prologue,
                body,
                epilogue,
                base_main_cost: _,
                allocation: _,
            } => {
                // TODO: Avoid these clones. Return an iterator of references.
                let mut v: Vec<Spec<Tgt>> = Vec::with_capacity(3);
                v.extend(prologue.clone());
                v.push(body.clone());
                v.extend(epilogue.clone());
                v.into_iter()
            }
            ActionSolver::Fallback(partial_impl) => {
                let mut partial_impl_subspecs = Vec::new();
                collect_nested_specs(partial_impl, &mut partial_impl_subspecs);
                partial_impl_subspecs.into_iter()
            }
        }
    }

    pub fn compute_cost<I>(&self, mut child_costs: I) -> Cost
    where
        I: Iterator<Item = Cost>,
    {
        match self {
            ActionSolver::PrimitiveTileOut {
                outer_spec,
                body_spec,
            } => {
                let parallel = !outer_spec.0.serial_only() && body_spec.0.serial_only();
                match &outer_spec.0 {
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ:
                                PrimitiveSpecType::Matmul { .. }
                                | PrimitiveSpecType::Move
                                | PrimitiveSpecType::Zero,
                            spec_shape,
                            ..
                        },
                        ..,
                    ) => {
                        let LogicalSpec::Primitive(
                            PrimitiveBasics {
                                spec_shape: body_shape,
                                ..
                            },
                            ..,
                        ) = &body_spec.0
                        else {
                            unreachable!();
                        };

                        let mut steps = 1;
                        let mut full_steps = 1;
                        for (o, t) in spec_shape.iter().zip(body_shape) {
                            steps *= o.get().div_ceil(t.get());
                            full_steps *= o.get() / t.get();
                        }
                        let mut cost = child_costs.next().unwrap();
                        cost.main =
                            compute_loop_main_cost::<Tgt>(steps, full_steps, parallel, cost.main);
                        cost.depth += 1;
                        cost
                    }
                    _ => unreachable!(),
                }
            }
            ActionSolver::Move {
                prologue: _,
                body: _,
                epilogue: _,
                base_main_cost,
                allocation,
            } => {
                let mut main = *base_main_cost;
                let mut child_peaks = vec![];
                let mut depth = 0;
                for child_cost in child_costs {
                    main = main.saturating_add(child_cost.main);
                    depth = depth.max(child_cost.depth);
                    child_peaks.push(child_cost.peaks);
                }
                depth += 1;
                // TODO: Is snap_memvec_up really needed or can we bake this into MemVec?
                let peaks = snap_memvec_up(
                    allocation.peak_memory_from_child_peaks::<Tgt>(&child_peaks),
                    false,
                );
                Cost { main, peaks, depth }
            }
            ActionSolver::Fallback(partial_impl) => {
                compute_impl_cost(partial_impl, &mut child_costs)
            }
        }
    }

    fn tiled_subspec_fast<B>(
        binds: B,
        original_spec: &Spec<Tgt>,
        tile_shape: &[DimSize],
        parallel: bool,
    ) -> Result<Spec<Tgt>, ApplyError>
    where
        B: Iterator<Item = (usize, usize)> + ExactSizeIterator,
    {
        let mut new_spec = original_spec.clone();
        match &mut new_spec.0 {
            LogicalSpec::Primitive(PrimitiveBasics { spec_shape, .. }, _, serial_only) => {
                for (o, t) in binds {
                    spec_shape[o] = tile_shape[t];
                }
                *serial_only = *serial_only || parallel;
            }
            _ => unreachable!(),
        }

        let outer_parameters = original_spec.0.parameters();
        let new_parameters = new_spec.0.parameters();
        let LogicalSpec::Primitive(_, new_auxes, _) = &mut new_spec.0 else {
            unreachable!();
        };

        // TODO: Should the following be optimized with `tile_shape_is_valid`?
        for ((outer, inner), new_aux) in outer_parameters
            .into_iter()
            .zip(new_parameters)
            .zip(new_auxes)
        {
            if outer.shape() == inner.shape() {
                continue;
            }

            // TODO: Need is_valid_tile_shape if we're calling update_for_tiling?
            if !outer.is_valid_tile_shape(inner.shape(), parallel) {
                return Err(ApplyError::NotApplicable(
                    NotApplicableReason::TileShapeInvalid,
                ));
            }
            let Ok((new_layout, new_contig)) = outer.layout().update_for_tiling(
                outer.shape(),
                inner.shape(),
                outer.contiguous_abs(),
            ) else {
                todo!();
            };
            new_aux.aligned = aligned_approx(inner.shape(), inner.shape(), &outer).unwrap();
            new_aux.layout = new_layout;
            new_aux.contig = new_contig;
        }

        if new_spec.canonicalize().is_err() {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::TileShapeInvalid,
            ));
        }

        Ok(new_spec)
    }
}

/// Return an error if the tile shape is invalid for a given tensor.
///
/// This can return TileShapeMatchesOriginal, TileShapeIsLarger, TileShapeInvalid, or
/// Other.
fn check_tile_out_applies<Tgt: Target>(
    current_out_shape: &[DimSize],
    output_shape: &[DimSize],
    current_output: &TensorSpec<Tgt>,
    parallel: bool,
) -> Result<(), ApplyError> {
    if current_out_shape == output_shape {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeMatchesOriginal,
        ));
    }
    if output_shape
        .iter()
        .enumerate()
        .any(|(dim, dim_size)| *dim_size > current_out_shape[dim])
    {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeIsLarger,
        ));
    }

    // Abort if it's invalid to tile the original output tensor
    // to the new shape (e.g., due to layout constraints).
    if !current_output.is_valid_tile_shape(output_shape, parallel) {
        return Err(ApplyError::NotApplicable(
            NotApplicableReason::TileShapeInvalid,
        ));
    }

    if output_shape
        .iter()
        .enumerate()
        .any(|(dim, out_size)| current_out_shape[dim].get() % out_size.get() != 0)
    {
        return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
            "Original size is not a multiple of tile size",
        ))));
    }

    Ok(())
}

// TODO: Can we replace this function with a more general `utils` crate fn. or something?
/// Push all nested [Spec]s in an Impl into a given [Vec], left to right.
fn collect_nested_specs<Tgt: Target>(imp: &ImplNode<Tgt>, out: &mut Vec<Spec<Tgt>>) {
    match imp {
        ImplNode::SpecApp(spec_app) => {
            out.push(spec_app.0.clone());
        }
        _ => {
            for child in imp.children() {
                collect_nested_specs(child, out);
            }
        }
    }
}

fn compute_impl_cost<Tgt, I>(imp: &ImplNode<Tgt>, costs: &mut I) -> Cost
where
    Tgt: Target,
    I: Iterator<Item = Cost>,
{
    match imp {
        ImplNode::SpecApp(_) => costs.next().unwrap(),
        _ => {
            let child_costs = imp
                .children()
                .iter()
                .map(|child| compute_impl_cost(child, costs))
                .collect::<Vec<_>>();
            Cost::from_node_and_child_costs(imp, &child_costs)
        }
    }
}

impl TileOut {
    pub fn tiled_output_shape(
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

    pub fn parallel(&self) -> bool {
        match self {
            TileOut::SingleLoop { parallel, .. } | TileOut::MultiLoop { parallel, .. } => *parallel,
        }
    }
}

impl Display for NotApplicableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotApplicableReason::OutOfMemory(lvl) => {
                write!(f, "Insufficient memory in {lvl}")
            }
            NotApplicableReason::TileShapeMatchesOriginal => {
                write!(f, "Tile shape matches original")
            }
            NotApplicableReason::TileShapeIsLarger => {
                write!(f, "Tile shape is larger than original")
            }
            NotApplicableReason::TileShapeInvalid => {
                write!(f, "Invalid tile shape")
            }
            NotApplicableReason::ParallelPrevented => {
                write!(f, "Cannot implement serial-only Spec with parallel tile")
            }
            NotApplicableReason::LayoutIncompatible => {
                write!(f, "Layout does not apply to tile size")
            }
            NotApplicableReason::SelfMove => {
                write!(
                    f,
                    "Source and destination TensorSpecs were equal after canonicalization"
                )
            }
            NotApplicableReason::VectorSizeInvalid(dtype, size) => {
                write!(
                    f,
                    "Target does not support {dtype} vectors with {size} values"
                )
            }
            NotApplicableReason::MultipleOutputs => {
                write!(f, "Spec has multiple outputs")
            }
            NotApplicableReason::Other(Some(reason_string)) => write!(f, "{}", reason_string),
            NotApplicableReason::Other(None) => write!(f, "Unknown reason"),
        }
    }
}

/// Build the inner Compose (or Primitive) sub-Spec appliction of a Pipeline.
///
/// The inner sub-Spec is the sub-Spec which is executed first.
fn make_inner_compose<Tgt: Target>(
    index: usize,
    components: &[PrimitiveBasics],
    parent_operand_auxes: &[TensorSpecAux<Tgt>],
    serial_only: bool,
    intermediate_tensor: Tensor<Tgt>,
    new_limits: MemoryLimits,
) -> SpecApp<ViewE<Tgt>> {
    let inner_components = &components[(index + 1)..];
    let inner_input_count = 1 + inner_components
        .iter()
        .map(|c| c.typ.input_count() - 1)
        .sum::<usize>();

    // Collect parameter auxes from the parent Spec and add an aux for the intermediate/output.
    let offset = parent_operand_auxes.len() - inner_input_count - 1;
    let offset_u8 = u8::try_from(offset).unwrap();
    let passthrough_auxes = &parent_operand_auxes[offset..(parent_operand_auxes.len() - 1)];
    let mut auxes = vec![];
    auxes.reserve_exact(passthrough_auxes.len() + 1);
    auxes.extend_from_slice(passthrough_auxes);
    auxes.push(intermediate_tensor.spec().aux.clone());

    // Construct the Spec. (Next, we'll wrap this in a SpecApp.)
    let mut inner_compose = Spec(
        match inner_components {
            [] => unreachable!("should never be empty"),
            [single] => LogicalSpec::Primitive(single.clone(), auxes, serial_only),
            _ => LogicalSpec::Compose {
                components: inner_components.into(),
                operand_auxes: auxes,
                serial_only,
            },
        },
        new_limits,
    );
    inner_compose.canonicalize().unwrap();

    // Above, we inserted the output at the end. Check that's the real output position.
    debug_assert_eq!(
        inner_compose.0.unique_output_index(),
        Some(inner_compose.0.operand_count() - 1),
        "Inner Compose must have a single output for the intermediate to be in the correct position"
    );

    // Parameters
    let params = (0..u8::try_from(inner_compose.0.operand_count() - 1).unwrap())
        .map(|i| {
            ViewE::from(Param::new(
                i + offset_u8,
                inner_compose.0.parameter(i.into()),
            ))
        })
        .chain(std::iter::once(ViewE::from(intermediate_tensor)))
        .collect::<Vec<_>>();

    SpecApp::new(inner_compose, params)
}

/// Build the outer Compose (or Primitive) sub-Spec appliction of a Pipeline.
///
/// The outer sub-Spec is the sub-Spec which is executed second.
fn make_outer_compose<Tgt: Target>(
    index: usize,
    components: &[PrimitiveBasics],
    parent_operand_auxes: &[TensorSpecAux<Tgt>],
    serial_only: bool,
    intermediate_tensor: Tensor<Tgt>,
    new_limits: MemoryLimits,
) -> SpecApp<ViewE<Tgt>> {
    let outer_components = &components[..(index + 1)];
    let outer_input_count = outer_components
        .iter()
        .map(|c| c.typ.input_count() - 1)
        .sum::<usize>();
    let mut outer_operand_auxes = vec![];
    outer_operand_auxes.reserve_exact(outer_input_count + 2);
    outer_operand_auxes.extend_from_slice(&parent_operand_auxes[..outer_input_count]);
    let insertion_point =
        1 + outer_operand_auxes.len() - outer_components.last().unwrap().typ.input_count();
    outer_operand_auxes.insert(insertion_point, intermediate_tensor.0.aux.clone());
    outer_operand_auxes.push(parent_operand_auxes.last().unwrap().clone());
    let mut outer_compose = Spec(
        match outer_components {
            [] => unreachable!("should never be empty"),
            [single] => LogicalSpec::Primitive(single.clone(), outer_operand_auxes, serial_only),
            _ => LogicalSpec::Compose {
                components: outer_components.into(),
                operand_auxes: outer_operand_auxes,
                serial_only,
            },
        },
        new_limits.clone(),
    );
    outer_compose.canonicalize().unwrap();

    // TODO: Wrap in SpecApp
    let mut params = vec![];
    params.reserve_exact(outer_input_count + 2);
    params.extend((0..outer_input_count).map(|i| {
        ViewE::from(Param::new(i.try_into().unwrap(), {
            outer_compose.0.parameter(i)
        }))
    }));
    params.insert(insertion_point, ViewE::from(intermediate_tensor));
    params.push(ViewE::from(Param::new(
        (parent_operand_auxes.len() - 1).try_into().unwrap(),
        outer_compose.0.unique_output().unwrap(),
    )));
    SpecApp::new(outer_compose, params)
}

fn plan_movelet<'a, Tgt: Target>(
    spec: &Spec<Tgt>,
    operands: &'a [TensorSpec<Tgt>],
    source_idx: u8,
    destination_dtype: Dtype,
    destination_level: Tgt::Level,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> Result<MoveLetPlan<'a, Tgt>, ApplyError> {
    let outer_moved_operand_spec = &operands[usize::from(source_idx)];

    if !outer_moved_operand_spec.can_move_to(destination_layout, &destination_level) {
        return Err(ApplyError::NotApplicable(
            // TODO: Replace Other with a new NotApplicableReason variant.
            NotApplicableReason::Other(None),
        ));
    }

    if let Some(vs) = destination_vector_size {
        if !Tgt::vec_types().iter().any(|vec_type| {
            vec_type.dtype == destination_dtype && u32::from(vec_type.value_cnt) == vs.get()
        }) {
            return Err(ApplyError::NotApplicable(
                NotApplicableReason::VectorSizeInvalid(destination_dtype, vs),
            ));
        }
    }

    let destination_layout_canonicalized = destination_layout
        .canonicalize(outer_moved_operand_spec.shape())
        .unwrap();
    let mut new_spec = TensorSpec::<Tgt>::new_noncanon(
        outer_moved_operand_spec.shape().into(),
        destination_dtype,
        outer_moved_operand_spec.contiguous_abs(),
        outer_moved_operand_spec.aligned(),
        destination_level,
        destination_layout_canonicalized,
        destination_vector_size,
    );

    let is_cache_miss = move_is_cache_miss(
        outer_moved_operand_spec,
        destination_dtype,
        &destination_level,
        new_spec.layout(),
    );
    // If this is anything other than a simple cache miss, a new buffer will be allocated, so that
    // buffer will be aligned and fully contiguous.
    if !is_cache_miss {
        new_spec.set_aligned(true);
        new_spec.set_contiguous_abs(new_spec.layout().contiguous_full());
        new_spec.canonicalize().unwrap();
    }
    debug_assert_eq!(
        is_cache_miss,
        move_is_cache_miss(
            outer_moved_operand_spec,
            new_spec.dtype(),
            &new_spec.level(),
            new_spec.layout(),
        ),
        "simple_cache_miss changes after updating alignment and contiguousness"
    );

    assert!(
        new_spec.layout().applies_to_shape(new_spec.shape()),
        "Destination layout {:?} does not apply to shape {:?}",
        new_spec.layout(),
        new_spec.shape()
    );

    // Filter cases where, after canonicalization, the source and destination
    // TensorSpecs match (i.e., within-level copies).
    if outer_moved_operand_spec == &new_spec {
        return Err(ApplyError::NotApplicable(NotApplicableReason::SelfMove));
    }

    let lower_limits: MemoryLimits = {
        let additional = u64::from(destination_dtype.size())
            * u64::from(operands[usize::from(source_idx)].volume().get());
        match &spec.1 {
            MemoryLimits::Standard(base) => {
                let levels = Tgt::levels();
                let updated_level_idx =
                    levels.iter().position(|l| l == &destination_level).unwrap();
                let mut new_limits = base.clone();
                let Some(level_updated) = new_limits
                    .get_unscaled(updated_level_idx)
                    .checked_sub(additional)
                else {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                        levels[updated_level_idx].to_string(),
                    )));
                };
                new_limits.set_unscaled(updated_level_idx, prev_power_of_two(level_updated));
                MemoryLimits::Standard(new_limits)
            }
        }
    };

    // Closure which makes a prologue or epilogue sub-Spec.
    let make_logue = |flip, f: &dyn Fn(_, _, _) -> bool| {
        if f(source_idx, &spec.0, is_cache_miss) {
            let mut left_spec = outer_moved_operand_spec;
            let mut right_spec = &new_spec;
            if flip {
                mem::swap(&mut left_spec, &mut right_spec);
            }
            let mut logue_spec = Spec(
                LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Move,
                        spec_shape: left_spec.shape().into(),
                        dtypes: vec![left_spec.dtype(), right_spec.dtype()],
                    },
                    vec![left_spec.aux.clone(), right_spec.aux.clone()],
                    spec.0.serial_only(),
                ),
                lower_limits.clone(),
            );
            logue_spec.canonicalize().unwrap();
            Some(logue_spec)
        } else {
            None
        }
    };
    let prologue_spec = make_logue(false, &move_gens_prologue);
    let epilogue_spec = make_logue(true, &move_gens_epilogue);

    let mut new_operands = operands.to_vec();
    new_operands[usize::from(source_idx)] = new_spec.clone();
    let mut new_body_spec = Spec(spec.0.clone(), lower_limits);
    new_body_spec.0.replace_io(&new_operands);
    new_body_spec.canonicalize().unwrap();

    Ok(MoveLetPlan {
        outer_moved_operand_spec,
        new_spec,
        prologue_spec,
        epilogue_spec,
        new_body_spec,
        new_operands,
        is_cache_miss,
    })
}

fn move_gens_prologue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    let parameters = logical_spec.parameters();
    let is_output = usize::from(source_idx) == parameters.len() - 1;
    let is_read = !is_output || logical_spec.output_is_read();
    is_read && !is_cache_miss
}

fn move_gens_epilogue<Tgt: Target>(
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
    is_cache_miss: bool,
) -> bool {
    let source_idx_usize = usize::from(source_idx);
    let is_output = source_idx_usize == logical_spec.operand_count() - 1;
    is_output && !is_cache_miss
}

/// Returns `true` if the move is a simple cache miss.
///
/// This is true if the destination is a hardware cache and the layout and data type are
/// unchanged.
fn move_is_cache_miss<Tgt: Target>(
    operand: &TensorSpec<Tgt>,
    destination_dtype: Dtype,
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
) -> bool {
    !destination_level.is_addressed()
        && operand.layout() == destination_layout
        && operand.dtype() == destination_dtype
}

/// Converts an internal [TileError] to an external [ApplyError].
fn tile_to_apply_err(err: TileError) -> ApplyError {
    match err {
        TileError::LayoutIncompatible(_) => {
            ApplyError::NotApplicable(NotApplicableReason::LayoutIncompatible)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layout::{col_major, row_major, PhysDim},
        lspec,
        memorylimits::MemVec,
        shape,
        spec::arb_canonical_spec,
        target::{CpuMemoryLevel, X86Target},
    };
    use proptest::prelude::*;

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_solver() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            action.solver(spec).unwrap().subspecs().collect()
        })
    }

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_apply() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            child_impls_into_specs(&action.apply(spec).unwrap())
        })
    }

    #[test]
    fn test_subspecs_when_moving_into_degenerate_packed_layout_apply_unchecked() {
        shared_test_subspecs_when_moving_into_degenerate_packed_layout(|spec, action| {
            child_impls_into_specs(&action.apply_unchecked_canon(spec).unwrap())
        })
    }

    // TODO: Add a variant where only physically innermost dimension is contiguous.
    #[test]
    fn test_move_planning_into_cache_with_extra_degenerate_dims_preserves_layout_and_contig() {
        let fixed_layout = col_major(2);
        let degenerate_layout = Layout::new(vec![
            (0, PhysDim::Dynamic),
            (1, PhysDim::Dynamic),
            (0, PhysDim::Packed(nz!(8u32))),
        ]);
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [8, 128, 8],
            (f32, CpuMemoryLevel::GL, fixed_layout.clone()),
            (f32, CpuMemoryLevel::GL, row_major(2)),
            (f32, CpuMemoryLevel::GL, row_major(2))
        ));
        let spec = Spec(logical_spec, X86Target::max_mem());
        let parameters = spec.0.parameters();
        let plan = plan_movelet(
            &spec,
            &parameters,
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            &degenerate_layout,
            None,
        )
        .unwrap();
        assert_eq!(plan.new_spec.layout(), &fixed_layout);
        assert_eq!(
            plan.new_spec.contiguous_abs(),
            fixed_layout.contiguous_full()
        );
    }

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
            output_shape: shape![4, 6],
            parallel: false,
        }))
    }

    /// Test that a TileOut::Split with a non-multiple size fails to apply.
    ///
    /// These aren't implemented yet in Impl, so the action should fail.
    #[test]
    fn test_non_multiple_split_returns_error() {
        shared_test_non_multiple_tiling_returns_error(Action::Split { k: nz!(6u32) })
    }

    /// Test that bufferizing a chain of 3 Matmuls produces the correct sub-Spec applications.
    #[test]
    fn test_bufferize_matmul_chain() {
        let basics0 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![1, 2, 4],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let basics1 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![1, 4, 8],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let basics2 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![1, 8, 16],
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let aux = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: true,
            level: CpuMemoryLevel::GL,
            layout: row_major(2),
            vector_size: None,
        };
        let mut spec = Spec::<X86Target>(
            LogicalSpec::Compose {
                components: vec![basics2.clone(), basics1.clone(), basics0.clone()],
                operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux.clone(), aux],
                serial_only: true,
            },
            X86Target::max_mem(),
        );
        spec.canonicalize().unwrap();
        let imp = spec.bufferize(1, CpuMemoryLevel::GL, row_major(2), None);

        assert_eq!(imp.children().len(), 2);
        assert!(matches!(imp, ImplNode::Pipeline(_)));

        let ImplNode::SpecApp(SpecApp(Spec(first_child_lspec, _), first_child_params)) =
            &imp.children()[0]
        else {
            panic!("expected a SpecApp child, got {:?}", imp.children()[0])
        };
        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: first_child_shape,
                dtypes: first_child_dtypes,
            },
            _,
            true,
        ) = first_child_lspec
        else {
            panic!("expected a serial, non-accum Matmul Primitive, got {first_child_lspec:?}");
        };
        assert_eq!(first_child_shape, &basics0.spec_shape);
        assert_eq!(first_child_dtypes, &basics0.dtypes);
        assert_eq!(first_child_params.len(), 3);
        assert!(matches!(
            first_child_params[0],
            ViewE::Param(Param(2, _, _))
        ));
        assert!(matches!(
            first_child_params[1],
            ViewE::Param(Param(3, _, _))
        ));
        assert!(matches!(first_child_params[2], ViewE::Tensor(_)));

        let ImplNode::SpecApp(SpecApp(Spec(second_child_lspec, _), second_child_params)) =
            &imp.children()[1]
        else {
            panic!("expected a SpecApp child, got {:?}", imp.children()[1])
        };
        let LogicalSpec::Compose {
            components,
            operand_auxes: _,
            serial_only: true,
        } = second_child_lspec
        else {
            panic!("expected a serial Compose LogicalSpec, got {second_child_lspec:?}");
        };
        assert_eq!(components, &[basics2.clone(), basics1.clone()]);
        assert_eq!(second_child_params.len(), 4);
        assert!(matches!(
            second_child_params[0],
            ViewE::Param(Param(0, _, _))
        ));
        assert!(matches!(second_child_params[1], ViewE::Tensor(_)));
        assert!(matches!(
            second_child_params[2],
            ViewE::Param(Param(1, _, _))
        ));
        assert!(matches!(
            second_child_params[3],
            ViewE::Param(Param(4, _, _))
        ));
    }

    fn shared_test_non_multiple_tiling_returns_error(action: Action<X86Target>) {
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [8, 8, 8],
            (f32, CpuMemoryLevel::GL, col_major(2)),
            (f32, CpuMemoryLevel::GL, row_major(2)),
            (f32, CpuMemoryLevel::GL, row_major(2))
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

    fn child_impls_into_specs(imp: &ImplNode<X86Target>) -> Vec<Spec<X86Target>> {
        imp.children()
            .iter()
            .map(|child| {
                if let ImplNode::SpecApp(SpecApp(spec, _)) = child {
                    spec.clone()
                } else {
                    panic!("expected a SpecApp child, got {:?}", child)
                }
            })
            .collect()
    }

    fn shared_test_subspecs_when_moving_into_degenerate_packed_layout(
        child_get: impl FnOnce(&Spec<X86Target>, Action<X86Target>) -> Vec<Spec<X86Target>>,
    ) {
        let logical: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [8, 128, 8],
            (f32, CpuMemoryLevel::GL, col_major(2)),
            (f32, CpuMemoryLevel::GL, row_major(2)),
            (f32, CpuMemoryLevel::GL, row_major(2))
        ));
        let spec = Spec(logical, X86Target::max_mem());
        let action = Action::Move {
            source_idx: 0,
            destination_dtype: Dtype::Float32,
            destination_level: CpuMemoryLevel::L1,
            destination_layout: Layout::new(vec![
                (0, PhysDim::Dynamic),
                (1, PhysDim::Dynamic),
                (0, PhysDim::Packed(nz!(8u32))),
            ]),
            destination_vector_size: None,
        };
        match child_get(&spec, action).as_slice() {
            [Spec(logical_spec, _)] => {
                assert!(matches!(
                    logical_spec,
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Matmul { .. },
                            ..
                        },
                        ..
                    )
                ));
            }
            children => panic!("expected one Matmul Spec child, got {:?}", children),
        };
    }

    proptest! {
        // TODO: Add an ARM variant
        #[test]
        fn test_fast_path_is_equivalent_to_slow(spec in arb_canonical_spec::<X86Target>(None, None)) {
            for action in X86Target::actions(&spec.0, None) {
                match (action.solver(&spec), action.apply(&spec)) {
                    (Ok(solver), Ok(applied)) => {
                        let subspecs = solver.subspecs().collect::<Vec<_>>();
                        let mut applied_subspecs = Vec::new();
                        collect_nested_specs(&applied, &mut applied_subspecs);
                        prop_assert_eq!(&subspecs, &applied_subspecs);

                        // Generate some quick-n'-dirty sub-Spec costs and confirm that the fast
                        // and slow paths yield the same final cost.
                        let subspec_costs = (0..u8::try_from(subspecs.len()).unwrap())
                            .map(|subspec_idx| {
                                Cost {
                                    main: subspec_idx.into(),
                                    peaks: MemVec::zero::<X86Target>(),
                                    depth: subspec_idx,
                                }
                            })
                            .collect::<Vec<_>>();
                        let solver_cost = solver.compute_cost(subspec_costs.iter().cloned());
                        let applied_cost = compute_impl_cost(&applied, &mut subspec_costs.into_iter());
                        prop_assert_eq!(solver_cost, applied_cost);
                    },
                    (Err(a), Err(b)) => prop_assert_eq!(a, b),
                    (l, r) => {
                        prop_assert!(false, "solver returned {l:?} but apply returned {r:?}");
                    }
                }
            }
        }
    }
}
