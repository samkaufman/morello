use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::fmt::Display;
use std::rc::Rc;
use std::{iter, mem};

use crate::alignment::aligned_approx;
use crate::common::{DimSize, Dtype, Shape};
use crate::imp::blocks::Block;
use crate::imp::kernels::KernelApp;
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::moves::{MoveLet, TensorOrCacheView};
use crate::imp::pipeline::Pipeline;
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::memorylimits::MemoryLimits;
use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::TensorSpec;
use crate::tiling::Tiling;
use crate::utils::prev_power_of_two;
use crate::views::{CacheView, Param, Tensor, Tile, TileError, View, ViewExt};

/// A scheduling decision which can be applied to a Spec to produce an Impl.
///
/// [Action]s contain the minimal amount of information needed to distinguish a one scheduling
/// decision from another, which makes it appropriate for storing in a database so that the
/// corresponding Impl node can be computed given the Spec.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[cfg_attr(test, derive(Hash))]
pub enum Action<Tgt: Target> {
    TileOut {
        output_shape: Shape,
        parallel: bool,
    },
    Split {
        k: DimSize,
    },
    Move {
        source_idx: u8,
        destination_dtype: Dtype,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    },
    ToAccum,
    Peel {
        layout: Layout,
        level: Tgt::Level,
        vector_size: Option<DimSize>,
    },
    SpatialSplit,
    Place(Tgt::Kernel),
}

#[derive(thiserror::Error, Debug)]
pub enum ApplyError {
    #[error("Cannot apply action to non-canonical Spec")]
    SpecNotCanonical,
    #[error("Insufficient memory to apply action")]
    OutOfMemory,
    #[error("Action does not apply to this Spec: {0}")]
    ActionNotApplicable(ActionNotApplicableReason),
}

#[derive(Debug)]
pub enum ActionNotApplicableReason {
    TileShapeInvalid,
    LayoutIncompatible,
    SelfMove,
    Other,
}

impl<Tgt: Target> Action<Tgt> {
    pub fn child_count(&self) -> usize {
        match self {
            Action::TileOut { .. } => 1,
            Action::Split { .. } => 1,
            Action::ToAccum => 2,
            Action::SpatialSplit => 1,
            Action::Place(_) => 0,
            Action::Move { .. } => unimplemented!(),
            Action::Peel { .. } => 2,
        }
    }

    pub fn apply(&self, spec: &Spec<Tgt>) -> Result<ImplNode<Tgt>, ApplyError> {
        if !spec.is_canonical() {
            return Err(ApplyError::SpecNotCanonical);
        }

        let logical_spec = &spec.0;
        let operands = logical_spec.parameters();

        match self {
            Action::TileOut { .. } | Action::Split { .. } => {
                // TODO: Refactor this huge case body into flattened cases for TileOut and Split.
                let (tiles, parallel) = {
                    match self {
                        Action::TileOut {
                            output_shape,
                            parallel,
                        } => {
                            let current_output = &operands[logical_spec.output_idx()];

                            let current_out_shape = current_output.shape();
                            assert!(
                                !(*parallel && logical_spec.serial_only()),
                                "Serial-only Spec prevents parallel tiling"
                            );
                            assert_eq!(
                                output_shape.len(),
                                current_out_shape.len(),
                                "Expected {} dimensions; got {}",
                                current_out_shape.len(),
                                output_shape.len()
                            );
                            assert!(output_shape
                                .iter()
                                .enumerate()
                                .all(|(dim, dim_size)| *dim_size <= current_out_shape[dim]));
                            assert_ne!(
                                current_out_shape,
                                &output_shape[..],
                                "Cannot tile to same shape: {:?}",
                                output_shape
                            );

                            // Abort if it's invalid to tile the original output tensor
                            // to the new shape (e.g., the new shape is larger).
                            if !current_output.is_valid_tile_shape(output_shape, *parallel) {
                                return Err(ApplyError::ActionNotApplicable(
                                    ActionNotApplicableReason::TileShapeInvalid,
                                ));
                            }

                            // Tiling happens in three steps:
                            // 1. Construct the simple tile corresponding to the new output shape.
                            let out_idx: u8 = logical_spec.output_idx().try_into().unwrap();
                            let smaller_output_tiling = Tiling::new_simple(output_shape.clone());
                            let smaller_output = LoopTile {
                                axes: (0..u8::try_from(output_shape.len()).unwrap()).collect(),
                                tile: smaller_output_tiling
                                    .apply(Param::new(out_idx, current_output.clone()))
                                    .map_err(tile_to_apply_err)?,
                            };

                            // 2. Construct tilings which respect the data deps. of the new output tile.
                            let updated_input_tilings =
                                logical_spec.input_tilings_for_tile_out(&smaller_output_tiling);

                            // 3. Reify the tilings into Tiles we'll store with this action. Tiles
                            //    objects track the index and shape of the Impl parameter being
                            //    tiled.
                            let mut next_fresh_loop_dim = u8::try_from(output_shape.len()).unwrap();
                            let mut new_tiles: Vec<LoopTile<Tgt>> = vec![];
                            for (
                                operand_idx,
                                (original_input, (updated_input_tiling, updated_input_axes)),
                            ) in operands.iter().zip(updated_input_tilings.0).enumerate()
                            {
                                if operand_idx == logical_spec.output_idx() {
                                    continue;
                                }

                                let tiling_shape = updated_input_tiling.shape();
                                if !original_input.is_valid_tile_shape(tiling_shape, *parallel) {
                                    return Err(ApplyError::ActionNotApplicable(
                                        ActionNotApplicableReason::TileShapeInvalid,
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
                                    new_tiles.push(LoopTile {
                                        axes,
                                        tile: updated_input_tiling
                                            .apply(Param::new(
                                                operand_idx.try_into().unwrap(),
                                                original_input.clone(),
                                            ))
                                            .map_err(tile_to_apply_err)?,
                                    });
                                }
                            }
                            new_tiles.push(smaller_output);
                            (new_tiles, *parallel)
                        }
                        Action::Split { k } => match logical_spec {
                            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => {
                                match typ {
                                    PrimitiveSpecType::Matmul { accum: _ } => {
                                        let [lhs, rhs] = &operands[..2] else {
                                            panic!();
                                        };
                                        assert!(*k < lhs.shape()[1]);

                                        let tiles = vec![
                                            LoopTile {
                                                axes: vec![0, 1],
                                                tile: Tile::new(
                                                    vec![lhs.shape()[0], *k],
                                                    vec![lhs.shape()[0], *k],
                                                    Param::new(0, lhs.clone()),
                                                )
                                                .map_err(tile_to_apply_err)?,
                                            },
                                            LoopTile {
                                                axes: vec![1, 2],
                                                tile: Tile::new(
                                                    vec![*k, rhs.shape()[1]],
                                                    vec![*k, rhs.shape()[1]],
                                                    Param::new(1, rhs.clone()),
                                                )
                                                .map_err(tile_to_apply_err)?,
                                            },
                                        ];
                                        (tiles, false)
                                    }
                                    _ => unimplemented!("Split not implemented for {:?}", typ),
                                }
                            }
                            LogicalSpec::Compose { .. } => todo!(),
                        },
                        _ => unreachable!(),
                    }
                };

                let mut new_operands = operands;
                for loop_tile in &tiles {
                    let ref_op = &mut new_operands[usize::from(loop_tile.tile.view.0)];
                    let aligned =
                        aligned_approx(loop_tile.tile.shape(), loop_tile.tile.step_sizes(), ref_op)
                            .unwrap();
                    ref_op.shrink(loop_tile.tile.shape(), aligned).unwrap();
                }

                let mut inner_spec = logical_spec.clone();
                inner_spec.replace_io(&new_operands);
                inner_spec.set_serial_only(inner_spec.serial_only() || parallel);
                inner_spec.canonicalize().unwrap();
                match self {
                    Action::TileOut { .. } => {}
                    Action::Split { .. } => {
                        if !matches!(
                            &inner_spec,
                            LogicalSpec::Primitive(
                                PrimitiveBasics {
                                    typ: PrimitiveSpecType::Matmul { accum: true },
                                    ..
                                },
                                ..
                            )
                        ) {
                            // TODO: Should return an error instead?
                            panic!("Can only split an accumulating Matmul");
                        };
                    }
                    _ => unreachable!(),
                };
                let body = Box::new(SpecApp::default_app(Spec(inner_spec, spec.1.clone())).into());

                Ok(ImplNode::Loop(Loop {
                    tiles,
                    body,
                    parallel,
                    spec: Some(spec.clone()),
                }))
            }
            Action::Peel {
                layout,
                level,
                vector_size,
            } => {
                let LogicalSpec::Compose {
                    components,
                    operand_auxes,
                    serial_only,
                } = logical_spec
                else {
                    panic!();
                };
                debug_assert!(components.len() >= 2);

                // Determine the output shape of the next-to-outermost component Spec.
                // This is the shape of the intermediate tensor.
                let next_to_outer_basics = &components[1];
                let out_idx = next_to_outer_basics.typ.output_idx();
                let intermediate_tensorspec = TensorSpec::<Tgt>::new_canon(
                    next_to_outer_basics.parameter_shapes().swap_remove(out_idx),
                    next_to_outer_basics.dtypes[out_idx],
                    layout.contiguous_full(),
                    true,
                    *level,
                    layout.clone(),
                    *vector_size,
                );
                let intermediate_tensor = Rc::new(Tensor::new(intermediate_tensorspec.clone()));

                // The head of a Compose is the final function evaluated. Build
                // a full Spec so that it can be further scheduled independently.
                let external_head_input_cnt = components[0].typ.input_count() - 1;
                let head_spec = {
                    let head_operand_auxes = iter::once(&intermediate_tensorspec.aux)
                        .chain(&operand_auxes[..external_head_input_cnt])
                        .chain(iter::once(&operand_auxes[logical_spec.output_idx()]));
                    let head_basics = &components[0];
                    LogicalSpec::Primitive(
                        head_basics.clone(),
                        head_basics.aux_from_operand_auxes(head_operand_auxes),
                        *serial_only,
                    )
                };

                // The "remainder" (inner/first) portion of the pipeline will be
                // a primitive Spec or a smaller Compose, depending on the initial
                // length of the Compose.
                let next_to_head_input_auxes = &operand_auxes[external_head_input_cnt
                    ..external_head_input_cnt + next_to_outer_basics.typ.input_count()];
                let remainder: LogicalSpec<Tgt> = if components.len() == 2 {
                    LogicalSpec::Primitive(
                        next_to_outer_basics.clone(),
                        next_to_outer_basics.aux_from_operand_auxes(
                            next_to_head_input_auxes
                                .iter()
                                .chain(iter::once(&intermediate_tensorspec.aux)),
                        ),
                        *serial_only,
                    )
                } else {
                    let remainder_inputs =
                        &operands[external_head_input_cnt..next_to_head_input_auxes.len() - 1];
                    let remainder_operand_auxes = remainder_inputs
                        .iter()
                        .map(|t| t.aux.clone())
                        .chain(iter::once(intermediate_tensorspec.aux))
                        .collect::<Vec<_>>();
                    LogicalSpec::Compose {
                        components: components[1..].to_vec(),
                        operand_auxes: remainder_operand_auxes,
                        serial_only: *serial_only,
                    }
                };

                // Compute the memory limits for the two new children.
                let new_limits = {
                    // Compute the amount of memory consumed by the new, intermediate
                    // tensor.
                    // TODO: This shouldn't need to be both here and in `memory_allocated`.
                    let next_to_outer_basics = &components[1];
                    let ntob_out_idx = next_to_outer_basics.typ.output_idx();
                    let output_shape = &next_to_outer_basics.parameter_shapes()[ntob_out_idx];
                    let intermediate_mem_consumed_nondiscrete = Tgt::levels().map(|l| {
                        if level == &l {
                            u64::from(next_to_outer_basics.dtypes[ntob_out_idx].size())
                                * u64::from(output_shape.iter().map(|d| d.get()).product::<u32>())
                        } else {
                            0u64
                        }
                    });

                    // TODO: Use MemoryLimits::Pipeline where appropriate instead.
                    let mut m = MemoryLimits::Standard(match &spec.1 {
                        MemoryLimits::Standard(v) => {
                            let Some(r) = v
                                .clone()
                                .checked_sub_snap_down(&intermediate_mem_consumed_nondiscrete)
                            else {
                                return Err(ApplyError::OutOfMemory);
                            };
                            r
                        }
                    });
                    m.discretize();
                    m
                };

                // Reify the new Specs and TensorSpecs into applications we can
                // nest in the Pipeline body.
                let remainder_spec_application = {
                    let mut params: Vec<Rc<dyn View<Tgt = Tgt>>> = vec![];
                    params.extend(remainder.inputs().iter().enumerate().map(|(i, inp)| {
                        Rc::new(Param::new(i.try_into().unwrap(), inp.clone())) as _
                    }));
                    params.push(Rc::new(intermediate_tensor.clone()) as _);
                    ImplNode::SpecApp(SpecApp::new(Spec(remainder, new_limits.clone()), params))
                };
                let head_spec_application = {
                    let mut params: Vec<Rc<dyn View<Tgt = Tgt>>> = vec![];
                    // TODO: Fill in.
                    params.extend(head_spec.parameters().iter().skip(1).map(|_operand| {
                        todo!();
                    }));
                    ImplNode::SpecApp(SpecApp::new(Spec(head_spec, new_limits), params))
                };

                Ok(ImplNode::Pipeline(Pipeline {
                    intermediates: vec![intermediate_tensor],
                    stages: vec![remainder_spec_application, head_spec_application],
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
                    panic!();
                };
                if !*conv_accum {
                    panic!("Can only spatially split accumulating convolutions");
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
                            axes: vec![7, 8, 0, 1],
                            tile: outer_image_tile,
                        },
                        LoopTile {
                            axes: vec![9, 8, 0, 1],
                            tile: outer_filters_tile,
                        },
                    ],
                    body: Box::new(
                        SpecApp::new(
                            Spec(body_spec, spec.1.clone()),
                            [
                                Rc::new(inner_image_tile) as Rc<dyn View<Tgt = Tgt>>,
                                Rc::new(inner_filters_tile) as Rc<dyn View<Tgt = Tgt>>,
                                Rc::new(inner_output_view) as Rc<dyn View<Tgt = Tgt>>,
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
                let outer_moved_operand_spec = &operands[usize::from(*source_idx)];

                let new_spec = movelet_inner_tensorspec(
                    outer_moved_operand_spec,
                    *destination_dtype,
                    destination_level,
                    destination_layout,
                    *destination_vector_size,
                );

                assert!(
                    destination_layout.applies_to_shape(new_spec.shape()),
                    "Destination layout {:?} does not apply to shape {:?}",
                    destination_layout,
                    new_spec.shape()
                );

                // Filter cases where, after canonicalization, the source and destination
                // TensorSpecs match (i.e., within-level copies).
                if outer_moved_operand_spec == &new_spec {
                    return Err(ApplyError::ActionNotApplicable(
                        ActionNotApplicableReason::SelfMove,
                    ));
                }

                let inner_moved_operand = if move_is_cache_miss(
                    outer_moved_operand_spec,
                    *destination_dtype,
                    destination_level,
                    destination_layout,
                ) {
                    let source = Param::new(*source_idx, outer_moved_operand_spec.clone());
                    TensorOrCacheView::CacheView(Rc::new(CacheView::new(source, new_spec)))
                } else {
                    TensorOrCacheView::Tensor(Rc::new(Tensor::new(new_spec)))
                };

                let lower_limits: MemoryLimits = {
                    let additional = u64::from(destination_dtype.size())
                        * u64::from(operands[usize::from(*source_idx)].volume().get());
                    match &spec.1 {
                        MemoryLimits::Standard(base) => {
                            let updated_level_idx = Tgt::levels()
                                .iter()
                                .position(|l| l == destination_level)
                                .unwrap();
                            let mut new_limits = base.clone();
                            let Some(level_updated) = new_limits
                                .get_unscaled(updated_level_idx)
                                .checked_sub(additional)
                            else {
                                return Err(ApplyError::OutOfMemory);
                            };
                            new_limits
                                .set_unscaled(updated_level_idx, prev_power_of_two(level_updated));
                            MemoryLimits::Standard(new_limits)
                        }
                    }
                };

                // Closure which makes a prologue or epilogue for this Spec.
                let make_logue = |flip, f: &dyn Fn(_, _, _, _, _) -> bool| {
                    if f(
                        destination_level,
                        destination_layout,
                        *destination_dtype,
                        *source_idx,
                        logical_spec,
                    ) {
                        let mut left_spec = outer_moved_operand_spec;
                        let mut right_spec = inner_moved_operand.spec();
                        let mut args: [Rc<dyn View<Tgt = Tgt>>; 2] = [
                            Rc::new(Param::new(0, outer_moved_operand_spec.clone())) as _,
                            Rc::new(Param::new(1, inner_moved_operand.spec().clone())) as _,
                        ];
                        if flip {
                            mem::swap(&mut left_spec, &mut right_spec);
                            args.swap(0, 1);
                        }
                        let mut logue_spec = Spec(
                            LogicalSpec::Primitive(
                                PrimitiveBasics {
                                    typ: PrimitiveSpecType::Move,
                                    spec_shape: left_spec.shape().into(),
                                    dtypes: vec![left_spec.dtype(), right_spec.dtype()],
                                },
                                vec![left_spec.aux.clone(), right_spec.aux.clone()],
                                logical_spec.serial_only(),
                            ),
                            lower_limits.clone(),
                        );
                        logue_spec.canonicalize().unwrap();
                        Some(SpecApp::new(logue_spec, args))
                    } else {
                        None
                    }
                };
                let prologue = make_logue(false, &move_gens_prologue);
                let epilogue = make_logue(true, &move_gens_epilogue);

                let new_body_app = {
                    let mut new_operands = operands.clone();
                    new_operands[usize::from(*source_idx)] = inner_moved_operand.spec().clone();
                    let new_inner_spec = {
                        let mut new_spec = logical_spec.clone();
                        new_spec.replace_io(&new_operands);
                        new_spec
                    };
                    let mut spec = Spec(new_inner_spec, lower_limits.clone());
                    spec.canonicalize().unwrap();
                    let inner_operands = new_operands.iter().enumerate().map(|(i, o)| {
                        Rc::new(Param::new(u8::try_from(i).unwrap(), o.clone()))
                            as Rc<dyn View<Tgt = Tgt>>
                    });
                    SpecApp::new(spec, inner_operands)
                };

                Ok(ImplNode::MoveLet(MoveLet::new(
                    *source_idx,
                    outer_moved_operand_spec.clone(),
                    inner_moved_operand,
                    prologue.map(|i| i.into()),
                    new_body_app.into(),
                    epilogue.map(|i| i.into()),
                    Some(spec.clone()),
                )))
            }
            Action::ToAccum => {
                let LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, ..) = logical_spec else {
                    panic!();
                };
                let (PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum }) = typ
                else {
                    panic!();
                };
                if *accum {
                    panic!("Spec is already accumulating");
                }

                let TensorSpec {
                    shape: output_shape,
                    dtype: output_dtype,
                    aux: output_aux,
                } = logical_spec.output();

                let zero_app = {
                    let subspec = LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Zero,
                            spec_shape: output_shape,
                            dtypes: vec![output_dtype],
                        },
                        vec![output_aux],
                        logical_spec.serial_only(),
                    );
                    let mut spec = Spec(subspec, spec.1.clone());
                    spec.canonicalize()
                        .expect("ToAccum's introduced Zero should be canonicalizable");
                    let app_arguments = [Param::new(0, logical_spec.output())];
                    SpecApp::new(spec, app_arguments).into()
                };
                let accum_app = {
                    let mut spec = Spec(logical_spec.clone_as_accum(), spec.1.clone());
                    spec.canonicalize()
                        .expect("ToAccum's introduced accumulating Spec should be canonicalizable");
                    let app_arguments = operands
                        .iter()
                        .enumerate()
                        .map(|(i, t)| Param::new(i.try_into().unwrap(), t.clone()));
                    SpecApp::new(spec, app_arguments).into()
                };

                Ok(ImplNode::Block(Block {
                    stages: vec![zero_app, accum_app],
                    bindings: vec![vec![2], vec![0, 1, 2]],
                    parameters: operands,
                    spec: Some(spec.clone()),
                }))
            }
            Action::Place(k) => {
                // TODO: Add: debug_assert!(k.applies_to_parameters(&operands));
                Ok(ImplNode::Kernel(KernelApp {
                    kernel_type: *k,
                    arguments: operands
                        .iter()
                        .enumerate()
                        .map(|(i, p)| Param::new(i.try_into().unwrap(), p.clone()))
                        .collect(),
                    spec: Some(spec.clone()),
                }))
            }
        }
    }
}

impl Display for ActionNotApplicableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionNotApplicableReason::TileShapeInvalid => {
                write!(f, "Invalid tile shape")
            }
            ActionNotApplicableReason::LayoutIncompatible => {
                write!(f, "Layout does not apply to tile size")
            }
            ActionNotApplicableReason::SelfMove => {
                write!(
                    f,
                    "Source and destination TensorSpecs were equal after canonicalization"
                )
            }
            ActionNotApplicableReason::Other => write!(f, "Unknown reason"),
        }
    }
}

fn move_gens_prologue<Tgt: Target>(
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
    destination_dtype: Dtype,
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
) -> bool {
    let source_idx_usize = usize::from(source_idx);
    let parameters = logical_spec.parameters();
    let is_output = usize::from(source_idx) == parameters.len() - 1;
    let is_read = !is_output || logical_spec.output_is_read();
    is_read
        && !move_is_cache_miss(
            &parameters[source_idx_usize],
            destination_dtype,
            destination_level,
            destination_layout,
        )
}

fn move_gens_epilogue<Tgt: Target>(
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
    destination_dtype: Dtype,
    source_idx: u8,
    logical_spec: &LogicalSpec<Tgt>,
) -> bool {
    let source_idx_usize = usize::from(source_idx);
    let parameters = logical_spec.parameters();
    let is_output = source_idx_usize == logical_spec.operand_count() - 1;
    is_output
        && !move_is_cache_miss(
            &parameters[source_idx_usize],
            destination_dtype,
            destination_level,
            destination_layout,
        )
}

pub(crate) fn movelet_inner_tensorspec<Tgt: Target>(
    operand: &TensorSpec<Tgt>,
    destination_dtype: Dtype,
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
    destination_vector_size: Option<DimSize>,
) -> TensorSpec<Tgt> {
    let simple_cache_miss = move_is_cache_miss(
        operand,
        destination_dtype,
        destination_level,
        destination_layout,
    );

    // When moving into an addressed bank, we'll generate an aligned destination.
    // If it's into a cache level, alignment won't change.
    let aligned = if !simple_cache_miss {
        true
    } else {
        operand.aligned()
    };

    // Will the result be contiguous? If the move is into a cache, it might be.
    // If it's into memory bank with its own address space, then yes.
    let contiguous_abs = if !simple_cache_miss {
        destination_layout.contiguous_full()
    } else {
        operand.contiguous_abs()
    };

    TensorSpec::<Tgt>::new_canon(
        operand.shape().into(),
        destination_dtype,
        contiguous_abs,
        aligned,
        *destination_level,
        destination_layout.clone(),
        destination_vector_size,
    )
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
            ApplyError::ActionNotApplicable(ActionNotApplicableReason::LayoutIncompatible)
        }
    }
}
