use crate::common::Dtype;
use crate::common::{DimSize, Shape};
use crate::cost::NormalizedCost;
use crate::db::DbKey;
use crate::grid::general::BiMap;
use crate::imp::loops::{Loop, LoopTile};
use crate::imp::subspecs::SpecApp;
use crate::imp::ImplNode;
use crate::layout::row_major;
use crate::scheduling::{
    check_tile_out_applies, collect_nested_specs, tile_to_apply_err, Action, ActionT,
    ActionTopDownSolver, ApplyError, BottomUpSolver, NaiveBottomUpActionProvider,
    NaiveBottomUpSolver, NotApplicableReason, PrimitiveTileOutSolver, VisitUpdater,
};
use crate::spec::{
    CanonicalizeError, FillValue, LogicalSpec, LogicalSpecInputTilingInference, PrimitiveBasics,
    PrimitiveSpecType, Spec,
};
use crate::target::common_actions::{split_actions, tile_out_actions};
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::views::{BoundaryTile, Param, Tile, View, ViewE};
use itertools::{izip, Either, Itertools};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::iter;
use std::iter::once;
use std::num::{NonZeroU32, NonZeroUsize};

/// Whether `tile_out` actions should tile in all dimensions per Spec.
pub(crate) const MULTI_DIM_TILING: bool = false;

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
            TileOut::SingleLoop { parallel, .. } => *parallel,
            TileOut::MultiLoop { parallel, .. } => *parallel,
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
                .apply_main(Param::new(output_idx_u8, current_output.clone()))
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
            input_tiles_for_tile_out(&operands, input_tilings.0, output_idx, parallel)?;
        new_tiles.push(smaller_output);
        tile_out_loop_spec_with_shrunken_tiles(component_input_tilings, new_tiles, parallel, spec)
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

                // Check if any dimension will create boundary regions
                // This happens when tile size doesn't evenly divide the untiled size
                let will_have_boundaries = tile_shape
                    .iter()
                    .zip(untiled_output_shape.iter())
                    .any(|(tile_size, untiled_size)| untiled_size.get() % tile_size.get() != 0);

                match basics.typ {
                    PrimitiveSpecType::Matmul { .. } => {
                        if will_have_boundaries {
                            // TODO: Speed up this path.
                            let slow_path_impl = self.apply_unchecked_canon(spec)?;
                            let mut slow_path_subspecs = Vec::new();
                            collect_nested_specs(&slow_path_impl, &mut slow_path_subspecs);
                            return Ok(PrimitiveTileOutSolver {
                                outer_spec: spec.clone(),
                                body_specs: slow_path_subspecs,
                            }
                            .into());
                        } else {
                            let main_body_spec = ActionTopDownSolver::tiled_subspec_fast(
                                [(0, 0), (1, 1), (3, 2)].into_iter(),
                                spec,
                                &tile_shape,
                                parallel,
                            )?;

                            return Ok(PrimitiveTileOutSolver {
                                outer_spec: spec.clone(),
                                body_specs: vec![main_body_spec],
                            }
                            .into());
                        }
                    }
                    PrimitiveSpecType::Move
                    | PrimitiveSpecType::Fill {
                        value: FillValue::Zero,
                    } => {
                        if will_have_boundaries {
                            // TODO: Speed up this path.
                            let slow_path_impl = self.apply_unchecked_canon(spec)?;
                            let mut slow_path_subspecs = Vec::new();
                            collect_nested_specs(&slow_path_impl, &mut slow_path_subspecs);
                            return Ok(PrimitiveTileOutSolver {
                                outer_spec: spec.clone(),
                                body_specs: slow_path_subspecs,
                            }
                            .into());
                        } else {
                            let rank = basics.spec_shape.len();
                            let main_body_spec = ActionTopDownSolver::tiled_subspec_fast(
                                (0..rank).map(|i| (i, i)),
                                spec,
                                &tile_shape,
                                parallel,
                            )?;

                            return Ok(PrimitiveTileOutSolver {
                                outer_spec: spec.clone(),
                                body_specs: vec![main_body_spec],
                            }
                            .into());
                        }
                    }
                    _ => {}
                }
            }
            LogicalSpec::Compose { .. } => {}
        };

        self.apply_unchecked_canon(spec)
            .map(|applied| ActionTopDownSolver::Fallback(Box::new(applied)))
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
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
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

                    let main_lhs_shape = smallvec![lhs.shape()[0], lhs.shape()[1], self.k];
                    let main_rhs_shape = smallvec![rhs.shape()[0], self.k, rhs.shape()[2]];
                    let tiles = vec![
                        LoopTile {
                            parameter_index: 0,
                            axes: vec![0, 1, 2],
                            tile: Tiling::new_simple(main_lhs_shape.clone())
                                .apply_main(Param::new(0, lhs.clone()))
                                .map(|v| v.boxed_viewe())
                                .map_err(tile_to_apply_err)?,
                        },
                        LoopTile {
                            parameter_index: 1,
                            axes: vec![0, 2, 3],
                            tile: Tiling::new_simple(main_rhs_shape.clone())
                                .apply_main(Param::new(1, rhs.clone()))
                                .map(|v| v.boxed_viewe())
                                .map_err(tile_to_apply_err)?,
                        },
                    ];

                    // Construct the main body.
                    let mut main_body_logicalspec = spec.0.clone();
                    let main_body_component_parameter_shapes = vec![vec![
                        main_lhs_shape,
                        main_rhs_shape,
                        operands[2].shape().into(),
                    ]];
                    // TODO: Inline update_component_shapes?
                    update_component_shapes(
                        &mut main_body_logicalspec,
                        &main_body_component_parameter_shapes,
                    )?;
                    main_body_logicalspec
                        .canonicalize()
                        .map_err(spec_canonicalize_to_apply_err)?;
                    let main_body = SpecApp::new(
                        Spec(main_body_logicalspec, spec.1.clone()),
                        [
                            ViewE::Tile(tiles[0].tile.clone()),
                            ViewE::Tile(tiles[1].tile.clone()),
                            ViewE::Param(Param::new(2, operands[2].clone())),
                        ],
                    );

                    // Make the boundary body if k isn't a multiple. There's, at most, one.
                    let mut boundary_bodies = Vec::new();
                    if let Some(remainder) = DimSize::new(lhs.shape()[2].get() % self.k) {
                        let boundary_lhs_tiling = Tiling::new_simple(smallvec![
                            lhs.shape()[0],
                            lhs.shape()[1],
                            remainder
                        ]);
                        let boundary_rhs_tiling = Tiling::new_simple(smallvec![
                            rhs.shape()[0],
                            remainder,
                            rhs.shape()[2]
                        ]);

                        let mut boundary_body_logicalspec = spec.0.clone();
                        let boundary_component_parameter_shapes = vec![vec![
                            boundary_lhs_tiling.shape().clone(),
                            boundary_rhs_tiling.shape().clone(),
                            operands[2].shape().into(),
                        ]];
                        update_component_shapes(
                            &mut boundary_body_logicalspec,
                            &boundary_component_parameter_shapes,
                        )?;
                        boundary_body_logicalspec
                            .canonicalize()
                            .map_err(spec_canonicalize_to_apply_err)?;

                        let boundary_spec = Spec(boundary_body_logicalspec, spec.1.clone());
                        let boundary_body = SpecApp::new(
                            boundary_spec,
                            [
                                ViewE::BoundaryTile(convert_tile_to_boundarytile(
                                    boundary_lhs_tiling
                                        .apply_main(Param::new(0, operands[0].clone()))
                                        .map(|v| v.boxed_viewe())
                                        .map_err(tile_to_apply_err)?,
                                    nz!(4usize), // 1 << 2
                                    operands[0].shape(),
                                    &[Some(0), Some(1), None],
                                )),
                                ViewE::BoundaryTile(convert_tile_to_boundarytile(
                                    boundary_rhs_tiling
                                        .apply_main(Param::new(1, operands[1].clone()))
                                        .map(|v| v.boxed_viewe())
                                        .map_err(tile_to_apply_err)?,
                                    nz!(4usize), // 1 << 2
                                    operands[1].shape(),
                                    &[Some(0), None, Some(2)],
                                )),
                                ViewE::Param(Param::new(2, operands[2].clone())),
                            ],
                        );
                        boundary_bodies.push(boundary_body.into());
                    }

                    let bodies: Vec<_> = once(main_body.into()).chain(boundary_bodies).collect();
                    Ok(ImplNode::Loop(Loop {
                        tiles,
                        bodies,
                        parallel: false,
                        spec: Some(spec.clone()),
                    }))
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

                    let mut split_shape = Shape::from_slice(in_tensor_spec.shape());
                    split_shape[usize::from(*dim)] = self.k;
                    let split_tiling = Tiling::new_simple(split_shape.clone());

                    let (main_tile, boundary_tiles) = split_tiling
                        .apply_with_boundaries(Param::new(0, in_tensor_spec.clone()))
                        .map_err(tile_to_apply_err)?;

                    let tiles = vec![LoopTile {
                        parameter_index: 0,
                        axes: (0..u8::try_from(in_tensor_spec.shape().len()).unwrap()).collect(),
                        tile: main_tile.boxed_viewe(),
                    }];

                    let new_shapes: Vec<_> = once(split_shape)
                        .chain(operands.iter().skip(1).map(|o| o.shape().into()))
                        .collect();

                    let mut bodies = Vec::<ImplNode<_>>::with_capacity(boundary_tiles.len() + 1);
                    bodies.push(create_main_body(&tiles, &[new_shapes], false, spec)?.into());

                    for boundary_tile in boundary_tiles {
                        let boundary_new_shapes: Vec<_> =
                            once(Shape::from_slice(boundary_tile.shape()))
                                .chain(operands.iter().skip(1).map(|o| o.shape().into()))
                                .collect();

                        // Create the boundary spec directly (no LoopTiles needed for boundary regions)
                        let mut boundary_body_logicalspec = spec.0.clone();
                        update_component_shapes(
                            &mut boundary_body_logicalspec,
                            &[boundary_new_shapes],
                        )?;
                        let mut boundary_spec = Spec(boundary_body_logicalspec, spec.1.clone());
                        boundary_spec
                            .canonicalize()
                            .map_err(spec_canonicalize_to_apply_err)?;
                        let boundary_args = std::iter::once(ViewE::from(boundary_tile.clone()))
                            .chain(operands.iter().skip(1).enumerate().map(|(i, operand)| {
                                ViewE::Param(Param::new(
                                    u8::try_from(i + 1).unwrap(),
                                    operand.clone(),
                                ))
                            }))
                            .collect::<Vec<_>>();
                        bodies.push(SpecApp::new(boundary_spec, boundary_args).into());
                    }

                    Ok(ImplNode::Loop(Loop {
                        tiles,
                        bodies,
                        parallel: false,
                        spec: Some(spec.clone()),
                    }))
                }
                _ => Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                    "Split only applies to Matmul, Max, and SoftmaxDenominator",
                )))),
            },
            LogicalSpec::Compose {
                components,
                operand_auxes: _,
                serial_only: _,
            } if matches!(
                components[0],
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: true },
                    ..
                }
            ) =>
            {
                // TODO: Implement Split for Compose(MatmulAccum, ..)
                Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                    "Split with Compose unimplemented",
                ))))
            }
            LogicalSpec::Compose { .. } => todo!(),
        }
    }

    fn bottom_up_solvers() -> Self::BSolverIter {
        iter::once(Self::BSolver::default())
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
        match smaller_output_tiling.apply_main(Param::new(3, current_output.clone())) {
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
    match input_tiles_for_tile_out(&operands, tilings_and_bindings, 3, false) {
        Ok(mut new_tiles) => {
            new_tiles.push(smaller_output);
            let component_parameter_shapes = vec![vec![
                new_output_shape.clone().either_into(), // parameter 0: input
                onescan_shape.clone(),                  // parameter 1: max
                onescan_shape,                          // parameter 2: denominator output
                new_output_shape.either_into(),         // parameter 3: unscaled output
            ]];
            let main_body =
                match create_main_body(&new_tiles, &component_parameter_shapes, false, spec) {
                    Ok(body) => body,
                    Err(e) => return Some(Err(e)),
                };
            let bodies = vec![main_body.into()];
            Some(Ok(ImplNode::Loop(Loop {
                tiles: new_tiles,
                bodies,
                parallel: false,
                spec: Some(spec.clone()),
            })))
        }
        Err(e) => Some(Err(e)),
    }
}

/// Returns a [Loop] with a smaller sub-Spec application.
fn tile_out_loop_spec_with_shrunken_tiles<Tgt: Target>(
    component_parameter_shapes: Vec<Vec<Shape>>,
    tiles: Vec<LoopTile<Tgt>>,
    parallel: bool,
    spec: &Spec<Tgt>,
) -> Result<ImplNode<Tgt>, ApplyError> {
    let main_body = create_main_body(&tiles, &component_parameter_shapes, parallel, spec)?;
    let boundary_bodies =
        create_tile_out_boundary_regions(&tiles, &component_parameter_shapes, spec)?;
    let bodies: Vec<_> = once(main_body.into()).chain(boundary_bodies).collect();
    Ok(ImplNode::Loop(Loop {
        tiles,
        bodies,
        parallel,
        spec: Some(spec.clone()),
    }))
}

impl<Tgt: Target> BottomUpSolver for TileOutSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_spec(&mut self, spec: &Spec<Tgt>) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        todo!()
    }

    fn dependencies_for_range<B>(
        &mut self,
        bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
    {
        todo!("Call complete_spec for whatever has no dependencies");

        // TODO: If top is parallel and bottom is serial-only, split into two two ranges.
        //       Right now, if they differ, we'll over-visit the serial-only children.

        let both_serial_only = low.0.serial_only() && high.0.serial_only();
        let multi_dim = MULTI_DIM_TILING || !both_serial_only;

        let parameter_shapes = high.0.parameter_shapes();
        let output_shape = &parameter_shapes[high.0.unique_output_index().unwrap()];
        let output_rank = output_shape.len();
        let output_rank_u8 = u8::try_from(output_rank).unwrap();

        if multi_dim {
            let unit_output_shape = smallvec![nz!(1u32); output_rank];
            let one_tile_action = TileOut::MultiLoop {
                output_shape: unit_output_shape,
                parallel: false,
            };
            let one_tiled_spec = one_tile_action.top_down_solver(low).unwrap();
            let Ok(unit_subspec) = one_tiled_spec.subspecs().exactly_one() else {
                panic!("Expected exactly one subspec");
            };
            vec![(unit_subspec, high.clone())]
        } else {
            let mut result = Vec::with_capacity(output_rank);
            for dim in 0..output_rank_u8 {
                if output_shape[usize::from(dim)] == nz!(1u32) {
                    continue;
                }

                let one_tile_action = TileOut::SingleLoop {
                    dim,
                    size: nz!(1u32),
                    parallel: !low.0.serial_only() && !high.0.serial_only(),
                };

                // TODO: Simplify the below.
                // let one_tiled_spec = one_tile_action.top_down_solver(low).unwrap();
                let one_tiled_spec = match one_tile_action.top_down_solver(high) {
                    Ok(s) => s,
                    Err(e) => {
                        panic!(
                            "Expected top_down_solver to succeed: {:?}\n---\n{}\n{:?}",
                            e, low, one_tile_action
                        );
                    }
                };

                let Ok(unit_subspec) = one_tiled_spec.subspecs().exactly_one() else {
                    panic!("Expected exactly one subspec");
                };
                result.push((unit_subspec, high.clone()));
            }
            result
        }
    }

    fn visit_dependency<U>(&mut self, spec: &Spec<Tgt>, cost: &[NormalizedCost], updater: &mut U)
    where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }

    fn apply_no_dependency_updates<U>(&mut self, spec: &Spec<Self::Tgt>, updater: &mut U)
    where
        U: VisitUpdater<Self::Tgt>,
    {
        todo!()
    }
}

impl<Tgt: Target> BottomUpSolver for SplitSolver<Tgt> {
    type Tgt = Tgt;

    fn dependencies_for_spec(&mut self, spec: &Spec<Tgt>) -> Vec<(Spec<Tgt>, Spec<Tgt>)> {
        todo!()
    }

    fn dependencies_for_range<B>(
        &mut self,
        bimap: &B,
        low: &Spec<Self::Tgt>,
        high: &Spec<Self::Tgt>,
    ) -> Vec<(Spec<Self::Tgt>, Spec<Self::Tgt>)>
    where
        B: BiMap<Domain = Spec<Self::Tgt>, Codomain = DbKey>,
    {
        // MatmulAccum only.
        let LogicalSpec::Primitive(ref basics, ref auxes, serial) = low.0 else {
            return vec![];
        };
        if !matches!(basics.typ, PrimitiveSpecType::Matmul { accum: true }) {
            return vec![];
        }

        // Range from `low` to a clone with k = 1.
        let mut smallest_basics = basics.clone();
        smallest_basics.spec_shape[1] = nz!(1u32);
        let smallest_dependency = Spec(
            LogicalSpec::Primitive(smallest_basics, auxes.clone(), serial),
            low.1.clone(),
        );
        self.requests
            .push((smallest_dependency.clone(), high.clone()));
        vec![(smallest_dependency, high.clone())]
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

    fn visit_dependency<U>(&mut self, spec: &Spec<Tgt>, cost: &[NormalizedCost], updater: &mut U)
    where
        U: VisitUpdater<Tgt>,
    {
        todo!()
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for TileOutActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        tile_out_actions(logical_spec).collect()
    }

    fn debugging() -> Option<String> {
        Some("TileOut".to_string())
    }
}

impl<Tgt: Target> NaiveBottomUpActionProvider<Tgt> for SplitActionProvider<Tgt> {
    fn actions(logical_spec: &LogicalSpec<Tgt>) -> Vec<Action<Tgt>> {
        split_actions(logical_spec).collect()
    }

    fn debugging() -> Option<String> {
        Some("Split".to_string())
    }
}

/// Creates a main body ImplNode by cloning the spec, updating component shapes, setting serial_only
/// flag, and canonicalizing.
///
/// Helper for [tile_out_loop_spec_with_shrunken_tiles].
fn create_main_body<Tgt: Target>(
    tiles: &[LoopTile<Tgt>],
    component_parameter_shapes: &[Vec<Shape>],
    parallel: bool,
    spec: &Spec<Tgt>,
) -> Result<SpecApp<ViewE<Tgt>>, ApplyError> {
    // Clone the parent and update its parameters with new shapes (passed in) from
    // [LogicalSpec::input_tilings_for_tile_out] as well as the given `parallel` flag.
    let mut main_body_logicalspec = spec.0.clone();
    update_component_shapes(&mut main_body_logicalspec, component_parameter_shapes)?;
    main_body_logicalspec.set_serial_only(main_body_logicalspec.serial_only() || parallel);
    main_body_logicalspec
        .canonicalize()
        .map_err(spec_canonicalize_to_apply_err)?;
    let mut spec_app_arguments = main_body_logicalspec
        .parameters()
        .into_iter()
        .enumerate()
        .map(|(i, spec)| ViewE::Param(Param::new(u8::try_from(i).unwrap(), spec)))
        .collect::<Vec<ViewE<Tgt>>>();
    for loop_tile in tiles {
        spec_app_arguments[usize::from(loop_tile.parameter_index)] =
            ViewE::Tile(loop_tile.tile.clone());
    }
    Ok(SpecApp::new(
        Spec(main_body_logicalspec, spec.1.clone()),
        spec_app_arguments,
    ))
}

/// Generate bodies for different boundary regions based on the loop tiles.
///
/// Returns a vector of boundary region bodies.  The main region is not included in the
/// returned vector.
fn create_tile_out_boundary_regions<Tgt: Target>(
    tiles: &[LoopTile<Tgt>],
    component_parameter_shapes: &[Vec<Shape>],
    spec: &Spec<Tgt>,
) -> Result<Vec<ImplNode<Tgt>>, ApplyError> {
    let output_index = spec
        .0
        .unique_output_index()
        .expect("Spec should have a unique output");
    let output_tile = tiles
        .iter()
        .find(|t| usize::from(t.parameter_index) == output_index)
        .expect("a LoopTile should be attached to the output parameter");
    let original_shape = spec.0.parameter_shape(output_index);

    // per_axis_bitvectors contains values with a single bit set. The bit's index is a
    // Spec axis that has a remainder when tiled.
    let per_axis_bitvectors: Vec<_> = output_tile
        .tile
        .shape()
        .iter()
        .enumerate()
        .filter_map(|(dim_idx, &tile_size)| {
            // Check if this dimension has a remainder when tiled
            let original_size = original_shape[dim_idx].get();
            if original_size % tile_size.get() != 0 {
                let actual_axis = output_tile.axes[dim_idx];
                assert!(
                    (actual_axis as u32) < usize::BITS,
                    "Axis {} is too large for region ID representation (max {})",
                    actual_axis,
                    usize::BITS - 1
                );
                Some(1_usize << actual_axis)
            } else {
                None
            }
        })
        .collect();

    let boundary_region_ids: Vec<usize> = bitvector_combinations(&per_axis_bitvectors).collect();

    // Reconstruct the inner_spec from the component_parameter_shapes
    let mut inner_spec = spec.0.clone();
    update_component_shapes(&mut inner_spec, component_parameter_shapes)?;

    // Generate bodies for boundary regions
    let mut bodies = Vec::with_capacity(boundary_region_ids.len());
    for &region_id in &boundary_region_ids {
        // Create tilings for this specific region
        let region_output_tiling =
            create_region_output_tiling(tiles, region_id.try_into().unwrap(), spec)?;

        // Use input_tilings_for_tile_out to get the correct component parameter shapes
        let Some(input_tilings_result) = spec.0.input_tilings_for_tile_out(&region_output_tiling)
        else {
            return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                "Tiling doesn't apply to logical Spec for boundary region",
            ))));
        };

        // Create the region spec with the adjusted shapes (without recursing into boundary regions)
        let mut region_logical_spec = inner_spec.clone();
        update_component_shapes(
            &mut region_logical_spec,
            &input_tilings_result.component_parameter_shapes,
        )?;
        region_logical_spec
            .canonicalize()
            .map_err(spec_canonicalize_to_apply_err)?;

        let operands = spec.0.parameters();
        let mut boundary_tiles = Vec::<(u8, BoundaryTile<_>)>::new();
        for (operand_idx, (original_input, (tiling, io_dim_bindings))) in operands
            .iter()
            .zip(input_tilings_result.input_tilings.0)
            .enumerate()
        {
            if operand_idx == output_index {
                continue;
            }
            if !original_input.is_valid_tile_shape(tiling.shape(), false) {
                return Err(ApplyError::NotApplicable(
                    NotApplicableReason::TileShapeInvalid,
                ));
            }

            // TODO: Can we avoid this check by construction?
            if original_input.shape() == &tiling.shape()[..] {
                continue;
            }

            let operand_idx_u8 = u8::try_from(operand_idx).unwrap();
            let input_tile_axes: Vec<Option<u8>> = io_dim_bindings
                .iter()
                .map(|&o| o.map(|output_dim| output_tile.axes[usize::from(output_dim)]))
                .collect();
            boundary_tiles.push((
                operand_idx_u8,
                convert_tile_to_boundarytile(
                    tiling
                        .apply_main(Param::new(operand_idx_u8, original_input.clone()))
                        .map(|v| v.boxed_viewe())
                        .map_err(tile_to_apply_err)?,
                    region_id.try_into().unwrap(),
                    original_input.shape(),
                    &input_tile_axes,
                ),
            ));
        }
        boundary_tiles.push((
            u8::try_from(output_index).unwrap(),
            convert_tile_to_boundarytile(
                region_output_tiling
                    .apply_main(Param::new(
                        u8::try_from(output_index).unwrap(),
                        spec.0.parameters()[output_index].clone(),
                    ))
                    .map(|v| v.boxed_viewe())
                    .map_err(tile_to_apply_err)?,
                region_id.try_into().unwrap(),
                spec.0.parameters()[output_index].shape(),
                &output_tile
                    .axes
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<Vec<_>>(),
            ),
        ));

        let mut body_logicalspec = spec.0.clone();
        update_component_shapes(
            &mut body_logicalspec,
            &input_tilings_result.component_parameter_shapes,
        )?;
        body_logicalspec.set_serial_only(body_logicalspec.serial_only());
        body_logicalspec
            .canonicalize()
            .map_err(spec_canonicalize_to_apply_err)?;
        let mut spec_app_arguments = body_logicalspec
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, spec)| ViewE::Param(Param::new(u8::try_from(i).unwrap(), spec)))
            .collect::<Vec<ViewE<Tgt>>>();
        for (parameter_idx, boundary_tile) in boundary_tiles {
            spec_app_arguments[usize::from(parameter_idx)] = ViewE::BoundaryTile(boundary_tile);
        }
        bodies.push(SpecApp::new(Spec(body_logicalspec, spec.1.clone()), spec_app_arguments).into())
    }

    debug_assert_eq!(bodies.len(), boundary_region_ids.len());
    Ok(bodies)
}

/// Create [LoopTile]s corresponding to a set of input [Tiling]s and bindings.
fn input_tiles_for_tile_out<Tgt: Target>(
    operands: &[TensorSpec<Tgt>],
    tilings_and_bindings: Vec<(Tiling, Vec<Option<u8>>)>,
    output_idx: usize,
    parallel: bool,
) -> Result<Vec<LoopTile<Tgt>>, ApplyError> {
    let output_tensor_rank = operands[output_idx].shape().len();
    let mut next_fresh_loop_dim = u8::try_from(output_tensor_rank).unwrap();
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
                .apply_main(Param::new(operand_idx_u8, original_input.clone()))
                .map(|v| v.boxed_viewe())
                .map_err(tile_to_apply_err)?,
        });
    }
    Ok(new_tiles)
}

/// Create an output tiling for a specific boundary region
fn create_region_output_tiling<Tgt: Target>(
    tiles: &[LoopTile<Tgt>],
    region_id: NonZeroUsize,
    spec: &Spec<Tgt>,
) -> Result<crate::tiling::Tiling, ApplyError> {
    let output_index = spec.0.unique_output_index().unwrap();
    let output_tile = tiles
        .iter()
        .find(|t| usize::from(t.parameter_index) == output_index)
        .unwrap();

    let original_param = &spec.0.parameters()[output_tile.parameter_index as usize];
    let original_param_shape = original_param.shape();

    let mut region_tile_shape: Vec<DimSize> = vec![nz!(1u32); original_param_shape.len()];
    for (dim_idx, &tile_size) in output_tile.tile.shape().iter().enumerate() {
        let original_size = original_param_shape[dim_idx];
        let actual_axis = output_tile.axes[dim_idx];

        if region_id.get() & (1_usize << actual_axis) != 0 {
            let remainder = NonZeroU32::new(original_size.get() % tile_size.get()).unwrap();
            region_tile_shape[dim_idx] = remainder;
        } else {
            let main_extent = original_size.get() - (original_size.get() % tile_size.get());
            region_tile_shape[dim_idx] = main_extent.try_into().unwrap();
        }
    }
    debug_assert_ne!(region_tile_shape, original_param_shape);
    Ok(Tiling::new_simple(region_tile_shape.into()))
}

/// Updates the parameter shapes of a LogicalSpec to match the provided shapes.
///
/// This takes a shape for each parameter of each component of the LogicalSpec. There is only one
/// "component" in the case of a [LogicalSpec::Primitive]. For a [LogicalSpec::Compose], this will
/// update all parameters, including "internal" parameters, of the components.
pub(crate) fn update_component_shapes<Tgt: Target>(
    spec: &mut LogicalSpec<Tgt>,
    component_parameter_shapes: &[Vec<Shape>],
) -> Result<(), ApplyError> {
    match spec {
        LogicalSpec::Primitive(prim_basics, primitive_aux, _) => {
            let [new_shapes] = component_parameter_shapes else {
                panic!(
                    "Expected exactly one component for primitive spec, got {}",
                    component_parameter_shapes.len()
                );
            };
            debug_assert_eq!(new_shapes.len(), prim_basics.dtypes.len());

            let original_shapes = prim_basics.parameter_shapes();
            prim_basics.replace_io(
                &new_shapes
                    .iter()
                    .zip(&prim_basics.dtypes)
                    .map(|(shape, &dtype)| (&shape[..], dtype))
                    .collect::<Vec<_>>(),
            );

            for (original_shape, new_shape, aux) in
                izip!(original_shapes, new_shapes, primitive_aux.iter_mut())
            {
                if !aux.level.has_layout() {
                    continue;
                }
                if let Ok(new_layout) = aux.layout.update_for_tiling(&original_shape, new_shape) {
                    aux.layout = new_layout;
                }
            }
        }
        LogicalSpec::Compose {
            components,
            operand_auxes,
            ..
        } => {
            if components.len() != component_parameter_shapes.len() {
                return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                    "Component count mismatch for compose spec",
                ))));
            }

            let original_component_shapes: Vec<Vec<Shape>> = components
                .iter()
                .map(|subspec| subspec.parameter_shapes())
                .collect();

            for (subspec, shapes) in components.iter_mut().zip(component_parameter_shapes) {
                if shapes.len() != subspec.typ.operand_count()
                    || shapes.len() != subspec.dtypes.len()
                {
                    return Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(
                        "Shape count mismatch for compose component",
                    ))));
                }

                let new_operands: Vec<(&[DimSize], Dtype)> = shapes
                    .iter()
                    .zip(&subspec.dtypes)
                    .map(|(shape, &dtype)| (&shape[..], dtype))
                    .collect();
                subspec.replace_io(&new_operands);
            }
            update_compose_aux_for_tiling(components, &original_component_shapes, operand_auxes);
        }
    }

    Ok(())
}

/// Updates the layout of a [TensorSpecAux], given a shape change and updated step
/// sizes.
fn update_aux_for_tiling<Tgt: Target>(
    aux: &mut TensorSpecAux<Tgt>,
    original_shape: &[DimSize],
    new_shape: &[DimSize],
) {
    if !aux.level.has_layout() {
        aux.layout = row_major(u8::try_from(new_shape.len()).expect("rank fits in u8"));
    } else if let Ok(new_layout) = aux.layout.update_for_tiling(original_shape, new_shape) {
        aux.layout = new_layout;
    }
}

/// Updates the layout and alignment the Compose's [TensorSpecAux]s for new shapes and
/// steps. New shapes will be taken from `components`, so `components` must already be
/// resized.
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
            update_aux_for_tiling(&mut operand_auxes[aux_idx], original_shape, &new_shape);
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
                update_aux_for_tiling(&mut operand_auxes[aux_idx], original_shape, &new_shape);
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
            update_aux_for_tiling(&mut operand_auxes[aux_idx], original_shape, &new_shape);
            aux_idx += 1;
        }
    }

    // Finally, handle the compose output (which is the first component's output)
    let original_shape = &original_component_shapes[0][c0_output_idx];
    let new_shape = components[0].parameter_shape(c0_output_idx);
    update_aux_for_tiling(&mut operand_auxes[aux_idx], original_shape, &new_shape);
}

// TODO: Get rid of this function. It's a design problem.
fn convert_tile_to_boundarytile<V: View>(
    tile: Tile<V>,
    region_id: NonZeroUsize, // non-zero because boundary
    original_shape: &[DimSize],
    axes: &[Option<u8>],
) -> BoundaryTile<V> {
    debug_assert_eq!(tile.shape().len(), axes.len());
    let mut offsets = vec![0u32; tile.shape().len()];
    for (dim_idx, &axis_option) in axes.iter().enumerate() {
        let Some(axis) = axis_option else {
            continue;
        };
        if region_id.get() & (1_usize << axis) != 0 {
            offsets[dim_idx] = original_shape[dim_idx].get() - tile.shape()[dim_idx].get();
        }
    }
    BoundaryTile::new(Shape::from_slice(tile.shape()), offsets, tile.view).unwrap()
}

fn bitvector_combinations(bitvectors: &[usize]) -> impl Iterator<Item = usize> + '_ {
    // Generate all non-empty subsets (1 to 2^n - 1)
    (1..1_usize << bitvectors.len()).map(move |subset_mask| {
        bitvectors
            .iter()
            .enumerate()
            .filter_map(|(i, &bitvector)| {
                if subset_mask & (1 << i) != 0 {
                    Some(bitvector)
                } else {
                    None
                }
            })
            .fold(0, |acc, bitvector| acc | bitvector)
    })
}

/// Converts a [CanonicalizeError] to an [ApplyError] for tiling operations.
fn spec_canonicalize_to_apply_err(canon_error: CanonicalizeError) -> ApplyError {
    match canon_error {
        CanonicalizeError::TensorSpecAuxCanonicalizeError(_) => {
            ApplyError::NotApplicable(NotApplicableReason::TileShapeInvalid)
        }
        CanonicalizeError::SideEffectingComponent => {
            unreachable!("Compose-to-tile should not have side-effecting components")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::imp::{loops::Loop, Impl, ImplNode};
    use crate::layout::row_major;
    use crate::scheduling::{Action, ActionT, ApplyError, NotApplicableReason};
    use crate::shape;
    use crate::spec;
    use crate::spec::Spec;
    use crate::spec::{arb_canonical_spec, LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
    use crate::target::{
        CpuMemoryLevel::{self, GL},
        X86Target,
    };
    use crate::tensorspec::{TensorSpec, TensorSpecArbMaxShape, TensorSpecAux};
    use crate::views::{View, ViewE};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;

    #[test]
    fn test_non_multiple_tile_out_single_succeeds() {
        shared_test_non_multiple_tiling_succeeds(Action::TileOut(TileOut::SingleLoop {
            dim: 0,
            size: nz!(3u32),
            parallel: false,
        }))
    }

    #[test]
    fn test_non_multiple_tile_out_multi_succeeds() {
        shared_test_non_multiple_tiling_succeeds(Action::TileOut(TileOut::MultiLoop {
            output_shape: shape![1, 4, 6],
            parallel: false,
        }))
    }

    #[test]
    fn test_non_multiple_split_succeeds() {
        shared_test_non_multiple_tiling_succeeds(Action::Split(Split { k: nz!(6u32) }))
    }

    /// Test that tiling a Conv inside a Compose is disallowed when the tile is smaller than the filter.
    #[test]
    fn test_conv_in_compose_tile_out_disallowed_when_filter_larger_than_tile() {
        let softmax_denominator = PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominator {
                scan_dim: 0,
                accum: false,
            },
            spec_shape: shape![1, 5, 2, 2],
            dtypes: vec![Dtype::Uint8, Dtype::Uint8, Dtype::Uint8],
        };
        let conv = PrimitiveBasics {
            typ: PrimitiveSpecType::Conv { accum: false },
            spec_shape: shape![1, 5, 6, 4, 2, 3, 1],
            dtypes: vec![Dtype::Sint8, Dtype::Uint16, Dtype::Uint8],
        };
        let rm4 = row_major(4);
        let rm7 = row_major(7);
        let l1rm4 = TensorSpecAux {
            level: CpuMemoryLevel::L1,
            layout: rm4.clone(),
            vector_size: None,
        };
        let rfrm4 = TensorSpecAux {
            level: CpuMemoryLevel::RF,
            layout: rm4.clone(),
            vector_size: None,
        };
        let operand_auxes: Vec<TensorSpecAux<X86Target>> = vec![
            l1rm4.clone(),
            rfrm4.clone(),
            l1rm4,
            TensorSpecAux {
                level: CpuMemoryLevel::L1,
                layout: rm7.clone(),
                vector_size: None,
            },
            TensorSpecAux {
                level: CpuMemoryLevel::RF,
                layout: rm7.clone(),
                vector_size: None,
            },
            TensorSpecAux {
                level: CpuMemoryLevel::L1,
                layout: rm7,
                vector_size: None,
            },
        ];
        let compose_spec = Spec(
            LogicalSpec::Compose {
                components: vec![softmax_denominator, conv],
                operand_auxes,
                serial_only: false,
            },
            X86Target::max_mem(),
        );

        // Compose them together and attempt to tile the Conv's loop (dim index 3) with size=1
        let action = Action::TileOut(TileOut::SingleLoop {
            dim: 3,
            size: nz!(1u32),
            parallel: false,
        });
        let result = action.apply(&compose_spec);
        assert!(
            result.is_err(),
            "tile_out to be disallowed when tile < filter, got {result:?}"
        );
    }

    fn shared_test_non_multiple_tiling_succeeds(action: Action<X86Target>) {
        let bcm_layout = crate::layout![0, 2, 1];
        let spec: Spec<X86Target> = spec!(MatmulAccum(
            [4, 8, 8, 8],
            (f32, CpuMemoryLevel::GL, bcm_layout),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let application = action.apply(&spec);
        assert!(
            application.is_ok(),
            "expected successful application but got {application:?}",
        );
    }

    /// Test that non-even tiling produces a valid loop structure with the correct cost.
    #[test]
    fn test_non_even_tiling_loop_and_cost() {
        // Test tiling of 10x10 to 3x3
        let spec: Spec<X86Target> = spec!(MatmulAccum(
            [4, 10, 10, 10],
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major),
            (f32, CpuMemoryLevel::GL, row_major)
        ));
        let tile_action = Action::TileOut(TileOut::MultiLoop {
            output_shape: shape![4, 3, 3],
            parallel: false,
        });
        match tile_action.apply(&spec) {
            Ok(ImplNode::Loop(loop_impl)) => {
                assert!(!loop_impl.tiles.is_empty(), "Loop should have tiles");
                assert_eq!(loop_impl.compute_main_cost(&[100, 9, 11, 2]), 922);
            }
            Ok(n) => panic!("Expected Loop implementation, got {n:?}"),
            Err(err) => panic!("Expected successful tiling application, got {err:?}"),
        };
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
            bodies,
            parallel,
            spec: loop_spec,
        }) = tile_action
            .apply(&spec)
            .expect("TileOut should apply successfully")
        else {
            panic!("Expected ImplNode::Loop")
        };
        assert_eq!(
            bodies.len(),
            1,
            "Expected one body in the loop (no boundary regions)"
        );
        assert!(!parallel);
        assert_eq!(loop_spec, Some(spec));

        // Parameters 0 and 3 should be tiled
        let param_indices: Vec<u8> = tiles.iter().map(|t| t.parameter_index).collect();
        assert_eq!(param_indices, [0, 3]);

        // Check shapes of the Loop's Params, which won't be tiled.
        let loop_params = bodies[0].collect_unbound_parameters();
        let expected_param_shapes = [
            &[nz!(4u32), nz!(8u32), nz!(16u32)][..],
            &[nz!(4u32), nz!(8u32), nz!(1u32)][..],
            &[nz!(4u32), nz!(8u32), nz!(1u32)][..],
            &[nz!(4u32), nz!(8u32), nz!(16u32)][..],
        ];
        assert_eq!(
            loop_params.iter().map(|p| p.shape()).collect::<Vec<_>>(),
            expected_param_shapes
        );

        let expected_body_argument_shapes = [
            shape![4, 8, 8], // Parameter 0: input (tiled)
            shape![4, 8, 1], // Parameter 1: max (reduced)
            shape![4, 8, 1], // Parameter 2: denominator (reduced)
            shape![4, 8, 8], // Parameter 3: output (tiled)
        ];
        let ImplNode::SpecApp(inner_specapp) = &bodies[0] else {
            panic!("Expected ImplNode::SpecApp");
        };
        assert_eq!(
            inner_specapp.0 .0.parameter_shapes(),
            expected_body_argument_shapes,
        );
        assert_eq!(
            inner_specapp
                .1
                .iter()
                .map(|a| a.spec().shape())
                .collect::<Vec<_>>(),
            expected_body_argument_shapes
                .iter()
                .map(|s| &s[..])
                .collect::<Vec<_>>(),
        );
    }

    proptest! {
        /// Test that [update_aux_for_tiling] preserves auxiliary information when shapes are
        /// identical.
        #[test]
        fn test_update_aux_for_tiling_identical_shapes_preserves_aux(
            mut tspec in any_with::<TensorSpec<X86Target>>(TensorSpecArbMaxShape(shape![8, 8, 8]))
        ) {
            let shape = tspec.shape().to_vec();
            let original_aux = tspec.aux.clone();
            update_aux_for_tiling(&mut tspec.aux, &shape, &shape);
            prop_assert_eq!(tspec.aux.layout, original_aux.layout);
            prop_assert_eq!(tspec.aux.level, original_aux.level);
            prop_assert_eq!(tspec.aux.vector_size, original_aux.vector_size);
        }

        /// Test that any tiling operation (TileOut or Split) producing boundary regions results in SpecApps
        /// with at least one BoundaryTile argument for each boundary body.
        #[test]
        fn test_tiling_operations_with_boundary_regions_have_boundary_tiles(
            (spec, action) in arb_canonical_spec::<X86Target>(Some(nz!(8u32)), None)
                .prop_filter_map("Must have unique output", |spec| {
                    spec.0.unique_output_index().map(|idx| (spec, idx))
                })
                .prop_flat_map(|(spec, output_idx)| {
                    let output_shape = spec.0.parameters()[output_idx].shape().to_vec();

                    // Strategy 1: TileOut actions on output tensor dimensions
                    let tileout_strategy = if output_shape.is_empty() {
                        unreachable!();
                    } else {
                        let dim_range = 0u8..u8::try_from(output_shape.len().min(4)).unwrap();
                        let size_range = 2u32..5;
                        (dim_range, size_range)
                            .prop_map(|(dim, size)| {
                                Action::TileOut(TileOut::SingleLoop {
                                    dim,
                                    size: NonZeroU32::new(size).unwrap(),
                                    parallel: false,
                                })
                            })
                            .boxed()
                    };

                    // Strategy 2: Split actions on k dimension of Matmul or Compose(Matmul, ..) (only if applicable)
                    let split_strategy = match &spec.0 {
                        LogicalSpec::Primitive(PrimitiveBasics { typ: PrimitiveSpecType::Matmul { accum: true }, .. }, ..) => {
                            let operands = spec.0.parameters();
                            let k_dim_size = operands[0].shape()[2].get();
                            if k_dim_size > 1 {
                                let k_range = 1u32..(k_dim_size.min(6));
                                Some(k_range
                                    .prop_map(|k| Action::Split(Split { k: NonZeroU32::new(k).unwrap() }))
                                    .boxed())
                            } else {
                                None
                            }
                        }
                        LogicalSpec::Compose { components, .. } if matches!(components.first(), Some(PrimitiveBasics { typ: PrimitiveSpecType::Matmul { accum: true }, .. })) => {
                            let operands = spec.0.parameters();
                            if operands.len() >= 2 && operands[0].shape().len() >= 3 {
                                let k_dim_size = operands[0].shape()[2].get();
                                if k_dim_size > 1 {
                                    let k_range = 1u32..(k_dim_size.min(6));
                                    Some(k_range
                                        .prop_map(|k| Action::Split(Split { k: NonZeroU32::new(k).unwrap() }))
                                        .boxed())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    let combined_strategy = if let Some(split_strat) = split_strategy {
                        prop_oneof![tileout_strategy, split_strat].boxed()
                    } else {
                        tileout_strategy
                    };
                    combined_strategy.prop_map(move |action| (spec.clone(), action))
                })
        ) {
            let result = action.apply(&spec);
            let Ok(impl_node) = result else {
                return Ok(());
            };
            let ImplNode::Loop(loop_impl) = impl_node else {
                unreachable!();
            };

            // Check that every non-main body is a SpecApp with at least one BoundaryTile argument
            for (body_idx, body) in loop_impl.bodies.iter().enumerate().skip(1) {
                let ImplNode::SpecApp(spec_app) = body else {
                    unreachable!("Boundary body {body_idx} should be a SpecApp, got: {body:?}");
                };
                let has_boundary_tile = spec_app.1.iter().any(|arg| matches!(arg, ViewE::BoundaryTile(_)));
                prop_assert!(
                    has_boundary_tile,
                    "Boundary body {body_idx} should have at least one BoundaryTile argument"
                );
            }
        }

        #[test]
        fn test_split_with_multiple_k_produces_no_boundaries(
            (spec, action) in arb_canonical_spec::<X86Target>(Some(nz!(16u32)), None)
                .prop_filter_map("Must be Matmul with accum", |spec| {
                    match &spec.0 {
                        LogicalSpec::Primitive(PrimitiveBasics {
                            typ: PrimitiveSpecType::Matmul { accum: true },
                            ..
                        }, ..) => Some(spec),
                        _ => None
                    }
                })
                .prop_filter("Matmul must have k > 1", |s| s.0.parameter_shape(0)[2].get() > 1)
                .prop_flat_map(|spec| {
                    let operands = spec.0.parameters();
                    let k_dim_size = operands[0].shape()[2].get();
                    let factors: Vec<u32> = (1..=(k_dim_size/2)).filter(|&k| k_dim_size % k == 0).collect();
                    let split_actions = prop::sample::select(factors)
                        .prop_map(|k| Action::Split(Split { k: k.try_into().unwrap() }))
                        .boxed();
                    (Just(spec), split_actions)
                })
        ) {
            match action.apply(&spec) {
                Ok(ImplNode::Loop(loop_impl)) => {
                    assert_eq!(loop_impl.bodies.len(), 1, "Split with multiple k should produce no boundary regions");
                }
                Ok(_) => unreachable!(),
                Err(_) => {} // skip apply failures
            }
        }
    }

    #[test]
    fn test_split_with_compose_matmul_accum_fails_with_not_yet_implemented() {
        let matmul_accum = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: true },
            spec_shape: shape![1, 4, 8, 4], // [b, m, k, n]
            dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
        };
        let move_component = PrimitiveBasics {
            typ: PrimitiveSpecType::Move,
            spec_shape: shape![1, 4, 4], // [b, m, n] - matches MatmulAccum output
            dtypes: vec![Dtype::Float32, Dtype::Float32],
        };
        let aux: TensorSpecAux<X86Target> = TensorSpecAux {
            level: CpuMemoryLevel::GL,
            layout: row_major(3),
            vector_size: None,
        };
        let mut compose_spec = Spec(
            LogicalSpec::Compose {
                components: vec![matmul_accum, move_component],
                operand_auxes: vec![aux.clone(), aux.clone(), aux],
                serial_only: false,
            },
            X86Target::max_mem(),
        );
        compose_spec.canonicalize().unwrap();

        let split_action = Action::Split(Split { k: nz!(4u32) });
        let result = split_action.apply(&compose_spec);

        // Once we implement the feature, we can change this to assert success
        match result {
            Err(ApplyError::NotApplicable(NotApplicableReason::Other(Some(msg)))) => {
                assert!(
                    msg.contains("Split with Compose unimplemented"),
                    "Expected 'unimplemented' error, got: {msg}"
                );
            }
            Ok(_) => {
                panic!("Split with Compose unexpectedly succeeded - feature may have been implemented!");
            }
            Err(other_err) => {
                panic!("Expected 'not yet implemented' error, got: {other_err:?}");
            }
        }
    }

    #[test]
    fn test_conv_tiling_produces_no_boundary_regions() {
        let conv_spec: Spec<X86Target> = spec!(
            Conv(
                [4, 7, 5, 8, 6, 4, 8],
                (f32, CpuMemoryLevel::RF, row_major),
                (u8, CpuMemoryLevel::L1, row_major),
                (u8, CpuMemoryLevel::RF, row_major)
            ),
            [16, 16, 32768, 0]
        );
        assert!(conv_spec.is_canonical());

        let tile_action = Action::TileOut(TileOut::SingleLoop {
            dim: 0,
            size: nz!(2u32), // tile 4÷2=2 (no remainder)
            parallel: false,
        });
        let resulting_impl = tile_action
            .apply(&conv_spec)
            .expect("TileOut should apply successfully");
        let ImplNode::Loop(loop_impl) = resulting_impl else {
            panic!("Expected ImplNode::Loop, got: {resulting_impl:?}");
        };
        assert_eq!(loop_impl.bodies.len(), 1, "Expected one body in the loop");
        // TODO: Assert on the sub-Spec
    }

    #[test]
    fn test_bitvector_combinations() {
        let empty_bitvectors = vec![];
        let empty_result: Vec<usize> = bitvector_combinations(&empty_bitvectors).collect();
        assert_eq!(empty_result, vec![]);

        let single_bitvectors = vec![0b100]; // bit 2 set
        let single_result: Vec<usize> = bitvector_combinations(&single_bitvectors).collect();
        assert_eq!(single_result, vec![0b100]);

        let two_bitvectors = vec![0b001, 0b010]; // bits 0 and 1 set respectively
        let mut two_result: Vec<usize> = bitvector_combinations(&two_bitvectors).collect();
        two_result.sort();
        assert_eq!(two_result, vec![0b001, 0b010, 0b011]);

        let three_bitvectors = vec![0b001, 0b010, 0b100]; // bits 0, 1, 2 set respectively
        let mut three_result: Vec<usize> = bitvector_combinations(&three_bitvectors).collect();
        three_result.sort();
        assert_eq!(
            three_result,
            vec![0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111]
        );

        // Test with missing middle bit (gap in axes)
        let gap_bitvectors = vec![0b001, 0b100]; // bits 0 and 2 set, bit 1 missing
        let mut gap_result: Vec<usize> = bitvector_combinations(&gap_bitvectors).collect();
        gap_result.sort();
        assert_eq!(gap_result, vec![0b001, 0b100, 0b101]); // OR combinations: 0b001, 0b100, 0b001|0b100
    }

    /// Test that splitting a Matmul produces a nested SpecApp applied to two Tiles and
    /// a Param.
    #[test]
    fn test_split_apply_yields_tile_bound_specapp() {
        let spec: Spec<X86Target> = spec!(MatmulAccum(
            [1, 64, 64, 64],
            (f32, GL, row_major),
            (f32, GL, row_major),
            (f32, GL, row_major),
            serial
        ));

        let split_action = Split { k: nz!(32u32) };
        let result = split_action.apply_unchecked_canon(&spec);
        assert!(result.is_ok(), "Split should apply successfully");

        let impl_node = result.unwrap();
        let ImplNode::Loop(loop_impl) = impl_node else {
            panic!("Expected Loop implementation, got: {impl_node:?}");
        };

        // First two arguments should be Tile views and third a Param
        let ImplNode::SpecApp(spec_app) = &loop_impl.bodies[0] else {
            panic!(
                "Expected SpecApp in loop body, got: {:?}",
                loop_impl.bodies[0]
            );
        };
        assert!(
            matches!(spec_app.1[0], ViewE::Tile(_)),
            "First operand should be a Tile view, got: {:?}",
            spec_app.1[0]
        );
        assert!(
            matches!(spec_app.1[1], ViewE::Tile(_)),
            "Second operand should be a Tile view, got: {:?}",
            spec_app.1[1]
        );
        assert!(
            matches!(spec_app.1[2], ViewE::Param(_)),
            "Third operand should be a Param view, got: {:?}",
            spec_app.1[2]
        );
    }

    /// Test that TileOut produces a SpecApp which uses Tile views for tiled operands,
    /// not Param views.
    #[test]
    fn test_tileout_apply_yields_tile_bound_specapp() {
        let spec: Spec<X86Target> = spec!(MatmulAccum(
            [1, 64, 64, 64],
            (f32, GL, row_major),
            (f32, GL, row_major),
            (f32, GL, row_major),
            serial
        ));

        let action = TileOut::SingleLoop {
            dim: 1,
            size: nz!(32u32),
            parallel: false,
        };
        let result = action.apply_unchecked_canon(&spec);
        assert!(result.is_ok(), "TileOut should apply successfully");
        let impl_node = result.unwrap();

        let ImplNode::Loop(loop_impl) = impl_node else {
            panic!("Expected Loop implementation, got: {impl_node:?}");
        };
        let ImplNode::SpecApp(spec_app) = &loop_impl.bodies[0] else {
            panic!(
                "Expected SpecApp in loop body, got: {:?}",
                loop_impl.bodies[0]
            );
        };

        assert!(
            matches!(spec_app.1[0], ViewE::Tile(_)),
            "First operand should be a Tile view, got: {:?}",
            spec_app.1[0]
        );
        assert!(
            matches!(spec_app.1[1], ViewE::Param(Param(1, ..))),
            "Second operand should be a Param view, got: {:?}",
            spec_app.1[1]
        );
        assert!(
            matches!(spec_app.1[2], ViewE::Tile(_)),
            "Third operand should be a Tile view, got: {:?}",
            spec_app.1[2]
        );
    }
}
