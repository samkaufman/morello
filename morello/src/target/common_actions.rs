//! Scheduling actions shared between targets.

use super::{MemoryLevel, Target};
use crate::{
    common::{DimSize, Shape},
    scheduling::{
        bufferize::Bufferize,
        moves::Move,
        tiling::{Split, TileOut},
        Action,
    },
    spec::{dim_range, LogicalSpec, PrimitiveBasics, PrimitiveSpecType},
    tensorspec::{gen_vector_sizes, gen_vector_sizes_opt},
};
use itertools::{Either, Itertools};
use std::iter;

/// Policy controlling when `tile_out` actions may tile in multiple dimensions.
const MULTI_DIM_TILING_POLICY: MultiDimTilingPolicy = MultiDimTilingPolicy::ParallelOnly(2);
/// An empirically chosen initial capacity for the [LogicalSpec::move_actions] results buffer.
const MOVE_RESULTS_CAPACITY: usize = 16;

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
enum MultiDimTilingPolicy {
    /// Never generate multi-dimension tile actions; only use single-loop tiles.
    Never,
    /// Only generate multi-dimension tile actions for parallel loops and enforce
    /// the contained maximum number of tiled dimensions.
    ParallelOnly(usize),
    /// Always generate multi-dimension tile actions and do not enforce
    /// any max tiled-dimension limit.
    Always,
}

pub fn tile_out_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> impl Iterator<Item = Action<Tgt>> + '_ {
    let serial_only = spec.serial_only();
    if let Some(output_idx) = spec.unique_output_index() {
        let output_shape = spec.parameter_shape(output_idx);
        Either::Left(tile_out_actions_for_shape::<Tgt>(
            &output_shape,
            serial_only,
            MULTI_DIM_TILING_POLICY,
        ))
    } else {
        Either::Right(iter::empty())
    }
}

fn tile_out_actions_for_shape<Tgt: Target>(
    output_shape: &[DimSize],
    serial_only: bool,
    multi_dim_tiling: MultiDimTilingPolicy,
) -> impl Iterator<Item = Action<Tgt>> + 'static {
    let serial_iter = tile_out_actions_for_mode::<Tgt>(
        output_shape,
        false,
        multi_dim_tiling == MultiDimTilingPolicy::Always,
        multi_dim_tiling,
    );

    if serial_only {
        Either::Left(serial_iter)
    } else {
        let parallel_iter = tile_out_actions_for_mode::<Tgt>(
            output_shape,
            true,
            multi_dim_tiling != MultiDimTilingPolicy::Never,
            multi_dim_tiling,
        );

        Either::Right(serial_iter.chain(parallel_iter))
    }
}

fn tile_out_actions_for_mode<Tgt: Target>(
    output_shape: &[DimSize],
    parallel: bool,
    multi_dim: bool,
    multi_dim_tiling: MultiDimTilingPolicy,
) -> impl Iterator<Item = Action<Tgt>> + 'static {
    let output_shape = Shape::from(output_shape);
    if multi_dim {
        let max_tiled_dims = match (parallel, multi_dim_tiling) {
            (true, MultiDimTilingPolicy::ParallelOnly(max_dims)) => max_dims,
            _ => output_shape.len(),
        };
        let tile_iter = gen_tile_sizes(&output_shape, max_tiled_dims);
        Either::Left(tile_iter.map(move |tile_shape| {
            Action::TileOut(TileOut::MultiLoop {
                output_shape: tile_shape,
                parallel,
            })
        }))
    } else {
        // Yield all output tilings up to the *maximum* dimension size so that the actions have
        // relatively stable order between Specs.
        let max_dim_size =
            DimSize::try_from(output_shape.iter().map(|d| d.get()).max().unwrap()).unwrap();
        Either::Right(dim_range(max_dim_size, true).flat_map(move |size| {
            (0..output_shape.len()).map(move |dim| {
                Action::TileOut(TileOut::SingleLoop {
                    dim: u8::try_from(dim).unwrap(),
                    size,
                    parallel,
                })
            })
        }))
    }
}

pub fn split_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> impl Iterator<Item = Action<Tgt>> + '_ {
    let LogicalSpec::Primitive(
        PrimitiveBasics {
            typ, spec_shape, ..
        },
        ..,
    ) = spec
    else {
        panic!("split_actions called on non-primitive Spec");
    };

    let operands = spec.parameters();

    match typ {
        PrimitiveSpecType::Matmul { accum: true } => {
            let [b, m, orig_k, n] = spec_shape[..] else {
                unreachable!();
            };
            Either::Left(
                dim_range(orig_k, false)
                    .filter(move |&new_k| {
                        // TODO: Shouldn't this be rejected during application instead?
                        operands[0].is_valid_tile_shape(&[b, m, new_k], false)
                            && operands[1].is_valid_tile_shape(&[b, new_k, n], false)
                    })
                    .map(|k| Action::Split(Split { k })),
            )
        }
        PrimitiveSpecType::Max { dim, accum: true }
        | PrimitiveSpecType::SoftmaxDenominator {
            scan_dim: dim,
            accum: true,
        } => {
            let scan_dim_idx = usize::from(*dim);
            if scan_dim_idx >= spec_shape.len() {
                panic!(
                    "scan_dim {} is out of bounds for spec_shape with length {}",
                    scan_dim_idx,
                    spec_shape.len()
                );
            }
            let orig_dim_size = spec_shape[scan_dim_idx];
            Either::Right(
                dim_range(orig_dim_size, false)
                    .filter(move |&new_k| {
                        // TODO: Shouldn't this be rejected during application instead?
                        operands.first().is_some_and(|input_spec| {
                            let mut tile_shape = spec_shape.clone();
                            tile_shape[scan_dim_idx] = new_k;
                            input_spec.is_valid_tile_shape(&tile_shape, false)
                        })
                    })
                    .map(|k| Action::Split(Split { k })),
            )
        }
        _ => {
            panic!("split_actions called on unsupported spec type: {typ:?}");
        }
    }
}

pub fn bufferize_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> impl Iterator<Item = Action<Tgt>> + '_ {
    let LogicalSpec::Compose {
        components,
        operand_auxes: _,
        serial_only: _,
    } = spec
    else {
        panic!("bufferize_actions called on non-Compose Spec");
    };

    let mut results = vec![];

    for index in 0..(components.len() - 1) {
        let comp = &components[index + 1];
        let comp_out_idx = comp.typ.unique_output_index().unwrap();
        let intermediate_shape = comp.parameter_shape(comp_out_idx);
        let intermediate_dtype = comp.dtypes[comp_out_idx];

        for level in Tgt::levels() {
            let vector_bytes = level.vector_bytes();

            for layout in Tgt::move_destination_layouts(&intermediate_shape, intermediate_dtype) {
                // TODO: Need to implement `can_move_to`-style logic here.

                if !vector_bytes.is_empty() {
                    for vector_size in gen_vector_sizes(intermediate_dtype, vector_bytes) {
                        results.push(Action::Bufferize(Bufferize {
                            index,
                            level,
                            layout: layout.clone(),
                            vector_size: Some(vector_size),
                        }));
                    }
                } else {
                    results.push(Action::Bufferize(Bufferize {
                        index,
                        level,
                        layout,
                        vector_size: None,
                    }));
                }
            }
        }
    }

    results.into_iter()
}

pub fn move_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> impl Iterator<Item = Action<Tgt>> + '_ {
    // TODO: Don't accumulate. Return an iterator.
    let mut results = Vec::with_capacity(MOVE_RESULTS_CAPACITY);

    for (i, operand) in spec.parameters().iter().enumerate() {
        // Yield actions for movement with register file destination, which
        // includes relayouts in registers and movements from level 1 to RF.
        let i = u8::try_from(i).unwrap();
        let operand_dtype = operand.dtype();
        let destination_layouts = Tgt::move_destination_layouts(operand.shape(), operand_dtype);
        let mut contains_source_layout = false;

        for layout in &destination_layouts {
            // TODO: Prevent moving into packed layouts where strip size equals the whole dim.
            if !contains_source_layout && layout == operand.layout() {
                contains_source_layout = true;
            }
            for level in Tgt::possible_destination_levels(operand.level()) {
                for &destination_dtype in
                    iter::once(&operand_dtype).chain(operand_dtype.higher_precision_types())
                {
                    results.extend(
                        gen_vector_sizes_opt(destination_dtype, level.vector_bytes()).map(
                            |vector_size| {
                                // This may return Moves with identical source and destination
                                // TensorSpecs (i.e., within-level copies). These will be filtered
                                // in [apply_with_aux].
                                Action::Move(Move {
                                    source_idx: i,
                                    destination_dtype,
                                    destination_level: level,
                                    destination_layout: layout.clone(),
                                    destination_vector_size: vector_size,
                                })
                            },
                        ),
                    )
                }
            }
        }

        // For cache destinations, generate moves that preserve original layout, unless
        // it was already added. This preserves the layout's contiguity when moving
        // into cache.
        let original_layout = operand.layout();
        if !contains_source_layout {
            for level in Tgt::possible_destination_levels(operand.level()) {
                if !level.is_addressed() {
                    // This is a cache level, generate a move with the original layout
                    for &destination_dtype in
                        iter::once(&operand_dtype).chain(operand_dtype.higher_precision_types())
                    {
                        results.extend(
                            gen_vector_sizes_opt(destination_dtype, level.vector_bytes()).map(
                                |vector_size| {
                                    Action::Move(Move {
                                        source_idx: i,
                                        destination_dtype,
                                        destination_level: level,
                                        destination_layout: original_layout.clone(),
                                        destination_vector_size: vector_size,
                                    })
                                },
                            ),
                        )
                    }
                }
            }
        }
    }

    results.into_iter()
}

/// Generate multi-dim tile sizes that differ from `tensor_shape` in at most
/// `max_tiled_dims` dimensions (and in at least one dimension).
fn gen_tile_sizes(
    tensor_shape: &[DimSize],
    max_tiled_dims: usize,
) -> impl Iterator<Item = Shape> + 'static {
    let tensor_shape = Shape::from_slice(tensor_shape);
    let effective_max = max_tiled_dims.min(tensor_shape.len());
    let n = tensor_shape.len();

    (1..=effective_max)
        .flat_map(move |k| (0..n).combinations(k))
        .flat_map(move |dims_to_tile| {
            let per_dim_sizes: Vec<Vec<DimSize>> = dims_to_tile
                .iter()
                .map(|&d| dim_range(tensor_shape[d], false).collect())
                .collect();
            let tensor_shape = tensor_shape.clone();
            per_dim_sizes
                .into_iter()
                .multi_cartesian_product()
                .map(move |chosen_sizes| {
                    let mut shape = tensor_shape.clone();
                    for (&dim, size) in dims_to_tile.iter().zip(chosen_sizes) {
                        shape[dim] = size;
                    }
                    shape
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{moves::Move, Action};
    use crate::shape;
    use crate::spec::arb_canonical_logical_spec;
    use crate::target::Avx2Target;
    use itertools::Itertools;
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::collections::HashSet;

    const MAX_SYNTHESIS_PARALLEL_TILE_DIMS_FOR_TEST: usize = 2;

    #[test]
    fn test_gen_tile_sizes_empty() {
        assert_eq!(gen_tile_sizes(&[], 0).count(), 0);
        assert_eq!(gen_tile_sizes(&[], 1).count(), 0);
    }

    #[test]
    fn test_gen_tile_sizes_zero_max() {
        assert_eq!(gen_tile_sizes(&shape![2, 2], 0).count(), 0);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1() {
        assert_gen_tile_sizes(shape![16], [shape![1], shape![2], shape![4], shape![8]], 1);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1_size_1() {
        assert_gen_tile_sizes(shape![1], [], 1);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_max_1() {
        assert_gen_tile_sizes(shape![2, 2], [shape![1, 2], shape![2, 1]], 1);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_max_2() {
        assert_gen_tile_sizes(shape![2, 2], [shape![1, 1], shape![1, 2], shape![2, 1]], 2);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_non_powers_of_two() {
        assert_gen_tile_sizes(
            shape![2, 3],
            [
                shape![1, 1],
                shape![1, 2],
                shape![1, 3],
                shape![2, 1],
                shape![2, 2],
            ],
            2,
        );
    }

    #[test]
    fn test_gen_tile_sizes_dim_3_max_2() {
        assert_gen_tile_sizes(
            shape![2, 2, 2],
            [
                // 1-dim tiles
                shape![1, 2, 2],
                shape![2, 1, 2],
                shape![2, 2, 1],
                // 2-dim tiles
                shape![1, 1, 2],
                shape![1, 2, 1],
                shape![2, 1, 1],
            ],
            2,
        );
    }

    fn assert_gen_tile_sizes(
        tensor_shape: Shape,
        expected: impl IntoIterator<Item = Shape>,
        max_tiled_dims: usize,
    ) {
        let expected: Vec<Shape> = expected.into_iter().sorted().collect();
        let actual: Vec<Shape> = gen_tile_sizes(&tensor_shape, max_tiled_dims)
            .sorted()
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "gen_tile_sizes({tensor_shape:?}, {max_tiled_dims}) returned {actual:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_tile_out_actions_for_2x2x2_parallel_only() {
        let output_shape = shape![2, 2, 2];
        let actions: HashSet<Action<Avx2Target>> = tile_out_actions_for_shape::<Avx2Target>(
            &output_shape,
            false,
            MultiDimTilingPolicy::ParallelOnly(2),
        )
        .collect();

        let mut expected: HashSet<Action<Avx2Target>> = HashSet::new();
        for &size in &[nz!(1u32), nz!(2u32)] {
            for dim in 0u8..3 {
                expected.insert(Action::TileOut(TileOut::SingleLoop {
                    dim,
                    size,
                    parallel: false,
                }));
            }
        }
        for tile in [
            shape![1, 2, 2],
            shape![2, 1, 2],
            shape![2, 2, 1],
            shape![1, 1, 2],
            shape![1, 2, 1],
            shape![2, 1, 1],
        ] {
            expected.insert(Action::TileOut(TileOut::MultiLoop {
                output_shape: tile,
                parallel: true,
            }));
        }

        assert_eq!(
            actions,
            expected,
            "Mismatch for tile_out_actions on [2,2,2] with ParallelOnly(2).\n\
             Extra: {:?}\nMissing: {:?}",
            actions.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&actions).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_tile_out_actions_for_2x2x2_never() {
        let output_shape = shape![2, 2, 2];
        let actions: HashSet<Action<Avx2Target>> = tile_out_actions_for_shape::<Avx2Target>(
            &output_shape,
            false,
            MultiDimTilingPolicy::Never,
        )
        .collect();

        let mut expected: HashSet<Action<Avx2Target>> = HashSet::new();
        for parallel in [false, true] {
            for &size in &[nz!(1u32), nz!(2u32)] {
                for dim in 0u8..3 {
                    expected.insert(Action::TileOut(TileOut::SingleLoop {
                        dim,
                        size,
                        parallel,
                    }));
                }
            }
        }

        assert_eq!(
            actions,
            expected,
            "Mismatch for tile_out_actions on [2,2,2] with Never.\n\
             Extra: {:?}\nMissing: {:?}",
            actions.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&actions).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_tile_out_actions_for_2x2x2_always() {
        let output_shape = shape![2, 2, 2];
        let actions: HashSet<Action<Avx2Target>> = tile_out_actions_for_shape::<Avx2Target>(
            &output_shape,
            false,
            MultiDimTilingPolicy::Always,
        )
        .collect();

        // Always mode: MultiLoop actions for both serial and parallel.
        // gen_tile_sizes with drop_given=true, multi_dim=true yields all
        // elements of {1,2}^3 except [2,2,2] itself.
        let mut expected: HashSet<Action<Avx2Target>> = HashSet::new();
        let tiles = [
            shape![1, 1, 1],
            shape![1, 1, 2],
            shape![1, 2, 1],
            shape![2, 1, 1],
            shape![1, 2, 2],
            shape![2, 1, 2],
            shape![2, 2, 1],
        ];
        for parallel in [false, true] {
            for tile in &tiles {
                expected.insert(Action::TileOut(TileOut::MultiLoop {
                    output_shape: tile.clone(),
                    parallel,
                }));
            }
        }

        assert_eq!(
            actions,
            expected,
            "Mismatch for tile_out_actions on [2,2,2] with Always.\n\
             Extra: {:?}\nMissing: {:?}",
            actions.difference(&expected).collect::<Vec<_>>(),
            expected.difference(&actions).collect::<Vec<_>>(),
        );
    }

    proptest! {
        #[test]
        fn test_multi_dim_tiling_never_mode_uses_single_loop_actions(
            spec in arb_canonical_logical_spec::<Avx2Target>(None)
                .prop_filter("Spec must have unique output", |spec| spec.unique_output_index().is_some())
        ) {
            let output_shape = spec.parameter_shape(spec.unique_output_index().unwrap());

            prop_assert!(
                tile_out_actions_for_shape::<Avx2Target>(
                    &output_shape,
                    spec.serial_only(),
                    MultiDimTilingPolicy::Never,
                )
                .all(|action| matches!(action, Action::TileOut(TileOut::SingleLoop { .. }))),
                "expected only single-loop tile actions in Never mode"
            );
        }

        #[test]
        fn test_multi_dim_tiling_parallel_only_mode_limits_parallel_multi_loop_actions(
            spec in arb_canonical_logical_spec::<Avx2Target>(None)
                .prop_filter("Spec must have unique output", |spec| spec.unique_output_index().is_some())
        ) {
            let output_shape = spec.parameter_shape(spec.unique_output_index().unwrap());

            prop_assert!(
                tile_out_actions_for_shape::<Avx2Target>(
                    &output_shape,
                    spec.serial_only(),
                    MultiDimTilingPolicy::ParallelOnly(MAX_SYNTHESIS_PARALLEL_TILE_DIMS_FOR_TEST),
                )
                .all(|action| {
                    if let Action::TileOut(TileOut::MultiLoop {
                        output_shape: tile_shape,
                        parallel,
                    }) = action
                    {
                        parallel
                            && count_tiled_dims(&output_shape, &tile_shape)
                                <= MAX_SYNTHESIS_PARALLEL_TILE_DIMS_FOR_TEST
                    } else {
                        true
                    }
                }),
                "parallel multi-loop actions must respect ParallelOnly limit"
            );
        }

        /// Test that [MultiDimTilingPolicy::ParallelOnly] prevents MultiLoops
        /// with too many tiled dimensions.
        #[test]
        fn test_parallel_only_mode_never_visits_too_many_dim_parallel_multi_loop_actions(
            spec in arb_canonical_logical_spec::<Avx2Target>(None)
                .prop_filter("Spec must have unique output", |spec| spec.unique_output_index().is_some())
        ) {
            let output_shape = spec.parameter_shape(spec.unique_output_index().unwrap());
            let actions = tile_out_actions_for_shape::<Avx2Target>(
                &output_shape,
                spec.serial_only(),
                MultiDimTilingPolicy::ParallelOnly(MAX_SYNTHESIS_PARALLEL_TILE_DIMS_FOR_TEST),
            );
            for action in actions {
                if let Action::TileOut(TileOut::MultiLoop {
                    output_shape: tile_shape,
                    parallel: true,
                }) = action
                {
                    prop_assert!(
                        count_tiled_dims(&output_shape, &tile_shape)
                            <= MAX_SYNTHESIS_PARALLEL_TILE_DIMS_FOR_TEST
                    );
                }
            }
        }

        #[test]
        fn test_move_actions_never_returns_duplicates(
            spec in arb_canonical_logical_spec::<Avx2Target>(None)
        ) {
            let actions = move_actions::<Avx2Target>(&spec).collect::<Vec<_>>();
            let mut seen_moves = HashSet::new();
            let mut duplicate_count = 0;
            for action in &actions {
                if let Action::Move(Move {
                    source_idx,
                    destination_level,
                    destination_layout,
                    destination_dtype,
                    destination_vector_size,
                }) = action
                {
                    let move_key = (
                        source_idx,
                        destination_level,
                        destination_layout,
                        destination_dtype,
                        destination_vector_size,
                    );
                    if !seen_moves.insert(move_key) {
                        duplicate_count += 1;
                    }
                }
            }
            assert_eq!(
                duplicate_count, 0,
                "Found {duplicate_count} duplicate Move actions"
            );
        }

        #[test]
        fn test_move_actions_preserves_layout_for_cache_destinations(
            spec in arb_canonical_logical_spec::<Avx2Target>(None)
        ) {
            let actions = move_actions::<Avx2Target>(&spec).collect::<Vec<_>>();
            let operands = spec.parameters();

            let mut cache_moves_by_operand = vec![HashSet::new(); operands.len()];
            for action in &actions {
                if let Action::Move(Move {
                    source_idx,
                    destination_level,
                    destination_layout,
                    ..
                }) = action
                {
                    if !destination_level.is_addressed() {
                        cache_moves_by_operand[usize::from(*source_idx)].insert((
                            *destination_level,
                            destination_layout.clone(),
                        ));
                    }
                }
            }

            for (operand_idx, operand) in operands.iter().enumerate() {
                let seen_destination_levels: HashSet<_> = cache_moves_by_operand[operand_idx]
                    .iter()
                    .map(|(destination_level, _)| destination_level)
                    .collect();
                for destination_level in seen_destination_levels {
                    if cache_moves_by_operand[operand_idx].iter().any(|(_, layout)| {
                        layout == operand.layout()
                    }) {
                        continue;
                    }
                    prop_assert!(
                        false,
                        "No layout-preserving move found for operand {} (layout: {:?}) to cache level {:?}.",
                        operand_idx,
                        operand.layout(),
                        destination_level
                    );
                }
            }
        }
    }

    fn count_tiled_dims(output_shape: &[DimSize], tile_shape: &[DimSize]) -> usize {
        output_shape
            .iter()
            .zip(tile_shape)
            .filter(|(output_dim, tile_dim)| output_dim != tile_dim)
            .count()
    }
}
