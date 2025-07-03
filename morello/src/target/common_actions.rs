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
    utils::is_power_of_two_u32,
};
use itertools::Either;
use smallvec::smallvec;
use std::iter::{self, once};

/// Whether `tile_out` actions should tile in all dimensions per Spec.
const MULTI_DIM_TILING: bool = false;
/// An empirically chosen initial capacity for the [LogicalSpec::move_actions] results buffer.
const MOVE_RESULTS_CAPACITY: usize = 16;

// TODO: Avoid boxed trait object return type
pub fn tile_out_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
    let serial_only = spec.serial_only();
    let Some(output_idx) = spec.unique_output_index() else {
        return Box::new(iter::empty());
    };
    let output_shape = spec.parameter_shape(output_idx);
    let multi_dim = MULTI_DIM_TILING || !serial_only;
    if multi_dim {
        // TODO: Simplfy following, knowing multi_dim is true.
        Box::new(
            gen_tile_sizes::<Tgt>(&output_shape, true, multi_dim).flat_map(move |tile_shape| {
                let left = once(Action::TileOut(TileOut::MultiLoop {
                    output_shape: tile_shape.clone(),
                    parallel: false,
                }));
                let mut right = None;
                if !serial_only {
                    right = Some(Action::TileOut(TileOut::MultiLoop {
                        output_shape: tile_shape,
                        parallel: true,
                    }));
                }
                left.into_iter().chain(right)
            }),
        )
    } else {
        // Yield all output tilings up to the *maximum* dimension size so that the actions have
        // relatively stable order between Specs.
        let output_tensor_rank = output_shape.len();
        let max_dim_size =
            DimSize::try_from(output_shape.iter().map(|d| d.get()).max().unwrap()).unwrap();
        Box::new(dim_range(max_dim_size, true).flat_map(move |size| {
            (0..output_tensor_rank).flat_map(move |dim| {
                let dim = u8::try_from(dim).unwrap();
                let left = once(Action::TileOut(TileOut::SingleLoop {
                    dim,
                    size,
                    parallel: false,
                }));
                let mut right = None;
                if !serial_only {
                    right = Some(Action::TileOut(TileOut::SingleLoop {
                        dim,
                        size,
                        parallel: true,
                    }));
                }
                left.into_iter().chain(right)
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
    let PrimitiveSpecType::Matmul { accum } = typ else {
        panic!("split_actions called on non-Matmul");
    };
    if !accum {
        panic!("split_actions called on non-accumulating Matmul");
    }
    let [b, m, orig_k, n] = spec_shape[..] else {
        unreachable!();
    };

    let operands = spec.parameters();
    dim_range(orig_k, false)
        .filter(move |&new_k| {
            // TODO: Shouldn't this be rejected during application instead?
            operands[0].is_valid_tile_shape(&[b, m, new_k], false)
                && operands[1].is_valid_tile_shape(&[b, new_k, n], false)
        })
        .map(|k| Action::Split(Split { k }))
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

// TODO: Modify to return an `impl Iterator` of some kind instead of a `Box`.
fn gen_tile_sizes<Tgt: Target>(
    tensor_shape: &[DimSize],
    drop_given: bool,
    multi_dim: bool,
) -> Box<dyn Iterator<Item = Shape> + 'static> {
    if tensor_shape.is_empty() {
        return Box::new(iter::empty());
    } else if tensor_shape.len() == 1 {
        let one_dim = tensor_shape[0];
        return Box::new(dim_range(one_dim, true).filter_map(move |d| {
            if drop_given && d == one_dim {
                None
            } else {
                Some(smallvec![d])
            }
        }));
    }

    if multi_dim {
        let tensor_shape = Shape::from_slice(tensor_shape);
        Box::new(
            gen_tile_sizes::<Tgt>(&tensor_shape[1..], false, multi_dim).flat_map(move |rest| {
                let tensor_shape = tensor_shape.clone();
                dim_range(tensor_shape[0], true).flat_map(move |d| {
                    let mut new_shape = smallvec![d];
                    new_shape.extend(rest.clone());
                    if drop_given && tensor_shape[..] == new_shape[..] {
                        None
                    } else {
                        Some(new_shape)
                    }
                })
            }),
        )
    } else {
        let tensor_shape = Shape::from_slice(tensor_shape);
        let own_shape_iter = if !drop_given
            && tensor_shape
                .iter()
                .map(|d: &DimSize| d.get())
                .all(is_power_of_two_u32)
        {
            Either::Left(once(tensor_shape.clone()))
        } else {
            Either::Right(iter::empty())
        };
        let smaller_tiles_iter = (0..tensor_shape.len()).flat_map(move |dim| {
            let tensor_shape = tensor_shape.clone();
            dim_range(tensor_shape[dim], false).map(move |d| {
                let mut new_shape = tensor_shape.clone();
                new_shape[dim] = d;
                new_shape
            })
        });
        Box::new(smaller_tiles_iter.chain(own_shape_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::{moves::Move, Action};
    use crate::shape;
    use crate::spec::arb_canonical_logical_spec;
    use crate::target::X86Target;
    use itertools::Itertools;
    use proptest::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn test_gen_tile_sizes_empty() {
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, false).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], true, false).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, true).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, false).count(), 0);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1_multi_dim() {
        shared_test_gen_tile_sizes_dim_1(true);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1_single_dim() {
        shared_test_gen_tile_sizes_dim_1(false);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_multi_dim() {
        assert_gen_tile_sizes(
            shape![2, 2],
            [shape![1, 1], shape![1, 2], shape![2, 1], shape![2, 2]],
            false,
            true,
        );
        assert_gen_tile_sizes(
            shape![2, 2],
            [shape![1, 1], shape![1, 2], shape![2, 1]],
            true,
            true,
        );
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_multi_dim_non_powers_of_two() {
        assert_gen_tile_sizes(
            shape![2, 3],
            [
                shape![1, 1],
                shape![1, 2],
                shape![1, 3],
                shape![2, 1],
                shape![2, 2],
                shape![2, 3],
            ],
            false,
            true,
        );
        assert_gen_tile_sizes(
            shape![2, 3],
            [
                shape![1, 1],
                shape![1, 2],
                shape![1, 3],
                shape![2, 1],
                shape![2, 2],
            ],
            true,
            true,
        );
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_single_dim() {
        assert_gen_tile_sizes(
            shape![2, 2],
            [shape![1, 2], shape![2, 1], shape![2, 2]],
            false,
            false,
        );
        assert_gen_tile_sizes(shape![2, 2], [shape![1, 2], shape![2, 1]], true, false);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_single_dim_non_powers_of_two() {
        for drop_given in [true, false] {
            assert_gen_tile_sizes(
                shape![2, 3],
                [shape![1, 3], shape![2, 1], shape![2, 2]],
                drop_given,
                false,
            );
        }
    }

    fn shared_test_gen_tile_sizes_dim_1(multi_dim: bool) {
        assert_gen_tile_sizes(shape![1], [shape![1]], false, multi_dim);
        assert_gen_tile_sizes(shape![1], [], true, multi_dim);
        assert_gen_tile_sizes(
            shape![16],
            [shape![1], shape![2], shape![4], shape![8], shape![16]],
            false,
            multi_dim,
        );
        assert_gen_tile_sizes(
            shape![16],
            [shape![1], shape![2], shape![4], shape![8]],
            true,
            multi_dim,
        );
    }

    fn assert_gen_tile_sizes(
        tensor_shape: Shape,
        expected: impl IntoIterator<Item = Shape>,
        drop_given: bool,
        multi_dim: bool,
    ) {
        let expected: Vec<Shape> = expected.into_iter().sorted().collect();
        let d = expected.first().map_or(0, |shape| shape.len());
        assert!(expected.iter().all(|shape| shape.len() == d));

        let actual: Vec<Shape> = gen_tile_sizes::<X86Target>(&tensor_shape, drop_given, multi_dim)
            .inspect(|s| assert_eq!(s.len(), d))
            .sorted()
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "gen_tile_sizes({tensor_shape:?}, drop_given={drop_given}, serial={multi_dim}) returned {actual:?}, expected {expected:?}"
        );
    }

    proptest! {
        #[test]
        fn test_move_actions_never_returns_duplicates(
            spec in arb_canonical_logical_spec::<X86Target>(None)
        ) {
            let actions = move_actions::<X86Target>(&spec).collect::<Vec<_>>();
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
            spec in arb_canonical_logical_spec::<X86Target>(None)
        ) {
            let actions = move_actions::<X86Target>(&spec).collect::<Vec<_>>();
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
}
