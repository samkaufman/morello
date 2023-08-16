use itertools::{iproduct, izip, Itertools};
use smallvec::{smallvec, SmallVec};
use std::hash::Hash;
use std::iter;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::layout::Layout;
use crate::spec::{
    conv_infer_output_shape, gen_vector_sizes, LogicalSpec, PrimitiveBasics, PrimitiveSpecType,
};
use crate::target::{CpuMemoryLevel, MemoryLevel, Target, X86Target};
use crate::tensorspec::TensorSpecAux;

/// A type which has a canonical map to a grid of data dependencies.
///
/// A "grid" is a product of non-negative integers ordered such that a product of zeros is the
/// bottom and there is no top. An object is mapped to a lattice such that it depends on all
/// other objects mapped to points with no greater coordinate.
///
/// The specific grid to which the type maps is identified by the `Key` type parameter.
///
/// The grid is not a *complete* description of the type's data dependencies. There may be data
/// dependencies between grids or entirely undescribed by the trait implementation.
pub trait ToFromDependencyLatticeCoordinate: Sized {
    type Key: Eq + Hash;

    fn to_grid(&self) -> Option<(Self::Key, Vec<u32>)>;

    // TODO: Return an iterator instead.
    fn objects_in_grid_pt(key: &Self::Key, pt: &[u32]) -> Vec<Self>;
}

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SpecKey {
    Matmul { dtype: Dtype },
    Conv { dtype: Dtype },
    Move { is_load: bool, dtype: Dtype },
    Zero { dtype: Dtype },
}

impl ToFromDependencyLatticeCoordinate for LogicalSpec<X86Target> {
    type Key = SpecKey;

    fn to_grid(&self) -> Option<(SpecKey, Vec<u32>)> {
        match self {
            LogicalSpec::Primitive(basics, auxes, serial_only) => match basics.typ {
                PrimitiveSpecType::Matmul { accum } => Some((
                    SpecKey::Matmul {
                        dtype: basics.dtype,
                    },
                    [
                        if accum { 0 } else { 1 },
                        to_log2_dim_space(basics.spec_shape[0])?,
                        to_log2_dim_space(basics.spec_shape[1])?,
                        to_log2_dim_space(basics.spec_shape[2])?,
                        if *serial_only { 0 } else { 1 },
                    ]
                    .into_iter()
                    .chain(auxes.iter().map(|a| level_to_int(&a.level).into()))
                    .collect(),
                )),
                PrimitiveSpecType::Conv { accum } => {
                    let [b, f, c, h, w, fh, fw] = basics.spec_shape[..] else {
                        panic!("Convolution must have 7 Spec dimensions")
                    };
                    let shape_vec = [b - 1, f - 1, c - 1, h - fh, w - fw, fh - 1, fw - 1];
                    Some((
                        SpecKey::Conv {
                            dtype: basics.dtype,
                        },
                        iter::once(if accum { 0 } else { 1 })
                            .chain(shape_vec.into_iter())
                            .chain(iter::once(if *serial_only { 0 } else { 1 }))
                            .chain(auxes.iter().map(|a| level_to_int(&a.level).into()))
                            .collect(),
                    ))
                }
                PrimitiveSpecType::Move | PrimitiveSpecType::Zero => {
                    let mapping_level = &auxes[0].level;
                    Some((
                        match basics.typ {
                            PrimitiveSpecType::Move => SpecKey::Move {
                                is_load: {
                                    // geometry breaks loads and stores into separate tables, which
                                    // makes sense after combining the two into a single Move Spec.
                                    // For now, we preserve this separation by determining whether
                                    // or not this is a load or a store by comparing operands.
                                    mapping_level < &auxes[0].level
                                },
                                dtype: basics.dtype,
                            },
                            PrimitiveSpecType::Zero => SpecKey::Zero {
                                dtype: basics.dtype,
                            },
                            _ => unreachable!(),
                        },
                        basics
                            .spec_shape
                            .iter()
                            .map(|d| to_log2_dim_space(*d).unwrap())
                            .chain(iter::once(DimSize::from(level_to_int(mapping_level))))
                            .chain(iter::once(if *serial_only { 0 } else { 1 }))
                            .collect(),
                    ))
                }
            },
            LogicalSpec::Compose { .. } => None,
        }
    }

    fn objects_in_grid_pt(key: &Self::Key, pt: &[u32]) -> Vec<Self> {
        // TODO: Relying on indices in the below implementations is fragile. Fix that.
        match key {
            SpecKey::Matmul { dtype } => {
                let m = pt[1] + 1;
                let k = pt[2] + 1;
                let n = pt[3] + 1;
                let levels = pt[5..8]
                    .iter()
                    .map(|&i| int_to_level(i))
                    .collect::<Vec<_>>();

                let shapes = [smallvec![m, k], smallvec![k, n], smallvec![m, n]];

                align_layout_contig_vector_size_product::<X86Target>(&shapes, *dtype, &levels)
                    .map(|(alignments, layouts, contigs, vector_sizes)| {
                        LogicalSpec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::Matmul { accum: pt[0] == 0 },
                                spec_shape: smallvec![m, k, n],
                                dtype: *dtype,
                            },
                            izip!(contigs, alignments, layouts, vector_sizes, &levels)
                                .map(|(contig, aligned, layout, vector_size, level)| {
                                    TensorSpecAux {
                                        contig,
                                        aligned,
                                        layout,
                                        vector_size,
                                        level: *level,
                                    }
                                })
                                .collect::<Vec<_>>(),
                            pt[4] == 0,
                        )
                    })
                    .collect()
            }
            SpecKey::Conv { dtype } => {
                let accum = pt[0] == 0;
                let [pb, pf, pc, ph, pw, pfh, pfw] = pt[..] else {
                    panic!();
                };

                let spec_shape = smallvec![
                    pb + 1,
                    pf + 1,
                    pc + 1,
                    ph + pfh + 1,
                    pw + pfw + 1,
                    pfh + 1,
                    pfw + 1
                ];

                let image_shape =
                    smallvec![spec_shape[0], spec_shape[2], spec_shape[3], spec_shape[4]];
                let filters_shape =
                    smallvec![spec_shape[1], spec_shape[2], spec_shape[5], spec_shape[6]];
                let output_shape = conv_infer_output_shape(&image_shape, &filters_shape);
                debug_assert_eq!(
                    output_shape,
                    PrimitiveSpecType::Conv { accum }
                        .infer_output_shape(&[&image_shape, &filters_shape])
                );
                let shapes = [image_shape, filters_shape, output_shape];

                let levels = pt[9..12]
                    .iter()
                    .map(|&i| int_to_level(i))
                    .collect::<Vec<_>>();

                align_layout_contig_vector_size_product::<X86Target>(&shapes, *dtype, &levels)
                    .map(|(alignments, layouts, contigs, vector_sizes)| {
                        LogicalSpec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::Conv { accum },
                                spec_shape: spec_shape.clone(),
                                dtype: *dtype,
                            },
                            izip!(contigs, alignments, layouts, vector_sizes, &levels)
                                .map(|(contig, aligned, layout, vector_size, level)| {
                                    TensorSpecAux {
                                        contig,
                                        aligned,
                                        layout,
                                        vector_size,
                                        level: *level,
                                    }
                                })
                                .collect::<Vec<_>>(),
                            pt[9] == 0,
                        )
                    })
                    .collect()
            }
            SpecKey::Move { is_load: _, dtype } => {
                let source_level = int_to_level(pt[pt.len() - 2]);
                let shape = &pt[..pt.len() - 2]
                    .iter()
                    .map(|&d| from_log2_dim_space(d))
                    .collect::<Shape>();
                let serial_only = pt[pt.len() - 1] == 0;

                let alignments = [true, false];
                let viable_layouts = X86Target::all_layouts_for_shape(shape);

                alignments
                    .into_iter()
                    .cartesian_product(alignments)
                    .cartesian_product(viable_layouts.iter().cloned())
                    .cartesian_product(viable_layouts.iter().cloned())
                    .flat_map(
                        move |(
                            ((source_aligned, dest_aligned), source_layout),
                            destination_layout,
                        )| {
                            X86Target::possible_destination_levels(source_level)
                                .into_iter()
                                .cartesian_product(source_layout.all_contiguous_abs().collect_vec())
                                .cartesian_product(
                                    destination_layout.all_contiguous_abs().collect_vec(),
                                )
                                .flat_map(move |((dest_level, source_contig), dest_contig)| {
                                    let source_layout = source_layout.clone();
                                    let destination_layout = destination_layout.clone();
                                    [source_level, dest_level]
                                        .map(|lvl| {
                                            if lvl.vector_rf() {
                                                gen_vector_sizes(
                                                    Some(shape),
                                                    *dtype,
                                                    lvl.vector_bytes(),
                                                )
                                                .map(Some)
                                                .collect::<Vec<_>>()
                                            } else {
                                                vec![None]
                                            }
                                        })
                                        .into_iter()
                                        .multi_cartesian_product()
                                        .map(move |vector_size_pair| match &vector_size_pair[..] {
                                            [source_vector_size, destination_vector_size] => {
                                                LogicalSpec::Primitive(
                                                    PrimitiveBasics {
                                                        typ: PrimitiveSpecType::Move,
                                                        spec_shape: shape.clone(),
                                                        dtype: *dtype,
                                                    },
                                                    vec![
                                                        TensorSpecAux {
                                                            contig: source_contig,
                                                            aligned: source_aligned,
                                                            level: source_level,
                                                            layout: source_layout.clone(),
                                                            vector_size: *source_vector_size,
                                                        },
                                                        TensorSpecAux {
                                                            contig: dest_contig,
                                                            aligned: dest_aligned,
                                                            level: dest_level,
                                                            layout: destination_layout.clone(),
                                                            vector_size: *destination_vector_size,
                                                        },
                                                    ],
                                                    serial_only,
                                                )
                                            }
                                            _ => unreachable!(),
                                        })
                                })
                        },
                    )
                    .collect::<Vec<_>>()
            }
            SpecKey::Zero { dtype } => {
                let serial_only = pt[pt.len() - 1] == 0;
                let level = int_to_level(pt[pt.len() - 2]);
                let shape = pt[..pt.len() - 2]
                    .iter()
                    .map(|&d| from_log2_dim_space(d))
                    .collect::<Shape>();
                align_layout_contig_vector_size_product::<X86Target>(
                    &[shape.clone()],
                    *dtype,
                    &[level],
                )
                .map(|(alignments, layouts, contigs, vector_sizes)| {
                    debug_assert_eq!(alignments.len(), 1);
                    debug_assert_eq!(layouts.len(), 1);
                    debug_assert_eq!(contigs.len(), 1);
                    debug_assert_eq!(vector_sizes.len(), 1);
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Zero,
                            spec_shape: shape.clone(),
                            dtype: *dtype,
                        },
                        vec![TensorSpecAux {
                            contig: contigs[0],
                            aligned: alignments[0],
                            level,
                            layout: layouts[0].clone(),
                            vector_size: vector_sizes[0],
                        }],
                        serial_only,
                    )
                })
                .collect()
            }
        }
    }
}

fn align_layout_contig_vector_size_product<'s, Tgt: Target>(
    shapes: &'s [Shape],
    dtype: Dtype,
    levels: &'s [Tgt::Level],
) -> impl Iterator<
    Item = (
        SmallVec<[bool; 3]>,
        SmallVec<[Layout; 3]>,
        SmallVec<[Contig; 3]>,
        SmallVec<[Option<DimSize>; 3]>,
    ),
> + 's {
    assert_eq!(shapes.len(), levels.len());
    let align_prod = iter::repeat([true, false])
        .take(shapes.len())
        .multi_cartesian_product();
    let layout_prod = shapes
        .iter()
        .map(|s| X86Target::all_layouts_for_shape(s))
        .multi_cartesian_product();
    align_prod
        .cartesian_product(layout_prod)
        .flat_map(move |(alignments, layouts)| {
            let contigs = layouts
                .iter()
                // TODO: Make iterator cloneable instead of collecting into Vec.
                .map(|l| l.all_contiguous_abs().collect::<Vec<_>>())
                .multi_cartesian_product();
            let vector_sizes = levels
                .iter()
                // TODO: Make iterator cloneable instead of collecting into Vec.
                .enumerate()
                .map(|(idx, lvl)| {
                    //  TODO: Avoid this collection.
                    if lvl.vector_rf() {
                        gen_vector_sizes(Some(&shapes[idx]), dtype, lvl.vector_bytes())
                            .map(Some)
                            .collect::<SmallVec<[_; 3]>>()
                    } else {
                        smallvec![None]
                    }
                })
                .multi_cartesian_product();
            iproduct!(
                iter::once(alignments),
                iter::once(layouts),
                contigs,
                vector_sizes
            )
        })
        .map(|(alignments, layouts, contigs, vector_sizes)| {
            // TODO: Collect into SmallVecs immediately instead of converting.
            (
                SmallVec::<[_; 3]>::from(alignments),
                SmallVec::<[_; 3]>::from(layouts),
                SmallVec::<[_; 3]>::from(contigs),
                SmallVec::<[_; 3]>::from(vector_sizes),
            )
        })
}

fn level_to_int(lvl: &CpuMemoryLevel) -> u8 {
    match &lvl {
        CpuMemoryLevel::GL => 3,
        CpuMemoryLevel::L1 => 2,
        CpuMemoryLevel::VRF => 1,
        CpuMemoryLevel::RF => 0,
    }
}

fn int_to_level(i: u32) -> CpuMemoryLevel {
    match i {
        0 => CpuMemoryLevel::RF,
        1 => CpuMemoryLevel::VRF,
        2 => CpuMemoryLevel::L1,
        3 => CpuMemoryLevel::GL,
        _ => panic!("Invalid level"),
    }
}

fn to_log2_dim_space(dim: DimSize) -> Option<u32> {
    assert!(dim > 0);
    Some(dim - 1)
    // let r = bit_length_u32(dim) - 1;
    // if from_log2_dim_space(r) == dim {
    //     Some(r)
    // } else {
    //     None
    // }
}

fn from_log2_dim_space(log2_dim: u32) -> DimSize {
    // 1 << log2_dim
    log2_dim + 1
}
