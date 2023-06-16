use itertools::{iproduct, izip, Itertools};
use smallvec::{smallvec, SmallVec};

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::layout::Layout;
use crate::spec::{
    conv_infer_output_shape, gen_vector_shapes, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType,
    Spec,
};
use crate::target::{MemoryLevel, Target, X86Target};
use crate::tensorspec::TensorSpecAux;

use crate::X86MemoryLevel;

use std::hash::Hash;
use std::iter;

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

impl ToFromDependencyLatticeCoordinate for Spec<X86Target> {
    type Key = SpecKey;

    fn to_grid(&self) -> Option<(SpecKey, Vec<u32>)> {
        match self {
            Spec::Primitive(basics, primitive_aux, serial_only) => match basics.typ {
                PrimitiveSpecType::Matmul { accum } => {
                    let PrimitiveAux::Standard(auxes) = primitive_aux else {
                        unreachable!();
                    };
                    Some((
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
                    ))
                }
                PrimitiveSpecType::Conv { accum } => {
                    let PrimitiveAux::Standard(auxes) = primitive_aux else {
                        unreachable!();
                    };

                    let [b, f, c, h, w, fh, fw] = basics.spec_shape[..] else {
                        panic!("Convolution must have 7 Spec dimensions")
                    };
                    let _image_shape = [b, c, h, w];
                    let _filters_shape = [f, c, fh, fw];

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
                PrimitiveSpecType::Load | PrimitiveSpecType::Store | PrimitiveSpecType::Zero => {
                    let mapping_level = match primitive_aux {
                        PrimitiveAux::Standard(auxes) => &auxes[0].level,
                        PrimitiveAux::Move {
                            outer_aux: TensorSpecAux { level, .. },
                            ..
                        } => level,
                    };
                    Some((
                        match basics.typ {
                            PrimitiveSpecType::Load => SpecKey::Move {
                                is_load: true,
                                dtype: basics.dtype,
                            },
                            PrimitiveSpecType::Store => SpecKey::Move {
                                is_load: false,
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
            Spec::Compose { .. } => None,
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

                align_layout_contig_vector_shape_product::<X86Target>(&shapes, *dtype, &levels)
                    .map(|(alignments, layouts, contigs, vector_shapes)| {
                        Spec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::Matmul { accum: pt[0] == 0 },
                                spec_shape: smallvec![m, k, n],
                                dtype: *dtype,
                            },
                            PrimitiveAux::Standard(
                                izip!(contigs, alignments, layouts, vector_shapes, &levels)
                                    .map(|(contig, aligned, layout, vector_shape, level)| {
                                        TensorSpecAux {
                                            contig,
                                            aligned,
                                            layout,
                                            vector_shape,
                                            level: *level,
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                            ),
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

                align_layout_contig_vector_shape_product::<X86Target>(&shapes, *dtype, &levels)
                    .map(|(alignments, layouts, contigs, vector_shapes)| {
                        Spec::Primitive(
                            PrimitiveBasics {
                                typ: PrimitiveSpecType::Conv { accum },
                                spec_shape: spec_shape.clone(),
                                dtype: *dtype,
                            },
                            PrimitiveAux::Standard(
                                izip!(contigs, alignments, layouts, vector_shapes, &levels)
                                    .map(|(contig, aligned, layout, vector_shape, level)| {
                                        TensorSpecAux {
                                            contig,
                                            aligned,
                                            layout,
                                            vector_shape,
                                            level: *level,
                                        }
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                            pt[9] == 0,
                        )
                    })
                    .collect()
            }
            SpecKey::Move { is_load, dtype } => {
                let source_level = int_to_level(pt[pt.len() - 2]);
                let dim_sizes = &pt[..pt.len() - 2]
                    .iter()
                    .map(|&d| from_log2_dim_space(d))
                    .collect::<Shape>();
                let serial_only = pt[pt.len() - 1] == 0;

                let alignments = [true, false];
                let viable_layouts = X86Target::all_layouts_for_shape(dim_sizes);

                alignments
                    .into_iter()
                    .cartesian_product(viable_layouts.iter().cloned())
                    .cartesian_product(viable_layouts.iter().cloned())
                    .flat_map(
                        move |((source_aligned, source_layout), destination_layout)| {
                            let allowed_destination_levels =
                                X86Target::faster_destination_levels(source_level);
                            allowed_destination_levels
                                .into_iter()
                                .cartesian_product(source_layout.all_contiguous_abs().collect_vec())
                                .flat_map(move |(destination_level, source_contiguous_abs)| {
                                    let source_layout = source_layout.clone();
                                    let destination_layout = destination_layout.clone();
                                    [source_level, destination_level]
                                        .map(|lvl| {
                                            if lvl.vector_rf() {
                                                gen_vector_shapes(
                                                    Some(dim_sizes),
                                                    *dtype,
                                                    lvl.vector_bytes(),
                                                    None,
                                                )
                                                .map(Some)
                                                .collect::<Vec<_>>()
                                            } else {
                                                vec![None]
                                            }
                                        })
                                        .into_iter()
                                        .multi_cartesian_product()
                                        .map(
                                            move |vector_shape_pair| match &vector_shape_pair[..] {
                                                [source_vector_shape, destination_vector_shape] => {
                                                    Spec::Primitive(
                                                        PrimitiveBasics {
                                                            typ: if *is_load {
                                                                PrimitiveSpecType::Load
                                                            } else {
                                                                PrimitiveSpecType::Store
                                                            },
                                                            spec_shape: dim_sizes.clone(),
                                                            dtype: *dtype,
                                                        },
                                                        PrimitiveAux::Move {
                                                            outer_aux: TensorSpecAux {
                                                                contig: source_contiguous_abs,
                                                                aligned: source_aligned,
                                                                level: source_level,
                                                                layout: source_layout.clone(),
                                                                vector_shape: source_vector_shape
                                                                    .clone(),
                                                            },
                                                            inner_level: destination_level,
                                                            inner_layout: destination_layout
                                                                .clone(),
                                                            inner_vector_shape:
                                                                destination_vector_shape.clone(),
                                                        },
                                                        serial_only,
                                                    )
                                                }
                                                _ => unreachable!(),
                                            },
                                        )
                                })
                        },
                    )
                    .collect::<Vec<_>>()
            }
            SpecKey::Zero { dtype } => {
                let serial_only = pt[pt.len() - 1] == 0;
                let level = int_to_level(pt[pt.len() - 2]);
                let dim_sizes = pt[..pt.len() - 2]
                    .iter()
                    .map(|&d| from_log2_dim_space(d))
                    .collect::<Shape>();
                align_layout_contig_vector_shape_product::<X86Target>(
                    &[dim_sizes.clone()],
                    *dtype,
                    &[level],
                )
                .map(|(alignments, layouts, contigs, vector_shapes)| {
                    debug_assert_eq!(alignments.len(), 1);
                    debug_assert_eq!(layouts.len(), 1);
                    debug_assert_eq!(contigs.len(), 1);
                    debug_assert_eq!(vector_shapes.len(), 1);
                    Spec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Zero,
                            spec_shape: dim_sizes.clone(),
                            dtype: *dtype,
                        },
                        PrimitiveAux::Standard(vec![TensorSpecAux {
                            contig: contigs[0],
                            aligned: alignments[0],
                            level,
                            layout: layouts[0].clone(),
                            vector_shape: vector_shapes[0].clone(),
                        }]),
                        serial_only,
                    )
                })
                .collect()
            }
        }
    }
}

fn align_layout_contig_vector_shape_product<'s, Tgt: Target>(
    shapes: &'s [Shape],
    dtype: Dtype,
    levels: &'s [Tgt::Level],
) -> impl Iterator<
    Item = (
        SmallVec<[bool; 3]>,
        SmallVec<[Layout; 3]>,
        SmallVec<[Contig; 3]>,
        SmallVec<[Option<Shape>; 3]>,
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
            // - contig.
            let contigs = layouts
                .iter()
                // TODO: Make iterator cloneable instead of collecting into Vec.
                .map(|l| l.all_contiguous_abs().collect::<Vec<_>>())
                .multi_cartesian_product();
            // - vector shape
            let vector_shapes = levels
                .iter()
                // TODO: Make iterator cloneable instead of collecting into Vec.
                .enumerate()
                .map(|(idx, lvl)| {
                    //  TODO: Avoid this collection.
                    if lvl.vector_rf() {
                        gen_vector_shapes(Some(&shapes[idx]), dtype, lvl.vector_bytes(), None)
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
                vector_shapes
            )
        })
        .map(|(alignments, layouts, contigs, vector_shapes)| {
            // TODO: Collect into SmallVecs immediately instead of converting.
            (
                SmallVec::<[_; 3]>::from(alignments),
                SmallVec::<[_; 3]>::from(layouts),
                SmallVec::<[_; 3]>::from(contigs),
                SmallVec::<[_; 3]>::from(vector_shapes),
            )
        })
}

fn level_to_int(lvl: &X86MemoryLevel) -> u8 {
    match &lvl {
        X86MemoryLevel::GL => 3,
        X86MemoryLevel::L1 => 2,
        X86MemoryLevel::VRF => 1,
        X86MemoryLevel::RF => 0,
    }
}

fn int_to_level(i: u32) -> X86MemoryLevel {
    match i {
        0 => X86MemoryLevel::RF,
        1 => X86MemoryLevel::VRF,
        2 => X86MemoryLevel::L1,
        3 => X86MemoryLevel::GL,
        _ => panic!("Invalid level"),
    }
}

fn iter_vector_shape_args<M: MemoryLevel>(
    level: M,
    outer_shape: &[usize],
    dtype: Dtype,
) -> Box<dyn Iterator<Item = Option<Shape>>> {
    if level.vector_bytes() == 0 {
        Box::new(iter::once(None))
    } else {
        Box::new(
            gen_vector_shapes(
                None,
                dtype,
                level.vector_bytes(),
                Some(outer_shape.len().try_into().unwrap()),
            )
            .map(Some),
        )
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
