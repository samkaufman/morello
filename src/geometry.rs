use itertools::{iproduct, Itertools};
use smallvec::{smallvec, SmallVec, ToSmallVec};

use crate::common::{Contig, DimSize, Dtype, Shape};

use crate::layout::Layout;
use crate::spec::{gen_vector_shapes, Spec};
use crate::target::{MemoryLevel, Target, X86Target};
use crate::tensorspec::TensorSpec;
use crate::utils::bit_length_u32;
use crate::X86MemoryLevel;

use std::hash::Hash;
use std::iter;

pub trait ToFromDependencyLatticeCoordinate {
    type Key: Eq + Hash;
    type InnerKey: Eq + Hash;

    fn to_grid(&self) -> Option<(Self::Key, Vec<u32>, Self::InnerKey)>;
    fn from_grid(key: &Self::Key, pt: &[u32], inner_key: &Self::InnerKey) -> Self;
    // TODO: Return an iterator instead.
    fn inner_keys_for_grid_pt(key: &Self::Key, pt: &[u32]) -> Vec<Self::InnerKey>;
}

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SpecKey {
    Matmul {
        dtype: Dtype,
    },
    Move {
        is_load: bool,
        outer_tensor_spec_key: TensorSpecKey,
    },
    Zero,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SpecInnerKey {
    Matmul {
        contiguous_abstractions: SmallVec<[Contig; 3]>,
        alignments: SmallVec<[bool; 3]>,
        layouts: SmallVec<[Layout; 3]>,
        vector_shapes: SmallVec<[Option<Shape>; 3]>,
    },
    Move {
        outer_tensor_spec_inner_key: TensorSpecInnerKey,
        inner_layout: Layout,
        inner_vector_shape: Option<Shape>,
    },
    Zero,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TensorSpecKey {
    dtype: Dtype,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TensorSpecInnerKey {
    layout: Layout,
    vector_shape: Option<Shape>,
}

impl ToFromDependencyLatticeCoordinate for Spec<X86Target> {
    type Key = SpecKey;
    type InnerKey = SpecInnerKey;

    fn to_grid(&self) -> Option<(SpecKey, Vec<u32>, SpecInnerKey)> {
        match self {
            Spec::Matmul {
                accum,
                m,
                k,
                n,
                dtype,
                contiguous_abstractions,
                alignments,
                levels,
                layouts,
                vector_shapes,
                serial_only,
            } => Some((
                SpecKey::Matmul { dtype: *dtype },
                [
                    if *accum { 0 } else { 1 },
                    to_log2_dim_space(*m)?,
                    to_log2_dim_space(*k)?,
                    to_log2_dim_space(*n)?,
                    if *serial_only { 0 } else { 1 },
                ]
                .into_iter()
                .chain(levels.iter().map(|l| level_to_int(l).into()))
                .collect(),
                SpecInnerKey::Matmul {
                    contiguous_abstractions: contiguous_abstractions.clone(),
                    alignments: alignments.clone(),
                    layouts: layouts.clone(),
                    vector_shapes: vector_shapes.clone(),
                },
            )),
            Spec::Load {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only,
            }
            | Spec::Store {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only,
            } => {
                let (ts_key, ts_pt, ts_inner_key) = tensorspec_to_lattice(outer_tensor_spec);
                Some((
                    SpecKey::Move {
                        is_load: matches!(self, Spec::Load { .. }),
                        outer_tensor_spec_key: ts_key,
                    },
                    ts_pt
                        .iter()
                        .chain(iter::once(&level_to_int(inner_level).into()))
                        .chain(iter::once(if *serial_only { &0 } else { &1 }))
                        .cloned()
                        .collect(),
                    SpecInnerKey::Move {
                        outer_tensor_spec_inner_key: ts_inner_key,
                        inner_layout: inner_layout.clone(),
                        inner_vector_shape: inner_vector_shape.clone(),
                    },
                ))
            }
            Spec::Zero {
                tensor_spec: _,
                serial_only: _,
            } => todo!(),
        }
    }

    fn from_grid(key: &SpecKey, pt: &[u32], inner_key: &SpecInnerKey) -> Self {
        match (key, inner_key) {
            (
                SpecKey::Matmul { dtype },
                SpecInnerKey::Matmul {
                    contiguous_abstractions,
                    alignments,
                    layouts,
                    vector_shapes,
                },
            ) => match pt[..] {
                [accum, m, k, n, serial_only, lvl0, lvl1, lvl2] => Spec::Matmul {
                    accum: accum == 0,
                    m: from_log2_dim_space(m),
                    k: from_log2_dim_space(k),
                    n: from_log2_dim_space(n),
                    dtype: *dtype,
                    contiguous_abstractions: contiguous_abstractions.clone(),
                    alignments: alignments.clone(),
                    levels: smallvec![int_to_level(lvl0), int_to_level(lvl1), int_to_level(lvl2)],
                    layouts: layouts.clone(),
                    vector_shapes: {
                        debug_assert_eq!(vector_shapes.len(), 3);
                        debug_assert!(vector_shapes
                            .iter()
                            .all(|v| v.is_none() || v.as_ref().unwrap().len() == 2));
                        vector_shapes.clone()
                    },
                    serial_only: serial_only == 0,
                },
                _ => panic!("Grid point had unexpected length"),
            },
            (
                SpecKey::Move {
                    is_load: _,
                    outer_tensor_spec_key: _,
                },
                SpecInnerKey::Move {
                    outer_tensor_spec_inner_key: _,
                    inner_layout: _,
                    inner_vector_shape: _,
                },
            ) => todo!(),
            (SpecKey::Zero, SpecInnerKey::Zero) => todo!(),
            _ => panic!("Mismatched key and inner key"),
        }
    }

    fn inner_keys_for_grid_pt(key: &Self::Key, pt: &[u32]) -> Vec<Self::InnerKey> {
        match key {
            SpecKey::Matmul { dtype } => {
                // TODO: Relying on indices below is fragile.
                let m = pt[1] + 1;
                let k = pt[2] + 1;
                let n = pt[3] + 1;
                let levels = pt[5..8]
                    .iter()
                    .map(|&i| int_to_level(i))
                    .collect::<Vec<_>>();

                let shapes = [[m, k], [k, n], [m, n]];

                // For each operand:
                // - alignment
                // - layout
                let align_prod = iter::repeat([true, false])
                    .take(3)
                    .multi_cartesian_product();
                let layout_prod = shapes
                    .iter()
                    .map(|s| X86Target::all_layouts_for_shape(s))
                    .multi_cartesian_product();
                align_prod
                    .cartesian_product(layout_prod)
                    .flat_map(|(alignments, layouts)| {
                        // - contig.
                        let contigs = layouts
                            .iter()
                            // TODO: Make cloneable instead of collecting into Vec.
                            .map(|l| l.all_contiguous_abs().collect::<Vec<_>>())
                            .multi_cartesian_product();
                        // - vector shape
                        let vector_shapes = levels
                            .iter()
                            // TODO: Make cloneable instead of collecting into Vec.
                            .map(|lvl| {
                                //  TODO: Avoid this collection.
                                let mut out = vec![];
                                if lvl.vector_rf() {
                                    out.extend(
                                        gen_vector_shapes(
                                            None,
                                            *dtype,
                                            lvl.vector_bytes(),
                                            Some(2),
                                        )
                                        .map(Some),
                                    )
                                } else {
                                    out.push(None);
                                }
                                out
                            })
                            .multi_cartesian_product();
                        iproduct!(
                            iter::once(alignments),
                            iter::once(layouts),
                            contigs,
                            vector_shapes
                        )
                    })
                    .map(
                        |(alignments, layouts, contigs, vector_shapes)| SpecInnerKey::Matmul {
                            contiguous_abstractions: contigs.into_iter().collect(),
                            alignments: alignments.into(),
                            layouts: layouts.into(),
                            vector_shapes: vector_shapes
                                .into_iter()
                                .map(|v| v.as_ref().map(|v| v.to_smallvec()))
                                .collect(),
                        },
                    )
                    .collect()
            }
            SpecKey::Move {
                is_load: _,
                outer_tensor_spec_key: _,
            } => todo!(),
            SpecKey::Zero => todo!(),
        }
    }
}

fn tensorspec_to_lattice(
    ts: &TensorSpec<X86Target>,
) -> (TensorSpecKey, Vec<u32>, TensorSpecInnerKey) {
    (
        TensorSpecKey { dtype: ts.dtype() },
        ts.dim_sizes()
            .iter()
            .map(|d| to_log2_dim_space(*d).unwrap())
            .chain(iter::once(level_to_int(&ts.level()).into()))
            .collect(),
        TensorSpecInnerKey {
            layout: ts.layout(),
            vector_shape: ts.vector_shape().cloned(),
        },
    )
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
            .map(|s| Some(s.into())),
        )
    }
}

fn to_log2_dim_space(dim: DimSize) -> Option<u32> {
    let r = bit_length_u32(dim) - 1;
    if from_log2_dim_space(r) == dim {
        Some(r)
    } else {
        None
    }
}

fn from_log2_dim_space(log2_dim: u32) -> DimSize {
    1 << log2_dim
}
