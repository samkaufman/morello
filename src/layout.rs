use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{cmp::min, collections::HashSet, fmt::Display, hash::Hash};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    expr::{AffineExpr, Term},
    opaque_symbol::OpaqueSymbol,
    target::Target,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Deserialize, Serialize)]
pub enum Layout {
    Standard {
        dim_order: SmallVec<[u8; 5]>,
    },
    Packed {
        dim_count: u8,
        strip_dim: u8,
        strip_size: DimSize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BufferExprTerm {
    TileIdx(u8, OpaqueSymbol),
    // TODO: Probably don't need the OpaqueSymbol for Pt
    Pt(u8, OpaqueSymbol),
}

impl Layout {
    pub fn buffer_indexing_expr(
        &self,
        expr_id: &OpaqueSymbol,
        concrete_shape: &[DimSize],
    ) -> AffineExpr<BufferExprTerm> {
        match self {
            Layout::Standard { dim_order } => {
                debug_assert_eq!(dim_order.len(), concrete_shape.len());
                regular_index_expr(expr_id, dim_order, concrete_shape)
            }
            Layout::Packed {
                dim_count: _,
                strip_dim: _,
                strip_size: _,
            } => todo!(),
        }
    }

    pub fn contiguous_full(&self) -> Contig {
        match self {
            Layout::Standard { dim_order } => dim_order.len().try_into().unwrap(),
            Layout::Packed { dim_count, .. } => dim_count + 1,
        }
    }

    fn contiguous_lub(&self, _other_layout: &Layout, _a: Contig, _b: Contig) -> Contig {
        todo!()
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> {
        match self {
            Layout::Standard { dim_order } => 0u8..(dim_order.len() + 1).try_into().unwrap(),
            Layout::Packed { dim_count, .. } => 0u8..(*dim_count + 2),
        }
    }

    fn tile_is_contiguous(&self, _contiguous_abs: Contig) -> bool {
        todo!()
    }

    // TODO: Rename and change docs; this actually returns a contiguousness abstraction.
    pub fn tile_contiguity(
        &self,
        tile_shape: &[DimSize],
        parent_shape: &[DimSize],
        parent_contiguous: Contig,
    ) -> Contig {
        match self {
            Layout::Standard { .. } => {
                if tile_shape.iter().all(|&d| d == 1) {
                    return self.contiguous_full();
                }

                let mut cnt = 1; // Skip first.
                self.inner_loop(&mut cnt, tile_shape, false, |x| {
                    parent_shape[usize::try_from(x).unwrap()]
                });
                cnt = min(cnt, parent_contiguous);
                self.inner_loop(&mut cnt, tile_shape, true, |_| 1);
                cnt
            }
            Layout::Packed {
                dim_count: _,
                strip_dim: _,
                strip_size: _,
            } => todo!(),
        }
    }

    // TODO: Rename
    pub fn inner_loop(
        &self,
        cnt: &mut u8,
        tile_shape: &[DimSize],
        one_back: bool,
        comp: impl Fn(u32) -> u32,
    ) {
        if let Layout::Standard { dim_order } = self {
            while usize::from(*cnt) < tile_shape.len() {
                let mut rev_dim_idx = usize::from(*cnt);
                if one_back {
                    rev_dim_idx += 1;
                }

                let phys_idx = usize::from(dim_order[dim_order.len() - rev_dim_idx]);
                if tile_shape[phys_idx] != comp(phys_idx.try_into().unwrap()) {
                    break;
                }
                *cnt += 1;
            }
        } else {
            panic!("inner_loop is only applicable to Standard layout variant")
        }
    }

    pub fn estimate_cache_lines<Tgt: Target>(
        &self,
        shape: &Shape,
        dtype: Dtype,
        contiguous: bool,
    ) -> u32 {
        match &self {
            Layout::Standard { dim_order: _ } => {
                let line_size = Tgt::line_size();

                if contiguous {
                    divrem::DivCeil::div_ceil(
                        shape.into_iter().product::<DimSize>() * DimSize::from(dtype.size()),
                        line_size,
                    )
                } else {
                    let lodims = self.layout_ordered_dims(shape);
                    let mut real_dims: Vec<u32> = lodims.into_iter().filter(|&d| d > 1).collect();
                    if real_dims.is_empty() {
                        real_dims.push(1);
                    }
                    real_dims[..real_dims.len() - 1].iter().product::<DimSize>()
                        * divrem::DivCeil::div_ceil(
                            real_dims.last().unwrap() * DimSize::from(dtype.size()),
                            line_size,
                        )
                }
            }
            Layout::Packed { .. } => todo!(),
        }
    }

    pub fn applies_to_shape(&self, shape: &[DimSize]) -> bool {
        match &self {
            Layout::Standard { dim_order } => {
                if !self.super_applies(shape) {
                    return false;
                }
                if shape.len() != dim_order.len() {
                    return false;
                }
                true
            }
            Layout::Packed { .. } => todo!(),
        }
    }

    // TODO: Inline
    fn super_applies(&self, shape: &[DimSize]) -> bool {
        if shape.iter().all(|&d| d == 1) && !self.is_row_major() {
            return false;
        }
        true
    }

    pub fn is_row_major(&self) -> bool {
        match &self {
            Layout::Standard { dim_order } => {
                // Check if each of dim_order if is equal to its index.
                dim_order
                    .iter()
                    .enumerate()
                    .all(|(i, &d)| i == usize::from(d))
            }
            _ => false,
        }
    }

    // TODO: Do we really need callers to build a HashSet?
    pub fn dim_drop(&self, dropped_dims: &HashSet<u8>, contiguous_abs: Contig) -> (Layout, Contig) {
        if dropped_dims.is_empty() {
            return (self.clone(), contiguous_abs);
        }
        match self {
            Self::Standard { dim_order } => {
                let mut new_dim_order = vec![];
                for &logical_dim in dim_order {
                    if !dropped_dims.contains(&logical_dim) {
                        let offset: u8 = dropped_dims
                            .iter()
                            .filter(|&&d| d < logical_dim)
                            .count()
                            .try_into()
                            .unwrap();
                        new_dim_order.push(logical_dim - offset);
                    }
                }

                let mut new_contiguous = contiguous_abs;
                if contiguous_abs != 0 {
                    for &logical_dim_inside_contig in
                        &dim_order[dim_order.len() - usize::from(contiguous_abs)..]
                    {
                        if dropped_dims.contains(&logical_dim_inside_contig) {
                            new_contiguous -= 1;
                        }
                    }
                }

                (
                    Self::Standard {
                        dim_order: SmallVec::from(new_dim_order),
                    },
                    new_contiguous,
                )
            }
            Self::Packed {
                dim_count,
                strip_dim,
                strip_size,
            } => {
                if dropped_dims.contains(strip_dim) {
                    let rm_contig = contiguous_abs.saturating_sub(1);
                    return row_major(*dim_count).dim_drop(dropped_dims, rm_contig);
                }

                let after_strip_dims = (strip_dim + 1..*dim_count).collect::<HashSet<_>>();
                assert!(!after_strip_dims.is_empty(), "There must be dimensions after the strip dim., otherwise this is really a StandardLayout");

                if dropped_dims.is_superset(&after_strip_dims) {
                    return row_major(strip_dim + 1).dim_drop(
                        &dropped_dims
                            .difference(&after_strip_dims)
                            .cloned()
                            .collect(),
                        (contiguous_abs - u8::try_from(after_strip_dims.len()).unwrap() - 1).max(0),
                    );
                }

                let fifth_dim_contig = contiguous_abs.min(1);
                let standard_contig = contiguous_abs.saturating_sub(1);
                let contig_dropped: Contig = dropped_dims
                    .iter()
                    .filter(|&&d| (*dim_count - d) <= standard_contig)
                    .count()
                    .try_into()
                    .unwrap();

                (
                    Self::Packed {
                        dim_count: dim_count - u8::try_from(dropped_dims.len()).unwrap(),
                        strip_dim: *strip_dim,
                        strip_size: *strip_size,
                    },
                    fifth_dim_contig + standard_contig - contig_dropped,
                )
            }
        }
    }

    pub fn expand_shape(&self, shape: &[DimSize]) -> Shape {
        match self {
            Layout::Packed {
                dim_count: _,
                strip_dim,
                strip_size,
            } => {
                let mut new_shape = Shape::from_slice(shape);
                if let Ok(strip_dim_idx) = usize::try_from(*strip_dim) {
                    if let Some(strip_dim_val) = new_shape.get_mut(strip_dim_idx) {
                        *strip_dim_val /= *strip_size;
                    } else {
                        panic!("strip_dim index is out of bounds");
                    }
                    new_shape.push(*strip_size);
                    assert!(
                        new_shape.iter().all(|&d| d > 0),
                        "All dimensions must be greater than 0"
                    );
                    new_shape
                } else {
                    panic!("Unable to convert strip_dim to usize")
                }
            }
            _ => panic!("expand_shape method is only applicable to Packed layout variant"),
        }
    }

    pub fn transpose(&self, swap_dims: (u8, u8), contiguous_abs: Contig) -> (Layout, Contig) {
        match self {
            Layout::Standard { dim_order } => {
                let new_dim_order = dim_order
                    .iter()
                    .copied()
                    .map(|orig_dim| {
                        if orig_dim == swap_dims.0 {
                            swap_dims.1
                        } else if orig_dim == swap_dims.1 {
                            swap_dims.0
                        } else {
                            orig_dim
                        }
                    })
                    .collect();
                (
                    Layout::Standard {
                        dim_order: new_dim_order,
                    },
                    contiguous_abs,
                )
            }
            Layout::Packed { .. } => todo!(),
        }
    }

    fn flatten_inner_contiguous_dimensions(
        &self,
        _shape: Shape,
        _contiguous_abs: Contig,
    ) -> Option<(Vec<usize>, HashSet<usize>, usize)> {
        // TODO: Do we really want to return an HashSet? Those are expensive!
        todo!()
    }

    // Reorder the shape according to the physical order of the dimensions.
    fn layout_ordered_dims(&self, dim_sizes: &Shape) -> Shape {
        match &self {
            Layout::Standard { dim_order } => {
                assert_eq!(dim_sizes.len(), dim_order.len());
                dim_order
                    .iter()
                    .map(|&d| dim_sizes[usize::from(d)])
                    .collect()
            }
            _ => unimplemented!(),
        }
    }

    pub fn canonicalize_for_shape(&self, shape: &[DimSize]) -> Layout {
        if shape.iter().all(|d| *d == 1) {
            row_major(shape.len().try_into().unwrap())
        } else {
            self.clone()
        }
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Layout::Standard { dim_order } => {
                if self.is_row_major() {
                    write!(f, "RM")
                } else if dim_order.to_vec() == vec![0, 2, 3, 1] {
                    write!(f, "NHWC")
                } else {
                    write!(
                        f,
                        "<{}>",
                        dim_order
                            .iter()
                            .map(|d| d.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                }
            }
            Layout::Packed {
                dim_count: _,
                strip_dim: _,
                strip_size: _,
            } => todo!(),
        }
    }
}

fn regular_index_expr(
    expr_id: &OpaqueSymbol,
    logical_dims: &[u8],
    shape: &[DimSize],
) -> AffineExpr<BufferExprTerm> {
    assert!(!logical_dims.is_empty());
    let t = *logical_dims.last().unwrap();
    let remaining_dims = &logical_dims[..logical_dims.len() - 1];
    let s = i32::try_from(shape[usize::from(t)]).unwrap();
    let p = if s > 1 {
        AffineExpr(vec![Term(1, BufferExprTerm::Pt(t, expr_id.clone()))], 0)
    } else {
        AffineExpr(vec![], 0) // zero
    };
    if remaining_dims.is_empty() {
        return p;
    }
    regular_index_expr(expr_id, remaining_dims, shape) * s + p
}

pub fn row_major(rank: u8) -> Layout {
    Layout::Standard {
        dim_order: (0..rank).collect(),
    }
}

pub fn nhwc() -> Layout {
    Layout::Standard {
        dim_order: vec![0, 2, 3, 1].into(),
    }
}
