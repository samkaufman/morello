use divrem::DivCeil;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{cmp::min, collections::HashSet, fmt::Display, hash::Hash};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    expr::{AffineForm, Atom, Bounds, NonAffine, NonAffineExpr, Substitute, Term},
    opaque_symbol::OpaqueSymbol,
    target::Target,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deserialize, Serialize)]
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
pub enum BufferVar {
    TileIdx(u8, OpaqueSymbol),
    // TODO: Probably don't need the OpaqueSymbol for Pt
    Pt(u8, OpaqueSymbol),
}

impl Layout {
    pub fn new_standard(dim_order: SmallVec<[u8; 5]>) -> Layout {
        Layout::Standard { dim_order }
    }

    pub fn new_packed(dim_count: u8, strip_dim: u8, strip_size: DimSize) -> Layout {
        assert!(strip_dim < dim_count - 1);
        Layout::Packed {
            dim_count,
            strip_dim,
            strip_size,
        }
    }

    pub fn buffer_indexing_expr(
        &self,
        expr_id: &OpaqueSymbol,
        concrete_shape: &[DimSize],
    ) -> NonAffineExpr<BufferVar> {
        match self {
            Layout::Standard { dim_order } => {
                debug_assert_eq!(dim_order.len(), concrete_shape.len());
                regular_index_expr(expr_id, dim_order, concrete_shape)
            }
            Layout::Packed {
                dim_count,
                strip_dim,
                strip_size,
            } => {
                assert_eq!(
                    concrete_shape.len(),
                    usize::from(*dim_count),
                    "Expected rank-{} shape",
                    concrete_shape.len()
                );

                let expanded = self.expand_shape(concrete_shape);
                debug_assert_eq!(expanded.len(), usize::from(*dim_count) + 1);
                let idx_expr = row_major(dim_count + 1).buffer_indexing_expr(expr_id, &expanded);
                idx_expr.map_vars(&mut |v| match v {
                    BufferVar::Pt(d, _) if d == *strip_dim => {
                        AffineForm::from(NonAffine::FloorDiv(Box::new(v.into()), *strip_size))
                    }
                    BufferVar::Pt(d, _) if d == *dim_count => {
                        let packing_p = BufferVar::Pt(*strip_dim, expr_id.clone());
                        AffineForm::from(NonAffine::Mod(Box::new(packing_p.into()), *strip_size))
                    }
                    _ => AffineForm::from(v),
                })
            }
        }
    }

    pub fn contiguous_full(&self) -> Contig {
        match self {
            Layout::Standard { dim_order } => dim_order.len().try_into().unwrap(),
            Layout::Packed { dim_count, .. } => dim_count + 1,
        }
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> + Clone {
        match self {
            Layout::Standard { dim_order } => 0u8..(dim_order.len() + 1).try_into().unwrap(),
            Layout::Packed { dim_count, .. } => 0u8..(*dim_count + 2),
        }
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
                dim_count,
                strip_dim,
                strip_size,
            } => {
                assert_eq!(
                    parent_shape.len(),
                    usize::from(*dim_count),
                    "Expected rank-{dim_count} outer shape"
                );
                assert_eq!(
                    tile_shape.len(),
                    usize::from(*dim_count),
                    "Expected rank-{dim_count} tile"
                );

                // If we're tiling to 1x…x1, then we'll return full contig. for row-major, not
                // packed, since the caller is going to change the layout to row-major.
                // TODO: This really tightly couples contig. and layout logic. Improve the
                //    abstraction, perhaps by making contig. a property of Layout.
                if tile_shape.iter().all(|&d| d == 1) {
                    return row_major(*dim_count).contiguous_full();
                }

                // Contig. is zero if non-multiple breaking innermost/strip dimension. This avoids
                // calling expand_shape for, among other things, tiling to 1x...x1. More
                // importantly,
                if tile_shape[usize::from(*strip_dim)] % *strip_size != 0 {
                    return 0;
                }

                let expanded_parent_shape = self.expand_shape(parent_shape);
                let expanded_tile_shape = self.expand_shape(tile_shape);
                row_major(expanded_parent_shape.len().try_into().unwrap()).tile_contiguity(
                    &expanded_tile_shape,
                    &expanded_parent_shape,
                    parent_contiguous,
                )
            }
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
        let Layout::Standard { dim_order } = self else {
            unreachable!();
        };

        let tile_rank = u8::try_from(tile_shape.len()).unwrap();
        while *cnt < tile_rank {
            let mut rev_dim_idx = *cnt;
            if one_back {
                rev_dim_idx += 1;
            }

            let phys_idx = usize::from(dim_order[dim_order.len() - usize::from(rev_dim_idx)]);
            if tile_shape[phys_idx] != comp(phys_idx.try_into().unwrap()) {
                break;
            }
            *cnt += 1;
        }
    }

    pub fn estimate_cache_lines<Tgt: Target>(
        &self,
        shape: &[DimSize],
        dtype: Dtype,
        contiguous: bool,
    ) -> u32 {
        match &self {
            Layout::Standard { dim_order: _ } => {
                let line_size = Tgt::line_size();

                if contiguous {
                    divrem::DivCeil::div_ceil(
                        shape.iter().product::<DimSize>() * DimSize::from(dtype.size()),
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
            Layout::Packed { .. } => {
                // TODO: Use contiguous from the caller rather than assuming no contiguousness?
                let rm_like_shape = self.expand_shape(shape);
                let expanded = row_major(rm_like_shape.len().try_into().unwrap());
                expanded.estimate_cache_lines::<Tgt>(&rm_like_shape, dtype, false)
            }
        }
    }

    pub fn applies_to_shape(&self, shape: &[DimSize]) -> bool {
        if shape.iter().all(|&d| d == 1) && !self.is_row_major() {
            return false;
        }
        match &self {
            Layout::Standard { dim_order } => shape.len() == dim_order.len(),
            Layout::Packed { dim_count, .. } => shape.len() == usize::from(*dim_count),
        }
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
                    Self::new_standard(SmallVec::from(new_dim_order)),
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
                        contiguous_abs
                            .saturating_sub(u8::try_from(after_strip_dims.len()).unwrap() + 1),
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
                    Self::new_packed(
                        dim_count - u8::try_from(dropped_dims.len()).unwrap(),
                        *strip_dim,
                        *strip_size,
                    ),
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
                        *strip_dim_val = DivCeil::div_ceil(*strip_dim_val, *strip_size);
                    } else {
                        panic!("strip_dim index is out of bounds");
                    }
                    new_shape.push(*strip_size);
                    assert!(
                        new_shape.iter().all(|&d| d > 0),
                        "Expanded shape has a zero size dimension: {new_shape:?}"
                    );
                    new_shape
                } else {
                    panic!("Unable to convert strip_dim to usize")
                }
            }
            _ => panic!("expand_shape method is only applicable to Packed layouts"),
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
            Layout::Packed {
                dim_count,
                strip_dim,
                strip_size,
            } => {
                let new_packed = if *strip_dim == swap_dims.0 {
                    Layout::Packed {
                        dim_count: *dim_count,
                        strip_dim: swap_dims.1,
                        strip_size: *strip_size,
                    }
                } else if *strip_dim == swap_dims.1 {
                    Layout::Packed {
                        dim_count: *dim_count,
                        strip_dim: swap_dims.0,
                        strip_size: *strip_size,
                    }
                } else {
                    todo!("not expressible in general with Packed layouts")
                };
                (new_packed, contiguous_abs)
            }
        }
    }

    // Reorder the shape according to the physical order of the dimensions.
    fn layout_ordered_dims(&self, shape: &[DimSize]) -> Shape {
        match &self {
            Layout::Standard { dim_order } => {
                assert_eq!(shape.len(), dim_order.len());
                dim_order.iter().map(|&d| shape[usize::from(d)]).collect()
            }
            _ => unimplemented!(),
        }
    }

    #[must_use]
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
                dim_count,
                strip_dim,
                strip_size,
            } => write!(f, "pack({}, {}, {})", dim_count, strip_dim, strip_size),
        }
    }
}

impl Display for BufferVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferVar::TileIdx(dim, _) => write!(f, "t{}", dim),
            BufferVar::Pt(dim, _) => write!(f, "p{}", dim),
        }
    }
}

impl Atom for BufferVar {}
impl Bounds for BufferVar {}

fn regular_index_expr(
    expr_id: &OpaqueSymbol,
    logical_dims: &[u8],
    shape: &[DimSize],
) -> NonAffineExpr<BufferVar> {
    assert!(!logical_dims.is_empty());
    let t = *logical_dims.last().unwrap();
    let remaining_dims = &logical_dims[..logical_dims.len() - 1];
    let s = i32::try_from(shape[usize::from(t)]).unwrap();
    let p = if s > 1 {
        AffineForm(vec![Term(1, BufferVar::Pt(t, expr_id.clone()).into())], 0)
    } else {
        AffineForm(vec![], 0) // zero
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

pub fn col_major(rank: u8) -> Layout {
    Layout::Standard {
        dim_order: (0..rank).rev().collect(),
    }
}

pub fn nhwc() -> Layout {
    Layout::Standard {
        dim_order: vec![0, 2, 3, 1].into(),
    }
}

#[cfg(test)]
mod tests {
    use super::col_major;

    #[test]
    fn test_col_major_slices_are_non_contiguous() {
        let col_major = col_major(2);
        let inner_contig =
            col_major.tile_contiguity(&[1, 8], &[128, 128], col_major.contiguous_full());
        assert_eq!(inner_contig, 1);
    }
}
