use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{collections::HashSet, fmt::Display};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
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

impl Layout {
    fn buffer_indexing_expr(&self, _concrete_shape: Shape) -> f64 {
        todo!()
    }

    pub fn contiguous_full(&self) -> Contig {
        match &self {
            Layout::Standard { dim_order } => dim_order.len().try_into().unwrap(),
            Layout::Packed { dim_count, .. } => dim_count + 1,
        }
    }

    fn contiguous_lub(&self, _other_layout: &Layout, _a: Contig, _b: Contig) -> Contig {
        todo!()
    }

    pub fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> {
        match &self {
            Layout::Standard { dim_order } => 0u8..(dim_order.len() + 1).try_into().unwrap(),
            Layout::Packed { dim_count, .. } => 0u8..(*dim_count + 2),
        }
    }

    fn tile_is_contiguous(&self, _contiguous_abs: Contig) -> bool {
        todo!()
    }

    // TODO: Rename; this actually returns a contiguousness abstraction.
    fn check_tile_contiguity(
        &self,
        _tile_shape: Shape,
        _parent_shape: Shape,
        _parent_contiguous: Contig,
    ) -> Contig {
        todo!()
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
    fn dim_drop(&self, _dropped_dims: HashSet<usize>, _contiguous_abs: usize) -> (Layout, Contig) {
        todo!()
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

    fn transpose(&self, _swap_dims: (usize, usize), _contiguous_abs: usize) -> (Layout, usize) {
        todo!()
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

pub fn row_major(rank: u8) -> Layout {
    Layout::Standard {
        dim_order: (0..rank).collect(),
    }
}
