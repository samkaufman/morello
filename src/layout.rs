use smallvec::SmallVec;
use std::{collections::HashSet, fmt::Display};

use crate::{
    common::{Contig, DimSize, Dtype, Shape},
    target::Target,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
    fn buffer_indexing_expr(&self, concrete_shape: Shape) -> f64 {
        todo!()
    }

    pub fn contiguous_full(&self) -> Contig {
        match &self {
            Layout::Standard { dim_order } => dim_order.len().try_into().unwrap(),
            Layout::Packed { dim_count, .. } => dim_count + 1,
        }
    }

    fn contiguous_lub(&self, other_layout: &Layout, a: Contig, b: Contig) -> Contig {
        todo!()
    }

    fn all_contiguous_abs(&self) -> impl Iterator<Item = Contig> {
        match &self {
            Layout::Standard { dim_order } => (0u8..(dim_order.len() + 1).try_into().unwrap()),
            Layout::Packed { dim_count, .. } => (0u8..(*dim_count + 2)),
        }
    }

    fn tile_is_contiguous(&self, contiguous_abs: Contig) -> bool {
        todo!()
    }

    // TODO: Rename; this actually returns a contiguousness abstraction.
    fn check_tile_contiguity(
        &self,
        tile_shape: Shape,
        parent_shape: Shape,
        parent_contiguous: Contig,
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
            Layout::Standard { dim_order } => {
                let line_size = Tgt::LINE_SIZE;

                if contiguous {
                    divrem::DivCeil::div_ceil(
                        shape.into_iter().product::<DimSize>() * DimSize::from(dtype.size()),
                        line_size,
                    )
                } else {
                    let lodims = self.layout_ordered_dims(&shape);
                    let mut real_dims: Vec<u32> = lodims.into_iter().filter(|&d| d > 1).collect();
                    if real_dims.is_empty() {
                        real_dims.push(1);
                    }
                    real_dims[..real_dims.len() - 1].iter().product::<DimSize>()
                        * divrem::DivCeil::div_ceil(
                            real_dims[real_dims.len()] * DimSize::from(dtype.size()),
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
                dim_order.iter().enumerate().all(|(i, &d)| i == d.into())
            }
            _ => false,
        }
    }

    // TODO: Do we really need callers to build a HashSet?
    fn dim_drop(&self, dropped_dims: HashSet<usize>, contiguous_abs: usize) -> (Layout, Contig) {
        todo!()
    }

    pub fn expand_shape(&self, shape: &[DimSize]) -> Shape {
        match self {
            Layout::Packed {
                dim_count,
                strip_dim,
                strip_size,
            } => {
                let mut new_shape = Shape::from_slice(shape);
                if let Some(strip_dim_idx) = usize::try_from(*strip_dim).ok() {
                    if let Some(strip_dim_val) = new_shape.get_mut(strip_dim_idx) {
                        *strip_dim_val = *strip_dim_val / *strip_size;
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

    fn transpose(&self, swap_dims: (usize, usize), contiguous_abs: usize) -> (Layout, usize) {
        todo!()
    }

    fn flatten_inner_contiguous_dimensions(
        &self,
        shape: Shape,
        contiguous_abs: Contig,
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
            } => todo!(),
        }
    }
}

pub fn row_major(rank: u8) -> Layout {
    Layout::Standard {
        dim_order: (0..rank).collect(),
    }
}
