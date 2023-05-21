use crate::alignment::aligned_approx;
use crate::common::{DimSize, Shape};
use crate::spec::Spec;
use divrem::DivCeil;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::fmt::Debug;

use crate::target::Target;
use crate::tensorspec::TensorSpec;

// TODO: Rename to something like `Tiling`, as opposed to a tile.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub enum PartialTile {
    Simple(Shape),
    ConvImage(Shape, Shape),
}

#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub struct Tile {
    pub partial: PartialTile,
    pub source_idx: u8,
    pub aligned: bool,
}

impl PartialTile {
    // TODO: Rename to `apply_to_operand`.
    pub fn tile<Tgt: Target>(&self, source_idx: u8, source_spec: &TensorSpec<Tgt>) -> Tile {
        match self {
            PartialTile::Simple(dim_sizes) => simple_tile(source_spec, source_idx, dim_sizes),
            PartialTile::ConvImage(dim_sizes, filter_shape) => {
                conv_image_tile(source_spec, source_idx, dim_sizes, filter_shape)
            }
        }
    }

    pub fn dim_sizes(&self) -> &Shape {
        match &self {
            PartialTile::Simple(shp) | PartialTile::ConvImage(shp, _) => shp,
        }
    }

    pub fn steps_dim(&self, dim: u8, origin_size: DimSize) -> u32 {
        match &self {
            PartialTile::Simple(shape) => {
                divrem::DivCeil::div_ceil(origin_size, shape[usize::from(dim)])
            }
            PartialTile::ConvImage(img_shape, filters_shape) => {
                // Batch should be a normal tiling.
                if dim == 0 {
                    DivCeil::div_ceil(origin_size, self.dim_sizes()[usize::from(dim)])
                } else {
                    let inner = img_shape[usize::from(dim)];
                    let f = filters_shape[usize::from(dim - 1)];
                    DivCeil::div_ceil(_s(origin_size, f), _s(inner, f))
                }
            }
        }
    }

    pub fn boundary_size(&self, dim: u8, origin_size: DimSize) -> u32 {
        match &self {
            PartialTile::Simple(shape) => origin_size % shape[usize::from(dim)],
            PartialTile::ConvImage(dim_sizes, filter_shape) => {
                // Non-spatial dimensions (batch) should be simple tilings.
                if dim == 0 {
                    origin_size % dim_sizes[usize::from(dim)]
                } else {
                    let filt = filter_shape[usize::from(dim) - 1];
                    let total_filter_applications = 1 + origin_size - filt;
                    let tile_filter_applications = 1 + dim_sizes[usize::from(dim)] - filt;
                    let boundary_applications =
                        total_filter_applications % tile_filter_applications;
                    if boundary_applications == 0 {
                        0
                    } else {
                        boundary_applications + filt - 1
                    }
                }
            }
        }
    }
}

// TODO: Inline the following.
fn _s(img_size: DimSize, filter_size: DimSize) -> DimSize {
    1 + img_size - filter_size
}

/// Constructs a simple tile with alignment set correctly.
pub fn simple_tile<Tgt: Target>(
    source_spec: &TensorSpec<Tgt>,
    operand_idx: u8,
    new_dims: &[DimSize],
) -> Tile {
    make_tile_with_alignment(source_spec, operand_idx, new_dims, PartialTile::Simple)
}

fn conv_image_tile<Tgt: Target>(
    source_spec: &TensorSpec<Tgt>,
    operand_idx: u8,
    new_dims: &[DimSize],
    filter_shape: &[DimSize],
) -> Tile {
    make_tile_with_alignment(source_spec, operand_idx, new_dims, move |s| {
        PartialTile::ConvImage(s, filter_shape.into())
    })
}

fn make_tile_with_alignment<Tgt: Target>(
    source_spec: &TensorSpec<Tgt>,
    operand_idx: u8,
    new_dims: &[DimSize],
    tile_constructor: impl FnOnce(Shape) -> PartialTile,
) -> Tile {
    assert_eq!(
        new_dims.len(),
        source_spec.dim_sizes().len(),
        "Cannot produce rank-{} tile of shape {:?} for rank-{} tensor of shape {:?}",
        new_dims.len(),
        new_dims,
        source_spec.dim_sizes().len(),
        source_spec.dim_sizes(),
    );

    assert!(
        new_dims
            .iter()
            .zip(source_spec.dim_sizes().iter())
            .all(|(td, rd)| td <= rd),
        "Tile {:?} would be larger than tensor {:?}",
        new_dims,
        source_spec.dim_sizes(),
    );

    let partial = tile_constructor(Shape::from(new_dims));
    let aligned = aligned_approx(&partial, new_dims, source_spec);
    Tile {
        partial,
        source_idx: operand_idx,
        aligned,
    }
}

/// Map a Spec's type, input shapes, and output tile to PartialTiles for its inputs.
///
/// Note that input_shapes refers to original, untiled input shapes, while the
/// spec_output describes the final, already-tiled output.
///
/// Compose is not directly represented because tiling a Compose depends on its
/// sub-Specs, which are members of the Compose object. As a result, the tile_out logic
/// can't be fully defined by the *type* Compose; only by a specific Compose instance.
pub fn tile_out<Tgt: Target>(
    // TODO: Rename function to `input_tile_deps` or something.
    spec: &Spec<Tgt>,
    input_shapes: &[Vec<DimSize>],
    spec_output: &PartialTile,
) -> Vec<PartialTile> {
    match (spec, spec_output) {
        (Spec::Matmul { .. }, PartialTile::Simple(dim_sizes)) => {
            let m = dim_sizes[0];
            let k = input_shapes[0][1];
            let n = dim_sizes[1];
            vec![
                PartialTile::Simple(smallvec![m, k]),
                PartialTile::Simple(smallvec![k, n]),
            ]
        }
        (Spec::Load { .. }, PartialTile::Simple(dim_sizes))
        | (Spec::Store { .. }, PartialTile::Simple(dim_sizes)) => {
            vec![PartialTile::Simple(dim_sizes.clone())]
        }
        (Spec::Zero { .. }, _) => vec![],
        _ => unimplemented!(),
    }
}
