use divrem::DivCeil;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::fmt::Debug;

use crate::alignment::aligned_approx;
use crate::common::{DimSize, Shape};
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;

// TODO: Rename to something like `Tiling`, as opposed to a tile.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub enum Tiling {
    Simple(Shape),
    ConvImage(Shape, Shape),
}

#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub struct Tile {
    pub tiling: Tiling,
    pub source_idx: u8,
    pub aligned: bool,
}

impl Tiling {
    // TODO: Rename to `apply_to_operand`.
    pub fn tile<Tgt: Target>(&self, source_idx: u8, source_spec: &TensorSpec<Tgt>) -> Tile {
        match self {
            Tiling::Simple(dim_sizes) => {
                make_tile_with_alignment(source_spec, source_idx, dim_sizes, Tiling::Simple)
            }
            Tiling::ConvImage(dim_sizes, filter_shape) => {
                make_tile_with_alignment(source_spec, source_idx, dim_sizes, move |s| {
                    Tiling::ConvImage(s, filter_shape.to_owned())
                })
            }
        }
    }

    pub fn dim_sizes(&self) -> &Shape {
        match &self {
            Tiling::Simple(shp) | Tiling::ConvImage(shp, _) => shp,
        }
    }

    pub fn steps_dim(&self, dim: u8, origin_size: DimSize) -> u32 {
        match &self {
            Tiling::Simple(shape) => {
                divrem::DivCeil::div_ceil(origin_size, shape[usize::from(dim)])
            }
            Tiling::ConvImage(img_shape, filters_shape) => {
                // Batch should be a normal tiling.
                if dim == 0 {
                    DivCeil::div_ceil(origin_size, self.dim_sizes()[usize::from(dim)])
                } else {
                    let inner = img_shape[usize::from(dim)];
                    let f = filters_shape[usize::from(dim - 1)];
                    DivCeil::div_ceil(1 + origin_size - f, 1 + inner - f)
                }
            }
        }
    }

    pub fn boundary_size(&self, dim: u8, origin_size: DimSize) -> u32 {
        match &self {
            Tiling::Simple(shape) => origin_size % shape[usize::from(dim)],
            Tiling::ConvImage(dim_sizes, filter_shape) => {
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

fn make_tile_with_alignment<Tgt: Target>(
    source_spec: &TensorSpec<Tgt>,
    operand_idx: u8,
    new_dims: &[DimSize],
    tile_constructor: impl FnOnce(Shape) -> Tiling,
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

    let tiling = tile_constructor(Shape::from(new_dims));
    let aligned = aligned_approx(&tiling, new_dims, source_spec);
    Tile {
        tiling,
        source_idx: operand_idx,
        aligned,
    }
}

/// Infer data dependency-respecting Tilings for a Spec's inputs from a tiling.
///
/// This additionally requires the initial input shapes, which are used to
/// determine dimensions unaffected by the tiling (tiled to the same size).
///
/// Compose is not directly represented because tiling a Compose depends on its
/// sub-Specs, which are members of the Compose object. As a result, the tile_out logic
/// can't be fully defined by the *type* Compose; only by a specific Compose instance.
pub fn tile_out<Tgt: Target>(
    // TODO: Rename function to `input_tile_deps` or something.
    spec: &Spec<Tgt>,
    input_shapes: &[Vec<DimSize>],
    spec_output: &Tiling,
) -> Vec<Tiling> {
    match (spec, spec_output) {
        (Spec::Matmul { .. }, Tiling::Simple(dim_sizes)) => {
            let m = dim_sizes[0];
            let k = input_shapes[0][1];
            let n = dim_sizes[1];
            vec![
                Tiling::Simple(smallvec![m, k]),
                Tiling::Simple(smallvec![k, n]),
            ]
        }
        (Spec::Load { .. }, Tiling::Simple(dim_sizes))
        | (Spec::Store { .. }, Tiling::Simple(dim_sizes)) => {
            vec![Tiling::Simple(dim_sizes.clone())]
        }
        (Spec::Zero { .. }, _) => vec![],
        _ => unimplemented!(),
    }
}
