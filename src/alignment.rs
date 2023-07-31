use crate::common::{DimSize, Dtype};
use crate::layout::Layout;
use crate::target::Target;
use crate::tensorspec::TensorSpec;

use log::warn;

pub fn aligned_approx<Tgt: Target>(
    tile_shape: &[DimSize],
    tile_step_sizes: &[DimSize],
    parent: &TensorSpec<Tgt>,
) -> bool {
    debug_assert_eq!(tile_shape.len(), tile_step_sizes.len());
    if !parent.is_contiguous() || !parent.aligned() {
        return false;
    }

    let is_simple = tile_shape
        .iter()
        .zip(tile_step_sizes.iter())
        .all(|(s, t)| *s == *t);

    match (parent.layout(), is_simple) {
        (Layout::Standard { dim_order }, true) => aligned_approx_standard_simple::<Tgt>(
            tile_shape,
            dim_order.as_slice(),
            parent.dim_sizes(),
            &parent.dtype(),
        ),
        (Layout::Packed { .. }, true) => {
            let tile_expanded = parent.layout().expand_shape(tile_shape);
            aligned_approx_standard_simple::<Tgt>(
                &tile_expanded,
                (0u8..tile_expanded.len().try_into().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
                &parent.layout().expand_shape(parent.dim_sizes()),
                &parent.dtype(),
            )
        }
        (_, false) => {
            if tile_shape[1..] == parent.dim_sizes()[1..] {
                todo!()
            } else {
                warn!("No alignment analysis for non-batch convolution");
                false
            }
        }
    }
}

fn aligned_approx_standard_simple<Tgt: Target>(
    tile_shape: &[DimSize],
    dim_order: &[u8],
    parent_shape: &[DimSize],
    parent_dtype: &Dtype,
) -> bool {
    let mut cum_inner_volume = 1;
    for &physical_dim_idx in dim_order.iter().rev() {
        let pd_idx_usize = usize::from(physical_dim_idx);
        let step_values = cum_inner_volume * tile_shape[pd_idx_usize];
        cum_inner_volume *= parent_shape[pd_idx_usize];
        if parent_shape[pd_idx_usize] == tile_shape[pd_idx_usize] {
            continue;
        }
        if (step_values * u32::from(parent_dtype.size())) % Tgt::line_size() != 0 {
            return false;
        }
    }
    true
}
