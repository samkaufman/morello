use std::collections::HashSet;

use crate::{layout::row_major, target::Target, tensorspec::TensorSpec};

pub fn squeeze_tile<Tgt: Target>(
    inner_spec: &TensorSpec<Tgt>,
    dropped_dims: &[u8],
) -> TensorSpec<Tgt> {
    // Make sure dropped_dims is sorted.
    debug_assert!(dropped_dims.windows(2).all(|w| w[0] < w[1]));

    let mut new_dim_sizes = inner_spec.dim_sizes().clone();
    for &dim in dropped_dims.iter().rev() {
        assert_eq!(
            new_dim_sizes[usize::from(dim)],
            1,
            "Cannot drop non-degenerate dimension {} of shape {:?}",
            dim,
            new_dim_sizes
        );
        new_dim_sizes.remove(dim.into());
    }

    let mut new_vector_shape = None;
    if let Some(vector_shape) = inner_spec.vector_shape() {
        new_vector_shape = Some(vector_shape.to_vec());
        for &dim in dropped_dims.iter().rev() {
            assert_eq!(
                vector_shape[usize::from(dim)],
                1,
                "Cannot drop dimension {} of vector shape {:?}",
                dim,
                vector_shape
            );
            new_vector_shape.as_mut().unwrap().remove(dim.into());
        }
    }

    let (new_layout, new_contig) = if new_dim_sizes.iter().all(|&d| d == 1) {
        let new_layout = row_major(new_dim_sizes.len().try_into().unwrap());
        (new_layout.clone(), new_layout.contiguous_full())
    } else {
        let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
        inner_spec
            .layout()
            .dim_drop(&dropped_dims_set, inner_spec.contiguous_abs())
    };

    TensorSpec::new_canon(
        new_dim_sizes,
        inner_spec.dtype(),
        new_contig,
        inner_spec.aligned(),
        inner_spec.level(),
        new_layout,
        new_vector_shape.map(|v| v.into()),
    )
}
