use crate::common::DimSize;
use crate::layout::LayoutError;
use crate::target::Target;
use crate::tensorspec::TensorSpec;

use log::warn;

/// Checks if each tile of a tiling's earliest value is a multiple of the target's alignment size.
///
/// This current implementation of this function will say a tile aligned if the source tensor is
/// itself aligned and contiguous and the tile stride is a multiple of the target's alignment size.
/// The stride is, of course, determined by the tensor's layout. At the moment, this function always
/// returns `false` for sliding layouts (layouts where the step size in any dimension is not equal
/// to its tile size).
pub fn aligned_approx<Tgt: Target>(
    tile_shape: &[DimSize],
    tile_step_sizes: &[DimSize],
    parent: &TensorSpec<Tgt>,
) -> Result<bool, LayoutError> {
    debug_assert_eq!(tile_shape.len(), tile_step_sizes.len());

    // If the tile source is unaligned or not contiguous, we assume the tile is also un-aligned.
    if !parent.is_contiguous() || !parent.aligned() {
        return Ok(false);
    }

    // Check if the tile evenly divides each tensor dimension (i.e., that tile step sizes are the
    // same as the tile sizes).  If not, we assume (for now) the tile is unaligned.
    let is_simple = tile_shape
        .iter()
        .zip(tile_step_sizes)
        .all(|(s, t)| *s == *t);
    if !is_simple {
        warn!("No alignment analysis for sliding layouts");
        return Ok(false);
    }

    let parent_physical_shape = parent.layout().expand_physical_shape(parent.shape())?;
    let mut stride_bytes = u32::from(parent.dtype().size());
    for phys_dim_size in parent_physical_shape.iter().rev() {
        stride_bytes *= phys_dim_size;
        if stride_bytes % Tgt::line_size() != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use crate::{
        alignment::aligned_approx,
        common::Dtype,
        expr::{AffineForm, Bounds, NonAffine, NonAffineExpr, Substitute},
        layout::{row_major, BufferVar},
        opaque_symbol::OpaqueSymbol,
        target::{CpuMemoryLevel, Target, X86Target},
        tensorspec::TensorSpec,
        views::{Param, Tile, View},
    };
    use itertools::Itertools;
    use smallvec::smallvec;

    #[test]
    fn test_aligned_approx_x86() {
        // TODO: Make sure to test different dtypes.

        let parent = TensorSpec::<X86Target>::new_canon(
            smallvec![16, 16, 16],
            Dtype::Uint8,
            row_major(3).contiguous_full(),
            true,
            CpuMemoryLevel::GL,
            row_major(3),
            None,
        );
        let layout = parent.layout();

        // TODO: Allocate a buffer which is aligned, or not, according to `parent.aligned()`.
        // In the following, we'll index `buffer` with byte offsets, not element offsets.
        // TODO: We don't really need to allocate the buffer, do we? We just need the indices,
        //   shifted if parent is unaligned.
        let _buffer = {
            let volume = usize::try_from(parent.volume()).unwrap();
            if parent.aligned() {
                vec![0; volume]
            } else {
                debug_assert!(X86Target::line_size() % u32::from(parent.dtype().size()) > 1);
                vec![1; volume]
            }
        };

        let iexpr = layout.buffer_indexing_expr(&OpaqueSymbol::new(), parent.shape());
        println!("initial indexing expr = {}", iexpr);

        let parent_as_param = Param::new(0, parent);
        let parent = parent_as_param.spec();
        let tile = Tile::new(smallvec![4, 4, 4], smallvec![4, 4, 4], &parent_as_param).unwrap();
        let iexpr = tile.compose_buffer_indexing_expr(
            parent
                .layout()
                .buffer_indexing_expr(&OpaqueSymbol::new(), parent.shape()),
        );
        println!("tiled indexing expr = {}", iexpr);

        // TODO: The buffer indexing exporession should be tiled.

        let is_aligned = {
            let tile_coordinates = (0..tile.shape().len())
                .map(|dim| 0..tile.steps_dim(dim.try_into().unwrap()))
                .multi_cartesian_product();
            let mut per_tile_index_expressions =
                tile_coordinates.map(|tile_coordinate| -> AffineForm<NonAffine<BufferVar>> {
                    println!("{:?}", tile_coordinate);
                    iexpr.clone().map_vars(&mut |v| match v {
                        BufferVar::TileIdx(d, _) => NonAffineExpr::constant(
                            tile_coordinate[usize::from(d)].try_into().unwrap(),
                        ),
                        BufferVar::Pt(_, _) => NonAffineExpr::constant(0),
                    })
                });
            per_tile_index_expressions.all(|new_iexpr| {
                println!("new_iexpr = {}", new_iexpr);
                (new_iexpr.as_constant().unwrap() * i32::from(parent.dtype().size()))
                    % i32::try_from(X86Target::line_size()).unwrap()
                    == 0
            })
        };

        let analysis = aligned_approx(tile.shape(), tile.step_sizes(), parent).unwrap();
        assert_eq!(
            is_aligned,
            analysis,
            "Tile is aligned? {:?}; Analysis result = {:?}; {} tiled to {:?} with step sizes {:?}",
            is_aligned,
            analysis,
            parent,
            tile.shape(),
            tile.step_sizes()
        );
    }
}
