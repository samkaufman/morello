use serde::{Deserialize, Serialize};

use std::fmt::Debug;

use crate::common::{DimSize, Shape};
use crate::views::{BoundaryTile, Tile, TileError, View};

#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub struct Tiling {
    shape: Shape,
    step_sizes: Shape,
}

/// A tiling over either Spec or operand shapes.
///
/// Tiles have arbitrary sizes and steps in each dimension. One tile
/// will have its origin (0, ..) at the tiled view's origin. Not all
/// tiles are necessarily complete; a tile exists if any of its points
/// lies inside the tiled view, and the tile is otherwise truncated.
/// (These are called 'boundary' tiles.)
///
/// This is the basis of shape and tiling logic inference. A tiling over a Spec
/// can be converted into tilings over each operand and vice versa. As a result,
/// tilings can be inferred across composed Specs by tiling any of its
/// component Specs and then propagating tilings across tensors shared with
/// other component Specs.
impl Tiling {
    pub fn new_simple(shape: Shape) -> Tiling {
        let steps = shape.clone();
        Tiling::new_sliding(shape, steps)
    }

    pub fn new_sliding(shape: Shape, steps: Shape) -> Tiling {
        assert_eq!(shape.len(), steps.len());
        Tiling {
            shape,
            step_sizes: steps,
        }
    }

    pub fn is_simple(&self) -> bool {
        self.shape
            .iter()
            .zip(self.step_sizes.iter())
            .all(|(s, t)| *s == *t)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn step_sizes(&self) -> &[DimSize] {
        &self.step_sizes
    }

    /// Returns the total number of steps for a given tensor shape.
    ///
    /// The result includes boundary steps.
    pub fn steps(&self, origin_shape: &[DimSize]) -> u32 {
        assert_eq!(self.shape.len(), origin_shape.len());
        origin_shape
            .iter()
            .enumerate()
            .map(|(d, &s)| self.steps_dim(u8::try_from(d).unwrap(), s))
            .product()
    }

    /// Returns the number of steps taken by a window sliding over a dimension.
    ///
    /// The result includes boundary steps.
    pub fn steps_dim(&self, dim: u8, origin_size: DimSize) -> u32 {
        let d = usize::from(dim);
        debug_assert!(
            origin_size >= self.shape[d],
            "origin_size {} was smaller than tile dimension size {}",
            origin_size,
            self.shape[d]
        );
        divrem::DivCeil::div_ceil(origin_size.get(), self.step_sizes[d].get())
    }

    /// Counts the boundary elements in a given Spec dimension of `origin_size`.
    ///
    /// Returns `0` if all iterations along this dimension are full in
    /// `origin_size`.
    pub fn boundary_size(&self, dim: u8, origin_size: DimSize) -> u32 {
        let span = self.shape[usize::from(dim)].get();
        let step = self.step_sizes[usize::from(dim)].get();
        let origin_size = origin_size.get();

        if step >= span {
            let remainder = origin_size % step;
            if remainder == 0 || remainder >= span {
                0
            } else {
                remainder
            }
        } else if step + 1 == span {
            // This case should be the same as the following, more general,
            // `step < span` case, but when there is only a difference of 1, we know
            // that only one boundary case will result. This lets us satisfy the `u32`
            // result type of this function, rather than returning a collection.
            // step.try_into().unwrap()
            let overlap = span - step;
            assert_eq!(overlap, 1);
            (origin_size - 1) % step + overlap
        } else {
            todo!("Arbitrary overlaps mean multiple boundary cases per dimension!");
        }
    }

    /// Construct a [Tile] over a given [View], ignoring boundary regions.
    pub fn apply_main<V: View>(&self, view: V) -> Result<Tile<V>, TileError> {
        Tile::new(self.shape.clone(), self.step_sizes.clone(), view)
    }

    /// Construct a [Tile] and any necessary [BoundaryTile]s over a given [View].
    pub fn apply_with_boundaries<V>(
        &self,
        view: V,
    ) -> Result<(Tile<V>, Vec<BoundaryTile<V>>), TileError>
    where
        V: View + Clone,
    {
        let main_tile = Tile::new(self.shape.clone(), self.step_sizes.clone(), view.clone())?;

        let rank = self.shape.len();
        let origin_shape = view.shape();
        let mut boundary_tiles = Vec::new();
        for dim_mask in 1..(1 << rank) {
            let mut has_boundary = false;
            let mut boundary_shape = self.shape.clone();
            let mut offsets = vec![0u32; rank];

            for dim in 0..rank {
                let remainder = origin_shape[dim].get() % self.step_sizes[dim].get();
                let main_region_size = origin_shape[dim].get() - remainder;

                if (dim_mask & (1 << dim)) != 0 {
                    // This dimension has a boundary
                    let boundary_size = self.boundary_size(dim as u8, origin_shape[dim]);
                    if boundary_size == 0 {
                        has_boundary = false;
                        break;
                    }
                    has_boundary = true;
                    boundary_shape[dim] = DimSize::new(boundary_size).unwrap();
                    offsets[dim] = main_region_size;
                } else {
                    boundary_shape[dim] = DimSize::new(main_region_size).unwrap();
                    offsets[dim] = 0;
                }
            }

            if has_boundary {
                boundary_tiles.push(BoundaryTile::new(boundary_shape, offsets, view.clone())?);
            }
        }

        Ok((main_tile, boundary_tiles))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Dtype;
    use crate::layout::row_major;
    use crate::shape;
    use crate::target::{Avx2Target, Target};
    use crate::tensorspec::TensorSpec;
    use crate::views::Param;
    use itertools::Itertools;
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use proptest::proptest;
    use std::cmp::max;

    const ALLOW_ARBITRARY_SLIDES: bool = false;
    const MAX_ORIGIN_SIZE: DimSize = nz!(20u32);

    /// A Strategy for generating valid Shapes.
    fn shape_strategy(max_size: DimSize, max_dims: u8) -> impl Strategy<Value = Shape> {
        prop::collection::vec(1u32..=max_size.get(), 1..=usize::from(max_dims))
            .prop_map(|v| v.into_iter().map(|x| DimSize::new(x).unwrap()).collect())
    }

    /// A Strategy for generating valid Tilings.
    fn tiling_strategy(dims: u8) -> impl Strategy<Value = Tiling> {
        shape_strategy(nz!(4u32), dims)
            .prop_flat_map(|shp| {
                let rank = shp.len();
                if ALLOW_ARBITRARY_SLIDES {
                    (Just(shp), prop::collection::vec(1u32..=4, rank).boxed())
                } else {
                    let steps = Strategy::boxed(
                        shp.iter()
                            .map(|&d| max(1, d.get() - 1)..10)
                            .collect::<Vec<_>>(),
                    );
                    (Just(shp), steps)
                }
            })
            .prop_map(|(shape, steps)| {
                Tiling::new_sliding(
                    shape,
                    steps
                        .into_iter()
                        .map(|x| DimSize::new(x).unwrap())
                        .collect(),
                )
            })
    }

    /// A Strategy for generating Tilings along with larger shapes.
    fn tiling_and_origin_shape_strategy() -> impl Strategy<Value = (Tiling, Shape)> {
        tiling_strategy(2).prop_flat_map(|tiling| {
            // let origin_shape = prop::collection::vec(1u32..=4, tiling.shape.len()).prop_map_into();
            let origin_shape = tiling
                .shape()
                .iter()
                .map(|&tile_dim_size| {
                    assert!(tile_dim_size <= MAX_ORIGIN_SIZE);
                    tile_dim_size.get()..=MAX_ORIGIN_SIZE.get()
                })
                .collect::<Vec<_>>()
                .prop_map(|v| v.into_iter().map(|x| DimSize::new(x).unwrap()).collect());
            (Just(tiling), origin_shape)
        })
    }

    /// A Strategy for generating simple Tilings along with larger shapes.
    fn simple_tiling_and_origin_shape_strategy() -> impl Strategy<Value = (Tiling, Shape)> {
        shape_strategy(nz!(4u32), 2).prop_flat_map(|tile_shape| {
            let origin_shape = tile_shape
                .iter()
                .map(|&tile_dim_size| {
                    assert!(tile_dim_size <= MAX_ORIGIN_SIZE);
                    tile_dim_size.get()..=MAX_ORIGIN_SIZE.get()
                })
                .collect::<Vec<_>>()
                .prop_map(|v| v.into_iter().map(|x| DimSize::new(x).unwrap()).collect());
            let tiling = Tiling::new_simple(tile_shape);
            (Just(tiling), origin_shape)
        })
    }

    #[test]
    fn test_tiling_sliding() {
        const OUTER_SHAPE: [DimSize; 5] = [nz!(1u32), nz!(4u32), nz!(2u32), nz!(4u32), nz!(4u32)];
        let t = Tiling::new_sliding(shape![1, 3, 1, 3, 3], shape![1, 3, 2, 1, 1]);
        assert_eq!(t.steps_dim(0, OUTER_SHAPE[0]), 1);
        assert_eq!(t.steps_dim(1, OUTER_SHAPE[1]), 2);
        assert_eq!(t.steps_dim(2, OUTER_SHAPE[2]), 1);
        assert_eq!(t.steps_dim(3, OUTER_SHAPE[3]), 4);
        assert_eq!(t.steps_dim(4, OUTER_SHAPE[4]), 4);
    }

    proptest! {
        #[test]
        fn test_tiling_steps_matches_interpretation(tup in tiling_and_origin_shape_strategy()) {
            let (tiling, origin_shape) = tup;

            // Naively, concretely traverse each tile and see if/when we hit a boundary,
            // test that number of full steps we traversed matches. Do this for each
            // dimension.
            #[allow(clippy::needless_range_loop)]
            for dim in 0..tiling.shape().len() {
                let mut steps_began = 0;
                for (tile_idx, pt_idx) in (0..).cartesian_product(0..tiling.shape()[dim].get()) {
                    let origin_idx = tile_idx * tiling.step_sizes[dim].get() + pt_idx;
                    if pt_idx == 0 && origin_idx >= origin_shape[dim].get() {
                        break
                    }
                    if origin_idx >= origin_shape[dim].get() {
                        continue
                    }
                    if pt_idx == 0 {
                        steps_began += 1;
                    }
                }

                let steps = tiling.steps_dim(dim as u8, origin_shape[dim]);
                assert_eq!(steps, steps_began, "Steps computed as {steps} but expected {steps_began}");
            }
        }

        #[test]
        fn test_tiling_boundary_size_matches_interpretation(tup in tiling_and_origin_shape_strategy()) {
            let (tiling, origin_shape) = tup;

            // Naively, concretely traverse each tile and see if/when we hit a boundary,
            // that boundary has the advertised size in the origin shape. Do this for
            // each dimension.
            #[allow(clippy::needless_range_loop)]
            for dim in 0..tiling.shape().len() {
                for (tile_idx, pt_idx) in (0..).cartesian_product(0..tiling.shape()[dim].get()) {
                    let origin_idx = tile_idx * tiling.step_sizes[dim].get() + pt_idx;
                    if origin_idx >= origin_shape[dim].get() {
                        // We've hit a boundary. Check that it has the right size.
                        let boundary_size = tiling.boundary_size(dim as u8, origin_shape[dim]);
                        assert_eq!(boundary_size, pt_idx, "Boundary size computed as {boundary_size} but expected {pt_idx}");
                        break;
                    }
                }
            }
        }

        /// Tests that when simple tiling is applied with boundaries, the sum of main tile
        /// and boundary tile volumes equals the original tensor volume.
        #[test]
        fn test_simple_tiling_volume_preserved(
            (tiling, origin_shape) in simple_tiling_and_origin_shape_strategy()
        ) {
            let spec = TensorSpec::<Avx2Target>::new_canon(
                origin_shape.clone(),
                Dtype::Uint32,
                Avx2Target::levels()[0],
                row_major(origin_shape.len().try_into().unwrap()),
                None,
            );

            if let Ok((main_tile, boundary_tiles)) = tiling.apply_with_boundaries(&Param::new(0, spec)) {
                let main_volume: u64 = main_tile.shape().iter()
                    .map(|&dim| u64::from(dim.get()))
                    .product();
                let main_steps: u64 = origin_shape.iter()
                    .enumerate()
                    .map(|(dim, &origin_dim)| {
                        u64::from(origin_dim.get() / tiling.step_sizes()[dim].get())
                    })
                    .product();
                let main_total_volume = main_volume * main_steps;

                let boundary_total_volume: u64 = boundary_tiles.iter()
                    .map(|boundary_tile| {
                        boundary_tile.shape().iter()
                            .map(|&dim| u64::from(dim.get()))
                            .product::<u64>()
                    })
                    .sum();

                let original_volume: u64 = origin_shape.iter()
                    .map(|&dim| u64::from(dim.get()))
                    .product();
                prop_assert_eq!(
                    main_total_volume + boundary_total_volume,
                    original_volume,
                    "Mismatching volumes after tiling {} with {}, yielding main {} and boundaries {}",
                    tiling.shape().iter().map(|d| d.get()).join("x"),
                    origin_shape.iter().map(|d| d.get()).join("x"),
                    main_tile.spec(),
                    boundary_tiles.iter()
                        .map(|b| b.spec().to_string())
                        .join(", ")
                );
            }
        }
    }

    #[test]
    fn test_apply_with_boundaries_simple() {
        // Tile 3x4x4 over 5x7x6 (non-multiple in every dimension)
        let spec = TensorSpec::<Avx2Target>::new_canon(
            shape![5, 7, 6],
            Dtype::Uint32,
            Avx2Target::levels()[0],
            row_major(3),
            None,
        );
        let param = Param::new(0, spec);
        let tiling = Tiling::new_simple(shape![3, 4, 4]);

        let result = tiling.apply_with_boundaries(param).unwrap();
        let (main_tile, boundary_tiles) = result;
        assert_eq!(boundary_tiles.len(), 7);

        assert_eq!(main_tile.shape(), &[nz!(3u32), nz!(4u32), nz!(4u32)]);

        let dim0_boundary = &boundary_tiles[0];
        assert_eq!(dim0_boundary.shape(), &[nz!(2u32), nz!(4u32), nz!(4u32)]);
        assert_eq!(dim0_boundary.offsets(), &[3, 0, 0]);

        let dim1_boundary = &boundary_tiles[1];
        assert_eq!(dim1_boundary.shape(), &[nz!(3u32), nz!(3u32), nz!(4u32)]);
        assert_eq!(dim1_boundary.offsets(), &[0, 4, 0]);

        let dim01_boundary = &boundary_tiles[2];
        assert_eq!(dim01_boundary.shape(), &[nz!(2u32), nz!(3u32), nz!(4u32)]);
        assert_eq!(dim01_boundary.offsets(), &[3, 4, 0]);

        let dim2_boundary = &boundary_tiles[3];
        assert_eq!(dim2_boundary.shape(), &[nz!(3u32), nz!(4u32), nz!(2u32)]);
        assert_eq!(dim2_boundary.offsets(), &[0, 0, 4]);

        let dim02_boundary = &boundary_tiles[4];
        assert_eq!(dim02_boundary.shape(), &[nz!(2u32), nz!(4u32), nz!(2u32)]);
        assert_eq!(dim02_boundary.offsets(), &[3, 0, 4]);

        let dim12_boundary = &boundary_tiles[5];
        assert_eq!(dim12_boundary.shape(), &[nz!(3u32), nz!(3u32), nz!(2u32)]);
        assert_eq!(dim12_boundary.offsets(), &[0, 4, 4]);

        let corner_boundary = &boundary_tiles[6];
        assert_eq!(corner_boundary.shape(), &[nz!(2u32), nz!(3u32), nz!(2u32)]);
        assert_eq!(corner_boundary.offsets(), &[3, 4, 4]);
    }

    #[test]
    fn test_apply_with_boundaries_no_boundaries() {
        // Tile 2x3 over 4x6 (multiples in all dimensions)
        let spec = TensorSpec::<Avx2Target>::new_canon(
            shape![4, 6],
            Dtype::Uint32,
            Avx2Target::levels()[0],
            row_major(2),
            None,
        );
        let param = Param::new(0, spec);
        let tiling = Tiling::new_simple(shape![2, 3]);
        let result = tiling.apply_with_boundaries(param).unwrap();
        let (main_tile, boundary_tiles) = result;
        assert_eq!(main_tile.shape(), &[nz!(2u32), nz!(3u32)]);
        assert_eq!(boundary_tiles.len(), 0);
    }

    #[test]
    fn test_apply_with_boundaries_some_nonmultiples() {
        // Tile 2x4x3 over 6x7x9 (multiple in dims 0 and 2, non-multiple in dim 1)
        let spec = TensorSpec::<Avx2Target>::new_canon(
            shape![6, 7, 9],
            Dtype::Uint32,
            Avx2Target::levels()[0],
            row_major(3),
            None,
        );
        let param = Param::new(0, spec);
        let tiling = Tiling::new_simple(shape![2, 4, 3]);

        let result = tiling.apply_with_boundaries(param).unwrap();
        let (main_tile, boundary_tiles) = result;

        assert_eq!(boundary_tiles.len(), 1);
        assert_eq!(main_tile.shape(), &[nz!(2u32), nz!(4u32), nz!(3u32)]);

        let dim1_boundary = &boundary_tiles[0];
        assert_eq!(dim1_boundary.shape(), &[nz!(6u32), nz!(3u32), nz!(9u32)]);
        assert_eq!(dim1_boundary.offsets(), &[0, 4, 0]);
    }
}
