use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::fmt::Debug;

use crate::common::{DimSize, Shape};

#[derive(Debug, Clone, Eq, PartialEq, Deserialize, Serialize)]
pub struct Tiling {
    shape: Shape,
    step_sizes: SmallVec<[DimSize; 5]>,
}

/// A tiling over either Spec or operand shapes.
///
/// This is the basis of shape and tiling logic inference. A tiling over a Spec
/// can be converted into tilings over each operand and vice versa. As a result,
/// tilings can be inferred across composed Specs by tiling any of its
/// component Specs and then propogating tilings across tensors shared with
/// other component Specs.
impl Tiling {
    pub fn new_simple(shape: Shape) -> Tiling {
        let steps = shape.clone();
        Tiling::new_sliding(shape, steps)
    }

    pub fn new_sliding(shape: Shape, steps: Shape) -> Tiling {
        assert_eq!(shape.len(), steps.len());
        assert!(steps.iter().all(|&s| s > 0));
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
    /// The result will include boundary steps.
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
    /// The result will include boundary steps.
    pub fn steps_dim(&self, dim: u8, origin_size: DimSize) -> u32 {
        let d = usize::from(dim);
        debug_assert!(
            origin_size >= self.shape[d],
            "origin_size is {} and tile shape is {}",
            origin_size,
            self.shape[d]
        );
        divrem::DivCeil::div_ceil(1 + origin_size - self.shape[d], self.step_sizes[d])
    }

    /// Counts the boundary elements in a given Spec dimension of `origin_size`.
    ///
    /// Returns `0` if all iterations along this dimension are full in
    /// `origin_size`.
    pub fn boundary_size(&self, dim: u8, origin_size: DimSize) -> u32 {
        let span = self.shape[usize::from(dim)];
        let step = self.step_sizes[usize::from(dim)];

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest::proptest;
    use smallvec::smallvec;
    use std::cmp::max;

    const ALLOW_ARBITRARY_SLIDES: bool = false;
    const MAX_ORIGIN_SIZE: DimSize = 20;

    /// A Strategy for generating valid Shapes.
    fn shape_strategy(max_size: DimSize, max_dims: u8) -> impl Strategy<Value = Shape> {
        prop::collection::vec(1u32..=max_size, 1..=usize::from(max_dims)).prop_map_into()
    }

    /// A Strategy for generating valid Tilings.
    fn tiling_strategy(dims: u8) -> impl Strategy<Value = Tiling> {
        shape_strategy(4, dims)
            .prop_flat_map(|shp| {
                let rank = shp.len();
                if ALLOW_ARBITRARY_SLIDES {
                    (Just(shp), prop::collection::vec(1u32..=4, rank).boxed())
                } else {
                    let steps =
                        Strategy::boxed(shp.iter().map(|&d| max(1, d - 1)..10).collect::<Vec<_>>());
                    (Just(shp), steps)
                }
            })
            .prop_map(|(shape, steps)| Tiling::new_sliding(shape, steps.into()))
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
                    tile_dim_size..=MAX_ORIGIN_SIZE
                })
                .collect::<Vec<_>>()
                .prop_map_into();
            (Just(tiling), origin_shape)
        })
    }

    #[test]
    fn test_tiling_sliding() {
        let t = Tiling::new_sliding(smallvec![1, 4, 3, 3], smallvec![1, 4, 1, 1]);
        assert_eq!(t.steps_dim(0, 1), 1);
        assert_eq!(t.steps_dim(1, 4), 1);
        assert_eq!(t.steps_dim(2, 4), 2);
        assert_eq!(t.steps_dim(3, 4), 2);
    }

    proptest! {
        #[test]
        fn test_tiling_steps_matches_interpretation(tup in tiling_and_origin_shape_strategy()) {
            let (tiling, origin_shape) = tup;

            // Naively, concretely traverse each tile and see if/when we hit a boundary,
            // test that number of full steps we traversed matches. Do this for each
            // dimension.
            for dim in 0..tiling.shape().len() {
                let mut steps_began = 0;
                for (tile_idx, pt_idx) in (0..).cartesian_product(0..tiling.shape()[dim]) {
                    let origin_idx = tile_idx * tiling.step_sizes[dim] + pt_idx;
                    if pt_idx == 0 && origin_idx >= origin_shape[dim] {
                        break
                    }
                    if origin_idx >= origin_shape[dim] {
                        continue
                    }
                    if pt_idx == 0 {
                        steps_began += 1;
                    }
                }

                let steps = tiling.steps_dim(dim as u8, origin_shape[dim]);
                assert_eq!(steps, steps_began, "Steps computed as {} but expected {}", steps, steps_began);
            }
        }

        #[test]
        fn test_tiling_boundary_size_matches_interpretation(tup in tiling_and_origin_shape_strategy()) {
            let (tiling, origin_shape) = tup;

            // Naively, concretely traverse each tile and see if/when we hit a boundary,
            // that boundary has the advertised size in the origin shape. Do this for
            // each dimension.
            for dim in 0..tiling.shape().len() {
                for (tile_idx, pt_idx) in (0..).cartesian_product(0..tiling.shape()[dim]) {
                    let origin_idx = tile_idx * tiling.step_sizes[dim] + pt_idx;
                    if origin_idx >= origin_shape[dim] {
                        // We've hit a boundary. Check that it has the right size.
                        let boundary_size = tiling.boundary_size(dim as u8, origin_shape[dim]);
                        assert_eq!(boundary_size, pt_idx, "Boundary size computed as {} but expected {}", boundary_size, pt_idx);
                        break;
                    }
                }
            }
        }
    }
}
