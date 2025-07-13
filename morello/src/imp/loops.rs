use crate::common::DimSize;
use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{BoundaryTile, Tile, TileError, View, ViewE};
use itertools::Itertools;
use std::fmt::{self, Debug};

const PAR_TILE_OVERHEAD: MainCost = 45_000; // rough cycle estimate

/// An Impl representing a loop over a set of zipped [`Tile`]s.
///
/// It has a form like:
/// ```text
/// (A, B, C) => {
///   tile (a: 1x1 <-[0,1]- A, b: 1x1 <-[2,0]- B) {
///     body(a, b, C)
///   }
/// }
/// ```
/// Or, before name binding, where [`Param`]s (`#n`) refer to the [`Loop`]'s
/// parent:
/// ```text
/// tile (a: 1x1 <-[0,1]- #0, b: 1x1 <-[2,0]- #1) {
///   body(a, b, #2)
/// }
/// ```
/// Notice that untiled arguments (`c`) are passed through to the body
/// [`Impl`] while tiled arguments are replaced by [`Tile`]s
/// (`a` and `b`).
#[derive(Clone)]
pub struct Loop<Tgt: Target> {
    pub tiles: Vec<LoopTile<Tgt>>,
    /// Implementations for different regions of the loop. This is never empty; the
    /// first body is the main, largest sub-Impl.
    ///
    /// The vector is sorted by a lexicographic ordering of bitvectors where each bit
    /// corresponds to an iteration dimension. A bit is set if that body is in the
    /// boundary region of that dimension. Not every iteration dimension has a boundary
    /// region, so not every region is represented in this Vec.
    pub bodies: Vec<ImplNode<Tgt>>,
    pub parallel: bool,
    /// The [Spec] implemented by this [Loop].
    pub spec: Option<Spec<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct LoopTile<Tgt: Target> {
    pub parameter_index: u8,
    pub axes: Vec<u8>,
    pub tile: Tile<Box<ViewE<Tgt>>>,
}

// #[derive(Debug, Clone)]
// pub struct BoundaryLoopTile<Tgt: Target> {
//     pub parameter_index: u8,
//     pub axes: Vec<u8>,
//     pub tile: BoundaryTile<Box<ViewE<Tgt>>>,
// }

impl<Tgt: Target> Impl<Tgt> for Loop<Tgt> {
    type BindOut = Self;

    fn children(&self) -> &[ImplNode<Tgt>] {
        &self.bodies
    }

    fn default_child(&self) -> Option<usize> {
        Some(0) // the full-tile sub-Spec
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        let all_dims: Vec<(u32, u32)> = unique_dims_per_axis(&self.tiles)
            .map(|(lt, dim)| (lt.tile.steps_dim(dim), lt.tile.full_steps_dim(dim)))
            .collect();
        compute_loop_main_cost::<Tgt>(&all_dims, self.parallel, child_costs)
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        let bodies: Vec<_> = new_children.collect();
        assert_eq!(bodies.len(), self.bodies.len());
        Loop {
            tiles: self.tiles.clone(),
            bodies,
            parallel: self.parallel,
            spec: self.spec.clone(),
        }
    }

    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Tgt>>) -> Self::BindOut {
        let mut new_tiles = Vec::with_capacity(self.tiles.len());
        for tile in &self.tiles {
            let ViewE::Tile(bound) = tile.tile.clone().bind(get_argument) else {
                unreachable!()
            };
            new_tiles.push(LoopTile {
                parameter_index: tile.parameter_index,
                axes: tile.axes.clone(),
                tile: bound,
            });
        }

        // TODO: Modify `self.bodies` in-place to avoid another heap allocation.
        let new_bodies = self
            .bodies
            .into_iter()
            .map(|b| b.bind(get_argument))
            .collect();
        Loop {
            tiles: new_tiles,
            bodies: new_bodies,
            parallel: self.parallel,
            spec: self.spec,
        }
    }

    fn pprint_line(&self, names: &mut NameEnv) -> Option<String> {
        Some(format!(
            "tile{} ({})",
            if self.parallel { "[p]" } else { "" },
            self.tiles
                .iter()
                .map(|t| {
                    let left = names.name(&t.tile).to_owned();
                    let right = names.get_name_or_display(&t.tile.view);
                    format!(
                        "{}: {} <-[{}]- {}",
                        left,
                        t.tile.spec(),
                        t.axes.iter().map(|v| v.to_string()).join(", "),
                        right
                    )
                })
                .join(", ")
        ))
    }

    fn spec(&self) -> Option<&Spec<Tgt>> {
        self.spec.as_ref()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>),
    {
        for tile in &self.tiles {
            tile.tile.visit_params(visitor);
        }
        for child in self.children() {
            child.visit_params(visitor);
        }
    }
}

impl<Tgt: Target> Debug for Loop<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loop")
            .field("tiles", &self.tiles)
            .field("parallel", &self.parallel)
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl<Tgt: Target> Loop<Tgt> {
    pub fn full_steps(&self) -> u32 {
        unique_dims_per_axis(&self.tiles)
            .map(|(loop_tile, dim_idx)| loop_tile.tile.full_steps_dim(dim_idx))
            .product()
    }

    /// Create a boundary tile for a specific region and loop tile
    fn create_boundary_tile_for_region(
        &self,
        loop_tile: &LoopTile<Tgt>,
        region_id: usize,
    ) -> Result<BoundaryTile<Box<ViewE<Tgt>>>, TileError> {
        let tile_shape = loop_tile.tile.shape();
        let original_shape = loop_tile.tile.view.shape();
        let axes = &loop_tile.axes;

        let mut boundary_shape = original_shape.to_vec();
        let mut offsets_raw = vec![0u32; original_shape.len()];

        // Check each tiled dimension for boundary regions
        for (dim_idx, &axis) in axes.iter().enumerate() {
            let axis_idx = usize::from(axis);

            // Check if this axis has a boundary region in this region_id
            if (region_id & (1 << axis_idx)) != 0 {
                // This dimension has a boundary region
                let tile_size = tile_shape[dim_idx];
                let original_size = original_shape[dim_idx];

                // Calculate the boundary size (remainder)
                if let Some(boundary_size) = DimSize::new(original_size.get() % tile_size.get()) {
                    boundary_shape[dim_idx] = boundary_size;

                    // Calculate the offset (where the boundary starts)
                    let full_tiles = original_size.get() / tile_size.get();
                    let offset = full_tiles * tile_size.get();
                    offsets_raw[dim_idx] = offset;
                }
            }
        }

        BoundaryTile::new(
            boundary_shape.into(),
            offsets_raw,
            loop_tile.tile.view.clone(),
        )
    }
}

/// Compute the main cost of a tile-loop by summing full and boundary regions.
///
/// Panics if any boundary dimension violates the constraint that `steps == full_steps + 1`.
///
/// # Arguments
///
/// * `dims` - slice of per-dimension `(steps, full_steps)` pairs.
///   **Note**: For boundary dimensions, `steps` must equal `full_steps + 1`
///   (boundary regions have exactly one iteration)
/// * `parallel` is `true` if this is a parellel loop.
/// * `body_costs` is the cost of each region's sub-Spec. The slice
///   must have length 2^B, where B is the number of boundary dimensions (axes with a
///   partial extra tile). Entry at index `m` gives the cost for region `m`, where bit 0
///   selects the innermost boundary dimension, bit 1 the next outer dimension, and so on.
///   For example, with two boundary dimensions:
///   - index 0 (`0b00`): full region (no boundaries),
///   - index 1 (`0b01`): innermost dimension's boundary only,
///   - index 2 (`0b10`): next outer dimension's boundary only,
///   - index 3 (`0b11`): the corner/boundary of both dimensions.
pub(crate) fn compute_loop_main_cost<Tgt: Target>(
    dims: &[(u32, u32)],
    parallel: bool,
    body_costs: &[MainCost],
) -> MainCost {
    debug_assert!(!body_costs.is_empty());

    // Initialize total with the main region's cost.
    let mut total = {
        let mut full_iterations = 1u32;
        for &(_, full_steps) in dims {
            full_iterations = full_iterations
                .checked_mul(full_steps)
                .expect("number of full iterations doesn't overflow");
        }
        if parallel {
            full_iterations = full_iterations.div_ceil(u32::from(Tgt::processors()))
        }
        body_costs[0].saturating_mul(full_iterations)
    };

    // Add boundary region costs (each boundary region is executed exactly once)
    for &cost in &body_costs[1..] {
        total = total.saturating_add(cost);
    }

    if parallel {
        total = total.saturating_add(PAR_TILE_OVERHEAD);
    }
    total
}

/// Returns an iterator of representative dimensions from given [LoopTile]s, returning
/// one per axis (across all tiles).
pub(crate) fn unique_dims_per_axis<Tgt: Target>(
    tiles: &[LoopTile<Tgt>],
) -> impl Iterator<Item = (&LoopTile<Tgt>, u8)> {
    tiles
        .iter()
        .flat_map(|loop_tile| {
            loop_tile
                .axes
                .iter()
                .enumerate()
                .map(move |(i, s)| (loop_tile, u8::try_from(i).unwrap(), *s))
        })
        .unique_by(|(_, _, axis)| *axis)
        .map(|(lt, dim, _)| (lt, dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::X86Target;

    #[test]
    fn test_compute_loop_main_cost_serial_1d() {
        // Single axis. Boundary has 1 step.
        let cost = compute_loop_main_cost::<X86Target>(&[(9, 8)], false, &[5, 4]);
        assert_eq!(cost, 5 * 8 + 4);
    }

    #[test]
    fn test_compute_loop_main_cost_parallel_1d() {
        // Single axis, parallel. Boundary has 1 step.
        let procs = u32::from(X86Target::processors());
        let cost = compute_loop_main_cost::<X86Target>(&[(9, 8)], true, &[5, 5]);
        let expected = 5 * 8u32.div_ceil(procs) + 5 + PAR_TILE_OVERHEAD;
        assert_eq!(cost, expected);
    }

    #[test]
    fn test_compute_loop_main_cost_serial_exact_div_1d() {
        // Single axis that divides evenly => no boundary contribution
        let cost = compute_loop_main_cost::<X86Target>(&[(8, 8)], false, &[4]);
        assert_eq!(cost, 4 * 8);
    }

    #[test]
    fn test_compute_loop_main_cost_serial_2d() {
        let dims = &[(3, 2), (5, 4)];
        let cost = compute_loop_main_cost::<X86Target>(dims, false, &[7, 6, 9, 10]);
        let expected = (7 * 2 * 4) + 6 + 9 + 10; // 3 boundary regions (each executed once)
        assert_eq!(cost, expected);
    }

    #[test]
    fn test_compute_loop_main_cost_serial_2d_with_one_exact_axis() {
        // Only axis 1 has boundary conditions.
        let dims = &[(8, 8), (4, 3)];
        let cost = compute_loop_main_cost::<X86Target>(dims, false, &[2, 3]);
        let expected = (2 * 8 * 3) + 3; // 1 boundary region
        assert_eq!(cost, expected);
    }
}
