use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::spec::Spec;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, Tile, View};

use itertools::Itertools;

use std::collections::HashMap;
use std::{iter, slice};

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
#[derive(Debug, Clone)]
pub struct Loop<Tgt: Target> {
    pub tiles: Vec<LoopTile<Tgt>>,
    pub body: Box<ImplNode<Tgt>>,
    pub parallel: bool,
    pub spec: Option<Spec<Tgt>>,
}

#[derive(Debug, Clone)]
pub struct LoopTile<Tgt: Target> {
    pub axes: Vec<u8>,
    pub tile: Tile<Param<Tgt>>,
}

impl<Tgt: Target> Impl<Tgt> for Loop<Tgt> {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorSpec<Tgt>> + '_> {
        debug_assert!(
            self.tiles
                .iter()
                .tuple_windows::<(_, _)>()
                .all(|(a, b)| a.tile.view.0 < b.tile.view.0),
            "tile weren't sorted"
        );

        // Return an iterator over parameters from loop tiles where they apply and the inner body
        // elsewhere.
        let mut next_tile_idx = 0;
        let mut body_parameters = self.body.parameters().enumerate();
        Box::new(iter::from_fn(move || match body_parameters.next() {
            None => {
                debug_assert_eq!(next_tile_idx, self.tiles.len());
                None
            }
            Some((i, body_param)) => match self.tiles.get(next_tile_idx) {
                Some(next_tile) if i == usize::from(next_tile.tile.view.0) => {
                    next_tile_idx += 1;
                    Some(next_tile.tile.view.spec())
                }
                _ => Some(body_param),
            },
        }))
    }

    fn children(&self) -> &[ImplNode<Tgt>] {
        slice::from_ref(&self.body)
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        compute_loop_main_cost::<Tgt>(
            self.steps(),
            self.full_steps(),
            self.parallel,
            child_costs[0],
        )
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt>>) -> Self {
        let mut new_children = new_children;
        let new_loop = Loop {
            tiles: self.tiles.clone(),
            body: Box::new(new_children.next().unwrap()),
            parallel: self.parallel,
            spec: self.spec.clone(),
        };
        debug_assert!(new_children.next().is_none());
        new_loop
    }

    fn bind<'i, 'j: 'i>(
        &'j self,
        args: &[&'j dyn View<Tgt = Tgt>],
        env: &'i mut HashMap<Param<Tgt>, &'j dyn View<Tgt = Tgt>>,
    ) {
        let mut inner_args = args.to_vec();
        for tile in &self.tiles {
            tile.tile.bind(args, env);
            inner_args[usize::from(tile.tile.view.0)] = &tile.tile;
        }
        self.body.bind(&inner_args, env);
    }

    fn pprint_line<'a>(
        &'a self,
        names: &mut NameEnv<'a, dyn View<Tgt = Tgt>>,
        param_bindings: &HashMap<Param<Tgt>, &dyn View<Tgt = Tgt>>,
    ) -> Option<String> {
        Some(format!(
            "tile{} ({})",
            if self.parallel { "[p]" } else { "" },
            self.tiles
                .iter()
                .map(|t| {
                    let source = param_bindings[&t.tile.view];
                    let left = names.name(&t.tile).to_owned();
                    let right = names.get_name_or_display(source);
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
}

impl<Tgt: Target> Loop<Tgt> {
    pub fn steps(&self) -> u32 {
        first_dim_per_axis(self)
            .map(|(loop_tile, dim_idx)| loop_tile.tile.steps_dim(dim_idx))
            .product()
    }

    pub fn full_steps(&self) -> u32 {
        first_dim_per_axis(self)
            .map(|(loop_tile, dim_idx)| loop_tile.tile.full_steps_dim(dim_idx))
            .product()
    }
}

pub(crate) fn compute_loop_main_cost<Tgt: Target>(
    steps: u32,
    full_steps: u32,
    parallel: bool,
    body_cost: MainCost,
) -> MainCost {
    let (factor, overhead) = if parallel {
        let processors = u32::from(Tgt::processors());
        let boundary_steps = steps - full_steps;
        let per_thread_factor = full_steps.div_ceil(processors) + boundary_steps;
        (per_thread_factor, PAR_TILE_OVERHEAD)
    } else {
        (steps, 0)
    };
    body_cost.saturating_mul(factor) + overhead
}

/// Yields the first tile and tile dimension seen for each unique axis.
fn first_dim_per_axis<Tgt: Target>(imp: &Loop<Tgt>) -> impl Iterator<Item = (&LoopTile<Tgt>, u8)> {
    imp.tiles
        .iter()
        .flat_map(|loop_tile| {
            loop_tile
                .axes
                .iter()
                .enumerate()
                .map(move |(i, s)| (loop_tile, u8::try_from(i).unwrap(), *s))
        })
        .unique_by(|(_, _, axis)| *axis)
        .map(|(i, s, _)| (i, s))
}
