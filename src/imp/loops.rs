use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::nameenv::NameEnv;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::{Param, Tile, View};

use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::{iter, slice};

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
pub struct Loop<Tgt: Target, Aux: Clone> {
    pub tiles: Vec<LoopTile<Tgt>>,
    pub body: Box<ImplNode<Tgt, Aux>>,
    pub parallel: bool,
    pub aux: Aux,
}

#[derive(Debug, Clone)]
pub struct LoopTile<Tgt: Target> {
    pub axes: SmallVec<[u8; 5]>,
    pub tile: Tile<Param<Tgt>>,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Loop<Tgt, Aux> {
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

    fn children(&self) -> &[ImplNode<Tgt, Aux>] {
        slice::from_ref(&self.body)
    }

    fn memory_allocated(&self) -> MemoryAllocation {
        MemoryAllocation::none::<Tgt>()
    }

    fn compute_main_cost(&self, child_costs: &[MainCost]) -> MainCost {
        let factor = if self.parallel {
            let processors = u32::from(Tgt::processors());
            let steps = self.steps();
            let main_steps = self.full_steps();
            ((main_steps + processors - 1) / processors) + (steps - main_steps)
        } else {
            self.steps()
        };
        child_costs[0].saturating_mul(factor)
    }

    fn replace_children(&self, new_children: impl Iterator<Item = ImplNode<Tgt, Aux>>) -> Self {
        let mut new_children = new_children;
        let new_loop = Loop {
            tiles: self.tiles.clone(),
            body: Box::new(new_children.next().unwrap()),
            parallel: self.parallel,
            aux: self.aux.clone(),
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

    fn aux(&self) -> &Aux {
        &self.aux
    }

    fn drop_aux(self) -> ImplNode<Tgt, ()> {
        ImplNode::Loop(Loop {
            tiles: self.tiles,
            body: Box::new(self.body.drop_aux()),
            parallel: self.parallel,
            aux: (),
        })
    }
}

impl<Tgt: Target, Aux: Clone> Loop<Tgt, Aux> {
    pub fn steps(&self) -> u32 {
        first_dim_per_axis(self)
            .map(|(loop_tile, dim_idx, _)| loop_tile.tile.steps_dim(dim_idx))
            .product()
    }

    pub fn full_steps(&self) -> u32 {
        first_dim_per_axis(self)
            .map(|(loop_tile, dim_idx, _)| loop_tile.tile.steps_dim(dim_idx))
            .product()
    }
}

/// Yields the first tile and tile dimension seen for each unique axis.
fn first_dim_per_axis<Tgt: Target, Aux: Clone>(
    imp: &Loop<Tgt, Aux>,
) -> impl Iterator<Item = (&LoopTile<Tgt>, u8, u8)> {
    imp.tiles
        .iter()
        .flat_map(|loop_tile| {
            loop_tile
                .axes
                .iter()
                .copied()
                .enumerate()
                .map(move |(i, s)| (loop_tile, u8::try_from(i).unwrap(), s))
        })
        .unique_by(|(_, _, axis)| *axis)
}
