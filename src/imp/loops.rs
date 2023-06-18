use crate::cost::MainCost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::MemoryAllocation;
use crate::pprint::NameEnv;
use crate::target::Target;
use crate::views::{Param, Tile, View};

use itertools::Itertools;
use smallvec::SmallVec;
use std::{cmp, slice};

const MAX_COST: MainCost = u64::MAX;

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
    pub subscripts: SmallVec<[u8; 5]>,
    pub tile: Tile<Param<Tgt>>,
}

impl<Tgt: Target, Aux: Clone> Impl<Tgt, Aux> for Loop<Tgt, Aux> {
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
        cmp::min(child_costs[0] * MainCost::from(factor), MAX_COST)
    }

    fn line_strs<'a>(
        &'a self,
        names: &mut NameEnv<'a, Tgt>,
        args: &[&dyn View<Tgt = Tgt>],
    ) -> Option<String> {
        Some(format!(
            "tile{} ({})",
            if self.parallel { "[p]" } else { "" },
            self.tiles
                .iter()
                .map(|t| {
                    let source = args[usize::from(t.tile.view.0)];
                    let left = names.name(&t.tile).to_owned();
                    let right = names.get_name_or_display(source);
                    format!(
                        "{}: {} <-[{}]- {}",
                        left,
                        t.tile.spec(),
                        t.subscripts.iter().map(|v| v.to_string()).join(", "),
                        right
                    )
                })
                .join(", ")
        ))
    }

    fn aux(&self) -> &Aux {
        &self.aux
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
}

impl<Tgt: Target, Aux: Clone> Loop<Tgt, Aux> {
    pub fn steps(&self) -> u32 {
        first_dim_per_subscript(self)
            .map(|(loop_tile, dim_idx, _)| loop_tile.tile.steps_dim(dim_idx))
            .product()
    }

    pub fn full_steps(&self) -> u32 {
        first_dim_per_subscript(self)
            .map(|(loop_tile, dim_idx, _)| {
                let s = loop_tile.tile.steps_dim(dim_idx);
                match loop_tile.tile.boundary_size(dim_idx) {
                    0 => s,
                    _ => s - 1,
                }
            })
            .product()
    }
}

/// Yields the first tile and tile dimension seen for each unique subscript.
fn first_dim_per_subscript<Tgt: Target, Aux: Clone>(
    imp: &Loop<Tgt, Aux>,
) -> impl Iterator<Item = (&LoopTile<Tgt>, u8, u8)> {
    imp.tiles
        .iter()
        .flat_map(|loop_tile| {
            loop_tile
                .subscripts
                .iter()
                .copied()
                .enumerate()
                .map(move |(i, s)| (loop_tile, u8::try_from(i).unwrap(), s))
        })
        .unique_by(|(_, _, subscript)| *subscript)
}
