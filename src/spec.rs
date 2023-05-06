use super::common::{DimSize, Shape};
use crate::common::{Contig, Dtype};
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::target::MemoryLevel;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::tiling::PartialTile;

use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::fmt::Display;
use std::iter::Iterator;
use std::iter::{self, once};

const LIMIT_VECTORS_TO_ONE_DIM: bool = true;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum Spec<Tgt: Target> {
    Matmul {
        accum: bool,
        m: DimSize,
        k: DimSize,
        n: DimSize,
        dtype: Dtype,
        contiguous_abstractions: SmallVec<[Contig; 3]>,
        alignments: SmallVec<[bool; 3]>,
        levels: SmallVec<[Tgt::Level; 3]>,
        layouts: SmallVec<[Layout; 3]>,
        vector_shapes: SmallVec<[Option<Shape>; 3]>,
        serial_only: bool,
    },
    Load {
        outer_tensor_spec: TensorSpec<Tgt>,
        inner_level: Tgt::Level,
        inner_layout: Layout,
        inner_vector_shape: Option<Shape>,
        serial_only: bool,
    },
    Store {
        outer_tensor_spec: TensorSpec<Tgt>,
        inner_level: Tgt::Level,
        inner_layout: Layout,
        inner_vector_shape: Option<Shape>,
        serial_only: bool,
    },
    Zero {
        tensor_spec: TensorSpec<Tgt>,
        serial_only: bool,
    },
}

impl<Tgt: Target> Spec<Tgt> {
    pub fn serial_only(&self) -> bool {
        match self {
            Spec::Matmul { serial_only, .. }
            | Spec::Load { serial_only, .. }
            | Spec::Store { serial_only, .. }
            | Spec::Zero { serial_only, .. } => *serial_only,
        }
    }

    pub fn operand_count(&self) -> usize {
        // TODO: This is slow.
        self.operands().len()
    }

    pub fn operands(&self) -> Vec<TensorSpec<Tgt>> {
        match &self {
            Spec::Matmul {
                accum: _,
                m,
                k,
                n,
                dtype,
                contiguous_abstractions,
                alignments,
                levels,
                layouts,
                vector_shapes,
                serial_only: _serial_only,
            } => {
                vec![
                    TensorSpec::new(
                        smallvec![*m, *k],
                        *dtype,
                        contiguous_abstractions[0],
                        alignments[0],
                        levels[0],
                        layouts[0].canonicalize_for_shape(&[*m, *k]),
                        vector_shapes[0].clone(),
                    ),
                    TensorSpec::new(
                        smallvec![*k, *n],
                        *dtype,
                        contiguous_abstractions[1],
                        alignments[1],
                        levels[1],
                        layouts[1].canonicalize_for_shape(&[*k, *n]),
                        vector_shapes[1].clone(),
                    ),
                    TensorSpec::new(
                        smallvec![*m, *n],
                        *dtype,
                        contiguous_abstractions[2],
                        alignments[2],
                        levels[2],
                        layouts[2].canonicalize_for_shape(&[*m, *n]),
                        vector_shapes[2].clone(),
                    ),
                ]
            }
            Spec::Load {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only: _,
            }
            | Spec::Store {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only: _,
            } => {
                let mut inner_tensor_spec = outer_tensor_spec.clone();
                inner_tensor_spec.set_level(*inner_level, inner_vector_shape.clone());
                inner_tensor_spec.set_layout(inner_layout.clone());
                vec![outer_tensor_spec.clone(), inner_tensor_spec]
            }
            Spec::Zero { tensor_spec, .. } => vec![tensor_spec.clone()],
        }
    }

    pub fn inputs(&self) -> Vec<TensorSpec<Tgt>> {
        match &self {
            Spec::Matmul { .. } => self.operands()[..2].to_vec(),
            Spec::Load { .. } | Spec::Store { .. } => {
                // TODO: Just grab the item instead of calling operands
                vec![self.operands()[0].clone()]
            }
            Spec::Zero { .. } => vec![],
        }
    }

    pub fn output(&self) -> TensorSpec<Tgt> {
        self.operands()[self.output_idx()].clone()
    }

    fn output_idx(&self) -> usize {
        match &self {
            Spec::Matmul { .. } => 2,
            Spec::Load { .. } | Spec::Store { .. } => 1,
            Spec::Zero { .. } => 0,
        }
    }

    pub fn expansions(&self) -> Box<dyn Iterator<Item = ImplNode<Tgt>> + '_> {
        let iter = self.tile_out_expansions();
        let iter = iter.chain(self.move_expansions());
        let iter = iter.chain(Tgt::expansions(self));

        match &self {
            Spec::Matmul { accum, .. } if !*accum => {
                Box::new(iter.chain(iter::once(ImplNode::MatmulAccumBlock)))
            }
            Spec::Matmul { accum, .. } if *accum => Box::new(iter.chain(self.split_expansions())),
            _ => Box::new(iter),
        }
    }

    fn tile_out_expansions(&self) -> impl Iterator<Item = ImplNode<Tgt>> + '_ {
        let serial_only = self.serial_only();
        let output = self.output();
        gen_tile_sizes::<Tgt>(output.dim_sizes(), true)
            .flat_map(move |tile_shape| {
                let mut ts = SmallVec::<[Option<ImplNode<Tgt>>; 2]>::new();
                ts.push(self.tile_out(&tile_shape, false));
                if !serial_only {
                    ts.push(self.tile_out(&tile_shape, true));
                }
                ts
            })
            .flatten()
    }

    fn split_expansions(&self) -> Box<dyn Iterator<Item = ImplNode<Tgt>> + '_> {
        match &self {
            Spec::Matmul { k, accum, .. } if *accum => Box::new(
                dim_range(*k, false)
                    .filter(|&new_k| self.split_valid(new_k))
                    .map(|k| self.split(k)),
            ),
            _ => panic!("split_expansions called on non-accumulating Matmul"),
        }
    }

    fn split_valid(&self, new_k: DimSize) -> bool {
        debug_assert!(matches!(&self, Spec::Matmul { .. }));
        let operands = self.operands();
        let m = operands[0].dim_sizes()[0];
        let orig_k = operands[0].dim_sizes()[1];
        let n = operands[1].dim_sizes()[1];

        // Special-case for splitting to single-element tensors, which will be normalized
        // to row-major. This is necessary for splits in any other layout to be
        // discovered by search.
        // TODO: This is pretty ad-hoc. Should there be an alternative to
        //   `is_valid_tile_shape` that includes this case?
        if m == 1 && new_k == 1 && n == 1 {
            return true;
        }

        if new_k >= orig_k || !operands[0].is_valid_tile_shape(&[m, new_k]) {
            false
        } else {
            operands[1].is_valid_tile_shape(&[new_k, n])
        }
    }

    fn move_expansions(&self) -> impl Iterator<Item = ImplNode<Tgt>> + '_ {
        // TODO: Enable prefetching moves.

        // TODO: Filter with `can_move_to`.

        let mut results = vec![]; // TODO: Don't accumulate. Return an iterator.
        if matches!(self, Spec::Load { .. } | Spec::Store { .. }) {
            return results.into_iter();
        }

        for (i, operand) in self.operands().iter().enumerate() {
            // Yield actions for movement with register file destination, which
            // includes relayouts in registers and movements from level 1 to RF.
            let i = u8::try_from(i).unwrap();
            for layout in Tgt::all_layouts_for_shape(operand.dim_sizes()) {
                for level in Tgt::faster_destination_levels(operand.level()) {
                    let vector_bytes = level.vector_bytes();
                    if vector_bytes > 0 {
                        for vector_shape in gen_vector_shapes(
                            Some(operand.dim_sizes()),
                            operand.dtype(),
                            vector_bytes,
                            None,
                        ) {
                            results.push(self.move_arg(
                                i,
                                level,
                                layout.clone(),
                                Some(&vector_shape),
                                false,
                            ))
                        }
                    } else {
                        results.push(self.move_arg(i, level, layout.clone(), None, false))
                    }
                }
            }
        }

        results.into_iter()
    }

    fn move_arg(
        &self,
        operand_idx: u8,
        dest_level: Tgt::Level,
        dest_layout: Layout,
        vector_shape: Option<&[DimSize]>,
        prefetch: bool,
    ) -> ImplNode<Tgt> {
        ImplNode::MoveLet {
            source_idx: operand_idx,
            destination_level: dest_level,
            destination_layout: dest_layout,
            destination_vector_shape: vector_shape.map(SmallVec::from),
            prefetch,
        }
    }

    /// Produces an ImplNode::Loop from this Spec.
    ///
    /// If the Spec cannot be tiled to that shape, returns None.
    pub fn tile_out(&self, output_shape: &[DimSize], parallel: bool) -> Option<ImplNode<Tgt>> {
        let current_output = self.output();
        let current_out_shape: &Shape = current_output.dim_sizes();

        assert!(
            !(parallel && self.serial_only()),
            "Serial-only Spec prevents parallel tiling"
        );
        assert_eq!(
            output_shape.len(),
            current_out_shape.len(),
            "Expected {} dimensions; got {}",
            current_out_shape.len(),
            output_shape.len()
        );
        assert!(output_shape
            .iter()
            .enumerate()
            .all(|(dim, dim_size)| { *dim_size > 0 && *dim_size <= current_out_shape[dim] }));
        assert_ne!(&current_out_shape[..], output_shape);

        if !current_output.is_valid_tile_shape(output_shape) {
            return None;
        }

        // Tiling happens in three steps:
        // 1. Construct the simple tile corresponding to the new output shape.
        let smaller_output = PartialTile::Simple(output_shape.into())
            .tile(self.output_idx().try_into().unwrap(), &current_output);

        // 2. Construct the *partial* tiles which respect the data deps. of the
        //    new output tile.
        let updated_inputs = self.partial_inputs_for_tile_out(&smaller_output.partial);

        // 3. Reify the partial tiles into Tile objects we'll store with this
        //    ImplNode. Tile objects basically just track the parameter index of
        //    the tensor they tile.
        let mut new_tiles = vec![];
        for (input_idx, (original_input, updated_input)) in
            self.inputs().iter().zip(&updated_inputs).enumerate()
        {
            // Toss out partial tiles with the same TensorSpec as their source,
            // since these weren't affected by the output tiling.
            if !original_input.is_valid_tile_shape(updated_input.dim_sizes()) {
                return None;
            }
            if original_input.dim_sizes() != updated_input.dim_sizes() {
                let new_input_tile =
                    updated_input.tile(input_idx.try_into().unwrap(), original_input);
                new_tiles.push(new_input_tile);
            }
        }
        new_tiles.push(smaller_output);

        Some(ImplNode::Loop {
            subscripts: self
                .operands_dim_subscripts()
                .last()
                .unwrap()
                .clone()
                .to_smallvec(),
            tiles: new_tiles.to_vec(),
            parallel,
        })
    }

    fn split(&self, size: DimSize) -> ImplNode<Tgt> {
        debug_assert_ne!(size, 0);

        // lhs, rhs = self.spec.inputs
        let operands = self.operands();
        let lhs = &operands[0];
        let rhs = &operands[1];
        assert!(size < lhs.dim_sizes()[1]);

        let left_view = PartialTile::Simple(smallvec![lhs.dim_sizes()[0], size]).tile(0, lhs);
        let right_view = PartialTile::Simple(smallvec![size, rhs.dim_sizes()[1]]).tile(1, rhs);

        let split_subscript = *self.operands_dim_subscripts()[0].last().unwrap();

        ImplNode::Loop {
            subscripts: smallvec![split_subscript],
            tiles: vec![left_view, right_view],
            parallel: false,
        }
    }

    fn partial_inputs_for_tile_out(&self, smaller_output: &PartialTile) -> Vec<PartialTile> {
        match (&self, smaller_output) {
            (Spec::Matmul { k, .. }, PartialTile::Simple(dim_sizes)) => {
                let m = dim_sizes[0];
                let n = dim_sizes[1];
                vec![
                    PartialTile::Simple(smallvec![m, *k]),
                    PartialTile::Simple(smallvec![*k, n]),
                ]
            }
            (Spec::Load { .. }, PartialTile::Simple(dim_sizes))
            | (Spec::Store { .. }, PartialTile::Simple(dim_sizes)) => {
                vec![PartialTile::Simple(dim_sizes.clone())]
            }
            (Spec::Zero { .. }, _) => vec![],
            _ => unimplemented!(),
        }
    }

    pub fn operands_dim_subscripts(&self) -> Vec<SmallVec<[u8; 4]>> {
        match &self {
            Spec::Matmul { .. } => vec![smallvec![0, 2], smallvec![2, 1], smallvec![0, 1]],
            Spec::Load { .. } | Spec::Store { .. } | Spec::Zero { .. } => {
                // TODO: Calling self.operands() is slow. Don't do it.
                self.operands()
                    .iter()
                    .map(|o| (0..u8::try_from(o.dim_sizes().len()).unwrap()))
                    .map(|rng| rng.collect())
                    .collect()
            }
        }
    }

    pub fn replace_io(&mut self, new_operands: &[TensorSpec<Tgt>]) {
        match self {
            Spec::Matmul {
                accum: _,
                m,
                k,
                n,
                dtype,
                contiguous_abstractions,
                alignments,
                levels,
                layouts,
                vector_shapes,
                serial_only: _,
            } => {
                *m = new_operands[0].dim_sizes()[0];
                *k = new_operands[0].dim_sizes()[1];
                *n = new_operands[1].dim_sizes()[1];
                *dtype = new_operands[0].dtype();
                *contiguous_abstractions =
                    new_operands.iter().map(|o| o.contiguous_abs()).collect();
                *alignments = new_operands.iter().map(|o| o.aligned()).collect();
                *levels = new_operands.iter().map(|o| o.level()).collect();
                *layouts = new_operands.iter().map(|o| o.layout()).collect();
                *vector_shapes = new_operands
                    .iter()
                    .map(|o| o.vector_shape().cloned())
                    .collect();
            }
            Spec::Load {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only: _,
            }
            | Spec::Store {
                outer_tensor_spec,
                inner_level,
                inner_layout,
                inner_vector_shape,
                serial_only: _,
            } => {
                assert_eq!(new_operands.len(), 2);
                let src = &new_operands[0];
                let dest = &new_operands[1];
                // TODO: Assert that everything else is equal.
                *outer_tensor_spec = src.clone();
                *inner_level = dest.level();
                *inner_layout = dest.layout();
                *inner_vector_shape = dest.vector_shape().cloned();
            }
            Spec::Zero {
                tensor_spec,
                serial_only: _,
            } => {
                *tensor_spec = new_operands[0].clone();
            }
        }
    }

    pub fn output_is_read(&self) -> bool {
        match self {
            Spec::Matmul { accum, .. } => *accum,
            _ => false,
        }
    }
}

impl<Tgt: Target> Display for Spec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = match &self {
            Spec::Matmul { accum, .. } if *accum => "MatmulAccum",
            Spec::Matmul { .. } => "Matmul",
            Spec::Load { .. } => "Load",
            Spec::Store { .. } => "Store",
            Spec::Zero { .. } => "Zero",
        };

        let operand_str = self
            .operands()
            .iter()
            .map(|o| format!("{}", o))
            .collect::<Vec<_>>()
            .join(", ");
        let serial_str = if self.serial_only() { ", serial" } else { "" };

        write!(f, "{}({}{})", header, operand_str, serial_str)
    }
}

// TODO: Modify to return an `impl Iterator` of some kind instead of a `Box`.
fn gen_tile_sizes<Tgt: Target>(
    tensor_shape: &[DimSize],
    drop_given: bool,
) -> Box<dyn Iterator<Item = Shape>> {
    let tensor_shape = tensor_shape.to_vec();

    if tensor_shape.is_empty() {
        Box::new(std::iter::empty())
    } else if tensor_shape.len() == 1 {
        Box::new(dim_range(tensor_shape[0], true).filter_map(move |d| {
            if drop_given && d == tensor_shape[0] {
                return None;
            }
            Some(smallvec![d])
        }))
    } else {
        Box::new(
            gen_tile_sizes::<Tgt>(&tensor_shape[1..], false).flat_map(move |rest| {
                let tensor_shape = tensor_shape.clone();
                dim_range(tensor_shape[0], true).flat_map(move |d| {
                    let mut new_shape = smallvec![d];
                    new_shape.extend(rest.clone());
                    if drop_given && tensor_shape == new_shape[..] {
                        None
                    } else {
                        Some(new_shape)
                    }
                })
            }),
        )
    }
}

pub fn gen_vector_shapes(
    outer_shape: Option<&[DimSize]>,
    dtype: Dtype,
    vector_bytes: u32,
    rank: Option<u8>,
) -> Box<dyn Iterator<Item = Vec<DimSize>>> {
    assert_ne!(
        outer_shape.is_some(),
        rank.is_some(),
        "Must specify either outer_shape or rank, but not both"
    );
    assert!(outer_shape.is_none() || outer_shape.unwrap().iter().all(|&d| d > 0));
    assert_ne!(vector_bytes, 0, "vector_bytes must be greater than 0");
    assert_eq!(
        vector_bytes % u32::from(dtype.size()),
        0,
        "vector_bytes must be a multiple of dtype size"
    );

    let rank = rank.unwrap_or_else(|| outer_shape.unwrap().len().try_into().unwrap());
    let mut adjusted_vector_bytes: u32 = vector_bytes;
    if dtype.size() != 1 {
        adjusted_vector_bytes /= u32::from(dtype.size());
    }
    debug_assert!(adjusted_vector_bytes > 0);

    if LIMIT_VECTORS_TO_ONE_DIM {
        if adjusted_vector_bytes == 1 {
            return Box::new(std::iter::once(vec![1; rank.into()]));
        }
        let outer_shape = outer_shape.map(Vec::from);
        Box::new(
            (0..rank)
                .rev()
                .map(usize::from)
                .filter(move |&i| {
                    outer_shape.is_none()
                        || outer_shape.as_ref().unwrap()[i] >= adjusted_vector_bytes
                })
                .map(move |i| {
                    let mut v = vec![1; rank.into()];
                    v[i] = adjusted_vector_bytes;
                    v
                }),
        )
    } else {
        todo!()
    }
}

pub fn dim_range(dim: DimSize, include_end: bool) -> impl Iterator<Item = DimSize> {
    let it = (0..)
        .map(|power| 2u32.pow(power))
        .take_while(move |x| *x < dim);

    it.chain(
        once(if include_end && dim != 0 {
            Some(dim)
        } else {
            None
        })
        .flatten(),
    )
}
