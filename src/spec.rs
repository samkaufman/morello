use super::common::{DimSize, Shape};
use crate::common::Dtype;
use crate::imp::ImplNode;
use crate::layout::Layout;
use crate::target::MemoryLevel;
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::join_into_string;

use core::panic;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::fmt;
use std::fmt::Display;
use std::iter::Iterator;
use std::iter::{self, once};

use std::{assert_eq, debug_assert_eq};

const LIMIT_VECTORS_TO_ONE_DIM: bool = true;

// The following should probably just be Spec::Primitive and Spec::Compose variants once
// there are good conversions to/from image/filter shapes for Conv.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum Spec<Tgt: Target> {
    // TODO: Enforce `typ` and PrimitiveAux compatibility at the type level.
    Primitive(PrimitiveBasics, PrimitiveAux<Tgt>, bool),
    Compose {
        // Components contain Spec shapes, which can be partially inferred, so
        // the following stores a little bit of redundant information.
        components: Vec<PrimitiveBasics>,
        operand_auxes: Vec<TensorSpecAux<Tgt>>,
        serial_only: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PrimitiveBasics {
    pub typ: PrimitiveSpecType,
    pub spec_shape: Shape,
    pub dtype: Dtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum PrimitiveSpecType {
    Matmul { accum: bool },
    Conv { accum: bool },
    Load,
    Store,
    Zero,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum PrimitiveAux<Tgt: Target> {
    Standard(Vec<TensorSpecAux<Tgt>>),
    Move {
        outer_aux: TensorSpecAux<Tgt>,
        inner_level: Tgt::Level,
        inner_layout: Layout,
        inner_vector_shape: Option<Shape>,
    },
}

impl PrimitiveBasics {
    pub fn aux_from_operand_auxes<'a, Tgt, I>(&self, operand_auxes: I) -> PrimitiveAux<Tgt>
    where
        Tgt: Target,
        I: IntoIterator<Item = &'a TensorSpecAux<Tgt>> + 'a,
    {
        match self.typ {
            PrimitiveSpecType::Matmul { .. }
            | PrimitiveSpecType::Conv { .. }
            | PrimitiveSpecType::Zero => {
                PrimitiveAux::Standard(operand_auxes.into_iter().cloned().collect())
            }
            PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                let mut operand_auxes_iter = operand_auxes.into_iter();
                let first: &TensorSpecAux<_> = operand_auxes_iter.next().unwrap();
                let second: &TensorSpecAux<_> = operand_auxes_iter.next().unwrap();
                PrimitiveAux::Move {
                    outer_aux: first.clone(),
                    inner_level: second.level,
                    inner_layout: second.layout.clone(),
                    inner_vector_shape: second.vector_shape.clone(),
                }
            }
        }
    }

    pub fn operand_shapes(&self) -> Vec<Shape> {
        match self.typ {
            PrimitiveSpecType::Matmul { .. } => {
                let [m, k, n] = self.spec_shape[..] else {
                    panic!("Matmul spec_shape must have length 3")
                };
                vec![smallvec![m, k], smallvec![k, n], smallvec![m, n]]
            }
            PrimitiveSpecType::Conv { .. } => {
                let [b, f, c, h, w, fh, fw] = self.spec_shape[..] else {
                    unreachable!()
                };
                debug_assert!(h >= fh && w >= fw);
                let img = smallvec![b, c, h, w];
                let filt = smallvec![f, c, fh, fw];
                let out = conv_infer_output_shape(&img, &filt);
                vec![img, filt, out]
            }
            PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                vec![self.spec_shape.clone(), self.spec_shape.clone()]
            }
            PrimitiveSpecType::Zero => vec![self.spec_shape.clone()],
        }
    }

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> Vec<Tiling> {
        match (self, smaller_output.is_simple()) {
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { .. },
                    spec_shape,
                    ..
                },
                true,
            ) => vec![
                Tiling::new_sliding(
                    smallvec![smaller_output.shape()[0], spec_shape[1]],
                    smallvec![smaller_output.step_sizes()[0], spec_shape[1]],
                ),
                Tiling::new_sliding(
                    smallvec![spec_shape[1], smaller_output.shape()[1]],
                    smallvec![spec_shape[1], smaller_output.step_sizes()[1]],
                ),
            ],
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Conv { .. },
                    spec_shape,
                    ..
                },
                _,
            ) => {
                let [_, _, channels, _, _, fh, fw] = spec_shape[..] else {
                    unreachable!()
                };

                // Compute the new input image Tiling.
                let new_image_shape: Shape = [smaller_output.shape()[0], channels]
                    .into_iter()
                    .chain(
                        smaller_output.shape()[2..]
                            .iter()
                            .zip([fh, fw])
                            .map(|(&o, f)| o + f - 1),
                    )
                    .collect();
                let mut new_image_steps: Shape = smaller_output.step_sizes().into();
                new_image_steps[1] = channels;

                // Compute the new filters Tiling.
                let new_filters_shape: Shape = [smaller_output.shape()[1], channels]
                    .into_iter()
                    .chain([fh, fw].into_iter())
                    .collect();
                let mut new_filters_steps: Shape = new_filters_shape.clone();
                new_filters_steps[0] = smaller_output.step_sizes()[1];
                vec![
                    Tiling::new_sliding(new_image_shape, new_image_steps),
                    Tiling::new_sliding(new_filters_shape, new_filters_steps),
                ]
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Load,
                    ..
                },
                true,
            )
            | (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Store,
                    ..
                },
                true,
            ) => vec![smaller_output.clone()],
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    ..
                },
                true,
            ) => vec![],
            _ => unimplemented!(
                "Output tiling not implemented for {:?} and {:?}",
                self,
                smaller_output
            ),
        }
    }
}

impl Display for PrimitiveBasics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str = join_into_string(&self.spec_shape, "×");
        write!(f, "{}({}, {})", self.typ, shape_str, self.dtype)
    }
}

impl PrimitiveSpecType {
    pub fn operand_count(&self) -> usize {
        self.input_count() + 1
    }

    pub fn input_count(&self) -> usize {
        match self {
            PrimitiveSpecType::Matmul { .. } => 2,
            PrimitiveSpecType::Conv { .. } => 2,
            PrimitiveSpecType::Load => 1,
            PrimitiveSpecType::Store => 1,
            PrimitiveSpecType::Zero => 0,
        }
    }

    pub fn output_idx(&self) -> usize {
        match self {
            PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => 2,
            PrimitiveSpecType::Load { .. } | PrimitiveSpecType::Store { .. } => 1,
            PrimitiveSpecType::Zero { .. } => 0,
        }
    }

    pub fn output_is_read(&self) -> bool {
        match self {
            PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum } => *accum,
            _ => false,
        }
    }

    // TODO: Wrap output in a TensorSpecBasics struct.
    pub fn infer_output_shape(&self, inputs: &[&[DimSize]]) -> Shape {
        // TODO: Can this be rewritten as output inference + `from_io` call?
        debug_assert_eq!(inputs.len(), self.operand_count());
        match self {
            PrimitiveSpecType::Matmul { .. } => {
                let ([m, _k], [_, n]) = (inputs[0], inputs[1]) else {
                    panic!("Matmul inputs must have 2 dimensions each");
                };
                smallvec![*m, *n]
            }
            PrimitiveSpecType::Conv { .. } => {
                let ([b, _, h, w], [f, _, fh, fw]) = (inputs[0], inputs[1]) else {
                    panic!("Conv inputs must have 4 dimensions each");
                };
                debug_assert!(h >= fh && w >= fw);
                smallvec![*b, *f, 1 + h - fh, 1 + w - fw]
            }
            PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                // The shape and dtype match for moves.
                inputs[0].to_smallvec()
            }
            PrimitiveSpecType::Zero => todo!(),
        }
    }
}

impl Display for PrimitiveSpecType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveSpecType::Matmul { accum, .. } if *accum => write!(f, "MatmulAccum"),
            PrimitiveSpecType::Matmul { .. } => write!(f, "Matmul"),
            PrimitiveSpecType::Conv { accum, .. } if *accum => write!(f, "ConvAccum"),
            PrimitiveSpecType::Conv { .. } => write!(f, "Conv"),
            PrimitiveSpecType::Load { .. } => write!(f, "Load"),
            PrimitiveSpecType::Store { .. } => write!(f, "Store"),
            PrimitiveSpecType::Zero { .. } => write!(f, "Zero"),
        }
    }
}

impl<Tgt: Target> Spec<Tgt> {
    // TODO: Do we really need this?
    pub fn primitive_type(&self) -> PrimitiveSpecType {
        match self {
            Spec::Primitive(basics, _, _) => basics.typ,
            Spec::Compose { .. } => panic!("Spec::Compose has no primitive type"),
        }
    }

    pub fn serial_only(&self) -> bool {
        match self {
            Spec::Primitive(_, _, serial_only) => *serial_only,
            Spec::Compose { serial_only, .. } => *serial_only,
        }
    }

    pub fn operand_count(&self) -> usize {
        match self {
            Spec::Compose { components, .. } => {
                let (innermost_component, outer_components) = components.split_last().unwrap();
                let mut cnt = innermost_component.typ.operand_count();
                cnt += outer_components
                    .iter()
                    .map(|p| p.typ.operand_count() - 2)
                    .sum::<usize>();
                cnt
            }
            Spec::Primitive(basics, _, _) => basics.typ.operand_count(),
        }
    }

    pub fn operands(&self) -> Vec<TensorSpec<Tgt>> {
        match self {
            Spec::Primitive(basics, aux, _) => match basics.typ {
                PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => {
                    let PrimitiveAux::Standard(taux) = aux else {
                        unreachable!();
                    };
                    basics
                        .operand_shapes()
                        .into_iter()
                        .zip(taux)
                        .map(|(s, a)| TensorSpec::new_noncanon_with_aux(s, basics.dtype, a.clone()))
                        .collect()
                }
                PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                    let PrimitiveAux::Move { outer_aux, inner_level, inner_layout, inner_vector_shape } = aux else {
                        unreachable!();
                    };
                    let outer_tensor_spec = TensorSpec::new_noncanon_with_aux(
                        basics.spec_shape.clone(),
                        basics.dtype,
                        outer_aux.clone(),
                    );
                    let mut inner_tensor_spec = outer_tensor_spec.clone();
                    inner_tensor_spec.set_level(*inner_level, inner_vector_shape.to_owned());
                    inner_tensor_spec.set_layout(inner_layout.to_owned());
                    inner_tensor_spec.canonicalize();
                    vec![outer_tensor_spec, inner_tensor_spec]
                }
                PrimitiveSpecType::Zero => {
                    let PrimitiveAux::Standard(taux) = aux else {
                        unreachable!();
                    };
                    vec![TensorSpec::new_noncanon_with_aux(
                        basics.spec_shape.clone(),
                        basics.dtype,
                        taux[0].clone(),
                    )]
                }
            },
            Spec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                let mut result_basics = Vec::with_capacity(self.operand_count());
                let mut last_seen_output = None;
                for (i, c) in components.iter().rev().enumerate() {
                    let mut operand_basics: Vec<(Shape, Dtype)> = c
                        .operand_shapes()
                        .into_iter()
                        .zip(iter::repeat(c.dtype))
                        .collect::<Vec<_>>();
                    last_seen_output = operand_basics.pop();
                    debug_assert!(last_seen_output.is_some());
                    operand_basics.reverse();
                    if i != 0 {
                        operand_basics.pop();
                    }
                    result_basics.append(&mut operand_basics);
                }
                result_basics.reverse();
                result_basics.push(last_seen_output.unwrap());
                debug_assert_eq!(result_basics.len(), operand_auxes.len());
                result_basics
                    .into_iter()
                    .zip(operand_auxes)
                    .map(|((s, d), a)| TensorSpec::new_noncanon_with_aux(s, d, a.clone()))
                    .collect()
            }
        }
    }

    pub fn inputs(&self) -> Vec<TensorSpec<Tgt>> {
        // TODO: Can this be unified across variants? Always drop last?
        match &self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
                PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => {
                    self.operands()[..2].to_vec()
                }
                PrimitiveSpecType::Load { .. } | PrimitiveSpecType::Store { .. } => {
                    // TODO: Just grab the item instead of calling operands
                    vec![self.operands()[0].clone()]
                }
                PrimitiveSpecType::Zero { .. } => vec![],
            },
            Spec::Compose { .. } => {
                let mut ops = self.operands();
                ops.pop();
                ops
            }
        }
    }

    pub fn output(&self) -> TensorSpec<Tgt> {
        self.operands()[self.output_idx()].clone()
    }

    pub fn output_idx(&self) -> usize {
        match &self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => typ.output_idx(),
            Spec::Compose { .. } => self.operand_count() - 1,
        }
    }

    pub fn canonicalize(&mut self) {
        // TODO: This is expensive. Make an operand_shapes() method instead.
        let operands = self.operands();

        match self {
            Spec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape,
                    dtype: _,
                },
                primitive_aux,
                _,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum: _ } | PrimitiveSpecType::Conv { accum: _ } => {
                    let PrimitiveAux::Standard(aux) = primitive_aux else {
                        unreachable!();
                    };
                    for i in 0..aux.len() {
                        aux[i].contig = aux[i].layout.tile_contiguity(
                            operands[i].dim_sizes(),
                            operands[i].dim_sizes(),
                            aux[i].contig,
                        );
                        aux[i].layout = aux[i]
                            .layout
                            .canonicalize_for_shape(operands[i].dim_sizes());
                    }
                }
                PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                    let PrimitiveAux::Move {
                        outer_aux,
                        inner_level: _,
                        inner_layout,
                        inner_vector_shape: _,
                    } = primitive_aux else {
                        unreachable!();
                    };
                    outer_aux.canonicalize(spec_shape, outer_aux.aligned);
                    inner_layout.canonicalize_for_shape(spec_shape);
                }
                PrimitiveSpecType::Zero => {
                    let PrimitiveAux::Standard(aux) = primitive_aux else {
                        unreachable!();
                    };
                    let aligned = aux[0].aligned;
                    aux[0].canonicalize(spec_shape, aligned);
                }
            },
            Spec::Compose { .. } => todo!(),
        }

        // TODO: What if you want to call `operands` on a non-canon Spec?
        debug_assert_eq!(
            self.operands(),
            self.operands()
                .iter()
                .map(|o| {
                    let mut o = o.clone();
                    o.canonicalize();
                    o
                })
                .collect::<Vec<_>>()
        );
    }

    pub fn is_canonical(&self) -> bool {
        // TODO: Probably slow.
        let mut c = self.clone();
        c.canonicalize();
        self == &c
    }

    pub fn expansions(&self) -> Box<dyn Iterator<Item = ImplNode<Tgt>> + '_> {
        let iter = self.tile_out_expansions();
        let iter = iter.chain(self.move_expansions());
        let iter = iter.chain(Tgt::expansions(self));

        match &self {
            Spec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape: _,
                    dtype: _,
                },
                _primitive_aux,
                _serial_only,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum } if !*accum => {
                    Box::new(iter.chain(iter::once(ImplNode::AccumBlock)))
                }
                PrimitiveSpecType::Matmul { accum } if *accum => {
                    Box::new(iter.chain(self.split_expansions()))
                }
                PrimitiveSpecType::Conv { accum } => {
                    if *accum {
                        if self.can_spatial_split() {
                            Box::new(iter.chain(iter::once(ImplNode::SpatialSplit)))
                        } else {
                            Box::new(iter)
                        }
                    } else {
                        Box::new(iter.chain(iter::once(ImplNode::AccumBlock)))
                    }
                }
                _ => Box::new(iter),
            },
            Spec::Compose {
                components: _,
                operand_auxes: _,
                serial_only: _,
            } => {
                // TODO: Add head reduce split expansions as well.
                Box::new(iter.chain(self.peel_expansions()))
            }
        }
    }

    fn can_spatial_split(&self) -> bool {
        let Spec::Primitive(PrimitiveBasics { typ, .. }, primitive_aux, _) = self else {
            panic!("can_spatial_split called on non-Primitive spec");
        };
        let PrimitiveSpecType::Conv { .. } = typ else {
            panic!("can_spatial_split called on non-Conv spec");
        };
        let PrimitiveAux::Standard(aux) = primitive_aux else {
            unreachable!();
        };

        let operands = self.operands();
        let image_shape = operands[0].dim_sizes();
        let filters_shape = operands[1].dim_sizes();

        if image_shape[2..] != filters_shape[2..] {
            return false;
        }
        for a in aux {
            if let Some(vector_shape) = &a.vector_shape {
                if vector_shape[2..].iter().any(|&d| d != 1) {
                    return false;
                }
            }
        }
        true
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
        let Spec::Primitive(PrimitiveBasics { typ, spec_shape, .. }, _, _) = self else {
            panic!("split_expansions called on non-primitive Spec");
        };
        let PrimitiveSpecType::Matmul { accum } = typ else {
            panic!("split_expansions called on non-Matmul");
        };
        if !accum {
            panic!("split_expansions called on non-accumulating Matmul");
        }
        let k = spec_shape[1];
        Box::new(
            dim_range(k, false)
                .filter(|&new_k| self.split_valid(new_k))
                .map(|k| self.split(k)),
        )
    }

    fn peel_expansions(&self) -> Box<dyn Iterator<Item = ImplNode<Tgt>> + '_> {
        let Spec::Compose { components, operand_auxes: _, serial_only: _ } = self else {
            panic!("peel_expansions called on non-Compose Spec");
        };

        let mut results = vec![];

        let o = components[1].operand_shapes();
        let intermediate_shape = &o[components[1].typ.output_idx()];

        for layout in Tgt::all_layouts_for_shape(intermediate_shape) {
            for level in Tgt::levels() {
                // TODO: Need to implement `can_move_to`-style logic here.

                let vector_bytes = level.vector_bytes();
                if vector_bytes > 0 {
                    for vector_shape in gen_vector_shapes(
                        Some(intermediate_shape),
                        components[1].dtype,
                        vector_bytes,
                        None,
                    ) {
                        results.push(self.peel(layout.clone(), level, Some(vector_shape)));
                    }
                } else {
                    results.push(self.peel(layout.clone(), level, None));
                }
            }
        }

        Box::new(results.into_iter())
    }

    fn split_valid(&self, new_k: DimSize) -> bool {
        let Spec::Primitive(PrimitiveBasics { typ, spec_shape, dtype: _ }, _, _) = self else {
            panic!();
        };
        debug_assert!(matches!(typ, PrimitiveSpecType::Matmul { .. }));

        let [m, orig_k, n] = spec_shape[..] else {
            unreachable!();
        };

        // Special-case for splitting to single-element tensors, which will be normalized
        // to row-major. This is necessary for splits in any other layout to be
        // discovered by search.
        // TODO: This is pretty ad-hoc. Should there be an alternative to
        //   `is_valid_tile_shape` that includes this case?
        if m == 1 && new_k == 1 && n == 1 {
            return true;
        }

        let operands = self.operands();
        if new_k >= orig_k || !operands[0].is_valid_tile_shape(&[m, new_k]) {
            false
        } else {
            operands[1].is_valid_tile_shape(&[new_k, n])
        }
    }

    fn move_expansions(&self) -> impl Iterator<Item = ImplNode<Tgt>> + '_ {
        // TODO: Add prefetching moves.

        let mut results = vec![]; // TODO: Don't accumulate. Return an iterator.
        if matches!(self, Spec::Primitive(_, _, _))
            && matches!(
                self.primitive_type(),
                PrimitiveSpecType::Load | PrimitiveSpecType::Store
            )
        {
            return results.into_iter();
        }

        for (i, operand) in self.operands().iter().enumerate() {
            // Yield actions for movement with register file destination, which
            // includes relayouts in registers and movements from level 1 to RF.
            let i = u8::try_from(i).unwrap();
            for layout in Tgt::all_layouts_for_shape(operand.dim_sizes()) {
                for level in Tgt::faster_destination_levels(operand.level()) {
                    if !operand.can_move_to(&layout, &level) {
                        continue;
                    }

                    let vector_bytes = level.vector_bytes();
                    if vector_bytes > 0 {
                        for vector_shape in gen_vector_shapes(
                            Some(operand.dim_sizes()),
                            operand.dtype(),
                            vector_bytes,
                            None,
                        ) {
                            results.push({
                                let _this = self;
                                let dest_layout = layout.clone();
                                let vector_shape = Some(&vector_shape);
                                ImplNode::MoveLet {
                                    source_idx: i,
                                    destination_level: level,
                                    destination_layout: dest_layout,
                                    destination_vector_shape: vector_shape.cloned(),
                                    prefetch: false,
                                }
                            });
                        }
                    } else {
                        results.push({
                            let _this = self;
                            let dest_layout = layout.clone();
                            ImplNode::MoveLet {
                                source_idx: i,
                                destination_level: level,
                                destination_layout: dest_layout,
                                destination_vector_shape: None,
                                prefetch: false,
                            }
                        });
                    }
                }
            }
        }

        results.into_iter()
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
        let smaller_output = Tiling::new_simple(output_shape.into())
            .into_operand_tile(self.output_idx().try_into().unwrap(), &current_output);

        // 2. Construct tilings which respect the data deps. of the new output tile.
        let updated_inputs = self.input_tilings_for_tile_out(&smaller_output.tiling);

        // 3. Reify the tilings into OperandTiles we'll store with this ImplNode.
        //    OperandTile objects basically just track the parameter index of the tensor
        //    they tile.
        let mut new_tiles = vec![];
        for (input_idx, (original_input, updated_input)) in
            self.inputs().iter().zip(updated_inputs).enumerate()
        {
            // Toss out partial tiles with the same TensorSpec as their source,
            // since these weren't affected by the output tiling.
            if !original_input.is_valid_tile_shape(updated_input.shape()) {
                return None;
            }
            if original_input.dim_sizes() != updated_input.shape() {
                let new_input_tile =
                    updated_input.into_operand_tile(input_idx.try_into().unwrap(), original_input);
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

        match self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
                PrimitiveSpecType::Matmul { accum: _ } => {
                    let operands = self.operands();
                    let lhs = &operands[0];
                    let rhs = &operands[1];
                    assert!(size < lhs.dim_sizes()[1]);

                    let left_view = Tiling::new_simple(smallvec![lhs.dim_sizes()[0], size])
                        .into_operand_tile(0, lhs);
                    let right_view = Tiling::new_simple(smallvec![size, rhs.dim_sizes()[1]])
                        .into_operand_tile(1, rhs);

                    let split_subscript = *self.operands_dim_subscripts()[0].last().unwrap();

                    ImplNode::Loop {
                        subscripts: smallvec![split_subscript],
                        tiles: vec![left_view, right_view],
                        parallel: false,
                    }
                }
                _ => unimplemented!(),
            },
            Spec::Compose { .. } => todo!(),
        }
    }

    fn peel(
        &self,
        layout: Layout,
        level: Tgt::Level,
        vector_shape: Option<Shape>,
    ) -> ImplNode<Tgt> {
        ImplNode::Pipeline {
            layout,
            level,
            vector_shape,
        }
    }

    fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> Vec<Tiling> {
        match self {
            Spec::Primitive(basics, _, _) => basics.input_tilings_for_tile_out(smaller_output),
            Spec::Compose { components, .. } => {
                let mut accumulated_input_tilings = Vec::with_capacity(self.operand_count() - 1);
                let mut last_output_tiling = smaller_output.clone();
                for (i, subspec) in components.iter().enumerate().rev() {
                    let mut subspec_input_tilings =
                        subspec.input_tilings_for_tile_out(&last_output_tiling);
                    debug_assert!(
                        !subspec_input_tilings.is_empty(),
                        "Compose contains {:?}, which has no inputs",
                        subspec
                    );
                    if i == 0 {
                        accumulated_input_tilings.extend(subspec_input_tilings);
                    } else {
                        accumulated_input_tilings.extend(subspec_input_tilings.drain(1..));
                        last_output_tiling = subspec_input_tilings.remove(0);
                    }
                }
                accumulated_input_tilings
            }
        }
    }

    // TODO: Can we replace this entirely with Spec shapes?
    pub fn operands_dim_subscripts(&self) -> Vec<SmallVec<[u8; 4]>> {
        match self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => match typ {
                PrimitiveSpecType::Matmul { .. } => {
                    vec![smallvec![0, 2], smallvec![2, 1], smallvec![0, 1]]
                }
                PrimitiveSpecType::Conv { .. } => {
                    // Only supports 2 spatial dimensions.
                    // TODO: Extend this to arbitrary number of spatial dimensions.
                    let (b, f, c, h, w, fh, fw) = (0, 1, 2, 3, 4, 5, 6);
                    let img = smallvec![b, c, h, w];
                    let filt = smallvec![f, c, fh, fw];
                    let out = smallvec![b, f, h, w];
                    vec![img, filt, out]
                }
                PrimitiveSpecType::Load { .. }
                | PrimitiveSpecType::Store { .. }
                | PrimitiveSpecType::Zero { .. } => {
                    // TODO: Calling self.operands() is slow. Don't do it.
                    self.operands()
                        .iter()
                        .map(|o| (0..u8::try_from(o.dim_sizes().len()).unwrap()))
                        .map(|rng| rng.collect())
                        .collect()
                }
            },
            Spec::Compose { .. } => todo!(),
        }
    }

    // TODO: Should move new_operands in.
    pub fn replace_io(&mut self, new_operands: &[TensorSpec<Tgt>]) {
        assert_eq!(new_operands.len(), self.operand_count());
        match self {
            Spec::Compose { operand_auxes, .. } => {
                *operand_auxes = new_operands.iter().map(|t| t.aux.clone()).collect();
            }
            Spec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape,
                    dtype,
                },
                primitive_aux,
                _,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum: _ } => {
                    debug_assert_eq!(new_operands.len(), 3);
                    debug_assert_eq!(
                        new_operands[0].dim_sizes()[0],
                        new_operands[2].dim_sizes()[0]
                    );
                    debug_assert_eq!(
                        new_operands[1].dim_sizes()[1],
                        new_operands[2].dim_sizes()[1]
                    );
                    debug_assert_eq!(
                        new_operands[0].dim_sizes()[1],
                        new_operands[1].dim_sizes()[0]
                    );
                    let PrimitiveAux::Standard(aux) = primitive_aux else {
                        unreachable!();
                    };
                    *spec_shape = smallvec![
                        new_operands[0].dim_sizes()[0],
                        new_operands[0].dim_sizes()[1],
                        new_operands[1].dim_sizes()[1],
                    ];
                    *dtype = new_operands[0].dtype();
                    for i in 0..aux.len() {
                        let _o = &new_operands[i];
                        aux[i] = new_operands[i].aux.clone();
                    }
                }
                PrimitiveSpecType::Conv { accum: _ } => {
                    let PrimitiveAux::Standard(aux) = primitive_aux else {
                        unreachable!();
                    };
                    assert_eq!(*dtype, new_operands[1].dtype());
                    let [b, c, h, w] = new_operands[0].dim_sizes()[..] else { panic!(); };
                    let [f, alt_c, fh, fw] = new_operands[1].dim_sizes()[..] else { panic!() };
                    assert_eq!(c, alt_c);
                    *spec_shape = smallvec![b, f, c, h, w, fh, fw];
                    *dtype = new_operands[0].dtype();
                    assert_eq!(*dtype, new_operands[1].dtype());
                    assert_eq!(*dtype, new_operands[2].dtype());
                    // TODO: Assert output shape is expected.
                    for i in 0..aux.len() {
                        let o = &new_operands[i];
                        aux[i] = TensorSpecAux {
                            contig: o.contiguous_abs(),
                            aligned: o.aligned(),
                            level: o.level(),
                            layout: o.layout(),
                            vector_shape: o.vector_shape().cloned(),
                        };
                    }
                }
                PrimitiveSpecType::Load | PrimitiveSpecType::Store => {
                    let PrimitiveAux::Move { outer_aux, inner_level, inner_layout, inner_vector_shape } = primitive_aux else {
                        unreachable!();
                    };
                    let [src, dest] = new_operands else {
                        panic!("Load/Store must have 2 operands");
                    };

                    assert_eq!(src.dim_sizes(), dest.dim_sizes());
                    assert_eq!(src.dtype(), dest.dtype());
                    *spec_shape = src.dim_sizes().clone();
                    *dtype = src.dtype();
                    *outer_aux = src.aux.clone();
                    *inner_level = dest.level();
                    *inner_layout = dest.layout();
                    *inner_vector_shape = dest.vector_shape().cloned();
                }
                PrimitiveSpecType::Zero => {
                    let PrimitiveAux::Standard(aux) = primitive_aux else {
                        unreachable!();
                    };
                    let [o] = new_operands else {
                        panic!();
                    };
                    *spec_shape = o.dim_sizes().clone();
                    *dtype = o.dtype();
                    aux[0] = o.aux.clone(); // TODO: Does this move!?
                }
            },
        }
    }

    pub fn output_is_read(&self) -> bool {
        match self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => typ.output_is_read(),
            Spec::Compose { components, .. } => components[0].typ.output_is_read(),
        }
    }

    pub fn clone_as_accum(&self) -> Self {
        let mut cloned = self.clone();
        match &mut cloned {
            Spec::Primitive(basics, _, _) => match &mut basics.typ {
                PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum } => {
                    *accum = true;
                }
                _ => panic!("Cannot clone_as_accum for {:?}", self),
            },
            Spec::Compose { .. } => todo!("Compose can accumulate if head can."),
        }
        cloned
    }
}

impl<Tgt: Target> Display for Spec<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Spec::Compose {
            components,
            operand_auxes: _,
            serial_only,
        } = self
        {
            let operands = self.operands();
            let (output, external_inputs) = operands.split_last().unwrap();
            debug_assert_eq!(self.output_idx(), external_inputs.len());
            return write!(
                f,
                "Compose(({}), [{}, out={}], ({}){})",
                join_into_string(components.iter().map(|c| c.typ), ", "),
                join_into_string(external_inputs, ", "),
                output.to_string(),
                join_into_string(components.iter().map(|c| c.dtype), ", "),
                if *serial_only { ", serial" } else { "" }
            );
        }

        let header = match self {
            Spec::Primitive(PrimitiveBasics { typ, .. }, _, _) => format!("{}", typ),
            Spec::Compose { .. } => todo!(),
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

impl<Tgt: Target> TensorSpecAux<Tgt> {
    fn make_tensorspec_noncanon(&self, dim_sizes: Shape, dtype: Dtype) -> TensorSpec<Tgt> {
        TensorSpec::new_noncanon(
            dim_sizes,
            dtype,
            self.contig,
            self.aligned,
            self.level,
            self.layout.clone(),
            self.vector_shape.clone(),
        )
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
) -> Box<dyn Iterator<Item = Shape>> {
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
            return Box::new(std::iter::once(smallvec![1; rank.into()]));
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
                    let mut v = smallvec![1; rank.into()];
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

// TODO: Drop in favor of primary output shape inference.
pub fn conv_infer_output_shape(image_shape: &[u32], filters_shape: &[u32]) -> Shape {
    let batch_cnt = image_shape[0];
    let channels = image_shape[1];
    let filter_cnt = filters_shape[0];
    // TODO: We don't need to store this dimension twice.
    assert_eq!(
        channels, filters_shape[1],
        "Image had {} channels and filters had {}",
        channels, filters_shape[1]
    );
    vec![batch_cnt, filter_cnt]
        .into_iter()
        .chain(image_shape[2..].iter().zip(filters_shape[2..].iter()).map(
            |(&img_dim, &filt_dim)| {
                assert!(
                    img_dim >= filt_dim,
                    "Image dimension {} was smaller than filter dimension {}",
                    img_dim,
                    filt_dim
                );
                img_dim - filt_dim + 1
            },
        ))
        .collect()
}
