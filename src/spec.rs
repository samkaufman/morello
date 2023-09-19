use super::common::{DimSize, Shape};
use crate::action_seq::ActionSeq;
use crate::common::Dtype;
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::memorylimits::{MemoryLimits, MemoryLimitsBimap};
use crate::scheduling::Action;
use crate::target::MemoryLevel;
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::join_into_string;
use crate::utils::{bit_length_u32, is_power_of_two_u32};

use itertools::Either;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::iter::Iterator;
use std::iter::{self, once};
use std::mem;
use std::panic;
use std::{assert_eq, debug_assert_eq};

/// Whether `tile_out` actions should tile in all dimensions per Spec.
const MULTI_DIM_TILING: bool = false;

/// An empirically chosen initial capacity for the [LogicalSpec::move_actions] results buffer.
const MOVE_RESULTS_CAPACITY: usize = 12;

const LOG_SCALE_SHAPES_BIMAP: bool = true;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct Spec<Tgt: Target>(pub LogicalSpec<Tgt>, pub MemoryLimits);

// The following should probably just be Spec::Primitive and Spec::Compose variants once
// there are good conversions to/from image/filter shapes for Conv.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LogicalSpec<Tgt: Target> {
    Primitive(PrimitiveBasics, Vec<TensorSpecAux<Tgt>>, bool),
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
    Move,
    Zero,
}

/// Tilings and dimension bindings for a particular output tiling.
///
/// Each dimension of an input tensor/tiling may have a binding to an output
/// tensor dimension. This means that loops should zip those dimensions of each
/// tensor to ensure data dependencies are correct. As an example, a matrix
/// multiplicaton will give the bindings `smallvec![Some(0), None]` and
/// `smallvec![None, Some(1)]` for each of its inputs, indicating that the first
/// dimension of the first input (the m dimension) is bound to the m dimension
/// of the output, and so on for the n dimension.
pub struct TilingInference(pub Vec<(Tiling, SmallVec<[Option<u8>; 5]>)>);

/// A [BiMap] which extends [LogicalSpecBimap] with memory limits dimensions.
///
/// Memory limits are represented identically in the codomain. They are not scaled logarithmically
/// or inverted to be in data dependency order.
#[derive(Default)]
pub struct SpecBimap<Tgt, A>
where
    Tgt: Target,
    A: BiMap<Domain = TensorSpecAux<Tgt>>,
{
    pub logical_spec_bimap: LogicalSpecBimap<Tgt, A>,
    pub memory_limits_bimap: MemoryLimitsBimap<Tgt>,
}

#[derive(Default)]
pub struct LogicalSpecBimap<Tgt, A>
where
    Tgt: Target,
    A: BiMap<Domain = TensorSpecAux<Tgt>>,
{
    pub aux_bimap: A,
}

pub struct PrimitiveBasicsBimap;

impl<Tgt: Target> Display for Spec<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

impl PrimitiveBasics {
    pub fn replace_io(&mut self, new_operands: &[(&[DimSize], Dtype)]) {
        match self.typ {
            PrimitiveSpecType::Matmul { accum: _ } => {
                debug_assert_eq!(new_operands.len(), 3);
                debug_assert_eq!(new_operands[0].0[0], new_operands[2].0[0]);
                debug_assert_eq!(new_operands[1].0[1], new_operands[2].0[1]);
                debug_assert_eq!(new_operands[0].0[1], new_operands[1].0[0]);
                self.spec_shape = smallvec![
                    new_operands[0].0[0],
                    new_operands[0].0[1],
                    new_operands[1].0[1],
                ];
                self.dtype = new_operands[0].1;
            }
            PrimitiveSpecType::Conv { accum: _ } => {
                assert_eq!(self.dtype, new_operands[0].1);
                let [b, c, h, w] = new_operands[0].0[..] else {
                    panic!();
                };
                let [f, alt_c, fh, fw] = new_operands[1].0[..] else {
                    panic!()
                };
                assert_eq!(c, alt_c);
                self.spec_shape = smallvec![b, f, c, h, w, fh, fw];
                self.dtype = new_operands[0].1;
                assert_eq!(self.dtype, new_operands[1].1);
                assert_eq!(self.dtype, new_operands[2].1);
                // TODO: Assert output shape is expected.
            }
            PrimitiveSpecType::Move => {
                let [src, dest] = new_operands else {
                    panic!("Move must have 2 operands");
                };
                assert_eq!(src, dest);
                self.spec_shape = src.0.into();
                self.dtype = src.1;
            }
            PrimitiveSpecType::Zero => {
                assert_eq!(new_operands.len(), 1);
                self.spec_shape = new_operands[0].0.into();
                self.dtype = new_operands[0].1;
            }
        }
    }

    pub fn aux_from_operand_auxes<'a, Tgt, I>(&self, operand_auxes: I) -> Vec<TensorSpecAux<Tgt>>
    where
        Tgt: Target,
        I: IntoIterator<Item = &'a TensorSpecAux<Tgt>> + 'a,
    {
        operand_auxes.into_iter().cloned().collect()
    }

    pub fn parameter_shapes(&self) -> SmallVec<[Shape; 3]> {
        match self.typ {
            PrimitiveSpecType::Matmul { .. } => {
                let [m, k, n] = self.spec_shape[..] else {
                    panic!("Matmul spec_shape must have length 3")
                };
                smallvec![smallvec![m, k], smallvec![k, n], smallvec![m, n]]
            }
            PrimitiveSpecType::Conv { .. } => {
                let [b, f, c, h, w, fh, fw] = self.spec_shape[..] else {
                    panic!("Conv must have rank 7")
                };
                debug_assert!(
                    h >= fh && w >= fw,
                    "Conv spatial dims. {}x{} were larger than filter {}x{}",
                    h,
                    w,
                    fh,
                    fw
                );
                let img = smallvec![b, c, h, w];
                let filt = smallvec![f, c, fh, fw];
                let out = conv_infer_output_shape(&img, &filt);
                smallvec![img, filt, out]
            }
            PrimitiveSpecType::Move => {
                smallvec![self.spec_shape.clone(), self.spec_shape.clone()]
            }
            PrimitiveSpecType::Zero => smallvec![self.spec_shape.clone()],
        }
    }

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> TilingInference {
        match (self, smaller_output.is_simple()) {
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { .. },
                    spec_shape,
                    ..
                },
                true,
            ) => TilingInference(vec![
                (
                    Tiling::new_sliding(
                        smallvec![smaller_output.shape()[0], spec_shape[1]],
                        smallvec![smaller_output.step_sizes()[0], spec_shape[1]],
                    ),
                    smallvec![Some(0), None],
                ),
                (
                    Tiling::new_sliding(
                        smallvec![spec_shape[1], smaller_output.shape()[1]],
                        smallvec![spec_shape[1], smaller_output.step_sizes()[1]],
                    ),
                    smallvec![None, Some(1)],
                ),
            ]),
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

                // Construct the bindings Vecs.
                let image_bindings = smallvec![Some(0), None, None, None];
                let filter_bindings = smallvec![None, Some(1), None, None];

                TilingInference(vec![
                    (
                        Tiling::new_sliding(new_image_shape, new_image_steps),
                        image_bindings,
                    ),
                    (
                        Tiling::new_sliding(new_filters_shape, new_filters_steps),
                        filter_bindings,
                    ),
                ])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    ..
                },
                true,
            ) => TilingInference(vec![(
                smaller_output.clone(),
                (0..smaller_output.shape().len())
                    .map(|d| Some(d.try_into().unwrap()))
                    .collect(),
            )]),
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Zero,
                    ..
                },
                true,
            ) => TilingInference(vec![]),
            _ => unimplemented!(
                "Output tiling not implemented for {:?} and {:?}",
                self,
                smaller_output
            ),
        }
    }

    pub fn parameter_dim_axes(&self) -> Vec<SmallVec<[u8; 4]>> {
        match self.typ {
            PrimitiveSpecType::Matmul { .. } => {
                vec![smallvec![0, 2], smallvec![2, 1], smallvec![0, 1]]
            }
            PrimitiveSpecType::Conv { .. } => {
                // Only correct for 2 spatial dimensions.
                // TODO: Extend this to arbitrary number of spatial dimensions.
                let (b, f, c, h, w, fh, fw) = (0, 1, 2, 3, 4, 5, 6);
                let img = smallvec![b, c, h, w];
                let filt = smallvec![f, c, fh, fw];
                let out = smallvec![b, f, h, w];
                vec![img, filt, out]
            }
            PrimitiveSpecType::Move { .. } | PrimitiveSpecType::Zero { .. } => self
                .parameter_shapes()
                .iter()
                .map(|o| (0..u8::try_from(o.len()).unwrap()).collect())
                .collect(),
        }
    }
}

impl Display for PrimitiveBasics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str = join_into_string(&self.spec_shape, "×");
        write!(f, "{}({}, {})", self.typ, shape_str, self.dtype)
    }
}

impl CanonicalBimap for PrimitiveBasics {
    type Bimap = PrimitiveBasicsBimap;

    fn bimap() -> Self::Bimap {
        PrimitiveBasicsBimap
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
            PrimitiveSpecType::Move => 1,
            PrimitiveSpecType::Zero => 0,
        }
    }

    pub fn output_idx(&self) -> usize {
        match self {
            PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => 2,
            PrimitiveSpecType::Move { .. } => 1,
            PrimitiveSpecType::Zero { .. } => 0,
        }
    }

    pub fn output_is_read(&self) -> bool {
        match self {
            PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum } => *accum,
            _ => false,
        }
    }

    pub fn infer_output_shape(&self, inputs: &[&[DimSize]]) -> Shape {
        // TODO: Can this be rewritten as output inference + `from_io` call?
        debug_assert_eq!(inputs.len(), self.input_count());
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
            PrimitiveSpecType::Move | PrimitiveSpecType::Zero => {
                // The shape and dtype match for moves and zero.
                inputs[0].to_smallvec()
            }
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
            PrimitiveSpecType::Move { .. } => write!(f, "Move"),
            PrimitiveSpecType::Zero { .. } => write!(f, "Zero"),
        }
    }
}

impl<Tgt: Target> LogicalSpec<Tgt> {
    pub fn serial_only(&self) -> bool {
        match self {
            LogicalSpec::Primitive(_, _, serial_only) => *serial_only,
            LogicalSpec::Compose { serial_only, .. } => *serial_only,
        }
    }

    pub fn operand_count(&self) -> usize {
        match self {
            LogicalSpec::Compose { components, .. } => {
                let (innermost_component, outer_components) = components.split_last().unwrap();
                let mut cnt = innermost_component.typ.operand_count();
                cnt += outer_components
                    .iter()
                    .map(|p| p.typ.operand_count() - 2)
                    .sum::<usize>();
                cnt
            }
            LogicalSpec::Primitive(basics, _, _) => basics.typ.operand_count(),
        }
    }

    pub fn parameters(&self) -> SmallVec<[TensorSpec<Tgt>; 3]> {
        match self {
            LogicalSpec::Primitive(basics, auxes, _) => match basics.typ {
                PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => basics
                    .parameter_shapes()
                    .into_iter()
                    .zip(auxes)
                    .map(|(s, a)| TensorSpec::new_noncanon_with_aux(s, basics.dtype, a.clone()))
                    .collect(),
                PrimitiveSpecType::Move | PrimitiveSpecType::Zero => auxes
                    .iter()
                    .map(|a| {
                        TensorSpec::new_noncanon_with_aux(
                            basics.spec_shape.clone(),
                            basics.dtype,
                            a.clone(),
                        )
                    })
                    .collect(),
            },
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                let mut result_basics = Vec::with_capacity(self.operand_count());
                let mut last_seen_output = None;
                for (i, c) in components.iter().rev().enumerate() {
                    let mut operand_basics: Vec<(Shape, Dtype)> = c
                        .parameter_shapes()
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

    pub fn inputs(&self) -> SmallVec<[TensorSpec<Tgt>; 3]> {
        let mut operands = self.parameters();
        operands.remove(self.output_idx());
        operands
    }

    pub fn output(&self) -> TensorSpec<Tgt> {
        self.parameters()[self.output_idx()].clone()
    }

    pub fn output_idx(&self) -> usize {
        match &self {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => typ.output_idx(),
            LogicalSpec::Compose { .. } => self.operand_count() - 1,
        }
    }

    pub fn canonicalize(&mut self) {
        // TODO: This is expensive. Make an operand_shapes() method instead.
        let operands = self.parameters();

        match self {
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape,
                    dtype: _,
                },
                primitive_aux,
                _,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum: _ } | PrimitiveSpecType::Conv { accum: _ } => {
                    for i in 0..primitive_aux.len() {
                        primitive_aux[i].contig = primitive_aux[i].layout.tile_contiguity(
                            operands[i].shape(),
                            operands[i].shape(),
                            primitive_aux[i].contig,
                        );
                        primitive_aux[i].layout = primitive_aux[i]
                            .layout
                            .canonicalize_for_shape(operands[i].shape());
                    }
                }
                PrimitiveSpecType::Move => {
                    let [outer_aux, inner_aux] = &mut primitive_aux[..] else {
                        unreachable!();
                    };
                    outer_aux.canonicalize(spec_shape, outer_aux.aligned);
                    inner_aux.contig = inner_aux.layout.tile_contiguity(
                        operands[1].shape(),
                        operands[1].shape(),
                        inner_aux.contig,
                    );
                    inner_aux.layout = inner_aux.layout.canonicalize_for_shape(spec_shape);
                }
                PrimitiveSpecType::Zero => {
                    let aligned = primitive_aux[0].aligned;
                    primitive_aux[0].canonicalize(spec_shape, aligned);
                }
            },
            LogicalSpec::Compose { .. } => todo!(),
        }

        // After a LogicalSpec is made canonical, its parameters should also be canonical.
        debug_assert_eq!(
            self.parameters().to_vec(),
            self.parameters()
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

    pub fn actions(&self) -> impl ActionSeq<Tgt> + '_ {
        let iter = self.tile_out_actions();
        let iter = iter.chain(self.move_actions());
        let iter = iter.chain(Tgt::actions(self));

        match &self {
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape: _,
                    dtype: _,
                },
                _primitive_aux,
                _serial_only,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum } if !*accum => {
                    iter.chain(once(Action::ToAccum)).collect::<Vec<_>>()
                }
                PrimitiveSpecType::Matmul { accum } if *accum => {
                    iter.chain(self.split_actions()).collect::<Vec<_>>()
                }
                PrimitiveSpecType::Conv { accum } => {
                    if *accum {
                        if self.can_spatial_split() {
                            iter.chain(once(Action::SpatialSplit)).collect::<Vec<_>>()
                        } else {
                            iter.collect::<Vec<_>>()
                        }
                    } else {
                        iter.chain(once(Action::ToAccum)).collect::<Vec<_>>()
                    }
                }
                _ => iter.collect::<Vec<_>>(),
            },
            LogicalSpec::Compose {
                components: _,
                operand_auxes: _,
                serial_only: _,
            } => {
                // TODO: Add head reduce split actions as well.
                iter.chain(self.peel_actions()).collect::<Vec<_>>()
            }
        }
    }

    fn can_spatial_split(&self) -> bool {
        let LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, primitive_aux, _) = self else {
            panic!("can_spatial_split called on non-Primitive spec");
        };
        let PrimitiveSpecType::Conv { accum } = typ else {
            panic!("can_spatial_split called on non-Conv spec");
        };
        if !*accum {
            panic!("can_spatial_split called on non-accum Conv spec");
        };

        let parameters = self.parameters();
        let image_shape = parameters[0].shape();
        let filters_shape = parameters[1].shape();

        if image_shape[2..] != filters_shape[2..] {
            return false;
        }
        for a in primitive_aux {
            if let Some(vector_size) = a.vector_size {
                if vector_size != 1 {
                    return false;
                }
            }
        }
        true
    }

    fn tile_out_actions(&self) -> impl Iterator<Item = Action<Tgt>> + '_ {
        let serial_only = self.serial_only();
        let output = self.output();
        let multi_dim = MULTI_DIM_TILING || !serial_only;
        gen_tile_sizes::<Tgt>(output.shape(), true, multi_dim).flat_map(move |tile_shape| {
            let left = once(Action::TileOut {
                output_shape: tile_shape.clone(),
                parallel: false,
            });
            let mut right = None;
            if !serial_only {
                right = Some(Action::TileOut {
                    output_shape: tile_shape,
                    parallel: true,
                });
            }
            left.into_iter().chain(right.into_iter())
        })
    }

    fn split_actions(&self) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ, spec_shape, ..
            },
            ..,
        ) = self
        else {
            panic!("split_actions called on non-primitive Spec");
        };
        let PrimitiveSpecType::Matmul { accum } = typ else {
            panic!("split_actions called on non-Matmul");
        };
        if !accum {
            panic!("split_actions called on non-accumulating Matmul");
        }
        let [m, orig_k, n] = spec_shape[..] else {
            unreachable!();
        };

        let operands = self.parameters();
        Box::new(
            dim_range(orig_k, false)
                .filter(move |&new_k| {
                    let r = operands[0].is_valid_tile_shape(&[m, new_k])
                        && operands[1].is_valid_tile_shape(&[new_k, n]);
                    debug_assert!(m > 1 || new_k > 1 || n > 1 || r); // TODO: Remove
                    r
                })
                .map(|k| Action::Split { k }),
        )
    }

    fn peel_actions(&self) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
        let LogicalSpec::Compose {
            components,
            operand_auxes: _,
            serial_only: _,
        } = self
        else {
            panic!("peel_actions called on non-Compose Spec");
        };

        let mut results = vec![];

        let o = components[1].parameter_shapes();
        let intermediate_shape = &o[components[1].typ.output_idx()];

        for layout in Tgt::move_destination_layouts(intermediate_shape) {
            for level in Tgt::levels() {
                // TODO: Need to implement `can_move_to`-style logic here.

                let vector_bytes = level.vector_bytes();
                if !vector_bytes.is_empty() {
                    for vector_size in gen_vector_sizes(
                        Some(intermediate_shape),
                        components[1].dtype,
                        vector_bytes,
                    ) {
                        results.push(Action::Peel {
                            layout: layout.clone(),
                            level,
                            vector_size: Some(vector_size),
                        });
                    }
                } else {
                    results.push(Action::Peel {
                        layout: layout.clone(),
                        level,
                        vector_size: None,
                    });
                }
            }
        }

        Box::new(results.into_iter())
    }

    fn move_actions(&self) -> impl Iterator<Item = Action<Tgt>> + '_ {
        // TODO: Don't accumulate. Return an iterator.
        let mut results = Vec::with_capacity(MOVE_RESULTS_CAPACITY);

        for (i, operand) in self.parameters().iter().enumerate() {
            // Yield actions for movement with register file destination, which
            // includes relayouts in registers and movements from level 1 to RF.
            let i = u8::try_from(i).unwrap();
            for layout in Tgt::move_destination_layouts(operand.shape()) {
                // TODO: Prevent moving into packed layouts where strip size equals the whole dim.
                for level in Tgt::possible_destination_levels(operand.level()) {
                    if !operand.can_move_to(&layout, &level) {
                        continue;
                    }

                    results.extend(
                        gen_vector_sizes_opt(
                            Some(operand.shape()),
                            operand.dtype(),
                            level.vector_bytes(),
                        )
                        .filter_map(|vector_size| {
                            // Don't yield moves which don't change anything. If yielded, it would be
                            // pruned immediately, but this is a bit quicker.
                            if operand.layout() == layout
                                && operand.level() == level
                                && operand.vector_size() == vector_size
                            {
                                None
                            } else {
                                Some(Action::Move {
                                    source_idx: i,
                                    destination_level: level,
                                    destination_layout: layout.clone(),
                                    destination_vector_size: vector_size,
                                })
                            }
                        }),
                    )
                }
            }
        }

        results.into_iter()
    }

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> TilingInference {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                basics.input_tilings_for_tile_out(smaller_output)
            }
            LogicalSpec::Compose { .. } => {
                todo!("Resolve axes.");
                // let mut accumulated_input_tilings = Vec::with_capacity(self.operand_count() - 1);
                // let mut last_output_tiling = smaller_output.clone();
                // for (i, subspec) in components.iter().enumerate().rev() {
                //     let mut subspec_input_tilings =
                //         subspec.input_tilings_for_tile_out(&last_output_tiling);
                //     debug_assert!(
                //         !subspec_input_tilings.is_empty(),
                //         "Compose contains {:?}, which has no inputs",
                //         subspec
                //     );
                //     if i == 0 {
                //         accumulated_input_tilings.extend(subspec_input_tilings);
                //     } else {
                //         accumulated_input_tilings.extend(subspec_input_tilings.drain(1..));
                //         last_output_tiling = subspec_input_tilings.remove(0);
                //     }
                // }
                // accumulated_input_tilings
            }
        }
    }

    // TODO: Can we replace this entirely with Spec shapes?
    pub fn operands_dim_axes(&self) -> Vec<SmallVec<[u8; 4]>> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.parameter_dim_axes(),
            LogicalSpec::Compose { components, .. } => {
                let mut max_seen = 0;
                let mut accum: Vec<SmallVec<[u8; 4]>> = Vec::new();
                let mut last_out_subs: Option<SmallVec<[u8; 4]>> = None;

                for compose_subspec in components.iter().rev() {
                    let mut kls_axes = Self::increment_dims_axes(
                        &compose_subspec.parameter_dim_axes(),
                        &mut max_seen,
                    );
                    if accum.is_empty() {
                        // Drop the output only
                        accum.extend_from_slice(&kls_axes[..kls_axes.len() - 1]);
                        last_out_subs = Some(kls_axes.last().unwrap().clone());
                    } else {
                        assert!(last_out_subs.is_some());
                        assert_eq!(last_out_subs.as_ref().unwrap().len(), kls_axes[0].len());
                        let substitution_dict = kls_axes
                            .first()
                            .unwrap()
                            .iter()
                            .copied()
                            .zip(last_out_subs.unwrap())
                            .collect::<HashMap<_, _>>();
                        kls_axes = Self::sub_axis(&kls_axes, &substitution_dict);
                        last_out_subs = Some(kls_axes.last().unwrap().clone());
                        let mut new_accum = Vec::with_capacity(accum.len() + kls_axes.len());
                        new_accum.extend_from_slice(&kls_axes[1..kls_axes.len() - 1]);
                        new_accum.extend(accum.drain(..accum.len()));
                        mem::swap(&mut accum, &mut new_accum);
                    }
                    max_seen = kls_axes.into_iter().flatten().max().unwrap();
                }

                // Add the Compose' output
                assert!(last_out_subs.is_some());
                accum.push(last_out_subs.unwrap());
                accum
            }
        }
    }

    fn increment_dims_axes(subs: &[SmallVec<[u8; 4]>], inc: &mut u8) -> Vec<SmallVec<[u8; 4]>> {
        let mut result = Vec::new();
        for dims in subs {
            let mut subresult = SmallVec::new();
            for &d in dims {
                *inc = (*inc).max(d);
                subresult.push(d + *inc);
            }
            result.push(subresult);
        }
        *inc += 1;
        result
    }

    fn sub_axis(
        source: &[SmallVec<[u8; 4]>],
        substitutions: &HashMap<u8, u8>,
    ) -> Vec<SmallVec<[u8; 4]>> {
        let mut result = Vec::new();
        for dims in source {
            let mut subresult = SmallVec::new();
            for &d in dims {
                subresult.push(*substitutions.get(&d).unwrap_or(&d));
            }
            result.push(subresult);
        }
        result
    }

    // TODO: Need IO? Would inputs alone be sufficient? Caller can check inferred output.
    // TODO: Should move new_operands in.
    pub fn replace_io(&mut self, new_operands: &[TensorSpec<Tgt>]) {
        assert_eq!(new_operands.len(), self.operand_count());
        let replaced = match self {
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                let new_inputs = &new_operands[..new_operands.len() - 1];
                let mut remaining_inputs = new_inputs
                    .iter()
                    .map(|t| (t.shape(), t.dtype()))
                    .collect::<Vec<_>>();
                let mut component_inputs: Vec<(Shape, Dtype)> = vec![];
                for (_i, component) in components.iter_mut().enumerate().rev() {
                    // Any missing inputs? Gather them here.
                    let needed = component.typ.input_count() - component_inputs.len();
                    component_inputs.extend(
                        remaining_inputs
                            .drain(remaining_inputs.len() - needed..)
                            .map(|(shape, dtype)| (Shape::from(shape), dtype)),
                    );

                    let new_output_shape = {
                        let inp_shapes = component_inputs
                            .iter()
                            .map(|t| t.0.as_slice())
                            .collect::<Vec<_>>();
                        component.typ.infer_output_shape(&inp_shapes)
                    };
                    let mut new_operands = component_inputs.clone();
                    new_operands.push((new_output_shape, component.dtype));
                    component.replace_io(
                        new_operands
                            .iter()
                            .map(|(s, d)| (&s[..], *d))
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );

                    // Next loop iteration should have have the output as its own argument.
                    component_inputs.clear();
                    component_inputs.push(new_operands.pop().unwrap());
                }

                // At termination, component_inputs should contain exactly the
                // provided replacement output. If it differs, then the replacement
                // output has an invalid shape.
                debug_assert_eq!(component_inputs.len(), 1);
                debug_assert_eq!(
                    new_operands.last().unwrap().shape(),
                    &component_inputs[0].0[..]
                );
                debug_assert_eq!(component_inputs[0].1, new_operands.last().unwrap().dtype());

                *operand_auxes = new_operands.iter().map(|t| t.aux.clone()).collect();
            }
            LogicalSpec::Primitive(basics, primitive_aux, _) => {
                basics.replace_io(
                    &new_operands
                        .iter()
                        .map(|o| (o.shape(), o.dtype))
                        .collect::<Vec<_>>(),
                );

                debug_assert_eq!(primitive_aux.len(), new_operands.len());
                for i in 0..primitive_aux.len() {
                    primitive_aux[i] = new_operands[i].aux.clone();
                }
            }
        };
        debug_assert!(
            self.parameters()
                .iter()
                .zip(new_operands)
                .all(|(a, b)| { a == b }),
            "Parameter mismatch after replace_io; Spec is {} after replacing with [{}]",
            self,
            new_operands.iter().map(|o| o.to_string()).join(", "),
        );
        replaced
    }

    pub fn output_is_read(&self) -> bool {
        match self {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => typ.output_is_read(),
            LogicalSpec::Compose { components, .. } => components[0].typ.output_is_read(),
        }
    }

    pub fn clone_as_accum(&self) -> Self {
        let mut cloned = self.clone();
        match &mut cloned {
            LogicalSpec::Primitive(basics, _, _) => match &mut basics.typ {
                PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum } => {
                    *accum = true;
                }
                _ => panic!("Cannot clone_as_accum for {:?}", self),
            },
            LogicalSpec::Compose { .. } => todo!("Compose can accumulate if head can."),
        }
        cloned
    }

    #[cfg(feature = "verification")]
    pub fn execute<S>(&self, args: &mut [ndarray::ArrayViewMutD<S>])
    where
        S: num_traits::Zero
            + num_traits::NumAssign
            + ndarray::LinalgScalar
            + Clone
            + std::fmt::Debug,
    {
        use ndarray::{s, Ix2, Ix4};

        match self {
            LogicalSpec::Primitive(basics, _, _) => match basics.typ {
                PrimitiveSpecType::Matmul { accum } => {
                    let [lhs, rhs, out] = args else {
                        panic!("Matmul requires 3 arguments");
                    };
                    // TODO: Check shapes and dtypes are correct for this Spec.
                    if !accum {
                        out.fill(S::zero());
                    }
                    let lhs = lhs
                        .view_mut()
                        .into_dimensionality::<Ix2>()
                        .expect("lhs should be rank 2");
                    let rhs = rhs
                        .view_mut()
                        .into_dimensionality::<Ix2>()
                        .expect("rhs should be rank 2");
                    out.assign(&lhs.dot(&rhs));
                }
                PrimitiveSpecType::Conv { accum } => {
                    use ndarray_conv::*;

                    let [lhs, rhs, out] = args else {
                        panic!("Conv requires 3 arguments");
                    };

                    // TODO: Check shapes and dtypes are correct for this Spec.
                    if !accum {
                        out.fill(S::zero());
                    }
                    let lhs = lhs
                        .view_mut()
                        .into_dimensionality::<Ix4>()
                        .expect("lhs should be rank 4");
                    let rhs = rhs
                        .view_mut()
                        .into_dimensionality::<Ix4>()
                        .expect("rhs should be rank 4");
                    for b in 0..lhs.shape()[0] {
                        for c in 0..lhs.shape()[1] {
                            for f in 0..rhs.shape()[0] {
                                let single_img_ch = lhs.slice(s![b, c, .., ..]);
                                let filter_ch = rhs.slice(s![f, c, .., ..]);
                                out.slice_mut(s![b, c, .., ..]).assign(
                                    &Conv2DExt::conv_2d(
                                        &single_img_ch,
                                        &filter_ch,
                                        PaddingSize::Valid,
                                        PaddingMode::Zeros,
                                    )
                                    .unwrap(),
                                );
                            }
                        }
                    }
                }
                PrimitiveSpecType::Move => {
                    let [inp, out] = args else {
                        panic!("Move requires 2 arguments");
                    };
                    // TODO: Check shape and dtype match.
                    out.assign(inp);
                }
                PrimitiveSpecType::Zero => {
                    // TODO: Check shape and dtype are correct for this Spec.
                    args[0].fill(S::zero());
                }
            },
            LogicalSpec::Compose { .. } => todo!(),
        }
    }
}

impl<Tgt: Target> Display for LogicalSpec<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let LogicalSpec::Compose {
            components,
            operand_auxes: _,
            serial_only,
        } = self
        {
            let operands = self.parameters();
            let (output, external_inputs) = operands.split_last().unwrap();
            debug_assert_eq!(self.output_idx(), external_inputs.len());
            return write!(
                f,
                "Compose(({}), [{}, out={}], ({}){})",
                join_into_string(components.iter().map(|c| c.typ), ", "),
                join_into_string(external_inputs, ", "),
                output,
                join_into_string(components.iter().map(|c| c.dtype), ", "),
                if *serial_only { ", serial" } else { "" }
            );
        }

        let header = match self {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => format!("{}", typ),
            LogicalSpec::Compose { .. } => todo!(),
        };

        let operand_str = self
            .parameters()
            .iter()
            .map(|o| format!("{}", o))
            .collect::<Vec<_>>()
            .join(", ");
        let serial_str = if self.serial_only() { ", serial" } else { "" };

        write!(f, "{}({}{})", header, operand_str, serial_str)
    }
}

impl<Tgt, A, Aa, Ab> BiMap for SpecBimap<Tgt, A>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = BimapInt>,
    A: BiMap<Domain = TensorSpecAux<Tgt>, Codomain = (Aa, Ab)>,
    Aa: Clone,
    Ab: IntoIterator<Item = BimapInt>,
{
    type Domain = Spec<Tgt>;
    type Codomain = <LogicalSpecBimap<Tgt, A> as SurMap>::Codomain;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        let mut initial = BiMap::apply(&self.logical_spec_bimap, &t.0);
        initial
            .1
            .extend(BiMap::apply(&self.memory_limits_bimap, &t.1));
        initial
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        let (left, right) = i;
        let (inner_right, memory_right) = right.split_at(i.1.len() - Tgt::levels().len());

        let remaining_value = (
            left.clone(),
            inner_right.iter().copied().map_into().collect(),
        );
        Spec(
            BiMap::apply_inverse(&self.logical_spec_bimap, &remaining_value),
            BiMap::apply_inverse(&self.memory_limits_bimap, &memory_right.into()),
        )
    }
}

impl<Tgt, A, Aa, Ab> BiMap for LogicalSpecBimap<Tgt, A>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = BimapInt>,
    A: BiMap<Domain = TensorSpecAux<Tgt>, Codomain = (Aa, Ab)>,
    Aa: Clone,
    Ab: IntoIterator<Item = BimapInt>,
{
    type Domain = LogicalSpec<Tgt>;
    type Codomain = ((SpecKey, SmallVec<[Aa; 3]>), SmallVec<[BimapInt; 10]>);

    fn apply(&self, spec: &LogicalSpec<Tgt>) -> Self::Codomain {
        match spec {
            LogicalSpec::Primitive(basics, auxes, serial_only) => {
                let (key, mut pt) = BiMap::apply(&PrimitiveBasics::bimap(), basics);
                let aux_keys = auxes
                    .iter()
                    .map(|aux| {
                        let (aux_key, aux_pt) = self.aux_bimap.apply(aux);
                        pt.extend(aux_pt);
                        aux_key
                    })
                    .collect();
                pt.push(!*serial_only as _);
                ((key, aux_keys), pt)
            }
            LogicalSpec::Compose { .. } => todo!(),
        }
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        todo!()
    }
}

impl BiMap for PrimitiveBasicsBimap {
    type Domain = PrimitiveBasics;
    type Codomain = (SpecKey, SmallVec<[BimapInt; 10]>);

    fn apply(&self, basics: &PrimitiveBasics) -> Self::Codomain {
        let PrimitiveBasics {
            typ,
            spec_shape,
            dtype,
        } = basics;
        let shifted_shape = spec_shape.iter().map(|&d| {
            debug_assert!(is_power_of_two_u32(d));
            if LOG_SCALE_SHAPES_BIMAP {
                bit_length_u32(d - 1)
            } else {
                d - 1
            }
        });
        match *typ {
            PrimitiveSpecType::Matmul { accum } => {
                let v = once(!accum as _).chain(shifted_shape).collect();
                (SpecKey::Matmul { dtype: *dtype }, v)
            }
            PrimitiveSpecType::Conv { accum } => {
                let v = once(!accum as _).chain(shifted_shape).collect();
                (SpecKey::Conv { dtype: *dtype }, v)
            }
            PrimitiveSpecType::Move => (SpecKey::Move { dtype: *dtype }, shifted_shape.collect()),
            PrimitiveSpecType::Zero => (SpecKey::Zero { dtype: *dtype }, shifted_shape.collect()),
        }
    }

    fn apply_inverse(&self, c: &Self::Codomain) -> Self::Domain {
        let (key, v) = c;
        let basics = match key {
            SpecKey::Matmul { dtype } | SpecKey::Conv { dtype } => {
                let accum = v[0] == 0;
                let typ = match key {
                    SpecKey::Matmul { .. } => PrimitiveSpecType::Matmul { accum },
                    SpecKey::Conv { .. } => PrimitiveSpecType::Conv { accum },
                    _ => unreachable!(),
                };
                let spec_shape = v.iter().skip(1);
                let spec_shape = if LOG_SCALE_SHAPES_BIMAP {
                    todo!()
                } else {
                    spec_shape.map(|d| d + 1)
                };
                PrimitiveBasics {
                    typ,
                    spec_shape: spec_shape.collect(),
                    dtype: *dtype,
                }
            }
            SpecKey::Move { dtype } => {
                todo!()
            }
            SpecKey::Zero { dtype } => todo!(),
        };
        basics
    }
}

// TODO: Modify to return an `impl Iterator` of some kind instead of a `Box`.
fn gen_tile_sizes<Tgt: Target>(
    tensor_shape: &[DimSize],
    drop_given: bool,
    multi_dim: bool,
) -> Box<dyn Iterator<Item = Shape>> {
    if tensor_shape.is_empty() {
        return Box::new(iter::empty());
    } else if tensor_shape.len() == 1 {
        let one_dim = tensor_shape[0];
        return Box::new(dim_range(one_dim, true).filter_map(move |d| {
            if drop_given && d == one_dim {
                None
            } else {
                Some(smallvec![d])
            }
        }));
    }

    if multi_dim {
        let tensor_shape = tensor_shape.to_vec();
        Box::new(
            gen_tile_sizes::<Tgt>(&tensor_shape[1..], false, multi_dim).flat_map(move |rest| {
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
    } else {
        let tensor_shape = tensor_shape.to_smallvec();
        let own_shape_iter = if !drop_given && tensor_shape.iter().copied().all(is_power_of_two_u32)
        {
            Either::Left(once(tensor_shape.clone()))
        } else {
            Either::Right(iter::empty())
        };
        let smaller_tiles_iter = (0..tensor_shape.len()).flat_map(move |dim| {
            let tensor_shape = tensor_shape.clone();
            dim_range(tensor_shape[dim], false).map(move |d| {
                let mut new_shape = tensor_shape.clone();
                new_shape[dim] = d;
                new_shape
            })
        });
        Box::new(smaller_tiles_iter.chain(own_shape_iter))
    }
}

pub fn gen_vector_sizes<'a>(
    outer_shape: Option<&'a [DimSize]>,
    dtype: Dtype,
    vector_bytes: &'a [u32],
) -> impl Iterator<Item = DimSize> + 'a {
    assert!(outer_shape.is_none() || outer_shape.unwrap().iter().all(|&d| d > 0));
    assert!(!vector_bytes.is_empty());
    assert!(
        vector_bytes
            .iter()
            .all(|&vb| vb % u32::from(dtype.size()) == 0),
        "vector_bytes must be a multiple of dtype size"
    );
    vector_bytes.iter().map(move |&vb| {
        let value_cnt = vb / DimSize::from(dtype.size());
        debug_assert!(value_cnt > 0);
        value_cnt
    })
}

pub fn gen_vector_sizes_opt<'a>(
    outer_shape: Option<&'a [DimSize]>,
    dtype: Dtype,
    vector_bytes: &'a [u32],
) -> impl Iterator<Item = Option<DimSize>> + 'a {
    let mut iter_a = None;
    let mut iter_b = None;
    if vector_bytes.is_empty() {
        iter_a = Some(once(None));
    } else {
        iter_b = Some(gen_vector_sizes(outer_shape, dtype, vector_bytes).map(Some));
    }
    iter_a
        .into_iter()
        .flatten()
        .chain(iter_b.into_iter().flatten())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::X86Target;

    #[test]
    fn test_gen_tile_sizes_empty() {
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, false).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], true, false).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, true).count(), 0);
        assert_eq!(gen_tile_sizes::<X86Target>(&[], false, false).count(), 0);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1_multi_dim() {
        shared_test_gen_tile_sizes_dim_1(true);
    }

    #[test]
    fn test_gen_tile_sizes_dim_1_single_dim() {
        shared_test_gen_tile_sizes_dim_1(false);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_multi_dim() {
        assert_gen_tile_sizes(&[2, 2], [[1, 1], [1, 2], [2, 1], [2, 2]], false, true);
        assert_gen_tile_sizes(&[2, 2], [[1, 1], [1, 2], [2, 1]], true, true);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_multi_dim_non_powers_of_two() {
        assert_gen_tile_sizes(
            &[2, 3],
            [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]],
            false,
            true,
        );
        assert_gen_tile_sizes(
            &[2, 3],
            [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2]],
            true,
            true,
        );
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_single_dim() {
        assert_gen_tile_sizes(&[2, 2], [[1, 2], [2, 1], [2, 2]], false, false);
        assert_gen_tile_sizes(&[2, 2], [[1, 2], [2, 1]], true, false);
    }

    #[test]
    fn test_gen_tile_sizes_dim_2_single_dim_non_powers_of_two() {
        for drop_given in [true, false] {
            assert_gen_tile_sizes(&[2, 3], [[1, 3], [2, 1], [2, 2]], drop_given, false);
        }
    }

    fn shared_test_gen_tile_sizes_dim_1(multi_dim: bool) {
        assert_gen_tile_sizes(&[1], [[1]], false, multi_dim);
        assert_gen_tile_sizes::<0>(&[1], [], true, multi_dim);
        assert_gen_tile_sizes(&[16], [[1], [2], [4], [8], [16]], false, multi_dim);
        assert_gen_tile_sizes(&[16], [[1], [2], [4], [8]], true, multi_dim);
    }

    fn assert_gen_tile_sizes<const D: usize>(
        tensor_shape: &[DimSize],
        expected: impl IntoIterator<Item = [DimSize; D]>,
        drop_given: bool,
        multi_dim: bool,
    ) {
        let actual: Vec<[DimSize; D]> =
            gen_tile_sizes::<X86Target>(tensor_shape, drop_given, multi_dim)
                .map(|s| {
                    assert_eq!(s.len(), D);
                    s.into_iter().collect::<Vec<_>>().try_into().unwrap()
                })
                .sorted()
                .collect::<Vec<_>>();
        let expected = expected.into_iter().sorted().collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "gen_tile_sizes({:?}, drop_given={}, serial={}) returned {:?}, expected {:?}",
            tensor_shape, drop_given, multi_dim, actual, expected
        );
    }
}
