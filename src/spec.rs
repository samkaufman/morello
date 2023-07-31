use super::common::{DimSize, Shape};
use crate::common::Dtype;
use crate::layout::Layout;
use crate::scheduling::Action;
use crate::target::MemoryLevel;
use crate::target::Target;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::join_into_string;

use core::panic;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::iter::Iterator;
use std::iter::{self, once};
use std::mem;

use log::warn;
use std::{assert_eq, debug_assert_eq};

// The following should probably just be Spec::Primitive and Spec::Compose variants once
// there are good conversions to/from image/filter shapes for Conv.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum LogicalSpec<Tgt: Target> {
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
    Move,
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
        inner_vector_size: Option<DimSize>,
    },
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
                let [b, c, h, w] = new_operands[0].0[..] else { panic!(); };
                let [f, alt_c, fh, fw] = new_operands[1].0[..] else { panic!() };
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
            PrimitiveSpecType::Move => {
                let mut operand_auxes_iter = operand_auxes.into_iter();
                let first: &TensorSpecAux<_> = operand_auxes_iter.next().unwrap();
                let second: &TensorSpecAux<_> = operand_auxes_iter.next().unwrap();
                PrimitiveAux::Move {
                    outer_aux: first.clone(),
                    inner_level: second.level,
                    inner_layout: second.layout.clone(),
                    inner_vector_size: second.vector_size.clone(),
                }
            }
        }
    }

    pub fn parameter_shapes(&self) -> Vec<Shape> {
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
            PrimitiveSpecType::Move => {
                vec![self.spec_shape.clone(), self.spec_shape.clone()]
            }
            PrimitiveSpecType::Zero => vec![self.spec_shape.clone()],
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
                let image_bindings = smallvec![Some(0), None];
                let filter_bindings = smallvec![None, Some(1)];

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
    // TODO: Do we really need this?
    pub fn primitive_type(&self) -> PrimitiveSpecType {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.typ,
            LogicalSpec::Compose { .. } => panic!("Spec::Compose has no primitive type"),
        }
    }

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
            LogicalSpec::Primitive(basics, aux, _) => match basics.typ {
                PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => {
                    let PrimitiveAux::Standard(taux) = aux else {
                        unreachable!();
                    };
                    basics
                        .parameter_shapes()
                        .into_iter()
                        .zip(taux)
                        .map(|(s, a)| TensorSpec::new_noncanon_with_aux(s, basics.dtype, a.clone()))
                        .collect()
                }
                PrimitiveSpecType::Move => {
                    let PrimitiveAux::Move { outer_aux, inner_level, inner_layout, inner_vector_size } = aux else {
                        unreachable!();
                    };
                    let outer_tensor_spec = TensorSpec::new_noncanon_with_aux(
                        basics.spec_shape.clone(),
                        basics.dtype,
                        outer_aux.clone(),
                    );
                    let mut inner_tensor_spec = outer_tensor_spec.clone();
                    inner_tensor_spec.set_level(*inner_level, *inner_vector_size);
                    inner_tensor_spec.set_layout(inner_layout.to_owned());
                    inner_tensor_spec.canonicalize();
                    smallvec![outer_tensor_spec, inner_tensor_spec]
                }
                PrimitiveSpecType::Zero => {
                    let PrimitiveAux::Standard(taux) = aux else {
                        unreachable!();
                    };
                    smallvec![TensorSpec::new_noncanon_with_aux(
                        basics.spec_shape.clone(),
                        basics.dtype,
                        taux[0].clone(),
                    )]
                }
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
                PrimitiveSpecType::Move => {
                    let PrimitiveAux::Move {
                        outer_aux,
                        inner_level: _,
                        inner_layout,
                        inner_vector_size: _,
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
            LogicalSpec::Compose { .. } => todo!(),
        }

        // TODO: What if you want to call `operands` on a non-canon Spec?
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

    pub fn actions(&self) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
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
                    Box::new(iter.chain(once(Action::ToAccum)))
                }
                PrimitiveSpecType::Matmul { accum } if *accum => {
                    Box::new(iter.chain(self.split_actions()))
                }
                PrimitiveSpecType::Conv { accum } => {
                    if *accum {
                        if self.can_spatial_split() {
                            Box::new(iter.chain(once(Action::SpatialSplit)))
                        } else {
                            Box::new(iter)
                        }
                    } else {
                        Box::new(iter.chain(once(Action::ToAccum)))
                    }
                }
                _ => Box::new(iter),
            },
            LogicalSpec::Compose {
                components: _,
                operand_auxes: _,
                serial_only: _,
            } => {
                // TODO: Add head reduce split actions as well.
                Box::new(iter.chain(self.peel_actions()))
            }
        }
    }

    fn can_spatial_split(&self) -> bool {
        warn!("spatial split disabled");
        return false;

        let LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, primitive_aux, _) = self else {
            panic!("can_spatial_split called on non-Primitive spec");
        };
        let PrimitiveSpecType::Conv { accum } = typ else {
            panic!("can_spatial_split called on non-Conv spec");
        };
        if !*accum {
            panic!("can_spatial_split called on non-accum Conv spec");
        };
        let PrimitiveAux::Standard(aux) = primitive_aux else {
            unreachable!();
        };

        let operands = self.parameters();
        let image_shape = operands[0].dim_sizes();
        let filters_shape = operands[1].dim_sizes();

        if image_shape[2..] != filters_shape[2..] {
            return false;
        }
        for a in aux {
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
        gen_tile_sizes::<Tgt>(output.dim_sizes(), true).flat_map(move |tile_shape| {
            let mut ts = SmallVec::<[Action<Tgt>; 2]>::new();
            ts.push(self.tile_out(&tile_shape, false));
            if !serial_only {
                ts.push(self.tile_out(&tile_shape, true));
            }
            ts
        })
    }

    fn split_actions(&self) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
        let LogicalSpec::Primitive(PrimitiveBasics { typ, spec_shape, .. }, _, _) = self else {
            panic!("split_actions called on non-primitive Spec");
        };
        let PrimitiveSpecType::Matmul { accum } = typ else {
            panic!("split_actions called on non-Matmul");
        };
        if !accum {
            panic!("split_actions called on non-accumulating Matmul");
        }
        let k = spec_shape[1];
        Box::new(
            dim_range(k, false)
                .filter(|&new_k| self.split_valid(new_k))
                .map(|k| self.split(k)),
        )
    }

    fn peel_actions(&self) -> Box<dyn Iterator<Item = Action<Tgt>> + '_> {
        let LogicalSpec::Compose { components, operand_auxes: _, serial_only: _ } = self else {
            panic!("peel_actions called on non-Compose Spec");
        };

        let mut results = vec![];

        let o = components[1].parameter_shapes();
        let intermediate_shape = &o[components[1].typ.output_idx()];

        for layout in Tgt::all_layouts_for_shape(intermediate_shape) {
            for level in Tgt::levels() {
                // TODO: Need to implement `can_move_to`-style logic here.

                let vector_bytes = level.vector_bytes();
                if !vector_bytes.is_empty() {
                    for vector_size in gen_vector_sizes(
                        Some(intermediate_shape),
                        components[1].dtype,
                        vector_bytes,
                    ) {
                        results.push(self.peel(layout.clone(), level, Some(vector_size)));
                    }
                } else {
                    results.push(self.peel(layout.clone(), level, None));
                }
            }
        }

        Box::new(results.into_iter())
    }

    fn split_valid(&self, new_k: DimSize) -> bool {
        let LogicalSpec::Primitive(PrimitiveBasics { typ, spec_shape, dtype: _ }, _, _) = self else {
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

        let operands = self.parameters();
        if new_k >= orig_k || !operands[0].is_valid_tile_shape(&[m, new_k]) {
            false
        } else {
            operands[1].is_valid_tile_shape(&[new_k, n])
        }
    }

    fn move_actions(&self) -> impl Iterator<Item = Action<Tgt>> + '_ {
        // TODO: Add prefetching moves.

        let mut results = vec![]; // TODO: Don't accumulate. Return an iterator.
        if matches!(self, LogicalSpec::Primitive(_, _, _))
            && matches!(self.primitive_type(), PrimitiveSpecType::Move)
        {
            return results.into_iter();
        }

        for (i, operand) in self.parameters().iter().enumerate() {
            // Yield actions for movement with register file destination, which
            // includes relayouts in registers and movements from level 1 to RF.
            let i = u8::try_from(i).unwrap();
            for layout in Tgt::all_layouts_for_shape(operand.dim_sizes()) {
                for level in Tgt::faster_destination_levels(operand.level()) {
                    if !operand.can_move_to(&layout, &level) {
                        continue;
                    }

                    let vector_bytes = level.vector_bytes();
                    if !vector_bytes.is_empty() {
                        for vector_size in gen_vector_sizes(
                            Some(operand.dim_sizes()),
                            operand.dtype(),
                            vector_bytes,
                        ) {
                            results.push(Action::Move {
                                source_idx: i,
                                destination_level: level,
                                destination_layout: layout.clone(),
                                destination_vector_size: Some(vector_size),
                                prefetch: false,
                            });
                        }
                    } else {
                        results.push({
                            let dest_layout = layout.clone();
                            Action::Move {
                                source_idx: i,
                                destination_level: level,
                                destination_layout: dest_layout,
                                destination_vector_size: None,
                                prefetch: false,
                            }
                        });
                    }
                }
            }
        }

        results.into_iter()
    }

    /// Produces a loop.
    ///
    /// If the Spec cannot be tiled to that shape, returns None.
    pub fn tile_out(&self, output_shape: &[DimSize], parallel: bool) -> Action<Tgt> {
        Action::TileOut {
            output_shape: Shape::from(output_shape),
            parallel,
        }
    }

    fn split(&self, size: DimSize) -> Action<Tgt> {
        Action::Split { k: size }
    }

    fn peel(&self, layout: Layout, level: Tgt::Level, vector_size: Option<DimSize>) -> Action<Tgt> {
        Action::Peel {
            layout,
            level,
            vector_size,
        }
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
        match self {
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                let new_inputs = &new_operands[..new_operands.len() - 1];
                let mut remaining_inputs = new_inputs
                    .iter()
                    .map(|t| (t.dim_sizes(), t.dtype()))
                    .collect::<Vec<_>>();
                let mut component_inputs: Vec<(Shape, Dtype)> = vec![];
                for (_i, component) in components.iter_mut().enumerate().rev() {
                    // Any missing inputs? Gather them here.
                    let needed = component.typ.input_count() - component_inputs.len();
                    eprintln!("input_count is {}", component.typ.input_count());
                    eprintln!("component_inputs.len(): {}", component_inputs.len());
                    eprintln!("remaining_inputs.len(): {}", remaining_inputs.len());
                    eprintln!("needed: {}", needed);
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
                    new_operands.last().unwrap().dim_sizes(),
                    &component_inputs[0].0[..]
                );
                debug_assert_eq!(component_inputs[0].1, new_operands.last().unwrap().dtype());

                *operand_auxes = new_operands.iter().map(|t| t.aux.clone()).collect();
            }
            LogicalSpec::Primitive(basics, primitive_aux, _) => {
                basics.replace_io(
                    &new_operands
                        .iter()
                        .map(|o| (o.dim_sizes(), o.dtype))
                        .collect::<Vec<_>>(),
                );

                match primitive_aux {
                    PrimitiveAux::Standard(aux) => {
                        debug_assert_eq!(aux.len(), new_operands.len());
                        for i in 0..aux.len() {
                            aux[i] = new_operands[i].aux.clone();
                        }
                    }
                    PrimitiveAux::Move {
                        outer_aux,
                        inner_level,
                        inner_layout,
                        inner_vector_size,
                    } => {
                        let [src, dest] = new_operands else {
                            panic!("Moves must have 2 operands");
                        };
                        *outer_aux = src.aux.clone();
                        *inner_level = dest.level();
                        *inner_layout = dest.layout();
                        *inner_vector_size = dest.vector_size();
                    }
                }
            }
        }
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

impl<Tgt: Target> TensorSpecAux<Tgt> {
    fn make_tensorspec_noncanon(&self, dim_sizes: Shape, dtype: Dtype) -> TensorSpec<Tgt> {
        TensorSpec::new_noncanon(
            dim_sizes,
            dtype,
            self.contig,
            self.aligned,
            self.level,
            self.layout.clone(),
            self.vector_size.clone(),
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
        Box::new(iter::empty())
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
