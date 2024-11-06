use super::common::{DimSize, Shape};
use crate::common::Dtype;
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::row_major;
use crate::memorylimits::{MemoryLimits, MemoryLimitsBimap};
use crate::target::Target;
use crate::tensorspec::{self, TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::{bit_length_inverse, bit_length_u32, join_into_string, prev_power_of_two_u32};

use itertools::Itertools;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::fmt;
use std::fmt::Display;
use std::iter::once;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::panic;
use std::{assert_eq, debug_assert_eq};

#[cfg(test)]
const ARBITRARY_SPEC_MAX_SIZE: DimSize = nonzero::nonzero!(8u32);

#[derive(Clone, PartialEq, Eq, Hash, Debug, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct Spec<Tgt: Target>(pub LogicalSpec<Tgt>, pub MemoryLimits);

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
    pub dtypes: Vec<Dtype>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum PrimitiveSpecType {
    Zero,
    Move,
    Matmul { accum: bool },
    Conv { accum: bool },
}

/// Tilings and dimension bindings for a particular output tiling.
///
/// Each dimension of an input tensor/tiling may have a binding to an output
/// tensor dimension. This means that loops should zip those dimensions of each
/// tensor to ensure data dependencies are correct. As an example, a matrix
/// multiplication will give the bindings `vec![Some(0), None]` and
/// `vec![None, Some(1)]` for each of its inputs, indicating that the first
/// dimension of the first input (the m dimension) is bound to the m dimension
/// of the output, and so on for the n dimension.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct TilingInference(pub Vec<(Tiling, Vec<Option<u8>>)>);

/// A [BiMap] which extends [LogicalSpecSurMap] with memory limits dimensions.
///
/// Memory limits are represented identically in the codomain. They are not scaled logarithmically
/// or inverted to be in data dependency order.
pub struct SpecSurMap<Tgt: Target, F, A, Aa> {
    pub logical_spec_surmap: LogicalSpecSurMap<Tgt, F, A, Aa>,
    pub memory_limits_bimap: MemoryLimitsBimap<Tgt>,
}

#[derive(Clone)]
pub struct LogicalSpecSurMap<Tgt, F, A, Aa> {
    pub primitive_basics_bimap: PrimitiveBasicsBimap,
    pub aux_surmap_fn: F,
    marker: std::marker::PhantomData<(Tgt, A, Aa)>,
}

#[derive(Clone)]
pub struct PrimitiveBasicsBimap {
    pub binary_scale_shapes: bool,
}

pub struct ShapeBimap(pub bool);

#[derive(thiserror::Error, Debug)]
pub enum CanonicalizeError {
    #[error("Failed to canonicalize the TensorSpecAux: {0}")]
    TensorSpecAuxCanonicalizeError(tensorspec::CanonicalizeError),
}

#[cfg(test)]
#[derive(Debug, Default)]
pub struct PrimitiveBasicsArbParams {
    max_size: Option<DimSize>,
    first_input_shape: Option<Shape>,
    first_input_dtype: Option<Dtype>,
    allowed_types: Option<Vec<PrimitiveSpecType>>,
}

#[cfg(test)]
impl From<DimSize> for PrimitiveBasicsArbParams {
    fn from(max_size: DimSize) -> Self {
        Some(max_size).into()
    }
}

#[cfg(test)]
impl From<Option<DimSize>> for PrimitiveBasicsArbParams {
    fn from(max_size: Option<DimSize>) -> Self {
        PrimitiveBasicsArbParams {
            max_size,
            ..Default::default()
        }
    }
}

impl<Tgt: Target> Spec<Tgt> {
    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        let parameters = self.0.parameters();
        let levels = parameters.iter().map(|p| p.level()).collect::<Vec<_>>();
        self.1.zero_levels_slower_than_all::<Tgt>(&levels);
        self.0.canonicalize()
    }

    pub fn is_canonical(&self) -> bool {
        let parameters = self.0.parameters();
        let levels = parameters.iter().map(|p| p.level()).collect::<Vec<_>>();
        !self.1.any_nonzero_levels_slower_than::<Tgt>(&levels) && self.0.is_canonical()
    }

    /// Returns the FLOPs required to implement this Spec, if appropriate.
    pub fn flops(&self) -> Option<u64> {
        match self {
            Spec(LogicalSpec::Primitive(basics, _, _), _) => match basics.typ {
                PrimitiveSpecType::Matmul { .. } => {
                    let [m, k, n] = basics.spec_shape[..] else {
                        unreachable!();
                    };
                    Some(2 * u64::from(m.get()) * u64::from(k.get()) * u64::from(n.get()))
                }
                PrimitiveSpecType::Conv { .. } => {
                    // TODO: Implement for floating-pt. Convs.
                    None
                }
                PrimitiveSpecType::Move | PrimitiveSpecType::Zero => None,
            },
            Spec(LogicalSpec::Compose { .. }, _) => None,
        }
    }
}

impl<Tgt: Target> Display for Spec<Tgt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[cfg(test)]
impl<Tgt: Target> proptest::arbitrary::Arbitrary for Spec<Tgt> {
    type Parameters = (Option<DimSize>, Option<u64>);
    type Strategy = proptest::strategy::BoxedStrategy<Spec<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use crate::memorylimits::arb_memorylimits;
        use proptest::prelude::*;

        // Optionally lower the max memory limits.
        let MemoryLimits::Standard(mut max_memory) = Tgt::max_mem();
        if let Some(lower_max) = args.1 {
            max_memory = max_memory.map(|v| v.min(lower_max));
        }

        (
            any_with::<LogicalSpec<Tgt>>(args.0.into()),
            arb_memorylimits::<Tgt>(&max_memory),
        )
            .prop_map(|(logical_spec, mem_limits)| Spec(logical_spec, mem_limits))
            .boxed()
    }
}

#[cfg(test)]
pub fn arb_canonical_spec<Tgt: Target>(
    max_size: Option<DimSize>,
    max_memory: Option<u64>,
) -> impl proptest::strategy::Strategy<Value = Spec<Tgt>> {
    use proptest::prelude::*;

    any_with::<Spec<Tgt>>((max_size, max_memory)).prop_filter_map(
        "Must be possible to canonicalize Spec",
        |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        },
    )
}

impl PrimitiveBasics {
    pub fn replace_io(&mut self, new_operands: &[(&[DimSize], Dtype)]) {
        self.dtypes = new_operands.iter().map(|o| o.1).collect();

        match self.typ {
            PrimitiveSpecType::Matmul { accum: _ } => {
                debug_assert_eq!(new_operands.len(), 3);
                debug_assert_eq!(new_operands[0].0[0], new_operands[2].0[0]);
                debug_assert_eq!(new_operands[1].0[1], new_operands[2].0[1]);
                debug_assert_eq!(new_operands[0].0[1], new_operands[1].0[0]);
                self.spec_shape = vec![
                    new_operands[0].0[0],
                    new_operands[0].0[1],
                    new_operands[1].0[1],
                ];
            }
            PrimitiveSpecType::Conv { accum: _ } => {
                let [b, c, h, w] = new_operands[0].0[..] else {
                    panic!();
                };
                let [f, alt_c, fh, fw] = new_operands[1].0[..] else {
                    panic!()
                };
                assert_eq!(c, alt_c);
                self.spec_shape = vec![b, f, c, h, w, fh, fw];
                // TODO: Assert output shape is expected.
            }
            PrimitiveSpecType::Move => {
                let [src, dest] = new_operands else {
                    panic!("Move must have 2 operands");
                };
                assert_eq!(src.0, dest.0);
                self.spec_shape = src.0.into();
            }
            PrimitiveSpecType::Zero => {
                assert_eq!(new_operands.len(), 1);
                self.spec_shape = new_operands[0].0.into();
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

    // TODO: Avoid constructing output shape.
    pub fn input_shapes(&self) -> Vec<Shape> {
        let mut operands = self.parameter_shapes();
        operands.remove(self.typ.output_idx());
        operands
    }

    pub fn parameter_shapes(&self) -> Vec<Shape> {
        match self.typ {
            PrimitiveSpecType::Matmul { .. } => {
                let [m, k, n] = self.spec_shape[..] else {
                    panic!("Matmul spec_shape must have length 3")
                };
                vec![vec![m, k], vec![k, n], vec![m, n]]
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
                let img = vec![b, c, h, w];
                let filt = vec![f, c, fh, fw];
                let out = conv_infer_output_shape(&img, &filt);
                vec![img, filt, out]
            }
            PrimitiveSpecType::Move => {
                vec![self.spec_shape.clone(), self.spec_shape.clone()]
            }
            PrimitiveSpecType::Zero => vec![self.spec_shape.clone()],
        }
    }

    pub fn parameter_shape(&self, idx: usize) -> Shape {
        match self.typ {
            PrimitiveSpecType::Matmul { .. } => match idx {
                0 => vec![self.spec_shape[0], self.spec_shape[1]],
                1 => vec![self.spec_shape[1], self.spec_shape[2]],
                2 => vec![self.spec_shape[0], self.spec_shape[2]],
                _ => panic!("Matmul has only 3 parameters"),
            },
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
                match idx {
                    0 => vec![b, c, h, w],
                    1 => vec![f, c, fh, fw],
                    2 => conv_infer_output_shape(&[b, c, h, w], &[f, c, fh, fw]),
                    _ => panic!("Conv has only 3 parameters"),
                }
            }
            PrimitiveSpecType::Move => match idx {
                0 | 1 => self.spec_shape.clone(),
                _ => panic!("Move has only 2 parameters"),
            },
            PrimitiveSpecType::Zero => match idx {
                0 => self.spec_shape.clone(),
                _ => panic!("Zero has only 1 parameter"),
            },
        }
    }

    pub fn input_shape(&self, idx: usize) -> Shape {
        if idx < self.typ.output_idx() {
            self.parameter_shape(idx)
        } else {
            self.parameter_shape(idx + 1)
        }
    }

    pub fn output_shape(&self) -> Shape {
        self.parameter_shape(self.typ.output_idx())
    }

    pub fn parameter_dtypes(&self) -> Vec<Dtype> {
        self.dtypes.clone()
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
                        vec![smaller_output.shape()[0], spec_shape[1]],
                        vec![smaller_output.step_sizes()[0], spec_shape[1]],
                    ),
                    vec![Some(0), None],
                ),
                (
                    Tiling::new_sliding(
                        vec![spec_shape[1], smaller_output.shape()[1]],
                        vec![spec_shape[1], smaller_output.step_sizes()[1]],
                    ),
                    vec![None, Some(1)],
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
                            .map(|(&o, f)| o.get() + f.get() - 1)
                            .map(|d| DimSize::new(d).unwrap()),
                    )
                    .collect();
                let mut new_image_steps: Shape = smaller_output.step_sizes().into();
                new_image_steps[1] = channels;

                // Compute the new filters Tiling.
                let new_filters_shape: Shape = [smaller_output.shape()[1], channels]
                    .into_iter()
                    .chain([fh, fw])
                    .collect();
                let mut new_filters_steps: Shape = new_filters_shape.clone();
                new_filters_steps[0] = smaller_output.step_sizes()[1];

                TilingInference(vec![
                    (
                        Tiling::new_sliding(new_image_shape, new_image_steps),
                        vec![Some(0), None, None, None],
                    ),
                    (
                        Tiling::new_sliding(new_filters_shape, new_filters_steps),
                        vec![None, Some(1), None, None],
                    ),
                ])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    ..
                },
                _,
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
                _,
            ) => TilingInference(vec![]),
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
        write!(f, "{}({}, ", self.typ, shape_str)?;
        if self.dtypes.len() == 1 {
            write!(f, "{})", self.dtypes[0])
        } else {
            write!(f, "[{}])", self.dtypes.iter().join(", "))
        }
    }
}

#[cfg(test)]
impl proptest::arbitrary::Arbitrary for PrimitiveBasics {
    type Parameters = PrimitiveBasicsArbParams;
    type Strategy = proptest::strategy::BoxedStrategy<PrimitiveBasics>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let max_size = args.max_size.unwrap_or(ARBITRARY_SPEC_MAX_SIZE).get();

        let type_strategy = match args.allowed_types {
            Some(allowed_types) => proptest::sample::select(allowed_types).sboxed(),
            None => any::<PrimitiveSpecType>().sboxed(),
        };
        type_strategy
            .prop_flat_map(move |typ| {
                let cnt = typ.operand_count();
                let dtypes_strategy = match args.first_input_dtype {
                    Some(d) => proptest::collection::vec(any::<Dtype>(), cnt - 1)
                        .prop_map(move |mut v| {
                            v.insert(0, d);
                            v
                        })
                        .sboxed(),
                    None => proptest::collection::vec(any::<Dtype>(), cnt).sboxed(),
                };
                (Just(typ), dtypes_strategy)
            })
            .prop_flat_map(move |(typ, dtypes)| {
                let shape_strategy = match typ {
                    PrimitiveSpecType::Matmul { accum: _ } => {
                        let (m, k) = match args.first_input_shape.as_deref() {
                            Some([m, k]) => (Just(m.get()).sboxed(), Just(k.get()).sboxed()),
                            Some(_) => panic!("Matmul requires a rank-2 first input"),
                            None => ((1..=max_size).sboxed(), (1..=max_size).sboxed()),
                        };
                        vec![m, k, (1..=max_size).sboxed()].sboxed()
                    }
                    PrimitiveSpecType::Conv { accum: _ } => {
                        let (b, c, h, w) = match args.first_input_shape.as_deref() {
                            Some([b, c, h, w]) => (
                                Just(b.get()).sboxed(),
                                Just(c.get()).sboxed(),
                                Just(h.get()).sboxed(),
                                Just(w.get()).sboxed(),
                            ),
                            Some(_) => panic!("Conv requires a rank-4 first input"),
                            None => (
                                (1..=max_size).sboxed(),
                                (1..=max_size).sboxed(),
                                (1..=max_size).sboxed(),
                                (1..=max_size).sboxed(),
                            ),
                        };
                        (b, c, h, w)
                            .prop_flat_map(move |(b, c, h, w)| {
                                (
                                    Just(b),
                                    1..max_size,
                                    Just(c),
                                    Just(h),
                                    Just(w),
                                    1..=h,
                                    1..=w,
                                )
                            })
                            .prop_map(|(b, f, c, h, w, fh, fw)| vec![b, f, c, h, w, fh, fw])
                            .sboxed()
                    }
                    PrimitiveSpecType::Move | PrimitiveSpecType::Zero => {
                        match args.first_input_shape.as_deref() {
                            Some(s) => s.iter().map(|d| Just(d.get())).collect::<Vec<_>>().sboxed(),
                            None => (1..=4usize)
                                .prop_flat_map(move |tensor_rank| {
                                    proptest::collection::vec(1..=max_size, tensor_rank).sboxed()
                                })
                                .sboxed(),
                        }
                    }
                };
                (Just(typ), Just(dtypes), shape_strategy)
            })
            .prop_map(move |(typ, dtypes, spec_shape)| PrimitiveBasics {
                typ,
                spec_shape: spec_shape
                    .into_iter()
                    .map(|x| DimSize::new(x).unwrap())
                    .collect(),
                dtypes,
            })
            .boxed()
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
                vec![*m, *n]
            }
            PrimitiveSpecType::Conv { .. } => {
                let ([b, _, h, w], [f, _, fh, fw]) = (inputs[0], inputs[1]) else {
                    panic!("Conv inputs must have 4 dimensions each");
                };
                debug_assert!(h.get() >= fh.get() && w.get() >= fw.get());
                vec![
                    *b,
                    *f,
                    DimSize::new(1 + h.get() - fh.get()).unwrap(),
                    DimSize::new(1 + w.get() - fw.get()).unwrap(),
                ]
            }
            PrimitiveSpecType::Move | PrimitiveSpecType::Zero => {
                // The shape and dtype match for moves and zero.
                inputs[0].to_vec()
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

    pub fn set_serial_only(&mut self, serial_only: bool) {
        match self {
            LogicalSpec::Primitive(_, _, ref mut s) => *s = serial_only,
            LogicalSpec::Compose {
                serial_only: ref mut s,
                ..
            } => *s = serial_only,
        }
    }

    pub fn operand_count(&self) -> usize {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.typ.operand_count(),
            LogicalSpec::Compose { components, .. } => compose_parameter_count(components),
        }
    }

    pub fn parameters(&self) -> Vec<TensorSpec<Tgt>> {
        match self {
            LogicalSpec::Primitive(basics, auxes, _) => match basics.typ {
                PrimitiveSpecType::Matmul { .. } | PrimitiveSpecType::Conv { .. } => basics
                    .parameter_shapes()
                    .into_iter()
                    .zip(&basics.dtypes)
                    .zip(auxes)
                    .map(|((s, dt), a)| TensorSpec::new_noncanon_with_aux(s, *dt, a.clone()))
                    .collect(),
                PrimitiveSpecType::Move | PrimitiveSpecType::Zero => auxes
                    .iter()
                    .zip(&basics.dtypes)
                    .map(|(a, dtype)| {
                        TensorSpec::new_noncanon_with_aux(
                            basics.spec_shape.clone(),
                            *dtype,
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
                debug_assert!(
                    components.len() >= 2,
                    "Compose must have at least 2 components, but components are: {:?}",
                    components
                );

                let mut result = vec![];
                result.reserve_exact(compose_parameter_count(components));

                // Compute the parameters for the first component (executed last).
                let mut outermost_component_parameter_shapes = components[0].parameter_shapes();
                let mut outermost_component_dtypes = components[0].dtypes.clone();
                let outermost_component_output_shape =
                    outermost_component_parameter_shapes.remove(components[0].typ.output_idx());
                let outermost_component_output_dtype =
                    outermost_component_dtypes.remove(components[0].typ.output_idx());
                result.extend(
                    outermost_component_parameter_shapes[1..]
                        .iter()
                        .zip(&outermost_component_dtypes[1..])
                        .zip(&operand_auxes[..outermost_component_dtypes.len() - 1])
                        .map(|((s, dt), a)| {
                            TensorSpec::new_noncanon_with_aux(s.clone(), *dt, a.clone())
                        }),
                );

                for c in &components[1..components.len() - 1] {
                    let mut ps = c.parameter_shapes();
                    ps.remove(c.typ.output_idx());
                    ps.remove(0);
                    let mut dtypes = c.dtypes.clone();
                    dtypes.remove(c.typ.output_idx());
                    dtypes.remove(0);
                    result.extend(
                        ps.into_iter()
                            .zip(dtypes)
                            .zip(&operand_auxes[result.len()..])
                            .map(|((s, dt), a)| {
                                TensorSpec::new_noncanon_with_aux(s, dt, a.clone())
                            }),
                    );
                }

                // Fill in the innermost component
                let innermost_component = &components[components.len() - 1];
                let mut ps = innermost_component.parameter_shapes();
                ps.remove(innermost_component.typ.output_idx());
                let mut dtypes = innermost_component.dtypes.clone();
                dtypes.remove(innermost_component.typ.output_idx());
                result.extend(
                    ps.into_iter()
                        .zip(dtypes)
                        .zip(&operand_auxes[result.len()..])
                        .map(|((s, dt), a)| TensorSpec::new_noncanon_with_aux(s, dt, a.clone())),
                );

                result.push(TensorSpec::new_noncanon_with_aux(
                    outermost_component_output_shape,
                    outermost_component_output_dtype,
                    operand_auxes.last().unwrap().clone(),
                ));
                result
            }
        }
    }

    pub fn input_shapes(&self) -> Vec<Shape> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.input_shapes(),
            LogicalSpec::Compose { .. } => todo!(),
        }
    }

    pub fn output_shape(&self) -> Shape {
        self.parameter_shape(self.output_idx())
    }

    pub fn parameter_shapes(&self) -> Vec<Shape> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.parameter_shapes(),
            LogicalSpec::Compose { components, .. } => compose_parameter_shapes(components),
        }
    }

    pub fn parameter_shape(&self, index: usize) -> Shape {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.parameter_shape(index),
            LogicalSpec::Compose { components, .. } => compose_parameter_shape(components, index),
        }
    }

    pub fn inputs(&self) -> Vec<TensorSpec<Tgt>> {
        let mut operands = self.parameters();
        operands.remove(self.output_idx());
        operands
    }

    pub fn output(&self) -> TensorSpec<Tgt> {
        self.parameters().swap_remove(self.output_idx())
    }

    pub fn output_idx(&self) -> usize {
        match &self {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => typ.output_idx(),
            LogicalSpec::Compose { .. } => self.operand_count() - 1,
        }
    }

    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        match self {
            LogicalSpec::Primitive(basics, primitive_aux, _) => match &basics.typ {
                PrimitiveSpecType::Matmul { accum: _ } | PrimitiveSpecType::Conv { accum: _ } => {
                    for (shp, aux) in basics.parameter_shapes().iter().zip(primitive_aux) {
                        aux.canonicalize(shp)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                    }
                }
                PrimitiveSpecType::Move => {
                    for aux in primitive_aux.iter_mut() {
                        aux.canonicalize(&basics.spec_shape)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                    }

                    // It source and destination are fully contiguous and the dtypes and layouts
                    // match, then we can canonicalize to a row-major bitwise move. This is a
                    // workaround for not being able to split interleaved layouts with a tile, but
                    // can be generalized to be a useful symmetry-breaking predicate later on.
                    // TODO: Do just that: generalize this caonicalizaton rule.
                    if basics.dtypes.iter().all_equal()
                        && primitive_aux.iter().map(|a| &a.layout).all_equal()
                        && primitive_aux
                            .iter()
                            .all(|aux| aux.contig == aux.layout.contiguous_full())
                    {
                        let rm = row_major(basics.spec_shape.len().try_into().unwrap());
                        let new_contig = rm.contiguous_full();
                        for aux in primitive_aux.iter_mut() {
                            aux.layout = rm.clone();
                            aux.contig = new_contig;
                        }
                    }
                }
                PrimitiveSpecType::Zero => {
                    primitive_aux[0]
                        .canonicalize(&basics.spec_shape)
                        .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                }
            },
            LogicalSpec::Compose { .. } => {
                let shapes = self.parameter_shapes();
                let LogicalSpec::Compose { operand_auxes, .. } = self else {
                    unreachable!();
                };
                for (shp, aux) in shapes.into_iter().zip(operand_auxes) {
                    aux.canonicalize(&shp)
                        .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                }
            }
        }
        Ok(())
    }

    pub fn is_canonical(&self) -> bool {
        match self {
            LogicalSpec::Primitive(basics, primitive_aux, _) => match &basics.typ {
                PrimitiveSpecType::Matmul { accum: _ } | PrimitiveSpecType::Conv { accum: _ } => {
                    for (shp, aux) in basics.parameter_shapes().iter().zip(primitive_aux) {
                        if !aux.is_canonical(shp) {
                            return false;
                        }
                    }
                }
                PrimitiveSpecType::Move => {
                    for aux in primitive_aux {
                        if !aux.is_canonical(&basics.spec_shape) {
                            return false;
                        }
                    }

                    if basics.dtypes.iter().all_equal()
                        && primitive_aux.iter().map(|a| &a.layout).all_equal()
                        && primitive_aux
                            .iter()
                            .all(|aux| aux.contig == aux.layout.contiguous_full())
                        && primitive_aux.iter().any(|aux| {
                            !aux.layout.is_row_major() || aux.contig != aux.layout.contiguous_full()
                        })
                    {
                        return false;
                    }
                }
                PrimitiveSpecType::Zero => {
                    if !primitive_aux[0].is_canonical(&basics.spec_shape) {
                        return false;
                    }
                }
            },
            LogicalSpec::Compose { .. } => {
                let shapes = self.parameter_shapes();
                let LogicalSpec::Compose { operand_auxes, .. } = self else {
                    unreachable!();
                };
                for (shp, aux) in shapes.into_iter().zip(operand_auxes) {
                    if !aux.is_canonical(&shp) {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn can_spatial_split(&self) -> bool {
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
                if vector_size.get() != 1 {
                    return false;
                }
            }
        }
        true
    }

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> TilingInference {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                basics.input_tilings_for_tile_out(smaller_output)
            }
            LogicalSpec::Compose { components, .. } => {
                let mut accumulated_input_tilings = Vec::with_capacity(self.operand_count() - 1);

                // Compute [TilingInference] for the outermost (last-executed) component.
                // `accumulated_input_tilings` will contain the [Tiling]s and dimensions for all but
                // the first input of that outermost component (those which don't consume the next
                // component's output).
                let mut first_inference = components[0].input_tilings_for_tile_out(smaller_output);
                accumulated_input_tilings.extend(first_inference.0.drain(1..));

                let mut last_output_tiling = first_inference.0.remove(0).0;
                for subspec in &components[1..components.len() - 1] {
                    let mut subspec_input_tilings =
                        subspec.input_tilings_for_tile_out(&last_output_tiling);
                    accumulated_input_tilings.extend(subspec_input_tilings.0.drain(1..));
                    last_output_tiling = subspec_input_tilings.0.remove(0).0;
                }

                accumulated_input_tilings.extend(
                    components[components.len() - 1]
                        .input_tilings_for_tile_out(&last_output_tiling)
                        .0,
                );

                TilingInference(accumulated_input_tilings)
            }
        }
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
                    .map(|t| (t.shape(), t.dtype()))
                    .collect::<Vec<_>>();
                let mut component_inputs: Vec<(Shape, Dtype)> = vec![];
                for component in components.iter_mut().rev() {
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
                    new_operands.push((
                        new_output_shape,
                        component.dtypes[component.typ.output_idx()],
                    ));
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
                _ => panic!("Cannot clone_as_accum: {:?}", self),
            },
            LogicalSpec::Compose { components, .. } => match &mut components[0].typ {
                PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum } => {
                    *accum = true;
                }
                _ => panic!("Cannot clone_as_accum: {:?}", self),
            },
        }
        cloned
    }

    /// Returns the product of Spec dimensions.
    pub fn volume(&self) -> DimSize {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                DimSize::new(basics.spec_shape.iter().map(|d| d.get()).product()).unwrap()
            }
            LogicalSpec::Compose { .. } => {
                // Returning a 1 here basically disables intensity-scaling.
                // TODO: Return an actual volume.
                nz!(1u32)
            }
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
                "Compose(({}), [{}, out={}]{})",
                join_into_string(components.iter().map(|c| c.typ), ", "),
                join_into_string(external_inputs, ", "),
                output,
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

impl<Tgt, F, A, Aa, const N: usize> SurMap for SpecSurMap<Tgt, F, A, Aa>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    F: Fn(&[DimSize], Dtype) -> A,
    A: SurMap<Domain = TensorSpecAux<Tgt>, Codomain = (Aa, [BimapInt; N])>,
    A::DomainIter: 'static,
    Aa: Clone,
{
    type Domain = Spec<Tgt>;
    type Codomain = <LogicalSpecSurMap<Tgt, F, A, Aa> as SurMap>::Codomain;
    type DomainIter = Box<dyn Iterator<Item = Self::Domain>>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        let mut initial = SurMap::apply(&self.logical_spec_surmap, &t.0);
        initial
            .1
            .extend(BiMap::apply(&self.memory_limits_bimap, &t.1));
        initial
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let (left, right) = i;
        let (inner_right, memory_right) = right.split_at(i.1.len() - Tgt::levels().len());

        let remaining_value = (
            left.clone(),
            inner_right.iter().copied().map_into().collect(),
        );
        let m = BiMap::apply_inverse(&self.memory_limits_bimap, &memory_right.into());
        Box::new(
            self.logical_spec_surmap
                .apply_inverse(&remaining_value)
                .map(move |ls| Spec(ls, m.clone())),
        )
    }
}

impl<Tgt, F, A, Aa> LogicalSpecSurMap<Tgt, F, A, Aa> {
    pub fn new(primitive_basics_bimap: PrimitiveBasicsBimap, aux_surmap_fn: F) -> Self {
        Self {
            primitive_basics_bimap,
            aux_surmap_fn,
            marker: PhantomData,
        }
    }
}

impl<Tgt, F, A, Aa, const N: usize> SurMap for LogicalSpecSurMap<Tgt, F, A, Aa>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    F: Fn(&[DimSize], Dtype) -> A,
    A: SurMap<Domain = TensorSpecAux<Tgt>, Codomain = (Aa, [BimapInt; N])>,
    A::DomainIter: 'static,
    Aa: Clone,
{
    type Domain = LogicalSpec<Tgt>;
    type Codomain = ((SpecKey, Vec<Aa>), Vec<BimapInt>);
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + Send>;

    fn apply(&self, spec: &LogicalSpec<Tgt>) -> Self::Codomain {
        match spec {
            LogicalSpec::Primitive(basics, auxes, serial_only) => {
                let (key, mut pt) = BiMap::apply(&self.primitive_basics_bimap, basics);
                let aux_keys = auxes
                    .iter()
                    .zip(basics.parameter_shapes())
                    .zip(&basics.dtypes)
                    .map(|((tensor_aux, tensor_shape), dtype)| {
                        let aux_bimap = (self.aux_surmap_fn)(&tensor_shape, *dtype);
                        let (aux_key, aux_pt) = aux_bimap.apply(tensor_aux);
                        pt.extend(aux_pt);
                        aux_key
                    })
                    .collect();
                pt.push(!*serial_only as _);
                ((key, aux_keys), pt)
            }
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only,
            } => {
                let key = SpecKey::Compose {
                    components: components
                        .iter()
                        .map(|c| (c.typ, c.dtypes.clone()))
                        .collect(),
                };
                let shape_bimap = ShapeBimap(self.primitive_basics_bimap.binary_scale_shapes);
                let mut pt = components
                    .iter()
                    .flat_map(|c| BiMap::apply(&shape_bimap, &c.spec_shape))
                    .collect::<Vec<_>>();
                // TODO: Avoid calling self.parameters(), which is expensive, if possible
                let aux_keys = operand_auxes
                    .iter()
                    .zip(spec.parameters())
                    .map(|(tensor_aux, parameter)| {
                        let aux_bimap = (self.aux_surmap_fn)(parameter.shape(), parameter.dtype());
                        let (aux_key, aux_pt) = aux_bimap.apply(tensor_aux);
                        pt.extend(aux_pt);
                        aux_key
                    })
                    .collect();

                pt.push(!*serial_only as _);
                ((key, aux_keys), pt)
            }
        }
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let ((key, aux_keys), pt) = i;
        let dtypes = key.dtypes();
        let operand_count = aux_keys.len();

        let pt_without_serial = &pt[..pt.len() - 1];
        let (basics_pt, tensor_aux_pts) =
            pt_without_serial.split_at(pt.len() - (operand_count * N) - 1);
        let serial = pt[pt.len() - 1] == 0;

        let primitive_basics = BiMap::apply_inverse(
            &self.primitive_basics_bimap,
            &(key.clone(), basics_pt.into()),
        );
        let parameter_shapes = primitive_basics.parameter_shapes();

        Box::new(
            (0..operand_count)
                .map(move |i| {
                    let Ok(tap) = (&tensor_aux_pts[i * N..(i + 1) * N]).try_into() else {
                        panic!("Couldn't reverse the TensorSpecAux pt.");
                    };
                    let aux_surmap = (self.aux_surmap_fn)(&parameter_shapes[i], dtypes[i]);
                    // TODO: Avoid collect, which is here to avoid needing the iter to be Clone
                    aux_surmap
                        .apply_inverse(&(aux_keys[i].clone(), tap))
                        .collect::<Vec<_>>()
                })
                .multi_cartesian_product()
                .map(move |tensor_auxes| {
                    LogicalSpec::Primitive(primitive_basics.clone(), tensor_auxes, serial)
                }),
        )
    }
}

impl BiMap for PrimitiveBasicsBimap {
    type Domain = PrimitiveBasics;
    type Codomain = (SpecKey, Vec<BimapInt>);

    fn apply(&self, basics: &PrimitiveBasics) -> Self::Codomain {
        let PrimitiveBasics {
            typ,
            spec_shape,
            dtypes,
        } = basics;
        let shifted_shape = spec_shape.iter().map(|d| d.get()).map(|d| {
            if self.binary_scale_shapes {
                if !d.is_power_of_two() {
                    panic!("Given non-zero/power-of-two shape {}", d);
                }
                bit_length_u32(prev_power_of_two_u32(d - 1))
            } else {
                d - 1
            }
        });
        match *typ {
            PrimitiveSpecType::Matmul { accum } => {
                let v = once(!accum as _).chain(shifted_shape).collect();
                (
                    SpecKey::Matmul {
                        dtypes: dtypes.as_slice().try_into().unwrap(),
                    },
                    v,
                )
            }
            PrimitiveSpecType::Conv { accum } => {
                let mut v: Vec<_> = once(!accum as _).chain(shifted_shape).collect();
                // Conv's image dimensions must be larger than or equal to the corresponding filter
                // dimensions (the final two dimensions in `v`/`shifted_shape`), so we'll subtract
                // the filter sizes from the image sizes, thereby normalizing the image dims. to
                // zero.
                v[4] -= v[6];
                v[5] -= v[7];
                (
                    SpecKey::Conv {
                        dtypes: dtypes.as_slice().try_into().unwrap(),
                    },
                    v,
                )
            }
            PrimitiveSpecType::Move => (
                SpecKey::Move {
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Zero => {
                (SpecKey::Zero { dtype: dtypes[0] }, shifted_shape.collect())
            }
        }
    }

    fn apply_inverse(&self, c: &Self::Codomain) -> Self::Domain {
        let (key, v) = c;
        let basics = match key {
            SpecKey::Matmul { dtypes } | SpecKey::Conv { dtypes } => {
                let accum = v[0] == 0;
                let typ = match key {
                    SpecKey::Matmul { .. } => PrimitiveSpecType::Matmul { accum },
                    SpecKey::Conv { .. } => PrimitiveSpecType::Conv { accum },
                    _ => unreachable!(),
                };

                let mut spec_shape: Vec<BimapInt> = v.iter().skip(1).copied().collect();
                // Reverse the normalization of image dimensions (see `apply`).
                if matches!(key, SpecKey::Conv { .. }) {
                    spec_shape[3] += spec_shape[5];
                    spec_shape[4] += spec_shape[6];
                }
                for d in &mut spec_shape[..] {
                    if self.binary_scale_shapes {
                        *d = u32::try_from((bit_length_inverse(*d) + 1).next_power_of_two())
                            .unwrap();
                    } else {
                        *d += 1;
                    }
                }

                PrimitiveBasics {
                    typ,
                    spec_shape: spec_shape
                        .iter()
                        .map(|&d| DimSize::new(d).unwrap())
                        .collect(),
                    dtypes: dtypes.as_slice().into(),
                }
            }
            SpecKey::Move { dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::Move,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.as_slice().into(),
            },
            SpecKey::Zero { dtype } => PrimitiveBasics {
                typ: PrimitiveSpecType::Zero,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: vec![*dtype],
            },
            SpecKey::Compose { .. } => {
                panic!("PrimitiveBasicsBimap is not defined for Compose keys")
            }
        };
        basics
    }
}

impl BiMap for ShapeBimap {
    type Domain = Vec<DimSize>;
    type Codomain = Vec<BimapInt>;

    fn apply(&self, shape: &Self::Domain) -> Self::Codomain {
        shape
            .iter()
            .map(|d| d.get())
            .map(|d| {
                if self.0 {
                    if !d.is_power_of_two() {
                        panic!("Given non-zero/power-of-two shape {}", d);
                    }
                    bit_length_u32(prev_power_of_two_u32(d - 1))
                } else {
                    d - 1
                }
            })
            .collect()
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        i.iter()
            .map(|&d| {
                DimSize::new(if self.0 {
                    u32::try_from((bit_length_inverse(d) + 1).next_power_of_two()).unwrap()
                } else {
                    d + 1
                })
                .unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
impl<Tgt: Target> proptest::arbitrary::Arbitrary for LogicalSpec<Tgt> {
    type Parameters = PrimitiveBasicsArbParams;
    type Strategy = proptest::strategy::BoxedStrategy<LogicalSpec<Tgt>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use crate::tensorspec::TensorSpecArbMaxShape;
        use proptest::prelude::*;

        let primitive_arb = (any_with::<PrimitiveBasics>(args), any::<bool>())
            .prop_flat_map(|(basics, serial_only)| {
                // TODO: These don't all make sense. Are they canonical for shapes?
                let auxes_strategy = basics
                    .parameter_shapes()
                    .into_iter()
                    .map(|s| any_with::<TensorSpecAux<Tgt>>(TensorSpecArbMaxShape(s)))
                    .collect::<Vec<_>>();
                (Just(basics), auxes_strategy, Just(serial_only))
            })
            .prop_map(|(basics, auxes, serial_only)| {
                LogicalSpec::Primitive(basics, auxes, serial_only)
            })
            .prop_filter("Layout must be applicable to TensorSpec shape", |s| {
                s.clone().canonicalize().is_ok()
            });

        proptest::prop_oneof![primitive_arb, arb_compose_spec()].boxed()
    }
}

#[cfg(test)]
pub(crate) fn arb_compose_spec<Tgt>() -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>>
where
    Tgt: Target,
{
    use crate::tensorspec::TensorSpecArbMaxShape;
    use proptest::prelude::*;

    arb_compose_components()
        .prop_flat_map(|components| {
            let auxes_strategies = compose_parameter_shapes(&components)
                .into_iter()
                .map(|shape| any_with::<TensorSpecAux<Tgt>>(TensorSpecArbMaxShape(shape)))
                .collect::<Vec<_>>();
            (Just(components), auxes_strategies, any::<bool>())
        })
        .prop_map(
            |(components, operand_auxes, serial_only)| LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only,
            },
        )
}

#[cfg(test)]
fn arb_compose_component_innermost() -> impl proptest::strategy::Strategy<Value = PrimitiveBasics> {
    use proptest::prelude::*;

    any::<PrimitiveBasics>()
        .prop_filter("Must have at least two parameters to compose", |basics| {
            basics.typ.operand_count() > 1
        })
        .boxed()
}

#[cfg(test)]
fn arb_compose_component_successor(
    predecessor: &PrimitiveBasics,
) -> impl proptest::strategy::Strategy<Value = PrimitiveBasics> {
    use proptest::prelude::*;

    // Restrict the basic types to those which have a first input of a possible rank.
    let shapes = predecessor.parameter_shapes();
    let out_idx = predecessor.typ.output_idx();
    let mut allowed_types = vec![];
    match shapes[out_idx].len() {
        2 => {
            allowed_types.push(PrimitiveSpecType::Matmul { accum: true });
            allowed_types.push(PrimitiveSpecType::Matmul { accum: false });
        }
        4 => {
            allowed_types.push(PrimitiveSpecType::Conv { accum: true });
            allowed_types.push(PrimitiveSpecType::Conv { accum: false });
        }
        _ => {}
    }
    allowed_types.push(PrimitiveSpecType::Move);

    any_with::<PrimitiveBasics>(PrimitiveBasicsArbParams {
        max_size: None,
        first_input_shape: Some(shapes[out_idx].clone()),
        first_input_dtype: Some(predecessor.dtypes[out_idx]),
        allowed_types: Some(allowed_types),
    })
    .prop_filter("Must have at least two parameters to compose", |basics| {
        basics.typ.operand_count() > 1
    })
    .boxed()
}

/// Returns a strategy for generating arbitrary 2- and 3-long Compose Specs.
#[cfg(test)]
fn arb_compose_components() -> impl proptest::strategy::Strategy<Value = Vec<PrimitiveBasics>> {
    use proptest::prelude::*;

    prop_oneof![
        arb_compose_component_innermost()
            .prop_flat_map(|c| {
                let successor = arb_compose_component_successor(&c);
                (successor, Just(c))
            })
            .prop_map(|(s, c)| vec![s, c]),
        arb_compose_component_innermost()
            .prop_flat_map(|c| {
                let successor = arb_compose_component_successor(&c);
                (successor, Just(c))
            })
            .prop_flat_map(|(s, c)| {
                let successor2 = arb_compose_component_successor(&s);
                (successor2, Just(s), Just(c))
            })
            .prop_map(|(s2, s, c)| vec![s2, s, c]),
    ]
}

#[cfg(test)]
pub fn arb_canonical_logical_spec<Tgt: Target>(
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>> {
    use proptest::prelude::*;

    any_with::<LogicalSpec<Tgt>>(max_size.into()).prop_filter_map(
        "Must be possible to canonicalize LogicalSpec",
        |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        },
    )
}

pub(crate) fn dim_range(
    dim_size: DimSize,
    include_end: bool,
    depth: Option<NonZeroU32>,
) -> impl Iterator<Item = DimSize> {
    let start = depth
        .map(|d| dim_size.trailing_zeros().saturating_sub(d.get()))
        .unwrap_or(0);
    let it = (start..)
        .map(|power| 2u32.pow(power))
        .take_while(move |x| *x < dim_size.get())
        .map(|x| DimSize::new(x).unwrap());

    it.chain(once(if include_end { Some(dim_size) } else { None }).flatten())
}

// TODO: Drop in favor of primary output shape inference.
pub fn conv_infer_output_shape(image_shape: &[DimSize], filters_shape: &[DimSize]) -> Shape {
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
                DimSize::new(img_dim.get() - filt_dim.get() + 1).unwrap()
            },
        ))
        .collect()
}

fn compose_parameter_shapes(components: &[PrimitiveBasics]) -> Vec<Shape> {
    debug_assert!(components.len() >= 2);

    let mut result = vec![];
    result.reserve_exact(compose_parameter_count(components));
    compose_parameter_visit(components, |component_idx, parameter_idx| {
        let component = &components[component_idx];
        result.push(component.parameter_shape(parameter_idx));
    });
    result
}

fn compose_parameter_shape(components: &[PrimitiveBasics], mut index: usize) -> Shape {
    for component in &components[..components.len() - 1] {
        let exposed_input_count = component.typ.input_count() - 1;
        if index < exposed_input_count {
            return component.input_shape(index + 1);
        }
        index -= exposed_input_count;
    }
    let last_component = &components[components.len() - 1];
    let last_input_count = last_component.typ.input_count();
    if index < last_input_count {
        return last_component.input_shape(index);
    }
    debug_assert_eq!(index, last_input_count);
    components[0].output_shape()
}

fn compose_parameter_visit(components: &[PrimitiveBasics], mut visitor: impl FnMut(usize, usize)) {
    debug_assert!(components.len() >= 2);

    // TODO: Replace parameter_dtypes with some parameter_count method
    let c0_output_idx = components[0].typ.output_idx();
    for parameter in 1..components[0].parameter_dtypes().len() {
        if parameter != c0_output_idx {
            visitor(0, parameter);
        }
    }

    for (component_idx, c) in components
        .iter()
        .enumerate()
        .take(components.len() - 1)
        .skip(1)
    {
        let output_idx = c.typ.output_idx();
        for parameter in 1..c.parameter_dtypes().len() {
            if parameter != output_idx {
                visitor(component_idx, parameter);
            }
        }
    }

    let cl_output_idx = components[components.len() - 1].typ.output_idx();
    for parameter in 0..components[components.len() - 1].parameter_dtypes().len() {
        if parameter != cl_output_idx {
            visitor(components.len() - 1, parameter);
        }
    }

    visitor(0, c0_output_idx)
}

fn compose_parameter_count(components: &[PrimitiveBasics]) -> usize {
    components
        .iter()
        .map(|c| {
            // TODO: Don't bother with the panic
            c.typ
                .operand_count()
                .checked_sub(2)
                .unwrap_or_else(|| panic!("Component {:?} has too few operands", c))
        })
        .sum::<usize>()
        + 2
}

pub mod macros {
    pub mod internal {
        use crate::common::DimSize;
        pub trait IntoDimSize {
            fn into_dim_size(self) -> DimSize;
        }
        impl IntoDimSize for DimSize {
            fn into_dim_size(self) -> DimSize {
                self
            }
        }
        impl IntoDimSize for u32 {
            fn into_dim_size(self) -> DimSize {
                DimSize::new(self).unwrap()
            }
        }
    }

    #[macro_export]
    macro_rules! shape {
        ($dim:expr; $n:expr) => {{
            use $crate::spec::macros::internal::IntoDimSize;
            // Bind to a variable with an explicit type to help out type inference.
            let sv: $crate::common::Shape = vec![ ($dim).into_dim_size(); $n ];
            sv
        }};
        ($($dim:expr),*$(,)*) => {{
            use $crate::spec::macros::internal::IntoDimSize;
            // Bind to a variable with an explicit type to help out type inference.
            let sv: $crate::common::Shape = vec![ $( ($dim).into_dim_size() ),* ];
            sv
        }};
    }

    #[macro_export]
    macro_rules! lspec {
        ( $typ:tt( $shp:expr, $( ($($opterms:tt)*) ),+, serial ) ) => {{
            lspec!(@inner $typ($shp, $( ($($opterms)*) ),* , true))
        }};
        ( $typ:tt( $shp:expr, $( ($($opterms:tt)*) ),+ ) ) => {{
            lspec!(@inner $typ($shp, $( ($($opterms)*) ),* , false))
        }};
        ( @inner $typ:tt( $shp:expr, $( ($($opterms:tt)*) ),*, $s:literal ) ) => {{
            use $crate::spec::macros::internal::IntoDimSize;

            let auxes = [ $( lspec!(@tensorspecaux_tup $($opterms)*) ),* ];
            let dtypes = auxes.iter().map(|v| v.0.clone()).collect();
            let basics = $crate::spec::PrimitiveBasics {
                typ: lspec!(@primitive_spec_type $typ),
                spec_shape: ($shp).into_iter().map(|x| x.into_dim_size()).collect(),
                dtypes,
            };
            $crate::spec::LogicalSpec::Primitive(
                basics,
                auxes.into_iter().map(|v| v.1).collect(),
                $s,
            )
        }};

        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, c0, ua ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, None, false, false)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, c0 ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, None, false, true)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, ua ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, None, true, false)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, None, true, true)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, $vs:expr, c0, ua ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, Some($vs), false, false)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, $vs:expr, c0 ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, Some($vs), false, true)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, $vs:expr, ua ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, Some($vs), true, false)
        };
        ( @tensorspecaux_tup $dt:tt, $level:expr, $layout:expr, $vs:expr ) => {
            lspec!(@tensorspecaux_tup_inner $dt, $level, $layout, Some($vs), true, true)
        };

        // TODO: Accept contiguousnesses other than fully contig. or not at all.
        ( @tensorspecaux_tup_inner $dt:tt, $level:expr, $layout:expr, $vs:expr,
          $c:literal, $a:literal ) =>
        {{
            let layout: $crate::layout::Layout = $layout;
            let contig = if $c {
                layout.contiguous_full()
            } else {
                layout.contiguous_none()
            };
            (
                lspec!(@dt_convert $dt),
                $crate::tensorspec::TensorSpecAux {
                    contig,
                    aligned: $a,
                    level: $level,
                    layout,
                    vector_size: ($vs).map(|x: u32| {
                        $crate::common::DimSize::try_from(x).unwrap()
                    }),
                },
            )
        }};

        ( @primitive_spec_type Zero ) => {
            $crate::spec::PrimitiveSpecType::Zero
        };
        ( @primitive_spec_type Move ) => {
            $crate::spec::PrimitiveSpecType::Move
        };
        ( @primitive_spec_type Matmul ) => {
            $crate::spec::PrimitiveSpecType::Matmul { accum: false }
        };
        ( @primitive_spec_type MatmulAccum ) => {
            $crate::spec::PrimitiveSpecType::Matmul { accum: true }
        };
        ( @primitive_spec_type Conv ) => {
            $crate::spec::PrimitiveSpecType::Conv { accum: false }
        };
        ( @primitive_spec_type ConvAccum ) => {
            $crate::spec::PrimitiveSpecType::Conv { accum: true }
        };

        ( @dt_convert u8 ) => {
            $crate::common::Dtype::Uint8
        };
        ( @dt_convert i8 ) => {
            $crate::common::Dtype::Sint8
        };
        ( @dt_convert u16 ) => {
            $crate::common::Dtype::Uint16
        };
        ( @dt_convert i16 ) => {
            $crate::common::Dtype::Sint16
        };
        ( @dt_convert u32 ) => {
            $crate::common::Dtype::Uint32
        };
        ( @dt_convert i32 ) => {
            $crate::common::Dtype::Sint32
        };
        ( @dt_convert f32 ) => {
            $crate::common::Dtype::Float32
        };
        ( @dt_convert bf16 ) => {
            $crate::common::Dtype::Bfloat16
        };
        ( @dt_convert $val:expr ) => {
            $val
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imp::{visit_leaves, Impl, ImplExt, ImplNode};
    use crate::memorylimits::{arb_memorylimits_ext, MemVec, MemoryAllocation};
    use crate::scheduling::{Action, ApplyError, TileOut};
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::target::{ArmTarget, CpuMemoryLevel, Target, X86Target};
    use crate::tensorspec::TensorSpecArbMaxShape;
    use crate::utils::{next_binary_power, sum_seqs};
    use crate::{
        layout::row_major,
        target::CpuMemoryLevel::{GL, L1, RF},
    };
    use crate::{lspec, shape};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::iter;

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);

    #[test]
    fn test_lspec_1() {
        let spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [2, 3, 3],
            (u8, GL, row_major(2)),
            (i8, GL, row_major(2), c0),
            (u16, GL, row_major(2), ua),
            serial
        ));
        let lhs = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: true,
            level: GL,
            layout: row_major(2),
            vector_size: None,
        };
        let rhs = TensorSpecAux {
            contig: row_major(2).contiguous_none(),
            aligned: true,
            level: GL,
            layout: row_major(2),
            vector_size: None,
        };
        let out = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: false,
            level: GL,
            layout: row_major(2),
            vector_size: None,
        };
        let expected = LogicalSpec::<X86Target>::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: true },
                spec_shape: shape![2, 3, 3],
                dtypes: vec![Dtype::Uint8, Dtype::Sint8, Dtype::Uint16],
            },
            vec![lhs, rhs, out],
            true,
        );
        assert_eq!(spec, expected);
    }

    #[test]
    fn test_compose_parameters() {
        let spec = compose_logicalspec_test_data();
        let LogicalSpec::Compose {
            components,
            operand_auxes,
            serial_only: _,
        } = &spec
        else {
            unreachable!();
        };

        let expected_parameters: Vec<TensorSpec<X86Target>> = vec![
            TensorSpec::new_noncanon_with_aux(
                components[0].spec_shape[1..].to_vec(),
                components[0].dtypes[1],
                operand_auxes[0].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                components[1].spec_shape[1..].to_vec(),
                components[1].dtypes[1],
                operand_auxes[1].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                components[2].spec_shape[..2].to_vec(),
                components[2].dtypes[0],
                operand_auxes[2].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                components[2].spec_shape[1..].to_vec(),
                components[2].dtypes[1],
                operand_auxes[3].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                vec![components[0].spec_shape[0], components[0].spec_shape[2]],
                components[0].dtypes[2],
                operand_auxes.last().unwrap().clone(),
            ),
        ];

        assert_eq!(spec.parameters(), expected_parameters);
    }

    #[test]
    fn test_compose_input_tiling_inference() {
        let spec = compose_logicalspec_test_data();
        let output_tiling = Tiling::new_simple(shape![32, 128]);
        let expected = TilingInference(vec![
            (Tiling::new_simple(shape![128, 128]), vec![None, Some(1)]),
            (Tiling::new_simple(shape![128, 128]), vec![None, Some(1)]),
            (Tiling::new_simple(shape![32, 128]), vec![Some(0), None]),
            (Tiling::new_simple(shape![128, 128]), vec![None, Some(1)]),
        ]);
        assert_eq!(spec.input_tilings_for_tile_out(&output_tiling), expected);
    }

    proptest! {
        #[test]
        fn test_parameter_shapes_and_every_parameter_shape_matches(
            spec in any::<LogicalSpec<X86Target>>()
        ) {
            let shapes_left = spec.parameter_shapes();
            let shapes_right = (0..spec.operand_count())
                .map(|i| spec.parameter_shape(i))
                .collect::<Vec<_>>();
            prop_assert_eq!(shapes_left, shapes_right);
        }

        #[test]
        fn test_no_action_panics_x86(spec in any::<Spec<X86Target>>()) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        fn test_no_action_panics_arm(spec in any::<Spec<ArmTarget>>()) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        fn test_actions_are_valid_through_consumed_memory_x86(
            logical_spec in arb_canonical_logical_spec::<X86Target>(Some(TEST_SMALL_SIZE))
        ) {
            shared_test_actions_are_valid_through_consumed_memory(logical_spec)
        }

        #[test]
        fn test_actions_are_valid_through_consumed_memory_arm(
            logical_spec in arb_canonical_logical_spec::<X86Target>(Some(TEST_SMALL_SIZE))
        ) {
            shared_test_actions_are_valid_through_consumed_memory(logical_spec)
        }

        #[test]
        fn test_parameters_len_matches_operand_count(
            logical_spec in any::<LogicalSpec<X86Target>>()
        ) {
            prop_assert_eq!(logical_spec.parameters().len(), logical_spec.operand_count());
        }

        #[test]
        fn test_canonicalize_is_noop_if_already_canonical(
            logical_spec in any::<LogicalSpec<X86Target>>()
        ) {
            let mut canonicalized_logical_spec = logical_spec.clone();
            canonicalized_logical_spec.canonicalize().unwrap();
            if logical_spec == canonicalized_logical_spec {
                prop_assert!(
                    logical_spec.is_canonical(),
                    "LogicalSpec::is_canonical was false, but canonicalizing {} was a no-op",
                    logical_spec
                );
            } else {
                prop_assert!(
                    !logical_spec.is_canonical(),
                    "LogicalSpec::is_canonical was true, but {} was canonicalized to {}",
                    logical_spec, canonicalized_logical_spec
                );
            }
        }

        #[test]
        fn test_canonicalizing_specs_canonicalizes_parameters(
            logical_spec in any::<LogicalSpec<X86Target>>()
        ) {
            let mut logical_spec = logical_spec;
            match logical_spec.canonicalize() {
                Ok(()) => {
                    for p in logical_spec.parameters() {
                        let mut recanonicalized = p.clone();
                        recanonicalized.canonicalize().unwrap();
                        assert_eq!(p, recanonicalized);
                    }
                }
                Err(_) => {
                    // If we can't canonicalize, there's nothing to test.
                }
            }
        }

        #[test]
        fn test_canonicalizing_move_tiled_to_one_canonicalizes_parameters(
            spec in
                (1usize..=4)
                    .prop_flat_map(|rank| (Just(rank), 0..rank))
                    .prop_flat_map(|(rank, nonone_idx)| {
                        (vec![1u32..=4; nonone_idx],
                         2u32..=4,
                         vec![1u32..=4; rank - nonone_idx - 1],
                        any::<Dtype>())
                    })
                    .prop_flat_map(|(left, si, right, dtype)| {
                        let shape =
                            left.into_iter().chain(iter::once(si)).chain(right).collect::<Vec<_>>();
                        let basics = PrimitiveBasics {
                            typ: PrimitiveSpecType::Move,
                            spec_shape: Shape::from(shape.into_iter().map(|x| DimSize::new(x).unwrap()).collect::<Vec<_>>()),
                            dtypes: vec![dtype, dtype],
                        };
                        let auxes_strategy = basics
                            .parameter_shapes()
                            .into_iter()
                            .map(|s| any_with::<TensorSpecAux<X86Target>>(TensorSpecArbMaxShape(s)))
                            .collect::<Vec<_>>();
                        (Just(basics), auxes_strategy, any::<bool>())
                    })
                    .prop_filter_map("Spec should be canonical", |(basics, auxes, serial_only)| {
                        let s = Spec(LogicalSpec::Primitive(basics, auxes, serial_only), X86Target::max_mem());
                        if s.is_canonical() {
                            Some(s)
                        } else {
                            None
                        }
                    })
        ) {
            let LogicalSpec::Primitive(PrimitiveBasics { spec_shape, ..}, _, _) = &spec.0 else {
                unreachable!();
            };
            let tile_out_result = Action::TileOut(TileOut::MultiLoop { output_shape: shape![1; spec_shape.len()], parallel: false })
                .apply(&spec).unwrap_or_else(|_| panic!("Couldn't tile Spec {} to single value", spec));
            let ImplNode::SpecApp(child_spec_app) = &tile_out_result.children()[0] else {
                panic!("First child was not a SpecApp; was: {:?}", tile_out_result.children()[0]);
            };
            let mut tiled_logical_spec = child_spec_app.0.0.clone();
            tiled_logical_spec.canonicalize().unwrap();
            assert!(tiled_logical_spec.parameters().iter().all(|p| {
                p.shape().iter().all(|&d| d.get() == 1)
            }));
            assert!(tiled_logical_spec.parameters().iter().all(|p| {
                let mut c = p.clone();
                c.canonicalize().unwrap();
                p == &c
            }));
        }

        #[test]
        fn test_move_actions_never_returns_within_level_copy(spec in any::<Spec<X86Target>>()) {
            for action in X86Target::actions(&spec.0, None) {
                if let Ok(ImplNode::MoveLet(move_let)) = action.apply(&spec) {
                    assert_ne!(&move_let.source_spec, move_let.introduced.spec(),
                        "Copying MoveLet introduced by action {:?}", action);
                }
            }
        }

        #[test]
        fn test_action_applies_everywhere_down_through_peak_memory(
            (spec, action, _, lower_limit) in arb_spec_action_and_lower_limit::<X86Target>()
        ) {
            let lower_spec = Spec(spec.0.clone(), lower_limit);
            assert!(X86Target::actions(&lower_spec.0, None).contains(&action),
                "Action {:?} was not present in lower-limits Spec {:?}",
                action, lower_spec);
        }

        #[test]
        fn test_no_action_produces_same_spec_with_higher_memory_limit_x86(
            spec in any::<Spec<X86Target>>()
        ) {
            shared_test_no_action_produces_same_spec_with_higher_memory_limit(&spec)
        }

        #[test]
        fn test_no_action_produces_same_spec_with_higher_memory_limit_arm(
            spec in any::<Spec<ArmTarget>>()
        ) {
            shared_test_no_action_produces_same_spec_with_higher_memory_limit(&spec)
        }

        #[test]
        fn test_actions_produce_canonical_subspecs(
            spec in any::<Spec<X86Target>>()
        ) {
            X86Target::actions(&spec.0, None).for_each(|action| {
                let Ok(applied) = action.apply(&spec) else {
                    return;
                };
                visit_leaves(&applied, &mut |leaf| {
                    if let ImplNode::SpecApp(spec_app) = leaf {
                        assert!(
                            spec_app.0.is_canonical(),
                            "Action {:?} applied to {} produced non-canonical {} (should be {})",
                            action,
                            spec,
                            spec_app.0,
                            {
                                let mut c = spec_app.0.clone();
                                c.canonicalize().unwrap();
                                c
                            }
                        );
                    }
                    true
                });
            });
        }

        #[test]
        fn test_primitivebasicsbimap_is_invertible(basics in any::<PrimitiveBasics>()) {
            // TODO: Also test binary_scale_shapes = true
            let bimap = PrimitiveBasicsBimap {
                binary_scale_shapes: false,
            };
            let projection = BiMap::apply(&bimap, &basics);
            let reversed = BiMap::apply_inverse(&bimap, &projection);
            assert_eq!(basics, reversed);
        }

        #[test]
        fn test_parameters_match_parameter_shapes(spec in any::<LogicalSpec<X86Target>>()) {
            let parameters = spec.parameters();
            let parameter_shapes = spec.parameter_shapes();
            prop_assert_eq!(parameters.len(), parameter_shapes.len());
            for (p, s) in parameters.iter().zip(&parameter_shapes) {
                assert_eq!(p.shape(), s);
            }
        }

        #[test]
        fn test_replace_io_noop(logical_spec in any::<LogicalSpec<X86Target>>()) {
            let mut replaced = logical_spec.clone();
            replaced.replace_io(&logical_spec.parameters());
            prop_assert_eq!(logical_spec, replaced);
        }

        #[test]
        fn test_bufferized_compose_parameters_match_pipeline_parameters(
            tinp in any::<Spec<X86Target>>()
                .prop_filter("Spec was not Compose", |s| matches!(s.0, LogicalSpec::Compose { .. }))
                .prop_flat_map(|mut s| {
                    s.canonicalize().unwrap();
                    let LogicalSpec::Compose { components, .. } = &s.0 else {
                        unreachable!();
                    };
                    let components_len = components.len();
                    (Just(s), 0..(components_len - 1))
                })
                .prop_filter("Spec intermediate didn't fit in RF", |(s, index)| {
                    let LogicalSpec::Compose { components, .. } = &s.0 else {
                        unreachable!();
                    };
                    let buf_volume: u32 =
                        components[*index].input_shapes()[0].iter().map(|d| d.get()).product();
                    let value_size = u32::from(components[*index].dtypes[0].size());
                    let bytes_needed = buf_volume * value_size;
                    let rf_idx =
                        X86Target::levels().iter().position(|l| l == &CpuMemoryLevel::RF).unwrap();
                    let remaining_in_rf = match s.1.clone().into_standard() {
                        MemoryLimits::Standard(standard) => standard.get_unscaled(rf_idx),
                    };
                    u64::from(bytes_needed) <= remaining_in_rf
                })
        ) {
            let (compose_spec, index) = tinp;
            let LogicalSpec::Compose { components, .. } = &compose_spec.0 else {
                unreachable!();
            };

            let pipeline = compose_spec.bufferize(
                index,
                CpuMemoryLevel::RF,
                row_major(components[index].input_shapes()[0].len().try_into().unwrap()),
                None
            );
            let spec_parameters = compose_spec.0.parameters();
            let pipeline_parameters = pipeline.parameters().cloned().collect::<Vec<_>>();
            prop_assert_eq!(spec_parameters, pipeline_parameters);
        }
    }

    #[test]
    fn test_dim_range_with_odd_max() {
        assert_eq!(
            dim_range(nz!(3u32), false, None).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32)]
        );
        assert_eq!(
            dim_range(nz!(3u32), true, None).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(3u32)]
        );

        assert_eq!(
            dim_range(nz!(7u32), false, None).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(4u32)]
        );
        assert_eq!(
            dim_range(nz!(7u32), true, None).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(4u32), nz!(7u32)]
        );
    }

    fn shared_test_no_action_panics<Tgt: Target>(spec: Spec<Tgt>) {
        for action in Tgt::actions(&spec.0, None) {
            let _ = action.apply(&spec);
        }
    }

    fn shared_test_no_action_produces_same_spec_with_higher_memory_limit<Tgt: Target>(
        spec: &Spec<Tgt>,
    ) {
        Tgt::actions(&spec.0, None).for_each(|action| {
            let Ok(applied) = action.apply(spec) else {
                return;
            };
            visit_leaves(&applied, &mut |leaf| {
                if let ImplNode::SpecApp(spec_app) = leaf {
                    assert!(
                        spec.0 != spec_app.0 .0 || spec_app.0 .1 <= spec.1,
                        "Action {:?} produced the same Spec {} with higher memory limit {}",
                        action,
                        spec,
                        spec_app.0 .1
                    );
                }
                true
            });
        });
    }

    /// Asserts that actions appear at all memory limits at and above memory consumed.
    fn shared_test_actions_are_valid_through_consumed_memory<Tgt: Target>(
        logical_spec: LogicalSpec<Tgt>,
    ) {
        // If an action consumes x bytes, then it should be valid for any Spec with the same logical
        // Spec at that memory limit and up.
        let MemoryLimits::Standard(maxes_vec) = Tgt::max_mem();
        let mut maxes = Vec::with_capacity(maxes_vec.len());
        for binary_scaled in maxes_vec.iter_binary_scaled() {
            maxes.push(u32::from(binary_scaled));
        }
        // Zero out levels which are slower than all present operands' levels.
        let parameters = logical_spec.parameters();
        for (level_idx, level) in Tgt::levels().into_iter().enumerate() {
            if parameters.iter().all(|p| p.level() < level) {
                maxes[level_idx] = 0;
            }
        }

        // The list of actions depends only on the logical Spec. Filtering by memory limit happens
        // at application. So it's safe to just collect the list of actions once, up front.
        let mut unseen_actions = Tgt::actions(&logical_spec, None).collect::<Vec<_>>();

        let mut shared_spec = Spec(logical_spec, MemoryLimits::Standard(MemVec::zero::<Tgt>()));
        let mut diagonal_idx = 0;
        loop {
            let mut empty = true;
            for pt in sum_seqs(&maxes, diagonal_idx) {
                empty = false;
                shared_spec.1 = MemoryLimits::Standard(MemVec::new_from_binary_scaled(
                    pt.iter()
                        .map(|&p| u8::try_from(p).unwrap())
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                ));
                let MemoryLimits::Standard(limits_memvec) = &shared_spec.1;
                // TODO: Assert that nothing disappears?
                for i in (0..unseen_actions.len()).rev() {
                    match unseen_actions[i].apply(&shared_spec) {
                        Ok(applied) => {
                            unseen_actions.swap_remove(i);
                            // TODO: Should we also assert that applying the same action at each level
                            //   doesn't actually accumulate additional memory?.
                            // TODO: Can we assert that the change in peak memory is exactly the
                            //   additional amount at the limit?.
                            // TODO: Assert here that the min of each level-wise limit is zero.
                            assert_eq!(&applied.peak_memory(), limits_memvec);
                        }
                        Err(ApplyError::NotApplicable(_)) => {}
                        Err(ApplyError::SpecNotCanonical) => panic!(),
                    }
                }
            }
            if empty {
                break;
            }
            diagonal_idx += 1;
        }
    }

    fn arb_spec_action_and_lower_limit<Tgt: Target>(
    ) -> impl Strategy<Value = (Spec<Tgt>, Action<Tgt>, ImplNode<Tgt>, MemoryLimits)> {
        arb_canonical_spec::<Tgt>(None, None)
            .prop_filter_map("Spec had zero applicable actions", |spec| {
                let applied_actions = Tgt::actions(&spec.0, None)
                    .filter_map(|a| match a.apply(&spec) {
                        Ok(applied) => Some((a, applied)),
                        Err(ApplyError::NotApplicable(_)) => None,
                        Err(ApplyError::SpecNotCanonical) => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                if applied_actions.is_empty() {
                    None
                } else {
                    Some((spec, applied_actions))
                }
            })
            .prop_flat_map(|(spec, applied_actions)| {
                (Just(spec), proptest::sample::select(applied_actions))
            })
            .prop_flat_map(|(spec, (action, applied))| {
                let lower_bound = match applied.memory_allocated() {
                    MemoryAllocation::Simple(allocated) => allocated,
                    MemoryAllocation::Pipeline {
                        intermediate_consumption,
                    } if intermediate_consumption.len() == 1 => intermediate_consumption[0],
                    _ => todo!(),
                };
                let MemoryLimits::Standard(limits_memvec) = &spec.1;
                let lower_limit_strategy = arb_memorylimits_ext(
                    &MemVec::new(lower_bound.map(next_binary_power)),
                    limits_memvec,
                );
                (
                    Just(spec),
                    Just(action),
                    Just(applied),
                    lower_limit_strategy,
                )
            })
    }

    fn compose_logicalspec_test_data() -> LogicalSpec<X86Target> {
        let basic0 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![128, 128, 128],
            dtypes: vec![Dtype::Uint8, Dtype::Uint16, Dtype::Uint32],
        };
        let basic1 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![128, 128, 128],
            dtypes: vec![Dtype::Uint32, Dtype::Uint16, Dtype::Uint8],
        };
        let basic2 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![128, 128, 128],
            dtypes: vec![Dtype::Uint8, Dtype::Uint8, Dtype::Uint32],
        };

        let aux0_1 = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: true,
            level: GL,
            layout: row_major(2),
            vector_size: None,
        };
        let aux1_1 = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: true,
            level: L1,
            layout: row_major(2),
            vector_size: None,
        };
        let aux2_0 = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: false,
            level: GL,
            layout: row_major(2),
            vector_size: None,
        };
        let aux2_1 = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: false,
            level: L1,
            layout: row_major(2),
            vector_size: None,
        };
        let aux0_out = TensorSpecAux {
            contig: row_major(2).contiguous_full(),
            aligned: true,
            level: RF,
            layout: row_major(2),
            vector_size: None,
        };

        LogicalSpec::<X86Target>::Compose {
            components: vec![basic0.clone(), basic1.clone(), basic2.clone()],
            operand_auxes: vec![aux0_1, aux1_1, aux2_0, aux2_1, aux0_out],
            serial_only: false,
        }
    }
}
