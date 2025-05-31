use crate::common::{DimSize, Dtype, Shape};
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::row_major;
use crate::memorylimits::{MemoryLimits, MemoryLimitsBimap};
use crate::target::Target;
use crate::tensorspec::{self, check_tensor_vector_size, TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::{bit_length_inverse, bit_length_u32, join_into_string, prev_power_of_two_u32};

use itertools::{izip, Itertools};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};

use std::fmt;
use std::fmt::Display;
use std::iter::once;
use std::iter::Iterator;
use std::marker::PhantomData;
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
        // Components contain Spec shapes and dtypes, which can be partially inferred, so the
        // following stores a little bit of redundant information.
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
    /// Consumes a tensor of shape dim0 x .. x dimN and casts to 1 x dim0 x .. x dimN.
    OnePrefix,
    Fill {
        value: FillValue,
    },
    Move,
    Matmul {
        accum: bool,
    },
    Conv {
        accum: bool,
    },
    Broadcast {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        dim: u8,
    },
    /// Divides the first argument by the second, elementwise, into the third argument.
    DivideVec,
    /// Broadcasts a tensor (second argument) across `scan_dim`, dividing the first argument. Output
    /// is in the third argument.
    ///
    /// Implementations of this Spec do not multiply by the reciprocal of the divisor. They divide
    /// directly for improved precision.
    DivideVecScalar {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
    },
    /// Computes y_i = (e^(x_i - max_k x_k)) / (Σ_j e^(x_j - max_k x_k)).
    Softmax {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
    },
    /// Computes Softmax with x, the denominator, and `max_k x_k` given as inputs.
    SoftmaxComplete {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
    },
    /// Computes `Σ_j e^(x_j - max_k x_k)` and `max_k x_k` as outputs.
    ///
    /// The parameters are the input, which can be of arbitrary rank, followed by the output
    /// denominator and maximum.
    SoftmaxDenominatorAndMax {
        // TODO: Swap variant name order to match args
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
    },
    /// Computes `Σ_j e^(x_j - max_k x_k)` and each `e^(x_i - max_k x_k)` as outputs.
    ///
    /// First arg is input, second is denominator, then unscaled scores.
    SoftmaxDenominatorAndUnscaled {
        // TODO: Swap variant name order to match args
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
        /// Whether to sum into the denominator tensor.
        accum: bool, // TODO: Does this have a point?
    },
    /// Computes unnormalized softmax scores and the scaling denominator from the input and max.
    ///
    /// First arg is input, second is maxes, third is denominators, then unscaled scores.
    SoftmaxDenominatorAndUnscaledFromMax {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
        /// Whether to sum into the denominator tensor.
        accum: bool,
    },
    Max {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        dim: u8,
        accum: bool,
    },
    SoftmaxDenominator {
        #[cfg_attr(test, proptest(strategy = "0..4u8"))]
        scan_dim: u8,
        accum: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum FillValue {
    Zero,
    NegInf,
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
    marker: PhantomData<(Tgt, A, Aa)>,
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
    #[error("Non-head component of the Compose causes side effects")]
    SideEffectingComponent,
}

#[cfg(test)]
#[derive(Debug, Default, Clone)]
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
                    let [b, m, k, n] = basics.spec_shape[..] else {
                        unreachable!();
                    };
                    if basics.dtypes.iter().any(|&t| t != Dtype::Float32) {
                        return None;
                    }
                    Some(
                        2 * u64::from(b.get())
                            * u64::from(m.get())
                            * u64::from(k.get())
                            * u64::from(n.get()),
                    )
                }
                PrimitiveSpecType::Softmax { scan_dim } => {
                    const EXP_FLOPS: u64 = 13;
                    const OTHER_FLOPS: u64 = 2;
                    const OTHER_MIN_ONE_FLOPS: u64 = 2;
                    if basics.dtypes.iter().any(|&t| t != Dtype::Float32) {
                        return None;
                    }
                    let volume = basics
                        .spec_shape
                        .iter()
                        .map(|d| u64::from(d.get()))
                        .product::<u64>();
                    let volume_reduced = basics
                        .spec_shape
                        .iter()
                        .enumerate()
                        .map(|(dim, d)| {
                            if dim == usize::from(scan_dim) {
                                u64::from(d.get() - 1)
                            } else {
                                u64::from(d.get())
                            }
                        })
                        .product::<u64>();
                    Some(
                        ((EXP_FLOPS + OTHER_FLOPS) * volume)
                            + (OTHER_MIN_ONE_FLOPS * volume_reduced),
                    )
                }
                _ => None,
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

#[cfg(test)]
pub fn arb_canonical_primitive_spec<Tgt: Target>(
    max_size: Option<DimSize>,
    max_memory: Option<u64>,
) -> impl proptest::strategy::Strategy<Value = Spec<Tgt>> {
    use crate::memorylimits::arb_memorylimits;
    use proptest::prelude::*;

    // Optionally lower the max memory limits.
    let MemoryLimits::Standard(mut max_memory_limits) = Tgt::max_mem();
    if let Some(lower_max) = max_memory {
        max_memory_limits = max_memory_limits.map(|v| v.min(lower_max));
    }

    (
        arb_canonical_primitive_logical_spec::<Tgt>(max_size),
        arb_memorylimits::<Tgt>(&max_memory_limits),
    )
        .prop_map(|(logical_spec, mem_limits)| Spec(logical_spec, mem_limits))
        .prop_filter_map("Must be possible to canonicalize Spec", |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        })
}

#[cfg(test)]
pub fn arb_canonical_compose_spec<Tgt: Target>(
    max_size: Option<DimSize>,
    max_memory: Option<u64>,
) -> impl proptest::strategy::Strategy<Value = Spec<Tgt>> {
    use crate::memorylimits::arb_memorylimits;
    use proptest::prelude::*;

    // Optionally lower the max memory limits.
    let MemoryLimits::Standard(mut max_memory_limits) = Tgt::max_mem();
    if let Some(lower_max) = max_memory {
        max_memory_limits = max_memory_limits.map(|v| v.min(lower_max));
    }

    (
        arb_canonical_compose_logical_spec::<Tgt>(max_size),
        arb_memorylimits::<Tgt>(&max_memory_limits),
    )
        .prop_map(|(logical_spec, mem_limits)| Spec(logical_spec, mem_limits))
        .prop_filter_map("Must be possible to canonicalize Spec", |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        })
}

impl PrimitiveBasics {
    pub fn replace_io(&mut self, new_operands: &[(&[DimSize], Dtype)]) {
        self.dtypes = new_operands.iter().map(|o| o.1).collect();
        self.spec_shape = self
            .typ
            .shape_from_parameters(new_operands.iter().map(|t| t.0));
    }

    pub fn aux_from_operand_auxes<'a, Tgt, I>(&self, operand_auxes: I) -> Vec<TensorSpecAux<Tgt>>
    where
        Tgt: Target,
        I: IntoIterator<Item = &'a TensorSpecAux<Tgt>> + 'a,
    {
        operand_auxes.into_iter().cloned().collect()
    }

    pub(crate) fn input_shape(&self, index: usize) -> Shape {
        self.parameter_shape(self.input_idx(index))
    }

    pub(crate) fn input_dtype(&self, index: usize) -> Dtype {
        self.parameter_dtype(self.input_idx(index))
    }

    pub(crate) fn input_shapes(&self) -> Vec<Shape> {
        let param_count = self.typ.operand_count();
        let mut shapes = Vec::with_capacity(param_count);
        for i in 0..param_count {
            if !self.typ.parameter_is_output(i) {
                shapes.push(self.parameter_shape(i));
            }
        }
        shapes
    }

    pub(crate) fn input_idx(&self, index: usize) -> usize {
        if let Some(output_idx) = self.typ.unique_output_index() {
            if index < output_idx {
                index
            } else {
                index + 1
            }
        } else {
            todo!()
        }
    }

    pub fn parameter_shapes(&self) -> Vec<Shape> {
        (0..self.typ.operand_count())
            .map(|i| self.parameter_shape(i))
            .collect()
    }

    pub fn parameter_shape(&self, idx: usize) -> Shape {
        match self.typ {
            PrimitiveSpecType::OnePrefix => match idx {
                0 => self.spec_shape.clone(),
                1 => {
                    let mut shape = vec![];
                    shape.reserve_exact(self.spec_shape.len() + 1);
                    shape.push(nz!(1u32));
                    shape.extend_from_slice(&self.spec_shape);
                    shape
                }
                _ => panic!("OnePrefix has only 2 parameters"),
            },
            PrimitiveSpecType::Matmul { .. } => match idx {
                0 => vec![self.spec_shape[0], self.spec_shape[1], self.spec_shape[2]],
                1 => vec![self.spec_shape[0], self.spec_shape[2], self.spec_shape[3]],
                2 => vec![self.spec_shape[0], self.spec_shape[1], self.spec_shape[3]],
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
            PrimitiveSpecType::Broadcast { dim } => match idx {
                0 => {
                    let mut shape = self.spec_shape.clone();
                    shape[usize::from(dim)] = nz!(1u32);
                    shape
                }
                1 => self.spec_shape.clone(),
                _ => panic!("Broadcast has only 2 parameters"),
            },
            PrimitiveSpecType::DivideVec => {
                if idx > 3 {
                    panic!("DivideVec has only 3 parameters")
                }
                self.spec_shape.clone()
            }
            PrimitiveSpecType::DivideVecScalar { scan_dim } => match idx {
                0 | 2 => self.spec_shape.clone(),
                1 => {
                    let mut shape = self.spec_shape.clone();
                    shape[usize::from(scan_dim)] = nz!(1u32);
                    shape
                }
                _ => panic!("DivideVecScalar has only 2 parameters"),
            },
            PrimitiveSpecType::Softmax { .. } => match idx {
                0 | 1 => self.spec_shape.to_vec(),
                _ => panic!("Softmax has only 2 parameters"),
            },
            PrimitiveSpecType::SoftmaxComplete { scan_dim } => match idx {
                0 | 3 => self.spec_shape.clone(),
                1 | 2 => {
                    let mut reduced = self.spec_shape.clone();
                    reduced[usize::from(scan_dim)] = nz!(1u32);
                    reduced
                }
                _ => panic!("SoftmaxComplete has only 4 parameters"),
            },
            PrimitiveSpecType::SoftmaxDenominator { scan_dim: dim, .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim: dim, .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim: dim, .. }
            | PrimitiveSpecType::Max { dim, .. } => match idx {
                0 => self.spec_shape.clone(),
                x if x < self.typ.operand_count() => {
                    let mut reduced = self.spec_shape.clone();
                    reduced[usize::from(dim)] = nz!(1u32);
                    reduced
                }
                _ => panic!(),
            },
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, .. } => match idx {
                0 | 3 => self.spec_shape.clone(),
                1 | 2 => {
                    let mut reduced = self.spec_shape.clone();
                    reduced[usize::from(scan_dim)] = nz!(1u32);
                    reduced
                }
                _ => panic!("SoftmaxDenominatorAndUnscaledFromMax has only 4 parameters"),
            },
            PrimitiveSpecType::Move => match idx {
                0 | 1 => self.spec_shape.clone(),
                _ => panic!("Move has only 2 parameters"),
            },
            PrimitiveSpecType::Fill { value: _ } => match idx {
                0 => self.spec_shape.clone(),
                _ => panic!("Zero has only 1 parameter"),
            },
        }
    }

    pub fn unique_output_shape(&self) -> Option<Shape> {
        self.typ
            .unique_output_index()
            .map(|idx| self.parameter_shape(idx))
    }

    pub fn parameter_dtypes(&self) -> Vec<Dtype> {
        self.dtypes.clone()
    }

    pub fn parameter_dtype(&self, idx: usize) -> Dtype {
        self.dtypes[idx]
    }

    pub fn causes_side_effects(&self) -> bool {
        match self.typ {
            PrimitiveSpecType::Matmul { accum }
            | PrimitiveSpecType::Conv { accum }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { accum, .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. }
            | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
            | PrimitiveSpecType::Max { accum, .. } => accum,
            PrimitiveSpecType::Fill { .. } => true,
            PrimitiveSpecType::OnePrefix
            | PrimitiveSpecType::Softmax { .. }
            | PrimitiveSpecType::SoftmaxComplete { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { .. }
            | PrimitiveSpecType::DivideVec
            | PrimitiveSpecType::DivideVecScalar { .. }
            | PrimitiveSpecType::Broadcast { .. }
            | PrimitiveSpecType::Move => false,
        }
    }

    pub(crate) fn initial_accumulating_value_for_output(&self, index: usize) -> Option<FillValue> {
        let PrimitiveBasics {
            typ,
            spec_shape: _,
            dtypes,
        } = self;
        debug_assert!(typ.parameter_is_output(index));
        match typ {
            PrimitiveSpecType::Matmul { accum: true }
            | PrimitiveSpecType::Conv { accum: true }
            | PrimitiveSpecType::SoftmaxDenominator { accum: true, .. } => Some(FillValue::Zero),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum: true, .. } => {
                match index {
                    2 => Some(FillValue::Zero),
                    _ => None,
                }
            }
            PrimitiveSpecType::Max { accum: true, .. } => match dtypes[index] {
                Dtype::Sint8 | Dtype::Sint16 | Dtype::Sint32 => {
                    todo!("support min-value filling for signed integers")
                }
                Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 => Some(FillValue::Zero),
                Dtype::Float32 | Dtype::Bfloat16 => Some(FillValue::NegInf),
            },
            PrimitiveSpecType::OnePrefix
            | PrimitiveSpecType::Fill { .. }
            | PrimitiveSpecType::Move
            | PrimitiveSpecType::Matmul { accum: false }
            | PrimitiveSpecType::Conv { accum: false }
            | PrimitiveSpecType::Softmax { .. }
            | PrimitiveSpecType::SoftmaxComplete { .. }
            | PrimitiveSpecType::Max { accum: false, .. }
            | PrimitiveSpecType::SoftmaxDenominator { accum: false, .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum: false, .. }
            | PrimitiveSpecType::Broadcast { dim: _ } => panic!("Not an accumulating Spec"),
            PrimitiveSpecType::DivideVec => todo!(),
            PrimitiveSpecType::DivideVecScalar { .. } => todo!(),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. } => todo!(),
        }
    }

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> Option<TilingInference> {
        Some(match (self, smaller_output.is_simple()) {
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { .. },
                    spec_shape,
                    ..
                },
                true,
            )
            | (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: false },
                    spec_shape,
                    ..
                },
                _,
            ) => TilingInference(vec![
                (
                    Tiling::new_sliding(
                        vec![
                            smaller_output.shape()[0],
                            smaller_output.shape()[1],
                            spec_shape[2],
                        ],
                        vec![
                            smaller_output.step_sizes()[0],
                            smaller_output.step_sizes()[1],
                            spec_shape[2],
                        ],
                    ),
                    vec![Some(0), Some(1), None],
                ),
                (
                    Tiling::new_sliding(
                        vec![
                            smaller_output.shape()[0],
                            spec_shape[2],
                            smaller_output.shape()[2],
                        ],
                        vec![
                            smaller_output.step_sizes()[0],
                            spec_shape[2],
                            smaller_output.step_sizes()[2],
                        ],
                    ),
                    vec![Some(0), None, Some(2)],
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
                    typ: PrimitiveSpecType::Softmax { scan_dim, .. },
                    spec_shape,
                    ..
                },
                _,
            ) => {
                debug_assert_eq!(smaller_output.shape().len(), spec_shape.len());
                let scan_dim_us = usize::from(*scan_dim);
                let tiled_input = smaller_output.shape().clone();
                if tiled_input[scan_dim_us] != spec_shape[scan_dim_us] {
                    // Softmax's scan dimension cannot be tiled
                    return None;
                }
                let mut tiled_step_sizes = smaller_output.step_sizes().to_vec();
                tiled_step_sizes[scan_dim_us] = spec_shape[scan_dim_us];
                let mut bindings = (0..smaller_output.shape().len())
                    .map(|d| Some(d.try_into().unwrap()))
                    .collect::<Vec<_>>();
                bindings[scan_dim_us] = None;

                TilingInference(vec![(
                    Tiling::new_sliding(tiled_input, tiled_step_sizes),
                    bindings,
                )])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxComplete { scan_dim },
                    spec_shape,
                    ..
                },
                _,
            ) => {
                debug_assert_eq!(smaller_output.shape().len(), spec_shape.len());
                let x_tiling_tuple = passthrough_tiling_tuple(smaller_output);
                let denom_tiling_tuple =
                    one_reduced_dimension_tiling_tuple(smaller_output, *scan_dim);
                let max_tiling_tuple = denom_tiling_tuple.clone();
                TilingInference(vec![x_tiling_tuple, denom_tiling_tuple, max_tiling_tuple])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::SoftmaxDenominator { scan_dim, .. },
                    spec_shape,
                    ..
                },
                _,
            ) => {
                let bindings = (0..smaller_output.shape().len())
                    .map(|d| Some(d.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let mut x_shape = smaller_output.shape().clone();
                x_shape[usize::from(*scan_dim)] = spec_shape[usize::from(*scan_dim)];
                let mut x_steps = smaller_output.step_sizes().to_vec();
                x_steps[usize::from(*scan_dim)] = spec_shape[usize::from(*scan_dim)];
                let maxes_shape = smaller_output.shape().clone();
                let maxes_steps = smaller_output.step_sizes().to_vec();
                TilingInference(vec![
                    (Tiling::new_sliding(x_shape, x_steps), bindings.clone()),
                    (Tiling::new_sliding(maxes_shape, maxes_steps), bindings),
                ])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Max { dim, accum: _ },
                    spec_shape,
                    dtypes: _,
                },
                _,
            ) => {
                let mut input_shape = smaller_output.shape().clone();
                input_shape[usize::from(*dim)] = spec_shape[usize::from(*dim)];
                let mut input_steps = smaller_output.step_sizes().to_vec();
                input_steps[usize::from(*dim)] = spec_shape[usize::from(*dim)];
                TilingInference(vec![(
                    Tiling::new_sliding(input_shape, input_steps),
                    (0..smaller_output.shape().len())
                        .map(|d| Some(d.try_into().unwrap()))
                        .collect(),
                )])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::DivideVec,
                    spec_shape: _,
                    dtypes: _,
                },
                _,
            ) => TilingInference(vec![
                passthrough_tiling_tuple(smaller_output),
                passthrough_tiling_tuple(smaller_output),
            ]),
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::DivideVecScalar { scan_dim },
                    spec_shape: _,
                    dtypes: _,
                },
                _,
            ) => {
                let input_tiling = passthrough_tiling_tuple(smaller_output);
                let scalars_tiling = one_reduced_dimension_tiling_tuple(smaller_output, *scan_dim);
                TilingInference(vec![input_tiling, scalars_tiling])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::OnePrefix,
                    ..
                },
                _,
            ) => {
                let shape = smaller_output.shape()[1..].to_vec();
                let steps = smaller_output.step_sizes()[1..].to_vec();
                let bindings = (1..shape.len())
                    .map(|v| Some(v.try_into().unwrap()))
                    .collect();
                TilingInference(vec![(Tiling::new_sliding(shape, steps), bindings)])
            }
            (
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Broadcast { dim },
                    ..
                },
                _,
            ) => {
                let input_tiling = one_reduced_dimension_tiling_tuple(smaller_output, *dim);
                TilingInference(vec![input_tiling])
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
                    typ: PrimitiveSpecType::Fill { value: _ },
                    ..
                },
                _,
            ) => TilingInference(vec![]),
            _ => unimplemented!(
                "Output tiling not implemented for {:?} and {:?}",
                self,
                smaller_output
            ),
        })
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
            Some(allowed_types) => proptest::sample::select(allowed_types).boxed(),
            None => any::<PrimitiveSpecType>().boxed(),
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
                        let (b, m, k) = match args.first_input_shape.as_deref() {
                            Some([b, m, k]) => (
                                Just(b.get()).sboxed(),
                                Just(m.get()).sboxed(),
                                Just(k.get()).sboxed(),
                            ),
                            Some(_) => panic!("Matmul requires a rank-3 first input"),
                            None => (
                                (1..=max_size).sboxed(),
                                (1..=max_size).sboxed(),
                                (1..=max_size).sboxed(),
                            ),
                        };
                        vec![b, m, k, (1..=max_size).sboxed()].sboxed()
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
                    PrimitiveSpecType::Softmax { scan_dim, .. }
                    | PrimitiveSpecType::SoftmaxComplete { scan_dim, .. }
                    | PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim, .. }
                    | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, .. }
                    | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                        scan_dim, ..
                    }
                    | PrimitiveSpecType::SoftmaxDenominator { scan_dim, .. }
                    | PrimitiveSpecType::DivideVecScalar { scan_dim }
                    | PrimitiveSpecType::Max { dim: scan_dim, .. } => {
                        match args.first_input_shape.as_deref() {
                            Some(s) => {
                                assert!(usize::from(scan_dim) < s.len());
                                s.iter().map(|d| Just(d.get())).collect::<Vec<_>>().sboxed()
                            }
                            None => {
                                // TODO: assert reduction and max size relationship
                                let r = usize::from(scan_dim);
                                ((r + 2)..(r + 3))
                                    .prop_flat_map(move |tensor_rank| {
                                        proptest::collection::vec(1..=max_size, tensor_rank)
                                            .sboxed()
                                    })
                                    .sboxed()
                            }
                        }
                    }
                    PrimitiveSpecType::OnePrefix
                    | PrimitiveSpecType::Move
                    | PrimitiveSpecType::Fill { .. }
                    | PrimitiveSpecType::DivideVec => match args.first_input_shape.as_deref() {
                        Some(s) => s.iter().map(|d| Just(d.get())).collect::<Vec<_>>().sboxed(),
                        None => (1..=4usize)
                            .prop_flat_map(move |tensor_rank| {
                                proptest::collection::vec(1..=max_size, tensor_rank).sboxed()
                            })
                            .sboxed(),
                    },
                    PrimitiveSpecType::Broadcast { dim } => {
                        match args.first_input_shape.as_deref() {
                            Some(s) => {
                                assert!(usize::from(dim) < s.len());
                                s.iter()
                                    .enumerate()
                                    .map(|(i, d)| {
                                        if i == usize::from(dim) {
                                            (1..=8u32).sboxed()
                                        } else {
                                            Just(d.get()).sboxed()
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .sboxed()
                            }
                            None => {
                                let r = usize::from(dim);
                                ((r + 1)..(r + 2))
                                    .prop_flat_map(move |tensor_rank| {
                                        proptest::collection::vec(1..=max_size, tensor_rank)
                                            .sboxed()
                                    })
                                    .sboxed()
                            }
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

// TODO: Move
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandDirection {
    In,
    Out,
    InOut,
}

impl PrimitiveSpecType {
    /// Computes a spec_shape from an [Iterator] of parameter shapes.
    fn shape_from_parameters<'a, I: Iterator<Item = &'a [DimSize]>>(
        &self,
        mut parameter_shapes: I,
    ) -> Shape {
        match self {
            PrimitiveSpecType::OnePrefix => {
                let input = parameter_shapes.next().unwrap();
                let _output = parameter_shapes.next().unwrap();
                input.into()
            }
            PrimitiveSpecType::Matmul { accum: _ } => {
                let lhs = parameter_shapes.next().unwrap();
                let rhs = parameter_shapes.next().unwrap();
                let out = parameter_shapes.next().unwrap();
                assert_eq!(lhs[0], rhs[0]);
                assert_eq!(lhs[0], out[0]);
                assert_eq!(lhs[1], out[1]);
                assert_eq!(rhs[2], out[2]);
                assert_eq!(lhs[2], rhs[1]);
                vec![lhs[0], lhs[1], lhs[2], rhs[2]]
            }
            PrimitiveSpecType::Conv { accum: _ } => {
                let lhs = parameter_shapes.next().unwrap();
                let rhs = parameter_shapes.next().unwrap();
                let _out = parameter_shapes.next().unwrap();

                let [b, c, h, w] = *lhs else {
                    panic!();
                };
                let [f, alt_c, fh, fw] = *rhs else { panic!() };
                assert_eq!(c, alt_c);
                // TODO: Assert consistency with the output as well
                vec![b, f, c, h, w, fh, fw]
            }
            PrimitiveSpecType::Broadcast { dim } => {
                let inp = parameter_shapes.next().unwrap();
                let out = parameter_shapes.next().unwrap();
                assert_eq!(inp[..usize::from(*dim)], out[..usize::from(*dim)]);
                assert_eq!(inp[usize::from(*dim)], nz!(1u32));
                assert_eq!(inp[usize::from(*dim) + 1..], out[usize::from(*dim) + 1..]);
                out.into()
            }
            PrimitiveSpecType::DivideVec => {
                let numer = parameter_shapes.next().unwrap();
                let denom = parameter_shapes.next().unwrap();
                let out = parameter_shapes.next().unwrap();

                assert_eq!(numer, denom);
                assert_eq!(numer, out);
                numer.into()
            }
            PrimitiveSpecType::DivideVecScalar { scan_dim } => {
                let numer = parameter_shapes.next().unwrap();
                let denom = parameter_shapes.next().unwrap();
                let out = parameter_shapes.next().unwrap();
                assert_eq!(numer, out);
                assert!(
                    denom.iter().enumerate().all(|(i, &dim)| {
                        if i == usize::from(*scan_dim) {
                            dim == nz!(1u32)
                        } else {
                            dim == numer[i]
                        }
                    }),
                    "surprise second parameter shape: {denom:?}",
                );
                numer.into()
            }
            PrimitiveSpecType::Move => {
                let src = parameter_shapes.next().unwrap();
                let dest = parameter_shapes.next().unwrap();
                assert_eq!(src, dest);
                src.into()
            }
            PrimitiveSpecType::Softmax { .. }
            | PrimitiveSpecType::SoftmaxComplete { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { .. }
            | PrimitiveSpecType::SoftmaxDenominator { .. }
            | PrimitiveSpecType::Max { .. }
            | PrimitiveSpecType::Fill { .. } => parameter_shapes.next().unwrap().into(),
        }
    }

    pub fn operand_directions(&self) -> &'static [OperandDirection] {
        use OperandDirection::*;

        match self {
            PrimitiveSpecType::OnePrefix => &[In, Out],
            PrimitiveSpecType::Matmul { accum: true } => &[In, In, InOut],
            PrimitiveSpecType::Matmul { accum: false } => &[In, In, Out],
            PrimitiveSpecType::Conv { accum: true } => &[In, In, InOut],
            PrimitiveSpecType::Conv { accum: false } => &[In, In, Out],
            PrimitiveSpecType::Broadcast { .. } => &[In, Out],
            PrimitiveSpecType::Softmax { .. } => &[In, Out],
            PrimitiveSpecType::SoftmaxComplete { .. } => &[In, In, In, Out],
            PrimitiveSpecType::SoftmaxDenominator { accum: true, .. } => &[In, In, InOut],
            PrimitiveSpecType::SoftmaxDenominator { accum: false, .. } => &[In, In, Out],
            PrimitiveSpecType::SoftmaxDenominatorAndMax { .. } => &[In, Out, Out],
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { accum: true, .. } => {
                &[In, InOut, Out]
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { accum: false, .. } => {
                &[In, Out, Out]
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum: true, .. } => {
                &[In, In, InOut, Out]
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum: false, .. } => {
                &[In, In, Out, Out]
            }
            PrimitiveSpecType::Max { accum: true, .. } => &[In, InOut],
            PrimitiveSpecType::Max { accum: false, .. } => &[In, Out],
            PrimitiveSpecType::Move => &[In, Out],
            PrimitiveSpecType::Fill { .. } => &[Out],
            PrimitiveSpecType::DivideVec => &[In, In, Out],
            PrimitiveSpecType::DivideVecScalar { .. } => &[In, In, Out],
        }
    }

    pub fn operand_count(&self) -> usize {
        self.operand_directions().len()
    }

    // TODO: Check that this handles InOuts as expected by callers.
    // TODO: Rename this to `strict_input_count`
    pub fn input_count(&self) -> usize {
        self.operand_directions()
            .iter()
            .filter(|&&d| d == OperandDirection::In)
            .count()
    }

    // TODO: Rename to parameter_is_written
    fn parameter_is_output(&self, index: usize) -> bool {
        matches!(
            self.operand_directions()[index],
            OperandDirection::Out | OperandDirection::InOut
        )
    }

    // TODO: Rename
    pub fn unique_output_index(&self) -> Option<usize> {
        let mut directions = self.operand_directions().iter();
        let first_out = directions
            .position(|&d| matches!(d, OperandDirection::Out | OperandDirection::InOut))?;
        if directions.any(|&d| matches!(d, OperandDirection::Out | OperandDirection::InOut)) {
            return None;
        }
        Some(first_out)
    }

    // TODO: Rename
    // TODO: Needed?
    pub fn output_is_read(&self) -> bool {
        match self {
            PrimitiveSpecType::Matmul { accum }
            | PrimitiveSpecType::Conv { accum }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { accum, .. }
            | PrimitiveSpecType::Max { accum, .. }
            | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. } => *accum,
            PrimitiveSpecType::OnePrefix
            | PrimitiveSpecType::Fill { .. }
            | PrimitiveSpecType::Move
            | PrimitiveSpecType::Broadcast { .. }
            | PrimitiveSpecType::Softmax { .. }
            | PrimitiveSpecType::SoftmaxComplete { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { .. }
            | PrimitiveSpecType::DivideVec
            | PrimitiveSpecType::DivideVecScalar { .. } => false,
        }
    }

    /// Return the output shape of the primitive given the input shapes.
    ///
    /// This returns `None` if either there are multiple outputs or the output shape cannot be
    /// inferred from the inputs.
    pub fn infer_unique_output_shape(&self, inputs: &[&[DimSize]]) -> Option<Shape> {
        // TODO: Can this be rewritten as output inference + `from_io` call?
        debug_assert_eq!(inputs.len(), self.input_count());
        match self {
            PrimitiveSpecType::Matmul { .. } => {
                let ([b, m, _k], [_, _, n]) = (inputs[0], inputs[1]) else {
                    panic!("Matmul inputs must have 3 dimensions each");
                };
                Some(vec![*b, *m, *n])
            }
            PrimitiveSpecType::Conv { .. } => {
                let ([b, _, h, w], [f, _, fh, fw]) = (inputs[0], inputs[1]) else {
                    panic!("Conv inputs must have 4 dimensions each");
                };
                debug_assert!(h.get() >= fh.get() && w.get() >= fw.get());
                Some(vec![
                    *b,
                    *f,
                    DimSize::new(1 + h.get() - fh.get()).unwrap(),
                    DimSize::new(1 + w.get() - fw.get()).unwrap(),
                ])
            }
            PrimitiveSpecType::Max { dim, .. } => Some(
                inputs[0]
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        if i == usize::from(*dim) {
                            nz!(1u32)
                        } else {
                            *d
                        }
                    })
                    .collect(),
            ),
            PrimitiveSpecType::OnePrefix => {
                let input = inputs[0];
                let mut output = vec![];
                output.reserve_exact(input.len() + 1);
                output.push(nz!(1u32));
                output.extend(input);
                Some(output)
            }
            PrimitiveSpecType::SoftmaxDenominator { .. } => Some(inputs[1].into()),
            PrimitiveSpecType::Softmax { .. }
            | PrimitiveSpecType::SoftmaxComplete { .. }
            | PrimitiveSpecType::Move
            | PrimitiveSpecType::Fill { .. }
            | PrimitiveSpecType::DivideVec
            | PrimitiveSpecType::DivideVecScalar { .. } => {
                // The shape and dtype match for moves and zero.
                Some(inputs[0].to_vec())
            }
            PrimitiveSpecType::Broadcast { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndMax { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. }
            | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { .. } => None,
        }
    }
}

impl Display for PrimitiveSpecType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveSpecType::OnePrefix => write!(f, "OnePrefix"),
            PrimitiveSpecType::Matmul { accum, .. } if *accum => write!(f, "MatmulAccum"),
            PrimitiveSpecType::Matmul { .. } => write!(f, "Matmul"),
            PrimitiveSpecType::Conv { accum, .. } if *accum => write!(f, "ConvAccum"),
            PrimitiveSpecType::Conv { .. } => write!(f, "Conv"),
            PrimitiveSpecType::Broadcast { dim } => write!(f, "Broadcast{dim}"),
            PrimitiveSpecType::Softmax { scan_dim } => write!(f, "Softmax{scan_dim}"),
            PrimitiveSpecType::SoftmaxComplete { scan_dim } => {
                write!(f, "SoftmaxComplete{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim, .. } => {
                write!(f, "SoftmaxDenominatorAndMax{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, accum } if *accum => {
                write!(f, "SoftmaxDenominatorAndUnscaledAccum{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, accum: _ } => {
                write!(f, "SoftmaxDenominatorAndUnscaled{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, accum }
                if *accum =>
            {
                write!(f, "SoftmaxDenominatorAndUnscaledFromMaxAccum{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, .. } => {
                write!(f, "SoftmaxDenominatorAndUnscaledFromMax{scan_dim}")
            }
            PrimitiveSpecType::SoftmaxDenominator { accum, .. } if *accum => {
                write!(f, "SoftmaxDenominatorAccum")
            }
            PrimitiveSpecType::SoftmaxDenominator { .. } => write!(f, "SoftmaxDenominator"),
            PrimitiveSpecType::Max { dim, accum } if *accum => write!(f, "MaxAccum{dim}"),
            PrimitiveSpecType::Max { dim, .. } => write!(f, "Max{dim}"),
            PrimitiveSpecType::Move => write!(f, "Move"),
            PrimitiveSpecType::Fill {
                value: FillValue::Zero,
            } => write!(f, "FillZero"),
            PrimitiveSpecType::Fill {
                value: FillValue::NegInf,
            } => write!(f, "FillNegInf"),
            PrimitiveSpecType::DivideVec => write!(f, "DivideVec"),
            PrimitiveSpecType::DivideVecScalar { scan_dim } => {
                write!(f, "DivideVecScalar{scan_dim}")
            }
        }
    }
}

impl<Tgt: Target> LogicalSpec<Tgt> {
    pub fn primitive_from_parameters<P: IntoIterator<Item = TensorSpec<Tgt>>>(
        typ: PrimitiveSpecType,
        parameters: P,
        serial_only: bool,
    ) -> Self {
        let parameters_vec = parameters.into_iter().collect::<Vec<_>>();
        // Call a less generic version to rein in monomorphization.
        Self::primitive_from_parameters_vec(typ, parameters_vec, serial_only)
    }

    fn primitive_from_parameters_vec(
        typ: PrimitiveSpecType,
        parameters: Vec<TensorSpec<Tgt>>,
        serial_only: bool,
    ) -> Self {
        let spec_shape = typ.shape_from_parameters(parameters.iter().map(|t| t.shape()));
        // TODO: Avoid cloning and instead just move out of the Vec elements
        let dtypes = parameters.iter().map(|p| p.dtype).collect();
        let auxes = parameters.iter().map(|p| p.aux.clone()).collect();
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ,
                spec_shape,
                dtypes,
            },
            auxes,
            serial_only,
        )
    }

    pub fn operand_directions(&self) -> Vec<OperandDirection> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.typ.operand_directions().into(),
            LogicalSpec::Compose { components, .. } => compose_parameter_directions(components),
        }
    }

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
            LogicalSpec::Primitive(basics, auxes, _) => {
                debug_assert_eq!(basics.parameter_shapes().len(), basics.dtypes.len());
                debug_assert_eq!(basics.parameter_shapes().len(), auxes.len());
                basics
                    .parameter_shapes()
                    .into_iter()
                    .zip(&basics.dtypes)
                    .zip(auxes)
                    .map(|((s, dt), a)| TensorSpec::new_noncanon_with_aux(s, *dt, a.clone()))
                    .collect()
            }
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
                let outermost_component_output_idx =
                    components[0].typ.unique_output_index().unwrap();
                let mut outermost_component_parameter_shapes = components[0].parameter_shapes();
                let mut outermost_component_dtypes = components[0].dtypes.clone();
                let outermost_component_output_shape =
                    outermost_component_parameter_shapes.remove(outermost_component_output_idx);
                let outermost_component_output_dtype =
                    outermost_component_dtypes.remove(outermost_component_output_idx);
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
                    let c_out_idx = c.typ.unique_output_index().unwrap();
                    let mut ps = c.parameter_shapes();
                    ps.remove(c_out_idx);
                    ps.remove(0);
                    let mut dtypes = c.dtypes.clone();
                    dtypes.remove(c_out_idx);
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
                let innermost_component_output_idx =
                    innermost_component.typ.unique_output_index().unwrap();
                let mut ps = innermost_component.parameter_shapes();
                ps.remove(innermost_component_output_idx);
                let mut dtypes = innermost_component.dtypes.clone();
                dtypes.remove(innermost_component_output_idx);
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

    // TODO: Test that these are always canonical
    pub fn parameter(&self, index: usize) -> TensorSpec<Tgt> {
        match self {
            LogicalSpec::Primitive(basics, auxes, _) => TensorSpec::new_noncanon_with_aux(
                self.parameter_shape(index),
                basics.dtypes[index],
                auxes[index].clone(),
            ),
            LogicalSpec::Compose { .. } => {
                // TODO: Extremely inefficient. Specialize this.
                self.parameters().swap_remove(index)
            }
        }
    }

    pub fn input_shapes(&self) -> Vec<Shape> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.input_shapes(),
            LogicalSpec::Compose { .. } => todo!(),
        }
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
        (0..self.operand_count())
            .filter(|i| !self.parameter_is_output(*i))
            .map(|i| self.parameter(i))
            .collect()
    }

    /// Returns the output parameter if there is just one.
    pub fn unique_output(&self) -> Option<TensorSpec<Tgt>> {
        self.unique_output_index().map(|idx| self.parameter(idx))
    }

    /// Returns the index of the output parameter if there is just one.
    pub fn unique_output_index(&self) -> Option<usize> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.typ.unique_output_index(),
            LogicalSpec::Compose { .. } => {
                let mut found_output = None;
                let mut param_idx = self.operand_count();
                while param_idx > 0 {
                    param_idx -= 1;
                    if self.parameter_is_output(param_idx) {
                        found_output = Some(param_idx);
                        break;
                    }
                }
                while param_idx > 0 {
                    param_idx -= 1;
                    if self.parameter_is_output(param_idx) {
                        return None;
                    }
                }
                found_output
            }
        }
    }

    pub fn parameter_is_output(&self, index: usize) -> bool {
        match self {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => {
                typ.parameter_is_output(index)
            }
            LogicalSpec::Compose { .. } => index == (self.operand_count() - 1),
        }
    }

    pub(crate) fn initial_accumulating_value_for_output(&self, index: usize) -> Option<FillValue> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                basics.initial_accumulating_value_for_output(index)
            }
            LogicalSpec::Compose { components, .. } => {
                let component_output = components[0]
                    .typ
                    .unique_output_index()
                    .expect("Compose component should have a unique output");
                components[0].initial_accumulating_value_for_output(component_output)
            }
        }
    }

    // TODO: Rename. This is used to prevent these Specs in non-head Compose positions.
    pub fn causes_side_effects(&self) -> bool {
        match self {
            LogicalSpec::Primitive(basics, _, _) => basics.causes_side_effects(),
            LogicalSpec::Compose { components, .. } => components[0].causes_side_effects(),
        }
    }

    pub fn canonicalize(&mut self) -> Result<(), CanonicalizeError> {
        match self {
            LogicalSpec::Primitive(basics, primitive_aux, _) => match &basics.typ {
                PrimitiveSpecType::Move => {
                    let shape = &basics.spec_shape;
                    for (aux, dtype) in primitive_aux.iter_mut().zip(&basics.dtypes) {
                        let vs = aux.vector_size;
                        if !check_tensor_vector_size::<Tgt>(shape, *dtype, &aux.level, vs) {
                            return Err(CanonicalizeError::TensorSpecAuxCanonicalizeError(
                                tensorspec::CanonicalizeError::VectorSizeInvalid,
                            ));
                        }

                        aux.canonicalize(shape)
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
                        let rm = row_major(shape.len().try_into().unwrap());
                        let new_contig = rm.contiguous_full();
                        for aux in primitive_aux.iter_mut() {
                            aux.layout = rm.clone();
                            aux.contig = new_contig;
                        }
                    }
                }
                _ => {
                    for (shp, dtype, aux) in
                        izip!(basics.parameter_shapes(), &basics.dtypes, primitive_aux)
                    {
                        let vs = aux.vector_size;
                        if !check_tensor_vector_size::<Tgt>(&shp, *dtype, &aux.level, vs) {
                            return Err(CanonicalizeError::TensorSpecAuxCanonicalizeError(
                                tensorspec::CanonicalizeError::VectorSizeInvalid,
                            ));
                        }

                        aux.canonicalize(&shp)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                    }
                }
            },
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                for tail_component in &components[1..] {
                    if tail_component.causes_side_effects() {
                        return Err(CanonicalizeError::SideEffectingComponent);
                    }
                }

                let mut visited = 0;
                let mut status_to_return = Ok(());
                compose_parameter_visit(components, |component_idx, parameter_idx| {
                    visited += 1;

                    // TODO: Short-circuit this closure/loop instead of checking `is_err`.
                    if status_to_return.is_err() {
                        return;
                    }

                    let component = &components[component_idx];
                    let shape = component.parameter_shape(parameter_idx);
                    let aux = &mut operand_auxes[visited - 1];
                    status_to_return = aux
                        .canonicalize(&shape)
                        .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError);
                    if status_to_return.is_err() {
                        return;
                    }

                    let dtype = component.parameter_dtype(parameter_idx);
                    let level = &aux.level;
                    let vs = aux.vector_size;
                    if !check_tensor_vector_size::<Tgt>(&shape, dtype, level, vs) {
                        status_to_return = Err(CanonicalizeError::TensorSpecAuxCanonicalizeError(
                            tensorspec::CanonicalizeError::VectorSizeInvalid,
                        ));
                    }
                });
                assert_eq!(visited, operand_auxes.len());
                status_to_return?;
            }
        }
        Ok(())
    }

    pub fn is_canonical(&self) -> bool {
        match self {
            LogicalSpec::Primitive(basics, primitive_aux, _) => match &basics.typ {
                PrimitiveSpecType::Move => {
                    let shape = &basics.spec_shape;
                    for (aux, dtype) in primitive_aux.iter().zip(&basics.dtypes) {
                        let vs = aux.vector_size;
                        if !check_tensor_vector_size::<Tgt>(shape, *dtype, &aux.level, vs) {
                            return false;
                        }

                        if !aux.is_canonical(shape) {
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
                    true
                }
                _ => {
                    for (shp, dtype, aux) in
                        izip!(basics.parameter_shapes(), &basics.dtypes, primitive_aux)
                    {
                        let vs = aux.vector_size;
                        if !check_tensor_vector_size::<Tgt>(&shp, *dtype, &aux.level, vs) {
                            return false;
                        }

                        if !aux.is_canonical(&shp) {
                            return false;
                        }
                    }
                    true
                }
            },
            LogicalSpec::Compose {
                components,
                operand_auxes,
                ..
            } => {
                for tail_component in &components[1..] {
                    if tail_component.causes_side_effects() {
                        return false;
                    }
                }

                let mut visited = 0;
                let mut status_to_return = true;
                compose_parameter_visit(components, |component_idx, parameter_idx| {
                    visited += 1;

                    // TODO: Short-circuit this closure/loop instead of checking `is_err`.
                    if !status_to_return {
                        return;
                    }

                    let component = &components[component_idx];
                    let shape = component.parameter_shape(parameter_idx);
                    let aux = &operand_auxes[visited - 1];
                    if !aux.is_canonical(&shape) {
                        status_to_return = false;
                    }
                    if !status_to_return {
                        return;
                    }

                    let dtype = component.parameter_dtype(parameter_idx);
                    let level = &aux.level;
                    let vs = aux.vector_size;
                    if !check_tensor_vector_size::<Tgt>(&shape, dtype, level, vs) {
                        status_to_return = false;
                    }
                });
                assert_eq!(visited, operand_auxes.len());
                status_to_return
            }
        }
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

    pub fn input_tilings_for_tile_out(&self, smaller_output: &Tiling) -> Option<TilingInference> {
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
                let mut first_inference =
                    components[0].input_tilings_for_tile_out(smaller_output)?;
                accumulated_input_tilings.extend(first_inference.0.drain(1..));

                let mut last_output_tiling = first_inference.0.remove(0).0;
                for subspec in &components[1..components.len() - 1] {
                    let mut subspec_input_tilings =
                        subspec.input_tilings_for_tile_out(&last_output_tiling)?;
                    accumulated_input_tilings.extend(subspec_input_tilings.0.drain(1..));
                    last_output_tiling = subspec_input_tilings.0.remove(0).0;
                }

                accumulated_input_tilings.extend(
                    components[components.len() - 1]
                        .input_tilings_for_tile_out(&last_output_tiling)?
                        .0,
                );

                Some(TilingInference(accumulated_input_tilings))
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
                for (component_idx, component) in components.iter_mut().enumerate().rev() {
                    // Any missing inputs? Gather them here.
                    let needed = component.typ.input_count() - component_inputs.len();
                    component_inputs.extend(
                        remaining_inputs
                            .drain(remaining_inputs.len() - needed..)
                            .map(|(shape, dtype)| (Shape::from(shape), dtype)),
                    );

                    // The output shape might be ambiguous, so we get it from the given TensorSpec
                    // in the case of the head component (it is not given for the others).  This is
                    // important when the head component is, for example, a Softmax where the input
                    // shape is compatible with many output shapes.
                    let (new_output_shape, new_output_dtype) = if component_idx == 0 {
                        let last_operand = new_operands.last().unwrap();
                        (last_operand.shape().to_vec(), last_operand.dtype())
                    } else {
                        let inp_shapes = component_inputs
                            .iter()
                            .map(|t| t.0.as_slice())
                            .collect::<Vec<_>>();
                        (
                            component
                                .typ
                                .infer_unique_output_shape(&inp_shapes)
                                .unwrap(),
                            component.dtypes[component.typ.unique_output_index().unwrap()],
                        )
                    };
                    let mut new_operands = component_inputs.clone();
                    new_operands.push((new_output_shape, new_output_dtype));
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
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Conv { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. } => {
                    *accum = true;
                }
                _ => panic!("Cannot clone_as_accum: {:?}", self),
            },
            LogicalSpec::Compose { components, .. } => match &mut components[0].typ {
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Conv { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. } => {
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
            let output_idx = self
                .unique_output_index()
                .expect("Compose has a unique output");
            let (output, external_inputs) = operands.split_last().unwrap();
            debug_assert_eq!(output_idx, external_inputs.len());
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
                let shape_bimap = ShapeBimap(self.primitive_basics_bimap.binary_scale_shapes);

                let key = SpecKey::Compose {
                    components: components
                        .iter()
                        .map(|c| {
                            (
                                c.typ,
                                c.dtypes.clone(),
                                c.spec_shape.len().try_into().unwrap(),
                            )
                        })
                        .collect(),
                };
                let mut pt = components
                    .iter()
                    .flat_map(|c| {
                        let mapped_shape = BiMap::apply(&shape_bimap, &c.spec_shape);
                        // lengths must match for apply_inverse correctness
                        debug_assert_eq!(c.spec_shape.len(), mapped_shape.len());
                        mapped_shape
                    })
                    .collect::<Vec<_>>();
                // TODO: Avoid calling self.parameters(), which is expensive, if possible
                let aux_keys = operand_auxes
                    .iter()
                    .zip(spec.parameters())
                    .map(|(tensor_aux, parameter)| {
                        let aux_bimap = (self.aux_surmap_fn)(parameter.shape(), parameter.dtype());
                        let (aux_key, aux_pt) = aux_bimap.apply(tensor_aux);
                        debug_assert_eq!(aux_pt.len(), N);
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

        match key {
            SpecKey::Compose {
                components: components_proj,
            } => {
                let shape_bimap = ShapeBimap(self.primitive_basics_bimap.binary_scale_shapes);

                let mut remaining_pt = &pt[..pt.len() - 1];
                let serial_only = pt[pt.len() - 1] == 0;

                let mut components = vec![];
                components.reserve_exact(components_proj.len());
                for (typ, dtypes, rank) in components_proj {
                    debug_assert_eq!(dtypes.len(), typ.operand_count());
                    let spec_shape =
                        BiMap::apply_inverse(&shape_bimap, &eat(&mut remaining_pt, *rank).to_vec());
                    components.push(PrimitiveBasics {
                        typ: *typ,
                        spec_shape,
                        dtypes: dtypes.clone(),
                    });
                }

                let parameter_shapes = compose_parameter_shapes(&components);
                let parameter_dtypes = compose_parameter_dtypes(&components);
                debug_assert_eq!(aux_keys.len(), parameter_shapes.len());
                debug_assert_eq!(aux_keys.len(), parameter_dtypes.len());

                Box::new(
                    izip!(aux_keys, parameter_shapes, parameter_dtypes)
                        .map(|(aux_key, parameter_shape, parameter_dtype)| {
                            let aux_surmap =
                                (self.aux_surmap_fn)(&parameter_shape, parameter_dtype);
                            let aux_pt = eat(&mut remaining_pt, N);
                            SurMap::apply_inverse(
                                &aux_surmap,
                                &(aux_key.clone(), aux_pt.try_into().unwrap()),
                            )
                            // TODO: Avoid collect, used to avoid needing the Iter to be Clone
                            .collect::<Vec<_>>()
                        })
                        .multi_cartesian_product()
                        .map(move |auxes| LogicalSpec::Compose {
                            components: components.clone(),
                            operand_auxes: auxes,
                            serial_only,
                        }),
                )
            }
            _ => {
                let dtypes = key.dtypes().collect::<Vec<_>>();
                let operand_count = aux_keys.len();
                debug_assert_eq!(dtypes.len(), operand_count);

                let pt_without_serial = &pt[..pt.len() - 1];
                let (basics_pt, tensor_aux_pts) =
                    pt_without_serial.split_at(pt.len() - (operand_count * N) - 1);
                let serial = pt[pt.len() - 1] == 0;

                let primitive_basics = BiMap::apply_inverse(
                    &self.primitive_basics_bimap,
                    &(key.clone(), basics_pt.into()),
                );
                let parameter_shapes = primitive_basics.parameter_shapes();
                debug_assert_eq!(parameter_shapes.len(), operand_count);

                Box::new(
                    (0..operand_count)
                        .map(move |i| {
                            let Ok(tap) = (&tensor_aux_pts[i * N..(i + 1) * N]).try_into() else {
                                panic!("Couldn't reverse the TensorSpecAux pt.");
                            };
                            let aux_surmap = (self.aux_surmap_fn)(&parameter_shapes[i], dtypes[i]);
                            // TODO: Avoid collect, used to avoid needing the Iter to be Clone
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
            PrimitiveSpecType::Broadcast { dim } => (
                SpecKey::Broadcast {
                    dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Softmax { scan_dim } => {
                assert_eq!(dtypes.len(), 2);
                (
                    SpecKey::Softmax {
                        scan_dim,
                        dtypes: dtypes.as_slice().try_into().unwrap(),
                    },
                    shifted_shape.collect(),
                )
            }
            PrimitiveSpecType::SoftmaxComplete { scan_dim } => (
                SpecKey::SoftmaxComplete {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim } => (
                SpecKey::SoftmaxDenominatorAndMax {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, accum } => (
                SpecKey::SoftmaxDenominatorAndUnscaled {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, accum } => (
                SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominator { scan_dim, accum } => (
                SpecKey::SoftmaxDenominator {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::Max { dim, accum } => (
                SpecKey::Max {
                    dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::OnePrefix => (
                SpecKey::OnePrefix {
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Move => (
                SpecKey::Move {
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Fill { value } => (
                SpecKey::Fill {
                    value,
                    dtype: dtypes[0],
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::DivideVec => (
                SpecKey::DivideVec {
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::DivideVecScalar { scan_dim } => (
                SpecKey::DivideVecScalar {
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
        }
    }

    fn apply_inverse(&self, c: &Self::Codomain) -> Self::Domain {
        let (key, v) = c;
        let basics = match key {
            SpecKey::Matmul { dtypes: _ }
            | SpecKey::Conv { dtypes: _ }
            | SpecKey::SoftmaxDenominatorAndUnscaled {
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominator {
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::Max { dim: _, dtypes: _ } => {
                let accum = v[0] == 0;
                let (typ, dtypes) = match key {
                    SpecKey::Matmul { dtypes } => {
                        (PrimitiveSpecType::Matmul { accum }, dtypes.to_vec())
                    }
                    SpecKey::Conv { dtypes } => {
                        (PrimitiveSpecType::Conv { accum }, dtypes.to_vec())
                    }
                    SpecKey::SoftmaxDenominatorAndUnscaled { scan_dim, dtypes } => (
                        PrimitiveSpecType::SoftmaxDenominatorAndUnscaled {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, dtypes } => (
                        PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominator { scan_dim, dtypes } => (
                        PrimitiveSpecType::SoftmaxDenominator {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::Max { dim, dtypes } => {
                        (PrimitiveSpecType::Max { dim: *dim, accum }, dtypes.to_vec())
                    }
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
                    dtypes,
                }
            }
            SpecKey::Softmax {
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxComplete {
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominatorAndMax {
                scan_dim: _,
                dtypes: _,
            } => {
                let (typ, dtypes) = match key {
                    SpecKey::Softmax { scan_dim, dtypes } => (
                        PrimitiveSpecType::Softmax {
                            scan_dim: *scan_dim,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxComplete { scan_dim, dtypes } => (
                        PrimitiveSpecType::SoftmaxComplete {
                            scan_dim: *scan_dim,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominatorAndMax { scan_dim, dtypes } => (
                        PrimitiveSpecType::SoftmaxDenominatorAndMax {
                            scan_dim: *scan_dim,
                        },
                        dtypes.into(),
                    ),
                    _ => unreachable!(),
                };

                let mut spec_shape: Vec<BimapInt> = v.clone();
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
                    dtypes,
                }
            }
            SpecKey::OnePrefix { dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::OnePrefix,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Move { dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::Move,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Fill { value, dtype } => PrimitiveBasics {
                typ: PrimitiveSpecType::Fill { value: *value },
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: vec![*dtype],
            },
            SpecKey::DivideVec { dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::DivideVec,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::DivideVecScalar { scan_dim, dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::DivideVecScalar {
                    scan_dim: *scan_dim,
                },
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Broadcast { dim, dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::Broadcast { dim: *dim },
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.to_vec(),
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
        use proptest::prelude::*;
        prop_oneof![
            arb_primitive_logical_spec(args.clone()),
            arb_compose_spec(args.max_size)
        ]
        .boxed()
    }
}

#[cfg(test)]
pub fn arb_primitive_logical_spec<Tgt: Target>(
    args: PrimitiveBasicsArbParams,
) -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>> {
    use crate::tensorspec::TensorSpecArbMaxShape;
    use proptest::prelude::*;

    (any_with::<PrimitiveBasics>(args), any::<bool>())
        .prop_flat_map(|(basics, serial_only)| {
            let auxes_strategy = basics
                .parameter_shapes()
                .into_iter()
                .zip(&basics.dtypes)
                .map(|(s, &d)| any_with::<TensorSpecAux<Tgt>>((TensorSpecArbMaxShape(s), Some(d))))
                .collect::<Vec<_>>();
            (Just(basics), auxes_strategy, Just(serial_only))
        })
        .prop_map(|(basics, auxes, serial_only)| LogicalSpec::Primitive(basics, auxes, serial_only))
        .prop_filter("Layout must be applicable to TensorSpec shape", |s| {
            s.clone().canonicalize().is_ok()
        })
}

#[cfg(test)]
pub(crate) fn arb_compose_spec<Tgt>(
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>>
where
    Tgt: Target,
{
    use crate::tensorspec::TensorSpecArbMaxShape;
    use proptest::prelude::*;

    arb_compose_components(max_size)
        .prop_flat_map(|components| {
            let auxes_strategies = compose_parameter_shapes(&components)
                .into_iter()
                .zip(compose_parameter_dtypes(&components))
                .map(|(shape, dtype)| {
                    any_with::<TensorSpecAux<Tgt>>((TensorSpecArbMaxShape(shape), Some(dtype)))
                })
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
fn arb_compose_component_innermost(
    allow_broadcast: bool,
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = PrimitiveBasics> {
    use proptest::prelude::*;

    any_with::<PrimitiveBasics>(max_size.into())
        .prop_filter("Must not be a Broadcast", move |basics| {
            allow_broadcast || !matches!(basics.typ, PrimitiveSpecType::Broadcast { .. })
        })
        .prop_filter("Must have at least two parameters to compose", |basics| {
            basics.typ.operand_count() > 1
        })
        .prop_filter("Must have a unique output to compose", |basics| {
            basics.typ.unique_output_index().is_some()
        })
        .prop_filter("Must not cause side effects", |basics| {
            !basics.causes_side_effects()
        })
        .boxed()
}

#[cfg(test)]
fn arb_compose_component_successor(
    predecessor: &PrimitiveBasics,
    allow_broadcast: bool,
    allow_side_effects: bool,
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = PrimitiveBasics> {
    use proptest::prelude::*;

    // Restrict the basic types to those which have a first input of a possible rank.
    let shapes = predecessor.parameter_shapes();
    let out_idx = predecessor.typ.unique_output_index().unwrap();
    let mut allowed_types = vec![]; // TODO: Somehow gather these from type def.
    match shapes[out_idx].len() {
        3 => {
            if allow_side_effects {
                allowed_types.push(PrimitiveSpecType::Matmul { accum: true });
            }
            allowed_types.push(PrimitiveSpecType::Matmul { accum: false });
        }
        4 => {
            if allow_side_effects {
                allowed_types.push(PrimitiveSpecType::Conv { accum: true });
            }
            allowed_types.push(PrimitiveSpecType::Conv { accum: false });
        }
        _ => {}
    }
    allowed_types.push(PrimitiveSpecType::Move);
    allowed_types.push(PrimitiveSpecType::OnePrefix);
    for dim in 0..u8::try_from(shapes[out_idx].len()).unwrap() {
        if allow_broadcast && shapes[out_idx][usize::from(dim)].get() == 1 {
            allowed_types.push(PrimitiveSpecType::Broadcast { dim });
        }
        allowed_types.push(PrimitiveSpecType::Softmax { scan_dim: dim });
        allowed_types.push(PrimitiveSpecType::SoftmaxComplete { scan_dim: dim });
        for accum in [true, false] {
            if !allow_side_effects && accum {
                continue;
            }
            allowed_types.push(PrimitiveSpecType::Max { dim, accum });
            allowed_types.push(PrimitiveSpecType::SoftmaxDenominator {
                scan_dim: dim,
                accum,
            });
        }
    }
    // TODO: Update this so we don't need to manually modify `allowed_types` all the time.

    any_with::<PrimitiveBasics>(PrimitiveBasicsArbParams {
        max_size,
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
fn arb_compose_components(
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = Vec<PrimitiveBasics>> {
    use proptest::prelude::*;

    prop_oneof![
        arb_compose_component_innermost(false, max_size)
            .prop_flat_map(move |c| {
                let successor = arb_compose_component_successor(&c, true, true, max_size);
                (successor, Just(c))
            })
            .prop_map(|(s, c)| vec![s, c]),
        arb_compose_component_innermost(false, max_size)
            .prop_flat_map(move |c| {
                let successor = arb_compose_component_successor(&c, false, false, max_size);
                (successor, Just(c))
            })
            .prop_flat_map(move |(s, c)| {
                let successor2 = arb_compose_component_successor(&s, true, true, max_size);
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

#[cfg(test)]
pub fn arb_canonical_primitive_logical_spec<Tgt: Target>(
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>> {
    use proptest::prelude::*;

    let params = PrimitiveBasicsArbParams::from(max_size);

    arb_primitive_logical_spec(params).prop_filter_map(
        "Must be possible to canonicalize LogicalSpec",
        |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        },
    )
}

#[cfg(test)]
pub fn arb_canonical_compose_logical_spec<Tgt: Target>(
    max_size: Option<DimSize>,
) -> impl proptest::strategy::Strategy<Value = LogicalSpec<Tgt>> {
    use proptest::prelude::*;

    arb_compose_spec(max_size).prop_filter_map(
        "Must be possible to canonicalize LogicalSpec",
        |mut s| {
            if s.canonicalize().is_err() {
                return None;
            }
            Some(s)
        },
    )
}

pub fn dim_range(dim_size: DimSize, include_end: bool) -> impl Iterator<Item = DimSize> {
    (0..)
        .map(|power| 2u32.pow(power))
        .take_while(move |x| *x < dim_size.get())
        .map(|x| DimSize::new(x).unwrap())
        .chain(once(if include_end { Some(dim_size) } else { None }).flatten())
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
    if index < last_component.typ.input_count() {
        return last_component.input_shape(index);
    }
    components[0].unique_output_shape().unwrap()
}

fn compose_parameter_dtypes(components: &[PrimitiveBasics]) -> Vec<Dtype> {
    let mut result = vec![];
    result.reserve_exact(compose_parameter_count(components));
    compose_parameter_visit(components, |component_idx, parameter_idx| {
        let component = &components[component_idx];
        result.push(component.parameter_dtype(parameter_idx));
    });
    result
}

fn compose_parameter_directions(components: &[PrimitiveBasics]) -> Vec<OperandDirection> {
    let mut result = vec![];
    result.reserve_exact(compose_parameter_count(components));
    compose_parameter_visit(components, |component_idx, parameter_idx| {
        let component = &components[component_idx];
        // While we repeatedly call `operand_directions` on the same component, this should be
        // pretty cheap since [PrimitiveSpecType::operand_directions] returns a static slice instead
        // of allocating a Vec like LogicalSpec's operand_directions.
        result.push(component.typ.operand_directions()[parameter_idx]);
    });
    result
}

fn compose_parameter_visit(components: &[PrimitiveBasics], mut visitor: impl FnMut(usize, usize)) {
    debug_assert!(components.len() >= 2);

    // TODO: Replace parameter_dtypes with some parameter_count method
    let c0_output_idx = components[0].typ.unique_output_index().unwrap();
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
        let output_idx = c.typ.unique_output_index().unwrap();
        for parameter in 1..c.parameter_dtypes().len() {
            if parameter != output_idx {
                visitor(component_idx, parameter);
            }
        }
    }

    let cl_output_idx = components[components.len() - 1]
        .typ
        .unique_output_index()
        .unwrap();
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

fn passthrough_tiling_tuple(output: &Tiling) -> (Tiling, Vec<Option<u8>>) {
    let bindings = (0..output.shape().len())
        .map(|d| Some(d.try_into().unwrap()))
        .collect();
    (output.clone(), bindings)
}

fn one_reduced_dimension_tiling_tuple(
    output: &Tiling,
    reduction_dim: u8,
) -> (Tiling, Vec<Option<u8>>) {
    let dim_us = usize::from(reduction_dim);
    let mut shape = output.shape().clone();
    shape[dim_us] = nz!(1u32);
    let mut steps = output.step_sizes().to_vec();
    steps[dim_us] = nz!(1u32);
    let bindings = (0..output.shape().len())
        .map(|d| {
            if d == dim_us {
                None
            } else {
                Some(d.try_into().unwrap())
            }
        })
        .collect();
    (Tiling::new_sliding(shape, steps), bindings)
}

/// Return the prefix of a slice, and update that slice to be the tail.
fn eat<'a, T, U: Into<usize>>(slice: &mut &'a [T], idx: U) -> &'a [T] {
    let (head, tail) = slice.split_at(idx.into());
    *slice = tail;
    head
}

pub mod macros {
    pub mod internal {
        use crate::common::DimSize;
        use crate::spec::{LogicalSpec, Spec, Target};

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

        pub fn spec_with_max_mem<Tgt: Target>(logical_spec: LogicalSpec<Tgt>) -> Spec<Tgt> {
            Spec(logical_spec, Tgt::max_mem())
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
            $crate::lspec!(@inner $typ($shp, $( ($($opterms)*) ),* , true))
        }};
        ( $typ:tt( $shp:expr, $( ($($opterms:tt)*) ),+ ) ) => {{
            $crate::lspec!(@inner $typ($shp, $( ($($opterms)*) ),* , false))
        }};
        ( @inner $typ:tt( $shp:expr, $( ($($opterms:tt)*) ),*, $s:literal ) ) => {{
            use $crate::spec::macros::internal::IntoDimSize;

            let dtypes = [ $( $crate::lspec!(@dt_head $($opterms)*) ),* ];
            let basics = $crate::spec::PrimitiveBasics {
                typ: $crate::lspec!(@primitive_spec_type $typ),
                spec_shape: ($shp).into_iter().map(|x| x.into_dim_size()).collect(),
                dtypes: dtypes.try_into().unwrap(),
            };

            let mut parameter_shapes = basics.parameter_shapes();
            parameter_shapes.reverse();
            let auxes = [ $( $crate::lspec!(@tensorspecaux &parameter_shapes.pop().unwrap(), $($opterms)*) ),* ];
            $crate::spec::LogicalSpec::Primitive(
                basics,
                auxes.try_into().unwrap(),
                $s,
            )
        }};

        ( @dt_head $dt:tt, $($rest:tt)* ) => {
            $crate::lspec!(@dt_convert $dt)
        };

        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, c0, ua ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, false, false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, c0 ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, false, true)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, ua ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, true, false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, true, true)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr, c0, ua ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), false, false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr, c0 ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), false, true)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr, ua ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), true, false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), true, true)
        };

        // TODO: Accept contiguousnesses other than fully contig. or not at all.
        ( @tensorspecaux_inner $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr,
          $c:literal, $a:literal ) =>
        {{
            let layout = $crate::layout::LayoutBuilder::build($layout, $shp);
            let contig = if $c {
                layout.contiguous_full()
            } else {
                layout.contiguous_none()
            };
            $crate::tensorspec::TensorSpecAux {
                contig,
                aligned: $a,
                level: $level,
                layout,
                vector_size: ($vs).map(|x: u32| {
                    $crate::common::DimSize::try_from(x).unwrap()
                }),
            }
        }};

        ( @primitive_spec_type FillZero ) => {
            $crate::spec::PrimitiveSpecType::Fill {
                value: $crate::spec::FillValue::Zero,
            }
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
        // TODO: Add Softmax to lspec!. Need some notation for reduction dim.

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

    /// Construct a full `Spec<Tgt>`.
    ///
    /// Usage:
    ///   spec!(Move([m,n], (u32,GL,row_major), (u32,GL,row_major)));
    ///   spec!(Move([m,n], (u32,GL,row_major), (u32,GL,row_major)), [lim0, lim1, ...]);
    #[macro_export]
    macro_rules! spec {
        // Literal limits array: build a Standard MemoryLimits
        ( $typ:ident $args:tt , [ $($limits:expr),* $(,)? ] ) => {{
            let logical_spec = $crate::lspec!($typ $args);
            let limits_arr: [u64; $crate::target::LEVEL_COUNT] = [ $( $limits as u64 ),* ];
            $crate::spec::Spec(
                logical_spec,
                $crate::memorylimits::MemoryLimits::Standard(
                    $crate::memorylimits::MemVec::new(limits_arr)
                ),
            )
        }};
        // Dynamic limits expression: wrap directly with provided MemoryLimits
        ( $typ:ident $args:tt , $mem_limits:expr $(,)? ) => {{
            $crate::spec::Spec(
                $crate::lspec!($typ $args),
                $mem_limits,
            )
        }};
        // Default: no memory limits provided, so use the target's maximum memory limits
        ( $typ:ident $args:tt ) => {{
            $crate::spec::macros::internal::spec_with_max_mem($crate::lspec!($typ $args))
        }};
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::general::AsBimap;
    use crate::imp::{visit_leaves, Impl, ImplExt, ImplNode};
    use crate::layout::row_major;
    use crate::memorylimits::{arb_memorylimits_ext, MemVec, MemoryAllocation};
    use crate::scheduling::tiling::TileOut;
    use crate::scheduling::{Action, ActionT as _, ApplyError};
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::target::CpuMemoryLevel::{GL, L1, RF};
    use crate::target::{ArmTarget, CpuMemoryLevel, MemoryLevel, Target, X86Target};
    use crate::tensorspec::{TensorSpecArbMaxShape, TensorSpecAuxNonDepBimap};
    use crate::utils::{next_binary_power, sum_seqs};
    use crate::views::View;
    use crate::{lspec, shape, spec};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::iter;

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);

    #[test]
    fn test_lspec_1() {
        let spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [32, 2, 3, 3],
            (u8, GL, row_major),
            (i8, GL, row_major, c0),
            (u16, GL, row_major, ua),
            serial
        ));
        let lhs = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: true,
            level: GL,
            layout: row_major(3),
            vector_size: None,
        };
        let rhs = TensorSpecAux {
            contig: row_major(3).contiguous_none(),
            aligned: true,
            level: GL,
            layout: row_major(3),
            vector_size: None,
        };
        let out = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: false,
            level: GL,
            layout: row_major(3),
            vector_size: None,
        };
        let expected = LogicalSpec::<X86Target>::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: true },
                spec_shape: shape![32, 2, 3, 3],
                dtypes: vec![Dtype::Uint8, Dtype::Sint8, Dtype::Uint16],
            },
            vec![lhs, rhs, out],
            true,
        );
        assert_eq!(spec, expected);
    }

    #[test]
    fn test_spec_macro_shorthand() {
        let s: Spec<X86Target> = spec!(
            Move([2, 3], (u32, GL, row_major), (u32, GL, row_major)),
            [1024, 512, 256, 128]
        );
        let expected_ls: LogicalSpec<X86Target> =
            lspec!(Move([2, 3], (u32, GL, row_major), (u32, GL, row_major)));
        assert_eq!(s.0, expected_ls);
        assert_eq!(
            s.1,
            MemoryLimits::Standard(MemVec::new([1024, 512, 256, 128]))
        );
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
                [0, 2, 3].map(|i| components[0].spec_shape[i]).into(),
                components[0].dtypes[1],
                operand_auxes[0].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 2, 3].map(|i| components[1].spec_shape[i]).into(),
                components[1].dtypes[1],
                operand_auxes[1].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                components[2].spec_shape[..3].to_vec(),
                components[2].dtypes[0],
                operand_auxes[2].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 2, 3].map(|i| components[2].spec_shape[i]).into(),
                components[2].dtypes[1],
                operand_auxes[3].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 1, 3].map(|i| components[0].spec_shape[i]).into(),
                components[0].dtypes[2],
                operand_auxes.last().unwrap().clone(),
            ),
        ];

        assert_eq!(spec.parameters(), expected_parameters);
    }

    #[test]
    fn test_compose_input_tiling_inference() {
        let spec = compose_logicalspec_test_data();
        let output_tiling = Tiling::new_simple(shape![8, 32, 128]);
        let expected = TilingInference(vec![
            (
                Tiling::new_simple(shape![8, 128, 128]),
                vec![Some(0), None, Some(2)],
            ),
            (
                Tiling::new_simple(shape![8, 128, 128]),
                vec![Some(0), None, Some(2)],
            ),
            (
                Tiling::new_simple(shape![8, 32, 128]),
                vec![Some(0), Some(1), None],
            ),
            (
                Tiling::new_simple(shape![8, 128, 128]),
                vec![Some(0), None, Some(2)],
            ),
        ]);
        assert_eq!(
            spec.input_tilings_for_tile_out(&output_tiling),
            Some(expected)
        );
    }

    #[test]
    fn test_compose_canonicalization_accepts_accmulating_head() {
        let mut spec = compose_logicalspec_test_data();
        let LogicalSpec::Compose { components, .. } = &mut spec else {
            unreachable!();
        };
        components[0].typ = PrimitiveSpecType::Matmul { accum: true };
        assert!(spec.canonicalize().is_ok());
    }

    #[test]
    fn test_compose_canonicalization_rejects_accmulating_tail_components() {
        let mut spec = compose_logicalspec_test_data();
        let LogicalSpec::Compose { components, .. } = &mut spec else {
            unreachable!();
        };
        components[1].typ = PrimitiveSpecType::Matmul { accum: true };
        assert!(spec.canonicalize().is_err());
    }

    #[test]
    fn test_dim_range_with_odd_max() {
        assert_eq!(
            dim_range(nz!(3u32), false).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32)]
        );
        assert_eq!(
            dim_range(nz!(3u32), true).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(3u32)]
        );

        assert_eq!(
            dim_range(nz!(7u32), false).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(4u32)]
        );
        assert_eq!(
            dim_range(nz!(7u32), true).collect::<Vec<_>>(),
            vec![nz!(1u32), nz!(2u32), nz!(4u32), nz!(7u32)]
        );
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
        fn test_no_action_panics_x86(spec in arb_canonical_spec::<X86Target>(None, None)) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        fn test_no_action_panics_arm(spec in arb_canonical_spec::<ArmTarget>(None, None)) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        #[ignore]
        fn test_actions_are_valid_through_consumed_memory_x86(
            logical_spec in arb_canonical_logical_spec::<X86Target>(Some(TEST_SMALL_SIZE))
        ) {
            shared_test_actions_are_valid_through_consumed_memory(logical_spec)
        }

        #[test]
        #[ignore]
        fn test_actions_are_valid_through_consumed_memory_arm(
            logical_spec in arb_canonical_logical_spec::<X86Target>(Some(TEST_SMALL_SIZE))
        ) {
            shared_test_actions_are_valid_through_consumed_memory(logical_spec)
        }

        #[test]
        fn test_unique_output_index_matches_parameter_is_output(
            spec_type in any::<PrimitiveSpecType>(),
        ) {
            let output_idxs = (0..spec_type.operand_count())
                .filter(|&i| spec_type.parameter_is_output(i))
                .collect::<Vec<_>>();
            if let Some(unique_output_idx) = spec_type.unique_output_index() {
                assert_eq!(output_idxs, [unique_output_idx]);
            } else {
                assert_ne!(output_idxs.len(), 1);
            }
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
            if canonicalized_logical_spec.canonicalize().is_ok() {
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
                        recanonicalized.canonicalize().unwrap_or_else(|e| {
                            panic!("Couldn't canonicalize parameter {} even though {} was canon: {}", p, logical_spec, e);
                        });
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
                            .map(|s| any_with::<TensorSpecAux<X86Target>>((TensorSpecArbMaxShape(s), Some(dtype))))
                            .collect::<Vec<_>>();
                        (Just(basics), auxes_strategy, any::<bool>())
                    })
                    .prop_filter("No operand should be in a vector register file", |(_, auxes, _)| {
                        auxes.iter().all(|aux| !aux.level.vector_rf())
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
                .apply(&spec).unwrap_or_else(|e| panic!("Couldn't tile Spec {} to single value: {e:?}", spec));
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
        fn test_move_actions_never_returns_within_level_copy(
            spec in arb_canonical_spec::<X86Target>(None, None)
        ) {
            for action in X86Target::actions(&spec.0) {
                if let Ok(ImplNode::Alloc(alloc)) = action.apply(&spec) {
                    assert_ne!(&alloc.source_spec, alloc.introduced.spec(),
                        "Copying Alloc introduced by action {:?}", action);
                }
            }
        }

        #[test]
        fn test_action_applies_everywhere_down_through_peak_memory(
            (spec, action, _, lower_limit) in arb_spec_action_and_lower_limit::<X86Target>()
        ) {
            let lower_spec = Spec(spec.0.clone(), lower_limit);
            assert!(X86Target::actions(&lower_spec.0).contains(&action),
                "Action {:?} was not present in lower-limits Spec {:?}",
                action, lower_spec);
        }

        #[test]
        fn test_no_action_produces_same_spec_with_higher_memory_limit_x86(
            spec in arb_canonical_spec::<X86Target>(None, None)
        ) {
            shared_test_no_action_produces_same_spec_with_higher_memory_limit(&spec)
        }

        #[test]
        fn test_no_action_produces_same_spec_with_higher_memory_limit_arm(
            spec in arb_canonical_spec::<ArmTarget>(None, None)
        ) {
            shared_test_no_action_produces_same_spec_with_higher_memory_limit(&spec)
        }

        #[test]
        fn test_actions_produce_canonical_subspecs(
            spec in arb_canonical_spec::<X86Target>(None, None)
        ) {
            X86Target::actions(&spec.0).for_each(|action| {
                println!("spec: {}\naction: {action:?}", spec.0);  // TODO: Remove
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
        fn test_specbimap_is_invertible_x86(spec in any::<Spec<X86Target>>()) {
            // binary-scaled SurMap is not tested
            shared_test_specbimap_is_invertible(spec, false);
        }


        #[test]
        fn test_specbimap_is_invertible_arm(spec in any::<Spec<ArmTarget>>()) {
            // binary-scaled SurMap is not tested
            shared_test_specbimap_is_invertible(spec, false);
        }

        #[test]
        fn test_parameter_fn_matches_parameters_fn(spec in any::<LogicalSpec<X86Target>>()) {
            let parameters = spec.parameters();
            let individual_parameters = (0..parameters.len()).map(|i| spec.parameter(i)).collect::<Vec<_>>();
            prop_assert_eq!(parameters, individual_parameters);
        }

        #[test]
        #[should_panic]
        fn test_parameter_fn_panics_above_parameter_len(spec in any::<LogicalSpec<X86Target>>()) {
            spec.parameter(spec.parameters().len());
        }

        #[test]
        #[should_panic]
        fn test_parameter_fn_panics_above_parameter_count(spec in any::<LogicalSpec<X86Target>>()) {
            spec.parameter(spec.operand_count());
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
            tinp in arb_compose_spec::<X86Target>(None)
                .prop_filter_map("Spec was not canonical", |logical_spec| {
                    let mut s = Spec(logical_spec, X86Target::max_mem());
                    if s.canonicalize().is_err() {
                        return None;
                    }
                    Some(s)
                })
                .prop_flat_map(|s| {
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
            let LogicalSpec::Compose { .. } = &compose_spec.0 else {
                unreachable!();
            };

            let pipeline = compose_spec.bufferize(
                index,
                CpuMemoryLevel::RF,
                row_major,
                None
            );
            let spec_parameters = compose_spec.0.parameters();
            let pipeline_parameters = pipeline.parameters().cloned().collect::<Vec<_>>();
            prop_assert_eq!(spec_parameters, pipeline_parameters);
        }
    }

    fn shared_test_specbimap_is_invertible<Tgt>(spec: Spec<Tgt>, binary_scale_shapes: bool)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
        let surmap = SpecSurMap::<Tgt, _, _, _> {
            logical_spec_surmap: LogicalSpecSurMap::new(
                PrimitiveBasicsBimap {
                    binary_scale_shapes,
                },
                |_: &[DimSize], dt| TensorSpecAuxNonDepBimap::new(dt),
            ),
            memory_limits_bimap: MemoryLimitsBimap::default(),
        };
        let bimap = surmap.into_bimap();
        let projection = BiMap::apply(&bimap, &spec);
        let reversed = BiMap::apply_inverse(&bimap, &projection);
        assert_eq!(spec, reversed);
    }

    fn shared_test_no_action_panics<Tgt: Target>(spec: Spec<Tgt>) {
        for action in Tgt::actions(&spec.0) {
            let _ = action.apply(&spec);
        }
    }

    fn shared_test_no_action_produces_same_spec_with_higher_memory_limit<Tgt: Target>(
        spec: &Spec<Tgt>,
    ) {
        Tgt::actions(&spec.0).for_each(|action| {
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
        let mut maxes = maxes_vec.iter_binary_scaled().collect::<Vec<_>>();

        // Zero out levels which are slower than all present operands' levels.
        let parameters = logical_spec.parameters();
        for (level_idx, level) in Tgt::levels().into_iter().enumerate() {
            if parameters.iter().all(|p| p.level() < level) {
                maxes[level_idx] = 0;
            }
        }

        // The list of actions depends only on the logical Spec. Filtering by memory limit happens
        // at application. So it's safe to just collect the list of actions once, up front.
        let mut unseen_actions = Tgt::actions(&logical_spec).collect::<Vec<_>>();

        let mut shared_spec = Spec(logical_spec, MemoryLimits::Standard(MemVec::zero::<Tgt>()));
        let mut diagonal_idx = 0;
        loop {
            let mut empty = true;
            for pt in sum_seqs(&maxes, diagonal_idx) {
                empty = false;
                shared_spec.1 =
                    MemoryLimits::Standard(MemVec::new_from_binary_scaled(pt.try_into().unwrap()));
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
                let applied_actions = Tgt::actions(&spec.0)
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
            spec_shape: shape![16, 128, 128, 128],
            dtypes: vec![Dtype::Uint8, Dtype::Uint16, Dtype::Uint32],
        };
        let basic1 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![16, 128, 128, 128],
            dtypes: vec![Dtype::Uint32, Dtype::Uint16, Dtype::Uint8],
        };
        let basic2 = PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: shape![16, 128, 128, 128],
            dtypes: vec![Dtype::Uint8, Dtype::Uint8, Dtype::Uint32],
        };

        let aux0_1 = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: true,
            level: GL,
            layout: row_major(3),
            vector_size: None,
        };
        let aux1_1 = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: true,
            level: L1,
            layout: row_major(3),
            vector_size: None,
        };
        let aux2_0 = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: false,
            level: GL,
            layout: row_major(3),
            vector_size: None,
        };
        let aux2_1 = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: false,
            level: L1,
            layout: row_major(3),
            vector_size: None,
        };
        let aux0_out = TensorSpecAux {
            contig: row_major(3).contiguous_full(),
            aligned: true,
            level: RF,
            layout: row_major(3),
            vector_size: None,
        };

        LogicalSpec::Compose {
            components: vec![basic0.clone(), basic1.clone(), basic2.clone()],
            operand_auxes: vec![aux0_1, aux1_1, aux2_0, aux2_1, aux0_out],
            serial_only: false,
        }
    }
}
