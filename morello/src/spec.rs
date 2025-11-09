use crate::common::{DimSize, Dtype, Shape};
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{BiMap, SurMap};
use crate::grid::linear::BimapInt;
use crate::layout::row_major;
use crate::memorylimits::{MemoryLimits, MemoryLimitsBimap};
use crate::target::{MemoryLevel, Target};
use crate::tensorspec::{self, check_tensor_vector_size, TensorSpec, TensorSpecAux};
use crate::tiling::Tiling;
use crate::utils::{bit_length_inverse, bit_length_u32, join_into_string, prev_power_of_two_u32};

use itertools::{izip, Itertools};
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;

use std::fmt;
use std::fmt::Display;
use std::iter::once;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::num::NonZeroU64;
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
    /// The smallest finite value.
    Min,
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
#[derive(Debug, Clone, PartialEq)]
pub struct TilingInference(pub Vec<(Tiling, Vec<Option<u8>>)>);

/// Combines a [TilingInference] with the new shapes of all component [LogicalSpec]s.
///
/// For [LogicalSpec::Primitive], `component_parameter_shapes` is the same shapes as those in
/// `compose_input_tilings` with the addition of the output. For [LogicalSpec::Compose], it includes
/// "internal" shapes of internal components, which are shapes which aren't composed by parameters
/// to the `Compose`.
#[derive(Debug)]
pub(crate) struct LogicalSpecInputTilingInference {
    pub input_tilings: TilingInference,
    pub component_parameter_shapes: Vec<Vec<Shape>>,
}

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
        let levels = self.0.parameter_levels();
        self.1.zero_levels_slower_than_all::<Tgt>(&levels);
        self.0.canonicalize()
    }

    pub fn is_canonical(&self) -> bool {
        let levels = self.0.parameter_levels();
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
                    let mut shape = smallvec![];
                    shape.reserve_exact(self.spec_shape.len() + 1);
                    shape.push(nz!(1u32));
                    shape.extend_from_slice(&self.spec_shape);
                    shape
                }
                _ => panic!("OnePrefix has only 2 parameters"),
            },
            PrimitiveSpecType::Matmul { .. } => match idx {
                0 => smallvec![self.spec_shape[0], self.spec_shape[1], self.spec_shape[2]],
                1 => smallvec![self.spec_shape[0], self.spec_shape[2], self.spec_shape[3]],
                2 => smallvec![self.spec_shape[0], self.spec_shape[1], self.spec_shape[3]],
                _ => panic!("Matmul has only 3 parameters"),
            },
            PrimitiveSpecType::Conv { .. } => {
                let [b, f, c, h_add_1, w_add_1, fh, fw] = self.spec_shape[..] else {
                    panic!("Conv must have rank 7")
                };
                let h = DimSize::new(h_add_1.get() - 1 + fh.get()).unwrap();
                let w = DimSize::new(w_add_1.get() - 1 + fw.get()).unwrap();
                match idx {
                    0 => smallvec![b, c, h, w],
                    1 => smallvec![f, c, fh, fw],
                    2 => smallvec![b, f, h_add_1, w_add_1],
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
                0 | 1 => self.spec_shape.clone(),
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
            | PrimitiveSpecType::Max { dim, .. } => match idx {
                0 => self.spec_shape.clone(),
                x if x < self.typ.operand_count() => {
                    let mut reduced = self.spec_shape.clone();
                    reduced[usize::from(dim)] = nz!(1u32);
                    reduced
                }
                _ => panic!(),
            },
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim: dim, .. } => match idx {
                0 | 2 => self.spec_shape.clone(),
                1 => {
                    let mut reduced = self.spec_shape.clone();
                    reduced[usize::from(dim)] = nz!(1u32);
                    reduced
                }
                _ => panic!("SoftmaxDenominatorAndUnscaled has only 3 parameters"),
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
                Dtype::Uint8
                | Dtype::Uint16
                | Dtype::Uint32
                | Dtype::Sint8
                | Dtype::Sint16
                | Dtype::Sint32 => Some(FillValue::Min),
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
                        smallvec![
                            smaller_output.shape()[0],
                            smaller_output.shape()[1],
                            spec_shape[2],
                        ],
                        smallvec![
                            smaller_output.step_sizes()[0],
                            smaller_output.step_sizes()[1],
                            spec_shape[2],
                        ],
                    ),
                    vec![Some(0), Some(1), None],
                ),
                (
                    Tiling::new_sliding(
                        smallvec![
                            smaller_output.shape()[0],
                            spec_shape[2],
                            smaller_output.shape()[2],
                        ],
                        smallvec![
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
                let h = DimSize::new(smaller_output.shape()[2].get() + fh.get() - 1).unwrap();
                let w = DimSize::new(smaller_output.shape()[3].get() + fw.get() - 1).unwrap();
                let new_image_shape: Shape = smallvec![smaller_output.shape()[0], channels, h, w];
                let mut new_image_steps: Shape = smaller_output.step_sizes().into();
                new_image_steps[1] = channels;

                // Compute the new filters Tiling.
                let new_filters_shape: Shape =
                    smallvec![smaller_output.shape()[1], channels, fh, fw];
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
                let mut tiled_step_sizes = Shape::from_slice(smaller_output.step_sizes());
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
                let mut x_steps = Shape::from_slice(smaller_output.step_sizes());
                x_steps[usize::from(*scan_dim)] = spec_shape[usize::from(*scan_dim)];
                let maxes_shape = smaller_output.shape().clone();
                let maxes_steps = Shape::from_slice(smaller_output.step_sizes());
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
                let mut input_steps = Shape::from_slice(smaller_output.step_sizes());
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
                let shape = Shape::from_slice(&smaller_output.shape()[1..]);
                let steps = Shape::from_slice(&smaller_output.step_sizes()[1..]);
                let bindings = once(None)
                    .chain((1..shape.len()).map(|v| Some(v.try_into().unwrap())))
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
                        match args.first_input_shape.as_deref() {
                            Some([b, c, h, w]) => {
                                let (h_val, w_val, b_val, c_val) =
                                    (h.get(), w.get(), b.get(), c.get());
                                (1..=max_size.min(h_val), 1..=max_size.min(w_val))
                                    .prop_flat_map(move |(fh, fw)| {
                                        let h_add_1 = h_val - fh + 1;
                                        let w_add_1 = w_val - fw + 1;
                                        (1..=max_size).prop_map(move |f| {
                                            vec![b_val, f, c_val, h_add_1, w_add_1, fh, fw]
                                        })
                                    })
                                    .sboxed()
                            }
                            Some(_) => panic!("Conv requires a rank-4 first input"),
                            None => proptest::collection::vec(1..=max_size, 7).sboxed(),
                        }
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
                smallvec![lhs[0], lhs[1], lhs[2], rhs[2]]
            }
            PrimitiveSpecType::Conv { accum: _ } => {
                let lhs = parameter_shapes.next().unwrap();
                let rhs = parameter_shapes.next().unwrap();
                let out = parameter_shapes.next().unwrap();
                let [b, c, h, w] = *lhs else {
                    panic!();
                };
                let [f, alt_c, fh, fw] = *rhs else { panic!() };
                assert_eq!(c, alt_c);
                assert!(
                    h.get() >= fh.get(),
                    "Image height {} must be >= filter height {}",
                    h.get(),
                    fh.get()
                );
                assert!(
                    w.get() >= fw.get(),
                    "Image width {} must be >= filter width {}",
                    w.get(),
                    fw.get()
                );
                debug_assert_eq!(
                    out.iter().map(|v| v.get()).collect::<Vec<_>>(),
                    [
                        b.get(),
                        f.get(),
                        h.get() - fh.get() + 1,
                        w.get() - fw.get() + 1
                    ],
                    "unexpected output shape: {out:?}"
                );
                let h_add = h.get() - fh.get();
                let w_add = w.get() - fw.get();
                let h_add_1 = DimSize::new(h_add + 1).unwrap();
                let w_add_1 = DimSize::new(w_add + 1).unwrap();
                smallvec![b, f, c, h_add_1, w_add_1, fh, fw]
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

    #[inline]
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
                Some(smallvec![*b, *m, *n])
            }
            PrimitiveSpecType::Conv { .. } => {
                let ([b, _, h, w], [f, _, fh, fw]) = (inputs[0], inputs[1]) else {
                    panic!("Conv inputs must have 4 dimensions each");
                };
                assert!(
                    h.get() >= fh.get(),
                    "Image height {h} must be >= filter height {fh}",
                );
                assert!(
                    w.get() >= fw.get(),
                    "Image width {w} must be >= filter width {fw}",
                );
                Some(smallvec![
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
                let mut output = smallvec![];
                output.reserve_exact(input.len() + 1);
                output.push(nz!(1u32));
                output.extend_from_slice(input);
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
                Some(Shape::from_slice(inputs[0]))
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
            PrimitiveSpecType::Fill {
                value: FillValue::Min,
            } => write!(f, "FillMin"),
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

    pub fn parameter_levels(&self) -> Vec<Tgt::Level> {
        match self {
            LogicalSpec::Primitive(_, auxes, _) => auxes.iter().map(|aux| aux.level).collect(),
            LogicalSpec::Compose { operand_auxes, .. } => {
                operand_auxes.iter().map(|aux| aux.level).collect()
            }
        }
    }

    pub fn parameter_level(&self, idx: usize) -> Tgt::Level {
        match self {
            LogicalSpec::Primitive(_, auxes, _) => auxes[idx].level,
            LogicalSpec::Compose { operand_auxes, .. } => operand_auxes[idx].level,
        }
    }

    pub fn parameters(&self) -> Vec<TensorSpec<Tgt>> {
        match self {
            LogicalSpec::Primitive(basics, auxes, _) => {
                debug_assert_eq!(basics.typ.operand_count(), basics.dtypes.len());
                debug_assert_eq!(basics.typ.operand_count(), auxes.len());
                basics
                    .dtypes
                    .iter()
                    .zip(auxes)
                    .enumerate()
                    .map(|(i, (dt, a))| {
                        TensorSpec::new_noncanon_with_aux(basics.parameter_shape(i), *dt, a.clone())
                    })
                    .collect()
            }
            LogicalSpec::Compose {
                components,
                operand_auxes,
                serial_only: _,
            } => {
                debug_assert!(
                    components.len() >= 2,
                    "Compose must have at least 2 components, but components are: {components:?}"
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
                        check_tensor_vector_size::<Tgt>(shape, *dtype, &aux.level, vs)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;

                        aux.canonicalize(shape)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;
                    }

                    // It source and destination are fully contiguous and the dtypes and layouts
                    // match, then we can canonicalize to a row-major bitwise move (or an empty
                    // Layout for RF). This is a workaround for not being able to split interleaved
                    // layouts with a tile, but can be generalized to be a useful symmetry-breaking
                    // predicate later on.
                    // TODO: Do just that: generalize this canonicalization rule.
                    if basics.dtypes.iter().all_equal()
                        && primitive_aux.iter().map(|a| &a.layout).all_equal()
                        && primitive_aux
                            .iter()
                            .all(|aux| aux.layout.is_fully_contiguous())
                    {
                        for aux in primitive_aux.iter_mut() {
                            if aux.level.has_layout() {
                                aux.layout = row_major(shape);
                            }
                        }
                    }
                }
                _ => {
                    for (shp, dtype, aux) in
                        izip!(basics.parameter_shapes(), &basics.dtypes, primitive_aux)
                    {
                        let vs = aux.vector_size;
                        check_tensor_vector_size::<Tgt>(&shp, *dtype, &aux.level, vs)
                            .map_err(CanonicalizeError::TensorSpecAuxCanonicalizeError)?;

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
                    if let Err(e) = check_tensor_vector_size::<Tgt>(&shape, dtype, level, vs) {
                        status_to_return =
                            Err(CanonicalizeError::TensorSpecAuxCanonicalizeError(e));
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
                        if check_tensor_vector_size::<Tgt>(shape, *dtype, &aux.level, vs).is_err() {
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
                            .all(|aux| aux.layout.is_fully_contiguous())
                        && primitive_aux.iter().any(|aux| {
                            !aux.layout.is_row_major() || !aux.layout.is_fully_contiguous()
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
                        if check_tensor_vector_size::<Tgt>(&shp, *dtype, &aux.level, vs).is_err() {
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
                    if check_tensor_vector_size::<Tgt>(&shape, dtype, level, vs).is_err() {
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

    pub(crate) fn input_tilings_for_tile_out(
        &self,
        smaller_output: &Tiling,
    ) -> Option<LogicalSpecInputTilingInference> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                let ti = basics.input_tilings_for_tile_out(smaller_output)?;
                let mut component_parameter_shapes: Vec<Vec<Shape>> =
                    vec![ti.0.iter().map(|(t, _)| t.shape().clone()).collect()];
                component_parameter_shapes[0].insert(
                    basics.typ.unique_output_index().unwrap(),
                    smaller_output.shape().clone(),
                );
                Some(LogicalSpecInputTilingInference {
                    input_tilings: ti.clone(),
                    component_parameter_shapes,
                })
            }
            LogicalSpec::Compose { components, .. } => {
                let mut compose_input_tilings = Vec::with_capacity(self.operand_count() - 1);
                let mut component_input_tilings =
                    Vec::<Vec<Shape>>::with_capacity(components.len());

                // Compute [TilingInference] for the outermost (last-executed) component.
                // `accumulated_input_tilings` will contain the [Tiling]s and dimensions for all but
                // the first input of that outermost component (those which don't consume the next
                // component's output).
                let mut first_inference =
                    components[0].input_tilings_for_tile_out(smaller_output)?;
                compose_input_tilings.extend_from_slice(&first_inference.0[1..]);
                component_input_tilings.push(
                    first_inference
                        .0
                        .iter()
                        .map(|(t, _)| t.shape().clone())
                        .collect(),
                );
                component_input_tilings[0].insert(
                    components[0].typ.unique_output_index().unwrap(),
                    smaller_output.shape().clone(),
                );

                let mut last_output_tiling = first_inference.0.remove(0).0;
                for subspec in &components[1..components.len() - 1] {
                    let mut input_tilings =
                        subspec.input_tilings_for_tile_out(&last_output_tiling)?;
                    component_input_tilings.push(
                        input_tilings
                            .0
                            .iter()
                            .map(|(t, _)| t.shape().clone())
                            .collect(),
                    );
                    component_input_tilings.last_mut().unwrap().insert(
                        subspec.typ.unique_output_index().unwrap(),
                        last_output_tiling.shape().clone(),
                    );
                    compose_input_tilings.extend(input_tilings.0.drain(1..));
                    last_output_tiling = input_tilings.0.remove(0).0;
                }

                let innermost_tiling = components[components.len() - 1]
                    .input_tilings_for_tile_out(&last_output_tiling)?;
                compose_input_tilings.extend_from_slice(&innermost_tiling.0);
                component_input_tilings.push(
                    innermost_tiling
                        .0
                        .iter()
                        .map(|(t, _)| t.shape().clone())
                        .collect(),
                );
                component_input_tilings.last_mut().unwrap().insert(
                    components[components.len() - 1]
                        .typ
                        .unique_output_index()
                        .unwrap(),
                    last_output_tiling.shape().clone(),
                );

                Some(LogicalSpecInputTilingInference {
                    input_tilings: TilingInference(compose_input_tilings),
                    component_parameter_shapes: component_input_tilings,
                })
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
                        (
                            Shape::from_slice(last_operand.shape()),
                            last_operand.dtype(),
                        )
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
                _ => panic!("Cannot clone_as_accum: {self:?}"),
            },
            LogicalSpec::Compose { components, .. } => match &mut components[0].typ {
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Conv { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. } => {
                    *accum = true;
                }
                _ => panic!("Cannot clone_as_accum: {self:?}"),
            },
        }
        cloned
    }

    /// Returns the product of Spec dimensions.
    pub fn volume(&self) -> NonZeroU64 {
        match self {
            LogicalSpec::Primitive(basics, _, _) => {
                // Compute product in u64 to avoid overflow for large shapes.
                NonZeroU64::new(
                    basics
                        .spec_shape
                        .iter()
                        .map(|d| u64::from(d.get()))
                        .product(),
                )
                .unwrap()
            }
            LogicalSpec::Compose { .. } => {
                // Returning a 1 here basically disables intensity-scaling.
                // TODO: Return an actual volume.
                nz!(1u64)
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
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, _, _) => format!("{typ}"),
            LogicalSpec::Compose { .. } => todo!(),
        };

        let operand_str = self
            .parameters()
            .iter()
            .map(|o| format!("{o}"))
            .collect::<Vec<_>>()
            .join(", ");
        let serial_str = if self.serial_only() { ", serial" } else { "" };

        write!(f, "{header}({operand_str}{serial_str})")
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
                    panic!("Given non-zero/power-of-two shape {d}");
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
                let v: Vec<_> = once(!accum as _).chain(shifted_shape).collect();
                (
                    SpecKey::Conv {
                        dtypes: dtypes.as_slice().try_into().unwrap(),
                    },
                    v,
                )
            }
            PrimitiveSpecType::Broadcast { dim } => (
                SpecKey::Broadcast {
                    rank: spec_shape.len().try_into().unwrap(),
                    dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Softmax { scan_dim } => {
                assert_eq!(dtypes.len(), 2);
                (
                    SpecKey::Softmax {
                        rank: spec_shape.len().try_into().unwrap(),
                        scan_dim,
                        dtypes: dtypes.as_slice().try_into().unwrap(),
                    },
                    shifted_shape.collect(),
                )
            }
            PrimitiveSpecType::SoftmaxComplete { scan_dim } => (
                SpecKey::SoftmaxComplete {
                    rank: spec_shape.len().try_into().unwrap(),
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndMax { scan_dim } => (
                SpecKey::SoftmaxDenominatorAndMax {
                    rank: spec_shape.len().try_into().unwrap(),
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { scan_dim, accum } => (
                SpecKey::SoftmaxDenominatorAndUnscaled {
                    rank: spec_shape.len().try_into().unwrap(),
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, accum } => (
                SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                    rank: spec_shape.len().try_into().unwrap(),
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::SoftmaxDenominator { scan_dim, accum } => (
                SpecKey::SoftmaxDenominator {
                    rank: spec_shape.len().try_into().unwrap(),
                    scan_dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::Max { dim, accum } => (
                SpecKey::Max {
                    rank: spec_shape.len().try_into().unwrap(),
                    dim,
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                once(!accum as _).chain(shifted_shape).collect(),
            ),
            PrimitiveSpecType::OnePrefix => (
                SpecKey::OnePrefix {
                    rank: spec_shape.len().try_into().unwrap(),
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Move => (
                SpecKey::Move {
                    rank: spec_shape.len().try_into().unwrap(),
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::Fill { value } => (
                SpecKey::Fill {
                    rank: spec_shape.len().try_into().unwrap(),
                    value,
                    dtype: dtypes[0],
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::DivideVec => (
                SpecKey::DivideVec {
                    rank: spec_shape.len().try_into().unwrap(),
                    dtypes: dtypes.as_slice().try_into().unwrap(),
                },
                shifted_shape.collect(),
            ),
            PrimitiveSpecType::DivideVecScalar { scan_dim } => (
                SpecKey::DivideVecScalar {
                    rank: spec_shape.len().try_into().unwrap(),
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
                rank: _,
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominator {
                rank: _,
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                rank: _,
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::Max {
                rank: _,
                dim: _,
                dtypes: _,
            } => {
                let accum = v[0] == 0;
                let (typ, dtypes) = match key {
                    SpecKey::Matmul { dtypes } => {
                        (PrimitiveSpecType::Matmul { accum }, dtypes.to_vec())
                    }
                    SpecKey::Conv { dtypes } => {
                        (PrimitiveSpecType::Conv { accum }, dtypes.to_vec())
                    }
                    SpecKey::SoftmaxDenominatorAndUnscaled {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
                        PrimitiveSpecType::SoftmaxDenominatorAndUnscaled {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
                        PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominator {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
                        PrimitiveSpecType::SoftmaxDenominator {
                            scan_dim: *scan_dim,
                            accum,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::Max {
                        rank: _,
                        dim,
                        dtypes,
                    } => (PrimitiveSpecType::Max { dim: *dim, accum }, dtypes.to_vec()),
                    _ => unreachable!(),
                };

                let mut spec_shape: Vec<BimapInt> = v.iter().skip(1).copied().collect();
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
                rank: _,
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxComplete {
                rank: _,
                scan_dim: _,
                dtypes: _,
            }
            | SpecKey::SoftmaxDenominatorAndMax {
                rank: _,
                scan_dim: _,
                dtypes: _,
            } => {
                let (typ, dtypes) = match key {
                    SpecKey::Softmax {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
                        PrimitiveSpecType::Softmax {
                            scan_dim: *scan_dim,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxComplete {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
                        PrimitiveSpecType::SoftmaxComplete {
                            scan_dim: *scan_dim,
                        },
                        dtypes.into(),
                    ),
                    SpecKey::SoftmaxDenominatorAndMax {
                        rank: _,
                        scan_dim,
                        dtypes,
                    } => (
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
            SpecKey::OnePrefix { rank: _, dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::OnePrefix,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Move { rank: _, dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::Move,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Fill {
                rank: _,
                value,
                dtype,
            } => PrimitiveBasics {
                typ: PrimitiveSpecType::Fill { value: *value },
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: vec![*dtype],
            },
            SpecKey::DivideVec { rank: _, dtypes } => PrimitiveBasics {
                typ: PrimitiveSpecType::DivideVec,
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::DivideVecScalar {
                rank: _,
                scan_dim,
                dtypes,
            } => PrimitiveBasics {
                typ: PrimitiveSpecType::DivideVecScalar {
                    scan_dim: *scan_dim,
                },
                spec_shape: BiMap::apply_inverse(&ShapeBimap(self.binary_scale_shapes), v),
                dtypes: dtypes.into(),
            },
            SpecKey::Broadcast {
                rank: _,
                dim,
                dtypes,
            } => PrimitiveBasics {
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
    type Domain = Shape;
    type Codomain = Vec<BimapInt>;

    fn apply(&self, shape: &Self::Domain) -> Self::Codomain {
        shape
            .iter()
            .map(|d| d.get())
            .map(|d| {
                if self.0 {
                    if !d.is_power_of_two() {
                        panic!("Given non-zero/power-of-two shape {d}");
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
                .unwrap_or_else(|| panic!("Component {c:?} has too few operands"))
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
    let mut steps = Shape::from_slice(output.step_sizes());
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
            let sv: $crate::common::Shape = smallvec::smallvec![ ($dim).into_dim_size(); $n ];
            sv
        }};
        ($($dim:expr),*$(,)*) => {{
            use $crate::spec::macros::internal::IntoDimSize;
            // Bind to a variable with an explicit type to help out type inference.
            let sv: $crate::common::Shape = smallvec::smallvec![ $( ($dim).into_dim_size() ),* ];
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

        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, c0 ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, None, true)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr, c0 ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), false)
        };
        ( @tensorspecaux $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr ) => {
            $crate::lspec!(@tensorspecaux_inner $shp, $dt, $level, $layout, Some($vs), true)
        };

        // TODO: Accept contiguousnesses other than fully contig. or not at all.
        ( @tensorspecaux_inner $shp:expr, $dt:tt, $level:expr, $layout:expr, $vs:expr,
          $c:literal ) =>
        {{
            let mut layout;
            if !$crate::target::MemoryLevel::has_layout(&($level)) {
                let _ = $shp; // make sure to evaluate $shp in case its a pop()
                layout = $crate::layout::Layout::empty();
            } else {
                layout = $crate::layout::LayoutBuilder::build($layout, $shp);
                if !($c) {
                    layout.set_contiguous_none();
                }
            }
            $crate::tensorspec::TensorSpecAux {
                level: ($level).into(),
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
    ///
    /// When providing limits, register file limits are given as register counts, and
    /// the remaining limits are given as bytes. Bytes must be multiples of the target's
    /// cache line size.
    #[macro_export]
    macro_rules! spec {
        // Literal limits array: build a Standard MemoryLimits
        ( $typ:ident $args:tt , [ $($limits:expr),* $(,)? ] ) => {{
            let logical_spec = $crate::lspec!($typ $args);
            let limits_arr: [u64; $crate::target::LEVEL_COUNT] = [ $( $limits as u64 ),* ];
            $crate::spec::__private::new_with_standard_memory(logical_spec, limits_arr)
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

#[doc(hidden)] // not part of the public API surface
pub mod __private {
    use super::{LogicalSpec, Spec};
    use crate::memorylimits::{MemVec, MemoryLimits};
    use crate::target::{MemoryLevel, Target, LEVEL_COUNT};

    /// A helper constructor for the `spec!` macro. This exists so that the inferred `Tgt` can be
    /// given to [MemVec::new_for_target].
    pub fn new_with_standard_memory<Tgt: Target>(
        logical_spec: LogicalSpec<Tgt>,
        mut limits_arr: [u64; LEVEL_COUNT],
    ) -> Spec<Tgt> {
        // Since `spec!` takes bytes for non-register-file levels, convert those
        // to cache lines.
        let line_size = u64::from(Tgt::line_size());
        debug_assert!(line_size > 0);
        for (idx, level) in Tgt::levels().iter().enumerate() {
            if !level.counts_registers() {
                let bytes = limits_arr[idx];
                if !bytes.is_multiple_of(line_size) {
                    panic!("Bytes ({bytes}) must be a multiple of line size ({line_size})");
                }
                limits_arr[idx] = bytes / line_size;
            }
        }
        Spec(
            logical_spec,
            MemoryLimits::Standard(MemVec::new_for_target::<Tgt>(limits_arr)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::general::AsBimap;
    use crate::imp::subspecs::SpecApp;
    use crate::imp::{Impl, ImplNode};
    use crate::layout::row_major;
    use crate::memorylimits::{arb_memorylimits_ext, MemVec, MemoryAllocation};
    use crate::scheduling::tiling::TileOut;
    use crate::scheduling::{Action, ActionT as _, ApplyError};
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::target::CpuMemoryLevel::{GL, L1, RF};
    use crate::target::{ArmTarget, Avx2Target, CpuMemoryLevel, MemoryLevel, Target, LEVEL_COUNT};
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
        let spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [32, 2, 3, 3],
            (u8, GL, row_major),
            (i8, GL, row_major, c0),
            (u16, GL, row_major),
            serial
        ));
        let lhs = TensorSpecAux {
            level: GL,
            layout: row_major(&shape![32, 2, 3]),
            vector_size: None,
        };
        let mut rhs_layout = row_major(&shape![32, 3, 3]);
        rhs_layout.set_contiguous_none();
        let rhs = TensorSpecAux {
            level: GL,
            layout: rhs_layout,
            vector_size: None,
        };
        let out = TensorSpecAux {
            level: GL,
            layout: row_major(&shape![32, 2, 3]),
            vector_size: None,
        };
        let expected = LogicalSpec::<Avx2Target>::Primitive(
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
        let s: Spec<Avx2Target> = spec!(
            Move([2, 3], (u32, GL, row_major), (u32, GL, row_major)),
            [8, 8, 256, 128]
        );
        let expected_ls: LogicalSpec<Avx2Target> =
            lspec!(Move([2, 3], (u32, GL, row_major), (u32, GL, row_major)));
        assert_eq!(s.0, expected_ls);
        assert_eq!(
            s.1,
            MemoryLimits::Standard(MemVec::new_mixed([8, 8, 8, 4], [true, true, false, false]))
        );
    }

    #[test]
    fn test_compose_parameters() {
        let spec = matmul_chain_test_data();
        let LogicalSpec::Compose {
            components,
            operand_auxes,
            serial_only: _,
        } = &spec
        else {
            unreachable!();
        };

        let expected_parameters: Vec<TensorSpec<Avx2Target>> = vec![
            TensorSpec::new_noncanon_with_aux(
                [0, 2, 3]
                    .into_iter()
                    .map(|i| components[0].spec_shape[i])
                    .collect(),
                components[0].dtypes[1],
                operand_auxes[0].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 2, 3]
                    .into_iter()
                    .map(|i| components[1].spec_shape[i])
                    .collect(),
                components[1].dtypes[1],
                operand_auxes[1].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                Shape::from_slice(&components[2].spec_shape[..3]),
                components[2].dtypes[0],
                operand_auxes[2].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 2, 3]
                    .into_iter()
                    .map(|i| components[2].spec_shape[i])
                    .collect(),
                components[2].dtypes[1],
                operand_auxes[3].clone(),
            ),
            TensorSpec::new_noncanon_with_aux(
                [0, 1, 3]
                    .into_iter()
                    .map(|i| components[0].spec_shape[i])
                    .collect(),
                components[0].dtypes[2],
                operand_auxes.last().unwrap().clone(),
            ),
        ];

        assert_eq!(spec.parameters(), expected_parameters);
    }

    #[test]
    fn test_conv_input_tilings_for_tile_out() {
        let spec_shape = shape![
            1, // batch size
            2, // output filters
            3, // input channels
            4, // additional img. height + 1 (img. height = 2 + 4 - 1 = 5)
            5, // additional img. width + 1 (img. width = 2 + 5 - 1 = 6)
            2, // filter height
            2, // filter width
        ];
        let basics = PrimitiveBasics {
            typ: PrimitiveSpecType::Conv { accum: false },
            spec_shape,
            dtypes: vec![Dtype::Uint8, Dtype::Uint8, Dtype::Uint8],
        };
        assert_eq!(
            basics.unique_output_shape().unwrap(),
            shape![1, 2, 4, 5],
            "initial output shape should be [1, 2, 4, 5]"
        );

        let output_tiling = Tiling::new_simple(shape![1, 2, 4, 3]);

        let mut steps: Shape = output_tiling.step_sizes().into();
        steps[1] = nz!(3u32);
        let expected_img_tiling = Tiling::new_sliding(shape![1, 3, 5, 4], steps);

        let mut steps: Shape = shape![2, 3, 2, 2];
        steps[0] = output_tiling.step_sizes()[1];
        let expected_filt_tiling = Tiling::new_sliding(shape![2, 3, 2, 2], steps);

        assert_eq!(
            basics.input_tilings_for_tile_out(&output_tiling),
            Some(TilingInference(vec![
                (expected_img_tiling, vec![Some(0), None, None, None]),
                (expected_filt_tiling, vec![None, Some(1), None, None]),
            ]))
        );
    }

    #[test]
    fn test_compose_canonicalization_accepts_accmulating_head() {
        let mut spec = matmul_chain_test_data();
        let LogicalSpec::Compose { components, .. } = &mut spec else {
            unreachable!();
        };
        components[0].typ = PrimitiveSpecType::Matmul { accum: true };
        assert!(spec.canonicalize().is_ok());
    }

    #[test]
    fn test_compose_canonicalization_rejects_accmulating_tail_components() {
        let mut spec = matmul_chain_test_data();
        let LogicalSpec::Compose { components, .. } = &mut spec else {
            unreachable!();
        };
        components[1].typ = PrimitiveSpecType::Matmul { accum: true };
        assert!(spec.canonicalize().is_err());
    }

    #[test]
    fn test_compose_input_tiling_inference() {
        let spec = matmul_chain_test_data();
        let output_tiling = Tiling::new_simple(shape![8, 32, 128]);
        let ci = spec.input_tilings_for_tile_out(&output_tiling).unwrap();

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
        assert_eq!(ci.input_tilings, expected);
        assert_eq!(ci.component_parameter_shapes.len(), 3); // 3 components in the compose
        for component_tilings in &ci.component_parameter_shapes {
            assert_eq!(component_tilings.len(), 3);
        }

        let expected_comp0 = vec![
            shape![8, 32, 128],  // first input (from component 1's output)
            shape![8, 128, 128], // second input (external)
            shape![8, 32, 128],  // output (final result)
        ];
        assert_eq!(ci.component_parameter_shapes[0], expected_comp0);

        let expected_comp1 = vec![
            shape![8, 32, 128],  // first input (from component 2's output)
            shape![8, 128, 128], // second input (external)
            shape![8, 32, 128],  // output (feeds into component 0)
        ];
        assert_eq!(ci.component_parameter_shapes[1], expected_comp1);

        let expected_comp2 = vec![
            shape![8, 32, 128],  // first input (external)
            shape![8, 128, 128], // second input (external)
            shape![8, 32, 128],  // output (feeds into component 1)
        ];
        assert_eq!(ci.component_parameter_shapes[2], expected_comp2);
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
            spec in any::<LogicalSpec<Avx2Target>>()
        ) {
            let shapes_left = spec.parameter_shapes();
            let shapes_right = (0..spec.operand_count())
                .map(|i| spec.parameter_shape(i))
                .collect::<Vec<_>>();
            prop_assert_eq!(shapes_left, shapes_right);
        }

        #[test]
        fn test_parameter_levels_matches_parameters_levels(
            spec in any::<LogicalSpec<Avx2Target>>()
        ) {
            let levels_from_parameter_levels = spec.parameter_levels();
            let levels_from_parameters = spec.parameters().iter().map(|p| p.level()).collect::<Vec<_>>();
            prop_assert_eq!(levels_from_parameter_levels, levels_from_parameters);
        }

        #[test]
        fn test_parameter_levels_matches_parameters_levels_arm(
            spec in any::<LogicalSpec<ArmTarget>>()
        ) {
            let levels_from_parameter_levels = spec.parameter_levels();
            let levels_from_parameters = spec.parameters().iter().map(|p| p.level()).collect::<Vec<_>>();
            prop_assert_eq!(levels_from_parameter_levels, levels_from_parameters);
        }

        // TODO: Add ARM variant
        #[test]
        fn test_parameter_level_matches_parameter_levels_avx2(
            spec in any::<LogicalSpec<Avx2Target>>()
        ) {
            for (i, level) in spec.parameter_levels().into_iter().enumerate() {
                prop_assert_eq!(level, spec.parameter_level(i));
            }
        }

        #[test]
        fn test_no_action_panics_avx2(spec in arb_canonical_spec::<Avx2Target>(None, None)) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        fn test_no_action_panics_arm(spec in arb_canonical_spec::<ArmTarget>(None, None)) {
            shared_test_no_action_panics(spec);
        }

        #[test]
        #[ignore]
        fn test_actions_are_valid_through_consumed_memory_avx2(
            logical_spec in arb_canonical_logical_spec::<Avx2Target>(Some(TEST_SMALL_SIZE))
        ) {
            shared_test_actions_are_valid_through_consumed_memory(logical_spec)
        }

        #[test]
        #[ignore]
        fn test_actions_are_valid_through_consumed_memory_arm(
            logical_spec in arb_canonical_logical_spec::<Avx2Target>(Some(TEST_SMALL_SIZE))
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
            logical_spec in any::<LogicalSpec<Avx2Target>>()
        ) {
            prop_assert_eq!(logical_spec.parameters().len(), logical_spec.operand_count());
        }

        #[test]
        fn test_canonicalize_is_noop_if_already_canonical(
            logical_spec in any::<LogicalSpec<Avx2Target>>()
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
            logical_spec in any::<LogicalSpec<Avx2Target>>()
        ) {
            let mut logical_spec = logical_spec;
            match logical_spec.canonicalize() {
                Ok(()) => {
                    for p in logical_spec.parameters() {
                        let mut recanonicalized = p.clone();
                        recanonicalized.canonicalize().unwrap_or_else(|e| {
                            panic!("Couldn't canonicalize parameter {p} even though {logical_spec} was canon: {e}");
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
                            .map(|s| any_with::<TensorSpecAux<Avx2Target>>((TensorSpecArbMaxShape(s), Some(dtype))))
                            .collect::<Vec<_>>();
                        (Just(basics), auxes_strategy, any::<bool>())
                    })
                    .prop_filter("No operand should be in a vector register file", |(_, auxes, _)| {
                        auxes.iter().all(|aux| !aux.level.vector_rf())
                    })
                    .prop_filter_map("Spec should be canonical", |(basics, auxes, serial_only)| {
                        let s = Spec(LogicalSpec::Primitive(basics, auxes, serial_only), Avx2Target::max_mem());
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
                .apply(&spec).unwrap_or_else(|e| panic!("Couldn't tile Spec {spec} to single value: {e:?}"));
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
            spec in arb_canonical_spec::<Avx2Target>(None, None)
        ) {
            for action in Avx2Target::actions(&spec.0) {
                if let Ok(ImplNode::Alloc(alloc)) = action.apply(&spec) {
                    assert_ne!(&alloc.source_spec, alloc.introduced.spec(),
                        "Copying Alloc introduced by action {action:?}");
                }
            }
        }

        #[test]
        fn test_action_applies_everywhere_down_through_peak_memory(
            (spec, action, _, lower_limit) in arb_spec_action_and_lower_limit::<Avx2Target>()
        ) {
            let lower_spec = Spec(spec.0.clone(), lower_limit);
            assert!(Avx2Target::actions(&lower_spec.0).contains(&action),
                "Action {action:?} was not present in lower-limits Spec {lower_spec:?}");
        }

        #[test]
        fn test_no_action_produces_same_spec_with_higher_memory_limit_avx2(
            spec in arb_canonical_spec::<Avx2Target>(None, None)
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
            spec in arb_canonical_spec::<Avx2Target>(None, None)
        ) {
            Avx2Target::actions(&spec.0).for_each(|action| {
                let Ok(applied) = action.apply(&spec) else {
                    return;
                };
                applied.visit_leaves(&mut |leaf| {
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
        fn test_actions_produce_valid_layouts(
            spec in arb_canonical_spec::<Avx2Target>(None, None)
        ) {
            Avx2Target::actions(&spec.0).for_each(|action| {
                let Ok(applied) = action.apply(&spec) else {
                    return;
                };
                applied.visit_leaves(&mut |leaf| {
                    if let ImplNode::SpecApp(SpecApp(spec, args)) = leaf {
                        let chained_tensorspecs = spec.0.parameters().into_iter().chain(args.iter().map(|a| a.spec()).cloned());
                        for tspec in chained_tensorspecs {
                            if tspec.level().has_layout() {
                                assert!(!tspec.layout().is_empty() || tspec.shape().iter().all(|d| d.get() == 1));
                            } else {
                                assert!(tspec.layout().is_empty());
                            }
                        }
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
        fn test_specbimap_is_invertible_avx2(spec in any::<Spec<Avx2Target>>()) {
            // binary-scaled SurMap is not tested
            shared_test_specbimap_is_invertible(spec, false);
        }


        #[test]
        fn test_specbimap_is_invertible_arm(spec in any::<Spec<ArmTarget>>()) {
            // binary-scaled SurMap is not tested
            shared_test_specbimap_is_invertible(spec, false);
        }

        #[test]
        fn test_parameter_fn_matches_parameters_fn(spec in any::<LogicalSpec<Avx2Target>>()) {
            let parameters = spec.parameters();
            let individual_parameters = (0..parameters.len()).map(|i| spec.parameter(i)).collect::<Vec<_>>();
            prop_assert_eq!(parameters, individual_parameters);
        }

        #[test]
        #[should_panic]
        fn test_parameter_fn_panics_above_parameter_len(spec in any::<LogicalSpec<Avx2Target>>()) {
            spec.parameter(spec.parameters().len());
        }

        #[test]
        #[should_panic]
        fn test_parameter_fn_panics_above_parameter_count(spec in any::<LogicalSpec<Avx2Target>>()) {
            spec.parameter(spec.operand_count());
        }

        #[test]
        fn test_parameters_match_parameter_shapes(spec in any::<LogicalSpec<Avx2Target>>()) {
            let parameters = spec.parameters();
            let parameter_shapes = spec.parameter_shapes();
            prop_assert_eq!(parameters.len(), parameter_shapes.len());
            for (p, s) in parameters.iter().zip(&parameter_shapes) {
                assert_eq!(p.shape(), s.as_slice());
            }
        }

        #[test]
        fn test_replace_io_noop(logical_spec in any::<LogicalSpec<Avx2Target>>()) {
            let mut replaced = logical_spec.clone();
            replaced.replace_io(&logical_spec.parameters());
            prop_assert_eq!(logical_spec, replaced);
        }

        #[test]
        fn test_bufferized_compose_parameters_match_pipeline_parameters(
            tinp in arb_compose_spec::<Avx2Target>(None)
                .prop_filter_map("Spec was not canonical", |logical_spec| {
                    let mut s = Spec(logical_spec, Avx2Target::max_mem());
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
                        Avx2Target::levels().iter().position(|l| l == &CpuMemoryLevel::RF).unwrap();
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
            let pipeline_parameters = pipeline.collect_unbound_parameters();
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
            applied.visit_leaves(&mut |leaf| {
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
        fn peak_memory<Tgt: Target>(imp: &ImplNode<Tgt>) -> MemVec {
            let children = imp.children();
            let mut child_peaks = Vec::with_capacity(children.len());
            for child in children {
                child_peaks.push(peak_memory(child));
            }
            imp.memory_allocated()
                .peak_memory_from_child_peaks::<Tgt>(&child_peaks)
        }

        // If an action consumes x bytes, then it should be valid for any Spec
        // with the same logical Spec at that memory limit and up. To keep
        // performance acceptable, we limit memory.
        let mut maxes = vec![];
        let MemoryLimits::Standard(maxes_vec) = Tgt::max_mem();
        for v in maxes_vec.iter() {
            maxes.push(v.min(64));
        }

        // Zero out levels which are slower than all present operands' levels.
        let parameter_levels = logical_spec.parameter_levels();
        for (level_idx, level) in Tgt::levels().into_iter().enumerate() {
            if parameter_levels.iter().all(|l| *l < level) {
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
                let Ok(pt_arr): Result<[u64; LEVEL_COUNT], _> = pt.try_into() else {
                    panic!("Expected length {LEVEL_COUNT}");
                };
                shared_spec.1 = MemoryLimits::Standard(MemVec::new_for_target::<Tgt>(pt_arr));
                assert!(shared_spec.is_canonical());
                let MemoryLimits::Standard(limits_memvec) = &shared_spec.1;
                // TODO: Assert that nothing disappears?
                for i in (0..unseen_actions.len()).rev() {
                    // We could use `apply` for safety, but instead we use `apply_unchecked_canon`
                    // since we assert the shared_spec is canonical outside the loop.
                    match unseen_actions[i].apply_unchecked_canon(&shared_spec) {
                        Ok(applied) => {
                            unseen_actions.swap_remove(i);
                            // TODO: Should we also assert that applying the same action at each level
                            //   doesn't actually accumulate additional memory?.
                            // TODO: Can we assert that the change in peak memory is exactly the
                            //   additional amount at the limit?.
                            // TODO: Assert here that the min of each level-wise limit is zero.
                            assert_eq!(&peak_memory(&applied), limits_memvec);
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
                let levels = Tgt::levels();
                let mut corrected_lower_bound = lower_bound;
                for i in 0..LEVEL_COUNT {
                    if !levels[i].counts_registers() {
                        corrected_lower_bound[i] = next_binary_power(corrected_lower_bound[i]);
                    }
                }
                assert!(
                    corrected_lower_bound
                        .iter()
                        .copied()
                        .enumerate()
                        .all(|(i, a)| a <= limits_memvec.get_unscaled(i)),
                    "Lower bound {corrected_lower_bound:?} was greater than Spec's memory limits {limits_memvec:?}"
                );
                let candidate_lower = MemVec::new(corrected_lower_bound);
                let lower_limit_strategy =
                    arb_memorylimits_ext::<Tgt>(&candidate_lower, limits_memvec);
                (
                    Just(spec),
                    Just(action),
                    Just(applied),
                    lower_limit_strategy,
                )
            })
    }

    fn matmul_chain_test_data() -> LogicalSpec<Avx2Target> {
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
            level: GL,
            layout: row_major(&basic0.parameter_shape(0)),
            vector_size: None,
        };
        let aux1_1 = TensorSpecAux {
            level: L1,
            layout: row_major(&basic1.parameter_shape(1)),
            vector_size: None,
        };
        let aux2_0 = TensorSpecAux {
            level: GL,
            layout: row_major(&basic2.parameter_shape(0)),
            vector_size: None,
        };
        let aux2_1 = TensorSpecAux {
            level: L1,
            layout: row_major(&basic2.parameter_shape(1)),
            vector_size: None,
        };
        let aux0_out = TensorSpecAux {
            level: RF,
            layout: row_major(&basic0.parameter_shape(2)),
            vector_size: None,
        };

        LogicalSpec::Compose {
            components: vec![basic0.clone(), basic1.clone(), basic2.clone()],
            operand_auxes: vec![aux0_1, aux1_1, aux2_0, aux2_1, aux0_out],
            serial_only: false,
        }
    }
}
