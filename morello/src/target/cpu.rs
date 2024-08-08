use super::common_actions::{move_actions, split_actions, tile_out_actions};
use crate::codegen::c_utils::VecType;
use crate::common::{DimSize, Dtype};
use crate::cost::MainCost;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::layout::{batched_col_major, col_major, nhwc, row_major, Layout, PhysDim};
use crate::memorylimits::{MemVec, MemoryAllocation, MemoryLimits};
use crate::scheduling::bufferize::Bufferize;
use crate::scheduling::select::Select;
use crate::scheduling::spatial_split::SpatialSplit;
use crate::scheduling::to_accum::ToAccum;
use crate::scheduling::Action;
use crate::shape;
use crate::spec::{FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use crate::target::{Kernel, MemoryLevel, Target, TargetId, LEVEL_COUNT};
use crate::tensorspec::{gen_vector_sizes, TensorSpec, TensorSpecAux};
use crate::views::View;

use divrem::DivRem;
use itertools::Itertools;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::{self, once};

const INST_COST: MainCost = 100;
const ASSIGN_INST_COST: MainCost = 1;
const CPU_LEVELS: [CpuMemoryLevel; 4] = [
    CpuMemoryLevel::RF,
    CpuMemoryLevel::VRF,
    CpuMemoryLevel::L1,
    CpuMemoryLevel::GL,
];
pub(crate) const DOT_PRODUCT_STRIP_SIZE: DimSize = nz!(8u32);
pub(crate) const DOT_PRODUCT_ACCUM_COUNT: u32 = 4;
pub(crate) const DOT_PRODUCT_BF16_STRIP_SIZE: DimSize = nz!(16u32);
pub(crate) const DOT_PRODUCT_BF16_ACCUM_COUNT: u32 = 4;

pub trait CpuTarget: Clone + Copy + std::hash::Hash + Eq + Default + Debug + 'static {
    type Kernel: Kernel<Tgt = Self> + From<CpuKernel>;
    fn target_id() -> TargetId;
    fn vec_types() -> &'static [VecType; 16];
}

#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    strum::VariantArray,
    strum::IntoStaticStr,
)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum CpuKernel {
    /// Does nothing, but implements OnePrefix Specs, passing input through to output.
    OnePrefixNoOp,
    /// Simple scalar multiplication (`+=`).
    MultAdd,
    /// Lowers to Clang's scalar-vector multiply-accumulate.
    BroadcastVecMultAdd,
    BroadcastVecMultAddBf16F32, // TODO: Move into X86 kernel.
    /// Lowers to an outer product 1x2xN MatmulAccum implementation taking u8 and s8 inputs and
    /// producing a s16 output.
    ///
    /// The first argument is broadcast over the N dimension with
    /// [_mm256_set1_epi16](https://www.felixcloutier.com/x86/vpbroadcastb:vpbroadcastw:vpbroadcastd:vpbroadcastq):
    /// ```text
    ///                        0 1 2 3
    ///      0 1 2 3           4 5 6 7
    ///      4 5 6 7       a b
    /// a b            →   a b
    ///                    a b
    ///                    a b
    /// ```
    ///
    /// and then the Hadamard is accumulated into the output with
    /// [_mm256_maddubs_epi16](https://www.felixcloutier.com/x86/pmaddubsw)
    /// and [_mm256_add_epi16](https://www.felixcloutier.com/x86/paddb:paddw:paddd:paddq):
    /// ```text
    /// += 0a 1a 2a 3a
    ///    4b 5b 6b 7b
    /// ```
    TwoVecBroadcastVecMultAddU8S8S16,
    DotProductLoop,
    /// Lowers to a dot product loop with 4 accumulating registers which takes two bf16
    /// vectors and produces a vector of float32s. The inputs are dequantized immediately
    /// after being loaded and before multiplication.
    DotProductLoopBf16Bf16F32,
    /// Lowers to a dot product loop with 4 accumulating registers which scans a buffer
    /// of float32 values and a buffer of bfloat16 values, producing float32s. The f32
    /// inputs are dequantized immediately after being loaded and before multiplcation.
    DotProductLoopF32Bf16F32,
    DotProductLoopF32InterleavedBf16F32,
    PhysicalTransposeByte128,
    PhysicalTransposeByte256,
    VectorInterleaveBf16F32,
    VectorDeinterleaveF32Bf16,
    ValueSoftmaxComplete,
    ValueSoftmaxDenominator,
    VectorSoftmaxDenominator,
    VectorSoftmaxComplete,
    VectorSoftmaxDenominatorAndUnscaledF32,
    ValueMax,
    VectorMax, // TODO: Add F32 to name
    VecScalarAssign,
    DivideVec,
    ValueAssign,
    VectorAssign,
    MemsetZero,
    /// Lowers to Clang vector extensions' zero-assignment, which, on x86, should emit `vxorps`.
    VectorZero,
    ValueNegInf,
    VectorNegInf,
    CastBf16F32,
    VectorCastBf16F32,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Eq, PartialEq, Debug, Copy, Clone, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum CpuMemoryLevel {
    RF,
    VRF,
    L1,
    GL,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuMemoryLevelBimap;

impl<T: CpuTarget> Target for T {
    type Level = CpuMemoryLevel;
    type Kernel = <Self as CpuTarget>::Kernel;
    type ActionsIter<'a> = Box<dyn Iterator<Item = Action<Self>> + 'a>;

    fn line_size() -> u32 {
        32
    }

    fn max_mem() -> MemoryLimits {
        MemoryLimits::Standard(MemVec::new([64, 1024, 32_768, 1_073_741_824]))
    }

    fn processors() -> u8 {
        32
    }

    fn default_level() -> Self::Level {
        CpuMemoryLevel::GL
    }

    fn levels() -> [Self::Level; LEVEL_COUNT] {
        CPU_LEVELS
    }

    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => vec![slower],
            CpuMemoryLevel::L1 => vec![slower, CpuMemoryLevel::RF, CpuMemoryLevel::VRF],
            CpuMemoryLevel::GL => vec![slower, CpuMemoryLevel::L1],
        }
    }

    fn all_layouts_for_shape(shape: &[DimSize], dtype: Dtype) -> Vec<Layout> {
        assert!(!shape.is_empty());

        let all_target_vector_bytes = Self::levels()
            .into_iter()
            .flat_map(|lvl| lvl.vector_bytes().iter().copied())
            .collect::<Vec<_>>();

        // The following could be faster. It keeps two copies of the non-packed layouts
        // (`base` and the first few values in `result`) and it doesn't compute the size
        // of the `result` ahead-of-time, potentially causing some Vec resizes.
        let rank = u8::try_from(shape.len()).unwrap();
        let unpacked_layouts = match rank {
            2 => vec![row_major(2), col_major(2)],
            4 => vec![row_major(4), nhwc()],
            _ => vec![row_major(rank)],
        };

        // Extend with all possible packings.
        // TODO: Reduce code duplication in the following
        let mut result = unpacked_layouts.clone();

        let all_packing_sizes = pack_sizes_all(dtype, &all_target_vector_bytes).collect::<Vec<_>>();
        result.extend(unpacked_layouts.iter().flat_map(|original_layout| {
            let Layout(dims) = original_layout;
            (0..dims.len())
                .cartesian_product(&all_packing_sizes)
                .filter_map(|(packing_dim, &packing_size)| {
                    debug_assert_ne!(packing_size.get(), 1);
                    if packing_dim == usize::from(dims.last().unwrap().0)
                        || shape[packing_dim].get() % packing_size.get() != 0
                    {
                        return None;
                    }
                    Some(Layout::new(
                        dims.iter()
                            .cloned()
                            .chain(iter::once((
                                packing_dim.try_into().unwrap(),
                                PhysDim::Packed(packing_size),
                            )))
                            .collect(),
                    ))
                })
        }));

        let all_oddeven_sizes =
            oddeven_sizes_all(dtype, &all_target_vector_bytes).collect::<Vec<_>>();
        result.extend(unpacked_layouts.iter().flat_map(|original_layout| {
            let Layout(dims) = original_layout;
            (0..dims.len())
                .cartesian_product(&all_oddeven_sizes)
                .filter_map(|(packing_dim, &packing_size)| {
                    debug_assert_ne!(packing_size.get(), 1);
                    if packing_dim == usize::from(dims.last().unwrap().0)
                        || shape[packing_dim].get() % packing_size.get() != 0
                    {
                        return None;
                    }
                    Some(Layout::new(
                        dims.iter()
                            .cloned()
                            .chain(iter::once((
                                packing_dim.try_into().unwrap(),
                                PhysDim::OddEven(packing_size),
                            )))
                            .collect(),
                    ))
                })
        }));
        result
    }

    fn move_destination_layouts(shape: &[DimSize], dtype: Dtype) -> Vec<Layout> {
        let all_target_vector_bytes = Self::levels()
            .into_iter()
            .flat_map(|lvl| lvl.vector_bytes().iter().copied())
            .collect::<Vec<_>>();

        // The following could be faster. It keeps two copies of the non-packed layouts
        // (`base` and the first few values in `result`) and it doesn't compute the size
        // of the `result` ahead-of-time, potentially causing some Vec resizes.
        let only_ones = shape.iter().all(|&d| d.get() == 1);
        let base = match (shape.len(), only_ones) {
            (2, false) => vec![row_major(2), col_major(2)],
            (4, false) => vec![row_major(4), nhwc()],
            (r, _) => vec![row_major(r.try_into().unwrap())],
        };
        let mut result = base.clone();
        result.extend(base.iter().flat_map(|original_layout| {
            packed_layouts_for_standard_layout(
                original_layout,
                shape,
                dtype,
                &all_target_vector_bytes,
            )
        }));
        result.extend(base.iter().flat_map(|original_layout| {
            oddeven_layouts_for_standard_layout(
                original_layout,
                shape,
                dtype,
                &all_target_vector_bytes,
            )
        }));

        debug_assert!(
            result.iter().all(|r| r.applies_to_shape(shape)),
            "Some layouts don't apply to shape {:?}: {:?}",
            shape,
            result
                .iter()
                .filter(|r| !r.applies_to_shape(shape))
                .collect::<Vec<_>>(),
        );
        result
    }

    fn actions(spec: &LogicalSpec<Self>) -> Self::ActionsIter<'_> {
        let iter = move_actions(spec);
        let iter = iter.chain(tile_out_actions(spec));

        // OnePrefix is an unfortunate special case. The only viable action is applying
        // its no-op kernel.
        if matches!(
            spec,
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::OnePrefix,
                    ..
                },
                ..
            )
        ) {
            return Box::new(iter::once(Action::Select(Select(
                (CpuKernel::OnePrefixNoOp).into(),
                false,
            ))));
        }

        let possible_kernels: &[CpuKernel] = match spec {
            LogicalSpec::Primitive(PrimitiveBasics { typ, .. }, ..) => match typ {
                PrimitiveSpecType::OnePrefix => unreachable!(),
                PrimitiveSpecType::Matmul { accum } => {
                    if *accum {
                        const MATMUL_ACCUM_KERNELS: [CpuKernel; 8] = [
                            CpuKernel::MultAdd,
                            CpuKernel::BroadcastVecMultAdd,
                            CpuKernel::BroadcastVecMultAddBf16F32,
                            CpuKernel::TwoVecBroadcastVecMultAddU8S8S16,
                            CpuKernel::DotProductLoop,
                            CpuKernel::DotProductLoopBf16Bf16F32,
                            CpuKernel::DotProductLoopF32Bf16F32,
                            CpuKernel::DotProductLoopF32InterleavedBf16F32,
                        ];
                        &MATMUL_ACCUM_KERNELS
                    } else {
                        &[]
                    }
                }
                PrimitiveSpecType::Conv { .. } => &[],
                PrimitiveSpecType::Broadcast { .. } => &[CpuKernel::VecScalarAssign],
                PrimitiveSpecType::Softmax { .. } => &[],
                PrimitiveSpecType::SoftmaxComplete { .. } => {
                    const SOFTMAX_COMPLETE_KERNELS: [CpuKernel; 2] = [
                        CpuKernel::ValueSoftmaxComplete,
                        CpuKernel::VectorSoftmaxComplete,
                    ];
                    &SOFTMAX_COMPLETE_KERNELS
                }
                PrimitiveSpecType::SoftmaxDenominatorAndMax { .. } => &[],
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. } => &[],
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { .. } => {
                    &[CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32]
                }
                PrimitiveSpecType::SoftmaxDenominator { accum, .. } => {
                    if *accum {
                        const SOFTMAX_DENOMINATOR_KERNELS: [CpuKernel; 2] = [
                            CpuKernel::ValueSoftmaxDenominator,
                            CpuKernel::VectorSoftmaxDenominator,
                        ];
                        &SOFTMAX_DENOMINATOR_KERNELS
                    } else {
                        &[]
                    }
                }
                PrimitiveSpecType::DivideVec => &[CpuKernel::DivideVec],
                PrimitiveSpecType::DivideVecScalar { .. } => &[],
                PrimitiveSpecType::Max { accum, .. } => {
                    if *accum {
                        const MAX_KERNELS: [CpuKernel; 2] =
                            [CpuKernel::ValueMax, CpuKernel::VectorMax];
                        &MAX_KERNELS
                    } else {
                        &[]
                    }
                }
                PrimitiveSpecType::Move => {
                    const MOVE_KERNELS: [CpuKernel; 8] = [
                        CpuKernel::ValueAssign,
                        CpuKernel::VectorAssign,
                        CpuKernel::PhysicalTransposeByte128,
                        CpuKernel::PhysicalTransposeByte256,
                        CpuKernel::VectorInterleaveBf16F32,
                        CpuKernel::VectorDeinterleaveF32Bf16,
                        CpuKernel::CastBf16F32,
                        CpuKernel::VectorCastBf16F32,
                    ];
                    &MOVE_KERNELS
                }
                PrimitiveSpecType::Fill {
                    value: FillValue::Zero,
                } => {
                    const ZERO_KERNELS: [CpuKernel; 2] =
                        [CpuKernel::MemsetZero, CpuKernel::VectorZero];
                    &ZERO_KERNELS
                }
                PrimitiveSpecType::Fill {
                    value: FillValue::NegInf,
                } => {
                    const NEGINF_KERNELS: [CpuKernel; 2] =
                        [CpuKernel::VectorNegInf, CpuKernel::ValueNegInf];
                    &NEGINF_KERNELS
                }
            },
            LogicalSpec::Compose { .. } => &[],
        };
        let iter = iter.chain(possible_kernels.iter().filter_map(move |mk| {
            if mk.applies_to_logical_spec(spec) {
                Some(Action::Select(Select((*mk).into(), false)))
            } else {
                None
            }
        }));

        match spec {
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ,
                    spec_shape: _,
                    dtypes: _,
                },
                _primitive_aux,
                _serial_only,
            ) => match typ {
                PrimitiveSpecType::Matmul { accum } if !*accum => {
                    Box::new(iter.chain(once(ToAccum.into())))
                }
                PrimitiveSpecType::Matmul { accum } if *accum => {
                    Box::new(iter.chain(split_actions(spec)))
                }
                PrimitiveSpecType::Conv { accum } => {
                    if *accum {
                        if spec.can_spatial_split() {
                            Box::new(iter.chain(once(SpatialSplit.into())))
                        } else {
                            Box::new(iter)
                        }
                    } else {
                        Box::new(iter.chain(once(ToAccum.into())))
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
                Box::new(iter.chain(bufferize_actions(spec)))
            }
        }
    }

    fn target_id() -> TargetId {
        <Self as CpuTarget>::target_id()
    }

    fn vec_types() -> &'static [VecType; 16] {
        <Self as CpuTarget>::vec_types()
    }
}

impl CpuKernel {
    pub fn argument_count(&self) -> u8 {
        match self {
            CpuKernel::ValueSoftmaxComplete
            | CpuKernel::VectorSoftmaxComplete
            | CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32 => 4,
            CpuKernel::MultAdd
            | CpuKernel::ValueSoftmaxDenominator
            | CpuKernel::VectorSoftmaxDenominator
            | CpuKernel::BroadcastVecMultAdd
            | CpuKernel::BroadcastVecMultAddBf16F32
            | CpuKernel::TwoVecBroadcastVecMultAddU8S8S16
            | CpuKernel::DotProductLoop
            | CpuKernel::DotProductLoopF32Bf16F32
            | CpuKernel::DotProductLoopF32InterleavedBf16F32
            | CpuKernel::DotProductLoopBf16Bf16F32
            | CpuKernel::DivideVec => 3,
            CpuKernel::OnePrefixNoOp
            | CpuKernel::PhysicalTransposeByte128
            | CpuKernel::PhysicalTransposeByte256
            | CpuKernel::VectorInterleaveBf16F32
            | CpuKernel::VectorDeinterleaveF32Bf16
            | CpuKernel::ValueMax
            | CpuKernel::VectorMax
            | CpuKernel::ValueAssign
            | CpuKernel::VectorAssign
            | CpuKernel::CastBf16F32
            | CpuKernel::VectorCastBf16F32
            | CpuKernel::VecScalarAssign => 2,
            CpuKernel::MemsetZero
            | CpuKernel::VectorZero
            | CpuKernel::ValueNegInf
            | CpuKernel::VectorNegInf => 1,
        }
    }

    // TODO: Make into `applies_to_spec`
    // TODO: Rename to parameters
    pub fn applies_to_logical_spec<Tgt: CpuTarget>(&self, logical_spec: &LogicalSpec<Tgt>) -> bool {
        // None of these kernels apply to Compose, so match Primitive only.
        let LogicalSpec::Primitive(
            PrimitiveBasics {
                typ,
                spec_shape: _,
                dtypes: _,
            },
            _,
            _,
        ) = logical_spec
        else {
            return false;
        };
        let operands = logical_spec.parameters();

        match self {
            CpuKernel::OnePrefixNoOp => matches!(typ, PrimitiveSpecType::OnePrefix),
            CpuKernel::MultAdd => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && operands.iter().all(|o| {
                        o.level() == CpuMemoryLevel::RF && o.shape().iter().all(|&d| d == nz!(1u32))
                    })
                    && operands.iter().map(|o| o.dtype()).all_equal()
            }
            CpuKernel::BroadcastVecMultAdd => {
                if !matches!(typ, PrimitiveSpecType::Matmul { accum: true }) {
                    return false;
                }

                // Only integers and 32-bit floats, which Clang should be able to handle pretty well with
                // its vector type extension.
                let first_dtype = operands[0].dtype();
                if !matches!(
                    first_dtype,
                    Dtype::Sint8
                        | Dtype::Uint8
                        | Dtype::Sint16
                        | Dtype::Uint16
                        | Dtype::Sint32
                        | Dtype::Uint32
                        | Dtype::Float32
                ) {
                    return false;
                }
                operands.iter().skip(1).all(|o| o.dtype() == first_dtype)
                    && shared_broadcastvecmult_applies_to_operands(&operands)
            }
            CpuKernel::BroadcastVecMultAddBf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && operands[0].dtype() == Dtype::Bfloat16
                    && operands[1].dtype() == Dtype::Bfloat16
                    && operands[2].dtype() == Dtype::Float32
                    && operands[1].vector_size() == operands[2].vector_size()
                    && shared_broadcastvecmult_applies_to_operands(&operands)
            }
            CpuKernel::TwoVecBroadcastVecMultAddU8S8S16 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && matches!(
                        &operands[..],
                        [
                            lhs @ TensorSpec {
                                shape: lhs_shape,
                                dtype: Dtype::Uint8,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::L1,
                                    ..
                                },
                            },
                            rhs @ TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Sint8,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::VRF,
                                    vector_size: Some(rhs_vector_size),
                                    ..
                                },
                            },
                            out @ TensorSpec {
                                shape: out_shape,
                                dtype: Dtype::Sint16,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::VRF,
                                    ..
                                },
                            }
                        ] if lhs_shape[..] == [nz!(1u32), nz!(1u32), nz!(2u32)]
                          && rhs_shape[..2] == [nz!(1u32), nz!(2u32)]
                          && rhs_shape[2].get() * 2 == rhs_vector_size.get()
                          && out_shape[..2] == [nz!(1u32), nz!(1u32)]
                          && out_shape[2].get() * 2 == rhs_vector_size.get()
                          && rhs.layout() == &batched_col_major(3) && out.layout().is_row_major()
                          && lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous()
                    )
            }
            CpuKernel::DotProductLoop => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && matches!(
                        &operands[..],
                        [
                            lhs @ TensorSpec {
                                shape: lhs_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::L1,
                                    ..
                                },
                            },
                            rhs @ TensorSpec {
                                shape: _,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::L1, ..
                                },
                            },
                            TensorSpec {
                                shape: out_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::RF,
                                    ..
                                },
                            }
                        ] if lhs_shape[0] == nz!(1u32)
                          && lhs_shape[1].get() % (DOT_PRODUCT_STRIP_SIZE.get() * DOT_PRODUCT_ACCUM_COUNT) == 0
                          && out_shape[..] == [nz!(1u32), nz!(1u32), nz!(1u32)]
                          && lhs.layout().is_row_major()
                          && rhs.layout() == &col_major(2)
                          && lhs.is_contiguous() && rhs.is_contiguous()
                    )
            }
            CpuKernel::DotProductLoopF32Bf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && dotproductloop_applies(&operands, Dtype::Float32, &[row_major(2)])
            }
            CpuKernel::DotProductLoopF32InterleavedBf16F32 => {
                // TODO: Simplify with a closure instead of constructing all layouts AOT.
                if !matches!(typ, PrimitiveSpecType::Matmul { accum: true }) {
                    return false;
                }
                let layout0 = Layout::new(vec![
                    (0, PhysDim::Dynamic),
                    (1, PhysDim::Dynamic),
                    (2, PhysDim::Dynamic),
                    (2, PhysDim::OddEven(nz!(16u32))),
                ]);
                let layout1 = Layout::new(vec![
                    (0, PhysDim::Dynamic),
                    (2, PhysDim::Dynamic),
                    (1, PhysDim::Dynamic),
                    (1, PhysDim::OddEven(nz!(16u32))),
                ]);
                dotproductloop_applies(&operands, Dtype::Float32, &[layout0, layout1])
            }
            CpuKernel::DotProductLoopBf16Bf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && dotproductloop_applies(&operands, Dtype::Bfloat16, &[row_major(2)])
            }
            CpuKernel::PhysicalTransposeByte128 => {
                physicaltransposebyte_applies_to_operands(typ, &operands, 16)
            }
            CpuKernel::PhysicalTransposeByte256 => {
                physicaltransposebyte_applies_to_operands(typ, &operands, 32)
            }
            CpuKernel::VectorInterleaveBf16F32 => {
                if !matches!(typ, PrimitiveSpecType::Move) {
                    return false;
                }

                let leaved = PhysDim::OddEven(nz!(16u32));
                matches!(
                    &operands[..],
                    [
                        src @ TensorSpec {
                            shape: src_shape,
                            dtype: Dtype::Bfloat16,
                            aux: TensorSpecAux {
                                level: CpuMemoryLevel::VRF,
                                layout: src_layout,
                                vector_size: src_vs,
                                aligned: _,
                                contig: _
                            },
                        },
                        dest @ TensorSpec {
                            shape: dest_shape,
                            dtype: Dtype::Float32,
                            aux: TensorSpecAux {
                                level: CpuMemoryLevel::VRF,
                                layout: dest_layout,
                                vector_size: dest_vs,
                                aligned: _,
                                contig: _
                            },
                        },
                    ] if src.is_contiguous() && dest.is_contiguous()
                      && src_shape.iter().all(|d| d.get() == 1 || d.get() == 16)
                      && src_shape.iter().filter(|d| d.get() == 16).count() == 1
                      && src_shape == dest_shape
                      && src_vs == &Some(nz!(16u32))
                      && dest_vs == &Some(nz!(8u32))
                      && src_layout.is_row_major()
                      && dest_layout.0.iter().all(|(_, pd)| pd == &PhysDim::Dynamic || pd == &leaved)
                      && dest_layout.0.iter().filter(|(_, pd)| pd == &leaved).count() == 1
                )
            }
            CpuKernel::VectorDeinterleaveF32Bf16 => {
                if !matches!(typ, PrimitiveSpecType::Move) {
                    return false;
                }

                // TODO: Fill in
                false
            }
            CpuKernel::ValueSoftmaxComplete => {
                debug_assert_eq!(operands.len(), 4);
                if !matches!(typ, PrimitiveSpecType::SoftmaxComplete { .. }) {
                    return false;
                }
                if operands
                    .iter()
                    .flat_map(|o| o.shape())
                    .any(|d| d.get() != 1)
                {
                    return false;
                }
                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                    if o.level() != CpuMemoryLevel::RF {
                        return false;
                    }
                }
                true
            }
            CpuKernel::VectorSoftmaxComplete => {
                debug_assert_eq!(operands.len(), 4);
                if !matches!(typ, PrimitiveSpecType::SoftmaxComplete { .. }) {
                    return false;
                }

                if operands[0].level() != CpuMemoryLevel::L1 {
                    return false;
                }
                if operands[1].level() != CpuMemoryLevel::RF {
                    return false;
                }
                if operands[2].level() != CpuMemoryLevel::RF {
                    return false;
                }
                if operands[3].level() != CpuMemoryLevel::L1 {
                    return false;
                }

                if operands[0].shape() != operands[3].shape() {
                    return false;
                }
                for operand in &operands[1..=2] {
                    if operand.shape().iter().any(|d| d.get() != 1) {
                        return false;
                    }
                }

                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                }
                true
            }
            CpuKernel::ValueSoftmaxDenominator => {
                debug_assert_eq!(operands.len(), 3);
                if !matches!(
                    typ,
                    PrimitiveSpecType::SoftmaxDenominator { accum: true, .. }
                ) {
                    return false;
                }
                if operands
                    .iter()
                    .flat_map(|o| o.shape())
                    .any(|d| d.get() != 1)
                {
                    return false;
                }
                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                    if o.level() != CpuMemoryLevel::RF {
                        return false;
                    }
                }
                true
            }
            CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32 => {
                debug_assert_eq!(operands.len(), 4);
                let PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                    accum: true,
                    scan_dim,
                } = typ
                else {
                    return false;
                };

                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                }

                for i in [0, 3] {
                    let operand = &operands[i];
                    let shape = operand.shape();
                    let dim_us = usize::from(*scan_dim);
                    if shape[..dim_us].iter().any(|d| d.get() != 1) {
                        return false;
                    }
                    if shape[dim_us].get() % 8 != 0 {
                        return false;
                    }
                    if shape[(dim_us + 1)..].iter().any(|d| d.get() != 1) {
                        return false;
                    }

                    if operand.level() != CpuMemoryLevel::VRF {
                        return false;
                    }
                    if operand.vector_size().unwrap().get() != 8 {
                        return false;
                    }
                    if !operand.is_contiguous() {
                        return false;
                    }
                }
                for i in [1, 2] {
                    if operands[i].shape().iter().any(|d| d.get() != 1) {
                        return false;
                    }
                    if operands[i].level() != CpuMemoryLevel::RF {
                        return false;
                    }
                }

                true
            }
            CpuKernel::VectorSoftmaxDenominator => {
                debug_assert_eq!(operands.len(), 3);
                let PrimitiveSpecType::SoftmaxDenominator {
                    accum: true,
                    scan_dim,
                } = *typ
                else {
                    return false;
                };

                for o in &operands {
                    for (dim, size) in o.shape().iter().enumerate() {
                        if dim != usize::from(scan_dim) && size.get() != 1 {
                            return false;
                        }
                    }
                }
                if operands[0].level() != CpuMemoryLevel::VRF
                    || operands[1].level() != CpuMemoryLevel::RF
                    || operands[2].level() != CpuMemoryLevel::RF
                {
                    return false;
                }
                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                }
                if operands[0].volume().get() % operands[0].vector_size().unwrap().get() != 0 {
                    return false;
                }
                true
            }
            CpuKernel::ValueMax => {
                debug_assert_eq!(operands.len(), 2);
                if !matches!(typ, PrimitiveSpecType::Max { accum: true, .. }) {
                    return false;
                }
                if operands
                    .iter()
                    .flat_map(|o| o.shape())
                    .any(|d| d.get() != 1)
                {
                    return false;
                }
                for o in &operands {
                    if o.dtype() != Dtype::Float32 {
                        return false;
                    }
                    if o.level() != CpuMemoryLevel::RF {
                        return false;
                    }
                }
                true
            }
            CpuKernel::VectorMax => {
                debug_assert_eq!(operands.len(), 2);
                if !matches!(typ, PrimitiveSpecType::Max { accum: true, .. }) {
                    return false;
                }
                if operands[0].level() != CpuMemoryLevel::VRF {
                    return false;
                }
                if operands[1].level() != CpuMemoryLevel::RF {
                    return false;
                }
                if operands[0].dtype() != Dtype::Float32 {
                    return false;
                }
                if operands[1].dtype() != Dtype::Float32 {
                    return false;
                }
                if operands[0].volume().get() % operands[0].vector_size().unwrap().get() != 0 {
                    return false;
                }
                // 256-bit VRF
                if operands[0].vector_size().unwrap().get() != 8 {
                    return false;
                }
                if operands[0].shape().iter().filter(|d| d.get() > 1).count() >= 2 {
                    return false;
                }
                if operands[1].shape().iter().any(|d| d.get() != 1) {
                    return false;
                }
                if operands.iter().any(|o| !o.is_contiguous()) {
                    return false;
                }
                true
            }
            CpuKernel::VecScalarAssign => {
                let PrimitiveSpecType::Broadcast { dim } = *typ else {
                    return false;
                };

                if operands[0].dtype() != operands[1].dtype() {
                    return false;
                }

                if operands[0].level() != CpuMemoryLevel::RF {
                    return false;
                }
                if operands[1].level() != CpuMemoryLevel::VRF {
                    return false;
                }

                if !operands[0].is_contiguous() || !operands[1].is_contiguous() {
                    return false;
                }

                if !operands[0].shape().iter().all(|d| d.get() == 1) {
                    return false;
                }

                let dest_shape = operands[1].shape();
                if dest_shape[..usize::from(dim)].iter().any(|d| d.get() != 1) {
                    return false;
                }
                if dest_shape[(usize::from(dim) + 1)..]
                    .iter()
                    .any(|d| d.get() != 1)
                {
                    return false;
                }

                let Some(vs) = operands[1].vector_size() else {
                    return false;
                };
                if dest_shape[usize::from(dim)].get() % vs.get() != 0 {
                    return false;
                }

                true
            }
            CpuKernel::DivideVec => {
                let PrimitiveSpecType::DivideVec = *typ else {
                    return false;
                };
                debug_assert_eq!(operands.len(), 3);

                if operands[0].dtype() != operands[1].dtype()
                    || operands[0].dtype() != operands[1].dtype()
                {
                    return false;
                }

                if operands[0].level() != CpuMemoryLevel::VRF {
                    return false;
                }
                if operands[1].level() != CpuMemoryLevel::VRF {
                    return false;
                }
                if operands[2].level() != CpuMemoryLevel::VRF {
                    return false;
                }

                if operands[0].vector_size() != operands[1].vector_size() {
                    return false;
                }
                if operands[0].vector_size() != operands[2].vector_size() {
                    return false;
                }

                if !operands[0].is_contiguous()
                    || !operands[1].is_contiguous()
                    || !operands[2].is_contiguous()
                {
                    return false;
                }

                if operands[0].shape() != operands[1].shape() {
                    return false;
                }
                if operands[0].shape() != operands[2].shape() {
                    return false;
                }

                true
            }
            CpuKernel::ValueAssign => {
                debug_assert_eq!(operands.len(), 2);

                if !matches!(typ, PrimitiveSpecType::Move) {
                    return false;
                }

                if operands
                    .iter()
                    .flat_map(|o| o.shape())
                    .any(|d| d.get() != 1)
                {
                    return false;
                }

                for o in &operands[1..] {
                    if (o.dtype(), o.layout()) != (operands[0].dtype(), operands[0].layout()) {
                        return false;
                    }
                }

                operands.iter().any(|o| o.level() == CpuMemoryLevel::RF)
                    && operands
                        .iter()
                        .all(|o| o.level() == CpuMemoryLevel::RF || o.level() == CpuMemoryLevel::L1)
            }
            CpuKernel::VectorAssign => {
                if !matches!(typ, PrimitiveSpecType::Move) {
                    return false;
                }
                if operands.iter().any(|o| !o.is_contiguous()) {
                    return false;
                }
                if operands[0].dtype() != operands[1].dtype() {
                    return false;
                }
                if operands[0].shape() != operands[1].shape() {
                    return false;
                }
                if operands[0].layout() != operands[1].layout() {
                    return false;
                }

                let mut has_vrf = false;
                for o in operands {
                    if o.level().vector_rf() {
                        has_vrf = true;
                        match o.vector_size() {
                            Some(vector_size) => {
                                if o.volume().get() % vector_size.get() != 0 {
                                    return false;
                                }
                            }
                            None => {
                                panic!("No vector_size on operand in level {:?}", o.level());
                            }
                        }
                    }
                }
                has_vrf
            }
            CpuKernel::CastBf16F32 => {
                matches!(typ, PrimitiveSpecType::Move)
                    && matches!(
                        &operands[..],
                        [
                            TensorSpec {
                                shape: lhs_shape,
                                dtype: Dtype::Bfloat16,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::RF,
                                    vector_size: None,
                                    ..
                                },
                            },
                            TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::RF,
                                    vector_size: None,
                                    ..
                                },
                            }
                        ] if lhs_shape.iter().all(|d| d.get() == 1)
                          && rhs_shape.iter().all(|d| d.get() == 1)
                    )
            }
            CpuKernel::VectorCastBf16F32 => {
                matches!(typ, PrimitiveSpecType::Move)
                    && matches!(
                        &operands[..],
                        [
                            TensorSpec {
                                shape: lhs_shape,
                                dtype: Dtype::Bfloat16,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::VRF,
                                    layout: lhs_layout,
                                    vector_size: Some(lhs_vector_size),
                                    ..
                                },
                            },
                            TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: CpuMemoryLevel::VRF,
                                    layout: rhs_layout,
                                    vector_size: Some(rhs_vector_size),
                                    ..
                                },
                            }
                        ] if lhs_shape == rhs_shape
                          && lhs_shape.iter().all(|&d| d.get() == 1 || d.get() == 16)
                          && lhs_shape.iter().filter(|&d| d.get() == 16).count() == 1
                          && lhs_vector_size.get() == 16
                          && rhs_vector_size.get() == 8
                          && lhs_layout == rhs_layout
                    )
            }
            CpuKernel::MemsetZero => {
                matches!(
                    typ,
                    PrimitiveSpecType::Fill {
                        value: FillValue::Zero
                    }
                ) && operands[0].level() == CpuMemoryLevel::RF
                    && operands[0].is_contiguous()
            }
            CpuKernel::VectorZero | CpuKernel::VectorNegInf => {
                let expected_fill_value = match self {
                    CpuKernel::VectorZero => FillValue::Zero,
                    CpuKernel::VectorNegInf => FillValue::NegInf,
                    _ => unreachable!(),
                };
                match typ {
                    PrimitiveSpecType::Fill { value } if value == &expected_fill_value => {}
                    _ => return false,
                };
                if !operands[0].is_contiguous() {
                    return false;
                }
                if operands[0].level() != CpuMemoryLevel::VRF {
                    return false;
                }
                match operands[0].vector_size() {
                    None => false,
                    Some(vector_size) if vector_size != operands[0].volume() => false,
                    _ => true,
                }
            }
            CpuKernel::ValueNegInf => {
                debug_assert_eq!(operands.len(), 1);
                matches!(
                    typ,
                    PrimitiveSpecType::Fill {
                        value: FillValue::NegInf
                    }
                ) && operands[0].volume().get() == 1
                    && operands[0].level() == CpuMemoryLevel::RF
                    && operands[0].is_contiguous()
                    && operands[0].dtype() == Dtype::Float32
            }
        }
    }

    pub fn memory_allocated<P: View>(&self, parameters: &[P]) -> MemoryAllocation {
        match self {
            // TODO: Model memory correctly for BroadcastVecMultAddBf16F32
            CpuKernel::BroadcastVecMultAdd
            | CpuKernel::TwoVecBroadcastVecMultAddU8S8S16
            | CpuKernel::BroadcastVecMultAddBf16F32 => {
                let vec_tensor_spec = &parameters[1].spec();
                let vb = u64::from(vec_tensor_spec.vector_size().unwrap().get())
                    * u64::from(vec_tensor_spec.dtype().size());
                MemoryAllocation::Simple(CPU_LEVELS.map(
                    |level| {
                        if level.vector_rf() {
                            vb * 2
                        } else {
                            0
                        }
                    },
                ))
            }
            CpuKernel::DotProductLoop
            | CpuKernel::DotProductLoopBf16Bf16F32
            | CpuKernel::DotProductLoopF32InterleavedBf16F32
            | CpuKernel::DotProductLoopF32Bf16F32 => {
                // TODO: Count any additional peak memory from sum8.
                MemoryAllocation::Simple(CPU_LEVELS.map(|level| {
                    let mut used = 0;
                    if level.vector_rf() {
                        used = 128;
                        // TODO: Add intermediate consumption
                        match self {
                            CpuKernel::DotProductLoopBf16Bf16F32 => {}
                            CpuKernel::DotProductLoopF32InterleavedBf16F32 => {}
                            CpuKernel::DotProductLoopF32Bf16F32 => {}
                            _ => {}
                        }
                    }
                    used
                }))
            }
            CpuKernel::PhysicalTransposeByte256 => MemoryAllocation::Simple(CPU_LEVELS.map(
                |level| {
                    if level.vector_rf() {
                        64
                    } else {
                        0
                    }
                },
            )),
            CpuKernel::VectorInterleaveBf16F32 | CpuKernel::VectorDeinterleaveF32Bf16 => {
                MemoryAllocation::Simple(CPU_LEVELS.map(|_| {
                    // TODO: Count any intermediate vectors.
                    0
                }))
            }
            CpuKernel::VectorMax => {
                MemoryAllocation::Simple(
                    CPU_LEVELS.map(|level| if level.vector_rf() { 128 } else { 0 }),
                )
            }
            CpuKernel::VectorSoftmaxDenominator => {
                // TODO: Check if VectorSoftmaxDenominator allocates more than 2 vectors.
                MemoryAllocation::Simple(CPU_LEVELS.map(|level| {
                    if level.vector_rf() {
                        2 * 128
                    } else {
                        0
                    }
                }))
            }
            _ => MemoryAllocation::none(),
        }
    }

    pub fn main_cost<P: View>(&self, parameters: &[P]) -> MainCost {
        match self {
            CpuKernel::OnePrefixNoOp => 0,
            CpuKernel::BroadcastVecMultAdd
            | CpuKernel::TwoVecBroadcastVecMultAddU8S8S16
            | CpuKernel::BroadcastVecMultAddBf16F32 => {
                // TODO: Adjust the BroadcastVecMultAdd cost.
                // TODO: Adjust the TwoVecBroadcastVecMultAddU8S8S16 cost.
                // TODO: Model cost for BroadcastVecMultAddBf16F32 correctly.

                let vec_tensor_spec = &parameters[1].spec();
                let vector_size = vec_tensor_spec.vector_size().unwrap().get();
                let volume = vec_tensor_spec.volume().get();
                debug_assert_eq!(volume % vector_size, 0);
                let vector_count = volume / vector_size;
                let mut cost = INST_COST * ((vector_count * 2) + 1);

                // TwoVecBroadcastVecMultAdd takes an input from L1.
                if matches!(self, CpuKernel::TwoVecBroadcastVecMultAddU8S8S16) {
                    // TODO: Instead, call `move_cost`. Requires specializing kernel to X86/ARM.
                    let mut l1_hit_cost = CpuMemoryLevel::L1.cache_hit_cost();
                    if !parameters[0].spec().is_contiguous() {
                        l1_hit_cost *= 2;
                    }
                    cost += l1_hit_cost;
                }

                cost
            }
            CpuKernel::DotProductLoop => {
                // TODO: Measure throughput! This is a rough estimate.
                let value_cnt = parameters[0].spec().shape()[1].get();
                let d = DOT_PRODUCT_STRIP_SIZE.get() * DOT_PRODUCT_ACCUM_COUNT;
                8 * INST_COST * value_cnt / d
            }
            CpuKernel::DotProductLoopBf16Bf16F32 => {
                // TODO: Measure throughput! This is a rough estimate.
                let value_cnt = parameters[0].spec().shape()[1].get();
                let d = DOT_PRODUCT_BF16_STRIP_SIZE.get() * DOT_PRODUCT_BF16_ACCUM_COUNT;
                (12 * INST_COST * value_cnt) / d
            }
            CpuKernel::DotProductLoopF32Bf16F32
            | CpuKernel::DotProductLoopF32InterleavedBf16F32 => {
                // RThroughput = 8 or 16
                let value_cnt = parameters[0].spec().shape()[1].get();
                let d = DOT_PRODUCT_BF16_STRIP_SIZE.get() * DOT_PRODUCT_BF16_ACCUM_COUNT;
                let mut cost = 16 * INST_COST * value_cnt;
                if self == &CpuKernel::DotProductLoopF32Bf16F32 {
                    cost *= 2;
                }
                cost / d
            }
            CpuKernel::VectorInterleaveBf16F32
            | CpuKernel::VectorDeinterleaveF32Bf16
            | CpuKernel::ValueMax
            | CpuKernel::ValueNegInf => {
                // TODO: Measure throughput!
                INST_COST
            }
            CpuKernel::VectorMax | CpuKernel::VecScalarAssign | CpuKernel::DivideVec => {
                // TODO: Measure throughput!
                let vidx = match self {
                    CpuKernel::VectorMax | CpuKernel::DivideVec => 0,
                    CpuKernel::VecScalarAssign => 1,
                    _ => unreachable!(),
                };
                let value_cnt = parameters[vidx].spec().volume().get();
                let vector_cnt = value_cnt / parameters[vidx].spec().vector_size().unwrap().get();
                INST_COST * (vector_cnt + 1)
            }
            CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32
            | CpuKernel::ValueSoftmaxDenominator
            | CpuKernel::VectorSoftmaxDenominator => {
                // TODO: Measure throughput!
                INST_COST * 3
            }
            CpuKernel::ValueSoftmaxComplete | CpuKernel::VectorSoftmaxComplete => {
                // TODO: Measure throughput!
                INST_COST * 4
            }
            CpuKernel::MultAdd => {
                match parameters[0].spec().dtype() {
                    Dtype::Bfloat16 => 6 * INST_COST,
                    Dtype::Float32 => INST_COST, // RThroughput = .5
                    Dtype::Uint8
                    | Dtype::Sint8
                    | Dtype::Uint16
                    | Dtype::Sint16
                    | Dtype::Uint32
                    | Dtype::Sint32 => 2 * INST_COST, // RThroughput = 1
                }
            }
            CpuKernel::CastBf16F32 => 4 * INST_COST, // RThroughput = 2
            CpuKernel::VectorCastBf16F32 => 6 * INST_COST, // RThroughput = 3
            CpuKernel::PhysicalTransposeByte128 => ASSIGN_INST_COST * 2,
            CpuKernel::PhysicalTransposeByte256 => ASSIGN_INST_COST * 4,
            CpuKernel::VectorAssign => {
                let vector_value_count = parameters[0]
                    .spec()
                    .vector_size()
                    .unwrap_or_else(|| parameters[1].spec().vector_size().unwrap());
                let vec_count = parameters[0].spec().volume().get() / vector_value_count.get();
                ASSIGN_INST_COST * vec_count
            }
            CpuKernel::ValueAssign | CpuKernel::MemsetZero => ASSIGN_INST_COST,
            CpuKernel::VectorZero | CpuKernel::VectorNegInf => {
                debug_assert_eq!(
                    parameters[0].spec().volume().get(),
                    parameters[0].spec().vector_size().unwrap().get(),
                );
                INST_COST
            }
        }
    }

    pub fn name(&self) -> &'static str {
        self.into()
    }
}

fn dotproductloop_applies<Tgt: CpuTarget>(
    operands: &[TensorSpec<Tgt>],
    lhs_dtype: Dtype,
    allowed_lhs_layouts: &[Layout],
) -> bool {
    matches!(
        operands,
        [
            lhs @ TensorSpec {
                shape: lhs_shape,
                dtype: ldt,
                aux: TensorSpecAux {
                    level: CpuMemoryLevel::L1,
                    ..
                },
            },
            rhs @ TensorSpec {
                shape: _,
                dtype: Dtype::Bfloat16,
                aux: TensorSpecAux {
                    level: CpuMemoryLevel::L1, ..
                },
            },
            TensorSpec {
                shape: out_shape,
                dtype: Dtype::Float32,
                aux: TensorSpecAux {
                    level: CpuMemoryLevel::RF,
                    ..
                },
            }
        ] if *ldt == lhs_dtype
          && lhs_shape[2].get() % (DOT_PRODUCT_BF16_STRIP_SIZE.get() * DOT_PRODUCT_BF16_ACCUM_COUNT) == 0
          && out_shape[0] == nz!(1u32)  // means {lhs, rhs}_shape also equal 1
          && out_shape[1] == nz!(1u32)
          && out_shape[2] == nz!(1u32)
          && allowed_lhs_layouts.contains(lhs.layout())
          && rhs.layout() == &Layout::new([0, 2, 1].map(|d| (d, PhysDim::Dynamic)).into())
          && lhs.is_contiguous() && rhs.is_contiguous()
    )
}

impl MemoryLevel for CpuMemoryLevel {
    fn is_addressed(&self) -> bool {
        match &self {
            CpuMemoryLevel::RF => true,
            CpuMemoryLevel::VRF => true,
            CpuMemoryLevel::L1 => false,
            CpuMemoryLevel::GL => true,
        }
    }

    fn can_parallel_tile(&self) -> bool {
        match self {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => false,
            CpuMemoryLevel::GL | CpuMemoryLevel::L1 => true,
        }
    }

    fn cache_hit_cost(&self) -> MainCost {
        match &self {
            CpuMemoryLevel::RF => 0,
            CpuMemoryLevel::VRF => 0,
            CpuMemoryLevel::L1 => 10,
            CpuMemoryLevel::GL => 100,
        }
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match &self {
            CpuMemoryLevel::VRF => &[16, 32],
            _ => &[],
        }
    }
}

impl PartialOrd for CpuMemoryLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (CpuMemoryLevel::RF, CpuMemoryLevel::VRF) => None,
            (CpuMemoryLevel::VRF, CpuMemoryLevel::RF) => None,
            (CpuMemoryLevel::RF, _) => Some(Ordering::Less),
            (CpuMemoryLevel::VRF, _) => Some(Ordering::Less),
            (_, CpuMemoryLevel::RF) => Some(Ordering::Greater),
            (_, CpuMemoryLevel::VRF) => Some(Ordering::Greater),
            (CpuMemoryLevel::L1, CpuMemoryLevel::GL) => Some(Ordering::Less),
            (CpuMemoryLevel::GL, CpuMemoryLevel::L1) => Some(Ordering::Greater),
            (CpuMemoryLevel::L1, CpuMemoryLevel::L1) => unreachable!(),
            (CpuMemoryLevel::GL, CpuMemoryLevel::GL) => unreachable!(),
        }
    }
}

impl Display for CpuMemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                CpuMemoryLevel::RF => "RF",
                CpuMemoryLevel::VRF => "VRF",
                CpuMemoryLevel::L1 => "L1",
                CpuMemoryLevel::GL => "GL",
            }
        )
    }
}

impl BiMap for CpuMemoryLevelBimap {
    type Domain = CpuMemoryLevel;
    type Codomain = u8;

    fn apply(&self, level: &CpuMemoryLevel) -> u8 {
        match level {
            CpuMemoryLevel::RF => 0,
            CpuMemoryLevel::VRF => 1,
            CpuMemoryLevel::L1 => 2,
            CpuMemoryLevel::GL => 3,
        }
    }

    fn apply_inverse(&self, i: &u8) -> CpuMemoryLevel {
        match *i {
            0 => CpuMemoryLevel::RF,
            1 => CpuMemoryLevel::VRF,
            2 => CpuMemoryLevel::L1,
            3 => CpuMemoryLevel::GL,
            _ => panic!("Invalid index: {}", i),
        }
    }
}

impl CanonicalBimap for CpuMemoryLevel {
    type Bimap = CpuMemoryLevelBimap;

    fn bimap() -> Self::Bimap {
        CpuMemoryLevelBimap
    }
}

fn bufferize_actions<Tgt: Target>(
    spec: &LogicalSpec<Tgt>,
) -> impl Iterator<Item = Action<Tgt>> + '_ {
    let LogicalSpec::Compose {
        components,
        operand_auxes: _,
        serial_only: _,
    } = spec
    else {
        panic!("bufferize_actions called on non-Compose Spec");
    };

    let mut results = vec![];

    for index in 0..(components.len() - 1) {
        let comp = &components[index + 1];
        let comp_out_idx = comp.typ.unique_output_index().unwrap();
        let intermediate_shape = comp.parameter_shape(comp_out_idx);
        let intermediate_dtype = comp.dtypes[comp_out_idx];

        for level in Tgt::levels() {
            let vector_bytes = level.vector_bytes();

            for layout in Tgt::move_destination_layouts(&intermediate_shape, intermediate_dtype) {
                // TODO: Need to implement `can_move_to`-style logic here.

                if !vector_bytes.is_empty() {
                    for vector_size in gen_vector_sizes(intermediate_dtype, vector_bytes) {
                        results.push(Action::Bufferize(Bufferize {
                            index,
                            level,
                            layout: layout.clone(),
                            vector_size: Some(vector_size),
                        }));
                    }
                } else {
                    results.push(Action::Bufferize(Bufferize {
                        index,
                        level,
                        layout,
                        vector_size: None,
                    }));
                }
            }
        }
    }

    results.into_iter()
}

fn physicaltransposebyte_applies_to_operands<Tgt>(
    typ: &PrimitiveSpecType,
    operands: &[TensorSpec<Tgt>],
    vector_values: u32,
) -> bool
where
    Tgt: Target<Level = CpuMemoryLevel>,
{
    if !matches!(typ, PrimitiveSpecType::Move) {
        return false;
    }

    for op in operands {
        match op {
            TensorSpec {
                shape,
                dtype: Dtype::Uint8 | Dtype::Sint8,
                aux:
                    TensorSpecAux {
                        contig,
                        aligned: _,
                        level: CpuMemoryLevel::VRF,
                        layout,
                        vector_size: Some(v),
                    },
            } if shape[..] == *shape![2, vector_values]
                && *contig == layout.contiguous_full()
                && v.get() == vector_values => {}
            _ => return false,
        };
    }

    if !operands[0].layout().is_row_major() {
        return false;
    }
    if operands[1].layout() != &col_major(2) {
        return false;
    }
    true
}

pub fn shared_broadcastvecmult_applies_to_operands<Tgt: Target<Level = CpuMemoryLevel>>(
    operands: &[TensorSpec<Tgt>],
) -> bool {
    let scalar_level = operands[0].level();
    if scalar_level != CpuMemoryLevel::RF && scalar_level != CpuMemoryLevel::L1 {
        return false;
    }

    // Second and third parameters must be in VRF, vector size multiples, aligned, contig., and
    // have the same dtype as the first parameter.
    for o in &operands[1..] {
        if o.level() != CpuMemoryLevel::VRF {
            return false;
        }
        if !o.aligned() || !o.is_contiguous() {
            return false;
        }
        if o.volume().get() % o.vector_size().unwrap().get() != 0 {
            return false;
        }
    }

    // First parameter is a single value.
    if operands[0].shape().iter().any(|d| d.get() != 1) {
        return false;
    }

    // Second parameter must have shape 1x1xn.
    if operands[1].shape().len() != 3
        || operands[1].shape()[0].get() != 1
        || operands[1].shape()[1].get() != 1
    {
        return false;
    }

    // Third (output) parameter shape must match the second input parameter shape and
    // vector size.
    if operands[1].shape() != operands[2].shape() {
        return false;
    }
    if operands[1].vector_size() != operands[2].vector_size() {
        return false;
    }
    true
}

/// Yields versions of the given [Layout] with packed dimensions added.
///
/// The specific sizes of the inner/packed dimension depend on the given layout, tensor
/// shape, and tensor [Dtype].
fn packed_layouts_for_standard_layout<'a>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
) -> impl Iterator<Item = Layout> + 'a {
    generic_packed_layouts_for_standard_layout::<'a>(
        original_layout,
        shape,
        dtype,
        all_target_vector_bytes,
        pack_sizes_for_dim,
        &PhysDim::Packed,
    )
}

fn pack_sizes_all(
    dtype: Dtype,
    all_target_vector_bytes: &[u32],
) -> impl Iterator<Item = DimSize> + '_ {
    iter::once(nz!(2u32)).chain(all_target_vector_bytes.iter().filter_map(move |&bytes| {
        let (vector_value_count, vvc_rem) = bytes.div_rem(u32::from(dtype.size()));
        if vvc_rem == 0 {
            Some(DimSize::new(vector_value_count).unwrap())
        } else {
            None
        }
    }))
}

/// Like [pack_sizes_all] but filters out sizes that are not valid for the given dimension size.
fn pack_sizes_for_dim(
    dim_size: DimSize,
    dtype: Dtype,
    all_target_vector_bytes: &[u32],
) -> impl Iterator<Item = DimSize> + '_ {
    pack_sizes_all(dtype, all_target_vector_bytes)
        .filter(move |&pack_size| pack_size < dim_size && dim_size.get() % pack_size.get() == 0)
}

fn oddeven_layouts_for_standard_layout<'a>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
) -> impl Iterator<Item = Layout> + 'a {
    generic_packed_layouts_for_standard_layout::<'a>(
        original_layout,
        shape,
        dtype,
        all_target_vector_bytes,
        oddeven_sizes_for_dim,
        &PhysDim::OddEven,
    )
}

fn oddeven_sizes_all(
    dtype: Dtype,
    all_target_vector_bytes: &[u32],
) -> impl Iterator<Item = DimSize> + '_ {
    all_target_vector_bytes.iter().filter_map(move |&bytes| {
        let double_bytes = bytes * 2;
        let (vector_value_count, vvc_rem) = double_bytes.div_rem(u32::from(dtype.size()));
        if vvc_rem != 0 {
            return None;
        }
        Some(vector_value_count.try_into().unwrap())
    })
}

fn oddeven_sizes_for_dim(
    dim_size: DimSize,
    dtype: Dtype,
    all_target_vector_bytes: &[u32],
) -> impl Iterator<Item = DimSize> + '_ {
    oddeven_sizes_all(dtype, all_target_vector_bytes)
        .filter(move |&pack_size| pack_size <= dim_size && dim_size.get() % pack_size.get() == 0)
}

/// Implements logic common to [packed_layouts_for_standard_layout] and
/// [oddeven_layouts_for_standard_layout].
fn generic_packed_layouts_for_standard_layout<'a, F, Fr, G>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
    sizes_for_dim_fn: F,
    phys_dim_construct: &'a G,
) -> impl Iterator<Item = Layout> + 'a
where
    F: 'a + Fn(DimSize, Dtype, &'a [u32]) -> Fr,
    Fr: 'a + Iterator<Item = DimSize>,
    G: 'a + Fn(DimSize) -> PhysDim,
{
    let Layout(dims) = &original_layout;
    debug_assert!(dims.iter().all(|(_, s)| *s == PhysDim::Dynamic));

    let final_nonone_dim = {
        let mut d = dims.len() - 1;
        while d > 0 && shape[d].get() == 1 {
            d -= 1;
        }
        d
    };

    dims[..final_nonone_dim].iter().flat_map(move |&(dim, _)| {
        let dims = dims.clone();
        let mut it = None;
        if shape[usize::from(dim)].get() != 1 {
            it = Some(
                sizes_for_dim_fn(shape[usize::from(dim)], dtype, all_target_vector_bytes).map(
                    move |strip_size| {
                        Layout::new(
                            dims.iter()
                                .cloned()
                                .chain(iter::once((dim, phys_dim_construct(strip_size))))
                                .collect(),
                        )
                    },
                ),
            );
        }
        it.into_iter().flatten()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common::{DimSize, Dtype},
        layout::{row_major, Layout},
        lspec,
        scheduling::{moves::Move, ActionT, ApplyError, NotApplicableReason},
        spec::{arb_canonical_spec, Spec},
        target::{Target, X86Target},
        views::Param,
    };
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        #[test]
        fn test_all_layouts_for_shape_are_unique(
            shape in proptest::collection::vec(1..8u32, 1..=5),
            dtype in any::<Dtype>(),
        ) {
            let shape = shape.into_iter().map(|x| DimSize::new(x).unwrap()).collect::<Vec<_>>();
            assert_unique_layouts(&X86Target::all_layouts_for_shape(&shape, dtype));
        }

        #[test]
        fn test_all_layouts_for_shape_contains_move_destination_layouts(
            shape in proptest::collection::vec(1..8u32, 1..=5),
            dtype in any::<Dtype>(),
        ) {
            let shape = shape.into_iter().map(|x| DimSize::new(x).unwrap()).collect::<Vec<_>>();
            let superset: HashSet<_> = X86Target::all_layouts_for_shape(&shape, dtype)
                .into_iter()
                .collect();
            let subset: HashSet<_> = X86Target::move_destination_layouts(&shape, dtype)
                .into_iter()
                .collect();
            assert!(superset.is_superset(&subset));
        }

        // TODO: Re-enable the following. Enumerating identical layouts wastes time, so this would
        //   be good to have.
        // #[test]
        // fn test_move_destination_layouts_have_unique_index_expressions(
        //     shape in proptest::collection::vec(1..8u32, 1..=5),
        //     dtype in any::<Dtype>(),
        // ) {
        //     let expr_id = OpaqueSymbol::new();
        //     let mut index_exprs: Vec<(Vec<i32>, Layout)> = Vec::new();
        //     for layout in X86Target::move_destination_layouts(&shape, dtype) {
        //         let ie = layout.buffer_indexing_expr(&expr_id, &shape);
        //         let evaluated_pts = eval_all_index_expr_points(&ie, &shape);
        //         for (prev_pts, prev_layout) in index_exprs.iter() {
        //             if prev_pts == &evaluated_pts {
        //                 panic!(
        //                     "Layouts had identical indexing expressions {} \
        //                     for shape {:?}: {:?} and {:?}", ie, shape, prev_layout, layout
        //                 );
        //             }
        //         }
        //         index_exprs.push((evaluated_pts, layout));
        //     }
        // }

        #[test]
        #[should_panic]
        fn test_packed_layout_with_strip_size_one_panics(
            example in arb_test_packed_layout_with_strip_size_one_is_row_major()
        ) {
            let (shape, strip_dim) = example;
            Layout::new(
                (0..u8::try_from(shape.len()).unwrap())
                    .map(|dim| (dim, PhysDim::Dynamic))
                    .chain(std::iter::once((strip_dim, PhysDim::Packed(nz!(1u32)))))
                    .collect(),
            );
        }

        #[test]
        fn test_actions_include_move_actions_for_all_parameters(
            spec in arb_canonical_spec::<X86Target>(None, None),
        ) {
            let lspec = &spec.0;
            let mut seen_source_idxs = HashSet::new();
            for action in X86Target::actions(lspec) {
                if let Action::Move(Move { source_idx, .. }) = action {
                    seen_source_idxs.insert(source_idx);
                }
            }

            let mut moveable_parameter_idxs = HashSet::new();
            // OnePrefix is the exception. It should have no Moves.
            if !matches!(
                lspec,
                LogicalSpec::Primitive(PrimitiveBasics { typ: PrimitiveSpecType::OnePrefix, ..}, ..)
            ) {
                lspec
                    .parameters()
                    .into_iter()
                    .enumerate()
                    .for_each(|(idx, parameter)| {
                        let dest_layouts = X86Target::move_destination_layouts(
                            parameter.shape(), parameter.dtype()
                        );
                        if !dest_layouts.is_empty() {
                            moveable_parameter_idxs.insert(u8::try_from(idx).unwrap());
                        }
                    });
            }
            prop_assert_eq!(
                &seen_source_idxs,
                &moveable_parameter_idxs,
                "Only saw {:?} but expected {:?} for {}",
                seen_source_idxs,
                moveable_parameter_idxs,
                spec
            );
        }

    }

    prop_compose! {
        fn arb_test_packed_layout_with_strip_size_one_is_row_major()
            (shape in proptest::collection::vec(1..8u32, 1..=5))
            (strip_dim in 0..shape.len(), shape in Just(shape))
            -> (Vec<DimSize>, u8)
        {
            (shape.into_iter().map(|x| DimSize::new(x).unwrap()).collect(), strip_dim.try_into().unwrap())
        }
    }

    #[test]
    fn test_twovecbroadcastvecmult_applies_to_operands() {
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [1, 1, 2, 16],
            (u8, CpuMemoryLevel::L1, row_major),
            (i8, CpuMemoryLevel::VRF, batched_col_major(3), 32),
            (i16, CpuMemoryLevel::VRF, row_major, 16),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(CpuKernel::TwoVecBroadcastVecMultAddU8S8S16.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_kernel_memory_constrains_placement() {
        let logical_spec: LogicalSpec<X86Target> = lspec!(MatmulAccum(
            [1, 1, 1, 16],
            (u8, CpuMemoryLevel::RF, row_major),
            (u8, CpuMemoryLevel::VRF, row_major, 16),
            (u8, CpuMemoryLevel::VRF, row_major, 16),
            serial
        ));
        let spec = Spec(
            logical_spec,
            MemoryLimits::Standard(MemVec::zero::<X86Target>()),
        );
        assert!(spec.is_canonical());

        let act = Action::Select(Select(CpuKernel::BroadcastVecMultAdd.into(), false));
        let act_application = act.apply(&spec);
        let act_unchecked_application = act.apply_unchecked_canon(&spec);
        assert!(
            matches!(
                act_application,
                Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                    _
                )))
            ),
            "Expected OutOfMemory error, got {act_application:?}",
        );
        assert!(
            matches!(
                act_unchecked_application,
                Err(ApplyError::NotApplicable(NotApplicableReason::OutOfMemory(
                    _
                )))
            ),
            "Expected OutOfMemory error, got {act_unchecked_application:?}",
        );
    }

    #[test]
    fn test_vectorassign_main_cost() {
        let arg0 = TensorSpec::new_canon(
            shape![2, 32],
            Dtype::Uint32,
            row_major(2).contiguous_full(),
            true,
            CpuMemoryLevel::VRF,
            row_major(2),
            Some(nz!(8u32)),
        );
        let arg1 = TensorSpec::new_canon(
            shape![2, 32],
            Dtype::Uint32,
            row_major(2).contiguous_full(),
            true,
            CpuMemoryLevel::L1,
            row_major(2),
            None,
        );
        let parameter0: Param<X86Target> = Param::new(0, arg0);
        let parameter1: Param<X86Target> = Param::new(1, arg1);
        let cost = CpuKernel::VectorAssign.main_cost(&[parameter0, parameter1]);
        assert_eq!(cost, 8 * ASSIGN_INST_COST);
    }

    fn assert_unique_layouts(layouts: &[Layout]) {
        let layouts_set = layouts.iter().collect::<HashSet<_>>();
        assert_eq!(layouts.len(), layouts_set.len());
    }

    // fn eval_all_index_expr_points(expr: &NonAffineExpr<BufferVar>, shape: &[DimSize]) -> Vec<i32> {
    //     let mut results = vec![];
    //     for pt in shape.iter().map(|&d| 0..d.get()).multi_cartesian_product() {
    //         let evaluated: NonAffineExpr<&str> = expr.clone().map_vars(&mut |var| match var {
    //             BufferVar::TileIdx(_, _) => panic!("TileIdx in index expression"),
    //             BufferVar::Pt(dim, _) => NonAffineExpr::constant(pt[dim as usize] as i32),
    //         });
    //         assert!(
    //             evaluated.0.is_empty(),
    //             "Non-constant index expression: {:?}",
    //             evaluated
    //         );
    //         results.push(evaluated.1);
    //     }
    //     results
    // }
}
