use super::common_actions::{bufferize_actions, move_actions, split_actions, tile_out_actions};
use crate::codegen::c_utils::VecType;
use crate::common::{DimSize, Dtype};
use crate::cost::MainCost;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::layout;
use crate::layout::{batched_col_major, col_major, nhwc, row_major, Layout, PhysDim};
use crate::memorylimits::{MemoryAllocation, MemoryLimits};
use crate::scheduling::broadcast_first::BroadcastFirst;
use crate::scheduling::select::Select;
use crate::scheduling::spatial_split::SpatialSplit;
use crate::scheduling::to_accum::ToAccum;
use crate::scheduling::to_max_and_denom::ToMaxAndDenominator;
use crate::scheduling::to_max_and_unscaled::ToMaxAndUnscaled;
use crate::scheduling::to_softmax_parts::{ToSoftmaxParts, ToSoftmaxPartsRecompute};
use crate::scheduling::Action;
use crate::shape;
use crate::spec::{FillValue, LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use crate::target::{Kernel, MemoryLevel, Target, TargetId, LEVEL_COUNT};
use crate::tensorspec::gen_vector_sizes_opt;
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::views::View;

use divrem::DivRem;
use itertools::Itertools;
use nonzero::nonzero as nz;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::{self, once};

const INST_COST: MainCost = 1;
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
    type Level: MemoryLevel
        + From<CpuMemoryLevel>
        + Into<CpuMemoryLevel>
        + PartialEq<CpuMemoryLevel>;
    fn max_mem() -> MemoryLimits;
    fn target_id() -> TargetId;
    fn vec_types() -> &'static [VecType];
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
    Assign,
    MemsetZero,
    /// Lowers to Clang vector extensions' zero-assignment, which, on x86, should emit `vxorps`.
    VectorZero,
    ValueNegInf,
    VectorNegInf,
    ValueMin,
    VectorMin,
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
    type Level = <Self as CpuTarget>::Level;
    type Kernel = <Self as CpuTarget>::Kernel;
    type ActionsIter<'a> = Box<dyn Iterator<Item = Action<Self>> + 'a>;

    fn line_size() -> u32 {
        32
    }

    fn max_mem() -> MemoryLimits {
        <Self as CpuTarget>::max_mem()
    }

    fn processors() -> u8 {
        32
    }

    fn default_level() -> Self::Level {
        CpuMemoryLevel::GL.into()
    }

    fn levels() -> [Self::Level; LEVEL_COUNT] {
        CPU_LEVELS.map(Into::into)
    }

    fn possible_destination_levels(slower: Self::Level) -> Vec<Self::Level> {
        match slower.into() {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => vec![slower],
            CpuMemoryLevel::L1 => vec![
                slower,
                CpuMemoryLevel::RF.into(),
                CpuMemoryLevel::VRF.into(),
            ],
            CpuMemoryLevel::GL => vec![slower, CpuMemoryLevel::L1.into()],
        }
    }

    fn all_layouts_for_shape(shape: &[DimSize], dtype: Dtype) -> Vec<Layout> {
        let all_target_vector_bytes = Self::levels()
            .into_iter()
            .flat_map(|lvl| lvl.vector_bytes().iter().copied())
            .collect::<Vec<_>>();

        // The following could be faster. It keeps two copies of the non-packed layouts
        // (`base` and the first few values in `result`) and it doesn't compute the size
        // of the `result` ahead-of-time, potentially causing some Vec resizes.
        let only_ones = shape.iter().all(|&d| d.get() == 1);
        let mut base = Vec::with_capacity(2);
        base.push(row_major(shape));
        if !only_ones {
            if shape.len() == 2 {
                base.push(col_major(shape));
                if base[1] == base[0] {
                    base.pop();
                }
            } else if shape.len() == 4 {
                base.push(nhwc(shape));
                if base[1] == base[0] {
                    base.pop();
                }
            }
        }

        let mut result = base.clone();

        let all_packing_sizes = pack_sizes_all(dtype, &all_target_vector_bytes).collect::<Vec<_>>();
        result.extend(base.iter().flat_map(|original_layout| {
            let Layout { dims, contig: _ } = original_layout;
            dims.iter()
                .map(|(d, _)| *d)
                .unique()
                .cartesian_product(&all_packing_sizes)
                .filter_map(|(packing_dim, &packing_size)| {
                    debug_assert_ne!(packing_size.get(), 1);
                    let packing_dim = usize::from(packing_dim);
                    if packing_dim == usize::from(dims.last().unwrap().0)
                        || !shape[packing_dim].get().is_multiple_of(packing_size.get())
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
        result.extend(base.iter().flat_map(|original_layout| {
            let Layout { dims, contig: _ } = original_layout;
            dims.iter()
                .map(|dim| dim.0)
                .unique()
                .cartesian_product(&all_oddeven_sizes)
                .filter_map(|(packing_dim, &packing_size)| {
                    debug_assert_ne!(packing_size.get(), 1);
                    if !shape[usize::from(packing_dim)]
                        .get()
                        .is_multiple_of(packing_size.get())
                    {
                        return None;
                    }
                    Some(Layout::new(
                        dims.iter()
                            .cloned()
                            .chain(iter::once((packing_dim, PhysDim::OddEven(packing_size))))
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
        let mut base = Vec::with_capacity(2);
        base.push(row_major(shape));
        if !only_ones {
            if shape.len() == 2 {
                base.push(col_major(shape));
                if base[1] == base[0] {
                    base.pop();
                }
            } else if shape.len() == 4 {
                base.push(nhwc(shape));
                if base[1] == base[0] {
                    base.pop();
                }
            }
        }

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
                    const MOVE_KERNELS: [CpuKernel; 7] = [
                        CpuKernel::Assign,
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
                PrimitiveSpecType::Fill {
                    value: FillValue::Min,
                } => {
                    const MIN_KERNELS: [CpuKernel; 2] = [CpuKernel::VectorMin, CpuKernel::ValueMin];
                    &MIN_KERNELS
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
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { accum, .. }
                    if !*accum =>
                {
                    Box::new(iter.chain(once(ToAccum::default().into())))
                }
                PrimitiveSpecType::Matmul { accum }
                | PrimitiveSpecType::Max { accum, .. }
                | PrimitiveSpecType::SoftmaxDenominator { accum, .. }
                    if *accum =>
                {
                    Box::new(iter.chain(split_actions(spec)))
                }
                PrimitiveSpecType::Conv { accum } => {
                    if *accum {
                        if spec.can_spatial_split() {
                            Box::new(iter.chain(once(SpatialSplit::default().into())))
                        } else {
                            Box::new(iter)
                        }
                    } else {
                        Box::new(iter.chain(once(ToAccum::default().into())))
                    }
                }
                PrimitiveSpecType::DivideVecScalar { .. } => {
                    let mut broadcast_firsts = Vec::new();
                    if let LogicalSpec::Primitive(basics, _, _) = spec {
                        let spec_shape = &basics.spec_shape;
                        let dtype = basics.dtypes[1];
                        for level in Self::levels() {
                            for layout in Self::move_destination_layouts(spec_shape, dtype) {
                                gen_vector_sizes_opt(dtype, level.vector_bytes()).for_each(
                                    |vector_size| {
                                        broadcast_firsts.push(
                                            BroadcastFirst {
                                                broadcast_level: level,
                                                broadcast_layout: layout.clone(),
                                                broadcast_vector_size: vector_size,
                                            }
                                            .into(),
                                        );
                                    },
                                );
                            }
                        }
                    }
                    Box::new(iter.chain(broadcast_firsts))
                }
                PrimitiveSpecType::Softmax { .. } => {
                    let mut softmax_actions = Vec::new();
                    if let LogicalSpec::Primitive(basics, _, _) = spec {
                        let spec_shape = &basics.spec_shape;
                        let dtype = basics.dtypes[0];
                        let levels = Self::levels();
                        let layouts = Self::move_destination_layouts(spec_shape, dtype);

                        // Fully fused loops using shared alt_level for both max and exps
                        for denom_level in levels {
                            for denom_layout in &layouts {
                                gen_vector_sizes_opt(dtype, denom_level.vector_bytes()).for_each(
                                    |denominator_vector_size| {
                                        for alt_level in levels {
                                            for alt_layout in &layouts {
                                                gen_vector_sizes_opt(
                                                    dtype,
                                                    alt_level.vector_bytes(),
                                                )
                                                .for_each(|alt_vector_size| {
                                                    softmax_actions.push(Action::ToSoftmaxParts(
                                                        ToSoftmaxParts {
                                                            denominator_level: denom_level,
                                                            denominator_layout: denom_layout
                                                                .clone(),
                                                            denominator_vector_size,
                                                            exps_level: alt_level,
                                                            exps_layout: alt_layout.clone(),
                                                            exps_vector_size: alt_vector_size,
                                                        },
                                                    ));
                                                    softmax_actions.push(
                                                        Action::ToSoftmaxPartsRecompute(
                                                            ToSoftmaxPartsRecompute {
                                                                max_level: alt_level,
                                                                max_layout: alt_layout.clone(),
                                                                max_vector_size: alt_vector_size,
                                                                denominator_level: denom_level,
                                                                denominator_layout: denom_layout
                                                                    .clone(),
                                                                denominator_vector_size,
                                                            },
                                                        ),
                                                    );
                                                });
                                            }
                                        }
                                    },
                                );
                            }
                        }
                    }
                    Box::new(iter.chain(softmax_actions))
                }
                PrimitiveSpecType::SoftmaxDenominatorAndMax { .. } => Box::new(iter.chain(once(
                    Action::ToMaxAndDenominator(ToMaxAndDenominator::default()),
                ))),
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. } => {
                    let mut unscaled_actions = Vec::new();
                    if let LogicalSpec::Primitive(basics, _, _) = spec {
                        let spec_shape = &basics.spec_shape;
                        let dtype = basics.dtypes[0];

                        for level in Self::levels() {
                            for layout in Self::move_destination_layouts(spec_shape, dtype) {
                                gen_vector_sizes_opt(dtype, level.vector_bytes()).for_each(
                                    |vector_size| {
                                        unscaled_actions.push(Action::ToMaxAndUnscaled(
                                            ToMaxAndUnscaled {
                                                max_level: level,
                                                max_layout: layout.clone(),
                                                max_vector_size: vector_size,
                                            },
                                        ));
                                    },
                                );
                            }
                        }
                    }
                    Box::new(iter.chain(unscaled_actions))
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

    fn vec_types() -> &'static [VecType] {
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
            | CpuKernel::Assign
            | CpuKernel::CastBf16F32
            | CpuKernel::VectorCastBf16F32
            | CpuKernel::VecScalarAssign => 2,
            CpuKernel::MemsetZero
            | CpuKernel::VectorZero
            | CpuKernel::ValueNegInf
            | CpuKernel::VectorNegInf
            | CpuKernel::ValueMin
            | CpuKernel::VectorMin => 1,
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
                    && broadcastvecmult_side(&operands).is_some()
            }
            CpuKernel::BroadcastVecMultAddBf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && operands[0].dtype() == Dtype::Bfloat16
                    && operands[1].dtype() == Dtype::Bfloat16
                    && operands[2].dtype() == Dtype::Float32
                    && operands[1].vector_size() == operands[2].vector_size()
                    && broadcastvecmult_bf16_applies_to_operands(&operands)
            }
            CpuKernel::TwoVecBroadcastVecMultAddU8S8S16 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && matches!(
                        &operands[..],
                        [
                            lhs @ TensorSpec {
                                shape: lhs_shape,
                                dtype: Dtype::Uint8,
                                aux: _,
                            },
                            rhs @ TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Sint8,
                                aux: TensorSpecAux {
                                    vector_size: Some(rhs_vector_size),
                                    ..
                                },
                            },
                            out @ TensorSpec {
                                shape: out_shape,
                                dtype: Dtype::Sint16,
                                aux: _,
                            }
                        ] if lhs_shape[..] == [nz!(1u32), nz!(1u32), nz!(2u32)]
                          && rhs_shape[..2] == [nz!(1u32), nz!(2u32)]
                          && rhs_shape[2].get() * 2 == rhs_vector_size.get()
                          && out_shape[..2] == [nz!(1u32), nz!(1u32)]
                          && out_shape[2].get() * 2 == rhs_vector_size.get()
                          && rhs.layout() == &batched_col_major(rhs_shape)
                          && out.layout().is_row_major()
                          && lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous()
                          && lhs.level() == CpuMemoryLevel::L1
                          && rhs.level() == CpuMemoryLevel::VRF
                          && out.level() == CpuMemoryLevel::VRF
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
                                aux: _,
                            },
                            rhs @ TensorSpec {
                                shape: _,
                                dtype: Dtype::Float32,
                                aux: _,
                            },
                            TensorSpec {
                                shape: out_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: out_level,
                                    ..
                                },
                            }
                        ] if lhs_shape[0] == nz!(1u32)
                          && lhs_shape[1] == nz!(1u32)
                          && dotproduct_accum_count(lhs_shape[2].get()).is_some()
                          && out_shape[..] == [nz!(1u32), nz!(1u32), nz!(1u32)]
                          && lhs.layout().is_row_major()
                          && rhs.layout().is_col_major()
                          && lhs.is_contiguous() && rhs.is_contiguous()
                          && lhs.vector_size().is_none_or(|v| v == DOT_PRODUCT_STRIP_SIZE)
                          && rhs.vector_size().is_none_or(|v| v == DOT_PRODUCT_STRIP_SIZE)
                          // If lhs is already in VRF, require a full vector so we don't need pointer arithmetic.
                          && (lhs.level() == CpuMemoryLevel::L1
                              || (lhs.level() == CpuMemoryLevel::VRF
                                  && lhs.vector_size() == Some(DOT_PRODUCT_STRIP_SIZE)))
                          && rhs.level() == CpuMemoryLevel::L1
                          && *out_level == CpuMemoryLevel::RF
                    )
            }
            CpuKernel::DotProductLoopF32Bf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && dotproductloop_bf16_applies(
                        &operands,
                        Dtype::Float32,
                        [row_major(operands[0].shape())],
                    )
            }
            CpuKernel::DotProductLoopF32InterleavedBf16F32 => {
                // TODO: Simplify with a closure instead of constructing all layouts AOT.
                if !matches!(typ, PrimitiveSpecType::Matmul { accum: true }) {
                    return false;
                }
                let layout0 = layout![0, 1, 2, 2 oe(16)];
                let layout1 = layout![0, 2, 1, 1 oe(16)];
                dotproductloop_bf16_applies(&operands, Dtype::Float32, [layout0, layout1])
            }
            CpuKernel::DotProductLoopBf16Bf16F32 => {
                matches!(typ, PrimitiveSpecType::Matmul { accum: true })
                    && dotproductloop_bf16_applies(
                        &operands,
                        Dtype::Bfloat16,
                        [row_major(operands[0].shape())],
                    )
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
                                level: _,
                                layout: src_layout,
                                vector_size: src_vs,
                            },
                        },
                        dest @ TensorSpec {
                            shape: dest_shape,
                            dtype: Dtype::Float32,
                            aux: TensorSpecAux {
                                level: _,
                                layout: dest_layout,
                                vector_size: dest_vs,
                            },
                        },
                    ] if src.is_contiguous() && dest.is_contiguous()
                      && src_shape.iter().all(|d| d.get() == 1 || d.get() == 16)
                      && src_shape.iter().filter(|d| d.get() == 16).count() == 1
                      && src_shape == dest_shape
                      && src_vs == &Some(nz!(16u32))
                      && dest_vs == &Some(nz!(8u32))
                      && src_layout.is_row_major()
                      && dest_layout.dims.iter().all(|(_, pd)| pd == &PhysDim::Dynamic || pd == &leaved)
                      && dest_layout.dims.iter().filter(|(_, pd)| pd == &leaved).count() == 1
                      && src.level() == CpuMemoryLevel::VRF
                      && dest.level() == CpuMemoryLevel::VRF
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
                    if !shape[dim_us].get().is_multiple_of(8) {
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
                if !operands[0]
                    .volume()
                    .get()
                    .is_multiple_of(operands[0].vector_size().unwrap().get())
                {
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
                if !operands[0]
                    .volume()
                    .get()
                    .is_multiple_of(operands[0].vector_size().unwrap().get())
                {
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
                if !dest_shape[usize::from(dim)].get().is_multiple_of(vs.get()) {
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
            CpuKernel::Assign => {
                debug_assert_eq!(operands.len(), 2);

                if !matches!(typ, PrimitiveSpecType::Move) {
                    return false;
                }

                let is_scalar = operands
                    .iter()
                    .flat_map(|o| o.shape())
                    .all(|d| d.get() == 1);
                if is_scalar {
                    for o in &operands[1..] {
                        if (o.dtype(), o.layout()) != (operands[0].dtype(), operands[0].layout()) {
                            return false;
                        }
                    }

                    return operands.iter().any(|o| o.level() == CpuMemoryLevel::RF)
                        && operands.iter().all(|o| {
                            o.level() == CpuMemoryLevel::RF || o.level() == CpuMemoryLevel::L1
                        });
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
                                    level: lhs_level,
                                    vector_size: None,
                                    ..
                                },
                            },
                            TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: rhs_level,
                                    vector_size: None,
                                    ..
                                },
                            }
                        ] if lhs_shape.iter().all(|d| d.get() == 1)
                          && rhs_shape.iter().all(|d| d.get() == 1)
                          && *lhs_level == CpuMemoryLevel::RF
                          && *rhs_level == CpuMemoryLevel::RF
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
                                    level: lhs_level,
                                    layout: lhs_layout,
                                    vector_size: Some(lhs_vector_size),
                                    ..
                                },
                            },
                            TensorSpec {
                                shape: rhs_shape,
                                dtype: Dtype::Float32,
                                aux: TensorSpecAux {
                                    level: rhs_level,
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
                          && *lhs_level == CpuMemoryLevel::VRF
                          && *rhs_level == CpuMemoryLevel::VRF
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
            CpuKernel::VectorZero | CpuKernel::VectorNegInf | CpuKernel::VectorMin => {
                let expected_fill_value = match self {
                    CpuKernel::VectorZero => FillValue::Zero,
                    CpuKernel::VectorNegInf => FillValue::NegInf,
                    CpuKernel::VectorMin => FillValue::Min,
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
            CpuKernel::ValueMin => {
                debug_assert_eq!(operands.len(), 1);
                matches!(
                    typ,
                    PrimitiveSpecType::Fill {
                        value: FillValue::Min
                    }
                ) && operands[0].volume().get() == 1
                    && operands[0].level() == CpuMemoryLevel::RF
                    && operands[0].is_contiguous()
            }
        }
    }

    pub fn memory_allocated<P: View>(&self, parameters: &[P]) -> MemoryAllocation {
        match self {
            // TODO: Model memory correctly for BroadcastVecMultAddBf16F32
            CpuKernel::BroadcastVecMultAdd => {
                // BroadcastVecMult applies to 1x1xn Matmuls. It broadcasts into a single
                // vector register which is reused across all n-axis vectors.
                MemoryAllocation::Simple(CPU_LEVELS.map(
                    |level| {
                        if level.vector_rf() {
                            1
                        } else {
                            0
                        }
                    },
                ))
            }
            CpuKernel::TwoVecBroadcastVecMultAddU8S8S16 | CpuKernel::BroadcastVecMultAddBf16F32 => {
                let vec_tensor_spec = &parameters[1].spec();
                let regs = u64::from(
                    vec_tensor_spec.volume().get() / vec_tensor_spec.vector_size().unwrap().get(),
                );
                MemoryAllocation::Simple(CPU_LEVELS.map(|level| {
                    if level.vector_rf() {
                        regs * 2
                    } else {
                        0
                    }
                }))
            }
            CpuKernel::DotProductLoop
            | CpuKernel::DotProductLoopBf16Bf16F32
            | CpuKernel::DotProductLoopF32InterleavedBf16F32
            | CpuKernel::DotProductLoopF32Bf16F32 => {
                // TODO: Count any additional peak memory from sum8.
                MemoryAllocation::Simple(CPU_LEVELS.map(|level| {
                    let mut used = 0;
                    if level.vector_rf() {
                        used = 1;
                        // TODO: Add intermediate consumption
                    }
                    used
                }))
            }
            CpuKernel::PhysicalTransposeByte256 => MemoryAllocation::Simple(CPU_LEVELS.map(
                |level| {
                    if level.vector_rf() {
                        2
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
            CpuKernel::VectorMax => MemoryAllocation::Simple(CPU_LEVELS.map(|level| {
                if level.vector_rf() {
                    1 // one __m128 allocated by horizontal_max_f32
                } else {
                    0
                }
            })),
            CpuKernel::VectorSoftmaxDenominator => {
                // TODO: Check if VectorSoftmaxDenominator allocates more than 2 vectors.
                MemoryAllocation::Simple(CPU_LEVELS.map(
                    |level| {
                        if level.vector_rf() {
                            2
                        } else {
                            0
                        }
                    },
                ))
            }
            CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32 => MemoryAllocation::Simple(
                CPU_LEVELS.map(|level| if level.vector_rf() { 4 } else { 0 }),
            ),
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

                let out_spec = parameters.last().unwrap().spec();
                let vector_size = out_spec.vector_size().unwrap().get();
                let volume = out_spec.volume().get();
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
            | CpuKernel::ValueNegInf
            | CpuKernel::ValueMin => {
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
            CpuKernel::Assign => {
                let is_scalar = parameters
                    .iter()
                    .flat_map(|parameter| parameter.spec().shape())
                    .all(|d| d.get() == 1);
                if is_scalar {
                    ASSIGN_INST_COST
                } else {
                    let vector_value_count = parameters[0]
                        .spec()
                        .vector_size()
                        .unwrap_or_else(|| parameters[1].spec().vector_size().unwrap());
                    let vec_count = parameters[0].spec().volume().get() / vector_value_count.get();
                    ASSIGN_INST_COST * vec_count
                }
            }
            CpuKernel::MemsetZero => ASSIGN_INST_COST,
            CpuKernel::VectorZero | CpuKernel::VectorNegInf | CpuKernel::VectorMin => {
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

pub(crate) fn dotproduct_accum_count(k: u32) -> Option<u32> {
    let strip = DOT_PRODUCT_STRIP_SIZE.get();
    if !k.is_multiple_of(strip) {
        return None;
    }
    let chunk_count = k / strip;
    (1..=DOT_PRODUCT_ACCUM_COUNT)
        .rev()
        .find(|c| chunk_count.is_multiple_of(*c))
}

fn dotproductloop_bf16_applies<const N: usize, Tgt: CpuTarget>(
    operands: &[TensorSpec<Tgt>],
    lhs_dtype: Dtype,
    allowed_lhs_layouts: [Layout; N],
) -> bool {
    let Ok(expected_rhs_layout) = layout![0, 2, 1].canonicalize(operands[1].shape()) else {
        return false;
    };
    matches!(
        operands,
        [
            lhs @ TensorSpec {
                shape: lhs_shape,
                dtype: ldt,
                aux: _
            },
            rhs @ TensorSpec {
                shape: _,
                dtype: Dtype::Bfloat16,
                aux: _
            },
            out @ TensorSpec {
                shape: out_shape,
                dtype: Dtype::Float32,
                aux: _
            }
        ] if *ldt == lhs_dtype
          && lhs_shape[2].get() % (DOT_PRODUCT_BF16_STRIP_SIZE.get() * DOT_PRODUCT_BF16_ACCUM_COUNT) == 0
          && out_shape[0] == nz!(1u32)  // means {lhs, rhs}_shape also equal 1
          && out_shape[1] == nz!(1u32)
          && out_shape[2] == nz!(1u32)
          && allowed_lhs_layouts
               .into_iter()
               .filter_map(|l| l.canonicalize(lhs_shape).ok())
               .contains(lhs.layout())
          && rhs.layout() == &expected_rhs_layout
          && lhs.level() == CpuMemoryLevel::L1
          && rhs.level() == CpuMemoryLevel::L1
          && out.level() == CpuMemoryLevel::RF
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
            CpuMemoryLevel::L1 => 2,
            #[cfg(feature = "l2-speed-gl")]
            CpuMemoryLevel::GL => 10,
            #[cfg(not(feature = "l2-speed-gl"))]
            CpuMemoryLevel::GL => 20,
        }
    }

    fn vector_bytes(&self) -> &'static [u32] {
        match &self {
            CpuMemoryLevel::VRF => &[16, 32],
            _ => &[],
        }
    }

    fn counts_registers(&self) -> bool {
        match self {
            CpuMemoryLevel::RF | CpuMemoryLevel::VRF => true,
            CpuMemoryLevel::L1 | CpuMemoryLevel::GL => false,
        }
    }

    fn has_layout(&self) -> bool {
        !matches!(self, CpuMemoryLevel::RF)
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
            _ => panic!("Invalid index: {i}"),
        }
    }
}

impl CanonicalBimap for CpuMemoryLevel {
    type Bimap = CpuMemoryLevelBimap;

    fn bimap() -> Self::Bimap {
        CpuMemoryLevelBimap
    }
}

fn physicaltransposebyte_applies_to_operands<Tgt>(
    typ: &PrimitiveSpecType,
    operands: &[TensorSpec<Tgt>],
    vector_values: u32,
) -> bool
where
    Tgt: Target,
    Tgt::Level: PartialEq<CpuMemoryLevel>,
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
                        level: rhs_level,
                        layout,
                        vector_size: Some(v),
                    },
            } if shape[..] == *shape![2, vector_values]
                && layout.is_fully_contiguous()
                && v.get() == vector_values
                && *rhs_level == CpuMemoryLevel::VRF => {}
            _ => return false,
        };
    }

    if !operands[0].layout().is_row_major() {
        return false;
    }
    if !operands[1].layout().is_col_major() {
        return false;
    }
    true
}

pub fn broadcastvecmult_side<Tgt>(operands: &[TensorSpec<Tgt>]) -> Option<(usize, usize)>
where
    Tgt: Target,
    Tgt::Level: PartialEq<CpuMemoryLevel>,
{
    debug_assert_eq!(operands.len(), 3);

    let out = &operands[2];

    // Output must live in VRF and be vector-aligned/contiguous.
    if out.level() != CpuMemoryLevel::VRF || !out.is_contiguous() {
        return None;
    }

    // Both inputs must be contiguous regardless of role.
    if !operands[0].is_contiguous() || !operands[1].is_contiguous() {
        return None;
    }

    let &[batch0, op0_dim1, op0_dim2] = operands[0].shape() else {
        unreachable!()
    };
    let &[batch1, op1_dim1, op1_dim2] = operands[1].shape() else {
        unreachable!()
    };
    let &[batch_out, out_dim1, out_dim2] = out.shape() else {
        unreachable!()
    };

    // Batch must align across all operands independent of operand roles.
    if batch0 != batch_out || batch1 != batch_out {
        return None;
    }

    let out_vector_size = out.vector_size()?;
    if !out.volume().get().is_multiple_of(out_vector_size.get()) {
        return None;
    }

    for (scalar_idx, broadcast_idx) in [(0, 1), (1, 0)] {
        let scalar = &operands[scalar_idx];
        let broadcast = &operands[broadcast_idx];

        if !(scalar.level() == CpuMemoryLevel::RF || scalar.level() == CpuMemoryLevel::L1)
            || !scalar.is_contiguous()
            || broadcast.level() != CpuMemoryLevel::VRF
            || !broadcast.is_contiguous()
        {
            continue;
        }

        // The non-VRF side must be a scalar.
        if scalar.shape().iter().any(|d| d.get() != 1) {
            continue;
        }

        // Vector sizing must line up with the output.
        if broadcast.vector_size() != Some(out_vector_size)
            || !broadcast
                .volume()
                .get()
                .is_multiple_of(out_vector_size.get())
        {
            continue;
        }

        let (b_dim1, b_dim2) = match broadcast_idx {
            0 => (op0_dim1, op0_dim2),
            1 => (op1_dim1, op1_dim2),
            _ => unreachable!(),
        };
        let (s_dim1, s_dim2) = match scalar_idx {
            0 => (op0_dim1, op0_dim2),
            1 => (op1_dim1, op1_dim2),
            _ => unreachable!(),
        };

        // Two acceptable orientations:
        // 1) Scalar is lhs: scalar dims [B, M, 1], broadcast dims [B, 1, N].
        // 2) Scalar is rhs: scalar dims [B, 1, N], broadcast dims [B, M, 1].
        if s_dim1 == out_dim1 && s_dim2.get() == 1 && b_dim1.get() == 1 && b_dim2 == out_dim2 {
            return Some((scalar_idx, broadcast_idx));
        }
        if s_dim1.get() == 1 && s_dim2 == out_dim2 && b_dim1 == out_dim1 && b_dim2.get() == 1 {
            return Some((scalar_idx, broadcast_idx));
        }
    }

    None
}

pub fn broadcastvecmult_bf16_applies_to_operands<Tgt>(operands: &[TensorSpec<Tgt>]) -> bool
where
    Tgt: Target,
    Tgt::Level: PartialEq<CpuMemoryLevel>,
{
    debug_assert_eq!(operands.len(), 3);

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
        if !o.is_contiguous() {
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
#[allow(dead_code)]
fn packed_layouts_for_standard_layout<'a>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
) -> impl Iterator<Item = Layout> + 'a {
    generic_packed_layouts_for_standard_layout::<_, _, _, false>(
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
    pack_sizes_all(dtype, all_target_vector_bytes).filter(move |&pack_size| {
        pack_size < dim_size && dim_size.get().is_multiple_of(pack_size.get())
    })
}

fn oddeven_layouts_for_standard_layout<'a>(
    original_layout: &'a Layout,
    shape: &'a [DimSize],
    dtype: Dtype,
    all_target_vector_bytes: &'a [u32],
) -> impl Iterator<Item = Layout> + 'a {
    generic_packed_layouts_for_standard_layout::<_, _, _, true>(
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
    oddeven_sizes_all(dtype, all_target_vector_bytes).filter(move |&pack_size| {
        pack_size <= dim_size && dim_size.get().is_multiple_of(pack_size.get())
    })
}

/// Implements logic common to [packed_layouts_for_standard_layout] and
/// [oddeven_layouts_for_standard_layout].
///
/// When INCLUDE_FINAL_NONONE is true, the final non-one logical dimension is also considered;
/// otherwise it is excluded.
fn generic_packed_layouts_for_standard_layout<'a, F, Fr, G, const INCLUDE_FINAL_NONONE: bool>(
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
    let Layout { dims, contig: _ } = original_layout;
    debug_assert!(dims.iter().all(|(_, s)| *s == PhysDim::Dynamic));

    let end = if dims.is_empty() {
        0
    } else {
        let mut d = dims.len() - 1;
        while d > 0 && shape[d].get() == 1 {
            d -= 1;
        }
        if INCLUDE_FINAL_NONONE {
            d += 1;
        }
        d
    };

    dims[..end].iter().flat_map(move |&(dim, _)| {
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
        memorylimits::MemVec,
        scheduling::{moves::Move, Action, ActionT, ApplyError, NotApplicableReason},
        shape, spec,
        spec::{arb_canonical_spec, LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec},
        target::{Avx2Target, Target},
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
            assert_unique_layouts(&Avx2Target::all_layouts_for_shape(&shape, dtype));
        }

        #[test]
        fn test_all_layouts_for_shape_contains_move_destination_layouts(
            shape in proptest::collection::vec(1..8u32, 1..=5),
            dtype in any::<Dtype>(),
        ) {
            let shape = shape.into_iter().map(|x| DimSize::new(x).unwrap()).collect::<Vec<_>>();
            let superset: HashSet<_> = Avx2Target::all_layouts_for_shape(&shape, dtype)
                .into_iter()
                .collect();
            let subset: HashSet<_> = Avx2Target::move_destination_layouts(&shape, dtype)
                .into_iter()
                .collect();
            let diff: Vec<_> = subset.difference(&superset).collect();
            assert!(
                diff.is_empty(),
                "move layouts contained {diff:?} not in all_layouts_for_shape: {superset:?}",
            );
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
        //     for layout in Avx2Target::move_destination_layouts(&shape, dtype) {
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
            spec in arb_canonical_spec::<Avx2Target>(None, None),
        ) {
            let lspec = &spec.0;
            let mut seen_source_idxs = HashSet::new();
            for action in Avx2Target::actions(lspec) {
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
                        let dest_layouts = Avx2Target::move_destination_layouts(
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
        let logical_spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 1, 2, 16],
            (u8, CpuMemoryLevel::L1, row_major),
            (i8, CpuMemoryLevel::VRF, batched_col_major, 32),
            (i16, CpuMemoryLevel::VRF, row_major, 16),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(CpuKernel::TwoVecBroadcastVecMultAddU8S8S16.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_dotproductloop_applies_to_vrf_lhs_strip() {
        let logical_spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 1, 8, 1],
            (f32, CpuMemoryLevel::VRF, row_major, 8),
            (f32, CpuMemoryLevel::L1, col_major),
            (f32, CpuMemoryLevel::RF, row_major),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(CpuKernel::DotProductLoop.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_dotproductloop_applies_to_longer_k() {
        let logical_spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 1, 32, 1],
            (f32, CpuMemoryLevel::VRF, row_major, 8),
            (f32, CpuMemoryLevel::L1, col_major),
            (f32, CpuMemoryLevel::RF, row_major),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(CpuKernel::DotProductLoop.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_dotproductloop_rejects_wrong_vector_size() {
        let logical_spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 1, 8, 1],
            (f32, CpuMemoryLevel::VRF, row_major, 4),
            (f32, CpuMemoryLevel::L1, col_major),
            (f32, CpuMemoryLevel::RF, row_major),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(!CpuKernel::DotProductLoop.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_all_layouts_for_bf16_includes_interleaved_dim2_oddeven16() {
        let shape = vec![nz!(1u32), nz!(1u32), nz!(2048u32)];
        let want = layout![2, 2 oe(16)];
        let layouts = Avx2Target::all_layouts_for_shape(&shape, Dtype::Bfloat16);
        assert!(
            layouts.contains(&want),
            "Expected layouts to include {want:?} for shape {shape:?}, but was: {layouts:?}",
        );
    }

    #[test]
    fn test_oddeven_layouts_for_standard_layout_includes_dim2_oddeven16_f32() {
        let layouts = oddeven_layouts_for_standard_layout(
            &row_major(&shape![1, 1, 16]),
            &shape![1, 1, 16],
            Dtype::Float32,
            &[16u32, 32u32],
        )
        .collect::<Vec<_>>();
        let expect = layout![2, 2 oe(16)];
        assert!(
            layouts.contains(&expect),
            "{layouts:?} did not contain {expect:?}"
        );
    }

    #[test]
    fn test_kernel_memory_constrains_placement() {
        let spec: Spec<Avx2Target> = spec!(
            MatmulAccum(
                [1, 1, 1, 16],
                (u8, CpuMemoryLevel::RF, row_major),
                (u8, CpuMemoryLevel::VRF, row_major, 16),
                (u8, CpuMemoryLevel::VRF, row_major, 16),
                serial
            ),
            MemoryLimits::Standard(MemVec::zero::<Avx2Target>())
        );
        assert!(spec.is_canonical(), "{spec:?} not canonical");

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
    fn test_broadcastvecmultadd_rhs_broadcast_param_order() {
        let logical_spec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 8, 1, 1],
            (f32, CpuMemoryLevel::VRF, row_major, 8),
            (f32, CpuMemoryLevel::L1, row_major),
            (f32, CpuMemoryLevel::VRF, row_major, 8),
            serial
        ));
        assert!(logical_spec.is_canonical());
        assert!(CpuKernel::BroadcastVecMultAdd.applies_to_logical_spec(&logical_spec));
    }

    #[test]
    fn test_assign_main_cost() {
        let arg0 = TensorSpec::new_canon(
            shape![2, 32],
            Dtype::Uint32,
            CpuMemoryLevel::VRF,
            row_major,
            Some(nz!(8u32)),
        );
        let arg1 = TensorSpec::new_canon(
            shape![2, 32],
            Dtype::Uint32,
            CpuMemoryLevel::L1,
            row_major,
            None,
        );
        let parameter0: Param<Avx2Target> = Param::new(0, arg0);
        let parameter1: Param<Avx2Target> = Param::new(1, arg1);
        let cost = CpuKernel::Assign.main_cost(&[parameter0, parameter1]);
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
