use crate::{
    common::Dtype,
    spec::{FillValue, PrimitiveSpecType},
};
use serde::{Deserialize, Serialize};
use std::{hash::Hash, slice};

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum SpecKey {
    Matmul {
        dtypes: [Dtype; 3],
    },
    Conv {
        dtypes: [Dtype; 3],
    },
    DivideVec {
        dtypes: [Dtype; 3],
    },
    DivideVecScalar {
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    Softmax {
        scan_dim: u8,
        dtypes: [Dtype; 2],
    },
    SoftmaxComplete {
        scan_dim: u8,
        dtypes: [Dtype; 4],
    },
    SoftmaxDenominatorAndMax {
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    SoftmaxDenominatorAndUnscaled {
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    SoftmaxDenominatorAndUnscaledFromMax {
        scan_dim: u8,
        dtypes: [Dtype; 4],
    },
    SoftmaxDenominator {
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    Max {
        dim: u8,
        dtypes: [Dtype; 2],
    },
    Move {
        dtypes: [Dtype; 2],
    },
    Fill {
        value: FillValue,
        dtype: Dtype,
    },
    Compose {
        components: Vec<(PrimitiveSpecType, Vec<Dtype>, u8)>,
    },
}

impl SpecKey {
    pub(crate) fn dtypes(&self) -> &[Dtype] {
        match self {
            SpecKey::Matmul { dtypes } => dtypes,
            SpecKey::Conv { dtypes } => dtypes,
            SpecKey::Move { dtypes } => dtypes,
            SpecKey::Max { dtypes, .. } => dtypes,
            SpecKey::DivideVec { dtypes } => dtypes,
            SpecKey::DivideVecScalar { dtypes, .. } => dtypes,
            SpecKey::Softmax { dtypes, .. } => dtypes,
            SpecKey::SoftmaxComplete { dtypes, .. } => dtypes,
            SpecKey::SoftmaxDenominatorAndMax { dtypes, .. } => dtypes,
            SpecKey::SoftmaxDenominatorAndUnscaled { dtypes, .. } => dtypes,
            SpecKey::SoftmaxDenominatorAndUnscaledFromMax { dtypes, .. } => dtypes,
            SpecKey::SoftmaxDenominator { dtypes, .. } => dtypes,
            SpecKey::Fill { dtype, value: _ } => slice::from_ref(dtype),
            SpecKey::Compose { .. } => unimplemented!(),
        }
    }
}
