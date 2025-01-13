use crate::{
    common::Dtype,
    spec::{FillValue, PrimitiveSpecType},
};
use serde::{Deserialize, Serialize};
use std::hash::Hash;

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum SpecKey {
    OnePrefix {
        dtype: Dtype,
    },
    Matmul {
        dtypes: [Dtype; 3],
    },
    Conv {
        dtypes: [Dtype; 3],
    },
    Broadcast {
        dim: u8,
        dtypes: [Dtype; 2],
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
    pub(crate) fn dtypes(&self) -> Box<dyn Iterator<Item = Dtype> + '_> {
        match self {
            SpecKey::Matmul { dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Conv { dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Broadcast { dim: _, dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Move { dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Max { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::DivideVec { dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::DivideVecScalar { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::Softmax { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxComplete { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominatorAndMax { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominatorAndUnscaled { dtypes, .. } => {
                Box::new(dtypes.iter().copied())
            }
            SpecKey::SoftmaxDenominatorAndUnscaledFromMax { dtypes, .. } => {
                Box::new(dtypes.iter().copied())
            }
            SpecKey::SoftmaxDenominator { dtypes, .. } => Box::new(dtypes.iter().copied()),
            SpecKey::OnePrefix { dtype } => Box::new([*dtype, *dtype].into_iter()),
            SpecKey::Fill { dtype, value: _ } => Box::new(std::iter::once(*dtype)),
            SpecKey::Compose { .. } => unimplemented!(),
        }
    }
}
