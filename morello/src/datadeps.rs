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
        rank: u8,
        dtypes: [Dtype; 2],
    },
    Matmul {
        dtypes: [Dtype; 3],
    },
    Conv {
        dtypes: [Dtype; 3],
    },
    Broadcast {
        rank: u8,
        dim: u8,
        dtypes: [Dtype; 2],
    },
    DivideVec {
        rank: u8,
        dtypes: [Dtype; 3],
    },
    DivideVecScalar {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    Softmax {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 2],
    },
    SoftmaxComplete {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 4],
    },
    SoftmaxDenominatorAndMax {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    SoftmaxDenominatorAndUnscaled {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    SoftmaxDenominatorAndUnscaledFromMax {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 4],
    },
    SoftmaxDenominator {
        rank: u8,
        scan_dim: u8,
        dtypes: [Dtype; 3],
    },
    Max {
        rank: u8,
        dim: u8,
        dtypes: [Dtype; 2],
    },
    Move {
        rank: u8,
        dtypes: [Dtype; 2],
    },
    Fill {
        rank: u8,
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
            SpecKey::Broadcast {
                rank: _,
                dim: _,
                dtypes,
            } => Box::new(dtypes.iter().copied()),
            SpecKey::Move { rank: _, dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Max {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::DivideVec { rank: _, dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::DivideVecScalar {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::Softmax {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxComplete {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominatorAndMax {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominatorAndUnscaled {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::SoftmaxDenominator {
                rank: _, dtypes, ..
            } => Box::new(dtypes.iter().copied()),
            SpecKey::OnePrefix { rank: _, dtypes } => Box::new(dtypes.iter().copied()),
            SpecKey::Fill {
                rank: _,
                dtype,
                value: _,
            } => Box::new(std::iter::once(*dtype)),
            SpecKey::Compose { .. } => unimplemented!(),
        }
    }
}
