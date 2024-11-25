use crate::{common::Dtype, spec::PrimitiveSpecType};
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
    Softmax {
        scan_dim: u8,
        dtype: Dtype,
    },
    SoftmaxComplete {
        scan_dim: u8,
        dtype: Dtype,
    },
    SoftmaxDenominatorAndMax {
        scan_dim: u8,
        dtype: Dtype,
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
    Zero {
        dtype: Dtype,
    },
    Compose {
        components: Vec<(PrimitiveSpecType, Vec<Dtype>)>,
    },
}

impl SpecKey {
    pub fn dtypes(&self) -> &[Dtype] {
        match self {
            SpecKey::Matmul { dtypes }
            | SpecKey::Conv { dtypes }
            | SpecKey::SoftmaxDenominator { dtypes, .. } => dtypes,
            SpecKey::Move { dtypes } | SpecKey::Max { dtypes, .. } => dtypes,
            SpecKey::Softmax { dtype, .. }
            | SpecKey::SoftmaxComplete { dtype, .. }
            | SpecKey::SoftmaxDenominatorAndMax { dtype, .. }
            | SpecKey::Zero { dtype } => slice::from_ref(dtype),
            SpecKey::Compose { components: _ } => todo!(),
        }
    }
}
