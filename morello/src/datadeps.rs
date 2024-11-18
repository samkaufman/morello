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
        reduction_dim: u8,
        dtype: Dtype,
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
            SpecKey::Matmul { dtypes } | SpecKey::Conv { dtypes } => dtypes,
            SpecKey::Move { dtypes } => dtypes,
            SpecKey::Softmax { dtype, .. } | SpecKey::Zero { dtype } => slice::from_ref(dtype),
            SpecKey::Compose { components: _ } => todo!(),
        }
    }
}
