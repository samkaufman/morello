use serde::{Deserialize, Serialize};
use std::{hash::Hash, slice};

use crate::common::Dtype;

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum SpecKey {
    Matmul { dtypes: [Dtype; 3] },
    Conv { dtypes: [Dtype; 3] },
    Move { dtypes: [Dtype; 2] },
    Zero { dtype: Dtype },
}

impl SpecKey {
    pub fn dtypes(&self) -> &[Dtype] {
        match self {
            SpecKey::Matmul { dtypes } | SpecKey::Conv { dtypes } => dtypes,
            SpecKey::Move { dtypes } => dtypes,
            SpecKey::Zero { dtype } => slice::from_ref(dtype),
        }
    }
}
