use serde::{Deserialize, Serialize};
use std::hash::Hash;

use crate::common::Dtype;

// TODO: Simplify code by making this the foundation of our Spec enum.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum SpecKey {
    Matmul { dtype: Dtype },
    Conv { dtype: Dtype },
    Move { dtype: Dtype },
    Zero { dtype: Dtype },
}

impl SpecKey {
    pub fn dtype(&self) -> Dtype {
        match self {
            SpecKey::Matmul { dtype }
            | SpecKey::Conv { dtype }
            | SpecKey::Move { dtype }
            | SpecKey::Zero { dtype } => *dtype,
        }
    }
}
