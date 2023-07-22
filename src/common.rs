use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::{memorylimits::MemoryLimits, spec::LogicalSpec, target::Target};

pub type DimSize = u32;
pub type Shape = smallvec::SmallVec<[DimSize; 5]>;
pub type Contig = u8;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct Spec<Tgt: Target>(pub LogicalSpec<Tgt>, pub MemoryLimits);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum Dtype {
    Uint8,
    Uint32,
}

impl<Tgt: Target> Display for Spec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {:?})", self.0, self.1)
    }
}

impl Dtype {
    /// The bytes required to represent a value of this Dtype.
    pub fn size(&self) -> u8 {
        match self {
            Dtype::Uint8 => 1,
            Dtype::Uint32 => 4,
        }
    }

    pub fn c_type(&self) -> &'static str {
        match self {
            Dtype::Uint8 => "uint8_t",
            Dtype::Uint32 => "uint32_t",
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Uint8 => write!(f, "u8"),
            Dtype::Uint32 => write!(f, "u32"),
        }
    }
}
