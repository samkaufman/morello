use serde::{Deserialize, Serialize};
use std::fmt::Display;

pub type DimSize = u32; // TODO: Switch to NonZeroU32.
pub type Shape = smallvec::SmallVec<[DimSize; 5]>;
pub type Contig = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum Dtype {
    Uint8,
    Uint32,
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

    pub fn int_fmt_macro(&self) -> &'static str {
        match self {
            Dtype::Uint8 => "PRIu8",
            Dtype::Uint32 => "PRIu32",
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
