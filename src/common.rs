use serde::{Deserialize, Serialize};
use std::fmt::Display;

pub type DimSize = u32; // TODO: Switch to NonZeroU32.
pub type Shape = smallvec::SmallVec<[DimSize; 5]>;
pub type Contig = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub enum Dtype {
    Uint8,
    Sint8,
    Uint16,
    Sint16,
    Uint32,
    Sint32,
}

impl Dtype {
    /// The bytes required to represent a value of this Dtype.
    pub fn size(&self) -> u8 {
        match self {
            Dtype::Uint8 | Dtype::Sint8 => 1,
            Dtype::Uint16 | Dtype::Sint16 => 2,
            Dtype::Uint32 | Dtype::Sint32 => 4,
        }
    }

    pub fn c_type(&self) -> &'static str {
        match self {
            Dtype::Uint8 => "uint8_t",
            Dtype::Sint8 => "int8_t",
            Dtype::Uint16 => "uint16_t",
            Dtype::Sint16 => "int16_t",
            Dtype::Uint32 => "uint32_t",
            Dtype::Sint32 => "int32_t",
        }
    }

    pub fn int_fmt_macro(&self) -> &'static str {
        match self {
            Dtype::Uint8 => "PRIu8",
            Dtype::Sint8 => "PRIi8",
            Dtype::Uint16 => "PRIu16",
            Dtype::Sint16 => "PRIi16",
            Dtype::Uint32 => "PRIu32",
            Dtype::Sint32 => "PRIi32",
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Uint8 => write!(f, "u8"),
            Dtype::Sint8 => write!(f, "s8"),
            Dtype::Uint16 => write!(f, "u16"),
            Dtype::Sint16 => write!(f, "s16"),
            Dtype::Uint32 => write!(f, "u32"),
            Dtype::Sint32 => write!(f, "s32"),
        }
    }
}
