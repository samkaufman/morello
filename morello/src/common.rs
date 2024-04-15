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
    Float32,
    Bfloat16,
}

impl Dtype {
    /// The bytes required to represent a value of this Dtype.
    pub fn size(&self) -> u8 {
        match self {
            Dtype::Uint8 | Dtype::Sint8 => 1,
            Dtype::Uint16 | Dtype::Sint16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Sint32 | Dtype::Float32 => 4,
        }
    }

    pub fn higher_precision_types(&self) -> &[Dtype] {
        match self {
            // TODO: Enable the following once we have a more principled way of
            //   pruning useless casts.
            // Dtype::Uint8 => &[Dtype::Uint16, Dtype::Uint32],
            // Dtype::Sint8 => &[Dtype::Sint16, Dtype::Sint32],
            // Dtype::Uint16 => &[Dtype::Uint32],
            // Dtype::Sint16 => &[Dtype::Sint32],
            Dtype::Bfloat16 => &[Dtype::Float32],
            _ => &[],
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Uint8 => write!(f, "u8"),
            Dtype::Sint8 => write!(f, "i8"),
            Dtype::Uint16 => write!(f, "u16"),
            Dtype::Sint16 => write!(f, "i16"),
            Dtype::Uint32 => write!(f, "u32"),
            Dtype::Sint32 => write!(f, "i32"),
            Dtype::Float32 => write!(f, "f32"),
            Dtype::Bfloat16 => write!(f, "bf16"),
        }
    }
}
