use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::num::NonZeroU32;

use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::tablemeta::{DimensionType, TableMeta};

pub type DimSize = NonZeroU32;
pub type Shape = Vec<DimSize>;
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

pub struct DtypeBimap;

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

impl BiMap for DtypeBimap {
    type Domain = Dtype;
    type Codomain = u8;

    fn apply(&self, dtype: &Self::Domain) -> Self::Codomain {
        match dtype {
            Dtype::Uint8 => 0,
            Dtype::Sint8 => 1,
            Dtype::Uint16 => 2,
            Dtype::Sint16 => 3,
            Dtype::Uint32 => 4,
            Dtype::Sint32 => 5,
            Dtype::Float32 => 6,
            Dtype::Bfloat16 => 7,
        }
    }

    fn apply_inverse(&self, v: &Self::Codomain) -> Self::Domain {
        match *v {
            0 => Dtype::Uint8,
            1 => Dtype::Sint8,
            2 => Dtype::Uint16,
            3 => Dtype::Sint16,
            4 => Dtype::Uint32,
            5 => Dtype::Sint32,
            6 => Dtype::Float32,
            7 => Dtype::Bfloat16,
            _ => panic!(),
        }
    }
}

impl TableMeta for DtypeBimap {
    fn dimension_types(&self, _: &Self::Domain) -> Vec<DimensionType> {
        vec![DimensionType::Dtype]
    }
}

impl CanonicalBimap for Dtype {
    type Bimap = DtypeBimap;

    fn bimap() -> Self::Bimap {
        DtypeBimap
    }
}
