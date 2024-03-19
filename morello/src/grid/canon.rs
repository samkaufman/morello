use super::general::BiMap;
use super::linear::BoolBimap;

pub trait CanonicalBimap {
    type Bimap: BiMap<Domain = Self>;
    fn bimap() -> Self::Bimap;
}

impl CanonicalBimap for bool {
    type Bimap = BoolBimap;

    fn bimap() -> Self::Bimap {
        BoolBimap
    }
}
