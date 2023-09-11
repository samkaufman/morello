use super::general::Bimap;
use super::linear::BoolBimap;

pub trait CanonicalBimap {
    type Bimap: Bimap<Domain = Self>;
    fn bimap() -> Self::Bimap;
}

impl CanonicalBimap for bool {
    type Bimap = BoolBimap;

    fn bimap() -> Self::Bimap {
        BoolBimap
    }
}
