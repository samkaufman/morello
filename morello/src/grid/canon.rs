use super::general::{IntoSingleItemIter, SurMap};
use super::linear::BoolBimap;

pub trait CanonicalBimap {
    type Bimap: SurMap<Domain = Self, DomainIter: IntoSingleItemIter>;
    fn bimap() -> Self::Bimap;
}

impl CanonicalBimap for bool {
    type Bimap = BoolBimap;

    fn bimap() -> Self::Bimap {
        BoolBimap
    }
}
