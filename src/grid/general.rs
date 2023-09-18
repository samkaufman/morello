use crate::grid::linear::BimapInt;

pub trait SurMap {
    type Domain;
    type Codomain;
    type DomainIter: Iterator<Item = Self::Domain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;
    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter;
}

pub trait BiMap {
    type Domain;
    type Codomain;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;
    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain;
}

pub trait IntBiMap: SurMap {
    type Domain;

    fn apply(&self, t: &<Self as IntBiMap>::Domain) -> BimapInt;
    fn apply_inverse(&self, i: BimapInt) -> <Self as IntBiMap>::Domain;
}

impl<T: BiMap> SurMap for T {
    type Domain = <T as BiMap>::Domain;
    type Codomain = <T as BiMap>::Codomain;
    type DomainIter = std::iter::Once<Self::Domain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        <Self as BiMap>::apply(self, t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        std::iter::once(<Self as BiMap>::apply_inverse(self, i))
    }
}

impl<T: IntBiMap> BiMap for T {
    type Domain = <T as IntBiMap>::Domain;
    type Codomain = BimapInt;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        <Self as IntBiMap>::apply(self, t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        <Self as IntBiMap>::apply_inverse(self, *i)
    }
}
