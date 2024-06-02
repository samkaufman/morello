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

pub trait AsBimap: SurMap {
    type Wrapper: BiMap<Domain = Self::Domain, Codomain = Self::Codomain>;
    fn into_bimap(self) -> Self::Wrapper;
}

pub struct BimapEnforcer<T: SurMap>(T);

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

impl<T: SurMap> AsBimap for T {
    type Wrapper = BimapEnforcer<T>;

    fn into_bimap(self) -> Self::Wrapper {
        BimapEnforcer(self)
    }
}

impl<T: SurMap> BiMap for BimapEnforcer<T> {
    type Domain = <T as SurMap>::Domain;
    type Codomain = <T as SurMap>::Codomain;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        <T as SurMap>::apply(&self.0, t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        let mut i = <T as SurMap>::apply_inverse(&self.0, i);
        let result = i.next().unwrap();
        debug_assert!(i.next().is_none());
        result
    }
}

/// Tuples of [BiMap]s are [BiMap]s.
impl<T: BiMap, U: BiMap> BiMap for (T, U) {
    type Domain = (<T as BiMap>::Domain, <U as BiMap>::Domain);
    type Codomain = (<T as BiMap>::Codomain, <U as BiMap>::Codomain);

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (
            <T as BiMap>::apply(&self.0, &t.0),
            <U as BiMap>::apply(&self.1, &t.1),
        )
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain {
        (
            <T as BiMap>::apply_inverse(&self.0, &i.0),
            <U as BiMap>::apply_inverse(&self.1, &i.1),
        )
    }
}

// TODO: Add implementations for 3- and larger tuples with some kind of macro.
