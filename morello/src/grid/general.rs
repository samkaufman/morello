use itertools::Itertools;
use std::iter;

use crate::grid::linear::BimapInt;
pub trait SurMap {
    type Domain;
    type Codomain;
    type DomainIter: Iterator<Item = Self::Domain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;
    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter;

    /// Returns whether this SurMap is defined for the given domain value.
    ///
    /// The default implementation returns true for all inputs.
    fn defined_for(&self, _t: &Self::Domain) -> bool {
        true
    }
}

pub trait BiMap {
    type Domain;
    type Codomain;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;
    fn apply_inverse(&self, i: &Self::Codomain) -> Self::Domain;

    /// Returns whether this BiMap is defined for the given domain value.
    ///
    /// The default implementation returns true for all inputs.
    fn defined_for(&self, _t: &Self::Domain) -> bool {
        true
    }
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
    type DomainIter = iter::Once<Self::Domain>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        <Self as BiMap>::apply(self, t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        iter::once(<Self as BiMap>::apply_inverse(self, i))
    }

    fn defined_for(&self, t: &Self::Domain) -> bool {
        <Self as BiMap>::defined_for(self, t)
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

    fn defined_for(&self, t: &Self::Domain) -> bool {
        <T as SurMap>::defined_for(&self.0, t)
    }
}

/// Tuples of [SurMap]s are [SurMap]s. Inversion takes the Cartesian product.
impl<T, U> SurMap for (T, U)
where
    T: SurMap,
    U: SurMap,
    T::Domain: Clone,
    U::DomainIter: Clone,
{
    type Domain = (<T as SurMap>::Domain, <U as SurMap>::Domain);
    type Codomain = (<T as SurMap>::Codomain, <U as SurMap>::Codomain);
    type DomainIter = itertools::Product<<T as SurMap>::DomainIter, <U as SurMap>::DomainIter>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (
            <T as SurMap>::apply(&self.0, &t.0),
            <U as SurMap>::apply(&self.1, &t.1),
        )
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        <T as SurMap>::apply_inverse(&self.0, &i.0)
            .cartesian_product(<U as SurMap>::apply_inverse(&self.1, &i.1))
    }
}

// TODO: Add implementations for 3- and larger tuples with some kind of macro.
