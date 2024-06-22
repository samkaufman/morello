use std::iter;

use auto_impl::auto_impl;
use itertools::Itertools;

use super::tablemeta::{DimensionType, TableMeta};

#[auto_impl(&, &mut, Box, Arc, Rc)]
pub trait SurMap {
    type Domain;
    type Codomain;
    type DomainIter: IntoIterator<Item = Self::Domain>; // TODO: Rename Preimage

    fn apply(&self, t: &Self::Domain) -> Self::Codomain;

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter; // TODO: Rename: preimage
}

pub trait SurMapExt: SurMap {
    // TODO: This can lead to needlessly nested IntoBimaps.
    fn into_bimap(self) -> IntoBimap<Self>
    where
        Self: Sized;
}

// TODO: Rename back to BiMap
pub trait BiMapExt: SurMap {
    fn invert(&self, i: &Self::Codomain) -> Self::Domain;
}

pub trait IntoSingleItemIter: IntoIterator {}

#[derive(Clone)]
pub struct IntoBimap<T>(T);

impl<T: SurMap> SurMapExt for T {
    fn into_bimap(self) -> IntoBimap<Self>
    where
        Self: Sized,
    {
        IntoBimap(self)
    }
}

impl<T> BiMapExt for T
where
    T: SurMap,
    T::DomainIter: IntoSingleItemIter,
{
    fn invert(&self, i: &Self::Codomain) -> Self::Domain {
        self.apply_inverse(i).into_iter().next().unwrap()
    }
}

/// Tuples of [SurMap]s are [SurMap]s. Inversion takes the Cartesian product.
impl<T, U> SurMap for (T, U)
where
    T: SurMap,
    U: SurMap,
    T::Domain: Clone,
    <U::DomainIter as IntoIterator>::IntoIter: Clone,
{
    type Domain = (<T as SurMap>::Domain, <U as SurMap>::Domain);
    type Codomain = (<T as SurMap>::Codomain, <U as SurMap>::Codomain);
    type DomainIter = itertools::Product<
        <T::DomainIter as IntoIterator>::IntoIter,
        <U::DomainIter as IntoIterator>::IntoIter,
    >;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (
            <T as SurMap>::apply(&self.0, &t.0),
            <U as SurMap>::apply(&self.1, &t.1),
        )
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        <T as SurMap>::apply_inverse(&self.0, &i.0)
            .into_iter()
            .cartesian_product(self.1.apply_inverse(&i.1))
    }
}

// TODO: Add implementations for 3- and larger tuples with some kind of macro.

impl<T: SurMap> SurMap for IntoBimap<T> {
    type Domain = T::Domain;
    type Codomain = T::Codomain;
    type DomainIter = [T::Domain; 1];

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        self.0.apply(t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let Ok(single_item) = self.0.apply_inverse(i).into_iter().exactly_one() else {
            panic!("preimage did not contain a single item");
        };
        [single_item; 1]
    }
}

impl<T: TableMeta> TableMeta for IntoBimap<T> {
    fn dimension_types(&self, input: &Self::Domain) -> Vec<DimensionType> {
        self.0.dimension_types(input)
    }
}

impl<T> IntoSingleItemIter for iter::Once<T> {}
impl<T> IntoSingleItemIter for [T; 1] {}
