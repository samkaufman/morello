use std::{iter::FusedIterator, marker::PhantomData};

use super::general::SurMap;

// TODO: Tons of duplication in this file. Merge with a macro.

#[derive(Clone)]
pub struct LensLhs<S, T>(pub S, PhantomData<T>);

#[derive(Clone)]
pub struct LensRhs<S, T>(pub S, PhantomData<T>);

pub struct LensLhsDomainIter<L, R> {
    lhs_iter: L,
    rhs: R,
}

pub struct LensRhsDomainIter<L, R> {
    lhs: L,
    rhs_iter: R,
}

impl<S, T> LensLhs<S, T> {
    pub fn new(s: S) -> Self {
        Self(s, PhantomData)
    }
}

impl<S, T> LensRhs<S, T> {
    pub fn new(s: S) -> Self {
        Self(s, PhantomData)
    }
}

impl<S, T> SurMap for LensLhs<S, T>
where
    S: SurMap,
    S::DomainIter: Send + 'static,
    T: Clone + Send + 'static,
{
    type Domain = (S::Domain, T);
    type Codomain = (S::Codomain, T);
    type DomainIter = LensLhsDomainIter<<S::DomainIter as IntoIterator>::IntoIter, T>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (self.0.apply(&t.0), t.1.clone())
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let rhs = i.1.clone();
        LensLhsDomainIter {
            lhs_iter: self.0.apply_inverse(&i.0).into_iter(),
            rhs,
        }
    }
}

impl<S, T> SurMap for LensRhs<S, T>
where
    S: SurMap,
    S::DomainIter: Send + 'static,
    T: Clone + Send + 'static,
{
    type Domain = (T, S::Domain);
    type Codomain = (T, S::Codomain);
    type DomainIter = LensRhsDomainIter<T, <S::DomainIter as IntoIterator>::IntoIter>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (t.0.clone(), self.0.apply(&t.1))
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let lhs = i.0.clone();
        LensRhsDomainIter {
            lhs,
            rhs_iter: self.0.apply_inverse(&i.1).into_iter(),
        }
    }
}

impl<L: Iterator, R: Clone> Iterator for LensLhsDomainIter<L, R> {
    type Item = (L::Item, R);

    fn next(&mut self) -> Option<Self::Item> {
        self.lhs_iter.next().map(|lhs| (lhs, self.rhs.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lhs_iter.size_hint()
    }
}

impl<L: Clone, R: Iterator> Iterator for LensRhsDomainIter<L, R> {
    type Item = (L, R::Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.rhs_iter.next().map(|rhs| (self.lhs.clone(), rhs))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.rhs_iter.size_hint()
    }
}

impl<L: FusedIterator, R: Clone> FusedIterator for LensLhsDomainIter<L, R> {}

impl<L: Clone, R: FusedIterator> FusedIterator for LensRhsDomainIter<L, R> {}
