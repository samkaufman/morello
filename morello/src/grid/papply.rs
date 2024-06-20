use std::iter::FusedIterator;

use super::general::SurMap;

/// Partially apply a tuple [SurMap], fixing the second component.
pub struct PApplyRhs<S, V>(pub S, pub V);

pub struct PApplyRhsDomainIter<T>(T);

impl<S, A, V> SurMap for PApplyRhs<S, V>
where
    S: SurMap<Domain = (A, V)>,
    A: Clone,
    V: Clone,
{
    type Domain = A;
    type Codomain = S::Codomain;
    type DomainIter = PApplyRhsDomainIter<S::DomainIter>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        self.0.apply(&(t.clone(), self.1.clone()))
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        PApplyRhsDomainIter(self.0.apply_inverse(i))
    }
}

impl<T: Iterator<Item = (A, V)>, A, V> Iterator for PApplyRhsDomainIter<T> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(a, _)| a)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

// PApplyRhsDomainIter is a FusedIterator if its underlying iterator is a FusedIterator.
impl<T: FusedIterator<Item = (A, V)>, A, V> FusedIterator for PApplyRhsDomainIter<T> {}
