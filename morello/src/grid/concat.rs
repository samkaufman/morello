use std::marker::PhantomData;

use super::general::SurMap;

pub struct ConcatFixedRight<T>(pub usize, PhantomData<T>);

impl<T> ConcatFixedRight<T> {
    pub fn new(n: usize) -> Self {
        ConcatFixedRight(n, PhantomData)
    }
}

impl<T: Clone> SurMap for ConcatFixedRight<T> {
    type Domain = (Vec<T>, Vec<T>);
    type Codomain = Vec<T>;
    type DomainIter = [Self::Domain; 1];

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        debug_assert_eq!(t.1.len(), self.0);
        // TODO: It would be better to have a signature that doesn't require clones.
        t.0.iter().chain(t.1.iter()).cloned().collect()
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        debug_assert!(i.len() >= self.0);
        [(i[..self.0].to_vec(), i[self.0..].to_vec())]
    }
}
