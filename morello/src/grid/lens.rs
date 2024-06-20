use std::marker::PhantomData;

use super::general::SurMap;

pub struct LensRhs<S, T>(pub S, pub PhantomData<T>);

impl<S, T> SurMap for LensRhs<S, T>
where
    S: SurMap,
    S::DomainIter: Send + 'static,
    T: Clone + Send + 'static,
{
    type Domain = (T, S::Domain);
    type Codomain = (T, S::Codomain);
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + Send>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (t.0.clone(), self.0.apply(&t.1))
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let lhs = i.0.clone();
        Box::new(self.0.apply_inverse(&i.1).map(move |u| (lhs.clone(), u)))
    }
}
