use super::general::SurMap;
use std::iter::FusedIterator;

pub struct Compose<A, B>(pub A, pub B);

pub enum ComposeDomainIter<A, B>
where
    A: SurMap<Codomain = B::Domain>,
    B: SurMap,
{
    Active {
        intermediate_map: A, // TODO: Take ref. instead of clone
        inner_iter: B::DomainIter,
        outer_iter: A::DomainIter,
    },
    Done,
}

impl<A, B> SurMap for Compose<A, B>
where
    A: SurMap<Codomain = B::Domain> + Clone,
    B: SurMap,
{
    type Domain = A::Domain;
    type Codomain = B::Codomain;
    type DomainIter = ComposeDomainIter<A, B>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        self.1.apply(&self.0.apply(t))
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        ComposeDomainIter::new(self, self.0.clone(), &self.1, i)
    }
}

impl<A, B> ComposeDomainIter<A, B>
where
    A: SurMap<Codomain = B::Domain> + Clone,
    B: SurMap,
{
    fn new(compose: &Compose<A, B>, a: A, b: &B, initial_co: &B::Codomain) -> Self {
        let mut inner_iter = b.apply_inverse(initial_co);
        match inner_iter.next() {
            Some(first_intermediate_value) => {
                let outer_iter = a.apply_inverse(&first_intermediate_value);
                ComposeDomainIter::Active {
                    intermediate_map: compose.0.clone(),
                    inner_iter,
                    outer_iter,
                }
            }
            None => ComposeDomainIter::Done,
        }
    }
}

impl<A, B> Iterator for ComposeDomainIter<A, B>
where
    A: SurMap<Codomain = B::Domain>,
    B: SurMap,
{
    type Item = A::Domain;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            ComposeDomainIter::Active {
                intermediate_map,
                inner_iter,
                outer_iter,
            } => loop {
                if let Some(outer_value) = outer_iter.next() {
                    return Some(outer_value);
                }

                match inner_iter.next() {
                    Some(next_inner_value) => {
                        *outer_iter = intermediate_map.apply_inverse(&next_inner_value);
                    }
                    None => {
                        *self = ComposeDomainIter::Done;
                        return None;
                    }
                }
            },
            ComposeDomainIter::Done => None,
        }
    }
}

impl<A, B> FusedIterator for ComposeDomainIter<A, B>
where
    A: SurMap<Codomain = B::Domain>,
    B: SurMap,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::general::SurMap;

    #[derive(Clone)]
    struct Halve;

    impl SurMap for Halve {
        type Domain = u32;
        type Codomain = u32;
        type DomainIter = std::array::IntoIter<u32, 2>;

        fn apply(&self, t: &Self::Domain) -> Self::Codomain {
            *t / 2
        }

        fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
            [2 * i, 2 * i + 1].into_iter()
        }
    }

    #[test]
    fn test_composed_halves() {
        let halve_twice = Compose(Halve, Halve);
        assert_eq!(halve_twice.apply(&8), 2);
        assert_eq!(
            halve_twice.apply_inverse(&2).collect::<Vec<_>>(),
            vec![8, 9, 10, 11]
        );
    }
}
