use std::marker::PhantomData;

use super::general::SurMap;

pub struct Permute<T>(Vec<usize>, PhantomData<T>);

impl<T> Permute<T> {
    pub fn new(indices: Vec<usize>) -> Self {
        // TODO: Check that these are unique indices.
        Self(indices, PhantomData)
    }
}

impl<T: Default + Clone> SurMap for Permute<T> {
    type Domain = Vec<T>;
    type Codomain = Vec<T>;
    type DomainIter = [Vec<T>; 1];

    fn apply(&self, t: &Vec<T>) -> Self::Codomain {
        self.0.iter().map(|&index| t[index].clone()).collect()
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let mut reversed = vec![T::default(); i.len()];
        for idx in 0..i.len() {
            reversed[self.0[idx]] = i[idx].clone();
        }
        [reversed]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::general::BiMapExt;

    #[test]
    fn test_permute_forward() {
        let permute = Permute::new(vec![1, 0, 2]);
        let input = vec!['a', 'b', 'c'];
        let output = permute.apply(&input);
        assert_eq!(output, vec!['b', 'a', 'c']);
    }

    #[test]
    fn test_permute_backward() {
        let permute = Permute::new(vec![1, 0, 2]);
        let shuffled = vec!['b', 'a', 'c'];
        assert_eq!(permute.invert(&shuffled), ['a', 'b', 'c']);
    }
}
