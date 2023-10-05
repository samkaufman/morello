use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDArray<T> {
    pub data: Vec<T>,
    // TODO: Not necessary to store shape or strides, which can be stored externally in database.
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Default + Clone> NDArray<T> {
    pub fn new(shape: &[usize]) -> Self {
        let strides = calculate_strides(shape);
        let volume = shape.iter().product();
        Self {
            data: vec![T::default(); volume],
            shape: shape.to_vec(),
            strides,
        }
    }
}

impl<T> NDArray<T> {
    fn index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "Number of indices must match the number of dimensions."
        );
        indices
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<&[usize]> for NDArray<T> {
    type Output = T;

    fn index(&self, pt: &[usize]) -> &Self::Output {
        &self.data[self.index(pt)]
    }
}

impl<T> IndexMut<&[usize]> for NDArray<T> {
    fn index_mut(&mut self, pt: &[usize]) -> &mut Self::Output {
        let index = self.index(pt);
        &mut self.data[index]
    }
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::new();
    let mut stride = 1;
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();
    strides
}
