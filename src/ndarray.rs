use divrem::DivRem;
use rle_vec::RleVec;
use serde::{Deserialize, Serialize};

use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDArray<T> {
    pub data: RleVec<T>,
    // TODO: Not necessary to store shape or strides, which can be stored externally in database.
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> NDArray<T> {
    // TODO: Use the standard Rust name for this method.
    /// Return the index of an arbitrary value which matches `predicate`.
    pub fn find_index<P>(&self, mut predicate: P) -> Option<Vec<usize>>
    where
        Self: Sized,
        P: FnMut(&T) -> bool,
    {
        for (idx, val) in self.data.iter().enumerate() {
            if predicate(val) {
                let mut multidim_idx = Vec::with_capacity(self.shape.len());
                let mut remaining = idx;
                for &stride in &self.strides {
                    let (d, r) = remaining.div_rem(stride);
                    multidim_idx.push(d);
                    remaining = r;
                }
                debug_assert_eq!(remaining, 0);
                debug_assert_eq!(multidim_idx.len(), self.shape.len());
                return Some(multidim_idx);
            }
        }
        None
    }
}

impl<T: Clone + Eq> NDArray<T> {
    pub fn new_with_value(shape: &[usize], value: T) -> Self {
        Self::new_from_buffer(shape, vec![value; shape.iter().product()])
    }

    pub fn new_from_buffer(shape: &[usize], buffer: Vec<T>) -> Self {
        let strides = calculate_strides(shape);
        let volume = shape.iter().product();
        assert_eq!(buffer.len(), volume, "Buffer size must match shape.");
        Self {
            data: buffer.as_slice().into(),
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn set_pt(&mut self, pt: &[usize], value: T) {
        let index = self.data_offset(pt);
        self.data.set(index, value);
    }
}

impl<T: Default + Clone + Eq> NDArray<T> {
    pub fn new(shape: &[usize]) -> Self {
        Self::new_with_value(shape, T::default())
    }
}

impl<T> NDArray<T> {
    /// Convert a multi-dimensional index into an offset into `self.data`.
    fn data_offset(&self, indices: &[usize]) -> usize {
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

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<&[usize]> for NDArray<T> {
    type Output = T;

    fn index(&self, pt: &[usize]) -> &Self::Output {
        &self.data[self.data_offset(pt)]
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

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: proptest-ize this test.
    #[test]
    fn test_insert_single_value_and_retrieve_with_get_arbitrary() {
        let insertion_point = vec![1, 1];
        let mut arr = NDArray::new_with_value(&[4, 4], false);
        arr.set_pt(&insertion_point, true);
        let retrieved_idx = arr.find_index(|&v| v);
        assert_eq!(retrieved_idx, Some(insertion_point));
    }
}
