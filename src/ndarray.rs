use crate::utils::iter_multidim_range;
use divrem::DivRem;
use rle_vec::RleVec;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::ops::{Index, Range};

const NDARRAY_DEFAULT_RANK: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDArray<T> {
    pub data: RleVec<T>,
    // TODO: Not necessary to store shape or strides, which can be stored externally in database.
    shape: SmallVec<[usize; NDARRAY_DEFAULT_RANK]>,
    strides: SmallVec<[usize; NDARRAY_DEFAULT_RANK]>,
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

    pub fn runs_len(&self) -> usize {
        self.data.runs_len()
    }
}

impl<T: Clone + Eq> NDArray<T> {
    pub fn new_with_value(shape: &[usize], value: T) -> Self {
        let volume = shape.iter().product::<usize>();
        let mut buffer = RleVec::new();
        buffer.push_n(volume.try_into().unwrap(), value);
        Self::new_from_rlevec(shape, volume, buffer)
    }

    pub fn new_from_buffer(shape: &[usize], buffer: Vec<T>) -> Self {
        let volume = shape.iter().product();
        Self::new_from_rlevec(shape, volume, buffer.as_slice().into())
    }

    fn new_from_rlevec(shape: &[usize], volume: usize, buffer: RleVec<T>) -> Self {
        let strides = calculate_strides(shape);
        assert_eq!(buffer.len(), volume, "Buffer size must match shape.");
        Self {
            data: buffer,
            shape: SmallVec::from_slice(shape),
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

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn fill_region_counting(
        &mut self,
        dim_ranges: &[Range<u32>],
        value: &T,
        counting_value: &T,
    ) -> u32
    where
        T: Clone + Eq + std::fmt::Debug,
    {
        debug_assert_ne!(value, counting_value);
        let mut affected = 0;
        let mut last_run_idx = 0;
        iter_multidim_range(dim_ranges, &self.strides, |index, _| {
            if &self.data[index] == counting_value {
                affected += 1;
            }
            last_run_idx = self.data.set_hint(index, value.clone(), last_run_idx);
        });
        affected
    }

    pub fn fill_broadcast_1d<I>(&mut self, dim_ranges: &[Range<u32>], inner_slice_iter: I)
    where
        T: Clone + Eq,
        I: Clone + Iterator<Item = T>,
    {
        let inner_slice_iter = inner_slice_iter.fuse();

        let mut dim_ranges_ext = Vec::with_capacity(dim_ranges.len() + 1);
        dim_ranges_ext.extend_from_slice(dim_ranges);
        dim_ranges_ext.push(0..self.shape[self.shape.len() - 1].try_into().unwrap());

        let k = u32::try_from(self.shape[self.shape.len() - 1]).unwrap();
        let mut slice_iter = inner_slice_iter.clone();
        let mut last_run_idx = 0;
        iter_multidim_range(&dim_ranges_ext, &self.strides, |index, pt| {
            // TODO: This still iterates over k. Instead, this should skip remaining k.
            if let Some(next_value) = slice_iter.next() {
                last_run_idx = self.data.set_hint(index, next_value, last_run_idx);
            }
            if pt[pt.len() - 1] == k - 1 {
                slice_iter = inner_slice_iter.clone();
            }
        });
    }

    pub fn compact(&mut self) {
        self.data.shrink_to_fit();
    }
}

impl<T> Index<&[usize]> for NDArray<T> {
    type Output = T;

    fn index(&self, pt: &[usize]) -> &Self::Output {
        &self.data[self.data_offset(pt)]
    }
}

fn calculate_strides(shape: &[usize]) -> SmallVec<[usize; NDARRAY_DEFAULT_RANK]> {
    let mut strides = SmallVec::with_capacity(shape.len());
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

    // TODO: proptest-ize this test.
    #[test]
    fn test_fill_subarray() {
        let mut arr = NDArray::new_with_value(&[3, 2], false);
        #[allow(clippy::single_range_in_vec_init)]
        arr.fill_broadcast_1d(&[0..2], std::iter::once(true));
        assert!(arr[&[0, 0]]);
        assert!(!arr[&[0, 1]]);
        assert!(arr[&[1, 0]]);
        assert!(!arr[&[1, 1]]);
        assert!(!arr[&[2, 0]]);
        assert!(!arr[&[2, 1]]);
    }
}
