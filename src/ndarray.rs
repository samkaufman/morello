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

    pub fn get_with_neighbor(&self, pt: &[usize]) -> (&T, Option<&T>) {
        let data_idx = self.data_offset(pt);
        let run_idx: usize = self
            .data
            .run_index(data_idx.try_into().unwrap())
            .try_into()
            .unwrap();
        let center_run = rle_vec_get_run(&self.data, run_idx);
        let center_value = center_run.value;
        if self.data.runs_len() == 1 {
            (center_value, None)
        } else if (run_idx == 0) {
            (center_value, Some(rle_vec_get_run(&self.data, run_idx + 1).value))
        } else {
            (center_value, Some(rle_vec_get_run(&self.data, run_idx - 1).value))
        }
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

    pub fn fill_region(&mut self, dim_ranges: &[Range<u32>], value: &T)
    where
        T: Clone + Eq,
    {
        self.fill_region_ext(dim_ranges, value, None)
    }

    pub fn fill_region_ext(
        &mut self,
        dim_ranges: &[Range<u32>],
        value: &T,
        through_unfilled: Option<(u8, &NDArray<u8>)>,
    ) where
        T: Clone + Eq,
    {
        // Figure out the volume of contiguous inner tile.
        let mut prefix_size = self.shape.len();
        let mut step_size: u32 = 1;
        for (&sh, dr) in self.shape().iter().zip(dim_ranges).rev() {
            if sh != dr.len() {
                break;
            }
            step_size *= u32::try_from(sh).unwrap();
            prefix_size -= 1;
        }

        if prefix_size >= 1 {
            let next_rng = &dim_ranges[prefix_size - 1];
            let offset = next_rng.start * step_size;
            step_size *= next_rng.end - next_rng.start;

            if prefix_size == 1 {
                Self::fill_region_ext_inner(
                    &mut self.data,
                    0,
                    offset,
                    step_size,
                    value,
                    through_unfilled,
                );
            } else {
                let substrides = &self.strides[..(prefix_size - 1)];
                iter_multidim_range(&dim_ranges[..(prefix_size - 1)], substrides, |i, _| {
                    Self::fill_region_ext_inner(
                        &mut self.data,
                        i,
                        offset,
                        step_size,
                        value,
                        through_unfilled,
                    )
                });
            }
        } else {
            debug_assert_eq!(
                self.shape().iter().product::<usize>(),
                usize::try_from(step_size).unwrap()
            );
            self.data.set_range(0, step_size, value.clone());
        }
    }

    fn fill_region_ext_inner(
        data: &mut RleVec<T>,
        index: usize,
        offset: u32,
        step_size: u32,
        value: &T,
        through_unfilled: Option<(u8, &NDArray<u8>)>,
    ) where
        T: Clone + Eq,
    {
        // Fill beyond the step size if permitted by fill. Note that this
        // fills forwards only, not backward.
        let mut fill_len = step_size;
        if let Some((unfilled_max, filled)) = through_unfilled {
            // TODO: Map the following by dividing out any k. Using stride?
            //
            // TODO: The following can easily fill past step_size into the next
            //   step. For example, the first fill_region_ext is likely to fill
            //   the entire Vec with a single run. In this case, we don't
            //   really need all the subsequent calls to set_range at all. Can
            //   we cheaply avoid these?
            let mut extended_idx = u32::try_from(index).unwrap() + offset + step_size;
            if extended_idx < u32::try_from(filled.data.len()).unwrap() {
                let eiri = filled.data.run_index(extended_idx).try_into().unwrap();
                let mut ext_iter = filled.data.runs().skip(eiri);
                loop {
                    if filled.data.len() <= usize::try_from(extended_idx).unwrap() {
                        break;
                    }
                    let run = ext_iter.next().unwrap();
                    if *run.value > unfilled_max {
                        break;
                    }
                    // TODO: What about removing extra start?
                    extended_idx = run.start + run.len;
                }
                fill_len = extended_idx - u32::try_from(index).unwrap() - offset;
            }
        }
        data.set_range(
            u32::try_from(index).unwrap() + offset,
            fill_len,
            value.clone(),
        );
    }

    pub fn fill_broadcast_1d<I>(
        &mut self,
        dim_ranges: &[Range<u32>],
        mut inner_slice_iter: I,
        filled: Option<&NDArray<u8>>,
    ) where
        T: Clone + Eq,
        I: Clone + Iterator<Item = T>,
    {
        // Update dim_ranges with a Range for the k dimension, which will be (partially) filled with
        // repeated copies out of inner_slice_iter.
        let k = u32::try_from(self.shape[self.shape.len() - 1]).unwrap();
        let mut dim_ranges_ext = Vec::with_capacity(dim_ranges.len() + 1);
        dim_ranges_ext.extend_from_slice(dim_ranges);
        dim_ranges_ext.push(0..k);

        // TODO: If k = 1, we'll use the faster implementation: `fill_region`.
        // If may faster to always use this implementation, even when k > 1, which we
        // could accomplish by moving the k dimension to the front of the shape.
        if k == 1 {
            if let Some(single_value) = inner_slice_iter.next() {
                // TODO: The following `1` should be a function of the index in the iterator, not
                //   a constant. (It's okay to fill in parts of the table which are blocked by a
                //   too-low `filled` value.)
                self.fill_region_ext(&dim_ranges_ext, &single_value, filled.map(|f| (1, f)));
            }
            return;
        }

        let inner_slice_iter = inner_slice_iter.fuse();
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

    pub fn shrink_to_fit(&mut self) {
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

// TODO: This should be in the rle_vec crate
fn rle_vec_get_run<T>(v: &RleVec<T>, run_idx: usize) -> rle_vec::Run<&T> {
    v.runs().nth(run_idx).unwrap()
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

    // TODO: proptest-ize this test, and test with non-None filled.
    #[test]
    fn test_fill_subarray() {
        let mut arr = NDArray::new_with_value(&[3, 2], false);
        #[allow(clippy::single_range_in_vec_init)]
        arr.fill_broadcast_1d(&[0..2], std::iter::once(true), None);
        assert!(arr[&[0, 0]]);
        assert!(!arr[&[0, 1]]);
        assert!(arr[&[1, 0]]);
        assert!(!arr[&[1, 1]]);
        assert!(!arr[&[2, 0]]);
        assert!(!arr[&[2, 1]]);
    }

    #[test]
    fn test_fill_subarray_with_infill_1() {
        let filled = NDArray::new_with_value(&[3, 3], 0);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[0..1, 0..3, 0..1], &1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_fill_subarray_with_infill_2() {
        let filled = NDArray::new_with_value(&[3, 3], 1);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[1..2, 0..3, 0..1], &1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![0, 0, 0, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_fill_subarray_with_infill_3() {
        let filled = NDArray::new_with_value(&[3, 3], 1);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[0..3, 0..3, 0..1], &1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![1; 9]);
    }

    #[test]
    fn test_fill_subarray_with_infill_4() {
        let filled = NDArray::new_from_buffer(&[3, 3], vec![1, 1, 1, 1, 1, 0, 1, 0, 0]);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[0..3, 1..2, 0..1], &1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![0, 1, 0, 0, 1, 1, 0, 1, 1]);
    }
}
