use crate::utils::iter_multidim_range;
use rle_vec::RleVec;
use serde::{Deserialize, Serialize};
use std::ops::{Index, Range};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDArray<T> {
    pub data: RleVec<T>,
    // TODO: Not necessary to store shape or strides, which can be stored externally in database.
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> NDArray<T> {
    #[allow(dead_code)]
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

    fn new_from_rlevec(shape: &[usize], volume: usize, buffer: RleVec<T>) -> Self {
        let strides = calculate_strides(shape);
        assert_eq!(buffer.len(), volume, "Buffer size must match shape.");
        Self {
            data: buffer,
            shape: shape.to_vec(),
            strides,
        }
    }

    #[allow(dead_code)]
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

    pub fn volume(&self) -> usize {
        self.shape.iter().product()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn fill_region(&mut self, dim_ranges: &[Range<u32>], value: T)
    where
        T: Clone + Eq,
    {
        self.fill_region_ext(dim_ranges, value, None)
    }

    pub fn fill_region_ext(
        &mut self,
        dim_ranges: &[Range<u32>],
        value: T,
        through_unfilled: Option<(u8, &NDArray<u8>)>,
    ) where
        T: Clone + Eq,
    {
        // Compute the volume of contiguous inner tiles (`step_size`) in `self`. `prefix_len` will
        // be the number of dimensions at the head of `dim_ranges` outside contig. tiles.
        let mut prefix_size = self.shape.len();
        let mut step_size: u32 = 1;
        for (&sh, dr) in self.shape().iter().zip(dim_ranges).rev() {
            if sh != dr.len() {
                break;
            }
            step_size *= u32::try_from(sh).unwrap();
            prefix_size -= 1;
        }

        // If `prefix_size` is 0, then we'll filling the entire buffer, so we can just call
        // `set_range`.  Otherwise, we're going to repeatedly called `fill_region_ext_inner` on each
        // contiguous tile.
        if prefix_size == 0 {
            debug_assert_eq!(self.volume(), usize::try_from(step_size).unwrap());
            self.data.set_range(0, step_size, value);
            return;
        }

        let next_rng = &dim_ranges[prefix_size - 1];
        let offset = next_rng.start * step_size;
        step_size *= next_rng.end - next_rng.start;

        if prefix_size == 1 {
            Self::fill_region_ext_inner(
                &mut self.data,
                offset,
                step_size,
                value,
                through_unfilled.map(|(v, f)| (v, &f.data)),
                None,
            );
        } else {
            let mut run_index_hint = None;
            let substrides = &self.strides[..(prefix_size - 1)];
            iter_multidim_range(&dim_ranges[..(prefix_size - 1)], substrides, |i, _| {
                run_index_hint = Self::fill_region_ext_inner(
                    &mut self.data,
                    u32::try_from(i).unwrap() + offset,
                    step_size,
                    value.clone(),
                    through_unfilled.map(|(v, f)| (v, &f.data)),
                    run_index_hint,
                );
            });
        }
    }

    /// Fills a range of an [RleVec], starting at index, up to at least `fill_len` with `value`.
    ///
    /// If `through_unfilled` is given, then `data` may be filled even beyond the given length.  The
    /// given [RleVec] will be scanned for as long as it contains integers less than or equal to the
    /// given value (the first element of the given tuple). Naturally, `data` and `filled` must have
    /// the same number of elements.
    ///
    /// This function is intended to fill "flattened," contiguous inner tiles of an NDArray.
    fn fill_region_ext_inner(
        data: &mut RleVec<T>,
        index: u32,
        mut fill_len: u32,
        value: T,
        through_unfilled: Option<(u8, &RleVec<u8>)>, // TODO: Should also be RleVec?
        run_index_hint: Option<u32>,
    ) -> Option<u32>
    where
        T: Clone + Eq,
    {
        // Extend `fill_len` if permitted by `through_unfilled`.  (There's an opportunity
        // here to also fill *backwards* by updating `index`, but this isn't implemented.)
        let mut hint_to_return = None;
        if let Some((unfilled_max, filled)) = through_unfilled {
            // TODO: Map the following by dividing out any k. Using stride?
            //
            // TODO: The following can easily fill past step_size into the next
            //   step. For example, the first fill_region_ext is likely to fill
            //   the entire Vec with a single run. In this case, we don't
            //   really need all the subsequent calls to set_range at all. Can
            //   we cheaply avoid these?
            debug_assert_eq!(data.len(), filled.len());

            // Update `extended_idx` to be the first index where the value if greater than
            // `unfilled_max`.
            let mut extended_idx = index + fill_len;
            if extended_idx < u32::try_from(filled.len()).unwrap() {
                let extended_run_idx = run_index_frhint(filled, extended_idx, run_index_hint);
                hint_to_return = Some(extended_run_idx);
                let mut ext_iter = filled.runs().skip(extended_run_idx.try_into().unwrap());
                loop {
                    if filled.len() <= usize::try_from(extended_idx).unwrap() {
                        break;
                    }
                    let run = ext_iter.next().unwrap();
                    if *run.value > unfilled_max {
                        break;
                    }
                    // TODO: Remove extra start?
                    extended_idx = run.start + run.len;
                }
                fill_len = extended_idx - index;
            }
        }
        data.set_range(index, fill_len, value);
        hint_to_return
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
        if let Some(filled) = filled {
            assert_eq!(dim_ranges.len(), filled.shape().len());
        }

        // Update dim_ranges with a Range for the k dimension, which will be (partially) filled with
        // repeated copies out of inner_slice_iter.
        let k = u32::try_from(self.shape[self.shape.len() - 1]).unwrap();
        let mut dim_ranges_ext = Vec::with_capacity(dim_ranges.len() + 1);
        dim_ranges_ext.extend_from_slice(dim_ranges);
        dim_ranges_ext.push(0..k);

        // If may faster to always use this implementation, even when k > 1, which we
        // could accomplish by moving the k dimension to the front of the shape.
        if k == 1 {
            if let Some(single_value) = inner_slice_iter.next() {
                // A value of 1 in `filled` means that there are zero actions. 0 means empty.  So
                // this will fill through anywhere there isn't at least one action, which is safe.
                // TODO: The following `1` should be a function of the index in the iterator, not
                //   a constant. (It's okay to fill in parts of the table which are blocked by a
                //   too-low `filled` value.)
                // TODO: This is really a database-specific detail, not an NDArray detail. It
                //   shouldn't be decided here.
                self.fill_region_ext(&dim_ranges_ext, single_value, filled.map(|f| (1, f)));
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
}

impl<T> Index<&[usize]> for NDArray<T> {
    type Output = T;

    fn index(&self, pt: &[usize]) -> &Self::Output {
        &self.data[self.data_offset(pt)]
    }
}

fn run_index_frhint<T>(rle_vec: &RleVec<T>, index: u32, hint: Option<u32>) -> u32 {
    // First, test up to 8 ahead of the hint. This is just an empirically derived heurstic.
    // TODO: Tune this.
    if let Some(hint) = hint {
        for to_test in hint..hint + 8 {
            if let Some(tested_run) = rle_vec.runs().nth(to_test.try_into().unwrap()) {
                // TODO: Can abort faster below.
                if tested_run.start <= index && index < tested_run.start + tested_run.len {
                    return to_test;
                }
            }
        }
    }

    // TODO: Binary search at and ahead of the hint before a binary search behind.

    rle_vec.run_index(index)
}

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
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
        arr.fill_region_ext(&[0..1, 0..3, 0..1], 1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_fill_subarray_with_infill_2() {
        let filled = NDArray::new_with_value(&[3, 3], 1);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[1..2, 0..3, 0..1], 1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![0, 0, 0, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_fill_subarray_with_infill_3() {
        let filled = NDArray::new_with_value(&[3, 3], 1);
        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[0..3, 0..3, 0..1], 1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![1; 9]);
    }

    #[test]
    fn test_fill_subarray_with_infill_4() {
        let shape: &[usize] = &[3, 3];
        let buffer = [1, 1, 1, 1, 1, 0, 1, 0, 0];
        let volume = shape.iter().product();
        let filled = NDArray::new_from_rlevec(shape, volume, buffer.as_slice().into());

        let mut arr = NDArray::new_with_value(&[3, 3, 1], 0);
        arr.fill_region_ext(&[0..3, 1..2, 0..1], 1, Some((0, &filled)));
        assert_eq!(arr.data.to_vec(), vec![0, 1, 0, 0, 1, 1, 0, 1, 1]);
    }
}
