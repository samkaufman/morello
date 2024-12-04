use crate::memorylimits::MemVec;

use std::fmt;
use std::io;
use std::iter;
use std::ops::Range;

const INDENT_SIZE: usize = 2;

// If true, schedules will be saved as if they had memory limits, for all banks,
// that are the next highest power of 2. This discretizes the cache a bit, even
// though it
const SNAP_CAP_TO_POWER_OF_TWO: bool = true;

pub const ASCII_CHARS: [char; 26] = ascii_chars();
pub const ASCII_PAIRS: [[char; 2]; 676] = ascii_pairs();

/// Wraps an [io::Write] for use as a [fmt::Write].
pub struct ToWriteFmt<T: io::Write>(pub T);

// Wraps a [fmt::Write] to prepend [str] to each line.
pub struct LinePrefixWrite<'a, W: fmt::Write>(W, &'a str, bool);

pub struct SumSeqs(Box<dyn Iterator<Item = Vec<u32>> + Send>, bool);

pub struct Diagonals {
    maxes: Vec<u32>,
    stage: u32,
}

impl<T: io::Write> fmt::Write for ToWriteFmt<T> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write_all(s.as_bytes()).map_err(|_| fmt::Error)
    }
}

impl<'a, W: fmt::Write> LinePrefixWrite<'a, W> {
    pub fn new(inner: W, line_prefix: &'a str) -> Self {
        LinePrefixWrite(inner, line_prefix, true)
    }
}

impl<W: fmt::Write> fmt::Write for LinePrefixWrite<'_, W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.2 && !s.is_empty() {
            self.0.write_str(self.1)?;
        }

        let mut split_iter = s.split_inclusive('\n').peekable();
        while let Some(substring) = split_iter.next() {
            self.0.write_str(substring)?;
            if split_iter.peek().is_some() {
                self.0.write_str(self.1)?;
            }
        }
        self.2 = s.ends_with('\n');
        Ok(())
    }
}

impl SumSeqs {
    pub fn is_empty(&self) -> bool {
        self.1
    }
}

impl Iterator for SumSeqs {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl Iterator for Diagonals {
    type Item = SumSeqs;

    fn next(&mut self) -> Option<Self::Item> {
        let diagonal = sum_seqs(&self.maxes, self.stage);
        if diagonal.is_empty() {
            return None;
        }
        self.stage += 1;
        Some(diagonal)
    }
}

const fn ascii_chars() -> [char; 26] {
    let mut chars = ['\0'; 26];
    let mut c: u8 = b'a';
    while c <= b'z' {
        chars[(c - b'a') as usize] = c as char;
        c += 1;
    }
    chars
}

const fn ascii_pairs() -> [[char; 2]; 676] {
    let mut result = [['a', 'a']; 676];
    let mut idx = 0;

    let mut c1: u8 = b'a';
    while c1 <= b'z' {
        let mut c2: u8 = b'a';
        while c2 <= b'z' {
            result[idx] = [c1 as char, c2 as char];
            idx += 1;
            c2 += 1;
        }
        c1 += 1;
    }
    result
}

pub fn snap_memvec_up(available: MemVec, always: bool) -> MemVec {
    if !SNAP_CAP_TO_POWER_OF_TWO && !always {
        return available;
    }
    available.map(|v| if v == 0 { 0 } else { v.next_power_of_two() })
}

pub const fn bit_length(n: u64) -> u32 {
    debug_assert!(n == 0 || is_power_of_two(n));
    u64::BITS - n.leading_zeros()
}

pub const fn bit_length_u32(n: u32) -> u32 {
    debug_assert!(n == 0 || is_power_of_two_u32(n));
    u32::BITS - n.leading_zeros()
}

pub const fn bit_length_inverse(n: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    2u64.pow(n - 1)
}

pub const fn is_power_of_two(n: u64) -> bool {
    if n == 0 {
        return false;
    }
    n & (n - 1) == 0
}

pub const fn is_power_of_two_u32(n: u32) -> bool {
    if n == 0 {
        return false;
    }
    n & (n - 1) == 0
}

pub const fn next_binary_power(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    n.next_power_of_two()
}

// TODO: Rename
pub const fn prev_power_of_two(n: u64) -> u64 {
    let highest_bit_set_idx = (u64::BITS - 1) - (n | 1).leading_zeros();
    (1 << highest_bit_set_idx) & n
}

pub const fn prev_power_of_two_u32(n: u32) -> u32 {
    let highest_bit_set_idx = (u32::BITS - 1) - (n | 1).leading_zeros();
    (1 << highest_bit_set_idx) & n
}

pub fn iter_powers_of_two(
    n: u64,
    include_zero: bool,
) -> impl DoubleEndedIterator<Item = u64> + Clone {
    let start = if include_zero { 0 } else { 1 };
    let top_bits = bit_length(n);
    (start..top_bits + 1).map(|b| if b == 0 { 0 } else { 2u64.pow(b - 1) })
}

/// Returns the factors of an integer, in ascending order.
pub fn factors(x: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 1;
    while i * i <= x {
        if x % i == 0 {
            result.push(i);
            if x / i != i {
                result.push(x / i);
            }
        }
        i += 1;
    }
    result.sort_unstable();
    result
}

pub fn diagonals(maxes: &[u32]) -> Diagonals {
    Diagonals {
        maxes: maxes.to_vec(),
        stage: 0,
    }
}

pub fn sum_seqs(maxes: &[u32], total: u32) -> SumSeqs {
    let maxes = maxes.to_vec();
    let len = maxes.len();
    if len == 0 {
        SumSeqs(Box::new(iter::empty()), true)
    } else if len == 1 {
        if maxes[0] >= total {
            SumSeqs(Box::new(iter::once(vec![total])), false)
        } else {
            SumSeqs(Box::new(iter::empty()), true)
        }
    } else {
        let obligation = total.saturating_sub(maxes[1..].iter().sum::<u32>());
        let min_value = u32::min(maxes[0], total);

        let mut flat_map_iter = (obligation..=min_value)
            .flat_map(move |v| {
                let maxes_tail = &maxes[1..];
                sum_seqs(maxes_tail, total - v).map(move |mut suffix| {
                    let mut prefix = vec![v];
                    prefix.append(&mut suffix);
                    prefix
                })
            })
            .peekable();
        let is_empty = flat_map_iter.peek().is_none();

        SumSeqs(Box::new(flat_map_iter), is_empty)
    }
}

pub fn join_into_string(c: impl IntoIterator<Item = impl ToString>, separator: &str) -> String {
    c.into_iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(separator)
}

pub fn indent(depth: usize) -> String {
    " ".repeat(depth * INDENT_SIZE)
}

pub(crate) fn iter_multidim_range<F>(dim_ranges: &[Range<u32>], strides: &[usize], mut callback: F)
where
    F: FnMut(usize, &[u32]),
{
    assert_eq!(dim_ranges.len(), strides.len());
    if dim_ranges.is_empty() || dim_ranges.iter().any(|rng| rng.is_empty()) {
        return;
    }

    let mut current_pt = dim_ranges.iter().map(|rng| rng.start).collect::<Vec<_>>();

    let mut buffer_index = current_pt
        .iter()
        .zip(strides)
        .map(|(&dim, &stride)| usize::try_from(dim).unwrap() * stride)
        .sum();

    loop {
        callback(buffer_index, &current_pt);

        let mut dimension = current_pt.len() - 1; // Start with the innermost dimension
        loop {
            current_pt[dimension] += 1;
            buffer_index += strides[dimension];
            if current_pt[dimension] < dim_ranges[dimension].end {
                break;
            }
            if dimension == 0 {
                return;
            }

            buffer_index -= strides[dimension]
                * usize::try_from(current_pt[dimension] - dim_ranges[dimension].start).unwrap();
            current_pt[dimension] = dim_ranges[dimension].start;
            dimension -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use proptest::prelude::*;
    use std::fmt::Write;

    #[test]
    fn test_lineprefixwrite_prefixes_1() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        write!(write, "Oh, ").unwrap();
        writeln!(write, "hi.").unwrap();
        assert_eq!(write.0, "--Oh, hi.\n");
    }

    #[test]
    fn test_lineprefixwrite_prefixes_2() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        writeln!(write, "Line 1.").unwrap();
        writeln!(write, "Line 2.").unwrap();
        assert_eq!(write.0, "--Line 1.\n--Line 2.\n");
    }

    #[test]
    fn test_lineprefixwrite_supports_incremental_line_writing() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        write!(write, "a").unwrap();
        write!(write, "b").unwrap();
        assert_eq!(write.0, "--ab");
    }

    #[test]
    fn test_lineprefixwrite_multiline() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        writeln!(write, "Line 1.\nLine 2.").unwrap();
        assert_eq!(write.0, "--Line 1.\n--Line 2.\n");
    }

    #[test]
    fn test_lineprefixwrite_noop_with_empty_string() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        write!(write, "").unwrap();
        assert_eq!(write.0, "");
    }

    #[test]
    fn test_lineprefixwrite_print_prefix_with_empty_line() {
        let mut write = LinePrefixWrite::new(String::new(), "--");
        writeln!(write).unwrap();
        assert_eq!(write.0, "--\n");
    }

    proptest! {
        #[test]
        fn test_iter_multidim_range_matches_itertools_product(
            rngs in proptest::collection::vec((0u32..5, 0u32..5, 0u32..5), 1..4)
        ) {
            let dim_ranges: Vec<Range<u32>> = rngs
                .iter()
                .map(|(s, l, _)| *s..(*s + *l))
                .collect::<Vec<_>>();

            let shape = rngs
                .iter()
                .map(|(s, l, d)| usize::try_from(*s + *l + *d).unwrap())
                .collect::<Vec<_>>();
            let strides = (1..shape.len())
                .map(|i| shape[i..].iter().product())
                .chain(std::iter::once(1))
                .collect::<Vec<_>>();

            let mut fast_pts = Vec::new();
            let mut fast_indices = Vec::new();
            iter_multidim_range(&dim_ranges, &strides, |i, pt| {
                fast_indices.push(i);
                fast_pts.push(pt.to_vec());
            });

            let itertools_product_pts = dim_ranges
                .into_iter()
                .multi_cartesian_product()
                .collect::<Vec<_>>();
            let itertools_product_indices = itertools_product_pts
                .iter()
                .map(|pt| {
                    pt.iter()
                        .zip(&strides)
                        .map(|(&dim, &stride)| usize::try_from(dim).unwrap() * stride)
                        .sum()
                })
                .collect::<Vec<_>>();

            assert_eq!(fast_pts, itertools_product_pts);
            assert_eq!(fast_indices, itertools_product_indices);
        }
    }
}
