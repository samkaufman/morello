use num_traits::PrimInt;
use std::fmt;
use std::io;
use std::iter;

const INDENT_SIZE: usize = 2;

/// Wraps an [io::Write] for use as a [fmt::Write].
pub struct ToWriteFmt<T: io::Write>(pub T);

// Wraps a [fmt::Write] to prepend [str] to each line.
pub struct LinePrefixWrite<'a, W: fmt::Write>(W, &'a str, bool);

pub struct SumSeqs<T>(Box<dyn Iterator<Item = Vec<T>> + Send>, bool);

pub struct Diagonals<T> {
    maxes: Vec<T>,
    stage: T,
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

impl<T> SumSeqs<T> {
    pub fn is_empty(&self) -> bool {
        self.1
    }
}

impl<T> Iterator for SumSeqs<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<T: PrimInt + Send + 'static> Iterator for Diagonals<T> {
    type Item = SumSeqs<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let diagonal = sum_seqs(&self.maxes, self.stage);
        if diagonal.is_empty() {
            return None;
        }
        self.stage = self.stage.add(T::one());
        Some(diagonal)
    }
}

/// Generate a lowercase ASCII name from an index.
///
/// # Examples
/// ```
/// # use morello::utils::ascii_name;
/// assert_eq!(ascii_name(0), "a");
/// assert_eq!(ascii_name(25), "z");
/// assert_eq!(ascii_name(26), "aa");
/// assert_eq!(ascii_name(27), "ab");
/// assert_eq!(ascii_name(26*26 + 26), "aaa");
/// ```
pub fn ascii_name(idx: usize) -> String {
    let mut n = idx
        .checked_add(1)
        .expect("index too large for bijective base-26");
    let mut characters: Vec<u8> = Vec::with_capacity(3);
    while n > 0 {
        let q = (n - 1) / 26;
        let r = ((n - 1) % 26) as u8;
        characters.push(b'a' + r);
        n = q;
    }
    characters.reverse();
    String::from_utf8(characters).expect("generated only ASCII")
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

pub const fn bit_length_inverse_u32(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    2u32.pow(n - 1)
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

/// Like [iter_powers_of_two_range] where start is either 0 or 1.
///
/// Behavior is undefined and may panic if `n` is not 0 or a power of two.
pub fn iter_powers_of_two(
    n: u64,
    include_zero: bool,
) -> impl DoubleEndedIterator<Item = u64> + Clone {
    let start = if include_zero { 0 } else { 1 };
    iter_powers_of_two_range(start, n)
}

/// Returns an iterator over zero and the powers of two within the given range.
///
/// Behavior is undefined and may panic if `start` or `end` are not 0 or powers of two.
pub fn iter_powers_of_two_range(
    start: u64,
    end: u64,
) -> impl DoubleEndedIterator<Item = u64> + Clone {
    let start_bits = bit_length(start);
    let end_bits = bit_length(end);
    (start_bits..end_bits + 1).map(|b| if b == 0 { 0 } else { 2u64.pow(b - 1) })
}

/// Returns the factors of an integer, in ascending order.
pub fn factors(x: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 1;
    while i * i <= x {
        if x.is_multiple_of(i) {
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

pub fn diagonals<T: PrimInt>(maxes: &[T]) -> Diagonals<T> {
    Diagonals {
        maxes: maxes.to_vec(),
        stage: T::zero(),
    }
}

pub fn sum_seqs<T: PrimInt + Send + 'static>(maxes: &[T], total: T) -> SumSeqs<T> {
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
        let mut obligation = total;
        for &m in &maxes[1..] {
            obligation = obligation.saturating_sub(m).max(T::zero());
        }
        let min_value = maxes[0].min(total);

        let mut flat_map_iter = std::iter::from_fn({
            let mut i = obligation;
            move || {
                if i <= min_value {
                    let r = Some(i);
                    i = i.add(T::one());
                    r
                } else {
                    None
                }
            }
        })
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_diagonals_1() {
        let expected = vec![vec![vec![0, 0]]];
        let got = diagonals(&[0, 0])
            .map(|v| v.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_diagonals_2() {
        let expected = vec![vec![vec![0, 0]], vec![vec![0, 1]]];
        let got = diagonals(&[0, 1])
            .map(|v| v.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_diagonals_3() {
        let expected = vec![
            vec![vec![0, 0]],
            vec![vec![0, 1], vec![1, 0]],
            vec![vec![1, 1], vec![2, 0]],
            vec![vec![2, 1]],
        ];
        let got = diagonals(&[2, 1])
            .map(|v| v.collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
    }

    #[test]
    fn test_iter_powers_of_two() {
        assert_eq!(
            iter_powers_of_two(8, true).collect::<Vec<_>>(),
            vec![0, 1, 2, 4, 8]
        );
        assert_eq!(
            iter_powers_of_two(8, false).collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );

        assert_eq!(iter_powers_of_two(1, true).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(iter_powers_of_two(1, false).collect::<Vec<_>>(), vec![1]);

        assert_eq!(iter_powers_of_two(0, true).collect::<Vec<_>>(), vec![0]);
        assert_eq!(iter_powers_of_two(0, false).collect::<Vec<_>>(), vec![]);
    }

    #[test]
    fn test_iter_powers_of_two_range() {
        assert_eq!(
            iter_powers_of_two_range(0, 8).collect::<Vec<_>>(),
            vec![0, 1, 2, 4, 8]
        );
        assert_eq!(
            iter_powers_of_two_range(2, 16).collect::<Vec<_>>(),
            vec![2, 4, 8, 16]
        );
        assert_eq!(iter_powers_of_two_range(1, 1).collect::<Vec<_>>(), vec![1]);
        assert_eq!(iter_powers_of_two_range(16, 8).collect::<Vec<_>>(), vec![]);
    }

    proptest! {
        #[test]
        fn test_diagonals_always_returns_positive_points(
            maxes in proptest::collection::vec(0i8..=3, 1..=3)
        ) {
            for diag in diagonals(&maxes) {
                for pt in diag {
                    prop_assert!(pt.iter().all(|&v| v >= 0));
                }
            }
        }
    }
}
