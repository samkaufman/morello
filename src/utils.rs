use crate::memorylimits::MemVec;
use std::fmt;
use std::io;
use std::iter;

const INDENT_SIZE: usize = 2;

// If true, schedules will be saved as if they had memory limits, for all banks,
// that are the next highest power of 2. This discretizes the cache a bit, even
// though it
const SNAP_CAP_TO_POWER_OF_TWO: bool = true;

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

pub const ASCII_CHARS: [char; 26] = ascii_chars();
pub const ASCII_PAIRS: [[char; 2]; 676] = ascii_pairs();

pub struct ToWriteFmt(pub Vec<Box<dyn io::Write>>);

impl ToWriteFmt {
    pub fn new<W: io::Write + 'static>(w: W) -> Self {
        Self(vec![Box::new(w)])
    }

    pub fn push<W: io::Write + 'static>(&mut self, w: W) {
        self.0.push(Box::new(w));
    }
}

impl fmt::Write for ToWriteFmt {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0
            .iter_mut()
            .map(|w| w.write_all(s.as_bytes()))
            .collect::<Result<_, _>>()
            .map_err(|_| fmt::Error)
    }
}

pub fn snap_availables_up_memvec(available: MemVec, always: bool) -> MemVec {
    if !SNAP_CAP_TO_POWER_OF_TWO && !always {
        return available;
    }

    available
        .iter()
        .map(|&v| {
            if v == 0 {
                0
            } else {
                2u64.pow(bit_length(v - 1))
            }
        })
        .collect()
}

pub fn bit_length(n: u64) -> u32 {
    u64::BITS - n.leading_zeros()
}

pub fn bit_length_u32(n: u32) -> u32 {
    u32::BITS - n.leading_zeros()
}

pub const fn is_power_of_two_u32(n: u32) -> bool {
    n & (n - 1) == 0
}

pub const fn prev_power_of_two(n: u64) -> u64 {
    let highest_bit_set_idx = (u64::BITS - 1) - (n | 1).leading_zeros();
    (1 << highest_bit_set_idx) & n
}

pub fn iter_powers_of_two(
    n: u64,
    include_zero: bool,
) -> impl Iterator<Item = u64> + DoubleEndedIterator + Clone {
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

pub fn sum_seqs(maxes: &[u32], total: u32) -> Box<dyn Iterator<Item = Vec<u32>>> {
    let maxes = maxes.to_vec();
    let len = maxes.len();
    if len == 0 {
        Box::new(iter::empty())
    } else if len == 1 {
        if maxes[0] >= total {
            Box::new(iter::once(vec![total]))
        } else {
            Box::new(iter::empty())
        }
    } else {
        let obligation = total.saturating_sub(maxes[1..].iter().sum::<u32>());
        let min_value = u32::min(maxes[0], total);

        let flat_map_iter = (obligation..=min_value).flat_map(move |v| {
            let maxes_tail = &maxes[1..];
            sum_seqs(maxes_tail, total - v).map(move |mut suffix| {
                let mut prefix = vec![v];
                prefix.append(&mut suffix);
                prefix
            })
        });

        Box::new(flat_map_iter)
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
