use crate::memorylimits::MemVec;

// If true, schedules will be saved as if they had memory limits, for all banks,
// that are the next highest power of 2. This discretizes the cache a bit, even
// though it
const SNAP_CAP_TO_POWER_OF_TWO: bool = true;

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

fn bit_length(n: u64) -> u32 {
    let bits = u64::BITS - n.leading_zeros();
    if n == 0 {
        0
    } else {
        bits
    }
}

pub const fn prev_power_of_two(n: u64) -> u64 {
    let highest_bit_set_idx = (u64::BITS - 1) - (n | 1).leading_zeros();
    (1 << highest_bit_set_idx) & n
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
