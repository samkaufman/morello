//! Helpers to emit C loops over the strips of contiguous tensor views.
//!
//! A *strip* is a run of consecutive elements of a tensor view. It may live in memory or vector
//! registers.
//!
//! One loop can visit several related views in lockstep, such as a kernel's input and output: each
//! call of the body gets the same strip `k` of every view. Strips are numbered in buffer order, so
//! strip `k` of each view holds the same logical elements (e.g., for `out[j] = f(in[j])`) only if
//! the views share a layout.
//!
//! The views needn't be kernel arguments, but each must be fully contiguous
//! ([TensorSpec::is_contiguous](crate::tensorspec::TensorSpec::is_contiguous)) and backed by a
//! tensor with a C buffer in scope (i.e., not an unbound `Param`).
//!
//! The C loop is fully unrolled, calling the body once per strip, if any view is backed by a tensor
//! in register.
//!
//! # Example
//!
//! Suppose `x` is a 32-element `f32` tensor in `buf` and `accums` is a
//! [VecAccums](super::vec_accums::VecAccums) of `Add` accumulators `acc0` and `acc1`. The following
//! sums the strips of `x` into the accumulators, alternating between them:
//!
//! ```ignore
//! let strip_loop = self.strip_loop(nz!(8u32), 2, [x]);
//! strip_loop.emit(w, depth, |w, depth, lane, [ptr]| {
//!     accums.emit_update(w, depth, lane, &format!("_mm256_loadu_ps({ptr})"))
//! })?;
//! ```
//!
//! This emits:
//!
//! ```c
//! for (int n003 = 0; n003 < 32; n003 += 16) {
//!   acc0 += _mm256_loadu_ps(buf + (n003));
//!   acc1 += _mm256_loadu_ps(buf + (n003 + 8));
//! }
//! ```

use super::c_utils::CName;
use super::cpu::{vector_offsets, CpuCodeGenerator};
use crate::common::DimSize;
use crate::expr::NonAffineExpr;
use crate::target::CpuTarget;
use crate::views::{View, ViewE};
use std::fmt::{self, Write};

pub struct StripLoop<const N: usize> {
    /// The line opening the `for` loop, or `None` if unrolled.
    ///
    /// An example would be `Some("for (int n003 = 0; n003 < 32; n003 += 16) {")`.
    for_line: Option<String>,
    strips: Vec<Strip<N>>,
}

struct Strip<const N: usize> {
    lane: usize,
    /// A C pointer to the strip in each view (e.g., `buf + (n003 + 8)`).
    ptrs: [String; N],
}

impl<Tgt: CpuTarget> CpuCodeGenerator<Tgt> {
    /// Plans a loop over the `strip_size`-element strips of `args`, which must share a volume
    /// that is a multiple of `lanes * strip_size` unless a view is in vector registers.
    pub fn strip_loop<const N: usize>(
        &mut self,
        strip_size: DimSize,
        lanes: u32,
        args: [&ViewE<Tgt>; N],
    ) -> StripLoop<N> {
        let volume = args[0].spec().volume().get();
        for arg in &args {
            debug_assert!(arg.spec().is_contiguous());
            debug_assert_eq!(arg.spec().volume().get(), volume);
        }
        let strip_size = strip_size.get();
        // The body is unrolled over the `extent` elements starting at `base`: the whole view if
        // unrolled, otherwise one iteration's `step` elements from the loop counter.
        let (for_line, extent, base) = if args
            .iter()
            .any(|arg| self.c_buffer_for_view(arg).needs_unroll())
        {
            (None, volume, NonAffineExpr::zero())
        } else {
            let step = lanes * strip_size;
            debug_assert_eq!(volume % step, 0);
            let name = self.namer.fresh_name();
            let for_line = format!("for (int {name} = 0; {name} < {volume}; {name} += {step}) {{");
            (Some(for_line), step, NonAffineExpr::from(CName(name)))
        };
        let lanes = usize::try_from(lanes).unwrap();
        let strips = vector_offsets(extent, strip_size)
            .enumerate()
            .map(|(k, offset)| {
                let offset = base.clone() + offset;
                Strip {
                    lane: k % lanes,
                    ptrs: args.map(|arg| self.c_ptr_arg_add(arg, &offset)),
                }
            })
            .collect();
        StripLoop { for_line, strips }
    }
}

impl<const N: usize> StripLoop<N> {
    /// Emits the loop, calling `body` with the writer, depth, lane, and per-view pointers of each
    /// strip: once per lane at `depth + 1` inside the C `for` if rolled, otherwise once per strip
    /// at `depth`.
    pub fn emit<W, F>(&self, w: &mut W, depth: usize, mut body: F) -> fmt::Result
    where
        W: Write,
        F: FnMut(&mut W, usize, usize, [&str; N]) -> fmt::Result,
    {
        let mut body_depth = depth;
        if let Some(for_line) = &self.for_line {
            cline!(w, depth, "{for_line}")?;
            body_depth += 1;
        }
        for Strip { lane, ptrs } in &self.strips {
            body(w, body_depth, *lane, ptrs.each_ref().map(String::as_str))?;
        }
        if self.for_line.is_some() {
            cline!(w, depth, "}}")?;
        }
        Ok(())
    }
}
