use super::c_utils::VecType;
use super::cpu::CpuCodeGenerator;
use crate::common::Dtype;
use crate::target::CpuTarget;
use itertools::Itertools;
use std::fmt::{self, Write};

/// An associative operation over vector registers.
#[derive(Clone, Copy)]
pub enum VecReduce {
    Add,
    Max,
}

/// VRF accumulators that will be reduced to one.
#[derive(Clone)]
pub struct VecAccums {
    pub op: VecReduce,
    pub names: Vec<String>,
}

impl VecReduce {
    /// Returns a C statement which folds the term `term_expr` into the accumulator
    /// `accum_expr`, which must be an lvalue of the same vector type.
    pub fn update_stmt(self, accum_expr: &str, term_expr: &str) -> String {
        match self {
            VecReduce::Add => format!("{accum_expr} += {term_expr};"),
            VecReduce::Max => {
                format!("{accum_expr} = __builtin_elementwise_max({accum_expr}, {term_expr});")
            }
        }
    }
}

impl VecAccums {
    /// Emits declarations of `count` fresh accumulators of `vector_type`, each initialized to the
    /// identity value of `op`.
    pub fn declare<Tgt: CpuTarget, W: Write>(
        generator: &mut CpuCodeGenerator<Tgt>,
        w: &mut W,
        depth: usize,
        op: VecReduce,
        count: u32,
        vector_type: &'static VecType,
    ) -> Result<Self, fmt::Error> {
        debug_assert_ne!(count, 0);
        generator.headers.vector_type_defs.insert(vector_type);
        let init = match op {
            VecReduce::Add => format!("({}){{}}", vector_type.name),
            VecReduce::Max => {
                let identity = match vector_type.dtype {
                    Dtype::Uint8
                    | Dtype::Sint8
                    | Dtype::Uint16
                    | Dtype::Sint16
                    | Dtype::Uint32
                    | Dtype::Sint32 => generator.c_min_const_for_dtype(vector_type.dtype),
                    Dtype::Float32 => {
                        generator.headers.emit_math_include = true;
                        "-INFINITY"
                    }
                    Dtype::Bfloat16 => {
                        generator.headers.emit_math_include = true;
                        "(__bf16)(-INFINITY)"
                    }
                };
                format!("({identity}) - ({}){{}}", vector_type.name)
            }
        };
        let names = generator.namer.fresh_names(usize::try_from(count).unwrap());
        for name in &names {
            cline!(w, depth, "{} {name} = {init};", vector_type.name)?;
        }
        Ok(VecAccums { op, names })
    }

    /// Emits a statement folding the vector expression `x` into one of the accumulators.
    ///
    /// `input_idx` is the position of `x` in the sequence of vectors being reduced. Inputs are
    /// mapped to accumulators round-robin.
    pub fn emit_update<W: Write>(
        &self,
        w: &mut W,
        depth: usize,
        input_idx: usize,
        x: &str,
    ) -> fmt::Result {
        let accum = &self.names[input_idx % self.names.len()];
        cline!(w, depth, "{}", self.op.update_stmt(accum, x))
    }

    /// Emits a reduction of all accumulators into the first, and returns its name.
    ///
    /// Other accumulators are also modified. Accumulators are combined pairwise, halving the number
    /// remaining each round. For example, an [VecReduce::Add] reduction of accumulators `a`, `b`,
    /// `c`, and `d` emits the following and returns `a`:
    /// ```c
    /// a += b;
    /// c += d;
    /// a += c;
    /// ```
    pub fn emit_reduce<W: Write>(&self, w: &mut W, depth: usize) -> Result<&str, fmt::Error> {
        let mut stride = 1;
        while stride < self.names.len() {
            for (lhs, rhs) in self.names.iter().step_by(stride).tuples() {
                cline!(w, depth, "{}", self.op.update_stmt(lhs, rhs))?;
            }
            stride *= 2;
        }
        Ok(&self.names[0])
    }
}
