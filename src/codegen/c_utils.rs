use std::fmt;

use crate::{
    common::{DimSize, Dtype, Shape},
    expr::AffineExpr,
    layout::BufferExprTerm,
    utils::indent,
};

#[derive(Debug, Clone)]
pub enum CBuffer {
    HeapArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    StackArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    ValueVar {
        name: String,
        dtype: Dtype,
    },
    SingleVecVar {
        name: String,
        vec_type: &'static VecType,
    },
    VecVars {
        inner_vecs: Vec<CBuffer>,
    },
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct VecType {
    pub dtype: Dtype,
    pub value_cnt: u8,
    pub name: &'static str,
    pub native_type_name: &'static str,
    pub load_fn: &'static str,
    pub store_fn: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CExprTerm {
    Buffer(BufferExprTerm),
    CName(String),
}

impl CBuffer {
    /// Returns the C identifier name if the receiver has just one (i.e., is not distributed).
    pub fn name(&self) -> Option<&str> {
        match self {
            CBuffer::HeapArray { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::ValueVar { name, .. }
            | CBuffer::SingleVecVar { name, .. } => Some(name),
            CBuffer::VecVars { .. } => None,
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            CBuffer::HeapArray { dtype, .. }
            | CBuffer::StackArray { dtype, .. }
            | CBuffer::ValueVar { dtype, .. } => *dtype,
            CBuffer::SingleVecVar { vec_type, .. } => vec_type.dtype,
            CBuffer::VecVars { inner_vecs, .. } => {
                let CBuffer::SingleVecVar { vec_type, .. } = inner_vecs[0] else {
                    unreachable!();
                };
                vec_type.dtype
            }
        }
    }

    pub fn needs_unroll(&self) -> bool {
        matches!(self, CBuffer::VecVars { .. })
    }

    pub fn emit<W: fmt::Write>(&self, w: &mut W, zero_init: bool, depth: usize) -> fmt::Result {
        match self {
            CBuffer::HeapArray { name, size, dtype } => {
                writeln!(w, "{}{} *restrict {};", indent(depth), c_type(*dtype), name)?;
                writeln!(
                    w,
                    "{}posix_memalign((void **)&{}, 128, {}*sizeof({}));",
                    indent(depth),
                    name,
                    size,
                    c_type(*dtype)
                )?;

                if zero_init {
                    writeln!(
                        w,
                        "{}memset({}, 0, {}*sizeof({}));",
                        indent(depth),
                        name,
                        size,
                        c_type(*dtype)
                    )?;
                }

                Ok(())
            }
            CBuffer::StackArray { name, size, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(
                    w,
                    "{}{} {}[{}] __attribute__((aligned (128))){};",
                    indent(depth),
                    c_type(*dtype),
                    name,
                    size,
                    epi
                )
            }
            CBuffer::SingleVecVar { name, vec_type } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(w, "{}{} {}{};", indent(depth), vec_type.name, name, epi)
            }
            CBuffer::VecVars { inner_vecs, .. } => {
                for inner_vec in inner_vecs {
                    inner_vec.emit(w, zero_init, depth)?;
                }
                Ok(())
            }
            CBuffer::ValueVar { name, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(w, "{}{} {}{};", indent(depth), c_type(*dtype), name, epi)
            }
        }
    }

    pub fn emit_free<W: fmt::Write>(&self, w: &mut W, depth: usize) -> fmt::Result {
        match self {
            CBuffer::HeapArray { name, .. } => {
                writeln!(w, "{}free({name});", indent(depth))?;
                Ok(())
            }
            CBuffer::StackArray { .. }
            | CBuffer::ValueVar { .. }
            | CBuffer::SingleVecVar { .. } => Ok(()),
            CBuffer::VecVars { inner_vecs, .. } => {
                for inner_vec in inner_vecs {
                    inner_vec.emit_free(w, depth)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn inner_vec_from_expr(&self, expr: &AffineExpr<CExprTerm>) -> (&CBuffer, usize) {
        let CBuffer::VecVars { inner_vecs, .. } = self else {
            unreachable!();
        };
        let CBuffer::SingleVecVar { name: _, vec_type } = inner_vecs[0] else {
            unreachable!();
        };
        let AffineExpr(ref linear_terms, expr_constant) = *expr;
        debug_assert!(linear_terms.is_empty());
        let expr_constant = usize::try_from(expr_constant).unwrap();
        let vector_size = usize::try_from(vec_type.value_cnt).unwrap();
        let inner_vec_idx = expr_constant / vector_size;
        let inside_vec_offset = expr_constant % vector_size;
        (&inner_vecs[inner_vec_idx], inside_vec_offset)
    }
}

pub fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 => "uint8_t",
        Dtype::Uint32 => "uint32_t",
    }
}

fn dot_product<T>(lhs: &[T], rhs: &[T]) -> T
where
    T: std::iter::Sum<<T as std::ops::Mul<T>>::Output> + std::ops::Mul<T> + Copy,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

/// Compute, for each dimension, the volume of all later dimensions.
fn compute_step_sizes(shape: &[DimSize]) -> Shape {
    // TODO: Do this in one pass over shape.
    (0..shape.len())
        .map(|dim| shape[dim + 1..].iter().product())
        .collect()
}
