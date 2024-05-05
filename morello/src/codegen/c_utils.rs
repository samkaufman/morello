use std::fmt;

use crate::{
    common::Dtype,
    expr::{AffineForm, Atom, Bounds, NonAffine},
    layout::BufferVar,
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
    Ptr {
        name: String,
        dtype: Dtype,
    },
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct VecType {
    pub dtype: Dtype,
    pub value_cnt: u8,
    pub name: &'static str,
    pub native_type_name: &'static str,
    pub load_fn: &'static str,
    pub load_fn_arg0: &'static str,
    pub store_fn: &'static str,
    pub store_fn_arg0: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CExprVar {
    Buffer(BufferVar),
    CName(String),
}

#[derive(Clone, Copy, PartialEq)]
pub enum InitType {
    None,
    Zero,
    Random,
}

impl CBuffer {
    /// Returns the C identifier name if the receiver has just one (i.e., is not distributed).
    pub fn name(&self) -> Option<&str> {
        match self {
            CBuffer::HeapArray { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::ValueVar { name, .. }
            | CBuffer::SingleVecVar { name, .. }
            | CBuffer::Ptr { name, .. } => Some(name),
            CBuffer::VecVars { .. } => None,
        }
    }

    pub fn size(&self) -> Option<u32> {
        match self {
            CBuffer::HeapArray { size, .. } | CBuffer::StackArray { size, .. } => Some(*size),
            _ => None,
        }
    }

    pub fn needs_unroll(&self) -> bool {
        matches!(self, CBuffer::VecVars { .. })
    }

    pub fn emit<W: fmt::Write>(&self, w: &mut W, init_type: InitType, depth: usize) -> fmt::Result {
        match self {
            CBuffer::HeapArray { name, size, dtype } => {
                writeln!(
                    w,
                    "{}{} *__restrict__ {};",
                    indent(depth),
                    c_type(*dtype),
                    name
                )?;
                writeln!(
                    w,
                    "{}posix_memalign((void **)&{}, 128, {}*sizeof({}));",
                    indent(depth),
                    name,
                    size,
                    c_type(*dtype)
                )?;

                match init_type {
                    InitType::Zero => {
                        writeln!(
                            w,
                            "{}memset({}, 0, {}*sizeof({}));",
                            indent(depth),
                            name,
                            size,
                            c_type(*dtype)
                        )
                    }
                    InitType::Random => self.emit_rand_init(w, depth, *size, name, *dtype),
                    _ => Ok(()),
                }
            }
            CBuffer::StackArray { name, size, dtype } => {
                let epi = if init_type == InitType::Zero {
                    " = {0}"
                } else {
                    ""
                };
                writeln!(
                    w,
                    "{}{} {}[{}] __attribute__((aligned (128))){};",
                    indent(depth),
                    c_type(*dtype),
                    name,
                    size,
                    epi
                )?;
                if init_type == InitType::Random {
                    self.emit_rand_init(w, depth, *size, name, *dtype)?;
                }
                Ok(())
            }
            CBuffer::SingleVecVar { name, vec_type } => {
                let epi = if init_type == InitType::Zero {
                    " = {0}"
                } else {
                    ""
                };
                writeln!(w, "{}{} {}{};", indent(depth), vec_type.name, name, epi)
            }
            CBuffer::VecVars { inner_vecs, .. } => {
                for inner_vec in inner_vecs {
                    inner_vec.emit(w, init_type, depth)?;
                }
                Ok(())
            }
            CBuffer::ValueVar { name, dtype } => {
                let epi = if init_type == InitType::Zero {
                    " = {0}"
                } else {
                    ""
                };
                writeln!(w, "{}{} {}{};", indent(depth), c_type(*dtype), name, epi)
            }
            CBuffer::Ptr { .. } => unimplemented!(),
        }
    }

    /// Emit a loop that initializes buffer values with `rand()`.
    fn emit_rand_init<W: fmt::Write>(
        &self,
        w: &mut W,
        depth: usize,
        size: u32,
        name: &str,
        dtype: Dtype,
    ) -> fmt::Result {
        writeln!(
            w,
            "{}for (size_t idx = 0; idx < {}; idx++) {{",
            indent(depth),
            size,
        )?;
        // Special-base bf16 to mitigate the need for rtlib support for bf16 truncation,
        // which is a partial workaround for a Clang issue:
        //   https://github.com/llvm/llvm-project/pull/84192
        if dtype == Dtype::Bfloat16 {
            writeln!(w, "{}float fv = (float)rand();", indent(depth + 1))?;
            writeln!(w, "{}{}[idx] = *(__bf16 *)(&fv);", indent(depth + 1), name)?;
        } else {
            writeln!(
                w,
                "{}{}[idx] = ({})rand();",
                indent(depth + 1),
                name,
                c_type(dtype)
            )?;
        }
        writeln!(w, "{}}}", indent(depth))
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
            CBuffer::Ptr { .. } => unimplemented!(),
        }
    }

    pub(super) fn inner_vec_from_expr(
        &self,
        expr: &AffineForm<NonAffine<CExprVar>>,
    ) -> (&CBuffer, usize) {
        let CBuffer::VecVars { inner_vecs, .. } = self else {
            unreachable!();
        };
        let CBuffer::SingleVecVar { name: _, vec_type } = inner_vecs[0] else {
            unreachable!();
        };
        let AffineForm(ref linear_terms, expr_constant) = *expr;
        debug_assert!(
            linear_terms.is_empty(),
            "Linear terms was non-empty: {:?}",
            linear_terms
        );
        let expr_constant = usize::try_from(expr_constant).unwrap();
        let vector_size = usize::from(vec_type.value_cnt);
        let inner_vec_idx = expr_constant / vector_size;
        let inside_vec_offset = expr_constant % vector_size;
        (&inner_vecs[inner_vec_idx], inside_vec_offset)
    }
}

impl Atom for CExprVar {}
impl Bounds for CExprVar {}

pub fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 => "uint8_t",
        Dtype::Sint8 => "int8_t",
        Dtype::Uint16 => "uint16_t",
        Dtype::Sint16 => "int16_t",
        Dtype::Uint32 => "uint32_t",
        Dtype::Sint32 => "int32_t",
        Dtype::Float32 => "float",
        Dtype::Bfloat16 => "__bf16",
    }
}

pub fn printf_fmt(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 => "%\" PRIu8 \"",
        Dtype::Sint8 => "%\" PRIi8 \"",
        Dtype::Uint16 => "%\" PRIu16 \"",
        Dtype::Sint16 => "%\" PRIi16 \"",
        Dtype::Uint32 => "%\" PRIu32 \"",
        Dtype::Sint32 => "%\" PRIi32 \"",
        Dtype::Float32 | Dtype::Bfloat16 => "%f",
    }
}
