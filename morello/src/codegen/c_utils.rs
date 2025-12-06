use std::collections::HashMap;
use std::{fmt, ops::Rem};

use itertools::Either;

use crate::{
    common::Dtype,
    expr::{AffineForm, Atom, Bounds, NonAffineExpr},
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
    RegVars {
        inner_vecs: Vec<(String, Either<Dtype, &'static VecType>)>,
    },
    Ptr {
        name: String,
    },
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct VecType {
    pub dtype: Dtype,
    pub value_cnt: u8,
    pub name: &'static str,
    pub native_type_name: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CName(pub String);

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
            | CBuffer::Ptr { name, .. } => Some(name),
            CBuffer::RegVars { inner_vecs } if inner_vecs.len() == 1 => Some(&inner_vecs[0].0),
            CBuffer::RegVars { .. } => None,
        }
    }

    pub fn needs_unroll(&self) -> bool {
        matches!(self, CBuffer::RegVars { .. })
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
            CBuffer::RegVars { inner_vecs, .. } => {
                let epi = if init_type == InitType::Zero {
                    " = {0}"
                } else {
                    ""
                };
                let mut groups: HashMap<Either<Dtype, &'static VecType>, Vec<&String>> =
                    HashMap::new();
                for (name, typ) in inner_vecs {
                    groups.entry(*typ).or_default().push(name);
                }
                for (typ, names) in groups {
                    let type_name = match typ {
                        Either::Left(dtype) => c_type(dtype),
                        Either::Right(vec_type) => vec_type.name,
                    };
                    write!(w, "{}{}", indent(depth), type_name)?;
                    for (i, name) in names.iter().enumerate() {
                        let prefix = if i == 0 { " " } else { ", " };
                        write!(w, "{prefix}{name}{epi}")?;
                    }
                    writeln!(w, ";")?;
                }
                Ok(())
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
            CBuffer::StackArray { .. } | CBuffer::RegVars { .. } => Ok(()),
            CBuffer::Ptr { .. } => unimplemented!(),
        }
    }

    /// Returns a specific register name and its internal offset.
    ///
    /// This checks the upper and lower bounds of the given expression. If they both fall inside the
    /// same single register, this returns that register. Otherwise, it panics.
    pub(super) fn inner_reg_from_expr<T>(
        &self,
        expr: &NonAffineExpr<T>,
    ) -> (String, NonAffineExpr<T>)
    where
        T: Bounds + Clone + fmt::Debug,
        NonAffineExpr<T>: Rem<i32, Output = NonAffineExpr<T>>,
    {
        let AffineForm(linear_terms, _) = expr;
        assert!(
            linear_terms.is_empty(),
            "unexpectedly contained linear terms: {expr:?}",
        );
        let CBuffer::RegVars { inner_vecs } = self else {
            unreachable!();
        };
        let Some((bmin, bmax)) = expr.bounds() else {
            panic!("expr's bounds are undefined: {expr:?}");
        };

        match inner_vecs[0] {
            (_, Either::Left(_)) => {
                debug_assert!(inner_vecs.iter().all(|(_, t)| matches!(t, Either::Left(_))));
                assert_eq!(bmin, bmax, "expr spans multiple vectors: {expr:?}");
                let (iv_name, _) = &inner_vecs[usize::try_from(bmin).unwrap()];
                (iv_name.clone(), NonAffineExpr::zero())
            }
            (_, Either::Right(vec_type)) => {
                debug_assert!(inner_vecs
                    .iter()
                    .all(|(_, t)| matches!(t, Either::Right(_))));

                let vector_size = i32::from(vec_type.value_cnt);
                let min_vec_idx = bmin / vector_size;
                let max_vec_idx = bmax / vector_size;
                assert_eq!(
                    min_vec_idx, max_vec_idx,
                    "expr spans multiple vectors: {expr:?}"
                );

                let (iv_name, _) = &inner_vecs[usize::try_from(min_vec_idx).unwrap()];
                (iv_name.clone(), expr.clone() % vector_size)
            }
        }
    }
}

impl Atom for CName {}
impl Bounds for CName {}

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
        Dtype::Float32 | Dtype::Bfloat16 => "%.10f",
    }
}
