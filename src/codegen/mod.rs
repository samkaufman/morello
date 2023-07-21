use std::fmt;

use crate::target::Target;

mod c_utils;
mod header;
mod namegen;
mod x86;

pub trait CodeGen<Tgt: Target> {
    fn emit_kernel<W: fmt::Write>(&self, out: &mut W) -> fmt::Result;
}
