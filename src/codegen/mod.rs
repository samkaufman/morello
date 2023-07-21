use crate::syntax::ColorMode;
use crate::target::Target;

use std::fmt;

mod c_utils;
mod header;
mod namegen;
mod x86;

pub trait CodeGen<Tgt: Target> {
    fn emit_kernel<W: fmt::Write>(&self, out: &mut W, color: ColorMode) -> fmt::Result;
}
