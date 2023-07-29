use crate::imp::ImplNode;
use crate::sysdep::clang::get_path;
use crate::sysdep::compiler::Compiler;
use crate::target::X86Target;

use anyhow::Result;
use std::fmt::Debug;

const NUM_VEC_FLAGS: usize = 2;

impl<Aux: Clone + Debug> Compiler<X86Target, NUM_VEC_FLAGS> for ImplNode<X86Target, Aux> {
    const CLI_VEC_FLAGS: [&'static str; NUM_VEC_FLAGS] = ["-fopenmp", "-mavx2"];

    fn get_path() -> Result<String> {
        get_path()
    }
}
