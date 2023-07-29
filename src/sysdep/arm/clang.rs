// use crate::imp::ImplNode;
// use crate::sysdep::compiler::Compiler;
// use crate::target::ArmTarget;
//
// use crate::sysdep::clang::get_path;
// use anyhow::Result;
// use std::fmt::Debug;
//
// const NUM_VEC_FLAGS: usize = 1;
//
// impl<Aux: Clone + Debug> Compiler<X86Target, NUM_VEC_FLAGS> for ImplNode<ArmTarget, Aux> {
//     const CLI_VEC_FLAGS: [&'static str; NUM_VEC_FLAGS] = ["-fopenmp"];
//
//     fn get_path() -> Result<String> {
//         get_path()
//     }
// }
