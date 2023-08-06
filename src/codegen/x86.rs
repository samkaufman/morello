use std::fmt;
use std::fmt::Debug;
use std::rc::Rc;

use super::CodeGen;
use crate::codegen::cpu::CpuCodeGenerator;
use crate::imp::Impl;
use crate::imp::ImplNode;
use crate::target::X86Target;
use crate::views::Tensor;

const NUM_VEC_FLAGS: usize = 2;

impl<Aux: Clone + Debug> CodeGen<X86Target, NUM_VEC_FLAGS> for ImplNode<X86Target, Aux> {
    const CLI_VEC_FLAGS: [&'static str; NUM_VEC_FLAGS] = ["-fopenmp", "-mavx2"];

    fn emit<W: fmt::Write>(&self, out: &mut W) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<X86Target>::new();
        generator.emit_kernel(self, &top_arg_tensors, out)?;
        out.write_str("\n")?;
        generator.emit_main(&top_arg_tensors, out)?;
        Ok(())
    }
}
