use std::fmt;
use std::fmt::Debug;
use std::rc::Rc;

use super::CodeGen;
use crate::codegen::cpu::CpuCodeGenerator;
use crate::imp::Impl;
use crate::imp::ImplNode;
use crate::target::ArmTarget;
use crate::views::Tensor;

pub(crate) const CLI_VEC_FLAGS: [&'static str; 1] = ["-fopenmp"];

impl<Aux: Clone + Debug> CodeGen<ArmTarget> for ImplNode<ArmTarget, Aux> {
    fn emit<W: fmt::Write>(&self, out: &mut W) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<ArmTarget>::new();
        generator.emit_kernel(self, &top_arg_tensors, out)?;
        out.write_str("\n")?;
        generator.emit_main(&top_arg_tensors, out)?;
        Ok(())
    }
}
