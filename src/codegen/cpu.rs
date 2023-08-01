use itertools::Either;
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;

use super::namegen::NameGenerator;
use crate::codegen::c_utils::CBuffer;
use crate::codegen::header::HeaderEmitter;
use crate::layout::BufferExprTerm;
use crate::target::{ArmTarget, Target, Targets, X86Target};
use crate::views::{Param, Tensor, View};

#[derive(Default)]
pub struct CpuCodeGenerator<'a, Tgt: Target> {
    pub namer: NameGenerator,
    pub name_env: HashMap<Rc<Tensor<Tgt>>, CBuffer>,
    pub loop_iter_bindings: HashMap<BufferExprTerm, Either<String, i32>>,
    pub param_bindings: HashMap<Param<Tgt>, &'a dyn View<Tgt = Tgt>>,
    pub headers: HeaderEmitter,
}

impl<'a> CpuCodeGenerator<'a, X86Target> {
    pub fn new() -> Self {
        Self {
            headers: HeaderEmitter::new(Targets::X86),
            ..Self::default()
        }
    }
}

impl<'a> CpuCodeGenerator<'a, ArmTarget> {
    pub fn new() -> Self {
        Self {
            headers: HeaderEmitter::new(Targets::Arm),
            ..Self::default()
        }
    }
}
