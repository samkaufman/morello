use std::fmt;

pub struct HeaderEmitter {
    pub emit_x86: bool,
    pub emit_arm: bool,
    pub emit_benchmarking: bool,
    pub vector_type_defs: Vec<VectorDef>,
}

pub struct VectorDef {
    pub c_type: String,
    pub name: String,
    pub vec_bytes: String,
}

impl HeaderEmitter {
    pub fn new() -> Self {
        Self {
            emit_x86: false,
            emit_arm: false,
            emit_benchmarking: false,
            vector_type_defs: vec![],
        }
    }

    pub fn emit<W: fmt::Write>(&self, out: &mut W) -> Result<(), fmt::Error> {
        out.write_str(include_str!("../codegen_partials/std.c"))?;
        out.write_char('\n')?;
        if self.emit_x86 {
            out.write_str(include_str!("../codegen_partials/x86.c"))?;
            out.write_char('\n')?;
        }
        if self.emit_arm {
            out.write_str(include_str!("../codegen_partials/arm.c"))?;
            out.write_char('\n')?;
        }
        if self.emit_benchmarking {
            out.write_str(include_str!("../codegen_partials/benchmarking.c"))?;
            out.write_char('\n')?;
        }
        Ok(())
    }
}

impl Default for HeaderEmitter {
    fn default() -> Self {
        HeaderEmitter::new()
    }
}
