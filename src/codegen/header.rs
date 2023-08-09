use super::c_utils::VecType;
use crate::target::{Target, TargetId};

use std::{collections::HashSet, fmt};

pub struct HeaderEmitter {
    pub emit_benchmarking: bool,
    pub vector_type_defs: HashSet<&'static VecType>,
}

impl HeaderEmitter {
    pub fn new() -> Self {
        Self {
            emit_benchmarking: false,
            vector_type_defs: HashSet::new(),
        }
    }

    pub fn emit<Tgt: Target, W: fmt::Write>(&self, out: &mut W) -> Result<(), fmt::Error> {
        out.write_str(include_str!("../codegen/partials/std.c"))?;
        out.write_char('\n')?;
        match Tgt::target_id() {
            TargetId::X86 => {
                out.write_str(include_str!("../codegen/partials/x86.c"))?;
            }
            TargetId::Arm => {
                out.write_str(include_str!("../codegen/partials/arm.c"))?;
            }
        }
        out.write_char('\n')?;
        if self.emit_benchmarking {
            out.write_str(include_str!("../codegen/partials/benchmarking.c"))?;
            out.write_char('\n')?;
        }

        for vec_type in &self.vector_type_defs {
            // Declare a vector of {vec_bytes} bytes, divided into {dt.c_type}
            // values. (vec_bytes must be a multiple of the c_type size.)
            out.write_str(&format!(
                "typedef {} {} __attribute__ ((vector_size ({} * sizeof({}))));\n",
                vec_type.dtype.c_type(),
                vec_type.name,
                vec_type.value_cnt,
                vec_type.dtype.c_type()
            ))?;
        }
        if !self.vector_type_defs.is_empty() {
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
