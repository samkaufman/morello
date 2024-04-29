use super::c_utils::{c_type, VecType};
use crate::target::TargetId;

use std::{collections::HashSet, fmt};

pub struct HeaderEmitter {
    pub emit_benchmarking: bool,
    pub vector_type_defs: HashSet<&'static VecType>,
    pub emit_stdbool_and_assert_headers: bool,
}

impl HeaderEmitter {
    pub fn new() -> Self {
        Self {
            emit_benchmarking: false,
            vector_type_defs: HashSet::new(),
            emit_stdbool_and_assert_headers: false,
        }
    }

    pub fn emit<W: fmt::Write>(&self, target: TargetId, out: &mut W) -> Result<(), fmt::Error> {
        out.write_str(include_str!("../codegen/partials/std.c"))?;
        out.write_char('\n')?;
        if self.emit_stdbool_and_assert_headers {
            out.write_str("#include <assert.h>\n#include <stdbool.h>\n")?;
        }
        match target {
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
                "typedef {0} {1} __attribute__ ((vector_size ({2} * sizeof({0}))));\n",
                c_type(vec_type.dtype),
                vec_type.name,
                vec_type.value_cnt,
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
