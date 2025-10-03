use super::c_utils::{c_type, VecType};
use crate::target::TargetId;

use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Default)]
pub struct HeaderEmitter {
    pub emit_benchmarking: bool,
    pub vector_type_defs: HashSet<&'static VecType>,
    pub emit_stdbool_and_assert_includes: bool,
    pub emit_math_include: bool,  // math.h
    pub emit_float_include: bool, // float.h
    pub emit_expf_avx2: bool,
    pub emit_cores_clamp: bool,
    pub emit_sum8: bool,
    pub emit_cvtbf16_fp32: bool,
    pub emit_max: bool,
    pub benchmark_counters: BTreeMap<String, String>,
}

impl HeaderEmitter {
    pub fn emit<W: fmt::Write>(&self, target: TargetId, out: &mut W) -> Result<(), fmt::Error> {
        out.write_str(include_str!("../codegen/partials/std.c"))?;
        out.write_char('\n')?;
        if self.emit_stdbool_and_assert_includes {
            out.write_str("#include <assert.h>\n#include <stdbool.h>\n")?;
        }
        if self.emit_math_include {
            out.write_str("#include <math.h>\n")?;
        }
        if self.emit_float_include {
            out.write_str("#include <float.h>\n")?;
        }
        if !self.benchmark_counters.is_empty() {
            out.write_str("#include <stdatomic.h>\n")?;
        }
        match target {
            TargetId::Avx2 | TargetId::Avx512 => {
                out.write_str(include_str!("../codegen/partials/x86.c"))?;
            }
            TargetId::Arm => {
                out.write_str(include_str!("../codegen/partials/arm.c"))?;
            }
        }
        if self.emit_cores_clamp {
            out.write_str(include_str!("../codegen/partials/x86/cores_clamp.c"))?;
        }
        if self.emit_sum8 {
            out.write_str(include_str!("../codegen/partials/x86/sum8.c"))?;
        }
        if self.emit_cvtbf16_fp32 {
            out.write_str(include_str!("../codegen/partials/x86/cvtbf16_fp32.c"))?;
        }
        if self.emit_max {
            out.write_str(include_str!("../codegen/partials/x86/max.c"))?;
        }
        if self.emit_expf_avx2 {
            out.write_str(include_str!("../codegen/partials/x86/expf_avx2.c"))?;
        }
        out.write_char('\n')?;
        if self.emit_benchmarking {
            out.write_str(include_str!("../codegen/partials/benchmarking.c"))?;
            out.write_char('\n')?;
        }

        if !self.benchmark_counters.is_empty() {
            for counter_var in self.benchmark_counters.values() {
                out.write_str(&format!("static _Atomic long long {counter_var} = 0;\n"))?;
            }
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

    /// Return a unique C name for an Impl benchmark counter name.
    pub fn register_benchmark_counter(&mut self, counter: &str) -> String {
        self.emit_benchmarking = true;
        if let Some(existing) = self.benchmark_counters.get(counter) {
            return existing.clone();
        }

        let base_ident = sanitize_counter_identifier(counter);
        let mut candidate = base_ident.clone();
        let mut disambiguator = 1;
        while self
            .benchmark_counters
            .values()
            .any(|existing| existing == &candidate)
        {
            candidate = format!("{base_ident}_{disambiguator}");
            disambiguator += 1;
        }

        self.benchmark_counters
            .insert(counter.to_owned(), candidate.clone());
        candidate
    }
}

fn sanitize_counter_identifier(counter: &str) -> String {
    let mut ident = String::from("benchmark_");
    for ch in counter.chars() {
        if ch.is_ascii_alphanumeric() {
            ident.push(ch.to_ascii_lowercase());
        } else {
            ident.push('_');
        }
    }
    if ident.ends_with('_') {
        ident.push('0');
    }
    ident
}
