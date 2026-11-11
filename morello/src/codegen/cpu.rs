use itertools::{Either, Itertools};
use nonzero::nonzero as nz;
use std::collections::HashMap;
use std::fmt::{self, Write};
use std::iter;
use std::marker::PhantomData;
use std::rc::Rc;

use super::c_utils::{c_type, printf_fmt, CBuffer, CName, InitType, VecType};
use super::header::HeaderEmitter;
use super::namegen::NameGenerator;
use super::planmem::{StaticAllocationPlan, WorkspaceId};
use super::vec_accums::{VecAccums, VecReduce};
use super::ENABLE_STATIC_ALLOCATION;
use crate::common::{DimSize, Dtype};
use crate::expr::{AffineForm, Bounds as _, NonAffine, NonAffineExpr, Substitute, Term};
use crate::imp::blocks::Block;
use crate::imp::kernels::KernelApp;
use crate::imp::loops::{unique_dims_per_axis, Loop};
use crate::imp::pipeline::Pipeline;
use crate::imp::{Impl, ImplNode};
use crate::layout::BufferVar;
use crate::opaque_symbol::OpaqueSymbol;
use crate::pprint::{pprint_write, ImplPrintStyle};
use crate::shape;
use crate::spec::OperandDirection;
use crate::target::{
    cpu::{
        broadcastvecmult_side, divide_vec_scalar_reciprocal_vector_size, softmax_vector_size,
        vector_accum_count, vector_max_loop_properties, DOT_PRODUCT_BF16_STRIP_SIZE,
        DOT_PRODUCT_STRIP_SIZE, VECTOR_ACCUM_COUNT,
    },
    Avx512Kernel, CpuKernel, CpuMemory, CpuTarget, Kernel, Target, TargetId,
    DOT_PRODUCT_VDPBF16PS_STRIP_SIZE,
};
use crate::utils::{ascii_name, indent, LinePrefixWrite};
use crate::views::{Tensor, View, ViewE};

/// Alignment used for addressed heap buffers emitted by CPU codegen.
pub const HEAP_BUFFER_ALIGNMENT_BYTES: u64 = 128;
/// Maximum addressed buffer size that still stays on the stack.
pub const STACK_CUTOFF: u32 = 256;

pub struct CpuCodeGenerator<Tgt: Target> {
    pub namer: NameGenerator,
    pub name_env: HashMap<OpaqueSymbol, CBuffer>,
    pub loop_iter_bindings: HashMap<BufferVar, Either<String, i32>>,
    pub headers: HeaderEmitter,
    pub kernel_name: String,
    pub thread_style: CpuCodeGenThreadStyle,
    pub benchmark: bool,
    static_allocation_plan: Option<StaticAllocationPlan<Tgt>>,
    /// Maps each currently emitted workspace scope to its fresh C variable name.
    active_workspace_names: HashMap<WorkspaceId, String>,
    phantom: PhantomData<Tgt>,
}

#[derive(Default)]
pub enum CpuCodeGenThreadStyle {
    #[default]
    OpenMP,
    Highway, // TODO: Generalize to plug-in codelets
}

impl<Tgt: CpuTarget> CpuCodeGenerator<Tgt> {
    pub fn new(benchmark: bool) -> Self {
        Self {
            benchmark,
            ..Self::default()
        }
    }

    fn require_x86_f32_exp_helper(&mut self, vector_size: DimSize) -> &'static str {
        match vector_size.get() {
            4 => {
                self.headers.emit_expf_avx2 = true;
                "exp128_ps"
            }
            8 => {
                self.headers.emit_expf_avx2 = true;
                "exp256_ps"
            }
            16 => {
                self.headers.emit_expf_avx512 = true;
                "exp512_ps"
            }
            _ => panic!("Unsupported x86 float32 exp helper width {vector_size}"),
        }
    }

    fn require_x86_f32_sum_helper(&mut self, vector_size: DimSize) -> &'static str {
        match vector_size.get() {
            4 => {
                self.headers.emit_sum8 = true;
                "sum4"
            }
            8 => {
                self.headers.emit_sum8 = true;
                "sum8"
            }
            16 => "_mm512_reduce_add_ps",
            _ => panic!("Unsupported x86 float32 sum helper width {vector_size}"),
        }
    }

    fn require_x86_f32_horizontal_max_helper(&mut self, vector_size: DimSize) -> &'static str {
        match vector_size.get() {
            4 => {
                self.headers.emit_horizontal_max4 = true;
                "horizontal_max4_f32"
            }
            8 => {
                self.headers.emit_horizontal_max8 = true;
                "horizontal_max_f32"
            }
            16 => "_mm512_reduce_max_ps",
            _ => panic!("Unsupported x86 float32 max helper width {vector_size}"),
        }
    }

    /// Write a pretty-printed Impl as a C comment.
    pub fn emit_impl_comment<W: Write>(
        &mut self,
        imp: &ImplNode<Tgt>,
        impl_style: ImplPrintStyle,
        out: &mut W,
    ) -> fmt::Result {
        let mut commenting_out = LinePrefixWrite::new(out, "// ");
        pprint_write(&mut commenting_out, imp, impl_style)?;
        Ok(())
    }

    pub fn emit_kernel<W: Write>(
        &mut self,
        imp: &ImplNode<Tgt>,
        top_arg_tensors: &[Rc<Tensor<Tgt>>],
        bench: bool,
        out: &mut W,
    ) -> fmt::Result {
        let mut main_body_str = String::new();
        writeln!(
            main_body_str,
            "__attribute__((noinline))\nvoid {}(",
            self.kernel_name
        )?;

        let thread_extra_args = self.thread_style_extra_args();
        let parameter_count = top_arg_tensors.len();
        let fn_arg_count = parameter_count + thread_extra_args.len();
        let parameter_directions = imp.spec().map(|spec| spec.0.operand_directions());
        debug_assert!(parameter_directions
            .as_ref()
            .is_none_or(|d| d.len() == parameter_count));

        let mut operand_idx = 0;
        for tensor in top_arg_tensors {
            // Use the Spec's operand directions when available. Without a Spec,
            // conservatively allow every parameter to be written.
            // TODO: It'd be better to do a quick analysis of the Impl.
            let is_read_only = parameter_directions
                .as_ref()
                .is_some_and(|directions| directions[operand_idx] == OperandDirection::In);
            let parameter_name = self.namer.fresh_name();
            writeln!(
                main_body_str,
                "  {}{} *__restrict__ {}{}",
                if is_read_only { "const " } else { "" },
                c_type(tensor.spec().dtype()),
                parameter_name,
                if operand_idx + 1 < fn_arg_count {
                    ", "
                } else {
                    "\n) {"
                }
            )?;
            self.name_env.insert(
                tensor.identifier(),
                CBuffer::Ptr {
                    name: parameter_name,
                },
            );
            operand_idx += 1;
        }

        for extra_arg in self.thread_style_extra_args() {
            writeln!(
                main_body_str,
                "  {extra_arg}{}",
                if operand_idx + 1 < fn_arg_count {
                    ", "
                } else {
                    "\n) {"
                }
            )?;
            operand_idx += 1;
        }

        // Put the tensor->c_buffer binding into `self.name_env`. (And fill tensors_as_ptrs.)
        let tensors_as_viewe = top_arg_tensors
            .iter()
            .map(|tensor| ViewE::from(Rc::as_ref(tensor).clone()))
            .collect::<Vec<_>>();

        let beta_reduced_imp = imp
            .clone()
            .bind(&mut |param_idx| tensors_as_viewe.get(usize::from(param_idx)).cloned());
        if ENABLE_STATIC_ALLOCATION {
            self.static_allocation_plan = Some(StaticAllocationPlan::plan(&beta_reduced_imp));
        }
        self.emit(&mut main_body_str, &beta_reduced_imp, 1)?;

        writeln!(main_body_str, "}}")?;

        self.headers.emit(Tgt::target_id(), out)?;
        out.write_char('\n')?;
        if bench && !self.headers.emit_benchmarking {
            out.write_str(include_str!("../codegen/partials/benchmarking.c"))?;
            out.write_str("\n\n")?;
        }
        out.write_str(&main_body_str)
    }

    pub fn emit_load_inputs<W: Write>(
        &mut self,
        top_arg_tensors: &[Rc<Tensor<Tgt>>],
        out: &mut W,
    ) -> fmt::Result {
        if top_arg_tensors
            .iter()
            .any(|t| !t.spec().layout().is_row_major())
        {
            todo!("Support changing layout for non-row-major tensors");
        }

        write!(out, "int load_inputs(char *paths[]")?;
        for i in 0..top_arg_tensors.len() {
            write!(out, ", void *__restrict__ dest{i}")?;
        }
        writeln!(out, ") {{")?;

        cline!(out, 1, "int fd;")?;
        cline!(out, 1, "void *mapped;")?;

        for (idx, input_tensor) in top_arg_tensors.iter().enumerate() {
            // Open and mmap the data.
            let value_cnt = input_tensor.0.volume().get();
            let byte_cnt = value_cnt * u32::from(input_tensor.spec().dtype().size());
            writeln!(out)?;
            cline!(out, 1, "if ((fd = open(paths[{idx}], O_RDONLY)) == -1)")?;
            cline!(out, 2, "return 1;")?;
            cline!(
                out,
                1,
                "if ((mapped = mmap(NULL, {byte_cnt}, PROT_READ, MAP_SHARED, fd, 0)) == MAP_FAILED)"
            )?;
            cline!(out, 2, "return 2;")?;
            cline!(out, 1, "close(fd);")?;

            // Move into destination argument.
            let dtype = input_tensor.spec().dtype();
            cline!(out, 1, "for (int i = 0; i < {value_cnt}; i++) {{")?;
            emit_load_input_value(out, idx, dtype)?;
            cline!(out, 1, "}}")?;

            // Un-map.
            cline!(out, 1, "if (munmap(mapped, {byte_cnt}) != 0)")?;
            cline!(out, 2, "return 3;")?;
        }

        cline!(out, 1, "return 0;")?;
        writeln!(out, "}}")
    }

    pub fn emit_main<W: Write>(
        &mut self,
        top_arg_tensors: &[Rc<Tensor<Tgt>>],
        out: &mut W,
    ) -> fmt::Result {
        if self.benchmark {
            self.emit_benchmarking_main(top_arg_tensors, out)
        } else {
            self.emit_standard_main(top_arg_tensors, out)
        }
    }

    fn emit_standard_main<W: Write>(
        &mut self,
        top_arg_tensors: &[Rc<Tensor<Tgt>>],
        out: &mut W,
    ) -> fmt::Result {
        let mut main_body_str = String::new();
        let mut depth = 0_usize;

        writeln!(main_body_str, "int main(int argc, char *argv[]) {{")?;
        depth += 1;

        let counter_vars: Vec<String> = self.headers.benchmark_counters.values().cloned().collect();
        if !counter_vars.is_empty() {
            for counter_var in &counter_vars {
                cline!(
                    main_body_str,
                    depth,
                    "atomic_store_explicit(&{}, 0, memory_order_seq_cst);",
                    counter_var
                )?;
            }
            writeln!(main_body_str)?;
        }

        // Allocate a buffer for each Impl parameter and re-bind to a CBuffer corresponding to the
        // local-scope buffer. It will have been previously bound by emit_kernel to a CBuffer::Ptr.
        let mut parameter_buf_names = vec![];
        for kernel_argument in top_arg_tensors {
            let spec = kernel_argument.spec();
            let buf = self.make_buffer(
                Some(kernel_argument.identifier()),
                spec.shape(),
                spec.vector_size(),
                spec.dtype(),
                spec.memory(),
            );
            buf.emit(&mut main_body_str, InitType::Zero, depth)?;
            parameter_buf_names.push(self.c_index_ptr(&buf, &NonAffineExpr::zero()));
            self.name_env.insert(kernel_argument.identifier(), buf);
            writeln!(main_body_str)?;
        }

        // Load data, if provided.
        let full_argc = top_arg_tensors.len() + 1;
        cline!(main_body_str, depth, "if (argc == {}) {{", full_argc)?;
        depth += 1;
        cline!(
            main_body_str,
            depth,
            "int load_result = load_inputs(&argv[1]{});",
            parameter_buf_names
                .iter()
                .map(|n| format!(", {n}"))
                .join("")
        )?;
        cline!(main_body_str, depth, "if (load_result != 0) {{")?;
        depth += 1;
        cline!(
            main_body_str,
            depth,
            "fprintf(stderr, \"Error loading input tensors.\\n\");"
        )?;
        cline!(main_body_str, depth, "return 2;")?;
        depth -= 1;
        cline!(main_body_str, depth, "}}")?;
        depth -= 1;
        cline!(main_body_str, depth, "}} else if (argc != 1) {{")?;
        depth += 1;
        cline!(
            main_body_str,
            depth,
            "fprintf(stderr, \"Unexpected number of arguments.\\n\");"
        )?;
        cline!(main_body_str, depth, "return 1;")?;
        depth -= 1;
        cline!(main_body_str, depth, "}}")?;

        let kernel_call_str = self.make_kernel_call(top_arg_tensors)?;
        cline!(main_body_str, depth, "{}\n", kernel_call_str)?;
        self.emit_print_tensor(top_arg_tensors.last().unwrap(), depth, &mut main_body_str)?;
        writeln!(main_body_str)?;

        let counter_entries: Vec<(String, String)> = self
            .headers
            .benchmark_counters
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        if !counter_entries.is_empty() {
            for (idx, (counter_name, counter_var)) in counter_entries.iter().enumerate() {
                let value_var = format!("{counter_var}_value_{idx}");
                let escaped_name = escape_c_string(counter_name);
                cline!(main_body_str, depth, "{{")?;
                cline!(
                    main_body_str,
                    depth + 1,
                    "long long {value_var} = atomic_load_explicit(&{}, memory_order_seq_cst);",
                    counter_var
                )?;
                cline!(
                    main_body_str,
                    depth + 1,
                    "printf(\"benchmark[%s]: %lld ns (%.6f s)\\n\", \"{}\", {value_var}, (double){value_var} / 1000000000.0);",
                    escaped_name
                )?;
                cline!(main_body_str, depth, "}}")?;
            }
            writeln!(main_body_str)?;
        }

        // Free the buffers.
        for kernel_argument in top_arg_tensors {
            let buf = self.name_env.get(&kernel_argument.identifier()).unwrap();
            buf.emit_free(&mut main_body_str, depth)?;
        }

        writeln!(main_body_str, "\n{}return 0;", indent(depth))?;
        writeln!(main_body_str, "}}")?;

        out.write_str(&main_body_str)
    }

    fn emit_benchmarking_main<W: Write>(
        &mut self,
        top_arg_tensors: &[Rc<Tensor<Tgt>>],
        out: &mut W,
    ) -> fmt::Result {
        let mut main_body_str = String::new();
        let mut depth = 0_usize;

        writeln!(main_body_str, "int main(int argc, char *argv[]) {{")?;
        depth += 1;

        let counter_vars: Vec<String> = self.headers.benchmark_counters.values().cloned().collect();
        if !counter_vars.is_empty() {
            for counter_var in &counter_vars {
                cline!(
                    main_body_str,
                    depth,
                    "atomic_store_explicit(&{}, 0, memory_order_seq_cst);",
                    counter_var
                )?;
            }
            writeln!(main_body_str)?;
        }

        // Allocate a buffer for each Impl parameter and re-bind to a CBuffer corresponding to the
        // local-scope buffer. It will have been previously bound by emit_kernel to a CBuffer::Ptr.
        let mut parameter_buf_names = vec![];
        for kernel_argument in top_arg_tensors {
            let spec = kernel_argument.spec();
            let buf = self.make_buffer(
                Some(kernel_argument.identifier()),
                spec.shape(),
                spec.vector_size(),
                spec.dtype(),
                spec.memory(),
            );
            buf.emit(&mut main_body_str, InitType::Random, depth)?;
            parameter_buf_names.push(self.c_index_ptr(&buf, &NonAffineExpr::zero()));
            self.name_env.insert(kernel_argument.identifier(), buf);
            writeln!(main_body_str)?;
        }

        let kernel_call_str = self.make_kernel_call(top_arg_tensors)?;
        self.emit_benchmarking(&kernel_call_str, depth, &mut main_body_str)?;
        writeln!(main_body_str)?;

        let counter_entries: Vec<(String, String)> = self
            .headers
            .benchmark_counters
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        if !counter_entries.is_empty() {
            for (idx, (counter_name, counter_var)) in counter_entries.iter().enumerate() {
                let value_var = format!("{counter_var}_value_{idx}");
                let escaped_name = escape_c_string(counter_name);
                cline!(main_body_str, depth, "{{")?;
                cline!(
                    main_body_str,
                    depth + 1,
                    "long long {value_var} = atomic_load_explicit(&{}, memory_order_seq_cst);",
                    counter_var
                )?;
                cline!(
                    main_body_str,
                    depth + 1,
                    "printf(\"benchmark[%s]: %lld ns (%.6f s)\\n\", \"{}\", {value_var}, (double){value_var} / 1000000000.0);",
                    escaped_name
                )?;
                cline!(main_body_str, depth, "}}")?;
            }
            writeln!(main_body_str)?;
        }

        // Free the buffers.
        for kernel_argument in top_arg_tensors {
            let buf = self.name_env.get(&kernel_argument.identifier()).unwrap();
            buf.emit_free(&mut main_body_str, depth)?;
        }

        writeln!(main_body_str, "\n{}return 0;", indent(depth))?;
        writeln!(main_body_str, "}}")?;

        out.write_str(&main_body_str)
    }

    fn make_kernel_call(&self, top_arg_tensors: &[Rc<Tensor<Tgt>>]) -> Result<String, fmt::Error> {
        let mut kernel_call_str = String::new();
        write!(kernel_call_str, "{}(", self.kernel_name)?;
        for (i, kernel_argument) in top_arg_tensors.iter().enumerate() {
            let a = self.name_env.get(&kernel_argument.identifier()).unwrap();
            write!(
                kernel_call_str,
                "{}{}",
                self.c_index_ptr(a, &AffineForm::zero()),
                if i < top_arg_tensors.len() - 1 {
                    ", "
                } else {
                    ");"
                },
            )?;
        }
        Ok(kernel_call_str)
    }

    fn emit_benchmarking<W: Write>(
        &mut self,
        kernel_call_str: &str,
        mut depth: usize,
        out: &mut W,
    ) -> Result<(), fmt::Error> {
        cline!(
            out,
            depth,
            "const long long bench_samples = atoll(argv[1]);\n"
        )?;

        cline!(
            out,
            depth,
            "// Inlined kernel follows. This is for warm-up."
        )?;
        cline!(out, depth, "{}", kernel_call_str)?;

        cline!(out, depth, "struct timespec start, end;")?;
        cline!(out, depth, "clock_gettime(CLOCK_MONOTONIC, &start);")?;
        writeln!(out, "#pragma clang loop unroll(disable)")?; // preprocessor directives should not have indentation.
        cline!(
            out,
            depth,
            "for (long long bench_itr = 0; bench_itr < bench_samples; ++bench_itr) {{"
        )?;
        depth += 1;
        cline!(out, depth, "{}", kernel_call_str)?;
        depth -= 1;
        cline!(out, depth, "}}")?;
        cline!(out, depth, "clock_gettime(CLOCK_MONOTONIC, &end);")?;
        cline!(out, depth, "struct timespec delta = ts_diff(start, end);")?;
        cline!(
            out,
            depth,
            "printf(\"cpu: %llds %lldns\\n\", (long long)delta.tv_sec, (long long)delta.tv_nsec);"
        )?;

        Ok(())
    }

    fn emit_print_tensor<W: Write>(
        &mut self,
        tensor: &Tensor<Tgt>,
        mut depth: usize,
        out: &mut W,
    ) -> Result<(), fmt::Error> {
        let buffer = self.name_env.get(&tensor.identifier()).unwrap();
        if buffer.needs_unroll() {
            unimplemented!();
        }

        let rank = tensor.shape().len();

        let shape_str = tensor.shape().iter().map(ToString::to_string).join("x");
        cline!(out, depth, "printf(\"{}\\n\");", shape_str)?;

        for (dim, d) in tensor.shape().iter().copied().enumerate() {
            let n = ascii_name(dim);
            cline!(out, depth, "for (int {n} = 0; {n} < {d}; {n}++) {{")?;
            let insertion_result = self.loop_iter_bindings.insert(
                BufferVar::Pt(dim.try_into().unwrap(), tensor.identifier()),
                Either::Left(n.to_string()),
            );
            debug_assert!(insertion_result.is_none());
        }
        depth += 1;

        debug_assert_eq!(tensor, tensor.backing_tensor().unwrap());
        let buffer_indexing_expr = tensor.make_buffer_indexing_expr();
        cline!(
            out,
            depth,
            "printf(\"{} \", {});",
            printf_fmt(tensor.spec().dtype()),
            self.c_index(buffer, &buffer_indexing_expr)
        )?;
        depth -= 1;

        cline!(out, depth, "}}")?;
        for _ in 0..rank - 1 {
            cline!(out, depth, "printf(\"\\n\");")?;
            cline!(out, depth, "}}")?;
        }
        Ok(())
    }

    fn emit_avx512_kernel<W: Write>(
        &mut self,
        w: &mut W,
        kernel: Avx512Kernel,
        arguments: &[ViewE<Tgt>],
        depth: usize,
    ) -> fmt::Result {
        match kernel {
            Avx512Kernel::Cpu(_) => unreachable!("CPU kernels should use the generic emitter"),
            Avx512Kernel::DotProductLoopVdpbf16ps => {
                let out_expr = self.c_value_arg(&arguments[2]);
                let k = arguments[0].spec().shape()[2].get();
                let accum_count = vector_accum_count(k, DOT_PRODUCT_VDPBF16PS_STRIP_SIZE.get())
                    .expect("AVX512 VDPBF16PS applies should have ensured accum_count");
                let accum_names = self
                    .namer
                    .fresh_names(usize::try_from(accum_count).unwrap());

                cline!(w, depth, "// DotProductLoopVdpbf16ps")?;
                for accum_name in &accum_names {
                    cline!(w, depth, "__m512 {accum_name} = _mm512_setzero_ps();")?;
                }
                let strip_loop = self.strip_loop(
                    DOT_PRODUCT_VDPBF16PS_STRIP_SIZE,
                    accum_count,
                    [&arguments[0], &arguments[1]],
                );
                strip_loop.emit(w, depth, |w, depth, lane, [lhs, rhs]| {
                    let accum_name = &accum_names[lane];
                    cline!(
                        w,
                        depth,
                        "{accum_name} = _mm512_dpbf16_ps({accum_name}, (__m512bh)_mm512_loadu_si512((const void *)({lhs})), (__m512bh)_mm512_loadu_si512((const void *)({rhs})));"
                    )
                })?;
                for accum_name in &accum_names {
                    cline!(
                        w,
                        depth,
                        "{out_expr} += _mm512_reduce_add_ps({accum_name});"
                    )?;
                }
                Ok(())
            }
            Avx512Kernel::MatmulLoopVdpbf16ps => {
                let lhs_spec = arguments[0].spec();
                let rhs_spec = arguments[1].spec();
                let m = lhs_spec.shape()[1].get();
                let k = lhs_spec.shape()[2].get();
                let n = rhs_spec.shape()[2].get();
                let n_vector_count = usize::try_from(n / 16).unwrap();

                cline!(w, depth, "// MatmulLoopVdpbf16ps")?;

                let needs_k_unroll = arguments[..2]
                    .iter()
                    .any(|argument| self.c_buffer_for_view(argument).needs_unroll());
                self.emit_matmul_vdpbf16ps_body(
                    w,
                    arguments,
                    usize::try_from(m).unwrap(),
                    k,
                    needs_k_unroll,
                    n_vector_count,
                    depth,
                )?;
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_matmul_vdpbf16ps_body<W: Write>(
        &mut self,
        w: &mut W,
        arguments: &[ViewE<Tgt>],
        m_row_count: usize,
        k: u32,
        unroll_k: bool,
        n_vector_count: usize,
        depth: usize,
    ) -> fmt::Result {
        debug_assert!((1..=2).contains(&n_vector_count));
        debug_assert!(m_row_count >= 1);

        let m_exprs = (0..m_row_count)
            .map(|row_idx| NonAffineExpr::constant(i32::try_from(row_idx).unwrap()))
            .collect::<Vec<_>>();
        let n_start_exprs = (0..n_vector_count)
            .map(|vector_idx| NonAffineExpr::constant(i32::try_from(vector_idx * 16).unwrap()))
            .collect::<Vec<_>>();

        let stripe_count = if m_row_count == 1 && n_vector_count == 2 {
            5
        } else {
            1
        };
        let mut l1_output_acc_names_and_ptrs = Vec::new();
        let mut acc_name_groups = Vec::with_capacity(m_exprs.len());
        for m_expr in &m_exprs {
            let mut row_acc_groups = Vec::with_capacity(n_start_exprs.len());
            for n_expr in &n_start_exprs {
                let m_const = m_expr.as_constant().unwrap();
                let n_const = n_expr.as_constant().unwrap();
                let out_index_expr =
                    arguments[2]
                        .make_buffer_indexing_expr()
                        .map_vars(&mut |v| match v {
                            BufferVar::Pt(dim, _) => match dim {
                                1 => NonAffineExpr::constant(m_const),
                                2 => NonAffineExpr::constant(n_const),
                                _ => NonAffineExpr::zero(),
                            },
                            BufferVar::TileIdx(_, _) => v.into(),
                        });
                let out_acc_name = self.namer.fresh_name();
                let out_buffer = self.c_buffer_for_view(&arguments[2]);
                let out_ptr = self.c_index_ptr(out_buffer, &out_index_expr);
                l1_output_acc_names_and_ptrs.push((out_acc_name.clone(), out_ptr));
                let mut acc_names = vec![out_acc_name];
                acc_names.extend((1..stripe_count).map(|_| self.namer.fresh_name()));
                row_acc_groups.push(acc_names);
            }
            acc_name_groups.push(row_acc_groups);
        }

        // The first accumulator in each group starts at zero and accumulates only A*B.
        // The prior L1 output value is added immediately before the final writeback.
        for (out_acc_name, _) in &l1_output_acc_names_and_ptrs {
            cline!(w, depth, "__m512 {out_acc_name} = _mm512_setzero_ps();")?;
        }
        for row_acc_groups in &acc_name_groups {
            for acc_names in row_acc_groups {
                for acc_name in &acc_names[1..] {
                    cline!(w, depth, "__m512 {acc_name} = _mm512_setzero_ps();")?;
                }
            }
        }

        self.emit_matmul_vdpbf16ps_k_iterations(
            w,
            k,
            stripe_count,
            unroll_k,
            depth,
            |this, w, stripe_idx, k_expr, depth| {
                let mut rhs_vec_names = Vec::with_capacity(n_vector_count);
                for n_expr in &n_start_exprs {
                    let rhs_pair_ptr = this.c_index_ptr_with_point_exprs(
                        &arguments[1],
                        &[
                            (0, NonAffineExpr::zero()),
                            (1, k_expr.clone()),
                            (2, n_expr.clone()),
                        ],
                    );
                    let rhs_vec_name = this.namer.fresh_name();
                    cline!(
                        w,
                        depth,
                        "__m512bh {rhs_vec_name} = (__m512bh)_mm512_loadu_si512((const void *)({rhs_pair_ptr}));"
                    )?;
                    rhs_vec_names.push(rhs_vec_name);
                }

                let emit_dpbf16 =
                    |w: &mut W, acc_name: &str, lhs_vec_name: &str, rhs_vec_name: &str| {
                        cline!(
                            w,
                            depth,
                            "{0} = _mm512_dpbf16_ps({0}, {1}, {2});",
                            acc_name,
                            lhs_vec_name,
                            rhs_vec_name
                        )
                    };

                for (row_idx, m_expr) in m_exprs.iter().enumerate() {
                    let lhs_vec_name = this.namer.fresh_name();
                    let lhs_pair_ptr = this.c_index_ptr_with_point_exprs(
                        &arguments[0],
                        &[
                            (0, NonAffineExpr::zero()),
                            (1, m_expr.clone()),
                            (2, k_expr.clone()),
                        ],
                    );
                    let lhs_pair_name = this.namer.fresh_name();
                    cline!(w, depth, "uint32_t {lhs_pair_name};")?;
                    cline!(
                        w,
                        depth,
                        "__builtin_memcpy(&{lhs_pair_name}, {lhs_pair_ptr}, sizeof({lhs_pair_name}));"
                    )?;
                    cline!(
                        w,
                        depth,
                        "__m512bh {lhs_vec_name} = (__m512bh)_mm512_set1_epi32({lhs_pair_name});"
                    )?;
                    for (n_idx, rhs_vec_name) in rhs_vec_names.iter().enumerate() {
                        let acc_name = acc_name_groups[row_idx][n_idx][stripe_idx].as_str();
                        emit_dpbf16(w, acc_name, &lhs_vec_name, rhs_vec_name)?;
                    }
                }

                Ok(())
            },
        )?;

        // Emit accumulator group reductions.
        for row_acc_groups in &acc_name_groups {
            for acc_names in row_acc_groups {
                let (acc_name, rest_acc_names) = acc_names.split_first().unwrap();
                for acc_name_rhs in rest_acc_names {
                    cline!(
                        w,
                        depth,
                        "{} = _mm512_add_ps({}, {});",
                        acc_name,
                        acc_name,
                        acc_name_rhs
                    )?;
                }
            }
        }

        for (out_acc_name, out_ptr) in &l1_output_acc_names_and_ptrs {
            cline!(
                w,
                depth,
                "__asm__ __volatile__(\"\" ::: \"memory\");  // prevent hoisting of load"
            )?;
            cline!(
                w,
                depth,
                "{out_acc_name} = _mm512_add_ps({out_acc_name}, _mm512_loadu_ps((const void *)({out_ptr})));"
            )?;
            cline!(
                w,
                depth,
                "_mm512_storeu_ps((void *)({out_ptr}), {out_acc_name});"
            )?;
        }

        Ok(())
    }

    /// Drives the K dimension for the AVX512 BF16 matmul microkernel.
    ///
    /// When K must be unrolled, the callback receives constant K expressions. Otherwise
    /// this emits a runtime loop stepping by `2 * stripe_count`, calls the callback once
    /// per stripe inside the loop, and emits any remaining full K-pairs with constants.
    ///
    /// Example emitted C for a single-stripe runtime K loop:
    ///
    /// ```c
    /// for (size_t k0 = 0; k0 < 528; k0 += 2) {
    ///   // one BF16-pair matmul step at k0
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn emit_matmul_vdpbf16ps_k_iterations<W: Write, F>(
        &mut self,
        w: &mut W,
        k: u32,
        stripe_count: usize,
        unroll_k: bool,
        depth: usize,
        mut emit_step: F,
    ) -> fmt::Result
    where
        F: FnMut(&mut Self, &mut W, usize, NonAffineExpr<CName>, usize) -> fmt::Result,
    {
        if unroll_k {
            let step = u32::try_from(stripe_count).unwrap() * 2;
            for k_idx in (0..k).step_by(step as usize) {
                for stripe_idx in 0..stripe_count {
                    let lane_k = k_idx + u32::try_from(stripe_idx).unwrap() * 2;
                    if lane_k >= k {
                        break;
                    }
                    emit_step(
                        self,
                        w,
                        stripe_idx,
                        NonAffineExpr::constant(i32::try_from(lane_k).unwrap()),
                        depth,
                    )?;
                }
            }
        } else {
            let step = u32::try_from(stripe_count).unwrap() * 2;
            let full_k = (k / step) * step;
            if full_k != 0 {
                let k_idx_name = self.namer.fresh_name();
                cline!(
                    w,
                    depth,
                    "for (size_t {0} = 0; {0} < {1}; {0} += {step}) {{",
                    k_idx_name,
                    full_k,
                )?;
                for stripe_idx in 0..stripe_count {
                    let k_offset = i32::try_from(stripe_idx * 2).unwrap();
                    emit_step(
                        self,
                        w,
                        stripe_idx,
                        NonAffineExpr::from(CName(k_idx_name.clone())) + k_offset,
                        depth + 1,
                    )?;
                }
                cline!(w, depth, "}}")?;
            }
            for (stripe_idx, lane_k) in (full_k..k).step_by(2).enumerate() {
                emit_step(
                    self,
                    w,
                    stripe_idx,
                    NonAffineExpr::constant(i32::try_from(lane_k).unwrap()),
                    depth,
                )?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_avx512_bf16_broadcast_vec_mult_add<W: Write>(
        &mut self,
        w: &mut W,
        output_expr: &str,
        source_ptr: &str,
        source_vec_expr: Option<&str>,
        scalar_bits: &str,
        vector_size_bf16: u32,
        vector_type_name: &str,
        depth: usize,
    ) -> fmt::Result {
        // Use the BF16 dot-product instruction directly for:
        //   output[i] += scalar_bf16 * rhs_bf16[i]
        //
        // The RHS BF16 lanes are zero-extended into 32-bit slots, which reinterprets as BF16 pairs
        // [rhs_i, 0]. The scalar is broadcast to all BF16 lanes, so each dpbf16 lane computes
        // scalar * rhs_i + scalar * 0. This avoids separate BF16->F32 conversions and avoids the
        // manual shift/blend unpack sequence that can crash Clang's x86 instruction selector.
        match vector_size_bf16 {
            8 => {
                let rhs_raw_name = self.namer.fresh_name();
                let rhs_pairs_name = self.namer.fresh_name();
                let scalar_vec_name = self.namer.fresh_name();
                if let Some(source_vec_expr) = source_vec_expr {
                    cline!(w, depth, "__m128i {rhs_raw_name} = {source_vec_expr};")?;
                } else {
                    cline!(
                        w,
                        depth,
                        "__m128i {rhs_raw_name} = _mm_loadu_si128((const __m128i *)({source_ptr}));",
                    )?;
                }
                cline!(
                    w,
                    depth,
                    "__m256i {rhs_pairs_name} = _mm256_cvtepu16_epi32({rhs_raw_name});",
                )?;
                cline!(
                    w,
                    depth,
                    "__m256bh {scalar_vec_name} = (__m256bh)_mm256_set1_epi16({scalar_bits});",
                )?;
                // _mm256_dpbf16_ps requires -mavx512vl
                cline!(
                    w,
                    depth,
                    "{output_expr} = ({vector_type_name})_mm256_dpbf16_ps((__m256){output_expr}, {scalar_vec_name}, (__m256bh){rhs_pairs_name});",
                )
            }
            16 => {
                let rhs_raw_name = self.namer.fresh_name();
                let rhs_pairs_name = self.namer.fresh_name();
                let scalar_vec_name = self.namer.fresh_name();
                if let Some(source_vec_expr) = source_vec_expr {
                    cline!(w, depth, "__m256i {rhs_raw_name} = {source_vec_expr};")?;
                } else {
                    cline!(
                        w,
                        depth,
                        "__m256i {rhs_raw_name} = _mm256_loadu_si256((const __m256i *)({source_ptr}));",
                    )?;
                }
                cline!(
                    w,
                    depth,
                    "__m512i {rhs_pairs_name} = _mm512_cvtepu16_epi32({rhs_raw_name});",
                )?;
                cline!(
                    w,
                    depth,
                    "__m512bh {scalar_vec_name} = (__m512bh)_mm512_set1_epi16({scalar_bits});",
                )?;
                cline!(
                    w,
                    depth,
                    "{output_expr} = ({vector_type_name})_mm512_dpbf16_ps((__m512){output_expr}, {scalar_vec_name}, (__m512bh){rhs_pairs_name});",
                )
            }
            _ => unimplemented!(),
        }
    }

    pub fn c_buffer_for_view(&self, arg: &ViewE<Tgt>) -> &CBuffer {
        let backing_tensor = arg.backing_tensor().unwrap();
        self.name_env.get(&backing_tensor.identifier()).unwrap()
    }

    fn make_buffer(
        &mut self,
        tensor_identifier: Option<OpaqueSymbol>,
        shape: &[DimSize],
        vector_size: Option<DimSize>,
        dtype: Dtype,
        memory: Tgt::Memory,
    ) -> CBuffer {
        debug_assert_eq!(vector_size.is_some(), memory == CpuMemory::VRF);

        let size = shape.iter().map(|d| d.get()).product::<u32>();
        match memory.into() {
            CpuMemory::VRF => {
                let vector_size = vector_size.unwrap();
                let vec_type = get_vector(Tgt::vec_types(), dtype, vector_size);
                self.headers.vector_type_defs.insert(vec_type);

                let vector_size = vector_size.get();
                debug_assert_eq!(size % vector_size, 0);
                CBuffer::RegVars {
                    inner_vecs: (0..(size / vector_size))
                        .map(|_| (self.namer.fresh_name(), Either::Right(vec_type)))
                        .collect(),
                }
            }
            #[cfg(not(feature = "drop-rf"))]
            CpuMemory::RF => CBuffer::RegVars {
                inner_vecs: (0..size)
                    .map(|_| (self.namer.fresh_name(), Either::Left(dtype)))
                    .collect(),
            },
            CpuMemory::L1 | CpuMemory::GL => {
                let name = self.namer.fresh_name();
                if size * u32::from(dtype.size()) > STACK_CUTOFF {
                    if let Some((workspace_id, offset_bytes)) = self
                        .static_allocation_plan
                        .as_ref()
                        .zip(tensor_identifier)
                        .and_then(|(plan, symbol)| plan.placement(symbol))
                    {
                        CBuffer::WorkspaceSlice {
                            name,
                            size,
                            dtype,
                            workspace_name: self
                                .active_workspace_names
                                .get(&workspace_id)
                                .expect("workspace should be active")
                                .clone(),
                            offset_bytes,
                        }
                    } else {
                        CBuffer::HeapArray { name, size, dtype }
                    }
                } else {
                    CBuffer::StackArray { name, size, dtype }
                }
            }
        }
    }

    fn emit<W: Write>(&mut self, w: &mut W, imp: &ImplNode<Tgt>, depth: usize) -> fmt::Result {
        let workspace_id = self
            .static_allocation_plan
            .as_ref()
            .and_then(|plan| plan.workspace_for(imp));
        let workspace_scope = if let Some(workspace_id) = workspace_id {
            let workspace_name = self.namer.fresh_name();
            let alignment = HEAP_BUFFER_ALIGNMENT_BYTES.max(u64::from(Tgt::line_size()));
            let bytes = self
                .static_allocation_plan
                .as_ref()
                .expect("workspace plan should exist")
                .workspace_size(workspace_id);
            cline!(w, depth, "uint8_t *__restrict__ {workspace_name};")?;
            cline!(
                w,
                depth,
                "if (posix_memalign((void **)&{workspace_name}, {alignment}, {bytes}) != 0) abort();",
            )?;

            let old_name = self
                .active_workspace_names
                .insert(workspace_id, workspace_name.clone());
            debug_assert!(
                old_name.is_none(),
                "workspace ids should not be active in nested scopes"
            );
            Some((workspace_id, workspace_name))
        } else {
            None
        };

        let emit_result = match imp {
            ImplNode::Loop(l) => {
                // Emit a C loop nest or, if any tensor view is requires unrolling (i.e., vector
                // tensors), unroll the loop, emitting the body repeatedly.
                //
                // It's not always necessary to unroll every emitted C loop because not all axes
                // range over different vector registers. This is pretty rare though, so, for
                // simplicity, every Impl loop is emitted as either as a C loop nest or fully
                // unrolled.
                if l.tiles.iter().any(|loop_tile| {
                    let backing_tensor = loop_tile
                        .tile
                        .backing_tensor()
                        .expect("tile should have backing tensor");
                    let backing_tensor_name = self
                        .name_env
                        .get(&backing_tensor.identifier())
                        .expect("tile's backing tensor should have a name");
                    backing_tensor_name.needs_unroll()
                }) {
                    self.emit_unrolled_loop(w, l, depth)
                } else {
                    self.emit_rolled_loop_nest(w, l, depth)
                }
            }
            ImplNode::Alloc(alloc_binding) => {
                match &alloc_binding.introduced {
                    ViewE::Tensor(tensor) => {
                        // Emit variable declaration(s) and store association between the
                        // CBuffer and Tensor.
                        let spec = alloc_binding.introduced.spec();
                        let dest_buffer = self.make_buffer(
                            Some(tensor.identifier()),
                            spec.shape(),
                            spec.vector_size(),
                            spec.dtype(),
                            spec.memory(),
                        );
                        dest_buffer.emit(w, InitType::None, depth)?;
                        self.name_env.insert(tensor.identifier(), dest_buffer);
                    }
                    ViewE::CacheView(_) => (),
                    _ => unreachable!(),
                };

                if let Some(prologue) = alloc_binding.prologue() {
                    self.emit(w, prologue, depth)?;
                }
                self.emit(w, alloc_binding.main_stage(), depth)?;
                if let Some(epilogue) = alloc_binding.epilogue() {
                    self.emit(w, epilogue, depth)?;
                }

                if let ViewE::Tensor(tensor) = &alloc_binding.introduced {
                    self.name_env
                        .remove(&tensor.identifier())
                        .unwrap()
                        .emit_free(w, depth)?;
                }
                Ok(())
            }
            ImplNode::Block(Block { stages, .. }) => {
                for stage in stages {
                    self.emit(w, stage, depth)?;
                }
                Ok(())
            }
            ImplNode::Pipeline(Pipeline {
                stages, wirings, ..
            }) => {
                for (stage_idx, stage) in stages.iter().enumerate() {
                    // Emit variable declaration(s) and store association between the
                    // CBuffer and Tensor.
                    if let Some(wiring) = wirings.get(stage_idx) {
                        for tensor_wiring in &wiring.intermediate_tensors {
                            let intermediate_spec = tensor_wiring.spec();
                            let buffer = self.make_buffer(
                                Some(tensor_wiring.identifier()),
                                intermediate_spec.shape(),
                                intermediate_spec.vector_size(),
                                intermediate_spec.dtype(),
                                intermediate_spec.memory(),
                            );
                            buffer.emit(w, InitType::None, depth)?;
                            self.name_env.insert(tensor_wiring.identifier(), buffer);
                        }
                    }
                    self.emit(w, stage, depth)?;
                    if stage_idx > 0 {
                        for tensor in &wirings[stage_idx - 1].intermediate_tensors {
                            let consumed_buffer = self.name_env.get(&tensor.identifier()).unwrap();
                            consumed_buffer.emit_free(w, depth)?;
                        }
                    }
                }
                Ok(())
            }
            ImplNode::TimedRegion(region) => {
                if self.benchmark {
                    // TODO: Would be nice for register_benchmark_counter to return &str instead.
                    let counter_var = self
                        .headers
                        .register_benchmark_counter(region.counter_name());
                    let start_var = self.namer.fresh_name();
                    let end_var = self.namer.fresh_name();
                    let delta_var = self.namer.fresh_name();
                    let nanos_var = self.namer.fresh_name();

                    cline!(w, depth, "struct timespec {start_var}, {end_var};")?;
                    cline!(w, depth, "clock_gettime(CLOCK_MONOTONIC, &{start_var});")?;
                    self.emit(w, region.child(), depth)?;
                    cline!(w, depth, "clock_gettime(CLOCK_MONOTONIC, &{end_var});")?;
                    cline!(
                        w,
                        depth,
                        "struct timespec {delta_var} = ts_diff({start_var}, {end_var});",
                    )?;
                    cline!(
                        w,
                        depth,
                        "long long {nanos_var} = {delta_var}.tv_sec * 1000000000LL + {delta_var}.tv_nsec;",
                    )?;
                    cline!(
                        w,
                        depth,
                        "atomic_fetch_add_explicit(&{counter_var}, {nanos_var}, memory_order_relaxed);",
                    )
                } else {
                    self.emit(w, region.child(), depth)
                }
            }
            ImplNode::SpecApp(p) => {
                self.headers.emit_stdbool_and_assert_includes = true;
                cline!(
                    w,
                    depth,
                    "/* {}({}) */",
                    p.0,
                    iter::repeat_n("_", p.1.len()).join(", ")
                )?;
                let args: String =
                    p.1.iter()
                        .map(|arg| {
                            let backing_tensor = arg.backing_tensor().unwrap();
                            match self.name_env.get(&backing_tensor.identifier()).unwrap() {
                                CBuffer::HeapArray { name, .. }
                                | CBuffer::WorkspaceSlice { name, .. }
                                | CBuffer::StackArray { name, .. }
                                | CBuffer::Ptr { name, .. } => format!("{name}[_]"),
                                CBuffer::RegVars { inner_vecs, .. } => inner_vecs[0].0.clone(),
                            }
                        })
                        .join(", ");
                cline!(w, depth, "assert(false);  // missing imp({args})")
            }
            ImplNode::Kernel(KernelApp {
                kernel_type,
                arguments,
                spec: _,
            }) => {
                match kernel_type.into_cpu_kernel() {
                    Some(CpuKernel::OnePrefixNoOp) => {
                        let exprs = self.c_value_args(arguments);
                        cline!(w, depth, "{} = {};", exprs[1], exprs[0])
                    }
                    Some(CpuKernel::ValueSoftmaxComplete) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        cline!(
                            w,
                            depth,
                            "{} = expf({} - {}) / {};",
                            exprs[3],
                            exprs[0],
                            exprs[1],
                            exprs[2],
                        )
                    }
                    Some(CpuKernel::ValueSoftmaxDenominatorAndMax) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        let old_max_name = self.namer.fresh_name();
                        let new_max_name = self.namer.fresh_name();
                        cline!(w, depth, "float {old_max_name} = {};", exprs[1])?;
                        cline!(
                            w,
                            depth,
                            "float {new_max_name} = fmaxf({old_max_name}, {});",
                            exprs[0],
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = {} * expf({old_max_name} - {new_max_name}) + expf({} - {new_max_name});",
                            exprs[2],
                            exprs[2],
                            exprs[0],
                        )?;
                        cline!(w, depth, "{} = {new_max_name};", exprs[1])
                    }
                    Some(CpuKernel::ValueSoftmaxDenominatorAndMaxFromParts) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        let old_max_name = self.namer.fresh_name();
                        let new_max_name = self.namer.fresh_name();
                        cline!(w, depth, "float {old_max_name} = {};", exprs[2])?;
                        cline!(
                            w,
                            depth,
                            "float {new_max_name} = fmaxf({old_max_name}, {});",
                            exprs[0],
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = {} * expf({old_max_name} - {new_max_name}) + {} * expf({} - {new_max_name});",
                            exprs[3],
                            exprs[3],
                            exprs[1],
                            exprs[0],
                        )?;
                        cline!(w, depth, "{} = {new_max_name};", exprs[2])
                    }
                    Some(CpuKernel::ValueSoftmaxDenominator) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        cline!(
                            w,
                            depth,
                            "{} += expf({} - {});",
                            exprs[2],
                            exprs[0],
                            exprs[1],
                        )
                    }
                    Some(CpuKernel::ValueSoftmaxDenominatorAndUnscaledFromMax) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        let reg_name = self.namer.fresh_name();
                        cline!(
                            w,
                            depth,
                            "float {reg_name} = expf({} - {});",
                            exprs[0],
                            exprs[1],
                        )?;
                        cline!(w, depth, "{} += {reg_name};", exprs[2])?;
                        cline!(w, depth, "{} = {reg_name};", exprs[3])
                    }
                    Some(CpuKernel::VectorSoftmaxDenominator) => {
                        // TODO: Test that we don't mutate the input.

                        cline!(w, depth, "/* VectorSoftmaxDenominator */")?;

                        let input_vector_exprs = self.c_vec_exprs_for_tensor(&arguments[0]);
                        let vector_size = arguments[0].spec().vector_size().unwrap();
                        let exp_helper = self.require_x86_f32_exp_helper(vector_size);
                        let sum_helper = self.require_x86_f32_sum_helper(vector_size);
                        let max_expr = self.c_value_arg(&arguments[1]);
                        let out_expr = self.c_value_arg(&arguments[2]);

                        let accum_count = u32::try_from(input_vector_exprs.len())
                            .unwrap()
                            .min(VECTOR_ACCUM_COUNT);
                        let vector_type =
                            get_vector(Tgt::vec_types(), arguments[0].spec().dtype(), vector_size);
                        let accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Add,
                            accum_count,
                            vector_type,
                        )?;
                        // TODO: Don't fully unroll. Instead, read from L1.
                        for (i, input_vector_name) in input_vector_exprs.iter().enumerate() {
                            let exp = format!("{exp_helper}({input_vector_name} - {max_expr})");
                            accums.emit_update(w, depth, i, &exp)?;
                        }
                        let sum_vec = accums.emit_reduce(w, depth)?;
                        cline!(w, depth, "{out_expr} += {sum_helper}({sum_vec});")
                    }
                    Some(
                        cpu_kernel @ (CpuKernel::VectorSoftmaxDenominatorAndMax
                        | CpuKernel::VectorSoftmaxDenominatorAndMaxFromParts),
                    ) => {
                        // The arguments are either an input or partial maxes and denominators,
                        // followed by the max and denominator outputs.
                        cline!(w, depth, "/* {cpu_kernel:?} */")?;
                        self.headers.emit_math_include = true;

                        let input_vector_exprs = self.c_vec_exprs_for_tensor(&arguments[0]);
                        let part_denom_vector_exprs = (cpu_kernel
                            == CpuKernel::VectorSoftmaxDenominatorAndMaxFromParts)
                            .then(|| self.c_vec_exprs_for_tensor(&arguments[1]));
                        debug_assert!(part_denom_vector_exprs
                            .as_ref()
                            .is_none_or(|d| d.len() == input_vector_exprs.len()));
                        let vector_size = arguments[0].spec().vector_size().unwrap();
                        let vector_type =
                            get_vector(Tgt::vec_types(), arguments[0].spec().dtype(), vector_size);
                        let exp_helper = self.require_x86_f32_exp_helper(vector_size);
                        let sum_helper = self.require_x86_f32_sum_helper(vector_size);
                        let horizontal_max_helper =
                            self.require_x86_f32_horizontal_max_helper(vector_size);
                        let max_expr = self.c_value_arg(&arguments[arguments.len() - 2]);
                        let denom_expr = self.c_value_arg(&arguments[arguments.len() - 1]);

                        let accum_count = u32::try_from(input_vector_exprs.len())
                            .unwrap()
                            .min(VECTOR_ACCUM_COUNT);

                        let max_accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Max,
                            accum_count,
                            vector_type,
                        )?;
                        for (i, input_vector_name) in input_vector_exprs.iter().enumerate() {
                            max_accums.emit_update(w, depth, i, input_vector_name)?;
                        }
                        let max_vec = max_accums.emit_reduce(w, depth)?;
                        let chunk_max_name = self.namer.fresh_name();
                        cline!(
                            w,
                            depth,
                            "float {chunk_max_name} = {horizontal_max_helper}({max_vec});"
                        )?;

                        let denom_accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Add,
                            accum_count,
                            vector_type,
                        )?;
                        for (i, input_vector_name) in input_vector_exprs.iter().enumerate() {
                            let exp =
                                format!("{exp_helper}({input_vector_name} - {chunk_max_name})");
                            let term = match &part_denom_vector_exprs {
                                Some(part_denoms) => format!("{} * {exp}", part_denoms[i]),
                                None => exp,
                            };
                            denom_accums.emit_update(w, depth, i, &term)?;
                        }
                        let denom_vec = denom_accums.emit_reduce(w, depth)?;
                        let chunk_denom_name = self.namer.fresh_name();
                        cline!(
                            w,
                            depth,
                            "float {chunk_denom_name} = {sum_helper}({denom_vec});"
                        )?;

                        let old_max_name = self.namer.fresh_name();
                        let new_max_name = self.namer.fresh_name();
                        cline!(w, depth, "float {old_max_name} = {max_expr};")?;
                        cline!(
                            w,
                            depth,
                            "float {new_max_name} = fmaxf({old_max_name}, {chunk_max_name});"
                        )?;
                        cline!(
                            w,
                            depth,
                            "{denom_expr} = {denom_expr} * expf({old_max_name} - {new_max_name}) + {chunk_denom_name} * expf({chunk_max_name} - {new_max_name});"
                        )?;
                        cline!(w, depth, "{max_expr} = {new_max_name};")
                    }
                    Some(CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32) => {
                        cline!(w, depth, "/* VectorSoftmaxDenominatorAndUnscaledF32 */")?;

                        let vector_size =
                            softmax_vector_size::<Tgt>(arguments[0].spec(), arguments[3].spec())
                                .unwrap();
                        let exp_helper = self.require_x86_f32_exp_helper(vector_size);
                        let sum_helper = self.require_x86_f32_sum_helper(vector_size);

                        let max_expr = self.c_value_arg(&arguments[1]);
                        let denom_expr = self.c_value_arg(&arguments[2]);
                        let volume = arguments[0].spec().volume().get();
                        let accum_count = (volume / vector_size.get()).min(VECTOR_ACCUM_COUNT);
                        let vector_type = get_vector(Tgt::vec_types(), Dtype::Float32, vector_size);
                        let accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Add,
                            accum_count,
                            vector_type,
                        )?;
                        let native_type = vector_type.native_type_name;
                        let input_vector_name = self.namer.fresh_name();
                        cline!(w, depth, "{native_type} {input_vector_name};")?;

                        let strip_loop = self.strip_loop(
                            vector_size,
                            accum_count,
                            [&arguments[0], &arguments[3]],
                        );
                        strip_loop.emit(w, depth, |w, depth, lane, [input_ptr, output_ptr]| {
                            let unscaled = self.namer.fresh_name();
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{input_vector_name}, {input_ptr}, sizeof({input_vector_name}));"
                            )?;
                            cline!(
                                w,
                                depth,
                                "{native_type} {unscaled} = {exp_helper}({input_vector_name} - {max_expr});"
                            )?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy({output_ptr}, &{unscaled}, sizeof({unscaled}));"
                            )?;
                            accums.emit_update(w, depth, lane, &unscaled)
                        })?;
                        let sum_vec = accums.emit_reduce(w, depth)?;
                        cline!(w, depth, "{denom_expr} += {sum_helper}({sum_vec});")
                    }
                    Some(CpuKernel::VectorSoftmaxComplete) => {
                        let vector_size =
                            softmax_vector_size::<Tgt>(arguments[0].spec(), arguments[3].spec())
                                .unwrap();
                        let exp_helper = self.require_x86_f32_exp_helper(vector_size);
                        cline!(w, depth, "/* VectorSoftmaxComplete */")?;

                        let vtype = get_vector(Tgt::vec_types(), Dtype::Float32, vector_size)
                            .native_type_name;
                        let max_expr = self.c_value_arg(&arguments[1]);
                        let denom_expr = self.c_value_arg(&arguments[2]);
                        let reciprocal_vec = self.namer.fresh_name();
                        cline!(
                            w,
                            depth,
                            "{vtype} {reciprocal_vec} = (1.0f / ({denom_expr})) - ({vtype}){{}};"
                        )?;

                        let strip_loop =
                            self.strip_loop(vector_size, 1, [&arguments[0], &arguments[3]]);
                        strip_loop.emit(w, depth, |w, depth, _, [input_ptr, output_ptr]| {
                            let [input_vec, out_vec] =
                                [self.namer.fresh_name(), self.namer.fresh_name()];
                            cline!(w, depth, "{vtype} {input_vec};")?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{input_vec}, {input_ptr}, sizeof({input_vec}));"
                            )?;
                            cline!(
                                w,
                                depth,
                                "{vtype} {out_vec} = {exp_helper}({input_vec} - {max_expr});"
                            )?;
                            cline!(w, depth, "{out_vec} *= {reciprocal_vec};")?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy({output_ptr}, &{out_vec}, sizeof({out_vec}));"
                            )
                        })
                    }
                    Some(CpuKernel::ValueMax) => {
                        self.headers.emit_math_include = true;
                        let exprs = self.c_value_args(arguments);
                        cline!(
                            w,
                            depth,
                            "{} = fmaxf({}, {});",
                            exprs[1],
                            exprs[1],
                            exprs[0]
                        )
                    }
                    Some(CpuKernel::VectorMax) => {
                        let vector_size = arguments[0].spec().vector_size().unwrap();
                        self.headers.emit_math_include = true;
                        let horizontal_max_helper =
                            self.require_x86_f32_horizontal_max_helper(vector_size);
                        cline!(w, depth, "/* VectorMax */")?;
                        let out_expr = self.c_value_arg(&arguments[1]);
                        // Emit the vertical reductions, in place in the input vectors
                        let input_vectors = VecAccums {
                            op: VecReduce::Max,
                            names: self.c_vec_exprs_for_tensor(&arguments[0]),
                        };
                        let max_vec = input_vectors.emit_reduce(w, depth)?;

                        // Emit the single-vector reduction
                        cline!(
                            w,
                            depth,
                            "{out_expr} = fmaxf({out_expr}, {horizontal_max_helper}({max_vec}));"
                        )
                    }
                    Some(CpuKernel::VectorMaxLoop) => {
                        let value_cnt = arguments[0].spec().volume().get();
                        let (vector_size, accum_count) =
                            vector_max_loop_properties::<Tgt>(value_cnt).unwrap();

                        self.headers.emit_math_include = true;
                        let out_expr = self.c_value_arg(&arguments[1]);
                        let horizontal_max_helper =
                            self.require_x86_f32_horizontal_max_helper(vector_size);
                        let vector_type = get_vector(Tgt::vec_types(), Dtype::Float32, vector_size);
                        self.headers.vector_type_defs.insert(vector_type);

                        cline!(w, depth, "/* VectorMaxLoop */")?;
                        let accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Max,
                            accum_count,
                            vector_type,
                        )?;
                        let strip_loop = self.strip_loop(vector_size, accum_count, [&arguments[0]]);
                        strip_loop.emit(w, depth, |w, depth, lane, [ptr]| {
                            let input_vec = self.namer.fresh_name();
                            cline!(w, depth, "{} {input_vec};", vector_type.name)?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{input_vec}, {ptr}, sizeof({input_vec}));"
                            )?;
                            accums.emit_update(w, depth, lane, &input_vec)
                        })?;
                        let max_vec = accums.emit_reduce(w, depth)?;
                        cline!(
                            w,
                            depth,
                            "{out_expr} = fmaxf({out_expr}, {horizontal_max_helper}({max_vec}));"
                        )
                    }
                    Some(CpuKernel::MultAdd) => {
                        let exprs = self.c_value_args(arguments);
                        cline!(
                            w,
                            depth,
                            "{} += {} * {};  /* MultAdd */",
                            exprs[2],
                            exprs[0],
                            exprs[1]
                        )
                    }
                    Some(CpuKernel::Assign) => {
                        let is_scalar = arguments
                            .iter()
                            .flat_map(|arg| arg.spec().shape())
                            .all(|d| d.get() == 1);
                        if is_scalar {
                            let exprs = self.c_value_args(arguments);
                            return cline!(w, depth, "{} = {};", exprs[1], exprs[0]);
                        }

                        let first_spec = arguments[0].spec();
                        let second_spec = arguments[1].spec();

                        let vector_size = first_spec.vector_size().unwrap_or_else(|| {
                            second_spec.vector_size().unwrap_or_else(|| {
                                panic!(
                                    "neither Spec had a vector size: {first_spec} and {second_spec}"
                                )
                            })
                        });

                        let dtype = first_spec.dtype();
                        let byte_count =
                            usize::from(dtype.size()) * usize::try_from(vector_size.get()).unwrap();
                        let strip_loop =
                            self.strip_loop(vector_size, 1, [&arguments[0], &arguments[1]]);
                        strip_loop.emit(w, depth, |w, depth, _, [src, dst]| {
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy((void *)({dst}), (const void *)({src}), {byte_count});  /* Assign */"
                            )
                        })
                    }
                    Some(CpuKernel::CastBf16F32) => {
                        let exprs = self.c_value_args(arguments);
                        cline!(w, depth, "{} = (float){};", exprs[1], exprs[0])
                    }
                    Some(CpuKernel::VectorCastBf16F32) => {
                        let vector_size =
                            i32::try_from(arguments[1].spec().vector_size().unwrap().get())
                                .unwrap();

                        self.headers.emit_cvtbf16_fp32 = true;
                        cline!(
                            w,
                            depth,
                            "cvtbf16_fp32_256({}, &{}, &{});",
                            self.c_vec_arg(&arguments[0]),
                            self.c_vec_arg(&arguments[1]),
                            self.c_vec_arg_add(&arguments[1], vector_size)
                        )
                    }
                    Some(CpuKernel::MemsetZero) => {
                        debug_assert_eq!(arguments.len(), 1);
                        let arg_expr = self.c_ptr_arg(&arguments[0]);
                        cline!(
                            w,
                            depth,
                            "memset((void *)({arg_expr}), 0, {});",
                            arguments[0].spec().bytes_used()
                        )
                    }
                    Some(
                        cpu_kernel @ (CpuKernel::VectorZero
                        | CpuKernel::VectorNegInf
                        | CpuKernel::VectorMin),
                    ) => {
                        let exprs = self.c_vec_args(arguments);
                        let dtype = arguments[0].spec().dtype();
                        let volume = arguments[0].spec().volume();
                        let vtype = get_vector(Tgt::vec_types(), dtype, volume);
                        let (v, c) = match cpu_kernel {
                            CpuKernel::VectorZero => ("0", "VectorZero"),
                            CpuKernel::VectorNegInf => {
                                self.headers.emit_math_include = true;
                                ("-INFINITY", "VectorNegInf")
                            }
                            CpuKernel::VectorMin => {
                                (self.c_min_const_for_dtype(dtype), "VectorMin")
                            }
                            _ => unreachable!(),
                        };
                        cline!(
                            w,
                            depth,
                            "{} = ({}){{{v}}};  /* {c} */",
                            exprs[0],
                            vtype.name
                        )
                    }
                    Some(
                        cpu_kernel @ (CpuKernel::ValueZero
                        | CpuKernel::ValueNegInf
                        | CpuKernel::ValueMin),
                    ) => {
                        let exprs = self.c_value_args(arguments);
                        match cpu_kernel {
                            CpuKernel::ValueZero => {
                                cline!(w, depth, "{} = 0;  /* ValueZero */", exprs[0])
                            }
                            CpuKernel::ValueNegInf => {
                                self.headers.emit_math_include = true;
                                cline!(w, depth, "{} = -INFINITY;  /* ValueNegInf */", exprs[0])
                            }
                            CpuKernel::ValueMin => {
                                let dtype = arguments[0].spec().dtype();
                                let v = self.c_min_const_for_dtype(dtype);
                                cline!(w, depth, "{} = {v};  /* ValueMin */", exprs[0])
                            }
                            _ => unreachable!(),
                        }
                    }
                    Some(CpuKernel::BroadcastVecMultAdd) => {
                        let (scalar_idx, broadcast_idx) = broadcastvecmult_side(
                            &arguments
                                .iter()
                                .map(|a| a.spec().clone())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();

                        let vector_size = arguments[2].spec().vector_size().unwrap().get();
                        let output_shape = arguments[2].spec().shape();
                        debug_assert_eq!(output_shape.len(), 3);
                        let row_len = output_shape[2].get();
                        debug_assert!(row_len > 0);
                        let volume = arguments[2].spec().volume().get();
                        for offset in vector_offsets(volume, vector_size) {
                            let [lhs, rhs, out] = [0, 1, 2].map(|idx| {
                                if idx == scalar_idx {
                                    self.c_value_arg(&arguments[idx])
                                } else {
                                    assert!(idx == broadcast_idx || idx == 2);
                                    self.c_vec_arg_add(&arguments[idx], offset)
                                }
                            });
                            cline!(
                                w,
                                depth,
                                "{out} += {lhs} * {rhs}; /* BroadcastVecMultAdd */"
                            )?;
                        }
                        Ok(())
                    }
                    Some(CpuKernel::BroadcastVecMultAddBf16F32) => {
                        let vector_size_bf16 = arguments[1].spec().vector_size().unwrap().get();
                        let volume = arguments[1].spec().volume().get();
                        let vfc = get_vector(
                            Tgt::vec_types(),
                            Dtype::Float32,
                            vector_size_bf16.try_into().unwrap(),
                        );
                        self.headers.vector_type_defs.insert(vfc);
                        let scalar_ptr = self.c_ptr_arg(&arguments[0]);
                        // Copies the bf16 scalar's bits into a new variable, returning its name.
                        // TODO: Don't inline `short` below.
                        let load_scalar = |this: &mut Self, w: &mut W| {
                            let name = this.namer.fresh_name();
                            cline!(w, depth, "short {name};")?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{name}, {scalar_ptr}, sizeof({name}));"
                            )?;
                            Ok::<_, fmt::Error>(name)
                        };
                        cline!(w, depth, "/* BroadcastVecMultAddBf16F32 */")?;
                        for offset in vector_offsets(volume, vector_size_bf16) {
                            let output_expr = self.c_vec_arg_add(&arguments[2], offset);
                            if Tgt::target_id() == TargetId::Avx512 {
                                let (broadcast_buffer, broadcast_expr) =
                                    self.c_arg_parts_add(&arguments[1], offset);
                                let broadcast_ptr =
                                    self.c_index_ptr(&broadcast_buffer, &broadcast_expr);
                                let broadcast_vec_expr =
                                    matches!(&broadcast_buffer, CBuffer::RegVars { .. }).then(
                                        || self.c_index_vec(&broadcast_buffer, &broadcast_expr),
                                    );
                                let scalar_name = load_scalar(self, w)?;
                                self.emit_avx512_bf16_broadcast_vec_mult_add(
                                    w,
                                    &output_expr,
                                    &broadcast_ptr,
                                    broadcast_vec_expr.as_deref(),
                                    &scalar_name,
                                    vector_size_bf16,
                                    vfc.name,
                                    depth,
                                )?;
                            } else {
                                let broadcast_expr = self.c_vec_arg_add(&arguments[1], offset);
                                let [even_name, odd_name, concat_name, broad_name] =
                                    [(); 4].map(|_| self.namer.fresh_name());
                                let scalar_name = load_scalar(self, w)?;

                                let (shift_fn, _, _, _) = vec_func_names(vector_size_bf16);
                                let (shift_fn_out, _, _, broadcast16_out) =
                                    vec_func_names(vector_size_bf16 * 2);
                                let vf8 = get_vector(
                                    Tgt::vec_types(),
                                    Dtype::Float32,
                                    (vector_size_bf16 / 2).try_into().unwrap(),
                                );
                                self.headers.vector_type_defs.insert(vf8);
                                let even_expr = bf16_even_lanes_expr(
                                    vector_size_bf16,
                                    vf8.name,
                                    vf8.native_type_name,
                                    &broadcast_expr,
                                );
                                let shuffle_indices =
                                    interleave_vector_halves_shuffle_indices(vector_size_bf16);
                                let (vf8_ty, vf8_native_ty, vfc_ty) =
                                    (vf8.name, vf8.native_type_name, vfc.name);
                                cline!(
                                    w,
                                    depth,
                                    "{vf8_ty} {odd_name} = ({vf8_ty}){shift_fn}(*({vf8_native_ty}*)(&{broadcast_expr}), 16);"
                                )?;
                                cline!(w, depth, "{vf8_ty} {even_name} = {even_expr};")?;
                                cline!(
                                    w,
                                    depth,
                                    "{vfc_ty} {concat_name} = __builtin_shufflevector({odd_name}, {even_name}, {shuffle_indices});"
                                )?;
                                cline!(
                                    w,
                                    depth,
                                    "{vfc_ty} {broad_name} = ({vfc_ty}){shift_fn_out}({broadcast16_out}({scalar_name}), 16);"
                                )?;
                                cline!(w, depth, "{output_expr} += {broad_name} * {concat_name};")?;
                            }
                        }
                        Ok(())
                    }
                    Some(CpuKernel::TwoVecBroadcastVecMultAddU8S8S16) => {
                        let vector_size = arguments[2].spec().vector_size().unwrap().get();
                        let volume = arguments[2].spec().volume().get();
                        for offset in vector_offsets(volume, vector_size) {
                            let lhs = self.c_ptr_arg(&arguments[0]);
                            let [rhs, out] =
                                [1, 2].map(|i| self.c_vec_arg_add(&arguments[i], offset));

                            // TODO: Lift the broadcast out of this loop.
                            let broadcast_name = self.namer.fresh_name();
                            writeln!(w, "/* TwoVecBroadcastVecMultAddU8S8S16 */")?;
                            let broadcast_scalar = self.namer.fresh_name();
                            cline!(w, depth, "int16_t {broadcast_scalar};")?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{broadcast_scalar}, {lhs}, sizeof({broadcast_scalar}));"
                            )?;
                            cline!(
                                w,
                                depth,
                                "__m256i {broadcast_name} = _mm256_set1_epi16({broadcast_scalar});"
                            )?;

                            // matmul (k=2) the broadcast vector with the rhs vectors.
                            cline!(
                                w,
                                depth,
                                "{out} = _mm256_add_epi16({out}, _mm256_maddubs_epi16({broadcast_name}, {rhs}));"
                            )?;
                        }
                        Ok(())
                    }
                    Some(CpuKernel::DotProductLoop) => {
                        self.headers.emit_sum8 = true;
                        let out_expr = self.c_value_arg(&arguments[2]);

                        let k = arguments[0].spec().shape()[2].get();
                        let strip = DOT_PRODUCT_STRIP_SIZE.get();
                        debug_assert_eq!(k % strip, 0);
                        let accum_count = (1..=VECTOR_ACCUM_COUNT)
                            .rev()
                            .find(|c| (k / strip).is_multiple_of(*c))
                            .expect("DotProductLoop applies should have ensured accum_count");
                        let accum_names = self
                            .namer
                            .fresh_names(usize::try_from(accum_count).unwrap());

                        let vtype =
                            get_vector(Tgt::vec_types(), Dtype::Float32, DOT_PRODUCT_STRIP_SIZE);
                        self.headers.vector_type_defs.insert(vtype);
                        cline!(w, depth, "// DotProductLoop")?;
                        for accum_name in &accum_names {
                            cline!(w, depth, "{0} {accum_name} = ({0}){{0}};", vtype.name)?;
                        }
                        let strip_loop = self.strip_loop(
                            DOT_PRODUCT_STRIP_SIZE,
                            accum_count,
                            [&arguments[0], &arguments[1]],
                        );
                        strip_loop.emit(w, depth, |w, depth, lane, [lhs_ptr, rhs_ptr]| {
                            let [lhs_vec, rhs_vec] =
                                [self.namer.fresh_name(), self.namer.fresh_name()];
                            cline!(w, depth, "{} {lhs_vec};", vtype.name)?;
                            cline!(w, depth, "{} {rhs_vec};", vtype.name)?;
                            for (vec, ptr) in [(&lhs_vec, lhs_ptr), (&rhs_vec, rhs_ptr)] {
                                cline!(
                                    w,
                                    depth,
                                    "__builtin_memcpy(&{vec}, {ptr}, sizeof({vec}));"
                                )?;
                            }
                            cline!(w, depth, "{} += {lhs_vec} * {rhs_vec};", accum_names[lane])
                        })?;
                        for accum_name in &accum_names {
                            cline!(w, depth, "{out_expr} += sum8({accum_name});")?;
                        }
                        Ok(())
                    }
                    Some(
                        cpu_kernel @ (CpuKernel::DotProductLoopBf16Bf16F32
                        | CpuKernel::DotProductLoopF32Bf16F32
                        | CpuKernel::DotProductLoopF32InterleavedBf16F32),
                    ) => {
                        let out_expr = self.c_value_arg(&arguments[2]);

                        let (shift_fn, _, _, _) = vec_func_names(16);

                        let vf32 = get_vector(Tgt::vec_types(), Dtype::Float32, nz!(8u32));
                        let vbf16 = get_vector(
                            Tgt::vec_types(),
                            Dtype::Bfloat16,
                            DOT_PRODUCT_BF16_STRIP_SIZE,
                        );
                        self.headers.vector_type_defs.insert(vbf16);
                        self.headers.emit_sum8 = true;

                        cline!(w, depth, "// {cpu_kernel:?}")?;
                        let accums = VecAccums::declare(
                            self,
                            w,
                            depth,
                            VecReduce::Add,
                            VECTOR_ACCUM_COUNT,
                            vf32,
                        )?;
                        let strip_loop = self.strip_loop(
                            DOT_PRODUCT_BF16_STRIP_SIZE,
                            VECTOR_ACCUM_COUNT,
                            [&arguments[0], &arguments[1]],
                        );
                        strip_loop.emit(w, depth, |w, depth, lane, [lhs_ptr, rhs_ptr]| {
                            let [lhs0, lhs1, rhs0, rhs1] = [(); 4].map(|_| self.namer.fresh_name());

                            let load = |w: &mut W, name: &str, src: &str| {
                                cline!(w, depth, "{} {name};", vf32.name)?;
                                cline!(
                                    w,
                                    depth,
                                    "__builtin_memcpy(&{name}, {src}, sizeof({name}));"
                                )
                            };

                            let mut deinterleave = |w: &mut W, ptr: &str, odd: &str, even: &str| {
                                let compressed_name = self.namer.fresh_name();
                                load(w, &compressed_name, ptr)?;
                                cline!(
                                    w,
                                    depth,
                                    "{0} {odd} = ({0}){shift_fn}(({1}){compressed_name}, 16);",
                                    vf32.name,
                                    vf32.native_type_name,
                                )?;
                                cline!(
                                    w,
                                    depth,
                                    "{} {even} = {};",
                                    vf32.name,
                                    bf16_even_lanes_expr(
                                        16,
                                        vf32.name,
                                        vf32.native_type_name,
                                        &compressed_name,
                                    ),
                                )
                            };

                            match cpu_kernel {
                                CpuKernel::DotProductLoopBf16Bf16F32 => {
                                    deinterleave(w, lhs_ptr, &lhs1, &lhs0)?;
                                    deinterleave(w, rhs_ptr, &rhs1, &rhs0)?;
                                }
                                CpuKernel::DotProductLoopF32InterleavedBf16F32 => {
                                    // The lhs is already dequantized, odd elements first.
                                    load(w, &lhs0, lhs_ptr)?;
                                    load(w, &lhs1, &format!("{lhs_ptr} + 8"))?;
                                    deinterleave(w, rhs_ptr, &rhs0, &rhs1)?;
                                }
                                _ => {
                                    // The lhs is already dequantized, in order.
                                    load(w, &lhs0, lhs_ptr)?;
                                    load(w, &lhs1, &format!("{lhs_ptr} + 8"))?;
                                    let compressed_name = self.namer.fresh_name();
                                    load(w, &compressed_name, rhs_ptr)?;
                                    cline!(w, depth, "{} {rhs0};", vf32.name)?;
                                    cline!(w, depth, "{} {rhs1};", vf32.name)?;
                                    cline!(
                                        w,
                                        depth,
                                        "cvtbf16_fp32_256({compressed_name}, &{rhs0}, &{rhs1});"
                                    )?;
                                    self.headers.emit_cvtbf16_fp32 = true;
                                }
                            }
                            accums.emit_update(w, depth, lane, &format!("{lhs0} * {rhs0}"))?;
                            accums.emit_update(w, depth, lane + 2, &format!("{lhs1} * {rhs1}"))
                        })?;
                        for accum_name in &accums.names {
                            cline!(w, depth, "{out_expr} += sum8({accum_name});")?;
                        }
                        Ok(())
                    }
                    Some(CpuKernel::PhysicalTransposeByte128) => {
                        let [in_lower, in_higher, out_lower, out_higher]: [String; 4] = [
                            (&arguments[0], 0),
                            (&arguments[0], 16),
                            (&arguments[1], 0),
                            (&arguments[1], 16),
                        ]
                        .into_iter()
                        .map(|(arg, idx)| self.c_vec_arg_add(arg, idx))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                        cline!(
                            w,
                            depth,
                            "{} = _mm_unpacklo_epi8({}, {});",
                            out_lower,
                            in_lower,
                            in_higher,
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = _mm_unpackhi_epi8({}, {});",
                            out_higher,
                            in_lower,
                            in_higher,
                        )
                    }
                    Some(CpuKernel::PhysicalTransposeByte256) => {
                        use CpuMemory::VRF;

                        let [in_lower, in_higher, out_lower, out_higher]: [String; 4] = [
                            (&arguments[0], 0),
                            (&arguments[0], 32),
                            (&arguments[1], 0),
                            (&arguments[1], 32),
                        ]
                        .into_iter()
                        .map(|(arg, idx)| self.c_vec_arg_add(arg, idx))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                        let intermediate_dtype = arguments[0].spec().dtype();
                        let intermediate_lower = self.make_buffer(
                            None,
                            &shape![1, 32],
                            Some(nz!(32u32)),
                            intermediate_dtype,
                            VRF.into(),
                        );
                        let intermediate_higher = self.make_buffer(
                            None,
                            &shape![1, 32],
                            Some(nz!(32u32)),
                            intermediate_dtype,
                            VRF.into(),
                        );
                        intermediate_lower.emit(w, InitType::None, depth)?;
                        intermediate_higher.emit(w, InitType::None, depth)?;

                        cline!(
                            w,
                            depth,
                            "{} = _mm256_unpacklo_epi8({}, {});",
                            intermediate_lower.name().unwrap(),
                            in_lower,
                            in_higher,
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = _mm256_unpackhi_epi8({}, {});",
                            intermediate_higher.name().unwrap(),
                            in_lower,
                            in_higher,
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = _mm256_permute2f128_si256({}, {}, 0x20);",
                            out_lower,
                            intermediate_lower.name().unwrap(),
                            intermediate_higher.name().unwrap(),
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = _mm256_permute2f128_si256({}, {}, 0x31);",
                            out_higher,
                            intermediate_lower.name().unwrap(),
                            intermediate_higher.name().unwrap(),
                        )
                    }
                    Some(CpuKernel::VectorInterleaveBf16F32) => {
                        let src_name = self.c_vec_arg(&arguments[0]);
                        let even_name = self.c_vec_arg(&arguments[1]);
                        let odd_name = self.c_vec_arg_add(&arguments[1], 8);

                        let (shift_fn, _, _, _) = vec_func_names(16);

                        let vf8 = get_vector(Tgt::vec_types(), Dtype::Float32, nz!(8u32));
                        cline!(
                            w,
                            depth,
                            "{1} = ({0}){shift_fn}(*({2}*)(&{3}), 16);",
                            vf8.name,
                            even_name,
                            vf8.native_type_name,
                            src_name
                        )?;
                        cline!(
                            w,
                            depth,
                            "{} = {};",
                            odd_name,
                            bf16_even_lanes_expr(16, vf8.name, vf8.native_type_name, &src_name),
                        )?;
                        Ok(())
                    }
                    Some(CpuKernel::VectorDeinterleaveF32Bf16) => todo!(),
                    Some(CpuKernel::VecScalarAssign) => {
                        let vector_size = arguments[1].spec().vector_size().unwrap();
                        let out_vector_type =
                            get_vector(Tgt::vec_types(), arguments[1].spec().dtype(), vector_size);

                        let volume = arguments[1].spec().volume().get();
                        let value = self.c_value_arg(&arguments[0]);
                        for offset in vector_offsets(volume, vector_size.get()) {
                            let out = self.c_vec_arg_add(&arguments[1], offset);
                            let ty = out_vector_type.name;
                            cline!(
                                w,
                                depth,
                                "{out} = ({value} - ({ty}){{}}); /* VecScalarAssign */"
                            )?;
                        }
                        Ok(())
                    }
                    Some(CpuKernel::DivideVec) => {
                        let vector_size = arguments[0].spec().vector_size().unwrap();
                        debug_assert_eq!(arguments[1].spec().vector_size(), Some(vector_size));
                        let volume = arguments[1].spec().volume().get();
                        for offset in vector_offsets(volume, vector_size.get()) {
                            let [lhs, rhs] =
                                [0, 1].map(|i| self.c_vec_arg_add(&arguments[i], offset));
                            cline!(w, depth, "{lhs} = {lhs} / {rhs}; /* DivideVec */")?;
                        }
                        Ok(())
                    }
                    Some(CpuKernel::ValueDivideVecScalar) => {
                        let exprs = self.c_value_args(arguments);
                        cline!(
                            w,
                            depth,
                            "{} = {} / {}; /* ValueDivideVecScalar */",
                            exprs[0],
                            exprs[0],
                            exprs[1],
                        )
                    }
                    Some(CpuKernel::DivideVecScalarReciprocal) => {
                        // scan_dim is the only non-matching dimension size
                        let scan_dim = {
                            let input_shape = arguments[0].spec().shape();
                            let denominator_shape = arguments[1].spec().shape();
                            input_shape
                                .iter()
                                .zip(denominator_shape)
                                .position(|(&in_size, &denom_size)| in_size != denom_size)
                                .map(|dim| u8::try_from(dim).unwrap())
                                .unwrap()
                        };

                        let vector_size =
                            divide_vec_scalar_reciprocal_vector_size::<Tgt>(arguments[0].spec())
                                .unwrap();
                        let vector_type =
                            get_vector(Tgt::vec_types(), arguments[0].spec().dtype(), vector_size);
                        self.headers.vector_type_defs.insert(vector_type);
                        let volume = arguments[0].spec().volume().get();
                        debug_assert_eq!(volume % vector_size.get(), 0);
                        let (denom_buffer, _) = self.c_arg_parts(&arguments[1]);

                        let reciprocal_vec = self.namer.fresh_name();
                        let denom_expr =
                            arguments[1]
                                .make_buffer_indexing_expr()
                                .map_vars(&mut |v| match v {
                                    BufferVar::TileIdx(dim, _) if dim != scan_dim => {
                                        AffineForm::from(v)
                                    }
                                    _ => AffineForm::zero(),
                                });
                        // Lowers to vdivss + vbroadcastss.
                        cline!(w, depth, "/* DivideVecScalarReciprocal */")?;
                        cline!(
                            w,
                            depth,
                            "{0} {reciprocal_vec} = (1.0f / ({1})) - ({0}){{}};",
                            vector_type.name,
                            self.c_index(&denom_buffer, &denom_expr),
                        )?;

                        let vector_temp = self.namer.fresh_name();
                        cline!(w, depth, "{} {vector_temp};", vector_type.name)?;
                        let strip_loop = self.strip_loop(vector_size, 1, [&arguments[0]]);
                        strip_loop.emit(w, depth, |w, depth, _, [ptr]| {
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy(&{vector_temp}, {ptr}, sizeof({vector_temp}));"
                            )?;
                            cline!(w, depth, "{vector_temp} *= {reciprocal_vec};")?;
                            cline!(
                                w,
                                depth,
                                "__builtin_memcpy({ptr}, &{vector_temp}, sizeof({vector_temp}));"
                            )
                        })
                    }
                    None => match kernel_type.into_avx512_kernel() {
                        Some(avx512_kernel) => {
                            self.emit_avx512_kernel(w, avx512_kernel, arguments, depth)
                        }
                        None => unimplemented!(),
                    },
                }
            }
            ImplNode::FunctionApp(_) => {
                panic!("FunctionApp should have been inlined");
            }
        };

        if let Some((workspace_id, workspace_name)) = workspace_scope {
            let removed_name = self.active_workspace_names.remove(&workspace_id);
            debug_assert_eq!(removed_name.as_deref(), Some(workspace_name.as_str()));
            emit_result?;
            cline!(w, depth, "free({workspace_name});")
        } else {
            emit_result
        }
    }

    fn emit_rolled_loop_nest<W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt>,
        depth: usize,
    ) -> fmt::Result {
        let nontrivial_axes = get_axis_steps(l);
        self.emit_main_loop_nest(w, l, &nontrivial_axes, depth)?;
        self.emit_boundary_regions(w, l, depth)
    }

    fn emit_main_loop_nest<W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt>,
        axes_to_emit: &[(u8, u32, u32)],
        depth: usize,
    ) -> fmt::Result {
        let mut iter_var_names = HashMap::new();
        for (axis, _, _) in axes_to_emit {
            iter_var_names
                .entry(*axis)
                .or_insert_with(|| self.namer.fresh_name());
        }

        // Bind tile indices to the axis' C iterator name in `self.loop_iter_names`.
        self.bind_tile_dimensions(l, &iter_var_names);

        if l.parallel {
            match self.thread_style {
                CpuCodeGenThreadStyle::OpenMP => {
                    self.headers.emit_cores_clamp = true;
                    cline!(
                        w,
                        depth,
                        "#pragma omp parallel for collapse({}) num_threads(cores_clamp({})) proc_bind(spread) schedule(static)",
                        axes_to_emit.len(),
                        l.full_steps(),
                    )?;
                }
                CpuCodeGenThreadStyle::Highway => {
                    if axes_to_emit.len() != 1 {
                        todo!("collapse loops for Highway parallel-for");
                    }
                    for (axis, full_steps, _) in axes_to_emit {
                        let var_name = iter_var_names.get(axis).unwrap();
                        // TODO: Lift the pool out into a kernel argument.
                        cline!(
                            w,
                            depth,
                            "pool.Run(0, {}, [&](const uint64_t {}, size_t) HWY_ATTR {{",
                            full_steps,
                            var_name,
                        )?;
                    }
                    self.emit(w, &l.bodies[0], depth + 1)?;
                    for _ in 0..axes_to_emit.len() {
                        cline!(w, depth, "}});")?;
                    }
                    return Ok(());
                }
            }
        }
        for (axis, full_steps, _) in axes_to_emit {
            let var_name = iter_var_names.get(axis).unwrap();
            cline!(
                w,
                depth,
                "for (int {var_name} = 0; {var_name} < {full_steps}; {var_name}++) {{"
            )?;
        }

        self.emit(w, &l.bodies[0], depth + 1)?;

        for _ in 0..axes_to_emit.len() {
            cline!(w, depth, "}}")?;
        }

        Ok(())
    }

    fn emit_boundary_regions<W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt>,
        depth: usize,
    ) -> fmt::Result {
        for body in &l.bodies[1..] {
            self.emit(w, body, depth)?;
        }
        Ok(())
    }

    fn emit_unrolled_loop<W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt>,
        depth: usize,
    ) -> fmt::Result {
        if l.parallel {
            todo!("Support parallel, unrolled loops");
        }

        // Main tiles
        let axis_steps = get_axis_steps(l);
        for axis_positions in axis_steps
            .iter()
            .map(|&(_, full_steps, _)| 0..full_steps)
            .multi_cartesian_product()
        {
            let axes_to_indices: HashMap<u8, u32> = axis_steps
                .iter()
                .enumerate()
                .map(|(i, (axis, _, _))| (*axis, axis_positions[i]))
                .collect();
            self.bind_tile_dimensions_for_unrolled(l, &axes_to_indices);
            self.emit(w, &l.bodies[0], depth)?;
        }

        // Emit boundary regions
        for body in &l.bodies[1..] {
            self.emit(w, body, depth)?;
        }

        Ok(())
    }

    /// Binds each dimension of each tile to the corresponding axis' C iterator name.
    /// If an axis is present in the [Loop] but not in `iter_var_names`, it is set to
    /// zero.
    fn bind_tile_dimensions(&mut self, l: &Loop<Tgt>, iter_var_names: &HashMap<u8, String>) {
        for loop_tile in &l.tiles {
            debug_assert_eq!(loop_tile.axes.len(), loop_tile.tile.shape().len());
            for tt in loop_tile.tile.tile_dim_terms() {
                let BufferVar::TileIdx(dim, _) = tt else {
                    unreachable!();
                };
                let axis = loop_tile.axes[usize::from(dim)];
                if let Some(var_name) = iter_var_names.get(&axis) {
                    self.loop_iter_bindings
                        .insert(tt.clone(), Either::Left(var_name.clone()));
                } else {
                    self.loop_iter_bindings.insert(tt.clone(), Either::Right(0));
                }
            }
        }
    }

    /// Binds each dimension of each tile to the corresponding axis step value for unrolled loops.
    /// If an axis is not present in `axes_to_indices`, it is set to zero.
    fn bind_tile_dimensions_for_unrolled(
        &mut self,
        l: &Loop<Tgt>,
        axes_to_indices: &HashMap<u8, u32>,
    ) {
        for loop_tile in &l.tiles {
            debug_assert_eq!(loop_tile.axes.len(), loop_tile.tile.shape().len());
            for tt in loop_tile.tile.tile_dim_terms() {
                let BufferVar::TileIdx(dim, _) = &tt else {
                    unreachable!();
                };
                let axis = loop_tile.axes[usize::from(*dim)];
                if let Some(&axis_step) = axes_to_indices.get(&axis) {
                    self.loop_iter_bindings
                        .insert(tt.clone(), Either::Right(i32::try_from(axis_step).unwrap()));
                } else {
                    self.loop_iter_bindings.insert(tt.clone(), Either::Right(0));
                }
            }
        }
    }

    /// Returns the C buffer and zero-based indexing expression for `arg`.
    ///
    /// This helper performs the common codegen lookup from view to backing tensor, then to the
    /// tensor's named C buffer, and finally to the flattened buffer indexing expression. It also
    /// zeroes point variables. For example, an argument at logical index `i` produces the parts
    /// that lower to C expressions such as `buf[i]` or `buf + i`.
    fn c_arg_parts<A>(&self, arg: &A) -> (CBuffer, NonAffineExpr<BufferVar>)
    where
        A: View<Tgt = Tgt>,
    {
        self.c_arg_parts_add(arg, 0)
    }

    /// Returns the C buffer and indexing expression for `arg` plus `elem_offset` elements.
    ///
    /// Like [`Self::c_arg_parts`], but adds `elem_offset` to the flattened element index. For
    /// example, an argument at logical index `i` with `elem_offset == 8` produces the parts that
    /// lower to C expressions such as `buf[i + 8]` or `buf + i + 8`.
    fn c_arg_parts_add<A>(&self, arg: &A, elem_offset: i32) -> (CBuffer, NonAffineExpr<BufferVar>)
    where
        A: View<Tgt = Tgt>,
    {
        let backing_tensor = arg.backing_tensor().unwrap();
        let buffer = self.name_env.get(&backing_tensor.identifier()).unwrap();
        let expr = zero_points(arg.make_buffer_indexing_expr()) + elem_offset;
        (buffer.clone(), expr)
    }

    /// Returns a C lvalue expression for `arg`.
    ///
    /// Use this helper when emitted C should read or write the value itself. For normal memory
    /// buffers it returns expressions such as `buf[i]`; for register-backed buffers it returns a
    /// register variable or lane expression such as `v012` or `v012[1]`.
    fn c_value_arg<A>(&self, arg: &A) -> String
    where
        A: View<Tgt = Tgt>,
    {
        self.c_value_arg_add(arg, 0)
    }

    /// Returns a C lvalue expression for `arg` plus `elem_offset` elements.
    ///
    /// Like [`Self::c_value_arg`], but adds `elem_offset` to the flattened element index. For
    /// example, an argument at logical index `i` with `elem_offset == 8` can lower to `buf[i + 8]`.
    fn c_value_arg_add<A>(&self, arg: &A, elem_offset: i32) -> String
    where
        A: View<Tgt = Tgt>,
    {
        let (buffer, expr) = self.c_arg_parts_add(arg, elem_offset);
        self.c_index(&buffer, &expr)
    }

    /// Returns C lvalue expressions for each argument.
    ///
    /// Like [`Self::c_value_arg`], but maps over a full argument list.
    fn c_value_args<A>(&self, arguments: &[A]) -> Vec<String>
    where
        A: View<Tgt = Tgt>,
    {
        arguments.iter().map(|arg| self.c_value_arg(arg)).collect()
    }

    /// Returns a C pointer expression for `arg`.
    ///
    /// Use this helper when emitted C should pass an address or perform pointer arithmetic. For
    /// heap, workspace, and pointer-backed buffers it returns expressions such as `buf + i`; for
    /// stack or register-backed buffers it returns address expressions such as `&buf[i]` or
    /// `&v012[1]`.
    pub fn c_ptr_arg<A>(&self, arg: &A) -> String
    where
        A: View<Tgt = Tgt>,
    {
        self.c_ptr_arg_add(arg, &NonAffineExpr::zero())
    }

    /// Returns a C pointer expression for `arg` plus `elem_offset` elements.
    ///
    /// Like [`Self::c_ptr_arg`], but adds `elem_offset` to the flattened element index. For
    /// example, an argument at logical index `i` with `elem_offset == 8` can lower to `buf + i + 8`
    /// or `&buf[i + 8]`.
    pub fn c_ptr_arg_add<A>(&self, arg: &A, elem_offset: &NonAffineExpr<CName>) -> String
    where
        A: View<Tgt = Tgt>,
    {
        let (buffer, expr) = self.c_arg_parts(arg);
        let expr = self.sub_expr_bindings(expr) + elem_offset.clone();
        self.c_index_ptr_bound(&buffer, &expr)
    }

    /// Returns a C vector-register expression for `arg`.
    ///
    /// Use this helper when emitted C should name the vector register that stores the argument.
    /// The expression must point at the start of a register vector; otherwise [`Self::c_index_vec`]
    /// panics. It returns expressions such as `v012`, not memory expressions such as `buf[i]`.
    fn c_vec_arg<A>(&self, arg: &A) -> String
    where
        A: View<Tgt = Tgt>,
    {
        self.c_vec_arg_add(arg, 0)
    }

    /// Returns a C vector-register expression for `arg` plus `elem_offset` elements.
    ///
    /// Like [`Self::c_vec_arg`], but adds `elem_offset` to the flattened element index before
    /// selecting the vector register.
    fn c_vec_arg_add<A>(&self, arg: &A, elem_offset: i32) -> String
    where
        A: View<Tgt = Tgt>,
    {
        let (buffer, expr) = self.c_arg_parts_add(arg, elem_offset);
        self.c_index_vec(&buffer, &expr)
    }

    /// Returns C vector-register expressions for each argument.
    ///
    /// Like [`Self::c_vec_arg`], but maps over a full argument list.
    fn c_vec_args<A>(&self, arguments: &[A]) -> Vec<String>
    where
        A: View<Tgt = Tgt>,
    {
        arguments.iter().map(|arg| self.c_vec_arg(arg)).collect()
    }

    /// Lowers [BufferVar]s from index expressions to corresponding C loop iterator names.
    ///
    /// Panics if a [BufferVar] isn't in `loop_iter_bindings`.
    fn sub_expr_bindings(&self, unbound_expr: NonAffineExpr<BufferVar>) -> NonAffineExpr<CName> {
        unbound_expr.map_vars(&mut |v| match self.loop_iter_bindings.get(&v) {
            Some(Either::Left(var_name)) => {
                AffineForm::from(NonAffine::Leaf(CName(var_name.clone())))
            }
            Some(Either::Right(c)) => NonAffineExpr::constant(*c),
            None => {
                // This shouldn't happen. If it does, it is probably because emit_rolled_loop didn't
                // encounter the BufferVar while enumerating `tile_dim_terms()` or `axes_to_emit`.
                panic!("var was not bound: {v:?}")
            }
        })
    }

    /// Returns a C expression referring to the value at a given expression.
    fn c_index(&self, buffer: &CBuffer, expr: &NonAffineExpr<BufferVar>) -> String {
        match buffer {
            CBuffer::HeapArray { name, .. }
            | CBuffer::WorkspaceSlice { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::Ptr { name, .. } => format!(
                "{}[{}]",
                name,
                expr_to_c(&self.sub_expr_bindings(expr.clone()))
            ),
            CBuffer::RegVars { .. } => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (name, vec_offset) = buffer.inner_reg_from_expr(&subbed_expr);
                debug_assert!(vec_offset.as_constant().is_some());
                if vec_offset == 0 {
                    name
                } else {
                    format!("{}[{}]", name, expr_to_c(&vec_offset))
                }
            }
        }
    }

    fn c_index_vec(&self, buffer: &CBuffer, expr: &NonAffineExpr<BufferVar>) -> String {
        // self is essentially unused, but included for consistency with c_index.
        #![allow(clippy::only_used_in_recursion)]

        match buffer {
            CBuffer::HeapArray { .. }
            | CBuffer::WorkspaceSlice { .. }
            | CBuffer::StackArray { .. }
            | CBuffer::Ptr { .. } => {
                unimplemented!()
            }
            CBuffer::RegVars { .. } => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (name, vec_offset) = buffer.inner_reg_from_expr(&subbed_expr);
                debug_assert!(vec_offset.as_constant().is_some());
                if vec_offset != 0 {
                    panic!("vec_offset must be 0, but was: {vec_offset:?}");
                }
                name.clone()
            }
        }
    }

    fn c_index_ptr(&self, buffer: &CBuffer, expr: &NonAffineExpr<BufferVar>) -> String {
        self.c_index_ptr_bound(buffer, &self.sub_expr_bindings(expr.clone()))
    }

    /// Like [`Self::c_index_ptr`], but `expr`'s variables are left as-is.
    fn c_index_ptr_bound(&self, buffer: &CBuffer, expr: &NonAffineExpr<CName>) -> String {
        match buffer {
            CBuffer::HeapArray { name, .. }
            | CBuffer::WorkspaceSlice { name, .. }
            | CBuffer::Ptr { name, .. } => format!("{name} + {}", expr_to_c(expr)),
            CBuffer::StackArray { name, .. } => format!("&{name}[{}]", expr_to_c(expr)),
            CBuffer::RegVars { .. } => {
                let (name, vec_offset) = buffer.inner_reg_from_expr(expr);
                debug_assert!(vec_offset.as_constant().is_some());
                if vec_offset == 0 {
                    format!("&{name}")
                } else {
                    format!("&{name}[{}]", expr_to_c(&vec_offset))
                }
            }
        }
    }

    /// Returns a pointer expression for `arg` at a logical point.
    ///
    /// This covers the AVX512 BF16 matmul A/B accesses where the K point is a C loop
    /// variable introduced inside the kernel emitter. That value is not represented as
    /// a scheduled `BufferVar::TileIdx`, so the normal `c_index_ptr` path cannot name it
    /// directly.
    fn c_index_ptr_with_point_exprs(
        &self,
        arg: &ViewE<Tgt>,
        point_exprs: &[(u8, NonAffineExpr<CName>)],
    ) -> String {
        let buffer = self.c_buffer_for_view(arg);
        let buffer_indexing_expr = arg.make_buffer_indexing_expr().map_vars(&mut |v| match v {
            BufferVar::Pt(dim, _) => point_exprs
                .iter()
                .find_map(|(point_dim, expr)| (*point_dim == dim).then(|| expr.clone()))
                .unwrap_or_else(NonAffineExpr::zero),
            BufferVar::TileIdx(_, _) => match self.loop_iter_bindings.get(&v) {
                Some(Either::Left(var_name)) => CName(var_name.clone()).into(),
                Some(Either::Right(c)) => NonAffineExpr::constant(*c),
                None => panic!("var was not bound: {v:?}"),
            },
        });
        self.c_index_ptr_bound(buffer, &buffer_indexing_expr)
    }

    /// Returns C lvalue expressions for each vector in VRF tensor `tensor`.
    ///
    /// Expressions are ordered by increasing vector offset in the view's
    /// flattened element order and are resolved through the tensor's named
    /// backing buffer in `name_env`.
    ///
    /// Panics if `tensor` is not in vector registers or if its backing buffer has no name.
    fn c_vec_exprs_for_tensor(&self, tensor: &ViewE<Tgt>) -> Vec<String> {
        let vector_volume = tensor.spec().volume().get();
        let vector_size = tensor
            .spec()
            .vector_size()
            .expect("tensor must be in vector registers");
        vector_offsets(vector_volume, vector_size.get())
            .map(|offset| self.c_vec_arg_add(tensor, offset))
            .collect()
    }

    /// Returns a C expression for the minimum finite value of a given Dtype.
    ///
    /// This will modify `self.headers` if a returned constant is defined in
    /// that header (e.g., float.h).
    pub fn c_min_const_for_dtype(&mut self, dt: Dtype) -> &'static str {
        match dt {
            Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 => "0",
            Dtype::Sint8 => "INT8_MIN",
            Dtype::Sint16 => "INT16_MIN",
            Dtype::Sint32 => "INT32_MIN",
            Dtype::Float32 => {
                self.headers.emit_float_include = true;
                "-FLT_MAX"
            }
            Dtype::Bfloat16 => {
                self.headers.emit_float_include = true;
                "(__bf16)(-FLT_MAX)"
            }
        }
    }

    fn thread_style_extra_args(&self) -> &[&'static str] {
        match self.thread_style {
            CpuCodeGenThreadStyle::OpenMP => &[],
            CpuCodeGenThreadStyle::Highway => &["hwy::ThreadPool& pool"],
        }
    }
}

impl<Tgt: Target> Default for CpuCodeGenerator<Tgt> {
    fn default() -> Self {
        CpuCodeGenerator {
            namer: Default::default(),
            name_env: Default::default(),
            loop_iter_bindings: Default::default(),
            headers: Default::default(),
            kernel_name: String::from("kernel"),
            thread_style: Default::default(),
            benchmark: false,
            static_allocation_plan: None,
            active_workspace_names: Default::default(),
            phantom: PhantomData,
        }
    }
}

/// Yields the element offset of each vector in a tensor with `volume`.
pub fn vector_offsets(volume: u32, vector_size: u32) -> impl Iterator<Item = i32> {
    debug_assert_eq!(volume % vector_size, 0);
    (0..volume)
        .step_by(usize::try_from(vector_size).unwrap())
        .map(|offset| i32::try_from(offset).unwrap())
}

const fn vec_func_names(
    vector_size_bf16: u32,
) -> (&'static str, &'static str, &'static str, &'static str) {
    match vector_size_bf16 {
        8 => (
            "_mm_slli_epi32",
            "_mm_blend_epi16",
            "_mm_setzero_si128",
            "_mm_set1_epi16",
        ),
        16 => (
            "_mm256_slli_epi32",
            "_mm256_blend_epi16",
            "_mm256_setzero_si256",
            "_mm256_set1_epi16",
        ),
        32 => (
            "_mm512_slli_epi32",
            "_mm512_mask_blend_epi16",
            "_mm512_setzero_si512",
            "_mm512_set1_epi16",
        ),
        _ => unimplemented!(),
    }
}

/// Builds `__builtin_shufflevector` indices that interleave the lower and upper
/// halves of two vectors into logical lane order.
fn interleave_vector_halves_shuffle_indices(vector_size: u32) -> String {
    debug_assert_eq!(vector_size % 2, 0);
    let half = vector_size / 2;
    (0..half)
        .flat_map(|i| [i, i + half])
        .map(|idx| idx.to_string())
        .join(", ")
}

/// Returns a vector expression that keeps bf16 values from even logical lanes
/// and zeros the odd lanes before the result is reinterpreted as f32.
fn bf16_even_lanes_expr(
    vector_size_bf16: u32,
    vector_type_name: &str,
    native_type_name: &str,
    input_expr: &str,
) -> String {
    let (_, blend_fn, zero_fn, _) = vec_func_names(vector_size_bf16);
    if vector_size_bf16 == 32 {
        format!(
            "({vector_type_name}){blend_fn}((__mmask32)0xAAAAAAAAu, {zero_fn}(), *({native_type_name}*)(&{input_expr}))"
        )
    } else {
        format!(
            "({vector_type_name}){blend_fn}({zero_fn}(), *({native_type_name}*)(&{input_expr}), 0xAA)"
        )
    }
}

/// Computes (axis, full_steps, total_steps) for each Spec axis with at least 2
/// steps.
fn get_axis_steps<Tgt: Target>(l: &Loop<Tgt>) -> Vec<(u8, u32, u32)> {
    let mut axis_steps = Vec::new();

    for (lt, dim) in unique_dims_per_axis(&l.tiles) {
        let axis = lt.axes[usize::from(dim)];
        let total_steps = lt.tile.steps_dim(dim);
        let full_steps = lt.tile.full_steps_dim(dim);

        if total_steps > 1 {
            axis_steps.push((axis, full_steps, total_steps));
        }
    }

    axis_steps
}

fn get_vector(
    vec_types: &'static [VecType],
    dtype: Dtype,
    vector_size: DimSize,
) -> &'static VecType {
    vec_types
        .iter()
        .find(|vec_type| {
            vec_type.dtype == dtype
                && vec_type.value_cnt == u8::try_from(vector_size.get()).unwrap()
        })
        .unwrap_or_else(|| {
            panic!("Vectors of type {dtype} and size {vector_size} are not supported")
        })
}

fn escape_c_string(input: &str) -> String {
    let mut escaped = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_ascii_graphic() || c == ' ' => escaped.push(c),
            c => escaped.push_str(&format!("\\x{:02x}", c as u32)),
        }
    }
    escaped
}

fn expr_to_c(e: &AffineForm<NonAffine<CName>>) -> String {
    let mut buf =
        e.0.iter()
            .map(|Term(coef, sym)| {
                let sym_string = cexpr_subexpr_to_c(sym);
                if *coef == 1 {
                    sym_string
                } else {
                    format!("{coef} * {sym_string}")
                }
            })
            .join(" + ");
    if e.1 != 0 {
        if buf.is_empty() {
            buf = e.1.to_string();
        } else {
            buf += &format!(" + {}", e.1);
        }
    }
    if buf.is_empty() {
        buf = String::from("0");
    }
    format!("({buf})")
}

fn cexpr_subexpr_to_c(subexpr: &NonAffine<CName>) -> String {
    match subexpr {
        NonAffine::Constant(c) => c.to_string(),
        NonAffine::Leaf(CName(name)) => name.clone(),
        NonAffine::FloorDiv(v, d) => format!("({} / {})", expr_to_c(v), d),
        NonAffine::Mod(v, m) => format!("({} % {})", expr_to_c(v), m),
    }
}

fn zero_points(expr: NonAffineExpr<BufferVar>) -> NonAffineExpr<BufferVar> {
    expr.map_vars(&mut |v| match v {
        BufferVar::TileIdx(_, _) => AffineForm::from(v),
        BufferVar::Pt(_, _) => AffineForm::zero(),
    })
}

fn emit_load_input_value<W: Write>(out: &mut W, idx: usize, dtype: Dtype) -> fmt::Result {
    match dtype {
        Dtype::Float32 => {
            cline!(
                out,
                2,
                "uint32_t converted = LE_TO_CPU32(((uint32_t *)mapped)[i]);"
            )?;
            cline!(
                out,
                2,
                "__builtin_memcpy(&((float *)dest{idx})[i], &converted, sizeof(converted));"
            )
        }
        Dtype::Bfloat16 => {
            cline!(
                out,
                2,
                "uint16_t converted = LE_TO_CPU16(((uint16_t *)mapped)[i]);"
            )?;
            cline!(
                out,
                2,
                "__builtin_memcpy(&((__bf16 *)dest{idx})[i], &converted, sizeof(converted));"
            )
        }
        Dtype::Uint8
        | Dtype::Sint8
        | Dtype::Uint16
        | Dtype::Sint16
        | Dtype::Uint32
        | Dtype::Sint32 => {
            let conv = match dtype {
                Dtype::Uint8 | Dtype::Sint8 => "",
                Dtype::Uint16 | Dtype::Sint16 => "LE_TO_CPU16",
                Dtype::Uint32 | Dtype::Sint32 => "LE_TO_CPU32",
                Dtype::Float32 | Dtype::Bfloat16 => unreachable!(
                    "float-like dtypes must be loaded through their integer bit storage"
                ),
            };
            cline!(
                out,
                2,
                "(({0} *)dest{idx})[i] = {1}((({0} *)mapped)[i]);",
                c_type(dtype),
                conv
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::c_utils::CName;
    use crate::codegen::CodeGen;
    use crate::common::{DimSize, Dtype};
    use crate::expr::{AffineForm, NonAffine, Term};
    use crate::layout::row_major;
    use crate::scheduling::parallel_softmax::ParallelSplitSoftmaxDenominatorAndMax;
    use crate::scheduling::ActionT;
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::shape;
    use crate::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
    use crate::target::{Avx2Target, CpuMemory, CpuTarget};
    use crate::tensorspec::TensorSpecAux;

    #[test]
    fn test_expr_zero_emitted() {
        assert_eq!(expr_to_c(&AffineForm(vec![], 0)), "(0)");
    }

    #[test]
    fn test_intercept_zero_not_emitted() {
        let x = CName(String::from("x"));
        let xa = NonAffine::Leaf(x);
        assert_eq!(expr_to_c(&AffineForm(vec![Term(2, xa)], 0)), "(2 * x)")
    }

    #[test]
    fn test_lower_to_c_expr() {
        let x = CName(String::from("x"));
        let y = CName(String::from("y"));
        assert_eq!(expr_to_c(&AffineForm(vec![], 1)), "(1)");
        assert_eq!(
            expr_to_c(&AffineForm(vec![Term(1, NonAffine::Leaf(x))], 1)),
            "(x + 1)"
        );
        assert_eq!(
            expr_to_c(&AffineForm(vec![Term(2, NonAffine::Leaf(y.clone()))], 3)),
            "(2 * y + 3)"
        );
        assert_eq!(
            expr_to_c(&AffineForm(
                vec![Term(
                    2,
                    NonAffine::FloorDiv(Box::new(NonAffine::Leaf(y.clone()).into()), 4)
                )],
                3
            )),
            "(2 * ((y) / 4) + 3)"
        );
        assert_eq!(
            expr_to_c(&AffineForm(
                vec![Term(
                    2,
                    NonAffine::Mod(Box::new(NonAffine::Leaf(y).into()), 4)
                )],
                3
            )),
            "(2 * ((y) % 4) + 3)"
        );
    }

    #[test]
    fn test_parallel_loop_codegen_omits_unit_axes_from_openmp_collapse() {
        let spec = softmax_denominator_and_max_spec(1, 64);
        let partial_shape = shape![1, 16];
        let implementation = ParallelSplitSoftmaxDenominatorAndMax {
            k: DimSize::new(4).unwrap(),
            partials_level: CpuMemory::L1,
            partials_layout: row_major(&partial_shape),
            partials_vector_size: None,
        }
        .apply(&spec)
        .expect("parallel split should apply");

        let mut source = String::new();
        implementation
            .emit(false, None, &mut source)
            .expect("codegen should succeed");

        assert!(source.contains("#pragma omp parallel for collapse(1)"));
    }

    #[test]
    fn test_interleave_vector_halves_shuffle_indices_8() {
        assert_eq!(
            interleave_vector_halves_shuffle_indices(8),
            "0, 4, 1, 5, 2, 6, 3, 7"
        );
    }

    #[test]
    fn test_interleave_vector_halves_shuffle_indices_16() {
        assert_eq!(
            interleave_vector_halves_shuffle_indices(16),
            "0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15"
        );
    }

    #[test]
    fn test_bf16_dotproductloops_iterate_over_all_of_k() {
        use crate::{layout, lspec};
        use CpuMemory::L1;

        let lspecs: [(CpuKernel, LogicalSpec<Avx2Target>); 3] = [
            (
                CpuKernel::DotProductLoopBf16Bf16F32,
                lspec!(MatmulAccum(
                    [1, 1, 128, 1],
                    (bf16, L1, row_major),
                    (bf16, L1, layout![0, 2, 1]),
                    (f32, L1, row_major),
                    serial
                )),
            ),
            (
                CpuKernel::DotProductLoopF32Bf16F32,
                lspec!(MatmulAccum(
                    [1, 1, 128, 1],
                    (f32, L1, row_major),
                    (bf16, L1, layout![0, 2, 1]),
                    (f32, L1, row_major),
                    serial
                )),
            ),
            (
                CpuKernel::DotProductLoopF32InterleavedBf16F32,
                lspec!(MatmulAccum(
                    [1, 1, 128, 1],
                    (f32, L1, layout![0, 1, 2, 2 oe(16)]),
                    (bf16, L1, layout![0, 2, 1]),
                    (f32, L1, row_major),
                    serial
                )),
            ),
        ];
        for (kernel, mut lspec) in lspecs {
            lspec.canonicalize().unwrap();
            let spec = Spec(lspec, <Avx2Target as CpuTarget>::max_mem());
            let mut source = String::new();
            // Emit the benchmark harness; the other can't yet fill odd-even layouts.
            spec.select(kernel)
                .emit(true, None, &mut source)
                .expect("codegen should succeed");
            let kernel_loop = source
                .lines()
                .find(|l| l.contains("+= 64) {"))
                .unwrap_or_else(|| panic!("{kernel:?} should emit a loop stepping by 64"));
            assert!(kernel_loop.contains("< 128;"), "{kernel:?}: {kernel_loop}");
        }
    }

    #[test]
    fn test_dotproductloop_accumulates_into_output() {
        use crate::layout::col_major;
        use crate::lspec;
        use CpuMemory::L1;

        let mut lspec: LogicalSpec<Avx2Target> = lspec!(MatmulAccum(
            [1, 1, 64, 1],
            (f32, L1, row_major),
            (f32, L1, col_major),
            (f32, L1, row_major),
            serial
        ));
        lspec.canonicalize().unwrap();
        let spec = Spec(lspec, <Avx2Target as CpuTarget>::max_mem());
        let mut source = String::new();
        spec.select(CpuKernel::DotProductLoop)
            .emit(false, None, &mut source)
            .expect("codegen should succeed");
        assert_eq!(source.matches("+= sum8(").count(), 4, "{source}");
        assert!(!source.contains(" = sum8("), "{source}");
    }

    fn softmax_denominator_and_max_spec(rows: u32, cols: u32) -> Spec<Avx2Target> {
        let shape = shape![rows, cols];
        let basics = PrimitiveBasics {
            typ: PrimitiveSpecType::SoftmaxDenominatorAndMax {
                scan_dim: 1,
                accum: false,
            },
            spec_shape: shape.clone(),
            dtypes: vec![Dtype::Float32; 3],
        };
        let auxes = basics
            .parameter_shapes()
            .into_iter()
            .map(|shape| TensorSpecAux {
                memory: CpuMemory::GL,
                layout: row_major(&shape),
                vector_size: None,
            })
            .collect();
        let mut spec = Spec(
            LogicalSpec::Primitive(basics, auxes, false),
            <Avx2Target as CpuTarget>::max_mem(),
        );
        spec.canonicalize()
            .expect("softmax denominator/max spec should canonicalize");
        spec
    }
}
