use itertools::{Either, Itertools};
use nonzero::nonzero as nz;
use std::collections::HashMap;
use std::fmt::{self, Debug, Write};
use std::iter;
use std::rc::Rc;

use super::namegen::NameGenerator;
use crate::codegen::c_utils::{c_type, CBuffer, CExprVar, InitType, VecType};
use crate::codegen::header::HeaderEmitter;
use crate::common::{DimSize, Dtype};
use crate::expr::{AffineForm, NonAffine, NonAffineExpr, Substitute, Term};
use crate::imp::blocks::Block;
use crate::imp::kernels::{Kernel, KernelType};
use crate::imp::loops::Loop;
use crate::imp::moves::TensorOrCacheView;
use crate::imp::Impl;
use crate::imp::ImplNode;
use crate::layout::BufferVar;
use crate::pprint::{pprint_write, ImplPrintStyle, PrintableAux};
use crate::shape;
use crate::target::{CpuMemoryLevel, Target};
use crate::utils::{indent, LinePrefixWrite, ASCII_CHARS};
use crate::views::{Param, Tensor, View};

const STACK_CUTOFF: u32 = 256;

#[derive(Default)]
pub struct CpuCodeGenerator<'a, Tgt: Target> {
    pub namer: NameGenerator,
    pub name_env: HashMap<Rc<Tensor<Tgt>>, CBuffer>,
    pub loop_iter_bindings: HashMap<BufferVar, Either<String, i32>>,
    pub param_bindings: HashMap<Param<Tgt>, &'a dyn View<Tgt = Tgt>>,
    pub headers: HeaderEmitter,
}

impl<'a, Tgt: Target<Level = CpuMemoryLevel>> CpuCodeGenerator<'a, Tgt> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Write a pretty-printed Impl as a C comment.
    pub fn emit_impl_comment<W: Write, Aux: PrintableAux + Debug>(
        &mut self,
        imp: &'a ImplNode<Tgt, Aux>,
        impl_style: ImplPrintStyle,
        out: &mut W,
    ) -> fmt::Result {
        let mut commenting_out = LinePrefixWrite::new(out, "// ");
        pprint_write(&mut commenting_out, imp, impl_style)?;
        Ok(())
    }

    pub fn emit_kernel<W: Write, Aux: PrintableAux + Debug>(
        &mut self,
        imp: &'a ImplNode<Tgt, Aux>,
        top_arg_tensors: &'a [Rc<Tensor<Tgt>>],
        bench: bool,
        out: &mut W,
    ) -> fmt::Result {
        debug_assert_eq!(top_arg_tensors.len(), usize::from(imp.parameter_count()));

        let mut main_body_str = String::new();
        writeln!(main_body_str, "__attribute__((noinline))\nvoid kernel(")?;
        for ((operand_idx, operand), tensor) in imp.parameters().enumerate().zip(top_arg_tensors) {
            let spec = tensor.spec();
            let parameter_name = self.namer.fresh_name();
            writeln!(
                main_body_str,
                "  {} *restrict {}{}",
                c_type(operand.dtype),
                parameter_name,
                if operand_idx + 1 < imp.parameter_count().into() {
                    ", "
                } else {
                    "\n) {"
                }
            )?;
            self.name_env.insert(
                Rc::clone(tensor),
                CBuffer::Ptr {
                    name: parameter_name,
                    dtype: spec.dtype(),
                },
            );
        }

        // Put the tensor->c_buffer binding into `self.name_env`. (And fill
        // tensors_as_trait_obj_ptrs.)
        let tensors_as_trait_obj_ptrs = top_arg_tensors
            .iter()
            .map(|tensor| tensor.as_ref() as &dyn View<Tgt = Tgt>)
            .collect::<Vec<_>>();

        imp.bind(&tensors_as_trait_obj_ptrs, &mut self.param_bindings);
        let depth = 1_usize;
        self.emit(&mut main_body_str, imp, depth)?;

        writeln!(main_body_str, "}}")?;

        self.headers.emit(Tgt::target_id(), out)?;
        out.write_char('\n')?;
        if bench {
            out.write_str(include_str!("../codegen/partials/benchmarking.c"))?;
            out.write_str("\n\n")?;
        }
        out.write_str(&main_body_str)
    }

    pub fn emit_load_inputs<W: Write>(
        &mut self,
        top_arg_tensors: &'a [Rc<Tensor<Tgt>>],
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
            write!(out, ", void *restrict dest{i}")?;
        }
        writeln!(out, ") {{")?;

        writeln!(out, "{}int fd;", indent(1))?;
        writeln!(out, "{}void *mapped;", indent(1))?;

        for (idx, input_tensor) in top_arg_tensors.iter().enumerate() {
            // Open and mmap the data.
            let value_cnt = input_tensor.0.volume().get();
            let byte_cnt = value_cnt * u32::from(input_tensor.spec().dtype().size());
            writeln!(out)?;
            writeln!(
                out,
                "{}if ((fd = open(paths[{idx}], O_RDONLY)) == -1)",
                indent(1)
            )?;
            writeln!(out, "{}return 1;", indent(2))?;
            writeln!(out, "{}if ((mapped = mmap(NULL, {byte_cnt}, PROT_READ, MAP_SHARED, fd, 0)) == MAP_FAILED)", indent(1))?;
            writeln!(out, "{}return 2;", indent(2))?;
            writeln!(out, "{}close(fd);", indent(1))?;

            // Move into destination argument.
            writeln!(out, "{}for (int i = 0; i < {value_cnt}; i++)", indent(1))?;
            writeln!(
                out,
                "{}(({1} *)dest{idx})[i] = {2}((({1} *)mapped)[i]);",
                indent(2),
                input_tensor.spec().dtype().c_type(),
                endian_convert_fn(input_tensor.spec().dtype())
            )?;

            // Un-map.
            writeln!(out, "{}if (munmap(mapped, {byte_cnt}) != 0)", indent(1))?;
            writeln!(out, "{}return 3;", indent(2))?;
        }

        writeln!(out, "{}return 0;", indent(1))?;
        writeln!(out, "}}")
    }

    pub fn emit_main<W: Write>(
        &mut self,
        top_arg_tensors: &'a [Rc<Tensor<Tgt>>],
        benchmark: bool,
        out: &mut W,
    ) -> fmt::Result {
        let mut main_body_str = String::new();
        let mut depth = 0_usize;

        writeln!(main_body_str, "int main(int argc, char *argv[]) {{")?;
        depth += 1;

        // Allocate a buffer for each Impl parameter and re-bind to a CBuffer corresponding to the
        // local-scope buffer. It will have been previously bound by emit_kernel to a CBuffer::Ptr.
        let mut parameter_buf_names = vec![];
        for kernel_argument in top_arg_tensors {
            let spec = kernel_argument.spec();
            let buf =
                self.make_buffer(spec.shape(), spec.vector_size(), spec.dtype(), spec.level());
            buf.emit(
                &mut main_body_str,
                if benchmark {
                    InitType::Random
                } else {
                    InitType::Zero
                },
                depth,
            )?;
            parameter_buf_names.push(self.c_index_ptr(&buf, &NonAffineExpr::zero(), None));
            self.name_env.insert(Rc::clone(kernel_argument), buf);
            writeln!(main_body_str)?;
        }

        // Load data, if provided.
        let (bottom_argc, full_argc) = if benchmark {
            (2, top_arg_tensors.len() + 2)
        } else {
            (1, top_arg_tensors.len() + 1)
        };
        writeln!(
            main_body_str,
            "{}if (argc == {}) {{",
            indent(depth),
            full_argc
        )?;
        depth += 1;
        writeln!(
            main_body_str,
            "{}int load_result = load_inputs(&argv[{}]{});",
            indent(depth),
            bottom_argc,
            parameter_buf_names
                .iter()
                .map(|n| format!(", {}", n))
                .join("")
        )?;
        writeln!(main_body_str, "{}if (load_result != 0) {{", indent(depth))?;
        depth += 1;
        writeln!(
            main_body_str,
            "{}fprintf(stderr, \"Error loading input tensors.\\n\");",
            indent(depth)
        )?;
        writeln!(main_body_str, "{}return 2;", indent(depth))?;
        depth -= 1;
        writeln!(main_body_str, "{}}}", indent(depth))?;
        depth -= 1;
        writeln!(
            main_body_str,
            "{}}} else if (argc != {bottom_argc}) {{",
            indent(depth)
        )?;
        depth += 1;
        writeln!(
            main_body_str,
            "{}fprintf(stderr, \"Unexpected number of arguments.\\n\");",
            indent(depth)
        )?;
        writeln!(main_body_str, "{}return 1;", indent(depth))?;
        depth -= 1;
        writeln!(main_body_str, "{}}}", indent(depth))?;

        let kernel_call_str = self.make_kernel_call(top_arg_tensors)?;
        if benchmark {
            self.emit_benchmarking(&kernel_call_str, depth, &mut main_body_str)?;
        } else {
            writeln!(main_body_str, "{}{}\n", indent(depth), kernel_call_str)?;
            self.emit_print_tensor(top_arg_tensors.last().unwrap(), depth, &mut main_body_str)?;
        }
        writeln!(main_body_str)?;

        // Free the buffers.
        for kernel_argument in top_arg_tensors {
            let buf = self.name_env.get(kernel_argument).unwrap();
            buf.emit_free(&mut main_body_str, depth)?;
        }

        writeln!(main_body_str, "\n{}return 0;", indent(depth))?;
        writeln!(main_body_str, "}}")?;

        out.write_str(&main_body_str)
    }

    fn make_kernel_call(
        &self,
        top_arg_tensors: &'a [Rc<Tensor<Tgt>>],
    ) -> Result<String, fmt::Error> {
        let mut kernel_call_str = String::new();
        write!(kernel_call_str, "kernel(")?;
        for (i, kernel_argument) in top_arg_tensors.iter().enumerate() {
            let a = self.name_env.get(kernel_argument).unwrap();
            write!(
                kernel_call_str,
                "{}{}",
                self.c_index_ptr(a, &AffineForm::zero(), None),
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
        writeln!(
            out,
            "{}const long long bench_samples = atoll(argv[1]);\n",
            indent(depth)
        )?;

        writeln!(
            out,
            "{}// Inlined kernel follows. This is for warm-up.",
            indent(depth)
        )?;
        writeln!(out, "{}{}", indent(depth), kernel_call_str)?;

        writeln!(out, "{}struct timespec start, end;", indent(depth))?;
        writeln!(
            out,
            "{}clock_gettime(CLOCK_MONOTONIC, &start);",
            indent(depth)
        )?;
        writeln!(out, "#pragma clang loop unroll(disable)")?; // preprocessor directives should not have indentation.
        writeln!(
            out,
            "{}for (long long bench_itr = 0; bench_itr < bench_samples; ++bench_itr) {{",
            indent(depth),
        )?;
        depth += 1;
        writeln!(out, "{}{}", indent(depth), kernel_call_str)?;
        depth -= 1;
        writeln!(out, "{}}}", indent(depth))?;
        writeln!(
            out,
            "{}clock_gettime(CLOCK_MONOTONIC, &end);",
            indent(depth)
        )?;
        writeln!(
            out,
            "{}struct timespec delta = ts_diff(start, end);",
            indent(depth)
        )?;
        writeln!(
            out,
            "{}printf(\"cpu: %llds %lldns\\n\", (long long)delta.tv_sec, (long long)delta.tv_nsec);",
            indent(depth)
        )?;

        Ok(())
    }

    fn emit_print_tensor<W: Write>(
        &mut self,
        tensor: &Tensor<Tgt>,
        mut depth: usize,
        out: &mut W,
    ) -> Result<(), fmt::Error> {
        let rank = tensor.shape().len();

        let shape_str = tensor.shape().iter().map(ToString::to_string).join("x");
        writeln!(out, "{}printf(\"{}\\n\");", indent(depth), shape_str)?;

        // TODO: Generate unique names. ASCII_CHARS could lead to conflicts in the future.
        for (dim, (d, n)) in tensor.shape().iter().copied().zip(ASCII_CHARS).enumerate() {
            writeln!(
                out,
                "{}for (int {n} = 0; {n} < {d}; {n}++) {{",
                indent(depth)
            )?;
            self.loop_iter_bindings.insert(
                BufferVar::Pt(dim.try_into().unwrap(), tensor.identifier()),
                Either::Left(n.to_string()),
            );
        }

        depth += 1;

        debug_assert_eq!(tensor, tensor.backing_tensor(&self.param_bindings).unwrap());
        let buffer = self.name_env.get(tensor).unwrap();
        let buffer_indexing_expr = tensor.make_buffer_indexing_expr(&self.param_bindings);
        writeln!(
            out,
            "{}printf(\"%\" {} \" \", {});",
            indent(depth),
            tensor.spec().dtype().int_fmt_macro(),
            self.c_index(buffer, &buffer_indexing_expr, None),
        )?;
        depth -= 1;

        writeln!(out, "{}}}", indent(depth))?;
        for _ in 0..rank - 1 {
            writeln!(out, "{}printf(\"\\n\");", indent(depth))?;
            writeln!(out, "{}}}", indent(depth))?;
        }
        Ok(())
    }

    fn make_buffer(
        &mut self,
        shape: &[DimSize],
        vector_size: Option<DimSize>,
        dtype: Dtype,
        level: CpuMemoryLevel,
    ) -> CBuffer {
        debug_assert_eq!(vector_size.is_some(), level == CpuMemoryLevel::VRF);

        let name = self.namer.fresh_name();
        let size = shape.iter().map(|d| d.get()).product::<u32>();
        match level {
            CpuMemoryLevel::VRF => {
                let vector_size = vector_size.unwrap();
                let vec_type = get_vector(Tgt::vec_types(), dtype, vector_size);
                self.headers.vector_type_defs.insert(vec_type);

                let vector_size = vector_size.get();
                debug_assert_eq!(size % vector_size, 0);
                let name = self.namer.fresh_name();
                if size == vector_size {
                    CBuffer::SingleVecVar { name, vec_type }
                } else {
                    let inner_vecs = (0..(size / vector_size))
                        .map(|_| CBuffer::SingleVecVar {
                            name: self.namer.fresh_name(),
                            vec_type,
                        })
                        .collect::<Vec<_>>();

                    CBuffer::VecVars { inner_vecs }
                }
            }
            CpuMemoryLevel::RF => {
                if size > 1 {
                    CBuffer::StackArray { name, size, dtype }
                } else {
                    CBuffer::ValueVar { name, dtype }
                }
            }
            CpuMemoryLevel::L1 | CpuMemoryLevel::GL => {
                if size * u32::from(dtype.size()) > STACK_CUTOFF {
                    CBuffer::HeapArray { name, size, dtype }
                } else {
                    CBuffer::StackArray { name, size, dtype }
                }
            }
        }
    }

    fn emit<Aux: PrintableAux + Debug, W: Write>(
        &mut self,
        w: &mut W,
        imp: &ImplNode<Tgt, Aux>,
        depth: usize,
    ) -> fmt::Result {
        if let Some(h) = imp.aux().c_header() {
            writeln!(w, "{}// {}", indent(depth), h)?;
        }
        match imp {
            ImplNode::Loop(l) => {
                // Emit a C loop nest or, if any tensor view is requires unrolling (i.e., vector
                // tensors), unroll the loop, emitting the body repeatedly.
                //
                // It's not always necessary to unroll every emitted C loop because not all axes
                // range over different vector registers. This is pretty rare though, so, for
                // simplicity, every Impl loop is emitted as either as a C loop nest or fully
                // unrolled.
                if l.tiles.iter().any(|loop_tile| {
                    self.name_env
                        .get(loop_tile.tile.backing_tensor(&self.param_bindings).unwrap())
                        .unwrap()
                        .needs_unroll()
                }) {
                    self.emit_unrolled_loop(w, l, depth)
                } else {
                    self.emit_rolled_loop(w, l, depth)
                }
            }
            ImplNode::MoveLet(move_let) => {
                match &move_let.introduced {
                    TensorOrCacheView::Tensor(tensor) => {
                        // Emit variable declaration(s) and store association between the
                        // CBuffer and Tensor.
                        let spec = move_let.introduced.spec();
                        let dest_buffer = self.make_buffer(
                            spec.shape(),
                            spec.vector_size(),
                            spec.dtype(),
                            spec.level(),
                        );
                        dest_buffer.emit(w, InitType::None, depth)?;

                        self.name_env.insert(Rc::clone(tensor), dest_buffer);
                    }
                    TensorOrCacheView::CacheView(_) => (),
                };

                if let Some(prologue) = move_let.prologue() {
                    self.emit(w, prologue, depth)?;
                }
                self.emit(w, move_let.main_stage(), depth)?;
                if let Some(epilogue) = move_let.epilogue() {
                    self.emit(w, epilogue, depth)?;
                }
                Ok(())
            }
            ImplNode::Block(Block {
                stages,
                bindings: _,
                parameters: _,
                aux: _,
            }) => {
                for stage in stages {
                    self.emit(w, stage, depth)?;
                }
                Ok(())
            }
            ImplNode::Pipeline(_) => todo!("Emit code for Pipeline"),
            ImplNode::SpecApp(p) => {
                self.headers.emit_stdbool_and_assert_headers = true;
                writeln!(
                    w,
                    "{}/* {}({}) */",
                    indent(depth),
                    p.0,
                    iter::repeat("_").take(p.1.len()).join(", ")
                )?;
                writeln!(w, "{}assert(false);  /* Missing Impl */", indent(depth))
            }
            ImplNode::Kernel(Kernel {
                kernel_type,
                arguments,
                aux: _,
            }) => {
                match kernel_type {
                    KernelType::MultAdd => {
                        let exprs = self.param_args_to_c_indices(arguments, |_i, a, b| {
                            self.c_index(a, b, None)
                        });
                        writeln!(
                            w,
                            "{}{} += {} * {};  /* MultAdd */",
                            indent(depth),
                            exprs[2],
                            exprs[0],
                            exprs[1]
                        )
                    }
                    KernelType::ValueAssign => {
                        let exprs = self.param_args_to_c_indices(arguments, |_i, a, b| {
                            self.c_index(a, b, None)
                        });
                        writeln!(w, "{}{} = {};", indent(depth), exprs[1], exprs[0])
                    }
                    KernelType::MemsetZero => {
                        // TODO: Merge this duplicate `exprs` block. It's used also in the ValueAssign.
                        debug_assert_eq!(arguments.len(), 1);
                        let backing_tensor =
                            arguments[0].backing_tensor(&self.param_bindings).unwrap();
                        let buffer = self.name_env.get(backing_tensor).unwrap();
                        let mut buffer_indexing_expr =
                            arguments[0].make_buffer_indexing_expr(&self.param_bindings);
                        buffer_indexing_expr = zero_points(buffer_indexing_expr);
                        let arg_expr = self.c_index_ptr(buffer, &buffer_indexing_expr, None);
                        writeln!(
                            w,
                            "{}memset((void *)({arg_expr}), 0, {});",
                            indent(depth),
                            arguments[0].1.bytes_used()
                        )
                    }
                    KernelType::VectorZero => {
                        let exprs = self.param_args_to_c_indices(arguments, |_, a, b| {
                            self.c_index_vec(a, b, None)
                        });
                        writeln!(w, "{}{} *= 0;  /* VectorZero */", indent(depth), exprs[0])
                    }
                    KernelType::VectorAssign => {
                        let dtype = arguments[0].spec().dtype();
                        let volume = arguments[0].spec().volume();

                        let exprs = self.param_args_to_c_indices(arguments, |_, a, b| {
                            self.c_index_ptr(a, b, None)
                        });
                        let vtype = get_vector(Tgt::vec_types(), dtype, volume);
                        let itype = vtype.native_type_name;
                        if arguments.iter().all(|a| a.1.aligned()) {
                            writeln!(
                                w,
                                "{}*({} *)({}) = (*({} *)({}));  /* VectorAssign */",
                                indent(depth),
                                itype,
                                exprs[1],
                                itype,
                                exprs[0]
                            )
                        } else {
                            writeln!(
                                w,
                                "{}{}(({} *)({}), {}(({} *)({})));  /* VectorAssign */",
                                indent(depth),
                                vtype.store_fn,
                                itype,
                                exprs[1],
                                vtype.load_fn,
                                itype,
                                exprs[0]
                            )
                        }
                    }
                    KernelType::BroadcastVecMultAdd => {
                        let vector_size = arguments[2].spec().vector_size().unwrap().get();
                        let volume = arguments[2].spec().volume().get();
                        debug_assert_eq!(volume % vector_size, 0);
                        let vector_count = volume / vector_size;
                        for vector_idx in 0..vector_count {
                            let exprs =
                                self.param_args_to_c_indices(arguments, |i, a, b| match i {
                                    0 => self.c_index(a, b, None),
                                    1 | 2 => self.c_index_vec(
                                        a,
                                        &(b.clone()
                                            + i32::try_from(vector_idx * vector_size).unwrap()),
                                        None,
                                    ),
                                    _ => unreachable!(),
                                });
                            writeln!(
                                w,
                                "{}{} += {} * {}; /* BroadcastVecMultAdd */",
                                indent(depth),
                                exprs[2],
                                exprs[0],
                                exprs[1]
                            )?;
                        }
                        Ok(())
                    }
                    KernelType::TwoVecBroadcastVecMultAdd => {
                        let vector_size = arguments[2].spec().vector_size().unwrap().get();
                        let volume = arguments[2].spec().volume().get();
                        debug_assert_eq!(volume % vector_size, 0);
                        let vector_count = volume / vector_size;

                        for vector_idx in 0..vector_count {
                            let exprs =
                                self.param_args_to_c_indices(arguments, |i, a, b| match i {
                                    0 => self.c_index_ptr(a, b, None),
                                    1 | 2 => self.c_index_vec(
                                        a,
                                        &(b.clone()
                                            + i32::try_from(vector_idx * vector_size).unwrap()),
                                        None,
                                    ),
                                    _ => unreachable!(),
                                });

                            // TODO: Lift the broadcast out of this loop.
                            let broadcast_name = self.namer.fresh_name();
                            writeln!(w, "/* TwoVecBroadcastVecMultAdd */")?;
                            writeln!(
                                w,
                                "{}__m256i {} = _mm256_set1_epi16(*(int16_t *)({}));",
                                indent(depth),
                                broadcast_name,
                                exprs[0]
                            )?;

                            // matmul (k=2) the broadcast vector with the rhs vectors.
                            writeln!(
                                w,
                                "{}{} = _mm256_add_epi16({}, _mm256_maddubs_epi16({}, {}));",
                                indent(depth),
                                exprs[2],
                                exprs[2],
                                broadcast_name,
                                exprs[1]
                            )?;
                        }
                        Ok(())
                    }
                    KernelType::PhysicalTransposeByte128 => {
                        let [in_lower, in_higher, out_lower, out_higher]: [String; 4] = [
                            (&arguments[0], 0),
                            (&arguments[0], 16),
                            (&arguments[1], 0),
                            (&arguments[1], 16),
                        ]
                        .into_iter()
                        .map(|(arg, idx)| {
                            let buffer = self
                                .name_env
                                .get(arg.backing_tensor(&self.param_bindings).unwrap())
                                .unwrap();
                            let buffer_indexing_expr =
                                zero_points(arg.make_buffer_indexing_expr(&self.param_bindings))
                                    + idx;
                            println!("About to index with expr: {:?}", buffer_indexing_expr);
                            self.c_index_vec(buffer, &buffer_indexing_expr, None)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                        writeln!(
                            w,
                            "{}{} = _mm_unpacklo_epi8({}, {});",
                            indent(depth),
                            out_lower,
                            in_lower,
                            in_higher,
                        )?;
                        writeln!(
                            w,
                            "{}{} = _mm_unpackhi_epi8({}, {});",
                            indent(depth),
                            out_higher,
                            in_lower,
                            in_higher,
                        )
                    }
                    KernelType::PhysicalTransposeByte256 => {
                        use CpuMemoryLevel::VRF;

                        let [in_lower, in_higher, out_lower, out_higher]: [String; 4] = [
                            (&arguments[0], 0),
                            (&arguments[0], 32),
                            (&arguments[1], 0),
                            (&arguments[1], 32),
                        ]
                        .into_iter()
                        .map(|(arg, idx)| {
                            let buffer = self
                                .name_env
                                .get(arg.backing_tensor(&self.param_bindings).unwrap())
                                .unwrap();
                            let buffer_indexing_expr =
                                zero_points(arg.make_buffer_indexing_expr(&self.param_bindings))
                                    + idx;
                            self.c_index_vec(buffer, &buffer_indexing_expr, None)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                        let intermediate_dtype = arguments[0].spec().dtype();
                        let intermediate_lower = self.make_buffer(
                            &shape![1, 32],
                            Some(nz!(32u32)),
                            intermediate_dtype,
                            VRF,
                        );
                        let intermediate_higher = self.make_buffer(
                            &shape![1, 32],
                            Some(nz!(32u32)),
                            intermediate_dtype,
                            VRF,
                        );
                        intermediate_lower.emit(w, InitType::None, depth)?;
                        intermediate_higher.emit(w, InitType::None, depth)?;

                        writeln!(
                            w,
                            "{}{} = _mm256_unpacklo_epi8({}, {});",
                            indent(depth),
                            intermediate_lower.name().unwrap(),
                            in_lower,
                            in_higher,
                        )?;
                        writeln!(
                            w,
                            "{}{} = _mm256_unpackhi_epi8({}, {});",
                            indent(depth),
                            intermediate_higher.name().unwrap(),
                            in_lower,
                            in_higher,
                        )?;
                        writeln!(
                            w,
                            "{}{} = _mm256_permute2f128_si256({}, {}, 0x20);",
                            indent(depth),
                            out_lower,
                            intermediate_lower.name().unwrap(),
                            intermediate_higher.name().unwrap(),
                        )?;
                        writeln!(
                            w,
                            "{}{} = _mm256_permute2f128_si256({}, {}, 0x31);",
                            indent(depth),
                            out_higher,
                            intermediate_lower.name().unwrap(),
                            intermediate_higher.name().unwrap(),
                        )
                    }
                }
            }
        }
    }

    fn emit_rolled_loop<Aux: PrintableAux + Debug, W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt, Aux>,
        depth: usize,
    ) -> fmt::Result {
        let axes_to_emit = axis_order_and_steps(l).collect::<Vec<_>>();

        // Map non-degen. axis names to fresh loop iterator names.
        let iter_var_names = axes_to_emit
            .iter()
            .map(|(axis, _)| (*axis, self.namer.fresh_name()))
            .collect::<HashMap<_, _>>();

        // Associate each of the tile indices in each LoopTile with the correct
        // name and store that association in the `self.loop_iter_names`.
        for loop_tile in &l.tiles {
            for tt in loop_tile.tile.tile_dim_terms() {
                let BufferVar::TileIdx(dim, _) = tt else {
                    unreachable!();
                };
                let axis = loop_tile.axes[usize::from(dim)];
                if let Some(axis_loop_iter_name) = iter_var_names.get(&axis) {
                    self.loop_iter_bindings
                        .insert(tt.clone(), Either::Left(axis_loop_iter_name.clone()));
                }
            }
        }

        if l.parallel {
            writeln!(
                w,
                "{}#pragma omp parallel for collapse({}) schedule(static)",
                indent(depth),
                axes_to_emit.len()
            )?;
        }

        for (axis, steps) in &axes_to_emit {
            let var_name = iter_var_names.get(axis).unwrap();
            writeln!(
                w,
                "{}for (int {} = 0; {} < {}; {}++) {{",
                indent(depth),
                var_name,
                var_name,
                steps,
                var_name
            )?;
        }

        self.emit(w, &l.body, depth + 1)?;

        for _ in 0..axes_to_emit.len() {
            writeln!(w, "{}}}", indent(depth))?;
        }
        Ok(())
    }

    fn emit_unrolled_loop<Aux: PrintableAux + Debug, W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<Tgt, Aux>,
        depth: usize,
    ) -> fmt::Result {
        if l.parallel {
            todo!("Support parallel, unrolled loops");
        }

        let axes_to_emit = axis_order_and_steps(l).collect::<Vec<_>>();

        for pt in axes_to_emit
            .iter()
            .map(|&(_, steps)| 0..steps)
            .multi_cartesian_product()
        {
            // Map the axes we'll emit to their index for a single step of the unrolled loop.
            // TODO: Allocating a HashMap is overkill.
            let axes_to_indices = axes_to_emit
                .iter()
                .zip(pt)
                .map(|((axis, _), axis_step)| (*axis, axis_step))
                .collect::<HashMap<_, _>>();

            // Bind all in loop_iter_bindings. On subsequent loop iterations, this will
            // overwrite.
            for loop_tile in &l.tiles {
                for tt in loop_tile.tile.tile_dim_terms() {
                    let BufferVar::TileIdx(dim, _) = &tt else {
                        unreachable!();
                    };
                    let axis = loop_tile.axes[usize::from(*dim)];
                    if let Some(axis_step) = axes_to_indices.get(&axis) {
                        self.loop_iter_bindings.insert(
                            tt.clone(),
                            Either::Right(i32::try_from(*axis_step).unwrap()),
                        );
                    }
                }
            }

            // Emit the body once for each step
            self.emit(w, &l.body, depth)?;
        }
        Ok(())
    }

    fn param_args_to_c_indices<F>(&self, arguments: &[Param<Tgt>], f: F) -> Vec<String>
    where
        F: Fn(usize, &CBuffer, &NonAffineExpr<BufferVar>) -> String,
    {
        arguments
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                let backing_tensor = arg.backing_tensor(&self.param_bindings).unwrap();
                let buffer = self.name_env.get(backing_tensor).unwrap();
                let mut buffer_indexing_expr = arg.make_buffer_indexing_expr(&self.param_bindings);
                buffer_indexing_expr = zero_points(buffer_indexing_expr);
                f(idx, buffer, &buffer_indexing_expr)
            })
            .collect()
    }

    fn sub_expr_bindings(&self, unbound_expr: NonAffineExpr<BufferVar>) -> NonAffineExpr<CExprVar> {
        unbound_expr.map_vars(&mut |v| match self.loop_iter_bindings.get(&v) {
            Some(Either::Left(var_name)) => {
                AffineForm::from(NonAffine::Leaf(CExprVar::CName(var_name.clone())))
            }
            Some(Either::Right(c)) => NonAffineExpr::constant(*c),
            None => AffineForm::from(NonAffine::Leaf(CExprVar::Buffer(v))),
        })
    }

    /// Returns a C expression referring to the value at a given expression.
    ///
    /// Additionally, `reinterpret` may be provided to introduce a type cast.
    /// This is useful for interpreting a (partial) buffer as a vector type.
    fn c_index(
        &self,
        buffer: &CBuffer,
        expr: &NonAffineExpr<BufferVar>,
        reinterpret: Option<String>,
    ) -> String {
        match buffer {
            CBuffer::HeapArray { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::Ptr { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => format!(
                    "{}[{}]",
                    name,
                    expr_to_c(&self.sub_expr_bindings(expr.clone()))
                ),
            },
            CBuffer::ValueVar { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => name.clone(),
            },
            CBuffer::SingleVecVar { name, .. } => {
                if let Some(reinterpret) = reinterpret {
                    debug_assert_eq!(expr, 0);
                    format!("*({} *)(&{})", reinterpret, name)
                } else {
                    format!(
                        "{}[{}]",
                        name,
                        expr_to_c(&self.sub_expr_bindings(expr.clone()))
                    )
                }
            }
            CBuffer::VecVars { .. } => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index(
                    inner_vec_buffer,
                    &AffineForm::constant(vec_offset.try_into().unwrap()),
                    reinterpret,
                )
            }
        }
    }

    fn c_index_vec(
        &self,
        buffer: &CBuffer,
        expr: &NonAffineExpr<BufferVar>,
        reinterpret: Option<String>,
    ) -> String {
        // self is essentially unused, but included for consistency with c_index.
        #![allow(clippy::only_used_in_recursion)]

        match buffer {
            CBuffer::HeapArray { .. }
            | CBuffer::StackArray { .. }
            | CBuffer::ValueVar { .. }
            | CBuffer::Ptr { .. } => {
                unimplemented!()
            }
            CBuffer::SingleVecVar { name, .. } => {
                if expr != 0 {
                    panic!("expr must be 0, but was: {:?}", expr);
                }
                if reinterpret.is_some() {
                    unimplemented!();
                }
                name.clone()
            }
            CBuffer::VecVars { .. } => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index_vec(
                    inner_vec_buffer,
                    &AffineForm::constant(vec_offset.try_into().unwrap()),
                    reinterpret,
                )
            }
        }
    }

    fn c_index_ptr(
        &self,
        buffer: &CBuffer,
        expr: &NonAffineExpr<BufferVar>,
        reinterpret: Option<String>,
    ) -> String {
        match buffer {
            CBuffer::HeapArray { name, .. } | CBuffer::Ptr { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => {
                    format!(
                        "{} + {}",
                        name,
                        expr_to_c(&self.sub_expr_bindings(expr.clone()))
                    )
                }
            },
            CBuffer::StackArray { .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => format!("&{}", self.c_index(buffer, expr, None)),
            },
            CBuffer::ValueVar { .. } => {
                if reinterpret.is_some() {
                    unimplemented!();
                };
                let mut ptr_str = format!("&{}", self.c_index(buffer, expr, None));
                if ptr_str.ends_with("[0]") {
                    ptr_str = ptr_str[..ptr_str.len() - 3].to_string();
                }
                ptr_str
            }
            CBuffer::SingleVecVar { name, .. } => {
                if reinterpret.is_some() {
                    unimplemented!();
                };
                if expr == 0 {
                    format!("&{}", name)
                } else {
                    format!("&{}", self.c_index(buffer, expr, None))
                }
            }
            CBuffer::VecVars { .. } => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index_ptr(
                    inner_vec_buffer,
                    &AffineForm::constant(vec_offset.try_into().unwrap()),
                    reinterpret,
                )
            }
        }
    }
}

fn axis_order_and_steps<Tgt: Target, Aux: Clone>(
    l: &Loop<Tgt, Aux>,
) -> impl Iterator<Item = (u8, u32)> + '_ {
    // TODO: Choose according to a skip-minimizing heuristic.
    let result = l
        .tiles
        .iter()
        .flat_map(|t| {
            t.axes.iter().enumerate().filter_map(|(dim_idx, axis)| {
                let steps = t.tile.steps_dim(dim_idx.try_into().unwrap());
                debug_assert_ne!(steps, 0);
                if steps == 1 {
                    None
                } else {
                    Some((*axis, steps))
                }
            })
        })
        .unique();

    // Assert that `r` doesn't contain duplicate axes. This is expensive, so only do so in debug
    // builds.
    #[cfg(debug_assertions)]
    {
        let mut seen = std::collections::HashSet::new();
        let rv = result.clone().collect::<Vec<_>>();
        for (axis, _) in rv.clone() {
            if !seen.insert(axis) {
                panic!("Duplicate axis in result: {}", axis);
            }
        }
    }

    result
}

fn get_vector(
    vec_types: &'static [VecType; 12],
    dtype: Dtype,
    vector_size: DimSize,
) -> &'static VecType {
    vec_types
        .iter()
        .find(|vec_type| {
            vec_type.dtype == dtype
                && vec_type.value_cnt == u8::try_from(vector_size.get()).unwrap()
        })
        .expect("VecType to match dtype and volume of vector_size")
}

fn expr_to_c(e: &AffineForm<NonAffine<CExprVar>>) -> String {
    let mut buf =
        e.0.iter()
            .map(|Term(coef, sym)| {
                let sym_string = cexpr_subexpr_to_c(sym);
                if *coef == 1 {
                    sym_string
                } else {
                    format!("{} * {}", coef, sym_string)
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
    format!("({})", buf)
}

fn cexpr_subexpr_to_c(subexpr: &NonAffine<CExprVar>) -> String {
    match subexpr {
        NonAffine::Constant(c) => c.to_string(),
        NonAffine::Leaf(v) => match v {
            CExprVar::CName(name) => name.clone(),
            CExprVar::Buffer(_) => {
                // TODO: Guarantee all terms are C names at the type level.
                unreachable!("Expected all terms to be C names");
            }
        },
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

/// Returns the function/macro name for converting a value of some type to processor byte order.
///
/// The functions/macros are included via `partials/cpu.c`.
const fn endian_convert_fn(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 | Dtype::Sint8 => "",
        Dtype::Uint16 | Dtype::Sint16 => "LE_TO_CPU16",
        Dtype::Uint32 | Dtype::Sint32 => "LE_TO_CPU32",
    }
}

#[cfg(test)]
mod tests {
    use super::expr_to_c;
    use crate::codegen::c_utils::CExprVar;
    use crate::expr::{AffineForm, NonAffine, Term};

    #[test]
    fn test_expr_zero_emitted() {
        assert_eq!(expr_to_c(&AffineForm(vec![], 0)), "(0)");
    }

    #[test]
    fn test_intercept_zero_not_emitted() {
        let x = CExprVar::CName(String::from("x"));
        let xa = NonAffine::Leaf(x);
        assert_eq!(expr_to_c(&AffineForm(vec![Term(2, xa)], 0)), "(2 * x)")
    }

    #[test]
    fn test_lower_to_c_expr() {
        let x = CExprVar::CName(String::from("x"));
        let y = CExprVar::CName(String::from("y"));
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
}
