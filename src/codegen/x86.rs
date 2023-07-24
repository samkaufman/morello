use itertools::{Either, Itertools};
use std::collections::HashMap;
use std::fmt::{self, Debug, Write};
use std::rc::Rc;

use super::c_utils::{CBuffer, CExprTerm, VecType};
use super::namegen::NameGenerator;
use super::CodeGen;
use crate::codegen::c_utils::{c_type, VecVarsDerived, VecVarsMain};
use crate::codegen::header::HeaderEmitter;
use crate::color::do_color;
use crate::common::{DimSize, Dtype};
use crate::expr::{AffineExpr, Term};
use crate::highlight;
use crate::imp::blocks::Block;
use crate::imp::kernels::{Kernel, KernelType};
use crate::imp::loops::Loop;
use crate::imp::moves::TensorOrCacheView;
use crate::imp::{Impl, ImplNode};
use crate::layout::BufferExprTerm;
use crate::target::{Target, X86MemoryLevel, X86Target};
use crate::utils::indent;
use crate::views::{Param, Tensor, View};

const STACK_CUTOFF: u32 = 256;

const X86_VEC_TYPES: [VecType; 4] = [
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 8,
        name: "vui8",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
    },
    VecType {
        dtype: Dtype::Uint32,
        value_cnt: 4,
        name: "vui4",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 32,
        name: "vub32",
        native_type_name: "__m256i",
        load_fn: "_mm256_loadu_si256",
        store_fn: "_mm256_storeu_si256",
    },
    VecType {
        dtype: Dtype::Uint8,
        value_cnt: 16,
        name: "vub16",
        native_type_name: "__m128i",
        load_fn: "_mm_loadu_si128",
        store_fn: "_mm_storeu_si128",
    },
];

#[derive(Default)]
struct X86CodeGenerator<'a> {
    namer: NameGenerator,
    name_env: HashMap<Rc<Tensor<X86Target>>, CBuffer>,
    loop_iter_bindings: HashMap<BufferExprTerm, Either<String, i32>>,
    param_bindings: HashMap<Param<X86Target>, &'a dyn View<Tgt = X86Target>>,
    headers: HeaderEmitter,
}

impl<Aux: Clone + Debug> CodeGen<X86Target> for ImplNode<X86Target, Aux> {
    fn emit_kernel<W: Write>(&self, out: &mut W) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = X86CodeGenerator::default();
        generator.headers.emit_x86 = true;
        generator.emit_kernel(self, &top_arg_tensors, out)?;
        Ok(())
    }
}

impl<'a> X86CodeGenerator<'a> {
    fn emit_kernel<W: Write, Aux: Clone + Debug>(
        &mut self,
        imp: &'a ImplNode<X86Target, Aux>,
        top_arg_tensors: &'a [Rc<Tensor<X86Target>>],
        out: &mut W,
    ) -> fmt::Result {
        debug_assert_eq!(top_arg_tensors.len(), usize::from(imp.parameter_count()));

        let mut main_body_str = String::new();
        writeln!(main_body_str)?;
        writeln!(main_body_str, "__attribute__((noinline))\nvoid kernel(")?;
        for ((operand_idx, operand), tensor) in imp.parameters().enumerate().zip(top_arg_tensors) {
            let spec = tensor.spec();
            let new_c_buffer = self.make_buffer(
                spec.dim_sizes(),
                spec.vector_shape(),
                spec.dtype(),
                spec.level(),
            );
            writeln!(
                main_body_str,
                "  {} *restrict {}{}",
                c_type(operand.dtype),
                new_c_buffer.name().unwrap(),
                if operand_idx + 1 < imp.parameter_count().into() {
                    ", "
                } else {
                    "\n) {"
                }
            )?;
            self.name_env.insert(Rc::clone(tensor), new_c_buffer);
        }

        // Put the tensor->c_buffer binding into `self.name_env`. (And fill
        // tensors_as_trait_obj_ptrs.)
        let tensors_as_trait_obj_ptrs = top_arg_tensors
            .iter()
            .map(|tensor| tensor.as_ref() as &dyn View<Tgt = X86Target>)
            .collect::<Vec<_>>();

        imp.bind(&tensors_as_trait_obj_ptrs, &mut self.param_bindings);
        let depth = 1_usize;
        self.emit(&mut main_body_str, imp, depth)?;

        writeln!(main_body_str, "}}")?;

        self.headers.emit(out)?;
        if do_color() {
            highlight::c(&main_body_str);
        } else {
            out.write_str(&main_body_str)?;
        }
        Ok(())
    }

    fn make_buffer(
        &mut self,
        shape: &[DimSize],
        vector_shape: Option<&[DimSize]>,
        dtype: Dtype,
        level: X86MemoryLevel,
    ) -> CBuffer {
        debug_assert_eq!(vector_shape.is_some(), level == X86MemoryLevel::VRF);

        let name = self.namer.fresh_name();
        let size = shape.iter().product::<DimSize>();
        match level {
            X86MemoryLevel::VRF => {
                let vector_shape = vector_shape.unwrap();
                let vec_type = get_vector(dtype, vector_shape);
                self.headers.vector_type_defs.insert(vec_type);
                let main = VecVarsMain {
                    name,
                    vec_type,
                    tensor_shape: shape.into(),
                    vector_shape: vector_shape.into(),
                };
                let derived = VecVarsDerived::compute(&main, &mut self.namer);
                CBuffer::VecVars(main, derived)
            }
            X86MemoryLevel::RF => {
                if size > 1 {
                    CBuffer::StackArray { name, size, dtype }
                } else {
                    CBuffer::ValueVar { name, dtype }
                }
            }
            X86MemoryLevel::L1 | X86MemoryLevel::GL => {
                if size * u32::from(dtype.size()) < STACK_CUTOFF {
                    CBuffer::HeapArray { name, size, dtype }
                } else {
                    CBuffer::StackArray { name, size, dtype }
                }
            }
        }
    }

    fn emit<Aux: Clone + Debug, W: Write>(
        &mut self,
        w: &mut W,
        imp: &ImplNode<X86Target, Aux>,
        mut depth: usize,
    ) -> fmt::Result {
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
                            spec.dim_sizes(),
                            spec.vector_shape(),
                            spec.dtype(),
                            spec.level(),
                        );
                        dest_buffer.emit(w, false, depth)?;

                        if self
                            .name_env
                            .insert(Rc::clone(tensor), dest_buffer)
                            .is_some()
                        {
                            panic!("Duplicate name for buffer");
                        }
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
            ImplNode::ProblemApp(p) => {
                writeln!(w, "{}assert(false);  /* {p:?} */", indent(depth))
            }
            ImplNode::Kernel(Kernel {
                kernel_type,
                arguments,
                aux: _,
            }) => {
                match kernel_type {
                    KernelType::Mult => {
                        let exprs = self.param_args_to_c_indices(arguments, |_i, a, b| {
                            self.c_index(a, b, None)
                        });
                        writeln!(
                            w,
                            "{}{} += {} * {};  /* Mult */",
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
                        zero_points(&mut buffer_indexing_expr);
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
                        writeln!(w, "{} *= 0;  /* VectorZero */", exprs[0])
                    }
                    KernelType::VectorAssign => {
                        let shape = arguments[0].shape();
                        let dtype = arguments[0].spec().dtype();
                        debug_assert_eq!(shape, arguments[1].shape());

                        let exprs = self.param_args_to_c_indices(arguments, |_, a, b| {
                            self.c_index_ptr(a, b, None)
                        });
                        let vtype = get_vector(dtype, shape);
                        let itype = vtype.native_type_name;
                        if arguments.iter().all(|a| a.1.aligned()) {
                            writeln!(
                                w,
                                "*({} *)({}) = (*({} *)({}));  /* VectorAssign */",
                                itype, exprs[1], itype, exprs[0]
                            )
                        } else {
                            writeln!(
                                w,
                                "{}(({} *)({}), {}(({} *)({})));  /* VectorAssign */",
                                vtype.store_fn, itype, exprs[1], vtype.load_fn, itype, exprs[0]
                            )
                        }
                    }
                    KernelType::BroadcastVecMult => {
                        let shape = arguments[2].shape();
                        let dtype = arguments[2].spec().dtype();
                        let itype = get_vector(dtype, shape).native_type_name;
                        let exprs = self.param_args_to_c_indices(arguments, |i, a, b| match i {
                            0 => self.c_index(a, b, None),
                            1 | 2 => self.c_index_ptr(a, b, None),
                            _ => unreachable!(),
                        });
                        writeln!(
                            w,
                            "*({} *)({}) += {} * (*({} *)({})); /* BroadcastVecMult */",
                            itype, exprs[2], exprs[0], itype, exprs[1]
                        )
                    }
                    KernelType::CacheAccess => Ok(()),
                }
            }
        }
    }

    fn emit_rolled_loop<Aux: Clone + Debug, W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<X86Target, Aux>,
        mut depth: usize,
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
                let BufferExprTerm::TileIdx(dim, _) = &tt else {
                    unreachable!();
                };
                let subscript = loop_tile.subscripts[usize::from(*dim)];
                if let Some(axis_loop_iter_name) = iter_var_names.get(&subscript) {
                    if self
                        .loop_iter_bindings
                        .insert(tt.clone(), Either::Left(axis_loop_iter_name.clone()))
                        .is_some()
                    {
                        panic!("Symbol {:?} already assigned a loop iterator", tt);
                    }
                }
            }
        }

        if l.parallel {
            writeln!(
                w,
                "#pragma omp parallel for collapse({}) schedule(static)",
                axes_to_emit.len()
            )?;
        }

        for (var_name, (_, steps)) in iter_var_names.values().zip(&axes_to_emit) {
            writeln!(
                w,
                "for (int {} = 0; {} < {}; {}++) {{",
                var_name, var_name, steps, var_name
            )?;
        }

        depth += 1;
        self.emit(w, &l.body, depth)?;
        depth -= 1;

        for _ in 0..axes_to_emit.len() {
            writeln!(w, "}}")?;
        }
        Ok(())
    }

    fn emit_unrolled_loop<Aux: Clone + Debug, W: Write>(
        &mut self,
        w: &mut W,
        l: &Loop<X86Target, Aux>,
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
                    let BufferExprTerm::TileIdx(dim, _) = &tt else {
                        unreachable!();
                    };
                    let subscript = loop_tile.subscripts[usize::from(*dim)];
                    if let Some(axis_step) = axes_to_indices.get(&subscript) {
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

    fn param_args_to_c_indices<F>(&self, arguments: &[Param<X86Target>], f: F) -> Vec<String>
    where
        F: Fn(usize, &CBuffer, &AffineExpr<BufferExprTerm>) -> String,
    {
        arguments
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                let backing_tensor = arg.backing_tensor(&self.param_bindings).unwrap();
                let buffer = self.name_env.get(backing_tensor).unwrap();
                let mut buffer_indexing_expr = arg.make_buffer_indexing_expr(&self.param_bindings);
                zero_points(&mut buffer_indexing_expr);
                f(idx, buffer, &buffer_indexing_expr)
            })
            .collect()
    }

    fn sub_expr_bindings(&self, unbound_expr: AffineExpr<BufferExprTerm>) -> AffineExpr<CExprTerm> {
        let init = AffineExpr(vec![], unbound_expr.1);
        unbound_expr
            .0
            .into_iter()
            .fold(init, |base, Term(coef, sym)| {
                match self.loop_iter_bindings.get(&sym) {
                    Some(Either::Left(var_name)) => {
                        base + Term(coef, CExprTerm::CName(var_name.clone()))
                    }
                    Some(Either::Right(bound_constant)) => base + (coef * bound_constant),
                    None => base + Term(coef, CExprTerm::Buffer(sym)),
                }
            })
    }

    /// Returns a C expression referring to the value at a given expression.
    ///
    /// Additionally, `reinterpret` may be provided to introduce a type cast.
    /// This is useful for interpreting a (partial) buffer as a vector type.
    fn c_index(
        &self,
        buffer: &CBuffer,
        expr: &AffineExpr<BufferExprTerm>,
        reinterpret: Option<String>,
    ) -> String {
        match buffer {
            CBuffer::Ptr { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => format!(
                    "{}[{}]",
                    name,
                    expr_to_c(&self.sub_expr_bindings(expr.clone()))
                ),
            },
            CBuffer::UnsizedHeapArray { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => format!(
                    "{}[{}]",
                    name,
                    expr_to_c(&self.sub_expr_bindings(expr.clone()))
                ),
            },
            CBuffer::HeapArray { name, .. } => match reinterpret {
                Some(_) => unimplemented!(),
                None => format!(
                    "{}[{}]",
                    name,
                    expr_to_c(&self.sub_expr_bindings(expr.clone()))
                ), // assuming expr.c_expr() is available in scope
            },
            CBuffer::StackArray { name, .. } => match reinterpret {
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
            CBuffer::VecVars(_, _) => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index(
                    inner_vec_buffer,
                    &vec_offset.try_into().unwrap(),
                    reinterpret,
                )
            }
        }
    }

    fn c_index_vec(
        &self,
        buffer: &CBuffer,
        expr: &AffineExpr<BufferExprTerm>,
        reinterpret: Option<String>,
    ) -> String {
        // self is essentially unused, but included for consistency with c_index.
        #![allow(clippy::only_used_in_recursion)]

        match buffer {
            CBuffer::Ptr { .. }
            | CBuffer::UnsizedHeapArray { .. }
            | CBuffer::HeapArray { .. }
            | CBuffer::StackArray { .. }
            | CBuffer::ValueVar { .. } => unimplemented!(),
            CBuffer::SingleVecVar { name, .. } => {
                if expr != 0 {
                    panic!("expr must be 0, but was: {:?}", expr);
                }
                if reinterpret.is_some() {
                    unimplemented!();
                }
                name.clone()
            }
            CBuffer::VecVars(_, _) => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index_vec(
                    inner_vec_buffer,
                    &vec_offset.try_into().unwrap(),
                    reinterpret,
                )
            }
        }
    }

    fn c_index_ptr(
        &self,
        buffer: &CBuffer,
        expr: &AffineExpr<BufferExprTerm>,
        reinterpret: Option<String>,
    ) -> String {
        match buffer {
            CBuffer::Ptr { name, .. }
            | CBuffer::UnsizedHeapArray { name, .. }
            | CBuffer::HeapArray { name, .. } => match reinterpret {
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
            CBuffer::VecVars(_, _) => {
                let subbed_expr = self.sub_expr_bindings(expr.clone());
                let (inner_vec_buffer, vec_offset) = buffer.inner_vec_from_expr(&subbed_expr);
                self.c_index_ptr(
                    inner_vec_buffer,
                    &vec_offset.try_into().unwrap(),
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
            t.subscripts
                .iter()
                .enumerate()
                .filter_map(|(dim_idx, subscript)| {
                    let s = t.tile.steps_dim(dim_idx.try_into().unwrap());
                    debug_assert_ne!(s, 0);
                    if s == 1 {
                        None
                    } else {
                        Some((*subscript, s))
                    }
                })
        })
        .unique();

    // For debug builds, assert that `r` doesn't contain duplicate subscripts.
    #[cfg(debug_assertions)]
    {
        let mut seen = std::collections::HashSet::new();
        for (axis, _steps) in result.clone() {
            assert!(seen.insert(axis));
        }
    }

    result
}

fn get_vector(dtype: Dtype, vector_shape: &[DimSize]) -> &'static VecType {
    X86_VEC_TYPES
        .iter()
        .find(|vec_type| {
            vec_type.dtype == dtype
                && vec_type.value_cnt
                    == u8::try_from(vector_shape.iter().product::<DimSize>()).unwrap()
        })
        .expect("VecType to match dtype and volume of vector_shape")
}

fn expr_to_c(e: &AffineExpr<CExprTerm>) -> String {
    let mut buf =
        e.0.iter()
            .map(|Term(coef, sym)| match sym {
                CExprTerm::CName(name) => {
                    if *coef == 1 {
                        name.clone()
                    } else {
                        format!("{} * {}", coef, name)
                    }
                }
                CExprTerm::Buffer(_) => {
                    // TODO: Guarantee all terms are C names at the type level.
                    panic!("Expected all terms to be C names");
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
    buf
}

fn zero_points(expr: &mut AffineExpr<BufferExprTerm>) {
    expr.0.retain(|t| match t.1 {
        BufferExprTerm::Pt(_, _) => false,
        BufferExprTerm::TileIdx(_, _) => true,
    });
}

#[cfg(test)]
mod tests {
    use super::expr_to_c;
    use crate::codegen::c_utils::CExprTerm;
    use crate::expr::{AffineExpr, Term};

    #[test]
    fn test_expr_zero_emitted() {
        assert_eq!(expr_to_c(&AffineExpr(vec![], 0)), "0");
    }

    #[test]
    fn test_intercept_zero_not_emitted() {
        let x = CExprTerm::CName(String::from("x"));
        assert_eq!(expr_to_c(&AffineExpr(vec![Term(2, x)], 0)), "2 * x")
    }

    #[test]
    fn test_lower_to_c_expr() {
        let x = CExprTerm::CName(String::from("x"));
        let y = CExprTerm::CName(String::from("y"));
        assert_eq!(expr_to_c(&AffineExpr(vec![], 1)), "1");
        assert_eq!(expr_to_c(&AffineExpr(vec![Term(1, x)], 1)), "x + 1");
        assert_eq!(expr_to_c(&AffineExpr(vec![Term(2, y)], 3)), "2 * y + 3");
    }
}
