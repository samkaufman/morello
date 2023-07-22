use itertools::Itertools;
use std::{fmt, iter};

use super::namegen::NameGenerator;
use crate::{
    common::{DimSize, Dtype, Shape},
    expr::AffineExpr,
    layout::BufferExprTerm,
};

#[derive(Debug, Clone)]
pub enum CBuffer {
    Ptr {
        name: String,
        backing_buffer: Box<CBuffer>,
    },
    UnsizedHeapArray {
        // TODO: Merge UnsizedHeapArray with HeapArray.
        name: String,
        dtype: Dtype,
    },
    HeapArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    StackArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    ValueVar {
        name: String,
        dtype: Dtype,
    },
    SingleVecVar {
        name: String,
        vec_type: &'static VecType,
    },
    VecVars(VecVarsMain, VecVarsDerived),
}

#[derive(Debug, Clone)]
pub struct VecVarsMain {
    pub name: String,
    pub vec_type: &'static VecType,
    pub tensor_shape: Shape,
    pub vector_shape: Shape,
}

#[derive(Debug, Clone)]
pub struct VecVarsDerived {
    inner_vecs: Vec<CBuffer>,
    tensor_step_sizes: Shape,
    vectors_in_tensor_step_sizes: Shape,
    vector_step_sizes: Shape,
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub struct VecType {
    pub dtype: Dtype,
    pub value_cnt: u8,
    pub name: &'static str,
    pub native_type_name: &'static str,
    pub load_fn: &'static str,
    pub store_fn: &'static str,
}

impl CBuffer {
    /// Returns the C identifier name if the receiver has just one (i.e., is not distributed).
    pub fn name(&self) -> Option<&str> {
        match self {
            CBuffer::Ptr { name, .. }
            | CBuffer::UnsizedHeapArray { name, .. }
            | CBuffer::HeapArray { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::ValueVar { name, .. }
            | CBuffer::SingleVecVar { name, .. }
            | CBuffer::VecVars(VecVarsMain { name, .. }, _) => Some(name),
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            CBuffer::Ptr {
                backing_buffer: backing_tensor,
                ..
            } => backing_tensor.dtype(),
            CBuffer::UnsizedHeapArray { dtype, .. }
            | CBuffer::HeapArray { dtype, .. }
            | CBuffer::StackArray { dtype, .. }
            | CBuffer::ValueVar { dtype, .. } => *dtype,
            CBuffer::SingleVecVar { vec_type, .. }
            | CBuffer::VecVars(VecVarsMain { vec_type, .. }, _) => vec_type.dtype,
        }
    }

    pub fn emit<W: fmt::Write>(&self, w: &mut W, zero_init: bool) -> fmt::Result {
        match self {
            CBuffer::Ptr { .. } => unimplemented!(),
            CBuffer::UnsizedHeapArray { .. } => unimplemented!(),
            CBuffer::HeapArray { name, size, dtype } => {
                writeln!(w, "{} *restrict {};", c_type(*dtype), name)?;
                writeln!(
                    w,
                    "posix_memalign((void **)&{}, 128, {}*sizeof({}));",
                    name,
                    size,
                    c_type(*dtype)
                )?;

                if zero_init {
                    writeln!(
                        w,
                        "memset({}, 0, {}*sizeof({}));",
                        name,
                        size,
                        c_type(*dtype)
                    )?;
                }

                Ok(())
            }
            CBuffer::StackArray { name, size, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(
                    w,
                    "{} {}[{}] __attribute__((aligned (128))){};",
                    c_type(*dtype),
                    name,
                    size,
                    epi
                )
            }
            CBuffer::SingleVecVar { name, vec_type } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(w, "{} {}{};", vec_type.name, name, epi)
            }
            CBuffer::VecVars(_, VecVarsDerived { inner_vecs, .. }) => {
                for inner_vec in inner_vecs {
                    inner_vec.emit(w, zero_init)?;
                }
                Ok(())
            }
            CBuffer::ValueVar { name, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(w, "{} {}{};", c_type(*dtype), name, epi)
            }
        }
    }

    pub fn emit_free<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        match self {
            CBuffer::Ptr { .. } => unimplemented!(),
            CBuffer::UnsizedHeapArray { name, .. } | CBuffer::HeapArray { name, .. } => {
                writeln!(w, "free({});", name)?;
                Ok(())
            }
            CBuffer::StackArray { .. }
            | CBuffer::ValueVar { .. }
            | CBuffer::SingleVecVar { .. } => Ok(()),
            CBuffer::VecVars(_, VecVarsDerived { inner_vecs, .. }) => {
                for inner_vec in inner_vecs {
                    inner_vec.emit_free(w)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn inner_vec_from_expr(
        &self,
        expr: &AffineExpr<BufferExprTerm>,
    ) -> (&CBuffer, usize) {
        let CBuffer::VecVars(VecVarsMain { tensor_shape, vector_shape, .. }, VecVarsDerived { tensor_step_sizes, vector_step_sizes, vectors_in_tensor_step_sizes,  inner_vecs, .. }) = self else {
            unreachable!();
        };

        // CVecVars reinterprets the (linearized) offset given by `expr` as though it were an offset
        // into a row-major tensor with a simple tiling applied. We convert expr into a row-major
        // tensor coordinate, then choose the vector (tile) into which the coordinate falls assuming
        // all symbols are zero (i.e. based on the constant alone), and finally apply a row-major
        // layout to the vector "tile" to determine the individual vector offset expression.
        let AffineExpr(ref terms, expr_constant) = *expr;
        debug_assert!(
            terms.is_empty(),
            "expr should have no terms, but had: {:?}",
            expr
        );
        let (tensor_coord, vector_coord) = {
            let mut remaining_offset = u32::try_from(expr_constant).unwrap();
            let mut tensor_coord = Vec::with_capacity(tensor_step_sizes.len());
            let mut vector_coord = Vec::with_capacity(tensor_step_sizes.len());
            debug_assert_eq!(tensor_step_sizes.len(), vector_shape.len());
            for (&step_size, &v) in tensor_step_sizes.iter().zip(vector_shape) {
                let t = remaining_offset / step_size;
                tensor_coord.push(t);
                remaining_offset -= tensor_coord.last().unwrap() * step_size;
                vector_coord.push(t / v);
            }
            debug_assert_eq!(tensor_coord.len(), tensor_shape.len());
            debug_assert_eq!(remaining_offset, 0);
            (tensor_coord, vector_coord)
        };

        let idx = dot_product(&vector_coord, vectors_in_tensor_step_sizes);
        let inner_vec = &inner_vecs[usize::try_from(idx).unwrap()];

        let inside_vec_coord = tensor_coord
            .iter()
            .zip(vector_shape)
            .map(|(t, v)| t % v)
            .collect::<Shape>();
        let inside_vec_offset = dot_product(&inside_vec_coord, vector_step_sizes);

        let inside_vec_expr = inside_vec_offset.try_into().unwrap();
        (inner_vec, inside_vec_expr)
    }

    fn offset_to_rm_tensor_coordinate(&self, offset: u32) -> Vec<u32> {
        let CBuffer::VecVars(VecVarsMain { tensor_shape, .. }, VecVarsDerived { tensor_step_sizes, .. }) = self else {
            unreachable!();
        };

        let mut remaining_offset = offset;
        let mut result = vec![];
        for step_size in tensor_step_sizes {
            result.push(remaining_offset / step_size);
            remaining_offset -= result.last().unwrap() * step_size;
        }
        debug_assert_eq!(result.len(), tensor_shape.len());
        debug_assert_eq!(remaining_offset, 0);
        result
    }
}

impl VecVarsDerived {
    pub fn compute(source: &VecVarsMain, namer: &mut NameGenerator) -> Self {
        let VecVarsMain {
            vec_type,
            ref tensor_shape,
            ref vector_shape,
            ..
        } = *source;

        let tensor_step_sizes = compute_step_sizes(tensor_shape);
        let vectors_in_tensor_step_sizes = compute_step_sizes(
            &tensor_shape
                .iter()
                .zip(vector_shape.iter())
                .map(|(&t, &v)| (t + v - 1) / v)
                .collect::<Shape>(),
        );
        let vector_step_sizes = compute_step_sizes(vector_shape);

        let mut inner_vecs = Vec::new();
        for pair in (0..tensor_shape.len())
            .map(|d| Self::range_vectors_single_dim(tensor_shape, vector_shape, d))
            .multi_cartesian_product()
        {
            // Check if all second values in pair are None
            if pair.iter().all(|&(_, size)| size.is_none()) {
                inner_vecs.push(CBuffer::SingleVecVar {
                    name: namer.fresh_name(),
                    vec_type,
                });
                continue;
            }

            // TODO: Remove this possibility from range_vectors_single_dim signature.
            panic!("Reached boundary case, which we're not implementing.");
        }

        Self {
            inner_vecs,
            tensor_step_sizes,
            vectors_in_tensor_step_sizes,
            vector_step_sizes,
        }
    }

    fn range_vectors_single_dim(
        tensor_shape: &[DimSize],
        vector_shape: &[DimSize],
        dim: usize,
    ) -> impl Iterator<Item = (DimSize, Option<DimSize>)> + Clone {
        // TODO: Remove the boundary logic.
        let full_steps = tensor_shape[dim] / vector_shape[dim];
        let boundary_size = tensor_shape[dim] % vector_shape[dim];

        let main_iter = (0..full_steps).map(|step| (step, None));
        let boundary_iter =
            iter::once((full_steps, Some(boundary_size))).filter(|(_, b)| b.unwrap() > 0);
        main_iter.chain(boundary_iter)
    }
}

pub fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 => "uint8_t",
        Dtype::Uint32 => "uint32_t",
    }
}

fn dot_product<T>(lhs: &[T], rhs: &[T]) -> T
where
    T: std::iter::Sum<<T as std::ops::Mul<T>>::Output> + std::ops::Mul<T> + Copy,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

/// Compute, for each dimension, the volume of all later dimensions.
fn compute_step_sizes(shape: &[DimSize]) -> Shape {
    // TODO: Do this in one pass over shape.
    (0..shape.len())
        .map(|dim| shape[dim + 1..].iter().product())
        .collect()
}
