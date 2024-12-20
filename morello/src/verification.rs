//! Extends [crate::codegen::BuiltArtifact] with methods to check correctness of lowered code.

use crate::{
    codegen::BuiltArtifact,
    common::{DimSize, Dtype},
    spec::{FillValue, LogicalSpec, PrimitiveSpecType, Spec},
    target::Target,
    tensorspec::TensorSpec,
};
use ndarray::{linalg::general_mat_mul, prelude::*, RemoveAxis};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use num_traits::{real::Real, AsPrimitive};
use std::{
    fmt::{self, Debug, Formatter},
    io::{self, BufWriter, Write},
    ops::{DivAssign, Sub},
    process::Command,
    str::FromStr,
};
use tempfile::NamedTempFile;

#[derive(Clone)]
pub enum DynArray<D> {
    Uint8(Array<u8, D>),
    Sint8(Array<i8, D>),
    Uint16(Array<u16, D>),
    Sint16(Array<i16, D>),
    Uint32(Array<u32, D>),
    Sint32(Array<i32, D>),
    Float32(Array<f32, D>),
    Bfloat16(Array<half::bf16, D>),
}

pub enum DynArrayViewMut<'a, D> {
    Uint8(ArrayViewMut<'a, u8, D>),
    Sint8(ArrayViewMut<'a, i8, D>),
    Uint16(ArrayViewMut<'a, u16, D>),
    Sint16(ArrayViewMut<'a, i16, D>),
    Uint32(ArrayViewMut<'a, u32, D>),
    Sint32(ArrayViewMut<'a, i32, D>),
    Float32(ArrayViewMut<'a, f32, D>),
    Bfloat16(ArrayViewMut<'a, half::bf16, D>),
}

#[derive(thiserror::Error, Debug)]
pub enum RunError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
}

impl BuiltArtifact {
    /// Check whether the artifact correctly implements a [Spec].
    ///
    /// This method can be used for a little extra defense against bugs in Morello or the underlying
    /// C compiler.
    pub fn check_correctness<Tgt: Target>(&self, spec: &Spec<Tgt>) -> bool {
        let test_result = test_artifact_correct_inner(spec, self);
        if test_result {
            log::debug!("Artifact passed correctness check");
        } else {
            log::debug!("Artifact failed correctness check");
        }
        test_result
    }

    /// Run the binary with provided input data and return the output.
    ///
    /// Input data should be in row-major layout. Output data will also be row-major.
    pub fn run_with_input_data(
        &self,
        arguments: &[DynArray<IxDyn>],
    ) -> Result<DynArray<IxDyn>, RunError> {
        assert_eq!(arguments.len(), self.parameter_dtypes().len());

        // Write tensor data to temporary files. These will be loaded by the generated
        // implementation to initialize the parameters. Note this includes the output tensor.
        let mut files = Vec::with_capacity(arguments.len());
        for arg in arguments {
            let mut buffered_writer = BufWriter::new(tempfile::NamedTempFile::new()?);
            write_inputs(arg, &mut buffered_writer)?;
            buffered_writer.flush()?;
            files.push(
                buffered_writer
                    .into_inner()
                    .expect("already flushed, so into_inner should be safe"),
            );
        }

        // Pass the filename as an argument.
        let mut cmd = Command::new(&self.binary_path);
        cmd.args(files.iter().map(|f| f.path()));
        log::debug!("Running {:?}", cmd);
        let output = cmd.output()?;

        Ok(read_output(
            *self.parameter_dtypes().last().unwrap(),
            &output.stdout,
        )?)
    }
}

impl<Tgt: Target> LogicalSpec<Tgt> {
    #[must_use]
    pub fn execute(&self, mut args: Vec<DynArray<IxDyn>>) -> Vec<DynArray<IxDyn>> {
        match self {
            LogicalSpec::Primitive(basics, _, _) => match basics.typ {
                PrimitiveSpecType::Matmul { accum } => {
                    // TODO: Check shapes and dtypes are correct for this Spec.
                    let [lhs, rhs, out] = args
                        .try_into()
                        .unwrap_or_else(|_| panic!("expected 3 args"));
                    let lhs = lhs
                        .into_dimensionality::<Ix2>()
                        .expect("lhs should be rank 2");
                    let rhs = rhs
                        .into_dimensionality::<Ix2>()
                        .expect("rhs should be rank 2");
                    let mut out = out
                        .into_dimensionality::<Ix2>()
                        .expect("out should be rank 2");
                    if !accum {
                        out.zero();
                    }
                    lhs.dot_inplace(&rhs, &mut out);
                    vec![lhs.into_dyn(), rhs.into_dyn(), out.into_dyn()]
                }
                PrimitiveSpecType::Conv { accum } => {
                    let [lhs, rhs, out] = args
                        .try_into()
                        .unwrap_or_else(|_| panic!("expected 3 args"));

                    // TODO: Implement views to avoid cloning below.
                    let lhs = lhs
                        .into_dimensionality::<Ix4>()
                        .expect("lhs should be rank 4");
                    let rhs = rhs
                        .into_dimensionality::<Ix4>()
                        .expect("rhs should be rank 4");
                    let mut out = out
                        .into_dimensionality::<Ix4>()
                        .expect("out should be rank 4");
                    if !accum {
                        out.zero();
                    }
                    for b in 0..lhs.shape()[0] {
                        for c in 0..lhs.shape()[1] {
                            for f in 0..rhs.shape()[0] {
                                let single_img_ch = lhs.slice_copy(s![b, c, .., ..]);
                                let filter_ch = rhs.slice_copy(s![f, c, .., ..]);
                                out.slice_mut(s![b, c, .., ..])
                                    .assign(&single_img_ch.conv_2d(&filter_ch));
                            }
                        }
                    }
                    vec![lhs.into_dyn(), rhs.into_dyn(), out.into_dyn()]
                }
                PrimitiveSpecType::Softmax { scan_dim } => {
                    let [inp, mut out] = args
                        .try_into()
                        .unwrap_or_else(|_| panic!("expected 2 args"));
                    let maxes = inp.max_axis(Axis(scan_dim.into()));
                    out.assign(&(inp.clone() - maxes.broadcast(inp.shape()).unwrap()));
                    out.exp();
                    out /= &out.sum_axis(Axis(scan_dim.into()));
                    vec![inp, out]
                }
                PrimitiveSpecType::SoftmaxComplete { .. } => todo!(),
                PrimitiveSpecType::SoftmaxDenominatorAndMax { .. } => todo!(),
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaled { .. } => todo!(),
                PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax { .. } => todo!(),
                PrimitiveSpecType::SoftmaxDenominator { .. } => todo!(),
                PrimitiveSpecType::DivideVecScalar { .. } => todo!(),
                PrimitiveSpecType::Max { .. } => todo!(),
                PrimitiveSpecType::Move => {
                    let [inp, mut out] = args
                        .try_into()
                        .unwrap_or_else(|_| panic!("expected 2 args"));
                    // TODO: Check shape and dtype match.
                    out.assign(&inp);
                    vec![inp, out]
                }
                PrimitiveSpecType::Fill { value } => {
                    assert_eq!(args.len(), 1);
                    // TODO: Check shape and dtype are correct for this Spec.
                    match value {
                        FillValue::Zero => args[0].zero(),
                        FillValue::NegInf => args[0].fill_neg_inf(),
                    }
                    args
                }
            },
            LogicalSpec::Compose { components, .. }
                if components
                    .iter()
                    .all(|c| matches!(c.typ, PrimitiveSpecType::Matmul { accum: false })) =>
            {
                let mut result = vec![]; // TODO: reserve

                let mut outermost_out = args.pop().unwrap().into_dimensionality::<Ix2>().unwrap();
                outermost_out.zero();

                // Compute the innermost matmul.
                let innermost_component = components.last().unwrap();
                let innermost_output_idx = innermost_component.typ.unique_output_index().unwrap();
                let first_rhs = args.pop().unwrap().into_dimensionality::<Ix2>().unwrap();
                let first_lhs = args.pop().unwrap().into_dimensionality::<Ix2>().unwrap();
                let first_dtype = innermost_component.dtypes[innermost_output_idx];
                let mut first_out =
                    DynArray::zeros((first_lhs.shape()[0], first_rhs.shape()[1]), first_dtype);
                first_lhs.dot_inplace(&first_rhs, &mut first_out);
                result.push(first_rhs.into_dyn());
                result.push(first_lhs.into_dyn());

                // Compute the inner components
                let mut next_lhs = first_out;
                for component in components.iter().take(components.len() - 2).skip(1) {
                    let dtype = component.dtypes[component.typ.unique_output_index().unwrap()];
                    let rhs = args.pop().unwrap().into_dimensionality::<Ix2>().unwrap();
                    let mut out = DynArray::zeros((next_lhs.shape()[0], rhs.shape()[1]), dtype);
                    next_lhs.dot_inplace(&rhs, &mut out);
                    next_lhs = out;
                    result.push(rhs.into_dyn());
                }

                // Compute the final matmul.
                let rhs = args.pop().unwrap().into_dimensionality::<Ix2>().unwrap();
                next_lhs.dot_inplace(&rhs, &mut outermost_out);
                result.push(rhs.into_dyn());

                result.reverse();
                result.push(outermost_out.into_dyn());
                result
            }
            LogicalSpec::Compose { .. } => todo!(),
        }
    }
}

impl<D: ndarray::Dimension> DynArray<D> {
    pub fn shape(&self) -> &[usize] {
        match self {
            DynArray::Uint8(a) => a.shape(),
            DynArray::Sint8(a) => a.shape(),
            DynArray::Uint16(a) => a.shape(),
            DynArray::Sint16(a) => a.shape(),
            DynArray::Uint32(a) => a.shape(),
            DynArray::Sint32(a) => a.shape(),
            DynArray::Float32(a) => a.shape(),
            DynArray::Bfloat16(a) => a.shape(),
        }
    }

    pub fn slice_copy<I>(&self, info: I) -> DynArray<I::OutDim>
    where
        I: ndarray::SliceArg<D>,
    {
        match self {
            DynArray::Uint8(a) => DynArray::Uint8(a.slice(info).to_owned()),
            DynArray::Sint8(a) => DynArray::Sint8(a.slice(info).to_owned()),
            DynArray::Uint16(a) => DynArray::Uint16(a.slice(info).to_owned()),
            DynArray::Sint16(a) => DynArray::Sint16(a.slice(info).to_owned()),
            DynArray::Uint32(a) => DynArray::Uint32(a.slice(info).to_owned()),
            DynArray::Sint32(a) => DynArray::Sint32(a.slice(info).to_owned()),
            DynArray::Float32(a) => DynArray::Float32(a.slice(info).to_owned()),
            DynArray::Bfloat16(a) => DynArray::Bfloat16(a.slice(info).to_owned()),
        }
    }

    pub fn slice_mut<I>(&mut self, info: I) -> DynArrayViewMut<'_, I::OutDim>
    where
        I: ndarray::SliceArg<D>,
    {
        match self {
            DynArray::Uint8(a) => DynArrayViewMut::Uint8(a.slice_mut(info)),
            DynArray::Sint8(a) => DynArrayViewMut::Sint8(a.slice_mut(info)),
            DynArray::Uint16(a) => DynArrayViewMut::Uint16(a.slice_mut(info)),
            DynArray::Sint16(a) => DynArrayViewMut::Sint16(a.slice_mut(info)),
            DynArray::Uint32(a) => DynArrayViewMut::Uint32(a.slice_mut(info)),
            DynArray::Sint32(a) => DynArrayViewMut::Sint32(a.slice_mut(info)),
            DynArray::Float32(a) => DynArrayViewMut::Float32(a.slice_mut(info)),
            DynArray::Bfloat16(a) => DynArrayViewMut::Bfloat16(a.slice_mut(info)),
        }
    }

    pub fn into_dimensionality<A>(self) -> Result<DynArray<A>, ndarray::ShapeError>
    where
        A: ndarray::Dimension,
    {
        match self {
            DynArray::Uint8(a) => a.into_dimensionality::<A>().map(DynArray::Uint8),
            DynArray::Sint8(a) => a.into_dimensionality::<A>().map(DynArray::Sint8),
            DynArray::Uint16(a) => a.into_dimensionality::<A>().map(DynArray::Uint16),
            DynArray::Sint16(a) => a.into_dimensionality::<A>().map(DynArray::Sint16),
            DynArray::Uint32(a) => a.into_dimensionality::<A>().map(DynArray::Uint32),
            DynArray::Sint32(a) => a.into_dimensionality::<A>().map(DynArray::Sint32),
            DynArray::Float32(a) => a.into_dimensionality::<A>().map(DynArray::Float32),
            DynArray::Bfloat16(a) => a.into_dimensionality::<A>().map(DynArray::Bfloat16),
        }
    }

    pub fn into_dyn(self) -> DynArray<IxDyn> {
        self.into_dimensionality::<IxDyn>().unwrap()
    }

    pub fn zero(&mut self) {
        match self {
            DynArray::Uint8(a) => a.fill(0),
            DynArray::Sint8(a) => a.fill(0),
            DynArray::Uint16(a) => a.fill(0),
            DynArray::Sint16(a) => a.fill(0),
            DynArray::Uint32(a) => a.fill(0),
            DynArray::Sint32(a) => a.fill(0),
            DynArray::Float32(a) => a.fill(0.0),
            DynArray::Bfloat16(a) => a.fill(half::bf16::ZERO),
        }
    }

    pub fn exp(&mut self) {
        match self {
            DynArray::Uint8(_)
            | DynArray::Sint8(_)
            | DynArray::Uint16(_)
            | DynArray::Sint16(_)
            | DynArray::Uint32(_)
            | DynArray::Sint32(_) => unimplemented!("exp for integer types"),
            DynArray::Float32(a) => a.mapv_inplace(f32::exp),
            DynArray::Bfloat16(a) => a.mapv_inplace(half::bf16::exp),
        }
    }

    pub fn broadcast(&self, shape: &[usize]) -> Option<DynArray<IxDyn>> {
        match self {
            DynArray::Uint8(a) => a.broadcast(shape).map(|b| DynArray::Uint8(b.into_owned())),
            DynArray::Sint8(a) => a.broadcast(shape).map(|b| DynArray::Sint8(b.into_owned())),
            DynArray::Uint16(a) => a.broadcast(shape).map(|b| DynArray::Uint16(b.into_owned())),
            DynArray::Sint16(a) => a.broadcast(shape).map(|b| DynArray::Sint16(b.into_owned())),
            DynArray::Uint32(a) => a.broadcast(shape).map(|b| DynArray::Uint32(b.into_owned())),
            DynArray::Sint32(a) => a.broadcast(shape).map(|b| DynArray::Sint32(b.into_owned())),
            DynArray::Float32(a) => a
                .broadcast(shape)
                .map(|b| DynArray::Float32(b.into_owned())),
            DynArray::Bfloat16(a) => a
                .broadcast(shape)
                .map(|b| DynArray::Bfloat16(b.into_owned())),
        }
    }

    pub fn sum_axis(&self, axis: Axis) -> DynArray<D::Smaller>
    where
        D: RemoveAxis,
    {
        match self {
            DynArray::Uint8(a) => DynArray::Uint8(a.sum_axis(axis)),
            DynArray::Sint8(a) => DynArray::Sint8(a.sum_axis(axis)),
            DynArray::Uint16(a) => DynArray::Uint16(a.sum_axis(axis)),
            DynArray::Sint16(a) => DynArray::Sint16(a.sum_axis(axis)),
            DynArray::Uint32(a) => DynArray::Uint32(a.sum_axis(axis)),
            DynArray::Sint32(a) => DynArray::Sint32(a.sum_axis(axis)),
            DynArray::Float32(a) => DynArray::Float32(a.sum_axis(axis)),
            DynArray::Bfloat16(a) => DynArray::Bfloat16(a.sum_axis(axis)),
        }
    }

    pub fn max_axis(&self, axis: Axis) -> DynArray<D::Smaller>
    where
        D: RemoveAxis,
    {
        match self {
            DynArray::Uint8(a) => DynArray::Uint8(ndarray_max_axis(a, axis)),
            DynArray::Sint8(a) => DynArray::Sint8(ndarray_max_axis(a, axis)),
            DynArray::Uint16(a) => DynArray::Uint16(ndarray_max_axis(a, axis)),
            DynArray::Sint16(a) => DynArray::Sint16(ndarray_max_axis(a, axis)),
            DynArray::Uint32(a) => DynArray::Uint32(ndarray_max_axis(a, axis)),
            DynArray::Sint32(a) => DynArray::Sint32(ndarray_max_axis(a, axis)),
            DynArray::Float32(a) => DynArray::Float32(ndarray_f32_max_axis(a, axis)),
            DynArray::Bfloat16(_) => todo!("extend ndarray_max_axis to handle bf16"),
        }
    }

    pub fn fill_neg_inf(&mut self) {
        match self {
            DynArray::Uint8(_) => unimplemented!("fill_neg_inf for Uint8"),
            DynArray::Sint8(_) => unimplemented!("fill_neg_inf for Sint8"),
            DynArray::Uint16(_) => unimplemented!("fill_neg_inf for Uint16"),
            DynArray::Sint16(_) => unimplemented!("fill_neg_inf for Sint16"),
            DynArray::Uint32(_) => unimplemented!("fill_neg_inf for Uint32"),
            DynArray::Sint32(_) => unimplemented!("fill_neg_inf for Sint32"),
            DynArray::Float32(a) => a.fill(f32::NEG_INFINITY),
            DynArray::Bfloat16(a) => a.fill(half::bf16::NEG_INFINITY),
        }
    }

    pub fn assign(&mut self, rhs: &Self) {
        match (self, rhs) {
            (DynArray::Uint8(a), DynArray::Uint8(b)) => a.assign(b),
            (DynArray::Sint8(a), DynArray::Sint8(b)) => a.assign(b),
            (DynArray::Uint16(a), DynArray::Uint16(b)) => a.assign(b),
            (DynArray::Sint16(a), DynArray::Sint16(b)) => a.assign(b),
            (DynArray::Uint32(a), DynArray::Uint32(b)) => a.assign(b),
            (DynArray::Sint32(a), DynArray::Sint32(b)) => a.assign(b),
            (DynArray::Float32(a), DynArray::Float32(b)) => a.assign(b),
            (DynArray::Bfloat16(a), DynArray::Bfloat16(b)) => a.assign(b),
            _ => panic!("Mismatched types"),
        }
    }

    pub fn saturating_cast<T>(&self) -> Array<T, D>
    where
        T: Copy + 'static,
        u8: num_traits::AsPrimitive<T>,
        i8: num_traits::AsPrimitive<T>,
        u16: num_traits::AsPrimitive<T>,
        i16: num_traits::AsPrimitive<T>,
        u32: num_traits::AsPrimitive<T>,
        i32: num_traits::AsPrimitive<T>,
        f32: num_traits::AsPrimitive<T>,
        half::bf16: num_traits::AsPrimitive<T>,
    {
        match self {
            DynArray::Uint8(a) => a.mapv(|x| x.as_()),
            DynArray::Sint8(a) => a.mapv(|x| x.as_()),
            DynArray::Uint16(a) => a.mapv(|x| x.as_()),
            DynArray::Sint16(a) => a.mapv(|x| x.as_()),
            DynArray::Uint32(a) => a.mapv(|x| x.as_()),
            DynArray::Sint32(a) => a.mapv(|x| x.as_()),
            DynArray::Float32(a) => a.mapv(|x| x.as_()),
            DynArray::Bfloat16(a) => a.mapv(|x| x.as_()),
        }
    }

    // TODO: Generalize to D other than IxDyn.
    pub fn zeros<Sh>(shape: Sh, dtype: Dtype) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        match dtype {
            Dtype::Uint8 => DynArray::Uint8(Array::zeros(shape)),
            Dtype::Sint8 => DynArray::Sint8(Array::zeros(shape)),
            Dtype::Uint16 => DynArray::Uint16(Array::zeros(shape)),
            Dtype::Sint16 => DynArray::Sint16(Array::zeros(shape)),
            Dtype::Uint32 => DynArray::Uint32(Array::zeros(shape)),
            Dtype::Sint32 => DynArray::Sint32(Array::zeros(shape)),
            Dtype::Float32 => DynArray::Float32(Array::zeros(shape)),
            Dtype::Bfloat16 => DynArray::Bfloat16(Array::zeros(shape)),
        }
    }

    pub fn approx_eq(&self, other: &Self, tol: f32) -> bool {
        match (self, other) {
            (DynArray::Uint8(a), DynArray::Uint8(b)) => a == b,
            (DynArray::Sint8(a), DynArray::Sint8(b)) => a == b,
            (DynArray::Uint16(a), DynArray::Uint16(b)) => a == b,
            (DynArray::Sint16(a), DynArray::Sint16(b)) => a == b,
            (DynArray::Uint32(a), DynArray::Uint32(b)) => a == b,
            (DynArray::Sint32(a), DynArray::Sint32(b)) => a == b,
            (DynArray::Float32(a), DynArray::Float32(b)) => a.abs_diff_eq(b, tol),
            (DynArray::Bfloat16(_), DynArray::Bfloat16(_)) => todo!("approx_eq for bf16"),
            _ => false,
        }
    }
}

impl DynArray<Ix2> {
    pub fn dot_inplace(&self, rhs: &DynArray<Ix2>, out: &mut DynArray<Ix2>) {
        match out {
            DynArray::Uint8(o) => {
                let l = self.saturating_cast::<u8>();
                let r = rhs.saturating_cast::<u8>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Sint8(o) => {
                let l = self.saturating_cast::<i8>();
                let r = rhs.saturating_cast::<i8>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Uint16(o) => {
                let l = self.saturating_cast::<u16>();
                let r = rhs.saturating_cast::<u16>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Sint16(o) => {
                let l = self.saturating_cast::<i16>();
                let r = rhs.saturating_cast::<i16>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Uint32(o) => {
                let l = self.saturating_cast::<u32>();
                let r = rhs.saturating_cast::<u32>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Sint32(o) => {
                let l = self.saturating_cast::<i32>();
                let r = rhs.saturating_cast::<i32>();
                general_mat_mul(1, &l, &r, 1, o);
            }
            DynArray::Float32(o) => {
                let l = self.saturating_cast::<f32>();
                let r = rhs.saturating_cast::<f32>();
                general_mat_mul(1.0, &l, &r, 1.0, o);
            }
            DynArray::Bfloat16(o) => {
                let l = self.saturating_cast::<half::bf16>();
                let r = rhs.saturating_cast::<half::bf16>();
                let intermed = l.dot(&r);
                *o += &intermed;
            }
        }
    }

    pub fn conv_2d(&self, kernel: &DynArray<Ix2>) -> DynArray<Ix2> {
        match (self, kernel) {
            (DynArray::Uint8(img), DynArray::Uint8(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Uint8)
                .unwrap(),
            (DynArray::Sint8(img), DynArray::Sint8(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Sint8)
                .unwrap(),
            (DynArray::Uint16(img), DynArray::Uint16(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Uint16)
                .unwrap(),
            (DynArray::Sint16(img), DynArray::Sint16(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Sint16)
                .unwrap(),
            (DynArray::Uint32(img), DynArray::Uint32(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Uint32)
                .unwrap(),
            (DynArray::Sint32(img), DynArray::Sint32(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Sint32)
                .unwrap(),
            (DynArray::Float32(img), DynArray::Float32(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Float32)
                .unwrap(),
            (DynArray::Bfloat16(img), DynArray::Bfloat16(ker)) => img
                .conv(ker, ConvMode::Same, PaddingMode::Zeros)
                .map(DynArray::Bfloat16)
                .unwrap(),
            _ => panic!("Mismatched types"),
        }
    }
}

impl<D> Sub for DynArray<D>
where
    D: ndarray::Dimension,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DynArray::Uint8(a), DynArray::Uint8(b)) => DynArray::Uint8(a - b),
            (DynArray::Sint8(a), DynArray::Sint8(b)) => DynArray::Sint8(a - b),
            (DynArray::Uint16(a), DynArray::Uint16(b)) => DynArray::Uint16(a - b),
            (DynArray::Sint16(a), DynArray::Sint16(b)) => DynArray::Sint16(a - b),
            (DynArray::Uint32(a), DynArray::Uint32(b)) => DynArray::Uint32(a - b),
            (DynArray::Sint32(a), DynArray::Sint32(b)) => DynArray::Sint32(a - b),
            (DynArray::Float32(a), DynArray::Float32(b)) => DynArray::Float32(a - b),
            (DynArray::Bfloat16(a), DynArray::Bfloat16(b)) => DynArray::Bfloat16(a - b),
            _ => panic!("Mismatched types"),
        }
    }
}

impl<D> DivAssign<&DynArray<D>> for DynArray<D>
where
    D: ndarray::Dimension,
{
    fn div_assign(&mut self, rhs: &Self) {
        match (self, rhs) {
            (DynArray::Uint8(a), DynArray::Uint8(b)) => a.div_assign(b),
            (DynArray::Sint8(a), DynArray::Sint8(b)) => a.div_assign(b),
            (DynArray::Uint16(a), DynArray::Uint16(b)) => a.div_assign(b),
            (DynArray::Sint16(a), DynArray::Sint16(b)) => a.div_assign(b),
            (DynArray::Uint32(a), DynArray::Uint32(b)) => a.div_assign(b),
            (DynArray::Sint32(a), DynArray::Sint32(b)) => a.div_assign(b),
            (DynArray::Float32(a), DynArray::Float32(b)) => a.div_assign(b),
            (DynArray::Bfloat16(a), DynArray::Bfloat16(b)) => a.div_assign(b),
            _ => panic!("Mismatched types"),
        }
    }
}

impl<D: ndarray::Dimension> PartialEq for DynArray<D> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Uint8(l0), Self::Uint8(r0)) => l0 == r0,
            (Self::Sint8(l0), Self::Sint8(r0)) => l0 == r0,
            (Self::Uint16(l0), Self::Uint16(r0)) => l0 == r0,
            (Self::Sint16(l0), Self::Sint16(r0)) => l0 == r0,
            (Self::Uint32(l0), Self::Uint32(r0)) => l0 == r0,
            (Self::Sint32(l0), Self::Sint32(r0)) => l0 == r0,
            (Self::Float32(l0), Self::Float32(r0)) => l0 == r0,
            (Self::Bfloat16(l0), Self::Bfloat16(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<D> From<Array<u8, D>> for DynArray<D> {
    fn from(value: Array<u8, D>) -> Self {
        DynArray::Uint8(value)
    }
}

impl<D> From<Array<i8, D>> for DynArray<D> {
    fn from(value: Array<i8, D>) -> Self {
        DynArray::Sint8(value)
    }
}

impl<D> From<Array<u16, D>> for DynArray<D> {
    fn from(value: Array<u16, D>) -> Self {
        DynArray::Uint16(value)
    }
}

impl<D> From<Array<i16, D>> for DynArray<D> {
    fn from(value: Array<i16, D>) -> Self {
        DynArray::Sint16(value)
    }
}

impl<D> From<Array<u32, D>> for DynArray<D> {
    fn from(value: Array<u32, D>) -> Self {
        DynArray::Uint32(value)
    }
}

impl<D> From<Array<i32, D>> for DynArray<D> {
    fn from(value: Array<i32, D>) -> Self {
        DynArray::Sint32(value)
    }
}

impl<D> From<Array<f32, D>> for DynArray<D> {
    fn from(value: Array<f32, D>) -> Self {
        DynArray::Float32(value)
    }
}

impl<D> From<Array<half::bf16, D>> for DynArray<D> {
    fn from(value: Array<half::bf16, D>) -> Self {
        DynArray::Bfloat16(value)
    }
}

impl<D> Debug for DynArray<D>
where
    D: ndarray::Dimension,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DynArray::Uint8(a) => write!(f, "DynArray::Uint8({:?})", a),
            DynArray::Sint8(a) => write!(f, "DynArray::Sint8({:?})", a),
            DynArray::Uint16(a) => write!(f, "DynArray::Uint16({:?})", a),
            DynArray::Sint16(a) => write!(f, "DynArray::Sint16({:?})", a),
            DynArray::Uint32(a) => write!(f, "DynArray::Uint32({:?})", a),
            DynArray::Sint32(a) => write!(f, "DynArray::Sint32({:?})", a),
            DynArray::Float32(a) => write!(f, "DynArray::Float32({:?})", a),
            DynArray::Bfloat16(a) => write!(f, "DynArray::Bfloat16({:?})", a),
        }
    }
}

impl<D> DynArrayViewMut<'_, D>
where
    D: ndarray::Dimension,
{
    pub fn assign(&mut self, rhs: &DynArray<D>) {
        match (self, rhs) {
            (DynArrayViewMut::Uint8(l), DynArray::Uint8(r)) => l.assign(r),
            (DynArrayViewMut::Sint8(l), DynArray::Sint8(r)) => l.assign(r),
            (DynArrayViewMut::Uint16(l), DynArray::Uint16(r)) => l.assign(r),
            (DynArrayViewMut::Sint16(l), DynArray::Sint16(r)) => l.assign(r),
            (DynArrayViewMut::Uint32(l), DynArray::Uint32(r)) => l.assign(r),
            (DynArrayViewMut::Sint32(l), DynArray::Sint32(r)) => l.assign(r),
            (DynArrayViewMut::Float32(l), DynArray::Float32(r)) => l.assign(r),
            (DynArrayViewMut::Bfloat16(l), DynArray::Bfloat16(r)) => l.assign(r),
            _ => panic!("Mismatched types"),
        }
    }
}

fn ndarray_max_axis<A, S, D>(array: &ndarray::ArrayBase<S, D>, axis: Axis) -> Array<A, D::Smaller>
where
    A: Copy + Clone + Ord + num_traits::bounds::LowerBounded,
    S: ndarray::Data<Elem = A>,
    D: Dimension + ndarray::RemoveAxis,
{
    let mut res = Array::from_elem(array.raw_dim().remove_axis(axis), A::min_value());
    for subview in array.axis_iter(axis) {
        res.zip_mut_with(&subview, |a, &b| {
            *a = (*a).max(b);
        });
    }
    res
}

fn ndarray_f32_max_axis<S, D>(
    array: &ndarray::ArrayBase<S, D>,
    axis: Axis,
) -> Array<f32, D::Smaller>
where
    S: ndarray::Data<Elem = f32>,
    D: Dimension + ndarray::RemoveAxis,
{
    let mut res = Array::from_elem(array.raw_dim().remove_axis(axis), f32::MIN);
    for subview in array.axis_iter(axis) {
        res.zip_mut_with(&subview, |a, &b| {
            *a = if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                (*a).max(b)
            };
        });
    }
    res
}

fn test_artifact_correct_inner<Tgt>(spec: &Spec<Tgt>, built_artifact: &BuiltArtifact) -> bool
where
    Tgt: Target,
{
    let Some(output_idx) = spec.0.unique_output_index() else {
        todo!("Support Specs with multiple outputs");
    };

    // Generate some test inputs (and output).
    let parameters = spec.0.parameters();
    let mut concrete_tensors = parameters
        .iter()
        .map(make_array_input_dyn::<Tgt>)
        .collect::<Vec<_>>();

    // Gather output from program. Do this before execution so that the expected output
    // isn't given to the generated program.
    let lowered_output = built_artifact
        .run_with_input_data(&concrete_tensors)
        .unwrap();

    // Compute expected output
    concrete_tensors = spec.0.execute(concrete_tensors);

    lowered_output.approx_eq(&concrete_tensors[output_idx], 1e-7)
}

fn make_array_input_dyn<Tgt: Target>(input: &TensorSpec<Tgt>) -> DynArray<IxDyn> {
    match input.dtype() {
        Dtype::Uint8 => DynArray::Uint8(make_array_input_static(input.shape())),
        Dtype::Sint8 => DynArray::Sint8(make_array_input_static(input.shape())),
        Dtype::Uint16 => DynArray::Uint16(make_array_input_static(input.shape())),
        Dtype::Sint16 => DynArray::Sint16(make_array_input_static(input.shape())),
        Dtype::Uint32 => DynArray::Uint32(make_array_input_static(input.shape())),
        Dtype::Sint32 => DynArray::Sint32(make_array_input_static(input.shape())),
        Dtype::Float32 => DynArray::Float32(make_array_input_static_f32(input.shape())),
        Dtype::Bfloat16 => DynArray::Bfloat16(make_array_input_static_bf16(input.shape())),
    }
}

fn make_array_input_static<T>(shape: &[DimSize]) -> ArrayD<T>
where
    T: num_traits::Bounded + num_traits::WrappingAdd + num_traits::Num + Copy + PartialOrd,
{
    make_array_input_static_bounds(shape, T::min_value(), T::max_value())
}

fn make_array_input_static_bounds<T>(shape: &[DimSize], min: T, max: T) -> ArrayD<T>
where
    T: num_traits::WrappingAdd + num_traits::Num + Copy + PartialOrd,
{
    let mut value_cnt = 1;
    let mut shp_usize = Vec::with_capacity(shape.len());
    for v in shape {
        let vc = usize::try_from(v.get()).unwrap();
        shp_usize.push(vc);
        value_cnt *= vc;
    }
    Array::from_iter(cycle_int_values::<T>(min, max).take(value_cnt))
        .into_shape(IxDyn(&shp_usize))
        .unwrap()
}

fn make_array_input_static_f32(shape: &[DimSize]) -> ArrayD<f32> {
    make_array_input_static_bounds::<u8>(shape, 0, 3).map(|x| x.as_())
}

fn make_array_input_static_bf16(shape: &[DimSize]) -> ArrayD<half::bf16> {
    make_array_input_static_bounds::<u8>(shape, 0, 3).map(|x| x.as_())
}

/// Returns an iterator that yields infinitely all values of a numeric type in ascending order.
fn cycle_int_values<T>(min: T, max: T) -> impl Iterator<Item = T>
where
    T: num_traits::WrappingAdd + num_traits::Num + Copy + PartialOrd,
{
    // The following can just be `T::min_value()..=T::max_value()` once std::iter::Step is stable.
    let mut v = min;
    std::iter::from_fn(move || {
        let r = v;
        v = v.wrapping_add(&T::one());
        if v > max {
            v = min;
        }
        Some(r)
    })
}

/// Writes an input tensor for consumption by emitted code.
///
/// Written values have little endian byte ordering.
fn write_inputs(input: &DynArray<IxDyn>, writer: &mut BufWriter<NamedTempFile>) -> io::Result<()> {
    match input {
        DynArray::Uint8(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Sint8(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Uint16(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Sint16(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Uint32(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Sint32(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Float32(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
        DynArray::Bfloat16(a) => {
            for value in a.iter() {
                writer.write_all(value.to_le_bytes().as_ref())?;
            }
        }
    };
    Ok(())
}

fn read_output(dtype: Dtype, source: &[u8]) -> io::Result<DynArray<IxDyn>> {
    // Read the shape from the first line. The remainder of the lines are the
    // flattened values; read until we have the reported number of values.
    let stdout = String::from_utf8_lossy(source);
    let mut lines = stdout.lines().filter_map(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            None
        } else {
            Some(trimmed)
        }
    });

    let first_line = lines.next().expect("stdout should not be empty");
    let shape = first_line
        .split('x')
        .map(|s| str::parse::<usize>(s).expect("first line should be shape"))
        .collect::<Vec<_>>();

    Ok(match dtype {
        Dtype::Uint8 => ndarray_from_lines::<u8, _, _>(&shape, &mut lines),
        Dtype::Sint8 => ndarray_from_lines::<i8, _, _>(&shape, &mut lines),
        Dtype::Uint16 => ndarray_from_lines::<u16, _, _>(&shape, &mut lines),
        Dtype::Sint16 => ndarray_from_lines::<i16, _, _>(&shape, &mut lines),
        Dtype::Uint32 => ndarray_from_lines::<u32, _, _>(&shape, &mut lines),
        Dtype::Sint32 => ndarray_from_lines::<i32, _, _>(&shape, &mut lines),
        Dtype::Float32 => ndarray_from_lines::<f32, _, _>(&shape, &mut lines),
        Dtype::Bfloat16 => ndarray_from_lines::<half::bf16, _, _>(&shape, &mut lines),
    })
}

fn ndarray_from_lines<T, I, V>(shape: &[usize], lines: &mut I) -> DynArray<IxDyn>
where
    T: FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    I: Iterator<Item = V>,
    V: AsRef<str>,
    Array<T, IxDyn>: Into<DynArray<IxDyn>>,
{
    let mut data = vec![];
    for line in lines {
        for s in line.as_ref().split(' ') {
            data.push(str::parse::<T>(s).expect("non-first lines should be data values"));
        }
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(shape), data)
        .unwrap()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_max_axis_single_element() {
        let array = array![1];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, arr0(1));
    }

    #[test]
    fn test_ndarray_max_axis_row_vector_axis_0() {
        let array = array![[1, 2, 3, 4]];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, array![1, 2, 3, 4]);
    }

    #[test]
    fn test_ndarray_max_axis_row_vector_axis_1() {
        let array = array![[1, 2, 3, 4]];
        let max_axis_1 = ndarray_max_axis(&array, Axis(1));
        assert_eq!(max_axis_1, array![4]);
    }

    #[test]
    fn test_ndarray_max_axis_column_vector_axis_0() {
        let array = array![[1], [2], [3], [4]];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, array![4]);
    }

    #[test]
    fn test_ndarray_max_axis_column_vector_axis_1() {
        let array = array![[1], [2], [3], [4]];
        let max_axis_1 = ndarray_max_axis(&array, Axis(1));
        assert_eq!(max_axis_1, array![1, 2, 3, 4]);
    }

    #[test]
    fn test_ndarray_max_axis_square_matrix_axis_0() {
        let array = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, array![7, 8, 9]);
    }

    #[test]
    fn test_ndarray_max_axis_square_matrix_axis_1() {
        let array = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let max_axis_1 = ndarray_max_axis(&array, Axis(1));
        assert_eq!(max_axis_1, array![3, 6, 9]);
    }

    #[test]
    fn test_ndarray_max_axis_rectangular_matrix_axis_0() {
        let array = array![[1, 2], [3, 4], [5, 6]];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, array![5, 6]);
    }

    #[test]
    fn test_ndarray_max_axis_rectangular_matrix_axis_1() {
        let array = array![[1, 2], [3, 4], [5, 6]];
        let max_axis_1 = ndarray_max_axis(&array, Axis(1));
        assert_eq!(max_axis_1, array![2, 4, 6]);
    }

    #[test]
    fn test_ndarray_max_axis_large_matrix_axis_0() {
        let array = array![[1, 2, 3, 4], [5, 6, 7, 8]];
        let max_axis_0 = ndarray_max_axis(&array, Axis(0));
        assert_eq!(max_axis_0, array![5, 6, 7, 8]);
    }

    #[test]
    fn test_ndarray_max_axis_large_matrix_axis_1() {
        let array = array![[1, 2, 3, 4], [5, 6, 7, 8]];
        let max_axis_1 = ndarray_max_axis(&array, Axis(1));
        assert_eq!(max_axis_1, array![4, 8]);
    }
}
