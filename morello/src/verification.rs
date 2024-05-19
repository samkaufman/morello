//! Extends [crate::codegen::BuiltArtifact] with methods to check correctness of lowered code.
#![cfg(feature = "verification")]

use crate::{
    codegen::BuiltArtifact,
    common::{DimSize, Dtype},
    spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec},
    target::Target,
    tensorspec::TensorSpec,
};
use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use num_traits::AsPrimitive;
use std::process::Command;
use std::{
    io::{self, BufWriter, Write},
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

impl BuiltArtifact {
    /// Check whether the artifact correctly implements a [Spec].
    ///
    /// This method can be used for a little extra defense against bugs in Morello or the underlying
    /// C compiler.
    pub fn check_correctness<Tgt: Target>(&self, spec: &Spec<Tgt>) -> bool {
        let test_result = match &spec.0 {
            LogicalSpec::Primitive(PrimitiveBasics { .. }, _, _) => {
                test_artifact_correct_inner(spec, self)
            }
            LogicalSpec::Compose { .. } => todo!(),
        };
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
    ) -> anyhow::Result<DynArray<IxDyn>> {
        assert_eq!(arguments.len(), self.parameter_dtypes().len());

        // Write tensor data to temporary files. These will be loaded by the generated
        // implementation to initialize the parameters. Note this includes the output tensor.
        let mut files = Vec::with_capacity(arguments.len());
        for arg in arguments {
            let mut buffered_writer = BufWriter::new(tempfile::NamedTempFile::new()?);
            write_inputs(arg, &mut buffered_writer)?;
            files.push(buffered_writer.into_inner()?);
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
                PrimitiveSpecType::Move => {
                    let [inp, mut out] = args
                        .try_into()
                        .unwrap_or_else(|_| panic!("expected 2 args"));
                    // TODO: Check shape and dtype match.
                    out.assign(&inp);
                    vec![inp, out]
                }
                PrimitiveSpecType::Zero => {
                    assert_eq!(args.len(), 1);
                    // TODO: Check shape and dtype are correct for this Spec.
                    args[0].zero();
                    args
                }
            },
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
}

impl DynArray<Ix2> {
    pub fn dot_inplace(&self, rhs: &DynArray<Ix2>, out: &mut DynArray<Ix2>) {
        match out {
            DynArray::Uint8(o) => {
                let l = self.saturating_cast::<u8>();
                let r = rhs.saturating_cast::<u8>();
                o.assign(&l.dot(&r));
            }
            DynArray::Sint8(o) => {
                let l = self.saturating_cast::<i8>();
                let r = rhs.saturating_cast::<i8>();
                o.assign(&l.dot(&r));
            }
            DynArray::Uint16(o) => {
                let l = self.saturating_cast::<u16>();
                let r = rhs.saturating_cast::<u16>();
                o.assign(&l.dot(&r));
            }
            DynArray::Sint16(o) => {
                let l = self.saturating_cast::<i16>();
                let r = rhs.saturating_cast::<i16>();
                o.assign(&l.dot(&r));
            }
            DynArray::Uint32(o) => {
                let l = self.saturating_cast::<u32>();
                let r = rhs.saturating_cast::<u32>();
                o.assign(&l.dot(&r));
            }
            DynArray::Sint32(o) => {
                let l = self.saturating_cast::<i32>();
                let r = rhs.saturating_cast::<i32>();
                o.assign(&l.dot(&r));
            }
            DynArray::Float32(o) => {
                let l = self.saturating_cast::<f32>();
                let r = rhs.saturating_cast::<f32>();
                o.assign(&l.dot(&r));
            }
            DynArray::Bfloat16(o) => {
                let l = self.saturating_cast::<half::bf16>();
                let r = rhs.saturating_cast::<half::bf16>();
                o.assign(&l.dot(&r));
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

impl<'a, D> DynArrayViewMut<'a, D>
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

fn test_artifact_correct_inner<Tgt>(spec: &Spec<Tgt>, built_artifact: &BuiltArtifact) -> bool
where
    Tgt: Target,
{
    // Generate some test inputs (and output).
    let parameters = spec.0.parameters();
    let mut concrete_tensors = parameters
        .iter()
        .map(make_array_input_dyn::<Tgt>)
        .collect::<Vec<_>>();

    // Gather output from program. Do this before execute so that the expected output
    // isn't given to the generated program.
    let lowered_output = built_artifact
        .run_with_input_data(&concrete_tensors)
        .unwrap();

    // Compute expected output
    concrete_tensors = spec.0.execute(concrete_tensors);

    lowered_output == concrete_tensors[spec.0.output_idx()]
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
    T: num_traits::Bounded + num_traits::WrappingAdd + num_traits::Num + Copy,
{
    let mut value_cnt = 1;
    let mut shp_usize = Vec::with_capacity(shape.len());
    for v in shape {
        let vc = usize::try_from(v.get()).unwrap();
        shp_usize.push(vc);
        value_cnt *= vc;
    }
    Array::from_iter(cycle_int_values::<T>().take(value_cnt))
        .into_shape(IxDyn(&shp_usize))
        .unwrap()
}

fn make_array_input_static_f32(shape: &[DimSize]) -> ArrayD<f32> {
    make_array_input_static::<u8>(shape).map(|x| x.as_())
}

fn make_array_input_static_bf16(shape: &[DimSize]) -> ArrayD<half::bf16> {
    make_array_input_static::<u8>(shape).map(|x| x.as_())
}

/// Returns an iterator that yields infinitely all values of a numeric type in ascending order.
fn cycle_int_values<T>() -> impl Iterator<Item = T>
where
    T: num_traits::Bounded + num_traits::WrappingAdd + num_traits::Num + Copy,
{
    // The following can just be `T::min_value()..=T::max_value()` once std::iter::Step is stable.
    let mut v = T::min_value();
    std::iter::from_fn(move || {
        let r = v;
        v = v.wrapping_add(&T::one());
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
