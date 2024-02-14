//! Extends [crate::codegen::BuiltArtifact] with methods to check correctness of lowered code.
#![cfg(feature = "verification")]

use crate::{
    codegen::BuiltArtifact,
    common::Dtype,
    spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec},
    target::Target,
    tensorspec::TensorSpec,
};
use ndarray::prelude::*;
use ndarray_conv::{Conv2DExt, PaddingMode, PaddingSize};
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
}

pub enum DynArrayViewMut<'a, D> {
    Uint8(ArrayViewMut<'a, u8, D>),
    Sint8(ArrayViewMut<'a, i8, D>),
    Uint16(ArrayViewMut<'a, u16, D>),
    Sint16(ArrayViewMut<'a, i16, D>),
    Uint32(ArrayViewMut<'a, u32, D>),
    Sint32(ArrayViewMut<'a, i32, D>),
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
    pub fn execute(&self, args: &mut [DynArray<IxDyn>]) {
        match self {
            LogicalSpec::Primitive(basics, _, _) => match basics.typ {
                PrimitiveSpecType::Matmul { accum } => {
                    let [lhs, rhs, out] = args else {
                        panic!("Matmul requires 3 arguments");
                    };
                    // TODO: Check shapes and dtypes are correct for this Spec.
                    if !accum {
                        out.zero();
                    }
                    // TODO: Implement views to avoid cloning below.
                    let lhs = lhs
                        .clone()
                        .into_dimensionality::<Ix2>()
                        .expect("lhs should be rank 2");
                    let rhs = rhs
                        .clone()
                        .into_dimensionality::<Ix2>()
                        .expect("rhs should be rank 2");
                    let mut out = out
                        .clone()
                        .into_dimensionality::<Ix2>()
                        .expect("out should be rank 2");
                    lhs.dot_inplace(&rhs, &mut out);
                }
                PrimitiveSpecType::Conv { accum } => {
                    use ndarray_conv::*;

                    let [lhs, rhs, out] = args else {
                        panic!("Conv requires 3 arguments");
                    };

                    // TODO: Check shapes and dtypes are correct for this Spec.
                    if !accum {
                        out.zero();
                    }
                    // TODO: Implement views to avoid cloning below.
                    let lhs = lhs
                        .clone()
                        .into_dimensionality::<Ix4>()
                        .expect("lhs should be rank 4");
                    let rhs = rhs
                        .clone()
                        .into_dimensionality::<Ix4>()
                        .expect("rhs should be rank 4");
                    for b in 0..lhs.shape()[0] {
                        for c in 0..lhs.shape()[1] {
                            for f in 0..rhs.shape()[0] {
                                let single_img_ch = lhs.slice_copy(s![b, c, .., ..]);
                                let filter_ch = rhs.slice_copy(s![f, c, .., ..]);
                                out.slice_mut(s![b, c, .., ..]).assign(
                                    &single_img_ch
                                        .conv_2d(&filter_ch, PaddingSize::Valid)
                                        .unwrap(),
                                );
                            }
                        }
                    }
                }
                PrimitiveSpecType::Move => {
                    let [inp, out] = args else {
                        panic!("Move requires 2 arguments");
                    };
                    // TODO: Check shape and dtype match.
                    out.assign(inp);
                }
                PrimitiveSpecType::Zero => {
                    // TODO: Check shape and dtype are correct for this Spec.
                    args[0].zero();
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
        }
    }

    pub fn zero(&mut self) {
        match self {
            DynArray::Uint8(a) => a.fill(0),
            DynArray::Sint8(a) => a.fill(0),
            DynArray::Uint16(a) => a.fill(0),
            DynArray::Sint16(a) => a.fill(0),
            DynArray::Uint32(a) => a.fill(0),
            DynArray::Sint32(a) => a.fill(0),
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
    {
        match self {
            DynArray::Uint8(a) => a.mapv(|x| x.as_()),
            DynArray::Sint8(a) => a.mapv(|x| x.as_()),
            DynArray::Uint16(a) => a.mapv(|x| x.as_()),
            DynArray::Sint16(a) => a.mapv(|x| x.as_()),
            DynArray::Uint32(a) => a.mapv(|x| x.as_()),
            DynArray::Sint32(a) => a.mapv(|x| x.as_()),
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
        }
    }

    pub fn conv_2d(
        &self,
        kernel: &DynArray<Ix2>,
        conv_type: PaddingSize<2>,
    ) -> Option<DynArray<Ix2>> {
        match (self, kernel) {
            (DynArray::Uint8(img), DynArray::Uint8(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Uint8),
            (DynArray::Sint8(img), DynArray::Sint8(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Sint8),
            (DynArray::Uint16(img), DynArray::Uint16(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Uint16),
            (DynArray::Sint16(img), DynArray::Sint16(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Sint16),
            (DynArray::Uint32(img), DynArray::Uint32(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Uint32),
            (DynArray::Sint32(img), DynArray::Sint32(ker)) => img
                .conv_2d(ker, conv_type, PaddingMode::Zeros)
                .map(DynArray::Sint32),
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

    // Compute expected output
    let output_idx = spec.0.output_idx();
    spec.0.execute(&mut concrete_tensors);

    // Evaluate output
    let lowered_output = built_artifact
        .run_with_input_data(&concrete_tensors)
        .unwrap();

    lowered_output == concrete_tensors[output_idx]
}

fn make_array_input_dyn<Tgt: Target>(input: &TensorSpec<Tgt>) -> DynArray<IxDyn> {
    match input.dtype() {
        Dtype::Uint8 => DynArray::Uint8(make_array_input_static::<Tgt, _>(input)),
        Dtype::Sint8 => DynArray::Sint8(make_array_input_static::<Tgt, _>(input)),
        Dtype::Uint16 => DynArray::Uint16(make_array_input_static::<Tgt, _>(input)),
        Dtype::Sint16 => DynArray::Sint16(make_array_input_static::<Tgt, _>(input)),
        Dtype::Uint32 => DynArray::Uint32(make_array_input_static::<Tgt, _>(input)),
        Dtype::Sint32 => DynArray::Sint32(make_array_input_static::<Tgt, _>(input)),
    }
}

fn make_array_input_static<Tgt, T>(input: &TensorSpec<Tgt>) -> ArrayD<T>
where
    Tgt: Target,
    T: num_traits::Bounded + num_traits::WrappingAdd + num_traits::Num + Copy,
{
    let mut value_cnt = 1;
    let mut shp_usize = Vec::with_capacity(input.shape().len());
    for v in input.shape() {
        let vc = usize::try_from(*v).unwrap();
        shp_usize.push(vc);
        value_cnt *= vc;
    }
    Array::from_iter(cycle_int_values::<T>().take(value_cnt))
        .into_shape(IxDyn(&shp_usize))
        .unwrap()
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
