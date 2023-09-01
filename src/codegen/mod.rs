pub mod c_utils;
mod clang;
mod cpu;
mod header;
mod namegen;

use crate::codegen::clang::clang_path;
use crate::codegen::cpu::CpuCodeGenerator;
use crate::color::do_color;
use crate::imp::Impl;
use crate::imp::ImplNode;
use crate::pprint::ImplPrintStyle;
use crate::pprint::PrintableAux;
use crate::target::{CpuMemoryLevel, Target, TargetId};
use crate::utils::ToWriteFmt;
use crate::views::Tensor;

use anyhow::{bail, Error, Result};
use log::info;
use std::cmp::max;
use std::fmt;
use std::fmt::Debug;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use std::ops::Div;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::rc::Rc;
use std::time::Duration;
use tempfile::tempdir;

const CLI_FLAGS: [&str; 3] = ["-std=gnu99", "-O3", "-o"];

const X86_CLI_VEC_FLAGS: [&str; 2] = ["-fopenmp", "-mavx2"];
const ARM_CLI_VEC_FLAGS: [&str; 1] = ["-fopenmp"];

const MIN_SAMPLES: u32 = 3;
const MIN_TRIAL_TIME_SECS: f32 = 2.5;

pub trait CodeGen<Tgt: Target> {
    fn compiler_path() -> Result<String> {
        clang_path()
    }

    fn cli_vec_flags() -> &'static [&'static str] {
        match Tgt::target_id() {
            TargetId::X86 => &X86_CLI_VEC_FLAGS,
            TargetId::Arm => &ARM_CLI_VEC_FLAGS,
        }
    }

    fn emit<W: fmt::Write>(
        &self,
        bench_samples: Option<u32>,
        include_impl: Option<ImplPrintStyle>,
        out: &mut W,
    ) -> fmt::Result;

    fn build(&self, bench_samples: Option<u32>) -> Result<BuiltArtifact> {
        let dirname = tempdir()?.into_path();
        let source_path = dirname.join("main.c");
        let binary_path = dirname.join("a.out");

        let source_file = std::fs::File::create(&source_path)?;
        self.emit(bench_samples, None, &mut ToWriteFmt(source_file))?;

        let mut clang_cmd = Command::new(Self::compiler_path()?);
        if do_color() {
            clang_cmd.arg("-fcolor-diagnostics");
        }
        let clang_proc = clang_cmd
            .args(Self::cli_vec_flags())
            .args(CLI_FLAGS)
            .arg(binary_path.to_string_lossy().to_string())
            .arg(source_path.to_string_lossy().to_string())
            .output()?;

        if !clang_proc.status.success() {
            bail!(
                "Clang exited with {}\n{}",
                clang_proc.status,
                String::from_utf8_lossy(&clang_proc.stderr).into_owned()
            );
        } else {
            // We still want to see warnings.
            io::stderr().write_all(&clang_proc.stderr)?;
        }

        Ok(BuiltArtifact::new(
            binary_path,
            source_path,
            dirname,
            bench_samples,
        ))
    }

    /// Estimate a good number of inner loop iterations.
    fn estimate_optimal_iters(&self) -> Result<u32> {
        // Collect a single rough sample.
        let time_check_artifact = self.build(Some(1))?;
        let rough_secs = time_check_artifact.measure_time()?;

        // Choose a good number of iterations for benchmarks' inner loop.
        Ok(max(
            MIN_SAMPLES,
            (MIN_TRIAL_TIME_SECS / rough_secs.as_secs_f32()).ceil() as u32,
        ))
    }

    /// Benchmark several times, returning the minimum of inner loop means.
    ///
    /// This will first estimate a good number of inner loop iterations, then
    /// build an executable which loops that number of times, returning the mean.
    /// The final `result` computed is the minimum of the means after running
    /// that executable `repeat` times.
    fn bench(&self, bench_samples: u32, repeat: Option<usize>) -> Result<RobustTimingResult> {
        let repeat = repeat.unwrap_or(10); // default: 10

        // Run main benchmark loop.
        info!("Goal iterations: {bench_samples}");
        let artifact = self.build(Some(bench_samples))?;
        let mut means = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let time = artifact.measure_time()?;
            info!("Sample runtime result {}s", time.as_secs_f32());
            means.push(time);
        }

        Ok(RobustTimingResult {
            result: *means.iter().min_by(|a, b| a.cmp(b)).unwrap(),
            outer_loop_samples: means,
            inner_loop_iterations: bench_samples,
            artifact,
        })
    }
}

impl<Tgt, Aux> CodeGen<Tgt> for ImplNode<Tgt, Aux>
where
    Tgt: Target<Level = CpuMemoryLevel>,
    Aux: PrintableAux + Debug,
{
    fn emit<W: fmt::Write>(
        &self,
        bench_samples: Option<u32>,
        include_impl: Option<ImplPrintStyle>,
        out: &mut W,
    ) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<Tgt>::new();
        if let Some(impl_style) = include_impl {
            generator.emit_impl_comment(self, impl_style, out)?;
            writeln!(out)?;
        }
        generator.emit_kernel(self, &top_arg_tensors, bench_samples.is_some(), out)?;
        out.write_char('\n')?;
        generator.emit_load_inputs(&top_arg_tensors, out)?;
        out.write_char('\n')?;
        generator.emit_main(&top_arg_tensors, bench_samples, out)?;
        Ok(())
    }
}

pub struct BuiltArtifact {
    binary_path: PathBuf,
    source_path: PathBuf,
    whole_dir: PathBuf,
    bench_samples: Option<u32>,
}

impl BuiltArtifact {
    pub fn new(
        binary_path: PathBuf,
        source_path: PathBuf,
        whole_dir: PathBuf,
        bench_samples: Option<u32>,
    ) -> Self {
        Self {
            binary_path,
            source_path,
            whole_dir,
            bench_samples,
        }
    }

    pub fn run(&self) -> Result<Output> {
        Command::new(&self.binary_path)
            .output()
            .map_err(|e| e.into())
    }

    /// Run the binary with provided input data and return the output.
    ///
    /// Input data should be in row-major layout. Output data will also be row-major.
    #[cfg(feature = "verification")]
    pub fn run_with_input_data<A>(
        &self,
        inputs: &[ndarray::ArrayViewD<A>],
    ) -> Result<ndarray::ArrayD<A>>
    where
        A: std::str::FromStr + num_traits::ToBytes,
        A::Err: Debug,
    {
        // Write input tensor data to temporary files.
        let input_files = write_inputs_to_files(inputs)?;

        // Pass the filename as an argument.
        let mut cmd = Command::new(&self.binary_path);
        cmd.args(input_files.iter().map(|f| f.path()));
        log::debug!("Running {:?}", cmd);
        let output = cmd.output()?;

        // Read the shape from the first line. The remainder of the lines are the
        // flattened values; read until we have the reported number of values.
        let stdout = String::from_utf8_lossy(&output.stdout);
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

        let values = lines
            .flat_map(|line| line.split(' '))
            .map(|n| str::parse::<A>(n).expect("non-first lines should be data values"))
            .collect::<Vec<_>>();
        Ok(ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&shape),
            values,
        )?)
    }

    /// Executes and benchmarks an Impl on the local machine using Clang.
    ///
    /// Returns the time in seconds. Measured by executing `self.bench_samples`
    /// times and returning the mean.
    pub fn measure_time(&self) -> Result<Duration> {
        assert!(self.bench_samples.is_some());

        let output = self.run()?;
        if !output.status.success() {
            io::stderr().write_all(&output.stderr)?;
            bail!("Failed to run the generated code: {}", output.status);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        let first_line = lines.next().unwrap();
        Ok(parse_benchmark_output(first_line)?.div(self.bench_samples.unwrap()))
    }
}

pub struct RobustTimingResult {
    pub result: Duration,
    pub outer_loop_samples: Vec<Duration>,
    pub inner_loop_iterations: u32,
    pub artifact: BuiltArtifact,
}

fn parse_benchmark_output(output: &str) -> Result<Duration> {
    let mut outs = output.split_whitespace();
    if outs.next() != Some("cpu:") {
        bail!("expected \"cpu:\" prefix in benchmark output");
    }

    let s_str = outs
        .next()
        .ok_or("invalid output format")
        .map_err(Error::msg)?;
    let ns_str = outs
        .next()
        .ok_or("invalid output format")
        .map_err(Error::msg)?;
    if !s_str.ends_with('s') || !ns_str.ends_with("ns") {
        bail!("invalid time unit");
    }

    let s = s_str.trim_end_matches('s');
    let ns = ns_str.trim_end_matches("ns");
    Ok(Duration::new(s.parse::<u64>()?, ns.parse::<u32>()?))
}

/// Writes input tensors for consumption by emitted code.
///
/// Values will be stored with little endian byte ordering.
#[cfg(feature = "verification")]
fn write_inputs_to_files<A>(
    input_data: &[ndarray::ArrayViewD<A>],
) -> io::Result<Vec<tempfile::NamedTempFile>>
where
    A: num_traits::ToBytes,
{
    let mut files = Vec::with_capacity(input_data.len());
    for input in input_data {
        let mut buffered_writer = BufWriter::new(tempfile::NamedTempFile::new()?);
        for value in input.iter() {
            buffered_writer.write_all(value.to_le_bytes().as_ref())?;
        }
        files.push(buffered_writer.into_inner()?);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_benchmark_output_valid_input() {
        assert_eq!(
            parse_benchmark_output("cpu: 10s 500ns").unwrap(),
            Duration::new(10, 500)
        );
        assert_eq!(
            parse_benchmark_output("cpu: 5s 0ns").unwrap(),
            Duration::new(5, 0)
        );
        assert_eq!(
            parse_benchmark_output("cpu: 0s 1000000000ns").unwrap(),
            Duration::new(1, 0)
        );
    }

    #[test]
    #[should_panic(expected = "expected \"cpu:\" prefix in benchmark output")]
    fn test_parse_benchmark_output_missing_cpu_prefix() {
        parse_benchmark_output("10s 500ns").unwrap();
    }

    #[test]
    fn test_parse_benchmark_output_extra_whitespace() {
        assert_eq!(
            parse_benchmark_output("cpu:   10s  500ns").unwrap(),
            Duration::new(10, 500)
        );
    }

    #[test]
    #[should_panic(expected = "invalid digit found in string")]
    fn test_parse_benchmark_output_negative_values() {
        parse_benchmark_output("cpu: -10s -500ns").unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid time unit")]
    fn test_parse_benchmark_output_missing_time_unit() {
        parse_benchmark_output("cpu: 10 500").unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid digit found in string")]
    fn test_parse_benchmark_output_invalid_time_value() {
        parse_benchmark_output("cpu: ten_s 500ns").unwrap();
    }

    #[test]
    #[should_panic(expected = "expected \"cpu:\" prefix in benchmark output")]
    fn test_parse_benchmark_output_empty_string() {
        parse_benchmark_output("").unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid output format")]
    fn test_parse_benchmark_output_missing_ns_part() {
        parse_benchmark_output("cpu: 10s").unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid output format")]
    fn test_parse_benchmark_output_missing_s_part() {
        parse_benchmark_output("cpu: 500ns").unwrap();
    }

    #[test]
    #[should_panic(expected = "expected \"cpu:\" prefix in benchmark output")]
    fn test_parse_benchmark_output_whitespace_only_input() {
        parse_benchmark_output("   ").unwrap();
    }
}
