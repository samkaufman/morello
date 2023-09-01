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
use crate::target::{CpuMemoryLevel, Target, TargetId};
use crate::utils::ToWriteFmt;
use crate::views::Tensor;

use anyhow::{bail, Error, Result};
use log::info;
use std::cmp::max;
use std::fmt;
use std::fmt::Debug;
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::rc::Rc;
use tempfile::tempdir;

const CLI_FLAGS: [&str; 3] = ["-std=gnu99", "-O3", "-o"];

const X86_CLI_VEC_FLAGS: [&str; 2] = ["-fopenmp", "-mavx2"];
const ARM_CLI_VEC_FLAGS: [&str; 1] = ["-fopenmp"];

const MIN_SAMPLES: usize = 3;
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

    fn emit<W: fmt::Write>(&self, bench_samples: Option<usize>, out: &mut W) -> fmt::Result;

    fn build(&self, print_code: bool, bench_samples: Option<usize>) -> Result<BuiltArtifact> {
        let dirname = tempdir()?.into_path();
        let source_path = dirname.join("main.c");
        let binary_path = dirname.join("a.out");

        let source_file = std::fs::File::create(&source_path)?;
        self.emit(bench_samples, &mut ToWriteFmt(source_file))?;
        if print_code {
            self.emit(bench_samples, &mut ToWriteFmt(io::stdout()))?;
        }

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
            println!("{}", String::from_utf8_lossy(&clang_proc.stderr));
        }

        Ok(BuiltArtifact::new(
            binary_path,
            source_path,
            dirname,
            bench_samples,
        ))
    }

    /// Benchmark several times, returning the minimum of inner loop means.
    //
    //  This will first estimate a good number of inner loop iterations, then
    //  build an executable which loops that number of times, returning the mean.
    //  The final `result` computed is the minimum of the means after running
    //  that executable `repeat` times.
    fn time_impl_robustly(
        &self,
        print_code: bool,
        repeat: Option<usize>,
    ) -> Result<RobustTimingResult> {
        let repeat = repeat.unwrap_or(10); // default: 10

        // Collect a single rough sample.
        let time_check_artifact = self.build(false, Some(1))?;
        let rough_secs = time_check_artifact.measure_time()?;

        // Choose a good number of iterations for benchmarks' inner loop.
        let inner_iters = max(
            MIN_SAMPLES,
            (MIN_TRIAL_TIME_SECS / rough_secs).ceil() as usize,
        );
        info!("Goal iterations: {inner_iters}");

        // Run main benchmark loop.
        let artifact = self.build(print_code, Some(inner_iters))?;
        let mut means = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let secs = artifact.measure_time()?;
            info!("Sample runtime result {secs}s");
            means.push(secs);
        }

        Ok(RobustTimingResult {
            result: *means
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            outer_loop_samples: means,
            inner_loop_iterations: inner_iters,
            artifact,
        })
    }
}

impl<Tgt: Target<Level = CpuMemoryLevel>, Aux: Clone + Debug> CodeGen<Tgt> for ImplNode<Tgt, Aux> {
    fn emit<W: fmt::Write>(&self, bench_samples: Option<usize>, out: &mut W) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<Tgt>::new();
        generator.emit_kernel(self, &top_arg_tensors, bench_samples.is_some(), out)?;
        out.write_char('\n')?;
        generator.emit_main(&top_arg_tensors, bench_samples, out)?;
        Ok(())
    }
}

pub struct BuiltArtifact {
    binary_path: PathBuf,
    source_path: PathBuf,
    whole_dir: PathBuf,
    bench_samples: Option<usize>,
}

impl BuiltArtifact {
    pub fn new(
        binary_path: PathBuf,
        source_path: PathBuf,
        whole_dir: PathBuf,
        bench_samples: Option<usize>,
    ) -> Self {
        Self {
            binary_path,
            source_path,
            whole_dir,
            bench_samples,
        }
    }

    pub fn run(&self) -> Result<Output> {
        Command::new(self.binary_path.as_os_str())
            .output()
            .map_err(|e| e.into())
    }

    /// Executes and benchmarks an Impl on the local machine using Clang.
    //
    //  Returns the time in seconds. Measured by executing `self.bench_samples`
    //  times and returning the mean.
    pub fn measure_time(&self) -> Result<f32> {
        assert!(self.bench_samples.is_some());

        let output = self.run()?;
        if !output.status.success() {
            io::stderr().write_all(&output.stderr)?;
            bail!("Failed to run the generated code: {}", output.status);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        let first_line = lines.next().unwrap();
        Ok(parse_benchmark_output(first_line)? / self.bench_samples.unwrap() as f32)
    }
}

pub struct RobustTimingResult {
    pub result: f32,
    pub outer_loop_samples: Vec<f32>,
    pub inner_loop_iterations: usize,
    pub artifact: BuiltArtifact,
}

fn parse_benchmark_output(output: &str) -> Result<f32> {
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
    Ok(s.parse::<f32>()? + (ns.parse::<f32>()? / 1e9))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_benchmark_output_valid_input() {
        assert_eq!(
            parse_benchmark_output("cpu: 10s 500ns").unwrap(),
            10.0000005
        );
        assert_eq!(parse_benchmark_output("cpu: 5s 0ns").unwrap(), 5.0);
        assert_eq!(parse_benchmark_output("cpu: 0s 1000000000ns").unwrap(), 1.0);
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
            10.0000005
        );
    }

    #[test]
    fn test_parse_benchmark_output_negative_values() {
        assert_eq!(
            parse_benchmark_output("cpu: -10s -500ns").unwrap(),
            -10.0000005
        );
    }

    #[test]
    #[should_panic(expected = "invalid time unit")]
    fn test_parse_benchmark_output_missing_time_unit() {
        parse_benchmark_output("cpu: 10 500").unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid float literal")]
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
