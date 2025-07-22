pub(crate) mod c_utils;
mod clang;
mod cpu;
mod header;
mod namegen;

use crate::codegen::clang::clang_path;
use crate::codegen::cpu::CpuCodeGenerator;
use crate::color::do_color;
use crate::common::Dtype;
use crate::imp::{Impl, ImplNode};
use crate::pprint::ImplPrintStyle;
use crate::target::CpuTarget;
use crate::target::{Target, TargetId};
use crate::utils::ToWriteFmt;
use crate::views::Tensor;

use log::{debug, info};
use std::cmp::max;
use std::env;
use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Output};
use std::rc::Rc;
use std::time::Duration;
use tempfile::tempdir;

pub use self::cpu::CpuCodeGenThreadStyle;

const CLI_FLAGS: [&str; 4] = ["-std=gnu99", "-O3", "-rtlib=compiler-rt", "-o"];
const OPENMP_FLAG: &str = "-fopenmp";

const X86_CLI_VEC_FLAGS: [&str; 2] = ["-mavx2", "-mfma"];
const ARM_CLI_VEC_FLAGS: [&str; 0] = [];
const X86_MAC_HOST_CLI_VEC_FLAGS: [&str; 2] = ["-arch", "x86_64"];
const ARM_MAC_HOST_CLI_VEC_FLAGS: [&str; 2] = ["-arch", "arm64"];
const X86_OTHER_HOST_CLI_VEC_FLAGS: [&str; 2] = ["-march=x86-64", "-lm"];
const ARM_OTHER_HOST_CLI_VEC_FLAGS: [&str; 2] = ["-march=arm64", "-lm"];

const MIN_SAMPLES: u32 = 3;
const MIN_TRIAL_TIME_SECS: f32 = 2.5;

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Compiler could not be found")]
    MissingCompiler,
    #[error("Compiler path was not given")]
    CompilerPathNotSet,
    #[error("Compiler exited with status code: {status}")]
    CompilerFailed {
        status: process::ExitStatus,
        stderr: String,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum RunError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Generated program exited with status code: {0}")]
    BadExitStatus(process::ExitStatus),
    #[error("Couldn't interpret generated program output: {0}")]
    MalformedOutput(String),
    #[error("Couldn't build: {0}")]
    BuildError(#[from] BuildError),
}

pub trait CodeGen<Tgt: Target> {
    fn compiler_path() -> Option<String> {
        clang_path()
    }

    fn cli_vec_flags() -> Box<dyn Iterator<Item = &'static str>> {
        let (target_flags, host_flags) = match Tgt::target_id() {
            TargetId::X86 => {
                let host_flags = if env::consts::OS == "macos" {
                    &X86_MAC_HOST_CLI_VEC_FLAGS
                } else {
                    X86_OTHER_HOST_CLI_VEC_FLAGS.as_slice()
                };
                (X86_CLI_VEC_FLAGS.as_slice(), host_flags)
            }
            TargetId::Arm => {
                let host_flags = if env::consts::OS == "macos" {
                    &ARM_MAC_HOST_CLI_VEC_FLAGS
                } else {
                    ARM_OTHER_HOST_CLI_VEC_FLAGS.as_slice()
                };
                (ARM_CLI_VEC_FLAGS.as_slice(), host_flags)
            }
        };
        Box::new(target_flags.iter().chain(host_flags.iter()).cloned())
    }

    fn emit_stdout(&self) -> fmt::Result {
        self.emit(false, None, &mut ToWriteFmt(io::stdout()))
    }

    fn emit<W: fmt::Write>(
        &self,
        benchmark: bool,
        include_impl: Option<ImplPrintStyle>,
        out: &mut W,
    ) -> fmt::Result {
        self.emit_ext(benchmark, include_impl, CpuCodeGenThreadStyle::OpenMP, out)
    }

    fn emit_ext<W: fmt::Write>(
        &self,
        benchmark: bool,
        include_impl: Option<ImplPrintStyle>,
        thread_style: CpuCodeGenThreadStyle, // TODO: CPU-specific type shouldn't be here
        out: &mut W,
    ) -> fmt::Result;

    fn build(&self, benchmark: bool) -> Result<BuiltArtifact, BuildError>;

    /// Estimate a good number of inner loop iterations.
    fn estimate_optimal_iters(&self) -> Result<u32, RunError> {
        // Collect a single rough sample.
        let time_check_artifact = self.build(true)?;
        let rough_secs = time_check_artifact.measure_time(1)?;

        // Choose a good number of iterations for benchmarks' inner loop.
        Ok(max(
            MIN_SAMPLES,
            (MIN_TRIAL_TIME_SECS / rough_secs.as_secs_f32()).ceil() as u32,
        ))
    }

    /// Benchmark `repeat` times.
    fn bench(
        &self,
        inner_loop_iters: u32,
        repeat: Option<usize>,
    ) -> Result<RobustTimingResult, RunError> {
        let repeat = repeat.unwrap_or(10); // default: 10

        // Run main benchmark loop.
        info!("Goal iterations: {inner_loop_iters}");
        let artifact = self.build(true)?;
        let mut inner_loop_runtimes = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let time = artifact.measure_time(inner_loop_iters)?;
            debug!("Sample runtime result {}s", time.as_secs_f32());
            inner_loop_runtimes.push(time);
        }

        Ok(RobustTimingResult {
            inner_loop_runtimes,
            inner_loop_iterations: inner_loop_iters,
            artifact,
        })
    }
}

impl<Tgt> CodeGen<Tgt> for ImplNode<Tgt>
where
    Tgt: CpuTarget,
{
    fn emit_ext<W: fmt::Write>(
        &self,
        benchmark: bool,
        include_impl: Option<ImplPrintStyle>,
        thread_style: CpuCodeGenThreadStyle,
        out: &mut W,
    ) -> fmt::Result {
        // TODO: Avoid this clone.
        let bound = self.clone().bind(&mut |_| None); // Apply internal bindings.

        let top_arg_tensors = bound
            .collect_unbound_parameters()
            .into_iter()
            .map(|parameter| Rc::new(Tensor::new(parameter)))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<Tgt>::new();
        generator.thread_style = thread_style;
        if let Some(impl_style) = include_impl {
            generator.emit_impl_comment(&bound, impl_style, out)?;
            writeln!(out)?;
        }
        generator.emit_kernel(&bound, &top_arg_tensors, benchmark, out)?;
        out.write_char('\n')?;
        if benchmark {
            generator.emit_benchmarking_main(&top_arg_tensors, out)?;
        } else {
            generator.emit_load_inputs(&top_arg_tensors, out)?;
            out.write_char('\n')?;
            generator.emit_standard_main(&top_arg_tensors, out)?;
        }
        Ok(())
    }

    fn build(&self, benchmark: bool) -> Result<BuiltArtifact, BuildError> {
        let dirname = tempdir()?.into_path();
        let source_path = dirname.join("main.c");
        let binary_path = dirname.join("a.out");

        let bound = self.clone().bind(&mut |_| None); // Apply internal bindings.

        let source_file = std::fs::File::create(&source_path)
            .expect("should be able to create source file in just-created temp. dir.");
        // TODO: The following may not prop. IO errors hidden by ToWriteFmt.
        bound
            .emit(benchmark, None, &mut ToWriteFmt(source_file))
            .expect("codegen should not fail");

        let Some(compiler_path) = Self::compiler_path() else {
            return Err(BuildError::CompilerPathNotSet);
        };
        let mut clang_cmd = Command::new(compiler_path);
        if do_color() {
            clang_cmd.arg("-fcolor-diagnostics");
        }
        clang_cmd.args(Self::cli_vec_flags());
        if has_openmp_parallel_loops(&bound) {
            clang_cmd.arg(OPENMP_FLAG);
        }
        let clang_proc_result = clang_cmd
            .args(CLI_FLAGS)
            .arg(binary_path.to_string_lossy().as_ref())
            .arg(source_path.to_string_lossy().as_ref())
            .output();
        let clang_proc = match clang_proc_result {
            Ok(r) => r,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(BuildError::MissingCompiler);
            }
            Err(e) => return Err(e.into()),
        };

        if !clang_proc.status.success() {
            return Err(BuildError::CompilerFailed {
                status: clang_proc.status,
                stderr: String::from_utf8_lossy(&clang_proc.stderr).into(),
            });
        } else {
            // We still want to see warnings.
            // TODO: Capture this for the caller.
            io::stderr().write_all(&clang_proc.stderr).unwrap();
        }

        Ok(BuiltArtifact::new(
            binary_path,
            source_path,
            dirname,
            bound
                .collect_unbound_parameters()
                .into_iter()
                .map(|p| p.dtype())
                .collect(),
        ))
    }
}

pub struct BuiltArtifact {
    pub(crate) binary_path: PathBuf,
    parameter_dtypes: Vec<Dtype>,
}

impl BuiltArtifact {
    pub(crate) fn new(
        binary_path: PathBuf,
        _source_path: PathBuf,
        _whole_dir: PathBuf,
        parameter_dtypes: Vec<Dtype>,
    ) -> Self {
        // While we accept `source_path` and `whole_dir`, we don't do anything with them.
        Self {
            binary_path,
            parameter_dtypes,
        }
    }

    pub fn binary_path(&self) -> &Path {
        &self.binary_path
    }

    pub fn parameter_dtypes(&self) -> &[Dtype] {
        &self.parameter_dtypes
    }

    pub fn run(&self) -> Result<Output, RunError> {
        Ok(Command::new(&self.binary_path).output()?)
    }

    /// Executes and benchmarks an Impl on the local machine using Clang.
    ///
    /// Measured by executing the kernel `steps` times and returning the total
    /// runtime of the loop in seconds.
    pub fn measure_time(&self, steps: u32) -> Result<Duration, RunError> {
        let output = Command::new(&self.binary_path)
            .arg(steps.to_string())
            .output()?;
        if !output.status.success() {
            io::stderr().write_all(&output.stderr).unwrap();
            return Err(RunError::BadExitStatus(output.status));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        let first_line = lines.next().unwrap();
        parse_benchmark_output(first_line)
            .map_err(|_| RunError::MalformedOutput(String::from_utf8_lossy(&output.stderr).into()))
    }
}

pub struct RobustTimingResult {
    pub inner_loop_runtimes: Vec<Duration>,
    pub inner_loop_iterations: u32,
    pub artifact: BuiltArtifact,
}

impl RobustTimingResult {
    pub fn best_inner_loop_runtime(&self) -> Duration {
        *self
            .inner_loop_runtimes
            .iter()
            .min_by(|a, b| a.cmp(b))
            .unwrap()
    }

    pub fn best_mean(&self) -> Duration {
        self.best_inner_loop_runtime() / self.inner_loop_iterations
    }
}

/// Check if an ImplNode tree contains any parallel loops that use OpenMP
fn has_openmp_parallel_loops<Tgt: Target>(node: &ImplNode<Tgt>) -> bool {
    if let ImplNode::Loop(loop_impl) = node {
        if loop_impl.parallel {
            return true;
        }
    }
    for child in node.children() {
        if has_openmp_parallel_loops(child) {
            return true;
        }
    }
    false
}

fn parse_benchmark_output(output: &str) -> Result<Duration, ()> {
    let mut outs = output.split_whitespace();
    if outs.next() != Some("cpu:") {
        return Err(());
    }

    let Some(s_str) = outs.next() else {
        return Err(());
    };
    let Some(ns_str) = outs.next() else {
        return Err(());
    };
    if !s_str.ends_with('s') || !ns_str.ends_with("ns") {
        return Err(());
    }

    let s = s_str.trim_end_matches('s');
    let ns = ns_str.trim_end_matches("ns");
    let Ok(s_parsed) = s.parse::<u64>() else {
        return Err(());
    };
    let Ok(ns_parsed) = ns.parse::<u32>() else {
        return Err(());
    };
    Ok(Duration::new(s_parsed, ns_parsed))
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
    fn test_parse_benchmark_output_missing_cpu_prefix() {
        assert!(parse_benchmark_output("10s 500ns").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_extra_whitespace() {
        assert_eq!(
            parse_benchmark_output("cpu:   10s  500ns").unwrap(),
            Duration::new(10, 500)
        );
    }

    #[test]
    fn test_parse_benchmark_output_negative_values() {
        assert!(parse_benchmark_output("cpu: -10s -500ns").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_missing_time_unit() {
        assert!(parse_benchmark_output("cpu: 10 500").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_invalid_time_value() {
        assert!(parse_benchmark_output("cpu: ten_s 500ns").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_empty_string() {
        assert!(parse_benchmark_output("").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_missing_ns_part() {
        assert!(parse_benchmark_output("cpu: 10s").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_missing_s_part() {
        assert!(parse_benchmark_output("cpu: 500ns").is_err());
    }

    #[test]
    fn test_parse_benchmark_output_whitespace_only_input() {
        assert!(parse_benchmark_output("   ").is_err());
    }
}
