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
use crate::target::{Target, Targets, X86MemoryLevel};
use crate::utils::ToWriteFmt;
use crate::views::Tensor;

use anyhow::{bail, Result};
use std::fmt;
use std::fmt::Debug;
use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::rc::Rc;
use tempfile::tempdir;

const CLI_FLAGS: [&str; 3] = ["-std=gnu99", "-O3", "-o"];

const X86_CLI_VEC_FLAGS: [&str; 2] = ["-fopenmp", "-mavx2"];
const ARM_CLI_VEC_FLAGS: [&str; 1] = ["-fopenmp"];

pub trait CodeGen<Tgt: Target> {
    fn compiler_path() -> Result<String> {
        clang_path() // Clang by default
    }
    fn cli_vec_flags() -> &'static [&'static str] {
        match Tgt::by_enum() {
            Targets::X86 => &X86_CLI_VEC_FLAGS,
            Targets::Arm => &ARM_CLI_VEC_FLAGS,
        }
    }

    fn emit<W: fmt::Write>(&self, out: &mut W) -> fmt::Result;

    fn build(&self, print_code: bool) -> Result<BuildArtifact> {
        let dirname = tempdir()?.into_path();
        let source_path = dirname.join("main.c");
        let binary_path = dirname.join("a.out");

        let source_file = std::fs::File::create(&source_path)?;
        self.emit(&mut ToWriteFmt(source_file))?;
        if print_code {
            self.emit(&mut ToWriteFmt(io::stdout()))?;
            println!();
        }
        // println!("Source file: {}", source_path.to_string_lossy());

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

        Ok(BuildArtifact::new(binary_path, source_path, dirname, None))
    }
}

impl<Tgt: Target<Level = X86MemoryLevel>, Aux: Clone + Debug> CodeGen<Tgt> for ImplNode<Tgt, Aux> {
    fn emit<W: fmt::Write>(&self, out: &mut W) -> fmt::Result {
        let top_arg_tensors = self
            .parameters()
            .map(|parameter| Rc::new(Tensor::new(parameter.clone())))
            .collect::<Vec<_>>();
        let mut generator = CpuCodeGenerator::<Tgt>::new();
        generator.emit_kernel(self, &top_arg_tensors, out)?;
        out.write_str("\n")?;
        generator.emit_main(&top_arg_tensors, out)?;
        Ok(())
    }
}

pub struct BuildArtifact {
    binary_path: PathBuf,
    source_path: PathBuf,
    whole_dir: PathBuf,
    benchmark_samples: Option<i32>,
}

impl BuildArtifact {
    pub fn new(
        binary_path: PathBuf,
        source_path: PathBuf,
        whole_dir: PathBuf,
        benchmark_samples: Option<i32>,
    ) -> Self {
        Self {
            binary_path,
            source_path,
            whole_dir,
            benchmark_samples,
        }
    }

    pub fn run(&self) -> Result<Output> {
        Command::new(self.binary_path.as_os_str())
            .output()
            .map_err(|e| e.into())
    }
}
