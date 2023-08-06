mod arm;
pub mod c_utils;
mod clang;
mod cpu;
mod header;
mod namegen;
mod x86;

use crate::target::{Target, Targets};
use crate::utils::ToWriteFmt;

use anyhow::{bail, Result};
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::tempdir;

const CLI_FLAGS: [&'static str; 3] = ["-std=gnu99", "-O3", "-o"];

pub trait CodeGen<Tgt: Target> {
    fn get_compiler_path() -> Result<String> {
        clang::get_path() // Clang by default
    }
    fn get_cli_vec_flags() -> &'static [&'static str] {
        match Tgt::by_enum() {
            Targets::X86 => &x86::CLI_VEC_FLAGS,
            Targets::Arm => &arm::CLI_VEC_FLAGS,
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

        let clang_proc = Command::new(Self::get_compiler_path()?)
            .args(Self::get_cli_vec_flags())
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
        }

        Ok(BuildArtifact::new(binary_path, source_path, dirname, None))
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
