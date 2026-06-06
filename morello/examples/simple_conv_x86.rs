//! This example shows how to manually schedule a simple convolution for X86.

#![cfg_attr(feature = "drop-rf", allow(unused_imports, dead_code))]

use morello::codegen::CodeGen;
use morello::cost::Cost;
use morello::imp::ImplNode;
use morello::layout::row_major;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec;
use morello::spec::Spec;
use morello::target::{
    Avx2Target, CpuKernel,
    CpuMemory::{self, GL},
    CpuTarget,
};
use morello::utils::ToWriteFmt;

use std::io;
use std::panic;

const B: u32 = 1;
const F: u32 = 8;
const C: u32 = 8;
const OH: u32 = 1;
const OW: u32 = 1;
const FH: u32 = 3;
const FW: u32 = 3;

#[cfg(feature = "drop-rf")]
fn main() {
    eprintln!("simple_conv_x86 requires RF and is not available with drop-rf");
}

#[cfg(not(feature = "drop-rf"))]
fn main() {
    env_logger::init();

    let spec: Spec<Avx2Target> = spec!(
        Conv(
            [B, F, C, OH, OW, FH, FW],
            (u32, GL, row_major),
            (u32, GL, row_major),
            (u32, GL, row_major),
            serial
        ),
        Avx2Target::max_mem(),
    );
    println!("Logical Spec: {}", spec.0);

    let implementation = spec
        .move_param(0, CpuMemory::L1)
        .move_param(1, CpuMemory::L1)
        .move_param(2, CpuMemory::L1)
        .to_accum()
        .subschedule(&[0], zero_schedule)
        .spatial_split()
        .tile_out(&[1, 1, 1])
        .split(1)
        .select(CpuKernel::MultAdd);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap_or_else(|e| panic!("Failed to generate code: {e}"));

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        match implementation.build(false) {
            Ok(artifact) => {
                if !artifact.check_correctness(&spec) {
                    panic!("Generated code returned incorrect output");
                }
            }
            Err(e) => {
                panic!("Failed to build generated code: {e}");
            }
        }
    }

    // Benchmark.
    const ITERS: u32 = 100;
    let result = implementation
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {e}"));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&implementation).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)");
}

#[cfg(not(feature = "drop-rf"))]
fn zero_schedule(zero: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    let output = zero.0.unique_output().unwrap();
    let zero = if output.shape().iter().any(|d| d.get() != 1) {
        let scalar_tile = vec![1; output.shape().len()];
        zero.tile_out(&scalar_tile).move_param(0, CpuMemory::RF)
    } else {
        zero.move_param(0, CpuMemory::RF)
    };
    zero.subschedule(&[0], |z| z.select(CpuKernel::ValueZero))
        .subschedule(&[1], |move_back| move_back.select(CpuKernel::Assign))
}
