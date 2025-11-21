use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec;
use morello::spec::Spec;
use morello::target::{
    Avx2Target, CpuKernel,
    CpuMemoryLevel::{GL, L1, RF, VRF},
};
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;

use std::io;

fn main() {
    const M: DimSize = nz!(1u32);
    const K: DimSize = nz!(2048u32);
    const N: DimSize = nz!(16384u32);

    // Let's construct a multi-threaded matrix-matrix multiplication which takes two bf16
    // matrices and produces a f32 matrix.
    let bcm_layout = layout![0, 2, 1];
    let mut spec: Spec<Avx2Target> = spec!(Matmul(
        [nz!(1u32), M, K, N],
        (bf16, GL, row_major),
        (bf16, GL, bcm_layout),
        (f32, GL, row_major)
    ));
    spec.canonicalize().unwrap();

    // Manually schedule the matrix multiplication.
    let interleaved = layout![0, 1, 2, 2 oe(16)];

    let implementation = spec
        .cast(0, Dtype::Float32, L1, interleaved.clone(), None)
        .subschedule(&[0], |z| {
            z.tile_out(&[1, 1, 16])
                .move_param(0, L1)
                .move_relayout(0, VRF, row_major, Some(16))
                .subschedule(&[0], |z| z.select(CpuKernel::VectorAssign))
                .subschedule(&[1], |z| {
                    z.move_relayout(1, VRF, interleaved.clone(), Some(8))
                        .subschedule(&[0], |z| z.select(CpuKernel::VectorInterleaveBf16F32))
                        .subschedule(&[1], |z| z.select(CpuKernel::VectorAssign))
                })
        })
        .tile_out_parallel(&[1, 1, 128])
        .tile_out(&[1, 1, 1])
        .move_param(2, L1)
        .move_param(2, RF)
        .to_accum()
        .subschedule(&[1, 0, 0], |z| z.select(CpuKernel::MemsetZero))
        .move_param(1, L1)
        .select(CpuKernel::DotProductLoopF32InterleavedBf16F32)
        .subschedule(&[1, 1], |body| body.select(CpuKernel::ValueAssign));

    implementation
        .emit(
            true,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap_or_else(|e| panic!("Failed to generate code: {e}"));

    // Benchmark.
    let skip_var = std::env::var("SKIP_BF16_EXECUTION");
    match skip_var.as_ref().map(|s| s.as_str()) {
        Ok("0") | Err(_) => {
            const ITERS: u32 = 100;
            let result = implementation
                .bench(ITERS, None)
                .unwrap_or_else(|e| panic!("Failed to benchmark: {e}"));
            let kernel_runtime =
                (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
            let throughput = result.inner_loop_iterations as f64
                / result.best_inner_loop_runtime().as_secs_f64();
            println!("\n// cost: {}", Cost::from_impl(&implementation).main);
            println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
            println!(
                "// {:.4} gigaFLOPs/sec ({:.1}% of Zen 1 theoretical max.)",
                (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
                (100.0 * spec.flops().unwrap() as f64 * throughput) / (52.0 * 1_000_000_000.0)
            );
        }
        _ => {}
    }
}
