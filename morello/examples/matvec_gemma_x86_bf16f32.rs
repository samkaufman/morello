use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::layout::{col_major, row_major, Layout, PhysDim};
use morello::lspec;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{
    CpuKernel,
    CpuMemoryLevel::{self, GL},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;

use std::io;

fn main() {
    const M: DimSize = nz!(1u32);
    const K: DimSize = nz!(2048u32);
    const N: DimSize = nz!(16384u32);

    // Let's construct a multi-threaded matrix-matrix multiplication which takes two bf16
    // matrices and produces a f32 matrix.
    let spec = Spec::<X86Target>(
        lspec!(Matmul(
            [M, K, N],
            (bf16, GL, row_major(2)),
            (bf16, GL, col_major(2)),
            (f32, GL, row_major(2))
        )),
        X86Target::max_mem(),
    );

    // Manually schedule the matrix multiplication.
    let interleaved = Layout::new(vec![
        (0, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (1, PhysDim::OddEven(nz!(16u32))),
    ]);

    let implementation = spec
        .cast(
            0,
            Dtype::Float32,
            CpuMemoryLevel::L1,
            interleaved.clone(),
            None,
        )
        .subschedule(&[0], &|z| {
            z.tile_out(&[1, 16], false)
                .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(0, CpuMemoryLevel::VRF, row_major(2), Some(nz!(16u32)))
                .subschedule(&[0], &|z| z.place(CpuKernel::VectorAssign))
                .subschedule(&[1], &|z| {
                    z.move_param(1, CpuMemoryLevel::VRF, interleaved.clone(), Some(nz!(8u32)))
                        .subschedule(&[0], &|z| z.place(CpuKernel::VectorInterleaveBf16F32))
                        .subschedule(&[1], &|z| {
                            z.tile_out(&[1, 8], false).place(CpuKernel::VectorAssign)
                        })
                })
        })
        .subschedule(&[1], &|body| {
            body.tile_out(&[1, 128], true)
                .tile_out(&[1, 1], false)
                .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                .subschedule(&[0], &|z| z.to_accum())
                .subschedule(&[0, 0], &|z| z.place(CpuKernel::MemsetZero))
                .subschedule(&[0, 1], &|body| {
                    body.move_param(1, CpuMemoryLevel::L1, col_major(2), None)
                        .place(CpuKernel::DotProductLoopF32InterleavedBf16F32)
                })
                .subschedule(&[1], &|body| {
                    body.move_param(0, CpuMemoryLevel::L1, col_major(2), None)
                        .subschedule(&[0], &|z| z.place(CpuKernel::ValueAssign))
                        .subschedule(&[1], &|z| z.place(CpuKernel::ValueAssign))
                })
        });

    implementation
        .emit(
            true,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

    // Benchmark.
    let skip_var = std::env::var("SKIP_BF16_EXECUTION");
    match skip_var.as_ref().map(|s| s.as_str()) {
        Ok("0") | Err(_) => {
            const ITERS: u32 = 100;
            let result = implementation
                .bench(ITERS, None)
                .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
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
