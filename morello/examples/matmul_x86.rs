use morello::codegen::CodeGen;
use morello::cost::Cost;
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec;
use morello::spec::Spec;
use morello::target::CpuKernel;
use morello::target::{
    Avx2Target,
    CpuMemoryLevel::{GL, L1, RF, VRF},
};
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;

use std::io;

fn main() {
    let mut spec: Spec<Avx2Target> = spec!(MatmulAccum(
        [1, 2048, 2048, 2048],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major),
        serial
    ));
    spec.canonicalize().unwrap();

    let mat1_pack_size = nz!(16u32);
    let layout_b = layout![0, 2, 1, 2 p(mat1_pack_size)];

    let implementation = spec
        .split(128)
        .move_relayout(1, GL, layout_b.clone(), None)
        .subschedule(&[0], |pack_b| {
            // TODO: This stinks. Use vectors at least.
            pack_b
                .tile_out(&[1, 1, 1])
                .move_param(0, L1)
                .move_param(1, L1)
                .move_param(0, RF)
                .subschedule(&[0], |m0| m0.select(CpuKernel::Assign))
                .subschedule(&[1], |m0| m0.select(CpuKernel::Assign))
        })
        .tile_out(&[1, 128, 1024])
        .tile_out(&[1, 6, 16])
        .move_param(0, L1)
        .move_param(1, L1)
        .move_param(2, L1)
        .move_vrf(2, VRF, 8)
        .split(1)
        .tile_out(&[1, 1, 16])
        .move_vrf(1, VRF, 8)
        .subschedule(&[1, 0, 1, 1], |m| m.select(CpuKernel::BroadcastVecMultAdd))
        .subschedule(&[1, 1], |m| {
            m.tile_out(&[1, 1, 16])
                .split(1)
                .move_param(0, L1)
                .move_param(1, L1)
                .move_param(2, L1)
                .move_vrf(1, VRF, 8)
                .move_vrf(2, VRF, 8)
                .select(CpuKernel::BroadcastVecMultAdd)
        })
        .subschedule(&[1, 0, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 0, 2], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 0, 1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 1, 1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        })
        .subschedule(&[1, 1, 1, 2], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::Assign)
        });

    implementation
        .emit(
            true,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
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
    const ITERS: u32 = 10;
    let result = implementation
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {e}"));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&implementation).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
    println!(
        "// {:.4} gigaFLOPs/sec (Spec is {} FLOPs)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        spec.flops().unwrap(),
    );
}
