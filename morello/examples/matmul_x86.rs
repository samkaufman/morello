use morello::codegen::CodeGen;
use morello::cost::Cost;
use morello::layout::{row_major, Layout, PhysDim};
use morello::lspec;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::Spec;
use morello::target::CpuKernel;
use morello::target::{
    CpuMemoryLevel::{GL, L1, RF, VRF},
    Target, X86Target,
};
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;

use std::io;

fn main() {
    let mut spec = Spec::<X86Target>(
        lspec!(MatmulAccum(
            [1, 2048, 2048, 2048],
            (f32, GL, row_major),
            (f32, GL, row_major),
            (f32, GL, row_major),
            serial
        )),
        X86Target::max_mem(),
    );
    spec.canonicalize().unwrap();

    let mat1_pack_size = nz!(16u32);
    let layout_b = Layout::new(vec![
        (0, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Packed(mat1_pack_size)),
    ]);

    let implementation = spec
        .split(128)
        .move_param(1, GL, layout_b.clone(), None)
        .subschedule(&[0], |pack_b| {
            // TODO: This stinks. Use vectors at least.
            pack_b
                .tile_out(&[1, 1, 1])
                .move_param(0, L1, row_major, None)
                .move_param(1, L1, row_major, None)
                .move_param(0, RF, row_major, None)
                .subschedule(&[0], |m0| m0.select(CpuKernel::ValueAssign))
                .subschedule(&[1], |m0| m0.select(CpuKernel::ValueAssign))
        })
        .tile_out(&[1, 128, 1024])
        .tile_out(&[1, 4, 16])
        .move_param(0, L1, row_major, None)
        .move_param(1, L1, layout_b.clone(), None)
        .move_param(2, L1, row_major, None)
        .move_param(2, VRF, row_major, Some(nz!(8u32)))
        .subschedule(&[1, 0], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::VectorAssign)
        })
        .subschedule(&[1, 2], |m| {
            m.tile_out(&[1, 1, 8]).select(CpuKernel::VectorAssign)
        })
        .split(1)
        .tile_out(&[1, 1, 16])
        .move_param(1, VRF, layout_b.clone(), Some(nz!(8u32)))
        .subschedule(&[1, 1, 0], |m| m.select(CpuKernel::VectorAssign))
        .subschedule(&[1, 1, 1], |m| m.select(CpuKernel::BroadcastVecMultAdd));

    implementation
        .emit(
            true,
            Some(ImplPrintStyle::Compact),
            &mut ToWriteFmt(io::stdout()),
        )
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

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
                panic!("Failed to build generated code: {}", e);
            }
        }
    }

    // Benchmark.
    const ITERS: u32 = 10;
    let result = implementation
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
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
