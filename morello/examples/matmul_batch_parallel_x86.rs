use morello::codegen::CodeGen;
use morello::imp::ImplNode;
use morello::layout::{row_major, Layout, PhysDim};
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec;
use morello::spec::Spec;
use morello::target::{CpuKernel, MemoryLevel};
use morello::target::{
    CpuMemoryLevel::{GL, L1, RF, VRF},
    X86Target,
};
use morello::utils::ToWriteFmt;
use nonzero::nonzero as nz;
use std::io;

fn main() {
    // Compute a batch=4 matrix multiplication. (Four independent matmuls!)
    let mut spec: Spec<X86Target> = spec!(MatmulAccum(
        [4, 2048, 2048, 2048],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major)
    ));
    spec.canonicalize().unwrap();

    let layout_a = Layout::new(vec![
        (0, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Packed(nz!(4u32))),
    ]);

    let mat1_pack_size = nz!(16u32);
    let layout_b = Layout::new(vec![
        (0, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Packed(mat1_pack_size)),
    ]);

    let implementation = spec
        .tile_out_parallel(&[1, 2048, 2048])
        .split(512) // Test larger split size
        .move_relayout(1, GL, layout_b.clone(), None)
        .move_relayout(0, GL, layout_a.clone(), None)
        .tile_out(&[1, 256, 128]) // Keep memory_opt's successful tile configuration
        .tile_out(&[1, 4, 16])
        .move_param(0, L1)
        .move_param(2, L1)
        .move_vrf(2, VRF, nz!(8u32))
        .split(1)
        .tile_out(&[1, 1, 16])
        .move_param(1, L1)
        .move_vrf(1, VRF, nz!(8u32))
        .select(CpuKernel::BroadcastVecMultAdd)
        .subschedule(&[0], naive_scalar_move_impl)
        .subschedule(&[1, 0], naive_scalar_move_impl)
        .subschedule(&[1, 1, 0], naive_vector_move_impl)
        .subschedule(&[1, 1, 1, 0], naive_vector_move_impl)
        .subschedule(&[1, 1, 2, 0], naive_vector_move_impl);

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
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
    println!(
        "// {:.4} gigaFLOPs/sec (Spec is {} FLOPs)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        spec.flops().unwrap(),
    );
}

fn naive_scalar_move_impl(move_spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    move_spec
        .tile_out(&[1, 1, 1])
        .move_relayout(0, L1, row_major, None)
        .move_relayout(1, L1, row_major, None)
        .move_relayout(0, RF, row_major, None)
        .subschedule(&[0], |m0| m0.select(CpuKernel::ValueAssign))
        .subschedule(&[1], |m0| m0.select(CpuKernel::ValueAssign))
}

fn naive_vector_move_impl(move_spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    let imp = move_spec.tile_out(&[1, 1, 8]);
    if move_spec
        .0
        .parameters()
        .into_iter()
        .any(|p| p.level().vector_rf())
    {
        imp.select(CpuKernel::VectorAssign)
    } else {
        imp
    }
}
