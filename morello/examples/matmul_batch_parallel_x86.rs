use morello::codegen::CodeGen;
use morello::imp::ImplNode;
use morello::layout;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec;
use morello::spec::Spec;
use morello::target::{
    Avx2Target,
    CpuMemory::{GL, L1, RF, VRF},
};
use morello::target::{CpuKernel, Memory};
use morello::utils::ToWriteFmt;
use std::io;

const M_C: u32 = 1020;
const K_C: u32 = 1024;
const N_C: u32 = 128;
const M_R: u32 = 6;
const N_R: u32 = 16;
const MOVE_TILE_SIZE: u32 = 32;

fn main() {
    // Compute a batch=4 matrix multiplication. (Four independent matmuls!)
    let mut spec: Spec<Avx2Target> = spec!(MatmulAccum(
        [4, 2048, 2048, 2048],
        (f32, GL, row_major),
        (f32, GL, row_major),
        (f32, GL, row_major)
    ));
    spec.canonicalize().unwrap();

    let layout_a = layout![0, 1, 2, 1 p(M_R)];
    let layout_b = layout![0, 1, 2, 1 p(K_C), 2 p(N_R)];

    let implementation = spec
        .tile_out_parallel(&[1, 2048, 2048])
        .tile_out(&[1, M_R * (2048 / M_R), 2048])
        .subschedule(&[0], |main| {
            main.move_relayout(1, GL, layout_b.clone(), None)
                .tile_out(&[1, M_C, 2048])
                .subschedule(&[0], naive_scalar_move_impl)
                .subschedule(&[1, 0], |main_main| {
                    main_main
                        .move_relayout(0, GL, layout_a.clone(), None)
                        // .tile_out(&[1, m_c, 2048])
                        .split(K_C)
                        // A' in 8 MiB L3 per CCX
                        .tile_out(&[1, M_C, N_C])
                        // B' fills L2 cache here
                        // Microkernel
                        .tile_out(&[1, M_R, N_R])
                        .move_param(0, L1)
                        .move_vrf(2, VRF, 8)
                        .split(1)
                        .tile_out(&[1, 1, 16])
                        .move_param(1, L1) // moves low to "skip" modeling
                        .move_vrf(1, VRF, 8)
                        .select(CpuKernel::BroadcastVecMultAdd)
                        // Moves
                        .subschedule(&[0], naive_scalar_move_impl)
                        .subschedule(&[1, 0], naive_vector_move_impl)
                        .subschedule(&[1, 1, 0], naive_vector_move_impl)
                        .subschedule(&[1, 2], naive_vector_move_impl)
                })
                .subschedule(&[1, 1], |main_secondary| {
                    main_secondary
                        // .move_relayout(1, GL, layout_b.clone(), None)
                        .move_relayout(0, GL, layout_a.clone(), None)
                        .split(K_C)
                        .tile_out(&[1, M_R, N_C])
                        // Microkernel
                        .tile_out(&[1, M_R, N_R])
                        .move_param(0, L1)
                        .move_vrf(2, VRF, 8)
                        .split(1)
                        .tile_out(&[1, 1, 16])
                        .move_param(1, L1) // moves low to "skip" modeling
                        .move_vrf(1, VRF, 8)
                        .select(CpuKernel::BroadcastVecMultAdd)
                        // Moves
                        .subschedule(&[0], naive_scalar_move_impl)
                        .subschedule(&[1, 0], naive_vector_move_impl)
                        .subschedule(&[1, 1, 0], naive_vector_move_impl)
                        .subschedule(&[1, 2], naive_vector_move_impl)
                })
        })
        .subschedule(&[1], |secondary| {
            secondary
                .move_relayout(1, GL, layout_b.clone(), None)
                .tile_out(&[1, 1, 2048])
                .split(K_C)
                // Microkernel
                .tile_out(&[1, 1, 16])
                .move_vrf(2, VRF, 8)
                .split(1)
                .move_param(1, L1) // moves low to "skip" modeling
                .move_param(0, L1)
                .move_vrf(1, VRF, 8)
                .select(CpuKernel::BroadcastVecMultAdd)
                // Moves
                .subschedule(&[1, 0], naive_vector_move_impl)
                .subschedule(&[0, 0, 0], naive_scalar_move_impl)
                .subschedule(&[1, 1, 0], naive_vector_move_impl)
                .subschedule(&[1, 2], naive_vector_move_impl)
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
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
    println!(
        "// {:.4} gigaFLOPs/sec (Spec is {} FLOPs)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        spec.flops().unwrap(),
    );
}

fn naive_scalar_move_impl(move_spec: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    move_spec
        .tile_out(&[1, 1, 1])
        .move_relayout(0, L1, row_major, None)
        .move_relayout(1, L1, row_major, None)
        .move_relayout(0, RF, row_major, None)
        .subschedule(&[0], |m0| m0.select(CpuKernel::Assign))
        .subschedule(&[1], |m0| m0.select(CpuKernel::Assign))
}

fn naive_vector_move_impl(move_spec: &Spec<Avx2Target>) -> ImplNode<Avx2Target> {
    let ot_h = MOVE_TILE_SIZE.min(move_spec.0.parameter_shape(0)[1].get());
    let ot_w = MOVE_TILE_SIZE.min(move_spec.0.parameter_shape(0)[2].get());
    let imp = if ot_h == move_spec.0.parameter_shape(0)[1].get()
        && ot_w == move_spec.0.parameter_shape(0)[2].get()
    {
        move_spec.tile_out(&[1, 1, 8])
    } else {
        move_spec.tile_out(&[1, ot_h, ot_w]).tile_out(&[1, 1, 8])
    };
    if move_spec
        .0
        .parameters()
        .into_iter()
        .any(|p| p.memory().vector_rf())
    {
        imp.select(CpuKernel::Assign)
    } else {
        imp
    }
}
