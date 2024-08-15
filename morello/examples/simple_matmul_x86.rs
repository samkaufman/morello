//! This example shows how to manually schedule a simple matrix multiplication for X86.

use morello::codegen::CodeGen;
use morello::cost::Cost;
use morello::layout::row_major;
use morello::lspec;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    CpuMemoryLevel::{self, GL},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;
use std::panic;

fn main() {
    // First, we'll define the Spec for the program we will implement: a 64x64x64 matrix
    // multiplication with unsigned, 32-bit integer inputs and output.
    //
    // This is a non-accumulating Spec (`Matmul` rather than `MatmulAccum`), which means that the
    // implementation will set rather then add values to the output tensor.
    let layout = row_major(2);

    let spec = Spec::<X86Target>(
        lspec!(Matmul(
            [64, 64, 64],
            (u32, GL, layout.clone()),
            (u32, GL, layout.clone()),
            (u32, GL, layout),
            serial
        )),
        X86Target::max_mem(),
    );
    println!("Logical Spec: {}", spec.0);

    // Manually schedule the matrix multiplication.
    let implementation = spec
        .tile_out(&[16, 16], false)
        .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
        .tile_out(&[1, 1], false)
        .to_accum()
        .subschedule(&[0], &|z| {
            z.move_param(0, CpuMemoryLevel::RF, row_major(2), None)
        })
        .subschedule(&[0, 0], &|z| z.place(CpuKernel::MemsetZero))
        .subschedule(&[0, 1], &|move_back| {
            move_back.place(CpuKernel::ValueAssign)
        })
        .subschedule(&[1], &|bat| {
            bat.split(4)
                .move_param(0, CpuMemoryLevel::RF, row_major(2), None)
                .subschedule(&[0], &|move_a| {
                    move_a
                        .tile_out(&[1, 1], false)
                        .place(CpuKernel::ValueAssign)
                })
                .subschedule(&[1], &|matmul_b| {
                    matmul_b.move_param(1, CpuMemoryLevel::RF, row_major(2), None)
                })
                .subschedule(&[1, 0], &|move_ba| {
                    move_ba
                        .tile_out(&[1, 1], false)
                        .place(CpuKernel::ValueAssign)
                })
                .subschedule(&[1, 1], &|matmul_bb| {
                    matmul_bb.move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                })
                .subschedule(&[1, 1, 0], &|s| s.place(CpuKernel::ValueAssign))
                .subschedule(&[1, 1, 1], &|s| s.split(1).place(CpuKernel::MultAdd))
                .subschedule(&[1, 1, 2], &|s| s.place(CpuKernel::ValueAssign))
        });

    // The resulting implementation, as encoded in our Impl intermediate representation, is:
    //   tile (aa: (16×64, u32) <-[0, 2]- #0, ab: (64×16, u32, c1) <-[3, 1]- #1, ac: (16×16, u32, c1) <-[0, 1]- #2)
    //     alloc ad: (16×64, u32, L1) <- aa
    //       alloc ae: (64×16, u32, L1, c1) <- ab
    //         alloc af: (16×16, u32, L1, c1) <- ac
    //           tile (ag: (1×64, u32, L1) <-[0, 2]- ad, ah: (64×1, u32, L1, c1, ua) <-[3, 1]- ae, ai: (1×1, u32, L1, ua) <-[0, 1]- af)
    //               alloc aj: (1×1, u32, RF)
    //                 MemsetZero(aj)
    //                 ValueAssign(aj, ai)
    //               tile (ak: (1×4, u32, L1) <-[0, 1]- ag, al: (4×1, u32, L1, c1, ua) <-[1, 2]- ah)
    //                 alloc am: (1×4, u32, RF)
    //                   tile (an: (1×1, u32, L1, ua) <-[0, 1]- ak, ao: (1×1, u32, RF, ua) <-[0, 1]- am)
    //                     ValueAssign(an, ao)
    //                   alloc ap: (4×1, u32, RF)
    //                     tile (aq: (1×1, u32, L1, ua) <-[0, 1]- al, ar: (1×1, u32, RF, ua) <-[0, 1]- ap)
    //                       ValueAssign(aq, ar)
    //                     alloc as: (1×1, u32, RF)
    //                       ValueAssign(ai, as)
    //                       tile (at: (1×1, u32, RF, ua) <-[0, 1]- am, au: (1×1, u32, RF, ua) <-[1, 2]- ap)
    //                         MultAdd(at, au, as)
    //                       ValueAssign(as, ai)
    //

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Compact);

    // Finally, we can lower that Impl to the following C kernel:
    //
    //   void kernel(
    //     const uint32_t *__restrict__ v000,
    //     const uint32_t *__restrict__ v001,
    //     uint32_t *__restrict__ v002
    //   ) {
    //     for (int v003 = 0; v003 < 4; v003++) {
    //     for (int v004 = 0; v004 < 4; v004++) {
    //       for (int v005 = 0; v005 < 16; v005++) {
    //       for (int v006 = 0; v006 < 16; v006++) {
    //         uint32_t v007;
    //         memset((void *)(&v007), 0, 4);
    //         v002[(64 * v005 + 1024 * v003 + v006 + 16 * v004)] = v007;
    //         for (int v008 = 0; v008 < 16; v008++) {
    //           uint32_t v009[4] __attribute__((aligned (128)));
    //           for (int v010 = 0; v010 < 4; v010++) {
    //             v009[(v010)] = v000[(64 * v005 + 1024 * v003 + v010 + 4 * v008)];
    //           }
    //           uint32_t v011[4] __attribute__((aligned (128)));
    //           for (int v012 = 0; v012 < 4; v012++) {
    //             v011[(v012)] = v001[(64 * v012 + 256 * v008 + v006 + 16 * v004)];
    //           }
    //           uint32_t v013;
    //           v013 = v002[(64 * v005 + 1024 * v003 + v006 + 16 * v004)];
    //           for (int v014 = 0; v014 < 4; v014++) {
    //             v013 += v009[(v014)] * v011[(v014)];  /* MultAdd */
    //           }
    //           v002[(64 * v005 + 1024 * v003 + v006 + 16 * v004)] = v013;
    //         }
    //       }
    //       }
    //     }
    //     }
    //   }
    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap();

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
    const ITERS: u32 = 100;
    let result = implementation.bench(ITERS, None).unwrap();
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&implementation).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)");
    println!(
        "// {:.4} gigaFLOPs/sec ({:.1}% of Zen 1 theoretical max.)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        (100.0 * spec.flops().unwrap() as f64 * throughput) / (52.0 * 1_000_000_000.0)
    );
}
