//! This example shows how to manually schedule a simple matrix multiplication for X86.

use morello::codegen::CodeGen;
use morello::imp::kernels::KernelType;
use morello::layout::row_major;
use morello::lspec;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
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
    // multiplication with unsigned, 8-bit integer inputs and output.
    //
    // This is a non-accumulating Spec (notice the `accum: false` below), which means that the
    // implementation will set rather then add values to the output tensor. Additionally, extra
    // details about each tensor are included in the `PrimitiveAux` structure: that the memory is
    // assumed to be fully contiguous, aligned, in global memory (not a cache or registers), and
    // laid out row-major. The `vector_size` field applies only to tensors in vector registers
    // (VRF), so it is `None` below.
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
        .subschedule(&[0, 0], &|z| z.place(KernelType::MemsetZero))
        .subschedule(&[0, 1], &|move_back| {
            move_back.place(KernelType::ValueAssign)
        })
        .subschedule(&[1], &|bat| {
            bat.split(4)
                .move_param(0, CpuMemoryLevel::RF, row_major(2), None)
                .subschedule(&[0], &|move_a| {
                    move_a
                        .tile_out(&[1, 1], false)
                        .place(KernelType::ValueAssign)
                })
                .subschedule(&[1], &|matmul_b| {
                    matmul_b.move_param(1, CpuMemoryLevel::RF, row_major(2), None)
                })
                .subschedule(&[1, 0], &|move_ba| {
                    move_ba
                        .tile_out(&[1, 1], false)
                        .place(KernelType::ValueAssign)
                })
                .subschedule(&[1, 1], &|matmul_bb| {
                    matmul_bb.move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                })
                .subschedule(&[1, 1, 0], &|s| s.place(KernelType::ValueAssign))
                .subschedule(&[1, 1, 1], &|s| s.split(1).place(KernelType::MultAdd))
                .subschedule(&[1, 1, 2], &|s| s.place(KernelType::ValueAssign))
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
    pprint(&implementation, ImplPrintStyle::Full);

    // Finally, we can lower that Impl to the following C kernel:
    //
    //    void kernel(
    //      uint32_t *__restrict__ aa,
    //      uint32_t *__restrict__ ab,
    //      uint32_t *__restrict__ ac
    //    ) {
    //      for (int ad = 0; ad < 4; ad++) {
    //      for (int ae = 0; ae < 4; ae++) {
    //        for (int af = 0; af < 16; af++) {
    //        for (int ag = 0; ag < 16; ag++) {
    //          uint32_t ah;
    //          memset((void *)(&ah), 0, 4);
    //          ac[(64 * af + 1024 * ad + ag + 16 * ae)] = ah;
    //          for (int ai = 0; ai < 16; ai++) {
    //            uint32_t aj[4] __attribute__((aligned (128)));
    //            for (int ak = 0; ak < 4; ak++) {
    //              aj[(ak)] = aa[(64 * af + 1024 * ad + ak + 4 * ai)];
    //            }
    //            uint32_t al[4] __attribute__((aligned (128)));
    //            for (int am = 0; am < 4; am++) {
    //              al[(am)] = ab[(64 * am + 256 * ai + ag + 16 * ae)];
    //            }
    //            uint32_t an;
    //            an = ac[(64 * af + 1024 * ad + ag + 16 * ae)];
    //            for (int ao = 0; ao < 4; ao++) {
    //              an += aj[(ao)] * al[(ao)];  /* MultAdd */
    //            }
    //            ac[(64 * af + 1024 * ad + ag + 16 * ae)] = an;
    //          }
    //        }
    //        }
    //      }
    //      }
    //    }

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap();

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        let artifact = implementation.build(false).unwrap();
        if !artifact.check_correctness(&spec) {
            panic!("Generated code returned incorrect output");
        }
    }
}
