//! This example shows how to manually schedule a simple matrix multiplication for X86.

use morello::codegen::CodeGen;
use morello::common::{Dtype, Shape};
use morello::imp::kernels::KernelType;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{CpuMemoryLevel, Target, X86Target};
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
    // assumed to be fully contiguous, in global memory (not a cache or registers), and laid out
    // row-major. The `vector_size` field applies only to tensors in vector registers (VRF), so it
    // is `None` below.
    let layout = row_major(2);
    let spec = Spec::<X86Target>(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: Shape::from([64, 64, 64].as_slice()),
                dtype: Dtype::Uint32,
            },
            vec![
                TensorSpecAux {
                    contig: layout.contiguous_full(),
                    level: CpuMemoryLevel::GL,
                    layout,
                    vector_size: None,
                };
                3
            ],
            true,
        ),
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
        .split(4)
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
        .subschedule(&[1, 1, 1], &|s| s.split(1).place(KernelType::Mult))
        .subschedule(&[1, 1, 2], &|s| s.place(KernelType::ValueAssign));

    // The resulting implementation, as encoded in our Impl intermediate representation, is:
    //
    //   tile (aa: (8×16, u32, c1) <-[3, 1]- #1, ab: (16×16, u32, c1) <-[0, 1]- #2)
    //   alloc ac: (16×8, u32, L1) <- #0
    //     alloc ad: (8×16, u32, L1, c1) <- aa
    //       alloc ae: (16×16, u32, L1, c1) <- ab
    //         tile (af: (1×8, u32, L1) <-[0, 2]- ac, ag: (8×1, u32, L1, c1, ua) <-[3, 1]- ad, ah: (1×1, u32, L1, ua) <-[0, 1]- ae)
    //           tile (ai: (1×4, u32, L1, ua) <-[0, 1]- af, aj: (4×1, u32, L1, c1, ua) <-[1, 2]- ag)
    //             alloc ak: (1×4, u32, RF)
    //               tile (al: (1×1, u32, L1, ua) <-[0, 1]- ai, am: (1×1, u32, RF, ua) <-[0, 1]- ak)
    //                 ValueAssign(al, am)
    //               alloc an: (4×1, u32, RF)
    //                 tile (ao: (1×1, u32, L1, ua) <-[0, 1]- aj, ap: (1×1, u32, RF, ua) <-[0, 1]- an)
    //                   ValueAssign(ao, ap)
    //                 alloc aq: (1×1, u32, RF)
    //                   ValueAssign(ah, aq)
    //                   tile (ar: (1×1, u32, RF, ua) <-[0, 1]- ak, as: (1×1, u32, RF, ua) <-[1, 2]- an)
    //                     Mult(ar, as, aq)
    //                   ValueAssign(aq, ah)
    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Full);

    // Finally, we can lower that Impl to the following C kernel:
    //
    //   __attribute__((noinline))
    //   void kernel(
    //     uint32_t *restrict aa,
    //     uint32_t *restrict ab,
    //     uint32_t *restrict ac
    //   ) {
    //     for (int ad = 0; ad < 2; ad++) {
    //       for (int ae = 0; ae < 16; ae++) {
    //       for (int af = 0; af < 16; af++) {
    //         for (int ag = 0; ag < 2; ag++) {
    //           uint32_t ah[4] __attribute__((aligned (128)));
    //           for (int ai = 0; ai < 4; ai++) {
    //             ah[(ai)] = aa[(8 * ae + ai + 4 * ag)];
    //           }
    //           uint32_t aj[4] __attribute__((aligned (128)));
    //           for (int ak = 0; ak < 4; ak++) {
    //             aj[(ak)] = ab[(32 * ak + 128 * ag + af + 16 * ad)];
    //           }
    //           uint32_t al;
    //           al = ac[(32 * ae + af + 16 * ad)];
    //           for (int am = 0; am < 4; am++) {
    //             al += ah[(am)] * aj[(am)];  /* Mult */
    //           }
    //           ac[(32 * ae + af + 16 * ad)] = al;
    //         }
    //       }
    //       }
    //     }
    //   }

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(None, None, &mut ToWriteFmt(io::stdout()))
        .unwrap();

    // If the verification flag is set, let's additionally double-check that the lowered
    // code builds and produces the correct results.
    #[cfg(feature = "verification")]
    {
        let artifact = implementation.build(None).unwrap();
        if !artifact.check_correctness(&spec) {
            panic!("Generated code returned incorrect output");
        }
    }
}
