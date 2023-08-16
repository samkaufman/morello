//! This example shows how to manually schedule a simple matrix multiplication for X86.

use morello::codegen::CodeGen;
use morello::common::{Dtype, Shape};
use morello::imp::kernels::KernelType;
use morello::layout::row_major;
use morello::pprint::{pprint, PrintMode};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{CpuMemoryLevel, Target, X86Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;

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
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: Shape::from([64, 64, 64].as_slice()),
                dtype: Dtype::Uint8,
            },
            vec![
                TensorSpecAux {
                    contig: layout.contiguous_full(),
                    aligned: true,
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
        .subschedule(&[1, 1, 1], &|s| s.place(KernelType::Mult))
        .subschedule(&[1, 1, 2], &|s| s.place(KernelType::ValueAssign));

    // The resulting implementation, as encoded in our Impl intermediate representation, is:
    //
    // tile (aa: (16×64, u8) <-[0, 2]- #0, ab: (64×16, u8, c1, ua) <-[3, 1]- #1, ac: (16×16, u8, c1, ua) <-[0, 1]- #2)
    // alloc ad: (16×64, u8, L1) <- aa
    //     alloc ae: (64×16, u8, L1, c1, ua) <- ab
    //     alloc af: (16×16, u8, L1, c1, ua) <- ac
    //         tile (ag: (1×64, u8, L1) <-[0, 2]- ad, ah: (64×1, u8, L1, c1, ua) <-[3, 1]- ae, ai: (1×1, u8, L1, ua) <-[0, 1]- af)
    //         tile (aj: (1×4, u8, L1, ua) <-[0, 1]- ag, ak: (4×1, u8, L1, c1, ua) <-[1, 2]- ah)
    //             alloc al: (1×4, u8, RF)
    //             tile (am: (1×1, u8, L1, ua) <-[0, 1]- aj, an: (1×1, u8, RF, ua) <-[0, 1]- al)
    //                 ValueAssign(am, an)
    //             alloc ao: (4×1, u8, RF)
    //                 tile (ap: (1×1, u8, L1, ua) <-[0, 1]- ak, aq: (1×1, u8, RF, ua) <-[0, 1]- ao)
    //                 ValueAssign(ap, aq)
    //                 alloc ar: (1×1, u8, RF)
    //                 ValueAssign(ai, ar)
    //                 Mult(al, ao, ar)
    //                 ValueAssign(ar, ai)
    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, PrintMode::Full);

    // Finally, we can lower that Impl to the following C code:
    //
    //   #include <inttypes.h>
    //   #include <stdlib.h>
    //   #include <stdint.h>
    //   #include <stdio.h>
    //   #include <string.h>
    //   #include <time.h>
    //   #include <immintrin.h>
    //   __attribute__((noinline))
    //   void kernel(
    //     uint8_t *restrict aa,
    //     uint8_t *restrict ab,
    //     uint8_t *restrict ac
    //   ) {
    //     for (int ad = 0; ad < 4; ad++) {
    //     for (int ae = 0; ae < 4; ae++) {
    //       for (int af = 0; af < 16; af++) {
    //       for (int ag = 0; ag < 16; ag++) {
    //         for (int ah = 0; ah < 16; ah++) {
    //           uint8_t ai[4] __attribute__((aligned (128)));
    //           for (int aj = 0; aj < 4; aj++) {
    //             ai[aj] = aa[1024 * ad + 64 * af + 4 * ah + aj];
    //           }
    //           uint8_t ak[4] __attribute__((aligned (128)));
    //           for (int al = 0; al < 4; al++) {
    //             ak[al] = ab[16 * ae + ag + 256 * ah + 64 * al];
    //           }
    //           uint8_t am;
    //           am = ac[1024 * ad + 16 * ae + 64 * af + ag];
    //           am += ai[0] * ak[0];  /* Mult */
    //           ac[1024 * ad + 16 * ae + 64 * af + ag] = am;
    //         }
    //       }
    //       }
    //     }
    //     }
    //   }
    //
    //   int main() {
    //     uint8_t an[4096] __attribute__((aligned (128))) = {0};
    //     uint8_t ao[4096] __attribute__((aligned (128))) = {0};
    //     uint8_t ap[4096] __attribute__((aligned (128))) = {0};
    //
    //     kernel(&an[0], &ao[0], &ap[0]);
    //
    //     printf("64x64\n");
    //     for (int a = 0; a < 64; a++) {
    //     for (int b = 0; b < 64; b++) {
    //       printf("%" PRIu8 " ", ap[64 * a + b]);
    //     }
    //     printf("\n");
    //     }
    //
    //     return 0;
    //   }

    println!("\nThe above Impl lowered to C:");
    implementation.emit(&mut ToWriteFmt(io::stdout())).unwrap();
}
