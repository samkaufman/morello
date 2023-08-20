// watch -w -d -c -n 1 RUSTFLAGS=-Awarnings cargo run --quiet --example BLIS
#![allow(non_snake_case)] // for crate name, BLIS

use anyhow::Result;
use std::io;

use morello::codegen::CodeGen;
use morello::color::{self, ColorMode};
use morello::common::{Dtype, Shape};
use morello::imp::kernels::KernelType;
use morello::layout::{row_major, Layout};
use morello::pprint::{pprint, PrintMode};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
#[cfg(target_arch = "aarch64")]
use morello::target::ArmTarget as CpuTarget;
#[cfg(target_arch = "x86_64")]
use morello::target::X86Target as CpuTarget;
use morello::target::{CpuMemoryLevel, Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

const HEADER_SIZE: usize = 80;

fn print_header(s: &str) {
    let total_len = (HEADER_SIZE - s.len() - 2) / 2;
    assert!(total_len > 0, "Header too long");
    print!("{1} {} {1}", s, "=".repeat(total_len));
    if s.len() % 2 != 0 {
        println!("=");
    } else {
        println!();
    }
}

fn main() -> Result<()> {
    color::set_color_mode(ColorMode::Auto);

    let size = 64;
    let layout = row_major(2);
    let spec = Spec::<CpuTarget>(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: Shape::from([size, size, size].as_slice()),
                dtype: Dtype::Uint8,
            },
            vec![
                TensorSpecAux {
                    contig: layout.contiguous_full(),
                    aligned: true,
                    level: CpuMemoryLevel::GL,
                    layout: layout.clone(),
                    vector_size: None,
                };
                3
            ],
            true,
        ),
        CpuTarget::max_mem(),
    );
    print_header("Logical Spec");
    println!("{}\n", spec.0);
    // Matmul((64×64, u8), (64×64, u8), (64×64, u8), serial)

    let imp = spec
        .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
        //
        .tile_out(&[4, 4], false)
        // B.pack_into
        .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
        //
        .split(4)
        // A.pack_into
        .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
        //
        // Macrokernel
        .tile_out(&[2, 2], false)
        //
        // Microkernel
        .tile_out(&[1, 1], false)
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
    print_header("Impl resulting from manual scheduling");
    pprint(&imp, PrintMode::Compact);
    println!();
    // alloc aa: (64×64, u8, L1) <- #0
    //   alloc ab: (64×64, u8, L1) <- #1
    //     alloc ac: (64×64, u8, L1) <- #2
    //       tile (ad: (4×64, u8, L1) <-[0, 2]- aa, ae: (64×4, u8, L1, c1, ua) <-[3, 1]- ab, af: (4×4, u8, L1, c1, ua) <-[0, 1]- ac)
    //         alloc ag: (64×4, u8, L1, c1, ua) <- ae
    //           tile (ah: (4×4, u8, L1, c1, ua) <-[0, 1]- ad, ai: (4×4, u8, L1, c1, ua) <-[1, 2]- ag)
    //             alloc aj: (4×4, u8, L1, c1, ua) <- ah
    //               tile (ak: (2×4, u8, L1, c1, ua) <-[0, 2]- aj, al: (4×2, u8, L1, c1, ua) <-[3, 1]- ai, am: (2×2, u8, L1, c1, ua) <-[0, 1]- af)
    //                 tile (an: (1×4, u8, L1, ua) <-[0, 2]- ak, ao: (4×1, u8, L1, c1, ua) <-[3, 1]- al, ap: (1×1, u8, L1, ua) <-[0, 1]- am)
    //                   alloc aq: (1×4, u8, RF)
    //                     tile (ar: (1×1, u8, L1, ua) <-[0, 1]- an, as: (1×1, u8, RF, ua) <-[0, 1]- aq)
    //                       ValueAssign(ar, as)
    //                     alloc at: (4×1, u8, RF)
    //                       tile (au: (1×1, u8, L1, ua) <-[0, 1]- ao, av: (1×1, u8, RF, ua) <-[0, 1]- at)
    //                         ValueAssign(au, av)
    //                       alloc aw: (1×1, u8, RF)
    //                         ValueAssign(ap, aw)
    //                         Mult(aq, at, aw)
    //                         ValueAssign(aw, ap)

    print_header("The above Impl lowered to C");
    imp.emit(&mut ToWriteFmt(io::stdout())).unwrap();
    // #include <inttypes.h>
    // #include <stdlib.h>
    // #include <stdint.h>
    // #include <stdio.h>
    // #include <string.h>
    // #include <time.h>
    // #include <arm_neon.h>
    //
    // __attribute__((noinline))
    // void kernel(
    //   uint8_t *restrict aa,
    //   uint8_t *restrict ab,
    //   uint8_t *restrict ac
    // ) {
    //   for (int ad = 0; ad < 16; ad++) {
    //   for (int ae = 0; ae < 16; ae++) {
    //     TODO: supposed to have let Bc = B.pack_into() here?
    //     for (int af = 0; af < 16; af++) {
    //       TODO: supposed to have let Ac = A.pack_into() here?
    //       for (int ag = 0; ag < 2; ag++) {
    //       for (int ah = 0; ah < 2; ah++) {
    //         for (int ai = 0; ai < 2; ai++) {
    //         for (int aj = 0; aj < 2; aj++) {
    //           uint8_t ak[4] __attribute__((aligned (128)));
    //           for (int al = 0; al < 4; al++) {
    //             ak[al] = aa[256 * ad + 4 * af + 128 * ag + 64 * ai + al];
    //           }
    //           uint8_t am[4] __attribute__((aligned (128)));
    //           for (int an = 0; an < 4; an++) {
    //             am[an] = ab[4 * ae + 256 * af + 2 * ah + aj + 64 * an];
    //           }
    //           uint8_t ao;
    //           ao = ac[256 * ad + 4 * ae + 128 * ag + 2 * ah + 64 * ai + aj];
    //           ao += ak[0] * am[0];  /* Mult */
    //           ac[256 * ad + 4 * ae + 128 * ag + 2 * ah + 64 * ai + aj] = ao;
    //         }
    //         }
    //       }
    //       }
    //     }
    //   }
    //   }
    // }
    //
    // int main() {
    //   uint8_t ap[4096] __attribute__((aligned (128))) = {0};
    //   uint8_t aq[4096] __attribute__((aligned (128))) = {0};
    //   uint8_t ar[4096] __attribute__((aligned (128))) = {0};
    //
    //   kernel(&ap[0], &aq[0], &ar[0]);
    //
    //   printf("64x64\n");
    //   for (int a = 0; a < 64; a++) {
    //   for (int b = 0; b < 64; b++) {
    //     printf("%" PRIu8 " ", ar[64 * a + b]);
    //   }
    //   printf("\n");
    //   }
    //
    //   return 0;

    Ok(())
}
