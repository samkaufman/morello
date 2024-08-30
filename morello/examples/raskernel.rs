use morello::codegen::CodeGen;
use morello::cost::Cost;
use morello::imp::ImplNode;
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
    const M: u32 = 64;
    const K: u32 = 64;
    const N: u32 = 64;
    let spec = Spec::<X86Target>(
        lspec!(Matmul(
            [M, K, N],
            (u32, GL, row_major(2)),
            (u32, GL, row_major(2)),
            (u32, GL, row_major(2)),
            serial
        )),
        X86Target::max_mem(),
    );
    println!("Logical Spec: {}", spec.0);

    // First, tile to a 16x64x16 matmul and move all operands into the L1 cache.  Tiling will
    // introduce two loops into the final C code, while the `.move_param` calls lower to nothing the
    // final code. Moves into the L1 cache instead model the behavior of the hardware cache and, by
    // changing the Spec, allow the remainder of the schedule to assume tensors are in L1.
    let implementation = spec
        .tile_out(&[16, 16], false)
        .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
        .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
        .to_accum()
        .subschedule(&[1], &|s| s.split(4).split(1).place(CpuKernel::RasKernel));

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
        "// {:.4} gigaFLOPs/sec",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
    );
}

/// Schedules the given 1x1 Zero Spec.
///
/// Specifically, this moves the Zero's tensor from L1 into registers, which introduces two
/// sub-Specs:
///  Zero((1×1, u32, RF), serial)
///  Move((1×1, u32, RF), (1×1, u32, L1, ua), serial)
/// These are then implemented with kernels which lower to `memset` and `=` respectively, like so:
/// ```
//  uint32_t v;
//  memset((void *)(&v), 0, 4);
//  l1_tile[index] = v;
/// ```
fn zero_schedule(zero: &Spec<X86Target>) -> ImplNode<X86Target> {
    zero.move_param(0, CpuMemoryLevel::RF, row_major(2), None)
        .subschedule(&[0], &|z| z.place(CpuKernel::MemsetZero))
        .subschedule(&[1], &|move_back| move_back.place(CpuKernel::ValueAssign))
}

/// Schedules the given Move Spec.
///
/// Specifically, this checks if the Move's tensor is a single value. If it is, it directly assigns
/// the value using the `ValueAssign` kernel. If not, it tiles the tensor and then assigns the values
/// using the `ValueAssign` kernel.
fn move_schedule(move_spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    let is_single_value = move_spec.0.parameter_shapes()[0]
        .iter()
        .all(|size| size.get() == 1);
    if is_single_value {
        move_spec.place(CpuKernel::ValueAssign)
    } else {
        move_spec
            .tile_out(&[1, 1], false)
            .place(CpuKernel::ValueAssign)
    }
}
