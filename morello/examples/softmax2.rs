use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::smallvec::smallvec;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    Avx2Target,
    CpuMemoryLevel::{self, GL, L1, RF, VRF},
    Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;
use std::io;
use std::panic;

fn main() {
    const RANK: u8 = 2;
    const SIZE: DimSize = nz!(128u32);
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim: 1 },
            spec_shape: smallvec![SIZE; usize::from(RANK)],
            dtypes: vec![Dtype::Float32; usize::from(RANK)],
        },
        vec![
            TensorSpecAux {
                level: CpuMemoryLevel::GL,
                layout: row_major(RANK),
                vector_size: None,
            };
            2
        ],
        true,
    );
    let spec = Spec::<Avx2Target>(logical_spec, Avx2Target::max_mem());
    println!("Logical Spec: {}", spec.0);

    let implementation = spec
        // Tile across the batch dimension. (We cannot tile across the scan dimension.)
        .tile_out(&[1, SIZE.get()])
        .to_softmax_parts(GL, row_major(RANK), None, GL, row_major(RANK), None)
        .subschedule(&[0], |subspec| {
            subspec.to_max_and_unscaled(GL, row_major(RANK), None)
        })
        .subschedule(&[0, 0], |subspec| subspec.to_accum().split(1))
        .subschedule(&[0, 0, 0], |subspec| {
            subspec
                .move_param(0, L1)
                .move_param(0, RF)
                .subschedule(&[0], |s| s.select(CpuKernel::ValueNegInf))
                .subschedule(&[1], |s| s.select(CpuKernel::ValueAssign))
        })
        .subschedule(&[0, 0, 1], |maxaccum| {
            maxaccum
                .move_param(0, L1)
                .move_param(0, RF)
                .move_param(1, L1)
                .move_param(1, RF)
                .select(CpuKernel::ValueMax)
                .subschedule(&[0], |s| s.select(CpuKernel::ValueAssign))
                .subschedule(&[1, 0], |s| s.select(CpuKernel::ValueAssign))
                .subschedule(&[1, 2], |s| s.select(CpuKernel::ValueAssign))
        })
        .subschedule(&[0, 1], |subspec| {
            subspec
                .to_accum()
                .subschedule(&[0], |s| {
                    s.move_param(0, L1)
                        .move_param(0, RF)
                        .subschedule(&[0], |s| s.select(CpuKernel::MemsetZero))
                        .subschedule(&[1], |s| s.select(CpuKernel::ValueAssign))
                })
                .subschedule(&[1], |s| {
                    s.tile_out(&[1, 32])
                        .move_param(0, CpuMemoryLevel::L1)
                        .move_param(1, CpuMemoryLevel::L1)
                        .move_param(2, CpuMemoryLevel::L1)
                        .move_param(3, CpuMemoryLevel::L1)
                        .move_vrf(0, CpuMemoryLevel::VRF, nz!(8u32))
                        .subschedule(&[0], |m| {
                            m.tile_out(&[1, 8]).select(CpuKernel::VectorAssign)
                        })
                        .move_param(1, CpuMemoryLevel::RF)
                        .subschedule(&[1, 0], |m| m.select(CpuKernel::ValueAssign))
                        .move_param(2, CpuMemoryLevel::RF)
                        .subschedule(&[1, 1, 0], |m| m.select(CpuKernel::ValueAssign))
                        .subschedule(&[1, 1, 2], |m| m.select(CpuKernel::ValueAssign))
                        .move_vrf(3, CpuMemoryLevel::VRF, nz!(8u32))
                        .select(CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32)
                        .subschedule(&[1, 1, 1, 1], |move_spec| {
                            move_spec.tile_out(&[1, 8]).select(CpuKernel::VectorAssign)
                        })
                })
        })
        .subschedule(&[1], |subspec| {
            subspec
                .tile_out(&[1, 4])
                .broadcast_first(VRF, row_major(RANK), Some(nz!(4u32)))
                .subschedule(&[0], |broadcast| {
                    broadcast
                        .move_param(0, CpuMemoryLevel::L1)
                        .move_param(0, CpuMemoryLevel::RF)
                        .subschedule(&[0], |s| s.select(CpuKernel::ValueAssign))
                        .select(CpuKernel::VecScalarAssign)
                })
                .subschedule(&[1], |d| {
                    d.move_param(0, L1)
                        .move_vrf(0, VRF, nz!(4u32))
                        .subschedule(&[0], |m| m.select(CpuKernel::VectorAssign))
                        .move_param(2, L1)
                        .move_vrf(2, VRF, nz!(4u32))
                        .subschedule(&[1, 0], |m| m.select(CpuKernel::DivideVec))
                        .subschedule(&[1, 1], |m| m.select(CpuKernel::VectorAssign))
                })
        });

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Compact);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
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
    const ITERS: u32 = 100;
    let result = implementation
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {e}"));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&implementation).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)");
    println!(
        "// {:.4} gigaFLOPs/sec (Spec is {} FLOPs)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        spec.flops().unwrap(),
    );
}
