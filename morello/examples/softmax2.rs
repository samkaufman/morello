use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    Avx2Target,
    CpuMemory::{self, GL, L1, RF},
    Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;
use std::panic;

fn main() {
    let shape = shape![8, 512];
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim: 1 },
            spec_shape: shape.clone(),
            dtypes: vec![Dtype::Float32; shape.len()],
        },
        vec![
            TensorSpecAux {
                memory: CpuMemory::GL,
                layout: row_major(&shape),
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
        .tile_out(&[1, shape[1].get()])
        .to_softmax_parts(GL, row_major, None, GL, row_major, None)
        .subschedule(&[0], |subspec| {
            subspec.to_max_and_unscaled(GL, row_major, None)
        })
        .subschedule(&[0, 0], |subspec| subspec.to_accum())
        .subschedule(&[0, 0, 0], |subspec| {
            subspec
                .move_param(0, L1)
                .move_param(0, RF)
                .subschedule(&[0], |s| s.select(CpuKernel::ValueNegInf))
                .subschedule(&[1], |s| s.select(CpuKernel::Assign))
        })
        .subschedule(&[0, 0, 1], |maxaccum| {
            maxaccum
                .move_param(0, L1)
                .move_param(1, L1)
                .move_param(1, RF)
                .select(CpuKernel::VectorMaxLoop)
                .subschedule(&[0], |s| s.select(CpuKernel::Assign))
                .subschedule(&[2], |s| s.select(CpuKernel::Assign))
        })
        .subschedule(&[0, 1], |subspec| {
            subspec
                .to_accum()
                .subschedule(&[0], |s| {
                    s.move_param(0, L1)
                        .move_param(0, RF)
                        .subschedule(&[0], |s| s.select(CpuKernel::MemsetZero))
                        .subschedule(&[1], |s| s.select(CpuKernel::Assign))
                })
                .subschedule(&[1], |s| {
                    s.move_param(0, CpuMemory::L1)
                        .move_param(1, CpuMemory::L1)
                        .move_param(2, CpuMemory::L1)
                        .move_param(3, CpuMemory::L1)
                        .move_param(1, CpuMemory::RF)
                        .subschedule(&[0], |m| m.select(CpuKernel::Assign))
                        .move_param(2, CpuMemory::RF)
                        .subschedule(&[1, 0], |m| m.select(CpuKernel::Assign))
                        .select(CpuKernel::VectorSoftmaxDenominatorAndUnscaledF32)
                        .subschedule(&[1, 2], |m| m.select(CpuKernel::Assign))
                })
        })
        .subschedule(&[1], |dvs| dvs.select(CpuKernel::DivideVecScalarReciprocal));

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
