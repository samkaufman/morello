use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::db::FilesDatabase;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{
    CpuMemoryLevel::{self, GL, VRF},
    Target, X86Target,
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
            spec_shape: vec![SIZE; usize::from(RANK)],
            dtypes: vec![Dtype::Float32; usize::from(RANK)],
        },
        vec![
            TensorSpecAux {
                contig: row_major(RANK).contiguous_full(),
                aligned: true,
                level: CpuMemoryLevel::GL,
                layout: row_major(RANK),
                vector_size: None,
            };
            2
        ],
        true,
    );
    let spec = Spec::<X86Target>(logical_spec, X86Target::max_mem());
    println!("Logical Spec: {}", spec.0);

    let db = FilesDatabase::new(None, true, 1, 10_000, 1, None);

    let implementation = spec
        // Tile across the batch dimension. (We cannot tile across the scan dimension.)
        .tile_out(&[1, SIZE.get()])
        .to_softmax_parts(GL, row_major(RANK), None, GL, row_major(RANK), None)
        .subschedule(&[0], |subspec| {
            subspec.to_max_and_unscaled(GL, row_major(RANK), None)
        })
        .subschedule(&[0, 0], |subspec| {
            subspec.to_accum().split(1).synthesize(&db, None)
        })
        .subschedule(&[0, 0, 0], |s| s.synthesize(&db, None))
        .subschedule(&[0, 1], |subspec| {
            subspec
                .to_accum()
                .subschedule(&[0], |s| s.synthesize(&db, None))
                .subschedule(&[1], |s| {
                    s.move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                        .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
                        .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
                        .move_param(3, CpuMemoryLevel::L1, row_major(2), None)
                        .move_param(0, CpuMemoryLevel::VRF, row_major(2), Some(nz!(8u32)))
                        .subschedule(&[0], |m| m.tile_out(&[1, 8]).synthesize(&db, None))
                        .move_param(1, CpuMemoryLevel::RF, row_major(2), None)
                        .subschedule(&[1, 0], |m| m.synthesize(&db, None))
                        .move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                        .subschedule(&[1, 1, 0], |m| m.synthesize(&db, None))
                        .subschedule(&[1, 1, 2], |m| m.synthesize(&db, None))
                        .move_param(3, CpuMemoryLevel::VRF, row_major(2), Some(nz!(8u32)))
                        .subschedule(&[1, 1, 1, 0], |m| m.synthesize(&db, None))
                        .subschedule(&[1, 1, 1, 1], |m| m.synthesize(&db, None))
                })
        })
        .subschedule(&[1], |subspec| {
            subspec
                .tile_out(&[1, 4])
                .broadcast_first(VRF, row_major(RANK), Some(nz!(4u32)))
                .subschedule(&[0], |broadcast| {
                    broadcast
                        .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                        .move_param(0, CpuMemoryLevel::RF, row_major(2), None)
                        .synthesize(&db, None)
                        .subschedule(&[0], |s| s.synthesize(&db, None))
                })
                .subschedule(&[1], |d| d.synthesize(&db, None))
        });

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Compact);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

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
    let result = implementation
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
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
