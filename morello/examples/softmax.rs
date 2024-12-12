use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::db::FilesDatabase;
use morello::layout::row_major;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule as _};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    CpuMemoryLevel::{self, GL},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use nonzero::nonzero as nz;
use std::io;
use std::panic;

fn main() {
    // First, we'll define the Spec for the program we will implement: a 64x64x64 matrix
    // multiplication with unsigned, 32-bit integer inputs and output.
    //
    // This is a non-accumulating Spec (`Matmul` rather than `MatmulAccum`), which means that the
    // implementation will set rather then add values to the output tensor.
    const RANK: u8 = 2;
    const SIZE: DimSize = nz!(1024u32);
    let layouts = [row_major(RANK), row_major(RANK)];
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Softmax { scan_dim: 1 },
            spec_shape: vec![SIZE; usize::from(RANK)],
            dtypes: vec![Dtype::Float32; usize::from(RANK)],
        },
        layouts
            .into_iter()
            .map(|layout| TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level: CpuMemoryLevel::GL,
                layout,
                vector_size: None,
            })
            .collect(),
        true,
    );
    let spec = Spec::<X86Target>(logical_spec, X86Target::max_mem());
    println!("Logical Spec: {}", spec.0);

    let db = FilesDatabase::new(None, true, 1, 10_000, 1, None);

    let implementation = spec
        // Tile across the batch dimension. (We cannot tile across the scan dimension.)
        .tile_out(&[1, SIZE.get()])
        // Split into sub-Specs for computing denominator-and-max and then one to complete softmax.
        .to_softmax_parts(GL, row_major, None, GL, row_major, None)
        // This [0] corresponds to SoftmaxDenominatorAndMax
        .subschedule(&[0], |subspec| subspec.to_max_and_denominator())
        // This [0, 0] corresponds to Max
        .subschedule(&[0, 0], |subspec| {
            subspec.to_accum().split(1).synthesize(&db, None)
        })
        .subschedule(&[0, 0, 0], |subspec| subspec.synthesize(&db, None))
        // This [0, 1] corresponds to the SoftmaxDenominator
        .subschedule(&[0, 1], |subspec| {
            subspec
                .to_accum()
                .split(32)
                .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(0, CpuMemoryLevel::VRF, row_major(2), Some(nz!(8u32)))
                .move_param(1, CpuMemoryLevel::RF, row_major(2), None)
                .move_param(2, CpuMemoryLevel::RF, row_major(2), None)
                .place(CpuKernel::VectorSoftmaxDenominator)
        })
        .subschedule(&[0, 1, 0], |subspec| subspec.synthesize(&db, None))
        .subschedule(&[0, 1, 1, 0], |subspec| subspec.synthesize(&db, None))
        .subschedule(&[0, 1, 1, 1, 0], |subspec| subspec.synthesize(&db, None))
        .subschedule(&[0, 1, 1, 1, 1, 1], |subspec| subspec.synthesize(&db, None))
        // This [1] corresponds to SoftmaxComplete
        .subschedule(&[1], |softmax_complete| {
            softmax_complete
                .move_param(0, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(1, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(2, CpuMemoryLevel::L1, row_major(2), None)
                .move_param(3, CpuMemoryLevel::L1, row_major(2), None)
                .tile_out(&[1, 32])
                .synthesize(&db, None)
        });

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, ImplPrintStyle::Compact);

    println!("\nThe above Impl lowered to C:");
    implementation
        .emit(false, None, &mut ToWriteFmt(io::stdout()))
        .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

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
}
