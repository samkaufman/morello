use morello::codegen::CodeGen;
use morello::common::Dtype;
use morello::cost::Cost;
use morello::imp::ImplNode;
use morello::layout::row_major;
use morello::pprint::ImplPrintStyle;
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::CpuKernel;
use morello::target::{
    CpuMemoryLevel::{GL, L1, RF},
    Target, X86Target,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

use std::io;

fn main() {
    let basics0 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![2048, 32, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let basics1 = PrimitiveBasics {
        typ: PrimitiveSpecType::Matmul { accum: false },
        spec_shape: shape![2048, 2048, 2048],
        dtypes: vec![Dtype::Float32, Dtype::Float32, Dtype::Float32],
    };
    let aux = TensorSpecAux {
        contig: row_major(2).contiguous_full(),
        aligned: true,
        level: GL,
        layout: row_major(2),
        vector_size: None,
    };

    let mut spec = Spec::<X86Target>(
        LogicalSpec::Compose {
            components: vec![basics1, basics0],
            operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux],
            serial_only: true,
        },
        X86Target::max_mem(),
    );
    spec.canonicalize().unwrap();

    let imp = spec.tile_out(&[128, 128]);
    let imp = imp
        .bufferize(0, GL, row_major(2), None)
        .subschedule(&[0], schedule_matmul)
        .subschedule(&[1], schedule_matmul);
    // pprint(&imp, ImplPrintStyle::Compact);
    // println!("\nThe above Impl lowered to C:");
    imp.emit(
        false,
        Some(ImplPrintStyle::Compact),
        &mut ToWriteFmt(io::stdout()),
    )
    .unwrap_or_else(|e| panic!("Failed to generate code: {}", e));

    // Benchmark.
    const ITERS: u32 = 10;
    let result = imp
        .bench(ITERS, None)
        .unwrap_or_else(|e| panic!("Failed to benchmark: {}", e));
    let kernel_runtime =
        (result.best_inner_loop_runtime() / result.inner_loop_iterations).as_secs_f64();
    let throughput =
        result.inner_loop_iterations as f64 / result.best_inner_loop_runtime().as_secs_f64();
    println!("\n// cost: {}", Cost::from_impl(&imp).main);
    println!("// kernel runtime: {kernel_runtime:.4}s ({throughput:.2}/sec)",);
    println!(
        "// {:.4} gigaFLOPs/sec (Spec is {} FLOPs)",
        (spec.flops().unwrap() as f64 * throughput) / 1_000_000_000.0,
        spec.flops().unwrap(),
    );
}

fn schedule_matmul(spec: &Spec<X86Target>) -> ImplNode<X86Target> {
    spec.tile_out(&[16, 16])
        .to_accum()
        .split(16)
        .move_param(0, L1, row_major(2), None)
        .move_param(1, L1, row_major(2), None)
        .move_param(2, L1, row_major(2), None)
        .tile_out(&[1, 1])
        .subschedule(&[0], zero_schedule)
        .split(4)
        .move_param(0, RF, row_major(2), None)
        .subschedule(&[1, 0], move_schedule)
        .move_param(1, RF, row_major(2), None)
        .subschedule(&[1, 1, 0], move_schedule)
        .move_param(2, RF, row_major(2), None)
        .subschedule(&[1, 1, 1, 0], move_schedule)
        .subschedule(&[1, 1, 1, 2], move_schedule)
        .split(1)
        .place(CpuKernel::MultAdd)
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
    zero.tile_out(&[1, 16])
        .move_param(0, RF, row_major(2), None)
        .subschedule(&[0], |z| z.place(CpuKernel::MemsetZero))
        .subschedule(&[1], |z| {
            z.move_param(1, L1, row_major(2), None)
                .tile_out(&[1, 1])
                .place(CpuKernel::ValueAssign)
        })
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
        move_spec.tile_out(&[1, 1]).place(CpuKernel::ValueAssign)
    }
}
