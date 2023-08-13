use morello::codegen::CodeGen;
use morello::common::{Dtype, Shape, Spec};
use morello::imp::kernels::KernelType;
use morello::layout::row_major;
use morello::pprint::{pprint, PrintMode};
use morello::scheduling_sugar::{SchedulingSugar, Subschedule};
use morello::spec::{LogicalSpec, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
use morello::target::{CpuMemoryLevel, Target, X86Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;
use std::fmt::Debug;
use std::io;

fn main() {
    // Define the spec for a 64x64x64 matrix multiplication.
    let layout = row_major(2);
    let spec = Spec::<X86Target>(
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: Shape::from([64, 64, 64].as_slice()),
                dtype: Dtype::Uint8,
            },
            PrimitiveAux::Standard(vec![
                TensorSpecAux {
                    contig: layout.contiguous_full(),
                    aligned: true,
                    level: CpuMemoryLevel::GL,
                    layout,
                    vector_size: None,
                };
                3
            ]),
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

    println!("\nImpl resulting from manual scheduling:");
    pprint(&implementation, PrintMode::Full);

    // Generate C code.
    println!("\nThe above Impl lowered to C:");
    implementation.emit(&mut ToWriteFmt(io::stdout())).unwrap();
}
