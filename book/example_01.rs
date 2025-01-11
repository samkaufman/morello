// ANCHOR: specdef_use
use morello::layout::row_major;
use morello::lspec;
use morello::spec::Spec;
use morello::target::{
    CpuMemoryLevel::{self, RF},
    Target, X86Target,
};
// ANCHOR_END: specdef_use
// ANCHOR: schedule_use
use morello::scheduling_sugar::SchedulingSugar;
use morello::target::CpuKernel;
// ANCHOR_END: schedule_use
// ANCHOR: emit_use
use morello::codegen::CodeGen;
// ANCHOR_END: emit_use

fn main() {
    // ANCHOR: specdef
    let mut spec = Spec::<X86Target>(
        lspec!(Matmul(
            [1, 1, 32, 1],
            (u32, RF, row_major), // `RF` = tensor is in register file
            (u32, RF, row_major),
            (u32, RF, row_major),
            serial
        )),
        X86Target::max_mem(),
    );
    spec.canonicalize().unwrap();
    // ANCHOR_END: specdef

    // ANCHOR: schedule
    let implementation = spec.to_accum().split(1).select(CpuKernel::MultAdd);
    // ANCHOR_END: schedule

    // ANCHOR: emit
    implementation.emit_stdout().unwrap();
    // ANCHOR_END: emit
}
