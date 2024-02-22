use iai_callgrind::{black_box, main};
use nonzero::nonzero as nz;

use morello::common::{DimSize, Dtype};
use morello::db::DashmapDiskDatabase;
use morello::layout::row_major;
use morello::lspec;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{Target, X86Target};
use morello::tensorspec::TensorSpecAux;

#[export_name = "morello_bench_synth::matmul_spec"]
fn matmul_spec<Tgt: Target>(size: DimSize) -> Spec<Tgt> {
    let rm2 = row_major(2);
    let logical_spec = lspec!(Matmul(
        [size, size, size],
        (u32, Tgt::default_level(), rm2.clone()),
        (u32, Tgt::default_level(), rm2.clone()),
        (u32, Tgt::default_level(), rm2),
        serial
    ));
    Spec(logical_spec, X86Target::max_mem())
}

fn synth(goal: &Spec<X86Target>) {
    let db = DashmapDiskDatabase::try_new(None, true, 1).unwrap();
    morello::search::top_down(&db, black_box(goal), 1, Some(nz!(1usize)));
}

#[inline(never)]
fn synth_matmul_benchmark_1() {
    synth(&matmul_spec(1));
}

main!(
    callgrind_args = "toggle-collect=morello_bench_synth::matmul_spec",
        "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = synth_matmul_benchmark_1
);
