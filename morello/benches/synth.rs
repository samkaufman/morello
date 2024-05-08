use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use nonzero::nonzero as nz;
use std::hint::black_box;

use morello::db::RocksDatabase;
use morello::layout::row_major;
use morello::lspec;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{Target, X86Target};
use morello::tensorspec::TensorSpecAux;

#[export_name = "morello_bench_synth::matmul_spec"]
fn matmul_spec<Tgt: Target>(size: u32) -> Spec<Tgt> {
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
    let db = RocksDatabase::try_new(None, true, 1).unwrap();
    morello::search::top_down(&db, black_box(goal), 1, Some(nz!(1usize)));
}

#[library_benchmark]
#[benches::multiple(1)]
fn synth_matmul(size: u32) {
    synth(&matmul_spec(black_box(size)));
}

library_benchmark_group!(
    name = synth_group;
    benchmarks = synth_matmul
);

main!(
    config = LibraryBenchmarkConfig::default()
                .raw_callgrind_args([
                    "toggle-collect=morello_bench_synth::matmul_spec",
                    "--simulate-wb=no", "--simulate-hwpref=yes",
                    "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64",
                ]);
    library_benchmark_groups = synth_group
);
