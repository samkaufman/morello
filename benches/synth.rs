use iai_callgrind::{black_box, main};
use smallvec::smallvec;
use std::sync::RwLock;

use morello::common::{DimSize, Dtype, Spec};
use morello::layout::row_major;
use morello::spec::{LogicalSpec, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
use morello::table::InMemDatabase;
use morello::target::{Target, X86Target};
use morello::tensorspec::TensorSpecAux;

#[export_name = "morello_bench_synth::matmul_spec"]
fn matmul_spec<Tgt: Target>(size: DimSize) -> Spec<Tgt> {
    let rm2 = row_major(2);
    let logical_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: smallvec![size, size, size],
            dtype: Dtype::Uint32,
        },
        PrimitiveAux::Standard(vec![
            TensorSpecAux {
                contig: rm2.contiguous_full(),
                aligned: true,
                level: Tgt::default_level(),
                layout: rm2,
                vector_size: None,
            };
            3
        ]),
        true,
    );
    Spec(logical_spec, X86Target::max_mem())
}

fn synth(goal: &Spec<X86Target>) {
    let db = InMemDatabase::<X86Target>::new();
    let db_lock = RwLock::new(db);
    morello::search::top_down(&db_lock, black_box(goal), 1);
}

#[inline(never)]
fn synth_matmul_benchmark_2() {
    synth(&matmul_spec(2));
}

#[inline(never)]
fn synth_matmul_benchmark_4() {
    synth(&matmul_spec(4));
}

main!(
    callgrind_args = "toggle-collect=morello_bench_synth::matmul_spec";
    functions = synth_matmul_benchmark_2, synth_matmul_benchmark_4
);
