use iai_callgrind::{black_box, main};
use morello::db::DashmapDiskDatabase;
use smallvec::smallvec;

use morello::common::{DimSize, Dtype};
use morello::layout::row_major;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
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
        vec![
            TensorSpecAux {
                contig: rm2.contiguous_full(),
                aligned: true,
                level: Tgt::default_level(),
                layout: rm2,
                vector_size: None,
            };
            3
        ],
        true,
    );
    Spec(logical_spec, X86Target::max_mem())
}

fn synth(goal: &Spec<X86Target>) {
    let db = DashmapDiskDatabase::new(None, true, 1);
    morello::search::top_down(&db, black_box(goal), 1, false);
}

#[inline(never)]
fn synth_matmul_benchmark_1() {
    synth(&matmul_spec(1));
}

#[inline(never)]
fn synth_matmul_benchmark_2() {
    synth(&matmul_spec(2));
}

main!(
    callgrind_args = "toggle-collect=morello_bench_synth::matmul_spec",
        "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = synth_matmul_benchmark_1, synth_matmul_benchmark_2
);
