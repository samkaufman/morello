use iai_callgrind::{black_box, main};
use smallvec::smallvec;

use morello::common::{DimSize, Dtype};
use morello::layout::row_major;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use morello::target::{Target, X86Target};
use morello::tensorspec::TensorSpecAux;

// TODO: Add a benchmark for Compose

#[export_name = "morello_bench_logicalspec_parameters::matmul_spec"]
fn matmul_spec<Tgt: Target>(size: DimSize) -> LogicalSpec<Tgt> {
    let rm2 = row_major(2);
    LogicalSpec::Primitive(
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
    )
}

#[export_name = "morello_bench_logicalspec_parameters::conv_spec"]
fn conv_spec<Tgt: Target>(size: DimSize) -> LogicalSpec<Tgt> {
    let rm4 = row_major(4);
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Conv { accum: false },
            spec_shape: smallvec![size; 7],
            dtype: Dtype::Uint32,
        },
        vec![
            TensorSpecAux {
                contig: rm4.contiguous_full(),
                aligned: true,
                level: Tgt::default_level(),
                layout: rm4,
                vector_size: None,
            };
            3
        ],
        true,
    )
}

#[export_name = "morello_bench_logicalspec_parameters::move_spec"]
fn move_spec<Tgt: Target>(size: DimSize) -> LogicalSpec<Tgt> {
    let rm2 = row_major(2);
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Move,
            spec_shape: smallvec![size; 2],
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
    )
}

#[inline(never)]
fn iter_logicalspec_parameters_matmul() {
    let sp = matmul_spec::<X86Target>(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

#[inline(never)]
fn iter_logicalspec_parameters_conv() {
    let sp = conv_spec::<X86Target>(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

#[inline(never)]
fn iter_logicalspec_parameters_move() {
    let sp = move_spec::<X86Target>(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

main!(
    callgrind_args = "toggle-collect=morello_bench_logicalspec_parameters::matmul_spec",
        "toggle-collect=morello_bench_logicalspec_parameters::conv_spec",
        "toggle-collect=morello_bench_logicalspec_parameters::move_spec",
        "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = iter_logicalspec_parameters_matmul, iter_logicalspec_parameters_conv,
        iter_logicalspec_parameters_move
);
