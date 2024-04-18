use iai_callgrind::{black_box, main};

use morello::layout::row_major;
use morello::lspec;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use morello::target::{Target, X86Target};
use morello::tensorspec::TensorSpecAux;

// TODO: Add a benchmark for Compose

#[export_name = "morello_bench_logicalspec_parameters::matmul_spec"]
fn matmul_spec<Tgt: Target>(size: u32) -> LogicalSpec<Tgt> {
    let rm2 = row_major(2);
    lspec!(Matmul(
        [size, size, size],
        (u32, Tgt::default_level(), rm2.clone()),
        (u32, Tgt::default_level(), rm2.clone()),
        (u32, Tgt::default_level(), rm2),
        serial
    ))
}

#[export_name = "morello_bench_logicalspec_parameters::conv_spec"]
fn conv_spec<Tgt: Target>(size: u32) -> LogicalSpec<Tgt> {
    let rm4 = row_major(4);
    lspec!(Conv(
        [size, size, size, size, size, size, size],
        (u32, Tgt::default_level(), rm4.clone()),
        (u32, Tgt::default_level(), rm4.clone()),
        (u32, Tgt::default_level(), rm4),
        serial
    ))
}

#[export_name = "morello_bench_logicalspec_parameters::move_spec"]
fn move_spec<Tgt: Target>(size: u32) -> LogicalSpec<Tgt> {
    let rm2 = row_major(2);
    lspec!(Move(
        [size, size],
        (u32, Tgt::default_level(), rm2.clone()),
        (u32, Tgt::default_level(), rm2),
        serial
    ))
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
