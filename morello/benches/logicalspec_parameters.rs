use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use std::hint::black_box;

use morello::layout::row_major;
use morello::lspec;
use morello::spec::LogicalSpec;
use morello::target::{Target, X86Target};

// TODO: Add a benchmark for Compose

#[export_name = "morello_bench_logicalspec_parameters::matmul_spec"]
fn matmul_spec(size: u32) -> LogicalSpec<X86Target> {
    lspec!(Matmul(
        [1, size, size, size],
        (u32, X86Target::default_level(), row_major),
        (u32, X86Target::default_level(), row_major),
        (u32, X86Target::default_level(), row_major),
        serial
    ))
}

#[export_name = "morello_bench_logicalspec_parameters::conv_spec"]
fn conv_spec(size: u32) -> LogicalSpec<X86Target> {
    lspec!(Conv(
        [size, size, size, 1, 1, size, size],
        (u32, X86Target::default_level(), row_major),
        (u32, X86Target::default_level(), row_major),
        (u32, X86Target::default_level(), row_major),
        serial
    ))
}

#[export_name = "morello_bench_logicalspec_parameters::move_spec"]
fn move_spec(size: u32) -> LogicalSpec<X86Target> {
    lspec!(Move(
        [size, size],
        (u32, X86Target::default_level(), row_major),
        (u32, X86Target::default_level(), row_major),
        serial
    ))
}

#[library_benchmark]
fn iter_logicalspec_parameters_matmul() {
    let sp = matmul_spec(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

#[library_benchmark]
fn iter_logicalspec_parameters_conv() {
    let sp = conv_spec(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

#[library_benchmark]
fn iter_logicalspec_parameters_move() {
    let sp = move_spec(32);
    for _ in 0..100 {
        for p in sp.parameters() {
            black_box(p);
        }
    }
}

library_benchmark_group!(
    name = logicalspec_parameters_group;
    benchmarks =
        iter_logicalspec_parameters_matmul,
        iter_logicalspec_parameters_conv,
        iter_logicalspec_parameters_move
);

main!(
    config = LibraryBenchmarkConfig::default()
                .raw_callgrind_args([
                    "toggle-collect=morello_bench_logicalspec_parameters::matmul_spec",
                    "toggle-collect=morello_bench_logicalspec_parameters::conv_spec",
                    "toggle-collect=morello_bench_logicalspec_parameters::move_spec",
                    "--simulate-wb=no", "--simulate-hwpref=yes",
                    "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64",
                ]);
    library_benchmark_groups = logicalspec_parameters_group
);
