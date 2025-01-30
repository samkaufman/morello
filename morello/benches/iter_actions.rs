use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use std::hint::black_box;

use morello::layout::row_major;
use morello::lspec;
use morello::spec::LogicalSpec;
use morello::target::{CpuMemoryLevel::GL, Target, X86Target};

#[library_benchmark]
fn copy_actions_into_vec() {
    let logical_spec: LogicalSpec<X86Target> = lspec!(Matmul(
        [64, 64, 64],
        (u32, GL, row_major),
        (u32, GL, row_major),
        (u32, GL, row_major),
        serial
    ));
    black_box(X86Target::actions(&logical_spec, None).collect::<Vec<_>>());
}

library_benchmark_group!(
    name = iter_actions_group;
    benchmarks = copy_actions_into_vec
);

main!(
    config = LibraryBenchmarkConfig::default()
                .raw_callgrind_args([
                    "--simulate-wb=no", "--simulate-hwpref=yes",
                    "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64",
                ]);
    library_benchmark_groups = iter_actions_group
);
