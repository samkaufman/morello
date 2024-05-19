use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use std::hint::black_box;

use morello::{layout, shape};

#[library_benchmark]
fn update_for_tiling() {
    let shape = shape![64, 64, 64];
    let tile_shape = shape![64, 8, 8];
    let layout = layout![
        (0, PhysDim::Dynamic),
        (1, PhysDim::Dynamic),
        (2, PhysDim::Dynamic),
        (1, PhysDim::Packed(8))
    ];
    let c = layout.contiguous_full();
    black_box(layout.update_for_tiling(&shape, &tile_shape, c)).unwrap();
}

library_benchmark_group!(
    name = update_for_tiling_group;
    benchmarks = update_for_tiling
);

main!(
    config = LibraryBenchmarkConfig::default()
                .raw_callgrind_args([
                    "--simulate-wb=no", "--simulate-hwpref=yes",
                    "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64",
                ]);
    library_benchmark_groups = update_for_tiling_group
);
