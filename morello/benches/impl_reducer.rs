use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use smallvec::smallvec;
use std::hint::black_box;

use morello::cost::Cost;
use morello::db::ActionIdx;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;

#[export_name = "morello_bench_impl_reducer::init_reduce_costs"]
fn init_reduce_costs(k: u16) -> (Vec<(ActionIdx, Cost)>, ImplReducer) {
    let reducer = ImplReducer::new(usize::from(k), smallvec![]);
    // Generate some "random" entries to reduce.
    let entries = (0..k + 10000)
        .map(|i| {
            (
                i % 10,
                Cost {
                    main: ((i + 11) % 13).into(),
                    peaks: MemVec::new_from_binary_scaled([
                        ((i + 2) % 5).try_into().unwrap(),
                        ((i + 3) % 4).try_into().unwrap(),
                        (i % 2).try_into().unwrap(),
                        (i % 13).try_into().unwrap(),
                    ]),
                    depth: ((i + 1) % 3 + 1).try_into().unwrap(),
                },
            )
        })
        .collect::<Vec<_>>();
    (entries, reducer)
}

#[library_benchmark]
#[benches::multiple(1, 2, 8, 100)]
fn reduce_costs(k: u16) {
    let (entries, mut reducer) = black_box(init_reduce_costs(black_box(k)));
    for (action_idx, cost) in entries {
        reducer.insert(black_box(action_idx), black_box(cost));
    }
}

library_benchmark_group!(
    name = impl_reducer_group;
    benchmarks = reduce_costs
);

main!(
    config = LibraryBenchmarkConfig::default()
                .raw_callgrind_args([
                    "toggle-collect=morello_bench_impl_reducer::init_reduce_costs",
                    "--simulate-wb=no", "--simulate-hwpref=yes",
                    "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64",
                ]);
    library_benchmark_groups = impl_reducer_group
);
