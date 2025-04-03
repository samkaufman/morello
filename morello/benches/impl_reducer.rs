use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use morello::cost::{CostIntensity, NormalizedCost};
use morello::db::ActionNum;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;
use morello::utils::bit_length_inverse;
use nonzero::nonzero as nz;
use std::hint::black_box;

#[export_name = "morello_bench_impl_reducer::init_reduce_costs"]
fn init_reduce_costs(k: u16) -> (Vec<(ActionNum, NormalizedCost)>, ImplReducer) {
    let reducer = ImplReducer::new(usize::from(k), vec![]);
    // Generate some "random" entries to reduce.
    let entries = (0..k + 10000)
        .map(|i| {
            (
                i % 10,
                NormalizedCost {
                    intensity: CostIntensity::new(((i + 11) % 13).into(), nz!(1u64)),
                    peaks: MemVec::new([
                        ((i + 2) % 5).into(),
                        ((i + 3) % 4).into(),
                        bit_length_inverse((i % 2).into()),
                        bit_length_inverse((i % 13).into()),
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
    for (action_num, cost) in entries {
        reducer.insert(black_box(action_num), black_box(cost));
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
