use iai_callgrind::main;

use std::hint::black_box;

use morello::cost::Cost;
use morello::db::ActionIdx;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;

#[export_name = "morello_bench_impl_reducer::init_reduce_costs"]
fn init_reduce_costs(k: u16) -> (Vec<(ActionIdx, Cost)>, ImplReducer) {
    let reducer = ImplReducer::new(usize::from(k), vec![]);
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

fn reduce_costs(k: u16) {
    let (entries, mut reducer) = black_box(init_reduce_costs(k));
    for (action_idx, cost) in entries {
        reducer.insert(black_box(action_idx), black_box(cost));
    }
}

#[inline(never)]
fn reduce_costs_1() {
    reduce_costs(1);
}

#[inline(never)]
fn reduce_costs_2() {
    reduce_costs(2);
}

#[inline(never)]
fn reduce_costs_8() {
    reduce_costs(8);
}

#[inline(never)]
fn reduce_costs_100() {
    reduce_costs(100);
}

main!(
    callgrind_args = "toggle-collect=morello_bench_impl_reducer::init_reduce_costs",
        "--simulate-wb=no", "--simulate-hwpref=yes",
        "--I1=32768,8,64", "--D1=32768,8,64", "--LL=8388608,16,64";
    functions = reduce_costs_1, reduce_costs_2, reduce_costs_8, reduce_costs_100
);
