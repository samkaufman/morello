use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;
use std::hint::black_box;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

use morello::cost::Cost;
use morello::db::ActionIdx;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;

#[inline(never)]
fn impl_reducer(top_k: u16) {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut reducer = ImplReducer::new(usize::from(top_k), smallvec![]);
    let mut actions: Vec<ActionIdx> = (0..top_k + 10000).collect();
    actions.shuffle(&mut rng);
    actions
        .into_iter()
        .map(|action_idx| {
            (
                action_idx,
                Cost {
                    main: rng.gen(),
                    peaks: MemVec::new([rng.gen(), rng.gen(), rng.gen(), rng.gen()]),
                    depth: rng.gen(),
                },
            )
        })
        .for_each(|(action_idx, cost)| reducer.insert(action_idx, cost));
}

fn impl_reducer_benchmarks() -> impl IntoBenchmarks {
    [
        benchmark_fn("impl_reducer: 1", |b| b.iter(|| impl_reducer(black_box(1)))),
        benchmark_fn("impl_reducer: 2", |b| b.iter(|| impl_reducer(black_box(2)))),
        benchmark_fn("impl_reducer: 8", |b| b.iter(|| impl_reducer(black_box(8)))),
        benchmark_fn("impl_reducer: 100", |b| {
            b.iter(|| impl_reducer(black_box(100)))
        }),
    ]
}

// TODO: Convert benchmark to use iai_callgrind.
tango_benchmarks!(impl_reducer_benchmarks());
tango_main!();
