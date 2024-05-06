use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use smallvec::smallvec;

use morello::cost::Cost;
use morello::db::ActionIdx;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;

#[inline(never)]
fn impl_reducer(top_k: usize, actions: &[(ActionIdx, Cost)]) {
    let mut reducer = ImplReducer::new(top_k, smallvec![]);
    actions
        .iter()
        .for_each(|(action_idx, cost)| reducer.insert(*action_idx, cost.clone()));
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ImplReducer");
    for top_k in [1, 2, 8, 100] {
        let mut rng = rand::thread_rng();
        let mut actions: Vec<ActionIdx> = (0..top_k + 10000).collect();
        actions.shuffle(&mut rng);
        let actions: Vec<_> = actions
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
            .collect();

        group.bench_with_input(
            BenchmarkId::new("impl_reducer", top_k),
            &(usize::from(top_k), &actions[..]),
            |b, (k, actions)| b.iter(|| impl_reducer(black_box(*k), black_box(actions))),
        );
    }
    group.finish();
}

// TODO: Convert benchmark to use iai_callgrind.
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
