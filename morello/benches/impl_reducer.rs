use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::seq::SliceRandom;
use rand::thread_rng;
use smallvec::smallvec;

use morello::cost::Cost;
use morello::db::ActionIdx;
use morello::memorylimits::MemVec;
use morello::search::ImplReducer;
use morello::target::X86Target;

#[inline(never)]
fn impl_reducer(top_k: usize, action_indices: Vec<ActionIdx>) {
    let mut reducer = ImplReducer::new(top_k, smallvec![]);
    action_indices.into_iter().for_each(|action_idx| {
        reducer.insert(
            action_idx,
            Cost {
                main: 1,
                peaks: MemVec::zero::<X86Target>(),
                depth: 0,
            },
        )
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ImplReducer");
    for top_k in [1, 2, 8, 100] {
        let mut action_indices: Vec<ActionIdx> = (0..top_k + 10000).collect();
        action_indices.shuffle(&mut thread_rng());
        group.bench_with_input(
            BenchmarkId::new("impl_reducer", top_k),
            &(top_k, action_indices),
            |b, (k, actions)| b.iter(|| impl_reducer((*k).into(), actions.clone())),
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
