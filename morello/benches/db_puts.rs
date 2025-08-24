use divan::counter::ItemsCount;
use morello::db::FilesDatabase;
use morello::layout::row_major;
use morello::spec::Spec;
use morello::target::{Target, X86Target};
use morello::{cost::Cost, memorylimits::MemVec};
use std::hint::black_box;

const BATCHES: [u32; 4] = [1, 2, 4, 8];
const SIZES: [u32; 6] = [32, 64, 128, 256, 512, 1024];

fn mk_specs_set() -> Vec<Spec<X86Target>> {
    let mut specs = Vec::new();
    for &b in &BATCHES {
        for &m in &SIZES {
            for &k in &SIZES {
                for &n in &SIZES {
                    specs.push(morello::spec!(Matmul(
                        [b, m, k, n],
                        (u32, X86Target::default_level(), row_major),
                        (u32, X86Target::default_level(), row_major),
                        (u32, X86Target::default_level(), row_major),
                        serial
                    )));
                }
            }
        }
    }
    specs
}

#[divan::bench]
fn db_puts_overlap(bencher: divan::Bencher) {
    let db = FilesDatabase::new(None, true, 1, 4096, 1);
    let specs = mk_specs_set();
    let decisions: Vec<_> = (0..specs.len())
        .map(|i| {
            // Keep peaks very small to always satisfy spec memory limits.
            let peaks = MemVec::zero::<X86Target>();
            let cost = Cost {
                main: 10_000 * (i as u32 % 4),
                peaks,
                depth: 1,
            };
            (0u16, cost)
        })
        .collect();

    bencher
        .counter(ItemsCount::new(specs.len() as u64))
        .bench(|| {
            for (spec, (_, cost)) in specs.iter().zip(decisions.iter()) {
                let d = vec![(0u16, cost.clone())];
                db.put(black_box(spec.clone()), black_box(d));
            }
        });
}

fn main() {
    divan::main();
}
