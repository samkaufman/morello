use crate::common::{Dtype, Problem};
use crate::pprint::pprint;
use crate::target::{Target, X86MemoryLevel, X86Target};
use crate::tensorspec::TensorSpec;
use smallvec::smallvec;

mod alignment;
mod common;
mod cost;
mod imp;
mod layout;
mod memorylimits;
mod pprint;
mod scheduling;
mod search;
mod spec;
mod table;
mod target;
mod tensorspec;
mod tiling;
mod utils;

fn main() {
    env_logger::init();

    let mut db = table::Database::<X86Target>::new();
    let rm = layout::row_major(2);

    let load_spec = spec::Spec::Load::<X86Target> {
        outer_tensor_spec: TensorSpec::new(
            smallvec![128, 128],
            Dtype::Uint32,
            rm.contiguous_full(),
            true,
            X86MemoryLevel::GL,
            rm.clone(),
            None,
        ),
        inner_level: X86MemoryLevel::L1,
        inner_layout: rm.clone(),
        inner_vector_shape: None,
        serial_only: true,
    };

    let matmul_spec = spec::Spec::Matmul::<X86Target> {
        accum: false,
        m: 512,
        k: 512,
        n: 512,
        dtype: Dtype::Uint32,
        contiguous_abstractions: smallvec![
            rm.contiguous_full(),
            rm.contiguous_full(),
            rm.contiguous_full()
        ],
        alignments: smallvec![true, true, true],
        levels: smallvec![X86MemoryLevel::GL, X86MemoryLevel::GL, X86MemoryLevel::GL],
        layouts: smallvec![rm.clone(), rm.clone(), rm.clone()],
        vector_shapes: smallvec![None, None, None],
        serial_only: true,
    };

    let problem = Problem(matmul_spec, X86Target::max_mem());

    let start_time = std::time::Instant::now();
    let (_, hits, misses) = search::top_down(&mut db, &problem, 1);
    println!("top_down took {:?}", start_time.elapsed());
    println!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    pprint(&db, &problem);
}
