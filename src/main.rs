use crate::common::{Dtype, Problem};
use crate::pprint::pprint;
use crate::table::{NullDatabaseIOStore, SqliteIOStore};
use crate::target::{Target, X86MemoryLevel, X86Target};

use clap::Parser;
use common::DimSize;
use log::info;
use smallvec::smallvec;

mod alignment;
mod common;
mod cost;
mod imp;
mod layout;
mod memorylimits;
mod pprint;
mod search;
mod spec;
mod table;
mod target;
mod tensorspec;
mod tiling;
mod utils;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    m: DimSize,
    k: DimSize,
    n: DimSize,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let mut db = table::Database::<X86Target, _>::new(SqliteIOStore::new(std::path::Path::new(
        "db.sqlite3",
    )));
    let rm = layout::row_major(2);

    let matmul_spec = spec::Spec::Matmul::<X86Target> {
        accum: false,
        m: args.m,
        k: args.k,
        n: args.n,
        dtype: Dtype::Uint32,
        contiguous_abstractions: smallvec![
            rm.contiguous_full(),
            rm.contiguous_full(),
            rm.contiguous_full()
        ],
        alignments: smallvec![true, true, true],
        levels: smallvec![X86MemoryLevel::GL, X86MemoryLevel::GL, X86MemoryLevel::GL],
        layouts: smallvec![rm.clone(), rm.clone(), rm],
        vector_shapes: smallvec![None, None, None],
        serial_only: true,
    };

    let problem = Problem(matmul_spec, X86Target::max_mem());

    let start_time = std::time::Instant::now();
    let (_, hits, misses) = search::top_down(&mut db, &problem, 1);
    info!("top_down took {:?}", start_time.elapsed());
    info!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    pprint(&mut db, &problem);
}
