use std::sync::RwLock;

use crate::common::{DimSize, Dtype, Problem};
use crate::pprint::pprint;
use crate::spec::SpecAux;
use crate::table::{InMemDatabase, SqliteDatabaseWrapper};
use crate::target::{Target, X86MemoryLevel, X86Target};

use clap::Parser;
use log::info;
use smallvec::smallvec;

mod alignment;
mod common;
mod cost;
mod geometry;
mod imp;
mod layout;
mod memorylimits;
mod pprint;
mod reinterpret;
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
    #[arg(long, short, default_value = "1")]
    batch: DimSize,
    #[arg(long, default_value = "4")]
    channels: DimSize,
    #[arg(long, default_value = "8")]
    filters: DimSize,
    #[arg(long, default_value = "3")]
    filters_size: DimSize,
    size: DimSize,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let db = RwLock::new(SqliteDatabaseWrapper::new(
        InMemDatabase::<X86Target>::new(),
        std::path::Path::new("db.sqlite3"),
    ));

    let rm = layout::row_major(4);
    let a = SpecAux {
        contig: rm.contiguous_full(),
        aligned: true,
        level: X86MemoryLevel::GL,
        layout: rm,
        vector_shape: None,
    };
    let conv_spec = spec::Spec::Conv {
        accum: false,
        image_shape: smallvec![args.batch, args.channels, args.size, args.size],
        filters_shape: smallvec![
            args.filters,
            args.channels,
            args.filters_size,
            args.filters_size
        ],
        dtype: Dtype::Uint32,
        aux: [a.clone(), a.clone(), a],
        serial_only: true,
    };

    let problem = Problem(conv_spec, X86Target::max_mem());

    let start_time = std::time::Instant::now();
    let (_, hits, misses) = search::top_down(&db, &problem, 1);
    info!("top_down took {:?}", start_time.elapsed());
    info!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    pprint(&*db.read().unwrap(), &problem);
}
