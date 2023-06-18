use std::sync::RwLock;

use crate::common::{DimSize, Dtype, Problem};
use crate::layout::row_major;
use crate::pprint::pprint;
use crate::spec::{PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::table::{DatabaseExt, InMemDatabase, SqliteDatabaseWrapper};
use crate::target::{Target, X86MemoryLevel, X86Target};
use crate::tensorspec::TensorSpecAux;

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
mod scheduling;
mod search;
mod spec;
mod table;
mod target;
mod tensorspec;
mod tiling;
mod utils;
mod views;

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

    let rm2 = row_major(2);
    let matmul_spec = Spec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: smallvec![args.size, args.size, args.size],
            dtype: Dtype::Uint32,
        },
        spec::PrimitiveAux::Standard(vec![
            TensorSpecAux {
                contig: rm2.contiguous_full(),
                aligned: true,
                level: X86MemoryLevel::GL,
                layout: rm2,
                vector_shape: None,
            };
            3
        ]),
        true,
    );

    // let rm = row_major(4);
    // let cnn_spec = Spec::Compose {
    //     components: vec![
    //         PrimitiveBasics {
    //             typ: PrimitiveSpecType::Conv { accum: false },
    //             spec_shape: smallvec![
    //                 args.batch,
    //                 args.filters,
    //                 args.filters,
    //                 args.size - args.filters_size + 1,
    //                 args.size - args.filters_size + 1,
    //                 args.filters_size,
    //                 args.filters_size
    //             ],
    //             dtype: Dtype::Uint32,
    //         },
    //         PrimitiveBasics {
    //             typ: PrimitiveSpecType::Conv { accum: false },
    //             spec_shape: smallvec![
    //                 args.batch,
    //                 args.filters,
    //                 args.channels,
    //                 args.size,
    //                 args.size,
    //                 args.filters_size,
    //                 args.filters_size
    //             ],
    //             dtype: Dtype::Uint32,
    //         },
    //     ],
    //     operand_auxes: vec![
    //         TensorSpecAux {
    //             contig: rm.contiguous_full(),
    //             aligned: true,
    //             level: X86MemoryLevel::GL,
    //             layout: rm,
    //             vector_shape: None,
    //         };
    //         4
    //     ],
    //     serial_only: true,
    // };

    let problem = Problem(matmul_spec, X86Target::max_mem());

    let start_time = std::time::Instant::now();
    let (_, hits, misses) = search::top_down(&db, &problem, 1);
    info!("top_down took {:?}", start_time.elapsed());
    info!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    let Some(results) = db.read().unwrap().get_impl(&problem) else {
        panic!("No Impl found");
    };
    assert_eq!(results.len(), 1);
    pprint(&results[0]);
}
