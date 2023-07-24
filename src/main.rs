use clap::Parser;
use log::info;
use smallvec::smallvec;
use std::io;
use std::path;
use std::sync::RwLock;

use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype, Spec};
use morello::layout::row_major;
use morello::pprint::pprint;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType};
use morello::table::{Database, DatabaseExt, InMemDatabase, SqliteDatabaseWrapper};
use morello::target::{Target, X86MemoryLevel, X86Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,
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
    match &args.db {
        Some(db_path) => main_per_db(
            &args,
            SqliteDatabaseWrapper::new(InMemDatabase::<X86Target>::new(), db_path),
        ),
        None => main_per_db(&args, InMemDatabase::<X86Target>::new()),
    };
}

fn main_per_db<D>(args: &Args, db: D)
where
    D: Database<X86Target> + Send + Sync,
{
    let rm2 = row_major(2);
    let matmul_spec = LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape: smallvec![args.size, args.size, args.size],
            dtype: Dtype::Uint32,
        },
        morello::spec::PrimitiveAux::Standard(vec![
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

    let spec = Spec(matmul_spec, X86Target::max_mem());

    let start_time = std::time::Instant::now();
    let db_lock = RwLock::new(db);
    let (_, hits, misses) = morello::search::top_down(&db_lock, &spec, 1);
    info!("top_down took {:?}", start_time.elapsed());
    info!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    let Some(results) = db_lock.read().unwrap().get_impl(&spec) else {
        panic!("No Impl found");
    };
    assert_eq!(results.len(), 1);
    pprint(&results[0]);
    results[0]
        .emit_kernel(&mut ToWriteFmt(io::stdout()))
        .unwrap();
}
