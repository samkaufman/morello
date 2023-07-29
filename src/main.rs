use clap::Parser;
use log::info;
use smallvec::smallvec;
use std::io;
use std::path;
use std::sync::RwLock;

use morello::codegen::CodeGen;
use morello::color::{self, ColorMode};
use morello::common::{DimSize, Dtype, Spec};
use morello::layout::row_major;
use morello::layout::Layout;
use morello::pprint::{pprint, PrintMode};
use morello::spec::{LogicalSpec, PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
use morello::table::{Database, DatabaseExt, InMemDatabase, SqliteDatabaseWrapper};
use morello::target::{Target, X86MemoryLevel, X86Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,

    /// Color mode
    #[arg(long, value_enum, default_value_t = ColorMode::Auto)]
    color: ColorMode,

    /// Print mode
    #[arg(long, value_enum, default_value_t = PrintMode::Full)]
    print: PrintMode,

    #[command(subcommand)]
    query_spec: QuerySpec,
}

#[derive(clap::Subcommand)]
enum QuerySpec {
    #[command(about = "Synthesize a row- to column-major transpose")]
    Transpose { size: DimSize },
    #[command(about = "Synthesize a matrix multiplication")]
    Matmul { size: DimSize },
    #[command(about = "Synthesize a convolution")]
    Conv {
        #[arg(long, short, default_value = "1")]
        batch: DimSize,
        #[arg(long, default_value = "4")]
        channels: DimSize,
        #[arg(long, default_value = "8")]
        filters: DimSize,
        #[arg(long, default_value = "3")]
        filters_size: DimSize,
        size: DimSize,
    },
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    color::set_color_mode(args.color);
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
    let query_spec = match &args.query_spec {
        QuerySpec::Transpose { size } => {
            let rm2 = row_major(2);
            let cm2 = Layout::Standard {
                dim_order: smallvec![1, 0],
            };
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Move,
                    spec_shape: smallvec![*size, *size],
                    dtype: Dtype::Uint32,
                },
                PrimitiveAux::Move {
                    outer_aux: TensorSpecAux {
                        contig: rm2.contiguous_full(),
                        aligned: true,
                        level: X86MemoryLevel::GL,
                        layout: rm2,
                        vector_shape: None,
                    },
                    inner_level: X86MemoryLevel::GL,
                    inner_layout: cm2,
                    inner_vector_shape: None,
                },
                true,
            )
        }
        QuerySpec::Matmul { size } => {
            let rm2 = row_major(2);
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: false },
                    spec_shape: smallvec![*size, *size, *size],
                    dtype: Dtype::Uint32,
                },
                PrimitiveAux::Standard(vec![
                    TensorSpecAux {
                        contig: rm2.contiguous_full(),
                        aligned: true,
                        level: X86MemoryLevel::GL,
                        layout: rm2,
                        vector_size: None,
                    };
                    3
                ]),
                true,
            )
        }
        QuerySpec::Conv {
            batch,
            channels,
            filters,
            filters_size,
            size,
        } => {
            let rm = row_major(4);
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Conv { accum: false },
                    spec_shape: smallvec![
                        *batch,
                        *filters,
                        *filters,
                        *size - *filters_size + 1,
                        *size - *filters_size + 1,
                        *filters_size,
                        *filters_size,
                    ],
                    dtype: Dtype::Uint32,
                },
                PrimitiveAux::Standard(vec![
                    TensorSpecAux {
                        contig: rm.contiguous_full(),
                        aligned: true,
                        level: X86MemoryLevel::GL,
                        layout: rm,
                        vector_size: None,
                    };
                    3
                ]),
                true,
            )
        }
    };

    let spec = Spec(query_spec, X86Target::max_mem());

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
    pprint(&results[0], args.print);
    println!();
    results[0]
        .emit_kernel(&mut ToWriteFmt(io::stdout()))
        .unwrap();
}
