use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::info;
use smallvec::smallvec;
use std::sync::RwLock;
use std::{io, path};

use morello::codegen::CodeGen;
use morello::color::{self, ColorMode};
use morello::common::{DimSize, Dtype};
use morello::layout::row_major;
use morello::layout::Layout;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::table::{Database, DatabaseExt, InMemDatabase, SqliteDatabaseWrapper};
use morello::target::{ArmTarget, CpuMemoryLevel, Target, TargetId, X86Target};
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

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::C)]
    format: OutputFormat,

    /// Impl style
    #[arg(long, value_enum, default_value_t = ImplPrintStyle::Full)]
    impl_style: ImplPrintStyle,

    /// Compile and run the synthesized implementation
    #[arg(short, long)]
    run: bool,

    /// Target architecture
    #[arg(long, value_enum, default_value_t = TargetId::X86)]
    target: TargetId,

    #[command(subcommand)]
    query_spec: QuerySpec,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    C,
    Impl,
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

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    color::set_color_mode(args.color);
    match &args.target {
        TargetId::X86 => {
            let db = InMemDatabase::<X86Target>::new();
            match &args.db {
                Some(db_path) => main_per_db(&args, SqliteDatabaseWrapper::new(db, db_path)),
                None => main_per_db(&args, db),
            }
        }
        TargetId::Arm => {
            let db = InMemDatabase::<ArmTarget>::new();
            match &args.db {
                Some(db_path) => main_per_db(&args, SqliteDatabaseWrapper::new(db, db_path)),
                None => main_per_db(&args, db),
            }
        }
    }
}

fn main_per_db<D, Tgt>(args: &Args, db: D) -> Result<()>
where
    D: Database<Tgt> + Send + Sync,
    Tgt: Target<Level = CpuMemoryLevel>,
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
                vec![
                    TensorSpecAux {
                        contig: rm2.contiguous_full(),
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: rm2,
                        vector_size: None,
                    },
                    TensorSpecAux {
                        contig: cm2.contiguous_full(),
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: cm2,
                        vector_size: None,
                    },
                ],
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
                vec![
                    TensorSpecAux {
                        contig: rm2.contiguous_full(),
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: rm2,
                        vector_size: None,
                    };
                    3
                ],
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
                vec![
                    TensorSpecAux {
                        contig: rm.contiguous_full(),
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: rm,
                        vector_size: None,
                    };
                    3
                ],
                true,
            )
        }
    };

    let spec = Spec(query_spec, Tgt::max_mem());

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
    let [synthesized_impl] = &results[..] else {
        unreachable!();
    };

    match args.format {
        OutputFormat::C => {
            synthesized_impl.emit(&mut ToWriteFmt(io::stdout()))?;
        }
        OutputFormat::Impl => {
            // TODO: How to use Compact? Should we?
            pprint(synthesized_impl, args.impl_style)
        }
    }
    if args.run {
        let output = synthesized_impl.build()?.run()?;
        println!("\nOutput:\n{}", String::from_utf8_lossy(&output.stdout));
    }

    Ok(())
}
