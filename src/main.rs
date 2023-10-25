use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::info;
use smallvec::smallvec;
use std::{io, path};

use morello::codegen::CodeGen;
use morello::color::{self, ColorMode};
use morello::common::{DimSize, Dtype};
use morello::db::{DashmapDiskDatabase, Database, DatabaseExt};
use morello::layout::row_major;
use morello::layout::Layout;
use morello::pprint::{pprint, ImplPrintStyle};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{ArmTarget, CpuMemoryLevel, Target, TargetId, X86Target};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;

const BINARY_SCALE_SHAPES: bool = true;
const K: u8 = 1;

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

    // Include Impl as a comment in generated C
    #[arg(long, default_value_t = false)]
    include_impl: bool,

    /// Impl style
    #[arg(long, value_enum, default_value_t = ImplPrintStyle::Compact)]
    impl_style: ImplPrintStyle,

    /// Target architecture
    #[arg(long, value_enum, default_value_t = TargetId::X86)]
    target: TargetId,

    #[command(subcommand)]
    subcmd: Subcommand,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    C,
    Impl,
}

#[derive(Parser)]
enum Subcommand {
    #[command(flatten)]
    Emit(QuerySpec),

    /// Compile and run the synthesized implementation
    Run(RunCmd),

    /// Compile and benchmark the synthesized implementation
    Bench(BenchCmd),
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

#[derive(Parser)]
struct RunCmd {
    #[command(subcommand)]
    query_spec: QuerySpec,
}

#[derive(Parser)]
struct BenchCmd {
    /// Number of benchmark samples
    #[arg(long, short)]
    bench_samples: Option<u32>,

    #[command(subcommand)]
    query_spec: QuerySpec,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    color::set_color_mode(args.color);
    match &args.target {
        TargetId::X86 => main_per_db::<_, X86Target>(
            &args,
            &DashmapDiskDatabase::new(args.db.as_deref(), BINARY_SCALE_SHAPES, K),
        ),
        TargetId::Arm => main_per_db::<_, ArmTarget>(
            &args,
            &DashmapDiskDatabase::new(args.db.as_deref(), BINARY_SCALE_SHAPES, K),
        ),
    }
}

fn main_per_db<'d, D, Tgt>(args: &Args, db: &'d D) -> Result<()>
where
    D: Database<'d> + Send + Sync,
    Tgt: Target<Level = CpuMemoryLevel>,
{
    let subcmd = &args.subcmd;
    let query_spec = match subcmd {
        Subcommand::Emit(query_spec) => query_spec,
        Subcommand::Run(run_cmd) => &run_cmd.query_spec,
        Subcommand::Bench(bench_cmd) => &bench_cmd.query_spec,
    };
    let logical_spec = match query_spec {
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
                    TensorSpecAux::<Tgt> {
                        contig: rm2.contiguous_full(),
                        aligned: true,
                        level: CpuMemoryLevel::GL,
                        layout: rm2,
                        vector_size: None,
                    },
                    TensorSpecAux::<Tgt> {
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
                        *channels,
                        *size,
                        *size,
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

    let spec = Spec(logical_spec, Tgt::max_mem());

    let start_time = std::time::Instant::now();
    let (_, hits, misses) = morello::search::top_down(db, &spec, K.into());
    info!("top_down took {:?}", start_time.elapsed());
    info!(
        "top_down missed {} times ({:.2}% of {})",
        misses,
        misses as f32 / (hits + misses) as f32,
        hits + misses
    );

    let Some(results) = db.get_impl(&spec) else {
        panic!("No Impl found");
    };
    let Some(synthesized_impl) = results.first() else {
        panic!("No Impl found");
    };

    let bench_samples = if let Subcommand::Bench(BenchCmd { bench_samples, .. }) = subcmd {
        // We need an exact number of samples when benchmarking.
        match *bench_samples {
            // The user specified a number of samples.
            Some(bench_samples) => Some(bench_samples),
            // The user didn't specify a number of samples, so we estimate
            // a good number of samples.
            None => Some(synthesized_impl.estimate_optimal_iters()?),
        }
    } else {
        None
    };

    match args.format {
        OutputFormat::C => {
            let impl_style = if args.include_impl {
                Some(args.impl_style)
            } else {
                None
            };
            synthesized_impl.emit(bench_samples, impl_style, &mut ToWriteFmt(io::stdout()))?;
        }
        OutputFormat::Impl => pprint(synthesized_impl, args.impl_style),
    }

    match subcmd {
        Subcommand::Run(_) => {
            let built_artifact = synthesized_impl.build(None)?;
            let output = built_artifact.run()?;
            println!("\nOutput:\n{}", String::from_utf8_lossy(&output.stdout));
            #[cfg(feature = "verification")]
            if !built_artifact.check_correctness(&spec) {
                panic!("Generated code returned incorrect output");
            }
        }
        Subcommand::Bench(_) => {
            // TODO: Test correctness (allow disabling with flag)
            let result = synthesized_impl.bench(bench_samples.unwrap(), None)?;
            println!("\nImpl Runtime: {:.4}s", result.result.as_secs_f32());
        }
        _ => {}
    }

    Ok(())
}
