#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::info;

use std::{io, path};

use morello::codegen::{BuildError, CodeGen};
use morello::color::{self, ColorMode};
use morello::common::{DimSize, Dtype};
use morello::db::FilesDatabase;
use morello::grid::canon::CanonicalBimap;
use morello::grid::general::BiMap;
use morello::layout::{col_major, row_major};
use morello::pprint::{pprint, ImplPrintStyle};
use morello::smallvec::smallvec;
use morello::target::{
    ArmTarget, Avx2Target, Avx512Target, CpuMemoryLevel::GL, CpuTarget, Target, TargetId,
};
use morello::tensorspec::TensorSpecAux;
use morello::utils::ToWriteFmt;
use morello::{
    lspec,
    spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec},
};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const BINARY_SCALE_SHAPES: bool = true;
const K: u8 = 1;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,

    #[arg(long, default_value = "128", help = "Cache size in database pages.")]
    cache_size: usize,

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
    #[arg(long, value_enum, hide_default_value = true, default_value_t = TargetId::default())]
    target: TargetId,

    /// Disable verification
    #[cfg(feature = "verification")]
    #[arg(long, default_value_t = false)]
    skip_check: bool,

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
    Transpose {
        size: DimSize,
    },
    #[command(about = "Synthesize a matrix multiplication")]
    Matmul {
        size: DimSize,
    },
    MatmulU8S8S16 {
        size: DimSize,
    },
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
    #[command(about = "Synthesize a softmax")]
    Softmax {
        #[arg(long, default_value = "2")]
        rank: u8,
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
    inner_loop_iters: Option<u32>,

    #[command(subcommand)]
    query_spec: QuerySpec,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    color::set_color_mode(args.color);
    let threads = rayon::current_num_threads();
    let db = FilesDatabase::new(
        args.db.as_deref(),
        BINARY_SCALE_SHAPES,
        K,
        args.cache_size,
        threads,
    );
    match &args.target {
        TargetId::Avx2 => main_per_db::<Avx2Target>(&args, &db),
        TargetId::Avx512 => main_per_db::<Avx512Target>(&args, &db),
        TargetId::Arm => main_per_db::<ArmTarget>(&args, &db),
    }
}

fn main_per_db<Tgt>(args: &Args, db: &FilesDatabase) -> Result<()>
where
    Tgt: CpuTarget,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    let subcmd = &args.subcmd;
    let query_spec = match subcmd {
        Subcommand::Emit(query_spec) => query_spec,
        Subcommand::Run(run_cmd) => &run_cmd.query_spec,
        Subcommand::Bench(bench_cmd) => &bench_cmd.query_spec,
    };
    let logical_spec = match query_spec {
        QuerySpec::Transpose { size } => {
            lspec!(Move(
                [*size, *size],
                (u32, GL, row_major),
                (u32, GL, col_major),
                serial
            ))
        }
        QuerySpec::Matmul { size } | QuerySpec::MatmulU8S8S16 { size } => {
            let [dt_a, dt_b, dt_c] = match query_spec {
                QuerySpec::Matmul { .. } => [Dtype::Uint32; 3],
                QuerySpec::MatmulU8S8S16 { .. } => [Dtype::Uint8, Dtype::Sint8, Dtype::Sint16],
                _ => unreachable!(),
            };
            lspec!(Matmul(
                [DimSize::new(1).unwrap(), *size, *size, *size],
                (dt_a, GL, row_major),
                (dt_b, GL, row_major),
                (dt_c, GL, row_major),
                serial
            ))
        }
        QuerySpec::Conv {
            batch,
            channels,
            filters,
            filters_size,
            size,
        } => {
            lspec!(Conv(
                [
                    *batch,
                    *filters,
                    *channels,
                    *size,
                    *size,
                    *filters_size,
                    *filters_size
                ],
                (u32, GL, row_major),
                (u32, GL, row_major),
                (u32, GL, row_major),
                serial
            ))
        }
        QuerySpec::Softmax { rank, size } => {
            let layouts = [row_major(*rank), row_major(*rank)];
            LogicalSpec::Primitive(
                PrimitiveBasics {
                    typ: PrimitiveSpecType::Softmax { scan_dim: rank - 1 },
                    spec_shape: smallvec![*size; usize::from(*rank)],
                    dtypes: vec![Dtype::Uint32; usize::from(*rank)],
                },
                layouts
                    .into_iter()
                    .map(|layout| {
                        debug_assert!(layout.is_fully_contiguous()); // row_major should be contig.
                        TensorSpecAux::<Tgt> {
                            level: GL.into(),
                            layout,
                            vector_size: None,
                        }
                    })
                    .collect(),
                true,
            )
        }
    };

    let mut spec = Spec(logical_spec, Tgt::max_mem());
    spec.canonicalize().unwrap();
    info!("Synthesizing {spec}");

    let start_time = std::time::Instant::now();

    morello::search::top_down(db, &spec, K.into());
    info!("top_down took {:?}", start_time.elapsed());

    let Some(results) = db.get_impl(&spec) else {
        panic!("No Impl found");
    };
    let Some(synthesized_impl) = results.first() else {
        panic!("No Impl found");
    };

    let bench_inner_loop_iters = if let Subcommand::Bench(BenchCmd {
        inner_loop_iters, ..
    }) = subcmd
    {
        // We need an exact number of samples when benchmarking.
        match *inner_loop_iters {
            // The user specified a number of samples.
            Some(s) => Some(s),
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
            synthesized_impl.emit(true, impl_style, &mut ToWriteFmt(io::stdout()))?;
        }
        OutputFormat::Impl => pprint(synthesized_impl, args.impl_style),
    }
    if let Subcommand::Emit(_) = subcmd {
        return Ok(());
    }

    let built_artifact = match synthesized_impl.build(false) {
        Ok(a) => a,
        Err(e) => {
            if let BuildError::CompilerFailed { ref stderr, .. } = e {
                eprintln!("Compiler stderr:\n\n{}", &stderr);
            }
            return Err(e.into());
        }
    };
    let output = built_artifact.run()?;
    if let Subcommand::Run(_) = subcmd {
        println!("\nOutput:\n{}", String::from_utf8_lossy(&output.stdout));
    }
    #[cfg(feature = "verification")]
    if !args.skip_check && !built_artifact.check_correctness(&spec) {
        panic!("Generated code returned incorrect output");
    }

    if let Subcommand::Bench(_) = subcmd {
        let result = synthesized_impl.bench(bench_inner_loop_iters.unwrap(), None)?;
        let inner_loop_runtime = result.best_inner_loop_runtime();
        let kernel_runtime = inner_loop_runtime / result.inner_loop_iterations;
        println!("\nkernel runtime: {:.8}s", kernel_runtime.as_secs_f32());
        println!("loop runtime: {}ns", inner_loop_runtime.as_nanos());
    }
    Ok(())
}
