use clap::Parser;
use log::{error, info};
use morello::codegen::CodeGen;
use morello::common::{DimSize, Dtype};
use morello::cost::Cost;
use morello::db::FilesDatabase;
use morello::imp::subspecs::SpecApp;
use morello::imp::ImplNode;
use morello::layout::row_major;
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::scheduling_sugar::SchedulingSugar;
use morello::shape;
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{
    Avx2Target,
    CpuMemoryLevel::{self, GL, L1, VRF},
};
use morello::target::{MemoryLevel, Target};
use morello::tensorspec::TensorSpecAux;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroU32;
use std::{path, vec};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, short)]
    db: Option<path::PathBuf>,
    #[arg(long, default_value_t = 1000, help = "Cache size in database pages.")]
    cache_size: usize,
    #[arg(long, short, help = "Output CSV file path for resume functionality.")]
    output: Option<path::PathBuf>,
}

const BENCHMARK_ITER_COEFF: u64 = 32_768_000_000;
// We enumerate multiple problem sizes instead of a single TOTAL_SIZE.
const TOTAL_SIZES: &[u32] = &[32, 64, 128, 256];
const TOTAL_K_DIMS: &[u32] = &[32, 64, 128, 256];
const INITIAL_TILE_SIZE_BITLEN: u32 = 0;
const INITIAL_SPLIT_SIZE_BITLEN: u32 = 0;
const BUFFER_LEVELS: &[CpuMemoryLevel] = &[VRF, L1, GL];
const RF_LIMIT: u64 = 0;
const DTYPE: Dtype = Dtype::Float32;
const CSV_HEADER: &str = "total_size,total_k_outer,total_k_inner,split_size,tile_size,buffer_level,cost,benchmark_iters,kernel_runtime,throughput,gflops_per_sec";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExperimentKey {
    total_size: u32,
    total_k_outer: u32,
    total_k_inner: u32,
    split_size: u32,
    tile_size: u32,
    buffer_level: String,
}

/// Load completed experiments from the output CSV file if it exists.
/// Returns a HashSet of experiment keys.
fn load_completed_experiments(output_path: Option<&path::Path>) -> HashSet<ExperimentKey> {
    let mut completed = HashSet::new();

    let output_path = match output_path {
        Some(p) => p,
        None => return completed,
    };

    let file = match File::open(output_path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Output file not found, starting fresh");
            return completed;
        }
    };

    let reader = BufReader::new(file);
    for (line_num, line) in reader.lines().enumerate() {
        if line_num == 0 {
            continue; // Skip header
        }

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading line {}: {}", line_num, e);
                continue;
            }
        };

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        // Parse the experiment key fields
        let total_size = parts[0].parse::<u32>().ok();
        let total_k_outer = parts[1].parse::<u32>().ok();
        let total_k_inner = parts[2].parse::<u32>().ok();
        let split_size = parts[3].parse::<u32>().ok();
        let tile_size = parts[4].parse::<u32>().ok();
        let buffer_level = parts[5].to_string();

        if let (Some(ts), Some(tko), Some(tki), Some(ss), Some(tls)) = (
            total_size,
            total_k_outer,
            total_k_inner,
            split_size,
            tile_size,
        ) {
            completed.insert(ExperimentKey {
                total_size: ts,
                total_k_outer: tko,
                total_k_inner: tki,
                split_size: ss,
                tile_size: tls,
                buffer_level,
            });
        }
    }

    completed
}

fn main() {
    env_logger::init();
    let args = Args::parse();

    // Load existing experiments from output CSV if it exists
    let completed_experiments = load_completed_experiments(args.output.as_deref());
    let num_completed = completed_experiments.len();
    let file_exists = num_completed > 0;

    if file_exists {
        eprintln!("Resuming: found {} completed experiments", num_completed);
    }

    // Open output file for appending, or create if it doesn't exist
    let mut output_file = if let Some(output_path) = args.output.as_deref() {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)
            .expect("Failed to open output file");

        // Write header if file is new
        if !file_exists {
            writeln!(file, "{}", CSV_HEADER).expect("Failed to write CSV header");
        }

        Some(file)
    } else {
        // Print CSV header to stdout if no output file specified
        println!("{}", CSV_HEADER);
        None
    };

    let db = FilesDatabase::new::<Avx2Target>(args.db.as_deref(), true, 1, args.cache_size, 1);

    for &total_size in TOTAL_SIZES {
        for &total_k_outer in TOTAL_K_DIMS {
            for &total_k_inner in TOTAL_K_DIMS {
                let total_size_nz: DimSize = NonZeroU32::new(total_size).unwrap();
                let total_k_outer_nz: DimSize = NonZeroU32::new(total_k_outer).unwrap();
                let total_k_inner_nz: DimSize = NonZeroU32::new(total_k_inner).unwrap();

                let inner_basics = PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: false },
                    spec_shape: shape![1, total_size_nz, total_k_inner_nz, total_k_outer_nz],
                    dtypes: vec![DTYPE; 3],
                };
                let outer_basics = PrimitiveBasics {
                    typ: PrimitiveSpecType::Matmul { accum: true },
                    spec_shape: shape![1, total_size_nz, total_k_outer_nz, total_size_nz],
                    dtypes: vec![DTYPE; 3],
                };
                let one_nz = NonZeroU32::new(1).unwrap();
                let aux = TensorSpecAux {
                    level: GL,
                    layout: row_major(&[one_nz, total_size_nz, total_size_nz]),
                    vector_size: None,
                };

                let mut spec = Spec::<Avx2Target>(
                    LogicalSpec::Compose {
                        components: vec![outer_basics, inner_basics],
                        operand_auxes: vec![aux.clone(), aux.clone(), aux.clone(), aux],
                        serial_only: true,
                    },
                    // No RF and GL shrunk to fit in L2
                    MemoryLimits::Standard(MemVec::new_for_target::<Avx2Target>([
                        RF_LIMIT, 16, 1024, 524288,
                    ])),
                );
                spec.canonicalize().unwrap();

                let split_sizes: Vec<u32> = (INITIAL_SPLIT_SIZE_BITLEN..u32::BITS)
                    .map(|k| 1u32 << k)
                    .take_while(|&s| s <= total_k_outer)
                    .collect();
                let tile_sizes: Vec<u32> = (INITIAL_TILE_SIZE_BITLEN..u32::BITS)
                    .map(|k| 1u32 << k)
                    .take_while(|&s| s <= total_size)
                    .collect();
                for &split_size in &split_sizes {
                    for &tile_size in &tile_sizes {
                        for buffer_level in BUFFER_LEVELS {
                            // Skip if this experiment was already completed
                            let buffer_level_name = format!("{:?}", buffer_level);
                            let experiment_key = ExperimentKey {
                                total_size,
                                total_k_outer,
                                total_k_inner,
                                split_size,
                                tile_size,
                                buffer_level: buffer_level_name.clone(),
                            };
                            if completed_experiments.contains(&experiment_key) {
                                info!("Skipping completed experiment: total_size={}, total_k_outer={}, total_k_inner={}, split_size={}, tile_size={}, buffer_level={:?}", 
                                    total_size, total_k_outer, total_k_inner, split_size, tile_size, buffer_level);
                                continue;
                            }

                            let vector_size = buffer_level
                                .vector_bytes()
                                .iter()
                                .map(|vb| NonZeroU32::new(*vb / DTYPE.size() as u32).unwrap())
                                .max();

                            // if let Some(vector_size) = vector_size {
                            //     let MemoryLimits::Standard(mem_vec) = &spec.1;
                            //     let vrf_idx = Avx2Target::levels()
                            //         .iter()
                            //         .position(|l| l == &CpuMemoryLevel::VRF)
                            //         .unwrap();
                            //     let max_vrf = mem_vec.get_unscaled(vrf_idx) as u32;
                            //     let tile_volume = tile_size * split_size;
                            //     if tile_volume.div_ceil(vector_size.get()) + 1 > max_vrf {
                            //         info!("Skipping: VRF too small for vectorization: tile_size={}, split_size={}, vector_size={}, max_vrf={}",
                            //             tile_size, split_size, vector_size.get(), max_vrf);
                            //         continue;
                            //     }
                            // }

                            let mut imp: ImplNode<_> =
                                SpecApp::new_with_default_params(spec.clone()).into();
                            // skip if bufferize would fail
                            if total_size != tile_size {
                                imp = imp.tile_out(&[1, tile_size, total_size_nz.get()]);
                            }
                            if total_k_outer > split_size {
                                imp = imp.split(split_size);
                            }
                            if total_size != tile_size {
                                imp = imp.tile_out(&[1, tile_size, tile_size]);
                            }

                            imp = match imp.checked_bufferize(
                                0,
                                *buffer_level,
                                row_major,
                                vector_size,
                            ) {
                                Some(i) => i,
                                None => {
                                    log::warn!("Bufferize not applicable: total_size={}, split_size={}, tile_size={}, buffer_level={:?}", total_size, split_size, tile_size, buffer_level);
                                    continue;
                                }
                            };
                            imp = match imp.checked_synthesize_all(&db) {
                                Some(s) => s,
                                None => {
                                    log::warn!("Failed to synthesize: total_size={}, split_size={}, tile_size={}, buffer_level={:?}", total_size, split_size, tile_size, buffer_level);
                                    continue;
                                }
                            };

                            // Benchmark.
                            // Total FLOPs: inner matmul + outer matmul
                            // = 2*total_size*total_k_inner*total_k_outer + 2*total_size*total_k_outer*total_size
                            let total_flops_u64 = 2u64
                                * total_size_nz.get() as u64
                                * total_k_outer_nz.get() as u64
                                * (total_k_inner_nz.get() as u64 + total_size_nz.get() as u64);
                            let benchmark_iters = BENCHMARK_ITER_COEFF
                                .div_ceil(total_flops_u64)
                                .try_into()
                                .unwrap();
                            let result = match imp.bench(benchmark_iters, None) {
                                Ok(r) => r,
                                Err(e) => {
                                    error!(
                                        "Benchmark failed: total_size={}, total_k_outer={}, total_k_inner={}, split_size={}, tile_size={}, buffer_level={:?}, error={}",
                                        total_size, total_k_outer,
                                        total_k_inner, split_size, tile_size, buffer_level, e,
                                    );
                                    continue;
                                }
                            };
                            // Log all individual inner loop runtimes (seconds) instead of only the best.
                            let inner_loop_runtimes_secs: Vec<f64> = result
                                .inner_loop_runtimes
                                .iter()
                                .map(|d| d.as_secs_f64())
                                .collect();
                            let total_flops = 2.0
                                * total_size_nz.get() as f64
                                * total_k_outer_nz.get() as f64
                                * (total_k_inner_nz.get() as f64 + total_size_nz.get() as f64);

                            let cost = Cost::from_impl(&imp).main;
                            // Emit one row per measured inner loop runtime.
                            for runtime in &result.inner_loop_runtimes {
                                let kernel_runtime_sample =
                                    (*runtime / result.inner_loop_iterations).as_secs_f64();
                                let throughput_sample =
                                    result.inner_loop_iterations as f64 / runtime.as_secs_f64();
                                let gflops_per_sec_sample =
                                    (total_flops / kernel_runtime_sample) / 1e9;

                                let output_line = format!(
                                    "{},{},{},{},{},{},{},{},{:.12},{:.2},{:.2}",
                                    total_size,
                                    total_k_outer,
                                    total_k_inner,
                                    split_size,
                                    tile_size,
                                    buffer_level_name,
                                    cost,
                                    benchmark_iters,
                                    kernel_runtime_sample,
                                    throughput_sample,
                                    gflops_per_sec_sample
                                );

                                if let Some(ref mut file) = output_file {
                                    writeln!(file, "{}", output_line)
                                        .expect("Failed to write to output file");
                                } else {
                                    println!("{}", output_line);
                                }
                            }
                            info!(
                                "Completed: total_size={}, total_k_outer={}, total_k_inner={}, split_size={}, tile_size={}, buffer_level={:?}, cost={}, inner_loop_runtimes_secs={:?}, inner_loop_iterations={}",
                                total_size,
                                total_k_outer,
                                total_k_inner,
                                split_size,
                                tile_size,
                                buffer_level,
                                cost,
                                inner_loop_runtimes_secs,
                                result.inner_loop_iterations
                            );
                        }
                    }
                }
            }
        }
    }
}
