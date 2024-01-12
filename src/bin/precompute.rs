#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use anyhow::Result;
use clap::Parser;
use log::{debug, info};
use nonzero::nonzero as nz;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use std::collections::VecDeque;
use std::time::Duration;
use std::{iter, path};

use morello::common::{DimSize, Dtype};
use morello::db::{DashmapDiskDatabase, Database};
use morello::grid::general::SurMap;
use morello::layout::row_major;
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::spec::{
    LogicalSpec, LogicalSpecSurMap, PrimitiveBasics, PrimitiveBasicsBimap, PrimitiveSpecType, Spec,
};
use morello::target::{CpuMemoryLevel, Target, X86Target};
use morello::tensorspec::{TensorSpecAux, TensorSpecAuxSurMap};
use morello::utils::bit_length;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const DB_SAVE_PERIOD: Duration = Duration::from_secs(10 * 60);
const K: u8 = 1;
const INITIAL_HASHMAP_CAPACITY: usize = 100_000_000;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, help = "Maximum number of stages to run.")]
    stages: Option<usize>,
    #[arg(long)]
    db: Option<path::PathBuf>,
    #[arg(long, default_value = "false")]
    include_conv: bool,
    #[arg(long, short, default_value = "1")]
    batch: DimSize,
    #[arg(long, default_value = "4")]
    channels: DimSize,
    #[arg(long, default_value = "8")]
    filters: DimSize,
    #[arg(long, default_value = "3")]
    filters_size: Vec<DimSize>,
    size: DimSize,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let db = DashmapDiskDatabase::try_with_capacity(
        args.db.as_deref(),
        true,
        K,
        INITIAL_HASHMAP_CAPACITY,
    )?;
    main_per_db(&args, &db);

    Ok(())
}

fn main_per_db<'a, D>(args: &Args, db: &'a D)
where
    D: Database<'a> + Send + Sync,
{
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    // TODO: Most of the following details aren't used in computing the bound.
    // It could be simplified.
    let mut bounds = vec![];
    let move_needed_rank = if args.include_conv { 4 } else { 2 };
    bounds.extend((1..=move_needed_rank).flat_map(|rank| [move_top(args.size, rank)]));
    bounds.extend((1..=move_needed_rank).map(|rank| {
        let layout = row_major(rank);
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Zero,
                spec_shape: smallvec![args.size; rank.into()],
                dtype: Dtype::Uint32,
            },
            vec![TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level: CpuMemoryLevel::GL,
                layout,
                vector_size: None,
            }],
            true,
        )
    }));
    bounds.push({
        let layout = row_major(2);
        let a = TensorSpecAux {
            contig: layout.contiguous_full(),
            aligned: true,
            level: CpuMemoryLevel::GL,
            layout,
            vector_size: None,
        };
        LogicalSpec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: smallvec![args.size, args.size, args.size],
                dtype: Dtype::Uint32,
            },
            vec![a.clone(), a.clone(), a],
            true,
        )
    });
    if args.include_conv {
        bounds.extend({
            let layout = row_major(4);
            let a = TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level: CpuMemoryLevel::GL,
                layout,
                vector_size: None,
            };
            args.filters_size
                .iter()
                .map(|&fs| {
                    LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Conv { accum: false },
                            spec_shape: smallvec![
                                args.batch,
                                args.filters,
                                args.channels,
                                args.size,
                                args.size,
                                fs,
                                fs
                            ],
                            dtype: Dtype::Uint32,
                        },
                        vec![a.clone(), a.clone(), a.clone()],
                        true,
                    )
                })
                .collect::<Vec<_>>()
        });
    }

    let mut rng = rand::thread_rng();
    let mut last_save_completion = std::time::Instant::now();
    let mut last_stage_results_saved = true;
    for (stage_idx, stage) in bounds.iter().flat_map(logical_specs_to_compute).enumerate() {
        info!(
            "Beginning stage {}, which has peak parallelism of {}",
            stage_idx,
            stage.len()
        );

        let nonempty_tasks = stage
            .into_iter()
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>();
        if let Some(example_spec) = nonempty_tasks
            .choose(&mut rng)
            .and_then(|v| v.choose(&mut rng))
        {
            info!("Example problem Spec: {}", example_spec);
        }

        let stage_start = std::time::Instant::now();
        nonempty_tasks.into_par_iter().for_each(|task| {
            let mut worklist = VecDeque::new();
            for logical_spec in task.iter() {
                worklist.push_back(top.clone());
                while let Some(job) = worklist.pop_front() {
                    let spec = Spec(logical_spec.clone(), MemoryLimits::Standard(job));
                    let result = morello::search::top_down(db, &spec, 1, Some(nz!(1usize)));
                    if let [(_, only_result_cost)] = &result.0[..] {
                        worklist.extend(next_limits(&spec.1, &only_result_cost.peaks));
                    }
                }
            }
        });
        db.compact();
        info!(
            "Stage (without saving) {} took {:?}",
            stage_idx,
            stage_start.elapsed()
        );
        info!("Database stats: {}", db.stats_str());

        last_stage_results_saved = false;
        if last_save_completion.elapsed() >= DB_SAVE_PERIOD {
            save_db(db);
            last_save_completion = std::time::Instant::now();
            last_stage_results_saved = true;
        }

        if Some(stage_idx) == args.stages {
            info!("Stopping early because --stages was passed");
            break;
        }
    }

    if !last_stage_results_saved {
        save_db(db);
    }
}

fn save_db<'a, D>(db: &'a D)
where
    D: Database<'a> + Send + Sync,
{
    let save_start = std::time::Instant::now();
    db.save().unwrap();
    info!("Saving took {:?}", save_start.elapsed());
}

/// Returns a logical Move Spec of given size and rank.
fn move_top(size: DimSize, rank: u8) -> LogicalSpec<X86Target> {
    let layout = row_major(rank);
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Move,
            spec_shape: smallvec![size; rank.into()],
            dtype: Dtype::Uint32,
        },
        [CpuMemoryLevel::GL, CpuMemoryLevel::L1]
            .into_iter()
            .map(|level| TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level,
                layout: layout.clone(),
                vector_size: None,
            })
            .collect(),
        true,
    )
}

fn next_limits<'a>(
    result_limits: &'a MemoryLimits,
    result_peak: &'a MemVec,
) -> impl Iterator<Item = MemVec> + 'a {
    let MemoryLimits::Standard(limits_vec) = result_limits;
    debug_assert!(limits_vec
        .iter()
        .zip(result_peak.iter())
        .all(|(l, p)| l >= p));
    (0..limits_vec.len()).filter_map(|idx| {
        let mut new_values = limits_vec.clone();
        if result_peak.get_unscaled(idx) == 0 {
            return None;
        }
        if result_peak.get_unscaled(idx) == 1 {
            new_values.set_unscaled(idx, 0);
        } else {
            new_values.set_unscaled(idx, 1 << (bit_length(result_peak.get_unscaled(idx)) - 2));
        }
        Some(new_values)
    })
}

/// Yield an [Iterator] over all [LogicalSpec]s to compute, in dependency order.
fn logical_specs_to_compute(
    bound_spec: &LogicalSpec<X86Target>,
) -> impl Iterator<Item = Vec<Vec<LogicalSpec<X86Target>>>> {
    let surmap = LogicalSpecSurMap::new(
        PrimitiveBasicsBimap {
            binary_scale_shapes: true,
        },
        TensorSpecAuxSurMap::new,
    );

    let (spec_key, bound_pt) = SurMap::apply(&surmap, bound_spec);
    debug!(
        "Grid shape is {:?}",
        bound_pt.iter().map(|d| d + 1).collect::<Vec<_>>()
    );
    let mut stage = 0u32;
    iter::from_fn(move || {
        let mut tasks = vec![];
        for pt in morello::utils::sum_seqs(&bound_pt, stage) {
            let mut task = vec![];
            // TODO: Factor out below key
            for sp in SurMap::apply_inverse(&surmap, &(spec_key.clone(), SmallVec::from_vec(pt))) {
                if sp.is_canonical() {
                    task.push(sp);
                }
            }
            tasks.push(task);
        }
        stage += 1;
        if tasks.is_empty() {
            None
        } else {
            Some(tasks)
        }
    })
}
