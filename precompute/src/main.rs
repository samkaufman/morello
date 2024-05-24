#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use anyhow::Result;
use clap::Parser;
use log::{debug, info};
use nonzero::nonzero as nz;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use std::collections::HashSet;
use std::{iter, path};

use morello::common::{DimSize, Dtype};
use morello::db::FilesDatabase;
use morello::grid::general::SurMap;
use morello::layout::row_major;
use morello::lspec;
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::search::top_down_many;
use morello::spec::{
    LogicalSpec, LogicalSpecSurMap, PrimitiveBasics, PrimitiveBasicsBimap, PrimitiveSpecType, Spec,
};
use morello::target::{
    CpuMemoryLevel::{self, GL},
    Target, X86Target,
};
use morello::tensorspec::{TensorSpecAux, TensorSpecAuxSurMap};
use morello::utils::bit_length;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const K: u8 = 1;

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
    let db = FilesDatabase::new(args.db.as_deref(), true, K);
    main_per_db(&args, &db);

    Ok(())
}

fn main_per_db(args: &Args, db: &FilesDatabase) {
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    // TODO: Most of the following details aren't used in computing the bound.
    // It could be simplified.
    let mut bounds = vec![];
    let move_needed_rank = if args.include_conv { 4 } else { 2 };
    bounds.extend((1..=move_needed_rank).flat_map(|rank| [move_top(args.size, rank)]));
    bounds.extend((1..=move_needed_rank).map(|rank| {
        lspec!(Zero(
            iter::repeat(args.size).take(rank.into()),
            (u32, GL, row_major(rank)),
            serial
        ))
    }));
    bounds.push(lspec!(Matmul(
        [args.size, args.size, args.size],
        (u32, CpuMemoryLevel::GL, row_major(2)),
        (u32, CpuMemoryLevel::GL, row_major(2)),
        (u32, CpuMemoryLevel::GL, row_major(2)),
        serial
    )));
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
                            spec_shape: vec![
                                args.batch,
                                args.filters,
                                args.channels,
                                args.size,
                                args.size,
                                fs,
                                fs,
                            ],
                            dtypes: vec![Dtype::Uint32; 3],
                        },
                        vec![a.clone(), a.clone(), a.clone()],
                        true,
                    )
                })
                .collect::<Vec<_>>()
        });
    }

    let mut rng = rand::thread_rng();
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
            let mut stage = task
                .into_iter()
                .map(|t| Spec(t, MemoryLimits::Standard(top.clone())))
                .collect::<Vec<_>>();
            let mut next_stage = HashSet::new();

            while !stage.is_empty() {
                let stage_results = top_down_many(db, &stage, 1, Some(nz!(1usize))).0;
                for (spec, result) in stage.iter().zip(stage_results) {
                    if let [(_, only_result_cost)] = &result[..] {
                        next_stage.extend(
                            next_limits(&spec.1, &only_result_cost.peaks)
                                .map(|l| Spec(spec.0.clone(), MemoryLimits::Standard(l))),
                        );
                    }
                }
                // TODO: Just swap data structures.
                stage = next_stage.drain().collect();
            }
        });
        info!(
            "Stage (without saving) {} took {:?}",
            stage_idx,
            stage_start.elapsed()
        );
        info!("Database stats: {}", db.stats_str());

        if Some(stage_idx) == args.stages {
            info!("Stopping early because --stages was passed");
            break;
        }
    }
}

/// Returns a logical Move Spec of given size and rank.
fn move_top(size: DimSize, rank: u8) -> LogicalSpec<X86Target> {
    lspec!(Move(
        iter::repeat(size).take(rank.into()),
        (u32, CpuMemoryLevel::GL, row_major(rank)),
        (u32, CpuMemoryLevel::L1, row_major(rank)),
        serial
    ))
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
            for sp in SurMap::apply_inverse(&surmap, &(spec_key.clone(), pt)) {
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
