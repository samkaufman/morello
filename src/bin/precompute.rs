use clap::Parser;
use log::{debug, info};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use smallvec::smallvec;
use std::collections::VecDeque;
use std::{iter, path};

use morello::common::{DimSize, Dtype};
use morello::datadeps::ToFromDependencyLatticeCoordinate;
use morello::db::{DashmapDiskDatabase, Database};
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::spec::{LogicalSpec, PrimitiveBasics, PrimitiveSpecType, Spec};
use morello::target::{CpuMemoryLevel, Target, X86Target};
use morello::tensorspec::TensorSpecAux;

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

fn main() {
    env_logger::init();
    let args = Args::parse();
    let db = DashmapDiskDatabase::<X86Target>::new(args.db.as_deref());
    main_per_db(&args, &db)
}

fn main_per_db<'a, D>(args: &Args, db: &'a D)
where
    D: Database<'a, X86Target> + Send + Sync,
{
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    // TODO: Most of the following details aren't used in computing the bound.
    // It could be simplified.
    let mut bounds = vec![];
    bounds.extend((1..5).flat_map(|rank| [move_top(args.size, rank)]));
    bounds.extend((1..5).map(|rank| {
        let layout = morello::layout::row_major(rank);
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
        let layout = morello::layout::row_major(2);
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
            let layout = morello::layout::row_major(4);
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
    for (stage_idx, stage) in bounds.iter().flat_map(logical_specs_to_compute).enumerate() {
        info!(
            "Beginning stage {}, which has peak parallelism of {}",
            stage_idx,
            stage.len()
        );

        let nonempty_tasks = stage.iter().filter(|v| !v.is_empty()).collect::<Vec<_>>();
        if let Some(example_spec) = nonempty_tasks
            .choose(&mut rng)
            .and_then(|v| v.choose(&mut rng))
        {
            info!("Example problem Spec: {}", example_spec);
        }

        let stage_start = std::time::Instant::now();
        let stage_iter = nonempty_tasks.into_par_iter().with_min_len(4);
        stage_iter.for_each(|task| {
            let mut worklist = VecDeque::new();
            // for (_task_idx, spec) in chunk.iter().flat_map(|e| *e).enumerate() {
            for (_task_idx, spec) in task.iter().enumerate() {
                let mut limits_iterator = make_memory_limit_iter::<X86Target>();
                debug_assert!(worklist.is_empty());
                worklist.push_back(top.clone());
                while let Some(job) = worklist.pop_front() {
                    let spec = Spec(spec.clone(), MemoryLimits::Standard(job));
                    let result = morello::search::top_down(db, &spec, 1);
                    if let [(_, only_result_cost)] = &result.0[..] {
                        worklist.extend(limits_iterator.next().into_iter().collect::<Vec<_>>());
                    }
                }
            }
        });
        db.save();
        info!("Stage {} took {:?}", stage_idx, stage_start.elapsed());
        if Some(stage_idx) == args.stages {
            info!("Stopping early because --stages was passed");
            break;
        }
    }
}

/// Returns a logical Move Spec of given size and rank.
fn move_top(size: DimSize, rank: u8) -> LogicalSpec<X86Target> {
    let layout = morello::layout::row_major(rank);
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

/// Returns an [Iterator] over power-of-two [MemVec]s from maximum memory down.
fn make_memory_limit_iter<T: Target>() -> impl Iterator<Item = MemVec> {
    // TOOD: Skip ahead rather than recursing for every limit.
    let MemoryLimits::Standard(top) = T::max_mem();
    top.iter_down_by_powers_of_two::<T>()
}

/// Yield an [Iterator] over all [LogicalSpec]s to compute, in dependency order.
fn logical_specs_to_compute(
    bound: &LogicalSpec<X86Target>,
) -> impl Iterator<Item = Vec<Vec<LogicalSpec<X86Target>>>> {
    let Some((spec_key, bound_pt)) = bound.to_grid() else {
        panic!("Could not map {:?} to grid", bound);
    };
    // TODO: Reintroduce a check like the following.
    // debug_assert_eq!(
    //     bound,
    //     &Spec::<X86Target>::from_grid(&spec_key, &bound_pt, &inner_key)
    // );
    debug!(
        "Grid shape is {:?}",
        bound_pt.iter().map(|d| d + 1).collect::<Vec<_>>()
    );
    let mut stage = 0u32;
    iter::from_fn(move || {
        let mut tasks = vec![];
        for pt in morello::utils::sum_seqs(&bound_pt, stage) {
            let mut task = vec![];
            for sp in LogicalSpec::<X86Target>::objects_in_grid_pt(&spec_key, &pt) {
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
