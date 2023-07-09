use std::collections::VecDeque;
use std::sync::RwLock;
use std::{iter, path};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use smallvec::smallvec;
use spec::{PrimitiveAux, PrimitiveBasics, PrimitiveSpecType};
use tensorspec::TensorSpecAux;

mod alignment;
mod common;
mod cost;
mod expr;
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

use crate::common::{DimSize, Dtype, Problem};
use crate::geometry::ToFromDependencyLatticeCoordinate;
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::spec::Spec;
use crate::table::{Database, InMemDatabase, SqliteDatabaseWrapper};
use crate::target::{Target, X86MemoryLevel, X86Target};
use crate::utils::iter_powers_of_two;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    stages: Option<usize>,
    #[arg(long)]
    timeout: Option<u64>,
    #[arg(long)]
    db: Option<path::PathBuf>,
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

    if let Some(db_path) = args.db.as_ref() {
        let db = SqliteDatabaseWrapper::new(InMemDatabase::<X86Target>::new(), db_path);
        main_per_db(&args, db)
    } else {
        let db = InMemDatabase::<X86Target>::new();
        main_per_db(&args, db)
    }
}

fn load_or_store(prim_type: PrimitiveSpecType, size: DimSize, rank: u8) -> Spec<X86Target> {
    let layout = layout::row_major(rank);
    Spec::Primitive(
        PrimitiveBasics {
            typ: prim_type,
            spec_shape: smallvec![size; rank.into()],
            dtype: Dtype::Uint32,
        },
        PrimitiveAux::Move {
            outer_aux: TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level: X86MemoryLevel::GL,
                layout: layout.clone(),
                vector_shape: None,
            },
            inner_level: X86MemoryLevel::L1,
            inner_layout: layout,
            inner_vector_shape: None,
        },
        true,
    )
}

fn main_per_db<D>(args: &Args, db: D)
where
    D: Database<X86Target> + Send + Sync,
{
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    let db = RwLock::new(db);

    // TODO: Most of the following details aren't used in computing the bound.
    // It could be simplified.
    let mut bounds = vec![];
    bounds.extend((1..5).flat_map(|rank| {
        [
            load_or_store(PrimitiveSpecType::Load, args.size, rank),
            load_or_store(PrimitiveSpecType::Store, args.size, rank),
        ]
    }));
    bounds.extend((1..5).map(|rank| {
        let layout = layout::row_major(rank);
        Spec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Zero,
                spec_shape: smallvec![args.size; rank.into()],
                dtype: Dtype::Uint32,
            },
            PrimitiveAux::Standard(vec![TensorSpecAux {
                contig: layout.contiguous_full(),
                aligned: true,
                level: X86MemoryLevel::GL,
                layout,
                vector_shape: None,
            }]),
            true,
        )
    }));
    bounds.push({
        let layout = layout::row_major(2);
        let a = TensorSpecAux {
            contig: layout.contiguous_full(),
            aligned: true,
            level: X86MemoryLevel::GL,
            layout,
            vector_shape: None,
        };
        Spec::Primitive(
            PrimitiveBasics {
                typ: PrimitiveSpecType::Matmul { accum: false },
                spec_shape: smallvec![args.size, args.size, args.size],
                dtype: Dtype::Uint32,
            },
            PrimitiveAux::Standard(vec![a.clone(), a.clone(), a]),
            true,
        )
    });
    bounds.extend({
        let layout = layout::row_major(4);
        let a = TensorSpecAux {
            contig: layout.contiguous_full(),
            aligned: true,
            level: X86MemoryLevel::GL,
            layout,
            vector_shape: None,
        };
        args.filters_size
            .iter()
            .map(|&fs| {
                Spec::Primitive(
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
                    PrimitiveAux::Standard(vec![a.clone(), a.clone(), a.clone()]),
                    true,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut rng = rand::thread_rng();
    for (stage_idx, stage) in bounds.iter().flat_map(specs_to_compute).enumerate() {
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
                let mut limits_iterator = problems_for_spec::<X86Target>();
                debug_assert!(worklist.is_empty());
                worklist.push_back(top.clone());
                while let Some(job) = worklist.pop_front() {
                    let problem = Problem(spec.clone(), MemoryLimits::Standard(job));
                    let MemoryLimits::Standard(job) = &problem.1;
                    let result = search::top_down(&db, &problem, 1);
                    if let [(_, only_result_cost)] = &result.0[..] {
                        worklist.extend(limits_iterator.next_vec(job, &only_result_cost.peaks));
                    }
                }
            }
            db.write().unwrap().flush();
        });
        info!("Stage {} took {:?}", stage_idx, stage_start.elapsed());
        if Some(stage_idx) == args.stages {
            info!("Stopping early because --stages was passed");
            break;
        }
    }
}

fn problems_for_spec<T: Target>() -> impl Iterator<Item = MemVec> {
    let MemoryLimits::Standard(top) = T::max_mem();
    top.into_iter()
        .map(|l| iter_powers_of_two(l, true).rev())
        .multi_cartesian_product()
        .map(move |prod| MemVec::new(prod.into_iter().collect()))
}

trait MemoryLimitsIterator {
    fn next_vec(&mut self, last_limits: &MemVec, last_peak: &MemVec) -> Vec<MemVec>;
}

struct SkippingLimitsIterator {}

impl<T: Iterator<Item = MemVec>> MemoryLimitsIterator for T {
    fn next_vec(&mut self, _last_limits: &MemVec, _last_peak: &MemVec) -> Vec<MemVec> {
        self.next().into_iter().collect()
    }
}

impl MemoryLimitsIterator for SkippingLimitsIterator {
    fn next_vec(&mut self, last_limits: &MemVec, last_peak: &MemVec) -> Vec<MemVec> {
        let mut new_limits = Vec::new();

        assert!(last_limits.iter().zip(last_peak).all(|(l, p)| *l >= p));

        for idx in 0..last_limits.len() {
            let mut new_values = last_limits.clone();
            if last_peak[idx] == 0 {
                continue;
            }
            if last_peak[idx] == 1 {
                new_values[idx] = 0;
            } else {
                new_values[idx] = 2u64.pow(last_peak[idx].leading_zeros() - 1);
            }
            new_limits.push(new_values);
        }

        new_limits
    }
}

fn specs_to_compute(bound: &Spec<X86Target>) -> impl Iterator<Item = Vec<Vec<Spec<X86Target>>>> {
    let grid_map_result = bound.to_grid();
    if grid_map_result.is_none() {
        panic!("Could not map {:?} to grid", bound);
    }
    let (spec_key, bound_pt) = grid_map_result.unwrap();
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
        for pt in utils::sum_seqs(&bound_pt, stage) {
            let mut task = vec![];
            for sp in Spec::<X86Target>::objects_in_grid_pt(&spec_key, &pt) {
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
