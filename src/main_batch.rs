use std::collections::VecDeque;
use std::sync::RwLock;
use std::{iter, path};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info};
use rayon::prelude::*;
use smallvec::smallvec;

mod alignment;
mod common;
mod cost;
mod geometry;
mod imp;
mod layout;
mod memorylimits;
mod pprint;
mod search;
mod spec;
mod table;
mod target;
mod tensorspec;
mod tiling;
mod utils;

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
    m: DimSize,
    k: DimSize,
    n: DimSize,
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

fn main_per_db<D>(args: &Args, db: D)
where
    D: Database<X86Target> + Send + Sync,
{
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    let db = RwLock::new(db);
    for (stage_idx, stage) in specs_to_compute_2(args).enumerate() {
        info!(
            "Beginning stage {}, which has peak parallelism of {}",
            stage_idx,
            stage.len()
        );
        let stage_start = std::time::Instant::now();
        let stage_iter = stage.into_par_iter();
        stage_iter.for_each(|task| {
            for (_task_idx, spec) in task.into_iter().enumerate() {
                let mut limits_iterator = problems_for_spec::<X86Target>();
                let mut worklist = VecDeque::from([top.clone()]);
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

fn specs_to_compute_2(args: &Args) -> impl Iterator<Item = Vec<Vec<spec::Spec<X86Target>>>> {
    // Most of the following details aren't used in computing the bound. It
    // could be simplified.
    let rm = layout::row_major(2);
    let bound = {
        spec::Spec::Matmul::<X86Target> {
            accum: false,
            m: args.m,
            k: args.k,
            n: args.n,
            dtype: Dtype::Uint32,
            contiguous_abstractions: smallvec![
                rm.contiguous_full(),
                rm.contiguous_full(),
                rm.contiguous_full()
            ],
            alignments: smallvec![true, true, true],
            levels: smallvec![X86MemoryLevel::GL, X86MemoryLevel::GL, X86MemoryLevel::GL],
            layouts: smallvec![rm.clone(), rm.clone(), rm],
            vector_shapes: smallvec![None, None, None],
            serial_only: true,
        }
    };
    let grid_map_result = bound.to_grid();
    if grid_map_result.is_none() {
        panic!("Could not map {:?} to grid", bound);
    }
    let (spec_key, bound_pt, _) = grid_map_result.unwrap();
    debug!(
        "Grid shape is {:?}",
        bound_pt.iter().map(|d| d + 1).collect::<Vec<_>>()
    );
    let mut stage = 0u32;
    iter::from_fn(move || {
        let mut tasks = vec![];
        for pt in utils::sum_seqs(&bound_pt, stage) {
            let mut task = vec![];
            let inner_keys = Spec::<X86Target>::inner_keys_for_grid_pt(&spec_key, &pt);
            for s in inner_keys {
                let sp = Spec::<X86Target>::from_grid(&spec_key, &pt, &s);
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
