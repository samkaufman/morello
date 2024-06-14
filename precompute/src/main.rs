#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use adler::Adler32;
use anyhow::Result;
use clap::Parser;
use log::info;
use nonzero::nonzero as nz;
use rayon::prelude::*;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::iter::Peekable;
use std::time::Instant;
use std::{fs, iter, path};

#[cfg(feature = "db-stats")]
use {
    std::sync::atomic::{self, AtomicU64},
    std::time::Duration,
};

use morello::common::{DimSize, Dtype};
use morello::datadeps::SpecKey;
use morello::db::FilesDatabase;
use morello::grid::general::SurMap;
use morello::grid::linear::BimapInt;
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
use morello::utils::{bit_length, SumSeqs};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

type JobFingerprint = (usize, u64);

const META_FILENAME: &str = "PRECOMPUTE";
const DB_PROGRESS_VERSION: usize = 1;
const K: u8 = 1;

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, help = "Maximum number of stages to run.")]
    stages: Option<usize>,
    #[arg(long)]
    db: Option<path::PathBuf>,
    #[arg(long, default_value = "32", help = "Cache size in database pages.")]
    cache_size: usize,
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

struct LogicalSpecGoals<S: SurMap> {
    logical_spec_surmap: S,
    bound: ((SpecKey, Vec<()>), Vec<BimapInt>), // TODO: Factor out the key
    stage: u32,
    done: bool,
}

struct TaskDiagonal<S> {
    logical_spec_surmap: S,
    spec_key: (SpecKey, Vec<()>), // TODO: Take reference
    sum_seqs_iter: SumSeqs,
}

struct TaskIter<S: SurMap>(S::DomainIter);

impl<S> LogicalSpecGoals<S>
where
    S: SurMap<Domain = LogicalSpec<X86Target>, Codomain = ((SpecKey, Vec<()>), Vec<BimapInt>)>,
{
    fn new(surmap: S, bound_spec: &LogicalSpec<X86Target>) -> Self {
        let bound = surmap.apply(bound_spec);
        Self {
            logical_spec_surmap: surmap,
            bound,
            stage: 0,
            done: false,
        }
    }
}

impl<S> Iterator for LogicalSpecGoals<S>
where
    S: SurMap<Domain = LogicalSpec<X86Target>, Codomain = ((SpecKey, Vec<()>), Vec<BimapInt>)>,
    S: Clone,
{
    type Item = Peekable<TaskDiagonal<S>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let (spec_key, bound_pt) = &self.bound;
        let mut diagonal_iter = TaskDiagonal {
            logical_spec_surmap: self.logical_spec_surmap.clone(),
            spec_key: spec_key.clone(),
            sum_seqs_iter: morello::utils::sum_seqs(bound_pt, self.stage),
        }
        .peekable();

        self.stage += 1;
        if diagonal_iter.peek().is_none() {
            self.done = true;
            return None;
        }
        Some(diagonal_iter)
    }
}

impl<S> TaskIter<S>
where
    S: SurMap,
{
    fn new(surmap: &S, co: &S::Codomain) -> Self {
        let inner = surmap.apply_inverse(co);
        Self(inner)
    }
}

impl<S> Iterator for TaskDiagonal<S>
where
    S: SurMap<Domain = LogicalSpec<X86Target>, Codomain = ((SpecKey, Vec<()>), Vec<BimapInt>)>,
{
    type Item = TaskIter<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let pt = self.sum_seqs_iter.next()?;
        Some(TaskIter::new(
            &self.logical_spec_surmap,
            &(self.spec_key.clone(), pt),
        ))
    }
}

impl<S, Ck> Iterator for TaskIter<S>
where
    S: SurMap<Domain = LogicalSpec<X86Target>, Codomain = (Ck, Vec<BimapInt>)>,
    Ck: Clone,
{
    type Item = LogicalSpec<X86Target>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let sp = self.0.next()?;
            if sp.is_canonical() {
                return Some(sp);
            }
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let threads = rayon::current_num_threads();

    #[cfg(feature = "db-stats")]
    log::info!("DB statistic collection enabled");

    let db = FilesDatabase::new(args.db.as_deref(), true, K, args.cache_size, threads);
    main_per_db(&args, db, args.db.as_deref());

    Ok(())
}

fn main_per_db(
    args: &Args,
    #[allow(unused_mut)] mut db: FilesDatabase, // mut when db-stats enabled
    db_path: Option<&path::Path>,
) {
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

    let fingerprint: JobFingerprint = {
        let mut progress_fingerprint_hasher = Adler32::new();
        bounds.hash(&mut progress_fingerprint_hasher);
        (DB_PROGRESS_VERSION, progress_fingerprint_hasher.finish())
    };
    let stages_completed = read_stages_to_skip(&fingerprint, db_path);
    if stages_completed != 0 {
        info!("First {} stages already computed", stages_completed);
    }
    if args.stages.map(|s| s <= stages_completed) == Some(true) {
        return;
    }

    let surmap = LogicalSpecSurMap::new(
        PrimitiveBasicsBimap {
            binary_scale_shapes: true,
        },
        TensorSpecAuxSurMap::new,
    );

    for (stage_idx, stage) in bounds
        .iter()
        .flat_map(|bound_spec| LogicalSpecGoals::new(surmap.clone(), bound_spec))
        .enumerate()
        .skip(stages_completed)
    {
        let mut stage = stage
            .filter_map(|task| {
                let mut p = task.peekable();
                if p.peek().is_some() {
                    Some(p)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        info!(
            "Beginning stage {}, which has peak parallelism of {}",
            stage_idx,
            stage.len()
        );

        if let Some(example_spec) = stage.first_mut().and_then(|v| v.peek()) {
            info!("Example problem Spec: {}", example_spec);
        }

        #[cfg(feature = "db-stats")]
        let total_synthesis_ms = AtomicU64::new(0);

        let stage_start = Instant::now();
        stage.into_par_iter().for_each(|task| {
            let mut stage = task
                .into_iter()
                .map(|t| Spec(t, MemoryLimits::Standard(top.clone())))
                .collect::<Vec<_>>();
            let mut next_stage = HashSet::new();

            #[cfg(feature = "db-stats")]
            let mut synthesis_time = Duration::ZERO;
            while !stage.is_empty() {
                #[cfg(feature = "db-stats")]
                let synthesis_start = Instant::now();
                let stage_results = top_down_many(&db, &stage, 1, Some(nz!(1usize))).0;
                #[cfg(feature = "db-stats")]
                {
                    synthesis_time += synthesis_start.elapsed();
                }
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

            #[cfg(feature = "db-stats")]
            total_synthesis_ms.fetch_add(
                synthesis_time.as_millis().try_into().unwrap(),
                atomic::Ordering::Relaxed,
            );
        });
        info!(
            "Stage (without saving) {} took {:?}",
            stage_idx,
            stage_start.elapsed()
        );

        #[cfg(feature = "db-stats")]
        {
            info!("DB stats: {}", db.basic_stats());
            let stime = total_synthesis_ms.load(atomic::Ordering::Relaxed);
            let btime = db.blocking_ms();
            info!(
                "synthesis: {stime}ms; blocking: {btime}ms ({:.0}%)",
                100.0 * btime as f64 / stime as f64
            );
            db.reset_basic_stats();
        }

        write_stages_completed(&fingerprint, db_path, stage_idx + 1);

        if let Some(max_stages) = args.stages {
            if stage_idx >= max_stages {
                info!("Stopping early because --stages was passed");
                break;
            }
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

fn read_stages_to_skip(
    current_job_fingerprint: &JobFingerprint,
    db_path: Option<&path::Path>,
) -> usize {
    let Some(db_path) = db_path else {
        return 0;
    };
    if !db_path.is_dir() {
        return 0;
    }
    let path = db_path.join(META_FILENAME);
    if !path.exists() {
        return 0;
    }

    let file = fs::File::open(&path).unwrap();
    let buf_reader = std::io::BufReader::new(file);
    let (read_fingerprint, stages_completed): (JobFingerprint, usize) =
        bincode::deserialize_from(buf_reader).unwrap();

    if current_job_fingerprint != &read_fingerprint {
        return 0;
    }
    stages_completed
}

fn write_stages_completed(
    current_job_fingerprint: &JobFingerprint,
    db_path: Option<&path::Path>,
    stages_completed: usize,
) {
    let Some(db_path) = db_path else {
        return;
    };
    fs::create_dir_all(db_path).unwrap();
    let path = db_path.join(META_FILENAME);
    let file = fs::File::create(path).unwrap();
    let buf_writer = std::io::BufWriter::new(file);
    bincode::serialize_into(buf_writer, &(current_job_fingerprint, stages_completed)).unwrap();
}
