#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use adler::Adler32;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::info;
use nonzero::nonzero as nz;
use rayon::prelude::*;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use std::{fs, iter, path};

#[cfg(feature = "db-stats")]
use {
    std::sync::atomic::{self, AtomicU64},
    std::time::Duration,
};

use morello::common::{DimSize, Dtype};
use morello::db::FilesDatabase;
use morello::grid::compose::Compose;
use morello::grid::downscale::DownscaleSurMap;
use morello::grid::general::SurMap;
use morello::grid::linear::BimapInt;
use morello::layout::row_major;
use morello::lspec;
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::search::top_down_many;
use morello::smallvec::smallvec;
use morello::spec::{
    LogicalSpec, LogicalSpecSurMap, PrimitiveBasics, PrimitiveBasicsBimap, PrimitiveSpecType, Spec,
};
use morello::target::{CpuMemoryLevel, MemoryLevel, Target, X86Target};
use morello::tensorspec::{TensorSpecAux, TensorSpecAuxSurMap};
use morello::utils::{bit_length, diagonals};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

type JobFingerprint = (usize, u64);

const META_FILENAME: &str = "PRECOMPUTE";
const DB_PROGRESS_VERSION: usize = 1;
const K: u8 = 1;
const SUBWORKLIST_MAX_SIZE: usize = 5000;
const TWOS: [u32; 32] = [2; 32];

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, help = "Show a progress bar.")]
    progress_bar: bool,
    #[arg(long, help = "Maximum number of stages to run.")]
    stages: Option<usize>,
    #[arg(long)]
    db: Option<path::PathBuf>,
    #[arg(long, default_value = "32", help = "Cache size in database pages.")]
    cache_size: usize,
    #[arg(long, default_value = "matmul")]
    through: ThroughSpec,
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

#[derive(ValueEnum, Debug, Clone, PartialEq, Eq)]
enum ThroughSpec {
    Move,
    Zero,
    Matmul,
    Conv,
}

struct ApplyRhs<S, T>(pub S, pub PhantomData<T>);

struct MaxVec<'a, S>(pub S, pub Arc<Vec<BimapInt>>, pub PhantomData<&'a ()>);

impl<S, T> SurMap for ApplyRhs<S, T>
where
    S: SurMap,
    S::DomainIter: Send + 'static,
    T: Clone + Send + 'static,
{
    type Domain = (T, S::Domain);
    type Codomain = (T, S::Codomain);
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + Send>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (t.0.clone(), self.0.apply(&t.1))
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let lhs = i.0.clone();
        Box::new(self.0.apply_inverse(&i.1).map(move |u| (lhs.clone(), u)))
    }
}

impl<'a, S> SurMap for MaxVec<'a, S>
where
    S: SurMap<Domain = Vec<BimapInt>, Codomain = Vec<BimapInt>> + Sync,
    S::DomainIter: Send + 'a,
{
    type Domain = S::Domain;
    type Codomain = S::Codomain;
    type DomainIter = Box<dyn Iterator<Item = Self::Domain> + Send + 'a>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        if t.iter().zip(&*self.1).any(|(v, m)| v > m) {
            panic!("input exceeds max");
        }
        self.0.apply(t)
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let a = self.0.apply_inverse(i);
        let maxes = Arc::clone(&self.1);
        Box::new(a.filter(move |d| d.iter().zip(&*maxes).all(|(v, m)| v <= m)))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let multi_opt = if args.progress_bar {
        let logger = env_logger::Builder::from_env(env_logger::Env::default()).build();
        let multi = MultiProgress::new();
        LogWrapper::new(multi.clone(), logger).try_init().unwrap();
        Some(multi)
    } else {
        env_logger::init();
        None
    };

    #[cfg(feature = "db-stats")]
    info!("DB statistic collection enabled");

    let threads = rayon::current_num_threads();
    let db = FilesDatabase::new(args.db.as_deref(), true, K, args.cache_size, threads);
    main_per_db(&args, db, args.db.as_deref(), multi_opt);

    Ok(())
}

fn main_per_db(
    args: &Args,
    #[allow(unused_mut)] mut db: FilesDatabase, // mut when db-stats enabled
    db_path: Option<&path::Path>,
    multi_opt: Option<MultiProgress>,
) {
    let levels = X86Target::levels();
    let MemoryLimits::Standard(top) = X86Target::max_mem();

    // TODO: Most of the following details aren't used in computing the bound.
    // It could be simplified.
    let bounds = goal_bounds(args);

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

    let mut stage_idx = 0;
    for bound_spec in &bounds {
        let unscaled_surmap = LogicalSpecSurMap::new(
            PrimitiveBasicsBimap {
                binary_scale_shapes: true,
            },
            TensorSpecAuxSurMap::new,
        );
        let (key, unscaled_bound_pt) = unscaled_surmap.apply(bound_spec);
        let surmap = Compose(
            unscaled_surmap,
            ApplyRhs(downscaler(unscaled_bound_pt), PhantomData),
        );
        let (_, bound_pt) = surmap.apply(bound_spec);

        // TODO: Implement `len` for Diagonals so we don't need to collect just to get a count.
        let stages = diagonals(&bound_pt).collect::<Vec<_>>();

        let table_progress_bar = multi_opt.as_ref().map(|m| {
            let bar = ProgressBar::new(stages.len().try_into().unwrap());
            bar.set_style(
                ProgressStyle::with_template(
                    "table: [{elapsed:>5}] {bar:20} {percent}% {pos:>9}/{len:9}",
                )
                .unwrap(),
            );
            m.add(bar)
        });

        for stage in stages {
            let stage_progress_bar = multi_opt.as_ref().map(|m| {
                let bar = ProgressBar::new(0);
                bar.set_style(
                    ProgressStyle::with_template(
                        "stage: [{elapsed:>5}] {bar:20} {percent}% {pos:>9}/{len:9} ({per_sec:.2}) {msg}",
                    )
                    .unwrap(),
                );
                m.add(bar)
            });

            if stage_idx < stages_completed {
                stage_idx += 1;
                if let Some(pb) = &table_progress_bar {
                    pb.inc(1);
                }
                continue;
            }

            // Construct the TaskIters, dropping empties. Materializing these up front simplifies
            // parallelization, makes out following "has a peak parallelism of" log message more
            // meaningful, and simplifies logging an example Spec.
            let mut stage = stage
                .filter_map(|task_pt| {
                    let mut p = surmap
                        .apply_inverse(&(key.clone(), task_pt))
                        .filter(|l| l.is_canonical())
                        .peekable();
                    p.peek()?;
                    Some(p)
                })
                .collect::<Vec<_>>();

            info!(
                "Beginning stage {stage_idx}, which has peak parallelism of {}",
                stage.len()
            );

            if let Some(example_spec) = stage.first_mut().and_then(|v| v.peek()) {
                info!("Example problem Spec: {}", example_spec);
            }

            #[cfg(feature = "db-stats")]
            let total_synthesis_ms = AtomicU64::new(0);

            let stage_start = Instant::now();
            stage.into_par_iter().for_each(|task| {
                let mut worklist = task
                    .into_iter()
                    .map(|t| Spec(t, MemoryLimits::Standard(top.clone())))
                    .collect::<Vec<_>>();
                if let Some(pb) = &stage_progress_bar {
                    pb.inc_length(worklist.len().try_into().unwrap());
                }
                let mut next_stage = HashSet::new();

                #[cfg(feature = "db-stats")]
                let mut synthesis_time = Duration::ZERO;
                while !worklist.is_empty() {
                    // Check that stage is all unique
                    {
                        let mut stage_set = HashSet::new();
                        for spec in &worklist {
                            if !stage_set.insert(spec) {
                                panic!("Duplicate spec in stage: {:?}", spec);
                            }
                        }
                    }

                    #[cfg(feature = "db-stats")]
                    let synthesis_start = Instant::now();

                    let mut subworklist_offset = 0;
                    let mut stage_results = Vec::with_capacity(worklist.len());
                    while subworklist_offset < worklist.len() {
                        let subworklist = &worklist[subworklist_offset
                            ..worklist
                                .len()
                                .min(subworklist_offset + SUBWORKLIST_MAX_SIZE)];
                        stage_results
                            .extend(top_down_many(&db, subworklist, 1, Some(nz!(1usize))).0);
                        if let Some(pb) = &stage_progress_bar {
                            pb.inc(subworklist.len().try_into().unwrap());
                        }
                        if let Some(pb) = &table_progress_bar {
                            pb.tick();
                        }
                        subworklist_offset += subworklist.len();
                    }

                    #[cfg(feature = "db-stats")]
                    {
                        synthesis_time += synthesis_start.elapsed();
                    }
                    for (spec, result) in worklist.iter().zip(stage_results) {
                        if let [(_, only_result_cost)] = &result[..] {
                            next_stage.extend(
                                next_limits(&spec.1, &only_result_cost.peaks, &levels)
                                    .map(|l| Spec(spec.0.clone(), MemoryLimits::Standard(l))),
                            );
                        }
                    }
                    if let Some(pb) = &stage_progress_bar {
                        pb.inc_length(next_stage.len().try_into().unwrap());
                    }
                    // TODO: Just swap data structures.
                    worklist = next_stage.drain().collect();
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

            if let Some(pb) = stage_progress_bar {
                pb.finish_and_clear();
            }

            let save_start = Instant::now();
            db.save();
            write_stages_completed(&fingerprint, db_path, stage_idx + 1);
            info!("Saving took {:?}", save_start.elapsed());

            if let Some(max_stages) = args.stages {
                if stage_idx >= max_stages {
                    info!("Stopping early because --stages was passed");
                    return;
                }
            }

            stage_idx += 1;
            if let Some(pb) = &table_progress_bar {
                pb.inc(1);
            }
        }

        if let Some(pb) = table_progress_bar {
            pb.finish_and_clear();
        }
    }
}

fn goal_bounds(args: &Args) -> Vec<LogicalSpec<X86Target>> {
    let mut bounds = vec![];
    let move_needed_rank = match args.through {
        ThroughSpec::Conv => 4,
        _ => 2,
    };

    bounds.extend((1..=move_needed_rank).flat_map(|rank| [move_top(args.size, rank)]));
    if args.through == ThroughSpec::Move {
        return bounds;
    }

    bounds.extend((1..=move_needed_rank).map(|rank| {
        lspec!(FillZero(
            iter::repeat_n(args.size, rank.into()),
            (u32, CpuMemoryLevel::GL, row_major),
            serial
        ))
    }));
    if args.through == ThroughSpec::Zero {
        return bounds;
    }

    bounds.push(lspec!(Matmul(
        [nz!(1u32), args.size, args.size, args.size],
        (u32, CpuMemoryLevel::GL, row_major),
        (u32, CpuMemoryLevel::GL, row_major),
        (u32, CpuMemoryLevel::GL, row_major),
        serial
    )));
    if args.through == ThroughSpec::Matmul {
        return bounds;
    }

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
    bounds
}

/// Returns a logical Move Spec of given size and rank.
fn move_top(size: DimSize, rank: u8) -> LogicalSpec<X86Target> {
    lspec!(Move(
        iter::repeat_n(size, rank.into()),
        (u32, CpuMemoryLevel::GL, row_major),
        (u32, CpuMemoryLevel::GL, row_major),
        serial
    ))
}

fn next_limits<'a>(
    result_limits: &'a MemoryLimits,
    result_peak: &'a MemVec,
    levels: &'a [CpuMemoryLevel],
) -> impl Iterator<Item = MemVec> + 'a {
    let MemoryLimits::Standard(limits_vec) = result_limits;
    debug_assert!(limits_vec
        .iter()
        .zip(result_peak.iter())
        .all(|(l, p)| l >= p));
    (0..limits_vec.len()).filter_map(|idx| {
        let mut new_values = limits_vec.clone();
        let lower = match result_peak.get_unscaled(idx) {
            0 => return None,
            1 => 0,
            prev if levels[idx].counts_registers() => prev - 1,
            prev => 1 << (bit_length(prev) - 2),
        };
        new_values.set(idx, lower);
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

fn downscaler<'a>(unscaled_bound: Vec<BimapInt>) -> MaxVec<'a, DownscaleSurMap<'static>> {
    let rank = unscaled_bound.len();
    debug_assert!(rank <= TWOS.len());
    MaxVec(
        DownscaleSurMap(&TWOS[..rank]),
        Arc::new(unscaled_bound),
        PhantomData,
    )
}
