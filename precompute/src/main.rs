#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

use adler::Adler32;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use log::info;
use nonzero::nonzero as nz;
use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use std::{fs, iter, path};

#[cfg(feature = "db-stats")]
use std::sync::atomic::{self, AtomicU64};

use morello::common::{DimSize, Dtype, Shape};
use morello::db::{ActionCostVec, FilesDatabase};
use morello::grid::compose::Compose;
use morello::grid::downscale::DownscaleSurMap;
use morello::grid::general::SurMap;
use morello::grid::linear::BimapInt;
use morello::layout::row_major;
use morello::memorylimits::{MemVec, MemoryLimits};
use morello::search::top_down_many;
use morello::smallvec::smallvec;
use morello::spec::{
    FillValue, LogicalSpec, LogicalSpecSurMap, PrimitiveBasics, PrimitiveBasicsBimap,
    PrimitiveSpecType, Spec,
};
use morello::target::{
    ArmTarget, Avx2Target, Avx512Target, CpuMemoryLevel, CpuTarget, MemoryLevel, Target, TargetId,
};
use morello::tensorspec::{TensorSpecAux, TensorSpecAuxSurMap};
use morello::utils::{bit_length, diagonals};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

type JobFingerprint = (usize, u64);

const META_FILENAME: &str = "PRECOMPUTE";
const DB_PROGRESS_VERSION: usize = 2;
const K: u8 = 1;
const SUBWORKLIST_MAX_SIZE: usize = 5000;
const TWOS: [u32; 32] = [2; 32];
const DTYPES: [Dtype; 2] = [Dtype::Uint32, Dtype::Float32];

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    db: Option<path::PathBuf>,
    #[arg(long, default_value = "2000", help = "Cache size in database pages.")]
    cache_size: usize,
    /// Target architecture
    #[arg(long, value_enum, hide_default_value = true, default_value_t = TargetId::default())]
    target: TargetId,
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
    match &args.target {
        TargetId::Avx2 => main_with_target::<Avx2Target>(&args),
        TargetId::Avx512 => main_with_target::<Avx512Target>(&args),
        TargetId::Arm => main_with_target::<ArmTarget>(&args),
    }
}

fn main_with_target<Tgt>(args: &Args) -> Result<()>
where
    Tgt: CpuTarget,
    Tgt::Level: morello::grid::canon::CanonicalBimap + Sync,
    <Tgt::Level as morello::grid::canon::CanonicalBimap>::Bimap:
        morello::grid::general::BiMap<Codomain = u8>,
{
    env_logger::init();

    #[cfg(feature = "db-stats")]
    info!("DB statistic collection enabled");

    let threads = rayon::current_num_threads();
    let db = FilesDatabase::new::<Tgt>(args.db.as_deref(), true, K, args.cache_size, threads);
    main_per_db::<Tgt>(args, db, args.db.as_deref());

    Ok(())
}

fn main_per_db<Tgt>(
    args: &Args,
    #[allow(unused_mut)] mut db: FilesDatabase, // mut when db-stats enabled
    db_path: Option<&path::Path>,
) where
    Tgt: CpuTarget,
    Tgt::Level: morello::grid::canon::CanonicalBimap + Sync,
    <Tgt::Level as morello::grid::canon::CanonicalBimap>::Bimap:
        morello::grid::general::BiMap<Codomain = u8>,
{
    let levels = Tgt::levels();
    let MemoryLimits::Standard(top) = Tgt::max_mem();

    let phases = goal_phases::<Tgt>(args);
    let bounds: Vec<_> = phases.iter().flatten().cloned().collect();

    let fingerprint = compute_job_fingerprint(&bounds);
    let progress_map = read_stages_to_skip(&fingerprint, db_path);
    if !progress_map.is_empty() {
        info!("Stages already computed for existing jobs:");
        let mut pairs: Vec<_> = progress_map
            .iter()
            .map(|(s, c)| (s.to_string(), *c))
            .collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (spec_str, completed) in pairs {
            info!("  {completed} stages for {spec_str}");
        }
    }

    // Launch thread responsible for writing updates to the PRECOMPUTE file
    let initial_progress_map = progress_map.clone();
    let (meta_update_tx, meta_update_rx) = mpsc::channel();
    let db_path_buf = db_path.map(|p| p.to_path_buf());
    let writer_handle = std::thread::spawn(move || {
        let mut progress_map = progress_map;
        for (spec, completed) in meta_update_rx {
            if progress_map
                .get(&spec)
                .is_some_and(|prev| *prev >= completed)
            {
                continue;
            }
            progress_map.insert(spec, completed);
            write_stages_completed(&fingerprint, db_path_buf.as_deref(), &progress_map);
        }
    });

    // Run each parallel phase
    for phase in phases {
        let phase_annotated = phase
            .into_iter()
            .map(|s| {
                let completed = initial_progress_map.get(&s).copied().unwrap_or(0);
                (s, completed)
            })
            .collect::<Vec<_>>();
        phase_annotated
            .into_par_iter()
            .for_each(|(bound_spec, spec_completed)| {
                process_spec(
                    &db,
                    bound_spec,
                    &levels,
                    &top,
                    spec_completed,
                    meta_update_tx.clone(),
                );
            });
    }

    drop(meta_update_tx);
    writer_handle.join().unwrap();
}

#[allow(clippy::too_many_arguments)]
fn process_spec<Tgt>(
    db: &FilesDatabase,
    bound_spec: LogicalSpec<Tgt>,
    levels: &[Tgt::Level],
    top: &MemVec,
    spec_completed: usize,
    progress_sender: mpsc::Sender<(LogicalSpec<Tgt>, usize)>,
) where
    Tgt: CpuTarget,
    Tgt::Level: morello::grid::canon::CanonicalBimap + Sync,
    <Tgt::Level as morello::grid::canon::CanonicalBimap>::Bimap:
        morello::grid::general::BiMap<Codomain = u8>,
{
    let unscaled_surmap = LogicalSpecSurMap::new(
        PrimitiveBasicsBimap {
            binary_scale_shapes: true,
        },
        TensorSpecAuxSurMap::new,
    );
    let (key, unscaled_bound_pt) = unscaled_surmap.apply(&bound_spec);
    let surmap = Compose(
        unscaled_surmap,
        ApplyRhs(downscaler(unscaled_bound_pt), PhantomData),
    );
    let (_, bound_pt) = surmap.apply(&bound_spec);

    let stages = diagonals(&bound_pt).collect::<Vec<_>>();
    let total_stages = stages.len();

    for (stage_number, stage) in stages.into_iter().enumerate() {
        if stage_number < spec_completed {
            continue;
        }

        // Construct the TaskIters, dropping empties. Materializing these up front simplifies
        // parallelization, makes out following "has a peak parallelism of" log message more
        // meaningful, and simplifies logging an example Spec.
        let stage = stage
            .filter_map(|task_pt| {
                let specs = surmap
                    .apply_inverse(&(key.clone(), task_pt))
                    .filter(|l| l.is_canonical())
                    .collect::<Vec<_>>();
                if specs.is_empty() {
                    None
                } else {
                    Some(specs)
                }
            })
            .collect::<Vec<_>>();

        let stage_within_spec = stage_number + 1;
        info!(
            "Beginning stage {} of {} for {} with peak parallelism of {}",
            stage_within_spec,
            total_stages,
            bound_spec,
            stage.len()
        );

        #[cfg(feature = "db-stats")]
        let total_synthesis_ms = AtomicU64::new(0);

        let stage_start = Instant::now();
        stage.into_par_iter().for_each(|task| {
            let mut worklist = task
                .into_iter()
                .map(|t| Spec(t, MemoryLimits::Standard(top.clone())))
                .collect::<Vec<_>>();
            let mut next_stage = HashSet::new();

            #[cfg(feature = "db-stats")]
            let mut synthesis_time = Duration::ZERO;
            while !worklist.is_empty() {
                validate_stage_worklist_unique(&worklist);

                #[cfg(feature = "db-stats")]
                let synthesis_start = Instant::now();

                let stage_results = process_worklist_chunks(db, &worklist);

                #[cfg(feature = "db-stats")]
                {
                    synthesis_time += synthesis_start.elapsed();
                }
                compute_next_stage(&worklist, stage_results, levels, &mut next_stage);
                worklist = next_stage.drain().collect();
            }

            #[cfg(feature = "db-stats")]
            total_synthesis_ms.fetch_add(
                synthesis_time.as_millis().try_into().unwrap(),
                atomic::Ordering::Relaxed,
            );
        });
        info!(
            "Completed stage {stage_within_spec} of {total_stages} for {bound_spec} in {:?}",
            stage_start.elapsed()
        );

        #[cfg(feature = "db-stats")]
        log_db_stats(db, &total_synthesis_ms);

        if let Err(err) = progress_sender.send((bound_spec.clone(), stage_within_spec)) {
            log::error!("Failed to enqueue progress update for {bound_spec}: {err:?}");
        }

        let save_start = Instant::now();
        db.save();
        info!("Saving took {:?}", save_start.elapsed());
    }
}

fn compute_job_fingerprint<Tgt: Target>(bounds: &[LogicalSpec<Tgt>]) -> JobFingerprint {
    let mut progress_fingerprint_hasher = Adler32::new();
    bounds.hash(&mut progress_fingerprint_hasher);
    (DB_PROGRESS_VERSION, progress_fingerprint_hasher.finish())
}

/// Check that given [Spec]s are all unique. Panics otherwise.
fn validate_stage_worklist_unique<Tgt: Target>(worklist: &[Spec<Tgt>]) {
    let mut stage_set = HashSet::new();
    for spec in worklist {
        if !stage_set.insert(spec) {
            panic!("Duplicate spec in stage: {spec:?}");
        }
    }
}

fn process_worklist_chunks<Tgt>(db: &FilesDatabase, worklist: &[Spec<Tgt>]) -> Vec<ActionCostVec>
where
    Tgt: CpuTarget,
    Tgt::Level: morello::grid::canon::CanonicalBimap + Sync,
    <Tgt::Level as morello::grid::canon::CanonicalBimap>::Bimap:
        morello::grid::general::BiMap<Codomain = u8>,
{
    let mut subworklist_offset = 0;
    let mut stage_results = Vec::with_capacity(worklist.len());

    while subworklist_offset < worklist.len() {
        let subworklist = &worklist[subworklist_offset
            ..worklist
                .len()
                .min(subworklist_offset + SUBWORKLIST_MAX_SIZE)];
        stage_results.extend(top_down_many(db, subworklist, 1));

        subworklist_offset += subworklist.len();
    }

    stage_results
}

fn compute_next_stage<Tgt: Target>(
    worklist: &[Spec<Tgt>],
    stage_results: Vec<morello::db::ActionCostVec>,
    levels: &[Tgt::Level],
    next_stage: &mut HashSet<Spec<Tgt>>,
) {
    for (spec, result) in worklist.iter().zip(stage_results) {
        if let [(_, only_result_cost)] = &result.0[..] {
            next_stage.extend(
                next_limits(&spec.1, &only_result_cost.peaks, levels)
                    .map(|l| Spec(spec.0.clone(), MemoryLimits::Standard(l))),
            );
        }
    }
}

#[cfg(feature = "db-stats")]
fn log_db_stats(db: &FilesDatabase, total_synthesis_ms: &AtomicU64) {
    info!("DB stats: {}", db.basic_stats());
    let stime = total_synthesis_ms.load(atomic::Ordering::Relaxed);
    let btime = db.blocking_ms();
    info!(
        "synthesis: {stime}ms; blocking: {btime}ms ({:.0}%)",
        100.0 * btime as f64 / stime as f64
    );
}

fn goal_phases<Tgt: CpuTarget>(args: &Args) -> Vec<Vec<LogicalSpec<Tgt>>> {
    let mut phases = vec![];
    let move_needed_rank = match args.through {
        ThroughSpec::Conv => 4,
        _ => 3,
    };

    phases.push({
        let mut move_phase = vec![];
        for rank in 1..=move_needed_rank {
            for &dtype in &DTYPES {
                move_phase.push(move_top::<Tgt>(args.size, rank, dtype));
            }
        }
        assert!(!move_phase.is_empty());
        move_phase
    });
    if args.through == ThroughSpec::Move {
        return phases;
    }

    phases.push({
        let mut zero_phase = Vec::with_capacity(DTYPES.len());
        for rank in 1..=move_needed_rank {
            for &dtype in &DTYPES {
                zero_phase.push(fill_zero_top(args.size, rank, dtype));
            }
        }
        zero_phase
    });

    phases.push({
        let mut matmul_phase = vec![];
        for &dtype in &DTYPES {
            matmul_phase.push(matmul_top(args.size, dtype));
        }
        for rank in 1..=move_needed_rank {
            let size_cube: Shape = iter::repeat_n(args.size, usize::from(rank)).collect();
            for &dtype in &DTYPES {
                for dim in 0..rank {
                    let mut reduced_shape = size_cube.clone();
                    reduced_shape[usize::from(dim)] = nz!(1u32);

                    matmul_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Broadcast { dim },
                            spec_shape: size_cube.clone(),
                            dtypes: vec![dtype; 2],
                        },
                        vec![taux_gl(&reduced_shape), taux_gl(&size_cube)],
                        true,
                    ));

                    matmul_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Max { dim, accum: false },
                            spec_shape: size_cube.clone(),
                            dtypes: vec![dtype; 2],
                        },
                        vec![taux_gl(&size_cube), taux_gl(&reduced_shape)],
                        true,
                    ));

                    matmul_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::SoftmaxDenominator {
                                scan_dim: dim,
                                accum: false,
                            },
                            spec_shape: size_cube.clone(),
                            dtypes: vec![dtype; 3],
                        },
                        vec![
                            taux_gl(&size_cube),
                            taux_gl(&reduced_shape),
                            taux_gl(&reduced_shape),
                        ],
                        true,
                    ));

                    matmul_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaledFromMax {
                                scan_dim: dim,
                                accum: false,
                            },
                            spec_shape: size_cube.clone(),
                            dtypes: vec![dtype; 4],
                        },
                        vec![
                            taux_gl(&size_cube),
                            taux_gl(&reduced_shape),
                            taux_gl(&reduced_shape),
                            taux_gl(&size_cube),
                        ],
                        true,
                    ));
                }

                matmul_phase.push(LogicalSpec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::DivideVec,
                        spec_shape: size_cube.clone(),
                        dtypes: vec![dtype; 3],
                    },
                    vec![
                        taux_gl(&size_cube),
                        taux_gl(&size_cube),
                        taux_gl(&size_cube),
                    ],
                    true,
                ));
            }
        }
        matmul_phase
    });
    if args.through == ThroughSpec::Matmul {
        return phases;
    }

    phases.push({
        let mut conv_phase = vec![];
        for &dtype in &DTYPES {
            for &fs in &args.filters_size {
                let s = DimSize::new(args.size.get() - 1 + fs.get()).unwrap();
                let img_aux = taux_gl(&[args.batch, args.channels, s, s]);
                let filters_aux = taux_gl(&[args.filters, args.channels, fs, fs]);
                let output_aux = taux_gl(&[args.batch, args.filters, args.size, args.size]);
                conv_phase.push(LogicalSpec::Primitive(
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
                        dtypes: vec![dtype; 3],
                    },
                    vec![img_aux, filters_aux, output_aux],
                    true,
                ));
            }
        }
        for rank in 1..=move_needed_rank {
            let numer_shape: Shape = iter::repeat_n(args.size, usize::from(rank)).collect();
            for &dtype in &DTYPES {
                for scan_dim in 0..rank {
                    let mut denom_shape = numer_shape.clone();
                    denom_shape[usize::from(scan_dim)] = nz!(1u32);

                    conv_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::SoftmaxDenominatorAndUnscaled {
                                scan_dim,
                                accum: false,
                            },
                            spec_shape: numer_shape.clone(),
                            dtypes: vec![dtype; 3],
                        },
                        vec![
                            taux_gl(&numer_shape),
                            taux_gl(&denom_shape),
                            taux_gl(&numer_shape),
                        ],
                        true,
                    ));

                    conv_phase.push(LogicalSpec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::DivideVecScalar { scan_dim },
                            spec_shape: numer_shape.clone(),
                            dtypes: vec![dtype; 3],
                        },
                        vec![
                            taux_gl(&numer_shape),
                            taux_gl(&denom_shape),
                            taux_gl(&numer_shape),
                        ],
                        true,
                    ));
                }
            }
        }
        conv_phase
    });
    phases
}

/// Returns a logical Move Spec of given size and rank.
fn matmul_top<Tgt: CpuTarget>(size: DimSize, dtype: Dtype) -> LogicalSpec<Tgt> {
    let spec_shape = smallvec![nz!(1u32), size, size, size];
    let param_shapes = [
        [nz!(1u32), size, size],
        [nz!(1u32), size, size],
        [nz!(1u32), size, size],
    ];
    let auxes = param_shapes
        .into_iter()
        .map(|shape| TensorSpecAux {
            level: CpuMemoryLevel::GL.into(),
            layout: row_major(&shape),
            vector_size: None,
        })
        .collect::<Vec<_>>();
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Matmul { accum: false },
            spec_shape,
            dtypes: vec![dtype; 3],
        },
        auxes,
        true,
    )
}

fn move_top<Tgt: CpuTarget>(size: DimSize, rank: u8, dtype: Dtype) -> LogicalSpec<Tgt> {
    let shape: Shape = iter::repeat_n(size, rank.into()).collect();
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Move,
            spec_shape: shape.clone(),
            dtypes: vec![dtype; 2],
        },
        vec![
            TensorSpecAux {
                level: CpuMemoryLevel::GL.into(),
                layout: row_major(&shape),
                vector_size: None,
            },
            TensorSpecAux {
                level: CpuMemoryLevel::GL.into(),
                layout: row_major(&shape),
                vector_size: None,
            },
        ],
        true,
    )
}

fn fill_zero_top<Tgt: CpuTarget>(size: DimSize, rank: u8, dtype: Dtype) -> LogicalSpec<Tgt> {
    let shape: Shape = iter::repeat_n(size, rank.into()).collect();
    LogicalSpec::Primitive(
        PrimitiveBasics {
            typ: PrimitiveSpecType::Fill {
                value: FillValue::Zero,
            },
            spec_shape: shape.clone(),
            dtypes: vec![dtype],
        },
        vec![taux_gl::<Tgt>(&shape)],
        true,
    )
}

fn next_limits<'a, L: MemoryLevel + 'a>(
    result_limits: &'a MemoryLimits,
    result_peak: &'a MemVec,
    levels: &'a [L],
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

fn read_stages_to_skip<Tgt: Target>(
    current_job_fingerprint: &JobFingerprint,
    db_path: Option<&path::Path>,
) -> HashMap<LogicalSpec<Tgt>, usize> {
    let Some(db_path) = db_path else {
        return HashMap::new();
    };
    if !db_path.is_dir() {
        return HashMap::new();
    }
    let path = db_path.join(META_FILENAME);
    let Ok(bytes) = fs::read(&path) else {
        return HashMap::new();
    };

    let Ok((read_fingerprint, stage_pairs)) =
        bincode::deserialize::<(JobFingerprint, Vec<(LogicalSpec<Tgt>, usize)>)>(&bytes)
    else {
        return HashMap::new();
    };

    if current_job_fingerprint != &read_fingerprint {
        return HashMap::new();
    }

    stage_pairs.into_iter().collect()
}

fn write_stages_completed<Tgt: Target>(
    current_job_fingerprint: &JobFingerprint,
    db_path: Option<&path::Path>,
    progress: &HashMap<LogicalSpec<Tgt>, usize>,
) {
    let Some(db_path) = db_path else {
        return;
    };
    fs::create_dir_all(db_path).unwrap();
    let path = db_path.join(META_FILENAME);
    let file = fs::File::create(path).unwrap();
    let buf_writer = std::io::BufWriter::new(file);
    let stage_pairs: Vec<_> = progress.iter().map(|(k, v)| (k.clone(), *v)).collect();
    bincode::serialize_into(buf_writer, &(*current_job_fingerprint, stage_pairs)).unwrap();
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

fn taux_gl<Tgt: CpuTarget>(shape: &[DimSize]) -> TensorSpecAux<Tgt> {
    TensorSpecAux {
        level: CpuMemoryLevel::GL.into(),
        layout: row_major(shape),
        vector_size: None,
    }
}
