mod blocks;

use crate::common::DimSize;
use crate::cost::{Cost, NormalizedCost};
use crate::datadeps::SpecKey;
use crate::db::blocks::{DbBlock, WholeBlock};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{AsBimap, BiMap};
use crate::grid::linear::BimapInt;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::spec::{LogicalSpecSurMap, PrimitiveBasicsBimap, Spec, SpecSurMap};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;
use blocks::{RTreeBlock, RTreeBlockGeneric};

use divrem::DivRem;
use itertools::Itertools;
use parking_lot::{Mutex, MutexGuard};
use prehash::{new_prehashed_set, DefaultPrehasher, Prehashed, PrehashedSet, Prehasher};
use serde::{Deserialize, Serialize};
use wtinylfu::WTinyLfuCache;

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, DerefMut, Range};
use std::path::{self, Path};
use std::sync::{mpsc, Arc};

#[cfg(feature = "db-stats")]
use {
    crate::common::Dtype,
    std::collections::HashSet,
    std::sync::atomic::{self, AtomicU64},
    std::time::Instant,
};

#[cfg(any(feature = "db-stats", test))]
use crate::layout::{row_major, PhysDim};

type DbKey = (TableKey, Vec<BimapInt>); // TODO: Rename to BlockKey for consistency?
type TableKey = (SpecKey, Vec<(Layout, u8, u32)>);
type SuperBlockKey = DbKey;
type SuperBlockContents = HashMap<Vec<BimapInt>, DbBlock>;
pub type ActionIdx = u16;

/// The number of shards/locks per thread.
const THREAD_SHARDS: usize = 2;
// TODO: Select these at runtime.
const SUPERBLOCK_FACTOR: BimapInt = 2;
const CHANNEL_SIZE: usize = 2;
/// Compress superblocks when writing to disk.
const COMPRESS_SUPERBLOCKS: bool = true;
const USE_RTREE: bool = true;

pub struct FilesDatabase {
    #[allow(dead_code)] // read only when db-stats enabled; otherwise only affects Drop
    dir_handle: Arc<DirPathHandle>,
    binary_scale_shapes: bool,
    k: u8,
    shards: ShardVec,
    prehasher: DefaultPrehasher,
    tiling_depth: Option<NonZeroU32>,
    #[cfg(feature = "db-stats")]
    stats: Arc<FilesDatabaseStats>,
}

#[cfg(feature = "db-stats")]
#[derive(Debug, Default)]
pub struct FilesDatabaseStats {
    disk_bytes_read: AtomicU64,
    disk_bytes_written: AtomicU64,
    gets: AtomicU64,
    puts: AtomicU64,
    /// Total time spent waiting for a database thread.
    blocking_ms: AtomicU64,
}

struct ShardVec(Vec<Mutex<Shard>>);

pub struct PageId<'a> {
    db: &'a FilesDatabase,
    pub(crate) table_key: TableKey,
    pub(crate) superblock_id: Vec<BimapInt>,
}

#[derive(Debug, Clone)]
struct SuperBlock {
    contents: SuperBlockContents,
    modified: bool,
}

struct Shard {
    cache: WTinyLfuCache<Prehashed<SuperBlockKey>, SuperBlock>,
    outstanding_gets: PrehashedSet<SuperBlockKey>,
    thread: Option<std::thread::JoinHandle<()>>,
    thread_tx: mpsc::SyncSender<ShardThreadMsg>,
    thread_rx: mpsc::Receiver<ShardThreadResponse>,
    #[cfg(feature = "db-stats")]
    stats: Arc<FilesDatabaseStats>,
}

#[cfg(feature = "db-stats")]
struct AnalyzeWriters {
    page_writer: csv::Writer<fs::File>,
    block_writer: csv::Writer<fs::File>,
    block_action_writer: csv::Writer<fs::File>,
}

enum ShardThreadMsg {
    Get(Prehashed<SuperBlockKey>),
    Put(SuperBlockKey, SuperBlockContents),
    Exit,
}

enum ShardThreadResponse {
    Loaded(Prehashed<SuperBlockKey>, SuperBlock),
}

/// Contains a path to a directory or a [tempfile::TempDir]. This facilitates deleting the
/// temporary directory on drop.
enum DirPathHandle {
    Persisted(path::PathBuf),
    TempDir(tempfile::TempDir),
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCostVec(pub Vec<(ActionIdx, Cost)>);

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ActionNormalizedCostVec(pub Vec<(ActionIdx, NormalizedCost)>);

pub enum GetPreference<T, V> {
    Hit(T),
    Miss(Option<V>),
}

#[derive(Debug, thiserror::Error)]
#[error("Error reading page; zstd error: '{zstd_error}', plain error: '{plain_error}'")]
struct ReadAnyFormatError {
    zstd_error: bincode::Error,
    plain_error: bincode::Error,
}

impl FilesDatabase {
    pub fn new(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
        k: u8,
        cache_size: usize,
        thread_count: usize,
        tiling_depth: Option<NonZeroU32>,
    ) -> Self {
        let dir_handle = Arc::new(match file_path {
            Some(path) => {
                fs::create_dir_all(path).unwrap();
                DirPathHandle::Persisted(path.to_owned())
            }
            None => DirPathHandle::TempDir(tempfile::TempDir::new().unwrap()),
        });
        log::info!("Opening database at: {}", dir_handle.path().display());

        // Check that the intended tiling depth matches the one logged on disk (if any).
        let tiling_depth_path = dir_handle.path().join("TILING_DEPTH");
        if tiling_depth_path.exists() {
            let raw_buf = fs::read_to_string(&tiling_depth_path).unwrap();
            let buf = raw_buf.trim();
            let file_depth = if buf == "ANY" {
                None
            } else {
                Some(NonZeroU32::new(buf.parse::<u32>().unwrap()).unwrap())
            };
            if tiling_depth != file_depth {
                panic!("Tiling depth mismatch: expected {tiling_depth:?}, found {file_depth:?}");
            }
        } else {
            let mut file = fs::File::create(&tiling_depth_path).unwrap();
            if let Some(depth) = tiling_depth {
                writeln!(file, "{}", depth.get()).unwrap();
            } else {
                writeln!(file, "ANY").unwrap();
            }
        }

        #[cfg(feature = "db-stats")]
        let stats = Arc::new(FilesDatabaseStats::default());

        let shard_count = thread_count * THREAD_SHARDS;
        let cache_size_per_shard = cache_size / shard_count;
        let cache_per_shard_samples = (cache_size_per_shard / 4).max(1);
        let cache_per_shard_size = cache_size_per_shard
            .saturating_sub(cache_per_shard_samples)
            .max(1);
        // TODO: Print the effective cache size
        let actual_total_cache_size =
            shard_count * (cache_per_shard_size + cache_per_shard_samples);
        if actual_total_cache_size != cache_size {
            log::warn!("Database using cache size: {}", actual_total_cache_size);
        }

        let shards = ShardVec(
            (0..shard_count)
                .map(|i| {
                    Mutex::new(Shard::new(
                        i,
                        Arc::clone(&dir_handle),
                        cache_per_shard_size,
                        cache_per_shard_samples,
                        #[cfg(feature = "db-stats")]
                        Arc::clone(&stats),
                    ))
                })
                .collect(),
        );
        Self {
            dir_handle,
            binary_scale_shapes,
            k,
            shards,
            prehasher: DefaultPrehasher::default(),
            tiling_depth,
            #[cfg(feature = "db-stats")]
            stats,
        }
    }

    pub fn get<Tgt>(&self, query: &Spec<Tgt>) -> Option<ActionCostVec>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        match self.get_with_preference(query) {
            GetPreference::Hit(v) => Some(v),
            GetPreference::Miss(_) => None,
        }
    }

    pub fn get_impl<Tgt>(&self, query: &Spec<Tgt>) -> Option<Vec<ImplNode<Tgt>>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let root_results = self.get(query)?;
        let actions = query.0.actions(self.tiling_depth);
        Some(
            root_results
                .as_ref()
                .iter()
                .map(|(action_idx, _cost)| {
                    let root = actions[(*action_idx).into()].apply(query).unwrap();
                    let children = root.children();
                    let new_children = children
                        .iter()
                        .map(|c| construct_impl(self, c))
                        .collect::<Vec<_>>();
                    root.replace_children(new_children.into_iter())
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
    ) -> GetPreference<ActionCostVec, Vec<ActionIdx>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        #[cfg(feature = "db-stats")]
        self.stats.gets.fetch_add(1, atomic::Ordering::Relaxed);

        let mut query = query.clone();
        query.canonicalize().unwrap();

        let bimap = self.spec_bimap();
        let (table_key, global_pt) = bimap.apply(&query);
        let (block_pt, inner_pt) = blockify_point(global_pt);

        let superblock_pt = superblockify_pt(&block_pt);
        let superblock_key = self.prehasher.prehash((table_key, superblock_pt));

        let superblock: &SuperBlock = &self.load_live_superblock(&superblock_key);
        let Some(b) = superblock.contents.get(&block_pt) else {
            return GetPreference::Miss(None);
        };
        b.get_with_preference(&query, &inner_pt, self.binary_scale_shapes)
    }

    pub fn prefetch<Tgt>(&self, query: &Spec<Tgt>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let mut query = query.clone();
        query.canonicalize().unwrap();

        let bimap = self.spec_bimap();
        let (table_key, global_pt) = bimap.apply(&query);
        let (block_pt, _) = blockify_point(global_pt);

        let superblock_pt = superblockify_pt(&block_pt);
        let superblock_key = self.prehasher.prehash((table_key, superblock_pt));

        let shard = &self.shards.0[self.shard_index(&superblock_key)];
        let mut shard_guard = shard.lock();
        shard_guard.process_available_bg_thread_msgs();
        if shard_guard.cache.peek(&superblock_key).is_none() {
            shard_guard.async_get(&superblock_key);
        }
    }

    pub fn page_id<Tgt>(&self, lhs: &Spec<Tgt>) -> PageId
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        assert!(lhs.is_canonical());

        let bimap = self.spec_bimap();
        let (table_key, global_pt_lhs) = bimap.apply(lhs);
        let (block_pt, _) = blockify_point(global_pt_lhs);
        PageId {
            db: self,
            table_key,
            superblock_id: superblockify_pt(&block_pt),
        }
    }

    pub fn put<Tgt>(&self, mut spec: Spec<Tgt>, decisions: Vec<(ActionIdx, Cost)>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        spec.canonicalize().unwrap();

        // Check that all costs in decisions have peak memory less than or equal to spec's
        // memory limits.
        debug_assert!(
            decisions.iter().all(|(_, c)| {
                let MemoryLimits::Standard(limits) = &spec.1;
                &c.peaks <= limits
            }),
            "peak memory of an action exceeds memory limits of {}: {:?}",
            spec,
            decisions
        );

        #[cfg(feature = "db-stats")]
        self.stats.puts.fetch_add(1, atomic::Ordering::Relaxed);

        let bimap = self.spec_bimap();
        let (db_key, (bottom, top)) = put_range_to_fill(&bimap, &spec, &decisions);

        // Construct an iterator over all blocks to fill.
        let rank = bottom.len();
        let blocks_iter = bottom
            .into_iter()
            .zip(&top)
            .enumerate()
            .map(|(dim, (b, t))| iter_blocks_in_single_dim_range(b, *t, block_size_dim(dim, rank)))
            .multi_cartesian_product();

        // Since this put is for a single Spec, we can normalize the cost with that Spec's volume.
        let normalized_decisions =
            ActionNormalizedCostVec::normalize(ActionCostVec(decisions), spec.0.volume());

        for joined_row in blocks_iter {
            let block_pt = joined_row
                .iter()
                .map(|(b, _)| *b)
                .collect::<Vec<BimapInt>>();
            // TODO: Factor out this tuple construction
            let mut superblock_guard = self.load_live_superblock_mut(
                &self
                    .prehasher
                    .prehash((db_key.clone(), superblockify_pt(&block_pt))),
            );
            superblock_guard.modified = true;

            // If the superblock already contains the block, mutate in place and continue the loop.
            if let Some(live_block) = superblock_guard.contents.get_mut(&block_pt) {
                let dim_ranges = joined_row
                    .iter()
                    .map(|(_, r)| r.clone())
                    .collect::<Vec<_>>();
                // Examine the table before updating.
                live_block.fill_region(self.k, &dim_ranges, &normalized_decisions);
                continue;
            }

            // If not, create the block and add it to the superblock.
            let db_shape = db_shape::<Tgt>(rank);
            let bs = block_shape(&block_pt, &db_shape, block_size_dim);
            let block_shape_usize = bs.map(|v| v.try_into().unwrap()).collect::<Vec<_>>();
            let dim_ranges = joined_row
                .iter()
                .map(|(_, r)| r.clone())
                .collect::<Vec<_>>();
            let new_block = if USE_RTREE {
                let mut tree = Box::new(RTreeBlock::empty(dim_ranges.len()));
                tree.fill_region(self.k, &dim_ranges, &normalized_decisions);
                DbBlock::RTree(tree)
            } else {
                DbBlock::Whole(Box::new(WholeBlock::partially_filled::<Tgt>(
                    self.k,
                    &block_shape_usize,
                    &dim_ranges,
                    &normalized_decisions,
                )))
            };
            superblock_guard.contents.insert(block_pt, new_block);
        }
    }

    /// Saves anything cached in memory to disk.
    pub fn save(&self) {
        for shard in &self.shards.0 {
            let mut shard_guard = shard.lock();
            shard_guard.save();
        }
    }

    pub fn max_k(&self) -> Option<usize> {
        Some(self.k.into())
    }

    /// Return a bidirectional map from [Spec]s to tuples of table keys and their coordinates.
    fn spec_bimap<Tgt>(&self) -> impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
        let surmap = SpecSurMap::<Tgt, _, _, _> {
            logical_spec_surmap: LogicalSpecSurMap::new(
                PrimitiveBasicsBimap {
                    binary_scale_shapes: self.binary_scale_shapes,
                },
                |_: &[DimSize], _| TensorSpecAuxNonDepBimap::<Tgt>::default(),
            ),
            memory_limits_bimap: MemoryLimitsBimap::default(),
        };
        surmap.into_bimap()
    }

    pub fn tiling_depth(&self) -> Option<NonZeroU32> {
        self.tiling_depth
    }

    fn load_live_superblock<'a>(
        &'a self,
        key: &Prehashed<SuperBlockKey>,
    ) -> impl Deref<Target = SuperBlock> + 'a {
        self.load_live_superblock_mut(key)
    }

    fn load_live_superblock_mut<'a>(
        &'a self,
        key: &Prehashed<SuperBlockKey>,
    ) -> impl DerefMut<Target = SuperBlock> + 'a {
        let shard = &self.shards.0[self.shard_index(key)];
        let mut shard_guard = shard.lock();

        shard_guard.process_available_bg_thread_msgs();

        let shard_guard = match MutexGuard::try_map(shard_guard, |s| s.cache.get_mut(key)) {
            Ok(mapped) => return mapped,
            Err(s) => s,
        };

        MutexGuard::map(shard_guard, |s| {
            s.async_get(key);
            s.process_bg_thread_msgs_until(|resp| match resp {
                ShardThreadResponse::Loaded(k, _) => k != key,
            });
            s.cache
                .get_mut(key)
                .unwrap_or_else(|| panic!("just-requested key in cache: {key:?}"))
        })
    }

    fn shard_index(&self, key: &Prehashed<SuperBlockKey>) -> usize {
        *Prehashed::as_hash(key) as usize % self.shards.0.len()
    }

    /// Return a string describing some basic counts.
    #[cfg(feature = "db-stats")]
    pub fn basic_stats(&self) -> String {
        let stats = &self.stats;
        let gets = stats.gets.load(atomic::Ordering::SeqCst);
        let puts = stats.puts.load(atomic::Ordering::SeqCst);
        let read = stats.disk_bytes_read.load(atomic::Ordering::SeqCst);
        let written = stats.disk_bytes_written.load(atomic::Ordering::SeqCst);
        format!(
            "gets={}, puts={}, bytes_read={}{}, bytes_written={}{}",
            gets,
            puts,
            read,
            if gets > 0 {
                format!(" ({:.3}/get)", read as f32 / gets as f32)
            } else {
                "".to_string()
            },
            written,
            if puts > 0 {
                format!(" ({:.3}/put)", written as f32 / puts as f32)
            } else {
                "".to_string()
            },
        )
    }

    #[cfg(feature = "db-stats")]
    pub fn reset_basic_stats(&mut self) {
        let stats = &mut self.stats;
        stats.gets.store(0, atomic::Ordering::SeqCst);
        stats.puts.store(0, atomic::Ordering::SeqCst);
        stats.disk_bytes_read.store(0, atomic::Ordering::SeqCst);
        stats.disk_bytes_written.store(0, atomic::Ordering::SeqCst);
        stats.blocking_ms.store(0, atomic::Ordering::SeqCst);
    }

    #[cfg(feature = "db-stats")]
    pub fn blocking_ms(&self) -> u64 {
        self.stats.blocking_ms.load(atomic::Ordering::SeqCst)
    }

    /// Write statistics about the database to stdout.
    ///
    /// This may be expensive and multi-threaded.
    #[cfg(feature = "db-stats")]
    pub fn analyze<Tgt>(&self, output_dir: &path::Path, sample: usize, skip_read_errors: bool)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
        // Read the tiling depth for applying actions.
        // TODO: Share code for loading this file.
        let tiling_depth_path = self.dir_handle.path().join("TILING_DEPTH");
        let tiling_depth_path_raw_buf = fs::read_to_string(tiling_depth_path).unwrap();
        let tiling_depth_buf = tiling_depth_path_raw_buf.trim();
        let tiling_depth = if tiling_depth_buf == "ANY" {
            None
        } else {
            Some(NonZeroU32::new(tiling_depth_buf.parse::<u32>().unwrap()).unwrap())
        };

        let page_csv_path = output_dir.join("pages.csv");
        let block_csv_path = output_dir.join("blocks.csv");
        let block_action_csv_path = output_dir.join("block_actions.csv");

        let mut writers = AnalyzeWriters {
            page_writer: csv::Writer::from_path(page_csv_path.clone()).unwrap(),
            block_writer: csv::Writer::from_path(block_csv_path.clone()).unwrap(),
            block_action_writer: csv::Writer::from_path(block_action_csv_path.clone()).unwrap(),
        };
        writers
            .page_writer
            .write_record(["superblock_path", "unique_actions", "unique_action_costs"])
            .unwrap();
        writers
            .block_writer
            .write_record([
                "superblock_path",
                "block_pt",
                "runs_filled",
                "runs_main_costs",
                "runs_peaks",
                "runs_depths_actions",
                "unique_actions",
                "unique_action_costs",
                "volume",
            ])
            .unwrap();
        writers
            .block_action_writer
            .write_record(["superblock_path", "block_pt", "action"])
            .unwrap();

        analyze_visit_dir::<Tgt>(
            self.dir_handle.path(),
            self.dir_handle.path(),
            &mut writers,
            sample,
            skip_read_errors,
            &self.spec_bimap(),
            tiling_depth,
            self.binary_scale_shapes,
        );

        writers.page_writer.flush().unwrap();
        writers.block_writer.flush().unwrap();
        writers.block_action_writer.flush().unwrap();
    }
}

impl Drop for FilesDatabase {
    fn drop(&mut self) {
        for shard in &mut self.shards.0 {
            let mut shard_guard = shard.lock();
            let drained = shard_guard.drain_cache().collect::<Vec<_>>();
            for (k, v) in drained {
                if v.modified {
                    shard_guard.async_put(k, v.contents);
                }
            }
        }
    }
}

impl<'a> PageId<'a> {
    pub fn contains<Tgt>(&self, spec: &Spec<Tgt>) -> bool
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        assert!(spec.is_canonical());

        let bimap = self.db.spec_bimap();
        let (table_key, global_pt) = bimap.apply(spec);
        if self.table_key != table_key {
            return false;
        }
        let (block_pt, _) = blockify_point(global_pt);
        self.superblock_id == superblockify_pt(&block_pt)
    }
}

impl Shard {
    fn new(
        idx: usize,
        db_root: Arc<DirPathHandle>,
        cache_per_shard_size: usize,
        cache_per_shard_samples: usize,
        #[cfg(feature = "db-stats")] stats: Arc<FilesDatabaseStats>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::sync_channel(CHANNEL_SIZE);
        let (response_tx, response_rx) = mpsc::sync_channel(CHANNEL_SIZE);

        #[cfg(feature = "db-stats")]
        let stats2 = Arc::clone(&stats);

        let thread = Some(
            std::thread::Builder::new()
                .name(format!("ShardThread-{idx}"))
                .spawn(move || loop {
                    match command_rx.recv() {
                        Ok(ShardThreadMsg::Get(key)) => {
                            let path = superblock_file_path(db_root.path(), &key);

                            #[cfg(feature = "db-stats")]
                            {
                                if let Ok(metadata) = path.metadata() {
                                    stats
                                        .disk_bytes_read
                                        .fetch_add(metadata.len(), atomic::Ordering::Relaxed);
                                }
                            }

                            let result = match fs::File::open(&path) {
                                Ok(file) => match read_any_format(file) {
                                    Ok(r) => r,
                                    Err(ReadAnyFormatError { zstd_error, plain_error }) => {
                                        log::error!(
                                            "Continuing after errors reading superblock; {:?} and {:?}",
                                            zstd_error,
                                            plain_error
                                        );
                                        SuperBlock { contents: HashMap::new(), modified: false }
                                    }
                                },
                                Err(_) => SuperBlock { contents: HashMap::new(), modified: false },
                            };
                            response_tx
                                .send(ShardThreadResponse::Loaded(key, result))
                                .unwrap();
                        }
                        Ok(ShardThreadMsg::Put(key, value)) => {
                            let path = superblock_file_path(db_root.path(), &key);

                            #[cfg(feature = "db-stats")]
                            {
                                log::debug!("Evicting superblock");
                            }

                            if let Some(parent) = path.parent() {
                                fs::create_dir_all(parent).unwrap();
                            }
                            let file = fs::File::create(&path).unwrap();
                            if COMPRESS_SUPERBLOCKS {
                                let mut zstd_writer = zstd::Encoder::new(file, 0).unwrap();
                                bincode::serialize_into(&mut zstd_writer, &value).unwrap();
                                zstd_writer.finish().unwrap();
                            } else {
                                let mut buf_writer = BufWriter::new(file);
                                bincode::serialize_into(&mut buf_writer, &value).unwrap();
                                buf_writer.flush().unwrap();
                            }

                            #[cfg(feature = "db-stats")]
                            {
                                stats.disk_bytes_written.fetch_add(
                                    path.metadata().unwrap().len(),
                                    atomic::Ordering::Relaxed,
                                );
                            }
                        }
                        Ok(ShardThreadMsg::Exit) => break,
                        Err(_) => unreachable!("expected Exit first"),
                    }
                })
                .unwrap(),
        );

        Self {
            cache: WTinyLfuCache::new(cache_per_shard_size, cache_per_shard_samples),
            outstanding_gets: new_prehashed_set(),
            thread,
            thread_tx: command_tx,
            thread_rx: response_rx,
            #[cfg(feature = "db-stats")]
            stats: stats2,
        }
    }

    /// Calls `self.thread_rx.recv()`, logging time taken if `db-stats` is enabled.
    fn blocking_recv(&mut self) -> Result<ShardThreadResponse, mpsc::RecvError> {
        #[cfg(feature = "db-stats")]
        let start = Instant::now();

        let received = self.thread_rx.recv();

        #[cfg(feature = "db-stats")]
        {
            let wait_duration = start.elapsed();
            self.stats.blocking_ms.fetch_add(
                wait_duration.as_millis().try_into().unwrap(),
                atomic::Ordering::Relaxed,
            );
        }

        received
    }

    fn process_bg_thread_msgs_until_close(&mut self) {
        while let Ok(msg) = self.blocking_recv() {
            self.process_bg_thread_msg_inner(msg);
        }
    }

    /// Process background thread messages until just *after* `cond` returns `false`.
    fn process_bg_thread_msgs_until<F>(&mut self, mut cond: F)
    where
        F: FnMut(&ShardThreadResponse) -> bool,
    {
        while let Ok(msg) = self.blocking_recv() {
            let should_continue = cond(&msg);
            self.process_bg_thread_msg_inner(msg);
            if !should_continue {
                return;
            }
        }
        panic!("process_bg_thread_msgs exited loop surprisingly!"); // TODO: Remove panic!
    }

    fn process_available_bg_thread_msgs(&mut self) {
        while let Ok(msg) = self.thread_rx.try_recv() {
            self.process_bg_thread_msg_inner(msg);
        }
    }

    fn process_bg_thread_msg_inner(&mut self, msg: ShardThreadResponse) {
        let ShardThreadResponse::Loaded(key, new_value) = msg;

        let was_present = self.outstanding_gets.remove(&key);
        debug_assert!(was_present);
        if let Some((evicted_key, evicted_value)) = self.cache.push(key, new_value) {
            if evicted_value.modified {
                self.async_put(Prehashed::into_inner(evicted_key), evicted_value.contents);
            }
        }
    }

    /// Start a background task to load a superblock. Do nothing if request already enqueued.
    ///
    /// This updates does not update cache statistics.
    ///
    /// This may panic if the key is already in the cache.
    fn async_get(&mut self, key: &Prehashed<SuperBlockKey>) -> bool {
        if self.outstanding_gets.contains(key) {
            return false;
        }
        debug_assert!(self.cache.peek(key).is_none(), "cache already had {key:?}");
        self.outstanding_gets.insert(prehashed_clone(key));
        self.thread_tx
            .send(ShardThreadMsg::Get(prehashed_clone(key)))
            .unwrap();
        true
    }

    fn async_put(&self, key: SuperBlockKey, value: SuperBlockContents) {
        self.thread_tx
            .send(ShardThreadMsg::Put(key, value))
            .unwrap();
    }

    fn save(&mut self) {
        let tx = self.thread_tx.clone();
        for (k, v) in self.cache.iter_mut() {
            // Clone so that the background thread has an immutable copy of the data to write, even
            // if the calling or another thread would update data in the cache.
            // TODO: Instead sync with the background thread and guarantee this is unchanging since
            //       we have the `&self` reference.
            if v.modified {
                v.modified = false;
                tx.send(ShardThreadMsg::Put(
                    Prehashed::as_inner(k).clone(),
                    v.contents.clone(),
                ))
                .unwrap();
            }
        }
    }

    fn drain_cache(&mut self) -> impl Iterator<Item = (SuperBlockKey, SuperBlock)> + '_ {
        std::iter::from_fn(move || {
            if let Some(popped) = self.cache.pop_lru_window() {
                return Some(popped);
            }
            self.cache.pop_lru_main()
        })
        .map(|(k, v)| (Prehashed::into_inner(k), v))
    }
}

impl Drop for Shard {
    fn drop(&mut self) {
        self.thread_tx.send(ShardThreadMsg::Exit).unwrap();
        self.process_bg_thread_msgs_until_close();
        self.thread.take().unwrap().join().unwrap();
    }
}

impl DirPathHandle {
    fn path(&self) -> &path::Path {
        match self {
            DirPathHandle::Persisted(p) => p.as_ref(),
            DirPathHandle::TempDir(t) => t.path(),
        }
    }
}

impl Deref for ActionCostVec {
    type Target = Vec<(ActionIdx, Cost)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Vec<(ActionIdx, Cost)>> for ActionCostVec {
    fn as_ref(&self) -> &Vec<(ActionIdx, Cost)> {
        self.deref()
    }
}

impl ActionNormalizedCostVec {
    pub fn normalize(action_costs: ActionCostVec, volume: DimSize) -> Self {
        ActionNormalizedCostVec(
            action_costs
                .0
                .into_iter()
                .map(|(action_idx, cost)| (action_idx, NormalizedCost::new(cost, volume)))
                .collect(),
        )
    }
}

/// Tries to read a zstd-compressed file. If that fails, tries to read it uncompressed.
fn read_any_format(file: fs::File) -> Result<SuperBlock, ReadAnyFormatError> {
    let mut zstd_reader = zstd::Decoder::new(file).unwrap();
    let contents: SuperBlockContents = match bincode::deserialize_from(&mut zstd_reader) {
        Ok(contents) => contents,
        Err(zstd_error) => {
            // Couldn't read as zstd? Try reading uncompressed.
            let mut file = zstd_reader.finish();
            file.seek(SeekFrom::Start(0)).unwrap();
            let buf_reader = BufReader::new(file);
            match bincode::deserialize_from(buf_reader) {
                Ok(superblock) => superblock,
                Err(plain_error) => {
                    return Err(ReadAnyFormatError {
                        zstd_error,
                        plain_error,
                    })
                }
            }
        }
    };
    Ok(SuperBlock {
        contents,
        modified: false,
    })
}

#[cfg(feature = "db-stats")]
fn analyze_visit_dir<Tgt>(
    root: &path::Path,
    path: &path::Path,
    writers: &mut AnalyzeWriters,
    sample: usize,
    skip_read_errors: bool,
    bimap: &impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>,
    tiling_depth: Option<DimSize>,
    binary_scale_shapes: bool,
) where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    let mut page_actions = HashSet::new();
    let mut page_action_costs = HashSet::new();
    let mut block_actions = HashSet::new();
    let mut block_action_costs = HashSet::new();

    // Since we don't revisit blocks, bypass the in-mem. cache and read from disk.
    for file_entry in fs::read_dir(path).unwrap() {
        page_actions.clear();
        page_action_costs.clear();

        let file_entry = file_entry.unwrap();
        match file_entry.path().file_name().unwrap().to_str() {
            Some("PRECOMPUTE") | Some("TILING_DEPTH") => continue,
            _ => {}
        }

        let entry_path = file_entry.path();
        if entry_path.is_dir() {
            analyze_visit_dir::<Tgt>(
                root,
                &file_entry.path(),
                writers,
                sample,
                skip_read_errors,
                bimap,
                tiling_depth,
                binary_scale_shapes,
            );
            continue;
        }

        // Choose with 1/sample probability to skip this file.
        if sample > 1 && rand::random::<usize>() % sample != 0 {
            continue;
        }

        let shortened_entry_path_str = entry_path.strip_prefix(root).unwrap();
        let entry_path_str = format!("{}", shortened_entry_path_str.display());

        let table_key = superblock_key_from_path(shortened_entry_path_str);

        let superblock_file = fs::File::open(entry_path).unwrap();
        let superblock = match read_any_format(superblock_file) {
            Ok(r) => r,
            Err(e) => {
                if skip_read_errors {
                    log::warn!("Error reading superblock: {:?}", e);
                    continue;
                }
                panic!("Error reading superblock: {:?}", e);
            }
        };

        for (block_pt, block) in &superblock.contents {
            let DbBlock::Whole(e) = block else {
                todo!("Update dbstats to support RTree-based blocks");
            };

            block_actions.clear();
            block_action_costs.clear();

            // Print all (partial) Impls.
            let rank = block_pt.len();
            let db_shape = db_shape::<Tgt>(rank);
            let bs = block_shape(block_pt, &db_shape, block_size_dim).collect::<Vec<_>>();
            let block_top = bs.iter().map(|d| *d - 1).collect::<Vec<_>>();
            for unshifted_local_pt in crate::utils::diagonals(&block_top).flatten() {
                let shifted_local_pt = unshifted_local_pt
                    .iter()
                    .zip(block_pt)
                    .zip(&bs)
                    .map(|((l, b), shp)| l + shp * b)
                    .collect::<Vec<_>>();
                let unshifted_local_pt_usize = unshifted_local_pt
                    .iter()
                    .map(|v| usize::try_from(*v).unwrap())
                    .collect::<Vec<_>>();
                let spec = bimap.apply_inverse(&(table_key.clone(), shifted_local_pt));
                if let Some(action_cost_vec) = e.get(
                    &unshifted_local_pt_usize,
                    spec.0.volume(),
                    binary_scale_shapes,
                ) {
                    if action_cost_vec.len() > 1 {
                        log::warn!("k > 1 but only k = 1 supported");
                    }
                    if let Some((action_idx, cost)) = action_cost_vec.first() {
                        let action = spec
                            .0
                            .actions(tiling_depth)
                            .into_iter()
                            .nth((*action_idx).into())
                            .unwrap();
                        if block_actions.insert(Some(action.clone())) {
                            writers
                                .block_action_writer
                                .write_record([
                                    &entry_path_str,
                                    &format!("{:?}", block_pt),
                                    &format!("{:?}", action),
                                ])
                                .unwrap();
                        }
                        block_action_costs.insert(Some((action.clone(), cost.clone())));
                    } else {
                        if block_actions.insert(None) {
                            writers
                                .block_action_writer
                                .write_record([&entry_path_str, &format!("{:?}", block_pt), "_"])
                                .unwrap();
                        }
                        block_action_costs.insert(None);
                    }
                    block_actions.insert(action_cost_vec.iter().next().map(|&(action_idx, _)| {
                        spec.0
                            .actions(tiling_depth)
                            .into_iter()
                            .nth(action_idx.into())
                            .unwrap()
                    }));
                }
            }

            page_actions.extend(block_actions.iter().cloned());
            page_action_costs.extend(block_action_costs.iter().cloned());

            // We're printing only a `volume` column, so the expectation is that these are
            // equal (by definition).
            assert_eq!(e.filled.len(), e.main_costs.len());
            assert_eq!(e.filled.len(), e.peaks.len());
            assert_eq!(e.filled.len(), e.depths_actions.len());

            // Write the CSV line.
            writers
                .block_writer
                .write_record([
                    &entry_path_str,
                    &format!("{:?}", block_pt),
                    &e.filled.runs_len().to_string(),
                    &e.main_costs.runs_len().to_string(),
                    &e.peaks.runs_len().to_string(),
                    &e.depths_actions.runs_len().to_string(),
                    &block_actions.len().to_string(),
                    &block_action_costs.len().to_string(),
                    &e.depths_actions.len().to_string(),
                ])
                .unwrap();
        }

        writers
            .page_writer
            .write_record([
                entry_path_str,
                page_actions.len().to_string(),
                page_action_costs.len().to_string(),
            ])
            .unwrap();
    }
}

fn superblockify_pt(block_pt: &[BimapInt]) -> Vec<BimapInt> {
    block_pt.iter().map(|&i| i / SUPERBLOCK_FACTOR).collect()
}

fn construct_impl<Tgt>(db: &FilesDatabase, imp: &ImplNode<Tgt>) -> ImplNode<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    match imp {
        ImplNode::SpecApp(p) => db
            .get_impl(&p.0)
            .unwrap_or_else(|| panic!("Database should have the sub-Spec: {}", p.0))
            .first()
            .unwrap_or_else(|| panic!("Database sub-Spec should be satisfiable: {}", p.0))
            .clone(),
        _ => imp.replace_children(imp.children().iter().map(|c| construct_impl(db, c))),
    }
}

fn block_size_dim(dim: usize, dim_count: usize) -> u32 {
    if dim >= dim_count - LEVEL_COUNT {
        31
    } else {
        4
    }
}

/// Convert a single dimension of a global point to a block and within-block index.
fn db_key_scale(dim: usize, value: BimapInt, dim_count: usize) -> (BimapInt, u8) {
    // TODO: Autotune rather than hardcode these arbitrary dimensions.
    let (quotient, remainder) = value.div_rem(&block_size_dim(dim, dim_count));
    (quotient, remainder.try_into().unwrap())
}

/// Computes the shape of a block in the database.
///
/// Given a maximum block shape and the coordinate of that block in the database, this will
/// truncate at the edges. If the given database shape is `None` in that dimension, that dimension
/// will be the full block size.
fn block_shape<'a, F>(
    block_pt: &'a [u32],
    db_shape: &'a [Option<NonZeroUsize>],
    block_max_size_fn: F,
) -> impl ExactSizeIterator<Item = BimapInt> + 'a
where
    F: Fn(usize, usize) -> u32 + 'a,
{
    let rank = db_shape.len();
    assert_eq!(block_pt.len(), rank);
    db_shape.iter().enumerate().map(move |(i, db_max_option)| {
        let full_block_size = block_max_size_fn(i, rank);
        if let Some(db_max) = db_max_option {
            let remaining_size =
                u32::try_from(db_max.get()).unwrap() - block_pt[i] * full_block_size;
            remaining_size.min(full_block_size)
        } else {
            full_block_size
        }
    })
}

fn db_shape<Tgt: Target>(rank: usize) -> Vec<Option<NonZeroUsize>> {
    let mut shape = vec![None; rank];
    let MemoryLimits::Standard(m) = Tgt::max_mem();
    for (level_idx, dest_idx) in ((rank - m.len())..rank).enumerate() {
        shape[dest_idx] =
            Some(NonZeroUsize::new((m.get_binary_scaled(level_idx) + 1).into()).unwrap());
    }
    shape
}

/// Converts a given global coordinate into block and within-block coordinates.
fn blockify_point(mut pt: Vec<BimapInt>) -> (Vec<BimapInt>, Vec<u8>) {
    let rank = pt.len();
    let mut inner_pt = Vec::with_capacity(rank);
    for (i, d) in pt.iter_mut().enumerate() {
        let (outer, inner) = db_key_scale(i, *d, rank);
        *d = outer;
        inner_pt.push(inner);
    }
    (pt, inner_pt)
}

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt, B>(
    bimap: &B,
    spec: &Spec<Tgt>,
    impls: &Vec<(ActionIdx, Cost)>,
) -> (TableKey, (Vec<BimapInt>, Vec<BimapInt>))
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    B: BiMap<Domain = Spec<Tgt>, Codomain = DbKey>,
{
    // Compute the per-level maximum limits of the solutions. This lower bounds the range.
    let mut per_level_peaks = [0; LEVEL_COUNT];
    for (_, cost) in impls {
        for (i, peak) in cost.peaks.iter().enumerate() {
            per_level_peaks[i] = per_level_peaks[i].max(peak);
        }
    }

    // Compute the complete upper and lower bounds from the given Spec and that Spec modified with
    // the peaks' bound (computed above).
    let upper_inclusive = bimap.apply(spec);
    let lower_inclusive = {
        let mut lower_bound_spec = spec.clone();
        lower_bound_spec.1 = MemoryLimits::Standard(MemVec::new(per_level_peaks));
        bimap.apply(&lower_bound_spec)
    };

    // TODO: This computes the non-memory dimensions of the key/coordinates twice. Avoid that.
    debug_assert_eq!(upper_inclusive.0, lower_inclusive.0);
    debug_assert_eq!(
        upper_inclusive.1[..upper_inclusive.1.len() - LEVEL_COUNT],
        lower_inclusive.1[..lower_inclusive.1.len() - LEVEL_COUNT]
    );

    (upper_inclusive.0, (lower_inclusive.1, upper_inclusive.1))
}

/// Iterate blocks of an integer range.
///
/// Yields block indices along with a range of within-block indices. For example:
/// ```
/// # use morello::db::iter_blocks_in_single_dim_range;
/// assert_eq!(iter_blocks_in_single_dim_range(0, 3, 4).collect::<Vec<_>>(),
///           vec![(0, 0..4)]);
/// assert_eq!(iter_blocks_in_single_dim_range(1, 7, 4).collect::<Vec<_>>(),
///            vec![(0, 1..4), (1, 0..4)]);
/// ```
///
/// Given indices `global_bottom` and `global_top` are inclusive, forming a
/// closed range. For example, the following yields a single incomplete block:
/// ```
/// # use morello::db::iter_blocks_in_single_dim_range;
/// assert_eq!(iter_blocks_in_single_dim_range(4, 4, 4).collect::<Vec<_>>(),
///            vec![(1, 0..1)]);
/// ```
// TODO: Make private. (Will break doctests.)
pub fn iter_blocks_in_single_dim_range(
    global_bottom: BimapInt,
    global_top: BimapInt,
    block_dim_size: BimapInt,
) -> impl Iterator<Item = (BimapInt, Range<BimapInt>)> + Clone {
    debug_assert_ne!(block_dim_size, 0);
    debug_assert!(global_bottom <= global_top);

    // Change global_top to a non-inclusive upper bound.
    let global_top = global_top + 1;
    debug_assert!(global_top > global_bottom);

    // Compute half-open range of blocks.
    let block_bottom = global_bottom / block_dim_size;
    let block_top = (global_top + block_dim_size - 1) / block_dim_size;
    debug_assert!(block_top > block_bottom);

    let last_block_is_complete = global_top % block_dim_size == 0;
    let one_block_only = block_bottom == block_top - 1;

    let prefix: Option<(BimapInt, Range<_>)>;
    let suffix: Option<(BimapInt, Range<_>)>;
    let s = global_bottom % block_dim_size;
    if one_block_only {
        let e = (s + (global_top - global_bottom)).min(block_dim_size);
        prefix = Some((block_bottom, s..e));
        suffix = None;
    } else {
        let e = global_top % block_dim_size;
        prefix = Some((block_bottom, s..block_dim_size));
        if e == 0 {
            suffix = None;
        } else {
            suffix = Some((block_top - 1, 0..e));
        }
    }

    let mut block_top_full = block_top;
    if !last_block_is_complete {
        block_top_full -= 1;
    }
    let full_blocks_iter =
        ((block_bottom + 1)..block_top_full).map(move |block_idx| (block_idx, 0..block_dim_size));

    prefix.into_iter().chain(full_blocks_iter).chain(suffix)
}

fn superblock_file_path(root: &Path, superblock_key: &SuperBlockKey) -> path::PathBuf {
    let ((spec_key, table_key_rest), block_pt) = superblock_key;
    let spec_key_dir_name = match spec_key {
        SpecKey::Matmul { dtypes } => root
            .join("Matmul")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Conv { dtypes } => root
            .join("Conv")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Move { dtypes } => root
            .join("Move")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Zero { dtype } => root.join("Zero").join(dtype.to_string()),
    };
    spec_key_dir_name
        .join(table_key_rest.iter().map(|(l, _, _)| l).join("_"))
        .join(table_key_rest.iter().map(|(_, d, _)| d).join("_"))
        .join(table_key_rest.iter().map(|(_, _, v)| v).join("_"))
        .join(block_pt.iter().map(|p| p.to_string()).join("_"))
}

#[cfg(feature = "db-stats")]
fn superblock_key_from_path(path: &path::Path) -> TableKey {
    let components = path.components().collect::<Vec<_>>();
    for start in 0..components.len() {
        if let Ok(k) = superblock_key_from_subpath(&components[start..]) {
            return k;
        }
    }
    panic!("Could not parse superblock key from path: {path:?}");
}

#[cfg(feature = "db-stats")]
fn superblock_key_from_subpath(components: &[path::Component]) -> Result<TableKey, ()> {
    if components.len() < 5 {
        return Err(());
    }

    // Collect dtypes.
    let mut dtypes = vec![];
    for subpart in into_normal_component(&components[1])?.split('_') {
        match subpart {
            "u8" => dtypes.push(Dtype::Uint8),
            "i8" => dtypes.push(Dtype::Sint8),
            "u16" => dtypes.push(Dtype::Uint16),
            "i16" => dtypes.push(Dtype::Sint16),
            "u32" => dtypes.push(Dtype::Uint32),
            "i32" => dtypes.push(Dtype::Sint32),
            "f32" => dtypes.push(Dtype::Float32),
            "bf16" => dtypes.push(Dtype::Bfloat16),
            _ => return Err(()),
        }
    }

    // Collect SpecKey.
    let spec_key = match into_normal_component(&components[0])? {
        "Matmul" => SpecKey::Matmul {
            dtypes: dtypes.try_into().unwrap(),
        },
        "Conv" => SpecKey::Conv {
            dtypes: dtypes.try_into().unwrap(),
        },
        "Move" => SpecKey::Move {
            dtypes: dtypes.try_into().unwrap(),
        },
        "Zero" => SpecKey::Zero {
            dtype: dtypes.into_iter().exactly_one().unwrap(),
        },
        _ => return Err(()),
    };

    // Collect Layouts.
    let layouts: Vec<Layout> = parse_layouts_component(into_normal_component(&components[2])?)?;

    // Collect levels-as-integers and vector sizes-as-integers.
    let levels = parse_underscored_int_tuple(into_normal_component(&components[3])?)?;
    let vector_sizes = parse_underscored_int_tuple(into_normal_component(&components[4])?)?;

    Ok((
        spec_key,
        layouts
            .into_iter()
            .zip(levels.into_iter().map(|x| x.try_into().unwrap()))
            .zip(vector_sizes)
            .map(|((la, le), v)| (la, le, v))
            .collect(),
    ))
}

#[cfg(feature = "db-stats")]
fn into_normal_component<'a>(component: &'a path::Component) -> Result<&'a str, ()> {
    match component {
        path::Component::Prefix(_)
        | path::Component::RootDir
        | path::Component::CurDir
        | path::Component::ParentDir => Err(()),
        path::Component::Normal(part) => {
            let Some(part) = part.to_str() else {
                return Err(());
            };
            Ok(part)
        }
    }
}

#[cfg(any(feature = "db-stats", test))]
fn parse_layouts_component(part: &str) -> Result<Vec<Layout>, ()> {
    let mut layouts = vec![];
    for layout_str in part.split('_') {
        if layout_str.len() < 2 {
            return Err(());
        }

        if layout_str == "RM" {
            // TODO: Use the correct tensor rank
            layouts.push(row_major(2));
        } else if layout_str.starts_with('[') {
            let mut layout_core = vec![];
            for physical_idx_str in layout_str[1..layout_str.len() - 1].split(',') {
                let physical_idx = physical_idx_str.parse().map_err(|_| ())?;
                layout_core.push((physical_idx, PhysDim::Dynamic));
            }
            layouts.push(Layout::new(layout_core));
        } else {
            let mut layout_split = layout_str
                .trim_start_matches("<[")
                .trim_end_matches("]>")
                .split("], [");

            let left = layout_split.next().unwrap();
            let mut order = vec![];
            for o_str in left.split(',') {
                order.push(o_str.parse().map_err(|_| ())?);
            }

            let right = layout_split.exactly_one().unwrap();
            let mut physdims = vec![];
            for physdim_str in right.split(", ") {
                if physdim_str == "Dynamic" {
                    physdims.push(PhysDim::Dynamic);
                } else if physdim_str.starts_with("Packed(") {
                    let packed_size = physdim_str[7..physdim_str.len() - 1]
                        .parse()
                        .map_err(|_| ())?;
                    physdims.push(PhysDim::Packed(packed_size));
                } else if physdim_str.starts_with("OddEven(") {
                    let oddeven_size = physdim_str[8..physdim_str.len() - 1]
                        .parse()
                        .map_err(|_| ())?;
                    physdims.push(PhysDim::OddEven(oddeven_size));
                } else {
                    return Err(());
                }
            }

            layouts.push(Layout::new(order.into_iter().zip(physdims).collect()));
        }
    }
    Ok(layouts)
}

#[cfg(feature = "db-stats")]
fn parse_underscored_int_tuple(input: &str) -> Result<Vec<u32>, ()> {
    input
        .split('_')
        .map(|s| s.parse().map_err(|_| ()))
        .collect()
}

// For some reason, [Prehashed]'s [Clone] impl requires that the value be [Copy].
fn prehashed_clone<T: Clone>(value: &Prehashed<T>) -> Prehashed<T> {
    let (inner, h) = Prehashed::as_parts(value);
    Prehashed::new(inner.clone(), *h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        imp::visit_leaves,
        memorylimits::{MemVec, MemoryLimits},
        scheduling::ApplyError,
        spec::arb_canonical_spec,
        target::X86Target,
        utils::{bit_length, bit_length_inverse},
    };
    use itertools::Itertools;
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::fmt;

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);
    const TEST_SMALL_MEM: u64 = 256;

    // TODO: What about leaves!? This shouldn't be called `Decision`.
    #[derive(Clone)]
    struct Decision<Tgt: Target> {
        spec: Spec<Tgt>,
        actions_costs: Vec<(ActionIdx, Cost)>,
        children: Vec<Decision<Tgt>>,
    }

    type DecisionNode<Tgt> = (Spec<Tgt>, Vec<(ActionIdx, Cost)>);

    impl<Tgt: Target> Decision<Tgt> {
        /// Yield all nested [Spec] and actions-costs, bottom-up.
        fn consume_decisions(self) -> Box<dyn Iterator<Item = DecisionNode<Tgt>>> {
            Box::new(
                self.children
                    .into_iter()
                    .flat_map(|c| c.consume_decisions())
                    .chain(std::iter::once((self.spec, self.actions_costs))),
            )
        }
    }

    impl<Tgt: Target> fmt::Debug for Decision<Tgt> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Decision")
                .field("spec", &self.spec)
                .field("actions_costs", &self.actions_costs)
                .finish_non_exhaustive()
        }
    }

    impl<Tgt: Target> fmt::Display for Decision<Tgt> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} := {:?}", self.spec, self.actions_costs)
        }
    }

    #[test]
    fn test_block_shape() {
        let db_shape = [4, 7]
            .into_iter()
            .map(|v| Some(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            block_shape(&[0, 0], &db_shape, |_, _| 2)
                .collect_vec()
                .as_slice(),
            &[2, 2]
        );
        assert_eq!(
            block_shape(&[1, 1], &db_shape, |_, _| 2)
                .collect_vec()
                .as_slice(),
            &[2, 2]
        );
        assert_eq!(
            block_shape(&[1, 3], &db_shape, |_, _| 2)
                .collect_vec()
                .as_slice(),
            &[2, 1]
        );
    }

    #[test]
    fn test_parse_layouts_component() {
        assert_eq!(
            parse_layouts_component("[1,0]_<[0,1,0], [Dynamic, Dynamic, OddEven(16)]>"),
            Ok(vec![
                Layout::new(vec![(1, PhysDim::Dynamic), (0, PhysDim::Dynamic)]),
                Layout::new(vec![
                    (0, PhysDim::Dynamic),
                    (1, PhysDim::Dynamic),
                    (0, PhysDim::OddEven(DimSize::new(16).unwrap()))
                ])
            ])
        );
    }

    proptest! {
        #[test]
        fn test_iter_blocks_in_single_dim_range(
            start in 0..12u32, extent in 0..8u32, block_dim_size in 1..4u32
        ) {
            let end = start + extent;
            let mut visited_indices = Vec::with_capacity(extent as usize + 1);
            for (block_idx, within_block_range) in
                iter_blocks_in_single_dim_range(start, end, block_dim_size)
            {
                for wpi in within_block_range {
                    let reversed_idx = block_dim_size * block_idx + wpi;
                    visited_indices.push(reversed_idx);
                }
            }
            assert_eq!(visited_indices, (start..=end).collect::<Vec<_>>());
        }

        #[test]
        fn test_iter_blocks_in_single_dim_range_yields_block_in_order(
            start in 0..12u32, extent in 0..8u32, block_dim_size in 1..4u32
        ) {
            let end = start + extent;
            let mut block_idxs =
                iter_blocks_in_single_dim_range(start, end, block_dim_size).map(|(block_idx, _)| block_idx);
            if let Some(mut last_block_idx) = block_idxs.next() {
                for block_idx in block_idxs {
                    assert!(block_idx == last_block_idx + 1);
                    last_block_idx = block_idx;
                }
            }
        }

        #[test]
        fn test_simple_put_then_get_works(
            decision in arb_spec_and_decision::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let top_spec = decision.spec.clone();
            let top_actions_costs = decision.actions_costs.clone();
            let db = FilesDatabase::new(None, false, 1, 2, 1, None);
            for (spec, actions_costs) in decision.consume_decisions() {
                db.put(spec, actions_costs);
            }
            let expected = ActionCostVec(top_actions_costs);
            let get_result = db.get(&top_spec).expect("Spec should be in database");
            assert_eq!(get_result, expected, "Entries differed at {}", top_spec);
        }

        // TODO: Add tests for top-2, etc. Impls
        #[test]
        fn test_put_then_get_fills_across_memory_limits(
            decision in arb_spec_and_decision::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let MemoryLimits::Standard(spec_limits) = decision.spec.1.clone();
            let db = FilesDatabase::new(None, false, 1, 128, 1, None);

            let top_logical_spec = decision.spec.0.clone();
            let top_actions_costs = decision.actions_costs.clone();

            // Put all decisions into database.
            for (spec, action_costs) in decision.consume_decisions() {
                db.put(spec, action_costs);
            }

            let peaks = if let Some((_, c)) = top_actions_costs.first() {
                c.peaks.clone()
            } else {
                MemVec::zero::<X86Target>()
            };
            let filled_limits_iter = spec_limits
                .iter()
                .zip(peaks.iter())
                .map(|(l, p)| {
                    assert!(l == 0 || l.is_power_of_two());
                    assert!(p == 0 || p.is_power_of_two());
                    bit_length(p)..=bit_length(l)
                })
                .multi_cartesian_product();
            let expected = ActionCostVec(top_actions_costs);
            for limit_to_check_bits in filled_limits_iter {
                let limit_to_check_vec = limit_to_check_bits.iter().copied().map(bit_length_inverse).collect::<Vec<_>>();
                let limit_to_check = MemoryLimits::Standard(MemVec::new(limit_to_check_vec.try_into().unwrap()));
                let spec_to_check = Spec(top_logical_spec.clone(), limit_to_check);
                let get_result = db.get(&spec_to_check).expect("Spec should be in database");
                assert_eq!(get_result, expected, "Entries differed at {}", spec_to_check);
            }
        }

        // TODO: Fix and re-enable this test.
        //
        // #[test]
        // fn test_two_puts_return_correct_gets_for_second_put(
        //     decision_pair in arb_spec_and_decision_pair::<X86Target>()
        // ) {
        //     let db = RocksDatabase::try_new(None, false, 1).unwrap();
        //     let (decision_a, decision_b) = decision_pair;
        //     assert!(decision_a.actions_costs.len() < 2 && decision_b.actions_costs.len() < 2);
        //     let logical_specs_match = decision_a.spec.0 == decision_b.spec.0;

        //     let MemoryLimits::Standard(spec_limits_a) = &decision_a.spec.1;
        //     let MemoryLimits::Standard(spec_limits_b) = &decision_b.spec.1;

        //     // TODO: This doesn't make sense because Database's are only supposed to take the
        //     //   optimal Impl per Spec. This will feed in arbitrary Impls, so we can get weird
        //     //   results where the put overwrites already-filled entries with worse solutions.
        //     for d in [&decision_a, &decision_b] {
        //         d.visit_decisions().for_each(|d| {
        //             db.put(d.spec.clone(), d.actions_costs.clone().into());
        //         });
        //     }

        //     let mut decision_peaks = Vec::with_capacity(2);
        //     for d in [&decision_a, &decision_b] {
        //         decision_peaks.push(if let Some((_, c)) = d.actions_costs.first() {
        //             c.peaks.clone()
        //         } else {
        //             MemVec::zero::<X86Target>()
        //         });
        //     }

        //     // TODO: Use the binary-scaled values directly rather than converting back and forth.
        //     let relevant_limits_iter = spec_limits_a.iter().zip(spec_limits_b.iter())
        //         .zip(decision_peaks[0].iter().zip(decision_peaks[1].iter()))
        //         .map(|((l_a, l_b), (p_a, p_b))| {
        //             (bit_length(p_a.min(p_b))..=bit_length(l_a.max(l_b))).map(bit_length_inverse)
        //         })
        //         .multi_cartesian_product();

        //     for limit_to_check in relevant_limits_iter {
        //         // b was put second, so if we're in its range, that should be the result. Check a
        //         // second.
        //         let limit_in_a = decision_peaks[0].iter().zip(spec_limits_a.iter()).zip(limit_to_check.iter()).all(|((bot, top), p)| {
        //             assert!(bot <= top);
        //             bot <= *p && *p <= top
        //         });
        //         let limit_in_b = decision_peaks[1].iter().zip(spec_limits_b.iter()).zip(limit_to_check.iter()).all(|((bot, top), p)| {
        //             assert!(bot <= top);
        //             bot <= *p && *p <= top
        //         });

        //         let expected_value = if limit_in_b {
        //             Some(ActionCostVec(decision_b.actions_costs.clone().into()))
        //         } else if limit_in_a && logical_specs_match {
        //             Some(ActionCostVec(decision_a.actions_costs.clone().into()))
        //         } else {
        //             None
        //         };

        //         let spec_to_check = Spec(
        //             decision_b.spec.0.clone(),
        //             MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())),
        //         );
        //         let get_result = db.get(&spec_to_check);
        //         match (get_result.as_ref(), expected_value.as_ref()) {
        //             (Some(g), Some(e)) if g == e => {}
        //             (None, None) => {}
        //             _ => {
        //                 eprintln!("First-inserted Spec: {}", decision_a.spec);
        //                 eprintln!("Last-inserted Spec: {}", decision_b.spec);
        //                 let expected_description = if limit_in_b {
        //                     "second value".to_string()
        //                 } else if limit_in_a {
        //                     "first value".to_string()
        //                 } else {
        //                     format!("{:?}", expected_value)
        //                 };
        //                 let result_description = match get_result {
        //                     Some(g) => format!("Some({g:?})"),
        //                     None => "None".to_string(),
        //                 };
        //                 panic!(
        //                     "Incorrect get result at {}. Expected {} but got {}",
        //                     spec_to_check, expected_description, result_description
        //                 )
        //             }
        //         }
        //     }
        // }
    }

    fn arb_spec_and_decision<Tgt: Target>(
        max_size: Option<DimSize>,
        max_memory: Option<u64>,
    ) -> impl Strategy<Value = Decision<Tgt>> {
        arb_canonical_spec::<Tgt>(max_size, max_memory)
            .prop_flat_map(|spec| {
                let valid_actions = spec
                    .0
                    .actions(None)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, a)| match a.apply(&spec) {
                        Ok(applied) => Some((ActionIdx::from(u16::try_from(i).unwrap()), applied)),
                        Err(ApplyError::ActionNotApplicable(_)) => None,
                        Err(ApplyError::OutOfMemory) => None,
                        Err(ApplyError::SpecNotCanonical) => {
                            unreachable!("Non-canonical Specs should be filtered")
                        }
                    })
                    .collect::<Vec<_>>();
                let action_idx_strategy = if valid_actions.is_empty() {
                    Just(None).boxed()
                } else {
                    prop_oneof![
                        1 => Just(None),
                        2 => proptest::sample::select(valid_actions).prop_map(Some),
                    ]
                    .boxed()
                };
                (Just(spec), action_idx_strategy)
            })
            .prop_map(|(spec, action_opt)| {
                if let Some((action_idx, imp)) = action_opt {
                    recursively_decide_with_action(&spec, action_idx, &imp)
                } else {
                    Decision {
                        spec,
                        actions_costs: vec![],
                        children: vec![],
                    }
                }
            })
    }

    // TODO: Implement arb_spec_and_decision_pair and use that everywhere instead.
    // fn arb_spec_and_decision_pair<Tgt: Target>(
    // ) -> impl Strategy<Value = (Decision<Tgt>, Decision<Tgt>)> {
    //     arb_spec_and_decision().prop_flat_map(|first| {
    //         let second_term = prop_oneof![
    //             2 => Just(first.clone()),
    //             2 => {
    //                 use crate::memorylimits::arb_memorylimits;
    //                 let MemoryLimits::Standard(max_memory) = Tgt::max_mem();
    //                 let first_logical = first.spec.0.clone();
    //                 arb_memorylimits::<Tgt>(&max_memory).prop_map(move |spec_limits| {
    //                     recursively_decide_actions(&Spec(first_logical.clone(), spec_limits))
    //                 })
    //             },
    //             1 => arb_spec_and_decision(),
    //         ];
    //         (Just(first), second_term).boxed()
    //     })
    // }

    /// Will return a [Decision] by choosing the first action for the Spec (if any) and recursively
    /// choosing the first action for all child Specs in the resulting partial Impl.
    fn recursively_decide_actions<Tgt: Target>(spec: &Spec<Tgt>) -> Decision<Tgt> {
        if let Some((action_idx, partial_impl)) = spec
            .0
            .actions(None)
            .into_iter()
            .enumerate()
            .filter_map(|(i, a)| match a.apply(spec) {
                Ok(imp) => Some((i, imp)),
                Err(ApplyError::ActionNotApplicable(_) | ApplyError::OutOfMemory) => None,
                Err(ApplyError::SpecNotCanonical) => panic!(),
            })
            .next()
        {
            recursively_decide_with_action(spec, action_idx.try_into().unwrap(), &partial_impl)
        } else {
            Decision {
                spec: spec.clone(),
                actions_costs: vec![],
                children: vec![],
            }
        }
    }

    fn recursively_decide_with_action<Tgt: Target>(
        spec: &Spec<Tgt>,
        action_idx: ActionIdx,
        partial_impl: &ImplNode<Tgt>,
    ) -> Decision<Tgt> {
        let mut children = Vec::new();
        let mut unsat = false;
        visit_leaves(partial_impl, &mut |leaf| {
            if let ImplNode::SpecApp(spec_app) = leaf {
                let cd = recursively_decide_actions(&spec_app.0);
                if cd.actions_costs.is_empty() {
                    unsat = true;
                    return false;
                }
                children.push(cd);
            }
            true
        });
        if unsat {
            return Decision {
                spec: spec.clone(),
                actions_costs: vec![],
                children: vec![],
            };
        }

        let cost = Cost::from_child_costs(
            partial_impl,
            &children
                .iter()
                .map(|c| {
                    assert_eq!(c.actions_costs.len(), 1);
                    c.actions_costs[0].1.clone()
                })
                .collect::<Vec<_>>(),
        );
        Decision {
            spec: spec.clone(),
            actions_costs: vec![(action_idx, cost)],
            children,
        }
    }
}
