mod pagecontents;

use crate::common::DimSize;
use crate::cost::{Cost, CostIntensity, NormalizedCost};
use crate::datadeps::SpecKey;
use crate::db::pagecontents::PageContents;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{AsBimap, BiMap, SurMap};
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::imp::ImplNode;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::reconstruct::reconstruct_impls_from_actions;
use crate::rtree::RTreeDyn;
use crate::spatial_query::SpatialQuery;
use crate::spec::{FillValue, LogicalSpecSurMap, PrimitiveBasicsBimap, Spec, SpecSurMap};
use crate::target::{Target, TargetId, MEMORY_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;
use crate::utils::multi_range_product;
use pagecontents::RTreePageContents;

use atomic_write_file::AtomicWriteFile;
use itertools::Itertools;
use parking_lot::{Mutex, MutexGuard};
use prehash::{new_prehashed_set, DefaultPrehasher, Prehashed, PrehashedSet, Prehasher};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::fs;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Range};
use std::path::{self, Path};
use std::sync::{
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
    mpsc, Arc,
};
use std::time::{Duration, Instant};
use wtinylfu::WTinyLfuCache;

#[cfg(feature = "db-stats")]
use std::sync::atomic::{self, AtomicU64};

type DbKey = (TableKey, Vec<BimapInt>);
type TableKey = (SpecKey, Vec<()>);
type PageKey = DbKey;
type SpatialQueryPageResult = (Vec<BimapSInt>, Vec<BimapSInt>, Option<NormalizedCost>);
type DbValue = Option<(CostIntensity, MemVec, u8, ActionNum)>;
pub type ActionNum = u16;

#[cfg(test)]
type MemoizedThroughputsByPoint =
    std::collections::HashMap<Vec<BimapInt>, Option<crate::cost::CostIntensity>>;
#[cfg(test)]
type MemoizedThroughputsByTable = std::collections::HashMap<TableKey, MemoizedThroughputsByPoint>;

/// The number of shards/locks per thread.
const THREAD_SHARDS: usize = 2;
const CHANNEL_SIZE: usize = 2;
/// Compress pages when writing to disk.
const COMPRESS_PAGES: bool = true;
const NONSPATIAL_CACHE_FILE: &str = "NONSPATIAL_CACHE";
const DEFAULT_PROACTIVE_SAVE_INTERVAL: Duration = Duration::from_secs(60);
const BACKGROUND_SAVE_SHARDS_PER_PUT: usize = THREAD_SHARDS;
const BACKGROUND_SAVE_PAGES_PER_SHARD: usize = CHANNEL_SIZE;
const TARGET_FILE: &str = "TARGET";
const TILESCALE_FILE: &str = "TILESCALE";

pub struct FilesDatabase {
    #[allow(dead_code)] // read only when db-stats enabled; otherwise only affects Drop
    dir_handle: Arc<DirPathHandle>,
    tile_scale: TileScale,
    k: u8,
    shards: ShardVec,
    nonspatial_cache: Mutex<NonSpatialCache>,
    prehasher: DefaultPrehasher,
    proactive_saves_enabled: bool,
    proactive_save_interval: Duration,
    next_proactive_save_shard: AtomicUsize,
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
    pub(crate) page_id: Vec<BimapInt>, // TODO: Rename to page_pt
}

#[derive(Debug, Clone)]
struct Page {
    contents: PageContents,
    modified: bool,
    last_save: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct NonSpatialKey {
    spec: Vec<u8>,
}

struct NonSpatialCache {
    entries: HashMap<NonSpatialKey, ActionCostVec>,
    modified: bool,
    last_save: Option<Instant>,
}

struct Shard {
    cache: WTinyLfuCache<Prehashed<PageKey>, Page>,
    modified_pages: PrehashedSet<PageKey>,
    modified_page_queue: VecDeque<Prehashed<PageKey>>,
    outstanding_gets: PrehashedSet<PageKey>,
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
    Get(Prehashed<PageKey>),
    Put(PageKey, PageContents),
    PutNonSpatial(path::PathBuf, HashMap<NonSpatialKey, ActionCostVec>),
    Flush(mpsc::Sender<()>),
    Exit,
}

enum ShardThreadResponse {
    Loaded(Prehashed<PageKey>, Page),
}

/// Contains a path to a directory or a [tempfile::TempDir]. This facilitates deleting the
/// temporary directory on drop.
enum DirPathHandle {
    Persisted(path::PathBuf),
    TempDir(tempfile::TempDir),
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCostVec(pub Vec<(ActionNum, Cost)>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TileScale {
    /// Encode each dimension `d` as `d - 1`.
    Linear,
    /// Use powers of two only.
    PowerOfTwo,
    /// Use sizes representable as `2^x` or `3 * 2^x`.
    PowerOrThreePower,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ActionNormalizedCostVec(pub Vec<(ActionNum, NormalizedCost)>);

#[derive(Debug, Clone)]
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
    pub fn new<Tgt: Target>(
        file_path: Option<&path::Path>,
        tile_scale: TileScale,
        k: u8,
        cache_size: usize,
        thread_count: usize,
    ) -> Self {
        let dir_handle = Arc::new(match file_path {
            Some(path) => {
                fs::create_dir_all(path).unwrap();
                DirPathHandle::Persisted(path.to_owned())
            }
            None => DirPathHandle::TempDir(tempfile::TempDir::new().unwrap()),
        });
        log::info!("Opening database at: {}", dir_handle.path().display());

        ensure_target_file::<Tgt>(dir_handle.path()).expect("Target validation failed");
        ensure_tilescale_file(dir_handle.path(), tile_scale).expect("TILESCALE validation failed");

        Self::new_with_dir_handle(
            dir_handle,
            tile_scale,
            k,
            cache_size,
            thread_count,
            file_path.is_some(),
        )
    }

    /// Open an existing database.
    ///
    /// Unlike [FilesDatabase::new], this does not require specifying the [Target], but
    /// will not create the database if it does not already exist.
    pub fn open(
        file_path: Option<&path::Path>,
        tile_scale: TileScale,
        k: u8,
        cache_size: usize,
        thread_count: usize,
    ) -> Result<Self, String> {
        let path = file_path
            .ok_or_else(|| "Cannot open non-existent database without a path".to_string())?;
        if !path.exists() {
            return Err(format!("Database path does not exist: {}", path.display()));
        }
        let dir_handle = Arc::new(DirPathHandle::Persisted(path.to_owned()));
        log::info!("Opening database at: {}", dir_handle.path().display());
        validate_tilescale_file(dir_handle.path(), tile_scale)?;
        Ok(Self::new_with_dir_handle(
            dir_handle,
            tile_scale,
            k,
            cache_size,
            thread_count,
            true,
        ))
    }

    /// Common logic for [FilesDatabase::new] and [FilesDatabase::open].
    fn new_with_dir_handle(
        dir_handle: Arc<DirPathHandle>,
        tile_scale: TileScale,
        k: u8,
        cache_size: usize,
        thread_count: usize,
        proactive_saves_enabled: bool,
    ) -> Self {
        #[cfg(feature = "db-stats")]
        let stats = Arc::new(FilesDatabaseStats::default());

        let shard_count = thread_count * THREAD_SHARDS;
        let cache_size_per_shard = cache_size / shard_count;
        let cache_per_shard_samples = (cache_size_per_shard / 4).max(1);
        let cache_per_shard_size = cache_size_per_shard
            .saturating_sub(cache_per_shard_samples)
            .max(1);
        let actual_total_cache_size =
            shard_count * (cache_per_shard_size + cache_per_shard_samples);
        if actual_total_cache_size != cache_size {
            log::warn!("Database using cache size: {actual_total_cache_size}");
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
        let nonspatial_cache = Mutex::new(NonSpatialCache {
            entries: read_nonspatial_cache(dir_handle.path()),
            modified: false,
            last_save: None,
        });
        Self {
            dir_handle,
            tile_scale,
            k,
            shards,
            nonspatial_cache,
            prehasher: DefaultPrehasher::default(),
            proactive_saves_enabled,
            proactive_save_interval: DEFAULT_PROACTIVE_SAVE_INTERVAL,
            next_proactive_save_shard: AtomicUsize::new(0),
            #[cfg(feature = "db-stats")]
            stats,
        }
    }

    pub fn get<Tgt>(&self, query: &Spec<Tgt>) -> Option<ActionCostVec>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        match self.get_with_preference(query) {
            GetPreference::Hit(v) => Some(v),
            GetPreference::Miss(_) => None,
        }
    }

    pub fn set_proactive_saves_enabled(&mut self, enabled: bool) {
        self.proactive_saves_enabled = enabled;
    }

    pub fn proactive_save_interval(&self) -> Duration {
        self.proactive_save_interval
    }

    pub fn set_proactive_save_interval(&mut self, interval: Duration) {
        self.proactive_save_interval = interval;
    }

    pub fn get_impl<Tgt>(&self, query: &Spec<Tgt>) -> Option<Vec<ImplNode<Tgt>>>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        Some(reconstruct_impls_from_actions(
            &|spec: &Spec<Tgt>| self.get(spec),
            query,
            self.get(query)?,
        ))
    }

    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
    ) -> GetPreference<ActionCostVec, Vec<ActionNum>>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        #[cfg(feature = "db-stats")]
        self.stats.gets.fetch_add(1, atomic::Ordering::Relaxed);

        let mut query = query.clone();
        query.canonicalize().unwrap();
        self.get_with_preference_canon(&query)
    }

    /// Like [Self::get_with_preference], but assumes `query` is already canonical.
    ///
    /// Passing a non-canonical [Spec] is a logic error.
    pub(crate) fn get_with_preference_canon<Tgt>(
        &self,
        query: &Spec<Tgt>,
    ) -> GetPreference<ActionCostVec, Vec<ActionNum>>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(query.is_canonical());

        if !self.can_memoize_efficiently(query) {
            let key = self.nonspatial_key(query);
            return match self.nonspatial_cache.lock().entries.get(&key).cloned() {
                Some(v) => GetPreference::Hit(v),
                None => GetPreference::Miss(None),
            };
        }

        let bimap = self.spec_bimap();
        let (table_key, global_pt) = BiMap::apply(&bimap, query);

        let page_pt = blockify_point(&global_pt);
        let page_key = self.prehasher.prehash((table_key, page_pt));

        let page: &Page = &self.load_live_page(&page_key);
        page.contents.get_with_preference(query, &global_pt)
    }

    /// Looks up optima for canonical [Spec]s that all belong to the same database page.
    ///
    /// Results will be returned in the same order as `queries`, and passing non-canoncial [Spec]s
    /// in `queries` is a logic error.
    ///
    /// This builds one query tree for the requested points and intersects it with the page's
    /// memoized result tree, avoiding one point lookup per query [Spec].
    pub(crate) fn get_same_page_many_canon<Tgt>(
        &self,
        queries: &[&Spec<Tgt>],
    ) -> Vec<GetPreference<ActionCostVec, Vec<ActionNum>>>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let Some(first_query) = queries.first() else {
            return Vec::new();
        };

        let bimap = self.spec_bimap();
        debug_assert!(first_query.is_canonical());
        let (table_key, first_global_pt) = BiMap::apply(&bimap, first_query);
        let page_key = self
            .prehasher
            .prehash((table_key, blockify_point(&first_global_pt)));

        let mut query_points = Vec::with_capacity(queries.len());
        let first_query_pt = first_global_pt
            .into_iter()
            .map(BimapSInt::from)
            .collect::<Vec<_>>();
        query_points.push(first_query_pt);

        #[cfg(debug_assertions)]
        let (first_table_key, first_page_pt) = Prehashed::as_inner(&page_key);

        for query in queries[1..].iter() {
            debug_assert!(query.is_canonical());
            let (_query_table_key, global_pt) = BiMap::apply(&bimap, *query);
            #[cfg(debug_assertions)]
            {
                debug_assert_eq!(
                    _query_table_key, *first_table_key,
                    "all queries must have the same table key"
                );
                debug_assert_eq!(blockify_point(&global_pt), *first_page_pt);
            }
            let query_pt = global_pt
                .into_iter()
                .map(BimapSInt::from)
                .collect::<Vec<_>>();
            query_points.push(query_pt);
        }
        let page: &Page = &self.load_live_page(&page_key);
        let PageContents::RTree(page_tree) = &page.contents;

        get_same_page_many_canon_spatial(queries, &query_points, page_tree.tree())
    }

    pub fn prefetch<Tgt>(&self, query: &Spec<Tgt>)
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let mut query = query.clone();
        query.canonicalize().unwrap();
        self.prefetch_canon(&query);
    }

    /// Like [Self::prefetch], but assumes `query` is already canonical.
    ///
    /// Passing a non-canonical [Spec] is a logic error.
    pub(crate) fn prefetch_canon<Tgt>(&self, query: &Spec<Tgt>)
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(query.is_canonical());

        if !self.can_memoize_efficiently(query) {
            return;
        }

        let bimap = self.spec_bimap();
        let (table_key, global_pt) = BiMap::apply(&bimap, query);
        let page_pt = blockify_point(&global_pt);
        let page_key = self.prehasher.prehash((table_key, page_pt));

        let shard = &self.shards.0[self.shard_index(&page_key)];
        let mut shard_guard = shard.lock();
        shard_guard.process_available_bg_thread_msgs();
        if shard_guard.cache.peek(&page_key).is_none() {
            shard_guard.async_get(&page_key);
        }
    }

    /// Return the [PageId] to which the given canonical [Spec] belongs.
    ///
    /// Passing a non-canonical [Spec] or a [Spec] for which
    /// [FilesDatabase::can_memoize_efficiently] is `false` is a logic error.
    pub fn page_id<Tgt>(&self, spec: &Spec<Tgt>) -> PageId<'_>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(spec.is_canonical());
        debug_assert!(self.can_memoize_efficiently(spec));

        let bimap = self.spec_bimap();
        let (table_key, global_pt_lhs) = BiMap::apply(&bimap, spec);
        let page_pt = blockify_point(&global_pt_lhs);
        PageId {
            db: self,
            table_key,
            page_id: page_pt,
        }
    }

    pub fn put<Tgt>(&self, mut spec: Spec<Tgt>, decisions: Vec<(ActionNum, Cost)>)
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        spec.canonicalize().unwrap();
        self.put_canon(&spec, decisions);
    }

    /// Like [Self::put], but assumes `spec` is already canonical.
    ///
    /// Passing a non-canonical [Spec] is a logic error.
    pub(crate) fn put_canon<Tgt>(&self, spec: &Spec<Tgt>, decisions: Vec<(ActionNum, Cost)>)
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(spec.is_canonical());

        // Check that all costs in decisions have peak memory less than or equal to spec's
        // memory limits.
        debug_assert!(
            decisions.iter().all(|(_, c)| {
                let MemoryLimits::Standard(limits) = &spec.1;
                &c.peaks <= limits
            }),
            "peak memory of an action exceeds memory limits of {spec}: {decisions:?}"
        );

        #[cfg(feature = "db-stats")]
        self.stats.puts.fetch_add(1, atomic::Ordering::Relaxed);

        if !self.can_memoize_efficiently(spec) {
            let key = self.nonspatial_key(spec);
            {
                let mut nonspatial_cache = self.nonspatial_cache.lock();
                nonspatial_cache
                    .entries
                    .insert(key, ActionCostVec(decisions));
                nonspatial_cache.modified = true;
            }
            if self.proactive_saves_enabled {
                self.try_save_nonspatial_cache_in_background();
            }
            return;
        }

        let bimap = self.spec_bimap();
        let (table_key, (bottom, top)) = put_range_to_fill(&bimap, spec, &decisions);

        // Construct the page-coordinate bounds for all pages (tiles) to fill.
        let rank = bottom.len();
        let page_bottom = bottom
            .iter()
            .enumerate()
            .map(|(dim, &bottom)| bottom / block_size_dim(dim, rank))
            .collect::<Vec<_>>();
        let page_top = top
            .iter()
            .enumerate()
            .map(|(dim, &top)| top / block_size_dim(dim, rank))
            .collect::<Vec<_>>();

        // Since this put is for a single Spec, we can normalize the cost with that Spec's volume.
        let normalized_decisions =
            ActionNormalizedCostVec::normalize(ActionCostVec(decisions), spec.0.volume());

        // Reuse the following two values to avoid some allocations.
        let mut key_tuple = Some((table_key, Vec::with_capacity(rank)));
        let mut dim_ranges = Vec::with_capacity(rank);
        multi_range_product(&page_bottom, &page_top, |page_point: &[BimapInt]| {
            {
                let key_tuple = key_tuple.as_mut().unwrap();
                key_tuple.1.clear();
                key_tuple.1.extend_from_slice(page_point);
            }

            dim_ranges.clear();
            for (dim, &page_idx) in page_point.iter().enumerate() {
                let block_dim_size = block_size_dim(dim, rank);
                let Some(global_top_noninc) = top[dim].checked_add(1) else {
                    todo!("support global_top equal to MAX");
                };
                let start = (page_idx * block_dim_size).max(bottom[dim]);
                let end = ((page_idx + 1) * block_dim_size).min(global_top_noninc);
                dim_ranges.push(start..end);
            }

            // Do some awkward mutation of `key_tuple.1` to avoid cloning `table_key`/ `key_tuple.0`
            // on each iteration.
            let key = self.prehasher.prehash(key_tuple.take().unwrap());
            // Load or wait for the page while holding its shard lock, then update both the page and
            // the shard's dirty-page bookkeeping together.
            {
                let shard = &self.shards.0[self.shard_index(&key)];
                let mut shard_guard = shard.lock();
                {
                    let page = shard_guard.load_live_page_mut(&key);
                    page.modified = true;
                    page.contents
                        .fill_region(self.k, &dim_ranges, &normalized_decisions);
                }

                if self.proactive_saves_enabled
                    && shard_guard.modified_pages.insert(prehashed_clone(&key))
                {
                    shard_guard
                        .modified_page_queue
                        .push_back(prehashed_clone(&key));
                }
            }
            key_tuple = Some(Prehashed::into_inner(key));
        });

        if self.proactive_saves_enabled {
            self.try_save_pages_in_background();
        }
    }

    /// Saves anything cached in memory to disk, blocking until complete.
    pub fn save(&self) {
        for shard in &self.shards.0 {
            let mut shard_guard = shard.lock();
            shard_guard.save();
        }
        self.save_nonspatial_cache();
    }

    fn try_save_pages_in_background(&self) {
        let shard_count = self.shards.0.len();
        let scan_count = BACKGROUND_SAVE_SHARDS_PER_PUT.min(shard_count);
        let start = self
            .next_proactive_save_shard
            .fetch_add(scan_count, AtomicOrdering::Relaxed);
        for offset in 0..scan_count {
            let shard_idx = (start + offset) % shard_count;
            if let Some(mut shard_guard) = self.shards.0[shard_idx].try_lock() {
                shard_guard.try_save_pages_in_background(self.proactive_save_interval);
            }
        }
    }

    fn try_save_nonspatial_cache_in_background(&self) {
        let now = Instant::now();
        let Some(mut nonspatial_cache) = self.nonspatial_cache.try_lock() else {
            return;
        };
        if !nonspatial_cache.modified
            || nonspatial_cache
                .last_save
                .is_some_and(|last| now.duration_since(last) < self.proactive_save_interval)
        {
            return;
        }

        let Some(shard_guard) = self.shards.0[0].try_lock() else {
            return;
        };
        let msg = ShardThreadMsg::PutNonSpatial(
            nonspatial_cache_file_path(self.dir_handle.path()),
            nonspatial_cache.entries.clone(),
        );
        match shard_guard.thread_tx.try_send(msg) {
            Ok(()) => {
                nonspatial_cache.modified = false;
                nonspatial_cache.last_save = Some(now);
            }
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => {
                panic!("shard thread exited before accepting proactive non-spatial cache write");
            }
        }
    }

    #[inline]
    pub fn max_k(&self) -> Option<usize> {
        Some(self.k.into())
    }

    // TODO: Make spec_bimap private again
    /// Return a bidirectional map from [Spec]s to tuples of table keys and their coordinates.
    pub(crate) fn spec_bimap<Tgt>(&self) -> impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Memory, Codomain = u8>,
    {
        let surmap = SpecSurMap::<Tgt, _, _, _> {
            logical_spec_surmap: LogicalSpecSurMap::new(
                PrimitiveBasicsBimap {
                    tile_scale: self.tile_scale,
                },
                |shape: &[DimSize], dtype| TensorSpecAuxNonDepBimap::new(shape, dtype),
            ),
            memory_limits_bimap: MemoryLimitsBimap::default(),
        };
        surmap.into_bimap()
    }

    /// Returns whether the database can memoize the given Spec in a spatial (compressed) table.
    ///
    /// A Spec can be memoized efficiently if the database's BiMap is defined for it. For example,
    /// when the database factorizes shapes, only dimensions of the form `2^x` or `3 * 2^x` can be
    /// memoized efficiently. Other Specs can still be stored in the non-spatial cache, but this is
    /// typically much less efficient.
    pub fn can_memoize_efficiently<Tgt>(&self, spec: &Spec<Tgt>) -> bool
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Memory, Codomain = u8>,
    {
        SurMap::defined_for(&self.spec_bimap(), spec)
    }

    fn nonspatial_key<Tgt>(&self, spec: &Spec<Tgt>) -> NonSpatialKey
    where
        Tgt: Target,
    {
        NonSpatialKey {
            // Serialize the key to erase the Tgt type parameter
            spec: bincode::serialize(spec)
                .expect("Spec should serialize for non-spatial cache key"),
        }
    }

    fn save_nonspatial_cache(&self) {
        let mut nonspatial_cache = self.nonspatial_cache.lock();
        if !nonspatial_cache.modified {
            return;
        }
        let path = nonspatial_cache_file_path(self.dir_handle.path());
        let mut shard_guard = self.shards.0[0].lock();

        let cache = nonspatial_cache.entries.clone();
        shard_guard.send_msg(ShardThreadMsg::PutNonSpatial(path, cache));
        shard_guard.flush();

        nonspatial_cache.modified = false;
        nonspatial_cache.last_save = Some(Instant::now());
    }

    pub fn spatial_query<Tgt, B>(
        &self,
        query: &SpatialQuery<Tgt, B, TableKey>,
        mut visit: impl FnMut(&TableKey, &[BimapSInt], &[BimapSInt], Option<NormalizedCost>),
    ) where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Memory, Codomain = u8>,
        B: BiMap<Domain = Spec<Tgt>, Codomain = DbKey>,
    {
        for (query_table_key, query_tree) in query.tables() {
            let rank = query_tree.dim_count();
            let mut visited_pages = HashSet::new();
            for (query_bottom, query_top, ()) in query_tree.iter() {
                let mut page_bottom = Vec::new();
                let mut page_top = Vec::new();
                page_bottom.reserve_exact(rank);
                page_top.reserve_exact(rank);
                for (dim, (&bottom, &top)) in query_bottom.iter().zip(query_top).enumerate() {
                    let bottom = BimapInt::try_from(bottom).unwrap();
                    let top = BimapInt::try_from(top).unwrap();
                    let block_dim_size = block_size_dim(dim, rank);
                    page_bottom.push(bottom / block_dim_size);
                    page_top.push(top / block_dim_size);
                }
                multi_range_product(&page_bottom, &page_top, |page_point: &[BimapInt]| {
                    if !visited_pages.insert(page_point.to_vec()) {
                        return;
                    }
                    self.for_each_spatial_query_page_result(
                        query_table_key,
                        query_tree,
                        page_point.to_vec(),
                        |(bottom, top, memoized_cost)| {
                            visit(query_table_key, &bottom, &top, memoized_cost);
                        },
                    );
                });
            }
        }
    }

    fn for_each_spatial_query_page_result(
        &self,
        query_table_key: &TableKey,
        query_tree: &RTreeDyn<()>,
        page_point: Vec<BimapInt>,
        mut visit: impl FnMut(SpatialQueryPageResult),
    ) {
        let page_key = self
            .prehasher
            .prehash((query_table_key.clone(), page_point));

        let Some(page) = self.try_load_live_page(&page_key) else {
            return;
        };
        let PageContents::RTree(page_tree) = &page.contents;
        query_tree
            .intersection_candidates_with_other_tree(page_tree.tree())
            .for_each(
                |((query_bottom, query_top, ()), (memo_bottom, memo_top, memo_value))| {
                    let bottom: Vec<_> = query_bottom
                        .iter()
                        .zip(memo_bottom)
                        .map(|(lhs, rhs)| *lhs.max(rhs))
                        .collect();
                    let top: Vec<_> = query_top
                        .iter()
                        .zip(memo_top)
                        .map(|(lhs, rhs)| *lhs.min(rhs))
                        .collect();
                    debug_assert!(bottom.iter().zip(&top).all(|(bottom, top)| bottom <= top));

                    let memoized_cost =
                        memo_value
                            .as_ref()
                            .map(|(intensity, peaks, depth, _)| NormalizedCost {
                                intensity: *intensity,
                                peaks: peaks.clone(), // TODO: Avoid this clone
                                depth: *depth,
                            });
                    visit((bottom, top, memoized_cost));
                },
            );
    }

    #[cfg(test)]
    pub fn assert_same_memoized_points_and_throughputs(&self, other: &Self) {
        use std::collections::HashSet;

        let lhs = self.all_throughputs();
        let rhs = other.all_throughputs();
        assert_eq!(
            lhs.keys().collect::<HashSet<_>>(),
            rhs.keys().collect::<HashSet<_>>(),
            "database table sets differ"
        );
        assert_eq!(lhs, rhs, "database memoized points or throughputs differ");
    }

    #[cfg(test)]
    fn all_throughputs(&self) -> MemoizedThroughputsByTable {
        let mut result = MemoizedThroughputsByTable::new();

        for shard in &self.shards.0 {
            let mut shard_guard = shard.lock();
            shard_guard.process_available_bg_thread_msgs();
            for (page_key, page) in shard_guard.cache.iter() {
                let (table_key, _) = Prehashed::as_inner(page_key);
                let PageContents::RTree(page_tree) = &page.contents;
                let table = result.entry(table_key.clone()).or_default();
                for (bottom, top, memoized_cost) in page_tree.tree().iter() {
                    let throughput = memoized_cost
                        .as_ref()
                        .map(|(intensity, _peaks, _depth, _action)| *intensity);
                    multi_range_product(bottom, top, |point| {
                        let point = point
                            .iter()
                            .map(|&coord| BimapInt::try_from(coord).unwrap())
                            .collect();
                        let previous = table.insert(point, throughput);
                        assert!(
                            previous.is_none_or(|previous| previous == throughput),
                            "overlapping memoized rectangles disagree on throughput"
                        );
                    });
                }
            }
        }

        result
    }

    /// Returns a [Page] if present or an empty [Page] if not.
    ///
    /// If an empty page is made, it's kept in the in-memory cache. Missing pages are
    /// not persisted unless later modified.
    fn load_live_page<'a>(&'a self, key: &Prehashed<PageKey>) -> impl Deref<Target = Page> + 'a {
        self.load_live_page_mut(key)
    }

    /// Like [Self::load_live_page], but returns a mutable reference.
    fn load_live_page_mut<'a>(
        &'a self,
        key: &Prehashed<PageKey>,
    ) -> impl DerefMut<Target = Page> + 'a {
        self.load_live_page_mut_inner(key, false).unwrap()
    }

    /// Returns a [Page] if it is already cached or exists on disk.
    ///
    /// Unlike [Self::load_live_page], this does not materialize a missing page as an empty cache
    /// entry.
    fn try_load_live_page<'a>(
        &'a self,
        key: &Prehashed<PageKey>,
    ) -> Option<impl Deref<Target = Page> + 'a> {
        self.load_live_page_mut_inner(key, true)
    }

    fn load_live_page_mut_inner<'a>(
        &'a self,
        key: &Prehashed<PageKey>,
        check_page_file_exists: bool,
    ) -> Option<impl DerefMut<Target = Page> + 'a> {
        let shard = &self.shards.0[self.shard_index(key)];
        let mut shard_guard = shard.lock();

        shard_guard.process_available_bg_thread_msgs();

        // Fast path if page is already in cache
        let shard_guard = match MutexGuard::try_map(shard_guard, |s| s.cache.get_mut(key)) {
            Ok(mapped) => return Some(mapped),
            Err(s) => s,
        };

        if check_page_file_exists
            && !page_file_path(self.dir_handle.path(), Prehashed::as_inner(key)).exists()
        {
            return None;
        }

        Some(MutexGuard::map(shard_guard, |s| s.load_live_page_mut(key)))
    }

    fn shard_index(&self, key: &Prehashed<PageKey>) -> usize {
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
    pub fn analyze(&self, output_dir: &path::Path, sample: usize, skip_read_errors: bool) {
        let page_csv_path = output_dir.join("pages.csv");
        let block_csv_path = output_dir.join("blocks.csv");
        let block_action_csv_path = output_dir.join("block_actions.csv");

        let mut writers = AnalyzeWriters {
            page_writer: csv::Writer::from_path(page_csv_path.clone()).unwrap(),
            block_writer: csv::Writer::from_path(block_csv_path.clone()).unwrap(),
            block_action_writer: csv::Writer::from_path(block_action_csv_path.clone()).unwrap(),
        };
        writers.page_writer.write_record(["page_path"]).unwrap();
        writers
            .block_writer
            .write_record(["page_path", "rects", "spec_count"])
            .unwrap();
        writers
            .block_action_writer
            .write_record(["page_path", "block_pt", "action"])
            .unwrap();

        analyze_visit_dir(
            self.dir_handle.path(),
            self.dir_handle.path(),
            &mut writers,
            sample,
            skip_read_errors,
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
        self.save_nonspatial_cache();
    }
}

impl PageId<'_> {
    /// Returns `true` if the page contains the given canonical [Spec].
    ///
    /// Passing a non-canonical [Spec] is a logic error.
    pub fn contains<Tgt>(&self, spec: &Spec<Tgt>) -> bool
    where
        Tgt: Target,
        Tgt::Memory: CanonicalBimap,
        <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(spec.is_canonical());

        let bimap = self.db.spec_bimap();
        let (table_key, global_pt) = BiMap::apply(&bimap, spec);
        if self.table_key != table_key {
            return false;
        }
        let page_pt = blockify_point(&global_pt);
        self.page_id == page_pt
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
                            let path = page_file_path(db_root.path(), &key);

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
                                    Err(ReadAnyFormatError {
                                        zstd_error,
                                        plain_error,
                                    }) => {
                                        log::error!(
                                            "Continuing after errors reading page; {zstd_error:?} and {plain_error:?}"
                                        );
                                        Page {
                                            contents: PageContents::RTree(Box::new(
                                                RTreePageContents::empty(key.1.len()),
                                            )),
                                            modified: false,
                                            last_save: None,
                                        }
                                    }
                                },
                                Err(_) => Page {
                                    contents: PageContents::RTree(Box::new(
                                        RTreePageContents::empty(key.1.len()),
                                    )),
                                    modified: false,
                                    last_save: None,
                                },
                            };
                            response_tx
                                .send(ShardThreadResponse::Loaded(key, result))
                                .unwrap();
                        }
                        Ok(ShardThreadMsg::Put(key, value)) => {
                            let path = page_file_path(db_root.path(), &key);

                            #[cfg(feature = "db-stats")]
                            {
                                log::debug!("Writing database page");
                            }

                            write_page_atomic(&path, &value);

                            #[cfg(feature = "db-stats")]
                            {
                                stats.disk_bytes_written.fetch_add(
                                    path.metadata().unwrap().len(),
                                    atomic::Ordering::Relaxed,
                                );
                            }
                        }
                        Ok(ShardThreadMsg::PutNonSpatial(path, cache)) => {
                            #[cfg(feature = "db-stats")]
                            {
                                log::debug!("Writing non-spatial database cache");
                            }

                            write_nonspatial_cache_atomic(&path, &cache);

                            #[cfg(feature = "db-stats")]
                            {
                                // TODO: This has a race condition, but it's not a big concern.
                                stats.disk_bytes_written.fetch_add(
                                    path.metadata().unwrap().len(),
                                    atomic::Ordering::Relaxed,
                                );
                            }
                        }
                        Ok(ShardThreadMsg::Flush(done_tx)) => {
                            done_tx.send(()).unwrap();
                        }
                        Ok(ShardThreadMsg::Exit) => break,
                        Err(_) => unreachable!("expected Exit first"),
                    }
                })
                .unwrap(),
        );

        Self {
            cache: WTinyLfuCache::new(cache_per_shard_size, cache_per_shard_samples),
            modified_pages: new_prehashed_set(),
            modified_page_queue: VecDeque::new(),
            outstanding_gets: new_prehashed_set(),
            thread,
            thread_tx: command_tx,
            thread_rx: response_rx,
            #[cfg(feature = "db-stats")]
            stats: stats2,
        }
    }

    /// Calls `self.thread_rx.recv()`, logging time taken if `db-stats` is enabled.
    #[allow(clippy::let_and_return)]
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
                self.modified_pages.remove(&evicted_key);
                self.async_put(Prehashed::into_inner(evicted_key), evicted_value.contents);
            }
        }
    }

    fn load_live_page_mut(&mut self, key: &Prehashed<PageKey>) -> &mut Page {
        self.process_available_bg_thread_msgs();
        if self.cache.peek(key).is_none() {
            self.async_get(key);
            self.process_bg_thread_msgs_until(|resp| match resp {
                ShardThreadResponse::Loaded(k, _) => k != key,
            });
        }
        self.cache
            .get_mut(key)
            .unwrap_or_else(|| panic!("just-requested key in cache: {key:?}"))
    }

    /// Start a background task to load a page. Do nothing if request already enqueued.
    ///
    /// This updates does not update cache statistics.
    ///
    /// This may panic if the key is already in the cache.
    fn async_get(&mut self, key: &Prehashed<PageKey>) -> bool {
        if self.outstanding_gets.contains(key) {
            return false;
        }
        debug_assert!(self.cache.peek(key).is_none(), "cache already had {key:?}");
        self.outstanding_gets.insert(prehashed_clone(key));
        self.send_msg(ShardThreadMsg::Get(prehashed_clone(key)));
        true
    }

    fn async_put(&mut self, key: PageKey, value: PageContents) {
        self.send_msg(ShardThreadMsg::Put(key, value));
    }

    fn save(&mut self) {
        // Finish outstanding gets before saving
        while !self.outstanding_gets.is_empty() {
            let msg = self
                .blocking_recv()
                .expect("shard thread exited before loading outstanding database pages");
            self.process_bg_thread_msg_inner(msg);
        }

        let now = Instant::now();
        let mut writes = Vec::new();
        for (k, v) in self.cache.iter_mut() {
            // Clone so that the background thread has an immutable copy of the data to write, even
            // if the calling or another thread would update data in the cache.
            // TODO: Instead sync with the background thread and guarantee this is unchanging since
            //       we have the `&self` reference.
            if v.modified {
                v.modified = false;
                v.last_save = Some(now);
                writes.push((Prehashed::as_inner(k).clone(), v.contents.clone()));
            }
        }
        for (key, value) in writes {
            self.async_put(key, value);
        }
        self.modified_pages.clear();
        self.modified_page_queue.clear();
        self.flush();
    }

    fn flush(&mut self) {
        let (done_tx, done_rx) = mpsc::channel();
        self.send_msg(ShardThreadMsg::Flush(done_tx));
        done_rx
            .recv()
            .expect("shard thread exited before flushing database writes");
    }

    fn try_save_pages_in_background(&mut self, proactive_save_interval: Duration) {
        let now = Instant::now();
        let tx = self.thread_tx.clone();

        for _ in 0..BACKGROUND_SAVE_PAGES_PER_SHARD {
            let Some(key) = self.modified_page_queue.pop_front() else {
                break;
            };
            if !self.modified_pages.contains(&key) {
                continue;
            }

            let mut remove_from_dirty = false;
            let mut requeue = false;
            let mut maybe_msg = None;
            {
                let Some(page) = self.cache.get_mut(&key) else {
                    self.modified_pages.remove(&key);
                    continue;
                };
                if !page.modified {
                    remove_from_dirty = true;
                } else if page.last_save.map_or_else(
                    || true,
                    |last| now.duration_since(last) >= proactive_save_interval,
                ) {
                    maybe_msg = Some(ShardThreadMsg::Put(
                        Prehashed::as_inner(&key).clone(),
                        page.contents.clone(),
                    ));
                } else {
                    requeue = true;
                }
            }

            let Some(msg) = maybe_msg else {
                if remove_from_dirty {
                    self.modified_pages.remove(&key);
                } else if requeue {
                    self.modified_page_queue.push_back(key);
                }
                continue;
            };

            match tx.try_send(msg) {
                Ok(()) => {
                    let page = self.cache.get_mut(&key).expect("just-saved page in cache");
                    page.modified = false;
                    page.last_save = Some(now);
                    self.modified_pages.remove(&key);
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    self.modified_page_queue.push_front(key);
                    return;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    panic!("shard thread exited before accepting proactive database write");
                }
            }
        }
    }

    fn send_msg(&mut self, mut msg: ShardThreadMsg) {
        // Send, processing background thread messages while the channel is full
        loop {
            match self.thread_tx.try_send(msg) {
                Ok(()) => return,
                Err(mpsc::TrySendError::Full(returned_msg)) => {
                    msg = returned_msg;
                    self.process_available_bg_thread_msgs();
                    std::thread::yield_now();
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    panic!("shard thread exited before accepting database command");
                }
            }
        }
    }

    fn drain_cache(&mut self) -> impl Iterator<Item = (PageKey, Page)> + '_ {
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
        self.send_msg(ShardThreadMsg::Exit);
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
    type Target = Vec<(ActionNum, Cost)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Vec<(ActionNum, Cost)>> for ActionCostVec {
    fn as_ref(&self) -> &Vec<(ActionNum, Cost)> {
        self.deref()
    }
}

impl TileScale {
    pub(crate) const fn codomain_len(self, rank: u8) -> usize {
        match self {
            TileScale::Linear => rank as usize,
            TileScale::PowerOfTwo => rank as usize,
            TileScale::PowerOrThreePower => rank as usize * 2,
        }
    }
}

impl fmt::Display for TileScale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileScale::Linear => write!(f, "linear"),
            TileScale::PowerOfTwo => write!(f, "power-of-two"),
            TileScale::PowerOrThreePower => write!(f, "power-or-three-power"),
        }
    }
}

impl std::str::FromStr for TileScale {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "linear" => Ok(TileScale::Linear),
            "power-of-two" => Ok(TileScale::PowerOfTwo),
            "power-or-three-power" => Ok(TileScale::PowerOrThreePower),
            _ => Err(format!("unknown TILESCALE value: {s}")),
        }
    }
}

impl ActionNormalizedCostVec {
    pub fn normalize(action_costs: ActionCostVec, volume: NonZeroU64) -> Self {
        ActionNormalizedCostVec(
            action_costs
                .0
                .into_iter()
                .map(|(action_num, cost)| (action_num, NormalizedCost::new(cost, volume)))
                .collect(),
        )
    }
}

/// Tries to read a zstd-compressed file. If that fails, tries to read it uncompressed.
fn read_any_format(file: fs::File) -> Result<Page, ReadAnyFormatError> {
    let mut zstd_reader = zstd::Decoder::new(file).unwrap();
    let contents: PageContents = match bincode::deserialize_from(&mut zstd_reader) {
        Ok(contents) => contents,
        Err(zstd_error) => {
            // Couldn't read as zstd? Try reading uncompressed.
            let mut file = zstd_reader.finish();
            file.seek(SeekFrom::Start(0)).unwrap();
            let buf_reader = BufReader::new(file);
            match bincode::deserialize_from(buf_reader) {
                Ok(page) => page,
                Err(plain_error) => {
                    return Err(ReadAnyFormatError {
                        zstd_error,
                        plain_error,
                    })
                }
            }
        }
    };
    Ok(Page {
        contents,
        modified: false,
        last_save: None,
    })
}

#[cfg(feature = "db-stats")]
fn analyze_visit_dir(
    root: &path::Path,
    path: &path::Path,
    writers: &mut AnalyzeWriters,
    sample: usize,
    skip_read_errors: bool,
) {
    // Since we don't revisit blocks, bypass the in-mem. cache and read from disk.
    for file_entry in fs::read_dir(path).unwrap() {
        let file_entry = file_entry.unwrap();
        if let Some("PRECOMPUTE") = file_entry.path().file_name().unwrap().to_str() {
            continue;
        }

        let entry_path = file_entry.path();
        if entry_path.is_dir() {
            analyze_visit_dir(root, &file_entry.path(), writers, sample, skip_read_errors);
            continue;
        }

        // Skip with probability 1/sample.
        if sample > 1 && !rand::random::<usize>().is_multiple_of(sample) {
            continue;
        }

        let shortened_entry_path_str = entry_path.strip_prefix(root).unwrap();
        let entry_path_str = format!("{}", shortened_entry_path_str.display());

        let page_file = fs::File::open(entry_path).unwrap();
        let page = match read_any_format(page_file) {
            Ok(r) => r,
            Err(e) => {
                if skip_read_errors {
                    log::warn!("Error reading page: {e:?}");
                    continue;
                }
                panic!("Error reading page: {e:?}");
            }
        };

        let PageContents::RTree(r) = &page.contents;
        writers
            .block_writer
            .write_record([
                &entry_path_str,
                &r.rect_count().to_string(),
                &r.spec_count().to_string(),
            ])
            .unwrap();

        writers.page_writer.write_record([entry_path_str]).unwrap();
    }
}

fn block_size_dim(dim: usize, dim_count: usize) -> u32 {
    if dim >= dim_count - MEMORY_COUNT {
        31
    } else {
        8
    }
}

/// Compute the block coordinate from a global coordinate.
///
/// If the value type is not [BimapInt], this will try to convert and panic if that fails.
fn blockify_point<T>(pt: &[T]) -> Vec<BimapInt>
where
    T: Copy + TryInto<BimapInt>,
    T::Error: Debug,
{
    let rank = pt.len();
    pt.iter()
        .enumerate()
        .map(|(i, &d)| d.try_into().unwrap() / block_size_dim(i, rank))
        .collect()
}

fn get_same_page_many_canon_spatial<Tgt>(
    queries: &[&Spec<Tgt>],
    query_points: &[Vec<BimapSInt>],
    page_tree: &RTreeDyn<DbValue>,
) -> Vec<GetPreference<ActionCostVec, Vec<ActionNum>>>
where
    Tgt: Target,
{
    let rank = page_tree.dim_count();
    debug_assert!(
        query_points.iter().all(|p| p.len() == rank),
        "query points must have same rank as page tree"
    );

    // Build the R*-Tree of dependencies from the given set of query points.
    let mut query_tree = RTreeDyn::empty(rank);
    query_tree.bulk_merge_insert(
        query_points
            .iter()
            .map(|p| (p.clone(), p.clone(), ()))
            .collect(),
    );

    // Build a side table to map intersecting points back to queries.
    // TODO: Building this is probably way too expensive.
    let mut query_indices_by_point = HashMap::<_, Vec<usize>>::with_capacity(query_points.len());
    for (query_idx, point) in query_points.iter().enumerate() {
        query_indices_by_point
            .entry(point.as_slice())
            .or_default()
            .push(query_idx);
    }

    let mut results = vec![GetPreference::Miss(None); queries.len()];
    let mut overlap_bottom_pt = Vec::new();
    let mut overlap_top_pt = Vec::new();
    overlap_bottom_pt.reserve_exact(rank);
    overlap_top_pt.reserve_exact(rank);
    query_tree
        .intersection_candidates_with_other_tree(page_tree)
        .for_each(
            |((query_bottom, query_top, ()), (memo_bottom, memo_top, memo_value))| {
                // Fill overlap_bottom and overlap_top with the bottom and top coordinates of the
                // intersection of the query rectangle and the memoized rectangle.
                overlap_bottom_pt.clear();
                overlap_top_pt.clear();
                for (((&lhs_bottom, &lhs_top), &rhs_bottom), &rhs_top) in query_bottom
                    .iter()
                    .zip(query_top)
                    .zip(memo_bottom)
                    .zip(memo_top)
                {
                    let dim_bottom = lhs_bottom.max(rhs_bottom);
                    let dim_top = lhs_top.min(rhs_top);
                    if dim_bottom > dim_top {
                        return;
                    }
                    overlap_bottom_pt.push(dim_bottom);
                    overlap_top_pt.push(dim_top);
                }

                multi_range_product(&overlap_bottom_pt, &overlap_top_pt, |point| {
                    let query_indices = query_indices_by_point
                        .get(point)
                        .expect("query geometry produced an unknown point");
                    for &query_idx in query_indices {
                        results[query_idx] = GetPreference::Hit(memo_value_to_action_cost_vec(
                            queries[query_idx],
                            memo_value,
                        ));
                    }
                });
            },
        );

    results
}

// TODO: This helper shouldn't really be needed. Ideally, instead, we could store ActionCostVecs (or
//       at least NormalizedCost values) directly in the R*-Tree.
fn memo_value_to_action_cost_vec<Tgt>(query: &Spec<Tgt>, memo_value: &DbValue) -> ActionCostVec
where
    Tgt: Target,
{
    ActionCostVec(
        memo_value
            .as_ref()
            .map(|(intensity, peaks, depth, action_num)| {
                let cost = NormalizedCost {
                    intensity: *intensity,
                    peaks: peaks.clone(),
                    depth: *depth,
                }
                .into_cost(query.0.volume());
                vec![(*action_num, cost)]
            })
            .unwrap_or_default(),
    )
}

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt, B>(
    bimap: &B,
    spec: &Spec<Tgt>,
    impls: &[(ActionNum, Cost)],
) -> (TableKey, (Vec<BimapInt>, Vec<BimapInt>))
where
    Tgt: Target,
    Tgt::Memory: CanonicalBimap,
    <Tgt::Memory as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    B: BiMap<Domain = Spec<Tgt>, Codomain = DbKey>,
{
    // Compute worst-case bound of solutions' memory usage. This lower bounds the range.
    let per_level_peaks = impls
        .iter()
        .fold(MemVec::zero::<Tgt>(), |acc, (_, cost)| acc.max(&cost.peaks));

    // Compute the complete upper and lower bounds from the given Spec and that Spec modified with
    // the peaks' bound (computed above).
    let upper_inclusive = BiMap::apply(bimap, spec);
    let lower_inclusive = {
        let mut lower_bound_spec = spec.clone();
        lower_bound_spec.1 = MemoryLimits::Standard(per_level_peaks);
        BiMap::apply(bimap, &lower_bound_spec)
    };

    // TODO: This computes the non-memory dimensions of the key/coordinates twice. Avoid that.
    debug_assert_eq!(upper_inclusive.0, lower_inclusive.0);
    debug_assert_eq!(
        upper_inclusive.1[..upper_inclusive.1.len() - MEMORY_COUNT],
        lower_inclusive.1[..lower_inclusive.1.len() - MEMORY_COUNT]
    );

    (upper_inclusive.0, (lower_inclusive.1, upper_inclusive.1))
}

/// Iterate blocks of an integer range.
///
/// Yields block indices along with a range of split global indices (the ranges are no in the block
/// coordinate space). For example:
/// ```
/// # use morello::db::iter_blocks_in_single_dim_range;
/// assert_eq!(iter_blocks_in_single_dim_range(0, 3, 4).collect::<Vec<_>>(),
///           vec![(0, 0..4)]);
/// assert_eq!(iter_blocks_in_single_dim_range(1, 7, 4).collect::<Vec<_>>(),
///            vec![(0, 1..4), (1, 4..8)]);
/// ```
///
/// `global_bottom` and `global_top` are inclusive, forming a closed range. For example,
/// the following yields a single range smaller than the block.
/// ```
/// # use morello::db::iter_blocks_in_single_dim_range;
/// assert_eq!(iter_blocks_in_single_dim_range(4, 4, 4).collect::<Vec<_>>(),
///            vec![(1, 4..5)]);
/// ```
///
// TODO: Make private. (Will break doctests.)
pub fn iter_blocks_in_single_dim_range(
    global_bottom: BimapInt,
    global_top: BimapInt,
    block_dim_size: BimapInt,
) -> impl Iterator<Item = (BimapInt, Range<BimapInt>)> + Clone {
    debug_assert_ne!(block_dim_size, 0);
    debug_assert!(global_bottom <= global_top);

    // Change global_top to a non-inclusive upper bound.
    let Some(global_top_noninc) = global_top.checked_add(1) else {
        todo!("support global_top equal to MAX");
    };

    // Compute half-open range of blocks.
    let block_bottom = global_bottom / block_dim_size;
    let block_top = global_top / block_dim_size;

    (block_bottom..=block_top).map(move |block_idx| {
        // Compute the largest possible range for the block, then clip to given bounds.
        let start = (block_idx * block_dim_size).max(global_bottom);
        let end = ((block_idx + 1) * block_dim_size).min(global_top_noninc);
        (block_idx, start..end)
    })
}

fn page_file_path(root: &Path, page_key: &PageKey) -> path::PathBuf {
    let ((spec_key, _), block_pt) = page_key;
    let spec_key_dir_name = match spec_key {
        SpecKey::OnePrefix { rank, dtypes } => root
            .join(format!("OnePrefix{}", rank))
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Matmul { dtypes } => root
            .join("Matmul")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Conv { dtypes } => root
            .join("Conv")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Broadcast { rank, dim, dtypes } => root
            .join(format!("Broadcast{}", rank))
            .join(dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Softmax {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("Softmax{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxComplete {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("SoftmaxComplete{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndMax {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("SoftmaxDenominatorAndMax{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndUnscaled {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("SoftmaxDenominatorAndUnscaled{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndUnscaledFromMax {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("SoftmaxDenominatorAndUnscaledFromMax{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominator {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("SoftmaxDenominator{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().join("_")),
        SpecKey::DivideVec { rank, dtypes } => root
            .join(format!("DivideVec{}", rank))
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::DivideVecScalar {
            rank,
            scan_dim,
            dtypes,
        } => root
            .join(format!("DivideVecScalar{}", rank))
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Max { rank, dtypes, dim } => root
            .join(format!("Max{}", rank))
            .join(dim.to_string())
            .join(dtypes.iter().join("_")),
        SpecKey::Move { rank, dtypes } => root
            .join(format!("Move{}", rank))
            .join(dtypes.iter().join("_")),
        SpecKey::Fill {
            rank,
            dtype,
            value: FillValue::Zero,
        } => root
            .join(format!("FillZero{}", rank))
            .join(dtype.to_string()),
        SpecKey::Fill {
            rank,
            dtype,
            value: FillValue::NegInf,
        } => root
            .join(format!("FillNegInf{}", rank))
            .join(dtype.to_string()),
        SpecKey::Fill {
            rank,
            dtype,
            value: FillValue::Min,
        } => root
            .join(format!("FillMin{}", rank))
            .join(dtype.to_string()),
        SpecKey::Compose { components } => root
            .join("Compose")
            .join(
                components
                    .iter()
                    .map(|(spec_type, _, _)| spec_type)
                    .join("_"),
            )
            .join(
                components
                    .iter()
                    .flat_map(|(_, dtypes, _)| dtypes)
                    .join("_"),
            ),
    };
    spec_key_dir_name.join(block_pt.iter().map(|p| p.to_string()).join("_"))
}

fn write_page_atomic(path: &Path, contents: &PageContents) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let mut file = AtomicWriteFile::options().open(path).unwrap();

    if COMPRESS_PAGES {
        let mut zstd_writer = zstd::Encoder::new(&mut file, 0).unwrap();
        bincode::serialize_into(&mut zstd_writer, contents).unwrap();
        zstd_writer.finish().unwrap();
    } else {
        let mut buf_writer = BufWriter::new(&mut file);
        bincode::serialize_into(&mut buf_writer, contents).unwrap();
        buf_writer.flush().unwrap();
    }

    file.commit().unwrap();
}

fn nonspatial_cache_file_path(root: &Path) -> path::PathBuf {
    root.join(NONSPATIAL_CACHE_FILE)
}

fn read_nonspatial_cache(root: &Path) -> HashMap<NonSpatialKey, ActionCostVec> {
    let path = nonspatial_cache_file_path(root);
    let Ok(file) = fs::File::open(&path) else {
        return HashMap::new();
    };
    bincode::deserialize_from(BufReader::new(file)).unwrap_or_else(|e| {
        panic!(
            "Failed to read non-spatial cache at {}: {e}",
            path.display()
        )
    })
}

fn write_nonspatial_cache_atomic(path: &Path, cache: &HashMap<NonSpatialKey, ActionCostVec>) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let mut file = AtomicWriteFile::options().open(path).unwrap();
    let mut buf_writer = BufWriter::new(&mut file);
    bincode::serialize_into(&mut buf_writer, cache).unwrap();
    buf_writer.flush().unwrap();
    drop(buf_writer);
    file.commit().unwrap();
}

/// Create TARGET file in `db_path` if it is missing, then validate it is
/// consistent with `Tgt`.
fn ensure_target_file<Tgt: Target>(db_path: &Path) -> Result<(), String> {
    let target_file_path = db_path.join(TARGET_FILE);

    if !target_file_path.exists() {
        fs::write(&target_file_path, Tgt::target_id().to_string())
            .map_err(|e| format!("Failed to write TARGET file: {}", e))?;
        return Ok(());
    }

    validate_target_file::<Tgt>(db_path)
}

/// Check TARGET file in `db_path` is consistent with `Tgt`.
fn validate_target_file<Tgt: Target>(db_path: &Path) -> Result<(), String> {
    let target_file_path = db_path.join(TARGET_FILE);

    let stored_target_str = fs::read_to_string(&target_file_path)
        .map_err(|e| format!("Failed to read TARGET file: {}", e))?;
    let stored_target = stored_target_str
        .trim()
        .parse::<TargetId>()
        .map_err(|e| format!("Failed to parse TARGET file: {}", e))?;
    let expected_target = Tgt::target_id();
    if stored_target != expected_target {
        return Err("Database created for a different target".to_string());
    }
    Ok(())
}

/// Create TILESCALE file in `db_path` if it is missing, then validate it
/// matches `tile_scale`.
fn ensure_tilescale_file(db_path: &Path, tile_scale: TileScale) -> Result<(), String> {
    let tilescale_file_path = db_path.join(TILESCALE_FILE);

    if !tilescale_file_path.exists() {
        fs::write(&tilescale_file_path, tile_scale.to_string())
            .map_err(|e| format!("Failed to write TILESCALE file: {}", e))?;
        return Ok(());
    }

    validate_tilescale_file(db_path, tile_scale)
}

/// Check TILESCALE file in `db_path` matches `tile_scale`.
fn validate_tilescale_file(db_path: &Path, tile_scale: TileScale) -> Result<(), String> {
    let tilescale_file_path = db_path.join(TILESCALE_FILE);

    let stored_tilescale_str = fs::read_to_string(&tilescale_file_path)
        .map_err(|e| format!("Failed to read TILESCALE file: {}", e))?;
    let stored_tilescale = stored_tilescale_str
        .trim()
        .parse::<TileScale>()
        .map_err(|e| format!("Failed to parse TILESCALE file: {}", e))?;
    if stored_tilescale != tile_scale {
        return Err("Database created for a different tile scale".to_string());
    }
    Ok(())
}

// For some reason, [Prehashed]'s [Clone] impl requires that the value be [Copy].
fn prehashed_clone<T: Clone>(value: &Prehashed<T>) -> Prehashed<T> {
    let (inner, h) = Prehashed::as_parts(value);
    Prehashed::new(inner.clone(), *h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::row_major;
    use crate::scheduling::ActionT as _;
    use crate::scheduling_sugar::SchedulingSugar;
    use crate::spec;
    use crate::{
        common::Dtype,
        imp::ImplNode,
        memorylimits::{MemVec, MemoryLimits},
        scheduling::ApplyError,
        spec::arb_canonical_spec,
        target::{Avx2Target, Avx512Target, CpuMemory::L1, Memory},
        utils::{bit_length, bit_length_inverse_u32},
    };
    use itertools::{izip, Itertools};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::{collections::HashSet, fmt};

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);
    const TEST_SMALL_MEM: u64 = 256;

    #[test]
    fn test_background_save_persists_without_explicit_save_or_drop() {
        let db_path = tempfile::tempdir().unwrap();

        let mut spec: Spec<Avx2Target> =
            spec!(Move([1, 1], (f32, L1, row_major), (f32, L1, row_major)));
        spec.canonicalize().unwrap();
        let decisions = vec![(
            0,
            Cost {
                main: 7,
                peaks: MemVec::zero::<Avx2Target>(),
                depth: 0,
            },
        )];

        let mut db =
            FilesDatabase::new::<Avx2Target>(Some(db_path.path()), TileScale::Linear, 1, 128, 1);
        db.set_proactive_save_interval(Duration::from_millis(50));
        db.put(spec.clone(), decisions.clone());

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let reopened = FilesDatabase::new::<Avx2Target>(
                Some(db_path.path()),
                TileScale::Linear,
                1,
                128,
                1,
            );
            if reopened.get(&spec) == Some(ActionCostVec(decisions.clone())) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background save did not persist the page before timeout"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// A proactive save triggered by one put must consider dirty pages from earlier puts, not just
    /// the page touched by the triggering put.
    #[test]
    fn test_background_save_checks_previously_modified_pages() {
        let db_path = tempfile::tempdir().unwrap();
        let mut db =
            FilesDatabase::new::<Avx2Target>(Some(db_path.path()), TileScale::Linear, 1, 128, 1);
        db.set_proactive_save_interval(Duration::from_millis(50));
        let dirty_key = db.prehasher.prehash((
            (
                SpecKey::Matmul {
                    dtypes: [Dtype::Float32; 3],
                },
                Vec::new(),
            ),
            vec![10, 20],
        ));
        {
            let shard = &db.shards.0[db.shard_index(&dirty_key)];
            let mut shard_guard = shard.lock();
            shard_guard.cache.push(
                prehashed_clone(&dirty_key),
                Page {
                    contents: PageContents::RTree(Box::new(RTreePageContents::empty(2))),
                    modified: true,
                    last_save: None,
                },
            );
            shard_guard
                .modified_pages
                .insert(prehashed_clone(&dirty_key));
            shard_guard
                .modified_page_queue
                .push_back(prehashed_clone(&dirty_key));
        }

        let mut spec: Spec<Avx2Target> =
            spec!(Move([1, 1], (f32, L1, row_major), (f32, L1, row_major)));
        spec.canonicalize().unwrap();
        db.put(
            spec,
            vec![(
                0,
                Cost {
                    main: 7,
                    peaks: MemVec::zero::<Avx2Target>(),
                    depth: 0,
                },
            )],
        );

        let page_path = page_file_path(db_path.path(), Prehashed::as_inner(&dirty_key));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if page_path.exists() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background save did not check the previously modified page"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn test_proactive_save_disabled_for_temporary_database() {
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);
        assert!(!db.proactive_saves_enabled);
    }

    #[test]
    fn test_proactive_save_enabled_for_new_path_backed_database() {
        let db_path = tempfile::tempdir().unwrap();
        let db =
            FilesDatabase::new::<Avx2Target>(Some(db_path.path()), TileScale::Linear, 1, 128, 1);
        assert!(db.proactive_saves_enabled);
    }

    #[test]
    fn test_proactive_save_enabled_for_existing_path_backed_database() {
        let db_path = tempfile::tempdir().unwrap();
        let db =
            FilesDatabase::new::<Avx2Target>(Some(db_path.path()), TileScale::Linear, 1, 128, 1);
        drop(db);

        let db = FilesDatabase::open(Some(db_path.path()), TileScale::Linear, 1, 128, 1).unwrap();
        assert!(db.proactive_saves_enabled);
    }

    // TODO: What about leaves!? This shouldn't be called `Decision`.
    #[derive(Clone)]
    struct Decision<Tgt: Target> {
        spec: Spec<Tgt>,
        actions_costs: Vec<(ActionNum, Cost)>,
        children: Vec<Decision<Tgt>>,
    }

    type DecisionNode<Tgt> = (Spec<Tgt>, Vec<(ActionNum, Cost)>);

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
    fn test_get_impl_after_reopen() {
        let db_path = tempfile::tempdir().unwrap();

        let mut spec: Spec<Avx2Target> =
            spec!(Move([1, 1], (f32, L1, row_major), (f32, L1, row_major)));
        spec.canonicalize().unwrap();

        {
            let db = FilesDatabase::new::<Avx2Target>(
                Some(db_path.path()),
                TileScale::PowerOrThreePower,
                1,
                2,
                1,
            );
            let _ = spec.synthesize(&db);
        }

        let db = FilesDatabase::new::<Avx2Target>(
            Some(db_path.path()),
            TileScale::PowerOrThreePower,
            1,
            2,
            1,
        );
        let result = db.get_impl(&spec);
        assert!(
            result.is_some(),
            "Database should have the spec after reopening"
        );
    }

    #[test]
    #[should_panic(expected = "Database created for a different target")]
    fn test_opening_database_with_wrong_target_panics() {
        let db_path = tempfile::tempdir().unwrap();

        // Save an AVX2 database
        {
            let mut spec_avx2: Spec<Avx2Target> =
                spec!(Move([2, 2], (f32, L1, row_major), (f32, L1, row_major)));
            spec_avx2.canonicalize().unwrap();
            let _db = FilesDatabase::new::<Avx2Target>(
                Some(db_path.path()),
                TileScale::PowerOrThreePower,
                1,
                2,
                1,
            );
        }

        // Read as an AVX-512 database
        let _db = FilesDatabase::new::<Avx512Target>(
            Some(db_path.path()),
            TileScale::PowerOrThreePower,
            1,
            2,
            1,
        );
    }

    #[test]
    fn test_opening_database_writes_tilescale_file() {
        let db_path = tempfile::tempdir().unwrap();
        let _db = FilesDatabase::new::<Avx2Target>(
            Some(db_path.path()),
            TileScale::PowerOrThreePower,
            1,
            2,
            1,
        );

        let stored_tilescale =
            std::fs::read_to_string(db_path.path().join(TILESCALE_FILE)).unwrap();
        assert_eq!(stored_tilescale, TileScale::PowerOrThreePower.to_string());
    }

    #[test]
    fn test_open_does_not_create_missing_tilescale_file() {
        let db_path = tempfile::tempdir().unwrap();

        let result =
            FilesDatabase::open(Some(db_path.path()), TileScale::PowerOrThreePower, 1, 2, 1);

        assert!(result.is_err());
        assert!(!db_path.path().join(TILESCALE_FILE).exists());
    }

    #[test]
    #[should_panic(expected = "Database created for a different tile scale")]
    fn test_opening_database_with_wrong_tilescale_panics() {
        let db_path = tempfile::tempdir().unwrap();
        let wrong_tilescale = TileScale::Linear;
        std::fs::write(
            db_path.path().join(TILESCALE_FILE),
            wrong_tilescale.to_string(),
        )
        .unwrap();

        let _db = FilesDatabase::new::<Avx2Target>(
            Some(db_path.path()),
            TileScale::PowerOrThreePower,
            1,
            2,
            1,
        );
    }

    #[test]
    fn test_nonspatial_put_then_get_works_for_non_factorizable_specs() {
        let mut spec: Spec<Avx2Target> = spec!(Move([5], (u8, L1, row_major), (u8, L1, row_major)));
        spec.canonicalize().unwrap();
        let db = FilesDatabase::new::<Avx2Target>(None, TileScale::PowerOrThreePower, 1, 2, 1);
        assert!(!db.can_memoize_efficiently(&spec));

        let decisions = vec![(
            0,
            Cost {
                main: 7,
                peaks: MemVec::zero::<Avx2Target>(),
                depth: 0,
            },
        )];
        db.put(spec.clone(), decisions.clone());

        assert_eq!(db.get(&spec), Some(ActionCostVec(decisions)));
    }

    #[test]
    fn test_can_spatially_memoize_three_vector_avx512_spec() {
        let mut spec: Spec<Avx512Target> =
            spec!(Move([48], (u8, L1, row_major), (u8, L1, row_major)));
        spec.canonicalize().unwrap();
        let db = FilesDatabase::new::<Avx512Target>(None, TileScale::PowerOrThreePower, 1, 2, 1);
        assert!(db.can_memoize_efficiently(&spec));
    }

    #[test]
    fn test_nonspatial_put_then_get_works_after_reopen() {
        let db_path = tempfile::tempdir().unwrap();
        let mut spec: Spec<Avx2Target> = spec!(Move([5], (u8, L1, row_major), (u8, L1, row_major)));
        spec.canonicalize().unwrap();
        let decisions = vec![(
            0,
            Cost {
                main: 7,
                peaks: MemVec::zero::<Avx2Target>(),
                depth: 0,
            },
        )];

        {
            let db = FilesDatabase::new::<Avx2Target>(
                Some(db_path.path()),
                TileScale::PowerOrThreePower,
                1,
                2,
                1,
            );
            assert!(!db.can_memoize_efficiently(&spec));
            db.put(spec.clone(), decisions.clone());
        }

        let db = FilesDatabase::new::<Avx2Target>(
            Some(db_path.path()),
            TileScale::PowerOrThreePower,
            1,
            2,
            1,
        );
        assert_eq!(db.get(&spec), Some(ActionCostVec(decisions)));
    }

    #[test]
    fn test_background_save_persists_nonspatial_cache_without_explicit_save_or_drop() {
        let mut spec: Spec<Avx2Target> = spec!(Move([5], (u8, L1, row_major), (u8, L1, row_major)));
        spec.canonicalize().unwrap();
        let decisions = vec![(
            0,
            Cost {
                main: 7,
                peaks: MemVec::zero::<Avx2Target>(),
                depth: 0,
            },
        )];

        let db_path = tempfile::tempdir().unwrap();
        let mut db =
            FilesDatabase::new::<Avx2Target>(Some(db_path.path()), TileScale::PowerOfTwo, 1, 2, 1);
        db.set_proactive_save_interval(Duration::from_millis(100));
        assert!(!db.can_memoize_efficiently(&spec));
        db.put(spec.clone(), decisions.clone());

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let reopened = FilesDatabase::new::<Avx2Target>(
                Some(db_path.path()),
                TileScale::PowerOfTwo,
                1,
                2,
                1,
            );
            if reopened.get(&spec) == Some(ActionCostVec(decisions.clone())) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background save did not persist the non-spatial cache before timeout"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    proptest! {
        #[test]
        fn test_iter_blocks_in_single_dim_range(
            start in 0..12u32, extent in 0..8u32, block_dim_size in 1..4u32
        ) {
            let end = start + extent;
            let mut visited_indices = Vec::with_capacity(extent as usize + 1);
            for (_, within_block_range) in
                iter_blocks_in_single_dim_range(start, end, block_dim_size)
            {
                visited_indices.extend(within_block_range);
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
                    prop_assert_eq!(block_idx, last_block_idx + 1);
                    last_block_idx = block_idx;
                }
            }
        }

        #[test]
        fn test_simple_put_then_get_works(
            decision in arb_spec_and_decision::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let top_spec = decision.spec.clone();
            let top_actions_costs = decision.actions_costs.clone();
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::PowerOrThreePower, 1, 2, 1);
            for (spec, actions_costs) in decision.consume_decisions() {
                db.put(spec, actions_costs);
            }
            let expected = ActionCostVec(top_actions_costs);
            let get_result = db.get(&top_spec).expect("Spec should be in database");
            assert_eq!(get_result, expected, "Entries differed at {top_spec}");
        }

        // TODO: Add tests for top-2, etc. Impls
        #[test]
        fn test_put_then_get_fills_across_memory_limits(
            decision in arb_spec_and_decision::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let MemoryLimits::Standard(spec_limits) = decision.spec.1.clone();
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 128, 1);

            let top_logical_spec = decision.spec.0.clone();
            let top_actions_costs = decision.actions_costs.clone();

            // Put all decisions into database.
            for (spec, action_costs) in decision.consume_decisions() {
                db.put(spec, action_costs);
            }

            let peaks = if let Some((_, c)) = top_actions_costs.first() {
                c.peaks.clone()
            } else {
                MemVec::zero::<Avx2Target>()
            };
            let expected = ActionCostVec(top_actions_costs);
            izip!(Avx2Target::memories(), spec_limits.iter(), peaks.iter())
                .map(|(memory, l, p)| {
                    if memory.counts_registers() {
                        (u32::try_from(p).unwrap()..=u32::try_from(l).unwrap()).collect::<Vec<_>>()
                    } else {
                        assert!(l == 0 || l.is_power_of_two());
                        assert!(p == 0 || p.is_power_of_two());
                        (bit_length(p)..=bit_length(l)).map(bit_length_inverse_u32).collect::<Vec<_>>()
                    }
                })
                .multi_cartesian_product()
                .for_each(|limit_to_check_bits| {
                    let limit_to_check_vec = limit_to_check_bits
                        .into_iter()
                        .map_into()
                        .collect::<Vec<u64>>();
                    let limit_to_check = MemoryLimits::Standard(MemVec::new(limit_to_check_vec.try_into().unwrap()));
                    let spec_to_check = Spec(top_logical_spec.clone(), limit_to_check);
                    let get_result = db.get(&spec_to_check).expect("Spec should be in database");
                    assert_eq!(get_result, expected, "Entries differed at {spec_to_check}");
                });
        }

        // TODO: Fix and re-enable this test.
        //
        // #[test]
        // fn test_two_puts_return_correct_gets_for_second_put(
        //     decision_pair in arb_spec_and_decision_pair::<Avx2Target>()
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
        //             MemVec::zero::<Avx2Target>()
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
                let valid_actions = Tgt::actions(&spec.0)
                    .enumerate()
                    .filter_map(|(i, a)| match a.apply(&spec) {
                        Ok(applied) => Some((ActionNum::from(u16::try_from(i).unwrap()), applied)),
                        Err(ApplyError::NotApplicable(_)) => None,
                        Err(ApplyError::SpecNotCanonical) => {
                            unreachable!("Non-canonical Spec should be filtered: {spec}")
                        }
                    })
                    .collect::<Vec<_>>();
                let action_num_strategy = if valid_actions.is_empty() {
                    Just(None).boxed()
                } else {
                    prop_oneof![
                        1 => Just(None),
                        2 => proptest::sample::select(valid_actions).prop_map(Some),
                    ]
                    .boxed()
                };
                (Just(spec), action_num_strategy)
            })
            .prop_map(|(spec, action_opt)| {
                if let Some((action_num, imp)) = action_opt {
                    recursively_decide_with_action(&spec, action_num, &imp)
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
    fn recursively_decide_actions_with_visited<Tgt: Target>(
        spec: &Spec<Tgt>,
        visited: &mut HashSet<Spec<Tgt>>,
    ) -> Decision<Tgt> {
        assert!(
            !visited.contains(spec),
            "detected cycle: {spec} already in call stack; whole stack is\n{}",
            visited.iter().map(|s| s.to_string()).join("\n")
        );
        visited.insert(spec.clone());

        let result = if let Some((action_num, partial_impl)) = Tgt::actions(&spec.0)
            .enumerate()
            .filter_map(|(i, a)| match a.apply(spec) {
                Ok(imp) => Some((i, imp)),
                Err(ApplyError::NotApplicable(_)) => None,
                Err(ApplyError::SpecNotCanonical) => panic!(),
            })
            .next()
        {
            recursively_decide_with_action_with_visited(
                spec,
                action_num.try_into().unwrap(),
                &partial_impl,
                visited,
            )
        } else {
            Decision {
                spec: spec.clone(),
                actions_costs: vec![],
                children: vec![],
            }
        };

        visited.remove(spec);
        result
    }

    fn recursively_decide_with_action<Tgt: Target>(
        spec: &Spec<Tgt>,
        action_num: ActionNum,
        partial_impl: &ImplNode<Tgt>,
    ) -> Decision<Tgt> {
        recursively_decide_with_action_with_visited(
            spec,
            action_num,
            partial_impl,
            &mut HashSet::new(),
        )
    }

    fn recursively_decide_with_action_with_visited<Tgt: Target>(
        spec: &Spec<Tgt>,
        action_num: ActionNum,
        partial_impl: &ImplNode<Tgt>,
        visited: &mut HashSet<Spec<Tgt>>,
    ) -> Decision<Tgt> {
        let mut children = Vec::new();
        let mut unsat = false;
        partial_impl.visit_leaves(&mut |leaf| {
            if let ImplNode::SpecApp(spec_app) = leaf {
                let cd = recursively_decide_actions_with_visited(&spec_app.0, visited);
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

        let cost = Cost::from_node_and_child_costs(
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
            actions_costs: vec![(action_num, cost)],
            children,
        }
    }

    proptest! {
        #[test]
        fn test_bimap_apply_panics_when_defined_for_is_false(
            spec in arb_canonical_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            // Binary-scaling database
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::PowerOrThreePower, 1, 2, 1);
            let bimap = db.spec_bimap::<Avx2Target>();

            if BiMap::defined_for(&bimap, &spec) {
                // If can_memoize_efficiently returns true, apply should not panic
                let _ = BiMap::apply(&bimap, &spec);
            } else {
                // If can_memoize_efficiently returns false, apply should panic
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    BiMap::apply(&bimap, &spec)
                }));
                assert!(result.is_err(), "Expected apply to panic when defined_for is false");
            }
        }

        #[test]
        fn test_can_memoize_efficiently_returns_true_for_representable_specs(
            spec in arb_canonical_spec::<Avx2Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let db = FilesDatabase::new::<Avx2Target>(None, TileScale::Linear, 1, 2, 1);
            assert!(
                db.can_memoize_efficiently(&spec),
                "All specs should be efficiently memoizable with TileScale::Linear, \
                but {spec} was not",
            );
        }
    }
}
