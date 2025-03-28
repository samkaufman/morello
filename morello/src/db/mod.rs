mod pagecontents;

use crate::common::DimSize;
use crate::cost::{Cost, NormalizedCost};
use crate::datadeps::SpecKey;
use crate::db::pagecontents::{PageContents, RTreePageContents};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{AsBimap, BiMap};
use crate::grid::linear::{BimapInt, BimapSInt};
use crate::imp::functions::FunctionApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::rtree::RTreeDyn;
use crate::scheduling::{Action, ActionEncodeDecode, ActionT as _, SpecGeometry};
use crate::spec::{FillValue, LogicalSpecSurMap, PrimitiveBasicsBimap, Spec, SpecSurMap};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use itertools::Itertools;
use parking_lot::{Mutex, MutexGuard};
use prehash::{new_prehashed_set, DefaultPrehasher, Prehashed, PrehashedSet, Prehasher};
use serde::{Deserialize, Serialize};
use wtinylfu::WTinyLfuCache;

use std::collections::HashSet;
use std::fs;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut, Range};
use std::path::{self, Path};
use std::sync::{mpsc, Arc};

#[cfg(feature = "db-stats")]
use {
    std::sync::atomic::{self, AtomicU64},
    std::time::Instant,
};

pub(crate) type DbKey = (TableKey, Vec<BimapInt>);
pub(crate) type TableKey = (SpecKey, Vec<(Layout,)>);
type PageKey = DbKey;
pub type ActionNum = u16;

/// The number of shards/locks per thread.
const THREAD_SHARDS: usize = 2;
const CHANNEL_SIZE: usize = 2;
/// Compress pages when writing to disk.
const COMPRESS_PAGES: bool = true;

pub struct FilesDatabase {
    #[allow(dead_code)] // read only when db-stats enabled; otherwise only affects Drop
    dir_handle: Arc<DirPathHandle>,
    binary_scale_shapes: bool,
    k: u8,
    shards: ShardVec,
    prehasher: DefaultPrehasher,
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
}

struct Shard {
    cache: WTinyLfuCache<Prehashed<PageKey>, Page>,
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

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionNormalizedCostVec(pub Vec<(ActionNum, NormalizedCost)>);

pub enum GetPreference<T, V> {
    Hit(T),
    Miss(Option<V>),
}

/// Intersections yielded by [FilesDatabase::intersect].
#[derive(Debug, Clone)]
pub(crate) struct Intersection<T> {
    pub(crate) bottom: Vec<BimapInt>,
    pub(crate) top: Vec<BimapInt>,
    pub(crate) action_costs: ActionNormalizedCostVec,
    pub(crate) dep_meta: T,
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
    ) -> Self {
        let dir_handle = Arc::new(match file_path {
            Some(path) => {
                fs::create_dir_all(path).unwrap();
                DirPathHandle::Persisted(path.to_owned())
            }
            None => DirPathHandle::TempDir(tempfile::TempDir::new().unwrap()),
        });
        log::info!("Opening database at: {}", dir_handle.path().display());

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
        Some(
            root_results
                .as_ref()
                .iter()
                .map(|(action_num, _cost)| {
                    let root = Action::decode(query, *action_num).apply(query).unwrap();
                    let children = root.children();
                    let new_children = children.iter().map(|c| construct_impl(self, c));
                    root.replace_children(new_children)
                })
                .collect(),
        )
    }

    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
    ) -> GetPreference<ActionCostVec, Vec<ActionNum>>
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
        let global_pt_u8 = global_pt
            .iter()
            .map(|&x| u8::try_from(x).unwrap())
            .collect::<Vec<_>>();

        let page_pt = blockify_point(&global_pt);
        let page_key = self.prehasher.prehash((table_key, page_pt));

        let page: &Page = &self.load_live_page(&page_key);
        page.contents.get_with_preference(&query, &global_pt_u8)
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
    /// Passing a non-canonical [Spec] is a logic error.
    pub fn page_id<Tgt>(&self, spec: &Spec<Tgt>) -> PageId<'_>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(spec.is_canonical());

        let bimap = self.spec_bimap();
        let (table_key, global_pt_lhs) = bimap.apply(spec);
        let page_pt = blockify_point(&global_pt_lhs);
        PageId {
            db: self,
            table_key,
            page_id: page_pt,
        }
    }

    pub fn put_normalized<Tgt>(
        &self,
        mut spec: Spec<Tgt>,
        normalized_decisions: ActionNormalizedCostVec,
    ) where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        spec.canonicalize().unwrap();

        // Check that all costs in decisions have peak memory less than or equal to spec's
        // memory limits.
        debug_assert!(
            normalized_decisions.0.iter().all(|(_, c)| {
                let MemoryLimits::Standard(limits) = &spec.1;
                &c.peaks <= limits
            }),
            "peak memory of an action exceeds memory limits of {spec}: {normalized_decisions:?}"
        );

        #[cfg(feature = "db-stats")]
        self.stats.puts.fetch_add(1, atomic::Ordering::Relaxed);

        let bimap = self.spec_bimap();
        let (table_key, (bottom, top)) = put_range_to_fill(&bimap, &spec, &normalized_decisions.0);

        // Construct an iterator over all pages (tiles) to fill.
        let rank = bottom.len();
        let pages_iter = bottom
            .into_iter()
            .zip(&top)
            .enumerate()
            .map(|(dim, (b, t))| {
                iter_blocks_in_single_dim_range(b, *t, block_size_dim(dim, rank).into())
            })
            .multi_cartesian_product();

        // Reuse the follow two values to avoid some allocations.
        let mut key_tuple = (table_key, vec![]);
        let mut dim_ranges = vec![];
        for joined_row in pages_iter {
            key_tuple.1.clear();
            key_tuple.1.reserve_exact(joined_row.len());
            dim_ranges.clear();
            dim_ranges.reserve_exact(joined_row.len());
            for (b, t) in joined_row {
                key_tuple.1.push(b);
                dim_ranges.push(t);
            }

            // Load `page_guard`. We do some awkward mutation of `key_tuple.1` to avoid cloning
            // `table_key`/`key_tuple.0` on each iteration.
            let key = self.prehasher.prehash(key_tuple);
            let mut page_guard = self.load_live_page_mut(&key);
            key_tuple = Prehashed::into_inner(key);

            page_guard.modified = true;
            page_guard
                .contents
                .fill_region(self.k, &dim_ranges, &normalized_decisions);
        }
    }

    pub fn put<Tgt>(&self, spec: Spec<Tgt>, decisions: Vec<(ActionNum, Cost)>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        // Since this put is for a single logical Spec, we can normalize the cost with that Spec's
        // volume.
        let normalized_decisions =
            ActionNormalizedCostVec::normalize(ActionCostVec(decisions), spec.0.volume());
        self.put_normalized(spec, normalized_decisions)
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

    // TODO: Return normalized cost or rescaled?
    // TODO: We should return ptrs. to T, not require it to be Clone.
    pub(crate) fn intersect<'a, T: Clone>(
        &'a self,
        table_key: &'a TableKey,
        rtree: &'a RTreeDyn<T>,
    ) -> impl Iterator<Item = Intersection<T>> + 'a {
        let rank = rtree.dim_count();
        let page_shape = (0..rank)
            .map(|dim| block_size_dim(dim, rank))
            .collect::<Vec<_>>();
        rtree_grid_intersections(rtree, page_shape).flat_map(move |page_pt| {
            let key = self.prehasher.prehash((table_key.clone(), page_pt.clone()));
            let Page {
                contents: PageContents::RTree(page_contents),
                modified: _,
            } = &*self.load_live_page(&key);
            let page_tree = &page_contents.as_ref().0;
            debug_assert_eq!(page_tree.dim_count(), rank);

            page_tree
                .intersection_candidates_with_other_tree(rtree)
                .map(
                    |((db_rect_bottom, db_rect_top, db_value), (b_bottom, b_top, dep_meta))| {
                        let (intersect_bottom, intersect_top) =
                            intersect(db_rect_bottom, db_rect_top, b_bottom, b_top);
                        let intersect_bottom_u32 = intersect_bottom
                            .iter()
                            .map(|&x| BimapInt::try_from(x).unwrap())
                            .collect::<Vec<_>>();
                        let intersect_top_u32 = intersect_top
                            .iter()
                            .map(|&x| BimapInt::try_from(x).unwrap())
                            .collect::<Vec<_>>();
                        let v = ActionNormalizedCostVec(match db_value {
                            Some((cost_intensity, peaks, depth, action_num)) => {
                                vec![(
                                    *action_num,
                                    NormalizedCost {
                                        intensity: *cost_intensity,
                                        peaks: peaks.clone(),
                                        depth: *depth,
                                    },
                                )]
                            }
                            None => vec![],
                        });
                        Intersection {
                            bottom: intersect_bottom_u32,
                            top: intersect_top_u32,
                            action_costs: v,
                            dep_meta: dep_meta.clone(),
                        }
                    },
                )
                // TODO: Avoid `collect`. Just return the iterator.
                .collect::<Vec<_>>()
                .into_iter()
        })
    }

    pub(crate) fn subtract_from<Tgt: Target>(&self, minuhend: &mut SpecGeometry<Tgt>) {
        // TODO: Somehow enforce that SpecGeometry's BiMap matches the databases's BiMap.
        for (table_key, tree) in minuhend.0.iter_mut() {
            self.subtract_from_rtree(table_key, tree)
        }
    }

    pub(crate) fn subtract_from_rtree<U: Clone>(
        &self,
        table_key: &TableKey,
        minuhend: &mut RTreeDyn<U>,
    ) {
        let rank = minuhend.dim_count();
        let page_shape = (0..rank)
            .map(|dim| block_size_dim(dim, rank))
            .collect::<Vec<_>>();
        let intersecting_pages = rtree_grid_intersections(minuhend, page_shape).collect::<Vec<_>>();
        for page_pt in intersecting_pages {
            let key = self.prehasher.prehash((table_key.clone(), page_pt.clone()));
            let Page {
                contents: PageContents::RTree(page_contents),
                modified: _,
            } = &*self.load_live_page(&key);
            let page_tree = &page_contents.as_ref().0;
            debug_assert_eq!(page_tree.dim_count(), rank);
            minuhend.subtract_tree(page_tree);
        }
    }

    /// Return a bidirectional map from [Spec]s to tuples of table keys and their coordinates.
    pub(crate) fn spec_bimap<Tgt>(&self) -> impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
        db_spec_bimap(self.binary_scale_shapes)
    }

    fn load_live_page<'a>(&'a self, key: &Prehashed<PageKey>) -> impl Deref<Target = Page> + 'a {
        self.load_live_page_mut(key)
    }

    fn load_live_page_mut<'a>(
        &'a self,
        key: &Prehashed<PageKey>,
    ) -> impl DerefMut<Target = Page> + 'a {
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
    pub fn analyze<Tgt>(&self, output_dir: &path::Path, sample: usize, skip_read_errors: bool)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
    {
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
            .write_record(["page_path", "rects"])
            .unwrap();
        writers
            .block_action_writer
            .write_record(["page_path", "block_pt", "action"])
            .unwrap();

        analyze_visit_dir::<Tgt>(
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
    }
}

impl PageId<'_> {
    /// Returns `true` if the page contains the given canonical [Spec].
    ///
    /// Passing a non-canonical [Spec] is a logic error.
    pub fn contains<Tgt>(&self, spec: &Spec<Tgt>) -> bool
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        debug_assert!(spec.is_canonical());

        let bimap = self.db.spec_bimap();
        let (table_key, global_pt) = bimap.apply(spec);
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
                                            "Continuing after errors reading page; {:?} and {:?}",
                                            zstd_error,
                                            plain_error
                                        );
                                        Page {
                                            contents: PageContents::RTree(Box::new(
                                                RTreePageContents::empty(key.1.len()),
                                            )),
                                            modified: false,
                                        }
                                    }
                                },
                                Err(_) => Page {
                                    contents: PageContents::RTree(Box::new(
                                        RTreePageContents::empty(key.1.len()),
                                    )),
                                    modified: false,
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
                                log::debug!("Evicting page");
                            }

                            if let Some(parent) = path.parent() {
                                fs::create_dir_all(parent).unwrap();
                            }
                            let file = fs::File::create(&path).unwrap();
                            if COMPRESS_PAGES {
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
                self.async_put(Prehashed::into_inner(evicted_key), evicted_value.contents);
            }
        }
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
        self.thread_tx
            .send(ShardThreadMsg::Get(prehashed_clone(key)))
            .unwrap();
        true
    }

    fn async_put(&self, key: PageKey, value: PageContents) {
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
    })
}

pub(crate) fn db_spec_bimap<Tgt>(
    binary_scale_shapes: bool,
) -> impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    let surmap = SpecSurMap::<Tgt, _, _, _> {
        logical_spec_surmap: LogicalSpecSurMap::new(
            PrimitiveBasicsBimap {
                binary_scale_shapes,
            },
            |_: &[DimSize], dtype| TensorSpecAuxNonDepBimap::new(dtype),
        ),
        memory_limits_bimap: MemoryLimitsBimap::default(),
    };
    surmap.into_bimap()
}

#[cfg(feature = "db-stats")]
fn analyze_visit_dir<Tgt>(
    root: &path::Path,
    path: &path::Path,
    writers: &mut AnalyzeWriters,
    sample: usize,
    skip_read_errors: bool,
) where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Domain = Tgt::Level, Codomain = u8>,
{
    // Since we don't revisit blocks, bypass the in-mem. cache and read from disk.
    for file_entry in fs::read_dir(path).unwrap() {
        let file_entry = file_entry.unwrap();
        if let Some("PRECOMPUTE") = file_entry.path().file_name().unwrap().to_str() {
            continue;
        }

        let entry_path = file_entry.path();
        if entry_path.is_dir() {
            analyze_visit_dir::<Tgt>(root, &file_entry.path(), writers, sample, skip_read_errors);
            continue;
        }

        // Choose with 1/sample probability to skip this file.
        if sample > 1 && rand::random::<usize>() % sample != 0 {
            continue;
        }

        let shortened_entry_path_str = entry_path.strip_prefix(root).unwrap();
        let entry_path_str = format!("{}", shortened_entry_path_str.display());

        let page_file = fs::File::open(entry_path).unwrap();
        let page = match read_any_format(page_file) {
            Ok(r) => r,
            Err(e) => {
                if skip_read_errors {
                    log::warn!("Error reading page: {:?}", e);
                    continue;
                }
                panic!("Error reading page: {e:?}");
            }
        };

        let PageContents::RTree(r) = &page.contents;
        writers
            .block_writer
            .write_record([&entry_path_str, &r.rect_count().to_string()])
            .unwrap();

        writers.page_writer.write_record([entry_path_str]).unwrap();
    }
}

fn construct_impl<Tgt>(db: &FilesDatabase, imp: &ImplNode<Tgt>) -> ImplNode<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    match imp {
        ImplNode::SpecApp(p) => {
            let body = db
                .get_impl(&p.0)
                .unwrap_or_else(|| panic!("Database should have the sub-Spec: {}", p.0))
                .first()
                .unwrap_or_else(|| panic!("Database sub-Spec should be satisfiable: {}", p.0))
                .clone();
            ImplNode::FunctionApp(FunctionApp {
                body: Box::new(body),
                parameters: p.1.clone(),
                spec: Some(p.0.clone()),
            })
        }
        _ => imp.replace_children(imp.children().iter().map(|c| construct_impl(db, c))),
    }
}

fn block_size_dim(dim: usize, dim_count: usize) -> u8 {
    if dim >= dim_count - LEVEL_COUNT {
        31
    } else {
        8
    }
}

/// Compute the block coordinate from a global coordinate.
fn blockify_point(pt: &[BimapInt]) -> Vec<BimapInt> {
    let rank = pt.len();
    pt.iter()
        .enumerate()
        .map(|(i, &d)| d / BimapInt::from(block_size_dim(i, rank)))
        .collect()
}

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt, B>(
    bimap: &B,
    spec: &Spec<Tgt>,
    impls: &[(ActionNum, NormalizedCost)],
) -> (TableKey, (Vec<BimapInt>, Vec<BimapInt>))
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    B: BiMap<Domain = Spec<Tgt>, Codomain = DbKey>,
{
    // Compute worst-case bound of solutions' memory usage. This lower bounds the range.
    let per_level_peaks = impls
        .iter()
        .fold(MemVec::zero::<Tgt>(), |acc, (_, cost)| acc.max(&cost.peaks));

    // Compute the complete upper and lower bounds from the given Spec and that Spec modified with
    // the peaks' bound (computed above).
    let upper_inclusive = bimap.apply(spec);
    let lower_inclusive = {
        let mut lower_bound_spec = spec.clone();
        lower_bound_spec.1 = MemoryLimits::Standard(per_level_peaks);
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
    let ((spec_key, table_key_rest), block_pt) = page_key;
    let mut spec_key_dir_name = match spec_key {
        SpecKey::OnePrefix { dtypes } => root
            .join("OnePrefix")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Matmul { dtypes } => root
            .join("Matmul")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Conv { dtypes } => root
            .join("Conv")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Broadcast { dim, dtypes } => root
            .join("Broadcast")
            .join(dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Softmax { scan_dim, dtypes } => root
            .join("Softmax")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxComplete { scan_dim, dtypes } => root
            .join("SoftmaxComplete")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndMax { scan_dim, dtypes } => root
            .join("SoftmaxDenominatorAndMax")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndUnscaled { scan_dim, dtypes } => root
            .join("SoftmaxDenominatorAndUnscaled")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominatorAndUnscaledFromMax { scan_dim, dtypes } => root
            .join("SoftmaxDenominatorAndUnscaledFromMax")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::SoftmaxDenominator { scan_dim, dtypes } => root
            .join("SoftmaxDenominator")
            .join(scan_dim.to_string())
            .join(dtypes.iter().join("_")),
        SpecKey::DivideVec { dtypes } => root
            .join("DivideVec")
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::DivideVecScalar { scan_dim, dtypes } => root
            .join("DivideVecScalar")
            .join(scan_dim.to_string())
            .join(dtypes.iter().map(|d| d.to_string()).join("_")),
        SpecKey::Max { dtypes, dim } => root
            .join("Max")
            .join(dim.to_string())
            .join(dtypes.iter().join("_")),
        SpecKey::Move { dtypes } => root.join("Move").join(dtypes.iter().join("_")),
        SpecKey::Fill {
            dtype,
            value: FillValue::Zero,
        } => root.join("FillZero").join(dtype.to_string()),
        SpecKey::Fill {
            dtype,
            value: FillValue::NegInf,
        } => root.join("FillNegInf").join(dtype.to_string()),
        SpecKey::Fill {
            dtype,
            value: FillValue::Min,
        } => root.join("FillMin").join(dtype.to_string()),
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
    for (l,) in table_key_rest {
        spec_key_dir_name = spec_key_dir_name.join(l.to_string());
    }
    spec_key_dir_name.join(block_pt.iter().map(|p| p.to_string()).join("_"))
}

// For some reason, [Prehashed]'s [Clone] impl requires that the value be [Copy].
fn prehashed_clone<T: Clone>(value: &Prehashed<T>) -> Prehashed<T> {
    let (inner, h) = Prehashed::as_parts(value);
    Prehashed::new(inner.clone(), *h)
}

/// Calculates the cells of a regular grid that intersect rectangles in an [RTreeDyn].
fn rtree_grid_intersections<T>(
    rtree: &RTreeDyn<T>,
    cell_size: Vec<u8>,
) -> impl Iterator<Item = Vec<BimapInt>> + '_ {
    // TODO: This is a slow implementation. Instead, use the structure of the tree to prune.
    debug_assert!(
        !cell_size.iter().any(|c| *c == 0),
        "cell cannot have size-zero dimension"
    );
    debug_assert_eq!(rtree.dim_count(), cell_size.len());
    let blocks_touched = rtree
        .iter()
        .flat_map(move |(bottom, top, _)| {
            if bottom.iter().any(|&v| v < 0) || top.iter().any(|&v| v < 0) {
                todo!("Negative values not supported");
            }
            let transposed = bottom.iter().zip(top.iter()).zip(&cell_size);
            transposed
                .map(|((&b, &t), &c)| {
                    let block_bottom = BimapInt::try_from(b).unwrap() / BimapInt::from(c);
                    let block_top = BimapInt::try_from(t).unwrap() / BimapInt::from(c);
                    block_bottom..=block_top
                })
                .multi_cartesian_product()
        })
        .collect::<HashSet<_>>();
    blocks_touched.into_iter()
}

// TODO: Shouldn't need this. Inline it.
fn intersect(
    db_rect_bottom: &[BimapSInt],
    db_rect_top: &[BimapSInt],
    b_bottom: &[BimapSInt],
    b_top: &[BimapSInt],
) -> (Vec<BimapSInt>, Vec<BimapSInt>) {
    let rank = db_rect_bottom.len();
    let mut intersection_bottom = Vec::with_capacity(rank);
    let mut intersection_top = Vec::with_capacity(rank);

    for dim in 0..rank {
        let bottom = db_rect_bottom[dim].max(b_bottom[dim]);
        let top = db_rect_top[dim].min(b_top[dim]);
        debug_assert!(bottom <= top);
        intersection_bottom.push(bottom);
        intersection_top.push(top);
    }

    (intersection_bottom, intersection_top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        imp::ImplNode,
        memorylimits::{MemVec, MemoryLimits},
        scheduling::ApplyError,
        spec::arb_canonical_spec,
        target::{MemoryLevel, X86Target},
        utils::{bit_length, bit_length_inverse_u32},
    };
    use itertools::{izip, Itertools};
    use nonzero::nonzero as nz;
    use proptest::prelude::*;
    use std::{collections::HashSet, fmt};

    const TEST_SMALL_SIZE: DimSize = nz!(2u32);
    const TEST_SMALL_MEM: u64 = 256;

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
            decision in arb_spec_and_decision::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let top_spec = decision.spec.clone();
            let top_actions_costs = decision.actions_costs.clone();
            let db = FilesDatabase::new(None, false, 1, 2, 1);
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
            decision in arb_spec_and_decision::<X86Target>(Some(TEST_SMALL_SIZE), Some(TEST_SMALL_MEM))
        ) {
            let MemoryLimits::Standard(spec_limits) = decision.spec.1.clone();
            let db = FilesDatabase::new(None, false, 1, 128, 1);

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
            let expected = ActionCostVec(top_actions_costs);
            izip!(X86Target::levels(), spec_limits.iter(), peaks.iter())
                .map(|(level, l, p)| {
                    if level.counts_registers() {
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

    #[test]
    #[should_panic(expected = "zero dimension")]
    fn test_rtree_grid_intersections_panics_on_empty_cells() {
        let t = RTreeDyn::empty(2);
        // should panic without needing to consume the iterator
        #[allow(unused_must_use)]
        {
            rtree_grid_intersections::<u8>(&t, vec![0, 1]);
        }
    }

    #[test]
    fn test_rtree_grid_intersections_empty() {
        let t = RTreeDyn::<u8>::empty(2);
        let output = rtree_grid_intersections(&t, vec![2, 2]).collect::<HashSet<_>>();
        assert!(output.is_empty());
    }

    #[test]
    fn test_rtree_grid_intersections_1() {
        let mut t = RTreeDyn::<u8>::empty(2);
        t.insert(&[0, 0], &[0, 0], 9);
        let output = rtree_grid_intersections(&t, vec![2, 2]).collect::<HashSet<_>>();
        assert_eq!(output, HashSet::from([vec![0, 0]]));
    }

    #[test]
    fn test_rtree_grid_intersections_2() {
        let mut t = RTreeDyn::<u8>::empty(2);
        t.insert(&[1, 1], &[1, 1], 9);
        let output = rtree_grid_intersections(&t, vec![2, 2]).collect::<HashSet<_>>();
        assert_eq!(output, HashSet::from([vec![0, 0]]));
    }

    #[test]
    fn test_rtree_grid_intersections_3() {
        let mut t = RTreeDyn::<u8>::empty(2);
        t.insert(&[1, 1], &[2, 1], 9);
        let output = rtree_grid_intersections(&t, vec![2, 2]).collect::<HashSet<_>>();
        assert_eq!(output, HashSet::from([vec![0, 0], vec![1, 0]]));
    }

    // #[test]
    // #[should_panic(expected = "already exists")]
    // fn test_db_overwriting_put_panics() {
    //     let rm2 = row_major(2);
    //     let spec: Spec<X86Target> = Spec(
    //         lspec!(Matmul(
    //             [64, 64, 64],
    //             (u32, GL, rm2.clone()),
    //             (u32, GL, rm2.clone()),
    //             (u32, GL, rm2),
    //             serial
    //         )),
    //         MemoryLimits::Standard(MemVec::zero::<X86Target>()),
    //     );
    //     let cost = Cost {
    //         main: 0,
    //         peaks: MemVec::zero::<X86Target>(),
    //         depth: 1,
    //     };
    //     let db = FilesDatabase::new(None, false, 1, 128, 1);
    //     db.put(spec.clone(), vec![]);
    //     db.put(spec.clone(), vec![(1, cost.clone())]);
    // }

    fn arb_spec_and_decision<Tgt: Target>(
        max_size: Option<DimSize>,
        max_memory: Option<u64>,
    ) -> impl Strategy<Value = Decision<Tgt>> {
        arb_canonical_spec::<Tgt>(max_size, max_memory)
            .prop_flat_map(|spec| {
                let valid_actions = Tgt::actions(&spec.0)
                    .filter_map(|a| match a.apply(&spec) {
                        Ok(applied) => Some((a.encode(&spec), applied)),
                        Err(ApplyError::NotApplicable(_)) => None,
                        Err(ApplyError::SpecNotCanonical) => {
                            unreachable!("Non-canonical Spec should be filtered: {spec}")
                        }
                    })
                    .collect::<Vec<_>>();
                let action_strategy = if valid_actions.is_empty() {
                    Just(None).boxed()
                } else {
                    prop_oneof![
                        1 => Just(None),
                        2 => proptest::sample::select(valid_actions).prop_map(Some),
                    ]
                    .boxed()
                };
                (Just(spec), action_strategy)
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
            .filter_map(|a| match a.apply(spec) {
                Ok(imp) => Some((a.encode(spec), imp)),
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
}
