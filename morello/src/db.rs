use crate::common::DimSize;
use crate::cost::{Cost, MainCost};
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{AsBimap, BiMap};
use crate::grid::linear::BimapInt;
use crate::imp::subspecs::SpecApp;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::ndarray::NDArray;
use crate::pprint::PrintableAux;
use crate::spec::{LogicalSpecSurMap, PrimitiveBasicsBimap, Spec, SpecSurMap};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use anyhow::Result;
use divrem::DivRem;
use itertools::Itertools;
use parking_lot::{Mutex, MutexGuard};
use prehash::{new_prehashed_set, DefaultPrehasher, Prehashed, PrehashedSet, Prehasher};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use wtinylfu::WTinyLfuCache;

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, DerefMut, Range};
use std::sync::Arc;
use std::{iter, path};

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

type DbKey = (TableKey, SmallVec<[BimapInt; 10]>); // TODO: Rename to BlockKey for consistency?
type TableKey = (SpecKey, SmallVec<[(Layout, u8, Option<NonZeroU32>); 3]>);
type SuperBlockKey = DbKey;
type SuperBlock = HashMap<SmallVec<[BimapInt; 10]>, DbBlock>;
pub type ActionIdx = u16;

// TODO: Select these at runtime.
const CONCURRENT_CACHE_SHARDS: usize = 16;
const CACHE_PER_SHARD_SIZE: usize = 64;
const CACHE_PER_SHARD_SAMPLES: usize = 8;
const SUPERBLOCK_FACTOR: BimapInt = 4;
const CHANNEL_SIZE: usize = 2;

#[derive(Clone, Debug)]
pub struct DbImplAux<Tgt: Target>(Option<(Spec<Tgt>, Cost)>);

pub struct RocksDatabase {
    db: Arc<rocksdb::DB>, // TODO: Arc shouldn't be necessary.
    binary_scale_shapes: bool,
    k: u8,
    shards: ShardArray,
    prehasher: DefaultPrehasher,
}

struct ShardArray([Mutex<Shard>; CONCURRENT_CACHE_SHARDS]);

pub struct PageId<'a> {
    db: &'a RocksDatabase,
    pub(crate) table_key: TableKey,
    pub(crate) superblock_id: SmallVec<[BimapInt; 10]>,
}

struct Shard {
    cache: WTinyLfuCache<Prehashed<SuperBlockKey>, SuperBlock>,
    outstanding_gets: PrehashedSet<SuperBlockKey>,
    thread: Option<std::thread::JoinHandle<()>>,
    thread_tx: std::sync::mpsc::SyncSender<ShardThreadMsg>,
    thread_rx: std::sync::mpsc::Receiver<ShardThreadResponse>,
}

enum ShardThreadMsg {
    Get(Prehashed<SuperBlockKey>),
    Put(SuperBlockKey, SuperBlock),
    Exit,
}

enum ShardThreadResponse {
    Loaded(Prehashed<SuperBlockKey>, SuperBlock),
}

/// Stores a [Database] block. This may be a single value if all block entries have been filled with
/// the same [ActionCostVec], or an n-dimensional array along with a count of identical entries
/// accumulated until the first differing entry.
///
/// This isn't designed to be compressed in the case that a non-identical value is added and later
/// overwritten with an identical value. In that case, `matches` will be `None` and all values
/// would need to be scanned to determine whether they are all identical and, to set `matches`, how
/// many there are.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum DbBlock {
    Whole(Box<WholeBlock>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WholeBlock {
    pub filled: NDArray<u8>, // 0 is empty; otherwise n - 1 = # of actions.
    pub main_costs: NDArray<MainCost>,
    pub peaks: NDArray<MemVec>,
    pub depths_actions: NDArray<(u8, ActionIdx)>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCostVec(pub SmallVec<[(ActionIdx, Cost); 1]>);

pub enum GetPreference<T, V> {
    Hit(T),
    Miss(Option<V>),
}

impl<Tgt: Target> PrintableAux for DbImplAux<Tgt> {
    fn extra_column_titles(&self) -> Vec<String> {
        iter::once("Logical Spec".to_owned())
            .chain(Tgt::levels().iter().map(|lvl| lvl.to_string()))
            .chain(iter::once("Cost".to_owned()))
            .collect()
    }

    fn extra_column_values(&self) -> Vec<String> {
        if let Some((spec, cost)) = &self.0 {
            iter::once(spec.0.to_string())
                .chain(cost.peaks.iter().map(|p| p.to_string()))
                .chain(iter::once(cost.main.to_string()))
                .collect()
        } else {
            vec![String::from(""); Tgt::levels().len() + 2]
        }
    }

    fn c_header(&self) -> Option<String> {
        self.0.as_ref().map(|(spec, _)| spec.to_string())
    }
}

impl<Tgt: Target> Default for DbImplAux<Tgt> {
    fn default() -> Self {
        DbImplAux(None)
    }
}

impl RocksDatabase {
    pub fn try_new(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
        k: u8,
    ) -> Result<Self> {
        let resolved_file_path = file_path
            .map(|p| p.to_owned())
            .unwrap_or_else(|| tempfile::TempDir::new().unwrap().into_path());
        log::info!("Opening database at: {}", resolved_file_path.display());
        let db = Arc::new(rocksdb::DB::open_default(resolved_file_path)?);
        let shards = ShardArray(std::array::from_fn(|i| Mutex::new(Shard::new(i, &db))));
        Ok(Self {
            db,
            binary_scale_shapes,
            k,
            shards,
            prehasher: DefaultPrehasher::default(),
        })
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

    pub fn get_impl<Tgt>(&self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let Some(root_results) = self.get(query) else {
            return None;
        };
        let actions = query.0.actions();
        Some(
            root_results
                .as_ref()
                .iter()
                .map(|(action_idx, cost)| {
                    let root = actions[(*action_idx).into()]
                        .apply_with_aux(query, DbImplAux(Some((query.clone(), cost.clone()))))
                        .unwrap();
                    let children = root.children();
                    let new_children = children
                        .iter()
                        .map(|c| construct_impl(self, c))
                        .collect::<Vec<_>>();
                    root.replace_children(new_children.into_iter())
                })
                .collect::<SmallVec<_>>(),
        )
    }

    pub fn get_with_preference<Tgt>(
        &self,
        query: &Spec<Tgt>,
    ) -> GetPreference<ActionCostVec, SmallVec<[ActionIdx; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let bimap = self.spec_bimap();
        let (table_key, global_pt) = bimap.apply(query);
        let (block_pt, inner_pt) = blockify_point(global_pt);

        let superblock_pt = superblockify_pt(&block_pt);
        let superblock_key = self.prehasher.prehash((table_key, superblock_pt));

        let superblock: &SuperBlock = &self.load_live_superblock(&superblock_key);
        let Some(b) = superblock.get(&block_pt) else {
            return GetPreference::Miss(None);
        };
        b.get_with_preference(self, query, &inner_pt)
    }

    pub fn prefetch<Tgt>(&self, query: &Spec<Tgt>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let bimap = self.spec_bimap();
        let (table_key, global_pt) = bimap.apply(query);
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
        let bimap = self.spec_bimap();
        let (table_key, global_pt_lhs) = bimap.apply(lhs);
        let (block_pt, _) = blockify_point(global_pt_lhs);
        PageId {
            db: &self,
            table_key,
            superblock_id: superblockify_pt(&block_pt),
        }
    }

    pub fn put<Tgt>(&self, spec: Spec<Tgt>, decisions: SmallVec<[(ActionIdx, Cost); 1]>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
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

        for joined_row in blocks_iter {
            let block_pt = joined_row
                .iter()
                .map(|(b, _)| *b)
                .collect::<SmallVec<[BimapInt; 10]>>();
            // TODO: Factor out this tuple construction
            let mut superblock_guard = self.load_live_superblock_mut(
                &self
                    .prehasher
                    .prehash((db_key.clone(), superblockify_pt(&block_pt))),
            );

            // If the superblock already contains the block, mutate in place and continue the loop.
            if let Some(live_block) = superblock_guard.get_mut(&block_pt) {
                let dim_ranges = joined_row
                    .iter()
                    .map(|(_, r)| r.clone())
                    .collect::<Vec<_>>();
                let DbBlock::Whole(e) = live_block;
                // Examine the table before updating.
                e.fill_region(self.k, &dim_ranges, &ActionCostVec(decisions.clone()));
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
            let new_block = DbBlock::Whole(Box::new(WholeBlock::partially_filled::<Tgt>(
                self.k,
                &block_shape_usize,
                &dim_ranges,
                &ActionCostVec(decisions.clone()),
            )));
            superblock_guard.insert(block_pt, new_block);
        }
    }

    pub fn flush(&self) {
        self.db.flush().unwrap();
    }

    pub fn max_k(&self) -> Option<usize> {
        Some(self.k.into())
    }

    pub fn stats_str(&self) -> String {
        // let start = Instant::now();
        // let mut runs_filled = 0;
        // let mut lens_filled = 0;
        // let mut runs_main_costs = 0;
        // let mut lens_main_costs = 0;
        // let mut runs_peaks = 0;
        // let mut lens_peaks = 0;
        // let mut runs_depths_actions = 0;
        // let mut lens_depths_actions = 0;
        // for block in &self.blocks {
        //     match block.value() {
        //         DbBlock::Whole(e) => {
        //             runs_filled += e.filled.runs_len();
        //             lens_filled += e.filled.len();
        //             runs_main_costs += e.main_costs.runs_len();
        //             lens_main_costs += e.main_costs.len();
        //             runs_peaks += e.peaks.runs_len();
        //             lens_peaks += e.peaks.len();
        //             runs_depths_actions += e.depths_actions.runs_len();
        //             lens_depths_actions += e.depths_actions.len();
        //         }
        //     }
        // }
        // let stat_duration = start.elapsed();
        // format!(
        //     "blocks={} \
        //     runs_filled={} runs_main_costs={} runs_peaks={} \
        //     runs_depthsactions={} cr_filled={:.5} \
        //     cr_main_costs={:.5} cr_peaks={:.5} cr_depthsactions={:.5} statms={}",
        //     self.blocks.len(),
        //     runs_filled,
        //     runs_main_costs,
        //     runs_peaks,
        //     runs_depths_actions,
        //     runs_filled as f32 / lens_filled as f32,
        //     runs_main_costs as f32 / lens_main_costs as f32,
        //     runs_peaks as f32 / lens_peaks as f32,
        //     runs_depths_actions as f32 / lens_depths_actions as f32,
        //     stat_duration.as_millis(),
        // )

        // TODO: Reimplement with RocksDB.
        "".to_owned()
    }

    pub fn spec_bimap<Tgt>(&self) -> impl BiMap<Domain = Spec<Tgt>, Codomain = DbKey>
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
}

impl Drop for RocksDatabase {
    fn drop(&mut self) {
        for shard in &mut self.shards.0 {
            let mut shard_guard = shard.lock();
            let drained = shard_guard.drain_cache().collect::<Vec<_>>();
            for (k, v) in drained {
                shard_guard.async_put(k, v);
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
    fn new(idx: usize, db: &Arc<rocksdb::DB>) -> Self {
        let db = Arc::clone(db);
        let (command_tx, command_rx) = std::sync::mpsc::sync_channel(CHANNEL_SIZE);
        let (response_tx, response_rx) = std::sync::mpsc::sync_channel(CHANNEL_SIZE);

        let thread = Some(
            std::thread::Builder::new()
                .name(format!("ShardThread-{idx}"))
                .spawn(move || loop {
                    match command_rx.recv() {
                        Ok(ShardThreadMsg::Get(key)) => {
                            let rocksdb_key = make_key(&key.0, &key.1); // TODO: Move inside.
                            let result = db
                                .get_pinned(rocksdb_key)
                                .unwrap()
                                .map(|pinnable_slice| {
                                    bincode::deserialize::<SuperBlock>(&pinnable_slice).unwrap()
                                })
                                .unwrap_or_default();
                            response_tx
                                .send(ShardThreadResponse::Loaded(key, result))
                                .unwrap();
                        }
                        Ok(ShardThreadMsg::Put(key, value)) => {
                            let mut put_options = rocksdb::WriteOptions::default();
                            put_options.disable_wal(true);

                            let rocksdb_key = make_key(&key.0, &key.1); // TODO: Move inside.
                            let rocksdb_value = bincode::serialize(&value).unwrap();
                            db.put_opt(rocksdb_key, &rocksdb_value, &put_options)
                                .unwrap();
                        }
                        Ok(ShardThreadMsg::Exit) => break,
                        Err(_) => unreachable!("expected Exit first"),
                    }
                })
                .unwrap(),
        );

        Self {
            cache: WTinyLfuCache::new(CACHE_PER_SHARD_SIZE, CACHE_PER_SHARD_SAMPLES),
            outstanding_gets: new_prehashed_set(),
            thread,
            thread_tx: command_tx,
            thread_rx: response_rx,
        }
    }

    fn process_bg_thread_msgs_until_close(&mut self) {
        while let Ok(msg) = self.thread_rx.recv() {
            self.process_bg_thread_msg_inner(msg);
        }
    }

    /// Process background thread messages until just *after* `cond` returns `false`.
    fn process_bg_thread_msgs_until<F>(&mut self, mut cond: F)
    where
        F: FnMut(&ShardThreadResponse) -> bool,
    {
        while let Ok(msg) = self.thread_rx.recv() {
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
            self.async_put(Prehashed::into_inner(evicted_key), evicted_value);
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

    fn async_put(&self, key: SuperBlockKey, value: SuperBlock) {
        self.thread_tx
            .send(ShardThreadMsg::Put(key, value))
            .unwrap();
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

impl DbBlock {
    pub fn get_with_preference<Tgt>(
        &self,
        containing_db: &RocksDatabase,
        query: &Spec<Tgt>,
        inner_pt: &[u8],
    ) -> GetPreference<ActionCostVec, SmallVec<[ActionIdx; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        match self {
            DbBlock::Whole(b) => {
                // TODO: Propogate an action index preference.
                let inner_pt_usize = inner_pt
                    .iter()
                    .map(|v| *v as usize)
                    .collect::<SmallVec<[_; 10]>>();
                match b.get(&inner_pt_usize) {
                    Some(r) => GetPreference::Hit(r),
                    None => GetPreference::Miss(None),
                }
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            DbBlock::Whole(e) => e.shape(),
        }
    }
}

impl WholeBlock {
    fn empty<Tgt: Target>(k: u8, shape: &[usize]) -> Self {
        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        WholeBlock {
            filled: NDArray::new_with_value(shape, 0),
            main_costs: NDArray::new(&shape_with_k),
            peaks: NDArray::new_with_value(&shape_with_k, MemVec::zero::<Tgt>()),
            depths_actions: NDArray::new(&shape_with_k),
        }
    }

    pub(crate) fn partially_filled<Tgt: Target>(
        k: u8,
        shape: &[usize],
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) -> Self {
        let mut e = Self::empty::<Tgt>(k, shape);
        e.fill_region_without_updating_match(k, dim_ranges, value);
        e
    }

    pub(crate) fn fill_region(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) {
        self.fill_region_without_updating_match(k, dim_ranges, value);
    }

    fn fill_region_without_updating_match(
        &mut self,
        k: u8,
        dim_ranges: &[Range<BimapInt>],
        value: &ActionCostVec,
    ) {
        let shape = self.filled.shape();
        debug_assert_eq!(dim_ranges.len(), shape.len());

        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        self.filled
            .fill_region(dim_ranges, &(u8::try_from(value.len()).unwrap() + 1));
        self.main_costs.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(_, c)| c.main),
            Some(&self.filled),
        );
        self.peaks.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(_, c)| c.peaks.clone()),
            Some(&self.filled),
        );
        self.depths_actions.fill_broadcast_1d(
            dim_ranges,
            value.0.iter().map(|(a, c)| (c.depth, *a)),
            Some(&self.filled),
        );
    }

    pub(crate) fn get(&self, pt: &[usize]) -> Option<ActionCostVec> {
        let f = self.filled[pt];
        if f == 0 {
            return None;
        }
        let local_k = f - 1;
        let mut pt_with_k = Vec::with_capacity(pt.len() + 1);
        pt_with_k.extend_from_slice(pt);
        pt_with_k.push(0);
        Some(ActionCostVec(
            (0..local_k)
                .map(move |i| {
                    *pt_with_k.last_mut().unwrap() = i.into();
                    let (depth, action_idx) = self.depths_actions[&pt_with_k];
                    (
                        action_idx,
                        Cost {
                            main: self.main_costs[&pt_with_k],
                            peaks: self.peaks[&pt_with_k].clone(),
                            depth,
                        },
                    )
                })
                .collect(),
        ))
    }

    pub(crate) fn compact(&mut self) {
        self.filled.shrink_to_fit();
        self.main_costs.shrink_to_fit();
        self.peaks.shrink_to_fit();
        self.depths_actions.shrink_to_fit();
    }

    pub fn shape(&self) -> &[usize] {
        self.filled.shape()
    }
}

impl Deref for ActionCostVec {
    type Target = SmallVec<[(ActionIdx, Cost); 1]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<SmallVec<[(ActionIdx, Cost); 1]>> for ActionCostVec {
    fn as_ref(&self) -> &SmallVec<[(ActionIdx, Cost); 1]> {
        self.deref()
    }
}

fn superblockify_pt(block_pt: &[BimapInt]) -> SmallVec<[BimapInt; 10]> {
    block_pt.iter().map(|&i| i / SUPERBLOCK_FACTOR).collect()
}

fn construct_impl<Tgt>(db: &RocksDatabase, imp: &DbImpl<Tgt>) -> DbImpl<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
{
    match imp {
        ImplNode::SpecApp(p) => db
            .get_impl(&p.0)
            .expect("Database should have the sub-Spec entry")
            .first()
            .expect("Database sub-Spec should be satisfiable")
            .clone(),
        _ => imp.replace_children(imp.children().iter().map(|c| construct_impl(db, c))),
    }
}

fn block_size_dim(dim: usize, dim_count: usize) -> u32 {
    if dim == 0 || dim == 1 || dim == dim_count - 5 {
        // The last case here is the serial_only dimension. Setting this to 1 will avoid empty
        // rows when computing serial_only, which is a common setting.
        // The first is just a shape dimension.
        1
    } else if dim == dim_count - 1 {
        31
    } else if dim < dim_count - 5 {
        3
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
) -> impl Iterator<Item = BimapInt> + ExactSizeIterator + 'a
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

fn db_shape<Tgt: Target>(rank: usize) -> SmallVec<[Option<NonZeroUsize>; 10]> {
    let mut shape = smallvec::smallvec![None; rank];
    let MemoryLimits::Standard(m) = Tgt::max_mem();
    for (level_idx, dest_idx) in ((rank - m.len())..rank).enumerate() {
        shape[dest_idx] =
            Some(NonZeroUsize::new((m.get_binary_scaled(level_idx) + 1).into()).unwrap());
    }
    shape
}

/// Converts a given global coordinate into block and within-block coordinates.
fn blockify_point(
    mut pt: SmallVec<[BimapInt; 10]>,
) -> (SmallVec<[BimapInt; 10]>, SmallVec<[u8; 10]>) {
    let rank = pt.len();
    let mut inner_pt = SmallVec::with_capacity(rank);
    for (i, d) in pt.iter_mut().enumerate() {
        let (outer, inner) = db_key_scale(i, *d, rank);
        *d = outer;
        inner_pt.push(inner);
    }
    (pt, inner_pt)
}

pub fn deblockify_points(a: &[BimapInt], b: &[u8]) -> SmallVec<[BimapInt; 10]> {
    debug_assert_eq!(a.len(), b.len());

    let rank = a.len();
    let mut result = SmallVec::with_capacity(rank);
    for i in 0..rank {
        let s = block_size_dim(i, rank);
        result.push(s * a[i] + BimapInt::from(b[i]));
    }
    result
}

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt, B>(
    bimap: &B,
    spec: &Spec<Tgt>,
    impls: &SmallVec<[(ActionIdx, Cost); 1]>,
) -> (
    TableKey,
    (SmallVec<[BimapInt; 10]>, SmallVec<[BimapInt; 10]>),
)
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

/// Compute the cost of an incomplete Impl.
fn compute_cost<Tgt, Aux, F>(imp: &ImplNode<Tgt, Aux>, lookup: &F) -> Cost
where
    Tgt: Target,
    Aux: Clone,
    F: Fn(&SpecApp<Tgt, Spec<Tgt>, Aux>) -> Cost,
{
    match imp {
        ImplNode::SpecApp(s) => lookup(s),
        _ => {
            let children = imp.children();
            let mut child_costs: SmallVec<[_; 3]> = SmallVec::with_capacity(children.len());
            for c in children {
                child_costs.push(compute_cost(c, lookup));
            }
            Cost::from_child_costs(imp, &child_costs)
        }
    }
}

fn make_key(table_key: &TableKey, block_pt: &[BimapInt]) -> String {
    // TODO: Use a faster (non-String?) and more stable encoding.
    format!("{table_key:?}/{block_pt:?}")
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
    use proptest::prelude::*;

    // TODO: What about leaves!? This shouldn't be called `Decision`.
    #[derive(Debug, Clone)]
    struct Decision<Tgt: Target> {
        spec: Spec<Tgt>,
        actions_costs: Vec<(ActionIdx, Cost)>,
        children: Vec<Decision<Tgt>>,
    }

    impl<Tgt: Target> Decision<Tgt> {
        /// Return an [Iterator] which visits all nested Decisions bottom up.
        fn visit_decisions(&self) -> Box<dyn Iterator<Item = &Decision<Tgt>> + '_> {
            Box::new(
                self.children
                    .iter()
                    .flat_map(|c| c.visit_decisions())
                    .chain(std::iter::once(self)),
            )
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
            let mut block_idxs = iter_blocks_in_single_dim_range(start, end, block_dim_size).map(|(block_idx, _)| block_idx);
            if let Some(mut last_block_idx) = block_idxs.next() {
                for block_idx in block_idxs {
                    assert!(block_idx == last_block_idx + 1);
                    last_block_idx = block_idx;
                }
            }
        }

        // TODO: Add tests for top-2, etc. Impls
        #[test]
        fn test_put_then_get_fills_across_memory_limits(decision in arb_spec_and_decision::<X86Target>()) {
            let MemoryLimits::Standard(spec_limits) = decision.spec.1.clone();
            let db = RocksDatabase::try_new(None, false, 1).unwrap();

            // Put all decisions into database.
            for d in decision.visit_decisions() {
                db.put(d.spec.clone(), d.actions_costs.clone().into());
            }

            let peaks = if let Some((_, c)) = decision.actions_costs.first() {
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
            let expected = ActionCostVec(decision.actions_costs.into());
            for limit_to_check_bits in filled_limits_iter {
                let limit_to_check_vec = limit_to_check_bits.iter().copied().map(bit_length_inverse).collect::<Vec<_>>();
                let limit_to_check = MemoryLimits::Standard(MemVec::new(limit_to_check_vec.try_into().unwrap()));
                let spec_to_check = Spec(decision.spec.0.clone(), limit_to_check);
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

    fn arb_spec_and_decision<Tgt: Target>() -> impl Strategy<Value = Decision<Tgt>> {
        arb_canonical_spec::<Tgt>(None, None)
            .prop_flat_map(|spec| {
                let valid_actions = spec
                    .0
                    .actions()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, a)| match a.apply(&spec) {
                        Ok(applied) => Some((ActionIdx::from(u16::try_from(i).unwrap()), applied)),
                        Err(ApplyError::ActionNotApplicable) => None,
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
    fn arb_spec_and_decision_pair<Tgt: Target>(
    ) -> impl Strategy<Value = (Decision<Tgt>, Decision<Tgt>)> {
        arb_spec_and_decision().prop_flat_map(|first| {
            let second_term = prop_oneof![
                2 => Just(first.clone()),
                2 => {
                    use crate::memorylimits::arb_memorylimits;
                    let MemoryLimits::Standard(max_memory) = Tgt::max_mem();
                    let first_logical = first.spec.0.clone();
                    arb_memorylimits::<Tgt>(&max_memory).prop_map(move |spec_limits| {
                        recursively_decide_actions(&Spec(first_logical.clone(), spec_limits))
                    })
                },
                1 => arb_spec_and_decision(),
            ];
            (Just(first), second_term).boxed()
        })
    }

    /// Return some complete Impl for the given Spec.
    fn complete_impl<Tgt: Target>(
        partial_impl: &ImplNode<Tgt, ()>,
    ) -> Option<(ImplNode<Tgt, ()>, Cost)> {
        match partial_impl {
            ImplNode::SpecApp(spec_app) => {
                for action in spec_app.0 .0.actions() {
                    match action.apply(&spec_app.0) {
                        Ok(applied) => {
                            if let Some(completed) = complete_impl(&applied) {
                                return Some(completed);
                            }
                        }
                        Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => {}
                        Err(ApplyError::SpecNotCanonical) => {
                            panic!("Spec-to-complete must be canon")
                        }
                    }
                }
                None
            }
            _ => {
                let old_children = partial_impl.children();
                let mut new_children = SmallVec::<[_; 3]>::with_capacity(old_children.len());
                let mut new_children_costs = SmallVec::<[_; 3]>::with_capacity(old_children.len());
                for child in old_children {
                    let (c1, c2) = complete_impl(child)?;
                    new_children.push(c1);
                    new_children_costs.push(c2);
                }
                Some((
                    partial_impl.replace_children(new_children.into_iter()),
                    Cost::from_child_costs(partial_impl, &new_children_costs),
                ))
            }
        }
    }

    /// Will return a [Decision] by choosing the first action for the Spec (if any) and recursively
    /// choosing the first action for all child Specs in the resulting partial Impl.
    fn recursively_decide_actions<Tgt: Target>(spec: &Spec<Tgt>) -> Decision<Tgt> {
        if let Some((action_idx, partial_impl)) = spec
            .0
            .actions()
            .into_iter()
            .enumerate()
            .filter_map(|(i, a)| match a.apply(spec) {
                Ok(imp) => Some((i, imp)),
                Err(ApplyError::ActionNotApplicable | ApplyError::OutOfMemory) => None,
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
        partial_impl: &ImplNode<Tgt, ()>,
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
