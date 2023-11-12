use crate::common::DimSize;
use crate::cost::{Cost, MainCost};
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::{AsBimap, BiMap};
use crate::grid::linear::BimapInt;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::pprint::PrintableAux;
use crate::spec::{LogicalSpecSurMap, PrimitiveBasicsBimap, Spec, SpecSurMap};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use anyhow::anyhow;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use divrem::DivRem;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, Range};
use std::time::Instant;
use std::{iter, path};

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

type DbKey = (
    (SpecKey, SmallVec<[(Layout, u8, Option<NonZeroU32>); 3]>),
    SmallVec<[BimapInt; 10]>,
);
pub type ActionIdx = u16;

pub trait Database<'a> {
    fn get<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<ActionCostVec>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;

    // TODO: Document interior mutability of put.
    fn put<Tgt>(&'a self, problem: Spec<Tgt>, impls: SmallVec<[(ActionIdx, Cost); 1]>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;

    fn flush(&'a self);

    fn save(&'a self) -> anyhow::Result<()>;

    fn compact(&'a self);

    /// Returns the maximum number of Impls this [Database] as store per Spec.
    ///
    /// If unlimited, returns `None`.
    fn max_k(&'a self) -> Option<usize>;

    fn stats_str(&self) -> String {
        "".to_string()
    }
}

pub trait DatabaseExt<'a>: Database<'a> {
    fn get_impl<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>;
}

#[derive(Clone, Debug)]
pub struct DbImplAux<Tgt: Target>(Option<(Spec<Tgt>, Cost)>);

pub struct DashmapDiskDatabase {
    file_path: Option<path::PathBuf>,
    pub blocks: DashMap<DbKey, DbBlock>,
    binary_scale_shapes: bool,
    k: u8,
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
    Rle(Box<RleBlock>),
}

// TODO: Flatten something to avoid two indirections from DbBlock to NDArray contents.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RleBlock {
    pub filled: crate::ndarray::NDArray<u8>, // 0 is empty; otherwise n - 1 = # of actions.
    pub main_costs: crate::ndarray::NDArray<MainCost>,
    pub peaks: crate::ndarray::NDArray<MemVec>,
    pub depths_actions: crate::ndarray::NDArray<(u8, ActionIdx)>,
    shape: SmallVec<[usize; 10]>,
    volume: NonZeroU32,
}

// TODO: Storing Spec and usize is too expensive.
pub struct DashmapDbRef<'a, Tgt: Target, S = RandomState>(
    dashmap::mapref::one::Ref<'a, DbKey, HashMap<Spec<Tgt>, ActionCostVec>, S>,
    Option<&'a ActionCostVec>,
);

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ActionCostVec(pub SmallVec<[(ActionIdx, Cost); 1]>);

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

impl DashmapDiskDatabase {
    pub fn new(file_path: Option<&path::Path>, binary_scale_shapes: bool, k: u8) -> Self {
        Self::new_with_dashmap_constructor(file_path, binary_scale_shapes, k, &DashMap::new)
    }

    pub fn with_capacity(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
        k: u8,
        capacity: usize,
    ) -> Self {
        Self::new_with_dashmap_constructor(file_path, binary_scale_shapes, k, &|| {
            DashMap::with_capacity(capacity)
        })
    }

    // TODO: This does I/O; it (and the pub constructors) should return errors, not panic.
    fn new_with_dashmap_constructor(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
        k: u8,
        dashmap_constructor: &dyn Fn() -> DashMap<DbKey, DbBlock>,
    ) -> Self {
        let grouped_entries = match file_path {
            Some(path) => match std::fs::File::open(path) {
                Ok(f) => {
                    let start = Instant::now();
                    let decoder = snap::read::FrameDecoder::new(f);
                    let result = bincode::deserialize_from(decoder).unwrap();
                    log::debug!("Loading database took {:?}", start.elapsed());
                    result
                }
                Err(err) => match err.kind() {
                    std::io::ErrorKind::NotFound => dashmap_constructor(),
                    _ => todo!("Handle other file errors"),
                },
            },
            None => Default::default(),
        };
        Self {
            file_path: file_path.map(|p| p.to_owned()),
            blocks: grouped_entries,
            binary_scale_shapes,
            k,
        }
    }

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
}

impl<'a> Database<'a> for DashmapDiskDatabase
where
    Self: 'a,
{
    fn get<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<ActionCostVec>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        let bimap = self.spec_bimap();
        let (table_key, global_pt) = bimap.apply(query);
        let (block_pt, inner_pt) = blockify_point(global_pt);
        let Some(group) = self.blocks.get(&(table_key, block_pt)) else {
            return None;
        };
        group.get(&inner_pt)
    }

    fn put<Tgt>(&'a self, spec: Spec<Tgt>, decisions: SmallVec<[(ActionIdx, Cost); 1]>)
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    {
        #[cfg(debug_assertions)]
        let original_spec = spec.clone();

        let bimap = self.spec_bimap();
        let (db_key, (bottom, top)) = put_range_to_fill(&bimap, &spec, &decisions);
        let spec_b = spec.clone();

        // Construct an iterator over all blocks to fill.
        let rank = bottom.len();
        let blocks_iter = bottom
            .into_iter()
            .zip(&top)
            .enumerate()
            .map(|(dim, (b, t))| iter_blocks_in_single_dim_range(b, *t, block_size_dim(dim, rank)))
            .multi_cartesian_product();

        for joined_row in blocks_iter {
            let block_pt = joined_row.iter().map(|(b, _)| *b).collect::<SmallVec<_>>();
            let block_pt_b = block_pt.clone();

            let block_entry = self.blocks.entry((db_key.clone(), block_pt.clone()));
            match block_entry {
                Entry::Occupied(mut existing_block) => {
                    let r = existing_block.get_mut();
                    match r {
                        DbBlock::Rle(e) => {
                            // Examine the table before updating.
                            e.fill_region(
                                self.k,
                                &joined_row
                                    .iter()
                                    .map(|(_, r)| r.clone())
                                    .collect::<Vec<_>>(),
                                &ActionCostVec(decisions.clone()),
                            );
                        }
                    }
                }
                Entry::Vacant(entry) => {
                    let db_shape = db_shape::<Tgt>(rank);
                    let bs = block_shape(&block_pt, &db_shape, block_size_dim);
                    let block_shape_usize = bs.map(|v| v.try_into().unwrap()).collect::<Vec<_>>();
                    entry.insert(DbBlock::Rle(Box::new(RleBlock::partially_filled::<Tgt>(
                        self.k,
                        &block_shape_usize,
                        &joined_row
                            .iter()
                            .map(|(_, r)| r.clone())
                            .collect::<Vec<_>>(),
                        &ActionCostVec(decisions.clone()),
                    ))));
                }
            }
        }
    }

    fn flush(&'a self) {}

    fn save(&self) -> anyhow::Result<()> {
        if let Some(path) = &self.file_path {
            let start = Instant::now();
            let dir = path
                .parent()
                .ok_or_else(|| anyhow!("path must have parent, but is: {:?}", path))?;
            let temp_file = tempfile::NamedTempFile::new_in(dir)?;
            let encoder = snap::write::FrameEncoder::new(&temp_file);
            bincode::serialize_into(encoder, &self.blocks)?;
            let temp_file_path = temp_file.keep()?.1;
            std::fs::rename(temp_file_path, path)?;
            log::debug!("Saving database took {:?}", start.elapsed());
        }
        Ok(())
    }

    fn compact(&'a self) {
        self.blocks.par_iter_mut().for_each(|mut block| {
            block.compact();
        });
    }

    fn max_k(&'a self) -> Option<usize> {
        Some(self.k.into())
    }

    fn stats_str(&self) -> String {
        let start = Instant::now();
        let mut runs_filled = 0;
        let mut lens_filled = 0;
        let mut runs_main_costs = 0;
        let mut lens_main_costs = 0;
        let mut runs_peaks = 0;
        let mut lens_peaks = 0;
        let mut runs_depths_actions = 0;
        let mut lens_depths_actions = 0;
        for block in &self.blocks {
            match block.value() {
                DbBlock::Rle(e) => {
                    runs_filled += e.filled.runs_len();
                    lens_filled += e.filled.len();
                    runs_main_costs += e.main_costs.runs_len();
                    lens_main_costs += e.main_costs.len();
                    runs_peaks += e.peaks.runs_len();
                    lens_peaks += e.peaks.len();
                    runs_depths_actions += e.depths_actions.runs_len();
                    lens_depths_actions += e.depths_actions.len();
                }
            }
        }
        let stat_duration = start.elapsed();
        format!(
            "blocks={} runs_filled={} runs_main_costs={} runs_peaks={} runs_depthsactions={} \
            cr_filled={:.5} cr_main_costs={:.5} cr_peaks={:.5} cr_depthsactions={:.5} statms={}",
            self.blocks.len(),
            runs_filled,
            runs_main_costs,
            runs_peaks,
            runs_depths_actions,
            runs_filled as f32 / lens_filled as f32,
            runs_main_costs as f32 / lens_main_costs as f32,
            runs_peaks as f32 / lens_peaks as f32,
            runs_depths_actions as f32 / lens_depths_actions as f32,
            stat_duration.as_millis(),
        )
    }
}

impl Drop for DashmapDiskDatabase {
    fn drop(&mut self) {
        self.save().unwrap();
    }
}

impl<'a, T: Database<'a>> DatabaseExt<'a> for T {
    fn get_impl<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>
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
}

impl DbBlock {
    fn get(&self, inner_pt: &[u8]) -> Option<ActionCostVec> {
        match self {
            DbBlock::Rle(e) => {
                let inner_pt_usize = inner_pt
                    .iter()
                    .map(|v| *v as usize)
                    .collect::<SmallVec<[_; 10]>>();
                e.get(&inner_pt_usize)
            }
        }
    }

    /// Returns the number of entries for which storage is allocated.
    ///
    /// This is a rough way to estimate memory consumption of the [DbBlock].
    pub fn storage_size(&self) -> u32 {
        match self {
            DbBlock::Rle(e) => e.volume.get(),
        }
    }

    pub fn compact(&mut self) {
        match self {
            DbBlock::Rle(e) => {
                e.compact();
            }
        }
    }
}

impl RleBlock {
    fn empty<Tgt: Target>(k: u8, shape: &[usize]) -> Self {
        let mut shape_with_k = Vec::with_capacity(shape.len() + 1);
        shape_with_k.extend_from_slice(shape);
        shape_with_k.push(k.into());

        RleBlock {
            filled: crate::ndarray::NDArray::new_with_value(shape, 0),
            main_costs: crate::ndarray::NDArray::new(&shape_with_k),
            peaks: crate::ndarray::NDArray::new_with_value(&shape_with_k, MemVec::zero::<Tgt>()),
            depths_actions: crate::ndarray::NDArray::new(&shape_with_k),
            shape: shape.into(),
            volume: shape
                .iter()
                .map(|&s| u32::try_from(s).unwrap())
                .product::<u32>()
                .try_into()
                .unwrap(),
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
}

impl<'a, Tgt: Target> Deref for DashmapDbRef<'a, Tgt> {
    type Target = ActionCostVec;

    fn deref(&self) -> &Self::Target {
        self.1.unwrap()
    }
}

impl<'a, Tgt: Target> AsRef<ActionCostVec> for DashmapDbRef<'a, Tgt> {
    fn as_ref(&self) -> &ActionCostVec {
        self.deref()
    }
}

impl<'a, Tgt: Target> AsRef<SmallVec<[(ActionIdx, Cost); 1]>> for DashmapDbRef<'a, Tgt> {
    fn as_ref(&self) -> &SmallVec<[(ActionIdx, Cost); 1]> {
        self.deref().as_ref()
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

fn construct_impl<'a, Tgt, D>(db: &'a D, imp: &DbImpl<Tgt>) -> DbImpl<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = u8>,
    D: Database<'a>,
{
    match imp {
        ImplNode::SpecApp(p) => db
            .get_impl(&p.0)
            .expect("Database should have the sub-Spec entry")
            .get(0)
            .expect("Database sub-Spec should be satisfiable")
            .clone(),
        _ => imp.replace_children(imp.children().iter().map(|c| construct_impl(db, c))),
    }
}

fn block_size_dim(dim: usize, dim_count: usize) -> u32 {
    if dim >= dim_count - 4 {
        4
    } else {
        3
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

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt, B>(
    bimap: &B,
    spec: &Spec<Tgt>,
    impls: &SmallVec<[(ActionIdx, Cost); 1]>,
) -> (
    (SpecKey, SmallVec<[(Layout, u8, Option<NonZeroU32>); 3]>),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cost::MainCost,
        memorylimits::{MemVec, MemoryLimits},
        target::X86Target,
        utils::{bit_length, bit_length_inverse},
    };
    use itertools::Itertools;
    use proptest::prelude::*;
    use smallvec::smallvec;

    #[test]
    fn test_block_shape() {
        let db_shape = [4, 7]
            .into_iter()
            .map(|v| Some(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            block_shape(&[0, 0], &db_shape, |i, _| 2)
                .collect_vec()
                .as_slice(),
            &[2, 2]
        );
        assert_eq!(
            block_shape(&[1, 1], &db_shape, |i, _| 2)
                .collect_vec()
                .as_slice(),
            &[2, 2]
        );
        assert_eq!(
            block_shape(&[1, 3], &db_shape, |i, _| 2)
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
        fn test_put_then_get_fills_across_memory_limits(entry in arb_spec_and_decision::<X86Target>()) {
            let (spec, decisions) = entry.clone();
            let (spec_b, decisions_b) = entry;
            let MemoryLimits::Standard(spec_limits) = spec.1.clone();
            let db = DashmapDiskDatabase::new(None, false, 1);
            db.put(spec, decisions.into());
            let peaks = if let Some((_, c)) = decisions_b.first() {
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
            let expected = ActionCostVec(decisions_b.into());
            for limit_to_check_bits in filled_limits_iter {
                let limit_to_check = limit_to_check_bits.iter().copied().map(bit_length_inverse).collect::<Vec<_>>();
                let spec_to_check = Spec(spec_b.0.clone(), MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())));
                let get_result = db.get(&spec_to_check).expect("Spec should be in database");
                assert_eq!(get_result, expected);
            }
        }

        #[test]
        fn test_database_empty_outside_range_after_one_put(
            entry in arb_spec_and_decision::<X86Target>()
        ) {
            test_db_put(entry, false);
        }

        #[test]
        fn test_database_empty_outside_range_after_one_put_with_compact(
            entry in arb_spec_and_decision::<X86Target>()
        ) {
            test_db_put(entry, true);
        }

        #[test]
        fn test_two_puts_return_correct_gets_for_second_put(
            entry_pair in arb_spec_and_action_pair::<X86Target>()
        ) {
            let db = DashmapDiskDatabase::new(None, false, 1);
            let ((spec_a, action_a, cost_a), (spec_b, action_b, cost_b)) = entry_pair;
            let logical_specs_match = spec_a.0 == spec_b.0;

            let spec_a_cloned = spec_a.clone();
            let spec_b_cloned = spec_b.clone();
            let MemoryLimits::Standard(spec_limits_a) = &spec_a_cloned.1;
            let MemoryLimits::Standard(spec_limits_b) = &spec_b_cloned.1;

            let cost_a_clone = cost_a.clone();
            let cost_b_clone = cost_b.clone();

            db.put(spec_a, smallvec![(action_a, cost_a)]);
            db.put(spec_b, smallvec![(action_b, cost_b)]);

            let vr_a_peaks = &cost_a_clone.peaks;
            let vr_b_peaks = &cost_b_clone.peaks;

            // TODO: Use the binary-scaled values directly rather than converting back and forth.
            let relevant_limits_iter = spec_limits_a.iter().zip(spec_limits_b.iter())
                .zip(vr_a_peaks.iter().zip(vr_b_peaks.iter()))
                .map(|((l_a, l_b), (p_a, p_b))| {
                    (bit_length(p_a.min(p_b))..=bit_length(l_a.max(l_b))).map(bit_length_inverse)
                })
                .multi_cartesian_product();

            for limit_to_check in relevant_limits_iter {
                // b was put second, so if we're in its range, that should be the result. Check a
                // second.
                let limit_in_a = vr_a_peaks.iter().zip(spec_limits_a.iter()).zip(limit_to_check.iter()).all(|((bot, top), p)| {
                    assert!(bot <= top);
                    bot <= *p && *p <= top
                });
                let limit_in_b = vr_b_peaks.iter().zip(spec_limits_b.iter()).zip(limit_to_check.iter()).all(|((bot, top), p)| {
                    assert!(bot <= top);
                    bot <= *p && *p <= top
                });

                let expected_value = if limit_in_b {
                    Some(ActionCostVec(smallvec![(action_b, cost_b_clone.clone())]))
                } else if limit_in_a && logical_specs_match {
                    Some(ActionCostVec(smallvec![(action_a, cost_a_clone.clone())]))
                } else {
                    None
                };

                let spec_to_check = Spec(
                    spec_b_cloned.0.clone(),
                    MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())),
                );
                let get_result = db.get(&spec_to_check);
                match (get_result.as_ref(), expected_value.as_ref()) {
                    (Some(g), Some(e)) if g == e => {}
                    (None, None) => {}
                    _ => {
                        eprintln!("First-inserted Spec: {}", spec_a_cloned);
                        eprintln!("Last-inserted Spec: {}", spec_b_cloned);
                        let expected_description = if limit_in_b {
                            "second value".to_string()
                        } else if limit_in_a {
                            "first value".to_string()
                        } else {
                            format!("{:?}", expected_value)
                        };
                        let result_description = match get_result {
                            Some(g) => format!("Some({g:?})"),
                            None => "None".to_string(),
                        };
                        panic!(
                            "Incorrect get result at {}. Expected {} but got {}.",
                            spec_to_check, expected_description, result_description
                        )
                    }
                }
            }
        }
    }

    fn test_db_put(entry: (Spec<X86Target>, Vec<(ActionIdx, Cost)>), compact_after_put: bool) {
        let spec_b = entry.0.clone();
        let (spec, decisions) = entry;
        let peaks = if let Some((_, c)) = decisions.first() {
            c.peaks.clone()
        } else {
            MemVec::zero::<X86Target>()
        };

        let db = DashmapDiskDatabase::new(None, false, 1);
        db.put(spec, decisions.into());

        if compact_after_put {
            db.compact();
        }

        let MemoryLimits::Standard(max_memory) = X86Target::max_mem();
        let MemoryLimits::Standard(spec_limits) = &spec_b.1;
        let filled_limits_iter = max_memory
            .iter_binary_scaled()
            .map(|l| (0..=u32::from(l)).map(bit_length_inverse))
            .multi_cartesian_product();
        let mut spec_to_check = spec_b.clone();
        for limit_to_check in filled_limits_iter {
            // Skip limits inside the put range
            if limit_to_check
                .iter()
                .zip(peaks.iter().zip(spec_limits.iter()))
                .any(|(c, (p, l))| c < &p || c > &l)
            {
                spec_to_check.1 =
                    MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap()));
                assert!(db.get(&spec_to_check).is_none());
            }
        }
    }

    fn arb_spec_and_decision<Tgt: Target>(
    ) -> impl Strategy<Value = (Spec<Tgt>, Vec<(ActionIdx, Cost)>)> {
        prop_oneof![
            arb_spec_and_action().prop_map(|(sp, a, c)| (sp, vec![(a, c)])),
            any::<Spec<Tgt>>().prop_map(|sp| (sp, vec![])),
        ]
    }

    fn arb_spec_and_action<Tgt: Target>() -> impl Strategy<Value = (Spec<Tgt>, ActionIdx, Cost)> {
        (any::<Spec<Tgt>>())
            .prop_flat_map(|spec| {
                let MemoryLimits::Standard(spec_limits) = &spec.1;
                let limits_strategy = spec_limits
                    .iter_binary_scaled()
                    .map(|l| (0..=l).prop_map(|bits| if bits == 0 { 0 } else { 1 << (bits - 1) }))
                    .collect::<Vec<_>>();
                (
                    Just(spec),
                    limits_strategy,
                    any::<ActionIdx>(),
                    any::<MainCost>(),
                    any::<u8>(),
                )
            })
            .prop_map(|(spec, peaks, action_idx, main_cost, depth)| {
                let cost = Cost {
                    main: main_cost,
                    peaks: MemVec::new(peaks.try_into().unwrap()),
                    depth,
                };
                (spec, action_idx, cost)
            })
    }

    // TODO: Implement arb_spec_and_decision_pair and use that everywhere instead.
    fn arb_spec_and_action_pair<Tgt: Target>(
    ) -> impl Strategy<Value = ((Spec<Tgt>, ActionIdx, Cost), (Spec<Tgt>, ActionIdx, Cost))> {
        arb_spec_and_action().prop_flat_map(|first| {
            let second_term = prop_oneof![
                2 => Just(first.clone()),
                2 => {
                    use crate::memorylimits::arb_memorylimits;
                    let MemoryLimits::Standard(max_memory) = Tgt::max_mem();
                    let first = first.clone();
                    (arb_memorylimits::<Tgt>(&max_memory), arb_memorylimits::<Tgt>(&max_memory)).prop_map(move |(limits_a, limits_b)| {
                        let MemoryLimits::Standard(limits_a) = limits_a;
                        let MemoryLimits::Standard(limits_b) = limits_b;

                        let spec_limits = limits_a.iter().zip(limits_b.iter()).map(|(a, b)| a.max(b)).collect::<Vec<_>>();
                        let peaks = limits_a.iter().zip(limits_b.iter()).map(|(a, b)| a.min(b)).collect::<Vec<_>>();

                        let new_spec = Spec(
                            first.0.0.clone(),
                            MemoryLimits::Standard(MemVec::new(spec_limits.try_into().unwrap()))
                        );
                        let mut new_cost = first.2.clone();
                        new_cost.peaks = MemVec::new(peaks.try_into().unwrap());
                        (new_spec, first.1, new_cost)
                    })
                },
                1 => arb_spec_and_action(),
            ];
            (Just(first), second_term)
        })
    }
}
