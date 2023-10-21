use crate::common::DimSize;
use crate::cost::Cost;
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::BiMap;
use crate::grid::linear::BimapInt;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits, MemoryLimitsBimap};
use crate::pprint::PrintableAux;
use crate::spec::{LogicalSpecBimap, PrimitiveBasicsBimap, Spec, SpecBimap};
use crate::target::{Target, LEVEL_COUNT};
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use anyhow::anyhow;
use dashmap::mapref::entry::Entry;
use dashmap::mapref::one::MappedRef;
use dashmap::DashMap;
use divrem::DivRem;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::ops::{Deref, Range};
use std::time::Instant;
use std::{iter, path};

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

type DbKey = ((SpecKey, SmallVec<[Layout; 3]>), SmallVec<[BimapInt; 10]>);
pub type ActionIdx = u16;

const INITIAL_HASHMAP_CAPACITY: usize = 100_000_000;

pub trait Database<'a> {
    type ValueRef: AsRef<SmallVec<[(ActionIdx, Cost); 1]>> + 'a;

    fn get<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<Self::ValueRef>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>;

    // TODO: Document interior mutability of put.
    fn put<Tgt>(
        &'a self,
        problem: Spec<Tgt>,
        impls: SmallVec<[(ActionIdx, Cost); 1]>,
    ) -> Self::ValueRef
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>;

    fn flush(&'a self);

    fn save(&'a self) -> anyhow::Result<()>;
}

pub trait DatabaseExt<'a>: Database<'a> {
    fn get_impl<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>;
}

#[derive(Clone, Debug)]
pub struct DbImplAux<Tgt: Target>(Option<(Spec<Tgt>, Cost)>);

pub struct DashmapDiskDatabase {
    file_path: Option<path::PathBuf>,
    pub blocks: DashMap<DbKey, DbBlock>,
    binary_scale_shapes: bool,
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
    Single(ActionCostVec),
    Expanded {
        // TODO: Optimize memory with Option + NonZeroU32
        actions: crate::ndarray::NDArray<Option<ActionCostVec>>,
        matches: Option<NonZeroU32>,
    },
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
    pub fn new(file_path: Option<&path::Path>, binary_scale_shapes: bool) -> Self {
        Self::new_with_dashmap_constructor(file_path, binary_scale_shapes, &DashMap::new)
    }

    pub fn new_with_shard_count(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
        shard_count: usize,
    ) -> Self {
        Self::new_with_dashmap_constructor(file_path, binary_scale_shapes, &|| {
            DashMap::with_capacity_and_hasher_and_shard_amount(
                INITIAL_HASHMAP_CAPACITY,
                RandomState::default(),
                shard_count,
            )
        })
    }

    // TODO: This does I/O; it (and the pub constructors) should return errors, not panic.
    fn new_with_dashmap_constructor(
        file_path: Option<&path::Path>,
        binary_scale_shapes: bool,
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
        }
    }
}

impl<'a> Database<'a> for DashmapDiskDatabase
where
    Self: 'a,
{
    type ValueRef = MappedRef<'a, DbKey, DbBlock, ActionCostVec>;

    fn get<Tgt>(&'a self, query: &Spec<Tgt>) -> Option<Self::ValueRef>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
    {
        let (table_key, global_pt) = compute_db_key(query, self.binary_scale_shapes);
        let (block_pt, inner_pt) = blockify_point(global_pt);
        let inner_pt_usize = inner_pt.iter().map(|v| *v as usize).collect::<Vec<_>>();
        let Some(group) = self.blocks.get(&(table_key, block_pt)) else {
            return None;
        };
        match group.deref() {
            DbBlock::Single(_) => {}
            DbBlock::Expanded { actions, .. } => {
                actions[&inner_pt_usize].as_ref()?;
            }
        }
        // TODO: Obviate need to hash and look up on the inner HashMap for each deref of the below.
        Some(group.map(|g| g.get(query, self.binary_scale_shapes).unwrap()))
    }

    fn put<Tgt>(
        &'a self,
        spec: Spec<Tgt>,
        decisions: SmallVec<[(ActionIdx, Cost); 1]>,
    ) -> Self::ValueRef
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
    {
        #[cfg(debug_assertions)]
        let original_spec = spec.clone();

        let (db_key, (bottom, top)) =
            put_range_to_fill(&spec, &decisions, self.binary_scale_shapes);
        let spec_b = spec.clone();

        // Compute the block shape for a table of this rank. Blocks are larger than necessary at the
        // boundaries when true dimensions aren't multiples of the block shape.
        let block_shape = block_shape(bottom.len());

        // Construct an iterator over all blocks to fill.
        let blocks_iter = bottom
            .into_iter()
            .zip(&top)
            .zip(&block_shape)
            .map(|((b, t), shp)| iter_blocks_in_single_dim_range(b, *t, (*shp).try_into().unwrap()))
            .multi_cartesian_product();

        let mut any_entry_ref = None;
        for joined_row in blocks_iter {
            // Drop any_entry_ref so the upcoming `.entry` call doesn't deadlock.
            drop(any_entry_ref);

            let block_pt = joined_row.iter().map(|(b, _)| *b).collect::<SmallVec<_>>();
            let fill_whole_block = joined_row
                .iter()
                .zip(&block_shape)
                .all(|((_, r), shp)| r.len() == usize::try_from(*shp).unwrap());

            let block_entry = self.blocks.entry((db_key.clone(), block_pt));
            let block_entry_ref = if fill_whole_block {
                block_entry.insert(DbBlock::Single(ActionCostVec(decisions.clone())))
            } else {
                match block_entry {
                    Entry::Occupied(mut existing_block) => {
                        let r = existing_block.get_mut();
                        match r {
                            DbBlock::Single(v) if v.0 != decisions => {
                                // Format `v.0` first we don't keep the mutable borrow of `v`.
                                let value_str = format!("{:?}", v.0);
                                let block_pt = &existing_block.key().1;
                                unimplemented!(
                                    "Replacement not supported. Tried to insert {:?} into \
                                    compressed block of {}. Block was ({:?}, {:?}). Spec was {}.",
                                    decisions,
                                    value_str,
                                    db_key,
                                    block_pt,
                                    spec
                                );
                            }
                            DbBlock::Single(_) => {}
                            DbBlock::Expanded { actions, matches } => {
                                // Examine the table before updating.
                                // TODO: Avoid the following scan.
                                let arbitrary_value =
                                    actions.data.iter().find_map(|v| v.as_ref()).unwrap();
                                let inserting_same_value = arbitrary_value.0 == decisions;

                                let values_updated = fill_ndarray_region(
                                    actions,
                                    joined_row.iter().map(|(_, r)| r.clone()),
                                    &Some(ActionCostVec(decisions.clone())),
                                );

                                if let Some(m) = matches {
                                    if inserting_same_value {
                                        *m = m.checked_add(values_updated).unwrap();
                                        debug_assert!(
                                            m.get() <= block_shape.iter().product::<DimSize>()
                                        );
                                    } else {
                                        *matches = None;
                                    }
                                }
                            }
                        }
                        try_compress_block(r, &block_shape);
                        existing_block.into_ref()
                    }
                    Entry::Vacant(entry) => {
                        let block_shape_usize = block_shape
                            .iter()
                            .copied()
                            .map(|v| v.try_into().unwrap())
                            .collect::<Vec<_>>();
                        let mut arr = crate::ndarray::NDArray::new(&block_shape_usize);
                        let values_updated = fill_ndarray_region(
                            &mut arr,
                            joined_row.iter().map(|(_, r)| r.clone()),
                            &Some(ActionCostVec(decisions.clone())), // TODO: Remove this clone
                        );
                        let r = entry.insert(DbBlock::Expanded {
                            actions: arr,
                            matches: Some(NonZeroU32::new(values_updated).unwrap()),
                        });
                        r
                    }
                }
            };
            any_entry_ref = Some(block_entry_ref);
        }

        // TODO: Re-hashing and re-querying the inner HashMap should be unnecessary. Don't.
        let impls_ref = any_entry_ref
            .unwrap()
            .downgrade()
            .map(|submap| submap.get(&spec_b, self.binary_scale_shapes).unwrap());
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.get(&original_spec).as_deref(),
            Some(impls_ref.deref()),
            "Original Spec {:?} was incorrect after put.",
            original_spec
        );
        impls_ref
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
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
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
    fn get<Tgt: Target>(
        &self,
        query: &Spec<Tgt>,
        binary_scale_shapes: bool,
    ) -> Option<&ActionCostVec>
    where
        Tgt: Target,
        Tgt::Level: CanonicalBimap,
        <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
    {
        match self {
            DbBlock::Single(v) => {
                // TODO: Confirm that v is in bounds
                Some(v)
            }
            DbBlock::Expanded { actions, .. } => {
                let (_, global_pt) = compute_db_key(query, binary_scale_shapes);
                let (_, inner_pt) = blockify_point(global_pt);
                let inner_pt_usize = inner_pt.iter().map(|v| *v as usize).collect::<Vec<_>>();
                actions[&inner_pt_usize].as_ref()
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DbBlock::Single(_) => 1,
            DbBlock::Expanded {
                actions,
                matches: _,
            } => actions.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

fn try_compress_block(block: &mut DbBlock, block_shape: &[DimSize]) {
    // TODO: Should just precompute block_shape as MAX_BLOCK_VOLUME
    let block_volume = block_shape.iter().product::<DimSize>();
    match block {
        DbBlock::Expanded {
            actions,
            matches: Some(m),
        } if m.get() == block_volume => {
            // log::debug!("Compressing block of size {}", block_volume);
            let new_value = DbBlock::Single(actions.data.pop().unwrap().unwrap());
            *block = new_value;
        }
        _ => {}
    }
}

fn construct_impl<'a, Tgt, D>(db: &'a D, imp: &DbImpl<Tgt>) -> DbImpl<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
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

/// Convert a [Spec] into a database key, block coordinate, and within-block coordinate.
fn compute_db_key<Tgt>(spec: &Spec<Tgt>, binary_scale_shapes: bool) -> DbKey
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
{
    let bimap = SpecBimap {
        logical_spec_bimap: LogicalSpecBimap {
            primitive_basics_bimap: PrimitiveBasicsBimap {
                binary_scale_shapes,
            },
            aux_bimap: TensorSpecAuxNonDepBimap::default(),
        },
        memory_limits_bimap: MemoryLimitsBimap::default(),
    };
    BiMap::apply(&bimap, spec)
}

/// Convert a single dimension of a global point to a block and within-block index.
fn db_key_scale(dim: usize, value: BimapInt, dim_count: usize) -> (BimapInt, u8) {
    // TODO: Autotune rather than hardcode these arbitrary dimensions.
    let scale_factor = if dim == 2 { 2 } else { 4 };
    if dim == 2 || dim >= dim_count - 4 {
        let (quotient, remainder) = value.div_rem(&scale_factor);
        (quotient, remainder.try_into().unwrap())
    } else {
        (value, 0)
    }
}

fn block_shape(rank: usize) -> SmallVec<[BimapInt; 10]> {
    (0..rank)
        .map(|i| {
            if i == 2 {
                2
            } else if i >= rank - 4 {
                4
            } else {
                1
            }
        })
        .collect()
}

/// Converts a given global coordinate into block and within-block coordinates.
fn blockify_point(
    mut pt: SmallVec<[BimapInt; 10]>,
) -> (SmallVec<[BimapInt; 10]>, SmallVec<[u8; 10]>) {
    let rank = pt.len();
    let mut inner_pt = SmallVec::new();
    for (i, d) in pt.iter_mut().enumerate() {
        let (outer, inner) = db_key_scale(i, *d, rank);
        *d = outer;
        inner_pt.push(inner);
    }
    (pt, inner_pt)
}

/// Iterate over all points in the hyper-rectangle between two points, inclusive.
fn iter_between_points<const N: usize>(
    bottom_pt: &[u64; N],
    top_pt: &[u64; N],
) -> impl Iterator<Item = [u64; N]> {
    debug_assert!(bottom_pt.iter().zip(top_pt).all(|(b, t)| b <= t));
    bottom_pt
        .iter()
        .zip(top_pt)
        .map(|(b, t)| (*b..=*t))
        .multi_cartesian_product()
        .map(|v| v.try_into().unwrap())
}

/// Compute the bottom and top points (inclusive) to fill in a database table.
///
/// Returned points are in global coordinates, not within-block coordinates.
fn put_range_to_fill<Tgt>(
    spec: &Spec<Tgt>,
    impls: &SmallVec<[(ActionIdx, Cost); 1]>,
    binary_scale_shapes: bool,
) -> (
    (SpecKey, SmallVec<[Layout; 3]>),
    (SmallVec<[BimapInt; 10]>, SmallVec<[BimapInt; 10]>),
)
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: BiMap<Codomain = BimapInt>,
{
    // Compute the per-level maximum limits of the solutions. This lower bounds the range.
    let mut per_level_peaks = [0; LEVEL_COUNT];
    for (_, cost) in impls {
        for (i, &peak) in cost.peaks.iter().enumerate() {
            per_level_peaks[i] = per_level_peaks[i].max(peak);
        }
    }

    // Compute the complete upper and lower bounds from the given Spec and that Spec modified with
    // the peaks' bound (computed above).
    let upper_inclusive = compute_db_key(spec, binary_scale_shapes);
    let lower_inclusive = {
        let mut lower_bound_spec = spec.clone();
        lower_bound_spec.1 = MemoryLimits::Standard(MemVec::new(per_level_peaks));
        compute_db_key(&lower_bound_spec, binary_scale_shapes)
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
// TODO: Make private.
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

fn fill_ndarray_region<T, I>(
    array: &mut crate::ndarray::NDArray<T>,
    dim_iterators: I,
    value: &T,
) -> u32
where
    T: Clone + PartialEq + std::fmt::Debug,
    I: Iterator<Item = Range<BimapInt>>,
{
    let mut values_updated = 0;
    for pt in dim_iterators.multi_cartesian_product() {
        let pt_usize = pt
            .iter()
            .map(|v| usize::try_from(*v).unwrap())
            .collect::<Vec<_>>();
        if &array[&pt_usize] != value {
            values_updated += 1;
            array[&pt_usize] = value.clone();
        }
    }
    values_updated
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
        fn test_put_then_get_fills_across_memory_limits(entry in arb_spec_and_action::<X86Target>()) {
            let (spec, action, cost) = entry.clone();
            let (spec_b, action_b, cost_b) = entry;
            let MemoryLimits::Standard(spec_limits) = spec.1.clone();
            let db = DashmapDiskDatabase::new(None, false);
            let value_ref = db.put(spec, smallvec![(action, cost)]);
            let filled_limits_iter = spec_limits
                .into_iter()
                .zip(&value_ref.value().0[0].1.peaks)
                .map(|(l, p)| {
                    assert!(l == 0 || l.is_power_of_two());
                    assert!(p == 0 || p.is_power_of_two());
                    bit_length(p)..=bit_length(l)
                })
                .multi_cartesian_product();
            let expected = ActionCostVec(smallvec![(action_b, cost_b)]);
            for limit_to_check_bits in filled_limits_iter {
                let limit_to_check = limit_to_check_bits.iter().copied().map(bit_length_inverse).collect::<Vec<_>>();
                let spec_to_check = Spec(spec_b.0.clone(), MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())));
                let get_result = db.get(&spec_to_check).expect("Spec should be in database");
                let g = get_result.value();
                assert_eq!(g, &expected);
            }
        }

        #[test]
        fn test_database_empty_outside_range_after_one_put(entry in arb_spec_and_action::<X86Target>()) {
            let spec_b = entry.0.clone();
            let (spec, action, cost) = entry;
            let db = DashmapDiskDatabase::new(None, false);
            let value_ref = db.put(spec, smallvec![(action, cost)]);
            let MemoryLimits::Standard(max_memory) = X86Target::max_mem();
            let MemoryLimits::Standard(spec_limits) = &spec_b.1;
            let filled_limits_iter = max_memory
                .into_iter()
                .map(|l| {
                    assert!(l == 0 || l.is_power_of_two());
                    (0..=bit_length(l)).map(bit_length_inverse)
                })
                .multi_cartesian_product();
            for limit_to_check in filled_limits_iter {
                // Skip limits inside the put range
                if limit_to_check.iter().zip(value_ref.value().0[0].1.peaks.iter().zip(spec_limits)).any(|(c, (p, l))| {
                    c < p || c > &l
                }) {
                    let spec_to_check = Spec(
                        spec_b.0.clone(),
                        MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())),
                    );
                    assert!(db.get(&spec_to_check).is_none());
                }
            }
        }

        #[test]
        fn test_two_puts_return_correct_gets_for_second_put(
            entry_pair in arb_spec_and_action_pair::<X86Target>()
        ) {
            let db = DashmapDiskDatabase::new(None, false);
            let ((spec_a, action_a, cost_a), (spec_b, action_b, cost_b)) = entry_pair;
            let logical_specs_match = spec_a.0 == spec_b.0;

            let spec_a_cloned = spec_a.clone();
            let spec_b_cloned = spec_b.clone();
            let MemoryLimits::Standard(spec_limits_a) = &spec_a_cloned.1;
            let MemoryLimits::Standard(spec_limits_b) = &spec_b_cloned.1;

            let value_ref_a = db.put(spec_a, smallvec![(action_a, cost_a)]);
            let (action_a, cost_a) = value_ref_a.value().0[0].clone();
            let vr_a_peaks = value_ref_a.value().0[0].1.peaks.clone();
            drop(value_ref_a);

            let value_ref_b = db.put(spec_b, smallvec![(action_b, cost_b)]);
            let (action_b, cost_b) = value_ref_b.value().0[0].clone();
            let vr_b_peaks = value_ref_b.value().0[0].1.peaks.clone();
            drop(value_ref_b);

            let relevant_limits_iter = spec_limits_a.iter().zip(spec_limits_b)
                .zip(vr_a_peaks.iter().zip(&vr_b_peaks))
                .map(|((l_a, l_b), (p_a, p_b))| {
                    let l_a = *l_a;
                    let p_a = *p_a;
                    assert!(l_a == 0 || l_a.is_power_of_two());
                    assert!(l_b == 0 || l_b.is_power_of_two());
                    assert!(p_a == 0 || p_a.is_power_of_two());
                    assert!(p_b == 0 || p_b.is_power_of_two());
                    (bit_length(p_a.min(p_b))..=bit_length(l_a.max(l_b))).map(bit_length_inverse)
                })
                .multi_cartesian_product();

            for limit_to_check in relevant_limits_iter {
                // b was put second, so if we're in its range, that should be the result. Check a
                // second.
                let limit_in_a = vr_a_peaks.iter().zip(spec_limits_a).zip(&limit_to_check).all(|((bot, top), p)| {
                    debug_assert!(*bot <= top);
                    *bot <= *p && *p <= top
                });
                let limit_in_b = vr_b_peaks.iter().zip(spec_limits_b).zip(&limit_to_check).all(|((bot, top), p)| {
                    debug_assert!(*bot <= top);
                    *bot <= *p && *p <= top
                });

                let expected_value = if limit_in_b {
                    Some(ActionCostVec(smallvec![(action_b, cost_b.clone())]))
                } else if limit_in_a && logical_specs_match {
                    Some(ActionCostVec(smallvec![(action_a, cost_a.clone())]))
                } else {
                    None
                };

                let spec_to_check = Spec(
                    spec_b_cloned.0.clone(),
                    MemoryLimits::Standard(MemVec::new(limit_to_check.try_into().unwrap())),
                );
                let get_result = db.get(&spec_to_check);
                match (get_result.as_ref(), expected_value.as_ref()) {
                    (Some(g), Some(e)) if g.value() == e => {}
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
                            Some(g) => format!("Some({:?})", g.value()),
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

    fn arb_spec_and_action<Tgt: Target>() -> impl Strategy<Value = (Spec<Tgt>, ActionIdx, Cost)> {
        (any::<Spec<Tgt>>())
            .prop_flat_map(|spec| {
                let MemoryLimits::Standard(spec_limits) = &spec.1;
                let limits_strategy = spec_limits
                    .iter()
                    .map(|&l| {
                        (0..=bit_length(l))
                            .prop_map(|bits| if bits == 0 { 0 } else { 1 << (bits - 1) })
                    })
                    .collect::<Vec<_>>();
                (
                    Just(spec),
                    limits_strategy,
                    any::<ActionIdx>(),
                    any::<MainCost>(),
                    any::<u32>(),
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

                        let spec_limits = limits_a.iter().zip(&limits_b).map(|(a, b)| (*a).max(b)).collect::<Vec<_>>();
                        let peaks = limits_a.iter().zip(&limits_b).map(|(a, b)| (*a).min(b)).collect::<Vec<_>>();

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
