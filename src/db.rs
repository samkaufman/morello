use crate::cost::Cost;
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::Bimap;
use crate::grid::linear::BimapInt;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::pprint::PrintableAux;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, LogicalSpecBimap, Spec};
use crate::target::Target;
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::time::Instant;
use std::{iter, path};

const SCALE_FACTOR: BimapInt = 2;

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

type DbKey = ((SpecKey, Vec<Layout>), Vec<BimapInt>);

pub trait Database<'a, Tgt: Target> {
    type Value: AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> + 'a;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<Self::Value>;
    // TODO: Document interior mutability of put.
    fn put(&'a self, problem: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>) -> Self::Value;
    fn flush(&'a self);
    // TODO: `save` should return Result
    fn save(&'a self);
}

pub trait DatabaseExt<'a, Tgt: Target>: Database<'a, Tgt> {
    fn get_impl(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>;
}

#[derive(Clone, Debug)]
pub struct DbImplAux<Tgt: Target>(Option<(Spec<Tgt>, Cost)>);

pub struct DashmapDiskDatabase<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    file_path: Option<path::PathBuf>,
    blocks: DashMap<DbKey, HashMap<LogicalSpec<Tgt>, LogicalSpecEntry<Tgt>>>,
}

// TODO: Entry should not need to be public.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(bound = "")]
pub struct LogicalSpecEntry<Tgt: Target> {
    ranges: Vec<(MemVec, MemVec)>,
    values: Vec<ActionCostVec<Tgt>>,
}

// TODO: Storing Spec and usize is too expensive.
pub struct LogicalSpecEntryRef<'a, Tgt: Target, S = RandomState>(
    dashmap::mapref::one::Ref<'a, DbKey, HashMap<LogicalSpec<Tgt>, LogicalSpecEntry<Tgt>>, S>,
    LogicalSpec<Tgt>,
    usize,
    PhantomData<Tgt>,
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ActionCostVec<Tgt: Target>(pub SmallVec<[(Action<Tgt>, Cost); 1]>);

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

impl<Tgt> DashmapDiskDatabase<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    // TODO: This does I/O; it should return errors, not panic.
    pub fn new(file_path: Option<&path::Path>) -> Self {
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
                    std::io::ErrorKind::NotFound => Default::default(),
                    _ => todo!("Handle other file errors"),
                },
            },
            None => Default::default(),
        };
        Self {
            file_path: file_path.map(|p| p.to_owned()),
            blocks: grouped_entries,
        }
    }
}

impl<Tgt: Target> AsRef<ActionCostVec<Tgt>> for ActionCostVec<Tgt> {
    fn as_ref(&self) -> &ActionCostVec<Tgt> {
        self
    }
}

impl<'a, Tgt> Database<'a, Tgt> for DashmapDiskDatabase<Tgt>
where
    Self: 'a,
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    type Value = LogicalSpecEntryRef<'a, Tgt>;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<LogicalSpecEntryRef<'a, Tgt>> {
        let db_key = compute_db_key(query);
        let Some(group) = self.blocks.get(&db_key) else {
            return None;
        };
        let Some(e) = group.get(&query.0) else {
            return None;
        };
        get_from_entry(&query.1, e)
            .map(move |i| LogicalSpecEntryRef::<'a, Tgt>(group, query.0.clone(), i, PhantomData))
    }

    fn put(
        &'a self,
        spec: Spec<Tgt>,
        impls: SmallVec<[(Action<Tgt>, Cost); 1]>,
    ) -> LogicalSpecEntryRef<'a, Tgt> {
        // TODO: How to treat non-powers of two memory bounds?

        #[cfg(debug_assertions)]
        let original_spec = spec.clone();

        let db_key = compute_db_key(&spec);

        assert!(impls.len() <= 1);
        let impls_ref = match spec.1 {
            MemoryLimits::Standard(lims) => {
                let mut block = self.blocks.entry(db_key).or_default();
                let existing = block.entry(spec.0.clone()).or_default();
                if impls.is_empty() {
                    existing.ranges.push((lims, MemVec::zero::<Tgt>()));
                } else {
                    existing.ranges.push((lims, impls[0].1.peaks.clone()));
                }
                let insertion_idx = existing.values.len();
                existing.values.push(ActionCostVec(impls));
                LogicalSpecEntryRef(
                    block.downgrade(),
                    spec.0.clone(),
                    insertion_idx,
                    PhantomData,
                )
            }
        };

        #[cfg(debug_assertions)]
        {
            let lhs = self.get(&original_spec);
            debug_assert_eq!(
                lhs.as_deref(),
                Some(impls_ref.deref()),
                "Original Spec {:?} was incorrect after put.",
                original_spec
            )
        }

        impls_ref
    }

    fn flush(&'a self) {}

    fn save(&'a self) {
        if let Some(path) = &self.file_path {
            let start = Instant::now();
            let temp_file_path = {
                let temp_file = tempfile::NamedTempFile::new().unwrap();
                let encoder = snap::write::FrameEncoder::new(&temp_file);
                bincode::serialize_into(encoder, &self.blocks).unwrap();
                temp_file.keep().unwrap().1
            };
            std::fs::rename(temp_file_path, path).unwrap();
            log::debug!("Saving database took {:?}", start.elapsed());
        }
    }
}

impl<Tgt> Drop for DashmapDiskDatabase<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    fn drop(&mut self) {
        self.save();
    }
}

impl<'a, Tgt: Target, T: Database<'a, Tgt>> DatabaseExt<'a, Tgt> for T {
    fn get_impl(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>> {
        let Some(root_results) = self.get(query) else {
            return None;
        };
        Some(
            root_results
                .as_ref()
                .iter()
                .map(|(action, cost)| {
                    let root = action
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

impl<Tgt: Target> Default for LogicalSpecEntry<Tgt> {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            values: Default::default(),
        }
    }
}

impl<Tgt: Target> AsRef<ActionCostVec<Tgt>> for LogicalSpecEntry<Tgt> {
    fn as_ref(&self) -> &ActionCostVec<Tgt> {
        &self.values[0]
    }
}

impl<'a, Tgt: Target> Deref for LogicalSpecEntryRef<'a, Tgt> {
    type Target = ActionCostVec<Tgt>;

    fn deref(&self) -> &Self::Target {
        &self.0.value().get(&self.1).unwrap().values[self.2]
    }
}

impl<'a, Tgt: Target> AsRef<ActionCostVec<Tgt>> for LogicalSpecEntryRef<'a, Tgt> {
    fn as_ref(&self) -> &ActionCostVec<Tgt> {
        self.deref()
    }
}

impl<'a, Tgt: Target> AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> for LogicalSpecEntryRef<'a, Tgt> {
    fn as_ref(&self) -> &SmallVec<[(Action<Tgt>, Cost); 1]> {
        self.deref().as_ref()
    }
}

impl<Tgt: Target> Deref for ActionCostVec<Tgt> {
    type Target = SmallVec<[(Action<Tgt>, Cost); 1]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Tgt: Target> AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> for ActionCostVec<Tgt> {
    fn as_ref(&self) -> &SmallVec<[(Action<Tgt>, Cost); 1]> {
        self.deref()
    }
}

fn construct_impl<'a, Tgt: Target, D: Database<'a, Tgt>>(
    db: &'a D,
    imp: &DbImpl<Tgt>,
) -> DbImpl<Tgt> {
    match imp {
        ImplNode::SpecApp(p) => db
            .get_impl(&p.0)
            .expect("Database should have the sub-Spec entry")
            .get(0)
            .expect("Database sub-Spec should be satisfiable")
            .clone(),
        _ => {
            let new_children = imp
                .children()
                .iter()
                .map(|c| construct_impl(db, c))
                .collect::<Vec<_>>();
            imp.replace_children(new_children.into_iter())
        }
    }
}

fn get_from_entry<Tgt: Target>(
    mlims: &MemoryLimits,
    entry: &LogicalSpecEntry<Tgt>,
) -> Option<usize> {
    match mlims {
        MemoryLimits::Standard(query_lims) => {
            for (i, (lims, peaks)) in entry.ranges.iter().enumerate() {
                if query_lims
                    .into_iter()
                    .zip(lims)
                    .zip(peaks)
                    .all(|((q, l), p)| l >= q && q >= p)
                {
                    return Some(i);
                }
            }
            None
        }
    }
}

/// Compute a key by which to group this [Spec]'s entries.
///
/// This key will not uniquely identify the [Spec].
fn compute_db_key<Tgt>(spec: &Spec<Tgt>) -> DbKey
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    // TODO: Inline this function and lift bimap construction.
    // TODO: Scale the coordinates to be more appropriate DB key.
    let bimap = LogicalSpecBimap {
        aux_bimap: TensorSpecAuxNonDepBimap::<Tgt>::default(),
    };
    let (table_key, mut pt) = bimap.apply(&spec.0);
    for d in pt.iter_mut() {
        *d = *d / SCALE_FACTOR;
    }
    (table_key, pt)
}
