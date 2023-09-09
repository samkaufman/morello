use crate::cost::Cost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::pprint::PrintableAux;
use crate::scheduling::Action;
use crate::spec::{LogicalSpec, Spec};
use crate::target::Target;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::hash_map::RandomState;
use std::marker::PhantomData;
use std::ops::Deref;
use std::time::Instant;
use std::{iter, path};

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

pub trait Database<'a, Tgt: Target> {
    type Value: AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> + 'a;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<Self::Value>;
    // TODO: Document interior mutability of put.
    fn put(&'a self, problem: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>) -> Self::Value;
    fn flush(&'a self);
    // TODO: `save` return Result
    fn save(&'a self);
}

pub trait DatabaseExt<'a, Tgt: Target> {
    fn get_impl(&'a self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>;
}

#[derive(Clone, Debug)]
pub struct DbImplAux<Tgt: Target>(Option<(Spec<Tgt>, Cost)>);

pub struct DashmapDiskDatabase<Tgt: Target> {
    file_path: Option<path::PathBuf>,
    grouped_entries: DashMap<LogicalSpec<Tgt>, LogicalSpecEntry<Tgt>>,
}

// TODO: Entry should not need to be public.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(bound = "")]
pub struct LogicalSpecEntry<Tgt: Target> {
    ranges: Vec<(MemVec, MemVec)>,
    values: Vec<ActionCostVec<Tgt>>,
}

pub struct LogicalSpecEntryRef<'a, Tgt: Target, S = RandomState>(
    dashmap::mapref::one::Ref<'a, LogicalSpec<Tgt>, LogicalSpecEntry<Tgt>, S>,
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

impl<Tgt: Target> DashmapDiskDatabase<Tgt> {
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
            grouped_entries,
        }
    }
}

impl<Tgt: Target> AsRef<ActionCostVec<Tgt>> for ActionCostVec<Tgt> {
    fn as_ref(&self) -> &ActionCostVec<Tgt> {
        self
    }
}

impl<'a, Tgt: Target> Database<'a, Tgt> for DashmapDiskDatabase<Tgt>
where
    Self: 'a,
{
    type Value = LogicalSpecEntryRef<'a, Tgt>;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<LogicalSpecEntryRef<'a, Tgt>> {
        let Some(e) = self.grouped_entries.get(&query.0) else {
            return None;
        };
        get_from_entry(&query.1, e.value())
            .map(move |i| LogicalSpecEntryRef::<'a, Tgt>(e, i, PhantomData))
    }

    fn put(
        &'a self,
        spec: Spec<Tgt>,
        impls: SmallVec<[(Action<Tgt>, Cost); 1]>,
    ) -> LogicalSpecEntryRef<'a, Tgt> {
        // TODO: How to treat non-powers of two memory bounds?
        let impls_ref = match spec.1 {
            MemoryLimits::Standard(lims) => {
                assert!(impls.len() <= 1);
                let mut existing = self.grouped_entries.entry(spec.0.clone()).or_default();
                if impls.is_empty() {
                    existing.ranges.push((lims, MemVec::zero::<Tgt>()));
                } else {
                    existing.ranges.push((lims, impls[0].1.peaks.clone()));
                }
                let insertion_idx = existing.values.len();
                existing.values.push(ActionCostVec(impls));
                LogicalSpecEntryRef(existing.downgrade(), insertion_idx, PhantomData)
            }
        };

        // debug_assert_eq!(
        //     self.get(&original_spec).map(|x| x.deref()),
        //     Some(impls_ref),
        //     "Original Spec {:?} was incorrect after put.",
        //     original_spec
        // );

        impls_ref
    }

    fn flush(&'a self) {}

    fn save(&'a self) {
        if let Some(path) = &self.file_path {
            let file = std::fs::File::create(path).unwrap();
            let encoder = snap::write::FrameEncoder::new(file);
            let start = Instant::now();
            bincode::serialize_into(encoder, &self.grouped_entries).unwrap();
            log::debug!("Saving database took {:?}", start.elapsed());
        }
    }
}

// TODO: Phase out save-on-Drop, since we may want to have read-only databases.
impl<T: Target> Drop for DashmapDiskDatabase<T> {
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
        &self.0.value().values[self.1]
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
