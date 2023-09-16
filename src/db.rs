use crate::cost::Cost;
use crate::datadeps::SpecKey;
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::Bimap;
use crate::grid::linear::BimapInt;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::pprint::PrintableAux;
use crate::scheduling::Action;
use crate::spec::{LogicalSpecBimap, Spec, SpecBimap};
use crate::target::Target;
use crate::tensorspec::TensorSpecAuxNonDepBimap;

use anyhow::anyhow;
use dashmap::mapref::one::MappedRef;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::ops::Deref;
use std::time::Instant;
use std::{iter, path};

const SCALE_FACTOR: BimapInt = 2;

pub type DbImpl<Tgt> = ImplNode<Tgt, DbImplAux<Tgt>>;

type DbKey = ((SpecKey, SmallVec<[Layout; 3]>), SmallVec<[BimapInt; 10]>);

pub trait Database<'a, Tgt: Target> {
    type ValueRef: AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> + 'a;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<Self::ValueRef>;
    // TODO: Document interior mutability of put.
    fn put(
        &'a self,
        problem: Spec<Tgt>,
        impls: SmallVec<[(Action<Tgt>, Cost); 1]>,
    ) -> Self::ValueRef;
    fn flush(&'a self);
    fn save(&'a self) -> anyhow::Result<()>;
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
    blocks: DashMap<DbKey, HashMap<Spec<Tgt>, ActionCostVec<Tgt>>>,
}

// TODO: Storing Spec and usize is too expensive.
pub struct DashmapDbRef<'a, Tgt: Target, S = RandomState>(
    dashmap::mapref::one::Ref<'a, DbKey, HashMap<Spec<Tgt>, ActionCostVec<Tgt>>, S>,
    Option<&'a ActionCostVec<Tgt>>,
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
    type ValueRef =
        MappedRef<'a, DbKey, HashMap<Spec<Tgt>, ActionCostVec<Tgt>>, ActionCostVec<Tgt>>;

    fn get(&'a self, query: &Spec<Tgt>) -> Option<Self::ValueRef> {
        let db_key = compute_db_key(query);
        let Some(group) = self.blocks.get(&db_key) else {
            return None;
        };
        if group.get(query).is_none() {
            return None;
        }
        // TODO: Obviate need to hash and look up on the inner HashMap for each deref of the below.
        Some(group.map(|g| g.get(query).unwrap()))
    }

    fn put(&'a self, spec: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>) -> Self::ValueRef {
        // TODO: How to treat non-powers of two memory bounds?

        #[cfg(debug_assertions)]
        let original_spec = spec.clone();

        let db_key = compute_db_key(&spec);
        let spec_b = spec.clone();

        assert!(impls.len() <= 1);
        let mut block = self.blocks.entry(db_key).or_default();
        block.insert(spec, ActionCostVec(impls));
        // TODO: Re-hashing and re-querying the inner HashMap is unnecessary. Avoid that.
        let impls_ref = block.downgrade().map(|submap| submap.get(&spec_b).unwrap());

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

    fn save(&'a self) -> anyhow::Result<()> {
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

impl<Tgt> Drop for DashmapDiskDatabase<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    fn drop(&mut self) {
        self.save().unwrap();
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

impl<'a, Tgt: Target> Deref for DashmapDbRef<'a, Tgt> {
    type Target = ActionCostVec<Tgt>;

    fn deref(&self) -> &Self::Target {
        self.1.unwrap()
    }
}

impl<'a, Tgt: Target> AsRef<ActionCostVec<Tgt>> for DashmapDbRef<'a, Tgt> {
    fn as_ref(&self) -> &ActionCostVec<Tgt> {
        self.deref()
    }
}

impl<'a, Tgt: Target> AsRef<SmallVec<[(Action<Tgt>, Cost); 1]>> for DashmapDbRef<'a, Tgt> {
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

/// Compute a key by which to group this [Spec]'s entries.
///
/// This key will not uniquely identify the [Spec].
fn compute_db_key<Tgt>(spec: &Spec<Tgt>) -> DbKey
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Codomain = BimapInt>,
{
    let bimap = SpecBimap {
        logical_spec_bimap: LogicalSpecBimap {
            aux_bimap: TensorSpecAuxNonDepBimap::<Tgt>::default(),
        },
    };
    let (table_key, mut pt) = bimap.apply(spec);
    for d in pt.iter_mut() {
        *d = (*d + 1).ilog2() / SCALE_FACTOR;
    }
    (table_key, pt)
}
