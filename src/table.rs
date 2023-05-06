use crate::common::Problem;
use crate::cost::Cost;
use crate::imp::ImplNode;
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::spec::Spec;
use crate::target::Target;

use log::warn;
use rusqlite::params_from_iter;
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::path::{self, PathBuf};

const SQLITE_BATCH_SIZE: usize = 8_000;

pub struct Database<Tgt: Target, S: DatabaseIOStore<Tgt>> {
    grouped_entries: HashMap<Spec<Tgt>, Entry<Tgt>>,
    store: std::sync::RwLock<S>,
}

// TODO: Entry should not need to be public.
#[derive(Debug, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct Entry<Tgt: Target> {
    ranges: Vec<(MemVec, MemVec)>,
    values: Vec<Vec<(ImplNode<Tgt>, Cost)>>,
}

pub trait DatabaseIOStore<Tgt: Target> {
    fn get(&mut self, spec: &Spec<Tgt>) -> Option<Entry<Tgt>>;
    fn put_entry(
        &mut self,
        _spec: &Spec<Tgt>,
        _entry: &Entry<Tgt>,
        grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>,
    );
    fn flush(&mut self, grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>);
}

#[derive(Default)]
pub struct NullDatabaseIOStore {}

pub struct SqliteIOStore<Tgt: Target> {
    path: PathBuf,
    conn_local: thread_local::ThreadLocal<rusqlite::Connection>,
    queued_spec_inserts: HashSet<Spec<Tgt>>,
}

impl<Tgt: Target, S: DatabaseIOStore<Tgt>> Database<Tgt, S> {
    pub fn new(store: S) -> Self {
        Database {
            grouped_entries: HashMap::new(),
            store: std::sync::RwLock::new(store),
        }
    }

    pub fn get(&self, query: &Problem<Tgt>) -> Option<&Vec<(ImplNode<Tgt>, Cost)>> {
        let gotten = self.grouped_entries.get(&query.0);
        if gotten.is_none() {
            if let Some(_rr) = self.store.write().unwrap().get(&query.0) {
                warn!("Not saving entry loaded from SQLite");
                // self.grouped_entries.entry(query.0.clone()).or_insert(rr);
                // TODO: Return
            }
        }
        gotten.and_then(|e| self.get_from_entry(&query.1, e))
    }

    fn get_from_entry<'a>(
        &'a self,
        mlims: &MemoryLimits,
        entry: &'a Entry<Tgt>,
    ) -> Option<&Vec<(ImplNode<Tgt>, Cost)>> {
        match mlims {
            MemoryLimits::Standard(query_lims) => {
                for (i, (lims, peaks)) in entry.ranges.iter().enumerate() {
                    if query_lims
                        .iter()
                        .zip(lims)
                        .zip(peaks)
                        .all(|((q, l), p)| l >= *q && *q >= p)
                    {
                        return Some(&entry.values[i]);
                    }
                }
                None
            }
        }
    }

    pub fn put(&mut self, problem: Problem<Tgt>, impls: Vec<(ImplNode<Tgt>, Cost)>) {
        // self.entries.insert(problem, impls);

        let orig_problem = problem.clone(); // Save for debug_assert_eq! postcondition.
        let orig_impls = impls.clone();

        // TODO: How to treat non-powers of two memory bounds?
        match problem.1 {
            MemoryLimits::Standard(lims) => {
                assert!(impls.len() <= 1);

                {
                    let existing: &mut Entry<Tgt> =
                        self.grouped_entries.entry(problem.0.clone()).or_default();
                    if impls.is_empty() {
                        existing.ranges.push((lims, MemVec::zero::<Tgt>()));
                    } else {
                        existing.ranges.push((lims, impls[0].1.peaks.clone()));
                    }
                    existing.values.push(impls);
                }

                self.store.write().unwrap().put_entry(
                    &problem.0,
                    self.grouped_entries.get(&problem.0).unwrap(),
                    &self.grouped_entries,
                );
            }
        }

        debug_assert_eq!(
            self.get(&orig_problem),
            Some(&orig_impls),
            "Original problem {:?} was incorrect after put.",
            orig_problem
        );
    }
}

impl<Tgt: Target, S: DatabaseIOStore<Tgt>> Drop for Database<Tgt, S> {
    fn drop(&mut self) {
        self.store.write().unwrap().flush(&self.grouped_entries);
    }
}

impl<Tgt: Target> Default for Entry<Tgt> {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            values: Default::default(),
        }
    }
}

impl<Tgt: Target> DatabaseIOStore<Tgt> for NullDatabaseIOStore {
    fn get(&mut self, _spec: &Spec<Tgt>) -> Option<Entry<Tgt>> {
        None
    }

    fn put_entry(
        &mut self,
        _spec: &Spec<Tgt>,
        _entry: &Entry<Tgt>,
        _grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>,
    ) {
        // Do nothing.
    }

    fn flush(&mut self, _grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>) {}
}

impl<Tgt: Target> SqliteIOStore<Tgt> {
    pub fn new(path: &path::Path) -> Self {
        // Set up the database, but then discard this connection. Connections
        // will subsequently be created lazily, per thread.
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.pragma_update(None, "journal_mode", "WAL").unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS impls (
            spec  BLOB PRIMARY KEY,
            entry BLOB
        )",
            (),
        )
        .unwrap();

        SqliteIOStore {
            path: path.to_owned(),
            conn_local: thread_local::ThreadLocal::new(),
            queued_spec_inserts: HashSet::with_capacity(SQLITE_BATCH_SIZE),
        }
    }
}

impl<Tgt: Target> DatabaseIOStore<Tgt> for SqliteIOStore<Tgt> {
    fn get(&mut self, spec: &Spec<Tgt>) -> Option<Entry<Tgt>> {
        let path = &self.path;
        let conn = self
            .conn_local
            .get_or(|| rusqlite::Connection::open(path).unwrap());

        let spec_str = serde_json::to_string(&spec).unwrap();

        let mut lookup_stmt = conn
            .prepare("SELECT entry FROM impls WHERE spec = ? LIMIT 1")
            .unwrap();
        let mut entry_rows = lookup_stmt.query([&spec_str]).unwrap();
        match entry_rows.next().unwrap() {
            Some(row) => {
                let entry_str: String = row.get(0).unwrap();
                Some(serde_json::from_str(&entry_str).unwrap())
            }
            None => None,
        }
    }

    fn put_entry(
        &mut self,
        spec: &Spec<Tgt>,
        _entry: &Entry<Tgt>,
        grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>,
    ) {
        self.queued_spec_inserts.insert(spec.clone());
        if self.queued_spec_inserts.len() >= SQLITE_BATCH_SIZE {
            self.flush(grouped_entries);
        }
    }

    fn flush(&mut self, grouped_entries: &HashMap<Spec<Tgt>, Entry<Tgt>>) {
        if self.queued_spec_inserts.is_empty() {
            return;
        }

        let post = vec!["(?, ?)"]
            .repeat(self.queued_spec_inserts.len())
            .join(", ");

        self.conn_local
            .get()
            .unwrap()
            .execute(
                &format!("INSERT OR REPLACE INTO impls (spec, entry) VALUES {}", post),
                params_from_iter(self.queued_spec_inserts.iter().flat_map(|spec| {
                    vec![
                        serde_json::to_string(spec).unwrap(),
                        serde_json::to_string(&grouped_entries.get(spec).unwrap()).unwrap(),
                    ]
                })),
            )
            .unwrap();

        self.queued_spec_inserts.clear();
    }
}
