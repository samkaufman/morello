use crate::common::Spec;
use crate::cost::Cost;
use crate::imp::{Impl, ImplNode};
use crate::memorylimits::{MemVec, MemoryLimits};
use crate::scheduling::Action;
use crate::spec::LogicalSpec;
use crate::target::Target;

use rusqlite::{params, params_from_iter, OptionalExtension};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::mem;
use std::path;

const SQLITE_BATCH_SIZE: usize = 1_000;

pub type DbImpl<Tgt> = ImplNode<Tgt, Option<(Spec<Tgt>, Cost)>>;

pub trait Database<Tgt: Target> {
    fn get(&self, query: &Spec<Tgt>) -> Option<SmallVec<[(Action<Tgt>, Cost); 1]>>;
    // TODO: Drop get_spec, which exists solely for SqliteDatabaseWrapper.
    fn get_spec(&self, spec: &LogicalSpec<Tgt>) -> Option<&Entry<Tgt>>;
    fn put(&mut self, problem: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>);
    fn flush(&mut self);
}

pub trait DatabaseExt<Tgt: Target> {
    fn get_impl(&self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>>;
}

pub struct InMemDatabase<Tgt: Target> {
    grouped_entries: HashMap<LogicalSpec<Tgt>, Entry<Tgt>>,
}

pub struct SqliteDatabaseWrapper<Tgt: Target, D: Database<Tgt>> {
    inner: D,
    tx: crossbeam_channel::Sender<SqliteDatabaseWrapperMsg<Tgt>>,
    rx: crossbeam_channel::Receiver<SqliteDatabaseWrapperResponse<Tgt>>,
    bg_thread_handle: Option<std::thread::JoinHandle<()>>,
}

#[allow(clippy::large_enum_variant)] // Because Flush and Stop are rare.
enum SqliteDatabaseWrapperMsg<Tgt: Target> {
    Get(Spec<Tgt>),
    Put(LogicalSpec<Tgt>, Entry<Tgt>),
    Flush { should_respond: bool },
    Stop,
}

enum SqliteDatabaseWrapperResponse<Tgt: Target> {
    GetResponse(Option<SmallVec<[(Action<Tgt>, Cost); 1]>>),
    FlushDone,
}

// TODO: Entry should not need to be public.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(bound = "")]
pub struct Entry<Tgt: Target> {
    ranges: Vec<(MemVec, MemVec)>,
    values: Vec<SmallVec<[(Action<Tgt>, Cost); 1]>>,
}

impl<Tgt: Target> InMemDatabase<Tgt> {
    pub fn new() -> Self {
        InMemDatabase {
            grouped_entries: HashMap::new(),
        }
    }
}

impl<Tgt: Target> Database<Tgt> for InMemDatabase<Tgt> {
    fn get(&self, query: &Spec<Tgt>) -> Option<SmallVec<[(Action<Tgt>, Cost); 1]>> {
        self.grouped_entries
            .get(&query.0)
            .and_then(|e| get_from_entry(&query.1, e).cloned())
    }

    fn get_spec(&self, spec: &LogicalSpec<Tgt>) -> Option<&Entry<Tgt>> {
        self.grouped_entries.get(spec)
    }

    fn put(&mut self, problem: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>) {
        // self.entries.insert(problem, impls);

        let orig_problem = problem.clone(); // Save for debug_assert_eq! postcondition.
        let orig_impls = impls.clone();

        // TODO: How to treat non-powers of two memory bounds?
        match problem.1 {
            MemoryLimits::Standard(lims) => {
                assert!(impls.len() <= 1);
                let existing: &mut Entry<Tgt> =
                    self.grouped_entries.entry(problem.0.clone()).or_default();
                if impls.is_empty() {
                    existing.ranges.push((lims, MemVec::zero::<Tgt>()));
                } else {
                    existing.ranges.push((lims, impls[0].1.peaks.clone()));
                }
                existing.values.push(impls);
            }
        }

        debug_assert_eq!(
            self.get(&orig_problem),
            Some(orig_impls),
            "Original problem {:?} was incorrect after put.",
            orig_problem
        );
    }

    fn flush(&mut self) {}
}

impl<Tgt: Target, D: Database<Tgt>> SqliteDatabaseWrapper<Tgt, D> {
    pub fn new(inner: D, db_path: &path::Path) -> Self {
        let db_path = db_path.to_owned();
        let (request_tx, request_rx) = crossbeam_channel::bounded(1);
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        SqliteDatabaseWrapper {
            inner,
            tx: request_tx,
            rx: response_rx,
            bg_thread_handle: Some(std::thread::spawn(move || {
                let conn = rusqlite::Connection::open(db_path).unwrap();
                conn.pragma_update(None, "journal_mode", "WAL").unwrap();
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS impls (
                    spec  BLOB PRIMARY KEY,
                    entry BLOB
                )",
                    (),
                )
                .unwrap();

                let mut pending_puts = HashMap::with_capacity(SQLITE_BATCH_SIZE);
                loop {
                    match request_rx.recv().unwrap() {
                        SqliteDatabaseWrapperMsg::Get(spec) => {
                            let db_result: Option<Entry<Tgt>> = conn
                                .query_row(
                                    "SELECT entry FROM impls WHERE spec = ?",
                                    [serde_json::to_string(&spec.0).unwrap()],
                                    |row| {
                                        Ok(serde_json::from_str(&row.get::<_, String>(0).unwrap())
                                            .unwrap())
                                    },
                                )
                                .optional()
                                .unwrap();
                            let response = match db_result {
                                Some(entry) => get_from_entry(&spec.1, &entry).cloned(),
                                None => None,
                            };
                            response_tx
                                .send(SqliteDatabaseWrapperResponse::GetResponse(response))
                                .unwrap();
                        }
                        SqliteDatabaseWrapperMsg::Put(problem, impls) => {
                            pending_puts.insert(problem, impls);
                            if pending_puts.len() >= SQLITE_BATCH_SIZE {
                                do_flush(&conn, &mut pending_puts);
                            }
                        }
                        SqliteDatabaseWrapperMsg::Flush { should_respond } => {
                            do_flush(&conn, &mut pending_puts);
                            if should_respond {
                                response_tx
                                    .send(SqliteDatabaseWrapperResponse::FlushDone)
                                    .unwrap();
                            }
                        }
                        SqliteDatabaseWrapperMsg::Stop => {
                            break;
                        }
                    }
                }
            })),
        }
    }

    fn stop(&mut self) {
        self.flush();
        self.tx.send(SqliteDatabaseWrapperMsg::Stop).unwrap();
        mem::take(&mut self.bg_thread_handle)
            .unwrap()
            .join()
            .unwrap();
    }
}

impl<Tgt: Target, D: Database<Tgt>> Database<Tgt> for SqliteDatabaseWrapper<Tgt, D> {
    fn get(&self, query: &Spec<Tgt>) -> Option<SmallVec<[(Action<Tgt>, Cost); 1]>> {
        if let Some(r) = self.inner.get(query) {
            return Some(r);
        };
        self.tx
            .send(SqliteDatabaseWrapperMsg::Get(query.clone()))
            .unwrap();
        match self.rx.recv().unwrap() {
            SqliteDatabaseWrapperResponse::GetResponse(r) => r,
            SqliteDatabaseWrapperResponse::FlushDone => {
                panic!("Unexpected FlushDone response to Get from background thread")
            }
        }
    }

    fn get_spec(&self, _spec: &LogicalSpec<Tgt>) -> Option<&Entry<Tgt>> {
        unimplemented!()
    }

    fn put(&mut self, problem: Spec<Tgt>, impls: SmallVec<[(Action<Tgt>, Cost); 1]>) {
        self.inner.put(problem.clone(), impls);
        let updated = self.inner.get_spec(&problem.0).unwrap();
        self.tx
            .send(SqliteDatabaseWrapperMsg::Put(problem.0, updated.clone()))
            .unwrap();
    }

    fn flush(&mut self) {
        self.inner.flush();
        self.tx
            .send(SqliteDatabaseWrapperMsg::Flush {
                should_respond: true,
            })
            .unwrap();
        self.rx.recv().unwrap();
    }
}

impl<Tgt: Target, D: Database<Tgt>> Drop for SqliteDatabaseWrapper<Tgt, D> {
    fn drop(&mut self) {
        self.stop();
    }
}

impl<Tgt: Target, T: Database<Tgt>> DatabaseExt<Tgt> for T {
    fn get_impl(&self, query: &Spec<Tgt>) -> Option<SmallVec<[DbImpl<Tgt>; 1]>> {
        let Some(root_results) = self.get(query) else {
            return None;
        };
        Some(
            root_results
                .iter()
                .map(|root_result| {
                    let root = root_result
                        .0
                        .apply_with_aux(query, Some((query.clone(), root_result.1.clone())))
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

impl<Tgt: Target> Default for Entry<Tgt> {
    fn default() -> Self {
        Self {
            ranges: Default::default(),
            values: Default::default(),
        }
    }
}

fn do_flush<Tgt: Target>(
    conn: &rusqlite::Connection,
    pending_puts: &mut HashMap<LogicalSpec<Tgt>, Entry<Tgt>>,
) {
    if !pending_puts.is_empty() {
        let post = vec!["(?, ?)"].repeat(pending_puts.len()).join(", ");

        conn.execute(
            &format!("INSERT OR REPLACE INTO impls (spec, entry) VALUES {}", post),
            params_from_iter(pending_puts.drain().flat_map(|(s, entry)| {
                vec![
                    serde_json::to_string(&s).unwrap(),
                    serde_json::to_string(&entry).unwrap(),
                ]
            })),
        )
        .unwrap();
    }
}

fn construct_impl<Tgt: Target, D: Database<Tgt>>(db: &D, imp: &DbImpl<Tgt>) -> DbImpl<Tgt> {
    match imp {
        ImplNode::ProblemApp(p) => db
            .get_impl(&p.0)
            .expect("Database should have the sub-problem entry")
            .get(0)
            .expect("Database sub-problem should be satisfiable")
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

fn get_from_entry<'a, Tgt: Target>(
    mlims: &MemoryLimits,
    entry: &'a Entry<Tgt>,
) -> Option<&'a SmallVec<[(Action<Tgt>, Cost); 1]>> {
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
