//! `ReadTxn::collection` uses per-snapshot catalog root.
//!
//! Regression test for the race the stress test surfaced.
//! Before the fix, a reader started before a writer added a new
//! collection would consult the writer's LIVE `Catalog.tree.root`
//! (which had COW-advanced past the snapshot's pinned LSN) and
//! either observe the post-write catalog state OR — once the
//! writer's COW-freed pages were recycled and re-staged in
//! `state.view` — surface as `Error::Corruption { page_id: 0 }`
//! from the B+tree codec.
//!
//! After the fix, the reader walks the catalog B+tree rooted at the
//! snapshot's pinned `root_catalog()` through
//! `ReaderSnapshot::read_page`, so the catalog state observed is
//! frozen at txn-begin time and a collection created post-snapshot
//! is invisible to that reader.

#![forbid(unsafe_code)]

use std::sync::mpsc;
use std::thread;

use obj::{Db, Document, Error};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Alpha {
    payload: u64,
}
impl Document for Alpha {
    const COLLECTION: &'static str = "alpha";
    const VERSION: u32 = 1;
}
impl obj::Schema for Alpha {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("payload", obj::DynamicSchema::U64)])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Beta {
    payload: u64,
}
impl Document for Beta {
    const COLLECTION: &'static str = "beta";
    const VERSION: u32 = 1;
}
impl obj::Schema for Beta {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("payload", obj::DynamicSchema::U64)])
    }
}

/// A reader started before the writer creates a new collection
/// must NOT observe that collection — `ReadTxn::collection::<Beta>`
/// returns `Error::CollectionNotFound`. Once the reader's snapshot
/// is dropped, a fresh `ReadTxn` DOES observe `Beta`.
///
/// Ordering is made deterministic via two channels: the writer
/// waits for the reader to confirm the snapshot is pinned before
/// creating `Beta`, and the reader waits for the writer to confirm
/// the `Beta` commit before issuing its read.
#[test]
fn reader_started_before_collection_creation_sees_collection_not_found() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("mvcc53.obj");
    let db = Db::open(&path).expect("open");

    let _id = db.insert(Alpha { payload: 7 }).expect("insert alpha");

    let (snap_tx, snap_rx) = mpsc::channel::<()>();
    let (commit_tx, commit_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let db_r = &db;
        let db_w = &db;

        s.spawn(move || {
            db_r.read_transaction(|tx| {
                snap_tx.send(()).expect("snap_tx");
                commit_rx.recv().expect("commit_rx");
                match tx.collection::<Beta>() {
                    Ok(_) => panic!(
                        "Beta must be invisible to a snapshot pinned \
                         before its creation (M6 #53 regression — \
                         reader walked writer's live catalog root)",
                    ),
                    Err(Error::CollectionNotFound { ref name }) => {
                        assert_eq!(name, "beta");
                    }
                    Err(other) => panic!(
                        "expected CollectionNotFound, got {other:?} \
                         (M6 #53 regression — reader walked writer's live \
                         catalog root)"
                    ),
                }
                Ok(())
            })
            .expect("read_transaction");
        });

        s.spawn(move || {
            snap_rx.recv().expect("snap_rx");
            let _id = db_w.insert(Beta { payload: 11 }).expect("insert beta");
            commit_tx.send(()).expect("commit_tx");
        });
    });

    let beta: Option<Beta> = db
        .read_transaction(|tx| tx.collection::<Beta>()?.all().map(|v| v.into_iter().next()))
        .expect("read beta")
        .map(|(_id, doc)| doc);
    let beta = beta.expect("Beta visible to a fresh reader");
    assert_eq!(beta.payload, 11);
}

/// On a memory pager the snapshot's frozen view is empty and the
/// pager has no WAL — but a memory pager has no MVCC surface, so a
/// `ReadTxn::collection` against an existing collection must still
/// work (the snapshot path falls through to the live cache).
#[test]
fn memory_pager_read_after_insert_round_trips() {
    let db = Db::memory().expect("memory");
    let id = db.insert(Alpha { payload: 42 }).expect("insert");
    let back: Option<Alpha> = db.get::<Alpha>(id).expect("get");
    let back = back.expect("present");
    assert_eq!(back.payload, 42);
}

/// Sanity check the negative case on a memory pager: a never-
/// registered collection produces `CollectionNotFound`, NOT a
/// corruption surfaced from the snapshot path.
#[test]
fn memory_pager_missing_collection_is_not_found() {
    let db = Db::memory().expect("memory");
    let err = db
        .read_transaction(|tx| {
            let _ = tx.collection::<Beta>()?;
            Ok(())
        })
        .expect_err("absent collection");
    match err {
        Error::CollectionNotFound { ref name } => assert_eq!(name, "beta"),
        other => panic!("expected CollectionNotFound, got {other:?}"),
    }
}
