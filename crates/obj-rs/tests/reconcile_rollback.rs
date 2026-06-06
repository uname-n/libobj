//! The per-process `reconciled` index-cache must be
//! TRANSACTIONAL — a collection's first-ever (lazy-create) txn that
//! ROLLS BACK must NOT poison the shared cache into skipping index
//! reconciliation on a later, committed txn in the same process.
//!
//! Before the fix, `reconcile_indexes_once` inserted the collection
//! name into the shared `reconciled` set DURING the txn. A rolled-back
//! first-ever txn unwound the catalog's index rows but left the name
//! in the set, so the next insert in the same process saw the poisoned
//! cache hit, skipped reconciliation, and wrote against a catalog with
//! NO index descriptors — the index was silently missing (writes
//! unindexed, `find_unique` misses).
//!
//! These tests drive only `obj::Db`'s public API; the index's presence
//! is observed through its visible behaviour (a `find_unique` hit on
//! the indexed field) plus a clean `integrity_check`.

#![forbid(unsafe_code)]

use obj::{Db, Document, IndexSpec};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    email: String,
    name: String,
}

impl Document for User {
    const COLLECTION: &'static str = "users";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_email", "email").expect("unique")]
    }
}

impl obj::Schema for User {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("email", obj::DynamicSchema::String),
            ("name", obj::DynamicSchema::String),
        ])
    }
}

/// File-backed `Db` (the rollback path needs a WAL to unwind) plus the
/// owning `TempDir` keepalive.
fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("reconcile-rollback.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

/// A collection's FIRST-EVER txn in the
/// process lazily creates the collection (running index
/// reconciliation) and then rolls back. A subsequent committed txn
/// must re-reconcile so the index is present + maintained.
///
/// Without the fix this fails: the rolled-back txn poisons the shared
/// `reconciled` set, the second insert skips reconciliation, the index
/// is never created, and `find_unique` returns `None`.
#[test]
fn rolled_back_lazy_create_does_not_poison_index_reconciliation() {
    let (db, _dir) = fresh_db();

    let rolled_back = db.transaction(|tx| {
        let users = tx.collection::<User>()?;
        let _ = users.insert(User {
            email: "ada@example.com".to_owned(),
            name: "Ada".to_owned(),
        })?;
        Err::<(), obj::Error>(obj::Error::InvalidArgument("force rollback"))
    });
    assert!(rolled_back.is_err(), "txn must have rolled back");

    let after_rollback = db.find_unique::<User>("by_email", "ada@example.com");
    match after_rollback {
        Ok(None) | Err(obj::Error::CollectionNotFound { .. }) => {}
        other => panic!("rolled-back lazy-create must leave no document, got {other:?}"),
    }

    let id = db
        .insert(User {
            email: "grace@example.com".to_owned(),
            name: "Grace".to_owned(),
        })
        .expect("second insert");

    let found: Option<User> = db
        .find_unique::<User>("by_email", "grace@example.com")
        .expect("find_unique");
    let found = found.expect("indexed doc must be found via by_email");
    assert_eq!(found.email, "grace@example.com");
    assert_eq!(found.name, "Grace");

    let dup = db.insert(User {
        email: "grace@example.com".to_owned(),
        name: "Imposter".to_owned(),
    });
    assert!(
        matches!(dup, Err(obj::Error::UniqueConstraintViolation { .. })),
        "unique index must be active and reject the duplicate, got {dup:?}"
    );

    let report = db.integrity_check().expect("integrity_check");
    assert!(report.is_ok(), "integrity check must pass: {report:?}");

    let _ = id;
}

/// The common (commit) path still reconciles EXACTLY once and the
/// index works end-to-end. A second open of the same collection in a
/// later txn hits the shared cache (no double-reconcile, no error) and
/// the index continues to function.
#[test]
fn committed_lazy_create_reconciles_and_repeat_open_is_a_cache_hit() {
    let (db, _dir) = fresh_db();

    db.insert(User {
        email: "first@example.com".to_owned(),
        name: "First".to_owned(),
    })
    .expect("first insert (commit)");

    db.insert(User {
        email: "second@example.com".to_owned(),
        name: "Second".to_owned(),
    })
    .expect("second insert (cache hit)");

    let a: Option<User> = db
        .find_unique::<User>("by_email", "first@example.com")
        .expect("find first");
    let b: Option<User> = db
        .find_unique::<User>("by_email", "second@example.com")
        .expect("find second");
    assert!(a.is_some(), "first doc must be indexed");
    assert!(b.is_some(), "second doc must be indexed");

    db.transaction(|tx| {
        let _first_handle = tx.collection::<User>()?;
        let second_handle = tx.collection::<User>()?;
        let _ = second_handle.insert(User {
            email: "third@example.com".to_owned(),
            name: "Third".to_owned(),
        })?;
        Ok::<(), obj::Error>(())
    })
    .expect("repeat-open in one txn");

    let report = db.integrity_check().expect("integrity_check");
    assert!(report.is_ok(), "integrity check must pass: {report:?}");
}
