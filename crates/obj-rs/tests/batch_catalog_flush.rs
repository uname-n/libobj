//! Batch-aware catalog flush — coalesce the per-doc catalog COW
//! mutations into one `Catalog::update` per touched collection at
//! commit, while keeping the per-txn cached descriptor the SOLE
//! mid-txn source of truth.
//!
//! These tests assert the load-bearing invariants of the change:
//!
//! - **Same-txn duplicate unique key** (THE regression test): two
//!   docs sharing one `Unique` key inserted in ONE transaction must
//!   trip `UniqueConstraintViolation` and roll back to 0 docs. This
//!   proves the unique pre-check (`tree.get`) descends the cached,
//!   in-memory-advanced index root and therefore sees the first
//!   doc's eager index write inside the uncommitted txn.
//! - **Multi-collection single txn**: inserting into two collections
//!   in one txn commits BOTH descriptors; a rolled-back txn commits
//!   NEITHER (one flush per touched collection, all-or-nothing).
//! - **Rollback leaves no catalog side effects**: a rolled-back batch
//!   leaves `next_id` + the primary root unadvanced on reopen — the
//!   coalesced flush never ran, so the catalog is byte-for-byte the
//!   pre-txn state.
//! - **Read-after-write inside one txn**: a `get` / `all` / lookup on
//!   the same handle after inserts in the same txn observes those
//!   writes (the read path descends the cached live roots).

#![forbid(unsafe_code)]

use obj::{Db, Document, IndexSpec};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Account {
    key: String,
    note: String,
}

impl Document for Account {
    const COLLECTION: &'static str = "accounts";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_key", "key").expect("unique spec")]
    }
}

impl obj::Schema for Account {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("key", obj::DynamicSchema::String),
            ("note", obj::DynamicSchema::String),
        ])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Ledger {
    code: u64,
}

impl Document for Ledger {
    const COLLECTION: &'static str = "ledgers";
    const VERSION: u32 = 1;
}

impl obj::Schema for Ledger {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("code", obj::DynamicSchema::U64)])
    }
}

/// File-backed `Db` plus its owning `TempDir`. File backing is
/// required: the rollback / reopen assertions exercise the WAL unwind
/// path, which the in-memory pager does not have.
fn fresh_db(name: &str) -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join(format!("{name}.obj"));
    let db = Db::open(&path).expect("open");
    (db, dir)
}

fn account(key: &str, note: &str) -> Account {
    Account {
        key: key.to_owned(),
        note: note.to_owned(),
    }
}

/// THE regression test: two docs with the
/// SAME unique key in ONE transaction must error
/// `UniqueConstraintViolation` and roll back to 0 docs.
///
/// This only passes if the second insert's unique pre-check (`tree.get`
/// on the index B-tree) descends the CACHED, in-memory-advanced index
/// root — i.e. the root the FIRST insert advanced in the same txn. A
/// stale catalog-tree re-read would see the empty pre-txn index root,
/// miss the first entry, and let the duplicate through.
#[test]
fn same_txn_duplicate_unique_key_rolls_back_to_zero() {
    let (db, _dir) = fresh_db("same_txn_dup");
    let result: obj::Result<()> = db.transaction(|tx| {
        let coll = tx.collection::<Account>()?;
        let _ = coll.insert(account("alpha", "first"))?;
        let _ = coll.insert(account("alpha", "second"))?;
        Ok(())
    });
    match result {
        Err(obj::Error::UniqueConstraintViolation {
            index, collection, ..
        }) => {
            assert_eq!(index, "by_key");
            assert_eq!(collection, Account::COLLECTION);
        }
        other => panic!("expected UniqueConstraintViolation, got {other:?}"),
    }
    let count = db
        .read_transaction(|tx| {
            Ok(match tx.collection::<Account>() {
                Ok(c) => c.all()?.len(),
                Err(obj::Error::CollectionNotFound { .. }) => 0,
                Err(e) => return Err(e),
            })
        })
        .expect("count");
    assert_eq!(count, 0, "rolled-back batch must leave 0 docs");
}

/// Companion to the same-txn regression: the duplicate-key pre-check
/// must ALSO fire when the FIRST holder of the key was committed in a
/// PRIOR txn and the second insert lands in a fresh txn (a sanity
/// check that the cached descriptor is seeded from the committed
/// catalog row, not assumed-empty). The second insert rolls back,
/// leaving the original holder intact.
#[test]
fn duplicate_unique_key_across_txns_keeps_original() {
    let (db, _dir) = fresh_db("dup_across_txn");
    let _ = db.insert(account("alpha", "first")).expect("first commit");
    let err = db
        .transaction(|tx| {
            let coll = tx.collection::<Account>()?;
            coll.insert(account("alpha", "second")).map(|_| ())
        })
        .expect_err("dup across txns");
    assert!(matches!(err, obj::Error::UniqueConstraintViolation { .. }));
    let found = db
        .find_unique::<Account>("by_key", "alpha")
        .expect("find_unique")
        .expect("present");
    assert_eq!(found.note, "first");
    let count = db
        .read_transaction(|tx| Ok(tx.collection::<Account>()?.all()?.len()))
        .expect("count");
    assert_eq!(count, 1, "only the original committed doc remains");
}

/// A successful 64-doc batch with distinct unique keys commits once
/// and every doc + index entry is durable. Exercises the coalesced
/// `next_id` / `primary_root` / index-root advances across many docs
/// in one txn, flushed once at commit.
#[test]
fn batch_of_distinct_unique_keys_all_persist() {
    let (db, _dir) = fresh_db("batch_distinct");
    db.transaction(|tx| {
        let coll = tx.collection::<Account>()?;
        for i in 0..64u32 {
            let _ = coll.insert(account(&format!("k-{i}"), &format!("n-{i}")))?;
        }
        Ok(())
    })
    .expect("batch insert");
    let all = db
        .read_transaction(|tx| tx.collection::<Account>()?.all())
        .expect("all");
    assert_eq!(all.len(), 64, "every distinct-key doc must persist");
    for i in 0..64u32 {
        let got = db
            .find_unique::<Account>("by_key", format!("k-{i}"))
            .expect("find_unique")
            .expect("present");
        assert_eq!(got.note, format!("n-{i}"));
    }
}

/// Multi-collection single txn: inserting into two collections in one
/// transaction commits BOTH descriptors (one flush apiece).
#[test]
fn multi_collection_single_txn_commits_both() {
    let (db, _dir) = fresh_db("multi_commit");
    db.transaction(|tx| {
        let accounts = tx.collection::<Account>()?;
        let _ = accounts.insert(account("a1", "n1"))?;
        let _ = accounts.insert(account("a2", "n2"))?;
        let ledgers = tx.collection::<Ledger>()?;
        let _ = ledgers.insert(Ledger { code: 100 })?;
        let _ = ledgers.insert(Ledger { code: 200 })?;
        let _ = ledgers.insert(Ledger { code: 300 })?;
        Ok(())
    })
    .expect("multi-collection txn");
    let n_accounts = db
        .read_transaction(|tx| Ok(tx.collection::<Account>()?.all()?.len()))
        .expect("accounts");
    let n_ledgers = db
        .read_transaction(|tx| Ok(tx.collection::<Ledger>()?.all()?.len()))
        .expect("ledgers");
    assert_eq!(n_accounts, 2, "both account descriptors flushed");
    assert_eq!(n_ledgers, 3, "both ledger descriptors flushed");
}

/// Multi-collection single txn rollback: a closure that errors after
/// writing to BOTH collections must commit NEITHER descriptor.
#[test]
fn multi_collection_single_txn_rollback_commits_neither() {
    let (db, _dir) = fresh_db("multi_rollback");
    let result: obj::Result<()> = db.transaction(|tx| {
        let accounts = tx.collection::<Account>()?;
        let _ = accounts.insert(account("a1", "n1"))?;
        let ledgers = tx.collection::<Ledger>()?;
        let _ = ledgers.insert(Ledger { code: 100 })?;
        Err(obj::Error::InvalidArgument("synthetic rollback"))
    });
    assert!(matches!(result, Err(obj::Error::InvalidArgument(_))));
    let accounts_after = db.read_transaction(|tx| {
        Ok(match tx.collection::<Account>() {
            Ok(c) => c.all()?.len(),
            Err(obj::Error::CollectionNotFound { .. }) => 0,
            Err(e) => return Err(e),
        })
    });
    let ledgers_after = db.read_transaction(|tx| {
        Ok(match tx.collection::<Ledger>() {
            Ok(c) => c.all()?.len(),
            Err(obj::Error::CollectionNotFound { .. }) => 0,
            Err(e) => return Err(e),
        })
    });
    assert_eq!(accounts_after.expect("accounts"), 0, "no account persisted");
    assert_eq!(ledgers_after.expect("ledgers"), 0, "no ledger persisted");
}

/// Rollback leaves no catalog side effects: after a collection is
/// seeded (so it exists), a rolled-back batch must leave `next_id` and
/// the primary root UNADVANCED on reopen — the coalesced descriptor
/// flush never ran.
#[test]
fn rollback_leaves_next_id_and_roots_unadvanced_on_reopen() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("rollback_reopen.obj");
    let baseline_next_id;
    let baseline_root;
    {
        let db = Db::open(&path).expect("open");
        let _ = db.insert(account("seed", "seed-note")).expect("seed");
        let (next_id, root) = db
            .read_transaction(|tx| {
                let d = tx.collection::<Account>()?.descriptor().clone();
                Ok((d.next_id, d.primary_root))
            })
            .expect("descriptor snapshot");
        baseline_next_id = next_id;
        baseline_root = root;
        let result: obj::Result<()> = db.transaction(|tx| {
            let coll = tx.collection::<Account>()?;
            for i in 0..16u32 {
                let _ = coll.insert(account(&format!("r-{i}"), "x"))?;
            }
            Err(obj::Error::InvalidArgument("synthetic rollback"))
        });
        assert!(matches!(result, Err(obj::Error::InvalidArgument(_))));
    }
    let db = Db::open(&path).expect("reopen");
    let (next_id, root, count) = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Account>()?;
            let d = coll.descriptor().clone();
            Ok((d.next_id, d.primary_root, coll.all()?.len()))
        })
        .expect("reopen snapshot");
    assert_eq!(count, 1, "only the committed seed survives reopen");
    assert_eq!(
        next_id, baseline_next_id,
        "next_id must be unadvanced by the rolled-back batch",
    );
    assert_eq!(
        root, baseline_root,
        "primary_root must be unadvanced by the rolled-back batch",
    );
}

/// Read-after-write inside one txn: a `get` / `all` / `find_unique` on
/// the same handle after inserts in the same txn observes those
/// uncommitted writes — proving the write-side read paths descend the
/// cached live roots, not the stale open-time descriptor.
#[test]
fn read_after_write_in_same_txn_sees_uncommitted_writes() {
    let (db, _dir) = fresh_db("read_after_write");
    db.transaction(|tx| {
        let coll = tx.collection::<Account>()?;
        let id1 = coll.insert(account("x1", "v1"))?;
        let id2 = coll.insert(account("x2", "v2"))?;
        let g1 = coll.get(id1)?.expect("id1 present mid-txn");
        let g2 = coll.get(id2)?.expect("id2 present mid-txn");
        assert_eq!(g1.note, "v1");
        assert_eq!(g2.note, "v2");
        assert_eq!(coll.all()?.len(), 2, "all() sees both mid-txn inserts");
        let viaidx = coll.find_unique("by_key", "x2")?.expect("idx lookup");
        assert_eq!(viaidx.note, "v2");
        Ok(())
    })
    .expect("read-after-write txn");
}
