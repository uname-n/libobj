//! `obj::Db` integration test.
//!
//! Covers:
//! - opening + per-op API (`insert`, `get`, `update`, `delete`,
//!   `upsert`),
//! - explicit `transaction` closures,
//! - rollback on `Err` return,
//! - `read_transaction` snapshot isolation,
//! - reopen-then-read durability,
//! - `Db::open_readonly` rejects mutating ops,
//! - `Db::memory` provides an ephemeral handle.

use obj::{Db, Document, Error};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    total_cents: u64,
    status: String,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("total_cents", obj::DynamicSchema::U64),
            ("status", obj::DynamicSchema::String),
        ])
    }
}

#[test]
fn insert_get_update_delete_upsert_smoke() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("app.obj");
    let db = Db::open(&path).expect("open");

    let id = db
        .insert(Order {
            customer_id: 1,
            total_cents: 100,
            status: "pending".to_owned(),
        })
        .expect("insert");

    let back: Option<Order> = db.get(id).expect("get");
    assert_eq!(
        back.expect("present").total_cents,
        100,
        "round-trip total preserved",
    );

    db.update::<Order, _>(id, |o| {
        o.status = "shipped".to_owned();
    })
    .expect("update");

    let updated: Order = db.get::<Order>(id).expect("get").expect("present");
    assert_eq!(updated.status, "shipped");

    let existed = db.delete::<Order>(id).expect("delete");
    assert!(existed);
    let missing: Option<Order> = db.get(id).expect("get after delete");
    assert!(missing.is_none(), "delete must remove the row");

    let id2 = obj::Id::try_new(424_242).expect("non-zero");
    db.upsert::<Order>(
        id2,
        Order {
            customer_id: 7,
            total_cents: 999,
            status: "new".to_owned(),
        },
    )
    .expect("upsert");
    let upserted: Order = db.get::<Order>(id2).expect("get").expect("present");
    assert_eq!(upserted.total_cents, 999);
}

#[test]
fn explicit_transaction_commit() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("tx.obj");
    let db = Db::open(&path).expect("open");

    let ids = db
        .transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let a = coll.insert(Order {
                customer_id: 1,
                total_cents: 50,
                status: "pending".to_owned(),
            })?;
            let b = coll.insert(Order {
                customer_id: 2,
                total_cents: 200,
                status: "pending".to_owned(),
            })?;
            Ok((a, b))
        })
        .expect("transaction");

    let a = db.get::<Order>(ids.0).expect("get a").expect("present");
    let b = db.get::<Order>(ids.1).expect("get b").expect("present");
    assert_eq!(a.total_cents, 50);
    assert_eq!(b.total_cents, 200);
}

#[test]
fn explicit_transaction_rollback_on_err() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("rb.obj");
    let db = Db::open(&path).expect("open");

    let id = db
        .insert(Order {
            customer_id: 1,
            total_cents: 10,
            status: "pending".to_owned(),
        })
        .expect("seed");

    let result: obj::Result<()> = db.transaction(|tx| {
        let coll = tx.collection::<Order>()?;
        coll.update(id, |o: &mut Order| {
            o.total_cents = 99_999;
        })?;
        Err(Error::InvalidArgument("synthetic"))
    });
    assert!(matches!(result, Err(Error::InvalidArgument(_))));

    let after = db.get::<Order>(id).expect("get").expect("present");
    assert_eq!(
        after.total_cents, 10,
        "rolled-back update must not be visible",
    );
}

#[test]
fn read_transaction_sees_consistent_snapshot() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("rt.obj");
    let db = Db::open(&path).expect("open");
    let id = db
        .insert(Order {
            customer_id: 1,
            total_cents: 10,
            status: "pending".to_owned(),
        })
        .expect("insert");

    let pair: (Option<Order>, Option<Order>) = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let a = coll.get(id)?;
            let b = coll.get(id)?;
            Ok((a, b))
        })
        .expect("read txn");
    assert_eq!(pair.0, pair.1);
}

#[test]
fn reopen_observes_committed_data() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("persist.obj");
    let id = {
        let db = Db::open(&path).expect("open");
        db.insert(Order {
            customer_id: 99,
            total_cents: 1234,
            status: "pending".to_owned(),
        })
        .expect("insert")
    };
    let db = Db::open(&path).expect("reopen");
    let back: Order = db.get::<Order>(id).expect("get").expect("present");
    assert_eq!(back.total_cents, 1234);
}

#[test]
fn open_readonly_rejects_mutations() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("ro.obj");
    {
        let db = Db::open(&path).expect("open writer");
        let _ = db
            .insert(Order {
                customer_id: 1,
                total_cents: 1,
                status: "pending".to_owned(),
            })
            .expect("insert");
    }
    let db = Db::open_readonly(&path).expect("open ro");
    let err = db
        .insert::<Order>(Order {
            customer_id: 2,
            total_cents: 2,
            status: "pending".to_owned(),
        })
        .expect_err("readonly insert must fail");
    assert!(
        matches!(
            err,
            Error::ReadOnly {
                operation: "transaction"
            }
        ),
        "unexpected: {err:?}",
    );
}

#[test]
fn memory_db_is_ephemeral_but_works() {
    let db = Db::memory().expect("memory");
    let id = db
        .insert(Order {
            customer_id: 1,
            total_cents: 5,
            status: "pending".to_owned(),
        })
        .expect("insert");
    let back: Order = db.get::<Order>(id).expect("get").expect("present");
    assert_eq!(back.total_cents, 5);
}

#[test]
fn collection_all_lists_inserted_docs() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("all.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..5 {
        let _ = db
            .insert(Order {
                customer_id: i,
                total_cents: i * 100,
                status: "pending".to_owned(),
            })
            .expect("insert");
    }
    let listed: Vec<(obj::Id, Order)> = db
        .read_transaction(|tx| tx.collection::<Order>()?.all())
        .expect("all");
    assert_eq!(listed.len(), 5);
    let totals: Vec<u64> = listed.iter().map(|(_, o)| o.total_cents).collect();
    assert!(totals.contains(&0));
    assert!(totals.contains(&400));
}

#[test]
fn read_transaction_missing_collection_errors() {
    let db = Db::memory().expect("memory");
    let err = db
        .read_transaction(|tx| {
            let _ = tx.collection::<Order>()?;
            Ok(())
        })
        .expect_err("absent collection on read side must err");
    assert!(matches!(err, Error::CollectionNotFound { ref name } if name == "orders"));
}
