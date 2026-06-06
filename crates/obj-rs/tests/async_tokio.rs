//! Async surface integration tests driven by
//! the Tokio multi-thread runtime.
//!
//! Mirrors the smoke surface against
//! [`obj::asynchronous::AsyncDb`]: open + insert + get + transaction +
//! `read_transaction` + query. The smol variant lives next door in
//! `async_smol.rs`; both prove the wrapper is runtime-agnostic via the
//! `blocking` crate.

#![cfg(feature = "async")]

use obj::asynchronous::AsyncDb;
use obj::{Document, Id};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Order {
    customer_id: u64,
    total_cents: u64,
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
        ])
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn open_memory_round_trip() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 1,
            total_cents: 999,
        })
        .await?;
    let back: Option<Order> = db.get(id).await?;
    assert_eq!(
        back,
        Some(Order {
            customer_id: 1,
            total_cents: 999
        })
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_inserts_via_join() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let db1 = db.clone();
    let db2 = db.clone();
    let db3 = db.clone();
    let (a, b, c) = tokio::join!(
        db1.insert(Order {
            customer_id: 1,
            total_cents: 100
        }),
        db2.insert(Order {
            customer_id: 2,
            total_cents: 200
        }),
        db3.insert(Order {
            customer_id: 3,
            total_cents: 300
        }),
    );
    let ids: Vec<Id> = vec![a?, b?, c?];
    for id in &ids {
        let back: Option<Order> = db.get(*id).await?;
        assert!(back.is_some(), "id {id:?} missing after concurrent insert");
    }
    let all: Vec<Order> = db.all().await?;
    assert_eq!(all.len(), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn transaction_and_read_transaction() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let ids: Vec<Id> = db
        .transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let id1 = coll.insert(Order {
                customer_id: 10,
                total_cents: 1_000,
            })?;
            let id2 = coll.insert(Order {
                customer_id: 11,
                total_cents: 2_000,
            })?;
            Ok(vec![id1, id2])
        })
        .await?;
    assert_eq!(ids.len(), 2);
    let total: u64 = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let mut acc: u64 = 0;
            for (_id, doc) in coll.all()? {
                acc = acc.saturating_add(doc.total_cents);
            }
            Ok(acc)
        })
        .await?;
    assert_eq!(total, 3_000);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_fetch_and_count() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    for n in 0..10u64 {
        let _ = db
            .insert(Order {
                customer_id: n,
                total_cents: n * 100,
            })
            .await?;
    }
    let total = db.query::<Order>().count().await?;
    assert_eq!(total, 10);
    let big: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.total_cents >= 500)
        .limit(3)
        .fetch()
        .await?;
    assert!(big.len() <= 3);
    for o in &big {
        assert!(o.total_cents >= 500, "filter not applied: {o:?}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn attach_detach_round_trip() -> obj::Result<()> {
    let dir = tempfile::tempdir()?;
    let archive_path = dir.path().join("archive.obj");
    {
        let archive = obj::Db::open(&archive_path)?;
        let _ = archive.insert(Order {
            customer_id: 1,
            total_cents: 999,
        })?;
    }
    let main_path = dir.path().join("main.obj");
    let mut db = AsyncDb::open(&main_path).await?;
    db.attach(&archive_path, "archive").await?;
    db.detach("archive").await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn from_blocking_round_trip() -> obj::Result<()> {
    let blocking_db = obj::Db::memory()?;
    let db = AsyncDb::from_blocking(blocking_db);
    let id = db
        .insert(Order {
            customer_id: 42,
            total_cents: 42,
        })
        .await?;
    assert!(db.get::<Order>(id).await?.is_some());
    Ok(())
}
