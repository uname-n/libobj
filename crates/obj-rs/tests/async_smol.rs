//! Async surface integration test driven by
//! the smol runtime.
//!
//! Companion to `async_tokio.rs`. Both tests run against the same
//! [`obj::asynchronous::AsyncDb`] surface — proving the wrapper is
//! runtime-agnostic via the `blocking` crate.
//!
//! smol (unlike tokio) ships no `#[test]` attribute macro, so each
//! test is a plain `#[test]` that blocks the current thread on the
//! smol executor via [`smol::block_on`].

#![cfg(feature = "async")]

use obj::asynchronous::AsyncDb;
use obj::Document;
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

#[test]
fn open_memory_round_trip_under_smol() -> obj::Result<()> {
    smol::block_on(async {
        let db = AsyncDb::memory().await?;
        let id = db
            .insert(Order {
                customer_id: 7,
                total_cents: 700,
            })
            .await?;
        let back: Option<Order> = db.get(id).await?;
        assert_eq!(
            back,
            Some(Order {
                customer_id: 7,
                total_cents: 700
            })
        );
        Ok(())
    })
}

#[test]
fn transaction_under_smol() -> obj::Result<()> {
    smol::block_on(async {
        let db = AsyncDb::memory().await?;
        let id = db
            .transaction(|tx| {
                tx.collection::<Order>()?.insert(Order {
                    customer_id: 1,
                    total_cents: 50,
                })
            })
            .await?;
        let back = db.get::<Order>(id).await?;
        assert!(back.is_some());
        Ok(())
    })
}

#[test]
fn query_under_smol() -> obj::Result<()> {
    smol::block_on(async {
        let db = AsyncDb::memory().await?;
        for n in 0..5u64 {
            let _ = db
                .insert(Order {
                    customer_id: n,
                    total_cents: n * 10,
                })
                .await?;
        }
        let count = db.query::<Order>().count().await?;
        assert_eq!(count, 5);
        let docs: Vec<Order> = db
            .query::<Order>()
            .filter(|o| o.total_cents > 10)
            .fetch()
            .await?;
        for o in &docs {
            assert!(o.total_cents > 10);
        }
        Ok(())
    })
}
