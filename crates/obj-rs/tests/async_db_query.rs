//! Extended async db and query coverage tests.
//!
//! Exercises the async surface not already reached by `async_tokio.rs`,
//! `async_smol.rs`, and `async_collection.rs`:
//!
//! **`AsyncDb`:**
//! - `open` / `open_with` / `memory_with` / `open_readonly`
//! - `update` / `delete` / `upsert` / `find_unique`
//! - `backup_to` / `stat` / `integrity_check`
//! - `as_blocking`
//! - `attach` while Arc is shared → `Error::Busy`
//!
//! **`AsyncQuery`:**
//! - `sort_by` / `sort_by_bytes` / `sort_buffer_limit`
//! - `index_range` happy path + empty window + chained filter
//! - `count` with filter, with limit, and with `index_range`
//! - `Debug` formatting (structural smoke-test)
//! - `sort_by` with embedded-NUL → `Error::SortKeyEncode`
//! - `sort_buffer_limit` overflow → `Error::SortBufferExceeded`
//! - `count` on empty collection → `Error::CollectionNotFound`

#![cfg(feature = "async")]

use std::ops::Bound;

use obj::asynchronous::AsyncDb;
use obj::{Config, Document, Dynamic, IndexSpec};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Test document: Order (with a Standard index on `placed_at`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    status: String,
    placed_at: u64,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::standard("placed_at", "placed_at").expect("standard")]
    }
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("status", obj::DynamicSchema::String),
            ("placed_at", obj::DynamicSchema::U64),
        ])
    }
}

// ---------------------------------------------------------------------------
// Test document: Widget (Unique index on `name`, for `find_unique`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Widget {
    name: String,
    weight_g: u64,
}

impl Document for Widget {
    const COLLECTION: &'static str = "widgets";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_name", "name").expect("unique")]
    }
}

impl obj::Schema for Widget {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("weight_g", obj::DynamicSchema::U64),
        ])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Seed `n` orders into `db` asynchronously (each is a separate call).
async fn seed_orders(db: &AsyncDb, n: u64) -> obj::Result<()> {
    for i in 0..n {
        let _ = db
            .insert(Order {
                customer_id: i,
                status: if i % 2 == 0 {
                    "pending".to_owned()
                } else {
                    "shipped".to_owned()
                },
                placed_at: i,
            })
            .await?;
    }
    Ok(())
}

// ===========================================================================
// AsyncDb: constructors
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn open_with_default_config_round_trip() -> obj::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("app.obj");
    let db = AsyncDb::open_with(&path, Config::default()).await?;
    let id = db
        .insert(Order {
            customer_id: 1,
            status: "pending".to_owned(),
            placed_at: 0,
        })
        .await?;
    let back: Option<Order> = db.get(id).await?;
    assert!(back.is_some(), "document should survive open_with round-trip");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_with_default_config_round_trip() -> obj::Result<()> {
    let db = AsyncDb::memory_with(Config::default()).await?;
    let id = db
        .insert(Order {
            customer_id: 2,
            status: "shipped".to_owned(),
            placed_at: 99,
        })
        .await?;
    let back: Option<Order> = db.get(id).await?;
    assert_eq!(back.map(|o| o.customer_id), Some(2));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn open_readonly_rejects_insert() -> obj::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ro.obj");
    // Create + populate via a read-write db first.
    {
        let rw = AsyncDb::open(&path).await?;
        let _ = rw
            .insert(Order {
                customer_id: 3,
                status: "pending".to_owned(),
                placed_at: 1,
            })
            .await?;
    }
    // Re-open read-only; insert must fail.
    let ro = AsyncDb::open_readonly(&path).await?;
    let err = ro
        .insert(Order {
            customer_id: 4,
            status: "pending".to_owned(),
            placed_at: 2,
        })
        .await;
    assert!(err.is_err(), "insert on read-only db must return Err");
    Ok(())
}

// ===========================================================================
// AsyncDb: per-op CRUD paths
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn update_modifies_document_in_place() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 5,
            status: "pending".to_owned(),
            placed_at: 10,
        })
        .await?;
    db.update::<Order, _>(id, |o| {
        o.status = "shipped".to_owned();
    })
    .await?;
    let back: Option<Order> = db.get(id).await?;
    assert_eq!(back.as_ref().map(|o| o.status.as_str()), Some("shipped"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_removes_document_returns_true() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 6,
            status: "pending".to_owned(),
            placed_at: 20,
        })
        .await?;
    let removed = db.delete::<Order>(id).await?;
    assert!(removed, "delete must return true when document existed");
    let back: Option<Order> = db.get(id).await?;
    assert_eq!(back, None, "document must be absent after delete");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_absent_returns_false() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 7,
            status: "pending".to_owned(),
            placed_at: 30,
        })
        .await?;
    // Manufacture an id that was never inserted.
    let missing_raw = id.get().saturating_add(9999);
    let missing_id = obj::Id::try_new(missing_raw).expect("non-zero");
    let removed = db.delete::<Order>(missing_id).await?;
    assert!(!removed, "delete on absent id must return false");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn upsert_inserts_then_overwrites() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 8,
            status: "pending".to_owned(),
            placed_at: 40,
        })
        .await?;
    db.upsert(
        id,
        Order {
            customer_id: 8,
            status: "archived".to_owned(),
            placed_at: 40,
        },
    )
    .await?;
    let back: Option<Order> = db.get(id).await?;
    assert_eq!(
        back.as_ref().map(|o| o.status.as_str()),
        Some("archived"),
        "upsert must overwrite the existing document"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn find_unique_returns_document_by_index_key() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let _ = db
        .insert(Widget {
            name: "Sprocket".to_owned(),
            weight_g: 42,
        })
        .await?;
    let found: Option<Widget> = db.find_unique("by_name", "Sprocket".to_owned()).await?;
    assert_eq!(
        found,
        Some(Widget {
            name: "Sprocket".to_owned(),
            weight_g: 42
        }),
        "find_unique must locate the document by index key"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn find_unique_returns_none_for_absent_key() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let _ = db
        .insert(Widget {
            name: "Bolt".to_owned(),
            weight_g: 5,
        })
        .await?;
    let found: Option<Widget> = db.find_unique("by_name", "NoSuchWidget".to_owned()).await?;
    assert_eq!(found, None);
    Ok(())
}

// ===========================================================================
// AsyncDb: backup / stat / integrity_check / as_blocking
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn backup_to_produces_readable_copy() -> obj::Result<()> {
    let dir = tempfile::tempdir()?;
    let src = dir.path().join("src.obj");
    let dst = dir.path().join("dst.obj");
    let db = AsyncDb::open(&src).await?;
    let _ = db
        .insert(Order {
            customer_id: 9,
            status: "pending".to_owned(),
            placed_at: 50,
        })
        .await?;
    db.backup_to(&dst).await?;
    // Reopen the backup (blocking) and verify the doc is there.
    let backup_db = obj::Db::open(&dst)?;
    let all: Vec<Order> = backup_db.all::<Order>()?;
    assert_eq!(all.len(), 1, "backup must contain the one inserted order");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn stat_returns_collection_info() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 5).await?;
    let stat = db.stat().await?;
    let collections: Vec<&str> = stat.collections.iter().map(|c| c.name.as_str()).collect();
    assert!(
        collections.contains(&Order::COLLECTION),
        "stat must list the 'orders' collection; got {collections:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn integrity_check_passes_on_healthy_db() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 10).await?;
    let report = db.integrity_check().await?;
    assert!(
        report.failures.is_empty(),
        "integrity_check must pass on a freshly-written db; failures: {:?}",
        report.failures
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn as_blocking_returns_underlying_db() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let id = db
        .insert(Order {
            customer_id: 10,
            status: "pending".to_owned(),
            placed_at: 60,
        })
        .await?;
    // as_blocking gives a &Db borrow valid for this scope.
    let blocking = db.as_blocking();
    let back: Option<Order> = blocking.get(id)?;
    assert!(back.is_some(), "as_blocking must expose the same data");
    Ok(())
}

// ===========================================================================
// AsyncDb: attach while Arc is shared → Error::Busy
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn attach_while_arc_is_shared_returns_busy() {
    let dir = tempfile::tempdir().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let attached_path = dir.path().join("attached.obj");

    // Create the file that will be attached.
    obj::Db::open(&attached_path).expect("create attached db");

    let mut db = AsyncDb::open(&main_path).await.expect("open main");
    // Clone to share the Arc — attach must now fail.
    let _clone = db.clone();
    let err = db.attach(&attached_path, "ns").await;
    assert!(
        err.is_err(),
        "attach with a shared Arc must return Err(Busy)"
    );
}

// ===========================================================================
// AsyncQuery: sort_by / sort_by_bytes / sort_buffer_limit
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_sort_by_dynamic_ascending() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    // Insert in reverse order so an unsorted result would fail the check.
    for i in (0..10u64).rev() {
        let _ = db
            .insert(Order {
                customer_id: i,
                status: "pending".to_owned(),
                placed_at: i,
            })
            .await?;
    }
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .fetch()
        .await?;
    assert_eq!(sorted.len(), 10);
    for w in sorted.windows(2) {
        assert!(
            w[0].placed_at <= w[1].placed_at,
            "sort_by must produce ascending order; got {:?} then {:?}",
            w[0].placed_at,
            w[1].placed_at
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_sort_by_bytes_ascending() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    for i in (0..8u64).rev() {
        let _ = db
            .insert(Order {
                customer_id: i,
                status: "pending".to_owned(),
                placed_at: i,
            })
            .await?;
    }
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by_bytes(|o| o.placed_at.to_be_bytes().to_vec())
        .fetch()
        .await?;
    assert_eq!(sorted.len(), 8);
    for w in sorted.windows(2) {
        assert!(
            w[0].placed_at <= w[1].placed_at,
            "sort_by_bytes must produce ascending order"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_sort_buffer_limit_cap_fires() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 20).await?;
    let err = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .sort_buffer_limit(5)
        .fetch()
        .await
        .expect_err("buffer of 5 must overflow with 20 docs");
    assert!(
        matches!(err, obj::Error::SortBufferExceeded { limit: 5 }),
        "expected SortBufferExceeded{{ limit: 5 }}, got {err:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_sort_buffer_limit_raised_lets_through() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 20).await?;
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .sort_buffer_limit(100)
        .fetch()
        .await?;
    assert_eq!(sorted.len(), 20);
    Ok(())
}

// ===========================================================================
// AsyncQuery: index_range paths
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_index_range_happy_path() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 50).await?;
    let mid: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(10)),
                Bound::Excluded(Dynamic::U64(30)),
            ),
        )
        .fetch()
        .await?;
    assert_eq!(mid.len(), 20, "[10, 30) must match exactly 20 docs");
    assert!(mid.iter().all(|o| (10..30).contains(&o.placed_at)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_index_range_empty_window_is_empty() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 20).await?;
    let empty: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(1_000)),
                Bound::Excluded(Dynamic::U64(2_000)),
            ),
        )
        .fetch()
        .await?;
    assert!(empty.is_empty(), "out-of-range window must return empty vec");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_index_range_chained_filter() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 40).await?;
    // [10, 30) has 20 docs; half are "pending" (even placed_at).
    let pending: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(10)),
                Bound::Excluded(Dynamic::U64(30)),
            ),
        )
        .filter(|o| o.status == "pending")
        .fetch()
        .await?;
    assert_eq!(pending.len(), 10, "10 even docs in [10,30) match 'pending'");
    assert!(pending.iter().all(|o| o.status == "pending"));
    Ok(())
}

// ===========================================================================
// AsyncQuery: count variants
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_count_with_filter() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 10).await?;
    let n = db
        .query::<Order>()
        .filter(|o| o.status == "pending")
        .count()
        .await?;
    assert_eq!(n, 5, "5 even-indexed orders match 'pending'");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_count_with_limit() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 20).await?;
    let n = db.query::<Order>().limit(7).count().await?;
    assert_eq!(n, 7, "count with limit must respect the cap");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_count_with_index_range() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 100).await?;
    let n = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(30)),
                Bound::Excluded(Dynamic::U64(70)),
            ),
        )
        .count()
        .await?;
    assert_eq!(n, 40, "[30, 70) covers exactly 40 docs");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_count_on_empty_collection_is_collection_not_found() {
    let db = AsyncDb::memory().await.expect("memory db");
    let err = db
        .query::<Order>()
        .count()
        .await
        .expect_err("count on empty collection must return Err");
    assert!(
        matches!(err, obj::Error::CollectionNotFound { ref name } if name == Order::COLLECTION),
        "expected CollectionNotFound(orders), got {err:?}"
    );
}

// ===========================================================================
// AsyncQuery: error paths
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_sort_by_with_nul_string_returns_sort_key_encode_error() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    seed_orders(&db, 5).await?;
    let err = db
        .query::<Order>()
        .sort_by(|_o| Dynamic::String("has\0nul".to_owned()))
        .fetch()
        .await
        .expect_err("embedded NUL must produce SortKeyEncode");
    assert!(
        matches!(err, obj::Error::SortKeyEncode { .. }),
        "expected SortKeyEncode, got {err:?}"
    );
    Ok(())
}

// ===========================================================================
// AsyncQuery: Debug impl smoke-test
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
async fn query_debug_format_is_non_empty() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let q = db
        .query::<Order>()
        .filter(|o| o.placed_at > 0)
        .limit(10)
        .sort_by(|o| Dynamic::U64(o.placed_at));
    let debug_str = format!("{q:?}");
    assert!(!debug_str.is_empty(), "Debug format of AsyncQuery must not be empty");
    Ok(())
}
