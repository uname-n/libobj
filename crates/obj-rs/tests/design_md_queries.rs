//! Querying examples, exit-gate integration.
//!
//! Each test mirrors one querying example — the exact text is cited in
//! the comment block above each test function. The acceptance
//! criterion is "compiles AND produces the expected result"; every
//! test asserts behaviour, not just compilation.
//!
//! Hand-impl `Document` is used. Types are kept minimal — `Timestamp`
//! is a thin newtype around `u64`.

#![forbid(unsafe_code)]

use obj::{Db, Document, IndexSpec};
use obj_core::codec::Dynamic;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Stand-in `Timestamp`; the concrete type is not pinned, so an opaque
/// newtype over `u64` (epoch milliseconds, conceptually) is sufficient
/// for the query examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct Timestamp(u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum OrderStatus {
    Pending,
    Shipped,
    Archived,
}

/// The `Order` type — fields trimmed to what the query examples
/// reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    status: OrderStatus,
    placed_at: Timestamp,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![
            IndexSpec::standard("placed_at", "placed_at").expect("placed_at standard"),
            IndexSpec::standard("customer_id", "customer_id").expect("customer_id standard"),
        ]
    }
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        use obj::{DynamicSchema, EnumVariantSchema};
        DynamicSchema::map([
            ("customer_id", DynamicSchema::U64),
            (
                "status",
                DynamicSchema::enumeration([
                    EnumVariantSchema::new(0, "Pending", DynamicSchema::Null),
                    EnumVariantSchema::new(1, "Shipped", DynamicSchema::Null),
                    EnumVariantSchema::new(2, "Archived", DynamicSchema::Null),
                ]),
            ),
            ("placed_at", DynamicSchema::U64),
        ])
    }
}

/// The `Customer` type — `email` is unique-indexed for the
/// `find_unique` example.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Customer {
    name: String,
    email: String,
}

impl Document for Customer {
    const COLLECTION: &'static str = "customers";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("email", "email").expect("email unique")]
    }
}

impl obj::Schema for Customer {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("email", obj::DynamicSchema::String),
        ])
    }
}

/// Hermetic file-backed `Db` for each test.
fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("design.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

/// Seed `n` orders. `customer_id = (i % 3)`, `placed_at = i * 1000`,
/// status alternates Pending / Shipped so every query has matches.
fn seed_orders(db: &Db, n: u64) {
    for i in 0..n {
        let _ = db
            .insert(Order {
                customer_id: i % 3,
                status: if i % 2 == 0 {
                    OrderStatus::Pending
                } else {
                    OrderStatus::Shipped
                },
                placed_at: Timestamp(i * 1_000),
            })
            .expect("seed");
    }
}

/// Querying example 1:
/// ```text
/// // All documents in a collection.
/// for order in db.all::<Order>()? {
///     let order = order?;
///     // ...
/// }
/// ```
///
/// `Db::all` returns an owned-`Vec<T>` shape. We assert the for-loop's
/// `Vec::into_iter()` behaviour against the expected set.
#[test]
fn all_orders_iterates_every_doc() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 20);
    let mut seen: Vec<Order> = Vec::new();
    for order in db.all::<Order>().expect("all") {
        seen.push(order);
    }
    assert_eq!(seen.len(), 20);
    assert!(seen.iter().any(|o| o.customer_id == 0));
    assert!(seen.iter().any(|o| o.customer_id == 1));
    assert!(seen.iter().any(|o| o.customer_id == 2));
}

/// Querying example 2:
/// ```text
/// // Filtered scan.
/// let pending: Vec<Order> = db
///     .query::<Order>()
///     .filter(|o| o.status == OrderStatus::Pending)
///     .sort_by(|o| o.placed_at)
///     .limit(50)
///     .fetch()?;
/// ```
///
/// We mirror the shape exactly; the `sort_by` closure returns a
/// `Dynamic` because the surface erases sort keys via the
/// order-preserving encoder.
#[test]
fn filtered_sorted_limited_scan_returns_top_n_pending() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 200);
    let pending: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.status == OrderStatus::Pending)
        .sort_by(|o| Dynamic::U64(o.placed_at.0))
        .limit(50)
        .fetch()
        .expect("fetch");
    assert_eq!(pending.len(), 50);
    assert!(pending.iter().all(|o| o.status == OrderStatus::Pending));
    for w in pending.windows(2) {
        assert!(w[0].placed_at <= w[1].placed_at);
    }
    assert_eq!(pending[0].placed_at, Timestamp(0));
}

/// Querying example 3:
/// ```text
/// // Index lookup — O(log n), no collection scan.
/// let customer: Option<Customer> =
///     db.find_unique::<Customer>("email", "ada@example.com")?;
/// ```
///
/// This exit-gate test verifies the `find_unique` example shape
/// against a real `Customer` row.
#[test]
fn find_unique_by_email_returns_the_customer() {
    let (db, _dir) = fresh_db();
    let _id = db
        .insert(Customer {
            name: "Ada Lovelace".to_owned(),
            email: "ada@example.com".to_owned(),
        })
        .expect("insert");
    let customer: Option<Customer> = db
        .find_unique::<Customer>("email", "ada@example.com")
        .expect("find_unique");
    let c = customer.expect("present");
    assert_eq!(c.name, "Ada Lovelace");
    assert_eq!(c.email, "ada@example.com");

    let missing: Option<Customer> = db
        .find_unique::<Customer>("email", "nobody@example.com")
        .expect("find_unique missing");
    assert!(missing.is_none());
}

/// Querying example 4:
/// ```text
/// // Range query on an indexed field.
/// let recent: Vec<Order> = db
///     .query::<Order>()
///     .index_range("placed_at", last_week..now)
///     .fetch()?;
/// ```
///
/// `last_week..now` is a half-open `Range<Timestamp>`-shape;
/// `Query::index_range` takes `impl DynamicRange`, so the bounds may
/// be bare `Dynamic` values (as here) or any scalar that converts
/// into one (`u64`, `&str`, …) — `40u64..60` works without wrapping.
#[test]
fn index_range_returns_recent_orders() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let last_week = Dynamic::U64(30_000);
    let now = Dynamic::U64(60_000);
    let recent: Vec<Order> = db
        .query::<Order>()
        .index_range("placed_at", last_week..now)
        .expect("index_range")
        .fetch()
        .expect("fetch");
    assert_eq!(recent.len(), 30);
    assert!(recent
        .iter()
        .all(|o| (30..60).contains(&(o.placed_at.0 / 1_000))));
}

/// Querying example 5:
/// ```text
/// // Count without materializing documents.
/// let n: u64 = db
///     .query::<Order>()
///     .filter(|o| o.customer_id == customer_id)
///     .count()?;
/// ```
#[test]
fn count_per_customer_id_matches_filtered_total() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 30);
    let customer_id: u64 = 1;
    let n: u64 = db
        .query::<Order>()
        .filter(move |o| o.customer_id == customer_id)
        .count()
        .expect("count");
    assert_eq!(n, 10);
}
