//! `Query<T>` builder integration tests.
//!
//! Each acceptance criterion gets a happy-path test and
//! an empty-result test. Later commits extend this file:
//!
//! - `Query::sort_by` with bounded sort buffer.
//! - `Query::count` no-decode fast path.
//!
//! The hand-impl `Document` for `Order` lives here. Indexes are
//! declared via the [`obj::Document::indexes`] override so
//! `.index_range` has something to walk.

#![forbid(unsafe_code)]

use std::ops::Bound;

use obj::{Db, Document, IndexSpec};
use obj_core::codec::Dynamic;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Hand-written `Document` for the `Order` example.
/// Carries one indexed field (`placed_at`, `Standard` kind) so the
/// `.index_range` path has something to walk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    /// `Id` field referencing the customer record.
    customer_id: u64,
    /// Order status (pending/shipped/...).
    status: String,
    /// Timestamp the order was placed. Indexed.
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

/// Hermetic file-backed `Db` plus the owning `TempDir`. The temp
/// dir lives as long as the returned tuple to keep the file alive.
fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("query.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

/// Seed `n` orders. `placed_at = i` so the index range is dense
/// from 0..n. `status = "pending"` for even `i`, `"shipped"` for odd
/// — so filters have a 50/50 split to exercise.
fn seed_orders(db: &Db, n: u64) {
    for i in 0..n {
        let _ = db
            .insert(Order {
                customer_id: i,
                status: if i % 2 == 0 { "pending" } else { "shipped" }.to_owned(),
                placed_at: i,
            })
            .expect("seed insert");
    }
}

#[test]
fn db_all_returns_every_doc() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 5);
    let all: Vec<Order> = db.all::<Order>().expect("all");
    assert_eq!(all.len(), 5, "Db::all must return every inserted doc");
}

#[test]
fn db_all_on_empty_collection_is_collection_not_found() {
    let (db, _dir) = fresh_db();
    let err = db.all::<Order>().expect_err("all on absent collection");
    assert!(
        matches!(err, obj::Error::CollectionNotFound { ref name } if name == "orders"),
        "expected CollectionNotFound, got {err:?}",
    );
}

#[test]
fn query_filter_returns_matching_subset() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 10);
    let pending: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.status == "pending")
        .fetch()
        .expect("filter fetch");
    assert_eq!(pending.len(), 5, "5 even-indexed docs match 'pending'");
    assert!(pending.iter().all(|o| o.status == "pending"));
}

#[test]
fn query_filter_empty_result_is_empty_vec() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 4);
    let none: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.status == "archived")
        .fetch()
        .expect("filter fetch");
    assert!(none.is_empty(), "no doc has status 'archived'");
}

#[test]
fn query_multiple_filters_compose_with_and() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 20);
    let hits: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.status == "pending")
        .filter(|o| o.placed_at >= 10)
        .fetch()
        .expect("multi-filter fetch");
    assert_eq!(hits.len(), 5);
    assert!(hits
        .iter()
        .all(|o| o.status == "pending" && o.placed_at >= 10));
}

#[test]
fn query_limit_caps_result_set() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let first_ten: Vec<Order> = db.query::<Order>().limit(10).fetch().expect("limit fetch");
    assert_eq!(first_ten.len(), 10);
}

#[test]
fn query_limit_zero_is_empty() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 5);
    let none: Vec<Order> = db.query::<Order>().limit(0).fetch().expect("limit 0 fetch");
    assert!(none.is_empty());
}

#[test]
fn query_index_range_walks_index_slice() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let mid: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(40)),
                Bound::Excluded(Dynamic::U64(60)),
            ),
        )        .fetch()
        .expect("index_range fetch");
    assert_eq!(mid.len(), 20);
    assert!(mid.iter().all(|o| (40..60).contains(&o.placed_at)));
}

/// A scalar-typed range (`40u64..60`) must yield byte-for-byte the
/// same documents as the equivalent explicitly-`Dynamic`-wrapped
/// bound tuple — the ergonomic form is pure sugar over the same
/// encoding. Exercises the sync query path, the `Collection`
/// range/count APIs, and the `..=` / open-ended forms.
#[test]
fn query_index_range_scalar_matches_dynamic_wrapped() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);

    let wrapped: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(40)),
                Bound::Excluded(Dynamic::U64(60)),
            ),
        )
        .fetch()
        .expect("wrapped fetch");

    let scalar: Vec<Order> = db
        .query::<Order>()
        .index_range("placed_at", 40u64..60)
        .fetch()
        .expect("scalar fetch");

    assert_eq!(scalar, wrapped, "scalar range must equal Dynamic-wrapped");
    assert_eq!(scalar.len(), 20);

    // Inclusive scalar form covers one extra row at the top end.
    let inclusive: Vec<Order> = db
        .query::<Order>()
        .index_range("placed_at", 40u64..=60)
        .fetch()
        .expect("inclusive fetch");
    assert_eq!(inclusive.len(), 21);

    // Collection-level APIs accept the same scalar range.
    let entries = db
        .read_transaction(|txn| {
            let coll = txn.collection::<Order>()?;
            coll.count_index_range("placed_at", 40u64..60)
        })
        .expect("count_index_range");
    assert_eq!(entries, 20);
}

#[test]
fn query_index_range_empty_window_is_empty() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 50);
    let empty: Vec<Order> = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(1_000)),
                Bound::Excluded(Dynamic::U64(2_000)),
            ),
        )        .fetch()
        .expect("index_range fetch");
    assert!(empty.is_empty());
}

#[test]
fn query_index_range_with_filter() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 50);
    let hits: Vec<Order> = db
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
        .expect("fetch");
    assert_eq!(hits.len(), 10);
}

/// `Query::index_range` returns `Self`, not `Result<Self>`, so the
/// fluent chain needs no mid-chain `?` / `.expect(...)` — it composes
/// with `.filter(...)` and `.fetch()` exactly like every other builder
/// step. This is the ergonomic win issue #7 delivers.
#[test]
fn index_range_is_infallible_and_chains_fluently() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let mid: Vec<Order> = db
        .query::<Order>()
        .index_range("placed_at", 40u64..60)
        .filter(|o| o.placed_at % 2 == 0)
        .fetch()
        .expect("fetch");
    assert_eq!(mid.len(), 10, "even placed_at in [40, 60) is 40,42,..,58");
}

/// Deferred-error path: an unencodable bound (a `Dynamic::String`
/// carrying an embedded NUL, which the order-preserving encoder
/// rejects) does NOT fail at the infallible `.index_range(...)` step.
/// The encode runs at terminal time, so the error surfaces from
/// `.fetch()`.
#[test]
fn index_range_unencodable_bound_surfaces_err_from_fetch() {
    let (db, _dir) = fresh_db();
    for _ in 0..3 {
        let _ = db
            .insert(Ticket {
                status: "urgent".to_owned(),
            })
            .expect("insert");
    }
    // The builder step itself succeeds — no `?` here.
    let q = db.query::<Ticket>().index_range(
        "by_status",
        (
            Bound::Included(Dynamic::String("has\0nul".to_owned())),
            Bound::Unbounded,
        ),
    );
    let err = q
        .fetch()
        .expect_err("embedded NUL must be rejected at fetch time");
    assert!(
        matches!(err, obj::Error::InvalidArgument(_)),
        "expected InvalidArgument from deferred encode, got {err:?}",
    );
}

/// Same deferred-error contract for the `.count()` terminal: the
/// encode error fires from `count()`, not from `index_range`.
#[test]
fn index_range_unencodable_bound_surfaces_err_from_count() {
    let (db, _dir) = fresh_db();
    for _ in 0..3 {
        let _ = db
            .insert(Ticket {
                status: "urgent".to_owned(),
            })
            .expect("insert");
    }
    let err = db
        .query::<Ticket>()
        .index_range(
            "by_status",
            (
                Bound::Included(Dynamic::String("has\0nul".to_owned())),
                Bound::Unbounded,
            ),
        )
        .count()
        .expect_err("embedded NUL must be rejected at count time");
    assert!(
        matches!(err, obj::Error::InvalidArgument(_)),
        "expected InvalidArgument from deferred encode, got {err:?}",
    );
}

/// Seed docs whose `placed_at` is the REVERSE of insertion order so
/// the index/sort comparison is non-trivial. Returns the populated
/// vector for the assertion side.
fn seed_reversed(db: &Db, n: u64) {
    for i in 0..n {
        let _ = db
            .insert(Order {
                customer_id: i,
                status: "pending".to_owned(),
                placed_at: n - i - 1,
            })
            .expect("seed insert");
    }
}

#[test]
fn query_sort_by_int_field_ascends() {
    let (db, _dir) = fresh_db();
    seed_reversed(&db, 100);
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .fetch()
        .expect("sort fetch");
    assert_eq!(sorted.len(), 100);
    for w in sorted.windows(2) {
        assert!(
            w[0].placed_at <= w[1].placed_at,
            "ascending sort_by violated at {:?} / {:?}",
            w[0].placed_at,
            w[1].placed_at,
        );
    }
    assert_eq!(sorted[0].placed_at, 0);
    assert_eq!(sorted[99].placed_at, 99);
}

#[test]
fn query_sort_by_string_field_ascends() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 10);
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::String(o.status.clone()))
        .fetch()
        .expect("sort fetch");
    assert_eq!(sorted.len(), 10);
    for (i, doc) in sorted.iter().enumerate() {
        let expected = if i < 5 { "pending" } else { "shipped" };
        assert_eq!(doc.status, expected);
    }
}

#[test]
fn query_filter_sort_by_limit_returns_top_n_sorted() {
    let (db, _dir) = fresh_db();
    seed_reversed(&db, 100);
    let top: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.placed_at >= 50)
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .limit(10)
        .fetch()
        .expect("fetch");
    assert_eq!(top.len(), 10);
    for (i, doc) in top.iter().enumerate() {
        assert_eq!(doc.placed_at, 50 + i as u64);
    }
}

#[test]
fn query_sort_buffer_exceeded_fires_below_cap_then_passes_above() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 250);

    let err = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .sort_buffer_limit(100)
        .fetch()
        .expect_err("explicit cap of 100 must overflow at 250 docs");
    assert!(
        matches!(err, obj::Error::SortBufferExceeded { limit: 100 }),
        "expected SortBufferExceeded{{ limit: 100 }}, got {err:?}",
    );

    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .sort_buffer_limit(300)
        .fetch()
        .expect("raised cap fetch");
    assert_eq!(sorted.len(), 250);
    for (i, doc) in sorted.iter().enumerate() {
        assert_eq!(doc.placed_at, i as u64);
    }
}

#[test]
fn query_sort_buffer_default_constant_is_one_hundred_thousand() {
    assert_eq!(obj::MAX_SORT_BUFFER, 100_000);
}

#[test]
fn query_count_full_matches_db_all_len() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 42);
    let count = db.query::<Order>().count().expect("count");
    let all = db.all::<Order>().expect("all");
    assert_eq!(count, 42);
    assert_eq!(count, all.len() as u64);
}

#[test]
fn query_count_with_filter_matches_filtered_fetch_len() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let q = db.query::<Order>().filter(|o| o.status == "pending");
    let n = q.count().expect("count with filter");
    let docs: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.status == "pending")
        .fetch()
        .expect("fetch with filter");
    assert_eq!(n, 50, "50 even-indexed docs match 'pending'");
    assert_eq!(n, docs.len() as u64);
}

#[test]
fn query_count_with_limit_returns_min_total_limit() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let n_under = db.query::<Order>().limit(10).count().expect("count under");
    assert_eq!(n_under, 10);
    let n_over = db.query::<Order>().limit(500).count().expect("count over");
    assert_eq!(n_over, 100, "limit > total returns total");
}

#[test]
fn query_count_with_sort_and_limit_matches_filtered_total_min_limit() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let n = db
        .query::<Order>()
        .filter(|o| o.status == "pending")
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .limit(5)
        .count()
        .expect("count");
    assert_eq!(n, 5);
}

#[test]
fn query_count_can_be_called_then_fetch_via_separate_builders() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 10);
    let q = db.query::<Order>().filter(|o| o.placed_at >= 5);
    let n = q.count().expect("count");
    assert_eq!(n, 5);
    let docs: Vec<Order> = db
        .query::<Order>()
        .filter(|o| o.placed_at >= 5)
        .fetch()
        .expect("fetch");
    assert_eq!(docs.len() as u64, n);
    drop(q);
}

#[test]
fn query_count_index_range_fast_path() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 100);
    let n = db
        .query::<Order>()
        .index_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(30)),
                Bound::Excluded(Dynamic::U64(70)),
            ),
        )        .count()
        .expect("count");
    assert_eq!(n, 40, "[30, 70) covers 40 docs");
}

#[test]
fn query_count_empty_collection_is_collection_not_found() {
    let (db, _dir) = fresh_db();
    let err = db
        .query::<Order>()
        .count()
        .expect_err("count on absent collection");
    assert!(
        matches!(err, obj::Error::CollectionNotFound { ref name } if name == "orders"),
        "expected CollectionNotFound, got {err:?}",
    );
}

#[test]
fn iter_all_yields_docs_one_at_a_time() {
    let (db, _dir) = fresh_db();
    let total: u64 = 1_000;
    let batch: u64 = 100;
    let mut inserted: u64 = 0;
    while inserted < total {
        let end = (inserted + batch).min(total);
        db.transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            for i in inserted..end {
                let _ = coll.insert(Order {
                    customer_id: i,
                    status: "pending".to_owned(),
                    placed_at: i,
                })?;
            }
            Ok(())
        })
        .expect("batch insert");
        inserted = end;
    }

    let iter = db.iter_all::<Order>().expect("iter_all");
    let cap = usize::try_from(total).expect("usize fits u64 on test targets");
    let mut collected: Vec<Order> = Vec::with_capacity(cap);
    let mut ids: Vec<u64> = Vec::with_capacity(cap);
    for step in iter {
        let (id, doc) = step.expect("per-step");
        ids.push(id.get());
        collected.push(doc);
    }
    assert_eq!(collected.len() as u64, total, "every doc must be yielded");
    for (i, doc) in collected.iter().enumerate() {
        assert_eq!(doc.customer_id, i as u64);
    }
    assert!(
        std::mem::size_of::<obj::IterAll<'_, Order>>() < 4096,
        "IterAll struct should be small (independent of collection size)",
    );
    let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(unique.len() as u64, total, "every yielded id is distinct");
}

#[test]
fn iter_all_on_empty_collection_errors_at_construction() {
    let (db, _dir) = fresh_db();
    let err = db.iter_all::<Order>().expect_err("iter_all on absent");
    assert!(
        matches!(err, obj::Error::CollectionNotFound { ref name } if name == "orders"),
        "expected CollectionNotFound, got {err:?}",
    );
}

#[test]
fn db_all_now_collects_iter_all() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 5);
    let v: Vec<Order> = db.all::<Order>().expect("all");
    assert_eq!(v.len(), 5);
    let iter_v: Vec<Order> = db
        .iter_all::<Order>()
        .expect("iter_all")
        .map(|s| s.expect("step").1)
        .collect();
    assert_eq!(iter_v, v);
}

/// Tagged doc with an `Each` index over `tags`. The `Each` kind lets
/// us exercise the entry-vs-distinct-id divergence (a single doc may
/// contribute multiple entries under different tag keys).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Tagged {
    name: String,
    tags: Vec<String>,
}

impl Document for Tagged {
    const COLLECTION: &'static str = "tagged";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::each("by_tag", "tags").expect("each")]
    }
}

impl obj::Schema for Tagged {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("tags", obj::DynamicSchema::seq(obj::DynamicSchema::String)),
        ])
    }
}

/// Build the `Dynamic` lower / upper bounds for an equality lookup
/// against a non-unique index. Both bounds are the same
/// `Dynamic::String(key)` — the `Collection` range API encodes them
/// through `encode_field` and the `widen_bounds_for_kind` step
/// appends the `0xFF;8` id-suffix widening internally, so an
/// `Included(key)..=Included(key)` range matches every entry whose
/// user-key equals `key` regardless of its trailing `Id`.
fn equality_range(key: &str) -> (Dynamic, Dynamic) {
    (
        Dynamic::String(key.to_owned()),
        Dynamic::String(key.to_owned()),
    )
}

#[test]
fn each_index_count_distinct_ids() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-count.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..3 {
        let _ = db
            .insert(Tagged {
                name: format!("doc{i}"),
                tags: vec!["urgent".to_owned(), "review".to_owned()],
            })
            .expect("insert");
    }
    let (lower, upper) = equality_range("urgent");
    let (entries, distinct) = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Tagged>()?;
            let entries = coll.count_index_range(
                "by_tag",
                (
                    std::ops::Bound::Included(lower.clone()),
                    std::ops::Bound::Included(upper.clone()),
                ),
            )?;
            let distinct = coll.count_distinct_ids_in_range(
                "by_tag",
                (
                    std::ops::Bound::Included(lower),
                    std::ops::Bound::Included(upper),
                ),
            )?;
            Ok((entries, distinct))
        })
        .expect("read");
    assert_eq!(entries, 3, "3 entries with the 'urgent' tag");
    assert_eq!(distinct, 3, "3 distinct doc ids");
}

#[test]
fn each_index_count_distinct_dedups_within_doc() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-dedup.obj");
    let db = Db::open(&path).expect("open");
    let _ = db
        .insert(Tagged {
            name: "dup".to_owned(),
            tags: vec!["urgent".to_owned(), "urgent".to_owned()],
        })
        .expect("insert");
    let (lower, upper) = equality_range("urgent");
    let (entries, distinct) = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Tagged>()?;
            let entries = coll.count_index_range(
                "by_tag",
                (
                    std::ops::Bound::Included(lower.clone()),
                    std::ops::Bound::Included(upper.clone()),
                ),
            )?;
            let distinct = coll.count_distinct_ids_in_range(
                "by_tag",
                (
                    std::ops::Bound::Included(lower),
                    std::ops::Bound::Included(upper),
                ),
            )?;
            Ok((entries, distinct))
        })
        .expect("read");
    assert_eq!(entries, 1, "M7 de-dups duplicate tags within a doc");
    assert_eq!(distinct, 1);
}

#[test]
fn query_count_uses_distinct_path_for_each_index() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-querycount.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..3 {
        let _ = db
            .insert(Tagged {
                name: format!("doc{i}"),
                tags: vec!["urgent".to_owned(), "review".to_owned()],
            })
            .expect("insert");
    }
    let n = db
        .query::<Tagged>()
        .index_range(
            "by_tag",
            Dynamic::from("urgent".to_owned())..=Dynamic::from("urgent".to_owned()),
        )        .count()
        .expect("count");
    assert_eq!(
        n, 3,
        "Each-kind count_fast routes to count_distinct_ids_in_range \
         (3 distinct docs, not 6 raw entries)",
    );
    let docs: Vec<Tagged> = db
        .query::<Tagged>()
        .index_range(
            "by_tag",
            Dynamic::from("urgent".to_owned())..=Dynamic::from("urgent".to_owned()),
        )        .fetch()
        .expect("fetch");
    assert_eq!(docs.len() as u64, n);
}

#[test]
fn distinct_count_exceeded_when_above_cap() {
    assert_eq!(obj::MAX_DISTINCT_IDS, 100_000);
}

#[test]
#[ignore = "100k populate is slow; smoke-tested by distinct_count_exceeded_when_above_cap constant check"]
fn distinct_count_exceeded() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-over.obj");
    let db = Db::open(&path).expect("open");
    let total: u64 = (obj::MAX_DISTINCT_IDS as u64) + 1;
    let batch: u64 = 1_000;
    let mut inserted: u64 = 0;
    while inserted < total {
        let end = (inserted + batch).min(total);
        db.transaction(|tx| {
            let coll = tx.collection::<Tagged>()?;
            for i in inserted..end {
                let _ = coll.insert(Tagged {
                    name: format!("doc{i}"),
                    tags: vec!["urgent".to_owned()],
                })?;
            }
            Ok(())
        })
        .expect("batch insert");
        inserted = end;
    }
    let (lower, upper) = equality_range("urgent");
    let err = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Tagged>()?;
            coll.count_distinct_ids_in_range(
                "by_tag",
                (
                    std::ops::Bound::Included(lower),
                    std::ops::Bound::Included(upper),
                ),
            )
        })
        .expect_err("must exceed the cap");
    assert!(
        matches!(err, obj::Error::DistinctCountExceeded { limit: 100_000 }),
        "expected DistinctCountExceeded, got {err:?}",
    );
}

#[test]
fn sort_by_with_embedded_nul_string_returns_error() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 5);
    let err = db
        .query::<Order>()
        .sort_by(|_doc| Dynamic::String("has\0nul".to_owned()))
        .fetch()
        .expect_err("encode_field must reject embedded NUL");
    assert!(
        matches!(err, obj::Error::SortKeyEncode { .. }),
        "expected SortKeyEncode, got {err:?}",
    );
}

#[test]
fn sort_by_bytes_works_with_arbitrary_bytes() {
    let (db, _dir) = fresh_db();
    seed_reversed(&db, 100);
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by_bytes(|o| o.placed_at.to_be_bytes().to_vec())
        .fetch()
        .expect("sort_by_bytes fetch");
    assert_eq!(sorted.len(), 100);
    for w in sorted.windows(2) {
        assert!(
            w[0].placed_at <= w[1].placed_at,
            "ascending sort violated at {:?}/{:?}",
            w[0].placed_at,
            w[1].placed_at,
        );
    }
    assert_eq!(sorted[0].placed_at, 0);
    assert_eq!(sorted[99].placed_at, 99);
}

#[test]
fn sort_by_bytes_accepts_bytes_with_nul_unlike_sort_by() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 5);
    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by_bytes(|o| o.customer_id.to_be_bytes().to_vec())
        .fetch()
        .expect("sort_by_bytes with NUL-bearing keys");
    assert_eq!(sorted.len(), 5);
}

/// Spec test for "200k items survive filtering with default limit;
/// `.sort_buffer_limit(200_001)` lets it through".
/// Populates via 1 000-doc batches per the cleanup pattern so
/// the WAL stays inside its 64 MiB default; the test is `#[ignore]`d
/// by default because the populate takes tens of seconds — `cargo
/// test --workspace` skips it, the CI's `--include-ignored` run
/// exercises the full contract.
#[test]
#[ignore = "200k populate is slow; default-cap behaviour also covered by query_sort_buffer_default_constant_is_one_hundred_thousand"]
fn query_sort_buffer_exceeded_default_fires_at_200k() {
    let (db, _dir) = fresh_db();
    let total: u64 = 200_001;
    let batch: u64 = 1_000;
    let mut inserted: u64 = 0;
    while inserted < total {
        let end = (inserted + batch).min(total);
        db.transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            for i in inserted..end {
                let _ = coll.insert(Order {
                    customer_id: i,
                    status: "pending".to_owned(),
                    placed_at: i,
                })?;
            }
            Ok(())
        })
        .expect("batch insert");
        inserted = end;
    }

    let err = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .fetch()
        .expect_err("default buffer must overflow at >100k");
    assert!(
        matches!(err, obj::Error::SortBufferExceeded { limit: 100_000 }),
        "expected SortBufferExceeded{{ limit: 100_000 }}, got {err:?}",
    );

    let sorted: Vec<Order> = db
        .query::<Order>()
        .sort_by(|o| Dynamic::U64(o.placed_at))
        .sort_buffer_limit(200_001)
        .fetch()
        .expect("raised cap fetch");
    assert_eq!(sorted.len() as u64, total);
    assert_eq!(sorted[0].placed_at, 0);
    assert_eq!(sorted[sorted.len() - 1].placed_at, total - 1);
}

/// `Tagged` doc lives above. A doc with a Standard `by_status` index
/// exists on `Customer` in `index_maintenance.rs`; here we declare a
/// fresh shape so this test file stays self-contained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Ticket {
    /// Status field carries the indexed key. Multiple tickets may
    /// share a status (Standard index, non-unique).
    status: String,
}

impl Document for Ticket {
    const COLLECTION: &'static str = "tickets";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::standard("by_status", "status").expect("standard")]
    }
}

impl obj::Schema for Ticket {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("status", obj::DynamicSchema::String)])
    }
}

#[test]
fn standard_index_inclusive_equality_matches_all_entries() {
    let (db, _dir) = fresh_db();
    for _ in 0..3 {
        let _ = db
            .insert(Ticket {
                status: "urgent".to_owned(),
            })
            .expect("insert");
    }
    let hits: Vec<Ticket> = db
        .query::<Ticket>()
        .index_range(
            "by_status",
            Dynamic::from("urgent".to_owned())..=Dynamic::from("urgent".to_owned()),
        )        .fetch()
        .expect("fetch");
    assert_eq!(hits.len(), 3, "all 3 'urgent' docs must match");
    assert!(hits.iter().all(|t| t.status == "urgent"));
}

#[test]
fn each_index_inclusive_equality_matches_all_entries() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-eq74.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..2 {
        let _ = db
            .insert(Tagged {
                name: format!("doc{i}"),
                tags: vec!["urgent".to_owned()],
            })
            .expect("insert");
    }
    let hits: Vec<Tagged> = db
        .query::<Tagged>()
        .index_range(
            "by_tag",
            Dynamic::from("urgent".to_owned())..=Dynamic::from("urgent".to_owned()),
        )        .fetch()
        .expect("fetch");
    assert_eq!(hits.len(), 2, "both 'urgent'-tagged docs must match");
}

#[test]
fn excluded_lower_skips_user_key_entries_inclusive_upper_matches() {
    let (db, _dir) = fresh_db();
    for _ in 0..2 {
        let _ = db
            .insert(Ticket {
                status: "a".to_owned(),
            })
            .expect("a insert");
    }
    for _ in 0..3 {
        let _ = db
            .insert(Ticket {
                status: "b".to_owned(),
            })
            .expect("b insert");
    }
    let hits: Vec<Ticket> = db
        .query::<Ticket>()
        .index_range(
            "by_status",
            (
                Bound::Excluded(Dynamic::from("a".to_owned())),
                Bound::Included(Dynamic::from("b".to_owned())),
            ),
        )        .fetch()
        .expect("fetch");
    assert_eq!(hits.len(), 3, "only the 3 'b' docs match");
    assert!(hits.iter().all(|t| t.status == "b"));
}

#[test]
fn unique_inclusive_equality_still_matches_single_entry() {
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct UniqueDoc {
        email: String,
    }
    impl Document for UniqueDoc {
        const COLLECTION: &'static str = "uniq_docs_74";
        const VERSION: u32 = 1;
        fn indexes() -> Vec<IndexSpec> {
            vec![IndexSpec::unique("by_email", "email").expect("unique")]
        }
    }
    impl obj::Schema for UniqueDoc {
        fn schema() -> obj::DynamicSchema {
            obj::DynamicSchema::map([("email", obj::DynamicSchema::String)])
        }
    }
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("unique-eq74.obj");
    let db = Db::open(&path).expect("open");
    let _ = db
        .insert(UniqueDoc {
            email: "ada@example.com".to_owned(),
        })
        .expect("insert");
    let hits: Vec<UniqueDoc> = db
        .query::<UniqueDoc>()
        .index_range(
            "by_email",
            Dynamic::from("ada@example.com".to_owned())
                ..=Dynamic::from("ada@example.com".to_owned()),
        )        .fetch()
        .expect("fetch");
    assert_eq!(hits.len(), 1, "Unique single-key inclusive range matches");
    assert_eq!(hits[0].email, "ada@example.com");
}

#[test]
fn composite_inclusive_equality_matches_all_entries() {
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Pair {
        a: String,
        b: u64,
    }
    impl Document for Pair {
        const COLLECTION: &'static str = "pairs_74";
        const VERSION: u32 = 1;
        fn indexes() -> Vec<IndexSpec> {
            vec![IndexSpec::composite("by_ab", &["a", "b"]).expect("composite")]
        }
    }
    impl obj::Schema for Pair {
        fn schema() -> obj::DynamicSchema {
            obj::DynamicSchema::map([
                ("a", obj::DynamicSchema::String),
                ("b", obj::DynamicSchema::U64),
            ])
        }
    }
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("composite-eq74.obj");
    let db = Db::open(&path).expect("open");
    for b in 1u64..=3 {
        let _ = db
            .insert(Pair {
                a: "k".to_owned(),
                b,
            })
            .expect("insert");
    }
    let lo = Dynamic::Seq(vec![Dynamic::from("k".to_owned()), Dynamic::U64(1)]);
    let hi = Dynamic::Seq(vec![Dynamic::from("k".to_owned()), Dynamic::U64(3)]);
    let pairs: Vec<(Vec<u8>, Pair)> = db
        .read_transaction(|tx| {
            tx.collection::<Pair>()?
                .index_range(
                    "by_ab",
                    (Bound::Included(lo.clone()), Bound::Included(hi.clone())),
                )?
                .collect()
        })
        .expect("composite range");
    assert_eq!(pairs.len(), 3, "all 3 composite entries must match");
    let mut bs: Vec<u64> = pairs.iter().map(|(_k, p)| p.b).collect();
    bs.sort_unstable();
    assert_eq!(bs, vec![1, 2, 3]);
    assert!(pairs.iter().all(|(_k, p)| p.a == "k"));
}

#[test]
fn iter_range_yields_same_set_as_index_range() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 50);
    let want: Vec<(Vec<u8>, Order)> = db
        .read_transaction(|tx| {
            tx.collection::<Order>()?
                .index_range(
                    "placed_at",
                    (
                        Bound::Included(Dynamic::U64(10)),
                        Bound::Excluded(Dynamic::U64(40)),
                    ),
                )?
                .collect()
        })
        .expect("index_range eager");
    let got: Vec<(Vec<u8>, Order)> = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let iter = coll.iter_range(
                "placed_at",
                (
                    Bound::Included(Dynamic::U64(10)),
                    Bound::Excluded(Dynamic::U64(40)),
                ),
            )?;
            iter.collect::<Result<Vec<_>, _>>()
        })
        .expect("iter_range streaming");
    assert_eq!(got.len(), want.len(), "row counts must match");
    assert_eq!(got, want, "iter_range must yield index_range's set");
}

#[test]
fn iter_range_refills_across_batch_boundary() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 1_000);
    let count = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let iter = coll.iter_range::<(Bound<Dynamic>, Bound<Dynamic>)>(
                "placed_at",
                (Bound::Unbounded, Bound::Unbounded),
            )?;
            let mut n = 0usize;
            for step in iter {
                let _ = step?;
                n += 1;
            }
            Ok(n)
        })
        .expect("iter_range full scan");
    assert_eq!(count, 1_000, "every doc visible via iter_range");
}

#[test]
fn iter_range_empty_window_yields_nothing() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 20);
    let count = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let iter = coll.iter_range(
                "placed_at",
                (
                    Bound::Included(Dynamic::U64(1_000)),
                    Bound::Excluded(Dynamic::U64(2_000)),
                ),
            )?;
            let mut n = 0usize;
            for step in iter {
                let _ = step?;
                n += 1;
            }
            Ok(n)
        })
        .expect("iter_range empty");
    assert_eq!(count, 0);
}

#[test]
fn iter_range_each_kind_dedups_across_refills() {
    let (db, _dir) = fresh_db();
    let n: u64 = 300;
    db.transaction(|tx| {
        let coll = tx.collection::<Tagged>()?;
        for i in 0..n {
            let _ = coll.insert(Tagged {
                name: format!("d-{i}"),
                tags: vec!["urgent".to_owned(), format!("batch-{}", i % 3)],
            })?;
        }
        Ok(())
    })
    .expect("seed tagged");
    let (lo, hi) = equality_range("urgent");
    let docs: Vec<Tagged> = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Tagged>()?;
            let iter = coll.iter_range("by_tag", (Bound::Included(lo), Bound::Included(hi)))?;
            iter.map(|s| s.map(|(_k, doc)| doc))
                .collect::<Result<Vec<_>, _>>()
        })
        .expect("iter_range Each dedup");
    let usize_n = usize::try_from(n).expect("usize fits u64");
    assert_eq!(docs.len(), usize_n, "every urgent-tagged doc yielded once");
    let names: std::collections::HashSet<String> = docs.into_iter().map(|d| d.name).collect();
    assert_eq!(names.len(), usize_n, "no doc emitted twice");
}

#[test]
fn iter_range_lazy_mode_matches_streaming_mode() {
    let (db, _dir) = fresh_db();
    seed_orders(&db, 30);
    let coll = db.collection::<Order>("orders");
    let lazy_docs: Vec<Order> = coll
        .iter_range(
            "placed_at",
            (
                Bound::Included(Dynamic::U64(5)),
                Bound::Excluded(Dynamic::U64(25)),
            ),
        )
        .expect("iter_range lazy")
        .map(|s| s.expect("step").1)
        .collect();
    let streaming_docs: Vec<Order> = db
        .read_transaction(|tx| {
            tx.collection::<Order>()?
                .iter_range(
                    "placed_at",
                    (
                        Bound::Included(Dynamic::U64(5)),
                        Bound::Excluded(Dynamic::U64(25)),
                    ),
                )?
                .map(|s| s.map(|(_k, d)| d))
                .collect::<Result<Vec<_>, _>>()
        })
        .expect("iter_range streaming");
    assert_eq!(lazy_docs, streaming_docs, "lazy and streaming sets agree");
}
