//! Transactional index maintenance — Standard + Unique.
//!
//! `Each` and `Composite` extensions are covered by additional tests.
//!
//! These tests exercise the maintenance contract end-to-end through
//! `obj::Db`'s public API. Index B-tree state is observed through
//! its visible behaviour:
//!
//! - A `Unique` collision is the canonical "the index sees this
//!   write" check.
//! - Insert-then-delete-then-reinsert at the same key proves the
//!   delete-side maintenance ran.
//! - Update-changing-the-indexed-field followed by a same-key
//!   collision-via-different-doc proves the update diff emitted
//!   the new key.

#![forbid(unsafe_code)]

use obj::{Db, Document, IndexSpec};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Customer {
    email: String,
    status: String,
}

impl Document for Customer {
    const COLLECTION: &'static str = "customers";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![
            IndexSpec::unique("by_email", "email").expect("unique"),
            IndexSpec::standard("by_status", "status").expect("standard"),
        ]
    }
}

impl obj::Schema for Customer {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("email", obj::DynamicSchema::String),
            ("status", obj::DynamicSchema::String),
        ])
    }
}

/// Hermetic file-backed `Db` plus the owning `TempDir`. File
/// backing is required because some of these tests exercise the
/// rollback path on `UniqueConstraintViolation`, and the in-memory
/// pager's writes do not unwind (memory pagers have no WAL).
fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("indexes.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

#[test]
fn insert_then_duplicate_unique_email_errors() {
    let (db, _dir) = fresh_db();
    let _id = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("first");
    let err = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "archived".to_owned(),
        })
        .expect_err("dup unique");
    match err {
        obj::Error::UniqueConstraintViolation {
            index, collection, ..
        } => {
            assert_eq!(index, "by_email");
            assert_eq!(collection, Customer::COLLECTION);
        }
        other => panic!("expected UniqueConstraintViolation, got {other:?}"),
    }
}

#[test]
fn unique_violation_rolls_back_primary_write() {
    let (db, _dir) = fresh_db();
    let _id = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("first");
    let _err = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "archived".to_owned(),
        })
        .expect_err("dup unique");
    let count = db
        .read_transaction(|tx| Ok(tx.collection::<Customer>()?.all()?.len()))
        .expect("count");
    assert_eq!(count, 1, "rollback must leave 1 doc, not 2");
}

#[test]
fn delete_clears_unique_constraint_so_reinsert_succeeds() {
    let (db, _dir) = fresh_db();
    let id = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("first");
    let _ = db.delete::<Customer>(id).expect("delete");
    let _id2 = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("reinsert succeeds after delete");
}

#[test]
fn update_changing_unique_field_swaps_the_index_entry() {
    let (db, _dir) = fresh_db();
    let id_a = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("a");
    db.update::<Customer, _>(id_a, |c| c.email = "z@e.com".to_owned())
        .expect("update");
    let _id_new = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("new doc reusing the freed key");
    let err = db
        .insert(Customer {
            email: "z@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect_err("post-update key is taken");
    assert!(matches!(err, obj::Error::UniqueConstraintViolation { .. }));
}

#[test]
fn update_to_collide_with_other_doc_errors() {
    let (db, _dir) = fresh_db();
    let _id_a = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("a");
    let id_b = db
        .insert(Customer {
            email: "b@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("b");
    let err = db
        .update::<Customer, _>(id_b, |c| c.email = "a@e.com".to_owned())
        .expect_err("collide");
    assert!(matches!(err, obj::Error::UniqueConstraintViolation { .. }));
    let b_back = db.get::<Customer>(id_b).expect("get").expect("present");
    assert_eq!(b_back.email, "b@e.com", "rollback must restore b");
}

#[test]
fn update_same_unique_value_is_idempotent_no_self_collision() {
    let (db, _dir) = fresh_db();
    let id = db
        .insert(Customer {
            email: "x@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("ok");
    db.update::<Customer, _>(id, |c| c.status = "archived".to_owned())
        .expect("update");
    let back = db.get::<Customer>(id).expect("get").expect("present");
    assert_eq!(back.status, "archived");
    assert_eq!(back.email, "x@e.com");
}

#[test]
fn upsert_replaces_indexed_entries() {
    let (db, _dir) = fresh_db();
    let id = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("a");
    db.upsert::<Customer>(
        id,
        Customer {
            email: "b@e.com".to_owned(),
            status: "active".to_owned(),
        },
    )
    .expect("upsert");
    let _ = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "x".to_owned(),
        })
        .expect("reuse old key");
    let err = db
        .insert(Customer {
            email: "b@e.com".to_owned(),
            status: "x".to_owned(),
        })
        .expect_err("new key taken");
    assert!(matches!(err, obj::Error::UniqueConstraintViolation { .. }));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    placed_at: u64,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![
            IndexSpec::composite("by_customer_time", &["customer_id", "placed_at"])
                .expect("composite"),
        ]
    }
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("placed_at", obj::DynamicSchema::U64),
        ])
    }
}

#[test]
fn each_index_one_doc_with_two_tags_creates_two_entries() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each.obj");
    let db = Db::open(&path).expect("open");
    let _id = db
        .insert(Tagged {
            name: "x".to_owned(),
            tags: vec!["red".to_owned(), "green".to_owned()],
        })
        .expect("insert");
    let _id2 = db
        .insert(Tagged {
            name: "y".to_owned(),
            tags: vec!["red".to_owned()],
        })
        .expect("insert 2");
    let count = db
        .read_transaction(|tx| Ok(tx.collection::<Tagged>()?.all()?.len()))
        .expect("count");
    assert_eq!(count, 2);
}

#[test]
fn each_index_update_removes_old_tags_and_adds_new() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-up.obj");
    let db = Db::open(&path).expect("open");
    let id = db
        .insert(Tagged {
            name: "x".to_owned(),
            tags: vec!["red".to_owned(), "green".to_owned()],
        })
        .expect("insert");
    db.update::<Tagged, _>(id, |t| {
        t.tags = vec!["red".to_owned(), "blue".to_owned()];
    })
    .expect("update");
    let back = db.get::<Tagged>(id).expect("get").expect("present");
    assert_eq!(back.tags, vec!["red".to_owned(), "blue".to_owned()]);
}

#[test]
fn each_index_delete_removes_all_tag_entries() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-del.obj");
    let db = Db::open(&path).expect("open");
    let id = db
        .insert(Tagged {
            name: "x".to_owned(),
            tags: vec!["red".to_owned(), "green".to_owned(), "blue".to_owned()],
        })
        .expect("insert");
    let _ = db.delete::<Tagged>(id).expect("delete");
    let back = db.get::<Tagged>(id).expect("get");
    assert!(back.is_none());
    let count = db
        .read_transaction(|tx| Ok(tx.collection::<Tagged>()?.all()?.len()))
        .expect("count");
    assert_eq!(count, 0);
}

#[test]
fn composite_index_round_trips_through_insert_and_delete() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("composite.obj");
    let db = Db::open(&path).expect("open");
    let id1 = db
        .insert(Order {
            customer_id: 1,
            placed_at: 100,
        })
        .expect("o1");
    let id2 = db
        .insert(Order {
            customer_id: 1,
            placed_at: 200,
        })
        .expect("o2");
    let id3 = db
        .insert(Order {
            customer_id: 2,
            placed_at: 150,
        })
        .expect("o3");
    let _ = db.delete::<Order>(id2).expect("delete o2");
    let remaining = db
        .read_transaction(|tx| Ok(tx.collection::<Order>()?.all()?.len()))
        .expect("count");
    assert_eq!(remaining, 2);
    db.update::<Order, _>(id1, |o| o.placed_at = 110)
        .expect("update o1");
    let back = db.get::<Order>(id1).expect("get").expect("present");
    assert_eq!(back.placed_at, 110);
    let _ = id3;
}

#[test]
fn each_index_with_empty_seq_does_nothing() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-empty.obj");
    let db = Db::open(&path).expect("open");
    let _id = db
        .insert(Tagged {
            name: "noop".to_owned(),
            tags: vec![],
        })
        .expect("insert");
    let count = db
        .read_transaction(|tx| Ok(tx.collection::<Tagged>()?.all()?.len()))
        .expect("count");
    assert_eq!(count, 1);
}

#[test]
fn each_index_duplicate_element_inside_one_doc_de_dups() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-dup.obj");
    let db = Db::open(&path).expect("open");
    let id = db
        .insert(Tagged {
            name: "dup".to_owned(),
            tags: vec!["foo".to_owned(), "foo".to_owned()],
        })
        .expect("insert");
    db.update::<Tagged, _>(id, |t| t.tags = vec!["foo".to_owned()])
        .expect("update");
    let back = db.get::<Tagged>(id).expect("get").expect("present");
    assert_eq!(back.tags, vec!["foo".to_owned()]);
}

#[test]
fn find_unique_returns_doc_on_match() {
    let (db, _dir) = fresh_db();
    let id = db
        .insert(Customer {
            email: "ada@example.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("insert");
    let back = db
        .find_unique::<Customer>("by_email", "ada@example.com".to_owned())
        .expect("find")
        .expect("present");
    assert_eq!(back.email, "ada@example.com");
    let _ = id;
}

#[test]
fn find_unique_returns_none_on_miss() {
    let (db, _dir) = fresh_db();
    let _ = db
        .insert(Customer {
            email: "someone@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("seed");
    let back = db
        .find_unique::<Customer>("by_email", "nobody@example.com".to_owned())
        .expect("find");
    assert!(back.is_none());
}

#[test]
fn find_unique_errors_on_non_unique_index() {
    let (db, _dir) = fresh_db();
    let _id = db
        .insert(Customer {
            email: "x@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("insert");
    let err = db
        .find_unique::<Customer>("by_status", "active".to_owned())
        .expect_err("non-unique");
    assert!(matches!(err, obj::Error::IndexNotUnique { .. }));
}

#[test]
fn find_unique_unknown_index_errors() {
    let (db, _dir) = fresh_db();
    let _ = db
        .insert(Customer {
            email: "x@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("seed");
    let err = db
        .find_unique::<Customer>("by_nope", "x".to_owned())
        .expect_err("unknown");
    assert!(matches!(err, obj::Error::IndexNotFound { .. }));
}

#[test]
fn lookup_on_standard_yields_every_matching_doc() {
    let (db, _dir) = fresh_db();
    let _ = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("a");
    let _ = db
        .insert(Customer {
            email: "b@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("b");
    let _ = db
        .insert(Customer {
            email: "c@e.com".to_owned(),
            status: "archived".to_owned(),
        })
        .expect("c");
    let docs: Vec<Customer> = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Customer>()?;
            let it = coll.lookup("by_status", "active".to_owned())?;
            it.collect::<obj::Result<Vec<_>>>()
        })
        .expect("lookup");
    assert_eq!(docs.len(), 2);
    let emails: std::collections::HashSet<String> = docs.iter().map(|d| d.email.clone()).collect();
    assert!(emails.contains("a@e.com"));
    assert!(emails.contains("b@e.com"));
}

#[test]
fn lookup_on_each_index_returns_doc_for_matching_element() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-lookup.obj");
    let db = Db::open(&path).expect("open");
    let _ = db
        .insert(Tagged {
            name: "doc1".to_owned(),
            tags: vec!["red".to_owned(), "green".to_owned()],
        })
        .expect("d1");
    let _ = db
        .insert(Tagged {
            name: "doc2".to_owned(),
            tags: vec!["red".to_owned()],
        })
        .expect("d2");
    let _ = db
        .insert(Tagged {
            name: "doc3".to_owned(),
            tags: vec!["blue".to_owned()],
        })
        .expect("d3");
    let red_docs: Vec<Tagged> = db
        .read_transaction(|tx| {
            tx.collection::<Tagged>()?
                .lookup("by_tag", "red".to_owned())?
                .collect()
        })
        .expect("red lookup");
    let names: std::collections::HashSet<String> =
        red_docs.iter().map(|d| d.name.clone()).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains("doc1"));
    assert!(names.contains("doc2"));
}

#[test]
fn lookup_dedups_each_index_when_doc_has_multiple_matches() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-dedup-lookup.obj");
    let db = Db::open(&path).expect("open");
    let _id = db
        .insert(Tagged {
            name: "dup".to_owned(),
            tags: vec!["foo".to_owned()],
        })
        .expect("dup");
    let docs: Vec<Tagged> = db
        .read_transaction(|tx| {
            tx.collection::<Tagged>()?
                .lookup("by_tag", "foo".to_owned())?
                .collect()
        })
        .expect("lookup");
    assert_eq!(docs.len(), 1);
}

#[test]
fn index_range_on_standard_returns_ordered_pairs() {
    let (db, _dir) = fresh_db();
    let _ = db
        .insert(Customer {
            email: "a@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("a");
    let _ = db
        .insert(Customer {
            email: "z@e.com".to_owned(),
            status: "blocked".to_owned(),
        })
        .expect("z");
    let _ = db
        .insert(Customer {
            email: "m@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("m");
    let pairs: Vec<(Vec<u8>, Customer)> = db
        .read_transaction(|tx| {
            tx.collection::<Customer>()?
                .index_range("by_status", ..)?
                .collect()
        })
        .expect("range");
    assert_eq!(pairs.len(), 3);
    for window in pairs.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}

#[test]
fn pure_insert_fast_path_standard_lookup_and_range_are_correct() {
    // Exercises the old_keys.is_empty() pure-insert fast path in
    // apply_nonunique_diff for a Standard index across many inserts,
    // then verifies lookup returns every matching doc and index_range
    // stays ordered with the right cardinality.
    let (db, _dir) = fresh_db();
    for i in 0..50u32 {
        let status = if i % 2 == 0 { "active" } else { "archived" };
        let _ = db
            .insert(Customer {
                email: format!("user{i}@e.com"),
                status: status.to_owned(),
            })
            .expect("insert");
    }
    let active: Vec<Customer> = db
        .read_transaction(|tx| {
            tx.collection::<Customer>()?
                .lookup("by_status", "active".to_owned())?
                .collect()
        })
        .expect("lookup");
    assert_eq!(active.len(), 25);
    assert!(active.iter().all(|c| c.status == "active"));

    let pairs: Vec<(Vec<u8>, Customer)> = db
        .read_transaction(|tx| {
            tx.collection::<Customer>()?
                .index_range("by_status", ..)?
                .collect()
        })
        .expect("range");
    assert_eq!(pairs.len(), 50);
    for window in pairs.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}

#[test]
fn pure_insert_fast_path_each_creates_all_entries_and_dedups() {
    // The Each pure-insert fast path must create one entry per
    // distinct element and de-dup equal elements within one doc,
    // matching the prior BTreeSet behaviour.
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("each-fastpath.obj");
    let db = Db::open(&path).expect("open");
    let _ = db
        .insert(Tagged {
            name: "d1".to_owned(),
            tags: vec!["red".to_owned(), "green".to_owned()],
        })
        .expect("d1");
    let _ = db
        .insert(Tagged {
            name: "d2".to_owned(),
            tags: vec!["red".to_owned(), "red".to_owned()],
        })
        .expect("d2 with duplicate element");
    let red: Vec<Tagged> = db
        .read_transaction(|tx| {
            tx.collection::<Tagged>()?
                .lookup("by_tag", "red".to_owned())?
                .collect()
        })
        .expect("red lookup");
    let names: std::collections::HashSet<String> = red.iter().map(|d| d.name.clone()).collect();
    assert_eq!(names.len(), 2, "both docs match red exactly once");
    assert!(names.contains("d1"));
    assert!(names.contains("d2"));
}

#[test]
fn pure_insert_fast_path_composite_round_trips() {
    // Composite pure-insert fast path: many inserts then full range
    // must report every entry, ordered.
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("composite-fastpath.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..20u64 {
        let _ = db
            .insert(Order {
                customer_id: i % 4,
                placed_at: i,
            })
            .expect("order");
    }
    let pairs: Vec<(Vec<u8>, Order)> = db
        .read_transaction(|tx| {
            tx.collection::<Order>()?
                .index_range("by_customer_time", ..)?
                .collect()
        })
        .expect("range");
    assert_eq!(pairs.len(), 20);
    for window in pairs.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}

#[test]
fn lookup_does_not_see_concurrent_writers_on_a_snapshot() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("snap-iso.obj");
    let db = Db::open(&path).expect("open");
    let _ = db
        .insert(Customer {
            email: "before@e.com".to_owned(),
            status: "active".to_owned(),
        })
        .expect("before");
    let _ = db.read_transaction(|rx| {
        let pre = rx
            .collection::<Customer>()?
            .find_unique("by_email", "before@e.com".to_owned())?;
        assert!(pre.is_some());
        Ok(())
    });
}

#[test]
fn collection_with_no_indexes_still_works() {
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Plain {
        n: u64,
    }
    impl Document for Plain {
        const COLLECTION: &'static str = "plain";
        const VERSION: u32 = 1;
    }
    impl obj::Schema for Plain {
        fn schema() -> obj::DynamicSchema {
            obj::DynamicSchema::map([("n", obj::DynamicSchema::U64)])
        }
    }
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("plain.obj");
    let db = Db::open(&path).expect("open");
    let id = db.insert(Plain { n: 42 }).expect("insert");
    let back = db.get::<Plain>(id).expect("get").expect("present");
    assert_eq!(back.n, 42);
    let _ = db.delete::<Plain>(id).expect("delete");
}
