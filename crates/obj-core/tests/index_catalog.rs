//! Catalog wiring of secondary indexes — declare / drop /
//! reconcile lifecycle.
//!
//! Tests at the `obj-core::catalog::Catalog` API level (i.e.
//! independent of the `obj::Db` wrapping). They exercise:
//!
//! 1. Declaring two indexes through `Catalog::declare_index` and
//!    confirming both `IndexDescriptor`s persist with non-zero
//!    `root_page_id`s.
//! 2. Reconciling against a smaller spec set — the missing index
//!    flips to `DroppedPending` but its `index_id` stays consumed.
//! 3. Reconciliation idempotence: applying the same spec set twice
//!    produces no descriptor churn.

#![forbid(unsafe_code)]

use obj_core::btree::BTree;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor, IndexKind, IndexSpec, IndexStatus};

fn fresh_pager_in_txn() -> Pager<FileHandle> {
    Pager::<FileHandle>::memory(Config::default()).expect("pager")
}

fn register_collection(pager: &mut Pager<FileHandle>, catalog: &mut Catalog<FileHandle>) {
    let primary_root = BTree::<FileHandle>::empty(pager)
        .expect("primary tree")
        .root();
    let descriptor = CollectionDescriptor::new(0, primary_root.get(), 1);
    catalog
        .insert(pager, "customers", descriptor)
        .expect("register customers");
}

#[test]
fn declare_index_persists_descriptor_with_active_status() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let by_email = IndexSpec::unique("by_email", "email").expect("spec");
    let id = catalog
        .declare_index(&mut pager, "customers", &by_email)
        .expect("declare");
    assert_eq!(id, 1, "first index gets index_id 1");

    let descriptor = catalog
        .get(&mut pager, "customers")
        .expect("get")
        .expect("present");
    assert_eq!(descriptor.indexes.len(), 1);
    let stored = &descriptor.indexes[0];
    assert_eq!(stored.name, "by_email");
    assert_eq!(stored.kind, IndexKind::Unique);
    assert_eq!(stored.key_paths, vec!["email".to_owned()]);
    assert_eq!(stored.status, IndexStatus::Active);
    assert_ne!(stored.root_page_id, 0, "index B+tree root must be non-zero");
}

#[test]
fn two_index_declarations_each_get_distinct_root_pages() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let a = IndexSpec::standard("by_status", "status").expect("a");
    let b = IndexSpec::unique("by_email", "email").expect("b");
    catalog
        .declare_index(&mut pager, "customers", &a)
        .expect("a");
    catalog
        .declare_index(&mut pager, "customers", &b)
        .expect("b");

    let descriptor = catalog
        .get(&mut pager, "customers")
        .expect("get")
        .expect("present");
    assert_eq!(descriptor.indexes.len(), 2);
    let r1 = descriptor.indexes[0].root_page_id;
    let r2 = descriptor.indexes[1].root_page_id;
    assert_ne!(r1, 0);
    assert_ne!(r2, 0);
    assert_ne!(r1, r2, "each index gets its own B+tree root");
}

#[test]
fn drop_index_flips_to_dropped_pending_and_keeps_id() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let a = IndexSpec::standard("by_status", "status").expect("a");
    let id_a = catalog
        .declare_index(&mut pager, "customers", &a)
        .expect("a");
    catalog
        .drop_index(&mut pager, "customers", "by_status")
        .expect("drop");

    let descriptor = catalog
        .get(&mut pager, "customers")
        .expect("get")
        .expect("present");
    assert_eq!(descriptor.indexes.len(), 1);
    let stored = &descriptor.indexes[0];
    assert_eq!(stored.status, IndexStatus::DroppedPending);
    assert_eq!(stored.index_id, id_a, "index_id is NOT reused on drop");
}

#[test]
fn reconcile_declares_missing_specs_and_drops_extra_active() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let a = IndexSpec::standard("by_status", "status").expect("a");
    let b = IndexSpec::unique("by_email", "email").expect("b");
    let post1 = catalog
        .reconcile_indexes(&mut pager, "customers", &[a, b.clone()])
        .expect("round 1");
    assert_eq!(post1.len(), 2);
    assert!(post1.iter().all(|d| d.status == IndexStatus::Active));

    let post2 = catalog
        .reconcile_indexes(&mut pager, "customers", &[b])
        .expect("round 2");
    assert_eq!(post2.len(), 2, "DroppedPending stays in the vector");
    let by_status = post2.iter().find(|d| d.name == "by_status").expect("a");
    let by_email = post2.iter().find(|d| d.name == "by_email").expect("b");
    assert_eq!(by_status.status, IndexStatus::DroppedPending);
    assert_eq!(by_email.status, IndexStatus::Active);
}

#[test]
fn reconcile_is_idempotent_no_descriptor_churn() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let a = IndexSpec::standard("by_status", "status").expect("a");
    let b = IndexSpec::unique("by_email", "email").expect("b");
    let post1 = catalog
        .reconcile_indexes(&mut pager, "customers", &[a.clone(), b.clone()])
        .expect("round 1");
    let post2 = catalog
        .reconcile_indexes(&mut pager, "customers", &[a, b])
        .expect("round 2");
    assert_eq!(post1, post2);
}

#[test]
fn reconcile_rejects_kind_mismatch() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    register_collection(&mut pager, &mut catalog);

    let v1 = IndexSpec::standard("by_status", "status").expect("v1");
    catalog
        .reconcile_indexes(&mut pager, "customers", &[v1])
        .expect("round 1");
    let v2 = IndexSpec::unique("by_status", "status").expect("v2");
    let err = catalog
        .reconcile_indexes(&mut pager, "customers", &[v2])
        .expect_err("kind mismatch");
    assert!(matches!(err, obj_core::Error::IndexKindMismatch { .. }));
}

#[test]
fn declare_on_missing_collection_errors() {
    let mut pager = fresh_pager_in_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
    let spec = IndexSpec::standard("by_x", "x").expect("spec");
    let err = catalog
        .declare_index(&mut pager, "nope", &spec)
        .expect_err("missing");
    assert!(matches!(err, obj_core::Error::CollectionNotFound { .. }));
}

#[test]
fn descriptors_round_trip_through_catalog_reopen() {
    let mut pager = fresh_pager_in_txn();
    {
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        register_collection(&mut pager, &mut catalog);
        let a = IndexSpec::standard("by_status", "status").expect("a");
        catalog
            .declare_index(&mut pager, "customers", &a)
            .expect("declare");
    }
    let catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("reopen");
    let descriptor = catalog
        .get(&mut pager, "customers")
        .expect("get")
        .expect("present");
    assert_eq!(descriptor.indexes.len(), 1);
    let stored = &descriptor.indexes[0];
    assert_eq!(stored.name, "by_status");
    assert_eq!(stored.kind, IndexKind::Standard);
    assert_eq!(stored.status, IndexStatus::Active);
}
