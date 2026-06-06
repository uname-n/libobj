//! End-to-end coverage for the non-generic index-declaration seam
//! `WriteTxn::reconcile_indexes_raw`.
//!
//! This is the engine entry point the FFI index path will call: it
//! declares `IndexSpec`s into the catalog WITHOUT a Rust
//! `#[derive(Document)]` type, making each index `Active` BEFORE the
//! first index-maintaining raw write (`insert_raw_indexed` requires the
//! index already `Active`).
//!
//! The headline test proves the declaration-before-maintenance order
//! works end to end: declare a `Unique` + `Standard` index via
//! `reconcile_indexes_raw`, write two documents via `insert_raw_indexed`
//! with matching field-encoded entries, then read each back through its
//! index (`find_unique_raw` / `index_range_raw`).

#![forbid(unsafe_code)]

use std::ops::Bound;

use obj::{Db, IndexSpec};
use obj_core::codec::Dynamic;
use obj_core::index::encode_field;

const COLLECTION: &str = "people";
const BY_EMAIL: &str = "by_email";
const BY_STATUS: &str = "by_status";
const BY_AGE: &str = "by_age";
/// Schema version the headline / idempotency / mismatch tests declare
/// against. `reconcile_indexes_raw` keys its skip-cache by `(collection,
/// version)`.
const V1: u32 = 1;
/// The later schema version used by the cross-version index-addition
/// test — its spec set ADDS `by_age` on top of `V1`'s.
const V2: u32 = 2;

/// Field-encode a string value into the index-key byte shape the raw
/// write / read seams expect (the caller owns the order-preserving
/// encoding; the engine appends the per-kind storage suffix).
fn email_key(value: &str) -> Vec<u8> {
    encode_field(&Dynamic::String(value.to_owned()))
        .expect("encode_field")
        .into_bytes()
}

/// Field-encode an integer value into the index-key byte shape (used by
/// the `by_age` standard index in the cross-version addition test).
fn int_key(value: i64) -> Vec<u8> {
    encode_field(&Dynamic::I64(value))
        .expect("encode_field i64")
        .into_bytes()
}

/// The two specs the raw declaration path stamps into the catalog:
/// one `Unique` scalar index and one `Standard` scalar index.
fn specs() -> Vec<IndexSpec> {
    vec![
        IndexSpec::unique(BY_EMAIL, "email").expect("unique spec"),
        IndexSpec::standard(BY_STATUS, "status").expect("standard spec"),
    ]
}

/// A v2 spec set that ADDS a third `Standard` index (`by_age`) on top
/// of the v1 set — the cross-version index-addition scenario.
fn specs_v2() -> Vec<IndexSpec> {
    let mut v = specs();
    v.push(IndexSpec::standard(BY_AGE, "age").expect("standard age spec"));
    v
}

#[test]
fn declare_via_raw_then_insert_indexed_then_read_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reconcile-raw.obj");
    let db = Db::open(&path).expect("open");

    db.transaction(|tx| {
        tx.reconcile_indexes_raw(COLLECTION, V1, &specs())?;
        let ada = b"ada-payload";
        let grace = b"grace-payload";
        let _ada_id = tx.insert_raw_indexed(
            COLLECTION,
            ada,
            1,
            &[
                (BY_EMAIL.to_owned(), email_key("ada@x.test")),
                (BY_STATUS.to_owned(), email_key("active")),
            ],
        )?;
        let _grace_id = tx.insert_raw_indexed(
            COLLECTION,
            grace,
            1,
            &[
                (BY_EMAIL.to_owned(), email_key("grace@x.test")),
                (BY_STATUS.to_owned(), email_key("active")),
            ],
        )?;
        Ok(())
    })
    .expect("declare + indexed inserts commit");

    db.read_transaction(|tx| {
        let hit = tx.find_unique_raw(COLLECTION, BY_EMAIL, &email_key("ada@x.test"))?;
        let (_id, payload) = hit.expect("ada present via unique index");
        assert_eq!(payload.as_slice(), b"ada-payload");

        let miss = tx.find_unique_raw(COLLECTION, BY_EMAIL, &email_key("nobody@x.test"))?;
        assert!(miss.is_none(), "absent unique key yields None");

        let key = email_key("active");
        let rows = tx.index_range_raw(
            COLLECTION,
            BY_STATUS,
            Bound::Included(key.clone()),
            Bound::Included(key),
        )?;
        assert_eq!(rows.len(), 2, "both active docs visible via standard index");
        Ok(())
    })
    .expect("index read-back");
}

#[test]
fn second_reconcile_with_same_specs_is_noop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reconcile-idempotent.obj");
    let db = Db::open(&path).expect("open");

    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs()))
        .expect("first declare");

    db.transaction(|tx| {
        tx.reconcile_indexes_raw(COLLECTION, V1, &specs())?;
        tx.reconcile_indexes_raw(COLLECTION, V1, &specs())
    })
    .expect("second declare is a no-op");

    db.transaction(|tx| {
        tx.insert_raw_indexed(
            COLLECTION,
            b"x",
            1,
            &[
                (BY_EMAIL.to_owned(), email_key("solo@x.test")),
                (BY_STATUS.to_owned(), email_key("active")),
            ],
        )
        .map(|_id| ())
    })
    .expect("indexed write after idempotent re-declare");

    db.read_transaction(|tx| {
        let hit = tx.find_unique_raw(COLLECTION, BY_EMAIL, &email_key("solo@x.test"))?;
        assert!(hit.is_some(), "doc present after idempotent re-declare");
        Ok(())
    })
    .expect("read-back");
}

#[test]
fn re_declare_with_mismatched_kind_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reconcile-mismatch.obj");

    {
        let db = Db::open(&path).expect("open");
        db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs()))
            .expect("declare unique");
    }

    let db = Db::open(&path).expect("reopen");
    let mismatched = vec![IndexSpec::standard(BY_EMAIL, "email").expect("standard spec")];
    let err = db
        .transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &mismatched))
        .expect_err("kind mismatch must error");
    assert!(
        matches!(err, obj::Error::IndexKindMismatch { .. }),
        "expected IndexKindMismatch, got {err:?}"
    );
}

#[test]
fn manual_raw_indexed_error_then_commit_persists_staged_primary_by_contract() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("manual-error-contract.obj");
    let db = Db::open(&path).expect("open");

    let env = db.env_arc();
    let inner = obj_core::WriteTxn::begin(&env, db.busy_timeout()).expect("begin manual tx");
    let mut tx = obj::WriteTxn::from_parts(inner, db.catalog_arc(), db.reconciled_arc());
    tx.reconcile_indexes_raw(COLLECTION, V1, &specs())
        .expect("declare indexes");
    let _first = tx
        .insert_raw_indexed(
            COLLECTION,
            b"first",
            V1,
            &[(BY_EMAIL.to_owned(), email_key("dup@x.test"))],
        )
        .expect("first insert");
    let err = tx
        .insert_raw_indexed(
            COLLECTION,
            b"second-primary-only-after-error",
            V1,
            &[(BY_EMAIL.to_owned(), email_key("dup@x.test"))],
        )
        .expect_err("duplicate unique key");
    assert!(
        matches!(err, obj::Error::UniqueConstraintViolation { .. }),
        "expected UniqueConstraintViolation, got {err:?}"
    );
    tx.commit().expect("manual commit after error is allowed");

    db.read_transaction(|tx| {
        let rows = tx.all_raw(COLLECTION)?;
        assert_eq!(
            rows.len(),
            2,
            "manual commit preserves staged primary writes"
        );
        let duplicate_index_rows =
            tx.find_unique_raw(COLLECTION, BY_EMAIL, &email_key("dup@x.test"))?;
        assert!(
            duplicate_index_rows.is_some(),
            "the first unique index entry remains queryable"
        );
        Ok(())
    })
    .expect("read committed manual transaction");
}

/// Cross-version index ADDITION in ONE process.
///
/// Reconcile a v1 spec set (`by_email`, `by_status`) and insert a v1
/// doc; then, in the SAME `Db` (same process, warm cache), reconcile a
/// v2 spec set that ADDS `by_age` and insert a v2 doc. Before the
/// `(collection, version)` cache key, the v2 reconcile was skipped
/// (collection already reconciled at v1) so `by_age` never became
/// `Active` and the v2 `insert_raw_indexed` failed with
/// `IndexNotFound`. Now each version reconciles once: `by_age` is
/// `Active` and the v2 doc is queryable through it.
#[test]
fn cross_version_index_addition_same_process() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("reconcile-cross-version.obj");
    let db = Db::open(&path).expect("open");

    db.transaction(|tx| {
        tx.reconcile_indexes_raw(COLLECTION, V1, &specs())?;
        tx.insert_raw_indexed(
            COLLECTION,
            b"v1-payload",
            V1,
            &[
                (BY_EMAIL.to_owned(), email_key("v1@x.test")),
                (BY_STATUS.to_owned(), email_key("active")),
            ],
        )
        .map(|_id| ())
    })
    .expect("v1 declare + insert");

    db.transaction(|tx| {
        tx.reconcile_indexes_raw(COLLECTION, V2, &specs_v2())?;
        tx.insert_raw_indexed(
            COLLECTION,
            b"v2-payload",
            V2,
            &[
                (BY_EMAIL.to_owned(), email_key("v2@x.test")),
                (BY_STATUS.to_owned(), email_key("active")),
                (BY_AGE.to_owned(), int_key(42)),
            ],
        )
        .map(|_id| ())
    })
    .expect("v2 declare (adds by_age) + insert");

    db.read_transaction(|tx| {
        let key = int_key(42);
        let rows = tx.index_range_raw(
            COLLECTION,
            BY_AGE,
            Bound::Included(key.clone()),
            Bound::Included(key),
        )?;
        assert_eq!(
            rows.len(),
            1,
            "v2 doc visible via the newly-added by_age index"
        );
        let (_id, payload) = &rows[0];
        assert_eq!(payload.as_slice(), b"v2-payload");
        Ok(())
    })
    .expect("read-back via added index");
}
