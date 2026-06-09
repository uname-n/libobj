//! Coverage for the `ReadTxn` snapshot helpers and the scattered
//! raw-helper error arms in `obj-rs`'s `txn.rs` (issue #3):
//!
//! 1. `snapshot_index_descriptor` — `Active` hit, missing collection,
//!    unknown index, and an index present but `DroppedPending`.
//! 2. `index_range_raw_with_version` — populated index-range
//!    round-trip surfacing each record's stored `type_version`.
//! 3. The raw read / write helpers' `CollectionNotFound` /
//!    `IndexNotFound` / `IndexNotUnique` / namespaced-write guards
//!    where reachable through the public API, plus the previously
//!    untested write-side shims (`get_raw_bytes`, `count_all_raw`,
//!    `delete_raw_bytes`, `update_raw_bytes`, `upsert_*`,
//!    `update_raw_indexed`, `delete_raw_indexed`).

#![forbid(unsafe_code)]

use std::ops::Bound;

use obj::{Db, Error, Id, IndexSpec};
use obj_core::codec::Dynamic;
use obj_core::index::encode_field;
use tempfile::TempDir;

const COLLECTION: &str = "items";
const BY_EMAIL: &str = "by_email";
const BY_STATUS: &str = "by_status";
/// Schema version the initial spec set is declared under.
const V1: u32 = 1;
/// Later schema version used when re-declaring a REDUCED spec set
/// (the omitted index flips to `DroppedPending`).
const V2: u32 = 2;

/// Field-encode a string value into the raw index-key byte shape the
/// raw write / read seams expect.
fn skey(value: &str) -> Vec<u8> {
    encode_field(&Dynamic::String(value.to_owned()))
        .expect("encode_field")
        .into_bytes()
}

/// The v1 declaration: one `Unique` + one `Standard` index.
fn specs_v1() -> Vec<IndexSpec> {
    vec![
        IndexSpec::unique(BY_EMAIL, "email").expect("unique spec"),
        IndexSpec::standard(BY_STATUS, "status").expect("standard spec"),
    ]
}

/// Declare the v1 indexes and insert two indexed documents with
/// distinct payloads / type versions / status keys. Returns the ids.
fn seed_two_indexed_docs(db: &Db) -> (Id, Id) {
    db.transaction(|tx| {
        tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1())?;
        let a = tx.insert_raw_indexed(
            COLLECTION,
            b"payload-a",
            3,
            &[
                (BY_EMAIL.to_owned(), skey("a@x.test")),
                (BY_STATUS.to_owned(), skey("alpha")),
            ],
        )?;
        let b = tx.insert_raw_indexed(
            COLLECTION,
            b"payload-b",
            5,
            &[
                (BY_EMAIL.to_owned(), skey("b@x.test")),
                (BY_STATUS.to_owned(), skey("beta")),
            ],
        )?;
        Ok((a, b))
    })
    .expect("seed two indexed docs")
}

// ---------------------------------------------------------------------------
// snapshot_index_descriptor
// ---------------------------------------------------------------------------

#[test]
fn snapshot_index_descriptor_returns_active_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("sid_ok.obj")).expect("open");

    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1()))
        .expect("declare v1 indexes");

    let descriptor = db
        .read_transaction(|tx| tx.snapshot_index_descriptor(COLLECTION, BY_EMAIL))
        .expect("active index resolves");
    assert_eq!(descriptor.name, BY_EMAIL, "name matches");
    assert_eq!(
        descriptor.kind,
        obj_core::IndexKind::Unique,
        "kind matches the declared spec"
    );
    assert_eq!(
        descriptor.status,
        obj_core::IndexStatus::Active,
        "freshly declared index is Active"
    );
}

#[test]
fn snapshot_index_descriptor_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("sid_nocol.obj")).expect("open");

    let err = db
        .read_transaction(|tx| tx.snapshot_index_descriptor("ghost", BY_EMAIL))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn snapshot_index_descriptor_errors_for_unknown_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("sid_noidx.obj")).expect("open");

    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1()))
        .expect("declare v1 indexes");

    let err = db
        .read_transaction(|tx| tx.snapshot_index_descriptor(COLLECTION, "no_such_index"))
        .expect_err("unknown index must error");
    assert!(
        matches!(
            err,
            Error::IndexNotFound { ref collection, ref name }
            if collection == COLLECTION && name == "no_such_index"
        ),
        "expected IndexNotFound; got {err:?}",
    );
}

#[test]
fn snapshot_index_descriptor_errors_for_dropped_pending_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("sid_dropped.obj")).expect("open");

    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1()))
        .expect("declare v1 indexes");
    // Re-declare at v2 WITHOUT by_status: the omitted Active index is
    // flipped to DroppedPending (tombstoned, not removed), so the
    // descriptor entry still exists but must not resolve.
    let v2_specs = vec![IndexSpec::unique(BY_EMAIL, "email").expect("unique spec")];
    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V2, &v2_specs))
        .expect("v2 re-declare drops by_status");

    let err = db
        .read_transaction(|tx| tx.snapshot_index_descriptor(COLLECTION, BY_STATUS))
        .expect_err("DroppedPending index must not resolve");
    assert!(
        matches!(
            err,
            Error::IndexNotFound { ref collection, ref name }
            if collection == COLLECTION && name == BY_STATUS
        ),
        "expected IndexNotFound for DroppedPending index; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// index_range_raw_with_version
// ---------------------------------------------------------------------------

#[test]
fn index_range_raw_with_version_round_trips_versions() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("irrwv_rt.obj")).expect("open");
    let (id_a, id_b) = seed_two_indexed_docs(&db);

    db.read_transaction(|tx| {
        let rows = tx.index_range_raw_with_version(
            COLLECTION,
            BY_STATUS,
            Bound::Unbounded,
            Bound::Unbounded,
        )?;
        assert_eq!(rows.len(), 2, "full range yields both docs");
        // Rows arrive in index-key order: "alpha" < "beta".
        assert_eq!(rows[0].0, id_a, "first row is the alpha doc");
        assert_eq!(rows[0].1, 3, "alpha doc carries its stored version");
        assert_eq!(rows[0].2.as_slice(), b"payload-a");
        assert_eq!(rows[1].0, id_b, "second row is the beta doc");
        assert_eq!(rows[1].1, 5, "beta doc carries its stored version");
        assert_eq!(rows[1].2.as_slice(), b"payload-b");
        Ok(())
    })
    .expect("versioned range read-back");
}

#[test]
fn index_range_raw_with_version_bounded_range_filters() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("irrwv_bound.obj")).expect("open");
    let (id_a, _id_b) = seed_two_indexed_docs(&db);

    db.read_transaction(|tx| {
        let key = skey("alpha");
        let rows = tx.index_range_raw_with_version(
            COLLECTION,
            BY_STATUS,
            Bound::Included(key.clone()),
            Bound::Included(key),
        )?;
        assert_eq!(rows.len(), 1, "single-key range yields one doc");
        assert_eq!(rows[0].0, id_a);
        assert_eq!(rows[0].1, 3);
        Ok(())
    })
    .expect("bounded versioned range");
}

// ---------------------------------------------------------------------------
// index_range_raw / count_index_range_raw error arms + count round-trip
// ---------------------------------------------------------------------------

#[test]
fn index_range_raw_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("irr_nocol.obj")).expect("open");

    let err = db
        .read_transaction(|tx| {
            tx.index_range_raw("ghost", BY_STATUS, Bound::Unbounded, Bound::Unbounded)
        })
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn index_range_raw_errors_for_unknown_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("irr_noidx.obj")).expect("open");

    // Collection exists, but no index was ever declared on it.
    db.transaction(|tx| tx.insert_raw_bytes(COLLECTION, b"x").map(|_| ()))
        .expect("seed collection");

    let err = db
        .read_transaction(|tx| {
            tx.index_range_raw(COLLECTION, "no_such", Bound::Unbounded, Bound::Unbounded)
        })
        .expect_err("unknown index must error");
    assert!(
        matches!(
            err,
            Error::IndexNotFound { ref collection, ref name }
            if collection == COLLECTION && name == "no_such"
        ),
        "expected IndexNotFound; got {err:?}",
    );
}

#[test]
fn count_index_range_raw_counts_entries() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("cirr.obj")).expect("open");
    let _ids = seed_two_indexed_docs(&db);

    db.read_transaction(|tx| {
        let all =
            tx.count_index_range_raw(COLLECTION, BY_STATUS, Bound::Unbounded, Bound::Unbounded)?;
        assert_eq!(all, 2, "full range counts both index entries");
        let key = skey("alpha");
        let one = tx.count_index_range_raw(
            COLLECTION,
            BY_STATUS,
            Bound::Included(key.clone()),
            Bound::Included(key),
        )?;
        assert_eq!(one, 1, "single-key range counts one entry");
        Ok(())
    })
    .expect("count_index_range_raw");
}

// ---------------------------------------------------------------------------
// find_unique_raw / find_unique_with_version
// ---------------------------------------------------------------------------

#[test]
fn find_unique_raw_on_standard_index_errors_index_not_unique() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("fur_notuniq.obj")).expect("open");
    let _ids = seed_two_indexed_docs(&db);

    let err = db
        .read_transaction(|tx| tx.find_unique_raw(COLLECTION, BY_STATUS, &skey("alpha")))
        .expect_err("find_unique against a Standard index must error");
    assert!(
        matches!(
            err,
            Error::IndexNotUnique { ref collection, ref name }
            if collection == COLLECTION && name == BY_STATUS
        ),
        "expected IndexNotUnique; got {err:?}",
    );
}

#[test]
fn find_unique_with_version_returns_stored_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("fuwv.obj")).expect("open");
    let (id_a, _id_b) = seed_two_indexed_docs(&db);

    db.read_transaction(|tx| {
        let hit = tx.find_unique_with_version(COLLECTION, BY_EMAIL, &skey("a@x.test"))?;
        let (id, payload, version) = hit.expect("unique hit present");
        assert_eq!(id, id_a, "id matches the inserted doc");
        assert_eq!(payload.as_slice(), b"payload-a");
        assert_eq!(version, 3, "stored type_version surfaced");

        let miss = tx.find_unique_with_version(COLLECTION, BY_EMAIL, &skey("nobody@x.test"))?;
        assert!(miss.is_none(), "absent unique key yields None");
        Ok(())
    })
    .expect("find_unique_with_version");
}

// ---------------------------------------------------------------------------
// snapshot_descriptor
// ---------------------------------------------------------------------------

#[test]
fn snapshot_descriptor_resolves_existing_and_none_for_missing() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("sd.obj")).expect("open");

    db.transaction(|tx| tx.insert_raw_bytes(COLLECTION, b"x").map(|_| ()))
        .expect("seed collection");

    db.read_transaction(|tx| {
        let present = tx.snapshot_descriptor(COLLECTION)?;
        assert!(present.is_some(), "existing collection resolves");
        let absent = tx.snapshot_descriptor("ghost")?;
        assert!(absent.is_none(), "missing collection yields None");
        Ok(())
    })
    .expect("snapshot_descriptor");
}

// ---------------------------------------------------------------------------
// WriteTxn raw read helpers: get_raw_bytes / count_all_raw / get_schema
// ---------------------------------------------------------------------------

#[test]
fn write_txn_get_raw_bytes_round_trips_and_yields_none_for_absent_id() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("wtx_grb.obj")).expect("open");

    db.transaction(|tx| {
        let id = tx.insert_raw_bytes(COLLECTION, b"live-payload")?;
        let got = tx.get_raw_bytes(COLLECTION, id)?;
        assert_eq!(
            got.as_deref(),
            Some(b"live-payload".as_slice()),
            "uncommitted insert visible to the write txn's own read"
        );
        let absent = Id::try_new(99_999).expect("nonzero");
        assert!(tx.get_raw_bytes(COLLECTION, absent)?.is_none());
        Ok(())
    })
    .expect("write-side get_raw_bytes");
}

#[test]
fn write_txn_count_all_raw_counts_uncommitted_inserts() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("wtx_count.obj")).expect("open");

    db.transaction(|tx| {
        for payload in [b"a".as_slice(), b"b".as_slice(), b"c".as_slice()] {
            let _ = tx.insert_raw_bytes(COLLECTION, payload)?;
        }
        assert_eq!(
            tx.count_all_raw(COLLECTION)?,
            3,
            "count reflects this txn's uncommitted inserts"
        );
        Ok(())
    })
    .expect("write-side count_all_raw");
}

#[test]
fn write_txn_count_all_raw_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("wtx_count_miss.obj")).expect("open");

    let err = db
        .transaction(|tx| tx.count_all_raw("ghost"))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn write_txn_get_schema_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("wtx_schema_miss.obj")).expect("open");

    let err = db
        .transaction(|tx| tx.get_schema("ghost", 1))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// delete_raw_bytes
// ---------------------------------------------------------------------------

#[test]
fn delete_raw_bytes_returns_true_then_false() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("drb.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_raw_bytes(COLLECTION, b"doomed"))
        .expect("insert");

    db.transaction(|tx| {
        assert!(
            tx.delete_raw_bytes(COLLECTION, id)?,
            "existing doc deletes as true"
        );
        assert!(
            !tx.delete_raw_bytes(COLLECTION, id)?,
            "second delete of the same id is false"
        );
        Ok(())
    })
    .expect("delete_raw_bytes");
}

#[test]
fn delete_raw_bytes_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("drb_nocol.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.delete_raw_bytes("ghost", id))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn delete_raw_bytes_rejects_namespaced_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("drb_ns.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.delete_raw_bytes("ns.items", id))
        .expect_err("namespaced delete must be rejected");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly { ref namespace, ref collection }
            if namespace == "ns" && collection == "items"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// update_raw_bytes / upsert_raw_bytes / upsert_with_version
// ---------------------------------------------------------------------------

#[test]
fn update_raw_bytes_replaces_payload_and_stamps_raw_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("urb.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_with_version(COLLECTION, b"before", 9))
        .expect("insert");

    db.transaction(|tx| tx.update_raw_bytes(COLLECTION, id, b"after"))
        .expect("update_raw_bytes");

    db.read_transaction(|tx| {
        let (payload, version) = tx
            .get_with_version(COLLECTION, id)?
            .expect("document present");
        assert_eq!(payload.as_slice(), b"after", "payload replaced");
        assert_eq!(version, 1, "update_raw_bytes stamps RAW_BYTES_TYPE_VERSION");
        Ok(())
    })
    .expect("read after update");
}

#[test]
fn upsert_with_version_inserts_then_replaces() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uwv.obj")).expect("open");

    let id = Id::try_new(7).expect("nonzero");
    db.transaction(|tx| {
        // First upsert lazy-creates the collection and inserts.
        tx.upsert_with_version(COLLECTION, id, b"first", 4)?;
        // Second upsert replaces in place.
        tx.upsert_with_version(COLLECTION, id, b"second", 6)
    })
    .expect("upserts");

    db.read_transaction(|tx| {
        let (payload, version) = tx
            .get_with_version(COLLECTION, id)?
            .expect("document present");
        assert_eq!(payload.as_slice(), b"second", "second upsert wins");
        assert_eq!(version, 6, "version follows the latest upsert");
        Ok(())
    })
    .expect("read after upserts");
}

#[test]
fn upsert_raw_bytes_stamps_raw_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("urbts.obj")).expect("open");

    let id = Id::try_new(3).expect("nonzero");
    db.transaction(|tx| tx.upsert_raw_bytes(COLLECTION, id, b"raw-upsert"))
        .expect("upsert_raw_bytes");

    db.read_transaction(|tx| {
        let (payload, version) = tx
            .get_with_version(COLLECTION, id)?
            .expect("document present");
        assert_eq!(payload.as_slice(), b"raw-upsert");
        assert_eq!(version, 1, "upsert_raw_bytes stamps RAW_BYTES_TYPE_VERSION");
        Ok(())
    })
    .expect("read after raw upsert");
}

#[test]
fn upsert_with_version_rejects_namespaced_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uwv_ns.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.upsert_with_version("ns.items", id, b"x", 1))
        .expect_err("namespaced upsert must be rejected");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly { ref namespace, ref collection }
            if namespace == "ns" && collection == "items"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// Indexed raw writes: unknown / dropped index, update / delete churn
// ---------------------------------------------------------------------------

#[test]
fn insert_raw_indexed_errors_for_unknown_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("iri_noidx.obj")).expect("open");

    let err = db
        .transaction(|tx| {
            tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1())?;
            tx.insert_raw_indexed(COLLECTION, b"x", 1, &[("no_such".to_owned(), skey("v"))])
                .map(|_| ())
        })
        .expect_err("unknown index name must error");
    assert!(
        matches!(
            err,
            Error::IndexNotFound { ref collection, ref name }
            if collection == COLLECTION && name == "no_such"
        ),
        "expected IndexNotFound; got {err:?}",
    );
}

#[test]
fn insert_raw_indexed_errors_for_dropped_pending_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("iri_dropped.obj")).expect("open");

    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V1, &specs_v1()))
        .expect("declare v1 indexes");
    let v2_specs = vec![IndexSpec::unique(BY_EMAIL, "email").expect("unique spec")];
    db.transaction(|tx| tx.reconcile_indexes_raw(COLLECTION, V2, &v2_specs))
        .expect("v2 re-declare drops by_status");

    let err = db
        .transaction(|tx| {
            tx.insert_raw_indexed(
                COLLECTION,
                b"x",
                1,
                &[(BY_STATUS.to_owned(), skey("alpha"))],
            )
            .map(|_| ())
        })
        .expect_err("DroppedPending index must be refused");
    assert!(
        matches!(
            err,
            Error::IndexNotFound { ref collection, ref name }
            if collection == COLLECTION && name == BY_STATUS
        ),
        "expected IndexNotFound for DroppedPending index; got {err:?}",
    );
}

#[test]
fn update_raw_indexed_moves_index_entries() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uri.obj")).expect("open");
    let (id_a, _id_b) = seed_two_indexed_docs(&db);

    db.transaction(|tx| {
        tx.update_raw_indexed(
            COLLECTION,
            id_a,
            b"payload-a2",
            7,
            &[(BY_STATUS.to_owned(), skey("alpha"))],
            &[(BY_STATUS.to_owned(), skey("gamma"))],
        )
    })
    .expect("update_raw_indexed");

    db.read_transaction(|tx| {
        let old_key = skey("alpha");
        let old = tx.index_range_raw(
            COLLECTION,
            BY_STATUS,
            Bound::Included(old_key.clone()),
            Bound::Included(old_key),
        )?;
        assert!(old.is_empty(), "old index entry removed");

        let new_key = skey("gamma");
        let rows = tx.index_range_raw_with_version(
            COLLECTION,
            BY_STATUS,
            Bound::Included(new_key.clone()),
            Bound::Included(new_key),
        )?;
        assert_eq!(rows.len(), 1, "doc visible under the new key");
        assert_eq!(rows[0].0, id_a);
        assert_eq!(rows[0].1, 7, "updated type_version surfaced");
        assert_eq!(rows[0].2.as_slice(), b"payload-a2");
        Ok(())
    })
    .expect("read after indexed update");
}

#[test]
fn delete_raw_indexed_removes_primary_and_index_entry() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("dri.obj")).expect("open");
    let (id_a, _id_b) = seed_two_indexed_docs(&db);

    let removed = db
        .transaction(|tx| {
            tx.delete_raw_indexed(
                COLLECTION,
                id_a,
                &[
                    (BY_EMAIL.to_owned(), skey("a@x.test")),
                    (BY_STATUS.to_owned(), skey("alpha")),
                ],
            )
        })
        .expect("delete_raw_indexed");
    assert!(removed, "existing doc reports true");

    db.read_transaction(|tx| {
        assert!(
            tx.get_raw_bytes(COLLECTION, id_a)?.is_none(),
            "primary record gone"
        );
        let key = skey("alpha");
        let count = tx.count_index_range_raw(
            COLLECTION,
            BY_STATUS,
            Bound::Included(key.clone()),
            Bound::Included(key),
        )?;
        assert_eq!(count, 0, "index entry gone");
        Ok(())
    })
    .expect("read after indexed delete");

    let removed_again = db
        .transaction(|tx| tx.delete_raw_indexed(COLLECTION, id_a, &[]))
        .expect("second delete");
    assert!(!removed_again, "absent doc reports false");
}
