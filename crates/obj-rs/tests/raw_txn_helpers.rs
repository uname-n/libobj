//! Integration tests for `WriteTxn` / `ReadTxn` raw / versioned helper
//! APIs that were not yet exercised by existing test files.
//!
//! Coverage targets (issue #8):
//!
//! 1. `insert_with_version` / `update_with_version` / `get_with_version` —
//!    the version-stamped engine entry points for typed bindings.
//! 2. `put_schema` / `get_schema` on both the write and read sides,
//!    including the `SchemaShapeChanged` error path.
//! 3. Namespaced-write rejection: `insert_with_version`,
//!    `update_with_version`, `put_schema` all reject a `"ns.col"`
//!    collection name with `AttachedDatabaseIsReadOnly`.
//! 4. Missing-collection / missing-document behaviour in the raw
//!    read / write helpers.

#![forbid(unsafe_code)]

use obj::{Db, DynamicSchema, Error, Id};
use tempfile::TempDir;

/// A reusable small payload that fits well within the inline cap.
const PAYLOAD_V1: &[u8] = b"version-one-payload";
/// A different payload for v2 updates.
const PAYLOAD_V2: &[u8] = b"version-two-payload";
const COLLECTION: &str = "items";

// ---------------------------------------------------------------------------
// insert_with_version / get_with_version / update_with_version
// ---------------------------------------------------------------------------

#[test]
fn insert_with_version_stamps_caller_supplied_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("iv.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_with_version(COLLECTION, PAYLOAD_V1, 7))
        .expect("insert_with_version commit");

    db.read_transaction(|tx| {
        let result = tx.get_with_version(COLLECTION, id)?;
        let (payload, version) = result.expect("document present");
        assert_eq!(payload.as_slice(), PAYLOAD_V1, "payload round-trips");
        assert_eq!(version, 7, "version stamp matches caller-supplied value");
        Ok(())
    })
    .expect("read_transaction");
}

#[test]
fn get_with_version_on_write_txn_reflects_uncommitted_inserts() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("wv.obj")).expect("open");

    db.transaction(|tx| {
        let id = tx.insert_with_version(COLLECTION, PAYLOAD_V1, 3)?;
        // Read back inside the same txn before commit.
        let result = tx.get_with_version(COLLECTION, id)?;
        let (payload, version) = result.expect("live-write visible within txn");
        assert_eq!(payload.as_slice(), PAYLOAD_V1);
        assert_eq!(version, 3);
        Ok(())
    })
    .expect("transaction");
}

#[test]
fn get_with_version_returns_none_for_absent_id() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("gv_miss.obj")).expect("open");

    // Create the collection by inserting a doc, then look up a bogus id.
    db.transaction(|tx| tx.insert_with_version(COLLECTION, PAYLOAD_V1, 1).map(|_| ()))
        .expect("seed");

    let absent_id = Id::try_new(99_999).expect("nonzero");
    db.read_transaction(|tx| {
        let result = tx.get_with_version(COLLECTION, absent_id)?;
        assert!(result.is_none(), "absent id yields None");
        Ok(())
    })
    .expect("read_transaction");
}

#[test]
fn update_with_version_replaces_payload_and_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uv.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_with_version(COLLECTION, PAYLOAD_V1, 1))
        .expect("insert");

    db.transaction(|tx| tx.update_with_version(COLLECTION, id, PAYLOAD_V2, 2))
        .expect("update_with_version");

    db.read_transaction(|tx| {
        let (payload, version) = tx
            .get_with_version(COLLECTION, id)?
            .expect("document present after update");
        assert_eq!(payload.as_slice(), PAYLOAD_V2, "payload updated");
        assert_eq!(version, 2, "version updated");
        Ok(())
    })
    .expect("read after update");
}

#[test]
fn update_with_version_errors_on_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uv_nocol.obj")).expect("open");

    let some_id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.update_with_version("nonexistent", some_id, PAYLOAD_V1, 1))
        .expect_err("update on missing collection must err");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "nonexistent"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn update_with_version_errors_on_missing_document() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("uv_nodoc.obj")).expect("open");

    // Create the collection but don't insert the target id.
    db.transaction(|tx| tx.insert_with_version(COLLECTION, PAYLOAD_V1, 1).map(|_| ()))
        .expect("seed collection");

    let absent_id = Id::try_new(88_888).expect("nonzero");
    let err = db
        .transaction(|tx| tx.update_with_version(COLLECTION, absent_id, PAYLOAD_V2, 2))
        .expect_err("update on absent id must err");
    // The implementation returns CollectionNotFound with "items#88888".
    assert!(
        matches!(err, Error::CollectionNotFound { .. }),
        "expected CollectionNotFound for absent id; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// put_schema / get_schema (WriteTxn side)
// ---------------------------------------------------------------------------

#[test]
fn put_and_get_schema_on_write_txn() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_rw.obj")).expect("open");

    let schema_v1 = DynamicSchema::map([("label", DynamicSchema::String)]);

    db.transaction(|tx| {
        // put_schema implicitly creates the collection.
        tx.put_schema(COLLECTION, 1, &schema_v1)?;
        // get_schema from the same txn should see it immediately.
        let got = tx.get_schema(COLLECTION, 1)?;
        assert!(got.is_some(), "put_schema visible in same txn via get_schema");
        Ok(())
    })
    .expect("transaction");
}

#[test]
fn get_schema_returns_none_for_missing_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_miss.obj")).expect("open");

    let schema_v1 = DynamicSchema::map([("x", DynamicSchema::U64)]);

    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_v1))
        .expect("put v1 schema");

    db.transaction(|tx| {
        // Version 99 was never persisted.
        let got = tx.get_schema(COLLECTION, 99)?;
        assert!(got.is_none(), "absent version yields None");
        Ok(())
    })
    .expect("transaction");
}

#[test]
fn put_schema_shape_changed_errors() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_conflict.obj")).expect("open");

    let schema_v1 = DynamicSchema::map([("label", DynamicSchema::String)]);
    let schema_conflict = DynamicSchema::map([("label", DynamicSchema::U64)]);

    // Persist the v1 schema.
    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_v1))
        .expect("first put_schema");

    // Attempt to put a DIFFERENT shape under the same (collection, version).
    let err = db
        .transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_conflict))
        .expect_err("shape conflict must error");
    assert!(
        matches!(err, Error::SchemaShapeChanged { .. }),
        "expected SchemaShapeChanged; got {err:?}",
    );
}

#[test]
fn put_schema_idempotent_same_shape() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_idem.obj")).expect("open");

    let schema = DynamicSchema::map([("count", DynamicSchema::U64)]);

    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema))
        .expect("first put_schema");

    // Second put with identical shape must be a no-op (no error).
    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema))
        .expect("idempotent put_schema must succeed");
}

// ---------------------------------------------------------------------------
// get_schema (ReadTxn side)
// ---------------------------------------------------------------------------

#[test]
fn get_schema_on_read_txn_sees_committed_schema() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_rtxn.obj")).expect("open");

    let schema_v1 = DynamicSchema::map([("name", DynamicSchema::String)]);

    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_v1))
        .expect("put schema");

    db.read_transaction(|tx| {
        let got = tx.get_schema(COLLECTION, 1)?;
        assert!(got.is_some(), "committed schema visible in ReadTxn");
        Ok(())
    })
    .expect("read_transaction");
}

#[test]
fn get_schema_on_read_txn_returns_none_for_absent_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_rtxn_miss.obj")).expect("open");

    let schema_v1 = DynamicSchema::map([("name", DynamicSchema::String)]);
    db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_v1))
        .expect("put schema");

    db.read_transaction(|tx| {
        let got = tx.get_schema(COLLECTION, 42)?;
        assert!(got.is_none(), "absent version yields None in ReadTxn");
        Ok(())
    })
    .expect("read_transaction");
}

#[test]
fn get_schema_on_read_txn_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("schema_rtxn_nocol.obj")).expect("open");

    // No collection created at all.
    let err = db
        .read_transaction(|tx| tx.get_schema("ghost", 1))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// Namespaced-write rejection
// ---------------------------------------------------------------------------

#[test]
fn insert_with_version_rejects_namespaced_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("ns_insert.obj")).expect("open");

    let err = db
        .transaction(|tx| tx.insert_with_version("ns.col", PAYLOAD_V1, 1))
        .expect_err("namespaced write must be rejected");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly {
                ref namespace,
                ref collection,
            }
            if namespace == "ns" && collection == "col"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

#[test]
fn update_with_version_rejects_namespaced_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("ns_update.obj")).expect("open");

    let some_id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.update_with_version("ext.items", some_id, PAYLOAD_V1, 1))
        .expect_err("namespaced update must be rejected");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly {
                ref namespace,
                ref collection,
            }
            if namespace == "ext" && collection == "items"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

#[test]
fn put_schema_rejects_namespaced_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("ns_schema.obj")).expect("open");

    let schema = DynamicSchema::map([("x", DynamicSchema::U64)]);
    let err = db
        .transaction(|tx| tx.put_schema("ro.items", 1, &schema))
        .expect_err("namespaced put_schema must be rejected");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly {
                ref namespace,
                ref collection,
            }
            if namespace == "ro" && collection == "items"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// Missing-collection behaviour for raw read helpers (ReadTxn)
// ---------------------------------------------------------------------------

#[test]
fn get_raw_bytes_on_read_txn_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("rbytes_miss.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .read_transaction(|tx| tx.get_raw_bytes("ghost", id))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

#[test]
fn get_with_version_on_read_txn_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("gwv_miss.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .read_transaction(|tx| tx.get_with_version("ghost", id))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "ghost"),
        "expected CollectionNotFound; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// WriteTxn get_with_version: missing collection
// ---------------------------------------------------------------------------

#[test]
fn get_with_version_on_write_txn_errors_for_missing_collection() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("gwv_wtxn_miss.obj")).expect("open");

    let id = Id::try_new(1).expect("nonzero");
    let err = db
        .transaction(|tx| tx.get_with_version("nonexistent", id))
        .expect_err("missing collection must error");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "nonexistent"),
        "expected CollectionNotFound; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// insert_with_version: DocumentTooLarge
// ---------------------------------------------------------------------------

#[test]
fn insert_with_version_errors_on_oversized_payload() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("too_large.obj")).expect("open");

    // MAX_INLINE_DOC = 3022; payload > 3006 bytes total-record would overflow.
    let huge: Vec<u8> = vec![0xAB; 3100];
    let err = db
        .transaction(|tx| tx.insert_with_version(COLLECTION, &huge, 1))
        .expect_err("oversized payload must error");
    assert!(
        matches!(err, Error::DocumentTooLarge { .. }),
        "expected DocumentTooLarge; got {err:?}",
    );
}

// ---------------------------------------------------------------------------
// put_schema / get_schema round-trip across txn boundary
// ---------------------------------------------------------------------------

#[test]
fn schema_survives_commit_and_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("schema_persist.obj");

    let schema_v1 = DynamicSchema::map([("score", DynamicSchema::U64)]);

    {
        let db = Db::open(&path).expect("open");
        db.transaction(|tx| tx.put_schema(COLLECTION, 1, &schema_v1))
            .expect("put schema");
    }

    // Reopen and verify the row survived.
    let db = Db::open(&path).expect("reopen");
    db.read_transaction(|tx| {
        let got = tx.get_schema(COLLECTION, 1)?;
        assert!(got.is_some(), "schema row persisted across reopen");
        Ok(())
    })
    .expect("read_transaction after reopen");
}

// ---------------------------------------------------------------------------
// insert_raw_bytes / get_raw_bytes (shim forwarding tested here too)
// ---------------------------------------------------------------------------

#[test]
fn insert_raw_bytes_and_get_raw_bytes_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("rb_rt.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_raw_bytes(COLLECTION, PAYLOAD_V1))
        .expect("insert_raw_bytes");

    db.read_transaction(|tx| {
        let got = tx.get_raw_bytes(COLLECTION, id)?;
        assert_eq!(
            got.as_deref(),
            Some(PAYLOAD_V1),
            "get_raw_bytes payload matches insert_raw_bytes"
        );
        Ok(())
    })
    .expect("read_transaction");
}

#[test]
fn insert_raw_bytes_forwards_raw_bytes_type_version() {
    let dir = TempDir::new().expect("tempdir");
    let db = Db::open(dir.path().join("rb_ver.obj")).expect("open");

    let id = db
        .transaction(|tx| tx.insert_raw_bytes(COLLECTION, PAYLOAD_V1))
        .expect("insert");

    // `insert_raw_bytes` is documented to forward to `insert_with_version`
    // with `type_version = RAW_BYTES_TYPE_VERSION` (= 1).
    db.read_transaction(|tx| {
        let (_, version) = tx
            .get_with_version(COLLECTION, id)?
            .expect("document present");
        assert_eq!(version, 1, "insert_raw_bytes stamps type_version = 1");
        Ok(())
    })
    .expect("read_transaction");
}
