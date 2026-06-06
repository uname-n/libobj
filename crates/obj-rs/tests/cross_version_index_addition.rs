//! Cross-version index ADDITION regression through the typed
//! `#[derive(Document)]` path, in ONE process / one `Db`.
//!
//! The engine's index-reconcile skip-cache was keyed by collection
//! NAME only. `Catalog::reconcile_indexes` is a FULL reconcile (declare
//! missing + drop Active indexes absent from the spec set), so once a
//! process reconciled a collection at schema version 1, a LATER version
//! that ADDED an index never reconciled — the new index never became
//! `Active` and the next index-maintaining write failed with
//! `IndexNotFound`. The fix keys the cache by `(collection, version)`,
//! so each version reconciles its own spec set exactly once.
//!
//! Sequence (single `Db`, single process):
//!
//! 1. Use type `V1` (collection `c`, VERSION 1, `indexes() = {by_email}`);
//!    insert one `V1` doc.
//! 2. Use type `V2` (same collection `c`, VERSION 2,
//!    `indexes() = {by_email, by_status}` — `by_status` is NEW); insert
//!    one `V2` doc.
//! 3. Assert the added `by_status` index is `Active` and queryable: a
//!    `find_unique` / `index_range` on it returns the `V2` doc. Before
//!    the fix this failed at step 2 with `IndexNotFound`.

#![forbid(unsafe_code)]

use obj::{Db, Document, IndexSpec};
use obj_core::codec::{Dynamic, DynamicSchema, Schema};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

const COLLECTION: &str = "c";

fn email_status_schema() -> DynamicSchema {
    DynamicSchema::map([
        ("email", DynamicSchema::String),
        ("status", DynamicSchema::String),
    ])
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct V1 {
    email: String,
    status: String,
}

impl Schema for V1 {
    fn schema() -> DynamicSchema {
        email_status_schema()
    }
}

impl Document for V1 {
    const COLLECTION: &'static str = COLLECTION;
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_email", "email").expect("unique by_email")]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct V2 {
    email: String,
    status: String,
}

impl Schema for V2 {
    fn schema() -> DynamicSchema {
        email_status_schema()
    }
}

impl Document for V2 {
    const COLLECTION: &'static str = COLLECTION;
    const VERSION: u32 = 2;

    fn indexes() -> Vec<IndexSpec> {
        vec![
            IndexSpec::unique("by_email", "email").expect("unique by_email"),
            IndexSpec::standard("by_status", "status").expect("standard by_status"),
        ]
    }

    fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
        vec![(
            1,
            DynamicSchema::map([
                ("email", DynamicSchema::String),
                ("status", DynamicSchema::String),
            ]),
        )]
    }

    fn migrate(dynamic: Dynamic, _from_version: u32) -> obj::Result<Self> {
        let email = field_string(&dynamic, "email");
        let status = field_string(&dynamic, "status");
        Ok(V2 { email, status })
    }
}

/// Pull a string field out of a decoded `Dynamic::Map` (identity
/// migration helper). Missing / non-string fields default to empty —
/// the test records always carry both fields, so this is never hit.
fn field_string(dynamic: &Dynamic, key: &str) -> String {
    if let Dynamic::Map(entries) = dynamic {
        for (k, v) in entries {
            if k == key {
                if let Dynamic::String(s) = v {
                    return s.clone();
                }
            }
        }
    }
    String::new()
}

fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("cross-version.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

#[test]
fn cross_version_index_addition_typed_path() {
    let (db, _dir) = fresh_db();

    let _v1_id = db
        .insert(V1 {
            email: "v1@x.test".to_owned(),
            status: "active".to_owned(),
        })
        .expect("v1 insert reconciles {by_email}");

    let _v2_id = db
        .insert(V2 {
            email: "v2@x.test".to_owned(),
            status: "pending".to_owned(),
        })
        .expect("v2 insert reconciles the ADDED by_status index");

    let rows = db
        .read_transaction(|tx| {
            let coll = tx.collection::<V2>()?;
            let iter = coll.index_range(
                "by_status",
                Dynamic::String("pending".to_owned())..=Dynamic::String("pending".to_owned()),
            )?;
            let mut out = Vec::new();
            for item in iter {
                out.push(item?);
            }
            Ok(out)
        })
        .expect("range scan on the newly-added by_status index");
    assert_eq!(
        rows.len(),
        1,
        "v2 doc visible via the newly-added by_status index"
    );
    let (_key, doc) = &rows[0];
    assert_eq!(doc.email, "v2@x.test");
    assert_eq!(doc.status, "pending");

    let by_email = db
        .find_unique::<V2>("by_email", "v2@x.test".to_owned())
        .expect("find_unique by_email")
        .expect("v2 doc present via by_email");
    assert_eq!(by_email.status, "pending");
}
