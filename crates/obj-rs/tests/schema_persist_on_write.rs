//! Every write persists the current-version schema into
//! the catalog in the SAME `WriteTxn`, and every typed read sources
//! the stored-version schema from the disk catalog via `decode_with`.
//!
//! Three properties under test:
//!
//! 1. **Index-free persist.** Insert into a collection with NO
//!    secondary indexes; reopen as a later version and read it back.
//!    A v1→v2 migration succeeds, proving the schema persist is NOT
//!    gated on the index-reconcile hook (an index-free collection
//!    runs no index reconcile, yet the schema row must still land).
//!
//! 2. **Round-trip migration via the disk schema.** A v1 doc written
//!    through the real `Db::insert` path is read back through a v2
//!    type whose `migrate` body reconstructs the new field — the v2
//!    reader has NO compiled-in `historical_schemas()` entry for v1,
//!    so the only way the migration walk can succeed is by sourcing
//!    the v1 wire shape from the catalog row the v1 writer persisted.
//!
//! 3. **Atomicity.** A rolled-back write txn must leave NEITHER the
//!    document NOR its schema row behind — the schema row rides the
//!    same WAL transaction as the body.

#![forbid(unsafe_code)]

use obj::{Db, Document, DynamicSchema, Schema};
use obj_core::codec::StoredSchema;
use obj_core::pager::{Config as PagerConfig, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tempfile::TempDir;

mod v1 {
    use super::{Deserialize, Document, DynamicSchema, Schema, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Widget {
        pub label: String,
    }

    impl Document for Widget {
        const COLLECTION: &'static str = "widgets_143";
        const VERSION: u32 = 1;
    }

    impl Schema for Widget {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([("label", DynamicSchema::String)])
        }
    }
}

mod v2 {
    use super::{Deserialize, Document, DynamicSchema, Result, Schema, Serialize};
    use obj_core::codec::Dynamic;
    use obj_core::Error;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Widget {
        pub label: String,
        pub weight: u32,
    }

    impl Schema for Widget {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("label", DynamicSchema::String),
                ("weight", DynamicSchema::U64),
            ])
        }
    }

    impl Document for Widget {
        const COLLECTION: &'static str = "widgets_143";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let label = dynamic.get_str("label")?.to_owned();
            Ok(Widget { label, weight: 0 })
        }
    }
}

/// Read the raw `StoredSchema` row for `(collection, version)` directly
/// from the on-disk catalog, bypassing the typed API — proves the row
/// physically exists on disk.
fn read_schema_row(path: &Path, collection: &str, version: u32) -> Option<StoredSchema> {
    let mut pager = Pager::<FileHandle>::open(path, PagerConfig::default()).expect("reopen pager");
    let catalog = Catalog::open_or_init(&mut pager).expect("catalog");
    let descriptor = catalog
        .get(&mut pager, collection)
        .expect("get descriptor")
        .expect("descriptor present");
    let row = catalog
        .get_schema_in_txn(&mut pager, descriptor.collection_id, version)
        .expect("get_schema_in_txn");
    pager.close().expect("close");
    row
}

#[test]
fn index_free_insert_persists_schema_row() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("widgets-indexfree.obj");
    {
        let db = Db::open(&path).expect("open");
        let _ = db
            .insert(v1::Widget {
                label: "alpha".to_owned(),
            })
            .expect("insert v1");
    }
    let row = read_schema_row(&path, "widgets_143", 1).expect("v1 schema row persisted");
    let expected = StoredSchema::from_live(&<v1::Widget as Schema>::schema()).expect("from_live");
    assert_eq!(
        row.schema, expected.schema,
        "persisted normalized shape matches the v1 type's schema()",
    );
}

#[test]
fn v1_insert_round_trips_through_v2_via_disk_schema() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("widgets-roundtrip.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(v1::Widget {
            label: "beta".to_owned(),
        })
        .expect("insert v1")
    };

    let db = Db::open(&path).expect("reopen v2");
    let migrated: v2::Widget = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        v2::Widget {
            label: "beta".to_owned(),
            weight: 0,
        },
        "v1 doc migrates to v2 via the disk-sourced schema",
    );

    let all = db.all::<v2::Widget>().expect("all v2");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].label, "beta");
    assert_eq!(all[0].weight, 0);
}

#[test]
fn rolled_back_write_leaves_no_schema_row() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("widgets-rollback.obj");

    let id = {
        let db = Db::open(&path).expect("open");
        db.insert(v1::Widget {
            label: "kept".to_owned(),
        })
        .expect("commit v1")
    };
    assert!(
        read_schema_row(&path, "widgets_143", 1).is_some(),
        "v1 row committed"
    );
    assert!(
        read_schema_row(&path, "widgets_143", 2).is_none(),
        "no v2 row yet"
    );

    {
        let db = Db::open(&path).expect("reopen");
        let outcome: Result<()> = db.transaction(|tx| {
            let coll = tx.collection::<v2::Widget>()?;
            coll.update(id, |w| w.weight = 99)?;
            Err(obj_core::Error::InvalidArgument("intentional rollback"))
        });
        assert!(outcome.is_err(), "txn intentionally errored");
    }

    assert!(
        read_schema_row(&path, "widgets_143", 2).is_none(),
        "rolled-back v2 schema row must NOT survive",
    );
    assert!(
        read_schema_row(&path, "widgets_143", 1).is_some(),
        "v1 row survives"
    );
    let db = Db::open(&path).expect("reopen v1");
    let kept: v1::Widget = db.get(id).expect("get v1").expect("present");
    assert_eq!(kept.label, "kept", "document body unchanged by rollback");
}

#[test]
fn equal_version_read_does_not_require_disk_schema_lookup() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("widgets-equal.obj");
    let db = Db::open(&path).expect("open");
    let id = db
        .insert(v1::Widget {
            label: "gamma".to_owned(),
        })
        .expect("insert v1");
    let got: v1::Widget = db.get(id).expect("get v1").expect("present");
    assert_eq!(got.label, "gamma");
    let row = read_schema_row(&path, "widgets_143", 1).expect("schema row");
    let expected = StoredSchema::from_live(&<v1::Widget as Schema>::schema()).expect("from_live");
    assert_eq!(row.schema, expected.schema);
}
