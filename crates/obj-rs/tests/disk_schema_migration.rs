//! Disk-sourced schema migration through a DERIVED v1 `Document`.
//!
//! Historical schemas are persisted on disk on every write and sourced
//! from the catalog at migration time. This file pins that disk model
//! for the DERIVE path specifically:
//!
//! 1. A v1 type derives `Document` — which now ALWAYS derives `Schema`
//!    too. Inserting through `Db::insert` persists the derived v1
//!    schema row in the same write txn.
//! 2. A v2 type (same collection, higher version, with a `migrate`
//!    override but NO `historical_schemas()`) reads the v1 doc back
//!    through `Db::get`. The only source of the v1 wire shape is the
//!    disk catalog row the v1 writer persisted.
//!
//! The fully hand-written-`Document` variants of this flow are covered
//! by `schema_evolution.rs`, `lazy_migration.rs`, etc. Here the v1
//! type goes through `#[derive(obj::Document)]` so we ALSO pin that the
//! derived `Schema` impl produces a catalog row the migration walk can
//! consume.

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
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "orders_disk")]
    pub struct Order {
        pub customer_id: u64,
        pub total_cents: u64,
    }
}

mod v2 {
    use super::{Deserialize, Result, Schema, Serialize};
    use obj_core::codec::{Dynamic, DynamicSchema};
    use obj_core::{Document, Error};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Order {
        pub customer_id: u64,
        pub total_cents: u64,
        pub placed_at: u64,
    }

    impl Schema for Order {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("customer_id", DynamicSchema::U64),
                ("total_cents", DynamicSchema::U64),
                ("placed_at", DynamicSchema::U64),
            ])
        }
    }

    impl Document for Order {
        const COLLECTION: &'static str = "orders_disk";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let customer_id = read_u64(&dynamic, "customer_id")?;
            let total_cents = read_u64(&dynamic, "total_cents")?;
            Ok(Order {
                customer_id,
                total_cents,
                placed_at: 0,
            })
        }
    }

    /// Pull a `u64` field out of a decoded `Dynamic::Map`. The v1
    /// record always carries both fields, so the error arm is only a
    /// guard against a corrupt / unexpected wire shape.
    fn read_u64(dynamic: &Dynamic, field: &str) -> Result<u64> {
        match dynamic.get(field) {
            Some(Dynamic::U64(n)) => Ok(*n),
            _ => Err(Error::SchemaTypeMismatch {
                expected: "U64",
                found: "absent-or-wrong-shape",
                path: field.to_owned(),
            }),
        }
    }
}

/// Read the raw `StoredSchema` row for `(collection, version)` from the
/// on-disk catalog, bypassing the typed API — proves the derived
/// `Schema` row physically landed on disk.
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
fn derived_insert_persists_current_schema_row() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("orders-persist.obj");
    {
        let db = Db::open(&path).expect("open");
        let _ = db
            .insert(v1::Order {
                customer_id: 7,
                total_cents: 1_234,
            })
            .expect("insert v1");
    }
    let row = read_schema_row(&path, "orders_disk", 1).expect("v1 schema row persisted");
    let expected = StoredSchema::from_live(&<v1::Order as Schema>::schema()).expect("from_live");
    assert_eq!(
        row.schema, expected.schema,
        "persisted shape matches the DERIVED v1 Schema::schema()",
    );
    assert_eq!(
        <v1::Order as Schema>::schema(),
        DynamicSchema::map([
            ("customer_id", DynamicSchema::U64),
            ("total_cents", DynamicSchema::U64),
        ]),
    );
}

#[test]
fn v1_derived_doc_migrates_to_v2_via_disk_schema() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("orders-roundtrip.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(v1::Order {
            customer_id: 42,
            total_cents: 9_900,
        })
        .expect("insert v1")
    };

    assert!(
        read_schema_row(&path, "orders_disk", 1).is_some(),
        "v1 schema row persisted by the derived insert",
    );
    assert!(
        <v2::Order as Document>::historical_schemas().is_empty(),
        "v2 must NOT carry a compiled-in historical_schemas registry",
    );

    let db = Db::open(&path).expect("reopen v2");
    let migrated: v2::Order = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        v2::Order {
            customer_id: 42,
            total_cents: 9_900,
            placed_at: 0,
        },
        "v1 doc migrates to v2 via the disk-sourced schema",
    );

    let all = db.all::<v2::Order>().expect("all v2");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].customer_id, 42);
    assert_eq!(all[0].placed_at, 0);
}
