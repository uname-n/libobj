//! Worked examples for the disk-backed schema catalog.
//!
//! Every example follows the same shape the real engine takes:
//!
//! 1. A **v1** type (often a plain `#[derive(obj::Document)]`) inserts a
//!    document through the real `Db::insert` path. That write persists
//!    the v1 `StoredSchema` row into the catalog B+tree in the same WAL
//!    transaction as the document.
//! 2. A **v2 / v3** type (same collection, higher `VERSION`, with a
//!    `Document::migrate` override but NO `historical_schemas()`) reads
//!    the v1 document back through `Db::get` / `Db::all`. The ONLY
//!    source of the old wire shape is the disk catalog row — the new
//!    type carries no compiled-in history.
//!
//! These pin four scenarios:
//!
//! - **Field removal** — v2 drops a field v1 carried.
//! - **Type change / signedness** — a field changes signedness across
//!   versions, exercising the `respecialize` signedness-hint path so
//!   the old (zigzag) bytes still decode correctly.
//! - **Multi-version jump** — a v1 doc read by a v3 type calls `migrate`
//!   exactly once with `from_version = 1` (never chained).
//! - **Large schema** — a type whose `StoredSchema` row approaches
//!   `MAX_INLINE_DOC`, exercising the byte-aware multi-way leaf split +
//!   overflow pages on the schema row itself.

#![forbid(unsafe_code)]
// allow: test-support code — `expect`/`?` panics and error returns are the test's
// own failure signal, not a documented public-API contract worth a doc section.
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
// allow: the worked-example structs intentionally share a field-name prefix to
// mirror the real-world wide schema; renaming them would obscure what the test pins.
#![allow(clippy::struct_field_names)]

use std::sync::atomic::{AtomicU32, Ordering};

use obj::{Db, Document, DynamicSchema, Schema};
use obj_core::codec::{Dynamic, StoredSchema, MAX_INLINE_DOC};
use obj_core::pager::{Config as PagerConfig, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tempfile::TempDir;

/// Read the raw `StoredSchema` row for `(collection, version)` straight
/// from the on-disk catalog, bypassing the typed API — proves the row
/// physically landed (and lets us measure its encoded size).
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

/// Pull a `u64` field out of a decoded `Dynamic::Map`, failing loudly
/// on a wrong / absent shape.
fn get_u64(dynamic: &Dynamic, field: &str) -> Result<u64> {
    match dynamic.get(field) {
        Some(Dynamic::U64(n)) => Ok(*n),
        _ => Err(Error::SchemaTypeMismatch {
            expected: "U64",
            found: "absent-or-wrong-shape",
            path: field.to_owned(),
        }),
    }
}

/// Pull an `i64` field out of a decoded `Dynamic::Map`.
fn get_i64(dynamic: &Dynamic, field: &str) -> Result<i64> {
    match dynamic.get(field) {
        Some(Dynamic::I64(n)) => Ok(*n),
        _ => Err(Error::SchemaTypeMismatch {
            expected: "I64",
            found: "absent-or-wrong-shape",
            path: field.to_owned(),
        }),
    }
}

fn get_str(dynamic: &Dynamic, field: &str) -> Result<String> {
    dynamic.get_str(field).map(ToOwned::to_owned)
}

mod removal {
    use super::{get_str, get_u64};
    use super::{Deserialize, Document, Dynamic, Error, Result, Schema, Serialize};
    use obj_core::codec::DynamicSchema;

    /// v1: `{ id, label, deprecated_note }`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "removal_demo")]
    pub struct ItemV1 {
        pub id: u64,
        pub label: String,
        pub deprecated_note: String,
    }

    /// v2: drops `deprecated_note`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct ItemV2 {
        pub id: u64,
        pub label: String,
    }

    impl Schema for ItemV2 {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([("id", DynamicSchema::U64), ("label", DynamicSchema::String)])
        }
    }

    impl Document for ItemV2 {
        const COLLECTION: &'static str = "removal_demo";
        const VERSION: u32 = 2;

        fn migrate(mut dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let dropped = dynamic.remove("deprecated_note")?;
            assert!(
                dropped.is_some(),
                "v1 payload must have carried deprecated_note",
            );
            Ok(ItemV2 {
                id: get_u64(&dynamic, "id")?,
                label: get_str(&dynamic, "label")?,
            })
        }
    }
}

#[test]
fn field_removal_round_trips_via_disk_schema() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("removal.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(removal::ItemV1 {
            id: 11,
            label: "widget".to_owned(),
            deprecated_note: "remove me".to_owned(),
        })
        .expect("insert v1")
    };

    let row = read_schema_row(&path, "removal_demo", 1).expect("v1 row");
    assert_eq!(
        row.schema,
        DynamicSchema::map([
            ("id", DynamicSchema::U64),
            ("label", DynamicSchema::String),
            ("deprecated_note", DynamicSchema::String),
        ]),
    );

    let db = Db::open(&path).expect("reopen v2");
    let migrated: removal::ItemV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        removal::ItemV2 {
            id: 11,
            label: "widget".to_owned(),
        },
    );
}

mod signedness {
    use super::{get_i64, get_u64};
    use super::{Deserialize, Document, Dynamic, Error, Result, Schema, Serialize};
    use obj_core::codec::DynamicSchema;

    /// v1: `delta` is a SIGNED i64. A negative value forces zigzag
    /// encoding on the wire — the case a naive plain-varint read would
    /// corrupt.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "signedness_demo")]
    pub struct AccountV1 {
        pub account_id: u64,
        pub delta: i64,
    }

    /// v2: `delta` is now an UNSIGNED u64; the migration takes the
    /// signed value's absolute magnitude. The point of the test is that
    /// the OLD bytes (zigzag) decode correctly via the disk schema's
    /// signedness hint, surfacing as `Dynamic::I64`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct AccountV2 {
        pub account_id: u64,
        pub delta_abs: u64,
    }

    impl Schema for AccountV2 {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("account_id", DynamicSchema::U64),
                ("delta_abs", DynamicSchema::U64),
            ])
        }
    }

    impl Document for AccountV2 {
        const COLLECTION: &'static str = "signedness_demo";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let account_id = get_u64(&dynamic, "account_id")?;
            let delta = get_i64(&dynamic, "delta")?;
            Ok(AccountV2 {
                account_id,
                delta_abs: delta.unsigned_abs(),
            })
        }
    }
}

#[test]
fn signedness_change_decodes_old_zigzag_bytes() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("signedness.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(signedness::AccountV1 {
            account_id: 5,
            delta: -1_000_000,
        })
        .expect("insert v1")
    };

    let row = read_schema_row(&path, "signedness_demo", 1).expect("v1 row");
    assert_eq!(
        row.schema,
        DynamicSchema::map([
            ("account_id", DynamicSchema::U64),
            ("delta", DynamicSchema::U64),
        ]),
        "stored shape is signedness-agnostic",
    );
    assert_eq!(
        row.int_signed,
        vec![false, true],
        "signedness hint records account_id=unsigned, delta=signed",
    );

    let db = Db::open(&path).expect("reopen v2");
    let migrated: signedness::AccountV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        signedness::AccountV2 {
            account_id: 5,
            delta_abs: 1_000_000,
        },
        "old signed (zigzag) bytes decoded correctly via the hint",
    );
}

/// Records every `migrate` invocation on the v3 type: call count and
/// the last `from_version` seen.
static V3_MIGRATE_CALLS: AtomicU32 = AtomicU32::new(0);
static V3_LAST_FROM_VERSION: AtomicU32 = AtomicU32::new(0);

mod multiversion {
    use super::{get_u64, V3_LAST_FROM_VERSION, V3_MIGRATE_CALLS};
    use super::{Deserialize, Document, Dynamic, Error, Result, Schema, Serialize};
    use obj_core::codec::DynamicSchema;
    use std::sync::atomic::Ordering;

    /// v1: a single counter field.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "multiversion_demo")]
    pub struct CounterV1 {
        pub n: u64,
    }

    /// v3: two new fields. Reads a v1 doc directly — there is no v2 row
    /// on disk, and none is needed: the jump goes straight from the
    /// stored version (1) to the current version (3).
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct CounterV3 {
        pub n: u64,
        pub doubled: u64,
        pub generation: u64,
    }

    impl Schema for CounterV3 {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("n", DynamicSchema::U64),
                ("doubled", DynamicSchema::U64),
                ("generation", DynamicSchema::U64),
            ])
        }
    }

    impl Document for CounterV3 {
        const COLLECTION: &'static str = "multiversion_demo";
        const VERSION: u32 = 3;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            V3_MIGRATE_CALLS.fetch_add(1, Ordering::SeqCst);
            V3_LAST_FROM_VERSION.store(from_version, Ordering::SeqCst);
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let n = get_u64(&dynamic, "n")?;
            Ok(CounterV3 {
                n,
                doubled: n * 2,
                generation: 3,
            })
        }
    }
}

#[test]
fn multi_version_jump_calls_migrate_once_with_from_version_1() {
    V3_MIGRATE_CALLS.store(0, Ordering::SeqCst);
    V3_LAST_FROM_VERSION.store(0, Ordering::SeqCst);

    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("multiversion.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(multiversion::CounterV1 { n: 21 })
            .expect("insert v1")
    };

    assert!(read_schema_row(&path, "multiversion_demo", 1).is_some());
    assert!(
        read_schema_row(&path, "multiversion_demo", 2).is_none(),
        "no v2 build ever wrote here; the jump needs only the v1 row",
    );

    let db = Db::open(&path).expect("reopen v3");
    let migrated: multiversion::CounterV3 = db.get(id).expect("get v3").expect("present");
    assert_eq!(
        migrated,
        multiversion::CounterV3 {
            n: 21,
            doubled: 42,
            generation: 3,
        },
    );

    assert_eq!(
        V3_MIGRATE_CALLS.load(Ordering::SeqCst),
        1,
        "migrate called EXACTLY once (no chaining through v2)",
    );
    assert_eq!(
        V3_LAST_FROM_VERSION.load(Ordering::SeqCst),
        1,
        "migrate called with from_version = stored version (1)",
    );
}

mod large {
    use super::{get_str, get_u64};
    use super::{Deserialize, Document, Dynamic, Error, Result, Schema, Serialize};
    use obj_core::codec::DynamicSchema;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "large_schema_demo")]
    pub struct WideV1 {
        pub very_long_descriptive_field_name_aaaaaaaa: u64,
        pub very_long_descriptive_field_name_bbbbbbbb: u64,
        pub very_long_descriptive_field_name_cccccccc: u64,
        pub very_long_descriptive_field_name_dddddddd: u64,
        pub very_long_descriptive_field_name_eeeeeeee: u64,
        pub very_long_descriptive_field_name_ffffffff: u64,
        pub very_long_descriptive_field_name_gggggggg: u64,
        pub very_long_descriptive_field_name_hhhhhhhh: u64,
        pub very_long_descriptive_field_name_iiiiiiii: u64,
        pub very_long_descriptive_field_name_jjjjjjjj: u64,
        pub very_long_descriptive_field_name_kkkkkkkk: String,
        pub very_long_descriptive_field_name_llllllll: String,
        pub very_long_descriptive_field_name_mmmmmmmm: String,
        pub very_long_descriptive_field_name_nnnnnnnn: String,
        pub very_long_descriptive_field_name_oooooooo: String,
        pub nested_substructure_with_a_long_name_pppp: Inner,
        pub nested_substructure_with_a_long_name_qqqq: Inner,
        pub nested_substructure_with_a_long_name_rrrr: Inner,
        pub nested_substructure_with_a_long_name_ssss: Inner,
        pub nested_substructure_with_a_long_name_tttt: Inner,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(collection = "large_schema_inner")]
    pub struct Inner {
        pub inner_field_one_with_a_reasonably_long_descriptive_name: u64,
        pub inner_field_two_with_a_reasonably_long_descriptive_name: u64,
        pub inner_field_three_with_a_reasonably_long_descriptive_name: u64,
        pub inner_field_four_with_a_reasonably_long_descriptive_name: u64,
        pub inner_field_five_with_a_reasonably_long_descriptive_name: u64,
        pub inner_field_six_with_a_reasonably_long_descriptive_name: String,
    }

    /// v2 simply tacks on a new field; the migration reads a
    /// representative subset of the wide v1 shape from disk.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct WideV2 {
        pub very_long_descriptive_field_name_aaaaaaaa: u64,
        pub very_long_descriptive_field_name_kkkkkkkk: String,
        pub schema_revision: u64,
    }

    impl Schema for WideV2 {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                (
                    "very_long_descriptive_field_name_aaaaaaaa",
                    DynamicSchema::U64,
                ),
                (
                    "very_long_descriptive_field_name_kkkkkkkk",
                    DynamicSchema::String,
                ),
                ("schema_revision", DynamicSchema::U64),
            ])
        }
    }

    impl Document for WideV2 {
        const COLLECTION: &'static str = "large_schema_demo";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            Ok(WideV2 {
                very_long_descriptive_field_name_aaaaaaaa: get_u64(
                    &dynamic,
                    "very_long_descriptive_field_name_aaaaaaaa",
                )?,
                very_long_descriptive_field_name_kkkkkkkk: get_str(
                    &dynamic,
                    "very_long_descriptive_field_name_kkkkkkkk",
                )?,
                schema_revision: 2,
            })
        }
    }

    impl WideV1 {
        /// A fully-populated instance for the round-trip.
        pub fn sample() -> Self {
            let inner = || Inner {
                inner_field_one_with_a_reasonably_long_descriptive_name: 1,
                inner_field_two_with_a_reasonably_long_descriptive_name: 2,
                inner_field_three_with_a_reasonably_long_descriptive_name: 3,
                inner_field_four_with_a_reasonably_long_descriptive_name: 4,
                inner_field_five_with_a_reasonably_long_descriptive_name: 5,
                inner_field_six_with_a_reasonably_long_descriptive_name: "n".to_owned(),
            };
            WideV1 {
                very_long_descriptive_field_name_aaaaaaaa: 100,
                very_long_descriptive_field_name_bbbbbbbb: 0,
                very_long_descriptive_field_name_cccccccc: 0,
                very_long_descriptive_field_name_dddddddd: 0,
                very_long_descriptive_field_name_eeeeeeee: 0,
                very_long_descriptive_field_name_ffffffff: 0,
                very_long_descriptive_field_name_gggggggg: 0,
                very_long_descriptive_field_name_hhhhhhhh: 0,
                very_long_descriptive_field_name_iiiiiiii: 0,
                very_long_descriptive_field_name_jjjjjjjj: 0,
                very_long_descriptive_field_name_kkkkkkkk: "kk".to_owned(),
                very_long_descriptive_field_name_llllllll: String::new(),
                very_long_descriptive_field_name_mmmmmmmm: String::new(),
                very_long_descriptive_field_name_nnnnnnnn: String::new(),
                very_long_descriptive_field_name_oooooooo: String::new(),
                nested_substructure_with_a_long_name_pppp: inner(),
                nested_substructure_with_a_long_name_qqqq: inner(),
                nested_substructure_with_a_long_name_rrrr: inner(),
                nested_substructure_with_a_long_name_ssss: inner(),
                nested_substructure_with_a_long_name_tttt: inner(),
            }
        }
    }
}

#[test]
fn large_schema_row_persists_and_migrates() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("large_schema.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(large::WideV1::sample()).expect("insert wide v1")
    };

    let row = read_schema_row(&path, "large_schema_demo", 1).expect("wide v1 row");
    let encoded = row.to_postcard_bytes().expect("encode row");
    assert!(
        encoded.len() > MAX_INLINE_DOC / 2,
        "wide schema row ({} bytes) should be a large fraction of \
         MAX_INLINE_DOC ({} bytes) to exercise the byte-aware split",
        encoded.len(),
        MAX_INLINE_DOC,
    );
    assert!(
        encoded.len() <= MAX_INLINE_DOC,
        "schema row ({} bytes) must stay inline (<= MAX_INLINE_DOC {})",
        encoded.len(),
        MAX_INLINE_DOC,
    );

    let db = Db::open(&path).expect("reopen v2");
    let migrated: large::WideV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        large::WideV2 {
            very_long_descriptive_field_name_aaaaaaaa: 100,
            very_long_descriptive_field_name_kkkkkkkk: "kk".to_owned(),
            schema_revision: 2,
        },
    );

    let all = db.all::<large::WideV2>().expect("all v2");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].very_long_descriptive_field_name_aaaaaaaa, 100,);
}
