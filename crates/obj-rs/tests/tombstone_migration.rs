//! Tombstone semantics for removed fields.
//!
//! Pattern under test:
//!
//! - v1 stores `{a, b, c}`.
//! - v2 stores `{a, c, d}` (drops `b`, adds `d`).
//! - The migrate body calls `doc.remove("b")` and `doc.set("d", ...)`
//!   before `doc.deserialize()` to construct the v2 value.
//!
//! Confirms:
//!
//! 1. `Dynamic::remove` strips the field; `Dynamic::deserialize`
//!    via postcard accepts the resulting Map.
//! 2. The end-to-end `Db::get` of a v1-on-disk record through the
//!    v2 type returns the migrated value.

#![forbid(unsafe_code)]

use obj_core::codec::{Dynamic, DynamicSchema};
use obj_core::{Document, Result};
use serde::{Deserialize, Serialize};

mod v1 {
    use super::{Deserialize, Document, Serialize};
    use obj_core::codec::{DynamicSchema, Schema};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Row {
        pub a: u32,
        pub b: String,
        pub c: u32,
    }

    impl Document for Row {
        const COLLECTION: &'static str = "tombstone_rows";
        const VERSION: u32 = 1;
    }

    impl Schema for Row {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("a", DynamicSchema::U64),
                ("b", DynamicSchema::String),
                ("c", DynamicSchema::U64),
            ])
        }
    }
}

mod v2 {
    use super::{Deserialize, Document, Dynamic, DynamicSchema, Result, Serialize};
    use obj_core::Error;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Row {
        pub a: u32,
        pub c: u32,
        pub d: String,
    }

    impl Document for Row {
        const COLLECTION: &'static str = "tombstone_rows";
        const VERSION: u32 = 2;

        fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
            vec![(
                1,
                DynamicSchema::map([
                    ("a", DynamicSchema::U64),
                    ("b", DynamicSchema::String),
                    ("c", DynamicSchema::U64),
                ]),
            )]
        }

        fn migrate(mut dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let removed = dynamic.remove("b")?;
            assert!(removed.is_some(), "v1 payload must carry `b`");
            dynamic.set("d", "<migrated>");
            let a = match dynamic.get("a") {
                Some(Dynamic::U64(n)) => {
                    u32::try_from(*n).map_err(|_| Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    })?
                }
                _ => {
                    return Err(Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    });
                }
            };
            let c = match dynamic.get("c") {
                Some(Dynamic::U64(n)) => {
                    u32::try_from(*n).map_err(|_| Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    })?
                }
                _ => {
                    return Err(Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    });
                }
            };
            let d = dynamic.get_str("d")?.to_owned();
            Ok(Row { a, c, d })
        }
    }
}

#[test]
fn migration_drops_b_adds_d() {
    use tempfile::TempDir;
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("tombstone.obj");

    let id;
    {
        let db = obj::Db::open(&path).expect("open");
        id = db
            .insert(v1::Row {
                a: 11,
                b: "to-be-dropped".to_owned(),
                c: 33,
            })
            .expect("insert v1");
    }

    {
        let db = obj::Db::open(&path).expect("reopen");
        let migrated: v2::Row = db.get(id).expect("get").expect("present");
        assert_eq!(
            migrated,
            v2::Row {
                a: 11,
                c: 33,
                d: "<migrated>".to_owned(),
            }
        );
    }
}

#[test]
fn remove_on_non_map_in_migrate_is_dynamic_path_not_map() {
    let mut value = Dynamic::U64(5);
    let err = value.remove("anything").expect_err("non-map");
    assert!(matches!(err, obj_core::Error::DynamicPathNotMap { .. }));
}
