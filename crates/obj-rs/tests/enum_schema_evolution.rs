//! Enum-field and `Option<T>`-field schema migration through the
//! **disk-sourced** schema catalog.
//!
//! `enum_migration.rs` proves the `Dynamic::Enum` walk and a v1→v2
//! enum migration. This file pins the schema-catalog form of that
//! capability:
//!
//! 1. A v1 type carrying an enum field is written with a real
//!    `Db::insert`, which persists the v1 schema — including the enum —
//!    into the on-disk catalog. A v2 type (new field + extended enum)
//!    reads those records back. The codec sources the v1 schema from
//!    the catalog (`SchemaSource::Snapshot`), `respecialize`s it, walks
//!    the enum, and hands the `Dynamic::Enum` to v2's `migrate`. No
//!    `historical_schemas()` is declared — the disk is the source.
//!
//! 2. An `Option<u64>` field migrates the same way. `Option<T>` is a
//!    two-variant enum (`None`=0 / `Some`=1) under the hood, so the
//!    enum walk is exactly what makes Option-field migration work.
//!    Both the `Some` and `None` payloads are exercised.
//!
//! ## Schema impls: hand-written (Parts 1-2) AND derived (Part 3)
//!
//! The `#[derive(obj::Document)]` struct path maps a field's type to a
//! `DynamicSchema` *syntactically* for scalars, `Vec<T>`, and
//! `Option<T>` (lowered to the two-variant enum, recursing on the
//! inner type), and otherwise falls through to
//! `<FieldTy as Schema>::schema()`.
//!
//! Parts 1-2 hand-write their `Schema` impl, exactly as
//! `enum_migration.rs` does for its enum-bearing `Order`: Part 1
//! exercises a genuine user enum (no syntactic lowering exists for
//! arbitrary enums), and Part 2 uses a bare `Option<u64>` field that
//! falls through to `<Option<u64> as Schema>::schema()`, which needs
//! `u64: Schema`. Part 3 DERIVES `Document` with a bare `Option<u64>`
//! field — no hand-written `Schema` — and proves the derived schema
//! both migrates through the disk catalog and equals the Part-2
//! hand-written shape byte-for-byte. The v1 insert persists these
//! schemas to the catalog; the v2 read sources them from disk.

#![forbid(unsafe_code)]

use obj::{Db, DynamicSchema, EnumVariantSchema, Schema};
use obj_core::codec::Dynamic;
use obj_core::{Error, Id, Result};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

mod enum_v1 {
    use super::{Deserialize, DynamicSchema, Schema, Serialize};
    use obj_core::Document;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum Tier {
        Free,
        Pro { seats: u32 },
        Enterprise(String),
    }

    impl Schema for Tier {
        fn schema() -> DynamicSchema {
            super::tier_v1_schema()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Account {
        pub owner: String,
        pub tier: Tier,
    }

    impl Schema for Account {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("owner", DynamicSchema::String),
                ("tier", <Tier as Schema>::schema()),
            ])
        }
    }

    impl Document for Account {
        const COLLECTION: &'static str = "evolution_accounts";
        const VERSION: u32 = 1;
    }
}

/// v1 `Tier` schema — the exact shape `enum_v1::Account` persists on
/// insert. Declared at module scope so both the v1 `Schema` impl and
/// the v2 reader agree on the byte shape.
fn tier_v1_schema() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Free", DynamicSchema::Null),
        EnumVariantSchema::new(
            1,
            "Pro",
            DynamicSchema::map([("seats", DynamicSchema::U64)]),
        ),
        EnumVariantSchema::new(2, "Enterprise", DynamicSchema::String),
    ])
}

mod enum_v2 {
    use super::{Deserialize, Dynamic, DynamicSchema, Error, Result, Schema, Serialize};
    use obj_core::Document;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum Tier {
        Free,
        Pro { seats: u32 },
        Enterprise(String),
        Trial,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Account {
        pub owner: String,
        pub tier: Tier,
        pub region: String,
    }

    impl Schema for Account {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("owner", DynamicSchema::String),
                (
                    "tier",
                    DynamicSchema::enumeration([
                        super::EnumVariantSchema::new(0, "Free", DynamicSchema::Null),
                        super::EnumVariantSchema::new(
                            1,
                            "Pro",
                            DynamicSchema::map([("seats", DynamicSchema::U64)]),
                        ),
                        super::EnumVariantSchema::new(2, "Enterprise", DynamicSchema::String),
                        super::EnumVariantSchema::new(3, "Trial", DynamicSchema::Null),
                    ]),
                ),
                ("region", DynamicSchema::String),
            ])
        }
    }

    impl Document for Account {
        const COLLECTION: &'static str = "evolution_accounts";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let owner = dynamic.get_str("owner")?.to_owned();
            let tier_dyn = dynamic
                .get("tier")
                .ok_or_else(|| Error::SchemaTypeMismatch {
                    expected: "Enum",
                    found: "absent",
                    path: "tier".to_owned(),
                })?;
            let tier = decode_v1_tier(tier_dyn)?;
            Ok(Account {
                owner,
                tier,
                region: "unknown".to_owned(),
            })
        }
    }

    fn decode_v1_tier(value: &Dynamic) -> Result<Tier> {
        let variant = value
            .enum_variant()
            .ok_or_else(|| Error::SchemaTypeMismatch {
                expected: "Enum",
                found: "non-enum",
                path: "tier".to_owned(),
            })?;
        let payload = value
            .enum_payload()
            .ok_or_else(|| Error::SchemaTypeMismatch {
                expected: "Enum-payload",
                found: "absent",
                path: "tier".to_owned(),
            })?;
        match variant {
            "Free" => Ok(Tier::Free),
            "Pro" => {
                let seats = match payload.get("seats") {
                    Some(Dynamic::U64(n)) => {
                        u32::try_from(*n).map_err(|_| Error::SchemaTypeMismatch {
                            expected: "u32",
                            found: "out-of-range",
                            path: "tier.Pro.seats".to_owned(),
                        })?
                    }
                    _ => {
                        return Err(Error::SchemaTypeMismatch {
                            expected: "U64",
                            found: "absent-or-wrong-type",
                            path: "tier.Pro.seats".to_owned(),
                        })
                    }
                };
                Ok(Tier::Pro { seats })
            }
            "Enterprise" => match payload {
                Dynamic::String(name) => Ok(Tier::Enterprise(name.clone())),
                _ => Err(Error::SchemaTypeMismatch {
                    expected: "String",
                    found: "non-string",
                    path: "tier.Enterprise".to_owned(),
                }),
            },
            other => Err(Error::SchemaTypeMismatch {
                expected: "known variant",
                found: "unknown-variant",
                path: format!("tier.{other}"),
            }),
        }
    }
}

#[test]
fn enum_field_migrates_v1_to_v2_via_disk_catalog() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("evolution_enum.obj");

    let ids: Vec<Id> = {
        let db = Db::open(&path).expect("open v1");
        vec![
            db.insert(enum_v1::Account {
                owner: "alice".to_owned(),
                tier: enum_v1::Tier::Free,
            })
            .expect("insert free"),
            db.insert(enum_v1::Account {
                owner: "bob".to_owned(),
                tier: enum_v1::Tier::Pro { seats: 12 },
            })
            .expect("insert pro"),
            db.insert(enum_v1::Account {
                owner: "carol".to_owned(),
                tier: enum_v1::Tier::Enterprise("acme".to_owned()),
            })
            .expect("insert enterprise"),
        ]
    };

    let db = Db::open(&path).expect("reopen v2");
    let got: Vec<enum_v2::Account> = ids
        .iter()
        .map(|id| {
            db.get::<enum_v2::Account>(*id)
                .expect("get")
                .expect("present")
        })
        .collect();

    assert_eq!(got[0].owner, "alice");
    assert_eq!(got[0].tier, enum_v2::Tier::Free);
    assert_eq!(got[0].region, "unknown", "v1 → v2 defaults region");
    assert_eq!(got[1].owner, "bob");
    assert_eq!(got[1].tier, enum_v2::Tier::Pro { seats: 12 });
    assert_eq!(got[2].owner, "carol");
    assert_eq!(got[2].tier, enum_v2::Tier::Enterprise("acme".to_owned()));
}

/// `Option<u64>` schema — postcard's two-variant enum: `None`=0 (Null
/// payload) / `Some`=1 (the inner `u64`). This mirrors the
/// `Option<T>: Schema` blanket impl in obj-core exactly; we spell it
/// out so a *scalar* inner type (which has no `Schema` impl) still has
/// a concrete schema to persist.
fn optional_u64_schema() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "None", DynamicSchema::Null),
        EnumVariantSchema::new(1, "Some", DynamicSchema::U64),
    ])
}

mod opt_v1 {
    use super::{optional_u64_schema, Deserialize, DynamicSchema, Schema, Serialize};
    use obj_core::Document;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Profile {
        pub handle: String,
        pub badge: Option<u64>,
    }

    impl Schema for Profile {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("handle", DynamicSchema::String),
                ("badge", optional_u64_schema()),
            ])
        }
    }

    impl Document for Profile {
        const COLLECTION: &'static str = "evolution_profiles";
        const VERSION: u32 = 1;
    }
}

mod opt_v2 {
    use super::{
        optional_u64_schema, Deserialize, Dynamic, DynamicSchema, Error, Result, Schema, Serialize,
    };
    use obj_core::Document;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Profile {
        pub handle: String,
        pub badge: Option<u64>,
        pub verified: bool,
    }

    impl Schema for Profile {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("handle", DynamicSchema::String),
                ("badge", optional_u64_schema()),
                ("verified", DynamicSchema::Bool),
            ])
        }
    }

    impl Document for Profile {
        const COLLECTION: &'static str = "evolution_profiles";
        const VERSION: u32 = 2;

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let handle = dynamic.get_str("handle")?.to_owned();
            let badge_dyn = dynamic
                .get("badge")
                .ok_or_else(|| Error::SchemaTypeMismatch {
                    expected: "Enum",
                    found: "absent",
                    path: "badge".to_owned(),
                })?;
            let badge = decode_optional_u64(badge_dyn)?;
            Ok(Profile {
                handle,
                badge,
                verified: false,
            })
        }
    }

    fn decode_optional_u64(value: &Dynamic) -> Result<Option<u64>> {
        let variant = value
            .enum_variant()
            .ok_or_else(|| Error::SchemaTypeMismatch {
                expected: "Enum",
                found: "non-enum",
                path: "badge".to_owned(),
            })?;
        match variant {
            "None" => Ok(None),
            "Some" => match value.enum_payload() {
                Some(Dynamic::U64(n)) => Ok(Some(*n)),
                _ => Err(Error::SchemaTypeMismatch {
                    expected: "U64",
                    found: "absent-or-wrong-type",
                    path: "badge.Some".to_owned(),
                }),
            },
            other => Err(Error::SchemaTypeMismatch {
                expected: "None|Some",
                found: "unknown-variant",
                path: format!("badge.{other}"),
            }),
        }
    }
}

#[test]
fn option_field_migrates_both_some_and_none_via_disk_catalog() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("evolution_option.obj");

    let (some_id, none_id): (Id, Id) = {
        let db = Db::open(&path).expect("open v1");
        let some_id = db
            .insert(opt_v1::Profile {
                handle: "neo".to_owned(),
                badge: Some(99),
            })
            .expect("insert some");
        let none_id = db
            .insert(opt_v1::Profile {
                handle: "trinity".to_owned(),
                badge: None,
            })
            .expect("insert none");
        (some_id, none_id)
    };

    let db = Db::open(&path).expect("reopen v2");
    let some = db
        .get::<opt_v2::Profile>(some_id)
        .expect("get some")
        .expect("present");
    let none = db
        .get::<opt_v2::Profile>(none_id)
        .expect("get none")
        .expect("present");

    assert_eq!(some.handle, "neo");
    assert_eq!(
        some.badge,
        Some(99),
        "Some(u64) survives the disk-sourced walk",
    );
    assert!(!some.verified, "v1 → v2 defaults verified");
    assert_eq!(none.handle, "trinity");
    assert_eq!(none.badge, None, "None survives the disk-sourced walk");
    assert!(!none.verified);
}

mod derived_opt_v1 {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
    #[obj(version = 1, collection = "evolution_derived_opt")]
    pub struct Profile {
        pub handle: String,
        pub badge: Option<u64>,
    }
}

mod derived_opt_v2 {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
    #[obj(version = 2, collection = "evolution_derived_opt", auto_migrate)]
    pub struct Profile {
        pub handle: String,
        pub badge: Option<u64>,
        pub verified: bool,
    }
}

#[test]
fn derived_option_field_migrates_both_some_and_none_via_disk_catalog() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("evolution_derived_option.obj");

    let (some_id, none_id): (Id, Id) = {
        let db = Db::open(&path).expect("open v1");
        let some_id = db
            .insert(derived_opt_v1::Profile {
                handle: "morpheus".to_owned(),
                badge: Some(42),
            })
            .expect("insert some");
        let none_id = db
            .insert(derived_opt_v1::Profile {
                handle: "switch".to_owned(),
                badge: None,
            })
            .expect("insert none");
        (some_id, none_id)
    };

    let db = Db::open(&path).expect("reopen v2");
    let some = db
        .get::<derived_opt_v2::Profile>(some_id)
        .expect("get some")
        .expect("present");
    let none = db
        .get::<derived_opt_v2::Profile>(none_id)
        .expect("get none")
        .expect("present");

    assert_eq!(some.handle, "morpheus");
    assert_eq!(
        some.badge,
        Some(42),
        "derived Some(u64) survives auto_migrate via the disk catalog \
         (pre-#154 this mis-decoded to Some(4))",
    );
    assert!(!some.verified, "v1 → v2 auto_migrate defaults verified");
    assert_eq!(none.handle, "switch");
    assert_eq!(
        none.badge, None,
        "derived None survives auto_migrate via the disk catalog",
    );
    assert!(!none.verified);
}

/// The derive's syntactic `Option<u64>` lowering must produce the SAME
/// schema the Part-2 hand-written `optional_u64_schema()` does — the
/// byte-identical guarantee.
#[test]
fn derived_option_u64_schema_equals_handwritten() {
    let derived = <derived_opt_v1::Profile as Schema>::schema();
    let expected = DynamicSchema::map([
        ("handle", DynamicSchema::String),
        ("badge", optional_u64_schema()),
    ]);
    assert_eq!(
        derived, expected,
        "derived Option<u64> schema must equal the hand-written one",
    );
}
