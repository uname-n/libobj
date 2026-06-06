//! Integration tests for `#[obj(auto_migrate)]`.
//!
//! `auto_migrate` generates a `Document::migrate` for the
//! **pure-additive** evolution case: a version bump that only ADDS
//! fields. The generated body reads every current field from the
//! older record's `Dynamic::Map` by name — present fields carry over,
//! fields added in this version backfill with `Default::default()`
//! (or a per-field `#[obj(default = <expr>)]` override). No
//! hand-written `migrate` is needed.
//!
//! Each test mirrors the real engine flow `schema_catalog_worked_examples.rs`
//! uses: a v1 type inserts through the real `Db::insert` (which
//! persists the v1 `StoredSchema` row in the same WAL txn), then a v2
//! type — same collection, higher version, `#[obj(auto_migrate)]`,
//! one new field — reads the v1 doc back via `Db::get` / `Db::all`.
//! The ONLY source of the old wire shape is the on-disk catalog row.
//!
//! Every fallible step propagates via `?` / `expect`;
//! the generated migration reads fields per-name and fails
//! loudly on a wrong shape; no recursion / unbounded loops.

#![forbid(unsafe_code)]
// allow: test-support code — `expect`/`?` panics and error returns are the test's
// own failure signal, not a documented public-API contract worth a doc section.
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

use obj::{Db, Document};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

mod additive {
    use super::{Deserialize, Serialize};

    /// v1: `{ name, email }`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_additive")]
    pub struct CustomerV1 {
        pub name: String,
        pub email: String,
    }

    /// v2: adds `tier` (String) and `visits` (u64). NO hand-written
    /// migrate — `#[obj(auto_migrate)]` generates it. `name` / `email`
    /// carry over from the v1 record; `tier` / `visits` default.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_additive", auto_migrate)]
    pub struct CustomerV2 {
        pub name: String,
        pub email: String,
        pub tier: String,
        pub visits: u64,
    }
}

#[test]
fn additive_v1_to_v2_defaults_new_fields() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_additive.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(additive::CustomerV1 {
            name: "Ada".to_owned(),
            email: "ada@example.com".to_owned(),
        })
        .expect("insert v1")
    };

    let db = Db::open(&path).expect("reopen v2");
    let migrated: additive::CustomerV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        additive::CustomerV2 {
            name: "Ada".to_owned(),
            email: "ada@example.com".to_owned(),
            tier: String::new(),
            visits: 0,
        },
    );
}

#[test]
fn additive_v1_to_v2_full_scan_migrates_all() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_additive_scan.obj");

    {
        let db = Db::open(&path).expect("open v1");
        for i in 0..3u64 {
            db.insert(additive::CustomerV1 {
                name: format!("name-{i}"),
                email: format!("{i}@example.com"),
            })
            .expect("insert v1");
        }
    }

    let db = Db::open(&path).expect("reopen v2");
    let mut all = db.all::<additive::CustomerV2>().expect("all v2");
    all.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(all.len(), 3);
    for (i, c) in all.iter().enumerate() {
        assert_eq!(c.name, format!("name-{i}"));
        assert_eq!(c.email, format!("{i}@example.com"));
        assert_eq!(c.tier, "", "new field defaulted in every migrated doc");
        assert_eq!(c.visits, 0, "new field defaulted in every migrated doc");
    }
}

#[test]
fn current_version_does_not_route_through_generated_migrate() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_current.obj");
    let db = Db::open(&path).expect("open v2");

    let id = db
        .insert(additive::CustomerV2 {
            name: "Grace".to_owned(),
            email: "grace@example.com".to_owned(),
            tier: "gold".to_owned(),
            visits: 7,
        })
        .expect("insert v2");

    let back: additive::CustomerV2 = db.get(id).expect("get").expect("present");
    assert_eq!(
        back,
        additive::CustomerV2 {
            name: "Grace".to_owned(),
            email: "grace@example.com".to_owned(),
            tier: "gold".to_owned(),
            visits: 7,
        },
    );
}

mod custom_default {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_custom_default")]
    pub struct AccountV1 {
        pub id: u64,
    }

    /// v2 adds `tier` (custom string backfill) and `credits` (custom
    /// numeric backfill). The defaults are EXPRESSIONS, not the type's
    /// `Default` — so `tier` is `"standard"` (not "") and `credits` is
    /// `100` (not 0) for migrated v1 records.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_custom_default", auto_migrate)]
    pub struct AccountV2 {
        pub id: u64,
        #[obj(default = "standard".to_owned())]
        pub tier: String,
        #[obj(default = 100)]
        pub credits: u64,
    }
}

#[test]
fn custom_default_backfills_with_expression_not_type_default() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_custom_default.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(custom_default::AccountV1 { id: 42 })
            .expect("insert v1")
    };

    let db = Db::open(&path).expect("reopen v2");
    let migrated: custom_default::AccountV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        custom_default::AccountV2 {
            id: 42,
            tier: "standard".to_owned(),
            credits: 100,
        },
    );
}

mod with_index {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_indexed")]
    pub struct ProductV1 {
        pub sku: String,
        pub price_cents: u64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_indexed", auto_migrate)]
    pub struct ProductV2 {
        #[obj(index = unique)]
        pub sku: String,
        pub price_cents: u64,
        pub in_stock: bool,
    }
}

#[test]
fn auto_migrate_composes_with_indexes() {
    use obj::IndexKind;
    let specs = <with_index::ProductV2 as Document>::indexes();
    assert_eq!(specs.len(), 1, "the unique index survives auto_migrate");
    assert_eq!(specs[0].kind, IndexKind::Unique);
    assert_eq!(specs[0].name, "sku");

    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_indexed.obj");

    let v1_id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(with_index::ProductV1 {
            sku: "ABC-123".to_owned(),
            price_cents: 999,
        })
        .expect("insert v1")
    };

    let db = Db::open(&path).expect("reopen v2");
    let migrated: with_index::ProductV2 = db.get(v1_id).expect("get v2").expect("present");
    assert_eq!(migrated.sku, "ABC-123");
    assert_eq!(migrated.price_cents, 999);
    assert!(!migrated.in_stock, "new bool field defaulted to false");

    db.insert(with_index::ProductV2 {
        sku: "XYZ-789".to_owned(),
        price_cents: 1500,
        in_stock: true,
    })
    .expect("insert v2");
    let by_unique: Option<with_index::ProductV2> = db
        .find_unique::<with_index::ProductV2>("sku", "XYZ-789".to_owned())
        .expect("find_unique");
    assert_eq!(
        by_unique.expect("found via index").price_cents,
        1500,
        "the derived unique index is live alongside auto_migrate",
    );
}

mod rename_boundary {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_rename")]
    pub struct DocV1 {
        pub old_name: String,
    }

    /// v2 renames `old_name` -> `new_name`. `auto_migrate` cannot know
    /// they are "the same" field, so `new_name` is treated as a fresh
    /// (absent) field and backfills with its default. This is the
    /// documented boundary: renames are NOT additive.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_rename", auto_migrate)]
    pub struct DocV2 {
        pub new_name: String,
    }
}

#[test]
fn rename_is_not_auto_migrated_and_defaults_the_new_name() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_rename.obj");

    let id = {
        let db = Db::open(&path).expect("open v1");
        db.insert(rename_boundary::DocV1 {
            old_name: "carried-value".to_owned(),
        })
        .expect("insert v1")
    };

    let db = Db::open(&path).expect("reopen v2");
    let migrated: rename_boundary::DocV2 = db.get(id).expect("get v2").expect("present");
    assert_eq!(
        migrated,
        rename_boundary::DocV2 {
            new_name: String::new(),
        },
        "a rename is not additive: auto_migrate defaults the new name",
    );
}

mod opt_field {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_opt_field")]
    pub struct AccountV1 {
        pub name: String,
        pub badge: Option<u64>,
    }

    /// v2 keeps `badge: Option<u64>` and adds a defaulted `active`
    /// field. NO hand-written migrate — `#[obj(auto_migrate)]` carries
    /// the Option over via the shape-faithful `Dynamic::deserialize`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_opt_field", auto_migrate)]
    pub struct AccountV2 {
        pub name: String,
        pub badge: Option<u64>,
        pub active: bool,
    }
}

#[test]
fn auto_migrate_carries_option_field_some_and_none() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_opt_field.obj");

    let (some_id, none_id) = {
        let db = Db::open(&path).expect("open v1");
        let some_id = db
            .insert(opt_field::AccountV1 {
                name: "neo".to_owned(),
                badge: Some(42),
            })
            .expect("insert some");
        let none_id = db
            .insert(opt_field::AccountV1 {
                name: "trinity".to_owned(),
                badge: None,
            })
            .expect("insert none");
        (some_id, none_id)
    };

    let db = Db::open(&path).expect("reopen v2");
    let some: opt_field::AccountV2 = db.get(some_id).expect("get some").expect("present");
    let none: opt_field::AccountV2 = db.get(none_id).expect("get none").expect("present");

    assert_eq!(
        some,
        opt_field::AccountV2 {
            name: "neo".to_owned(),
            badge: Some(42),
            active: false,
        },
        "auto_migrate carries Some(42) over verbatim",
    );
    assert_eq!(
        none,
        opt_field::AccountV2 {
            name: "trinity".to_owned(),
            badge: None,
            active: false,
        },
        "auto_migrate carries None over verbatim",
    );
}

mod enum_field {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(schema)]
    pub enum Tier {
        #[default]
        Free,
        Pro {
            seats: u32,
        },
        Enterprise(String),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 1, collection = "auto_enum_field")]
    pub struct AccountV1 {
        pub owner: String,
        pub tier: Tier,
    }

    /// v2 keeps the `tier` enum field and adds a defaulted `region`
    /// scalar. `#[obj(auto_migrate)]` carries the enum over through the
    /// shape-faithful `Dynamic::deserialize`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(version = 2, collection = "auto_enum_field", auto_migrate)]
    pub struct AccountV2 {
        pub owner: String,
        pub tier: Tier,
        #[obj(default = "unknown".to_owned())]
        pub region: String,
    }
}

#[test]
fn auto_migrate_carries_real_enum_field() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("auto_enum_field.obj");

    let ids = {
        let db = Db::open(&path).expect("open v1");
        vec![
            db.insert(enum_field::AccountV1 {
                owner: "alice".to_owned(),
                tier: enum_field::Tier::Free,
            })
            .expect("insert free"),
            db.insert(enum_field::AccountV1 {
                owner: "bob".to_owned(),
                tier: enum_field::Tier::Pro { seats: 12 },
            })
            .expect("insert pro"),
            db.insert(enum_field::AccountV1 {
                owner: "carol".to_owned(),
                tier: enum_field::Tier::Enterprise("acme".to_owned()),
            })
            .expect("insert enterprise"),
        ]
    };

    let db = Db::open(&path).expect("reopen v2");
    let got: Vec<enum_field::AccountV2> = ids
        .iter()
        .map(|id| db.get(*id).expect("get").expect("present"))
        .collect();

    assert_eq!(got[0].owner, "alice");
    assert_eq!(got[0].tier, enum_field::Tier::Free, "unit variant survives");
    assert_eq!(got[0].region, "unknown", "new scalar field backfills");
    assert_eq!(got[1].owner, "bob");
    assert_eq!(
        got[1].tier,
        enum_field::Tier::Pro { seats: 12 },
        "struct variant survives",
    );
    assert_eq!(got[2].owner, "carol");
    assert_eq!(
        got[2].tier,
        enum_field::Tier::Enterprise("acme".to_owned()),
        "newtype variant survives",
    );
}
