//! Mixed-version coexistence.
//!
//! A v1 and v3 customer can coexist in the same collection. This
//! test doubles as a worked example — the type shape mirrors the
//! Customer example (name, email; v2 adds tier; v3 adds region).
//!
//! Sequence:
//!
//! 1. Open Db, insert 3 v1 Customer docs (no tier, no region).
//!    Reopen — the on-disk bytes carry `type_version = 1`.
//! 2. Switch to v2 Customer (adds `tier`, registers v1 schema).
//!    Insert 3 more docs. The first three docs are still v1 on
//!    disk; the new three are v2 on disk.
//! 3. Switch to v3 Customer (adds `region`, registers v1 + v2
//!    schemas). Insert 3 more docs.
//! 4. Read every doc through the v3 type — the v1 set returns
//!    with default tier + default region; the v2 set returns
//!    with their original tier + default region; the v3 set
//!    returns verbatim. All 9 docs decode as `v3::Customer`.
//! 5. The raw on-disk `type_version`s are still v1, v2, v3 for
//!    their respective writing eras — `Db::get` did NOT
//!    rewrite anything.
//! 6. `Db::update` one of the v1 docs through the v3 type. Its
//!    on-disk bytes are now v3. The other v1 docs are still v1.

#![forbid(unsafe_code)]

use obj_core::btree::BTree;
use obj_core::codec::{DocumentHeader, DOC_HEADER_SIZE};
use obj_core::pager::page::PageId;
use obj_core::pager::{Config as PagerConfig, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor};
use obj_core::{Id, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tempfile::TempDir;

mod v1 {
    use super::{Deserialize, Serialize};
    use obj_core::codec::{DynamicSchema, Schema};
    use obj_core::Document;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Customer {
        pub name: String,
        pub email: String,
    }

    impl Document for Customer {
        const COLLECTION: &'static str = "customers_evo";
        const VERSION: u32 = 1;
    }

    impl Schema for Customer {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("name", DynamicSchema::String),
                ("email", DynamicSchema::String),
            ])
        }
    }
}

mod v2 {
    use super::{Deserialize, Result, Serialize};
    use obj_core::codec::{Dynamic, DynamicSchema, Schema};
    use obj_core::{Document, Error};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Customer {
        pub name: String,
        pub email: String,
        pub tier: String,
    }

    impl Schema for Customer {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("name", DynamicSchema::String),
                ("email", DynamicSchema::String),
                ("tier", DynamicSchema::String),
            ])
        }
    }

    impl Document for Customer {
        const COLLECTION: &'static str = "customers_evo";
        const VERSION: u32 = 2;

        fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
            vec![(
                1,
                DynamicSchema::map([
                    ("name", DynamicSchema::String),
                    ("email", DynamicSchema::String),
                ]),
            )]
        }

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let name = dynamic.get_str("name")?.to_owned();
            let email = dynamic.get_str("email")?.to_owned();
            Ok(Customer {
                name,
                email,
                tier: "standard".to_owned(),
            })
        }
    }
}

mod v3 {
    use super::{Deserialize, Result, Serialize};
    use obj_core::codec::{Dynamic, DynamicSchema, Schema};
    use obj_core::{Document, Error};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Customer {
        pub name: String,
        pub email: String,
        pub tier: String,
        pub region: String,
    }

    impl Schema for Customer {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("name", DynamicSchema::String),
                ("email", DynamicSchema::String),
                ("tier", DynamicSchema::String),
                ("region", DynamicSchema::String),
            ])
        }
    }

    impl Document for Customer {
        const COLLECTION: &'static str = "customers_evo";
        const VERSION: u32 = 3;

        fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
            vec![
                (
                    1,
                    DynamicSchema::map([
                        ("name", DynamicSchema::String),
                        ("email", DynamicSchema::String),
                    ]),
                ),
                (
                    2,
                    DynamicSchema::map([
                        ("name", DynamicSchema::String),
                        ("email", DynamicSchema::String),
                        ("tier", DynamicSchema::String),
                    ]),
                ),
            ]
        }

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            match from_version {
                1 => {
                    let name = dynamic.get_str("name")?.to_owned();
                    let email = dynamic.get_str("email")?.to_owned();
                    Ok(Customer {
                        name,
                        email,
                        tier: "standard".to_owned(),
                        region: "us-east".to_owned(),
                    })
                }
                2 => {
                    let name = dynamic.get_str("name")?.to_owned();
                    let email = dynamic.get_str("email")?.to_owned();
                    let tier = dynamic.get_str("tier")?.to_owned();
                    Ok(Customer {
                        name,
                        email,
                        tier,
                        region: "us-east".to_owned(),
                    })
                }
                other => Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version: other,
                    to_version: Self::VERSION,
                }),
            }
        }
    }
}

fn read_doc_header(path: &Path, collection: &str, id: Id) -> DocumentHeader {
    let mut pager = Pager::<FileHandle>::open(path, PagerConfig::default()).expect("reopen pager");
    let catalog = Catalog::open_or_init(&mut pager).expect("catalog");
    let descriptor: CollectionDescriptor = catalog
        .get(&mut pager, collection)
        .expect("get descriptor")
        .expect("descriptor present");
    let primary_root = PageId::new(descriptor.primary_root).expect("non-zero root");
    let tree = BTree::<FileHandle>::open(&pager, primary_root).expect("open primary");
    let bytes = tree
        .get(&mut pager, &id.to_be_bytes())
        .expect("get record")
        .expect("record present");
    assert!(
        bytes.len() >= DOC_HEADER_SIZE,
        "record smaller than DocumentHeader",
    );
    let header = DocumentHeader::read_from(&bytes[..DOC_HEADER_SIZE]).expect("decode header");
    pager.close().expect("close");
    header
}

fn phase_insert_v1(path: &Path, count: usize) -> Vec<Id> {
    let db = obj::Db::open(path).expect("open");
    (0..count)
        .map(|i| {
            db.insert(v1::Customer {
                name: format!("v1-name-{i}"),
                email: format!("v1-{i}@example.com"),
            })
            .expect("insert v1")
        })
        .collect()
}

fn phase_insert_v2(path: &Path, count: usize, v1_ids: &[Id]) -> Vec<Id> {
    let db = obj::Db::open(path).expect("reopen v2");
    for id in v1_ids {
        let c: v2::Customer = db.get(*id).expect("get").expect("present");
        assert_eq!(c.tier, "standard", "v1 → v2 defaults tier");
    }
    (0..count)
        .map(|i| {
            db.insert(v2::Customer {
                name: format!("v2-name-{i}"),
                email: format!("v2-{i}@example.com"),
                tier: "gold".to_owned(),
            })
            .expect("insert v2")
        })
        .collect()
}

fn phase_insert_v3(path: &Path, count: usize) -> Vec<Id> {
    let db = obj::Db::open(path).expect("reopen v3");
    (0..count)
        .map(|i| {
            db.insert(v3::Customer {
                name: format!("v3-name-{i}"),
                email: format!("v3-{i}@example.com"),
                tier: "platinum".to_owned(),
                region: "eu-west".to_owned(),
            })
            .expect("insert v3")
        })
        .collect()
}

fn read_all_as_v3(path: &Path, ids: &[Id]) -> Vec<v3::Customer> {
    let db = obj::Db::open(path).expect("reopen v3 for full read");
    ids.iter()
        .map(|id| db.get::<v3::Customer>(*id).expect("get").expect("present"))
        .collect()
}

fn assert_v1_in_v3(c: &v3::Customer, i: usize) {
    assert_eq!(c.name, format!("v1-name-{i}"));
    assert_eq!(c.tier, "standard", "v1 → v3 inherits v1 → v2 default");
    assert_eq!(c.region, "us-east", "v1 → v3 defaults region");
}

fn assert_v2_in_v3(c: &v3::Customer, i: usize) {
    assert_eq!(c.name, format!("v2-name-{i}"));
    assert_eq!(c.tier, "gold", "v2 tier passed through to v3");
    assert_eq!(c.region, "us-east", "v2 → v3 defaults region");
}

fn assert_v3_in_v3(c: &v3::Customer, i: usize) {
    assert_eq!(c.name, format!("v3-name-{i}"));
    assert_eq!(c.tier, "platinum");
    assert_eq!(c.region, "eu-west");
}

fn assert_on_disk(path: &Path, v1_ids: &[Id], v2_ids: &[Id], v3_ids: &[Id]) {
    for id in v1_ids {
        assert_eq!(
            read_doc_header(path, "customers_evo", *id).type_version,
            1,
            "v1 doc {id:?} unrewritten",
        );
    }
    for id in v2_ids {
        assert_eq!(
            read_doc_header(path, "customers_evo", *id).type_version,
            2,
            "v2 doc {id:?} at v2",
        );
    }
    for id in v3_ids {
        assert_eq!(
            read_doc_header(path, "customers_evo", *id).type_version,
            3,
            "v3 doc {id:?} at v3",
        );
    }
}

#[test]
fn mixed_version_coexistence_v1_v2_v3() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("customers_evo.obj");

    let v1_ids = phase_insert_v1(&path, 3);
    assert_on_disk(&path, &v1_ids, &[], &[]);

    let v2_ids = phase_insert_v2(&path, 3, &v1_ids);
    assert_on_disk(&path, &v1_ids, &v2_ids, &[]);

    let v3_ids = phase_insert_v3(&path, 3);
    assert_on_disk(&path, &v1_ids, &v2_ids, &v3_ids);

    let all_ids: Vec<Id> = v1_ids
        .iter()
        .chain(v2_ids.iter())
        .chain(v3_ids.iter())
        .copied()
        .collect();
    let all = read_all_as_v3(&path, &all_ids);
    assert_eq!(all.len(), 9, "all 9 docs return as v3::Customer");
    for (i, c) in all.iter().take(3).enumerate() {
        assert_v1_in_v3(c, i);
    }
    for (i, c) in all.iter().skip(3).take(3).enumerate() {
        assert_v2_in_v3(c, i);
    }
    for (i, c) in all.iter().skip(6).take(3).enumerate() {
        assert_v3_in_v3(c, i);
    }
    assert_on_disk(&path, &v1_ids, &v2_ids, &v3_ids);

    let updated = v1_ids[0];
    {
        let db = obj::Db::open(&path).expect("reopen for update");
        db.update::<v3::Customer, _>(updated, |c| {
            c.region = "ap-south".to_owned();
        })
        .expect("update v1 → v3");
    }
    assert_eq!(
        read_doc_header(&path, "customers_evo", updated).type_version,
        3,
        "updated v1 doc is now v3 on disk",
    );
    for id in v1_ids.iter().skip(1) {
        assert_eq!(
            read_doc_header(&path, "customers_evo", *id).type_version,
            1,
            "untouched v1 docs are STILL v1 on disk",
        );
    }
}
