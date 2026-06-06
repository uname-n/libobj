//! Lazy migration — on-disk bytes do NOT rewrite until the next
//! `Collection::update` (or `upsert`).
//!
//! Verifies the schema-evolution contract:
//!
//! 1. Write a v1 record into a fresh database.
//! 2. Reopen via a v2 type whose `historical_schemas()` registers
//!    the v1 schema. `Db::get` returns the migrated value. The
//!    on-disk bytes are still v1 (the per-doc header carries
//!    `type_version = 1`).
//! 3. `Collection::update` the same doc. The on-disk bytes now
//!    carry `type_version = 2` because `codec::encode` always
//!    stamps `T::VERSION`.

#![forbid(unsafe_code)]

use obj_core::btree::BTree;
use obj_core::codec::{DocumentHeader, DOC_HEADER_SIZE};
use obj_core::pager::page::PageId;
use obj_core::pager::{Config as PagerConfig, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor};
use obj_core::{Document, Id, Result};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

mod v1 {
    use super::{Deserialize, Document, Serialize};
    use obj_core::codec::{DynamicSchema, Schema};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Note {
        pub body: String,
    }

    impl Document for Note {
        const COLLECTION: &'static str = "lazy_notes";
        const VERSION: u32 = 1;
    }

    impl Schema for Note {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([("body", DynamicSchema::String)])
        }
    }
}

mod v2 {
    use super::{Deserialize, Document, Result, Serialize};
    use obj_core::codec::{Dynamic, DynamicSchema, Schema};
    use obj_core::Error;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Note {
        pub body: String,
        pub tags: Vec<String>,
    }

    impl Schema for Note {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("body", DynamicSchema::String),
                ("tags", DynamicSchema::seq(DynamicSchema::String)),
            ])
        }
    }

    impl Document for Note {
        const COLLECTION: &'static str = "lazy_notes";
        const VERSION: u32 = 2;

        fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
            vec![(1, DynamicSchema::map([("body", DynamicSchema::String)]))]
        }

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let body = dynamic.get_str("body")?.to_owned();
            Ok(Note {
                body,
                tags: vec!["<migrated>".to_owned()],
            })
        }
    }
}

fn read_doc_header(path: &std::path::Path, collection: &str, id: Id) -> DocumentHeader {
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

#[test]
fn lazy_migration_keeps_v1_bytes_on_disk_until_update() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("lazy_migration.obj");

    let id;
    {
        let db = obj::Db::open(&path).expect("open db");
        id = db
            .insert(v1::Note {
                body: "v1-content".to_owned(),
            })
            .expect("insert v1");
        let back: Option<v1::Note> = db.get(id).expect("get v1");
        assert_eq!(
            back,
            Some(v1::Note {
                body: "v1-content".to_owned(),
            })
        );
    }

    let header = read_doc_header(&path, "lazy_notes", id);
    assert_eq!(header.type_version, 1, "freshly-written record is v1");

    let migrated;
    {
        let db = obj::Db::open(&path).expect("reopen db as v2");
        migrated = db.get::<v2::Note>(id).expect("get v2").expect("present");
        assert_eq!(
            migrated,
            v2::Note {
                body: "v1-content".to_owned(),
                tags: vec!["<migrated>".to_owned()],
            }
        );
    }
    let header = read_doc_header(&path, "lazy_notes", id);
    assert_eq!(
        header.type_version, 1,
        "Db::get must NOT write the migrated bytes back (lazy migration)",
    );

    {
        let db = obj::Db::open(&path).expect("reopen db");
        db.update::<v2::Note, _>(id, |n| {
            n.body = "v2-content".to_owned();
        })
        .expect("update v2");
    }
    let header = read_doc_header(&path, "lazy_notes", id);
    assert_eq!(
        header.type_version, 2,
        "Collection::update rewrites the doc at T::VERSION",
    );
}
