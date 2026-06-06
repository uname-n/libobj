//! `Db::attach` acceptance tests.
//!
//! Create two Dbs, populate both, attach one to the other, read
//! across both in a single `read_transaction`. Verify writes to the
//! attached collection error cleanly. Detach and confirm the
//! calling Db's own collections still work.

use obj::{Db, Document, Error};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    total_cents: u64,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("total_cents", obj::DynamicSchema::U64),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArchivedOrder {
    customer_id: u64,
    total_cents: u64,
    archived_at_ms: u64,
}

impl Document for ArchivedOrder {
    const COLLECTION: &'static str = "archive.orders";
    const VERSION: u32 = 1;
}

impl obj::Schema for ArchivedOrder {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("total_cents", obj::DynamicSchema::U64),
            ("archived_at_ms", obj::DynamicSchema::U64),
        ])
    }
}

#[test]
fn attached_db_visible_in_read_transaction() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");

    {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct ArchiveSide {
            customer_id: u64,
            total_cents: u64,
            archived_at_ms: u64,
        }
        impl Document for ArchiveSide {
            const COLLECTION: &'static str = "orders";
            const VERSION: u32 = 1;
        }
        impl obj::Schema for ArchiveSide {
            fn schema() -> obj::DynamicSchema {
                obj::DynamicSchema::map([
                    ("customer_id", obj::DynamicSchema::U64),
                    ("total_cents", obj::DynamicSchema::U64),
                    ("archived_at_ms", obj::DynamicSchema::U64),
                ])
            }
        }
        let archive_db = Db::open(&archive_path).expect("open archive");
        archive_db
            .insert(ArchiveSide {
                customer_id: 1,
                total_cents: 999,
                archived_at_ms: 42,
            })
            .expect("insert into archive");
    }

    let mut main_db = Db::open(&main_path).expect("open main");
    main_db
        .insert(Order {
            customer_id: 1,
            total_cents: 100,
        })
        .expect("insert live");
    main_db.attach(&archive_path, "archive").expect("attach");

    main_db
        .read_transaction(|tx| {
            let live = tx.collection::<Order>()?;
            let archived = tx.collection::<ArchivedOrder>()?;
            let live_docs = live.all()?;
            assert_eq!(live_docs.len(), 1);
            let arch_docs = archived.all()?;
            assert_eq!(arch_docs.len(), 1);
            assert_eq!(arch_docs[0].1.archived_at_ms, 42);
            Ok(())
        })
        .expect("read across attached");
}

#[test]
fn writes_to_attached_collection_are_rejected() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");

    {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct ArchiveSide {
            customer_id: u64,
            total_cents: u64,
            archived_at_ms: u64,
        }
        impl Document for ArchiveSide {
            const COLLECTION: &'static str = "orders";
            const VERSION: u32 = 1;
        }
        impl obj::Schema for ArchiveSide {
            fn schema() -> obj::DynamicSchema {
                obj::DynamicSchema::map([
                    ("customer_id", obj::DynamicSchema::U64),
                    ("total_cents", obj::DynamicSchema::U64),
                    ("archived_at_ms", obj::DynamicSchema::U64),
                ])
            }
        }
        let archive_db = Db::open(&archive_path).expect("open archive");
        archive_db
            .insert(ArchiveSide {
                customer_id: 1,
                total_cents: 999,
                archived_at_ms: 42,
            })
            .expect("insert");
    }
    let mut main_db = Db::open(&main_path).expect("open main");
    main_db.attach(&archive_path, "archive").expect("attach");

    let err = main_db
        .insert(ArchivedOrder {
            customer_id: 2,
            total_cents: 1,
            archived_at_ms: 0,
        })
        .expect_err("insert into attached must fail");
    assert!(
        matches!(
            err,
            Error::AttachedDatabaseIsReadOnly {
                ref namespace,
                ..
            } if namespace == "archive"
        ),
        "expected AttachedDatabaseIsReadOnly; got {err:?}",
    );
}

#[test]
fn duplicate_namespace_is_rejected() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");
    {
        let _ = Db::open(&archive_path).expect("create archive");
    }
    let mut main_db = Db::open(&main_path).expect("open");
    main_db
        .attach(&archive_path, "archive")
        .expect("first attach");
    let err = main_db
        .attach(&archive_path, "archive")
        .expect_err("second attach");
    assert!(
        matches!(
            err,
            Error::AttachmentAlreadyExists { ref namespace }
            if namespace == "archive"
        ),
        "expected AttachmentAlreadyExists; got {err:?}",
    );
}

#[test]
fn detach_removes_attachment() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");
    {
        let _ = Db::open(&archive_path).expect("create archive");
    }
    let mut main_db = Db::open(&main_path).expect("open");
    main_db.attach(&archive_path, "archive").expect("attach");
    main_db.detach("archive").expect("detach");
    let err = main_db
        .read_transaction(|tx| tx.collection::<ArchivedOrder>().map(|_| ()))
        .expect_err("read on detached namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "archive"
        ),
        "expected CollectionNamespaceUnknown; got {err:?}",
    );
    main_db.attach(&archive_path, "archive").expect("re-attach");
}

#[test]
fn detach_unknown_namespace_errors() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let mut main_db = Db::open(&main_path).expect("open");
    let err = main_db.detach("ghost").expect_err("unknown namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "ghost"
        ),
        "expected CollectionNamespaceUnknown; got {err:?}",
    );
}

/// `Db::collection::<T>(name)` reads from a
/// runtime-named collection (here a namespaced attached one)
/// without requiring `T::COLLECTION` to carry the namespace.
///
/// ```text
/// db.attach("archive.obj", "archive")?;
/// let archived: Vec<Order> = db
///     .collection::<Order>("archive.orders")
///     .all()?
///     .collect();
/// ```
///
/// `Order`'s `COLLECTION` is `"orders"` — the namespace prefix lives
/// only on the calling side, supplied at the runtime accessor's call
/// site.
#[test]
fn db_collection_reads_from_attached_namespace() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");

    {
        let archive_db = Db::open(&archive_path).expect("open archive");
        for i in 1..=3 {
            archive_db
                .insert(Order {
                    customer_id: i,
                    total_cents: i * 100,
                })
                .expect("seed archive");
        }
    }

    let mut main_db = Db::open(&main_path).expect("open main");
    main_db.attach(&archive_path, "archive").expect("attach");

    let archived: Vec<Order> = main_db
        .collection::<Order>("archive.orders")
        .all()
        .expect("all on attached")
        .into_iter()
        .map(|(_id, doc)| doc)
        .collect();
    assert_eq!(archived.len(), 3);
    let totals: Vec<u64> = archived.iter().map(|o| o.total_cents).collect();
    assert!(totals.contains(&100));
    assert!(totals.contains(&200));
    assert!(totals.contains(&300));

    let err = main_db
        .all::<Order>()
        .expect_err("calling-db `orders` was never written");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "orders"),
        "expected CollectionNotFound for calling-db `orders`; got {err:?}",
    );
}

/// `Db::collection::<T>(name)` against an unknown namespace
/// surfaces the namespace-unknown error at the first method call
/// (construction is infallible).
#[test]
fn db_collection_unknown_namespace_errors_at_call_site() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let main_db = Db::open(&main_path).expect("open main");
    let handle = main_db.collection::<Order>("ghost.orders");
    let err = handle.all().expect_err("unknown namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "ghost"
        ),
        "expected CollectionNamespaceUnknown; got {err:?}",
    );
}

/// `Db::collection::<T>(name)` reads from a runtime-named
/// collection on the **calling** Db (no namespace prefix). Useful
/// when the type's declared `COLLECTION` differs from the name the
/// caller wants to consult at runtime (e.g. multi-tenant schemas).
#[test]
fn db_collection_reads_from_calling_db_runtime_name() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let main_db = Db::open(&main_path).expect("open main");
    main_db
        .insert(Order {
            customer_id: 1,
            total_cents: 42,
        })
        .expect("insert");

    let docs: Vec<Order> = main_db
        .collection::<Order>("orders")
        .all()
        .expect("all on calling db")
        .into_iter()
        .map(|(_id, doc)| doc)
        .collect();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].total_cents, 42);
}

/// Writes through `Db::collection::<T>(name)` are rejected: the
/// runtime accessor is read-only by design (documented on the
/// rustdoc). Verified through `.insert` failing with
/// `Error::ReadOnly`.
#[test]
fn db_collection_rejects_writes() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let main_db = Db::open(&main_path).expect("open main");
    let handle = main_db.collection::<Order>("orders");
    let err = handle
        .insert(Order {
            customer_id: 1,
            total_cents: 7,
        })
        .expect_err("insert must be rejected");
    assert!(
        matches!(err, Error::ReadOnly { .. }),
        "expected ReadOnly; got {err:?}",
    );
}

#[test]
fn calling_db_collection_still_works_after_detach() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");
    {
        let _ = Db::open(&archive_path).expect("create archive");
    }
    let mut main_db = Db::open(&main_path).expect("open");
    let id = main_db
        .insert(Order {
            customer_id: 7,
            total_cents: 77,
        })
        .expect("insert");
    main_db.attach(&archive_path, "archive").expect("attach");
    main_db.detach("archive").expect("detach");
    let got: Option<Order> = main_db.get(id).expect("get after detach");
    assert!(got.is_some(), "main-db reads must still work after detach");
}

/// The fused one-shot `Db::get` path (single pager lock,
/// empty-attached fast path) is observably identical to the explicit
/// `read_transaction(|tx| tx.collection()?.get())` handle path — the
/// hit, the miss, and the unknown-collection arms all match, with no
/// database attached.
#[test]
fn fused_get_matches_handle_path_with_empty_attached() {
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct NeverWritten {
        x: u64,
    }
    impl Document for NeverWritten {
        const COLLECTION: &'static str = "never_written_collection";
        const VERSION: u32 = 1;
    }

    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let main_db = Db::open(&main_path).expect("open");
    let id = main_db
        .insert(Order {
            customer_id: 42,
            total_cents: 4_200,
        })
        .expect("insert");

    let fused: Option<Order> = main_db.get(id).expect("fused get");
    let via_handle: Option<Order> = main_db
        .read_transaction(|tx| tx.collection::<Order>()?.get(id))
        .expect("handle get");
    assert_eq!(fused, via_handle, "fused get must match the handle path");
    assert_eq!(
        fused,
        Some(Order {
            customer_id: 42,
            total_cents: 4_200,
        }),
        "fused get must return the inserted doc",
    );

    let absent_id = obj::Id::try_new(id.get() + 1_000).expect("nonzero id");
    let fused_miss: Option<Order> = main_db.get(absent_id).expect("fused miss");
    assert!(fused_miss.is_none(), "absent id must read as None");

    let probe = obj::Id::try_new(1).expect("nonzero id");
    let err = main_db
        .get::<NeverWritten>(probe)
        .expect_err("unknown collection");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "never_written_collection"),
        "fused get on an unknown collection must surface CollectionNotFound; got {err:?}",
    );
}
