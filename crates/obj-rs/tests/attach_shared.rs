//! `Db::attach_shared` / `Db::detach_shared` (the `&self`
//! attach/detach forms) + namespace-aware raw-bytes read shims.
//!
//! These exercise the namespace-aware engine read path: `attach_shared`
//! through a shared `Arc<Db>` handle, then namespaced raw-bytes reads
//! (`"archive.orders"`) resolve against the attached file while bare
//! names (`"orders"`) resolve locally. Unknown namespaces surface
//! `CollectionNamespaceUnknown`; `detach_shared` makes the namespace
//! unknown again while local reads keep working.

use std::sync::Arc;

use obj::{Db, Document, Error, Id};
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

/// Seed an archive db with `total` orders (collection name `"orders"`,
/// un-namespaced — the namespace prefix only matters on the calling
/// side) and return its path inside `dir`.
fn seed_archive(dir: &TempDir, total: u64) -> std::path::PathBuf {
    let archive_path = dir.path().join("archive.obj");
    let archive_db = Db::open(&archive_path).expect("open archive");
    for i in 1..=total {
        archive_db
            .insert(Order {
                customer_id: i,
                total_cents: i * 100,
            })
            .expect("seed archive");
    }
    archive_path
}

/// One-shot raw count for `collection` through a read transaction on
/// `db`. Wraps the raw `count_all_raw` engine entry point.
fn count_raw(db: &Db, collection: &str) -> obj::Result<u64> {
    db.read_transaction(|tx| tx.count_all_raw(collection))
}

/// Collect `(id, Order)` from the type-erased full-scan `Db::dump_raw`
/// — the engine method that binding `all()` / `query.fetch()` reads
/// route through. `DumpRecord::payload` is already header-stripped, so the
/// postcard decode is direct. `0` is the engine's unbounded mode,
/// capped here by the finite seed count.
fn dump_orders(db: &Db, collection: &str) -> obj::Result<Vec<(u64, Order)>> {
    let mut out = Vec::new();
    for step in db.dump_raw(collection, 0)? {
        let record = step?;
        let order: Order = postcard::from_bytes(&record.payload).expect("decode dump payload");
        out.push((record.id.get(), order));
    }
    Ok(out)
}

#[test]
fn attach_shared_namespaced_raw_reads_resolve_to_attached_file() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 3);
    let main_path = dir.path().join("main.obj");

    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.insert(Order {
        customer_id: 9,
        total_cents: 9_999,
    })
    .expect("insert local");
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    let archived = count_raw(&db, "archive.orders").expect("count archive.orders");
    assert_eq!(archived, 3, "archive.orders must read the attached file");

    let local = count_raw(&db, "orders").expect("count orders");
    assert_eq!(local, 1, "orders must read the local db");

    let archived_doc: Vec<Order> = db
        .read_transaction(|tx| {
            let pair = tx.get_with_version("archive.orders", Id::try_new(1).expect("nonzero"))?;
            Ok(pair
                .into_iter()
                .map(|(bytes, _v)| postcard::from_bytes::<Order>(&bytes).expect("decode"))
                .collect())
        })
        .expect("get archive.orders id 1");
    assert_eq!(archived_doc.len(), 1, "archive.orders id 1 must exist");
    assert_eq!(archived_doc[0].total_cents, 100);
}

#[test]
fn unknown_namespace_raw_read_errors() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));

    let err = count_raw(&db, "ghost.orders").expect_err("unknown namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "ghost"
        ),
        "expected CollectionNamespaceUnknown; got {err:?}",
    );
}

#[test]
fn detach_shared_makes_namespace_unknown_local_still_reads() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 2);
    let main_path = dir.path().join("main.obj");

    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.insert(Order {
        customer_id: 1,
        total_cents: 11,
    })
    .expect("insert local");
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    assert_eq!(count_raw(&db, "archive.orders").expect("pre-detach"), 2);

    db.detach_shared("archive").expect("detach_shared");

    let err = count_raw(&db, "archive.orders").expect_err("post-detach read");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "archive"
        ),
        "expected CollectionNamespaceUnknown after detach; got {err:?}",
    );

    assert_eq!(count_raw(&db, "orders").expect("local after detach"), 1);
}

#[test]
fn detach_shared_unknown_namespace_errors() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));
    let err = db.detach_shared("ghost").expect_err("unknown namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "ghost"
        ),
        "expected CollectionNamespaceUnknown; got {err:?}",
    );
}

#[test]
fn attach_shared_duplicate_namespace_rejected() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 1);
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.attach_shared(&archive_path, "archive")
        .expect("first attach_shared");
    let err = db
        .attach_shared(&archive_path, "archive")
        .expect_err("second attach_shared");
    assert!(
        matches!(
            err,
            Error::AttachmentAlreadyExists { ref namespace }
            if namespace == "archive"
        ),
        "expected AttachmentAlreadyExists; got {err:?}",
    );
}

/// A namespaced collection that does not exist in the attached file
/// surfaces `CollectionNotFound` (under the original namespaced name),
/// NOT a namespace error — the namespace itself resolved.
#[test]
fn known_namespace_unknown_collection_is_collection_not_found() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 1);
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    let err = count_raw(&db, "archive.widgets").expect_err("unknown collection");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "archive.widgets"),
        "expected CollectionNotFound for archive.widgets; got {err:?}",
    );
}

/// The raw FULL-SCAN that binding `all()` /
/// `query.fetch()` reads route through must resolve a namespaced name to the
/// attached file (like the point-read shims), AND the local bare-name
/// scan must stay byte-identical.
#[test]
fn dump_raw_namespaced_full_scan_reads_attached_local_independent() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 3);
    let main_path = dir.path().join("main.obj");

    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.insert(Order {
        customer_id: 42,
        total_cents: 4_200,
    })
    .expect("insert local");
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    let archived = dump_orders(&db, "archive.orders").expect("dump archive.orders");
    assert_eq!(
        archived.len(),
        3,
        "archive.orders full-scan must see 3 docs"
    );
    assert_eq!(
        archived[0],
        (
            1,
            Order {
                customer_id: 1,
                total_cents: 100
            }
        )
    );
    assert_eq!(
        archived[1],
        (
            2,
            Order {
                customer_id: 2,
                total_cents: 200
            }
        )
    );
    assert_eq!(
        archived[2],
        (
            3,
            Order {
                customer_id: 3,
                total_cents: 300
            }
        )
    );

    let local = dump_orders(&db, "orders").expect("dump orders");
    assert_eq!(
        local,
        vec![(
            1,
            Order {
                customer_id: 42,
                total_cents: 4_200
            }
        )]
    );
}

/// An unknown namespace on the full-scan path surfaces
/// `CollectionNamespaceUnknown` — consistent with the point-read shims.
#[test]
fn dump_raw_unknown_namespace_errors() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));

    let err = db
        .dump_raw("ghost.orders", 0)
        .expect_err("unknown namespace");
    assert!(
        matches!(
            err,
            Error::CollectionNamespaceUnknown { ref namespace }
            if namespace == "ghost"
        ),
        "expected CollectionNamespaceUnknown on dump_raw; got {err:?}",
    );
}

/// A known namespace but unknown collection surfaces `CollectionNotFound`
/// under the original namespaced name (the namespace itself resolved).
#[test]
fn dump_raw_known_namespace_unknown_collection_is_collection_not_found() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 1);
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    let err = db
        .dump_raw("archive.widgets", 0)
        .expect_err("unknown collection");
    assert!(
        matches!(err, Error::CollectionNotFound { ref name } if name == "archive.widgets"),
        "expected CollectionNotFound for archive.widgets full-scan; got {err:?}",
    );
}

/// A full-scan iterator pins its own snapshot at construction, so it
/// completes across a concurrent `detach_shared` (the registry entry is
/// gone, but the in-flight scan's pin keeps the attached env alive).
#[test]
fn dump_raw_namespaced_scan_survives_concurrent_detach() {
    let dir = TempDir::new().expect("tmp");
    let archive_path = seed_archive(&dir, 3);
    let main_path = dir.path().join("main.obj");
    let db = Arc::new(Db::open(&main_path).expect("open main"));
    db.attach_shared(&archive_path, "archive")
        .expect("attach_shared");

    let iter = db
        .dump_raw("archive.orders", 0)
        .expect("dump archive.orders");
    db.detach_shared("archive").expect("detach mid-scan");

    let mut seen = 0_u64;
    for step in iter {
        let record = step.expect("scan step after detach");
        let _: Order = postcard::from_bytes(&record.payload).expect("decode");
        seen += 1;
    }
    assert_eq!(
        seen, 3,
        "the pinned scan must still yield all 3 archived docs"
    );

    let err = db
        .dump_raw("archive.orders", 0)
        .expect_err("post-detach read");
    assert!(
        matches!(err, Error::CollectionNamespaceUnknown { ref namespace } if namespace == "archive"),
        "expected CollectionNamespaceUnknown after detach; got {err:?}",
    );
}
