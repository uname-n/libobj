//! `Db::integrity_check` acceptance tests.
//!
//! Clean DB passes; a hand-corrupted DB fails with `ChecksumMismatch`;
//! a manually-orphaned index entry fails with `OrphanIndexEntry`. Runs
//! under default `cargo test`.

use obj::{Db, Document, IndexSpec, IntegrityFailure, IntegrityReport};
use obj_core::pager::page::PAGE_SIZE;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct User {
    email: String,
    handle: String,
}

impl Document for User {
    const COLLECTION: &'static str = "users";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_email", "email").expect("static spec")]
    }
}

impl obj::Schema for User {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("email", obj::DynamicSchema::String),
            ("handle", obj::DynamicSchema::String),
        ])
    }
}

#[test]
fn clean_db_passes_integrity_check() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("clean.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..16u32 {
        db.insert(User {
            email: format!("user{i}@example.com"),
            handle: format!("u{i}"),
        })
        .expect("insert");
    }
    let report: IntegrityReport = db.integrity_check().expect("integrity_check");
    assert!(
        report.is_ok(),
        "clean db should pass; got: {:?}",
        report.failures
    );
    assert!(report.pages_checked > 0, "must have inspected some pages");
}

/// Regression: a collection populated past the first leaf-split
/// threshold must pass `integrity_check`. The walker must not treat
/// CoW-stale `next_sibling` pointers on left-neighbour leaves as chain
/// breakage. 200 docs at this payload guarantees several splits' worth
/// of stale pointers in the tree.
#[test]
fn integrity_check_passes_on_post_split_collection() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("post-split.obj");
    let db = Db::open(&path).expect("open");
    for i in 0..200u32 {
        db.insert(User {
            email: format!("user{i:05}@example.com"),
            handle: format!("u{i:05}"),
        })
        .expect("insert");
    }
    let report: IntegrityReport = db.integrity_check().expect("integrity_check");
    assert!(
        report.is_ok(),
        "post-split db should pass integrity_check; got: {:?}",
        report.failures,
    );
    assert!(
        report.pages_checked >= 8,
        "expected several pages (≥ 8) in a 200-doc collection; got {}",
        report.pages_checked,
    );
}

#[test]
fn corrupted_page_surfaces_checksum_mismatch() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("corrupted.obj");
    {
        let db = Db::open(&path).expect("open");
        for i in 0..32u32 {
            db.insert(User {
                email: format!("user{i}@example.com"),
                handle: format!("u{i}"),
            })
            .expect("insert");
        }
    }
    checkpoint_and_close(&path).expect("checkpoint");
    let target_pid = locate_primary_page(&path).expect("locate primary root");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open for corruption");
    let target_off = (target_pid * PAGE_SIZE as u64) + 64;
    file.seek(SeekFrom::Start(target_off)).expect("seek");
    file.write_all(&[0xFFu8]).expect("flip byte");
    drop(file);

    let db = Db::open(&path).expect("re-open");
    let report = db.integrity_check().expect("integrity_check completes");
    assert!(
        report.failures.iter().any(|f| matches!(
            f,
            IntegrityFailure::ChecksumMismatch { .. } | IntegrityFailure::BTreeSortViolation { .. }
        )),
        "expected a ChecksumMismatch or sort-violation failure; got {:?}",
        report.failures,
    );
}

/// Open the file at the pager layer and call `close()` so the WAL
/// is checkpointed into the main file and the sidecar removed.
/// Without this, a flipped byte on a main-file page is masked by
/// the WAL overlay on the next open.
fn checkpoint_and_close(path: &std::path::Path) -> obj::Result<()> {
    use obj_core::pager::{Config, Pager};
    use obj_core::platform::FileHandle;
    let pager = Pager::<FileHandle>::open(path, Config::default())?;
    pager.close()
}

/// Look up the primary B-tree root page-id of the `users` collection.
fn locate_primary_page(path: &std::path::Path) -> obj::Result<u64> {
    use obj_core::pager::{Config, Pager};
    use obj_core::platform::FileHandle;
    use obj_core::Catalog;

    let mut pager = Pager::<FileHandle>::open(path, Config::default())?;
    pager.begin_txn();
    let catalog = Catalog::<FileHandle>::open_or_init(&mut pager)?;
    let descriptor = catalog.get(&mut pager, "users")?.expect("users present");
    pager.end_txn();
    Ok(descriptor.primary_root)
}

#[test]
fn orphan_index_entry_surfaces() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("orphan.obj");
    {
        let db = Db::open(&path).expect("open");
        for i in 0..8u32 {
            db.insert(User {
                email: format!("user{i}@example.com"),
                handle: format!("u{i}"),
            })
            .expect("insert");
        }
    }
    inject_orphan_index_entry(&path).expect("inject");
    let db = Db::open(&path).expect("re-open");
    let report = db.integrity_check().expect("integrity_check");
    assert!(
        report
            .failures
            .iter()
            .any(|f| matches!(f, IntegrityFailure::OrphanIndexEntry { .. })),
        "expected OrphanIndexEntry; got {:?}",
        report.failures,
    );
}

#[test]
fn open_check_rejects_corrupted_catalog_page() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("bad-catalog.obj");
    {
        let db = Db::open(&path).expect("open");
        for i in 0..8u32 {
            db.insert(User {
                email: format!("user{i}@example.com"),
                handle: format!("u{i}"),
            })
            .expect("insert");
        }
    }
    checkpoint_and_close(&path).expect("checkpoint");

    let catalog_root = locate_catalog_root(&path).expect("catalog root");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open for corruption");
    let target_off = (catalog_root * PAGE_SIZE as u64) + 64;
    file.seek(SeekFrom::Start(target_off)).expect("seek");
    file.write_all(&[0xFFu8]).expect("flip byte");
    drop(file);

    let err = Db::open(&path).expect_err("open should fail on corrupted catalog");
    assert!(
        matches!(
            err,
            obj::Error::Corruption { .. } | obj::Error::BTreeDepthExceeded { .. }
        ),
        "expected Corruption / BTreeDepthExceeded; got {err:?}",
    );
}

#[test]
fn open_check_passes_on_clean_db() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("clean-open.obj");
    {
        let db = Db::open(&path).expect("open");
        db.insert(User {
            email: "a@b.com".to_owned(),
            handle: "u".to_owned(),
        })
        .expect("insert");
    }
    checkpoint_and_close(&path).expect("checkpoint");
    let _db = Db::open(&path).expect("clean db opens with default check");
}

#[test]
fn skip_open_check_bypasses_fast_walk_on_non_catalog_corruption() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("skip-open.obj");
    {
        let db = Db::open(&path).expect("open");
        for i in 0..16u32 {
            db.insert(User {
                email: format!("user{i}@example.com"),
                handle: format!("u{i}"),
            })
            .expect("insert");
        }
    }
    checkpoint_and_close(&path).expect("checkpoint");
    let target_pid = locate_primary_page(&path).expect("primary root");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open for corruption");
    let target_off = (target_pid * PAGE_SIZE as u64) + 64;
    file.seek(SeekFrom::Start(target_off)).expect("seek");
    file.write_all(&[0xFFu8]).expect("flip");
    drop(file);
    let db = Db::open(&path).expect("default open succeeds");
    drop(db);
    let _db = Db::open_with(&path, obj::Config::default().skip_open_check(true))
        .expect("skip_open_check accepted");
}

fn locate_catalog_root(path: &std::path::Path) -> obj::Result<u64> {
    use obj_core::pager::{Config, Pager};
    use obj_core::platform::FileHandle;
    let pager = Pager::<FileHandle>::open(path, Config::default())?;
    Ok(pager.root_catalog())
}

/// Manually inject an index B-tree entry pointing at an id that
/// has no primary row. We open the file at the obj-core layer,
/// look up the `by_email` index's root page-id from the catalog,
/// and call `BTree::insert` with a synthetic key pointing at
/// `Id(99_999)` — an id no insert ever produced.
fn inject_orphan_index_entry(path: &std::path::Path) -> obj::Result<()> {
    use obj_core::btree::BTree;
    use obj_core::pager::page::PageId;
    use obj_core::pager::{Config, Pager};
    use obj_core::platform::FileHandle;
    use obj_core::Catalog;

    let mut pager = Pager::<FileHandle>::open(path, Config::default())?;
    pager.begin_txn();
    let catalog = Catalog::<FileHandle>::open_or_init(&mut pager)?;
    let descriptor = catalog
        .get(&mut pager, "users")?
        .expect("users collection present");
    let by_email = descriptor
        .indexes
        .iter()
        .find(|d| d.name == "by_email")
        .expect("by_email index present")
        .clone();
    let root = PageId::new(by_email.root_page_id).expect("index root non-zero");
    let mut tree = BTree::<FileHandle>::open(&pager, root)?;
    let mut key: Vec<u8> = b"zzzz-orphan@example.com".to_vec();
    key.push(0u8);
    let bogus_id: u64 = 99_999;
    tree.insert(&mut pager, &key, &bogus_id.to_be_bytes())?;
    let mut new_desc = descriptor.clone();
    for d in &mut new_desc.indexes {
        if d.name == "by_email" {
            d.root_page_id = tree.root().get();
        }
    }
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager)?;
    catalog.update(&mut pager, "users", &new_desc)?;
    let _ = pager.commit()?;
    pager.end_txn();
    Ok(())
}
