//! Main-file bit-rot detection.
//!
//! The 10k-seed crash-cycle campaign (`crash_cycles.rs`) deliberately
//! clamps main-file bit-flips to zero: `FaultyOpener::main_plan` sets
//! `bit_flip_prob = 0.0`, and `verify_recovery_lenient` documents that
//! its `Error::Corruption` arm is therefore "unreachable in the
//! current configuration". A torn write or flipped bit on the MAIN
//! file is unrecoverable bit-rot — not a WAL-layer concern — so the
//! crash matrix never injects one. That leaves the page-trailer CRC
//! corruption-detection branch unexercised by the campaign.
//!
//! This is the missing small, deterministic test for that branch. It
//! is distinct from the torn-write / dropped-fsync cases the crash
//! matrix covers: here we let a clean write + commit + checkpoint land
//! the page durably on the MAIN file, then flip a single committed bit
//! on disk (post-fsync, simulating media rot / an out-of-band
//! mutation) and assert the integrity walk surfaces it as a
//! `ChecksumMismatch`.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use obj_core::btree::BTree;
use obj_core::catalog::{Catalog, CollectionDescriptor};
use obj_core::codec::encode;
use obj_core::id::Id;
use obj_core::integrity::{walk_btree, IntegrityFailure, TreeContext};
use obj_core::pager::page::{PageId, PAGE_SIZE};
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use obj_core::Document;

use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// A trivial stored document — exercised purely to give the primary
/// B-tree real, CRC-trailed leaf pages to corrupt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Widget {
    label: String,
    weight: u64,
}

impl Document for Widget {
    const COLLECTION: &'static str = "widgets";
    const VERSION: u32 = 1;
}

/// Number of documents inserted into the primary tree. Enough to fill
/// a leaf with a non-trivial slot directory so the flipped byte lands
/// inside real payload, not padding.
const DOC_COUNT: u64 = 24;

/// Build a single-collection database at the obj-core layer: init the
/// catalog, build a primary B-tree, insert `DOC_COUNT` encoded
/// documents, register the collection, commit, and checkpoint+close so
/// the bytes are durable on the MAIN file (no WAL overlay masks the
/// later flip). Returns the primary tree's root page-id.
fn build_db(path: &std::path::Path) -> u64 {
    let mut pager = Pager::<FileHandle>::open(path, Config::default()).expect("open pager");
    pager.begin_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init catalog");
    let mut primary = BTree::<FileHandle>::empty(&mut pager).expect("primary tree");
    let collection_id: u32 = 0;
    for n in 1..=DOC_COUNT {
        let id = Id::try_new(n).expect("non-zero id");
        let doc = Widget {
            label: format!("widget-{n:04}"),
            weight: n.wrapping_mul(7),
        };
        let value = encode(&doc, collection_id).expect("encode doc");
        primary
            .insert(&mut pager, &id.to_be_bytes(), &value)
            .expect("primary insert");
    }
    let root = primary.root().get();
    let descriptor = CollectionDescriptor::new(collection_id, root, Widget::VERSION);
    catalog
        .insert(&mut pager, Widget::COLLECTION, descriptor)
        .expect("register collection");
    let _ = pager.commit().expect("commit");
    pager.end_txn();
    pager.close().expect("checkpoint + close");
    root
}

/// Resolve the primary B-tree root of the `widgets` collection by
/// re-reading the catalog from the freshly-closed file. Confirms the
/// committed root id survived the checkpoint, independent of the
/// value `build_db` returned.
fn locate_primary_root(path: &std::path::Path) -> u64 {
    let mut pager = Pager::<FileHandle>::open(path, Config::default()).expect("reopen pager");
    pager.begin_txn();
    let catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("reopen catalog");
    let descriptor = catalog
        .get(&mut pager, Widget::COLLECTION)
        .expect("catalog get")
        .expect("widgets present");
    pager.end_txn();
    descriptor.primary_root
}

/// Flip a single bit at `byte_offset` within page `pid` on the MAIN
/// file (plaintext stride = `PAGE_SIZE`). Reads the current byte and
/// XORs one bit so the mutation is always an actual change.
fn flip_bit_in_page(path: &std::path::Path, pid: u64, byte_offset: u64) {
    let page_size = PAGE_SIZE as u64;
    debug_assert!(pid >= 1, "page 0 is the header, not a tree node");
    debug_assert!(
        byte_offset < page_size,
        "offset must land inside the page body",
    );
    let abs = pid.saturating_mul(page_size) + byte_offset;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open for corruption");
    file.seek(SeekFrom::Start(abs)).expect("seek");
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).expect("read byte");
    file.seek(SeekFrom::Start(abs)).expect("seek back");
    byte[0] ^= 0x01;
    file.write_all(&byte).expect("flip bit");
    file.flush().expect("flush");
}

/// Walk the primary B-tree from `root` and return its recorded
/// integrity failures.
fn walk_primary(path: &std::path::Path, root: u64) -> Vec<IntegrityFailure> {
    let mut pager = Pager::<FileHandle>::open(path, Config::default()).expect("reopen for walk");
    let root_pid = PageId::new(root).expect("non-zero primary root");
    let ctx = TreeContext {
        label: format!("primary:{}", Widget::COLLECTION),
        root: root_pid,
    };
    let mut reachable: HashSet<PageId> = HashSet::new();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    let _walked =
        walk_btree(&mut pager, &ctx, &mut reachable, &mut failures).expect("walk_btree completes");
    failures
}

#[test]
fn clean_main_file_walk_reports_no_checksum_mismatch() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("clean.obj");
    let root = build_db(&path);
    let failures = walk_primary(&path, root);
    assert!(
        !failures
            .iter()
            .any(|f| matches!(f, IntegrityFailure::ChecksumMismatch { .. })),
        "clean main file must not report a checksum mismatch; got {failures:?}",
    );
}

#[test]
fn flipped_committed_main_page_bit_surfaces_checksum_mismatch() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("bitrot.obj");
    let _root = build_db(&path);
    let root = locate_primary_root(&path);
    assert_ne!(root, 0, "primary root must be non-zero after checkpoint");

    flip_bit_in_page(&path, root, 64);

    let failures = walk_primary(&path, root);
    let detected = failures.iter().any(|f| {
        matches!(
            f,
            IntegrityFailure::ChecksumMismatch { page_id } if *page_id == root
        )
    });
    assert!(
        detected,
        "expected ChecksumMismatch on page {root}; got {failures:?}",
    );
}
