//! Integrity-walker tests for corrupted B-tree page states.
//!
//! Each test hand-crafts an on-page corruption the obj writer can
//! never produce (cyclic child pointers, an over-deep descent chain,
//! a damaged trailer, an undecodable node, an index entry whose id
//! cannot be recovered, a catalog root past the end of the file) and
//! asserts the walker reports the specific [`IntegrityFailure`]
//! variant rather than looping, panicking, or erroring out.
//!
//! Covers the previously untested failure arms in
//! `obj-core/src/integrity.rs`:
//!
//! - `walk_btree` — `BTreeDepthExceeded` via `MAX_BTREE_DEPTH`.
//! - `record_btree_visit` (via `walk_btree`) —
//!   `BTreeSiblingChainBroken` descent-graph cycle detection.
//! - `read_and_validate_node` (via `walk_btree`) —
//!   `ChecksumMismatch` on a bad trailer and on an undecodable node.
//! - `cross_reference_index` — `OrphanIndexEntry { id: 0 }` when the
//!   primary id cannot be recovered from a malformed entry.
//! - `quick_check` — `DanglingCatalogPointer` for an out-of-range
//!   catalog root.
//! - `walk_freelist` — every reachable `FreelistChainBroken` arm (a
//!   `next` link past the end of the file, a chain cycle, a link to a
//!   non-freelist page, the steps-vs-`page_count` budget) plus the
//!   `ChecksumMismatch` arm for a damaged freelist trailer.
//!
//! The `MAX_RANGE_NODES` page-walk budget and the
//! `BTreeLevelInvariantViolated` / `BTreeSortViolation` re-check arms
//! are unit-tested inside `integrity.rs` itself: the budget needs >1M
//! real pages to trip through the public API, and a bad node level or
//! unsorted keys are rejected by `decode_node` (surfacing as
//! `ChecksumMismatch`) before the re-check helpers can see them.

#![forbid(unsafe_code)]

use std::collections::HashSet;

use obj_core::btree::node::{encode_node, DecodedNode, InternalEntry, NodeKind};
use obj_core::btree::{BTree, MAX_BTREE_DEPTH};
use obj_core::catalog::{IndexDescriptor, IndexStatus};
use obj_core::index::IndexKind;
use obj_core::integrity::{
    cross_reference_index, quick_check, walk_btree, walk_freelist, IntegrityFailure, TreeContext,
};
use obj_core::pager::checksum::write_page_trailer;
use obj_core::pager::freelist::{encode as encode_freelist, FreeListPage};
use obj_core::pager::page::{Page, PageId};
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Open a memory-backed pager with a transaction already begun, so
/// tests can allocate and write raw pages directly.
fn fresh_pager() -> Pager<FileHandle> {
    let mut pager = Pager::<FileHandle>::memory(Config::default()).expect("pager");
    pager.begin_txn();
    pager
}

/// Encode an internal node page with the given child page-ids and
/// pivot keys. `children.len()` must equal `pivots.len() + 1`.
fn internal_node(children: &[u64], pivots: &[&[u8]], level: u8) -> Page {
    let node = DecodedNode {
        kind: NodeKind::Internal,
        level,
        next_sibling: 0,
        children: children.to_vec(),
        leaves: Vec::new(),
        internals: pivots
            .iter()
            .map(|k| InternalEntry { key: k.to_vec() })
            .collect(),
    };
    let mut page = Page::zeroed();
    encode_node(&node, &mut page).expect("encode internal node");
    page
}

/// Encode an empty leaf node page.
fn empty_leaf_node() -> Page {
    let node = DecodedNode {
        kind: NodeKind::Leaf,
        level: 0,
        next_sibling: 0,
        children: Vec::new(),
        leaves: Vec::new(),
        internals: Vec::new(),
    };
    let mut page = Page::zeroed();
    encode_node(&node, &mut page).expect("encode leaf node");
    page
}

/// Run `walk_btree` from `root` and return the recorded failures.
fn walk_from(pager: &mut Pager<FileHandle>, root: PageId) -> Vec<IntegrityFailure> {
    let ctx = TreeContext {
        label: "primary:widgets".to_owned(),
        root,
    };
    let mut reachable: HashSet<PageId> = HashSet::new();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    walk_btree(pager, &ctx, &mut reachable, &mut failures).expect("walk_btree");
    failures
}

// ---------------------------------------------------------------------------
// BTreeSiblingChainBroken — descent-graph cycles
// ---------------------------------------------------------------------------

#[test]
fn walk_btree_detects_self_referencing_internal_node() {
    let mut pager = fresh_pager();
    let pid = pager.alloc_page().expect("alloc");
    // Internal node whose only child is itself — the tightest cycle.
    let page = internal_node(&[pid.get()], &[], 1);
    pager.write_page(pid, &page).expect("write");

    let failures = walk_from(&mut pager, pid);
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::BTreeSiblingChainBroken { tree, page_id }
                if tree == "primary:widgets" && *page_id == pid.get()
        ),
        "expected BTreeSiblingChainBroken on the self-loop page; got {failures:?}",
    );
}

#[test]
fn walk_btree_detects_shared_child_cycle() {
    let mut pager = fresh_pager();
    let leaf = pager.alloc_page().expect("alloc leaf");
    let root = pager.alloc_page().expect("alloc root");
    pager.write_page(leaf, &empty_leaf_node()).expect("write leaf");
    // Two distinct child slots of the root point at the SAME leaf —
    // the page is reachable as the child of two ancestors.
    let page = internal_node(&[leaf.get(), leaf.get()], &[b"m"], 1);
    pager.write_page(root, &page).expect("write root");

    let failures = walk_from(&mut pager, root);
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::BTreeSiblingChainBroken { page_id, .. }
                if *page_id == leaf.get()
        ),
        "expected BTreeSiblingChainBroken on the doubly-referenced leaf; got {failures:?}",
    );
}

// ---------------------------------------------------------------------------
// BTreeDepthExceeded — descent chain deeper than MAX_BTREE_DEPTH
// ---------------------------------------------------------------------------

#[test]
fn walk_btree_detects_depth_exceeded() {
    let mut pager = fresh_pager();
    // Chain of MAX_BTREE_DEPTH + 2 pages: page[i] is an internal node
    // whose single child is page[i+1]. page[i] is popped at depth i,
    // so the last page is popped at depth MAX_BTREE_DEPTH + 1 and
    // trips the bound before it is ever read (it stays unwritten).
    let mut pids: Vec<PageId> = Vec::new();
    for _ in 0..=(MAX_BTREE_DEPTH + 1) {
        pids.push(pager.alloc_page().expect("alloc"));
    }
    for pair in pids.windows(2) {
        let page = internal_node(&[pair[1].get()], &[], 1);
        pager.write_page(pair[0], &page).expect("write");
    }

    let failures = walk_from(&mut pager, pids[0]);
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::BTreeDepthExceeded { tree, limit }
                if tree == "primary:widgets" && *limit == MAX_BTREE_DEPTH
        ),
        "expected BTreeDepthExceeded at limit {MAX_BTREE_DEPTH}; got {failures:?}",
    );
}

// ---------------------------------------------------------------------------
// ChecksumMismatch — bad trailer and undecodable node via walk_btree
// ---------------------------------------------------------------------------

#[test]
fn walk_btree_reports_checksum_mismatch_on_damaged_trailer() {
    let mut pager = fresh_pager();
    let pid = pager.alloc_page().expect("alloc");
    let mut page = empty_leaf_node();
    // Flip a payload byte AFTER the trailer was stamped so the CRC
    // no longer matches the page contents.
    page.as_bytes_mut()[64] ^= 0xFF;
    pager.write_page(pid, &page).expect("write");

    let failures = walk_from(&mut pager, pid);
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::ChecksumMismatch { page_id } if *page_id == pid.get()
        ),
        "expected ChecksumMismatch on the bit-flipped page; got {failures:?}",
    );
}

#[test]
fn walk_btree_reports_checksum_mismatch_on_undecodable_node() {
    let mut pager = fresh_pager();
    let pid = pager.alloc_page().expect("alloc");
    // Valid trailer over garbage content: the page-type tag is
    // neither leaf nor internal, so decode_node rejects it even
    // though the CRC verifies.
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0x7F;
    write_page_trailer(&mut page);
    pager.write_page(pid, &page).expect("write");

    let failures = walk_from(&mut pager, pid);
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::ChecksumMismatch { page_id } if *page_id == pid.get()
        ),
        "expected ChecksumMismatch on the undecodable node; got {failures:?}",
    );
}

// ---------------------------------------------------------------------------
// OrphanIndexEntry { id: 0 } — unrecoverable id in an index entry
// ---------------------------------------------------------------------------

fn active_index(name: &str, kind: IndexKind, root_page_id: u64) -> IndexDescriptor {
    IndexDescriptor {
        index_id: 1,
        name: name.to_owned(),
        kind,
        key_paths: vec!["field".to_owned()],
        root_page_id,
        status: IndexStatus::Active,
    }
}

#[test]
fn cross_reference_standard_index_short_key_reports_orphan_id_zero() {
    let mut pager = fresh_pager();
    let mut index_tree = BTree::<FileHandle>::empty(&mut pager).expect("index tree");
    // Standard entries carry the id as the trailing 8-byte key
    // suffix; a 3-byte key cannot hold one.
    index_tree
        .insert(&mut pager, b"abc", &[])
        .expect("insert malformed entry");
    let desc = active_index("by_field", IndexKind::Standard, index_tree.root().get());

    let primary_ids: HashSet<u64> = [1u64].into_iter().collect();
    let mut referenced: HashSet<u64> = HashSet::new();
    let mut failures = Vec::new();
    cross_reference_index(
        &mut pager,
        "widgets",
        &desc,
        &primary_ids,
        &mut referenced,
        &mut failures,
    )
    .expect("cross_reference_index");

    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::OrphanIndexEntry { collection, index, id: 0 }
                if collection == "widgets" && index == "by_field"
        ),
        "expected OrphanIndexEntry with id 0 for the short key; got {failures:?}",
    );
    assert!(referenced.is_empty(), "no id can be recovered");
}

#[test]
fn cross_reference_unique_index_malformed_value_reports_orphan_id_zero() {
    let mut pager = fresh_pager();
    let mut index_tree = BTree::<FileHandle>::empty(&mut pager).expect("index tree");
    // Unique entries carry the id as the value; a 3-byte value is
    // not a valid 8-byte big-endian id.
    index_tree
        .insert(&mut pager, b"k", &[1, 2, 3])
        .expect("insert malformed entry");
    let desc = active_index("by_email", IndexKind::Unique, index_tree.root().get());

    let primary_ids: HashSet<u64> = [1u64].into_iter().collect();
    let mut referenced: HashSet<u64> = HashSet::new();
    let mut failures = Vec::new();
    cross_reference_index(
        &mut pager,
        "widgets",
        &desc,
        &primary_ids,
        &mut referenced,
        &mut failures,
    )
    .expect("cross_reference_index");

    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::OrphanIndexEntry { id: 0, .. }
        ),
        "expected OrphanIndexEntry with id 0 for the malformed value; got {failures:?}",
    );
}

// ---------------------------------------------------------------------------
// quick_check — catalog root past the end of the file
// ---------------------------------------------------------------------------

#[test]
fn quick_check_reports_dangling_catalog_root() {
    let mut pager = fresh_pager();
    let bogus_root = pager.page_count() + 9;
    pager
        .set_root_catalog(bogus_root)
        .expect("set out-of-range catalog root");

    let report = quick_check(&mut pager).expect("quick_check");
    assert_eq!(
        report.failures.len(),
        1,
        "expected one failure; got {:?}",
        report.failures,
    );
    assert!(
        matches!(
            &report.failures[0],
            IntegrityFailure::DanglingCatalogPointer { collection, index: None, page_id }
                if collection == "<catalog>" && *page_id == bogus_root
        ),
        "expected DanglingCatalogPointer on the catalog root; got {:?}",
        report.failures,
    );
    assert_eq!(report.pages_checked, 1, "only the header is inspected");
}

// ---------------------------------------------------------------------------
// FreelistChainBroken / ChecksumMismatch — corrupted freelist chains
// ---------------------------------------------------------------------------

/// Write a freelist link page at `pid` whose `next` pointer is `next`.
fn write_freelist_page(pager: &mut Pager<FileHandle>, pid: PageId, next: u64) {
    let mut page = Page::zeroed();
    encode_freelist(FreeListPage::new(next), &mut page);
    write_page_trailer(&mut page);
    pager.write_page(pid, &page).expect("write freelist page");
}

/// Run `walk_freelist` from `head` with the pager's real page count
/// and return the recorded failures.
fn walk_freelist_from(pager: &mut Pager<FileHandle>, head: u64) -> Vec<IntegrityFailure> {
    let page_count = pager.page_count();
    let mut reachable: HashSet<PageId> = HashSet::new();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    walk_freelist(pager, head, page_count, &mut reachable, &mut failures)
        .expect("walk_freelist");
    failures
}

#[test]
fn walk_freelist_empty_head_reports_nothing() {
    let mut pager = fresh_pager();
    let page_count = pager.page_count();
    let mut reachable: HashSet<PageId> = HashSet::new();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    let steps = walk_freelist(&mut pager, 0, page_count, &mut reachable, &mut failures)
        .expect("walk_freelist");
    assert_eq!(steps, 0, "an empty freelist walks zero links");
    assert!(failures.is_empty(), "no failure expected: {failures:?}");
}

#[test]
fn walk_freelist_detects_out_of_range_next_link() {
    let mut pager = fresh_pager();
    let head = pager.alloc_page().expect("alloc");
    let bogus_next = pager.page_count() + 7;
    write_freelist_page(&mut pager, head, bogus_next);

    let failures = walk_freelist_from(&mut pager, head.get());
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::FreelistChainBroken { page_id } if *page_id == bogus_next
        ),
        "expected FreelistChainBroken on the out-of-range link; got {failures:?}",
    );
}

#[test]
fn walk_freelist_detects_chain_cycle() {
    let mut pager = fresh_pager();
    let a = pager.alloc_page().expect("alloc a");
    let b = pager.alloc_page().expect("alloc b");
    // a → b → a: the second visit of `a` is the broken link.
    write_freelist_page(&mut pager, a, b.get());
    write_freelist_page(&mut pager, b, a.get());

    let failures = walk_freelist_from(&mut pager, a.get());
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::FreelistChainBroken { page_id } if *page_id == a.get()
        ),
        "expected FreelistChainBroken on the revisited link; got {failures:?}",
    );
}

#[test]
fn walk_freelist_detects_link_to_non_freelist_page() {
    let mut pager = fresh_pager();
    let head = pager.alloc_page().expect("alloc head");
    let stray = pager.alloc_page().expect("alloc stray");
    // `stray` carries a valid-trailer B-tree leaf, not a freelist
    // link — the chain points outside the freelist.
    pager
        .write_page(stray, &empty_leaf_node())
        .expect("write stray");
    write_freelist_page(&mut pager, head, stray.get());

    let failures = walk_freelist_from(&mut pager, head.get());
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::FreelistChainBroken { page_id } if *page_id == stray.get()
        ),
        "expected FreelistChainBroken on the non-freelist page; got {failures:?}",
    );
}

#[test]
fn walk_freelist_reports_checksum_mismatch_on_damaged_trailer() {
    let mut pager = fresh_pager();
    let head = pager.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    encode_freelist(FreeListPage::new(0), &mut page);
    write_page_trailer(&mut page);
    // Flip a payload byte AFTER the trailer was stamped.
    page.as_bytes_mut()[12] ^= 0xFF;
    pager.write_page(head, &page).expect("write");

    let failures = walk_freelist_from(&mut pager, head.get());
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::ChecksumMismatch { page_id } if *page_id == head.get()
        ),
        "expected ChecksumMismatch on the damaged freelist page; got {failures:?}",
    );
}

/// The steps-vs-`page_count` budget is the outermost defense of the
/// freelist walk. With an honest `page_count` a revisit always trips
/// the cycle guard first, so the budget is pinned here with a
/// degenerate zero `page_count`: the very first step exceeds it.
#[test]
fn walk_freelist_detects_chain_longer_than_page_count() {
    let mut pager = fresh_pager();
    let head = pager.alloc_page().expect("alloc");
    write_freelist_page(&mut pager, head, 0);

    let mut reachable: HashSet<PageId> = HashSet::new();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    let steps = walk_freelist(&mut pager, head.get(), 0, &mut reachable, &mut failures)
        .expect("walk_freelist");
    assert_eq!(steps, 1, "the walk stops on the step that crossed the bound");
    assert_eq!(failures.len(), 1, "expected one failure; got {failures:?}");
    assert!(
        matches!(
            &failures[0],
            IntegrityFailure::FreelistChainBroken { page_id } if *page_id == head.get()
        ),
        "expected FreelistChainBroken when the chain outruns page_count; got {failures:?}",
    );
}
