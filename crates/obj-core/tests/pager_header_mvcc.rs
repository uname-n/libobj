//! `Pager::free_page` + `alloc_fresh` route their header
//! updates through the WAL.
//!
//! `set_root_catalog` routes through the same WAL pathway, extended to
//! `free_page` and `alloc_fresh` (via `stage_or_write_header`). This
//! test pair pins the contract: header mutations are atomic with the
//! Pager-txn they belong to, and rolling back a txn rolls back the
//! header as well.

#![forbid(unsafe_code)]

use obj_core::pager::{Config, Pager};
use tempfile::TempDir;

/// A `Db::open` → drop → `Db::open` cycle (no explicit user
/// `commit`) must produce a clean reopen, and the recovered header
/// must reflect the catalog-init txn's committed state (the wiring
/// runs the catalog init inside a Pager txn that
/// commits before drop). The `freelist_head` + `page_count`
/// fields likewise ride the WAL, so the catalog-init txn's effects
/// on those fields are durable across the open → drop → open cycle.
#[test]
fn open_drop_open_recovers_post_catalog_init_header_state() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("hdr_init.obj");

    let (page_count_after_init, freelist_head_after_init, root_after_init) = {
        let mut p = Pager::open(&path, Config::default()).expect("open fresh");
        p.begin_txn();
        let a = p.alloc_page().expect("alloc a");
        let _b = p.alloc_page().expect("alloc b");
        p.free_page(a).expect("free a");
        let _ = p.commit().expect("commit init txn");
        p.end_txn();
        (p.page_count(), p.freelist_head(), p.root_catalog())
    };
    assert!(
        page_count_after_init >= 3,
        "page_count grew to at least 3 (page 0 + a + b)"
    );
    assert_eq!(
        freelist_head_after_init, 1,
        "freed page a (=1) is the freelist head"
    );

    let p2 = Pager::open(&path, Config::default()).expect("reopen");
    assert_eq!(
        p2.page_count(),
        page_count_after_init,
        "page_count must survive the open → drop → open cycle",
    );
    assert_eq!(
        p2.freelist_head(),
        freelist_head_after_init,
        "freelist_head must survive the open → drop → open cycle",
    );
    assert_eq!(
        p2.root_catalog(),
        root_after_init,
        "root_catalog must survive the open → drop → open cycle",
    );
}

/// Open a Db. Start a Pager txn. Do `alloc_page` → `free_page`
/// inside it. Drop the txn without committing. Reopen. The recovered
/// header must roll back to the pre-txn state — i.e. the alloc and
/// the free must BOTH be discarded, not just the freelist link page.
#[test]
fn rolled_back_alloc_free_does_not_advance_header_on_reopen() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("hdr_rollback.obj");

    let (baseline_page_count, baseline_freelist_head) = {
        let mut p = Pager::open(&path, Config::default()).expect("open fresh");
        p.begin_txn();
        let _a = p.alloc_page().expect("alloc baseline a");
        let _ = p.commit().expect("commit baseline");
        p.end_txn();
        (p.page_count(), p.freelist_head())
    };
    assert!(
        baseline_page_count >= 2,
        "baseline page_count includes page 0 + a"
    );

    {
        let mut p = Pager::open(&path, Config::default()).expect("open phase 2");
        p.begin_txn();
        let new_page = p.alloc_page().expect("alloc inside uncommitted txn");
        p.free_page(new_page).expect("free inside uncommitted txn");
        drop(p);
    }

    let p = Pager::open(&path, Config::default()).expect("reopen");
    assert_eq!(
        p.page_count(),
        baseline_page_count,
        "uncommitted alloc must NOT advance page_count on disk",
    );
    assert_eq!(
        p.freelist_head(),
        baseline_freelist_head,
        "uncommitted free must NOT advance freelist_head on disk",
    );
}
