//! `Db::integrity_check` — full bidirectional walk.
//!
//! Thin orchestrator over the obj-core
//! [`obj_core::integrity`](obj_core::integrity) module: opens a read
//! snapshot, walks every tree, cross-references each `Active` index
//! against its primary, sweeps the freelist, and compares the set
//! of reachable pages to `0..page_count` to surface orphans.
//!
//! The walk holds the pager mutex for its duration — readers and
//! writers in other threads will queue behind it. Future revisions
//! may relax the locking via a snapshot-aware walker that does not
//! block writers.

use std::collections::HashSet;
use std::sync::MutexGuard;
use std::time::Instant;

use obj_core::integrity::{
    check_catalog_pointers, collect_primary_ids, cross_reference_index, walk_btree, walk_freelist,
    IntegrityFailure, IntegrityReport, TreeContext,
};
use obj_core::pager::page::PageId;
use obj_core::pager::Pager;
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor, Error, IndexKind, IndexStatus, Result};

use crate::Db;

impl Db {
    /// Run the on-demand full integrity walk and return a structured
    /// [`IntegrityReport`].
    ///
    /// The walk:
    /// 1. Opens a read snapshot (does NOT block writers).
    /// 2. Walks the catalog B-tree and every `Active` collection's
    ///    primary + index B-trees, validating per-page CRCs, sort
    ///    invariants, depth and sibling-chain invariants.
    /// 3. Cross-references each `Active` index against its primary:
    ///    every index entry must point at an extant primary id, and
    ///    every primary id must be referenced by at least one entry
    ///    in each non-`Each` `Active` index.
    /// 4. Sweeps the freelist chain.
    /// 5. Compares the set of reachable pages to `0..page_count`,
    ///    emitting [`IntegrityFailure::OrphanPage`] for each
    ///    unreferenced page id.
    ///
    /// I/O failures during the walk surface as `Err(_)`; content-
    /// level violations are accumulated into
    /// `report.failures` and the walk continues.
    ///
    /// A lighter-weight catalog-only walk runs automatically at
    /// [`Db::open`] time; opt out via
    /// [`Config::skip_open_check`](crate::Config::skip_open_check).
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "users_integrity_doc")]
    /// struct User { email: String }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("check.obj"))?;
    /// for i in 0..16u32 {
    ///     let _ = db.insert(User { email: format!("u{i}@example.com") })?;
    /// }
    /// let report = db.integrity_check()?;
    /// assert!(report.is_ok(), "clean db must pass: {:?}", report.failures);
    /// assert!(report.pages_checked > 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on cache-miss read failure during the walk.
    /// - [`Error::Busy`] if the pager mutex is poisoned.
    /// - Pager / B-tree errors propagated from the catalog walk.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "db.integrity_check", level = "info", skip_all)
    )]
    pub fn integrity_check(&self) -> Result<IntegrityReport> {
        let start = Instant::now();
        let mut state = IntegrityState::new();
        let mut pager = lock_pager(self)?;
        state.pages_checked = state.pages_checked.saturating_add(1);
        walk_catalog(&mut pager, &mut state)?;
        walk_collections(&mut pager, &mut state)?;
        walk_freelist_chain(&mut pager, &mut state)?;
        detect_orphan_pages(&pager, &mut state);
        Ok(IntegrityReport::new(
            state.failures,
            state.pages_checked,
            start.elapsed(),
        ))
    }
}

/// Working state for the integrity walk. Accumulated as the walk
/// progresses; consumed when the [`IntegrityReport`] is constructed.
struct IntegrityState {
    failures: Vec<IntegrityFailure>,
    reachable: HashSet<PageId>,
    pages_checked: u64,
}

impl IntegrityState {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
            reachable: HashSet::new(),
            pages_checked: 0,
        }
    }
}

fn lock_pager(db: &Db) -> Result<MutexGuard<'_, Pager<FileHandle>>> {
    db.env.pager().lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

fn walk_catalog(pager: &mut Pager<FileHandle>, state: &mut IntegrityState) -> Result<()> {
    let raw = pager.root_catalog();
    let Some(root) = PageId::new(raw) else {
        return Ok(());
    };
    let page_count = pager.page_count();
    if root.get() >= page_count {
        state
            .failures
            .push(IntegrityFailure::DanglingCatalogPointer {
                collection: "<catalog>".to_owned(),
                index: None,
                page_id: root.get(),
            });
        return Ok(());
    }
    let ctx = TreeContext {
        label: "catalog".to_owned(),
        root,
    };
    let walked = walk_btree(pager, &ctx, &mut state.reachable, &mut state.failures)?;
    state.pages_checked = state.pages_checked.saturating_add(walked);
    Ok(())
}

fn walk_collections(pager: &mut Pager<FileHandle>, state: &mut IntegrityState) -> Result<()> {
    let raw = pager.root_catalog();
    if PageId::new(raw).is_none() {
        return Ok(());
    }
    let catalog = match Catalog::<FileHandle>::open_or_init(pager) {
        Ok(c) => c,
        Err(Error::Corruption { .. }) => return Ok(()),
        Err(e) => return Err(e),
    };
    let rows = match catalog.list_collections(pager) {
        Ok(r) => r,
        Err(Error::Corruption { .. }) => return Ok(()),
        Err(e) => return Err(e),
    };
    let page_count = pager.page_count();
    for (name, descriptor) in rows {
        check_catalog_pointers(&name, &descriptor, page_count, &mut state.failures);
        walk_one_collection(pager, &name, &descriptor, state)?;
    }
    Ok(())
}

fn walk_one_collection(
    pager: &mut Pager<FileHandle>,
    name: &str,
    descriptor: &CollectionDescriptor,
    state: &mut IntegrityState,
) -> Result<()> {
    walk_primary_tree(pager, name, descriptor, state)?;
    let mut primary_ids: HashSet<u64> = HashSet::new();
    let _scanned = collect_primary_ids(pager, descriptor, &mut primary_ids)?;
    let mut per_index: Vec<(String, IndexKind, HashSet<u64>)> = Vec::new();
    for index in &descriptor.indexes {
        if index.status != IndexStatus::Active {
            continue;
        }
        walk_index_tree(pager, name, descriptor, index, state)?;
        let mut referenced: HashSet<u64> = HashSet::new();
        let _entries = cross_reference_index::<FileHandle>(
            pager,
            name,
            index,
            &primary_ids,
            &mut referenced,
            &mut state.failures,
        )?;
        per_index.push((index.name.clone(), index.kind, referenced));
    }
    obj_core::integrity::check_primary_to_index(
        name,
        descriptor,
        &primary_ids,
        &per_index,
        &mut state.failures,
    );
    Ok(())
}

fn walk_primary_tree(
    pager: &mut Pager<FileHandle>,
    name: &str,
    descriptor: &CollectionDescriptor,
    state: &mut IntegrityState,
) -> Result<()> {
    let page_count = pager.page_count();
    let Some(root) = PageId::new(descriptor.primary_root) else {
        return Ok(());
    };
    if root.get() >= page_count {
        return Ok(());
    }
    let ctx = TreeContext {
        label: format!("primary:{name}"),
        root,
    };
    let walked = walk_btree(pager, &ctx, &mut state.reachable, &mut state.failures)?;
    state.pages_checked = state.pages_checked.saturating_add(walked);
    Ok(())
}

fn walk_index_tree(
    pager: &mut Pager<FileHandle>,
    collection: &str,
    _descriptor: &CollectionDescriptor,
    index: &obj_core::IndexDescriptor,
    state: &mut IntegrityState,
) -> Result<()> {
    let page_count = pager.page_count();
    let Some(root) = PageId::new(index.root_page_id) else {
        return Ok(());
    };
    if root.get() >= page_count {
        return Ok(());
    }
    let ctx = TreeContext {
        label: format!("index:{}.{}", collection, index.name),
        root,
    };
    let walked = walk_btree(pager, &ctx, &mut state.reachable, &mut state.failures)?;
    state.pages_checked = state.pages_checked.saturating_add(walked);
    Ok(())
}

fn walk_freelist_chain(pager: &mut Pager<FileHandle>, state: &mut IntegrityState) -> Result<()> {
    let head = pager.freelist_head();
    let page_count = pager.page_count();
    let walked = walk_freelist(
        pager,
        head,
        page_count,
        &mut state.reachable,
        &mut state.failures,
    )?;
    state.pages_checked = state.pages_checked.saturating_add(walked);
    Ok(())
}

fn detect_orphan_pages(pager: &Pager<FileHandle>, state: &mut IntegrityState) {
    let page_count = pager.page_count();
    let mut id: u64 = 1;
    while id < page_count {
        if let Some(pid) = PageId::new(id) {
            if !state.reachable.contains(&pid) {
                state
                    .failures
                    .push(IntegrityFailure::OrphanPage { page_id: id });
            }
        }
        id = id.saturating_add(1);
    }
}
