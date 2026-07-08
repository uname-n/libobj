//! On-disk integrity check.
//!
//! [`IntegrityReport`] is the structured output of
//! [`crate::pager::Pager`]-level walks that verify every documented
//! invariant of the `.obj` file format. The public driver lives in
//! the `obj` crate (`obj::Db::integrity_check`); this module hosts
//! the data types and the obj-core-internal helpers that do the
//! heavy lifting.

#![forbid(unsafe_code)]
// allow: the public walk helpers take `HashSet<_>` with the default hasher by design; adding a generic `S` hasher param would leak into the public surface for no benefit.
#![allow(clippy::implicit_hasher)]

use std::collections::HashSet;
use std::time::Duration;

use crate::btree::node::{decode_node, NodeKind};
use crate::btree::{MAX_BTREE_DEPTH, MAX_RANGE_NODES};
use crate::catalog::{Catalog, CollectionDescriptor, IndexDescriptor, IndexStatus};
use crate::error::{Error, Result};
use crate::id::Id;
use crate::index::IndexKind;
use crate::pager::checksum::page_trailer_valid;
use crate::pager::freelist::decode as decode_freelist_page;
use crate::pager::page::PageId;
use crate::pager::Pager;
use crate::platform::FileBackend;

use heapless::Vec as HeaplessVec;

/// Categorical reasons an integrity walk records a failure. Every
/// variant carries the locus of the problem (page id, collection,
/// index name, document id) so an operator can root-cause without
/// re-running the check.
///
/// The integrity walker accumulates a `Vec<IntegrityFailure>` and
/// returns it inside an [`IntegrityReport`] — every failure here is
/// recoverable in the sense that the check **completed**; the
/// caller decides whether to repair the file or refuse to open it.
/// I/O failures during the walk surface as an outer `Result::Err`
/// instead (the walk could not finish).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum IntegrityFailure {
    /// A page's CRC32C trailer did not verify. The page-id is the
    /// locus; page `0` would be a header CRC failure (the file
    /// header carries its own CRC, not a page trailer — this
    /// variant is emitted for any page whose trailer fails the
    /// per-page integrity check).
    ChecksumMismatch {
        /// Page where the corruption was detected.
        page_id: u64,
    },
    /// A page in `1..page_count` is not reachable from any root the
    /// walker recognises (catalog, freelist, any primary or index
    /// B-tree). Indicates either a leaked page (allocated, never
    /// linked) or a corrupted root pointer somewhere upstream.
    OrphanPage {
        /// The unreferenced page id.
        page_id: u64,
    },
    /// An index B-tree entry references a primary id that does NOT
    /// exist in the collection's primary B-tree. Indicates the
    /// index is out-of-sync (likely a partial write that survived
    /// recovery, or a forensic mutation outside obj).
    OrphanIndexEntry {
        /// Owning collection name.
        collection: String,
        /// Owning index name.
        index: String,
        /// Primary id the index entry pointed at.
        id: u64,
    },
    /// A primary B-tree entry has no matching entry in one of the
    /// collection's `Active` indexes. The walker checks the
    /// per-index `(id-suffix-or-value → at-least-one-entry)`
    /// invariant; an `Each` index with an empty sequence is NOT
    /// reported (legal: the doc contributes no keys). For
    /// `Standard` / `Unique` / `Composite` the absence of any
    /// matching entry is a violation.
    MissingIndexEntry {
        /// Owning collection name.
        collection: String,
        /// Owning index name.
        index: String,
        /// Primary id with no matching index entry.
        id: u64,
    },
    /// A B+tree node failed the sort invariant: a key was followed
    /// by a strictly-lesser-or-equal key inside the same node.
    BTreeSortViolation {
        /// Page where the violation was detected.
        page_id: u64,
    },
    /// A B+tree's child-pointer graph contains a cycle — the same
    /// page is reachable as the child of two distinct ancestors (or
    /// of itself). The `tree` field names the containing B-tree by
    /// a human-readable label (`"catalog"`,
    /// `"primary:<collection>"`, `"index:<collection>.<index>"`).
    ///
    /// The variant name is preserved for compatibility
    /// — it now signals only the descent-graph cycle path.
    BTreeSiblingChainBroken {
        /// Label identifying which B-tree the violation surfaced on.
        tree: String,
        /// Page where the cycle was detected.
        page_id: u64,
    },
    /// A B+tree's depth exceeded
    /// [`crate::btree::MAX_BTREE_DEPTH`]. Indicates a runaway tree
    /// shape — by construction the obj writer cannot produce one;
    /// surfacing here means the file was mutated outside obj.
    BTreeDepthExceeded {
        /// Label identifying which B-tree tripped the depth bound.
        tree: String,
        /// The bound that was exceeded.
        limit: usize,
    },
    /// A B+tree node's `level` value was inconsistent with its
    /// kind — leaves must be level 0, internals strictly positive,
    /// and the level must decrease by exactly 1 when descending
    /// from an internal node to its child.
    BTreeLevelInvariantViolated {
        /// Label identifying the B-tree.
        tree: String,
        /// Page where the violation was detected.
        page_id: u64,
    },
    /// A `CollectionDescriptor` carried a `primary_root` or
    /// `IndexDescriptor.root_page_id` that does not point at a
    /// page id in `1..page_count` — the catalog references a page
    /// the file does not contain.
    DanglingCatalogPointer {
        /// Owning collection name.
        collection: String,
        /// Optional index name when the dangling pointer is on an
        /// index descriptor; `None` when it is the
        /// `primary_root` itself.
        index: Option<String>,
        /// The out-of-range page id.
        page_id: u64,
    },
    /// The freelist chain was broken — a `next` link pointed at a
    /// non-freelist page (or a freelist page failed to decode), at
    /// a page id past `page_count`, or the chain looped.
    FreelistChainBroken {
        /// Page id where the broken link was detected.
        page_id: u64,
    },
}

/// Structured result of an [integrity check](crate::integrity).
///
/// The public driver returns this inside `Ok(_)` whenever the walk
/// completes (whether or not it found any failures). An `Err`
/// outer result means the walk could not finish — an I/O failure,
/// not a content-level integrity violation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct IntegrityReport {
    /// Every detected violation, in walker-encounter order.
    pub failures: Vec<IntegrityFailure>,
    /// Pages the walker actually inspected (header + every tree
    /// node + every freelist link). Provides a sanity-check vs
    /// `page_count`: a check that ran without errors but visited
    /// noticeably fewer pages than the file holds suggests a
    /// reachability problem (the [`IntegrityFailure::OrphanPage`]
    /// failures cover the concrete cases).
    pub pages_checked: u64,
    /// Wall-clock duration the walk took. Useful for the operator
    /// to know whether the check finished within a maintenance
    /// window.
    pub elapsed: Duration,
}

impl IntegrityReport {
    /// Assemble a report from its parts.
    ///
    /// Provided so callers outside `obj-core` (the `obj` layer's
    /// `integrity_check` driver) can build a report now that the
    /// struct is `#[non_exhaustive]` and can no longer be
    /// struct-literal-constructed across crate boundaries.
    #[must_use]
    pub fn new(failures: Vec<IntegrityFailure>, pages_checked: u64, elapsed: Duration) -> Self {
        Self {
            failures,
            pages_checked,
            elapsed,
        }
    }

    /// `true` iff no failure was recorded — the report indicates a
    /// healthy database.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Snapshot of the on-disk shape needed to walk a single B-tree.
/// Used by the obj layer's wrapper to thread per-tree context
/// (label, root) into the obj-core helper.
#[derive(Debug, Clone)]
pub struct TreeContext {
    /// Human-readable label used in failure messages (`"catalog"`,
    /// `"primary:users"`, `"index:users.by_email"`).
    pub label: String,
    /// Root page id for the tree.
    pub root: PageId,
}

/// Walk a single B-tree from `ctx.root`, recording any per-node
/// invariant violations into `failures` and inserting every visited
/// node page-id into `reachable`. Returns the number of pages this
/// walk inspected.
///
/// The walk traverses the tree level by level via an explicit
/// `Vec<PageId>` queue and a `HashSet<PageId>` already-seen guard
/// (so a cycle introduced by an out-of-band mutation cannot
/// infinitely-loop the walker). Bounded by
/// [`MAX_RANGE_NODES`] per tree.
///
/// # Errors
///
/// Returns [`Error::Io`] on cache-miss read failure. Never returns
/// an `Error::Corruption` directly — corruption surfaces as a
/// [`IntegrityFailure`] entry in `failures`.
pub fn walk_btree<F: FileBackend>(
    pager: &mut Pager<F>,
    ctx: &TreeContext,
    reachable: &mut HashSet<PageId>,
    failures: &mut Vec<IntegrityFailure>,
) -> Result<u64> {
    let mut queue: Vec<(PageId, u32)> = Vec::new();
    queue.push((ctx.root, 0));
    let mut visited: HashSet<PageId> = HashSet::new();
    let mut pages_walked: u64 = 0;
    let mut visit = WalkVisitState {
        visited: &mut visited,
        reachable,
        pages_walked: &mut pages_walked,
        failures,
    };
    while let Some((pid, depth)) = queue.pop() {
        if depth as usize > MAX_BTREE_DEPTH {
            visit.failures.push(IntegrityFailure::BTreeDepthExceeded {
                tree: ctx.label.clone(),
                limit: MAX_BTREE_DEPTH,
            });
            return Ok(*visit.pages_walked);
        }
        match record_btree_visit(pid, ctx, &mut visit) {
            VisitStep::Cycle => continue,
            VisitStep::BudgetExceeded => return Ok(*visit.pages_walked),
            VisitStep::Walked => {}
        }
        let Some(decoded) = read_and_validate_node(pager, pid, &ctx.label, visit.failures)? else {
            continue;
        };
        if matches!(decoded.kind, NodeKind::Internal) {
            for raw in &decoded.children {
                if let Some(child) = PageId::new(*raw) {
                    queue.push((child, depth + 1));
                }
            }
        }
    }
    let final_pages_walked = *visit.pages_walked;
    Ok(final_pages_walked)
}

/// Mutable per-walk state shared by the [`walk_btree`] driver loop and
/// its per-node helper.  Grouped into one struct so the helper takes
/// a single reference rather than a long argument list.
struct WalkVisitState<'a> {
    /// Set of page-ids already popped off the queue during this walk;
    /// used to detect cycles introduced by out-of-band mutation.
    visited: &'a mut HashSet<PageId>,
    /// Caller-owned set of every page-id reachable from the root.
    reachable: &'a mut HashSet<PageId>,
    /// Per-tree count of pages successfully popped + counted, used to
    /// enforce the `MAX_RANGE_NODES` budget.
    pages_walked: &'a mut u64,
    /// Caller-owned failure log; every detected invariant violation is
    /// pushed here rather than surfaced as `Err`.
    failures: &'a mut Vec<IntegrityFailure>,
}

/// Outcome of [`record_btree_visit`]: the cycle guard, reachable-set
/// update, and per-tree node-count budget consolidated into a single
/// dispatch so the [`walk_btree`] driver loop reads as a flat
/// state-machine step.
enum VisitStep {
    /// The page was already visited in this walk — a cycle.  The
    /// failure has been recorded; the caller should skip this entry
    /// and pop the next queued page.
    Cycle,
    /// The per-tree node-count budget has just been crossed.  The
    /// failure has been recorded; the caller must stop the walk and
    /// return the current `pages_walked` count.
    BudgetExceeded,
    /// The page was newly visited and counted.  The caller may now
    /// validate the node and enqueue its children.
    Walked,
}

/// Apply the cycle guard, mark `pid` reachable, and tick the per-tree
/// node-count budget.  Records the appropriate [`IntegrityFailure`]
/// when a cycle is detected or the budget is exceeded.
fn record_btree_visit(pid: PageId, ctx: &TreeContext, visit: &mut WalkVisitState<'_>) -> VisitStep {
    if !visit.visited.insert(pid) {
        visit
            .failures
            .push(IntegrityFailure::BTreeSiblingChainBroken {
                tree: ctx.label.clone(),
                page_id: pid.get(),
            });
        return VisitStep::Cycle;
    }
    visit.reachable.insert(pid);
    *visit.pages_walked = visit.pages_walked.saturating_add(1);
    if *visit.pages_walked > MAX_RANGE_NODES as u64 {
        visit.failures.push(IntegrityFailure::BTreeDepthExceeded {
            tree: ctx.label.clone(),
            limit: MAX_RANGE_NODES,
        });
        return VisitStep::BudgetExceeded;
    }
    VisitStep::Walked
}

/// Read a B-tree node page and verify per-page invariants (trailer
/// CRC + decoder agreement + sort invariant). Returns the decoded
/// node when validation passes; returns `Ok(None)` and records the
/// failure when the page is so broken decoding cannot continue
/// (the caller treats this as "do not descend further" and moves
/// on to the next queued page).
fn read_and_validate_node<F: FileBackend>(
    pager: &mut Pager<F>,
    pid: PageId,
    tree_label: &str,
    failures: &mut Vec<IntegrityFailure>,
) -> Result<Option<crate::btree::node::DecodedNode>> {
    let page = match pager.read_page(pid) {
        Ok(pr) => pr.to_owned_page(),
        Err(Error::Corruption { page_id }) => {
            failures.push(IntegrityFailure::ChecksumMismatch { page_id });
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    if !page_trailer_valid(&page) {
        failures.push(IntegrityFailure::ChecksumMismatch { page_id: pid.get() });
        return Ok(None);
    }
    if let Ok(decoded) = decode_node(page.as_bytes()) {
        verify_sort_invariant(pid, &decoded, failures);
        verify_level_invariant(pid, &decoded, tree_label, failures);
        Ok(Some(decoded))
    } else {
        failures.push(IntegrityFailure::ChecksumMismatch { page_id: pid.get() });
        Ok(None)
    }
}

/// Confirm the node's keys are in strictly-ascending order.
fn verify_sort_invariant(
    pid: PageId,
    decoded: &crate::btree::node::DecodedNode,
    failures: &mut Vec<IntegrityFailure>,
) {
    let mut prev: Option<&[u8]> = None;
    let keys: Box<dyn Iterator<Item = &[u8]> + '_> = match decoded.kind {
        NodeKind::Leaf => Box::new(decoded.leaves.iter().map(|e| e.key.as_slice())),
        NodeKind::Internal => Box::new(decoded.internals.iter().map(|e| e.key.as_slice())),
    };
    for key in keys {
        if let Some(p) = prev {
            if key <= p {
                failures.push(IntegrityFailure::BTreeSortViolation { page_id: pid.get() });
                return;
            }
        }
        prev = Some(key);
    }
}

/// Confirm leaf-level = 0 and internal-level > 0.
fn verify_level_invariant(
    pid: PageId,
    decoded: &crate::btree::node::DecodedNode,
    tree_label: &str,
    failures: &mut Vec<IntegrityFailure>,
) {
    let level_ok = match decoded.kind {
        NodeKind::Leaf => decoded.level == 0,
        NodeKind::Internal => decoded.level > 0,
    };
    if !level_ok {
        failures.push(IntegrityFailure::BTreeLevelInvariantViolated {
            tree: tree_label.to_owned(),
            page_id: pid.get(),
        });
    }
}

/// Walk the freelist chain starting at `head_raw`, inserting every link
/// page into `reachable` and recording any chain breakage into
/// `failures`. Returns the number of freelist link pages walked.
///
/// Bounded by `page_count` (a freelist chain cannot have more entries
/// than the file has pages).
///
/// # Errors
///
/// Returns [`Error::Io`] on cache-miss read failure. Content-level
/// breakage is recorded into `failures` (the walk does NOT abort
/// on a broken link; the walker records the failure and stops the
/// freelist sweep so a subsequent walk does not loop).
pub fn walk_freelist<F: FileBackend>(
    pager: &mut Pager<F>,
    head_raw: u64,
    page_count: u64,
    reachable: &mut HashSet<PageId>,
    failures: &mut Vec<IntegrityFailure>,
) -> Result<u64> {
    if head_raw == 0 {
        return Ok(0);
    }
    let mut current = head_raw;
    let mut steps: u64 = 0;
    let mut seen: HashSet<PageId> = HashSet::new();
    while current != 0 {
        steps = steps.saturating_add(1);
        if steps > page_count {
            failures.push(IntegrityFailure::FreelistChainBroken { page_id: current });
            return Ok(steps);
        }
        let Some(pid) = PageId::new(current) else {
            failures.push(IntegrityFailure::FreelistChainBroken { page_id: current });
            return Ok(steps);
        };
        if pid.get() >= page_count {
            failures.push(IntegrityFailure::FreelistChainBroken { page_id: pid.get() });
            return Ok(steps);
        }
        if !seen.insert(pid) {
            failures.push(IntegrityFailure::FreelistChainBroken { page_id: pid.get() });
            return Ok(steps);
        }
        reachable.insert(pid);
        let page = match pager.read_page(pid) {
            Ok(pr) => pr.to_owned_page(),
            Err(Error::Corruption { page_id }) => {
                failures.push(IntegrityFailure::ChecksumMismatch { page_id });
                return Ok(steps);
            }
            Err(e) => return Err(e),
        };
        if !page_trailer_valid(&page) {
            failures.push(IntegrityFailure::ChecksumMismatch { page_id: pid.get() });
            return Ok(steps);
        }
        let Some(entry) = decode_freelist_page(&page) else {
            failures.push(IntegrityFailure::FreelistChainBroken { page_id: pid.get() });
            return Ok(steps);
        };
        current = entry.next;
    }
    Ok(steps)
}

/// Confirm the catalog descriptor's `primary_root` + each `Active`
/// index's `root_page_id` point at pages within `1..page_count`.
/// Records `DanglingCatalogPointer` failures for every miss.
pub fn check_catalog_pointers(
    name: &str,
    descriptor: &CollectionDescriptor,
    page_count: u64,
    failures: &mut Vec<IntegrityFailure>,
) {
    if descriptor.primary_root == 0 || descriptor.primary_root >= page_count {
        failures.push(IntegrityFailure::DanglingCatalogPointer {
            collection: name.to_owned(),
            index: None,
            page_id: descriptor.primary_root,
        });
    }
    for index in &descriptor.indexes {
        if index.status != IndexStatus::Active {
            continue;
        }
        if index.root_page_id == 0 || index.root_page_id >= page_count {
            failures.push(IntegrityFailure::DanglingCatalogPointer {
                collection: name.to_owned(),
                index: Some(index.name.clone()),
                page_id: index.root_page_id,
            });
        }
    }
}

/// Verify every entry in an `Active` index B-tree (`Standard`,
/// `Each`, `Composite`: the trailing 8-byte key suffix is the id;
/// `Unique`: the value is the id) references a primary id that
/// exists in the primary B-tree.
///
/// `primary_ids` is the set of every id present in the collection's
/// primary tree (pre-computed by the caller's primary walk).
/// `referenced_ids` (out-param) accumulates the set of primary ids
/// referenced by AT LEAST ONE index entry — the caller uses it to
/// run the Primary → Index `MissingIndexEntry` check.
///
/// # Errors
///
/// Returns [`Error::Io`] on cache-miss read failure. Content-level
/// breakage is recorded into `failures` and iteration continues.
pub fn cross_reference_index<F: FileBackend>(
    pager: &mut Pager<F>,
    collection: &str,
    index: &IndexDescriptor,
    primary_ids: &HashSet<u64>,
    referenced_ids: &mut HashSet<u64>,
    failures: &mut Vec<IntegrityFailure>,
) -> Result<u64> {
    let Some(root) = PageId::new(index.root_page_id) else {
        return Ok(0);
    };
    let tree = crate::btree::BTree::<F>::open(pager, root)?;
    let iter = match tree.range(pager, ..) {
        Ok(it) => it,
        Err(Error::Corruption { page_id }) => {
            failures.push(IntegrityFailure::ChecksumMismatch { page_id });
            return Ok(0);
        }
        Err(e) => return Err(e),
    };
    let mut scanned: u64 = 0;
    for step in iter {
        scanned = scanned
            .checked_add(1)
            .ok_or(Error::BTreeInvariantViolated {
                reason: "index entry count overflow",
            })?;
        let (full_key, value) = match step {
            Ok(kv) => kv,
            Err(Error::Corruption { page_id }) => {
                failures.push(IntegrityFailure::ChecksumMismatch { page_id });
                return Ok(scanned);
            }
            Err(e) => return Err(e),
        };
        let Some(raw_id) = recover_id_from_entry(&full_key, &value, index.kind) else {
            failures.push(IntegrityFailure::OrphanIndexEntry {
                collection: collection.to_owned(),
                index: index.name.clone(),
                id: 0,
            });
            continue;
        };
        if !primary_ids.contains(&raw_id) {
            failures.push(IntegrityFailure::OrphanIndexEntry {
                collection: collection.to_owned(),
                index: index.name.clone(),
                id: raw_id,
            });
            continue;
        }
        referenced_ids.insert(raw_id);
    }
    Ok(scanned)
}

/// Recover the `Id` from one index B-tree entry. Mirrors the
/// `Collection<T>::id_from_index_entry` private helper but is
/// type-agnostic. For `Unique` indexes the id is the value; for
/// every other kind it is the trailing 8-byte big-endian suffix of
/// the key.
fn recover_id_from_entry(full_key: &[u8], value: &[u8], kind: IndexKind) -> Option<u64> {
    let bytes: &[u8] = match kind {
        IndexKind::Unique => value,
        IndexKind::Standard | IndexKind::Each | IndexKind::Composite => {
            if full_key.len() < 8 {
                return None;
            }
            &full_key[full_key.len() - 8..]
        }
    };
    let id = Id::from_be_bytes(bytes)?;
    Some(id.get())
}

/// Walk every entry in the primary B-tree of `descriptor`,
/// inserting each id into `primary_ids` and recording any
/// per-page failures (CRC, sort, depth) along the way. Returns the
/// number of leaf entries scanned.
///
/// # Errors
///
/// Returns [`Error::Io`] on cache-miss read failure. Content-level
/// breakage is recorded into `failures` (via `walk_btree` if the
/// caller paired the two walks) or surfaced as an early return
/// from this function on a B-tree decode failure.
pub fn collect_primary_ids<F: FileBackend>(
    pager: &mut Pager<F>,
    descriptor: &CollectionDescriptor,
    primary_ids: &mut HashSet<u64>,
) -> Result<u64> {
    let Some(root) = PageId::new(descriptor.primary_root) else {
        return Ok(0);
    };
    let tree = crate::btree::BTree::<F>::open(pager, root)?;
    let iter = match tree.range(pager, ..) {
        Ok(it) => it,
        Err(Error::Corruption { .. }) => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut count: u64 = 0;
    for step in iter {
        count = count.checked_add(1).ok_or(Error::BTreeInvariantViolated {
            reason: "primary entry count overflow",
        })?;
        let (key, _value) = match step {
            Ok(kv) => kv,
            Err(Error::Corruption { .. }) => return Ok(count),
            Err(e) => return Err(e),
        };
        if let Some(id) = Id::from_be_bytes(&key) {
            primary_ids.insert(id.get());
        }
    }
    Ok(count)
}

/// Build the set of every primary id that has AT LEAST ONE
/// surviving index entry across the collection's `Active` indexes,
/// and emit `MissingIndexEntry` failures for any primary id that is
/// NOT referenced by an index whose kind is `Standard`, `Unique`,
/// or `Composite`. `Each` indexes are exempted — they legally
/// reference zero entries when the source sequence is empty.
pub fn check_primary_to_index(
    collection: &str,
    descriptor: &CollectionDescriptor,
    primary_ids: &HashSet<u64>,
    per_index_referenced: &[(String, IndexKind, HashSet<u64>)],
    failures: &mut Vec<IntegrityFailure>,
) {
    for (index_name, kind, referenced) in per_index_referenced {
        if matches!(*kind, IndexKind::Each) {
            continue;
        }
        for id in primary_ids {
            if !referenced.contains(id) {
                failures.push(IntegrityFailure::MissingIndexEntry {
                    collection: collection.to_owned(),
                    index: index_name.clone(),
                    id: *id,
                });
            }
        }
    }
    debug_assert!(
        descriptor.indexes.iter().all(|d| {
            d.status != IndexStatus::Active
                || per_index_referenced.iter().any(|(n, _, _)| n == &d.name)
        }),
        "every Active index must contribute a referenced-ids set",
    );
}

/// Compute the fast subset of the integrity walk: validate the
/// file-header invariants and walk the catalog tree (and only the
/// catalog tree), without descending into any per-collection
/// primary or index.
///
/// Used by the obj crate's open-time fast check and by
/// the obj crate's `Db::integrity_check` for the catalog portion
/// of the full walk.
///
/// # Errors
///
/// - [`Error::Io`] on cache-miss read failure.
///
/// The returned `IntegrityReport` carries failures encountered
/// during the catalog walk; the caller decides whether to surface
/// them as `Err(Error::Corruption { ... })` (the open-time fast
/// path does) or to merge them into a broader report (the on-
/// demand full walk does).
pub fn quick_check<F: FileBackend>(pager: &mut Pager<F>) -> Result<IntegrityReport> {
    let start = std::time::Instant::now();
    let mut failures: Vec<IntegrityFailure> = Vec::new();
    let mut reachable: HashSet<PageId> = HashSet::new();
    let page_count = pager.page_count();
    let mut pages_checked: u64 = 1;
    if let Some(root) = PageId::new(pager.root_catalog()) {
        if root.get() >= page_count {
            failures.push(IntegrityFailure::DanglingCatalogPointer {
                collection: "<catalog>".to_owned(),
                index: None,
                page_id: root.get(),
            });
        } else {
            let ctx = TreeContext {
                label: "catalog".to_owned(),
                root,
            };
            pages_checked = pages_checked.saturating_add(walk_btree(
                pager,
                &ctx,
                &mut reachable,
                &mut failures,
            )?);
            list_catalog_for_pointer_check(pager, page_count, &mut failures)?;
        }
    }
    Ok(IntegrityReport {
        failures,
        pages_checked,
        elapsed: start.elapsed(),
    })
}

/// Read every catalog row and check the `primary_root` +
/// `root_page_id` pointers without walking the per-collection trees.
/// Used by [`quick_check`].
fn list_catalog_for_pointer_check<F: FileBackend>(
    pager: &mut Pager<F>,
    page_count: u64,
    failures: &mut Vec<IntegrityFailure>,
) -> Result<()> {
    let raw = pager.root_catalog();
    if PageId::new(raw).is_none() {
        return Ok(());
    }
    let catalog = match Catalog::<F>::open_or_init(pager) {
        Ok(c) => c,
        Err(Error::Corruption { .. }) => return Ok(()),
        Err(e) => return Err(e),
    };
    let rows = match catalog.list_collections(pager) {
        Ok(r) => r,
        Err(Error::Corruption { .. }) => return Ok(()),
        Err(e) => return Err(e),
    };
    for (name, descriptor) in rows {
        check_catalog_pointers(&name, &descriptor, page_count, failures);
    }
    Ok(())
}

/// Helper: ensure `path` is the canonical descent stack shape used
/// by the integrity walk's caller. Re-exported for the obj crate's
/// per-T extension so it does not have to reimplement the
/// `HeaplessVec<PageId, MAX_BTREE_DEPTH>` pattern.
#[must_use]
pub fn fresh_descent_stack() -> HeaplessVec<PageId, MAX_BTREE_DEPTH> {
    HeaplessVec::new()
}

/// Unit tests for the failure arms that cannot be reached through the
/// public walk entry points:
///
/// - the `MAX_RANGE_NODES` node-count budget in [`record_btree_visit`]
///   would need a tree of more than one million REAL pages to trip via
///   [`walk_btree`];
/// - [`verify_level_invariant`] and [`verify_sort_invariant`] are
///   belt-and-suspenders re-checks of invariants `decode_node` already
///   rejects (as `Error::Corruption`, surfaced by
///   [`read_and_validate_node`] as `ChecksumMismatch`), so their
///   failure pushes only fire on a `DecodedNode` constructed in
///   memory.
///
/// The reachable corruption arms (depth bound, descent-graph cycle,
/// damaged trailer, undecodable node, unrecoverable index id, dangling
/// catalog root, broken freelist chain) are integration-tested against
/// a real pager in `obj-core/tests/integrity_corruption.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::btree::node::{DecodedNode, InternalEntry, LeafEntry};

    fn test_ctx() -> TreeContext {
        TreeContext {
            label: "primary:widgets".to_owned(),
            root: PageId::new(1).expect("non-zero page id"),
        }
    }

    fn empty_node(kind: NodeKind, level: u8) -> DecodedNode {
        DecodedNode {
            kind,
            level,
            next_sibling: 0,
            children: Vec::new(),
            leaves: Vec::new(),
            internals: Vec::new(),
        }
    }

    fn budget() -> u64 {
        u64::try_from(MAX_RANGE_NODES).expect("MAX_RANGE_NODES fits in u64")
    }

    #[test]
    fn record_btree_visit_walks_a_fresh_page() {
        let mut visited: HashSet<PageId> = HashSet::new();
        let mut reachable: HashSet<PageId> = HashSet::new();
        let mut pages_walked: u64 = 0;
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(7).expect("non-zero page id");
        let step = record_btree_visit(
            pid,
            &test_ctx(),
            &mut WalkVisitState {
                visited: &mut visited,
                reachable: &mut reachable,
                pages_walked: &mut pages_walked,
                failures: &mut failures,
            },
        );
        assert!(matches!(step, VisitStep::Walked));
        assert!(failures.is_empty(), "no failure expected: {failures:?}");
        assert!(reachable.contains(&pid));
        assert_eq!(pages_walked, 1);
    }

    #[test]
    fn record_btree_visit_reports_cycle_on_revisit() {
        let mut visited: HashSet<PageId> = HashSet::new();
        let mut reachable: HashSet<PageId> = HashSet::new();
        let mut pages_walked: u64 = 1;
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(7).expect("non-zero page id");
        visited.insert(pid);
        let step = record_btree_visit(
            pid,
            &test_ctx(),
            &mut WalkVisitState {
                visited: &mut visited,
                reachable: &mut reachable,
                pages_walked: &mut pages_walked,
                failures: &mut failures,
            },
        );
        assert!(matches!(step, VisitStep::Cycle));
        assert_eq!(pages_walked, 1, "a revisited page is not re-counted");
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeSiblingChainBroken { tree, page_id }]
                    if tree == "primary:widgets" && *page_id == pid.get()
            ),
            "expected BTreeSiblingChainBroken; got {failures:?}",
        );
    }

    /// The `MAX_RANGE_NODES` page-walk budget: the visit that crosses
    /// the bound records `BTreeDepthExceeded { limit: MAX_RANGE_NODES }`
    /// and tells the walk driver to stop.
    #[test]
    fn record_btree_visit_reports_exceeded_node_budget() {
        let mut visited: HashSet<PageId> = HashSet::new();
        let mut reachable: HashSet<PageId> = HashSet::new();
        let mut pages_walked: u64 = budget();
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(42).expect("non-zero page id");
        let step = record_btree_visit(
            pid,
            &test_ctx(),
            &mut WalkVisitState {
                visited: &mut visited,
                reachable: &mut reachable,
                pages_walked: &mut pages_walked,
                failures: &mut failures,
            },
        );
        assert!(matches!(step, VisitStep::BudgetExceeded));
        assert_eq!(pages_walked, budget() + 1);
        assert!(reachable.contains(&pid), "page is still marked reachable");
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeDepthExceeded { tree, limit }]
                    if tree == "primary:widgets" && *limit == MAX_RANGE_NODES
            ),
            "expected BTreeDepthExceeded at the node budget; got {failures:?}",
        );
    }

    #[test]
    fn verify_level_invariant_flags_nonzero_leaf_level() {
        let node = empty_node(NodeKind::Leaf, 3);
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(9).expect("non-zero page id");
        verify_level_invariant(pid, &node, "index:users.by_email", &mut failures);
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeLevelInvariantViolated { tree, page_id }]
                    if tree == "index:users.by_email" && *page_id == pid.get()
            ),
            "expected BTreeLevelInvariantViolated; got {failures:?}",
        );
    }

    #[test]
    fn verify_level_invariant_flags_zero_internal_level() {
        let mut node = empty_node(NodeKind::Internal, 0);
        node.children.push(2);
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(5).expect("non-zero page id");
        verify_level_invariant(pid, &node, "catalog", &mut failures);
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeLevelInvariantViolated { tree, page_id }]
                    if tree == "catalog" && *page_id == pid.get()
            ),
            "expected BTreeLevelInvariantViolated; got {failures:?}",
        );
    }

    #[test]
    fn verify_level_invariant_accepts_consistent_levels() {
        let pid = PageId::new(5).expect("non-zero page id");
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        verify_level_invariant(pid, &empty_node(NodeKind::Leaf, 0), "catalog", &mut failures);
        let mut internal = empty_node(NodeKind::Internal, 1);
        internal.children.push(2);
        verify_level_invariant(pid, &internal, "catalog", &mut failures);
        assert!(failures.is_empty(), "no failure expected: {failures:?}");
    }

    fn leaf_with_keys(keys: &[&[u8]]) -> DecodedNode {
        let mut node = empty_node(NodeKind::Leaf, 0);
        for key in keys {
            node.leaves.push(LeafEntry {
                key: key.to_vec(),
                value: Vec::new(),
            });
        }
        node
    }

    fn internal_with_pivots(pivots: &[&[u8]]) -> DecodedNode {
        let mut node = empty_node(NodeKind::Internal, 1);
        node.children.push(2);
        for key in pivots {
            node.internals.push(InternalEntry { key: key.to_vec() });
            node.children.push(3);
        }
        node
    }

    #[test]
    fn verify_sort_invariant_flags_descending_leaf_keys() {
        let node = leaf_with_keys(&[b"b", b"a"]);
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(4).expect("non-zero page id");
        verify_sort_invariant(pid, &node, &mut failures);
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeSortViolation { page_id }] if *page_id == pid.get()
            ),
            "expected BTreeSortViolation; got {failures:?}",
        );
    }

    #[test]
    fn verify_sort_invariant_flags_duplicate_internal_pivots() {
        let node = internal_with_pivots(&[b"k", b"k"]);
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(6).expect("non-zero page id");
        verify_sort_invariant(pid, &node, &mut failures);
        assert!(
            matches!(
                failures.as_slice(),
                [IntegrityFailure::BTreeSortViolation { page_id }] if *page_id == pid.get()
            ),
            "expected BTreeSortViolation; got {failures:?}",
        );
    }

    #[test]
    fn verify_sort_invariant_accepts_sorted_keys() {
        let mut failures: Vec<IntegrityFailure> = Vec::new();
        let pid = PageId::new(4).expect("non-zero page id");
        verify_sort_invariant(pid, &leaf_with_keys(&[b"a", b"b"]), &mut failures);
        verify_sort_invariant(pid, &internal_with_pivots(&[b"d", b"m"]), &mut failures);
        assert!(failures.is_empty(), "no failure expected: {failures:?}");
    }
}
