//! Forward range scan iterator.
//!
//! The iterator captures the tree's root page-id at construction;
//! subsequent writes (which COW the touched pages) cannot disturb the
//! iterator's traversal because every page the iterator references
//! remains intact on disk for the lifetime of the iterator.
//!
//! # Navigation strategy
//!
//! The iterator advances from one leaf to the next by **descending
//! from the snapshot root** with the search key set to "the
//! smallest key strictly greater than the last emitted key". It
//! does NOT follow leaf `next_sibling` pointers.
//!
//! Why not sibling pointers? In a copy-on-write B+tree the page-id
//! of a leaf changes every time the leaf is rewritten. A left-
//! neighbour leaf that was NOT touched by the rewrite still carries
//! the OLD `next_sibling` page-id, which now points at a freed page
//! (or, post-recycling, at a completely different node). Updating
//! every potentially-affected sibling on every COW would cascade
//! the rewrite up arbitrarily; descending from the snapshot root
//! is O(log n) per leaf-step but always correct.
//!
//! `next_sibling` is still part of the on-disk node header and is set
//! correctly by every split, but readers do not use it.

#![forbid(unsafe_code)]

use core::ops::{Bound, RangeBounds};

use crate::btree::node::{decode_node, read_leaf_slot, BorrowedLeaf, DecodedNode, NodeKind};
use crate::btree::{BTree, MAX_BTREE_DEPTH, MAX_RANGE_NODES};
use crate::error::{Error, Result};
use crate::pager::page::PageId;
use crate::pager::{PageHandle, Pager};
use crate::platform::FileBackend;

use heapless::Vec as HeaplessVec;

/// A forward iterator over `(key, value)` pairs in a B+tree.
///
/// Implements `Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>`.
/// Returns items in ascending key order over the half-open range
/// `start..end` (or `..end`, `start..`, `..` depending on the
/// bounds passed to [`BTree::range`]).
///
/// # Snapshot guarantee
///
/// The iterator captures the tree's root page-id at construction.
/// Because every B+tree mutation copies-on-write, any page the
/// iterator's current traversal references continues to exist on
/// disk until the iterator drops, *even if a concurrent
/// `BTree::insert` or `BTree::delete` has produced a new tree
/// shape*. The snapshot guarantee covers concurrent in-process
/// mutations only.
pub struct RangeIter<'a, F: FileBackend> {
    pager: &'a mut Pager<F>,
    /// Snapshot root captured at construction. Every leaf descent
    /// starts here so the iterator's view is stable across COW
    /// writes that may have happened since.
    root: PageId,
    current_leaf: Option<DecodedNode>,
    slot_index: usize,
    start_bound: Bound<Vec<u8>>,
    end_bound: Bound<Vec<u8>>,
    /// The last key emitted by `next`. Used to find the next leaf
    /// via a root-descent with start = `Excluded(last_key)`.
    last_emitted_key: Option<Vec<u8>>,
    nodes_visited: usize,
    finished: bool,
}

impl<F: FileBackend> std::fmt::Debug for RangeIter<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeIter")
            .field("slot_index", &self.slot_index)
            .field("start_bound", &self.start_bound)
            .field("end_bound", &self.end_bound)
            .field("nodes_visited", &self.nodes_visited)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<F: FileBackend> BTree<F> {
    /// Construct a forward iterator over the half-open range
    /// `range`. Bounds are inclusive on `start` (per `RangeBounds`)
    /// and exclusive on `end`. An unbounded `start`/`end` is
    /// permitted (yielding `..end`, `start..`, or `..`).
    ///
    /// Bounds are taken as `RangeBounds<Vec<u8>>` so the four
    /// idiomatic Rust range syntaxes — `..`, `start..`, `..end`,
    /// `start..end`, `start..=end` — and the `(Bound<Vec<u8>>,
    /// Bound<Vec<u8>>)` tuple all compose naturally.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BTreeDepthExceeded`], [`Error::Corruption`],
    /// or [`Error::Io`] propagated from the descent to the first
    /// leaf. Per-step errors during iteration are surfaced via
    /// `Some(Err(...))` from `next`.
    pub fn range<'a, R>(&self, pager: &'a mut Pager<F>, range: R) -> Result<RangeIter<'a, F>>
    where
        R: RangeBounds<Vec<u8>>,
    {
        let start_bound = clone_bound_vec(range.start_bound());
        let end_bound = clone_bound_vec(range.end_bound());
        Self::build_range_iter(pager, self.root, start_bound, end_bound)
    }

    /// Iterate over every `(key, value)` pair in the tree. Sugar
    /// for `range(..)`.
    ///
    /// The Clippy `iter_not_returning_iterator` warning would
    /// normally fire here because the return type is wrapped in a
    /// `Result` (construction can fail). We accept the deviation
    /// because the alternative — making `iter` infallible and
    /// returning errors on the first `next()` — surprises the
    /// caller more.
    ///
    /// # Errors
    ///
    /// As for [`BTree::range`].
    // allow: `iter` returns `Result<RangeIter>` by design — construction can fail, so it cannot return a bare Iterator as the lint expects.
    #[allow(clippy::iter_not_returning_iterator)]
    pub fn iter<'a>(&self, pager: &'a mut Pager<F>) -> Result<RangeIter<'a, F>> {
        self.range(pager, ..)
    }

    fn build_range_iter(
        pager: &mut Pager<F>,
        root: PageId,
        start_bound: Bound<Vec<u8>>,
        end_bound: Bound<Vec<u8>>,
    ) -> Result<RangeIter<'_, F>> {
        let (leaf, slot_index, nodes_visited) =
            locate_first_in_range_leaf(pager, root, &start_bound)?;
        let finished = slot_index >= leaf.leaves.len();
        Ok(RangeIter {
            pager,
            root,
            current_leaf: Some(leaf),
            slot_index,
            start_bound,
            end_bound,
            last_emitted_key: None,
            nodes_visited,
            finished,
        })
    }

    /// Snapshot-consistent variant of [`BTree::range`] — every page
    /// read during descent and leaf-scan is resolved as-of `snapshot`
    /// via [`crate::pager::ReaderSnapshot::read_page`] rather than the
    /// live `Pager::read_page` (which consults the WAL overlay,
    /// including a concurrent writer's post-snapshot commits).
    ///
    /// This is the range/count analogue of [`BTree::get_via_snapshot`]:
    /// a read transaction documented as snapshot-isolated must
    /// enumerate range and count results consistently with its pinned
    /// LSN. Walking the live pager would let a concurrent writer's
    /// post-snapshot index entries leak into the read txn's
    /// range/count.
    ///
    /// `root` is the B+tree root captured at the reader's
    /// snapshot-time view of the catalog (the descriptor read via
    /// [`crate::Catalog::lookup_via_snapshot`]); the caller is
    /// responsible for handing in the snapshot-time root.
    ///
    /// # Errors
    ///
    /// As [`BTree::range`], plus snapshot read errors propagated from
    /// [`crate::pager::ReaderSnapshot::read_page`].
    pub fn range_via_snapshot<'a, R>(
        pager: &'a Pager<F>,
        snapshot: &'a crate::pager::ReaderSnapshot<F>,
        root: PageId,
        range: R,
    ) -> Result<SnapshotRangeIter<'a, F>>
    where
        R: RangeBounds<Vec<u8>>,
    {
        let start_bound = clone_bound_vec(range.start_bound());
        let end_bound = clone_bound_vec(range.end_bound());
        let (leaf, slot_index, nodes_visited) =
            snap_locate_first_in_range_leaf(pager, snapshot, root, &start_bound)?;
        let finished = slot_index >= leaf.len;
        Ok(SnapshotRangeIter {
            pager,
            snapshot,
            root,
            current_leaf: Some(leaf),
            slot_index,
            start_bound,
            end_bound,
            last_emitted_key: None,
            nodes_visited,
            finished,
        })
    }
}

/// Find the leaf + slot that holds the first key satisfying
/// `start_bound`. Returns the leaf, the in-leaf slot index (which may
/// equal `leaf.leaves.len()` when the snapshot has no in-range keys
/// at all — the caller treats that as a finished iterator), and the
/// number of leaf nodes visited during the search (counted against
/// `MAX_RANGE_NODES`).
///
/// If every key in the landing leaf sorts BEFORE `start_bound`,
/// `position_in_leaf` returns past-end. We then walk forward through
/// right-sibling leaves (via root-descent on the current leaf's last
/// key — leaf page-ids are unstable under COW, so we re-descend
/// rather than follow `next_sibling`) until we find a leaf with an
/// in-bound slot, or the snapshot has no more leaves to scan.
///
/// Bound the walk via the same `MAX_RANGE_NODES` budget the iterator
/// uses per-step; an unbounded walk here would be incorrect. In
/// practice the loop executes at most twice —
/// by the B+tree's pivot invariant, a leaf adjacent-right of the
/// landing leaf has its first key strictly greater than every key in
/// the landing leaf, so `position_in_leaf` on the very next leaf
/// returns 0 for any start bound whose value is ≤ the landing leaf's
/// largest key.
fn locate_first_in_range_leaf<F: FileBackend>(
    pager: &mut Pager<F>,
    root: PageId,
    start_bound: &Bound<Vec<u8>>,
) -> Result<(DecodedNode, usize, usize)> {
    let descend_key = match start_bound {
        Bound::Included(k) | Bound::Excluded(k) => k.as_slice(),
        Bound::Unbounded => &[][..],
    };
    let leaf_id = descend_to_start_leaf(pager, root, descend_key)?;
    let mut leaf = read_leaf(pager, leaf_id)?;
    let mut slot_index = position_in_leaf(&leaf, start_bound);
    let mut nodes_visited: usize = 1;
    while slot_index >= leaf.leaves.len() {
        let Some(last_key) = leaf.leaves.last().map(|e| e.key.clone()) else {
            break;
        };
        nodes_visited += 1;
        if nodes_visited > MAX_RANGE_NODES {
            return Err(Error::BTreeScanLimitExceeded {
                limit: MAX_RANGE_NODES,
            });
        }
        let Some(next_id) = descend_to_leaf_after(pager, root, last_key.as_slice())? else {
            break;
        };
        leaf = read_leaf(pager, next_id)?;
        slot_index = position_in_leaf(&leaf, start_bound);
    }
    Ok((leaf, slot_index, nodes_visited))
}

fn clone_bound_vec(b: Bound<&Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(s) => Bound::Included(s.clone()),
        Bound::Excluded(s) => Bound::Excluded(s.clone()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

fn descend_to_start_leaf<F: FileBackend>(
    pager: &mut Pager<F>,
    root: PageId,
    key: &[u8],
) -> Result<PageId> {
    let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        path.push(current).map_err(|_| Error::BTreeDepthExceeded {
            limit: MAX_BTREE_DEPTH,
        })?;
        let decoded = {
            let pr = pager.read_page(current)?;
            decode_node(pr.as_bytes())?
        };
        match decoded.kind {
            NodeKind::Leaf => return Ok(current),
            NodeKind::Internal => {
                let idx = pivot_index_for_start(&decoded, key);
                current =
                    PageId::new(decoded.children[idx]).ok_or(Error::BTreeInvariantViolated {
                        reason: "range descent: zero child page-id",
                    })?;
            }
        }
    }
}

fn pivot_index_for_start(node: &DecodedNode, key: &[u8]) -> usize {
    let mut idx = node.internals.len();
    for (i, p) in node.internals.iter().enumerate() {
        if p.key.as_slice() > key {
            idx = i;
            break;
        }
    }
    idx
}

fn read_leaf<F: FileBackend>(pager: &mut Pager<F>, id: PageId) -> Result<DecodedNode> {
    let pr = pager.read_page(id)?;
    let decoded = decode_node(pr.as_bytes())?;
    if !matches!(decoded.kind, NodeKind::Leaf) {
        return Err(Error::BTreeInvariantViolated {
            reason: "range: expected leaf",
        });
    }
    Ok(decoded)
}

/// First slot index in `leaf` whose key satisfies the start bound.
fn position_in_leaf(leaf: &DecodedNode, start: &Bound<Vec<u8>>) -> usize {
    match start {
        Bound::Unbounded => 0,
        Bound::Included(k) => leaf
            .leaves
            .iter()
            .position(|e| e.key.as_slice() >= k.as_slice())
            .unwrap_or(leaf.leaves.len()),
        Bound::Excluded(k) => leaf
            .leaves
            .iter()
            .position(|e| e.key.as_slice() > k.as_slice())
            .unwrap_or(leaf.leaves.len()),
    }
}

fn within_end(key: &[u8], end: &Bound<Vec<u8>>) -> bool {
    match end {
        Bound::Unbounded => true,
        Bound::Included(k) => key <= k.as_slice(),
        Bound::Excluded(k) => key < k.as_slice(),
    }
}

impl<F: FileBackend> Iterator for RangeIter<'_, F> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let leaf = self.current_leaf.as_ref()?;
            if self.slot_index < leaf.leaves.len() {
                let entry = &leaf.leaves[self.slot_index];
                if !within_end(entry.key.as_slice(), &self.end_bound) {
                    self.finished = true;
                    return None;
                }
                let item = (entry.key.clone(), entry.value.clone());
                self.slot_index += 1;
                self.last_emitted_key = Some(item.0.clone());
                return Some(Ok(item));
            }
            match self.advance_to_next_leaf() {
                Ok(true) => (),
                Ok(false) => {
                    self.finished = true;
                    return None;
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

impl<F: FileBackend> RangeIter<'_, F> {
    /// Advance `current_leaf` to the next leaf in the snapshot.
    /// Returns `Ok(true)` if a new leaf with at least one key
    /// strictly greater than the last emitted was loaded; `Ok(false)`
    /// if no more keys remain in the snapshot.
    fn advance_to_next_leaf(&mut self) -> Result<bool> {
        let Some(last) = self.last_emitted_key.clone() else {
            return Ok(false);
        };
        self.nodes_visited += 1;
        if self.nodes_visited > MAX_RANGE_NODES {
            return Err(Error::BTreeScanLimitExceeded {
                limit: MAX_RANGE_NODES,
            });
        }
        let Some(leaf_id) = descend_to_leaf_after(self.pager, self.root, last.as_slice())? else {
            return Ok(false);
        };
        let leaf = read_leaf(self.pager, leaf_id)?;
        let slot_index = leaf
            .leaves
            .iter()
            .position(|e| e.key.as_slice() > last.as_slice())
            .unwrap_or(leaf.leaves.len());
        if slot_index == leaf.leaves.len() {
            return Ok(false);
        }
        self.current_leaf = Some(leaf);
        self.slot_index = slot_index;
        Ok(true)
    }
}

/// Descend from `root` to the leaf whose smallest key is strictly
/// greater than `key`. Returns `Ok(None)` if no such leaf exists.
///
/// Algorithm: descend to the leaf that would *contain* `key` while
/// recording the (internal, `child_index`) frames of the path. If
/// the landing leaf has any key > `key`, return it. Otherwise walk
/// the recorded path back up looking for an internal whose next
/// child (`child_index` + 1) exists; descend the leftmost leaf of
/// that subtree.
fn descend_to_leaf_after<F: FileBackend>(
    pager: &mut Pager<F>,
    root: PageId,
    key: &[u8],
) -> Result<Option<PageId>> {
    let mut frames: HeaplessVec<DescendFrame, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        let decoded = {
            let pr = pager.read_page(current)?;
            decode_node(pr.as_bytes())?
        };
        match decoded.kind {
            NodeKind::Leaf => {
                if decoded.leaves.iter().any(|e| e.key.as_slice() > key) {
                    return Ok(Some(current));
                }
                break;
            }
            NodeKind::Internal => {
                let child_index = pivot_index_for_start(&decoded, key);
                let raw = decoded.children[child_index];
                frames
                    .push(DescendFrame {
                        node: decoded,
                        child_index,
                    })
                    .map_err(|_| Error::BTreeDepthExceeded {
                        limit: MAX_BTREE_DEPTH,
                    })?;
                current = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                    reason: "descend_to_leaf_after: zero child id",
                })?;
            }
        }
    }
    while let Some(frame) = frames.pop() {
        let next_child = frame.child_index + 1;
        if next_child < frame.node.children.len() {
            let raw = frame.node.children[next_child];
            let next_root = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                reason: "descend_to_leaf_after: zero next-child id",
            })?;
            return descend_leftmost_leaf(pager, next_root).map(Some);
        }
    }
    Ok(None)
}

struct DescendFrame {
    node: DecodedNode,
    child_index: usize,
}

/// Descend from `root` to the leftmost leaf in its subtree.
fn descend_leftmost_leaf<F: FileBackend>(pager: &mut Pager<F>, root: PageId) -> Result<PageId> {
    let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        path.push(current).map_err(|_| Error::BTreeDepthExceeded {
            limit: MAX_BTREE_DEPTH,
        })?;
        let decoded = {
            let pr = pager.read_page(current)?;
            decode_node(pr.as_bytes())?
        };
        match decoded.kind {
            NodeKind::Leaf => return Ok(current),
            NodeKind::Internal => {
                let raw = decoded.children[0];
                current = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                    reason: "descend_leftmost_leaf: zero child id",
                })?;
            }
        }
    }
}

/// A leaf loaded for the snapshot range scan, held as the borrowing
/// page handle plus the leaf metadata validated once at load.
///
/// The iterator keeps the [`PageHandle`] alive and re-reads a single
/// entry per `next()` via [`read_leaf_slot`] on `handle.as_bytes()` —
/// no per-leaf [`DecodedNode`] allocation, and no per-step whole-leaf
/// re-validation (the slots were validated once when this struct was
/// built through [`BorrowedLeaf::new`]).
struct LoadedLeaf {
    handle: PageHandle,
    len: usize,
}

impl LoadedLeaf {
    /// Borrow the `(key, value)` of the `i`-th entry from the held
    /// page bytes. The borrow is tied to `&self`, so it cannot outlive
    /// the page handle.
    fn entry(&self, i: usize) -> Result<(&[u8], &[u8])> {
        read_leaf_slot(self.handle.as_bytes(), i)
    }

    /// First slot index whose key is strictly `> target`, or `len` if
    /// none.
    fn upper_bound(&self, target: &[u8]) -> Result<usize> {
        for i in 0..self.len {
            let (k, _) = self.entry(i)?;
            if k > target {
                return Ok(i);
            }
        }
        Ok(self.len)
    }
}

/// Snapshot-consistent forward iterator over `(key, value)` pairs.
///
/// Constructed by [`BTree::range_via_snapshot`]. Yields items in
/// ascending key order, identical in semantics to [`RangeIter`], but
/// every traversal read is resolved as-of the pinned snapshot.
///
/// # Borrowed-decode read path
///
/// Unlike [`RangeIter`], this iterator does NOT materialize each leaf
/// as a [`DecodedNode`] (which allocates 2N inner `Vec<u8>`s per
/// leaf). It holds the leaf's [`PageHandle`] and reads each entry as
/// borrowed `&[u8]` slices into the page bytes via [`read_leaf_slot`],
/// allocating an owned `(Vec, Vec)` only for the single pair it is
/// about to yield. See [`crate::btree::node::BorrowedLeaf`].
pub struct SnapshotRangeIter<'a, F: FileBackend> {
    pager: &'a Pager<F>,
    snapshot: &'a crate::pager::ReaderSnapshot<F>,
    /// Snapshot root captured at construction; every leaf descent
    /// restarts here, matching [`RangeIter`].
    root: PageId,
    current_leaf: Option<LoadedLeaf>,
    slot_index: usize,
    start_bound: Bound<Vec<u8>>,
    end_bound: Bound<Vec<u8>>,
    last_emitted_key: Option<Vec<u8>>,
    nodes_visited: usize,
    finished: bool,
}

impl<F: FileBackend> std::fmt::Debug for SnapshotRangeIter<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotRangeIter")
            .field("slot_index", &self.slot_index)
            .field("start_bound", &self.start_bound)
            .field("end_bound", &self.end_bound)
            .field("nodes_visited", &self.nodes_visited)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<F: FileBackend> Iterator for SnapshotRangeIter<'_, F> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            match self.try_emit_current_slot() {
                SlotStep::Yield(item) => return Some(item),
                SlotStep::Done => {
                    self.finished = true;
                    return None;
                }
                SlotStep::LeafExhausted => {}
            }
            match self.advance_to_next_leaf() {
                Ok(true) => (),
                Ok(false) => {
                    self.finished = true;
                    return None;
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Outcome of probing the current slot in [`SnapshotRangeIter`].
enum SlotStep {
    /// Emit this item from `next()`.
    Yield(Result<(Vec<u8>, Vec<u8>)>),
    /// The range is finished (past end bound, or no current leaf).
    Done,
    /// The current leaf is exhausted; advance to the next leaf.
    LeafExhausted,
}

impl<F: FileBackend> SnapshotRangeIter<'_, F> {
    /// Probe the entry at `slot_index` of the current leaf, reading it
    /// as borrowed slices and cloning only the pair to be yielded.
    fn try_emit_current_slot(&mut self) -> SlotStep {
        let Some(leaf) = self.current_leaf.as_ref() else {
            return SlotStep::Done;
        };
        if self.slot_index >= leaf.len {
            return SlotStep::LeafExhausted;
        }
        let (key, value) = match leaf.entry(self.slot_index) {
            Ok(kv) => kv,
            Err(e) => {
                self.finished = true;
                return SlotStep::Yield(Err(e));
            }
        };
        if !within_end(key, &self.end_bound) {
            return SlotStep::Done;
        }
        let item = (key.to_vec(), value.to_vec());
        self.slot_index += 1;
        self.last_emitted_key = Some(item.0.clone());
        SlotStep::Yield(Ok(item))
    }

    /// Advance `current_leaf` to the next leaf in the snapshot.
    /// Snapshot mirror of [`RangeIter::advance_to_next_leaf`].
    fn advance_to_next_leaf(&mut self) -> Result<bool> {
        let Some(last) = self.last_emitted_key.clone() else {
            return Ok(false);
        };
        self.nodes_visited += 1;
        if self.nodes_visited > MAX_RANGE_NODES {
            return Err(Error::BTreeScanLimitExceeded {
                limit: MAX_RANGE_NODES,
            });
        }
        let Some(leaf_id) =
            snap_descend_to_leaf_after(self.pager, self.snapshot, self.root, last.as_slice())?
        else {
            return Ok(false);
        };
        let leaf = snap_read_leaf(self.pager, self.snapshot, leaf_id)?;
        let slot_index = leaf.upper_bound(last.as_slice())?;
        if slot_index == leaf.len {
            return Ok(false);
        }
        self.current_leaf = Some(leaf);
        self.slot_index = slot_index;
        Ok(true)
    }
}

/// Snapshot mirror of [`locate_first_in_range_leaf`], using the
/// borrowed-leaf read path: each leaf is held as a [`LoadedLeaf`]
/// (page handle + validated length) and probed via borrowed slices —
/// no per-leaf [`DecodedNode`] allocation.
fn snap_locate_first_in_range_leaf<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &crate::pager::ReaderSnapshot<F>,
    root: PageId,
    start_bound: &Bound<Vec<u8>>,
) -> Result<(LoadedLeaf, usize, usize)> {
    let descend_key = match start_bound {
        Bound::Included(k) | Bound::Excluded(k) => k.as_slice(),
        Bound::Unbounded => &[][..],
    };
    let leaf_id = snap_descend_to_start_leaf(pager, snapshot, root, descend_key)?;
    let mut leaf = snap_read_leaf(pager, snapshot, leaf_id)?;
    let mut slot_index = position_in_loaded_leaf(&leaf, start_bound)?;
    let mut nodes_visited: usize = 1;
    while slot_index >= leaf.len {
        let Some(last_key) = last_key_of(&leaf)? else {
            break;
        };
        nodes_visited += 1;
        if nodes_visited > MAX_RANGE_NODES {
            return Err(Error::BTreeScanLimitExceeded {
                limit: MAX_RANGE_NODES,
            });
        }
        let Some(next_id) = snap_descend_to_leaf_after(pager, snapshot, root, last_key.as_slice())?
        else {
            break;
        };
        leaf = snap_read_leaf(pager, snapshot, next_id)?;
        slot_index = position_in_loaded_leaf(&leaf, start_bound)?;
    }
    Ok((leaf, slot_index, nodes_visited))
}

/// First slot index in `leaf` whose key satisfies the start bound,
/// reading keys as borrowed slices. Mirror of [`position_in_leaf`]
/// for the borrowed [`LoadedLeaf`].
fn position_in_loaded_leaf(leaf: &LoadedLeaf, start: &Bound<Vec<u8>>) -> Result<usize> {
    match start {
        Bound::Unbounded => Ok(0),
        Bound::Included(k) => loaded_leaf_position(leaf, |key| key >= k.as_slice()),
        Bound::Excluded(k) => loaded_leaf_position(leaf, |key| key > k.as_slice()),
    }
}

/// First slot index whose borrowed key satisfies `pred`, or `len`.
fn loaded_leaf_position(leaf: &LoadedLeaf, pred: impl Fn(&[u8]) -> bool) -> Result<usize> {
    for i in 0..leaf.len {
        let (key, _) = leaf.entry(i)?;
        if pred(key) {
            return Ok(i);
        }
    }
    Ok(leaf.len)
}

/// The last (largest) key of `leaf` as an owned `Vec`, or `None` if
/// the leaf is empty. Owned because the caller re-descends with it
/// after the borrowing leaf may have been replaced.
fn last_key_of(leaf: &LoadedLeaf) -> Result<Option<Vec<u8>>> {
    if leaf.len == 0 {
        return Ok(None);
    }
    let (key, _) = leaf.entry(leaf.len - 1)?;
    Ok(Some(key.to_vec()))
}

/// Snapshot mirror of [`descend_to_start_leaf`].
fn snap_descend_to_start_leaf<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &crate::pager::ReaderSnapshot<F>,
    root: PageId,
    key: &[u8],
) -> Result<PageId> {
    let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        path.push(current).map_err(|_| Error::BTreeDepthExceeded {
            limit: MAX_BTREE_DEPTH,
        })?;
        let page = snapshot.read_page(pager, current)?;
        match crate::btree::node::peek_node_kind(page.as_bytes())? {
            NodeKind::Leaf => return Ok(current),
            NodeKind::Internal => {
                let decoded = decode_node(page.as_bytes())?;
                let idx = pivot_index_for_start(&decoded, key);
                current =
                    PageId::new(decoded.children[idx]).ok_or(Error::BTreeInvariantViolated {
                        reason: "range descent: zero child page-id",
                    })?;
            }
        }
    }
}

/// Snapshot mirror of [`read_leaf`], borrowed-decode variant. Reads
/// the page through the snapshot, validates it as a leaf via
/// [`BorrowedLeaf::new`] (same per-slot integrity checks as a full
/// decode), and returns a [`LoadedLeaf`] holding the page handle plus
/// the validated entry count — without allocating a [`DecodedNode`].
fn snap_read_leaf<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &crate::pager::ReaderSnapshot<F>,
    id: PageId,
) -> Result<LoadedLeaf> {
    let handle = snapshot.read_page(pager, id)?;
    let len = {
        let view = BorrowedLeaf::new(handle.as_bytes()).map_err(|e| match e {
            Error::BTreeInvariantViolated { .. } => Error::BTreeInvariantViolated {
                reason: "range: expected leaf",
            },
            other => other,
        })?;
        view.len()
    };
    Ok(LoadedLeaf { handle, len })
}

/// Snapshot mirror of [`descend_to_leaf_after`].
fn snap_descend_to_leaf_after<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &crate::pager::ReaderSnapshot<F>,
    root: PageId,
    key: &[u8],
) -> Result<Option<PageId>> {
    let mut frames: HeaplessVec<DescendFrame, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        let page = snapshot.read_page(pager, current)?;
        match crate::btree::node::peek_node_kind(page.as_bytes())? {
            NodeKind::Leaf => {
                let leaf = BorrowedLeaf::new(page.as_bytes())?;
                if leaf.upper_bound(key) < leaf.len() {
                    return Ok(Some(current));
                }
                break;
            }
            NodeKind::Internal => {
                let decoded = decode_node(page.as_bytes())?;
                let child_index = pivot_index_for_start(&decoded, key);
                let raw = decoded.children[child_index];
                frames
                    .push(DescendFrame {
                        node: decoded,
                        child_index,
                    })
                    .map_err(|_| Error::BTreeDepthExceeded {
                        limit: MAX_BTREE_DEPTH,
                    })?;
                current = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                    reason: "descend_to_leaf_after: zero child id",
                })?;
            }
        }
    }
    while let Some(frame) = frames.pop() {
        let next_child = frame.child_index + 1;
        if next_child < frame.node.children.len() {
            let raw = frame.node.children[next_child];
            let next_root = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                reason: "descend_to_leaf_after: zero next-child id",
            })?;
            return snap_descend_leftmost_leaf(pager, snapshot, next_root).map(Some);
        }
    }
    Ok(None)
}

/// Snapshot mirror of [`descend_leftmost_leaf`].
fn snap_descend_leftmost_leaf<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &crate::pager::ReaderSnapshot<F>,
    root: PageId,
) -> Result<PageId> {
    let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        path.push(current).map_err(|_| Error::BTreeDepthExceeded {
            limit: MAX_BTREE_DEPTH,
        })?;
        let page = snapshot.read_page(pager, current)?;
        let decoded = decode_node(page.as_bytes())?;
        match decoded.kind {
            NodeKind::Leaf => return Ok(current),
            NodeKind::Internal => {
                let raw = decoded.children[0];
                current = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                    reason: "descend_leftmost_leaf: zero child id",
                })?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::{Config, Pager};
    use crate::platform::FileHandle;

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;
    use std::ops::Bound;

    fn config() -> Config {
        Config::default()
    }

    fn collect_all(
        pager: &mut Pager<FileHandle>,
        tree: &BTree<FileHandle>,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let iter = tree.iter(pager).expect("iter");
        iter.map(|r| r.expect("step")).collect()
    }

    #[test]
    fn iter_empty_tree() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let v = collect_all(&mut pager, &tree);
        assert!(v.is_empty());
    }

    #[test]
    fn iter_single_leaf() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        for (k, v) in [("alpha", "A"), ("bravo", "B"), ("charlie", "C")] {
            tree.insert(&mut pager, k.as_bytes(), v.as_bytes())
                .expect("ins");
        }
        let got = collect_all(&mut pager, &tree);
        let want: Vec<(Vec<u8>, Vec<u8>)> = [("alpha", "A"), ("bravo", "B"), ("charlie", "C")]
            .iter()
            .map(|(k, v)| (k.as_bytes().to_vec(), v.as_bytes().to_vec()))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn range_across_leaf_split_boundary() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let value = vec![0xAAu8; 256];
        for i in 0..50u32 {
            let key = format!("key-{i:08}");
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("ins");
        }
        let root = tree.root();
        let root_decoded = {
            let pr = pager.read_page(root).expect("read");
            decode_node(pr.as_bytes()).expect("dec")
        };
        assert!(root_decoded.level >= 1, "expected root split");
        let iter = tree.iter(&mut pager).expect("iter");
        let mut prev: Option<Vec<u8>> = None;
        let mut count = 0;
        for item in iter {
            let (k, _v) = item.expect("step");
            if let Some(p) = &prev {
                assert!(p.as_slice() < k.as_slice(), "non-monotonic at {k:?}");
            }
            prev = Some(k);
            count += 1;
        }
        assert_eq!(count, 50);
    }

    #[test]
    fn range_bounded_subset() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        for i in 0..20u32 {
            let key = format!("k-{i:03}");
            tree.insert(&mut pager, key.as_bytes(), b"v").expect("ins");
        }
        let iter = tree
            .range(
                &mut pager,
                (
                    Bound::Included(b"k-005".to_vec()),
                    Bound::Excluded(b"k-010".to_vec()),
                ),
            )
            .expect("range");
        let got: Vec<String> = iter
            .map(|r| String::from_utf8(r.expect("step").0).expect("utf8"))
            .collect();
        assert_eq!(got, vec!["k-005", "k-006", "k-007", "k-008", "k-009"]);
    }

    #[test]
    fn range_oracle_1000_ops() {
        for seed in 0..3u64 {
            run_range_oracle(seed, 1_000);
        }
    }

    /// Regression for the bug surfaced by the 1M-op oracle:
    /// when `RangeIter::build_range_iter` descends to the leaf that
    /// should hold the start key but every entry in that leaf sorts
    /// strictly before the start bound, the constructor's
    /// `position_in_leaf` returns past-end. The original
    /// implementation then short-circuited the first `next()` to
    /// `None` via `last_emitted_key.is_none()`, wrongly reporting an
    /// empty range even though right-sibling leaves contained
    /// matching keys.
    ///
    /// This test forces a tree split, then queries a range whose
    /// start key sorts AFTER every key in the leftmost leaf but
    /// BEFORE keys in the next leaf. The fixed iterator must walk
    /// to the right sibling and return those keys.
    #[test]
    fn range_start_bound_past_landing_leaf_end() {
        let (mut pager, tree, value) = build_split_fixture();
        let (start_key, want) = construct_past_landing_leaf_query(&mut pager, &tree, &value);
        let iter = tree
            .range(
                &mut pager,
                (Bound::Included(start_key), Bound::<Vec<u8>>::Unbounded),
            )
            .expect("range");
        let got: Vec<(Vec<u8>, Vec<u8>)> = iter.map(|r| r.expect("step")).collect();
        assert_eq!(got, want, "iterator must advance past empty landing leaf");
    }

    /// Builds a B+tree with enough 256-byte-value entries to force
    /// at least one leaf split (root becomes internal).
    fn build_split_fixture() -> (Pager<FileHandle>, BTree<FileHandle>, Vec<u8>) {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let value = vec![0xCDu8; 256];
        for i in 0..50u32 {
            let key = format!("key-{i:08}");
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("ins");
        }
        (pager, tree, value)
    }

    type ExpectedRangeResult = Vec<(Vec<u8>, Vec<u8>)>;

    /// Constructs a start key strictly greater than every key in the
    /// leftmost leaf but strictly less than the pivot above it, and
    /// returns the oracle-computed expected results for the
    /// `Included(start_key)..` range.
    fn construct_past_landing_leaf_query(
        pager: &mut Pager<FileHandle>,
        tree: &BTree<FileHandle>,
        value: &[u8],
    ) -> (Vec<u8>, ExpectedRangeResult) {
        let root_decoded = {
            let pr = pager.read_page(tree.root()).expect("read");
            decode_node(pr.as_bytes()).expect("dec")
        };
        assert!(root_decoded.level >= 1, "fixture must split the tree");
        let leftmost_id = PageId::new(root_decoded.children[0]).expect("nz");
        let leftmost = read_leaf(pager, leftmost_id).expect("leaf");
        let last_in_leftmost = leftmost.leaves.last().expect("non-empty").key.clone();
        let pivot = root_decoded.internals[0].key.clone();
        let mut start_key = last_in_leftmost.clone();
        start_key.push(0);
        assert!(start_key.as_slice() > last_in_leftmost.as_slice());
        assert!(start_key.as_slice() < pivot.as_slice());
        let oracle: BTreeMap<Vec<u8>, Vec<u8>> = (0..50u32)
            .map(|i| (format!("key-{i:08}").into_bytes(), value.to_vec()))
            .collect();
        let want: Vec<(Vec<u8>, Vec<u8>)> = oracle
            .range::<Vec<u8>, _>((Bound::Included(&start_key), Bound::<&Vec<u8>>::Unbounded))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert!(!want.is_empty(), "oracle must return sibling-leaf entries");
        (start_key, want)
    }

    fn run_range_oracle(seed: u64, ops: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for op in 0..ops {
            match rng.random_range(0u32..6) {
                0..=2 => insert_step(&mut pager, &mut tree, &mut oracle, &mut rng, seed, op),
                3 => delete_step(&mut pager, &mut tree, &mut oracle, &mut rng, seed, op),
                _ => range_check(&mut pager, &tree, &oracle, &mut rng, seed, op),
            }
        }
        let got = collect_all(&mut pager, &tree);
        let want: Vec<(Vec<u8>, Vec<u8>)> =
            oracle.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(got, want, "seed {seed}: final iter mismatch");
    }

    fn insert_step(
        pager: &mut Pager<FileHandle>,
        tree: &mut BTree<FileHandle>,
        oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
        rng: &mut ChaCha8Rng,
        seed: u64,
        op: usize,
    ) {
        let key = random_key(rng);
        let value = random_value(rng);
        if let std::collections::btree_map::Entry::Vacant(slot) = oracle.entry(key.clone()) {
            tree.insert(pager, &key, &value)
                .unwrap_or_else(|e| panic!("seed {seed} op {op} ins: {e:?}"));
            slot.insert(value);
        }
    }

    fn delete_step(
        pager: &mut Pager<FileHandle>,
        tree: &mut BTree<FileHandle>,
        oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
        rng: &mut ChaCha8Rng,
        seed: u64,
        op: usize,
    ) {
        if oracle.is_empty() {
            return;
        }
        let pick = rng.random_range(0..oracle.len());
        let key = oracle.keys().nth(pick).cloned().unwrap_or_default();
        oracle.remove(&key);
        tree.delete(pager, &key)
            .unwrap_or_else(|e| panic!("seed {seed} op {op} del: {e:?}"));
    }

    fn range_check(
        pager: &mut Pager<FileHandle>,
        tree: &BTree<FileHandle>,
        oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
        rng: &mut ChaCha8Rng,
        seed: u64,
        op: usize,
    ) {
        let (start, end) = random_bounds(rng);
        if !bounds_well_ordered(&start, &end) {
            return;
        }
        let iter = tree
            .range(pager, (start.clone(), end.clone()))
            .expect("range");
        let got: Vec<(Vec<u8>, Vec<u8>)> = iter.map(|r| r.expect("step")).collect();
        let want: Vec<(Vec<u8>, Vec<u8>)> = oracle
            .range::<Vec<u8>, _>((start.as_ref(), end.as_ref()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(got, want, "seed {seed} op {op}: range mismatch");
    }

    fn bounds_well_ordered(start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> bool {
        match (start, end) {
            (Bound::Unbounded, _) | (_, Bound::Unbounded) => true,
            (Bound::Excluded(s), Bound::Excluded(e)) => s.as_slice() < e.as_slice(),
            (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e)) => {
                s.as_slice() <= e.as_slice()
            }
        }
    }

    fn random_bounds(rng: &mut ChaCha8Rng) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
        let s = match rng.random_range(0u32..3) {
            0 => Bound::Unbounded,
            1 => Bound::Included(random_key(rng)),
            _ => Bound::Excluded(random_key(rng)),
        };
        let e = match rng.random_range(0u32..3) {
            0 => Bound::Unbounded,
            1 => Bound::Included(random_key(rng)),
            _ => Bound::Excluded(random_key(rng)),
        };
        (s, e)
    }

    fn random_key(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(1..6);
        (0..len).map(|_| rng.random_range(b'a'..=b'c')).collect()
    }

    fn random_value(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(0..16);
        (0..len).map(|_| rng.random()).collect()
    }
}
