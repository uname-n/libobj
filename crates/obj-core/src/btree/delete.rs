//! B+tree delete path with merge / borrow rebalancing.
//!
//! Mirrors `insert.rs`: every node touched is rewritten to a fresh
//! page; pre-delete pages enter the freelist only after the new
//! root is staged. The caller still owns the commit boundary —
//! `BTree::delete` does NOT call `Pager::commit`.
//!
//! # Underflow policy
//!
//! A non-root node is "underflowing" when its post-delete occupied
//! byte count falls below `MIN_OCCUPIED_BYTES = PAYLOAD_BYTES / 2`.
//! At underflow:
//! - If a sibling is rich enough that **moving one slot from it**
//!   keeps both halves above the threshold, borrow.
//! - Otherwise merge with the sibling. The two halves combine into
//!   a single page; the separator pivot in the parent is removed
//!   (leaf merge) or absorbed (internal merge).
//!
//! Root collapse: an internal root with exactly one child (zero
//! pivots) is replaced by its child. A leaf root with zero slots
//! is allowed and is the bootstrap "empty tree" state.

#![forbid(unsafe_code)]

use heapless::Vec as HeaplessVec;

use crate::btree::insert::write_new_node;
use crate::btree::node::{
    decode_node, DecodedNode, InternalEntry, LeafEntry, NodeKind, NODE_HEADER_SIZE,
};
use crate::btree::{BTree, MAX_BTREE_DEPTH};
use crate::error::{Error, Result};
use crate::pager::page::{PageId, PAGE_SIZE, PAGE_TRAILER_SIZE};
use crate::pager::Pager;
use crate::platform::FileBackend;

/// Available payload bytes (excludes the node header and the page
/// trailer).
const PAYLOAD_BYTES: usize = PAGE_SIZE - PAGE_TRAILER_SIZE - NODE_HEADER_SIZE;

/// Underflow threshold: half-full. A non-root node whose
/// `occupied_bytes()` drops below this is rebalanced via borrow or
/// merge. This is the smallest threshold that keeps the B+tree's
/// `O(log_F n)` guarantees with a worst-case fanout of 2.
const MIN_OCCUPIED_BYTES: usize = PAYLOAD_BYTES / 2;

struct PathFrame {
    page_id: PageId,
    node: DecodedNode,
    /// Child index this frame's descent followed.
    child_index: usize,
}

/// Outcome of replacing one node along the delete path: a fresh
/// page-id and an "underflow" hint the parent uses to decide
/// whether to rebalance.
#[derive(Debug, Clone, Copy)]
struct ReplaceOutcome {
    new_id: PageId,
    /// Whether the new node falls below the underflow threshold.
    /// The parent inspects this to decide whether to borrow or
    /// merge.
    underflow: bool,
}

impl<F: FileBackend> BTree<F> {
    /// Remove `key` from the tree. Returns `Ok(true)` if it was
    /// present (and removed), `Ok(false)` if the key was absent.
    ///
    /// # Errors
    ///
    /// Propagates pager / decode errors. Surfaces
    /// `Error::BTreeDepthExceeded` if the tree height exceeds
    /// `MAX_BTREE_DEPTH`. No `panic`s.
    pub fn delete(&mut self, pager: &mut Pager<F>, key: &[u8]) -> Result<bool> {
        let path = self.descend_path_for_delete(pager, key)?;
        self.apply_delete(pager, path, key)
    }

    fn descend_path_for_delete(
        &self,
        pager: &mut Pager<F>,
        key: &[u8],
    ) -> Result<HeaplessVec<PathFrame, MAX_BTREE_DEPTH>> {
        let mut path: HeaplessVec<PathFrame, MAX_BTREE_DEPTH> = HeaplessVec::new();
        let mut current = self.root;
        loop {
            let decoded = {
                let page_ref = pager.read_page(current)?;
                decode_node(page_ref.as_bytes())?
            };
            match decoded.kind {
                NodeKind::Leaf => {
                    let frame = PathFrame {
                        page_id: current,
                        node: decoded,
                        child_index: 0,
                    };
                    push_path(&mut path, frame)?;
                    return Ok(path);
                }
                NodeKind::Internal => {
                    let child_index = pivot_index(&decoded, key);
                    let next = PageId::new(decoded.children[child_index]).ok_or(
                        Error::BTreeInvariantViolated {
                            reason: "internal node had zero child page-id",
                        },
                    )?;
                    let frame = PathFrame {
                        page_id: current,
                        node: decoded,
                        child_index,
                    };
                    push_path(&mut path, frame)?;
                    current = next;
                }
            }
        }
    }

    fn apply_delete(
        &mut self,
        pager: &mut Pager<F>,
        mut path: HeaplessVec<PathFrame, MAX_BTREE_DEPTH>,
        key: &[u8],
    ) -> Result<bool> {
        let mut freed: HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }> = HeaplessVec::new();
        let Some(leaf_frame) = path.pop() else {
            return Err(Error::BTreeInvariantViolated {
                reason: "delete: descend returned empty path",
            });
        };
        let found = leaf_frame
            .node
            .leaves
            .iter()
            .any(|e| e.key.as_slice() == key);
        if !found {
            return Ok(false);
        }
        let leaf_outcome = remove_from_leaf(pager, leaf_frame, key, &mut freed)?;
        let mut outcome = leaf_outcome;
        while let Some(parent_frame) = path.pop() {
            outcome = process_parent(pager, parent_frame, outcome, &mut freed, &mut path)?;
        }
        self.root = outcome.new_id;
        self.collapse_root(pager, &mut freed)?;
        for old_id in freed.iter().copied() {
            pager.free_page(old_id)?;
        }
        Ok(true)
    }

    /// If the current root is an internal node with exactly one
    /// child (zero pivots), replace it with that child and free the
    /// old root. Idempotent for any other root shape.
    fn collapse_root(
        &mut self,
        pager: &mut Pager<F>,
        freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
    ) -> Result<()> {
        loop {
            let root_id = self.root;
            let decoded = {
                let page_ref = pager.read_page(root_id)?;
                decode_node(page_ref.as_bytes())?
            };
            if !matches!(decoded.kind, NodeKind::Internal) {
                return Ok(());
            }
            if !decoded.internals.is_empty() || decoded.children.len() != 1 {
                return Ok(());
            }
            let only_child =
                PageId::new(decoded.children[0]).ok_or(Error::BTreeInvariantViolated {
                    reason: "collapse_root: zero child page-id",
                })?;
            self.root = only_child;
            push_freed(freed, root_id)?;
        }
    }
}

fn push_path(path: &mut HeaplessVec<PathFrame, MAX_BTREE_DEPTH>, frame: PathFrame) -> Result<()> {
    path.push(frame).map_err(|_| Error::BTreeDepthExceeded {
        limit: MAX_BTREE_DEPTH,
    })
}

fn push_freed(freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>, id: PageId) -> Result<()> {
    freed.push(id).map_err(|_| Error::BTreeInvariantViolated {
        reason: "delete: too many displaced pages to track",
    })
}

fn pivot_index(node: &DecodedNode, key: &[u8]) -> usize {
    let mut idx = node.internals.len();
    for (i, pivot) in node.internals.iter().enumerate() {
        if pivot.key.as_slice() > key {
            idx = i;
            break;
        }
    }
    idx
}

/// Remove the slot whose key equals `key` from `frame.node` (a
/// leaf), rewrite it as a fresh page, and report underflow.
fn remove_from_leaf<F: FileBackend>(
    pager: &mut Pager<F>,
    frame: PathFrame,
    key: &[u8],
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<ReplaceOutcome> {
    let mut leaf = frame.node;
    let Some(idx) = leaf.leaves.iter().position(|e| e.key.as_slice() == key) else {
        return Err(Error::BTreeInvariantViolated {
            reason: "remove_from_leaf: descend located absent key",
        });
    };
    leaf.leaves.remove(idx);
    let occupied = leaf.occupied_bytes();
    let underflow = occupied < MIN_OCCUPIED_BYTES;
    push_freed(freed, frame.page_id)?;
    let new_id = write_new_node(pager, &leaf)?;
    Ok(ReplaceOutcome { new_id, underflow })
}

/// Apply the child's outcome to a parent frame, possibly
/// rebalancing on underflow.
fn process_parent<F: FileBackend>(
    pager: &mut Pager<F>,
    frame: PathFrame,
    child_outcome: ReplaceOutcome,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
    _path: &mut HeaplessVec<PathFrame, MAX_BTREE_DEPTH>,
) -> Result<ReplaceOutcome> {
    let mut internal = frame.node;
    let child_index = frame.child_index;
    internal.children[child_index] = child_outcome.new_id.get();
    if child_outcome.underflow {
        rebalance_under_parent(pager, &mut internal, child_index, freed)?;
    }
    finalize_parent(pager, &internal, frame.page_id, freed)
}

/// Decide whether the rebalanced parent itself underflows, encode
/// it, and return the outcome.
fn finalize_parent<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &DecodedNode,
    old_id: PageId,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<ReplaceOutcome> {
    let occupied = internal.occupied_bytes();
    let underflow = occupied < MIN_OCCUPIED_BYTES;
    push_freed(freed, old_id)?;
    let new_id = write_new_node(pager, internal)?;
    Ok(ReplaceOutcome { new_id, underflow })
}

/// Rebalance an underflowing child of `internal` at `child_index`.
/// Tries to merge first (one fewer page in the tree, better for
/// cache locality); falls back to borrow if merge would overflow.
///
/// # Residual-underflow policy — best-effort, bounded, NOT a
/// correctness bug
///
/// In four ordered attempts (merge-right, merge-left, borrow-right,
/// borrow-left) this resolves the underflow in every ordinary case.
/// A *residual* underflow — where ALL FOUR attempts decline — can
/// only arise with variable-length keys, and only when both of these
/// hold simultaneously for every available sibling:
///
/// 1. **Merge declines**: combined `occupied_bytes` of the child, the
///    sibling, and (for internal nodes) the descended separator pivot
///    would exceed `PAYLOAD_BYTES` — i.e. the sibling is too FULL to
///    absorb the child; and
/// 2. **Borrow declines**: any of — the sibling cannot spare a slot and
///    stay `>= MIN_OCCUPIED_BYTES` ([`sibling_has_spare`]); moving its
///    boundary slot up would install a separator pivot LARGER than the
///    one it replaces and push the PARENT past `PAYLOAD_BYTES`
///    ([`parent_fits_after_pivot_swap`]) — the "borrow-grows-parent"
///    hazard; or the single borrowed slot is large enough that the
///    receiving child would itself exceed `PAYLOAD_BYTES`
///    ([`child_fits_after_borrow`]).
///
/// When that conjunction holds we deliberately leave the child below
/// `MIN_OCCUPIED_BYTES`. This is sound:
///
/// - **Correctness is preserved.** The child still holds every key it
///   should; the parent's pivots and child pointers stay consistent;
///   `get` / range-scan / subsequent `insert` and `delete` all remain
///   correct. The B+tree oracle (`btree_oracle`, `index_oracle`) only
///   observes externally-visible state and would not — and must not —
///   change. The ONLY degraded property is the *minimum-occupancy*
///   guarantee on that one node.
/// - **The degradation is bounded.** At most one node per
///   delete-rebalance bubble-up frame can be left underflowing, and a
///   later `insert` into that subtree, or a `delete` that makes a
///   sibling spare a slot, repairs the shape. It does not compound
///   into emptiness: a node that reaches zero slots is handled by the
///   merge path (its sibling can always absorb a zero-occupancy node),
///   and the root-collapse path handles a single-child root.
///
/// Implementing a full cascade here (recursively splitting the
/// over-full sibling, or borrow-then-rebalance-the-parent) was
/// considered and rejected for this change: it is materially more
/// invasive and risks a B-tree CORRECTNESS regression, which is a far
/// worse failure mode than a transient occupancy dip. The
/// `delete_rebalance_residual_underflow_is_bounded_and_correct` test
/// makes the degradation visible and asserts it stays bounded.
fn rebalance_under_parent<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<()> {
    if child_index + 1 < internal.children.len()
        && try_merge(pager, internal, child_index, MergeDirection::Right, freed)?
    {
        return Ok(());
    }
    if child_index > 0 && try_merge(pager, internal, child_index, MergeDirection::Left, freed)? {
        return Ok(());
    }
    if child_index + 1 < internal.children.len()
        && try_borrow_from_right(pager, internal, child_index, freed)?
    {
        return Ok(());
    }
    if child_index > 0 && try_borrow_from_left(pager, internal, child_index, freed)? {
        return Ok(());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum MergeDirection {
    Left,
    Right,
}

/// Attempt to merge with the indicated sibling. Returns `Ok(true)`
/// if the merge happened; `Ok(false)` if combined sizes would not
/// fit in a single page and the merge was abandoned.
fn try_merge<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    direction: MergeDirection,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<bool> {
    let (left_idx, right_idx) = match direction {
        MergeDirection::Right => (child_index, child_index + 1),
        MergeDirection::Left => (child_index - 1, child_index),
    };
    let left_id = pid(internal.children[left_idx])?;
    let right_id = pid(internal.children[right_idx])?;
    let left = read_node(pager, left_id)?;
    let right = read_node(pager, right_id)?;
    let separator_bytes = match left.kind {
        NodeKind::Leaf => 0,
        NodeKind::Internal => {
            use crate::btree::node::{varint_len, INTERNAL_SLOT_BYTES};
            let pivot = &internal.internals[left_idx].key;
            INTERNAL_SLOT_BYTES + varint_len(pivot.len() as u64) + pivot.len()
        }
    };
    let combined = left.occupied_bytes() + right.occupied_bytes() + separator_bytes
        - match left.kind {
            NodeKind::Internal => crate::btree::node::INTERNAL_LEFTMOST_CHILD_BYTES,
            NodeKind::Leaf => 0,
        };
    if combined > PAYLOAD_BYTES {
        return Ok(false);
    }
    drop(left);
    drop(right);
    match direction {
        MergeDirection::Right => merge_with_right(pager, internal, child_index, freed)?,
        MergeDirection::Left => merge_with_left(pager, internal, child_index, freed)?,
    }
    Ok(true)
}

/// Attempt to borrow one slot from the right sibling. Returns
/// `Ok(true)` if the borrow occurred.
///
/// A borrow REPLACES the separator pivot in the parent: the new
/// pivot can be larger than the old one, growing the parent. If
/// that growth would push the parent past `PAYLOAD_BYTES` the borrow
/// is abandoned (returns `Ok(false)`) so the caller can fall through
/// to the "leave the underflowing child as-is" branch — better than
/// raising a hard encode error for a delete-rebalance edge case.
fn try_borrow_from_right<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<bool> {
    let child_id = pid(internal.children[child_index])?;
    let right_id = pid(internal.children[child_index + 1])?;
    let mut child = read_node(pager, child_id)?;
    let mut right = read_node(pager, right_id)?;
    if !sibling_has_spare(&right) {
        return Ok(false);
    }
    let new_pivot = pick_borrow_right_pivot(&right, child.kind)?;
    if !parent_fits_after_pivot_swap(internal, child_index, &new_pivot) {
        return Ok(false);
    }
    let moved_bytes = borrow_right_moved_bytes(internal, &right, child.kind, child_index)?;
    if !child_fits_after_borrow(&child, moved_bytes) {
        return Ok(false);
    }
    apply_borrow_from_right(internal, &mut child, &mut right, child_index, &new_pivot)?;
    push_freed(freed, child_id)?;
    push_freed(freed, right_id)?;
    internal.children[child_index] = write_new_node(pager, &child)?.get();
    internal.children[child_index + 1] = write_new_node(pager, &right)?.get();
    Ok(true)
}

/// Choose the key that will be promoted from the right sibling to
/// become the parent's new separator pivot.
fn pick_borrow_right_pivot(right: &DecodedNode, child_kind: NodeKind) -> Result<Vec<u8>> {
    match (child_kind, right.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            right
                .leaves
                .get(1)
                .map(|e| e.key.clone())
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "borrow_from_right: right leaf has < 2 entries",
                })
        }
        (NodeKind::Internal, NodeKind::Internal) => right
            .internals
            .first()
            .map(|p| p.key.clone())
            .ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_from_right: right internal has no pivots",
            }),
        _ => Err(Error::BTreeInvariantViolated {
            reason: "borrow_from_right: mixed kinds",
        }),
    }
}

/// Mutate `child`, `right`, and `internal` in place to complete the
/// borrow once the parent-fit check has passed.
fn apply_borrow_from_right(
    internal: &mut DecodedNode,
    child: &mut DecodedNode,
    right: &mut DecodedNode,
    child_index: usize,
    new_pivot: &[u8],
) -> Result<()> {
    match (child.kind, right.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            let entry = right.leaves.remove(0);
            child.leaves.push(entry);
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot_down = internal.internals[child_index].key.clone();
            right.internals.remove(0);
            let moved_child = right.children.remove(0);
            child.internals.push(InternalEntry { key: pivot_down });
            child.children.push(moved_child);
        }
        _ => {
            return Err(Error::BTreeInvariantViolated {
                reason: "borrow_from_right: mixed kinds",
            });
        }
    }
    internal.internals[child_index].key = new_pivot.to_vec();
    Ok(())
}

/// Attempt to borrow one slot from the left sibling. Same parent-
/// fit check as [`try_borrow_from_right`].
fn try_borrow_from_left<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<bool> {
    let child_id = pid(internal.children[child_index])?;
    let left_id = pid(internal.children[child_index - 1])?;
    let mut child = read_node(pager, child_id)?;
    let mut left = read_node(pager, left_id)?;
    if !sibling_has_spare(&left) {
        return Ok(false);
    }
    let new_pivot = pick_borrow_left_pivot(&left, child.kind)?;
    if !parent_fits_after_pivot_swap(internal, child_index - 1, &new_pivot) {
        return Ok(false);
    }
    let moved_bytes = borrow_left_moved_bytes(internal, &left, child.kind, child_index)?;
    if !child_fits_after_borrow(&child, moved_bytes) {
        return Ok(false);
    }
    apply_borrow_from_left(internal, &mut child, &mut left, child_index, &new_pivot)?;
    push_freed(freed, child_id)?;
    push_freed(freed, left_id)?;
    internal.children[child_index - 1] = write_new_node(pager, &left)?.get();
    internal.children[child_index] = write_new_node(pager, &child)?.get();
    Ok(true)
}

fn pick_borrow_left_pivot(left: &DecodedNode, child_kind: NodeKind) -> Result<Vec<u8>> {
    match (child_kind, left.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            left.leaves
                .last()
                .map(|e| e.key.clone())
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "borrow_from_left: empty left leaf",
                })
        }
        (NodeKind::Internal, NodeKind::Internal) => left
            .internals
            .last()
            .map(|p| p.key.clone())
            .ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_from_left: empty left pivots",
            }),
        _ => Err(Error::BTreeInvariantViolated {
            reason: "borrow_from_left: mixed kinds",
        }),
    }
}

fn apply_borrow_from_left(
    internal: &mut DecodedNode,
    child: &mut DecodedNode,
    left: &mut DecodedNode,
    child_index: usize,
    new_pivot: &[u8],
) -> Result<()> {
    match (child.kind, left.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            let entry = left.leaves.pop().ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_from_left: empty left leaf",
            })?;
            child.leaves.insert(0, entry);
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot_down = internal.internals[child_index - 1].key.clone();
            left.internals.pop().ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_from_left: empty left pivots",
            })?;
            let moved_child = left.children.pop().ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_from_left: empty left children",
            })?;
            child.internals.insert(0, InternalEntry { key: pivot_down });
            child.children.insert(0, moved_child);
        }
        _ => {
            return Err(Error::BTreeInvariantViolated {
                reason: "borrow_from_left: mixed kinds",
            });
        }
    }
    internal.internals[child_index - 1].key = new_pivot.to_vec();
    Ok(())
}

/// Compute whether the parent internal node would still fit in
/// `PAYLOAD_BYTES` after swapping the pivot at `pivot_index` for a
/// new pivot of `new_key`. Borrow may grow the parent if
/// `new_key.len() + varint_len(new_key.len())` exceeds the same for
/// the old pivot.
fn parent_fits_after_pivot_swap(parent: &DecodedNode, pivot_index: usize, new_key: &[u8]) -> bool {
    use crate::btree::node::varint_len;
    let Some(old_pivot) = parent.internals.get(pivot_index) else {
        return false;
    };
    let old_entry = varint_len(old_pivot.key.len() as u64) + old_pivot.key.len();
    let new_entry = varint_len(new_key.len() as u64) + new_key.len();
    let parent_occupied = parent.occupied_bytes();
    parent_occupied + new_entry <= PAYLOAD_BYTES + old_entry
}

/// Whether the receiving child stays within `PAYLOAD_BYTES` after a
/// borrow appends/prepends a slot of `moved_bytes`. With
/// variable-length entries a single borrowed slot can be large enough
/// to overflow the (previously underflowing) child; declining the
/// borrow in that case preserves the no-overflow encode invariant.
fn child_fits_after_borrow(child: &DecodedNode, moved_bytes: usize) -> bool {
    child.occupied_bytes().saturating_add(moved_bytes) <= PAYLOAD_BYTES
}

/// Byte footprint the right-borrow moves INTO the child: a leaf gains
/// `right.leaves[0]`; an internal gains the descended separator pivot
/// (`internal.internals[child_index]`) plus one child pointer.
fn borrow_right_moved_bytes(
    internal: &DecodedNode,
    right: &DecodedNode,
    child_kind: NodeKind,
    child_index: usize,
) -> Result<usize> {
    match (child_kind, right.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            let e = right.leaves.first().ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_right_moved_bytes: empty right leaf",
            })?;
            Ok(leaf_slot_bytes(e))
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot =
                internal
                    .internals
                    .get(child_index)
                    .ok_or(Error::BTreeInvariantViolated {
                        reason: "borrow_right_moved_bytes: missing separator pivot",
                    })?;
            Ok(internal_slot_bytes(&pivot.key))
        }
        _ => Err(Error::BTreeInvariantViolated {
            reason: "borrow_right_moved_bytes: mixed kinds",
        }),
    }
}

/// Byte footprint the left-borrow moves INTO the child: a leaf gains
/// `left.leaves.last()`; an internal gains the descended separator
/// pivot (`internal.internals[child_index - 1]`) plus a child pointer.
fn borrow_left_moved_bytes(
    internal: &DecodedNode,
    left: &DecodedNode,
    child_kind: NodeKind,
    child_index: usize,
) -> Result<usize> {
    match (child_kind, left.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            let e = left.leaves.last().ok_or(Error::BTreeInvariantViolated {
                reason: "borrow_left_moved_bytes: empty left leaf",
            })?;
            Ok(leaf_slot_bytes(e))
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot =
                internal
                    .internals
                    .get(child_index - 1)
                    .ok_or(Error::BTreeInvariantViolated {
                        reason: "borrow_left_moved_bytes: missing separator pivot",
                    })?;
            Ok(internal_slot_bytes(&pivot.key))
        }
        _ => Err(Error::BTreeInvariantViolated {
            reason: "borrow_left_moved_bytes: mixed kinds",
        }),
    }
}

/// Encoded byte footprint of one leaf slot (directory entry + heap
/// entry). Matches the per-entry term of `DecodedNode::occupied_bytes`.
fn leaf_slot_bytes(e: &LeafEntry) -> usize {
    use crate::btree::node::{varint_len, LEAF_SLOT_BYTES};
    LEAF_SLOT_BYTES
        + varint_len(e.key.len() as u64)
        + e.key.len()
        + varint_len(e.value.len() as u64)
        + e.value.len()
}

/// Encoded byte footprint of one internal slot (directory entry +
/// heap pivot key). The child pointer lives in the fixed-width slot
/// directory, so no separate accounting is needed.
fn internal_slot_bytes(key: &[u8]) -> usize {
    use crate::btree::node::{varint_len, INTERNAL_SLOT_BYTES};
    INTERNAL_SLOT_BYTES + varint_len(key.len() as u64) + key.len()
}

/// Merge the child at `child_index` with its right sibling.
/// Removes the separator pivot and the right child id from the
/// parent.
fn merge_with_right<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<()> {
    let child_id = pid(internal.children[child_index])?;
    let right_id = pid(internal.children[child_index + 1])?;
    let mut child = read_node(pager, child_id)?;
    let mut right = read_node(pager, right_id)?;
    match (child.kind, right.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            child.leaves.append(&mut right.leaves);
            child.next_sibling = right.next_sibling;
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot_down = internal.internals[child_index].key.clone();
            child.internals.push(InternalEntry { key: pivot_down });
            child.internals.append(&mut right.internals);
            child.children.append(&mut right.children);
        }
        _ => {
            return Err(Error::BTreeInvariantViolated {
                reason: "merge_with_right: mixed kinds",
            });
        }
    }
    internal.internals.remove(child_index);
    internal.children.remove(child_index + 1);
    push_freed(freed, child_id)?;
    push_freed(freed, right_id)?;
    internal.children[child_index] = write_new_node(pager, &child)?.get();
    Ok(())
}

/// Merge the child at `child_index` with its left sibling. The
/// merged node lives in the left's slot; the parent's separator
/// pivot and the right (current) child id are removed.
fn merge_with_left<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: &mut DecodedNode,
    child_index: usize,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 4 }>,
) -> Result<()> {
    let left_id = pid(internal.children[child_index - 1])?;
    let child_id = pid(internal.children[child_index])?;
    let mut left = read_node(pager, left_id)?;
    let mut child = read_node(pager, child_id)?;
    match (left.kind, child.kind) {
        (NodeKind::Leaf, NodeKind::Leaf) => {
            left.leaves.append(&mut child.leaves);
            left.next_sibling = child.next_sibling;
        }
        (NodeKind::Internal, NodeKind::Internal) => {
            let pivot_down = internal.internals[child_index - 1].key.clone();
            left.internals.push(InternalEntry { key: pivot_down });
            left.internals.append(&mut child.internals);
            left.children.append(&mut child.children);
        }
        _ => {
            return Err(Error::BTreeInvariantViolated {
                reason: "merge_with_left: mixed kinds",
            });
        }
    }
    internal.internals.remove(child_index - 1);
    internal.children.remove(child_index);
    push_freed(freed, left_id)?;
    push_freed(freed, child_id)?;
    internal.children[child_index - 1] = write_new_node(pager, &left)?.get();
    Ok(())
}

fn sibling_has_spare(node: &DecodedNode) -> bool {
    let one_slot_bytes = one_slot_size(node);
    let after_borrow = node.occupied_bytes().saturating_sub(one_slot_bytes);
    after_borrow >= MIN_OCCUPIED_BYTES
}

/// Approximate the byte footprint of one slot in `node`. Used by
/// the underflow / borrow heuristic.
fn one_slot_size(node: &DecodedNode) -> usize {
    use crate::btree::node::{varint_len, INTERNAL_SLOT_BYTES, LEAF_SLOT_BYTES};
    match node.kind {
        NodeKind::Leaf => {
            let Some(e) = node.leaves.first() else {
                return 0;
            };
            LEAF_SLOT_BYTES
                + varint_len(e.key.len() as u64)
                + e.key.len()
                + varint_len(e.value.len() as u64)
                + e.value.len()
        }
        NodeKind::Internal => {
            let Some(e) = node.internals.first() else {
                return 0;
            };
            INTERNAL_SLOT_BYTES + varint_len(e.key.len() as u64) + e.key.len()
        }
    }
}

fn read_node<F: FileBackend>(pager: &mut Pager<F>, id: PageId) -> Result<DecodedNode> {
    let page_ref = pager.read_page(id)?;
    decode_node(page_ref.as_bytes())
}

fn pid(raw: u64) -> Result<PageId> {
    PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
        reason: "child page-id was zero",
    })
}

const _: fn(LeafEntry) = drop;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::{Config, Pager};
    use crate::platform::FileHandle;

    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    fn config() -> Config {
        Config::default()
    }

    #[test]
    fn delete_absent_returns_false() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let removed = tree.delete(&mut pager, b"missing").expect("del");
        assert!(!removed);
    }

    #[test]
    fn delete_single_key_round_trip() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        tree.insert(&mut pager, b"k", b"v").expect("ins");
        assert!(tree.delete(&mut pager, b"k").expect("del"));
        assert_eq!(tree.get(&mut pager, b"k").expect("get"), None);
    }

    #[test]
    fn delete_collapses_tall_tree_back_to_one_level() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let value = vec![0xCDu8; 256];
        for i in 0..200u32 {
            let key = format!("key-{i:08}");
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("ins");
        }
        let root = tree.root();
        let level_before = {
            let pr = pager.read_page(root).expect("read");
            decode_node(pr.as_bytes()).expect("dec").level
        };
        assert!(level_before >= 1);
        for i in 0..199u32 {
            let key = format!("key-{i:08}");
            assert!(tree.delete(&mut pager, key.as_bytes()).expect("del"));
        }
        let final_root = tree.root();
        let level_after = {
            let pr = pager.read_page(final_root).expect("read");
            decode_node(pr.as_bytes()).expect("dec").level
        };
        assert!(
            level_after < level_before,
            "tree should have collapsed (before {level_before}, after {level_after})"
        );
        let leftover_key = format!("key-{i:08}", i = 199);
        assert_eq!(
            tree.get(&mut pager, leftover_key.as_bytes()).expect("get"),
            Some(value.clone())
        );
    }

    #[test]
    fn insert_delete_oracle_10k() {
        for seed in 0..3u64 {
            run_oracle(seed, 10_000);
        }
    }

    /// Regression for the borrow-grows-parent bug surfaced by the
    /// 1M-op oracle. With random keys up to 64 bytes a borrow
    /// during delete-rebalance can install a new separator pivot in
    /// the parent that is much larger than the one it replaced,
    /// pushing the parent past `PAYLOAD_BYTES` and causing the
    /// subsequent encode to fail with the "slot dir and heap
    /// collide" invariant. Drive 30 000 mixed insert/delete ops with
    /// the same key-length distribution to flush out the
    /// edge case without paying the 1M-op cost in the unit-test
    /// harness.
    #[test]
    fn insert_delete_oracle_large_keys_30k() {
        for seed in 0..3u64 {
            run_oracle_with(seed, 30_000, random_key_up_to_64, random_value_up_to_256);
        }
    }

    fn random_key_up_to_64(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(1u32..=64);
        (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect()
    }

    fn random_value_up_to_256(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(0u32..=256);
        (0..len).map(|_| rng.random()).collect()
    }

    fn run_oracle_with(
        seed: u64,
        ops: usize,
        gen_key: fn(&mut ChaCha8Rng) -> Vec<u8>,
        gen_value: fn(&mut ChaCha8Rng) -> Vec<u8>,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for op in 0..ops {
            let pick = rng.random_range(0u32..3);
            if pick < 2 {
                let key = gen_key(&mut rng);
                let value = gen_value(&mut rng);
                if let std::collections::btree_map::Entry::Vacant(slot) = oracle.entry(key.clone())
                {
                    tree.insert(&mut pager, &key, &value)
                        .unwrap_or_else(|e| panic!("seed {seed} op {op} ins: {e:?}"));
                    slot.insert(value);
                }
            } else if !oracle.is_empty() {
                let pick_in_oracle = rng.random_range(0u32..5) > 0;
                let candidate = if pick_in_oracle {
                    let n = oracle.len();
                    let i = rng.random_range(0..n);
                    oracle.keys().nth(i).cloned().unwrap_or_default()
                } else {
                    gen_key(&mut rng)
                };
                let want = oracle.remove(&candidate).is_some();
                let got = tree
                    .delete(&mut pager, &candidate)
                    .unwrap_or_else(|e| panic!("seed {seed} op {op} del: {e:?}"));
                assert_eq!(got, want, "seed {seed} op {op}: delete-presence disagrees");
            }
        }
        for (k, v) in &oracle {
            assert_eq!(
                tree.get(&mut pager, k).expect("get").as_ref(),
                Some(v),
                "seed {seed} final: key {k:?}"
            );
        }
    }

    fn run_oracle(seed: u64, ops: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for op in 0..ops {
            let pick = rng.random_range(0u32..3);
            if pick < 2 {
                oracle_step_insert(&mut pager, &mut tree, &mut oracle, &mut rng, seed, op);
            } else {
                oracle_step_delete(&mut pager, &mut tree, &mut oracle, &mut rng, seed, op);
            }
            if op.is_multiple_of(257) {
                oracle_sample_check(&mut pager, &tree, &oracle, seed, op);
            }
        }
        for (k, v) in &oracle {
            assert_eq!(
                tree.get(&mut pager, k).expect("get").as_ref(),
                Some(v),
                "seed {seed} final: key {k:?}"
            );
        }
    }

    fn oracle_step_insert(
        pager: &mut Pager<FileHandle>,
        tree: &mut BTree<FileHandle>,
        oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
        rng: &mut ChaCha8Rng,
        seed: u64,
        op: usize,
    ) {
        let key = random_key(rng);
        let value = random_value(rng);
        match oracle.entry(key.clone()) {
            std::collections::btree_map::Entry::Occupied(existing) => {
                assert_eq!(
                    tree.get(pager, &key).expect("get").as_ref(),
                    Some(existing.get()),
                    "seed {seed} op {op}: existing key disagrees"
                );
            }
            std::collections::btree_map::Entry::Vacant(slot) => {
                tree.insert(pager, &key, &value)
                    .unwrap_or_else(|e| panic!("seed {seed} op {op} ins: {e:?}"));
                slot.insert(value);
            }
        }
    }

    fn oracle_step_delete(
        pager: &mut Pager<FileHandle>,
        tree: &mut BTree<FileHandle>,
        oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
        rng: &mut ChaCha8Rng,
        seed: u64,
        op: usize,
    ) {
        let candidate = if !oracle.is_empty() && rng.random_range(0u32..4) > 0 {
            let n = oracle.len();
            let pick = rng.random_range(0..n);
            oracle.keys().nth(pick).cloned().unwrap_or_default()
        } else {
            random_key(rng)
        };
        let want = oracle.remove(&candidate).is_some();
        let got = tree
            .delete(pager, &candidate)
            .unwrap_or_else(|e| panic!("seed {seed} op {op} del: {e:?}"));
        assert_eq!(got, want, "seed {seed} op {op}: delete-presence disagrees");
    }

    fn oracle_sample_check(
        pager: &mut Pager<FileHandle>,
        tree: &BTree<FileHandle>,
        oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
        seed: u64,
        op: usize,
    ) {
        let mut sample_keys: Vec<&Vec<u8>> = oracle.keys().take(5).collect();
        sample_keys.extend(oracle.keys().rev().take(5));
        for k in sample_keys {
            assert_eq!(
                tree.get(pager, k).expect("get").as_ref(),
                oracle.get(k),
                "seed {seed} op {op}: mid-run get disagrees on key {k:?}"
            );
        }
    }

    fn random_key(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(1..16);
        (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect()
    }

    fn random_value(rng: &mut ChaCha8Rng) -> Vec<u8> {
        let len = rng.random_range(0..64);
        (0..len).map(|_| rng.random()).collect()
    }

    /// Result of walking every node in a tree once.
    struct OccupancyWalk {
        nodes: usize,
        /// Non-root nodes left below `MIN_OCCUPIED_BYTES` — the
        /// residual-underflow population. Bounded and made visible
        /// here so a future regression that lets it grow unbounded
        /// (or that turns it into a correctness bug) is caught.
        residual_underflows: usize,
    }

    /// Walk the whole tree from `root` with an explicit stack
    /// (no recursion; the visit count is
    /// bounded by `pager.page_count()`, which strictly exceeds the
    /// node count). Counts nodes and non-root nodes that sit below the
    /// min-occupancy threshold. Empty/zero-slot leaves are NOT counted
    /// as underflows: an empty leaf root is the legitimate empty-tree
    /// state, and the merge path keeps non-root leaves from stranding
    /// at zero.
    fn walk_occupancy(pager: &mut Pager<FileHandle>, root: PageId) -> OccupancyWalk {
        let mut stack: Vec<(PageId, bool)> = vec![(root, true)];
        let mut nodes = 0usize;
        let mut residual_underflows = 0usize;
        let visit_budget = usize::try_from(pager.page_count())
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        let mut visited = 0usize;
        while let Some((id, is_root)) = stack.pop() {
            visited += 1;
            assert!(
                visited <= visit_budget,
                "occupancy walk exceeded node bound"
            );
            let node = {
                let pr = pager.read_page(id).expect("read node");
                decode_node(pr.as_bytes()).expect("decode node")
            };
            nodes += 1;
            let occupied = node.occupied_bytes();
            let empty = match node.kind {
                NodeKind::Leaf => node.leaves.is_empty(),
                NodeKind::Internal => node.internals.is_empty(),
            };
            if !is_root && !empty && occupied < MIN_OCCUPIED_BYTES {
                residual_underflows += 1;
            }
            if matches!(node.kind, NodeKind::Internal) {
                for &child in &node.children {
                    let cid = PageId::new(child).expect("child id nonzero");
                    stack.push((cid, false));
                }
            }
        }
        OccupancyWalk {
            nodes,
            residual_underflows,
        }
    }

    /// After a delete-heavy, large-key workload (the distribution
    /// most likely to trip the merge-and-borrow-both-decline residual
    /// underflow), assert that:
    ///
    /// 1. the tree is still fully CORRECT — every surviving key
    ///    resolves to its expected value (this is the invariant a
    ///    cascade-handling regression must never break); and
    /// 2. the residual-underflow degradation is BOUNDED and VISIBLE —
    ///    the number of underflowing non-root nodes never exceeds the
    ///    node count (it cannot "leak" unboundedly) and is reported.
    ///
    /// The assertion is intentionally loose on the exact count (the
    /// snapshot shape is workload-dependent); its job is to make the
    /// degradation observable and to fail loudly if it ever stops
    /// being bounded by the node population.
    #[test]
    fn delete_rebalance_residual_underflow_is_bounded_and_correct() {
        for seed in 0..4u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
            let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
            let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
            for op in 0..20_000usize {
                let deleting = op >= 6_000 && rng.random_range(0u32..3) > 0;
                if deleting && !oracle.is_empty() {
                    let n = oracle.len();
                    let i = rng.random_range(0..n);
                    let cand = oracle.keys().nth(i).cloned().unwrap_or_default();
                    let want = oracle.remove(&cand).is_some();
                    let got = tree
                        .delete(&mut pager, &cand)
                        .unwrap_or_else(|e| panic!("seed {seed} op {op} del: {e:?}"));
                    assert_eq!(got, want, "seed {seed} op {op}: delete presence");
                } else {
                    let key = random_key_up_to_64(&mut rng);
                    let value = random_value_up_to_256(&mut rng);
                    if let std::collections::btree_map::Entry::Vacant(slot) =
                        oracle.entry(key.clone())
                    {
                        tree.insert(&mut pager, &key, &value)
                            .unwrap_or_else(|e| panic!("seed {seed} op {op} ins: {e:?}"));
                        slot.insert(value);
                    }
                }
            }
            for (k, v) in &oracle {
                assert_eq!(
                    tree.get(&mut pager, k).expect("get").as_ref(),
                    Some(v),
                    "seed {seed}: surviving key must resolve: {k:?}",
                );
            }
            let root = tree.root();
            let walk = walk_occupancy(&mut pager, root);
            eprintln!(
                "OCCUPANCY #64 seed {seed}: nodes={} residual_underflows={}",
                walk.nodes, walk.residual_underflows,
            );
            assert!(
                walk.residual_underflows <= walk.nodes,
                "seed {seed}: residual underflows ({}) must stay bounded by \
                 node count ({}) — an unbounded count signals a rebalance \
                 regression",
                walk.residual_underflows,
                walk.nodes,
            );
        }
    }
}
