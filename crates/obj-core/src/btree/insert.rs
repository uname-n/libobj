//! B+tree insert path with copy-on-write splits.
//!
//! Every node touched along the insert path is rewritten to a
//! freshly-allocated page; the displaced pages enter the freelist
//! only after the new root is staged in the pager's WAL transaction
//! buffer. The caller is responsible for calling `Pager::commit` to
//! make the insert durable.

#![forbid(unsafe_code)]

use heapless::Vec as HeaplessVec;

use crate::btree::node::{
    decode_node, encode_node, max_inline_value, max_key_len, DecodedNode, InternalEntry, LeafEntry,
    NodeKind, INTERNAL_LEFTMOST_CHILD_BYTES, INTERNAL_SLOT_BYTES, LEAF_SLOT_BYTES,
};
use crate::btree::{BTree, MAX_BTREE_DEPTH};
use crate::error::{Error, Result};
use crate::pager::page::{Page, PageId, PAGE_SIZE, PAGE_TRAILER_SIZE};
use crate::pager::Pager;
use crate::platform::FileBackend;

/// One step in a descending path: the node's id, its decoded
/// representation, and the index of the child we followed.
struct PathFrame {
    page_id: PageId,
    node: DecodedNode,
    /// For internal-node frames: the child index we recursed into.
    /// Unused for the leaf frame at the bottom of the stack.
    child_index: usize,
}

/// One promoted (pivot key, right-child page-id) pair produced by a
/// split. The parent inserts these in order immediately after the
/// slot the split node occupied.
struct Promoted {
    /// Smallest key of the right sibling (leaf split) or the pivot
    /// moved up (internal split).
    key: Vec<u8>,
    /// Page-id of the right sibling this pivot routes to.
    right_id: PageId,
}

/// Outcome of replacing one node along the insert path: either the
/// node still fits in a single page (we just COW it), or it had to
/// split into N siblings (N ≥ 2) with N-1 promoted pivot keys.
///
/// # Split invariant
///
/// A [`ReplaceOutcome::Split`] always reports a `left_id` plus a
/// non-empty `promoted` list. EVERY node referenced — `left_id` and
/// every `promoted[i].right_id` — was written by [`write_new_node`],
/// which calls [`encode_node`] and therefore guarantees
/// `occupied_bytes() <= PAYLOAD_BYTES`. The split routines below
/// (`split_leaf` / `split_internal`) pack entries by BYTES, never by
/// count, so no resulting node can overflow a page even when one
/// entry is large relative to the page. The parent
/// (`replace_internal` / `build_new_root`) absorbs the WHOLE
/// `promoted` list, which may itself overflow the parent and cascade
/// the split upward — that is handled by the same byte-aware
/// `split_internal`.
enum ReplaceOutcome {
    /// Node fits in one page; new page-id replaces the old one in
    /// its parent's child slot.
    Fits { new_id: PageId },
    /// Node split into `left_id` plus one or more right siblings. The
    /// `promoted` pivots/children are inserted, in order, immediately
    /// after the original slot in the parent.
    Split {
        left_id: PageId,
        promoted: Vec<Promoted>,
    },
}

/// Total bytes available for the slot directory + heap in a single
/// node (excludes the node header and the page trailer).
const PAYLOAD_BYTES: usize = PAGE_SIZE - PAGE_TRAILER_SIZE - crate::btree::node::NODE_HEADER_SIZE;

impl<F: FileBackend> BTree<F> {
    /// Insert `key → value` into the tree. Returns
    /// [`Error::BTreeKeyExists`] if `key` is already present.
    ///
    /// Copy-on-write: every node touched on the path is allocated as
    /// a fresh page through the pager. The pre-insert pages enter the
    /// freelist before this function returns, but only after every
    /// new page has been staged in the WAL transaction. The caller
    /// must call [`Pager::commit`] to make the insert durable.
    ///
    /// # Errors
    ///
    /// - [`Error::BTreeKeyTooLarge`] if the key exceeds `PAGE_SIZE / 4`.
    /// - [`Error::BTreeValueTooLarge`] if the (key, value) pair will
    ///   not fit inline in a leaf.
    /// - [`Error::BTreeKeyExists`] if `key` is already present.
    /// - [`Error::BTreeDepthExceeded`] if the tree height would
    ///   exceed `MAX_BTREE_DEPTH`.
    /// - [`Error::Corruption`] / [`Error::Io`] propagated from the
    ///   pager.
    pub fn insert(&mut self, pager: &mut Pager<F>, key: &[u8], value: &[u8]) -> Result<()> {
        check_key_value_size(key, value)?;
        let path = self.descend_with_path(pager, key)?;
        self.apply_insert(pager, path, key, value)
    }

    /// Walk root → leaf, recording each ancestor's page-id, decoded
    /// representation, and the child index that was followed.
    fn descend_with_path(
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
                    if path.push(frame).is_err() {
                        return Err(Error::BTreeDepthExceeded {
                            limit: MAX_BTREE_DEPTH,
                        });
                    }
                    return Ok(path);
                }
                NodeKind::Internal => {
                    let child_index = pivot_index(&decoded, key);
                    let raw = decoded.children[child_index];
                    let next = PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
                        reason: "internal node had zero child page-id",
                    })?;
                    let frame = PathFrame {
                        page_id: current,
                        node: decoded,
                        child_index,
                    };
                    if path.push(frame).is_err() {
                        return Err(Error::BTreeDepthExceeded {
                            limit: MAX_BTREE_DEPTH,
                        });
                    }
                    current = next;
                }
            }
        }
    }

    /// Apply the (key, value) insertion to the leaf at the bottom of
    /// the path, then bubble any split up to the root.
    fn apply_insert(
        &mut self,
        pager: &mut Pager<F>,
        mut path: HeaplessVec<PathFrame, MAX_BTREE_DEPTH>,
        key: &[u8],
        value: &[u8],
    ) -> Result<()> {
        let mut freed: HeaplessVec<PageId, { MAX_BTREE_DEPTH * 2 }> = HeaplessVec::new();
        let Some(leaf_frame) = path.pop() else {
            return Err(Error::BTreeInvariantViolated {
                reason: "insert: descend returned empty path",
            });
        };
        let mut outcome = replace_leaf(pager, leaf_frame, key, value, &mut freed)?;
        while let Some(parent_frame) = path.pop() {
            outcome = replace_internal(pager, parent_frame, outcome, &mut freed)?;
        }
        let new_root = build_new_root(pager, outcome)?;
        self.root = new_root;
        for old_id in freed.iter().copied() {
            pager.free_page(old_id)?;
        }
        Ok(())
    }
}

fn replace_leaf<F: FileBackend>(
    pager: &mut Pager<F>,
    frame: PathFrame,
    key: &[u8],
    value: &[u8],
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 2 }>,
) -> Result<ReplaceOutcome> {
    let mut leaf = frame.node;
    if leaf.leaves.iter().any(|e| e.key.as_slice() == key) {
        return Err(Error::BTreeKeyExists);
    }
    let insert_at = leaf
        .leaves
        .iter()
        .position(|e| e.key.as_slice() > key)
        .unwrap_or(leaf.leaves.len());
    leaf.leaves.insert(
        insert_at,
        LeafEntry {
            key: key.to_vec(),
            value: value.to_vec(),
        },
    );
    push_freed(freed, frame.page_id)?;
    if leaf.occupied_bytes() <= PAYLOAD_BYTES {
        let new_id = write_new_node(pager, &leaf)?;
        return Ok(ReplaceOutcome::Fits { new_id });
    }
    split_leaf(pager, leaf)
}

fn replace_internal<F: FileBackend>(
    pager: &mut Pager<F>,
    frame: PathFrame,
    child_outcome: ReplaceOutcome,
    freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 2 }>,
) -> Result<ReplaceOutcome> {
    let mut internal = frame.node;
    let idx = frame.child_index;
    match child_outcome {
        ReplaceOutcome::Fits { new_id } => {
            internal.children[idx] = new_id.get();
        }
        ReplaceOutcome::Split { left_id, promoted } => {
            internal.children[idx] = left_id.get();
            for (i, p) in promoted.into_iter().enumerate() {
                internal
                    .internals
                    .insert(idx + i, InternalEntry { key: p.key });
                internal.children.insert(idx + 1 + i, p.right_id.get());
            }
        }
    }
    push_freed(freed, frame.page_id)?;
    if internal.occupied_bytes() <= PAYLOAD_BYTES {
        let new_id = write_new_node(pager, &internal)?;
        return Ok(ReplaceOutcome::Fits { new_id });
    }
    split_internal(pager, internal)
}

/// Build the new root from the outcome of `apply_insert`'s final
/// bubble. If no split, the root is just the new id; if a split, we
/// allocate a fresh internal node holding (left, promoted...).
///
/// The new root absorbs the WHOLE promoted list. With a byte-aware
/// multi-way split the promoted list can carry several large pivots,
/// so the freshly-built root may itself overflow `PAYLOAD_BYTES`; in
/// that case it is split via [`split_internal`] and a level is added
/// per split. The loop is bounded by `MAX_BTREE_DEPTH`: each
/// iteration adds one tree level, and a tree deeper than the bound is
/// a [`Error::BTreeDepthExceeded`].
fn build_new_root<F: FileBackend>(
    pager: &mut Pager<F>,
    mut outcome: ReplaceOutcome,
) -> Result<PageId> {
    for _ in 0..MAX_BTREE_DEPTH {
        let (left_id, promoted) = match outcome {
            ReplaceOutcome::Fits { new_id } => return Ok(new_id),
            ReplaceOutcome::Split { left_id, promoted } => (left_id, promoted),
        };
        let level = node_level_after_split(pager, left_id)?;
        let next_level = level.checked_add(1).ok_or(Error::BTreeDepthExceeded {
            limit: MAX_BTREE_DEPTH,
        })?;
        let mut children = Vec::with_capacity(promoted.len() + 1);
        let mut internals = Vec::with_capacity(promoted.len());
        children.push(left_id.get());
        for p in promoted {
            internals.push(InternalEntry { key: p.key });
            children.push(p.right_id.get());
        }
        let root_node = DecodedNode {
            kind: NodeKind::Internal,
            level: next_level,
            next_sibling: 0,
            children,
            leaves: Vec::new(),
            internals,
        };
        if root_node.occupied_bytes() <= PAYLOAD_BYTES {
            return write_new_node(pager, &root_node);
        }
        outcome = split_internal(pager, root_node)?;
    }
    Err(Error::BTreeDepthExceeded {
        limit: MAX_BTREE_DEPTH,
    })
}

fn push_freed(freed: &mut HeaplessVec<PageId, { MAX_BTREE_DEPTH * 2 }>, id: PageId) -> Result<()> {
    freed.push(id).map_err(|_| Error::BTreeInvariantViolated {
        reason: "insert: too many displaced pages to track",
    })
}

/// Pick the child index in `node` whose subtree contains `key`.
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

/// Validate that `key` / `value` fit the format-version-0 bounds.
fn check_key_value_size(key: &[u8], value: &[u8]) -> Result<()> {
    if key.len() > max_key_len() {
        return Err(Error::BTreeKeyTooLarge {
            key_len: key.len(),
            max: max_key_len(),
        });
    }
    let v_max = max_inline_value(key.len());
    if value.len() > v_max {
        return Err(Error::BTreeValueTooLarge {
            value_len: value.len(),
            max: v_max,
        });
    }
    Ok(())
}

/// Encode `node` into a fresh page and write it through the pager.
/// Returns the newly-allocated `PageId`.
pub(crate) fn write_new_node<F: FileBackend>(
    pager: &mut Pager<F>,
    node: &DecodedNode,
) -> Result<PageId> {
    let new_id = pager.alloc_page()?;
    let mut page = Page::zeroed();
    encode_node(node, &mut page)?;
    pager.write_page(new_id, &page)?;
    Ok(new_id)
}

/// Byte footprint one leaf entry occupies in an encoded node: its
/// slot-directory entry plus its heap entry (two length-prefixed
/// byte strings). Mirrors the accounting in
/// [`DecodedNode::occupied_bytes`] for a leaf exactly.
fn leaf_entry_bytes(entry: &LeafEntry) -> usize {
    use crate::btree::node::varint_len;
    LEAF_SLOT_BYTES
        + varint_len(entry.key.len() as u64)
        + entry.key.len()
        + varint_len(entry.value.len() as u64)
        + entry.value.len()
}

/// Byte footprint one internal pivot occupies in an encoded node: its
/// slot-directory entry plus its length-prefixed heap key. Mirrors
/// the per-pivot term of [`DecodedNode::occupied_bytes`] for an
/// internal node. The leftmost-child term is accounted separately by
/// the caller because it is per-NODE, not per-pivot.
fn internal_pivot_bytes(entry: &InternalEntry) -> usize {
    use crate::btree::node::varint_len;
    INTERNAL_SLOT_BYTES + varint_len(entry.key.len() as u64) + entry.key.len()
}

/// Split an overflowing leaf into the minimum number of byte-bounded
/// siblings.
///
/// # Why by BYTES, not by count
///
/// Entries are variable-length. A count-midpoint split can leave a
/// half whose `occupied_bytes()` still exceeds `PAYLOAD_BYTES` (a
/// near-full leaf of small entries plus one large entry), which made
/// [`encode_node`] raise the "slot dir and heap collide" invariant
/// and surfaced as a spurious write error. This greedy packer walks
/// entries left → right, opening a new sibling whenever the next
/// entry would push the current sibling past `PAYLOAD_BYTES`.
///
/// # Split invariant
///
/// Every emitted sibling satisfies `occupied_bytes() <=
/// PAYLOAD_BYTES`. Each sibling holds at least one entry, because
/// [`check_key_value_size`] guarantees a single (key, value) pair
/// always fits in one node (`max_inline_value` reserves a slot plus
/// two varints), so a fresh-sibling never immediately overflows on
/// its first entry. The siblings' `next_sibling` chain is rebuilt
/// left → right so a forward range scan visits every entry in order.
fn split_leaf<F: FileBackend>(pager: &mut Pager<F>, leaf: DecodedNode) -> Result<ReplaceOutcome> {
    debug_assert!(matches!(leaf.kind, NodeKind::Leaf));
    debug_assert!(leaf.leaves.len() >= 2, "leaf split needs ≥ 2 entries");
    let original_sibling = leaf.next_sibling;
    let groups = pack_leaf_groups(leaf.leaves)?;
    write_leaf_groups(pager, groups, original_sibling)
}

/// Greedily pack leaf entries into byte-bounded groups. The first
/// group is the left node (keeps the original slot in the parent);
/// the rest become promoted right siblings. Bounded by the entry
/// count.
fn pack_leaf_groups(entries: Vec<LeafEntry>) -> Result<Vec<Vec<LeafEntry>>> {
    let mut groups: Vec<Vec<LeafEntry>> = Vec::new();
    let mut current: Vec<LeafEntry> = Vec::new();
    let mut current_bytes = 0usize;
    for entry in entries {
        let entry_bytes = leaf_entry_bytes(&entry);
        if !current.is_empty() && current_bytes + entry_bytes > PAYLOAD_BYTES {
            groups.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current_bytes += entry_bytes;
        current.push(entry);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    if groups.len() < 2 {
        return Err(Error::BTreeInvariantViolated {
            reason: "leaf split produced fewer than two groups",
        });
    }
    Ok(groups)
}

/// Write each leaf group to a fresh page, chaining `next_sibling`
/// left → right (the last group inherits the original right-sibling
/// pointer). Returns the left id plus the promoted right siblings.
/// Bounded by the group count.
fn write_leaf_groups<F: FileBackend>(
    pager: &mut Pager<F>,
    groups: Vec<Vec<LeafEntry>>,
    original_sibling: u64,
) -> Result<ReplaceOutcome> {
    let count = groups.len();
    let mut next_sibling = original_sibling;
    let mut ids: Vec<(PageId, Vec<u8>)> = Vec::with_capacity(count);
    for (i, entries) in groups.into_iter().enumerate().rev() {
        let promoted_key =
            entries
                .first()
                .map(|e| e.key.clone())
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "leaf split produced an empty group",
                })?;
        let node = DecodedNode {
            kind: NodeKind::Leaf,
            level: 0,
            next_sibling,
            children: Vec::new(),
            leaves: entries,
            internals: Vec::new(),
        };
        let id = write_new_node(pager, &node)?;
        next_sibling = id.get();
        let key = if i == 0 { Vec::new() } else { promoted_key };
        ids.push((id, key));
    }
    ids.reverse();
    assemble_split_outcome(ids)
}

/// Turn a left → right ordered `(page_id, promoted_key)` list into a
/// [`ReplaceOutcome::Split`]. The first element is the left node
/// (its key is unused); the rest are promoted right siblings.
fn assemble_split_outcome(mut ids: Vec<(PageId, Vec<u8>)>) -> Result<ReplaceOutcome> {
    if ids.len() < 2 {
        return Err(Error::BTreeInvariantViolated {
            reason: "split outcome needs a left node and ≥ 1 promoted sibling",
        });
    }
    let mut iter = ids.drain(..);
    let (left_id, _) = iter.next().ok_or(Error::BTreeInvariantViolated {
        reason: "split outcome missing left node",
    })?;
    let mut promoted = Vec::with_capacity(iter.len());
    for (right_id, key) in iter {
        promoted.push(Promoted { key, right_id });
    }
    Ok(ReplaceOutcome::Split { left_id, promoted })
}

/// Split an overflowing internal node into the minimum number of
/// byte-bounded siblings. Like the leaf split this packs
/// by BYTES, not by count, so no sibling overflows `PAYLOAD_BYTES`
/// even when a pivot key is large relative to the page.
///
/// An internal split PROMOTES the boundary pivot between two adjacent
/// siblings (it is not retained in either), exactly like the original
/// 2-way split — so for N siblings, N-1 pivots are promoted and the
/// remaining pivots are partitioned among the siblings.
///
/// # Split invariant
///
/// Every emitted sibling satisfies `occupied_bytes() <=
/// PAYLOAD_BYTES` and keeps `children.len() == internals.len() + 1`.
/// Each sibling holds at least one pivot (so each has ≥ 2 children),
/// because a single pivot entry plus the leftmost-child term always
/// fits in a node (`max_key_len = PAGE_SIZE / 4` ≪ `PAYLOAD_BYTES`).
fn split_internal<F: FileBackend>(
    pager: &mut Pager<F>,
    internal: DecodedNode,
) -> Result<ReplaceOutcome> {
    debug_assert!(matches!(internal.kind, NodeKind::Internal));
    debug_assert!(
        internal.internals.len() >= 2,
        "internal split needs ≥ 2 pivots"
    );
    let level = internal.level;
    let groups = pack_internal_groups(internal.internals, internal.children)?;
    write_internal_groups(pager, groups, level)
}

/// One byte-bounded internal sibling produced by the split: its
/// child pointers, its retained pivots, and (for siblings after the
/// first) the boundary pivot promoted to the parent.
struct InternalGroup {
    children: Vec<u64>,
    internals: Vec<InternalEntry>,
    /// `None` for the first group (left node); `Some(key)` is the
    /// pivot promoted to the parent ahead of this sibling.
    promoted: Option<Vec<u8>>,
}

/// Greedily pack internal pivots/children into byte-bounded groups,
/// promoting the boundary pivot between adjacent groups. Bounded by
/// the pivot count.
///
/// `children.len()` is `internals.len() + 1`. We walk pivots left →
/// right; child `i` always travels with pivot `i` (it is the child to
/// the pivot's LEFT), and the trailing child `K` closes the last
/// group. When the running group is non-empty and adding the next
/// pivot+child would exceed `PAYLOAD_BYTES`, that pivot is PROMOTED
/// (it becomes the parent separator) and a fresh group is opened
/// starting with the pivot's right child.
fn pack_internal_groups(
    pivots: Vec<InternalEntry>,
    children: Vec<u64>,
) -> Result<Vec<InternalGroup>> {
    if children.len() != pivots.len() + 1 {
        return Err(Error::BTreeInvariantViolated {
            reason: "internal split: children.len() != pivots+1",
        });
    }
    let mut groups: Vec<InternalGroup> = Vec::new();
    let mut cur_children: Vec<u64> = Vec::new();
    let mut cur_pivots: Vec<InternalEntry> = Vec::new();
    let mut cur_bytes = INTERNAL_LEFTMOST_CHILD_BYTES;
    let mut pending_promote: Option<Vec<u8>> = None;
    let mut child_iter = children.into_iter();
    let leftmost = child_iter.next().ok_or(Error::BTreeInvariantViolated {
        reason: "internal split: missing leftmost child",
    })?;
    cur_children.push(leftmost);
    for pivot in pivots {
        let right_child = child_iter.next().ok_or(Error::BTreeInvariantViolated {
            reason: "internal split: missing right child for pivot",
        })?;
        let pivot_bytes = internal_pivot_bytes(&pivot);
        if !cur_pivots.is_empty() && cur_bytes + pivot_bytes > PAYLOAD_BYTES {
            groups.push(InternalGroup {
                children: std::mem::take(&mut cur_children),
                internals: std::mem::take(&mut cur_pivots),
                promoted: pending_promote.take(),
            });
            pending_promote = Some(pivot.key);
            cur_children.push(right_child);
            cur_bytes = INTERNAL_LEFTMOST_CHILD_BYTES;
        } else {
            cur_bytes += pivot_bytes;
            cur_pivots.push(pivot);
            cur_children.push(right_child);
        }
    }
    groups.push(InternalGroup {
        children: cur_children,
        internals: cur_pivots,
        promoted: pending_promote,
    });
    if groups.len() < 2 {
        return Err(Error::BTreeInvariantViolated {
            reason: "internal split produced fewer than two groups",
        });
    }
    Ok(groups)
}

/// Write each internal group to a fresh page and assemble the
/// promoted (pivot, right-id) list. Bounded by the group count.
fn write_internal_groups<F: FileBackend>(
    pager: &mut Pager<F>,
    groups: Vec<InternalGroup>,
    level: u8,
) -> Result<ReplaceOutcome> {
    let mut left_id: Option<PageId> = None;
    let mut promoted: Vec<Promoted> = Vec::new();
    for group in groups {
        let node = DecodedNode {
            kind: NodeKind::Internal,
            level,
            next_sibling: 0,
            children: group.children,
            leaves: Vec::new(),
            internals: group.internals,
        };
        let id = write_new_node(pager, &node)?;
        match group.promoted {
            None => left_id = Some(id),
            Some(key) => promoted.push(Promoted { key, right_id: id }),
        }
    }
    let left_id = left_id.ok_or(Error::BTreeInvariantViolated {
        reason: "internal split produced no left node",
    })?;
    if promoted.is_empty() {
        return Err(Error::BTreeInvariantViolated {
            reason: "internal split produced no promoted siblings",
        });
    }
    Ok(ReplaceOutcome::Split { left_id, promoted })
}

/// Determine the level of a node that just emerged from a split. We
/// read the left half's page back through the pager (cheap — it was
/// just staged in the WAL view) to learn its level.
fn node_level_after_split<F: FileBackend>(pager: &mut Pager<F>, id: PageId) -> Result<u8> {
    let page_ref = pager.read_page(id)?;
    let decoded = decode_node(page_ref.as_bytes())?;
    Ok(decoded.level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::{Config, Pager};
    use crate::platform::FileHandle;

    use proptest::prelude::*;
    use rand::prelude::IndexedRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

    fn config() -> Config {
        Config::default()
    }

    #[test]
    fn insert_single_key_round_trip() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        tree.insert(&mut pager, b"hello", b"world").expect("ins");
        assert_eq!(
            tree.get(&mut pager, b"hello").expect("get"),
            Some(b"world".to_vec())
        );
    }

    #[test]
    fn duplicate_key_errors() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        tree.insert(&mut pager, b"k", b"v1").expect("ins");
        let err = tree
            .insert(&mut pager, b"k", b"v2")
            .expect_err("dup must fail");
        assert!(matches!(err, Error::BTreeKeyExists));
    }

    #[test]
    fn insert_growth_splits_root() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let value = vec![0xABu8; 256];
        for i in 0..200u32 {
            let key = format!("key-{i:08}");
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("ins");
        }
        for i in 0..200u32 {
            let key = format!("key-{i:08}");
            assert_eq!(
                tree.get(&mut pager, key.as_bytes()).expect("get"),
                Some(value.clone()),
                "key {key}"
            );
        }
        let root = tree.root();
        let page_ref = pager.read_page(root).expect("read root");
        let decoded = decode_node(page_ref.as_bytes()).expect("decode root");
        assert!(
            decoded.level >= 1,
            "expected internal root, got {decoded:?}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 16,
            max_shrink_iters: 32,
            .. ProptestConfig::default()
        })]

        #[test]
        fn insert_oracle_property(seed in any::<u64>()) {
            run_insert_oracle(seed, 200);
        }
    }

    /// Run 10k random insert operations against a `BTreeMap` oracle.
    /// This is the in-module sanity check; the full 1M-op oracle
    /// lives in `tests/btree_oracle.rs`.
    #[test]
    fn insert_oracle_10k() {
        for seed in 0..3u64 {
            run_insert_oracle(seed, 10_000);
        }
    }

    fn run_insert_oracle(seed: u64, ops: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for op in 0..ops {
            let key = random_key(&mut rng);
            let value = random_value(&mut rng);
            let key_already = oracle.contains_key(&key);
            let res = tree.insert(&mut pager, &key, &value);
            if key_already {
                assert!(
                    matches!(res, Err(Error::BTreeKeyExists)),
                    "seed {seed} op {op}: expected BTreeKeyExists, got {res:?}"
                );
            } else {
                res.unwrap_or_else(|e| panic!("seed {seed} op {op}: insert err {e:?}"));
                oracle.insert(key.clone(), value.clone());
            }
            if op.is_multiple_of(127) {
                let keys: Vec<&Vec<u8>> = oracle.keys().collect();
                if !keys.is_empty() {
                    let sample: Vec<&Vec<u8>> =
                        keys.choose_multiple(&mut rng, 4).copied().collect();
                    for k in sample {
                        assert_eq!(
                            tree.get(&mut pager, k).expect("get").as_ref(),
                            oracle.get(k),
                            "seed {seed} op {op}: key {k:?}"
                        );
                    }
                }
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

    fn random_key(rng: &mut ChaCha8Rng) -> Vec<u8> {
        use rand::Rng;
        let len = rng.random_range(1..16);
        (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect()
    }

    fn random_value(rng: &mut ChaCha8Rng) -> Vec<u8> {
        use rand::Rng;
        let len = rng.random_range(0..64);
        (0..len).map(|_| rng.random()).collect()
    }

    use crate::btree::node::{validate_node, NODE_HEADER_SIZE};

    /// Payload budget every encoded node must respect. Mirrors the
    /// `PAYLOAD_BYTES` const used by the insert path so the test holds
    /// the implementation to its own contract.
    const TEST_PAYLOAD_BYTES: usize = PAGE_SIZE - PAGE_TRAILER_SIZE - NODE_HEADER_SIZE;

    /// A random key in `[1, max_key_len()]`, heavily biased toward
    /// small keys but reaching the format cap a fraction of the time
    /// so large index-key entries exercise the internal-node split.
    fn fuzz_key(rng: &mut ChaCha8Rng) -> Vec<u8> {
        use rand::Rng;
        let cap = max_key_len();
        let len = match rng.random_range(0u32..100) {
            0..=84 => rng.random_range(1..=16usize),
            85..=96 => rng.random_range(16..=256usize),
            _ => rng.random_range(256..=cap),
        };
        let len = len.clamp(1, cap);
        (0..len).map(|_| rng.random::<u8>()).collect()
    }

    /// A random value whose length spans `[0, max_inline_value(key)]`,
    /// heavily sampling LARGE values (hundreds–thousands of bytes) so
    /// a populated leaf is repeatedly hit with an entry big enough to
    /// force a byte-aware (not count) split.
    fn fuzz_value(rng: &mut ChaCha8Rng, key_len: usize) -> Vec<u8> {
        use rand::Rng;
        let cap = max_inline_value(key_len);
        if cap == 0 {
            return Vec::new();
        }
        let len = match rng.random_range(0u32..100) {
            0..=49 => rng.random_range(0..=64usize),
            50..=79 => rng.random_range(64..=512usize),
            _ => rng.random_range(512..=cap.max(512)),
        };
        let len = len.min(cap);
        (0..len).map(|_| rng.random::<u8>()).collect()
    }

    /// Walk every node in the tree (explicit stack, no recursion;
    /// bounded by `page_count`) and assert the split
    /// invariant on each: it decodes (which re-runs
    /// `validate_node_release`), passes `validate_node`, and its
    /// `occupied_bytes()` never exceeds the per-node payload budget.
    fn assert_tree_invariants(pager: &mut Pager<FileHandle>, root: PageId) {
        let mut stack: Vec<PageId> = vec![root];
        let budget = usize::try_from(pager.page_count())
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        let mut visited = 0usize;
        while let Some(id) = stack.pop() {
            visited += 1;
            assert!(visited <= budget, "node walk exceeded page-count bound");
            let node = {
                let pr = pager.read_page(id).expect("read node");
                decode_node(pr.as_bytes()).expect("decode node")
            };
            validate_node(&node).expect("node must satisfy validate_node");
            assert!(
                node.occupied_bytes() <= TEST_PAYLOAD_BYTES,
                "node {id:?} occupies {} bytes > payload cap {} — a split \
                 left an overflowing node (issue #137 regression)",
                node.occupied_bytes(),
                TEST_PAYLOAD_BYTES,
            );
            if matches!(node.kind, NodeKind::Internal) {
                for &child in &node.children {
                    stack.push(PageId::new(child).expect("child id nonzero"));
                }
            }
        }
    }

    /// Drive `ops` random inserts of RANDOM-sized keys and values
    /// (heavily sampling large values) against a `BTreeMap` oracle.
    /// After EACH op the only tolerated error is a documented size
    /// guard (which `fuzz_key`/`fuzz_value` never trip); any
    /// `BTreeInvariantViolated` (the "slot dir and heap collide"
    /// class) fails the test. At the end every oracle key must
    /// round-trip with its EXACT value, an ordered range scan must
    /// match the oracle's order, and every page must satisfy the
    /// split invariant.
    fn run_split_oracle(seed: u64, ops: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for op in 0..ops {
            let key = fuzz_key(&mut rng);
            let value = fuzz_value(&mut rng, key.len());
            let already = oracle.contains_key(&key);
            let res = tree.insert(&mut pager, &key, &value);
            match res {
                Ok(()) => {
                    assert!(!already, "seed {seed} op {op}: dup insert unexpectedly Ok");
                    oracle.insert(key, value);
                }
                Err(Error::BTreeKeyExists) => {
                    assert!(already, "seed {seed} op {op}: spurious BTreeKeyExists");
                }
                Err(e) => panic!(
                    "seed {seed} op {op}: unexpected insert error {e:?} \
                     (key_len={}, value_len={})",
                    key.len(),
                    value.len(),
                ),
            }
            if op.is_multiple_of(521) {
                assert_tree_invariants(&mut pager, tree.root());
            }
        }
        for (k, v) in &oracle {
            assert_eq!(
                tree.get(&mut pager, k).expect("get").as_ref(),
                Some(v),
                "seed {seed} final: key {k:?} value mismatch",
            );
        }
        let scanned: Vec<(Vec<u8>, Vec<u8>)> = tree
            .range(&mut pager, ..)
            .expect("range")
            .map(|r| r.expect("range item"))
            .collect();
        let expected: Vec<(Vec<u8>, Vec<u8>)> =
            oracle.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(
            scanned.len(),
            expected.len(),
            "seed {seed}: range scan count disagrees with oracle",
        );
        assert_eq!(
            scanned, expected,
            "seed {seed}: ordered range scan mismatch"
        );
        assert_tree_invariants(&mut pager, tree.root());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 24,
            max_shrink_iters: 16,
            .. ProptestConfig::default()
        })]

        /// Many seeds × 600 large-value inserts each. The shrink-free
        /// 10k×many-seeds run lives in `split_oracle_large_values_10k`.
        #[test]
        fn split_oracle_property(seed in any::<u64>()) {
            run_split_oracle(seed, 600);
        }
    }

    /// ≥10k random ops across several seeds with random key+value
    /// sizes (heavy on large values). Asserts zero invariant
    /// violations and zero oracle mismatches.
    #[test]
    fn split_oracle_large_values_10k() {
        for seed in 0..5u64 {
            run_split_oracle(seed, 10_000);
        }
    }

    /// Deterministic reproduction: populate a leaf with
    /// ~30 small entries, then insert one ~1.8 KB-record entry that
    /// lands so a count-midpoint split would leave an
    /// overflowing half. Every insert succeeds and the tree
    /// satisfies the split invariant.
    #[test]
    fn issue_137_large_entry_into_populated_leaf() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        for i in 0..30u32 {
            let key = format!("k{i:04}");
            let value = vec![0x5Au8; 8];
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("small insert");
        }
        let big_key = vec![b'm'; 16];
        let big_value = vec![0xA5u8; 1_800];
        tree.insert(&mut pager, &big_key, &big_value)
            .expect("large insert must succeed (issue #137)");
        let big_key2 = vec![b'n'; 16];
        let big_value2 = vec![0xC3u8; max_inline_value(big_key2.len())];
        tree.insert(&mut pager, &big_key2, &big_value2)
            .expect("near-max large insert must succeed");
        assert_eq!(
            tree.get(&mut pager, &big_key).expect("get"),
            Some(big_value)
        );
        assert_eq!(
            tree.get(&mut pager, &big_key2).expect("get"),
            Some(big_value2)
        );
        for i in 0..30u32 {
            let key = format!("k{i:04}");
            assert_eq!(
                tree.get(&mut pager, key.as_bytes()).expect("get"),
                Some(vec![0x5Au8; 8]),
                "small key {key} must survive the split",
            );
        }
        assert_tree_invariants(&mut pager, tree.root());
    }

    /// Pathological size distribution: a leaf
    /// near-full of small entries with a large entry landing in the
    /// MIDDLE, such that a naive 2-way split would leave BOTH halves
    /// overflowing. The multi-way split must still emit only
    /// payload-bounded nodes.
    #[test]
    fn issue_137_large_entry_in_the_middle_multiway() {
        let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
        let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
        for i in 0..20u32 {
            let key = format!("a{i:03}");
            let value = vec![0x11u8; 180];
            tree.insert(&mut pager, key.as_bytes(), &value)
                .expect("fill");
        }
        let mid_key = b"a0095".to_vec();
        let mid_value = vec![0x22u8; max_inline_value(mid_key.len())];
        tree.insert(&mut pager, &mid_key, &mid_value)
            .expect("mid large insert must succeed");
        assert_eq!(
            tree.get(&mut pager, &mid_key).expect("get"),
            Some(mid_value)
        );
        assert_tree_invariants(&mut pager, tree.root());
    }

    /// Insert-then-delete mixed workload of random large/small sizes,
    /// then an integrity walk: every page satisfies the split
    /// invariant and `validate_node`, and every surviving key
    /// round-trips. Guards the interaction between the byte-aware
    /// split and the delete-rebalance path.
    #[test]
    fn split_then_delete_integrity_walk() {
        use rand::Rng;
        for seed in 0..4u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut pager = Pager::<FileHandle>::memory(config()).expect("pager");
            let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty");
            let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
            for _ in 0..6_000usize {
                let deleting = !oracle.is_empty() && rng.random_range(0u32..3) == 0;
                if deleting {
                    let n = oracle.len();
                    let i = rng.random_range(0..n);
                    let cand = oracle.keys().nth(i).cloned().unwrap_or_default();
                    let want = oracle.remove(&cand).is_some();
                    let got = tree.delete(&mut pager, &cand).expect("delete");
                    assert_eq!(got, want, "seed {seed}: delete presence");
                } else {
                    let key = fuzz_key(&mut rng);
                    let value = fuzz_value(&mut rng, key.len());
                    if let std::collections::btree_map::Entry::Vacant(slot) =
                        oracle.entry(key.clone())
                    {
                        tree.insert(&mut pager, &key, &value).expect("insert");
                        slot.insert(value);
                    }
                }
            }
            for (k, v) in &oracle {
                assert_eq!(
                    tree.get(&mut pager, k).expect("get").as_ref(),
                    Some(v),
                    "seed {seed}: surviving key {k:?}",
                );
            }
            assert_tree_invariants(&mut pager, tree.root());
        }
    }
}
