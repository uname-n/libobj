//! B+tree node encode / decode.
//!
//! This module is the reference implementation; the tests at the
//! bottom validate round-trip equality and rejection of every
//! documented invariant violation.
//!
//! # Decoded-node allocation note
//!
//! The in-memory [`DecodedNode`] uses `Vec`s, not `heapless::Vec`s of
//! the worst-case slot count: the worst-case slot count times the
//! worst-case per-key buffer (~1 MiB) would overflow a 2 MiB stack.
//! The B+tree's *traversal stack* uses `heapless::Vec` on the hot
//! read path; the *decoded representation* of a single node is a
//! transient per-page artifact whose heap allocation is unavoidable
//! and is not on a real-time path.
//!
//! The three `Vec`s in [`DecodedNode`] (`children`, `leaves`,
//! `internals`) look like a hot-path allocation cost, but the actual
//! cost is far smaller than it looks and a `heapless`-conversion is
//! not worth shipping. Four reasons:
//! (1) `Vec::new()` does not allocate — the first `reserve` does —
//! so a leaf decode allocates exactly ONE outer `Vec` (the `leaves`
//! spine) and an internal decode allocates TWO (`children` +
//! `internals`); the unused vecs stay at zero capacity.
//! (2) Each leaf entry's `key` and `value` are themselves `Vec<u8>`;
//! for an N-entry leaf that's 2N inner allocations vs. the 1
//! outer, so converting the outer saves at best 1/(2N+1) on a real
//! workload (~0.5 % on a 100-entry leaf).
//! (3) `LEAF_SLOT_CAP ≈ 1017` and `sizeof(LeafEntry) = 48` — a
//! stack-allocated `heapless::Vec<LeafEntry, LEAF_SLOT_CAP>` adds
//! ~48 KiB to every [`decode_node`] frame, a stack-overflow hazard
//! on small-stack targets (embedded, fiber runtimes) and a 48 KiB
//! zero-write per call under debug + stack-protectors. The
//! conversion makes the hot path *worse* in latency-sensitive
//! deployments.
//! (4) `index_lookup` (warm-cache point read) spends most of its
//! time in `pager.read_page` (page-cache hash lookup), CRC32C
//! verification of the page trailer, and the descend stack — not
//! in `Vec::reserve`. A 100k-leaf `collection_scan` does ~100 000
//! outer allocs vs. ~50 000 000 inner allocs (~500 bytes/doc with
//! 2 inner `Vec`s per entry); the outer count is a noise floor.
//!
//! The conclusion: the outer `Vec`s stay. A real perf win for
//! `decode_node` requires attacking the inner-vec allocations
//! (e.g. decode-in-place with `&[u8]` slices into the page buffer,
//! borrowing the pager's page-cache buffer for the iterator's
//! lifetime), which is a format-aware refactor of the node API.

#![forbid(unsafe_code)]

use crate::error::{Error, Result};
use crate::pager::page::{Page, PAGE_SIZE, PAGE_TRAILER_SIZE};

/// Page-type tag for an internal B+tree node.
pub const PAGE_TYPE_BTREE_INTERNAL: u8 = 0x02;
/// Page-type tag for a leaf B+tree node.
pub const PAGE_TYPE_BTREE_LEAF: u8 = 0x03;

/// Fixed-size B+tree node header.
pub const NODE_HEADER_SIZE: usize = 24;

const OFF_PAGE_TYPE: usize = 0;
const OFF_LEVEL: usize = 1;
const OFF_KEY_COUNT: usize = 2;
const OFF_FREE_SPACE_OFF: usize = 4;
const OFF_NEXT_SIBLING: usize = 8;
const OFF_RESERVED: usize = 16;

/// Bytes per slot in a leaf slot directory. The entry is a single
/// u32 LE giving the byte offset of the corresponding heap entry.
pub const LEAF_SLOT_BYTES: usize = 4;

/// Bytes per slot in an internal slot directory. The entry is a u32
/// LE offset followed by a u64 LE right-child page-id.
pub const INTERNAL_SLOT_BYTES: usize = 12;

/// Bytes the leftmost-child page-id occupies in an internal node,
/// immediately after the node header.
pub const INTERNAL_LEFTMOST_CHILD_BYTES: usize = 8;

/// Byte offset of the first byte after the node header (leaf slot
/// directory start, OR internal leftmost-child).
const PAYLOAD_OFFSET: usize = NODE_HEADER_SIZE;

/// Last byte index (exclusive) of the in-node payload. Beyond this
/// lies the page trailer.
const PAYLOAD_END: usize = PAGE_SIZE - PAGE_TRAILER_SIZE;

/// Maximum number of slots a leaf can hold at the worst-case empty
/// key/value sizes. Used to size the in-memory slot vector.
pub const LEAF_SLOT_CAP: usize = (PAYLOAD_END - PAYLOAD_OFFSET) / LEAF_SLOT_BYTES;

/// Maximum number of slots an internal node can hold at the worst
/// case. Used to size the in-memory slot vector.
pub const INTERNAL_SLOT_CAP: usize =
    (PAYLOAD_END - PAYLOAD_OFFSET - INTERNAL_LEFTMOST_CHILD_BYTES) / INTERNAL_SLOT_BYTES;

/// Maximum key length the format permits.
#[must_use]
pub const fn max_key_len() -> usize {
    PAGE_SIZE / 4
}

/// Maximum value length that still fits inline in a leaf alongside
/// at least one slot.
#[must_use]
pub const fn max_inline_value(key_len: usize) -> usize {
    let base = PAYLOAD_END - PAYLOAD_OFFSET - LEAF_SLOT_BYTES;
    let varints = 9 + 9;
    if key_len + varints >= base {
        0
    } else {
        base - key_len - varints
    }
}

/// Whether a node is an internal node or a leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Internal node — pivots + child pointers, no user values.
    Internal,
    /// Leaf node — `(key, value)` slots.
    Leaf,
}

/// One leaf slot in the decoded in-memory representation.
#[derive(Debug, Clone)]
pub struct LeafEntry {
    /// The key bytes.
    pub key: Vec<u8>,
    /// The value bytes.
    pub value: Vec<u8>,
}

/// One internal-node pivot entry in the decoded in-memory
/// representation. The corresponding right child is stored in
/// [`DecodedNode::children`] at index `i + 1`.
#[derive(Debug, Clone)]
pub struct InternalEntry {
    /// The pivot key.
    pub key: Vec<u8>,
}

/// Alias for "one of the right-child page-ids in an internal node".
/// We store children as raw `u64`s; the on-disk format permits `0`
/// only as the leaf `next_sibling` sentinel, never as an internal
/// child pointer.
pub type ChildEntry = u64;

/// Decoded in-memory representation of a B+tree node.
///
/// One [`DecodedNode`] is built per `Pager::read_page` of a B+tree
/// page; lifetime is "until the caller is done inspecting the
/// node." See the module-level decoded-node allocation note for why
/// we use `Vec` rather than `heapless::Vec` for the slot collections.
#[derive(Debug, Clone)]
pub struct DecodedNode {
    /// Node kind (internal or leaf).
    pub kind: NodeKind,
    /// Tree level. Leaves are level 0; internals are level ≥ 1.
    pub level: u8,
    /// Right-sibling leaf page-id (leaves only; `0` on the last leaf
    /// and on internal nodes).
    pub next_sibling: u64,
    /// Child page-ids (internal nodes only). Length is
    /// `internals.len() + 1` when the node is internal, `0` for
    /// leaves.
    pub children: Vec<ChildEntry>,
    /// Leaf slots (leaf nodes only).
    pub leaves: Vec<LeafEntry>,
    /// Internal-node pivots (internal nodes only).
    pub internals: Vec<InternalEntry>,
}

impl DecodedNode {
    /// Number of bytes the slot directory + key/value heap currently
    /// occupy in a freshly-encoded page. Used by the insert path to
    /// decide whether a node has room for one more entry.
    #[must_use]
    pub fn occupied_bytes(&self) -> usize {
        match self.kind {
            NodeKind::Leaf => {
                let slot_bytes = self.leaves.len() * LEAF_SLOT_BYTES;
                let heap_bytes: usize = self
                    .leaves
                    .iter()
                    .map(|e| {
                        varint_len(u64_from_usize(e.key.len()))
                            + e.key.len()
                            + varint_len(u64_from_usize(e.value.len()))
                            + e.value.len()
                    })
                    .sum();
                slot_bytes + heap_bytes
            }
            NodeKind::Internal => {
                let slot_bytes = self.internals.len() * INTERNAL_SLOT_BYTES;
                let heap_bytes: usize = self
                    .internals
                    .iter()
                    .map(|e| varint_len(u64_from_usize(e.key.len())) + e.key.len())
                    .sum();
                INTERNAL_LEFTMOST_CHILD_BYTES + slot_bytes + heap_bytes
            }
        }
    }

    /// Available payload bytes a fresh encode would still have.
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        let cap = PAYLOAD_END - PAYLOAD_OFFSET;
        cap.saturating_sub(self.occupied_bytes())
    }
}

/// Encode `node` into `page`. Stamps the page trailer.
///
/// # Errors
///
/// Returns [`Error::BTreeInvariantViolated`] if the node's slot count
/// or key/value sizes do not fit in a single page.
pub fn encode_node(node: &DecodedNode, page: &mut Page) -> Result<()> {
    validate_node_release(node)?;
    let buf = page.as_bytes_mut();
    buf.fill(0);
    write_node_header(buf, node);
    match node.kind {
        NodeKind::Leaf => encode_leaf_body(buf, node)?,
        NodeKind::Internal => encode_internal_body(buf, node)?,
    }
    crate::pager::checksum::write_page_trailer(page);
    Ok(())
}

fn write_node_header(buf: &mut [u8; PAGE_SIZE], node: &DecodedNode) {
    let page_type = match node.kind {
        NodeKind::Leaf => PAGE_TYPE_BTREE_LEAF,
        NodeKind::Internal => PAGE_TYPE_BTREE_INTERNAL,
    };
    buf[OFF_PAGE_TYPE] = page_type;
    buf[OFF_LEVEL] = node.level;
    let key_count = match node.kind {
        NodeKind::Leaf => node.leaves.len(),
        NodeKind::Internal => node.internals.len(),
    };
    let kc16 = u16_from_usize(key_count);
    buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&kc16.to_le_bytes());
    let heap_bytes: usize = match node.kind {
        NodeKind::Leaf => node
            .leaves
            .iter()
            .map(|e| {
                varint_len(u64_from_usize(e.key.len()))
                    + e.key.len()
                    + varint_len(u64_from_usize(e.value.len()))
                    + e.value.len()
            })
            .sum(),
        NodeKind::Internal => node
            .internals
            .iter()
            .map(|e| varint_len(u64_from_usize(e.key.len())) + e.key.len())
            .sum(),
    };
    let fso = u32_from_usize(PAYLOAD_END.saturating_sub(heap_bytes));
    buf[OFF_FREE_SPACE_OFF..OFF_FREE_SPACE_OFF + 4].copy_from_slice(&fso.to_le_bytes());
    buf[OFF_NEXT_SIBLING..OFF_NEXT_SIBLING + 8].copy_from_slice(&node.next_sibling.to_le_bytes());
    let _ = OFF_RESERVED;
}

fn encode_leaf_body(buf: &mut [u8; PAGE_SIZE], node: &DecodedNode) -> Result<()> {
    let mut heap_cursor = PAYLOAD_END;
    let mut slot_off = PAYLOAD_OFFSET;
    for entry in &node.leaves {
        let entry_len = varint_len(u64_from_usize(entry.key.len()))
            + entry.key.len()
            + varint_len(u64_from_usize(entry.value.len()))
            + entry.value.len();
        if heap_cursor < entry_len + slot_off + LEAF_SLOT_BYTES {
            return Err(Error::BTreeInvariantViolated {
                reason: "leaf encode: slot dir and heap collide",
            });
        }
        heap_cursor -= entry_len;
        let mut cur = heap_cursor;
        let kl = u64_from_usize(entry.key.len());
        cur += write_varint(&mut buf[cur..], kl);
        buf[cur..cur + entry.key.len()].copy_from_slice(&entry.key);
        cur += entry.key.len();
        let vl = u64_from_usize(entry.value.len());
        cur += write_varint(&mut buf[cur..], vl);
        buf[cur..cur + entry.value.len()].copy_from_slice(&entry.value);
        let off32 = u32_from_usize(heap_cursor);
        buf[slot_off..slot_off + LEAF_SLOT_BYTES].copy_from_slice(&off32.to_le_bytes());
        slot_off += LEAF_SLOT_BYTES;
    }
    Ok(())
}

fn encode_internal_body(buf: &mut [u8; PAGE_SIZE], node: &DecodedNode) -> Result<()> {
    if node.children.len() != node.internals.len() + 1 {
        return Err(Error::BTreeInvariantViolated {
            reason: "internal node has children.len() != pivots+1",
        });
    }
    let leftmost = node.children[0];
    buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + INTERNAL_LEFTMOST_CHILD_BYTES]
        .copy_from_slice(&leftmost.to_le_bytes());
    let mut heap_cursor = PAYLOAD_END;
    let mut slot_off = PAYLOAD_OFFSET + INTERNAL_LEFTMOST_CHILD_BYTES;
    for (i, pivot) in node.internals.iter().enumerate() {
        let right_child = node.children[i + 1];
        let entry_len = varint_len(u64_from_usize(pivot.key.len())) + pivot.key.len();
        if heap_cursor < entry_len + slot_off + INTERNAL_SLOT_BYTES {
            return Err(Error::BTreeInvariantViolated {
                reason: "internal encode: slot dir and heap collide",
            });
        }
        heap_cursor -= entry_len;
        let mut cur = heap_cursor;
        cur += write_varint(&mut buf[cur..], u64_from_usize(pivot.key.len()));
        buf[cur..cur + pivot.key.len()].copy_from_slice(&pivot.key);
        let off32 = u32_from_usize(heap_cursor);
        buf[slot_off..slot_off + 4].copy_from_slice(&off32.to_le_bytes());
        buf[slot_off + 4..slot_off + INTERNAL_SLOT_BYTES]
            .copy_from_slice(&right_child.to_le_bytes());
        slot_off += INTERNAL_SLOT_BYTES;
    }
    Ok(())
}

/// Decode a B+tree page into a [`DecodedNode`].
///
/// # Errors
///
/// Returns [`Error::Corruption`] if any field is malformed: bad
/// page-type tag, key-count exceeding the slot cap, slot offset
/// outside the page, varint length running past the page end, or
/// keys not in ascending order.
///
/// # Perf note
///
/// The three `Vec`s on [`DecodedNode`] (`children`, `leaves`,
/// `internals`) stay heap-allocated rather than `heapless::Vec`-
/// backed. See the module-level decoded-node allocation note for the
/// full allocator-vs-stack-frame analysis; in summary, the outer-vec
/// cost is one allocation per leaf / two per internal — dwarfed
/// by the 2N inner `Vec<u8>` allocations for the entry keys and
/// values. The inner allocations are the real hot-path tax and
/// require a borrowed-slice refactor of the node API to remove.
pub fn decode_node(buf: &[u8; PAGE_SIZE]) -> Result<DecodedNode> {
    let page_type = buf[OFF_PAGE_TYPE];
    let kind = match page_type {
        PAGE_TYPE_BTREE_LEAF => NodeKind::Leaf,
        PAGE_TYPE_BTREE_INTERNAL => NodeKind::Internal,
        _ => return Err(Error::Corruption { page_id: 0 }),
    };
    let level = buf[OFF_LEVEL];
    let key_count = u16::from_le_bytes([buf[OFF_KEY_COUNT], buf[OFF_KEY_COUNT + 1]]) as usize;
    let next_sibling = u64::from_le_bytes([
        buf[OFF_NEXT_SIBLING],
        buf[OFF_NEXT_SIBLING + 1],
        buf[OFF_NEXT_SIBLING + 2],
        buf[OFF_NEXT_SIBLING + 3],
        buf[OFF_NEXT_SIBLING + 4],
        buf[OFF_NEXT_SIBLING + 5],
        buf[OFF_NEXT_SIBLING + 6],
        buf[OFF_NEXT_SIBLING + 7],
    ]);
    let mut node = DecodedNode {
        kind,
        level,
        next_sibling,
        children: Vec::new(),
        leaves: Vec::new(),
        internals: Vec::new(),
    };
    match kind {
        NodeKind::Leaf => decode_leaf_body(buf, &mut node, key_count)?,
        NodeKind::Internal => decode_internal_body(buf, &mut node, key_count)?,
    }
    validate_node_release(&node)?;
    Ok(node)
}

/// Read only the node-kind tag from a B+tree page header.
///
/// O(1), allocation-free. Used by the snapshot read path to decide
/// whether to descend (full [`decode_node`] for an internal node, so
/// `choose_child` has the pivots) or to copy out a single value via
/// [`decode_node_find`] (a leaf). A malformed page-type tag surfaces
/// as [`Error::Corruption`] identically to [`decode_node`].
///
/// # Errors
///
/// [`Error::Corruption`] if the page-type tag is neither leaf nor
/// internal.
pub fn peek_node_kind(buf: &[u8; PAGE_SIZE]) -> Result<NodeKind> {
    match buf[OFF_PAGE_TYPE] {
        PAGE_TYPE_BTREE_LEAF => Ok(NodeKind::Leaf),
        PAGE_TYPE_BTREE_INTERNAL => Ok(NodeKind::Internal),
        _ => Err(Error::Corruption { page_id: 0 }),
    }
}

/// Find-only leaf decode: scan a leaf page for `target` and copy out
/// ONLY the matched value as an owned `Vec<u8>` — never the whole
/// 2N-entry node.
///
/// This is the read-path fast lane for [`crate::btree::BTree::get`] /
/// `get_via_snapshot`. The page must be a leaf; an internal-node tag
/// is rejected as [`Error::Corruption`] (a read never calls this on an
/// internal node — `get_via_snapshot` gates on [`peek_node_kind`] and
/// `BTree::get` descends to a leaf first).
///
/// # Safety / corruption posture
///
/// This path keeps EVERY per-slot integrity check that [`decode_node`]
/// applies to the slots it actually reads: the leaf header invariants
/// (`level == 0`, `key_count <= LEAF_SLOT_CAP`), the slot-offset range
/// check, `read_length_prefixed`'s `checked_add` overflow guard and
/// payload-end bound, the `key.len() <= max_key_len()` cap, and the
/// inline strictly-ascending-key check. A tampered page therefore
/// yields [`Error::Corruption`] identically to the full decode. The
/// ONLY check it skips is the whole-node `windows(2)` ordering
/// re-walk in `validate_node_release`, which is redundant because
/// the inline ascending check is re-derived on every key the scan
/// compares. No data-derived index touches `buf[i]` raw — every such
/// access goes through `read_length_prefixed`'s checked slicing.
///
/// # Errors
///
/// [`Error::Corruption`] on any malformed header, slot, or varint;
/// [`Error::BTreeInvariantViolated`] if the leaf header invariants
/// fail.
pub fn decode_node_find(buf: &[u8; PAGE_SIZE], target: &[u8]) -> Result<Option<Vec<u8>>> {
    match peek_node_kind(buf)? {
        NodeKind::Leaf => {}
        NodeKind::Internal => {
            return Err(Error::BTreeInvariantViolated {
                reason: "decode_node_find called on an internal node",
            });
        }
    }
    if buf[OFF_LEVEL] != 0 {
        return Err(Error::BTreeInvariantViolated {
            reason: "leaf has non-zero level",
        });
    }
    let key_count = u16::from_le_bytes([buf[OFF_KEY_COUNT], buf[OFF_KEY_COUNT + 1]]) as usize;
    if key_count > LEAF_SLOT_CAP {
        return Err(Error::Corruption { page_id: 0 });
    }
    find_leaf_value(buf, key_count, target)
}

/// Linear scan of a validated-header leaf for `target`. Applies the
/// identical per-slot bounds / varint / ascending-key checks as
/// [`decode_leaf_body`] on every slot it visits, comparing keys as
/// borrowed `&[u8]` slices into `buf` — no per-entry `to_vec`.
fn find_leaf_value(
    buf: &[u8; PAGE_SIZE],
    key_count: usize,
    target: &[u8],
) -> Result<Option<Vec<u8>>> {
    let mut prev_key: Option<&[u8]> = None;
    for i in 0..key_count {
        let slot_off = PAYLOAD_OFFSET + i * LEAF_SLOT_BYTES;
        if slot_off + LEAF_SLOT_BYTES > PAYLOAD_END {
            return Err(Error::Corruption { page_id: 0 });
        }
        let entry_off = u32::from_le_bytes([
            buf[slot_off],
            buf[slot_off + 1],
            buf[slot_off + 2],
            buf[slot_off + 3],
        ]) as usize;
        if !(PAYLOAD_OFFSET..PAYLOAD_END).contains(&entry_off) {
            return Err(Error::Corruption { page_id: 0 });
        }
        let (key, after_key) = read_length_prefixed(buf, entry_off)?;
        let (value, _) = read_length_prefixed(buf, after_key)?;
        if key.len() > max_key_len() {
            return Err(Error::Corruption { page_id: 0 });
        }
        if let Some(prev) = prev_key {
            if key <= prev {
                return Err(Error::Corruption { page_id: 0 });
            }
        }
        if key == target {
            return Ok(Some(value.to_vec()));
        }
        prev_key = Some(key);
    }
    Ok(None)
}

/// A borrowed, allocation-free view of a leaf B+tree page.
///
/// `BorrowedLeaf` is the read-path counterpart to [`DecodedNode`] for
/// leaves: it validates the leaf header and every slot up front (the
/// identical per-slot bounds / varint / over-long-key / strictly-
/// ascending checks `decode_leaf_body` applies), but it does **not**
/// copy any key or value into a `Vec`. Each entry is materialized on
/// demand as a `(&[u8], &[u8])` pair borrowing directly into the page
/// bytes.
///
/// # Lifetime safety
///
/// The borrow `'a` ties every entry returned by [`BorrowedLeaf::entry`]
/// / [`BorrowedLeaf::key`] to the page bytes the view was built from.
/// Because callers keep the backing page handle (`PageRef` /
/// `PageHandle` / `Arc<Page>`) alive for at least `'a`, no borrowed
/// entry can outlive the page it points into — the borrow checker
/// rejects any attempt to drop the page while a `BorrowedLeaf` (or any
/// slice it handed out) is still live.
///
/// This replaces the leaf-decode allocation on read-only scan paths:
/// a full [`decode_node`] of an N-entry leaf performs 2N inner
/// `Vec<u8>` allocations (one per key, one per value) plus the outer
/// `leaves` spine; building a `BorrowedLeaf` allocates nothing.
///
/// # Retained surface
///
/// This is the validated read-path leaf view used by the snapshot
/// range scan: `new`/`len`/`upper_bound` are its public entry points
/// (called from `range.rs`), with `entry`/`key` as the internal slot
/// accessors backing `upper_bound`. `is_empty` is present only to
/// satisfy the `clippy::len_without_is_empty` lint required because
/// `len` is public.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedLeaf<'a> {
    buf: &'a [u8; PAGE_SIZE],
    key_count: usize,
    /// Right-sibling leaf page-id (`0` on the last leaf). Mirrors
    /// [`DecodedNode::next_sibling`].
    pub next_sibling: u64,
}

impl<'a> BorrowedLeaf<'a> {
    /// Validate `buf` as a leaf page and return a borrowed view.
    ///
    /// Applies the same header and per-slot integrity checks as
    /// [`decode_node`]'s leaf path — bad page-type tag, non-zero level,
    /// key-count over the slot cap, slot offset out of range, varint
    /// overrun, over-long key, and the strictly-ascending key
    /// ordering — so a tampered page surfaces as [`Error::Corruption`]
    /// / [`Error::BTreeInvariantViolated`] identically to the owned
    /// decode. Validating once up front lets [`Self::entry`] read a
    /// slot with no per-call ordering re-derivation.
    ///
    /// # Errors
    ///
    /// [`Error::BTreeInvariantViolated`] if the page is not a leaf or
    /// its header is malformed; [`Error::Corruption`] on any malformed
    /// slot, varint, over-long key, or out-of-order key.
    pub fn new(buf: &'a [u8; PAGE_SIZE]) -> Result<Self> {
        match peek_node_kind(buf)? {
            NodeKind::Leaf => {}
            NodeKind::Internal => {
                return Err(Error::BTreeInvariantViolated {
                    reason: "BorrowedLeaf::new called on an internal node",
                });
            }
        }
        if buf[OFF_LEVEL] != 0 {
            return Err(Error::BTreeInvariantViolated {
                reason: "leaf has non-zero level",
            });
        }
        let key_count = u16::from_le_bytes([buf[OFF_KEY_COUNT], buf[OFF_KEY_COUNT + 1]]) as usize;
        if key_count > LEAF_SLOT_CAP {
            return Err(Error::Corruption { page_id: 0 });
        }
        let next_sibling = read_u64(buf, OFF_NEXT_SIBLING);
        validate_borrowed_leaf_slots(buf, key_count)?;
        Ok(Self {
            buf,
            key_count,
            next_sibling,
        })
    }

    /// Number of `(key, value)` entries in the leaf.
    #[must_use]
    pub fn len(&self) -> usize {
        self.key_count
    }

    /// Whether the leaf holds zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.key_count == 0
    }

    /// Borrow the `(key, value)` slices of the `i`-th entry.
    ///
    /// Returns `None` if `i >= len()`. Both slices borrow into the page
    /// bytes for the lifetime `'a`. The slot was bounds- and
    /// varint-validated at [`Self::new`], so this is an infallible
    /// re-read.
    #[must_use]
    pub fn entry(&self, i: usize) -> Option<(&'a [u8], &'a [u8])> {
        if i >= self.key_count {
            return None;
        }
        read_leaf_slot(self.buf, i).ok()
    }

    /// Borrow just the key slice of the `i`-th entry.
    #[must_use]
    pub fn key(&self, i: usize) -> Option<&'a [u8]> {
        self.entry(i).map(|(k, _)| k)
    }

    /// First slot index whose key is strictly `> target`, or
    /// [`Self::len`] if none. Linear scan.
    #[must_use]
    pub fn upper_bound(&self, target: &[u8]) -> usize {
        for i in 0..self.key_count {
            if let Some(k) = self.key(i) {
                if k > target {
                    return i;
                }
            }
        }
        self.key_count
    }
}

/// Read the `i`-th leaf slot of `buf` as borrowed `(key, value)`
/// slices. Applies the identical per-slot bounds / varint / over-long-
/// key checks as `decode_leaf_body`; does NOT re-check ordering
/// (the caller validated the whole leaf once via
/// `validate_borrowed_leaf_slots`).
///
/// Public so that read-path iterators can hold a validated leaf's
/// backing page bytes and re-read a single entry per step without
/// rebuilding (and re-validating) a whole [`BorrowedLeaf`]. The per-
/// slot bounds checks make a stale-index call safe — it returns
/// [`Error::Corruption`] rather than indexing out of range.
///
/// # Errors
///
/// [`Error::Corruption`] on a malformed slot offset, varint, or
/// over-long key.
pub fn read_leaf_slot(buf: &[u8; PAGE_SIZE], i: usize) -> Result<(&[u8], &[u8])> {
    let slot_off = PAYLOAD_OFFSET + i * LEAF_SLOT_BYTES;
    if slot_off + LEAF_SLOT_BYTES > PAYLOAD_END {
        return Err(Error::Corruption { page_id: 0 });
    }
    let entry_off = u32::from_le_bytes([
        buf[slot_off],
        buf[slot_off + 1],
        buf[slot_off + 2],
        buf[slot_off + 3],
    ]) as usize;
    if !(PAYLOAD_OFFSET..PAYLOAD_END).contains(&entry_off) {
        return Err(Error::Corruption { page_id: 0 });
    }
    let (key, after_key) = read_length_prefixed(buf, entry_off)?;
    let (value, _) = read_length_prefixed(buf, after_key)?;
    if key.len() > max_key_len() {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok((key, value))
}

/// Validate every slot of a leaf body, enforcing the strictly-
/// ascending key invariant inline. Run once at [`BorrowedLeaf::new`]
/// so that later [`read_leaf_slot`] re-reads need no ordering check.
fn validate_borrowed_leaf_slots(buf: &[u8; PAGE_SIZE], key_count: usize) -> Result<()> {
    let mut prev_key: Option<&[u8]> = None;
    for i in 0..key_count {
        let (key, _value) = read_leaf_slot(buf, i)?;
        if let Some(prev) = prev_key {
            if key <= prev {
                return Err(Error::Corruption { page_id: 0 });
            }
        }
        prev_key = Some(key);
    }
    Ok(())
}

fn decode_leaf_body(buf: &[u8; PAGE_SIZE], node: &mut DecodedNode, key_count: usize) -> Result<()> {
    if key_count > LEAF_SLOT_CAP {
        return Err(Error::Corruption { page_id: 0 });
    }
    node.leaves.reserve(key_count);
    for i in 0..key_count {
        let (key, value) = read_leaf_slot(buf, i)?;
        if let Some(prev) = node.leaves.last() {
            if key <= prev.key.as_slice() {
                return Err(Error::Corruption { page_id: 0 });
            }
        }
        node.leaves.push(LeafEntry {
            key: key.to_vec(),
            value: value.to_vec(),
        });
    }
    Ok(())
}

fn decode_internal_body(
    buf: &[u8; PAGE_SIZE],
    node: &mut DecodedNode,
    key_count: usize,
) -> Result<()> {
    if key_count > INTERNAL_SLOT_CAP {
        return Err(Error::Corruption { page_id: 0 });
    }
    let leftmost = read_u64(buf, PAYLOAD_OFFSET);
    if leftmost == 0 {
        return Err(Error::Corruption { page_id: 0 });
    }
    node.children.reserve(key_count + 1);
    node.internals.reserve(key_count);
    node.children.push(leftmost);
    for i in 0..key_count {
        let (key_vec, right_child) = decode_internal_slot(buf, i)?;
        if let Some(prev) = node.internals.last() {
            if key_vec.as_slice() <= prev.key.as_slice() {
                return Err(Error::Corruption { page_id: 0 });
            }
        }
        node.internals.push(InternalEntry { key: key_vec });
        node.children.push(right_child);
    }
    Ok(())
}

/// Decode the i-th slot of an internal-node body. Returns the pivot
/// key and the right-child page-id, or `Error::Corruption` if the
/// slot is malformed.
fn decode_internal_slot(buf: &[u8; PAGE_SIZE], i: usize) -> Result<(Vec<u8>, u64)> {
    let slot_off = PAYLOAD_OFFSET + INTERNAL_LEFTMOST_CHILD_BYTES + i * INTERNAL_SLOT_BYTES;
    if slot_off + INTERNAL_SLOT_BYTES > PAYLOAD_END {
        return Err(Error::Corruption { page_id: 0 });
    }
    let entry_off = u32::from_le_bytes([
        buf[slot_off],
        buf[slot_off + 1],
        buf[slot_off + 2],
        buf[slot_off + 3],
    ]) as usize;
    let right_child = read_u64(buf, slot_off + 4);
    if right_child == 0 || !(PAYLOAD_OFFSET..PAYLOAD_END).contains(&entry_off) {
        return Err(Error::Corruption { page_id: 0 });
    }
    let (key, _) = read_length_prefixed(buf, entry_off)?;
    if key.len() > max_key_len() {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok((key.to_vec(), right_child))
}

/// Read a u64 LE from `buf` at `offset`. Panics if `offset + 8` is
/// out of bounds; callers must validate the offset.
fn read_u64(buf: &[u8; PAGE_SIZE], offset: usize) -> u64 {
    u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

/// Read a `(length-prefixed bytes, next_offset)` pair from `buf`
/// starting at `offset`. Length is an unsigned LEB128 varint; the
/// payload is `length` raw bytes.
///
/// # Overflow posture
///
/// `len` is caller-controlled (it lives on disk in a page byte that
/// can be tampered with or random in a fuzz harness). The
/// `after + len_usize` end-of-slice computation MUST use
/// `checked_add`: with `overflow-checks = true` in release a raw `+`
/// panics on a wrap; without overflow-checks the bounds check is
/// silently bypassed. Either failure mode is a denial-of-service
/// hazard reachable from
/// `Pager::read_page → BTree::open → decode_node` on any tampered
/// B+tree page.
fn read_length_prefixed(buf: &[u8; PAGE_SIZE], offset: usize) -> Result<(&[u8], usize)> {
    if offset >= PAYLOAD_END {
        return Err(Error::Corruption { page_id: 0 });
    }
    let (len, after) = read_varint(buf, offset)?;
    let len_usize = usize_from_u64(len)?;
    let end = after
        .checked_add(len_usize)
        .ok_or(Error::Corruption { page_id: 0 })?;
    if end > PAYLOAD_END {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok((&buf[after..end], end))
}

/// Validate `node` against the format-version-0 invariants.
///
/// # Errors
///
/// Returns [`Error::BTreeInvariantViolated`] if any invariant fails.
/// Callers can `debug_assert!` on a wrapping helper for development
/// builds; release builds get the error surfaced cleanly.
pub fn validate_node(node: &DecodedNode) -> Result<()> {
    validate_node_release(node)
}

fn validate_node_release(node: &DecodedNode) -> Result<()> {
    validate_node_structural(node)?;
    validate_node_ordering(node)
}

/// The cheap O(1) header / capacity / children invariants that the
/// inline decode loop does *not* enforce: leaf `level == 0`, internal
/// `level != 0`, internal `next_sibling == 0`,
/// `children.len() == pivots + 1`, slot-cap bounds, and non-zero
/// internal child page-ids.
///
/// This is the only part of the full validate that the read find path
/// needs: the strictly-ascending key ordering ([`validate_node_ordering`])
/// is enforced inline during decode, so the find path — which already
/// re-derives the same ordering on every slot it touches — does not
/// repeat the whole-node `windows(2)` re-walk.
fn validate_node_structural(node: &DecodedNode) -> Result<()> {
    match node.kind {
        NodeKind::Leaf => {
            if node.level != 0 {
                return Err(Error::BTreeInvariantViolated {
                    reason: "leaf has non-zero level",
                });
            }
            if node.leaves.len() > LEAF_SLOT_CAP {
                return Err(Error::BTreeInvariantViolated {
                    reason: "leaf key count exceeds slot cap",
                });
            }
        }
        NodeKind::Internal => {
            if node.level == 0 {
                return Err(Error::BTreeInvariantViolated {
                    reason: "internal node at level 0",
                });
            }
            if node.internals.len() > INTERNAL_SLOT_CAP {
                return Err(Error::BTreeInvariantViolated {
                    reason: "internal key count exceeds slot cap",
                });
            }
            if node.children.len() != node.internals.len() + 1 {
                return Err(Error::BTreeInvariantViolated {
                    reason: "internal children.len() != pivots+1",
                });
            }
            for c in &node.children {
                if *c == 0 {
                    return Err(Error::BTreeInvariantViolated {
                        reason: "internal node has zero child page-id",
                    });
                }
            }
            if node.next_sibling != 0 {
                return Err(Error::BTreeInvariantViolated {
                    reason: "internal node has non-zero next_sibling",
                });
            }
        }
    }
    Ok(())
}

/// The strictly-ascending key-ordering re-walk. On the full decode and
/// every mutation/integrity caller this confirms the whole node is
/// sorted; the read find path skips it because [`decode_leaf_body`] /
/// [`decode_internal_body`] enforce the identical inline check on every
/// key as it is pushed (node.rs leaf/internal `last()` compares).
fn validate_node_ordering(node: &DecodedNode) -> Result<()> {
    match node.kind {
        NodeKind::Leaf => {
            for w in node.leaves.windows(2) {
                if w[0].key.as_slice() >= w[1].key.as_slice() {
                    return Err(Error::BTreeInvariantViolated {
                        reason: "leaf keys not strictly sorted",
                    });
                }
            }
        }
        NodeKind::Internal => {
            for w in node.internals.windows(2) {
                if w[0].key.as_slice() >= w[1].key.as_slice() {
                    return Err(Error::BTreeInvariantViolated {
                        reason: "internal keys not strictly sorted",
                    });
                }
            }
        }
    }
    Ok(())
}

/// Number of bytes an unsigned LEB128 varint would use for `v`.
#[must_use]
pub fn varint_len(v: u64) -> usize {
    let mut n: usize = 1;
    let mut x = v >> 7;
    while x != 0 {
        n += 1;
        x >>= 7;
    }
    n
}

/// Write an unsigned LEB128 varint into `dst` starting at byte 0.
/// Returns the number of bytes written.
fn write_varint(dst: &mut [u8], mut v: u64) -> usize {
    let mut i = 0;
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
            dst[i] = byte;
            i += 1;
        } else {
            dst[i] = byte;
            i += 1;
            return i;
        }
    }
}

/// Read an unsigned LEB128 varint from `buf` starting at `offset`.
/// Returns `(value, next_offset)`. The loop is bounded by 10 bytes
/// (`ceil(64 / 7)`), which is the maximum a 64-bit varint can occupy.
fn read_varint(buf: &[u8; PAGE_SIZE], offset: usize) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = offset;
    for _ in 0..10 {
        if i >= PAYLOAD_END {
            return Err(Error::Corruption { page_id: 0 });
        }
        let byte = buf[i];
        i += 1;
        let chunk = u64::from(byte & 0x7F);
        value |= chunk << shift;
        if (byte & 0x80) == 0 {
            return Ok((value, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::Corruption { page_id: 0 });
        }
    }
    Err(Error::Corruption { page_id: 0 })
}

fn u32_from_usize(v: usize) -> u32 {
    debug_assert!(u32::try_from(v).is_ok());
    u32::try_from(v).unwrap_or(u32::MAX)
}

fn u16_from_usize(v: usize) -> u16 {
    debug_assert!(u16::try_from(v).is_ok());
    u16::try_from(v).unwrap_or(u16::MAX)
}

fn u64_from_usize(v: usize) -> u64 {
    v as u64
}

fn usize_from_u64(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| Error::Corruption { page_id: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_leaf() -> DecodedNode {
        DecodedNode {
            kind: NodeKind::Leaf,
            level: 0,
            next_sibling: 0,
            children: Vec::new(),
            leaves: Vec::new(),
            internals: Vec::new(),
        }
    }

    fn leaf_with(entries: &[(&[u8], &[u8])]) -> DecodedNode {
        let mut leaf = empty_leaf();
        for (k, v) in entries {
            leaf.leaves.push(LeafEntry {
                key: k.to_vec(),
                value: v.to_vec(),
            });
        }
        leaf
    }

    #[test]
    fn round_trip_empty_leaf() {
        let leaf = empty_leaf();
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let decoded = decode_node(page.as_bytes()).expect("decode");
        assert_eq!(decoded.kind, NodeKind::Leaf);
        assert_eq!(decoded.level, 0);
        assert_eq!(decoded.next_sibling, 0);
        assert!(decoded.leaves.is_empty());
    }

    #[test]
    fn round_trip_populated_leaf() {
        let leaf = leaf_with(&[(b"alpha", b"AAA"), (b"bravo", b"BBB"), (b"charlie", b"CCC")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let decoded = decode_node(page.as_bytes()).expect("decode");
        assert_eq!(decoded.leaves.len(), 3);
        assert_eq!(decoded.leaves[0].key.as_slice(), b"alpha");
        assert_eq!(decoded.leaves[0].value.as_slice(), b"AAA");
        assert_eq!(decoded.leaves[2].key.as_slice(), b"charlie");
    }

    #[test]
    fn round_trip_internal() {
        let mut internal = DecodedNode {
            kind: NodeKind::Internal,
            level: 1,
            next_sibling: 0,
            children: Vec::new(),
            leaves: Vec::new(),
            internals: Vec::new(),
        };
        for raw in [10u64, 20, 30, 40] {
            internal.children.push(raw);
        }
        for k in [b"d".as_slice(), b"h", b"m"] {
            internal.internals.push(InternalEntry { key: k.to_vec() });
        }
        let mut page = Page::zeroed();
        encode_node(&internal, &mut page).expect("encode");
        let decoded = decode_node(page.as_bytes()).expect("decode");
        assert_eq!(decoded.kind, NodeKind::Internal);
        assert_eq!(decoded.level, 1);
        assert_eq!(decoded.internals.len(), 3);
        assert_eq!(decoded.children.as_slice(), &[10, 20, 30, 40]);
        assert_eq!(decoded.internals[0].key.as_slice(), b"d");
        assert_eq!(decoded.internals[2].key.as_slice(), b"m");
    }

    #[test]
    fn decode_rejects_unsorted_leaf() {
        let mut page = Page::zeroed();
        let buf = page.as_bytes_mut();
        buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_LEAF;
        buf[OFF_LEVEL] = 0;
        buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&2u16.to_le_bytes());
        let slot0 = u32_from_usize(PAYLOAD_END - 4);
        let slot1 = u32_from_usize(PAYLOAD_END - 8);
        buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&slot0.to_le_bytes());
        buf[PAYLOAD_OFFSET + 4..PAYLOAD_OFFSET + 8].copy_from_slice(&slot1.to_le_bytes());
        buf[PAYLOAD_END - 4] = 1;
        buf[PAYLOAD_END - 3] = b'c';
        buf[PAYLOAD_END - 2] = 1;
        buf[PAYLOAD_END - 1] = b'x';
        buf[PAYLOAD_END - 8] = 1;
        buf[PAYLOAD_END - 7] = b'a';
        buf[PAYLOAD_END - 6] = 1;
        buf[PAYLOAD_END - 5] = b'y';
        crate::pager::checksum::write_page_trailer(&mut page);
        let err = decode_node(page.as_bytes()).expect_err("decode must reject");
        assert!(matches!(err, Error::Corruption { .. }));
    }

    #[test]
    fn varint_round_trip() {
        for v in [0u64, 1, 127, 128, 0xFFFF, 0xDEAD_BEEF, u64::MAX] {
            let mut buf = [0u8; 16];
            let n = write_varint(&mut buf, v);
            assert_eq!(n, varint_len(v));
            let mut page = [0u8; PAGE_SIZE];
            page[..n].copy_from_slice(&buf[..n]);
            let (decoded, after) = read_varint(&page, 0).expect("decode varint");
            assert_eq!(decoded, v);
            assert_eq!(after, n);
        }
    }

    /// Regression: a tampered B+tree leaf carrying a length-prefix
    /// varint that encodes `u64::MAX` must return `Error::Corruption`,
    /// NOT panic in `read_length_prefixed` on the `after + len_usize`
    /// overflow.
    ///
    /// Critical: this test must also pass under `cargo test --release`
    /// — the workspace profile enables `overflow-checks = true`, which
    /// is the mechanism that turned the arithmetic wrap into a panic in
    /// the first place. A `checked_add` fix is the only thing that
    /// makes both debug and release happy.
    #[test]
    fn decode_node_varint_max_returns_corruption_not_panic() {
        let mut page = Page::zeroed();
        {
            let buf = page.as_bytes_mut();
            buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_LEAF;
            buf[OFF_LEVEL] = 0;
            buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&1u16.to_le_bytes());
            let entry_off = PAYLOAD_END - 16;
            let slot_off_bytes = u32_from_usize(entry_off).to_le_bytes();
            buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&slot_off_bytes);
            let varint = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
            buf[entry_off..entry_off + varint.len()].copy_from_slice(&varint);
        }
        crate::pager::checksum::write_page_trailer(&mut page);
        let result = decode_node(page.as_bytes());
        match result {
            Err(Error::Corruption { .. }) => {}
            other => panic!(
                "expected Err(Error::Corruption), got {:?} \
                 (a panic here would indicate the M13 #115 bug \
                  is still present in release builds)",
                other.map_or("non-corruption err", |_| "Ok(_)")
            ),
        }
    }

    /// Regression for the internal-node variant: a tampered
    /// internal-node entry pointing at a varint that encodes
    /// `u64::MAX` must surface as `Error::Corruption`, not panic via
    /// `decode_internal_slot → read_length_prefixed`.
    #[test]
    fn decode_internal_node_varint_max_returns_corruption() {
        let mut page = Page::zeroed();
        {
            let buf = page.as_bytes_mut();
            buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_INTERNAL;
            buf[OFF_LEVEL] = 1;
            buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&1u16.to_le_bytes());
            buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 8].copy_from_slice(&42u64.to_le_bytes());
            let slot_base = PAYLOAD_OFFSET + INTERNAL_LEFTMOST_CHILD_BYTES;
            let entry_off = PAYLOAD_END - 16;
            buf[slot_base..slot_base + 4].copy_from_slice(&u32_from_usize(entry_off).to_le_bytes());
            buf[slot_base + 4..slot_base + INTERNAL_SLOT_BYTES]
                .copy_from_slice(&7u64.to_le_bytes());
            let varint = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
            buf[entry_off..entry_off + varint.len()].copy_from_slice(&varint);
        }
        crate::pager::checksum::write_page_trailer(&mut page);
        let result = decode_node(page.as_bytes());
        assert!(
            matches!(result, Err(Error::Corruption { .. })),
            "expected Err(Error::Corruption) on u64::MAX varint in internal slot"
        );
    }

    #[test]
    fn max_inline_value_is_below_payload() {
        let k = 8;
        let v = max_inline_value(k);
        let entry_len = varint_len(u64_from_usize(k)) + k + varint_len(u64_from_usize(v)) + v;
        let payload = entry_len + LEAF_SLOT_BYTES;
        assert!(payload <= PAYLOAD_END - PAYLOAD_OFFSET);
    }

    /// The find-only path returns the matched value and `None` for a
    /// miss, agreeing with a full decode.
    #[test]
    fn decode_node_find_matches_full_decode() {
        let leaf = leaf_with(&[(b"alpha", b"AAA"), (b"bravo", b"BBB"), (b"charlie", b"CCC")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let found = decode_node_find(page.as_bytes(), b"bravo").expect("find");
        assert_eq!(found.as_deref(), Some(b"BBB".as_slice()));
        let missing = decode_node_find(page.as_bytes(), b"zeta").expect("find");
        assert_eq!(missing, None);
        assert_eq!(
            decode_node_find(page.as_bytes(), b"alpha")
                .expect("find")
                .as_deref(),
            Some(b"AAA".as_slice())
        );
        assert_eq!(
            decode_node_find(page.as_bytes(), b"charlie")
                .expect("find")
                .as_deref(),
            Some(b"CCC".as_slice())
        );
    }

    /// Safety regression (i): a tampered slot offset pointing
    /// outside the payload range must surface `Error::Corruption`
    /// THROUGH THE FIND PATH, not only through the full decode. The
    /// search targets `"b"`, which forces the scan to read the damaged
    /// slot 0 first.
    #[test]
    fn decode_node_find_rejects_corrupted_slot_offset() {
        let leaf = leaf_with(&[(b"a", b"x"), (b"b", b"y")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let buf = page.as_bytes_mut();
        let bad = u32_from_usize(PAYLOAD_END + 4);
        buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&bad.to_le_bytes());
        crate::pager::checksum::write_page_trailer(&mut page);
        let r = decode_node_find(page.as_bytes(), b"b");
        assert!(
            matches!(r, Err(Error::Corruption { .. })),
            "corrupted slot offset not caught by find path: {r:?}"
        );
    }

    /// Safety regression (ii): two out-of-order keys must surface
    /// `Error::Corruption` THROUGH THE FIND PATH. Built by hand since
    /// `encode_node` would reject a descending leaf; the search target
    /// `"z"` (a miss) forces the full scan to reach the descending
    /// pair at slot 1.
    #[test]
    fn decode_node_find_rejects_out_of_order_keys() {
        let mut page = Page::zeroed();
        {
            let buf = page.as_bytes_mut();
            buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_LEAF;
            buf[OFF_LEVEL] = 0;
            buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&2u16.to_le_bytes());
            let slot0 = u32_from_usize(PAYLOAD_END - 4);
            let slot1 = u32_from_usize(PAYLOAD_END - 8);
            buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&slot0.to_le_bytes());
            buf[PAYLOAD_OFFSET + 4..PAYLOAD_OFFSET + 8].copy_from_slice(&slot1.to_le_bytes());
            buf[PAYLOAD_END - 4] = 1;
            buf[PAYLOAD_END - 3] = b'c';
            buf[PAYLOAD_END - 2] = 1;
            buf[PAYLOAD_END - 1] = b'x';
            buf[PAYLOAD_END - 8] = 1;
            buf[PAYLOAD_END - 7] = b'a';
            buf[PAYLOAD_END - 6] = 1;
            buf[PAYLOAD_END - 5] = b'y';
        }
        crate::pager::checksum::write_page_trailer(&mut page);
        let r = decode_node_find(page.as_bytes(), b"z");
        assert!(
            matches!(r, Err(Error::Corruption { .. })),
            "out-of-order key not caught by find path: {r:?}"
        );
    }

    /// Safety regression (iii): a key whose length prefix claims
    /// `max_key_len() + 1` bytes (in-bounds for the page but over the
    /// format cap) must surface `Error::Corruption` THROUGH THE FIND
    /// PATH on the only slot it reads.
    #[test]
    fn decode_node_find_rejects_over_long_key() {
        let mut page = Page::zeroed();
        {
            let buf = page.as_bytes_mut();
            buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_LEAF;
            buf[OFF_LEVEL] = 0;
            buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&1u16.to_le_bytes());
            let over = max_key_len() + 1;
            let entry_off = PAYLOAD_OFFSET + LEAF_SLOT_BYTES;
            buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4]
                .copy_from_slice(&u32_from_usize(entry_off).to_le_bytes());
            let n = write_varint(&mut buf[entry_off..], u64_from_usize(over));
            assert_eq!(n, varint_len(u64_from_usize(over)));
        }
        crate::pager::checksum::write_page_trailer(&mut page);
        let r = decode_node_find(page.as_bytes(), b"anything");
        assert!(
            matches!(r, Err(Error::Corruption { .. })),
            "over-long key length not caught by find path: {r:?}"
        );
    }

    /// `BorrowedLeaf` agrees with a full owned decode: same entry
    /// count, and the borrowed `(key, value)` slices equal the owned
    /// `LeafEntry` bytes for every slot.
    #[test]
    fn borrowed_leaf_matches_full_decode() {
        let leaf = leaf_with(&[(b"alpha", b"AAA"), (b"bravo", b"BBB"), (b"charlie", b"CCC")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let owned = decode_node(page.as_bytes()).expect("decode");
        let borrowed = BorrowedLeaf::new(page.as_bytes()).expect("borrowed");
        assert_eq!(borrowed.len(), owned.leaves.len());
        assert!(!borrowed.is_empty());
        for (i, want) in owned.leaves.iter().enumerate() {
            let (k, v) = borrowed.entry(i).expect("entry");
            assert_eq!(k, want.key.as_slice());
            assert_eq!(v, want.value.as_slice());
            assert_eq!(borrowed.key(i), Some(want.key.as_slice()));
        }
        assert_eq!(borrowed.entry(owned.leaves.len()), None);
        assert_eq!(borrowed.next_sibling, owned.next_sibling);
    }

    /// `upper_bound` agrees with a linear `position` over the owned
    /// decode.
    #[test]
    fn borrowed_leaf_upper_bound() {
        let leaf = leaf_with(&[(b"a", b"1"), (b"c", b"2"), (b"e", b"3")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let b = BorrowedLeaf::new(page.as_bytes()).expect("borrowed");
        assert_eq!(b.upper_bound(b"a"), 1);
        assert_eq!(b.upper_bound(b"c"), 2);
        assert_eq!(b.upper_bound(b"e"), 3);
    }

    /// An empty leaf borrows as a zero-length view.
    #[test]
    fn borrowed_leaf_empty() {
        let leaf = empty_leaf();
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let b = BorrowedLeaf::new(page.as_bytes()).expect("borrowed");
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
        assert_eq!(b.entry(0), None);
    }

    /// `BorrowedLeaf::new` rejects an internal-node page, matching the
    /// find-path invariant.
    #[test]
    fn borrowed_leaf_rejects_internal_node() {
        let mut internal = DecodedNode {
            kind: NodeKind::Internal,
            level: 1,
            next_sibling: 0,
            children: Vec::new(),
            leaves: Vec::new(),
            internals: Vec::new(),
        };
        internal.children.push(10);
        internal.children.push(20);
        internal
            .internals
            .push(InternalEntry { key: b"m".to_vec() });
        let mut page = Page::zeroed();
        encode_node(&internal, &mut page).expect("encode");
        let r = BorrowedLeaf::new(page.as_bytes());
        assert!(matches!(r, Err(Error::BTreeInvariantViolated { .. })));
    }

    /// `BorrowedLeaf::new` rejects out-of-order keys, matching the full
    /// decode's `Error::Corruption`.
    #[test]
    fn borrowed_leaf_rejects_out_of_order_keys() {
        let mut page = Page::zeroed();
        {
            let buf = page.as_bytes_mut();
            buf[OFF_PAGE_TYPE] = PAGE_TYPE_BTREE_LEAF;
            buf[OFF_LEVEL] = 0;
            buf[OFF_KEY_COUNT..OFF_KEY_COUNT + 2].copy_from_slice(&2u16.to_le_bytes());
            let slot0 = u32_from_usize(PAYLOAD_END - 4);
            let slot1 = u32_from_usize(PAYLOAD_END - 8);
            buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&slot0.to_le_bytes());
            buf[PAYLOAD_OFFSET + 4..PAYLOAD_OFFSET + 8].copy_from_slice(&slot1.to_le_bytes());
            buf[PAYLOAD_END - 4] = 1;
            buf[PAYLOAD_END - 3] = b'c';
            buf[PAYLOAD_END - 2] = 1;
            buf[PAYLOAD_END - 1] = b'x';
            buf[PAYLOAD_END - 8] = 1;
            buf[PAYLOAD_END - 7] = b'a';
            buf[PAYLOAD_END - 6] = 1;
            buf[PAYLOAD_END - 5] = b'y';
        }
        crate::pager::checksum::write_page_trailer(&mut page);
        let r = BorrowedLeaf::new(page.as_bytes());
        assert!(matches!(r, Err(Error::Corruption { .. })));
    }

    /// `read_leaf_slot` bounds-checks a stale index: a slot offset
    /// outside the payload surfaces `Error::Corruption` rather than
    /// indexing out of range.
    #[test]
    fn read_leaf_slot_rejects_corrupted_offset() {
        let leaf = leaf_with(&[(b"a", b"x"), (b"b", b"y")]);
        let mut page = Page::zeroed();
        encode_node(&leaf, &mut page).expect("encode");
        let buf = page.as_bytes_mut();
        let bad = u32_from_usize(PAYLOAD_END + 4);
        buf[PAYLOAD_OFFSET..PAYLOAD_OFFSET + 4].copy_from_slice(&bad.to_le_bytes());
        crate::pager::checksum::write_page_trailer(&mut page);
        let r = read_leaf_slot(page.as_bytes(), 0);
        assert!(matches!(r, Err(Error::Corruption { .. })));
    }

    /// The find path rejects an internal-node page: a read never calls
    /// it on an internal node, and doing so is an invariant violation.
    #[test]
    fn decode_node_find_rejects_internal_node() {
        let mut internal = DecodedNode {
            kind: NodeKind::Internal,
            level: 1,
            next_sibling: 0,
            children: Vec::new(),
            leaves: Vec::new(),
            internals: Vec::new(),
        };
        internal.children.push(10);
        internal.children.push(20);
        internal
            .internals
            .push(InternalEntry { key: b"m".to_vec() });
        let mut page = Page::zeroed();
        encode_node(&internal, &mut page).expect("encode");
        let r = decode_node_find(page.as_bytes(), b"m");
        assert!(matches!(r, Err(Error::BTreeInvariantViolated { .. })));
    }
}
