//! B+tree — copy-on-write B+tree over the pager.
//!
//! The B+tree is the data structure both the primary store and every
//! secondary index compose over.

#![forbid(unsafe_code)]

pub mod delete;
pub mod insert;
pub mod node;
pub mod range;

pub use crate::btree::range::{RangeIter, SnapshotRangeIter};

use crate::error::{Error, Result};
use crate::pager::page::PageId;
use crate::pager::Pager;
use crate::platform::{FileBackend, FileHandle};

use heapless::Vec as HeaplessVec;

/// Maximum B+tree depth.
///
/// At fanout ≥ 4 a 32-level tree holds 2^64 entries, which exceeds
/// the page-id space. The bound exists so tree traversals always
/// terminate; a tree that somehow grew past 32 levels surfaces as
/// [`Error::BTreeDepthExceeded`] rather than a stack overflow.
pub const MAX_BTREE_DEPTH: usize = 32;

/// Maximum number of B+tree nodes visited in a single range scan.
///
/// Every loop has an explicit upper bound. 1 000 000 nodes × ~32
/// slots/node ≈ 32 M entries — comfortably above the "collection
/// scan" target.
pub const MAX_RANGE_NODES: usize = 1_000_000;

/// Borrowed view of the in-memory representation of a B+tree node.
pub use crate::btree::node::{
    decode_node, encode_node, max_inline_value, max_key_len, read_leaf_slot, validate_node,
    BorrowedLeaf, ChildEntry, DecodedNode, InternalEntry, LeafEntry, NodeKind, NODE_HEADER_SIZE,
};

/// A B+tree handle.
///
/// Construct via [`BTree::empty`] (fresh tree allocating a new root
/// leaf) or [`BTree::open`] (attach to an existing root page-id).
/// The handle does not own the pager; the caller passes a `&mut`
/// reference into mutating methods.
///
/// Generic over `F: FileBackend` so the same code works against the
/// production [`FileHandle`] and the fault-injection harness. The
/// type parameter is used as a marker — it appears in the
/// `Pager<F>` signature of every method.
///
/// Single-writer: every mutating method takes `&mut self`. The pager
/// already enforces single-writer at the file level; the B+tree
/// inherits the same property.
#[derive(Debug)]
pub struct BTree<F: FileBackend = FileHandle> {
    root: PageId,
    _phantom: std::marker::PhantomData<fn() -> F>,
}

impl<F: FileBackend> BTree<F> {
    /// Attach a B+tree handle to an existing root page-id.
    ///
    /// The root page is NOT read here — `BTree::open` is a pure
    /// constructor that records the root id. The first mutating or
    /// reading call will fault the root in via `Pager::read_page`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `root` is zero. (Zero
    /// is the sentinel "no tree"; the caller should use
    /// [`BTree::empty`] to bootstrap.)
    pub fn open(_pager: &Pager<F>, root: PageId) -> Result<Self> {
        Ok(Self {
            root,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Allocate a fresh empty B+tree.
    ///
    /// Allocates one new page through the pager, encodes an empty
    /// leaf node into it, and writes it through the WAL transaction
    /// buffer. The caller is responsible for calling
    /// [`Pager::commit`] before relying on the tree being durable.
    ///
    /// # Errors
    ///
    /// Propagates pager errors from [`Pager::alloc_page`] and
    /// [`Pager::write_page`].
    pub fn empty(pager: &mut Pager<F>) -> Result<Self> {
        let root = pager.alloc_page()?;
        let mut page = crate::pager::page::Page::zeroed();
        encode_node(
            &DecodedNode {
                kind: NodeKind::Leaf,
                level: 0,
                next_sibling: 0,
                children: Vec::new(),
                leaves: Vec::new(),
                internals: Vec::new(),
            },
            &mut page,
        )?;
        pager.write_page(root, &page)?;
        Ok(Self {
            root,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The current root page-id of this tree.
    ///
    /// Mutating methods update this in place after a successful
    /// write. The caller can persist this id externally so that a
    /// subsequent [`BTree::open`] reattaches to the same tree.
    #[must_use]
    pub fn root(&self) -> PageId {
        self.root
    }

    /// Look up `key` in the tree. Returns `None` if absent.
    ///
    /// The traversal walks root → leaf using an explicit stack of
    /// page-ids (no recursion, no heap allocation — the stack is a
    /// `heapless::Vec`).
    ///
    /// # Errors
    ///
    /// - [`Error::Corruption`] if any page along the path fails to
    ///   decode.
    /// - [`Error::BTreeDepthExceeded`] if the depth bound trips
    ///   (would require a tree height ≥ 32).
    /// - [`Error::Io`] / [`Error::InvalidArgument`] propagated from
    ///   the pager on a cache-miss read.
    pub fn get(&self, pager: &mut Pager<F>, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let leaf_id = descend_to_leaf(pager, self.root, key)?;
        let page_ref = pager.read_page(leaf_id)?;
        node::decode_node_find(page_ref.as_bytes(), key)
    }

    /// Snapshot-consistent variant of [`BTree::get`] — descends the
    /// tree using [`crate::pager::ReaderSnapshot::read_page`] rather
    /// than the live `Pager::read_page` (which consults the WAL
    /// overlay — including a concurrent writer's post-snapshot
    /// commits).
    ///
    /// A reader's `Collection<T>::get` must walk the primary B+tree
    /// consistently with the snapshot's pinned LSN. Without this, a
    /// writer that COW-frees a page and
    /// reallocates the same page-id can serve the reader a page
    /// whose `state.view` content is no longer a valid B+tree node,
    /// surfacing as `Error::Corruption { page_id: 0 }` from
    /// [`crate::btree::node::decode_node`].
    ///
    /// `root` is the B+tree root captured at the reader's relevant
    /// snapshot-time view of the catalog (i.e. the descriptor's
    /// `primary_root` read via [`crate::Catalog::lookup_via_snapshot`]).
    /// Walking from the writer's live root would defeat the
    /// snapshot read; the caller is responsible for handing in the
    /// snapshot-time root.
    ///
    /// # Errors
    ///
    /// - [`Error::BTreeDepthExceeded`] if the tree height exceeds
    ///   [`MAX_BTREE_DEPTH`].
    /// - [`Error::Corruption`] / [`Error::BTreeInvariantViolated`]
    ///   propagated from the descent.
    /// - Snapshot read errors propagated from
    ///   [`crate::pager::ReaderSnapshot::read_page`].
    pub fn get_via_snapshot(
        pager: &Pager<F>,
        snapshot: &crate::pager::ReaderSnapshot<F>,
        root: PageId,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
        let mut current = root;
        loop {
            if path.push(current).is_err() {
                return Err(Error::BTreeDepthExceeded {
                    limit: MAX_BTREE_DEPTH,
                });
            }
            let page = snapshot.read_page(pager, current)?;
            match node::peek_node_kind(page.as_bytes())? {
                NodeKind::Leaf => return node::decode_node_find(page.as_bytes(), key),
                NodeKind::Internal => {
                    let decoded = decode_node(page.as_bytes())?;
                    current = choose_child(&decoded, key)?;
                }
            }
        }
    }
}

/// Walk root → leaf, returning the leaf page-id that **would** contain
/// `key`. Used by every read/insert/delete path.
///
/// The walk is iterative and uses an explicit `heapless::Vec<PageId,
/// MAX_BTREE_DEPTH>` stack of visited ancestor ids. Exceeding the
/// depth bound returns [`Error::BTreeDepthExceeded`] rather than
/// recursing into a stack overflow.
pub(crate) fn descend_to_leaf<F: FileBackend>(
    pager: &mut Pager<F>,
    root: PageId,
    key: &[u8],
) -> Result<PageId> {
    let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
    let mut current = root;
    loop {
        if path.push(current).is_err() {
            return Err(Error::BTreeDepthExceeded {
                limit: MAX_BTREE_DEPTH,
            });
        }
        let page_ref = pager.read_page(current)?;
        let decoded = decode_node(page_ref.as_bytes())?;
        match decoded.kind {
            NodeKind::Leaf => return Ok(current),
            NodeKind::Internal => {
                current = choose_child(&decoded, key)?;
            }
        }
    }
}

/// Pick the child page-id of `node` whose subtree contains `key`.
/// Internal-node child selection: find the smallest `i` with
/// `pivot[i] > key`; descend into `child[i]`. If no such `i`,
/// descend into `child[K]` (the rightmost child).
pub(crate) fn choose_child(node: &DecodedNode, key: &[u8]) -> Result<PageId> {
    debug_assert!(matches!(node.kind, NodeKind::Internal));
    debug_assert_eq!(node.children.len(), node.internals.len() + 1);
    if node.children.is_empty() {
        return Err(Error::BTreeInvariantViolated {
            reason: "internal node with zero children",
        });
    }
    let mut idx = node.internals.len();
    for (i, pivot) in node.internals.iter().enumerate() {
        if pivot.key.as_slice() > key {
            idx = i;
            break;
        }
    }
    let raw = node.children[idx];
    PageId::new(raw).ok_or(Error::BTreeInvariantViolated {
        reason: "internal node had zero child page-id",
    })
}

#[cfg(test)]
mod tests;
