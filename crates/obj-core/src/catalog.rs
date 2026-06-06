//! Catalog (L5) — on-disk registry of collections.
//!
//! The catalog is a B+tree keyed by collection name and valued by
//! postcard-encoded [`CollectionDescriptor`]s. Its root page-id is
//! recorded in the file header field `root_catalog`; the default of
//! zero signals "no catalog yet" and [`Catalog::open_or_init`] creates
//! one on first open.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::btree::node::{decode_node, NodeKind};
use crate::btree::{choose_child, BTree, MAX_BTREE_DEPTH};
use crate::codec::schema::DynamicSchema;
use crate::codec::stored_schema::StoredSchema;
use crate::error::{Error, Result};
use crate::id::{bump_next_id, Id};
use crate::index::{IndexKind, IndexSpec};
use crate::pager::page::PageId;
use crate::pager::{Pager, ReaderSnapshot};
use crate::platform::{FileBackend, FileHandle};

use heapless::Vec as HeaplessVec;

/// Maximum number of collections a single catalog may carry. Bounds
/// [`Catalog::list_collections`] and the
/// next-collection-id allocator below. 1 << 20 (1 048 576) is a
/// generous ceiling — at 64-byte descriptor payloads the catalog
/// would still fit in ~64 MiB.
pub const MAX_COLLECTIONS: usize = 1 << 20;

/// The reserved catalog-name (empty UTF-8 bytes) under which the
/// next-collection-id watermark is stored. Empty names are
/// rejected on user-facing `insert`, so this row is private to the
/// catalog implementation.
const RESERVED_NEXT_ID_KEY: &[u8] = b"";

/// Leading sentinel byte marking a reserved **system** row inside the
/// catalog B+tree. User collection names can never begin
/// with this byte: [`validate_name`] rejects any name containing a NUL
/// or control character, and `b""` (the next-id row) sorts strictly
/// before any `0x00`-prefixed key.
const SYSTEM_ROW_SENTINEL: u8 = 0x00;

/// Subtype tag (the second byte of a [`SYSTEM_ROW_SENTINEL`] key)
/// identifying a "schema-by-`(collection_id, version)`" row. Keeping the
/// subtype explicit leaves room for future system-row families under the
/// same sentinel without ambiguity.
const SCHEMA_ROW_SUBTYPE: u8 = 0x01;

/// Width (in bytes) of a schema-catalog key: 1 sentinel + 1 subtype +
/// 4 (`collection_id`, big-endian) + 4 (`version`, big-endian).
const SCHEMA_KEY_LEN: usize = 10;

/// Build the fixed 10-byte catalog key for the schema row of
/// `(collection_id, version)`.
///
/// Layout:
///
/// ```text
/// [0]      0x00            sentinel — "system: schema row"
/// [1]      0x01            subtype — "schema-by-(collection_id, version)"
/// [2..6]   collection_id : u32 big-endian
/// [6..10]  version       : u32 big-endian
/// ```
///
/// Big-endian encoding makes `(collection_id, *)` a contiguous
/// lexicographic range whose entries sort by ascending `version`, and
/// keeps distinct collection ranges non-overlapping.
#[must_use]
pub fn schema_key(collection_id: u32, version: u32) -> [u8; SCHEMA_KEY_LEN] {
    let mut key = [0u8; SCHEMA_KEY_LEN];
    key[0] = SYSTEM_ROW_SENTINEL;
    key[1] = SCHEMA_ROW_SUBTYPE;
    key[2..6].copy_from_slice(&collection_id.to_be_bytes());
    key[6..10].copy_from_slice(&version.to_be_bytes());
    key
}

/// On-disk description of a collection.
///
/// Encoded with `postcard` as the value of a catalog B-tree row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionDescriptor {
    /// Catalog-assigned numeric id for this collection.
    pub collection_id: u32,
    /// Page-id of the collection's primary B-tree root.
    pub primary_root: u64,
    /// Current `Document::VERSION` for the collection's type.
    pub type_version: u32,
    /// Next-id watermark — the next [`Id`] the allocator will
    /// hand out for this collection.
    pub next_id: u64,
    /// Secondary indexes.
    pub indexes: Vec<IndexDescriptor>,
}

impl CollectionDescriptor {
    /// Construct a descriptor for a freshly-registered collection.
    /// `primary_root` is the page-id of the collection's empty
    /// primary B-tree (allocated by the caller before
    /// [`Catalog::insert`]); `collection_id` is the value the
    /// catalog will assign.
    #[must_use]
    pub const fn new(collection_id: u32, primary_root: u64, type_version: u32) -> Self {
        Self {
            collection_id,
            primary_root,
            type_version,
            next_id: 1,
            indexes: Vec::new(),
        }
    }
}

/// On-disk descriptor for a secondary index attached to a
/// collection.
///
/// Persisted inside the owning [`CollectionDescriptor::indexes`]
/// vector as part of the catalog row's postcard payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDescriptor {
    /// Catalog-assigned numeric id for this index. Stable across
    /// reopens; **never reused** — a `DroppedPending` descriptor
    /// retains its `index_id` so the page reclamation pass on the
    /// next checkpoint does not race with concurrent readers.
    pub index_id: u32,
    /// User-visible name. Stable across reopens; the reconciler
    /// matches a runtime [`IndexSpec`] to a stored descriptor by
    /// this name.
    pub name: String,
    /// Discriminator for the kind of index. See
    /// [`crate::index::IndexKind`].
    pub kind: IndexKind,
    /// Field path(s) the index is keyed by. Single-element for
    /// `Standard` / `Unique` / `Each`; ≥ 2 for `Composite`.
    pub key_paths: Vec<String>,
    /// Page-id of the index B+tree's root.
    pub root_page_id: u64,
    /// Lifecycle status — see [`IndexStatus`].
    pub status: IndexStatus,
}

/// Lifecycle state of an [`IndexDescriptor`].
///
/// `Active` indexes participate in writes and reads;
/// `DroppedPending` is a tombstone — the descriptor lingers so the
/// `index_id` is not reused and the next `Pager::checkpoint` can
/// reclaim the B+tree pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum IndexStatus {
    /// Index is live — every write maintains it, every read may
    /// consult it.
    Active = 0,
    /// Index was dropped by the reconciler; the descriptor remains
    /// so the `index_id` is not reused. Pages are reclaimed on the
    /// next `Pager::checkpoint`.
    DroppedPending = 1,
}

/// The catalog handle.
///
/// Owns the catalog B+tree's root page-id; mutating methods take
/// `&mut Pager<F>` to advance the underlying B-tree through
/// copy-on-write.
#[derive(Debug)]
pub struct Catalog<F: FileBackend = FileHandle> {
    tree: BTree<F>,
    /// Cached watermark for the next `collection_id` to allocate.
    /// Loaded from the reserved catalog row on open and re-persisted
    /// on every change.
    next_collection_id: u32,
}

impl<F: FileBackend> Catalog<F> {
    /// Open the catalog, creating it on first call.
    ///
    /// Reads the file header's `root_catalog` field via
    /// [`Pager::root_catalog`]:
    ///
    /// - If non-zero, attaches to the existing catalog B-tree and
    ///   loads the next-collection-id watermark from the reserved
    ///   row.
    /// - If zero, allocates a fresh B+tree root, seeds the reserved
    ///   row with `1`, persists the root via
    ///   [`Pager::set_root_catalog`], and commits.
    ///
    /// # Errors
    ///
    /// - [`Error::Corruption`] if an existing catalog's reserved
    ///   row is missing or malformed.
    /// - Pager / B-tree errors propagated as-is.
    pub fn open_or_init(pager: &mut Pager<F>) -> Result<Self> {
        let raw = pager.root_catalog();
        if let Some(existing) = PageId::new(raw) {
            return Self::open_existing(pager, existing);
        }
        Self::init_fresh(pager)
    }

    fn open_existing(pager: &mut Pager<F>, root: PageId) -> Result<Self> {
        let tree = BTree::<F>::open(pager, root)?;
        let watermark = match tree.get(pager, RESERVED_NEXT_ID_KEY)? {
            Some(bytes) => postcard::from_bytes::<u32>(&bytes).map_err(Error::from)?,
            None => {
                return Err(Error::Corruption {
                    page_id: root.get(),
                });
            }
        };
        Ok(Self {
            tree,
            next_collection_id: watermark,
        })
    }

    fn init_fresh(pager: &mut Pager<F>) -> Result<Self> {
        let mut tree = BTree::<F>::empty(pager)?;
        let watermark: u32 = 1;
        let encoded = postcard::to_allocvec(&watermark)?;
        tree.insert(pager, RESERVED_NEXT_ID_KEY, &encoded)?;
        pager.set_root_catalog(tree.root().get())?;
        Ok(Self {
            tree,
            next_collection_id: watermark,
        })
    }

    /// Get the descriptor for the named collection. Returns
    /// `Ok(None)` if no such collection exists.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty.
    /// - Pager / B-tree / postcard errors propagated as-is.
    pub fn get(&self, pager: &mut Pager<F>, name: &str) -> Result<Option<CollectionDescriptor>> {
        validate_name(name)?;
        match self.tree.get(pager, name.as_bytes())? {
            Some(bytes) => {
                let descriptor: CollectionDescriptor =
                    postcard::from_bytes(&bytes).map_err(Error::from)?;
                Ok(Some(descriptor))
            }
            None => Ok(None),
        }
    }

    /// Look up a collection descriptor as-of a [`ReaderSnapshot`]'s
    /// pinned LSN — i.e. observe the catalog state the reader's
    /// snapshot pinned, NOT the writer's live `Catalog.tree.root`.
    ///
    /// Walks the catalog B+tree rooted at `snapshot.root_catalog()`
    /// (the value captured by `Pager::reader_snapshot` at pin time).
    /// Every page read goes through
    /// [`ReaderSnapshot::read_page`], which consults the snapshot's
    /// frozen WAL view first and falls through to the main file —
    /// bypassing the live WAL overlay that may have been advanced by
    /// a concurrent writer since pin time. Without this, a reader's
    /// catalog descend can land on a freelist-
    /// recycled page-id whose `state.view` contents are no longer a
    /// valid B+tree node, surfacing as `Error::Corruption { page_id:
    /// 0 }` from the codec.
    ///
    /// When `snapshot.root_catalog() == 0` the catalog did not exist
    /// at the snapshot's pinned LSN; `Ok(None)` is returned and the
    /// caller should surface that as `Error::CollectionNotFound`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty.
    /// - [`Error::BTreeDepthExceeded`] if the catalog B+tree exceeds
    ///   `MAX_BTREE_DEPTH`.
    /// - [`Error::Corruption`] / [`Error::Codec`] propagated from the
    ///   snapshot read and postcard decode.
    pub fn lookup_via_snapshot(
        pager: &Pager<F>,
        snapshot: &ReaderSnapshot<F>,
        name: &str,
    ) -> Result<Option<CollectionDescriptor>> {
        validate_name(name)?;
        let Some(root) = PageId::new(snapshot.root_catalog()) else {
            return Ok(None);
        };
        let key = name.as_bytes();
        let mut path: HeaplessVec<PageId, MAX_BTREE_DEPTH> = HeaplessVec::new();
        let mut current = root;
        let leaf_node = loop {
            if path.push(current).is_err() {
                return Err(Error::BTreeDepthExceeded {
                    limit: MAX_BTREE_DEPTH,
                });
            }
            let page = snapshot.read_page(pager, current)?;
            let decoded = decode_node(page.as_bytes())?;
            match decoded.kind {
                NodeKind::Leaf => break decoded,
                NodeKind::Internal => {
                    current = choose_child(&decoded, key)?;
                }
            }
        };
        for entry in &leaf_node.leaves {
            if entry.key.as_slice() == key {
                let descriptor: CollectionDescriptor =
                    postcard::from_bytes(&entry.value).map_err(Error::from)?;
                return Ok(Some(descriptor));
            }
        }
        Ok(None)
    }

    /// Register a new collection.
    ///
    /// Allocates the next `collection_id`, sets it on `descriptor`,
    /// re-persists the next-collection-id watermark, and inserts
    /// the descriptor into the catalog B-tree. The descriptor that
    /// the caller passes in has its `collection_id` field
    /// **ignored** — the catalog assigns the canonical value.
    ///
    /// Call [`Pager::commit`] after this to make the registration
    /// durable.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty.
    /// - [`Error::CollectionAlreadyExists`] if `name` is already
    ///   registered.
    /// - [`Error::IdSpaceExhausted`] if the `u32` `collection_id`
    ///   space is exhausted.
    /// - Pager / B-tree / postcard errors propagated.
    pub fn insert(
        &mut self,
        pager: &mut Pager<F>,
        name: &str,
        mut descriptor: CollectionDescriptor,
    ) -> Result<u32> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::insert must run inside a WAL transaction \
             (Pager::begin_txn / WriteTxn::begin)",
        );
        validate_name(name)?;
        if self.tree.get(pager, name.as_bytes())?.is_some() {
            return Err(Error::CollectionAlreadyExists {
                name: name.to_owned(),
            });
        }
        let assigned = self.next_collection_id;
        descriptor.collection_id = assigned;
        let new_watermark =
            self.next_collection_id
                .checked_add(1)
                .ok_or_else(|| Error::IdSpaceExhausted {
                    collection: "<catalog>".to_owned(),
                })?;
        let encoded = postcard::to_allocvec(&descriptor)?;
        self.tree.insert(pager, name.as_bytes(), &encoded)?;
        self.persist_next_collection_id(pager, new_watermark)?;
        pager.set_root_catalog(self.tree.root().get())?;
        self.next_collection_id = new_watermark;
        Ok(assigned)
    }

    /// Update an existing collection's descriptor in place.
    ///
    /// Used when `next_id` advances, `type_version` is bumped, or
    /// secondary indexes change. The on-disk `collection_id` is
    /// preserved across the update; callers should not change it.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty.
    /// - [`Error::Corruption`] if the descriptor's `collection_id`
    ///   disagrees with the catalog's record (defensive check —
    ///   indicates a caller bug).
    /// - Pager / B-tree / postcard errors propagated.
    pub fn update(
        &mut self,
        pager: &mut Pager<F>,
        name: &str,
        descriptor: &CollectionDescriptor,
    ) -> Result<()> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::update must run inside a WAL transaction",
        );
        validate_name(name)?;
        let existing = self.get(pager, name)?.ok_or(Error::InvalidArgument(
            "catalog update: collection not registered",
        ))?;
        if existing.collection_id != descriptor.collection_id {
            return Err(Error::Corruption {
                page_id: self.tree.root().get(),
            });
        }
        let encoded = postcard::to_allocvec(descriptor)?;
        self.tree.delete(pager, name.as_bytes())?;
        self.tree.insert(pager, name.as_bytes(), &encoded)?;
        pager.set_root_catalog(self.tree.root().get())?;
        Ok(())
    }

    /// Persist the [`StoredSchema`] row for `(collection_id, version)`.
    ///
    /// Normalizes `schema` into its cross-language-canonical form via
    /// [`StoredSchema::from_live`], encodes it, and writes it under
    /// [`schema_key`]. The write is **idempotent and drift-guarded**:
    ///
    /// - If no row exists yet, the bytes are inserted and the catalog
    ///   root is re-persisted via [`Pager::set_root_catalog`] (required —
    ///   mirrors [`Catalog::insert`] / [`Catalog::update`]; without it
    ///   the header would point at the pre-insert page).
    /// - If a row already exists, the stored row's **normalized shape**
    ///   ([`StoredSchema::schema`]) is compared against the new one. The
    ///   per-field signedness hint ([`StoredSchema::int_signed`]) is
    ///   deliberately excluded: it is benign cross-language metadata, not
    ///   shape. On a shape match the call is a no-op (no write,
    ///   no WAL churn). On a shape mismatch it returns
    ///   [`Error::SchemaShapeChanged`], rolling back the whole write txn.
    ///
    /// # Errors
    ///
    /// - [`Error::SchemaShapeChanged`] if a differing shape is already
    ///   persisted under the same key.
    /// - [`Error::SchemaDepthExceeded`] if `schema` nests too deeply
    ///   for normalization.
    /// - [`Error::UnsupportedSchemaFormat`] if an existing row carries a
    ///   `format` this build does not understand.
    /// - Pager / B-tree / postcard errors propagated as-is.
    pub fn put_schema(
        &mut self,
        pager: &mut Pager<F>,
        collection_id: u32,
        version: u32,
        schema: &DynamicSchema,
    ) -> Result<()> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::put_schema must run inside a WAL transaction",
        );
        let key = schema_key(collection_id, version);
        let new_row = StoredSchema::from_live(schema)?;
        if let Some(existing_bytes) = self.tree.get(pager, &key)? {
            let existing = StoredSchema::from_postcard_bytes(&existing_bytes)?;
            if existing.schema != new_row.schema {
                return Err(Error::SchemaShapeChanged {
                    collection_id,
                    version,
                });
            }
            return Ok(());
        }
        let encoded = new_row.to_postcard_bytes()?;
        self.tree.insert(pager, &key, &encoded)?;
        pager.set_root_catalog(self.tree.root().get())?;
        Ok(())
    }

    /// Read the [`StoredSchema`] row for `(collection_id, version)` from
    /// the writer's own live / pending catalog tree.
    ///
    /// This is the **write-side** lookup: it descends `self.tree`, which
    /// reflects mutations staged in the current `WriteTxn` (including a
    /// `put_schema` from earlier in the same txn). For snapshot-isolated
    /// **reads** outside a write txn, use the free function
    /// [`lookup_schema_via_snapshot`] instead.
    ///
    /// Returns the full [`StoredSchema`] (normalized shape **and** the
    /// per-field signedness hint), not just the bare schema — the decode
    /// path needs the hint to re-pick plain-varint vs
    /// zigzag for fields the reader's live type no longer describes.
    ///
    /// # Errors
    ///
    /// - [`Error::UnsupportedSchemaFormat`] if the stored row carries a
    ///   `format` this build does not understand.
    /// - [`Error::Codec`] if the row is not a well-formed
    ///   [`StoredSchema`].
    /// - Pager / B-tree errors propagated as-is.
    pub fn get_schema_in_txn(
        &self,
        pager: &mut Pager<F>,
        collection_id: u32,
        version: u32,
    ) -> Result<Option<StoredSchema>> {
        let key = schema_key(collection_id, version);
        match self.tree.get(pager, &key)? {
            Some(bytes) => Ok(Some(StoredSchema::from_postcard_bytes(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Declare a new secondary index on the named collection.
    ///
    /// Allocates a fresh `index_id`, an empty index B+tree,
    /// and appends a new `IndexDescriptor { status: Active }` to
    /// the collection's `indexes` vector. The mutation rides inside
    /// the caller's WAL transaction; on rollback the descriptor +
    /// the empty B+tree are both discarded.
    ///
    /// `spec` is validated before any state mutation; an invalid
    /// spec surfaces as [`Error::InvalidArgument`] before the
    /// catalog touches the pager.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `spec.validate()` rejects
    ///   the spec.
    /// - [`Error::CollectionNotFound`] if `collection` is not
    ///   registered.
    /// - [`Error::IndexKindMismatch`] if an `Active` descriptor of
    ///   the same name has a different `(kind, key_paths)`.
    /// - [`Error::IdSpaceExhausted`] on `u32` `index_id` wraparound
    ///   (defense-in-depth — practically unreachable).
    /// - Pager / B-tree / postcard errors propagated.
    pub fn declare_index(
        &mut self,
        pager: &mut Pager<F>,
        collection: &str,
        spec: &IndexSpec,
    ) -> Result<u32> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::declare_index must run inside a WAL transaction",
        );
        spec.validate()?;
        validate_name(collection)?;
        let mut descriptor =
            self.get(pager, collection)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: collection.to_owned(),
                })?;
        if let Some(existing) = descriptor.indexes.iter().find(|d| d.name == spec.name) {
            return Self::reconcile_existing_index(existing, spec);
        }
        let index_id = next_index_id(&descriptor)?;
        let root_page_id = BTree::<F>::empty(pager)?.root().get();
        let new_descriptor = IndexDescriptor {
            index_id,
            name: spec.name.clone(),
            kind: spec.kind,
            key_paths: spec.key_paths.clone(),
            root_page_id,
            status: IndexStatus::Active,
        };
        descriptor.indexes.push(new_descriptor);
        self.update(pager, collection, &descriptor)?;
        Ok(index_id)
    }

    /// Reconcile a runtime [`IndexSpec`] against an already-stored
    /// `IndexDescriptor` of the same name. Returns the existing
    /// `index_id` if the `(kind, key_paths)` match; errors otherwise.
    fn reconcile_existing_index(existing: &IndexDescriptor, spec: &IndexSpec) -> Result<u32> {
        if existing.kind != spec.kind {
            return Err(Error::IndexKindMismatch {
                name: spec.name.clone(),
                expected: spec.kind,
                found: existing.kind,
            });
        }
        if existing.key_paths != spec.key_paths {
            return Err(Error::IndexKeyPathsMismatch {
                name: spec.name.clone(),
            });
        }
        Ok(existing.index_id)
    }

    /// Reconcile the runtime [`IndexSpec`] set for a collection
    /// against the catalog's stored descriptors.
    ///
    /// - Specs present in `specs` and absent from the descriptor are
    ///   **declared** (new `Active` descriptor + empty B+tree).
    /// - `Active` descriptors absent from `specs` are flipped to
    ///   `DroppedPending`.
    /// - Matching `(name, kind, key_paths)` pairs are left alone —
    ///   reconciliation is **idempotent**.
    ///
    /// Returns the descriptor's post-reconciliation index roster (a
    /// `Vec<IndexDescriptor>` clone) so the caller can build its
    /// maintenance plan without re-querying the catalog.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexKindMismatch`] /
    ///   [`Error::IndexKeyPathsMismatch`] on per-name structural
    ///   disagreement.
    /// - [`Error::IdSpaceExhausted`] on `u32` `index_id` wraparound.
    /// - Pager / B-tree / postcard errors propagated.
    pub fn reconcile_indexes(
        &mut self,
        pager: &mut Pager<F>,
        collection: &str,
        specs: &[IndexSpec],
    ) -> Result<Vec<IndexDescriptor>> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::reconcile_indexes must run inside a WAL transaction",
        );
        validate_name(collection)?;
        for spec in specs {
            spec.validate()?;
        }
        for spec in specs {
            let _ = self.declare_index(pager, collection, spec)?;
        }
        let descriptor = self
            .get(pager, collection)?
            .ok_or_else(|| Error::CollectionNotFound {
                name: collection.to_owned(),
            })?;
        let mut to_drop: Vec<String> = Vec::new();
        for d in &descriptor.indexes {
            if d.status == IndexStatus::Active && !specs.iter().any(|s| s.name == d.name) {
                to_drop.push(d.name.clone());
            }
        }
        for name in to_drop {
            self.drop_index(pager, collection, &name)?;
        }
        let final_descriptor =
            self.get(pager, collection)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: collection.to_owned(),
                })?;
        Ok(final_descriptor.indexes)
    }

    /// Drop the named index from the named collection — flips the
    /// descriptor's status to [`IndexStatus::DroppedPending`].
    ///
    /// The descriptor stays in the catalog so its `index_id` is not
    /// reused. The index B+tree pages are reclaimed on the next
    /// [`Pager::checkpoint`] pass (deferred reclamation: a
    /// concurrent reader's snapshot may still need to walk the
    /// pages until its pin is released).
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if `collection` is not
    ///   registered.
    /// - [`Error::IndexNotFound`] if `index_name` is not a known
    ///   descriptor on the collection.
    pub fn drop_index(
        &mut self,
        pager: &mut Pager<F>,
        collection: &str,
        index_name: &str,
    ) -> Result<()> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::drop_index must run inside a WAL transaction",
        );
        validate_name(collection)?;
        let mut descriptor =
            self.get(pager, collection)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: collection.to_owned(),
                })?;
        let entry = descriptor
            .indexes
            .iter_mut()
            .find(|d| d.name == index_name)
            .ok_or_else(|| Error::IndexNotFound {
                collection: collection.to_owned(),
                name: index_name.to_owned(),
            })?;
        if entry.status == IndexStatus::Active {
            entry.status = IndexStatus::DroppedPending;
        }
        self.update(pager, collection, &descriptor)?;
        Ok(())
    }

    /// Allocate the next [`Id`] for the named collection,
    /// persisting the bumped `next_id` watermark inside the
    /// catalog row.
    ///
    /// Re-reads the descriptor, bumps `next_id`, writes the new
    /// descriptor back via [`Catalog::update`], and returns the
    /// just-issued id.
    ///
    /// The id-bump is staged through the WAL exactly like every
    /// other catalog mutation; if the caller's surrounding
    /// transaction is later rolled back (no `Pager::commit`), the
    /// allocation is rolled back with it — the next open will
    /// re-issue the same id.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty or not
    ///   registered.
    /// - [`Error::IdSpaceExhausted`] on `u64` wraparound.
    /// - Pager / B-tree errors propagated.
    pub fn next_id(&mut self, pager: &mut Pager<F>, name: &str) -> Result<Id> {
        debug_assert!(
            pager.in_txn(),
            "Catalog::next_id must run inside a WAL transaction",
        );
        validate_name(name)?;
        let mut descriptor = self.get(pager, name)?.ok_or(Error::InvalidArgument(
            "catalog next_id: collection not registered",
        ))?;
        let issued = bump_next_id(&mut descriptor.next_id, || name.to_owned())?;
        self.update(pager, name, &descriptor)?;
        Ok(issued)
    }

    /// List every registered collection.
    ///
    /// Scans the full catalog B-tree. Bounded by
    /// [`MAX_COLLECTIONS`]. The reserved next-collection-id
    /// row is filtered out.
    ///
    /// # Errors
    ///
    /// - [`Error::BTreeScanLimitExceeded`] if the catalog has more
    ///   than [`MAX_COLLECTIONS`] entries.
    /// - Pager / B-tree / postcard errors propagated.
    pub fn list_collections(
        &self,
        pager: &mut Pager<F>,
    ) -> Result<Vec<(String, CollectionDescriptor)>> {
        let mut out: Vec<(String, CollectionDescriptor)> = Vec::new();
        let mut scanned = 0usize;
        let iter = self.tree.range(pager, ..)?;
        for entry in iter {
            scanned += 1;
            if scanned > MAX_COLLECTIONS {
                return Err(Error::BTreeScanLimitExceeded {
                    limit: MAX_COLLECTIONS,
                });
            }
            let (key, value) = entry?;
            if key.as_slice() == RESERVED_NEXT_ID_KEY {
                continue;
            }
            if key.first() == Some(&SYSTEM_ROW_SENTINEL) {
                continue;
            }
            let name = String::from_utf8(key).map_err(|_| Error::Corruption {
                page_id: self.tree.root().get(),
            })?;
            let descriptor: CollectionDescriptor =
                postcard::from_bytes(&value).map_err(Error::from)?;
            out.push((name, descriptor));
        }
        Ok(out)
    }

    fn persist_next_collection_id(&mut self, pager: &mut Pager<F>, watermark: u32) -> Result<()> {
        let encoded = postcard::to_allocvec(&watermark)?;
        self.tree.delete(pager, RESERVED_NEXT_ID_KEY)?;
        self.tree.insert(pager, RESERVED_NEXT_ID_KEY, &encoded)?;
        Ok(())
    }
}

/// Look up the [`StoredSchema`] row for `(collection_id, version)`
/// as-of a [`ReaderSnapshot`]'s pinned LSN.
///
/// This is the **read-side**, snapshot-isolated counterpart to
/// [`Catalog::get_schema_in_txn`], mirroring [`Catalog::lookup_via_snapshot`]:
/// it descends the catalog B+tree rooted at `snapshot.root_catalog()`
/// using [`ReaderSnapshot::read_page`], so it observes exactly the
/// catalog state the reader's snapshot pinned — never the writer's live
/// `Catalog.tree.root`, which a concurrent writer may have advanced (and
/// whose pages may have been freelist-recycled, see
/// [`Catalog::lookup_via_snapshot`]). The schema row is therefore read
/// at the same snapshot as the document bytes it describes, so a reader
/// never pairs a newer schema with older document bytes.
///
/// Returns the full [`StoredSchema`] (normalized shape **and** the
/// per-field signedness hint), not just the bare schema: the decode
/// path re-specializes signedness from the hint for fields
/// the reader's live type no longer describes.
///
/// When `snapshot.root_catalog() == 0` the catalog did not exist at the
/// snapshot's pinned LSN; `Ok(None)` is returned.
///
/// # Errors
///
/// - [`Error::BTreeDepthExceeded`] if the catalog B+tree exceeds
///   `MAX_BTREE_DEPTH`.
/// - [`Error::UnsupportedSchemaFormat`] if the stored row carries a
///   `format` this build does not understand.
/// - [`Error::Corruption`] / [`Error::Codec`] propagated from the
///   snapshot read and postcard decode.
pub fn lookup_schema_via_snapshot<F: FileBackend>(
    pager: &Pager<F>,
    snapshot: &ReaderSnapshot<F>,
    collection_id: u32,
    version: u32,
) -> Result<Option<StoredSchema>> {
    let Some(root) = PageId::new(snapshot.root_catalog()) else {
        return Ok(None);
    };
    let key = schema_key(collection_id, version);
    match BTree::<F>::get_via_snapshot(pager, snapshot, root, &key)? {
        Some(bytes) => Ok(Some(StoredSchema::from_postcard_bytes(&bytes)?)),
        None => Ok(None),
    }
}

/// Maximum collection-name length, in bytes. 255 is a
/// conservative, widely-compatible cap that comfortably exceeds any
/// reasonable name yet keeps catalog keys bounded. Loosening this
/// later is backward-compatible; tightening it would be breaking, so
/// we pick a generous-but-finite bound at the 1.0 freeze.
const MAX_COLLECTION_NAME_LEN: usize = 255;

/// Validate a collection name.
///
/// Policy:
/// - Reject the empty string: it would collide with the reserved
///   next-collection-id row and is a UX hazard (a Document with
///   `COLLECTION = ""` would be invisible to `list_collections`).
/// - Reject names longer than [`MAX_COLLECTION_NAME_LEN`] bytes so
///   catalog keys stay bounded.
/// - Reject names containing a NUL or any other ASCII/Unicode
///   control character: such bytes are a portability and
///   display-safety hazard (terminal injection, truncation at an
///   embedded NUL on FFI boundaries).
///
/// All rejections surface as [`Error::InvalidArgument`].
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidArgument("collection name must be non-empty"));
    }
    if name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(Error::InvalidArgument("collection name exceeds 255 bytes"));
    }
    if name.chars().any(char::is_control) {
        return Err(Error::InvalidArgument(
            "collection name must not contain NUL or control characters",
        ));
    }
    Ok(())
}

/// Compute the next `index_id` for the collection. Scans the
/// descriptor's existing indexes (including `DroppedPending` rows
/// — their ids are NEVER reused)
/// and returns one past the max. Wraps with [`Error::IdSpaceExhausted`]
/// on `u32::MAX`.
fn next_index_id(descriptor: &CollectionDescriptor) -> Result<u32> {
    let max = descriptor
        .indexes
        .iter()
        .map(|d| d.index_id)
        .max()
        .unwrap_or(0);
    max.checked_add(1).ok_or_else(|| Error::IdSpaceExhausted {
        collection: format!("<indexes:{}>", descriptor.collection_id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::{Config, Pager};
    use crate::platform::FileHandle;

    fn fresh_pager() -> Pager<FileHandle> {
        Pager::<FileHandle>::memory(Config::default()).expect("pager")
    }

    #[test]
    fn validate_name_policy() {
        assert!(validate_name("users").is_ok());
        let at_cap = "a".repeat(MAX_COLLECTION_NAME_LEN);
        assert!(validate_name(&at_cap).is_ok());

        assert!(matches!(validate_name(""), Err(Error::InvalidArgument(_))));
        let too_long = "a".repeat(MAX_COLLECTION_NAME_LEN + 1);
        assert!(matches!(
            validate_name(&too_long),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            validate_name("ab\0cd"),
            Err(Error::InvalidArgument(_))
        ));
        for bad in ["line\nbreak", "tab\tname", "del\u{7f}name"] {
            assert!(
                matches!(validate_name(bad), Err(Error::InvalidArgument(_))),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn open_or_init_on_fresh_pager_creates_catalog() {
        let mut pager = fresh_pager();
        assert_eq!(pager.root_catalog(), 0);
        let _catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init catalog");
        assert_ne!(pager.root_catalog(), 0, "catalog root must be installed");
    }

    #[test]
    fn insert_and_get_round_trip() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let primary_root = BTree::<FileHandle>::empty(&mut pager)
            .expect("primary tree")
            .root();
        let descriptor = CollectionDescriptor::new(0, primary_root.get(), 1);
        let assigned = catalog
            .insert(&mut pager, "users", descriptor.clone())
            .expect("insert users");
        assert_eq!(assigned, 1, "first collection gets id 1");

        let back = catalog
            .get(&mut pager, "users")
            .expect("get")
            .expect("present");
        assert_eq!(back.collection_id, assigned);
        assert_eq!(back.primary_root, primary_root.get());
        assert_eq!(back.type_version, 1);
        assert_eq!(back.next_id, 1);
        assert!(back.indexes.is_empty());
    }

    #[test]
    fn duplicate_insert_errors() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let primary_root = BTree::<FileHandle>::empty(&mut pager)
            .expect("primary tree")
            .root();
        let descriptor = CollectionDescriptor::new(0, primary_root.get(), 1);
        catalog
            .insert(&mut pager, "users", descriptor.clone())
            .expect("first insert");
        let err = catalog
            .insert(&mut pager, "users", descriptor)
            .expect_err("dup");
        assert!(matches!(err, Error::CollectionAlreadyExists { ref name } if name == "users"));
    }

    #[test]
    fn next_id_advances_and_persists() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let primary_root = BTree::<FileHandle>::empty(&mut pager)
            .expect("primary tree")
            .root();
        let _id = catalog
            .insert(
                &mut pager,
                "users",
                CollectionDescriptor::new(0, primary_root.get(), 1),
            )
            .expect("insert");
        let id1 = catalog.next_id(&mut pager, "users").expect("next 1");
        let id2 = catalog.next_id(&mut pager, "users").expect("next 2");
        assert_eq!(id1.get(), 1);
        assert_eq!(id2.get(), 2);
        let descriptor = catalog
            .get(&mut pager, "users")
            .expect("get")
            .expect("present");
        assert_eq!(descriptor.next_id, 3, "next_id watermark advanced");
    }

    #[test]
    fn cross_collection_id_isolation() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let p1 = BTree::<FileHandle>::empty(&mut pager).expect("p1").root();
        let p2 = BTree::<FileHandle>::empty(&mut pager).expect("p2").root();
        catalog
            .insert(&mut pager, "a", CollectionDescriptor::new(0, p1.get(), 1))
            .expect("a");
        catalog
            .insert(&mut pager, "b", CollectionDescriptor::new(0, p2.get(), 1))
            .expect("b");
        let _ = catalog.next_id(&mut pager, "a").expect("a1");
        let _ = catalog.next_id(&mut pager, "a").expect("a2");
        let _ = catalog.next_id(&mut pager, "a").expect("a3");
        let a = catalog
            .get(&mut pager, "a")
            .expect("get a")
            .expect("present");
        let b = catalog
            .get(&mut pager, "b")
            .expect("get b")
            .expect("present");
        assert_eq!(a.next_id, 4);
        assert_eq!(b.next_id, 1, "b's next_id unchanged");
    }

    #[test]
    fn list_collections_excludes_reserved_row() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let p1 = BTree::<FileHandle>::empty(&mut pager).expect("p1").root();
        let p2 = BTree::<FileHandle>::empty(&mut pager).expect("p2").root();
        catalog
            .insert(
                &mut pager,
                "alpha",
                CollectionDescriptor::new(0, p1.get(), 1),
            )
            .expect("alpha");
        catalog
            .insert(
                &mut pager,
                "beta",
                CollectionDescriptor::new(0, p2.get(), 1),
            )
            .expect("beta");
        let listed = catalog.list_collections(&mut pager).expect("list");
        assert_eq!(listed.len(), 2);
        let names: Vec<&str> = listed.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn reopen_preserves_watermark() {
        let mut pager = fresh_pager();
        let p1 = BTree::<FileHandle>::empty(&mut pager).expect("p1").root();
        let p2 = BTree::<FileHandle>::empty(&mut pager).expect("p2").root();
        {
            let mut catalog =
                Catalog::<FileHandle>::open_or_init(&mut pager).expect("init catalog");
            catalog
                .insert(
                    &mut pager,
                    "users",
                    CollectionDescriptor::new(0, p1.get(), 1),
                )
                .expect("users");
            catalog
                .insert(
                    &mut pager,
                    "posts",
                    CollectionDescriptor::new(0, p2.get(), 1),
                )
                .expect("posts");
            let _ = catalog.next_id(&mut pager, "users").expect("u1");
            let _ = catalog.next_id(&mut pager, "users").expect("u2");
        }
        let catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("reopen");
        let listed = catalog.list_collections(&mut pager).expect("list");
        assert_eq!(listed.len(), 2);
        let users = listed
            .iter()
            .find(|(n, _)| n == "users")
            .expect("users present");
        assert_eq!(users.1.next_id, 3, "users next_id survived reopen");
        let posts = listed
            .iter()
            .find(|(n, _)| n == "posts")
            .expect("posts present");
        assert_eq!(posts.1.next_id, 1);
    }

    #[test]
    fn empty_name_rejected() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let p1 = BTree::<FileHandle>::empty(&mut pager).expect("p1").root();
        let err = catalog
            .insert(&mut pager, "", CollectionDescriptor::new(0, p1.get(), 1))
            .expect_err("empty name rejected");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    fn schema_v1() -> DynamicSchema {
        DynamicSchema::map([
            ("customer_id", DynamicSchema::U64),
            ("total_cents", DynamicSchema::U64),
        ])
    }

    #[test]
    fn schema_key_byte_layout() {
        let key = schema_key(7, 1);
        assert_eq!(key.len(), 10);
        assert_eq!(key[0], 0x00, "sentinel byte");
        assert_eq!(key[1], 0x01, "schema-row subtype");
        assert_eq!(&key[2..6], &7u32.to_be_bytes(), "collection_id BE");
        assert_eq!(&key[6..10], &1u32.to_be_bytes(), "version BE");
    }

    #[test]
    fn schema_key_be_orders_versions_ascending_within_a_collection() {
        let k1 = schema_key(7, 1);
        let k2 = schema_key(7, 2);
        let k256 = schema_key(7, 256);
        assert!(k1 < k2, "v1 key < v2 key");
        assert!(k2 < k256, "BE means v2 < v256 (not byte-flipped)");
    }

    #[test]
    fn schema_key_distinct_collections_do_not_overlap() {
        let max_v_for_7 = schema_key(7, u32::MAX);
        let min_v_for_8 = schema_key(8, 0);
        assert!(
            max_v_for_7 < min_v_for_8,
            "cid range (7,*) must precede (8,*) with no overlap",
        );
    }

    #[test]
    fn put_then_get_round_trips_schema_and_hint() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let live = DynamicSchema::map([("id", DynamicSchema::U64), ("delta", DynamicSchema::I64)]);
        catalog
            .put_schema(&mut pager, 7, 1, &live)
            .expect("put_schema");
        let got = catalog
            .get_schema_in_txn(&mut pager, 7, 1)
            .expect("get_schema_in_txn")
            .expect("present");
        let expected_shape =
            DynamicSchema::map([("id", DynamicSchema::U64), ("delta", DynamicSchema::U64)]);
        assert_eq!(got.schema, expected_shape);
        assert_eq!(got.int_signed, vec![false, true]);
        assert_eq!(got.format, crate::codec::STORED_SCHEMA_FORMAT_V1);
    }

    #[test]
    fn get_schema_in_txn_absent_is_none() {
        let mut pager = fresh_pager();
        let catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let got = catalog
            .get_schema_in_txn(&mut pager, 42, 1)
            .expect("get_schema_in_txn");
        assert!(got.is_none(), "no row written yet");
    }

    #[test]
    fn re_put_identical_schema_is_idempotent_noop() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        catalog
            .put_schema(&mut pager, 7, 1, &schema_v1())
            .expect("first put");
        let root_after_first = pager.root_catalog();
        let value_after_first = catalog
            .tree
            .get(&mut pager, &schema_key(7, 1))
            .expect("get bytes")
            .expect("present");
        catalog
            .put_schema(&mut pager, 7, 1, &schema_v1())
            .expect("idempotent re-put");
        assert_eq!(
            pager.root_catalog(),
            root_after_first,
            "idempotent re-put must not advance the catalog root",
        );
        let value_after_second = catalog
            .tree
            .get(&mut pager, &schema_key(7, 1))
            .expect("get bytes")
            .expect("present");
        assert_eq!(value_after_first, value_after_second, "value unchanged");
    }

    #[test]
    fn re_put_different_shape_errors_schema_shape_changed() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        catalog
            .put_schema(&mut pager, 7, 1, &schema_v1())
            .expect("first put");
        let drifted = DynamicSchema::map([
            ("customer_id", DynamicSchema::U64),
            ("total_cents", DynamicSchema::U64),
            ("placed_at", DynamicSchema::U64),
        ]);
        let err = catalog
            .put_schema(&mut pager, 7, 1, &drifted)
            .expect_err("shape drift must error");
        assert!(matches!(
            err,
            Error::SchemaShapeChanged {
                collection_id: 7,
                version: 1
            }
        ));
    }

    #[test]
    fn re_put_benign_signedness_change_is_idempotent_not_drift() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let unsigned = DynamicSchema::map([("v", DynamicSchema::U64)]);
        let signed = DynamicSchema::map([("v", DynamicSchema::I64)]);
        catalog
            .put_schema(&mut pager, 9, 1, &unsigned)
            .expect("put unsigned");
        catalog
            .put_schema(&mut pager, 9, 1, &signed)
            .expect("benign signedness divergence must be idempotent Ok");
        let got = catalog
            .get_schema_in_txn(&mut pager, 9, 1)
            .expect("get")
            .expect("present");
        assert_eq!(got.int_signed, vec![false], "first writer's hint kept");
    }

    #[test]
    fn list_collections_does_not_surface_schema_rows() {
        let mut pager = fresh_pager();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let p1 = BTree::<FileHandle>::empty(&mut pager).expect("p1").root();
        let cid = catalog
            .insert(
                &mut pager,
                "orders",
                CollectionDescriptor::new(0, p1.get(), 1),
            )
            .expect("insert orders");
        catalog
            .put_schema(&mut pager, cid, 1, &schema_v1())
            .expect("put_schema");
        let listed = catalog.list_collections(&mut pager).expect("list");
        assert_eq!(listed.len(), 1, "only the user collection is listed");
        assert_eq!(listed[0].0, "orders");
    }

    #[test]
    fn lookup_schema_via_snapshot_is_mvcc_isolated() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("schema-snap.obj");
        let mut pager = Pager::<FileHandle>::open(&path, Config::default()).expect("open");
        pager.begin_txn();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        catalog
            .put_schema(&mut pager, 7, 1, &schema_v1())
            .expect("put v1");
        let _ = pager.commit().expect("commit 1");
        let snap = pager.reader_snapshot().expect("snapshot");

        let pinned = lookup_schema_via_snapshot(&pager, &snap, 7, 1)
            .expect("lookup")
            .expect("present at snapshot");
        let v1_shape = pinned.schema.clone();

        let v2 = DynamicSchema::map([
            ("customer_id", DynamicSchema::U64),
            ("total_cents", DynamicSchema::U64),
            ("placed_at", DynamicSchema::U64),
        ]);
        catalog.put_schema(&mut pager, 7, 2, &v2).expect("put v2");
        let _ = pager.commit().expect("commit 2");

        let still_absent =
            lookup_schema_via_snapshot(&pager, &snap, 7, 2).expect("lookup v2 at old snap");
        assert!(
            still_absent.is_none(),
            "snapshot pinned before the v2 write must not observe it",
        );
        let still_v1 = lookup_schema_via_snapshot(&pager, &snap, 7, 1)
            .expect("lookup v1 at old snap")
            .expect("v1 still present");
        assert_eq!(still_v1.schema, v1_shape, "snapshot view of v1 frozen");

        let fresh = pager.reader_snapshot().expect("fresh snapshot");
        let v2_seen =
            lookup_schema_via_snapshot(&pager, &fresh, 7, 2).expect("lookup v2 at fresh snap");
        assert!(v2_seen.is_some(), "fresh snapshot observes the v2 write");
    }
}
