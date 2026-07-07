//! Public `WriteTxn` / `ReadTxn` types.
//!
//! Thin wrappers over `obj_core::txn::{WriteTxn, ReadTxn}` that
//! attach a [`Catalog`] reference (the obj-core txn types are
//! catalog-agnostic; the catalog is the obj crate's responsibility).
//!
//! The catalog handle is `Arc<Mutex<Catalog<FileHandle>>>`.  Lock
//! ordering: **always acquire the pager mutex (via the txn env)
//! BEFORE the catalog mutex**.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use obj_core::btree::BTree;
use obj_core::codec::{DocumentHeader, DOC_HEADER_SIZE, MAX_INLINE_DOC};
use obj_core::index::EncodedIndexKey;
use obj_core::pager::checksum::crc32c;
use obj_core::pager::page::PageId;
use obj_core::pager::Pager;
use obj_core::platform::FileHandle;
use obj_core::{
    Catalog, CollectionDescriptor, Document, Error, Id, IndexStatus, ReaderSnapshot, Result, TxnEnv,
};

use crate::collection::Collection;

/// `type_version` stamped on documents written via the C ABI raw-
/// bytes path. The C caller has no Rust `Document::VERSION`; we
/// stamp 1 so the value is recognisable in dump output and the
/// existing schema-version-from-future / migration logic still
/// applies when a Rust-typed reader opens the same collection.
///
/// Bumping this constant is breaking for any consumer that has
/// written raw-bytes data with the old value, so leave at 1
/// pre-1.0.
pub(crate) const RAW_BYTES_TYPE_VERSION: u32 = 1;

/// Public write transaction.
///
/// Acquired by [`crate::Db::transaction`].  Holds the in-process
/// write-serialization mutex + cross-process `WRITER_LOCK` for its
/// entire lifetime.  `commit` / `rollback` consume `self`; dropping
/// without explicitly committing rolls back automatically.
pub struct WriteTxn<'db> {
    pub(crate) inner: obj_core::WriteTxn<'db, FileHandle>,
    pub(crate) catalog: Arc<Mutex<Catalog<FileHandle>>>,
    /// Per-process cache of `(collection_name, version)` keys whose
    /// `T::indexes()` reconciliation has already run. Reconciliation
    /// is idempotent but expensive (a catalog walk + index
    /// declarations); caching keeps the first
    /// `WriteTxn::collection::<T>()` call per-process per
    /// `(collection, version)` as the only one that pays the cost.
    ///
    /// The key includes the schema `version`: a later version of
    /// the same collection that ADDS an index reconciles on its first
    /// write rather than being skipped (the name-only key never let a
    /// cross-version index addition become `Active`). See
    /// [`crate::collection::reconcile_specs_once`] for the full
    /// rationale and the removal-interleaving caveat.
    ///
    /// Membership is promoted into this SHARED set ONLY after a
    /// successful [`Self::commit`]. During a txn the keys live in the
    /// per-txn [`Self::reconciled_staged`] set instead, so a
    /// rolled-back first-ever txn never poisons this set into skipping
    /// reconciliation on a later (committed) txn in the same process.
    pub(crate) reconciled: Arc<Mutex<HashSet<(String, u32)>>>,
    /// Per-transaction staged set of `(collection, version)` keys
    /// whose `T::indexes()` reconciliation has run INSIDE this (not-yet-
    /// committed) txn. The skip-check in
    /// [`crate::collection::reconcile_indexes_once`] is `shared ∪
    /// staged`, so a second handle of the same `(collection, version)`
    /// in one txn still skips the (idempotent) catalog walk — but the
    /// keys are only folded into the shared [`Self::reconciled`] set by
    /// [`Self::commit`] AFTER the WAL commit succeeds. On rollback /
    /// drop this set is discarded with NO shared-set mutation, so a
    /// rolled-back lazy-create leaves the shared cache untouched and
    /// the next txn re-reconciles correctly.
    ///
    /// Not behind a mutex: a `WriteTxn` is single-threaded (it holds
    /// the write-serialization lock for its whole life and is borrowed
    /// `&mut` by every write), so interior mutability via the staging
    /// helpers is sufficient.
    pub(crate) reconciled_staged: HashSet<(String, u32)>,
    /// Batch-aware catalog flush. Per-transaction cache of the
    /// LIVE [`CollectionDescriptor`] for every collection touched by
    /// a write, keyed by collection name. This is the SOLE mid-txn
    /// source of truth for `next_id`, `primary_root`, and each
    /// index's `root_page_id`: every write bumps / advances these
    /// IN-MEMORY rather than rewriting the catalog B-tree per doc.
    /// [`Self::commit`] flushes each entry back through
    /// `Catalog::update` exactly ONCE (see
    /// [`Self::flush_descriptors`]). On rollback / drop the cache is
    /// discarded with no catalog-tree side effects.
    ///
    /// The `Arc<Mutex<_>>` is cloned into each [`Collection`]'s
    /// `WriteRef` so two handles of the same collection opened in one
    /// txn share the single entry.
    pub(crate) descriptors: crate::collection::DescriptorCache,
}

impl<'db> WriteTxn<'db> {
    /// Construct a `WriteTxn` directly from its three pieces.
    /// Public so the FFI layer ([`libobj`](../../libobj/index.html))
    /// can build an owned write txn whose lifetime extends past a
    /// single `Db::transaction` closure call.
    ///
    /// User-Rust callers should reach for `Db::transaction` — the
    /// closure shape handles commit / rollback / drop semantics
    /// correctly without needing direct construction.
    #[doc(hidden)]
    #[must_use]
    pub fn from_parts(
        inner: obj_core::WriteTxn<'db, FileHandle>,
        catalog: Arc<Mutex<Catalog<FileHandle>>>,
        reconciled: Arc<Mutex<HashSet<(String, u32)>>>,
    ) -> Self {
        Self {
            inner,
            catalog,
            reconciled,
            reconciled_staged: HashSet::new(),
            descriptors: crate::collection::new_descriptor_cache(),
        }
    }

    pub(crate) fn new(
        inner: obj_core::WriteTxn<'db, FileHandle>,
        catalog: Arc<Mutex<Catalog<FileHandle>>>,
        reconciled: Arc<Mutex<HashSet<(String, u32)>>>,
    ) -> Self {
        Self::from_parts(inner, catalog, reconciled)
    }

    /// Open a typed handle to the collection `T` lives in.
    ///
    /// Lazily creates the catalog row + an empty primary B-tree on
    /// first call for a given `T` inside the current process.  The
    /// catalog mutation is staged in the same WAL transaction as
    /// the user's subsequent writes — a rolled-back txn leaves no
    /// half-created collection.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] if the pager / catalog mutex is poisoned.
    /// - Any error the pager / B-tree / postcard codec returns.
    pub fn collection<T: Document>(&mut self) -> Result<Collection<'_, T>> {
        if let (Some(namespace), tail) = crate::db::try_split_namespace(T::COLLECTION)? {
            return Err(Error::AttachedDatabaseIsReadOnly {
                namespace: namespace.to_owned(),
                collection: tail.to_owned(),
            });
        }
        Collection::open_or_create(self)
    }

    /// Commit the transaction.
    ///
    /// Flushes every cached [`CollectionDescriptor`] back to the
    /// catalog (one `Catalog::update` per touched collection) BEFORE
    /// the WAL commit, so the coalesced `next_id` / `primary_root` /
    /// index-root advances land durably in the same transaction as the
    /// document + index writes. A flush failure aborts the commit (the
    /// `?` propagates and `self` drops, rolling the WAL back) rather
    /// than committing a half-flushed catalog.
    ///
    /// AFTER the WAL commit succeeds, fold this txn's staged
    /// reconciled-collection names into the shared per-process
    /// `reconciled` set, so the expensive `T::indexes()`
    /// reconciliation is skipped for those collections on later txns.
    /// Promotion is deliberately POST-commit: a rolled-back txn never
    /// reaches here, so it cannot poison the shared cache into skipping
    /// reconciliation against a catalog whose index rows it just rolled
    /// back. A poisoned `reconciled` mutex maps to `Error::Busy` but
    /// does NOT un-commit the durable WAL state.
    ///
    /// # Errors
    ///
    /// As [`obj_core::WriteTxn::commit`], plus any catalog / pager /
    /// postcard error surfaced by the descriptor flush, plus
    /// [`Error::Busy`] if the shared `reconciled` mutex is poisoned at
    /// promotion time (after the commit has already landed durably).
    pub fn commit(self) -> Result<()> {
        self.flush_descriptors()?;
        let Self {
            inner,
            reconciled,
            reconciled_staged,
            ..
        } = self;
        inner.commit()?;
        promote_reconciled(&reconciled, reconciled_staged)
    }

    /// Persist every cached descriptor back to the catalog
    /// B-tree exactly once. Called by [`Self::commit`] before the WAL
    /// commit. Iterates the per-txn descriptor cache (one entry per
    /// touched collection) and issues one `Catalog::update` apiece,
    /// propagating the first failure via `?` so a partial flush aborts
    /// the commit.
    ///
    /// Lock order matches every write path: pager BEFORE catalog (the
    /// descriptor-cache lock is acquired and released first, so it is
    /// never held across the pager/catalog locks).
    fn flush_descriptors(&self) -> Result<()> {
        let entries: Vec<(String, CollectionDescriptor)> = {
            let cache = crate::collection::lock_descriptors(&self.descriptors)?;
            if cache.is_empty() {
                return Ok(());
            }
            cache
                .iter()
                .map(|(name, descriptor)| (name.clone(), descriptor.clone()))
                .collect()
        };
        let mut pager = lock_pager(self.inner.env())?;
        let mut catalog = lock_catalog(&self.catalog)?;
        for (name, descriptor) in &entries {
            catalog.update(&mut pager, name, descriptor)?;
        }
        Ok(())
    }

    /// Roll the transaction back.
    ///
    /// # Errors
    ///
    /// As [`obj_core::WriteTxn::rollback`].
    pub fn rollback(self) -> Result<()> {
        self.inner.rollback()
    }

    /// **FFI shim**: insert a raw-bytes document into `collection`,
    /// returning the freshly-allocated [`Id`].
    ///
    /// The payload is stored as-is; the on-disk record carries the
    /// standard 16-byte [`DocumentHeader`] with
    /// `type_version = RAW_BYTES_TYPE_VERSION` and a CRC32C of the
    /// payload. The collection is lazy-created in the same WAL
    /// transaction if it does not already exist.
    ///
    /// **Index maintenance does NOT run** on the raw-bytes path —
    /// the C ABI's caller has no schema introspection. Documents
    /// inserted through this path are invisible to indexes built
    /// by typed [`Document`] writers, until a Rust-side
    /// `WriteTxn::collection::<T>()` rewrites them.
    ///
    /// Forwards to [`Self::insert_with_version`] with
    /// `type_version = RAW_BYTES_TYPE_VERSION`. Callers that have
    /// a meaningful schema version to stamp (e.g. a typed binding
    /// layer that knows the document's declared version) should call
    /// [`Self::insert_with_version`] directly.
    ///
    /// # Errors
    ///
    /// - [`Error::AttachedDatabaseIsReadOnly`] for namespaced
    ///   collections (attached dbs are read-only).
    /// - [`Error::DocumentTooLarge`] if `payload.len() + 16` > the
    ///   B-tree inline cap.
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn insert_raw_bytes(&mut self, collection: &str, payload: &[u8]) -> Result<Id> {
        self.insert_with_version(collection, payload, RAW_BYTES_TYPE_VERSION)
    }

    /// **Engine API**: insert a raw-bytes document into `collection`
    /// with a caller-supplied `type_version` stamped in the per-doc
    /// header. Returns the freshly-allocated [`Id`].
    ///
    /// This is the byte-identity entry point for cross-language
    /// writers (a typed binding's `insert(instance)` calls this with
    /// the document's declared version, matching what Rust's
    /// `#[derive(Document)]` stamps via [`obj_core::codec::encode`]).
    /// Apart from the version source, the behaviour is identical to
    /// [`Self::insert_raw_bytes`].
    ///
    /// **Index maintenance does NOT run** — same caveat as the
    /// raw-bytes shim. Use [`Self::collection`] for typed writes
    /// that need index maintenance.
    ///
    /// # Errors
    ///
    /// - [`Error::AttachedDatabaseIsReadOnly`] for namespaced
    ///   collections.
    /// - [`Error::DocumentTooLarge`] if `payload.len() + 16` > the
    ///   B-tree inline cap.
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn insert_with_version(
        &mut self,
        collection: &str,
        payload: &[u8],
        type_version: u32,
    ) -> Result<Id> {
        reject_namespaced_write(collection)?;
        let _ = ensure_collection_raw(&self.inner, &self.catalog, collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let id = obj_core::id::bump_next_id(&mut descriptor.next_id, || collection.to_owned())?;
        let bytes = wrap_raw_payload_with_version(descriptor.collection_id, payload, type_version)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        Ok(id)
    }

    /// **Engine API**: persist the [`DynamicSchema`](obj_core::codec::DynamicSchema) for
    /// `(collection, version)` in the current write transaction.
    ///
    /// Resolves / lazily-creates `collection` exactly as
    /// [`Self::insert_with_version`] does (same per-txn descriptor
    /// cache → `collection_id`), then writes the schema row via
    /// [`Catalog::put_schema`]. The write rides the same WAL txn as the
    /// document body, so a rolled-back insert discards the schema row
    /// with it; the catalog-level write is idempotent + drift-guarded.
    ///
    /// A typed binding calls this on every write so the stored-version
    /// schema is available on disk for a later reader whose live type
    /// is a different (newer) version.
    ///
    /// # Errors
    ///
    /// - [`Error::SchemaShapeChanged`] if a differing shape is already
    ///   persisted under the same key.
    /// - [`Error::SchemaDepthExceeded`] / [`Error::UnsupportedSchemaFormat`]
    ///   from [`Catalog::put_schema`].
    /// - As [`Self::insert_with_version`] (namespaced / pager / catalog).
    pub fn put_schema(
        &mut self,
        collection: &str,
        version: u32,
        schema: &obj_core::codec::DynamicSchema,
    ) -> Result<()> {
        reject_namespaced_write(collection)?;
        let _ = ensure_collection_raw(&self.inner, &self.catalog, collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let mut catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let collection_id = descriptor.collection_id;
        catalog.put_schema(&mut pager, collection_id, version, schema)
    }

    /// **Engine API**: read the persisted [`DynamicSchema`](obj_core::codec::DynamicSchema) for
    /// `(collection, version)` from the writer's own live / pending
    /// catalog tree.
    ///
    /// Descends the catalog tree as mutated in THIS write txn (so it
    /// observes a [`Self::put_schema`] from earlier in the same txn),
    /// then re-specializes the [`StoredSchema`](obj_core::codec::StoredSchema)
    /// into a live [`DynamicSchema`](obj_core::codec::DynamicSchema) via [`obj_core::codec::respecialize`].
    /// Returns `Ok(None)` when no row exists for that key.
    ///
    /// # Errors
    ///
    /// - [`Error::UnsupportedSchemaFormat`] / [`Error::Codec`] if the
    ///   stored row is malformed or in an unknown format.
    /// - As [`Self::get_with_version`] (collection lookup / pager).
    pub fn get_schema(
        &mut self,
        collection: &str,
        version: u32,
    ) -> Result<Option<obj_core::codec::DynamicSchema>> {
        let descriptor = self.live_descriptor_required(collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        match catalog.get_schema_in_txn(&mut pager, descriptor.collection_id, version)? {
            Some(stored) => Ok(Some(obj_core::codec::respecialize(&stored)?)),
            None => Ok(None),
        }
    }

    /// **FFI shim**: fetch the raw payload of the document at `id`
    /// in `collection`. Returns `Ok(None)` if absent. The returned
    /// `Vec<u8>` is the payload only (the 16-byte per-doc header
    /// is stripped).
    ///
    /// Forwards to [`Self::get_with_version`] and discards the
    /// stored version. Use [`Self::get_with_version`] directly when
    /// the caller needs the header's `type_version` (e.g. a typed
    /// binding's read path, which dispatches schema migration on it).
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if the collection is unknown.
    /// - [`Error::Corruption`] if the on-disk record is malformed.
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn get_raw_bytes(&mut self, collection: &str, id: Id) -> Result<Option<Vec<u8>>> {
        Ok(self
            .get_with_version(collection, id)?
            .map(|(payload, _version)| payload))
    }

    /// **Engine API**: fetch the raw payload AND stored
    /// `type_version` of the document at `id` in `collection`.
    /// Returns `Ok(None)` if absent.
    ///
    /// Companion read accessor for the version-aware write path
    /// ([`Self::insert_with_version`]) — used by a typed binding's
    /// read pipeline to dispatch directly on the stored header
    /// version instead of the historical try-decode-walk heuristic.
    ///
    /// # Errors
    ///
    /// As [`Self::get_raw_bytes`].
    #[doc(hidden)]
    pub fn get_with_version(&mut self, collection: &str, id: Id) -> Result<Option<(Vec<u8>, u32)>> {
        let descriptor = self.live_descriptor_required(collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let tree = btree_handle(&pager, descriptor.primary_root)?;
        let key = id.to_be_bytes();
        match tree.get(&mut pager, &key)? {
            Some(bytes) => Ok(Some(strip_raw_payload_with_version(
                &bytes,
                descriptor.collection_id,
            )?)),
            None => Ok(None),
        }
    }

    /// Resolve the LIVE descriptor for `collection`, preferring
    /// the per-txn cache (with its in-memory root advances) over the
    /// catalog tree. Returns `CollectionNotFound` if neither has it.
    /// Returns an owned clone so the caller does not hold the cache
    /// lock across the subsequent pager work.
    fn live_descriptor_required(&self, collection: &str) -> Result<CollectionDescriptor> {
        {
            let cache = crate::collection::lock_descriptors(&self.descriptors)?;
            if let Some(descriptor) = cache.get(collection) {
                return Ok(descriptor.clone());
            }
        }
        catalog_get_required(&self.inner, &self.catalog, collection)
    }

    /// **FFI shim**: update the document at `id` in `collection`
    /// to `payload` bytes. Returns [`Error::DocumentNotFound`] if
    /// the id is absent.
    ///
    /// Forwards to [`Self::update_with_version`] with
    /// `type_version = RAW_BYTES_TYPE_VERSION`.
    ///
    /// # Errors
    ///
    /// - [`Error::DocumentNotFound`] if `id` does not exist.
    /// - [`Error::AttachedDatabaseIsReadOnly`] / [`Error::DocumentTooLarge`]
    ///   etc as [`Self::insert_raw_bytes`].
    #[doc(hidden)]
    pub fn update_raw_bytes(&mut self, collection: &str, id: Id, payload: &[u8]) -> Result<()> {
        self.update_with_version(collection, id, payload, RAW_BYTES_TYPE_VERSION)
    }

    /// **Engine API**: update the document at `id` in `collection`
    /// to `payload` bytes, stamping the per-doc header's
    /// `type_version` with the caller-supplied value.
    ///
    /// Companion to [`Self::insert_with_version`] for the typed
    /// write path.
    ///
    /// # Errors
    ///
    /// As [`Self::update_raw_bytes`].
    #[doc(hidden)]
    pub fn update_with_version(
        &mut self,
        collection: &str,
        id: Id,
        payload: &[u8],
        type_version: u32,
    ) -> Result<()> {
        reject_namespaced_write(collection)?;
        let exists = self.collection_exists(collection)?;
        if !exists {
            return Err(Error::CollectionNotFound {
                name: collection.to_owned(),
            });
        }
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        if tree.get(&mut pager, &key)?.is_none() {
            return Err(Error::CollectionNotFound {
                name: format!("{collection}#{}", id.get()),
            });
        }
        let bytes = wrap_raw_payload_with_version(descriptor.collection_id, payload, type_version)?;
        tree.delete(&mut pager, &key)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        Ok(())
    }

    /// **FFI shim**: delete the document at `id` in `collection`.
    /// Returns `Ok(true)` if it existed, `Ok(false)` if not.
    ///
    /// # Errors
    ///
    /// Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn delete_raw_bytes(&mut self, collection: &str, id: Id) -> Result<bool> {
        reject_namespaced_write(collection)?;
        if !self.collection_exists(collection)? {
            return Err(Error::CollectionNotFound {
                name: collection.to_owned(),
            });
        }
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        let removed = tree.delete(&mut pager, &key)?;
        descriptor.primary_root = tree.root().get();
        Ok(removed)
    }

    /// **Engine API**: count every document in `collection` as seen
    /// inside THIS write transaction — i.e. the count reflects the
    /// txn's own uncommitted inserts / deletes, consistent with
    /// [`Self::get_with_version`]'s live-read semantics.
    ///
    /// Companion to the snapshot-isolated read-side
    /// [`ReadTxn::count_all_raw`]: the write side descends the LIVE
    /// primary B-tree at the per-txn cache's current `primary_root`
    /// (preferring any in-memory root advance from a prior raw write
    /// in this txn) rather than a pinned reader snapshot. Used by
    /// a binding's `count_all()` so a typed collection
    /// handle can count uncommitted state.
    ///
    /// The full-tree scan does not decode records; the iteration
    /// count is bounded by the primary tree's entry count and the
    /// running total is `checked_add`-guarded so a
    /// `> u64::MAX` count surfaces as [`Error::BTreeInvariantViolated`]
    /// rather than wrapping.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if `collection` does not exist.
    /// - Pager / B-tree errors propagated from the descent / scan.
    #[doc(hidden)]
    pub fn count_all_raw(&mut self, collection: &str) -> Result<u64> {
        let descriptor = self.live_descriptor_required(collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let tree = btree_handle(&pager, descriptor.primary_root)?;
        let mut n: u64 = 0;
        for step in tree.range(&mut pager, ..)? {
            let _ = step?;
            n = n.checked_add(1).ok_or(Error::BTreeInvariantViolated {
                reason: "primary tree entry count exceeds u64",
            })?;
        }
        Ok(n)
    }

    /// **FFI shim**: insert-or-replace the document at `id` in
    /// `collection` to `payload` bytes.
    ///
    /// Forwards to [`Self::upsert_with_version`] with
    /// `type_version = RAW_BYTES_TYPE_VERSION`.
    ///
    /// # Errors
    ///
    /// As [`Self::insert_raw_bytes`].
    #[doc(hidden)]
    pub fn upsert_raw_bytes(&mut self, collection: &str, id: Id, payload: &[u8]) -> Result<()> {
        self.upsert_with_version(collection, id, payload, RAW_BYTES_TYPE_VERSION)
    }

    /// **Engine API**: insert-or-replace the document at `id` in
    /// `collection`, stamping the per-doc header's `type_version`
    /// with the caller-supplied value. Companion to
    /// [`Self::insert_with_version`] for the typed upsert path.
    ///
    /// # Errors
    ///
    /// As [`Self::insert_with_version`].
    #[doc(hidden)]
    pub fn upsert_with_version(
        &mut self,
        collection: &str,
        id: Id,
        payload: &[u8],
        type_version: u32,
    ) -> Result<()> {
        reject_namespaced_write(collection)?;
        let _ = ensure_collection_raw(&self.inner, &self.catalog, collection)?;
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let bytes = wrap_raw_payload_with_version(descriptor.collection_id, payload, type_version)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        let _ = tree.delete(&mut pager, &key)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        Ok(())
    }

    /// **Engine API**: insert a raw-bytes document into `collection`
    /// AND maintain the named secondary indexes from caller-supplied
    /// field-encoded keys. Returns the freshly-allocated [`Id`].
    ///
    /// Unlike [`Self::insert_raw_bytes`] (primary-only), this is the
    /// schema-bearing raw write: the C ABI cannot reflect index keys
    /// out of an opaque payload, so the CALLER supplies one
    /// `(index_name, field_key)` entry per index value, where
    /// `field_key` is the order-preserving encoding of the indexed
    /// field (produced by `obj_core::index::encode_field` —
    /// `libobj::obj_index_key_encode` wraps it). obj does the
    /// kind-specific STORAGE-key composition (append the `Id` suffix
    /// for `Standard` / `Each` / `Composite`; use the field key as-is
    /// with the `Id` as value + enforce uniqueness for `Unique`),
    /// matching the typed path byte-for-byte.
    ///
    /// Transaction contract: the primary insert + every index
    /// maintenance lands inside the same WAL transaction. If index
    /// maintenance fails after the primary insert, this method returns
    /// `Err` with the transaction still uncommitted but dirty. Preserve
    /// atomicity by returning that error from [`crate::Db::transaction`]
    /// (which rolls back) or by explicitly calling [`Self::rollback`] /
    /// dropping the manual transaction. Committing after an error commits
    /// whatever had already been staged.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if an entry names an index that does
    ///   not exist or is not `Active` on `collection`.
    /// - [`Error::UniqueConstraintViolation`] if a `Unique` entry's
    ///   key already maps to a different document.
    /// - As [`Self::insert_with_version`] (namespaced / too-large /
    ///   pager / catalog).
    pub fn insert_raw_indexed(
        &mut self,
        collection: &str,
        payload: &[u8],
        type_version: u32,
        entries: &[(String, Vec<u8>)],
    ) -> Result<Id> {
        let id = self.insert_with_version(collection, payload, type_version)?;
        self.maintain_raw_indexes(collection, id, &[], entries)?;
        Ok(id)
    }

    /// **Engine API**: update the document at `id` in `collection` to
    /// `payload` AND move its secondary-index entries from the OLD
    /// caller-supplied field keys to the NEW ones.
    ///
    /// obj cannot re-derive the OLD index keys from the stored opaque
    /// bytes, so the caller MUST supply BOTH the `remove` set (the
    /// field keys the document indexed under before this update) and
    /// the `add` set (the field keys it indexes under after). Each is
    /// one `(index_name, field_key)` entry per index value. The
    /// kind-specific composition + uniqueness enforcement matches
    /// [`Self::insert_raw_indexed`].
    ///
    /// Transaction contract: the primary update + every index
    /// removal/insertion lands in the same WAL transaction. If index
    /// maintenance fails after the primary update, this method returns
    /// `Err` with the transaction still uncommitted but dirty. Preserve
    /// atomicity by returning that error from [`crate::Db::transaction`]
    /// (which rolls back) or by explicitly calling [`Self::rollback`] /
    /// dropping the manual transaction. Committing after an error commits
    /// whatever had already been staged.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if `id` does not exist.
    /// - [`Error::IndexNotFound`] / [`Error::UniqueConstraintViolation`]
    ///   as [`Self::insert_raw_indexed`].
    /// - As [`Self::update_with_version`].
    pub fn update_raw_indexed(
        &mut self,
        collection: &str,
        id: Id,
        payload: &[u8],
        type_version: u32,
        remove: &[(String, Vec<u8>)],
        add: &[(String, Vec<u8>)],
    ) -> Result<()> {
        self.update_with_version(collection, id, payload, type_version)?;
        self.maintain_raw_indexes(collection, id, remove, add)?;
        Ok(())
    }

    /// **Engine API**: delete the document at `id` in `collection`
    /// AND remove its secondary-index entries given the caller-
    /// supplied OLD field keys. Returns `Ok(true)` if the primary
    /// record existed, `Ok(false)` if not.
    ///
    /// As with [`Self::update_raw_indexed`], obj cannot re-derive the
    /// index keys from stored bytes, so the caller supplies the
    /// `remove` set (one `(index_name, field_key)` per indexed value).
    /// The index removals always run (even on `Ok(false)`) so a caller
    /// can repair a known-stale index entry; this mirrors the typed
    /// `Collection::delete` which also diffs against the supplied OLD
    /// key set regardless of primary presence. If an index removal
    /// fails after the primary delete, the transaction is left
    /// uncommitted but dirty; rollback/drop it to preserve atomicity.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if a `remove` entry names an unknown
    ///   / non-`Active` index.
    /// - As [`Self::delete_raw_bytes`].
    pub fn delete_raw_indexed(
        &mut self,
        collection: &str,
        id: Id,
        remove: &[(String, Vec<u8>)],
    ) -> Result<bool> {
        let removed = self.delete_raw_bytes(collection, id)?;
        self.maintain_raw_indexes(collection, id, remove, &[])?;
        Ok(removed)
    }

    /// **Engine API**: declare / reconcile a runtime [`obj_core::IndexSpec`] set
    /// into the catalog for `collection`, making each `Active` BEFORE
    /// any index-maintaining raw write ([`Self::insert_raw_indexed`] &c.
    /// require the index already `Active`).
    ///
    /// This is the NON-generic equivalent of the `#[derive(Document)]`
    /// reconcile path (`WriteTxn::collection::<T>()`, which reflects
    /// `T::indexes()`): a caller that has no Rust `Document` type — the
    /// FFI index-declaration path — supplies the specs
    /// directly. Both share ONE body
    /// (`reconcile_specs_once`) so the cache /
    /// staging / validation / catalog-walk semantics never diverge.
    ///
    /// Lazy-creates the collection's catalog row + empty primary B-tree
    /// on first call (as the typed path does), then runs the same
    /// `shared ∪ staged` skip-cache: a SECOND call with the same
    /// `(collection, version)` is a no-op (the underlying
    /// `Catalog::reconcile_indexes` is itself idempotent for matching
    /// `(name, kind, key_paths)`). The catalog mutation is staged in the
    /// live WAL transaction — a rolled-back txn leaves no half-declared
    /// index, and the per-process reconciled cache is only promoted on a
    /// successful [`Self::commit`].
    ///
    /// # `version`
    ///
    /// The skip-cache is keyed by `(collection, version)`, not by
    /// `collection` alone, so a LATER schema `version` of the same
    /// collection that ADDS an index reconciles on its first call rather
    /// than being skipped. The caller passes the schema version the
    /// `specs` belong to (e.g. the typed `Document::VERSION` or a
    /// binding's declared version). One narrow caveat applies when two
    /// live versions of one collection declare DIFFERENT (conflicting)
    /// index sets and their writes interleave in a single process — see
    /// the `reconcile_specs_once` internal docs. Index ADDITION (the
    /// common monotonic case) is fully correct.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if any spec is malformed (validated
    ///   before any catalog mutation).
    /// - [`Error::IndexKindMismatch`] / [`Error::IndexKeyPathsMismatch`]
    ///   if a spec re-declares an existing `Active` index with a
    ///   different `(kind, key_paths)`.
    /// - [`Error::Busy`] on a poisoned pager / catalog mutex.
    /// - Pager / B-tree / postcard errors propagated.
    pub fn reconcile_indexes_raw(
        &mut self,
        collection: &str,
        version: u32,
        specs: &[obj_core::IndexSpec],
    ) -> Result<()> {
        let _descriptor = ensure_collection_raw(&self.inner, &self.catalog, collection)?;
        crate::collection::reconcile_specs_once(
            &self.inner,
            &self.catalog,
            &self.reconciled,
            &mut self.reconciled_staged,
            collection,
            version,
            specs,
        )
    }

    /// Apply the per-index removal (`old`) + addition (`new`) churn
    /// for a raw-bytes write, composing the storage key per index
    /// kind via the shared non-generic seam
    /// [`crate::index_maint::maintain_index_from_keys`].
    ///
    /// Resolves each `(index_name, field_key)` entry to its `Active`
    /// [`obj_core::IndexDescriptor`] by name, groups the OLD and NEW
    /// field keys per index, and maintains every index touched by
    /// either set. The (possibly COW-advanced) descriptor is persisted
    /// back to the catalog once. Runs entirely under the pager +
    /// catalog locks held for the whole call, inside the live WAL
    /// transaction. A mid-way error is atomic only if the caller
    /// rolls back/drops the transaction (or propagates it out of
    /// `Db::transaction`); committing after an error commits any work
    /// already staged before that error.
    fn maintain_raw_indexes(
        &mut self,
        collection: &str,
        id: Id,
        old: &[(String, Vec<u8>)],
        new: &[(String, Vec<u8>)],
    ) -> Result<()> {
        if old.is_empty() && new.is_empty() {
            return Ok(());
        }
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        let mut cache = crate::collection::lock_descriptors(&self.descriptors)?;
        let descriptor =
            crate::collection::cached_descriptor_mut(&mut cache, &mut pager, &catalog, collection)?;
        let touched = touched_index_names(old, new);
        for index_name in &touched {
            maintain_one_raw_index(&mut pager, descriptor, collection, index_name, old, new, id)?;
        }
        Ok(())
    }

    /// Does `collection` have a catalog row (either already in
    /// the per-txn descriptor cache, or in the live catalog tree)?
    /// Used by the raw update / delete paths to preserve their
    /// `CollectionNotFound` contract before routing the descriptor
    /// through the cache.
    fn collection_exists(&self, collection: &str) -> Result<bool> {
        {
            let cache = crate::collection::lock_descriptors(&self.descriptors)?;
            if cache.contains_key(collection) {
                return Ok(true);
            }
        }
        let mut pager = lock_pager(self.inner.env())?;
        let catalog = lock_catalog(&self.catalog)?;
        Ok(catalog.get(&mut pager, collection)?.is_some())
    }
}

/// Public read transaction.  Acquired by
/// [`crate::Db::read_transaction`].
///
/// Carries only the obj-core read-side handle — the writer's live
/// `Catalog` is NOT held by a `ReadTxn` because reads consult the
/// snapshot-pinned catalog root via
/// [`obj_core::Catalog::lookup_via_snapshot`], not the
/// live `Catalog.tree.root`.
///
/// A `ReadTxn` MAY also carry one `AttachedReadCtx` per
/// attached database registered on the calling [`crate::Db`]. The
/// per-attached snapshots are pinned at txn-begin time and released
/// when the `ReadTxn` drops; reads against `<namespace>.<collection>`
/// route through them.
pub struct ReadTxn<'db> {
    pub(crate) inner: obj_core::ReadTxn<'db, FileHandle>,
    /// Per-attached-database read contexts, keyed by namespace.
    /// Populated by [`crate::Db::read_transaction`] before the
    /// closure runs; emptied when the txn drops.
    pub(crate) attached: HashMap<String, AttachedReadCtx>,
}

impl<'db> ReadTxn<'db> {
    /// Construct a `ReadTxn` from a bare obj-core handle. Public so
    /// the FFI layer can build an owned read txn whose lifetime
    /// extends past a single `Db::read_transaction` closure call.
    ///
    /// User-Rust callers should reach for `Db::read_transaction`.
    #[doc(hidden)]
    #[must_use]
    pub fn from_parts(inner: obj_core::ReadTxn<'db, FileHandle>) -> Self {
        Self {
            inner,
            attached: HashMap::new(),
        }
    }

    pub(crate) fn new(inner: obj_core::ReadTxn<'db, FileHandle>) -> Self {
        Self::from_parts(inner)
    }

    pub(crate) fn with_attached(
        inner: obj_core::ReadTxn<'db, FileHandle>,
        attached: HashMap<String, AttachedReadCtx>,
    ) -> Self {
        Self { inner, attached }
    }

    /// Resolve a (possibly namespaced) collection name to the
    /// `(env, snapshot, lookup_name)` the raw-bytes read shims should
    /// read through. A bare `"collection"` resolves against the
    /// calling Db's own snapshot exactly as before; a
    /// `"<ns>.<tail>"` name resolves against the read-only database
    /// attached under `<ns>` (its pinned snapshot), with the namespace
    /// prefix stripped for the catalog lookup.
    ///
    /// Mirrors the namespace dispatch in
    /// [`crate::collection::Collection::open_readonly_named`] — the
    /// only other namespace-aware read path — so both honour the same
    /// `<ns>.<tail>` → attached-snapshot rule.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNamespaceUnknown`] if `collection`
    ///   carries a namespace prefix that is not attached.
    fn resolve_read_target<'a>(&'a self, collection: &'a str) -> Result<ReadTarget<'a>> {
        let (namespace, tail) = crate::db::try_split_namespace(collection)?;
        match namespace {
            None => Ok(ReadTarget {
                env: self.inner.env(),
                snapshot: self.inner.snapshot(),
                lookup_name: collection,
            }),
            Some(ns) => {
                let ctx =
                    self.attached
                        .get(ns)
                        .ok_or_else(|| Error::CollectionNamespaceUnknown {
                            namespace: ns.to_owned(),
                        })?;
                Ok(ReadTarget {
                    env: ctx.env.as_ref(),
                    snapshot: &ctx.snapshot,
                    lookup_name: tail,
                })
            }
        }
    }

    /// **FFI shim**: fetch the raw payload of the document at `id`
    /// in `collection`, snapshot-consistent against the read txn's
    /// pinned LSN. Returns `Ok(None)` if absent.
    ///
    /// Forwards to [`Self::get_with_version`] and discards the
    /// stored version.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if the collection is unknown.
    /// - [`Error::Corruption`] if the on-disk record is malformed.
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn get_raw_bytes(&self, collection: &str, id: Id) -> Result<Option<Vec<u8>>> {
        Ok(self
            .get_with_version(collection, id)?
            .map(|(payload, _version)| payload))
    }

    /// **Engine API**: fetch the raw payload AND stored
    /// `type_version` of the document at `id` in `collection`,
    /// snapshot-consistent against the read txn's pinned LSN.
    /// Returns `Ok(None)` if absent.
    ///
    /// Companion read accessor for the version-aware write path
    /// ([`WriteTxn::insert_with_version`]) — used by a typed binding's
    /// read pipeline to dispatch on the stored header
    /// version instead of the historical try-decode-walk heuristic.
    ///
    /// # Errors
    ///
    /// As [`Self::get_raw_bytes`].
    #[doc(hidden)]
    pub fn get_with_version(&self, collection: &str, id: Id) -> Result<Option<(Vec<u8>, u32)>> {
        let target = self.resolve_read_target(collection)?;
        let descriptor = target.collection_descriptor(collection)?;
        let pager = lock_pager(target.env)?;
        let root = PageId::new(descriptor.primary_root)
            .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
        let key = id.to_be_bytes();
        let bytes = obj_core::btree::BTree::<FileHandle>::get_via_snapshot(
            &pager,
            target.snapshot,
            root,
            &key,
        )?;
        match bytes {
            Some(b) => Ok(Some(strip_raw_payload_with_version(
                &b,
                descriptor.collection_id,
            )?)),
            None => Ok(None),
        }
    }

    /// **Engine API**: read the persisted [`DynamicSchema`](obj_core::codec::DynamicSchema) for
    /// `(collection, version)` as-of this read txn's pinned snapshot.
    ///
    /// Snapshot-isolated counterpart to [`WriteTxn::get_schema`]: it
    /// looks the [`StoredSchema`](obj_core::codec::StoredSchema) row up
    /// via [`obj_core::lookup_schema_via_snapshot`] against the pinned
    /// catalog root (honouring `<ns>.<tail>` attached-db resolution),
    /// then re-specializes it into a live [`DynamicSchema`](obj_core::codec::DynamicSchema) via
    /// [`obj_core::codec::respecialize`]. Returns `Ok(None)` when no row
    /// exists for that key. A binding's migration read path calls this only
    /// when the stored record's version differs from the live type's.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if the collection is unknown at
    ///   the snapshot.
    /// - [`Error::UnsupportedSchemaFormat`] / [`Error::Codec`] if the
    ///   stored row is malformed or in an unknown format.
    /// - Pager / B-tree errors propagated.
    pub fn get_schema(
        &self,
        collection: &str,
        version: u32,
    ) -> Result<Option<obj_core::codec::DynamicSchema>> {
        let target = self.resolve_read_target(collection)?;
        let descriptor = target.collection_descriptor(collection)?;
        let pager = lock_pager(target.env)?;
        let stored = obj_core::lookup_schema_via_snapshot(
            &pager,
            target.snapshot,
            descriptor.collection_id,
            version,
        )?;
        match stored {
            Some(s) => Ok(Some(obj_core::codec::respecialize(&s)?)),
            None => Ok(None),
        }
    }

    /// **FFI shim**: look up the descriptor for `collection`
    /// against the snapshot. Returns `Ok(None)` if absent.
    ///
    /// Used by [`libobj`](../../libobj/index.html) for query /
    /// iteration entry points.
    ///
    /// # Errors
    ///
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn snapshot_descriptor(&self, collection: &str) -> Result<Option<CollectionDescriptor>> {
        read_descriptor_via_snapshot(self.inner.env(), self.inner.snapshot(), collection)
    }

    /// **FFI shim**: borrow the wrapped obj-core read txn. Used by
    /// [`libobj`](../../libobj/index.html) iterators that need
    /// snapshot-aware B-tree access.
    #[doc(hidden)]
    #[must_use]
    pub fn inner(&self) -> &obj_core::ReadTxn<'db, FileHandle> {
        &self.inner
    }

    /// **FFI shim**: resolve an `Active` index descriptor by name
    /// on `collection`. Used by libobj's range / find_unique /
    /// count paths.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if the collection is absent.
    /// - [`Error::IndexNotFound`] if the index is unknown or
    ///   `DroppedPending`.
    /// - Pager / catalog errors propagated.
    #[doc(hidden)]
    pub fn snapshot_index_descriptor(
        &self,
        collection: &str,
        index: &str,
    ) -> Result<obj_core::IndexDescriptor> {
        let descriptor =
            self.snapshot_descriptor(collection)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: collection.to_owned(),
                })?;
        let entry = descriptor.indexes.iter().find(|d| d.name == index);
        match entry {
            Some(d) if d.status == obj_core::IndexStatus::Active => Ok(d.clone()),
            _ => Err(Error::IndexNotFound {
                collection: collection.to_owned(),
                name: index.to_owned(),
            }),
        }
    }

    /// **FFI shim**: count every doc in `collection` snapshot-
    /// consistently against the read txn's pinned LSN.
    ///
    /// # Errors
    ///
    /// As [`Self::snapshot_descriptor`] plus pager / B-tree.
    #[doc(hidden)]
    pub fn count_all_raw(&self, collection: &str) -> Result<u64> {
        let target = self.resolve_read_target(collection)?;
        let descriptor = target.collection_descriptor(collection)?;
        count_via_btree_range_full(target.env, target.snapshot, descriptor.primary_root)
    }

    /// **FFI shim**: collect every `(Id, raw_payload)` pair in
    /// `collection`, snapshot-consistently against this read txn's
    /// pinned LSN.
    ///
    /// Materialises the full result so the C iterator handle can
    /// outlive the read-txn pointer it was constructed from while
    /// still exposing only rows visible at this transaction's
    /// snapshot.
    ///
    /// # Errors
    ///
    /// As [`Self::count_all_raw`] plus record-header validation.
    #[doc(hidden)]
    pub fn all_raw(&self, collection: &str) -> Result<Vec<(Id, Vec<u8>)>> {
        let target = self.resolve_read_target(collection)?;
        let descriptor = target.collection_descriptor(collection)?;
        collect_all_raw_pairs(target.env, target.snapshot, &descriptor)
    }

    /// **FFI shim**: walk an index B-tree by raw-byte key range
    /// and collect the matching `(Id, raw_payload)` pairs. The
    /// caller is responsible for encoding `lower` / `upper` per the
    /// order-preserving encoding.
    ///
    /// `lower_bound` / `upper_bound` use Rust's `std::ops::Bound`
    /// shape (Included / Excluded / Unbounded).
    ///
    /// Materialises every result in a `Vec` — the result set is
    /// bounded by [`obj_core::btree::MAX_RANGE_NODES`] inherited
    /// from `BTree::range`. The libobj iterator yields these one
    /// at a time.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] / [`Error::CollectionNotFound`].
    /// - Pager / B-tree errors propagated.
    #[doc(hidden)]
    pub fn index_range_raw(
        &self,
        collection: &str,
        index: &str,
        lower: std::ops::Bound<Vec<u8>>,
        upper: std::ops::Bound<Vec<u8>>,
    ) -> Result<Vec<(Id, Vec<u8>)>> {
        let target = self.resolve_read_target(collection)?;
        let index_descriptor = target.index_descriptor(collection, index)?;
        let collection_descriptor = target.collection_descriptor(collection)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(lower, upper, index_descriptor.kind);
        let entries = collect_index_range_entries(
            target.env,
            target.snapshot,
            index_descriptor.root_page_id,
            start,
            end,
        )?;
        materialize_id_payload_pairs(
            target.env,
            target.snapshot,
            &collection_descriptor,
            &index_descriptor,
            entries,
        )
    }

    /// **Engine API**: walk an index B-tree by raw-byte key range and
    /// collect the matching `(Id, type_version, raw_payload)` rows.
    /// Companion to [`Self::index_range_raw`] that ALSO surfaces each
    /// record's stored `type_version` (read from the per-doc record
    /// header), so a typed range decode can dispatch schema migration at
    /// the version each record was actually written under — exactly as
    /// [`Self::find_unique_with_version`] does for the single-key path.
    ///
    /// `lower` / `upper` follow the same raw-byte convention as
    /// [`Self::index_range_raw`]; the result set is bounded identically.
    ///
    /// # Errors
    ///
    /// As [`Self::index_range_raw`].
    #[doc(hidden)]
    pub fn index_range_raw_with_version(
        &self,
        collection: &str,
        index: &str,
        lower: std::ops::Bound<Vec<u8>>,
        upper: std::ops::Bound<Vec<u8>>,
    ) -> Result<Vec<(Id, u32, Vec<u8>)>> {
        let target = self.resolve_read_target(collection)?;
        let index_descriptor = target.index_descriptor(collection, index)?;
        let collection_descriptor = target.collection_descriptor(collection)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(lower, upper, index_descriptor.kind);
        let entries = collect_index_range_entries(
            target.env,
            target.snapshot,
            index_descriptor.root_page_id,
            start,
            end,
        )?;
        materialize_id_version_payload_rows(
            target.env,
            target.snapshot,
            &collection_descriptor,
            &index_descriptor,
            entries,
        )
    }

    /// **FFI shim**: count index B-tree entries inside `range`.
    /// `lower` / `upper` follow the same raw-byte convention as
    /// [`Self::index_range_raw`].
    ///
    /// # Errors
    ///
    /// As [`Self::index_range_raw`].
    #[doc(hidden)]
    pub fn count_index_range_raw(
        &self,
        collection: &str,
        index: &str,
        lower: std::ops::Bound<Vec<u8>>,
        upper: std::ops::Bound<Vec<u8>>,
    ) -> Result<u64> {
        let target = self.resolve_read_target(collection)?;
        let index_descriptor = target.index_descriptor(collection, index)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(lower, upper, index_descriptor.kind);
        let entries = collect_index_range_entries(
            target.env,
            target.snapshot,
            index_descriptor.root_page_id,
            start,
            end,
        )?;
        u64::try_from(entries.len()).map_err(|_| Error::BTreeInvariantViolated {
            reason: "index range entry count exceeds u64",
        })
    }

    /// **FFI shim**: single-key lookup against a `Unique` index.
    /// Returns the matched `(Id, payload)` or `Ok(None)`.
    ///
    /// `key_bytes` is the index key, pre-encoded by the caller per
    /// the order-preserving scheme.
    ///
    /// Forwards to [`Self::find_unique_with_version`] and discards
    /// the stored version.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotUnique`] if the index is not `Unique`.
    /// - As [`Self::index_range_raw`].
    #[doc(hidden)]
    pub fn find_unique_raw(
        &self,
        collection: &str,
        index: &str,
        key_bytes: &[u8],
    ) -> Result<Option<(Id, Vec<u8>)>> {
        Ok(self
            .find_unique_with_version(collection, index, key_bytes)?
            .map(|(id, payload, _version)| (id, payload)))
    }

    /// **Engine API**: single-key lookup against a `Unique` index,
    /// returning the matched `(Id, payload, type_version)` or
    /// `Ok(None)`. Companion to [`Self::get_with_version`] for the
    /// typed find path.
    ///
    /// # Errors
    ///
    /// As [`Self::find_unique_raw`].
    #[doc(hidden)]
    pub fn find_unique_with_version(
        &self,
        collection: &str,
        index: &str,
        key_bytes: &[u8],
    ) -> Result<Option<(Id, Vec<u8>, u32)>> {
        let target = self.resolve_read_target(collection)?;
        let index_descriptor = target.index_descriptor(collection, index)?;
        if index_descriptor.kind != obj_core::IndexKind::Unique {
            return Err(Error::IndexNotUnique {
                collection: collection.to_owned(),
                name: index.to_owned(),
            });
        }
        let id_bytes = {
            let pager = lock_pager(target.env)?;
            let root = PageId::new(index_descriptor.root_page_id)
                .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
            BTree::<FileHandle>::get_via_snapshot(&pager, target.snapshot, root, key_bytes)?
        };
        match id_bytes {
            Some(bytes) => {
                let id = Id::from_be_bytes(&bytes).ok_or(Error::Corruption { page_id: 0 })?;
                match self.get_with_version(collection, id)? {
                    Some((payload, version)) => Ok(Some((id, payload, version))),
                    None => Err(Error::Corruption { page_id: 0 }),
                }
            }
            None => Ok(None),
        }
    }

    /// Open a typed handle to the collection `T` lives in.
    ///
    /// Read-only: returns [`Error::CollectionNotFound`] if the
    /// collection has never been registered AT THE SNAPSHOT'S
    /// PINNED LSN.
    ///
    /// If `T::COLLECTION` is of the form `<namespace>.<name>`, the
    /// txn dispatches against the attached database registered under
    /// `<namespace>` instead of the calling Db.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if `T::COLLECTION` is not
    ///   registered in the catalog as-of the snapshot's pinned LSN.
    /// - [`Error::CollectionNamespaceUnknown`] if `T::COLLECTION`
    ///   carries a namespace prefix that is not attached.
    /// - [`Error::Busy`] if the pager / catalog mutex is poisoned.
    pub fn collection<T: Document>(&self) -> Result<Collection<'_, T>> {
        Collection::open_readonly(self)
    }
}

/// The `(env, snapshot, lookup_name)` triple a raw-bytes read shim
/// should read through, produced by [`ReadTxn::resolve_read_target`].
///
/// For a bare collection name the fields point at the calling Db's
/// own env / pinned snapshot; for a `<ns>.<tail>` name they point at
/// the attached read-only Db registered under `<ns>` and `lookup_name`
/// is the namespace-stripped `<tail>`. Every field borrows the
/// `ReadTxn` for `'a`, so the descriptor / B-tree reads stay pinned to
/// the same snapshot for the whole shim call.
struct ReadTarget<'a> {
    env: &'a TxnEnv<FileHandle>,
    snapshot: &'a ReaderSnapshot<FileHandle>,
    lookup_name: &'a str,
}

impl ReadTarget<'_> {
    /// Resolve the collection descriptor for [`Self::lookup_name`]
    /// against this target's snapshot, surfacing
    /// [`Error::CollectionNotFound`] (under the ORIGINAL,
    /// possibly-namespaced name) when the collection is absent.
    fn collection_descriptor(&self, original: &str) -> Result<CollectionDescriptor> {
        read_descriptor_via_snapshot(self.env, self.snapshot, self.lookup_name)?.ok_or_else(|| {
            Error::CollectionNotFound {
                name: original.to_owned(),
            }
        })
    }

    /// Resolve the `Active` index descriptor named `index` on
    /// [`Self::lookup_name`] against this target's snapshot. The
    /// `original` (possibly-namespaced) name is reported in
    /// [`Error::CollectionNotFound`] / [`Error::IndexNotFound`] so
    /// the caller sees the name it asked for.
    fn index_descriptor(&self, original: &str, index: &str) -> Result<obj_core::IndexDescriptor> {
        let descriptor = self.collection_descriptor(original)?;
        let entry = descriptor.indexes.iter().find(|d| d.name == index);
        match entry {
            Some(d) if d.status == obj_core::IndexStatus::Active => Ok(d.clone()),
            _ => Err(Error::IndexNotFound {
                collection: original.to_owned(),
                name: index.to_owned(),
            }),
        }
    }
}

/// Per-attached-database read context carried inside a [`ReadTxn`].
/// Pins one [`ReaderSnapshot`] against the attached database for the
/// duration of the calling Db's read transaction.
pub(crate) struct AttachedReadCtx {
    /// Calling-side stable reference to the attached env. Cloned
    /// from the [`AttachedDb`]'s `env` at txn-begin time so the
    /// `ReadTxn` does not retain a borrow on the calling Db's
    /// attached-registry mutex.
    pub(crate) env: Arc<TxnEnv<FileHandle>>,
    /// Snapshot pinned at txn-begin against `env`.
    pub(crate) snapshot: ReaderSnapshot<FileHandle>,
}

/// One attached read-only database registered on the calling
/// [`crate::Db`] under a namespace.
///
/// Created by [`crate::Db::attach`]; stored inside the calling Db's
/// `attached: Arc<Mutex<HashMap<String, AttachedDb>>>` registry.
/// Removed by [`crate::Db::detach`] (or by the calling Db's drop,
/// which transitively drops all attachments).
pub(crate) struct AttachedDb {
    /// Cloned `Arc<TxnEnv>` of the attached db, so read transactions
    /// pin snapshots without re-locking the registry.
    pub(crate) env: Arc<TxnEnv<FileHandle>>,
    /// Calling-side keepalive for the attached `crate::Db`. The
    /// underscore prefix marks it as held-for-side-effect: dropping
    /// this `_db` releases file locks and any other resources the
    /// attached open acquired.
    pub(crate) _db: crate::Db,
}

/// Fold a committed txn's staged reconciled `(collection,
/// version)` keys into the shared per-process `reconciled` set. Called
/// by [`WriteTxn::commit`] only AFTER the WAL commit has landed, so the
/// shared cache is never poisoned by a rolled-back lazy-create.
///
/// A no-op (no lock acquired) when nothing was staged — the common
/// path for a txn that opened only already-reconciled collections, so
/// the fast "already reconciled" path pays nothing extra.
///
/// A poisoned `reconciled` mutex maps to [`Error::Busy`]; the
/// loop is bounded by the staged-key count (one entry per distinct
/// `(collection, version)` lazily reconciled in the txn).
fn promote_reconciled(
    reconciled: &Mutex<HashSet<(String, u32)>>,
    staged: HashSet<(String, u32)>,
) -> Result<()> {
    if staged.is_empty() {
        return Ok(());
    }
    let mut shared = reconciled.lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })?;
    shared.extend(staged);
    Ok(())
}

/// Acquire the catalog mutex; convert a poison error into a
/// `WriterInProcess` Busy.  Helper shared by the public txn wrappers
/// and the [`Collection`] internals.
pub(crate) fn lock_catalog(
    catalog: &Mutex<Catalog<FileHandle>>,
) -> Result<std::sync::MutexGuard<'_, Catalog<FileHandle>>> {
    catalog.lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

/// Reject a write against a namespaced collection. Attached
/// databases are read-only through the calling Db.
fn reject_namespaced_write(collection: &str) -> Result<()> {
    if let (Some(namespace), tail) = crate::db::try_split_namespace(collection)? {
        return Err(Error::AttachedDatabaseIsReadOnly {
            namespace: namespace.to_owned(),
            collection: tail.to_owned(),
        });
    }
    Ok(())
}

/// Acquire the env's pager mutex; map poison into Busy.
fn lock_pager(env: &TxnEnv<FileHandle>) -> Result<std::sync::MutexGuard<'_, Pager<FileHandle>>> {
    env.pager().lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

/// Collect the de-duplicated set of index names touched by either the
/// `old` (remove) or `new` (add) entry list, preserving first-seen
/// order. Bounded by `old.len() + new.len()` — the caller's supplied
/// entry count.
fn touched_index_names(old: &[(String, Vec<u8>)], new: &[(String, Vec<u8>)]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (name, _key) in old.iter().chain(new.iter()) {
        if !names.iter().any(|n| n == name) {
            names.push(name.clone());
        }
    }
    names
}

/// Gather every field-encoded key in `entries` whose index name
/// equals `index_name`, wrapped as [`EncodedIndexKey`] for the
/// composition seam. Order follows the caller's entry order.
fn keys_for_index(entries: &[(String, Vec<u8>)], index_name: &str) -> Vec<EncodedIndexKey> {
    entries
        .iter()
        .filter(|(name, _key)| name == index_name)
        .map(|(_name, key)| EncodedIndexKey::from_bytes(key.clone()))
        .collect()
}

/// Resolve `index_name` to its `Active` descriptor index on
/// `descriptor`, then maintain that one index B-tree by diffing the
/// `old` field keys against the `new` ones through the shared
/// non-generic composition seam
/// ([`crate::index_maint::maintain_index_from_keys`]).
///
/// An unknown or non-`Active` index name is [`Error::IndexNotFound`]
/// — the raw write refuses rather than silently dropping the entry.
fn maintain_one_raw_index(
    pager: &mut Pager<FileHandle>,
    descriptor: &mut CollectionDescriptor,
    collection: &str,
    index_name: &str,
    old: &[(String, Vec<u8>)],
    new: &[(String, Vec<u8>)],
    id: Id,
) -> Result<()> {
    let idx = descriptor
        .indexes
        .iter()
        .position(|d| d.name == index_name && d.status == IndexStatus::Active)
        .ok_or_else(|| Error::IndexNotFound {
            collection: collection.to_owned(),
            name: index_name.to_owned(),
        })?;
    let spec = crate::index_maint::descriptor_to_spec(&descriptor.indexes[idx])?;
    let old_keys = keys_for_index(old, index_name);
    let new_keys = keys_for_index(new, index_name);
    crate::index_maint::maintain_index_from_keys(
        pager, descriptor, idx, collection, &spec, &old_keys, &new_keys, id,
    )
}

/// Ensure a (raw-bytes) collection exists, lazy-creating an empty
/// primary B-tree on first call. Used by the C ABI's insert /
/// upsert paths. Distinct from `Collection::open_or_create` because
/// raw-bytes writes do NOT participate in the typed-reconciliation
/// path — no `T::indexes()` to walk.
fn ensure_collection_raw(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
    name: &str,
) -> Result<CollectionDescriptor> {
    let mut pager = inner.lock_pager()?;
    let mut catalog_guard = lock_catalog(catalog)?;
    if let Some(d) = catalog_guard.get(&mut pager, name)? {
        return Ok(d);
    }
    let tree = BTree::<FileHandle>::empty(&mut pager)?;
    let descriptor = CollectionDescriptor::new(0, tree.root().get(), RAW_BYTES_TYPE_VERSION);
    let _id = catalog_guard.insert(&mut pager, name, descriptor)?;
    catalog_guard
        .get(&mut pager, name)?
        .ok_or(Error::Corruption { page_id: 0 })
}

/// Look up the descriptor for `name`, returning
/// `Err(CollectionNotFound)` if absent. Used by `update` / `delete`
/// where lazy-create would mask the caller's mistake.
fn catalog_get_required(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
    name: &str,
) -> Result<CollectionDescriptor> {
    let mut pager = inner.lock_pager()?;
    let catalog_guard = lock_catalog(catalog)?;
    catalog_guard
        .get(&mut pager, name)?
        .ok_or_else(|| Error::CollectionNotFound {
            name: name.to_owned(),
        })
}

/// Snapshot-aware descriptor lookup on the read side. Returns
/// `Ok(None)` when the collection is absent at the snapshot's
/// pinned LSN.
fn read_descriptor_via_snapshot(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    name: &str,
) -> Result<Option<CollectionDescriptor>> {
    let pager = lock_pager(env)?;
    Catalog::<FileHandle>::lookup_via_snapshot(&pager, snapshot, name)
}

/// Open a primary-tree handle from a descriptor's `primary_root`.
fn btree_handle(pager: &Pager<FileHandle>, root: u64) -> Result<BTree<FileHandle>> {
    let root_pid =
        PageId::new(root).ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    BTree::<FileHandle>::open(pager, root_pid)
}

/// Wrap a raw payload with the on-disk [`DocumentHeader`] carrying
/// the caller-supplied `type_version`. The header stamps
/// `collection_id` (so a cross-collection forgery is detectable on
/// read), `type_version`, `payload_len`, and a CRC32C of the
/// payload.
///
/// The on-disk bytes produced here are byte-identical to what
/// [`obj_core::codec::encode`] emits for the same logical payload +
/// `type_version` — this is the key plumbing point that closes the
/// cross-language header-level interop gap.
///
/// Returns [`Error::DocumentTooLarge`] if `payload.len() + 16` would
/// not fit inline in a B-tree leaf — mirrors
/// [`obj_core::codec::encode`]'s overflow handling.
fn wrap_raw_payload_with_version(
    collection_id: u32,
    payload: &[u8],
    type_version: u32,
) -> Result<Vec<u8>> {
    let payload_len = u32::try_from(payload.len()).map_err(|_| Error::DocumentTooLarge {
        len: payload.len(),
        max: MAX_INLINE_DOC,
    })?;
    let total = DOC_HEADER_SIZE
        .checked_add(payload.len())
        .ok_or(Error::DocumentTooLarge {
            len: usize::MAX,
            max: MAX_INLINE_DOC,
        })?;
    if total > MAX_INLINE_DOC {
        return Err(Error::DocumentTooLarge {
            len: total,
            max: MAX_INLINE_DOC,
        });
    }
    let header = DocumentHeader {
        collection_id,
        type_version,
        payload_len,
        payload_crc32c: crc32c(payload),
    };
    let mut out = Vec::with_capacity(total);
    header.write_to(&mut out);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Strip the per-doc header and return `(payload, type_version)`.
/// Validates the header's `collection_id`, total length, and
/// payload CRC32C — surfaces [`Error::Corruption`] /
/// [`Error::CollectionIdMismatch`] on any mismatch. Exposes the
/// stored `type_version` so the typed read path can dispatch on
/// it directly.
fn strip_raw_payload_with_version(
    bytes: &[u8],
    expected_collection_id: u32,
) -> Result<(Vec<u8>, u32)> {
    let header = DocumentHeader::read_from(bytes)?;
    if header.collection_id != expected_collection_id {
        return Err(Error::CollectionIdMismatch {
            expected: expected_collection_id,
            found: header.collection_id,
        });
    }
    let payload_len =
        usize::try_from(header.payload_len).map_err(|_| Error::Corruption { page_id: 0 })?;
    let total = DOC_HEADER_SIZE
        .checked_add(payload_len)
        .ok_or(Error::Corruption { page_id: 0 })?;
    if bytes.len() != total {
        return Err(Error::Corruption { page_id: 0 });
    }
    let payload = &bytes[DOC_HEADER_SIZE..total];
    if crc32c(payload) != header.payload_crc32c {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok((payload.to_vec(), header.type_version))
}

/// Count every entry in a primary B-tree without decoding the
/// records. Used by the FFI `count_all_raw` path.
///
/// Snapshot-pinned: the full-tree scan resolves every page
/// read as-of the read txn's `snapshot`, so a concurrent writer's
/// post-snapshot inserts/deletes cannot perturb the count — it stays
/// consistent with the read txn's pinned LSN.
fn count_via_btree_range_full(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    primary_root: u64,
) -> Result<u64> {
    let pager = lock_pager(env)?;
    let root = PageId::new(primary_root)
        .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    let iter = BTree::<FileHandle>::range_via_snapshot(&pager, snapshot, root, ..)?;
    let mut n: u64 = 0;
    for step in iter {
        let _ = step?;
        n = n.checked_add(1).ok_or(Error::BTreeInvariantViolated {
            reason: "primary tree entry count exceeds u64",
        })?;
    }
    Ok(n)
}

/// Collect every primary-tree row as `(Id, raw_payload)`, reading the
/// tree through a supplied snapshot.
fn collect_all_raw_pairs(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    descriptor: &CollectionDescriptor,
) -> Result<Vec<(Id, Vec<u8>)>> {
    let pager = lock_pager(env)?;
    let root = PageId::new(descriptor.primary_root)
        .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    let iter = BTree::<FileHandle>::range_via_snapshot(&pager, snapshot, root, ..)?;
    let mut out: Vec<(Id, Vec<u8>)> = Vec::new();
    for step in iter {
        let (key, value) = step?;
        let id = Id::from_be_bytes(&key)
            .ok_or(Error::InvalidArgument("primary B-tree key is not an Id"))?;
        let (payload, _version) = strip_raw_payload_with_version(&value, descriptor.collection_id)?;
        out.push((id, payload));
    }
    Ok(out)
}

/// Walk an index B-tree by raw-byte key range and return every
/// (`full_key`, `value_bytes`) entry inside the range. Used by the
/// FFI `index_range_raw` / `count_index_range_raw` paths.
///
/// Snapshot-pinned: the descent and leaf-scan resolve every
/// page read as-of the read txn's `snapshot`, so a concurrent writer's
/// post-snapshot index entries do not leak into the range/count — the
/// enumeration stays consistent with the read txn's pinned LSN.
fn collect_index_range_entries(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    index_root: u64,
    start: std::ops::Bound<Vec<u8>>,
    end: std::ops::Bound<Vec<u8>>,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let pager = lock_pager(env)?;
    let root =
        PageId::new(index_root).ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
    let iter = BTree::<FileHandle>::range_via_snapshot(&pager, snapshot, root, (start, end))?;
    let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for step in iter {
        out.push(step?);
    }
    Ok(out)
}

/// Resolve a list of (`full_key`, `value_bytes`) index entries into
/// `(Id, raw_payload)` pairs. For `Unique` indexes the id is the
/// VALUE; for other kinds it is the trailing 8 bytes of the key.
///
/// Snapshot-aware: each primary lookup goes through the read
/// txn's pinned snapshot so the result set is consistent across
/// the call.
fn materialize_id_payload_pairs(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    collection: &CollectionDescriptor,
    index: &obj_core::IndexDescriptor,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<(Id, Vec<u8>)>> {
    let rows = materialize_id_version_payload_rows(env, snapshot, collection, index, entries)?;
    Ok(rows
        .into_iter()
        .map(|(id, _version, payload)| (id, payload))
        .collect())
}

/// Resolve a list of (`full_key`, `value_bytes`) index entries into
/// `(Id, type_version, raw_payload)` rows — the version-carrying form
/// of [`materialize_id_payload_pairs`]. The stored `type_version` comes
/// from each record's header (via [`strip_raw_payload_with_version`]),
/// letting the typed value-form `index_range` decode dispatch migration
/// at the version each record was written under. De-duplication +
/// id-resolution semantics are identical to the pairs form.
fn materialize_id_version_payload_rows(
    env: &TxnEnv<FileHandle>,
    snapshot: &ReaderSnapshot<FileHandle>,
    collection: &CollectionDescriptor,
    index: &obj_core::IndexDescriptor,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<(Id, u32, Vec<u8>)>> {
    let mut out: Vec<(Id, u32, Vec<u8>)> = Vec::with_capacity(entries.len());
    let mut emitted: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let primary_root = PageId::new(collection.primary_root)
        .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    let pager = lock_pager(env)?;
    for (full_key, value) in entries {
        let id_u64 = index_entry_id(index.kind, &full_key, &value)?;
        if index.kind == obj_core::IndexKind::Each && !emitted.insert(id_u64) {
            continue;
        }
        if index.kind != obj_core::IndexKind::Each {
            emitted.insert(id_u64);
        }
        let id = Id::try_new(id_u64).ok_or(Error::Corruption { page_id: 0 })?;
        let primary_bytes = BTree::<FileHandle>::get_via_snapshot(
            &pager,
            snapshot,
            primary_root,
            &id.to_be_bytes(),
        )?
        .ok_or(Error::Corruption { page_id: 0 })?;
        let (payload, version) =
            strip_raw_payload_with_version(&primary_bytes, collection.collection_id)?;
        out.push((id, version, payload));
    }
    Ok(out)
}

/// Extract the document `Id` (as `u64`) an index entry points at. For
/// `Unique` indexes the id is the entry VALUE; for every other kind it
/// is the trailing 8 bytes of the full key. Shared by both materialisers.
fn index_entry_id(kind: obj_core::IndexKind, full_key: &[u8], value: &[u8]) -> Result<u64> {
    if kind == obj_core::IndexKind::Unique {
        return Ok(Id::from_be_bytes(value)
            .ok_or(Error::Corruption { page_id: 0 })?
            .get());
    }
    if full_key.len() < 8 {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok(Id::from_be_bytes(&full_key[full_key.len() - 8..])
        .ok_or(Error::Corruption { page_id: 0 })?
        .get())
}
