//! `Collection<T>` — typed handle to one collection's primary
//! B-tree.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::{Arc, Mutex, MutexGuard};

use std::collections::BTreeMap;

use obj_core::btree::BTree;
use obj_core::codec::{decode_with, encode, DocumentHeader, Schema, SchemaSource, StoredSchema};
use obj_core::pager::page::PageId;
use obj_core::pager::Pager;
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor, Document, Error, Id, Result};

/// Boxed iterator alias used by [`Collection::lookup`] and
/// [`Collection::index_range`]. The iterator borrows from the
/// enclosing transaction; the `'tx` lifetime is bound on the
/// `Collection<T>` it was obtained from.
pub type IndexIter<'a, Item> = Box<dyn Iterator<Item = Result<Item>> + Send + 'a>;

/// Per-batch refill size for [`IterIndexRange`]. The iterator yields
/// one `(user_key, T)` pair at a time but pulls index B-tree entries
/// in fixed-size chunks so the per-step pager-lock acquisition cost
/// amortises across many `next()` calls. The buffer is fixed-size —
/// at ~8 bytes/key plus the trailing 8-byte id suffix that's ~4 KiB
/// peak for the staged batch. The buffer does NOT scale with the
/// range's total size.
const ITER_INDEX_RANGE_BATCH: usize = 256;

/// Per-call cap on the bounded `HashSet<Id>` used by
/// [`Collection::count_distinct_ids_in_range`] to count unique
/// document `Id`s under an `Each` index. The distinct set is
/// allocation-bounded; exceeding the cap surfaces
/// [`obj_core::Error::DistinctCountExceeded`] rather than chewing
/// arbitrary memory. The user can narrow the range via
/// `.index_range(...)` to fit inside the budget.
pub const MAX_DISTINCT_IDS: usize = 100_000;

use crate::txn::{lock_catalog, ReadTxn, WriteTxn};

/// Per-transaction descriptor cache. Maps collection name to
/// its LIVE [`CollectionDescriptor`] (with `next_id`, `primary_root`,
/// and every index `root_page_id` advanced IN-MEMORY across the
/// transaction). This is the single mid-txn source of truth for those
/// roots; [`crate::WriteTxn::commit`] flushes each entry back to the
/// catalog exactly once. Shared (via `Arc`) between the [`WriteTxn`]
/// and every [`Collection`] handle opened on it, so two handles of the
/// same collection observe the same advancing descriptor.
pub(crate) type DescriptorCache = Arc<Mutex<HashMap<String, CollectionDescriptor>>>;

/// Construct an empty [`DescriptorCache`].
#[must_use]
pub(crate) fn new_descriptor_cache() -> DescriptorCache {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Acquire the descriptor-cache mutex; map poison into `Busy` — no
/// panic on a poisoned lock.
pub(crate) fn lock_descriptors(
    cache: &Mutex<HashMap<String, CollectionDescriptor>>,
) -> Result<MutexGuard<'_, HashMap<String, CollectionDescriptor>>> {
    cache.lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

/// Resolve the live cached descriptor for `name`, lazily loading it
/// from the catalog B-tree on first touch in this transaction. After
/// the first load the catalog tree is NEVER re-read mid-txn for this
/// collection — every subsequent read/advance goes through the cache
/// entry, so the unique pre-check and every index-tree open observe
/// the in-memory-advanced roots (a load-bearing invariant).
///
/// Returns a mutable borrow into the cache so callers bump `next_id`,
/// advance `primary_root`, and let `apply_doc_change` advance index
/// roots in place — all without a per-doc `Catalog::update`.
pub(crate) fn cached_descriptor_mut<'g>(
    cache: &'g mut HashMap<String, CollectionDescriptor>,
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    name: &str,
) -> Result<&'g mut CollectionDescriptor> {
    if !cache.contains_key(name) {
        let descriptor = catalog_get_required(pager, catalog, name)?;
        cache.insert(name.to_owned(), descriptor);
    }
    cache.get_mut(name).ok_or(Error::Corruption { page_id: 0 })
}

/// Typed handle to a collection.
///
/// Construct via [`crate::WriteTxn::collection`] (lazy-create) or
/// [`crate::ReadTxn::collection`] (read-only; errors if absent), or
/// via [`crate::Db::collection`] for a one-shot read-only handle
/// bound to a runtime collection name.
///
/// All methods take `&self` because the underlying state lives
/// behind mutexes on the parent transaction; the handle itself is
/// stateless beyond the descriptor it caches.
pub struct Collection<'tx, T: Document> {
    /// `Mode::Write` carries the [`WriteTxn`] reference;
    /// `Mode::Read` the [`ReadTxn`] reference; `Mode::Lazy` carries
    /// a `&Db` and opens a private read transaction on each method
    /// call.  The [`Collection`]'s lifetime is bound to whichever
    /// it was constructed from.
    mode: CollectionMode<'tx>,
    /// Collection name resolved at construction. For handles built
    /// via [`crate::WriteTxn::collection`] / [`crate::ReadTxn::collection`]
    /// this equals `T::COLLECTION`. For handles built via
    /// [`crate::Db::collection`] this is the caller-supplied runtime
    /// name (which may differ from `T::COLLECTION`, e.g.
    /// `"archive.orders"` against a type whose declared `COLLECTION`
    /// is `"orders"`).
    // allow: field name intentionally repeats the `collection_*` prefix of
    // its `Collection<T>` container; renaming would lose clarity at use sites.
    #[allow(clippy::struct_field_names)]
    collection_name: Cow<'static, str>,
    /// Cached descriptor.  Populated at construction for `Write` /
    /// `Read` mode; never updated in place — a `Collection<T>`
    /// reflects the catalog row that existed when the handle was
    /// opened.  `update` / `delete` inside the same txn re-read the
    /// descriptor through the catalog lock to capture mutations
    /// from prior calls in the same txn (e.g. an `insert` that
    /// advanced `next_id`).
    ///
    /// For `Lazy` mode the descriptor is a sentinel — the real
    /// descriptor is loaded inside each method's private read
    /// transaction.
    descriptor: CollectionDescriptor,
    _phantom: PhantomData<fn() -> T>,
}

/// Backing-reference inside a [`Collection`].  Encodes whether
/// the txn is writable.
enum CollectionMode<'tx> {
    Write(WriteRef<'tx>),
    Read(ReadRef<'tx>),
    /// Read-only handle that opens a private read transaction on
    /// each method call. Constructed by [`crate::Db::collection`];
    /// the `'tx` lifetime is the borrow of `&Db`.
    Lazy(LazyRef<'tx>),
}

struct LazyRef<'db> {
    /// Borrowed `Db` — methods open one-shot read transactions on
    /// it. The `'db` lifetime keeps the borrow alive for as long
    /// as the [`Collection`] handle.
    db: &'db crate::Db,
}

struct WriteRef<'tx> {
    env: &'tx obj_core::TxnEnv<FileHandle>,
    catalog: Arc<Mutex<Catalog<FileHandle>>>,
    /// Shared per-txn descriptor cache (cloned from the owning
    /// [`WriteTxn`]). The single mid-txn source of truth for
    /// `next_id` / `primary_root` / index roots; flushed once at
    /// commit.
    descriptors: DescriptorCache,
}

struct ReadRef<'tx> {
    snapshot: &'tx obj_core::ReaderSnapshot<FileHandle>,
    env: &'tx obj_core::TxnEnv<FileHandle>,
}

impl<'tx, T: Document> Collection<'tx, T> {
    /// Open the collection on the write side, lazy-creating the
    /// catalog row + an empty primary B-tree on first call, and
    /// reconciling the type's declared `Document::indexes()` against
    /// the catalog's stored descriptors on the first call
    /// per process per collection.
    pub(crate) fn open_or_create(tx: &'tx mut WriteTxn<'_>) -> Result<Self> {
        let _initial = ensure_collection::<T>(&tx.inner, &tx.catalog)?;
        reconcile_indexes_once::<T>(
            &tx.inner,
            &tx.catalog,
            &tx.reconciled,
            &mut tx.reconciled_staged,
        )?;
        let descriptor = reread_descriptor::<T>(&tx.inner, &tx.catalog)?;
        Ok(Self {
            mode: CollectionMode::Write(WriteRef {
                env: tx.inner.env(),
                catalog: Arc::clone(&tx.catalog),
                descriptors: Arc::clone(&tx.descriptors),
            }),
            collection_name: Cow::Borrowed(T::COLLECTION),
            descriptor,
            _phantom: PhantomData,
        })
    }

    /// Open the collection on the read side.  Errors if the
    /// collection has not yet been registered AT THE SNAPSHOT'S
    /// PINNED LSN — a collection created by a concurrent writer
    /// AFTER the snapshot was pinned is invisible to this txn. The
    /// descriptor is read by walking the catalog B+tree
    /// rooted at `snapshot.root_catalog()` (the value captured by
    /// `Pager::reader_snapshot` at pin time) using the snapshot-
    /// aware [`Catalog::lookup_via_snapshot`] free function — NOT
    /// via the writer's live `Catalog.tree.root`, which a concurrent
    /// writer may have COW-advanced past the snapshot.
    ///
    /// If `T::COLLECTION` is of the form
    /// `<namespace>.<name>`, the lookup dispatches against the
    /// attached database registered under `<namespace>` instead of
    /// the calling Db. The attached snapshot is the one pinned by
    /// `Db::read_transaction` at txn-begin; the attached env is
    /// passed through the same way.
    pub(crate) fn open_readonly(tx: &'tx ReadTxn<'_>) -> Result<Self> {
        Self::open_readonly_named(tx, Cow::Borrowed(T::COLLECTION))
    }

    /// Open the collection on the read side against a caller-supplied
    /// runtime `name`. Like [`Self::open_readonly`] but the catalog
    /// lookup uses `name` instead of `T::COLLECTION` — required for
    /// [`crate::Db::collection`]'s accessor, which
    /// binds a runtime collection name (e.g. `"archive.orders"`) to
    /// the typed handle.
    ///
    /// Namespace dispatch (`<ns>.<tail>` → attached database) follows
    /// the same shape as [`Self::open_readonly`].
    pub(crate) fn open_readonly_named(
        tx: &'tx ReadTxn<'_>,
        name: Cow<'static, str>,
    ) -> Result<Self> {
        let (namespace, tail) = crate::db::split_namespace(&name);
        let (env, snapshot, lookup_name): (
            &'tx obj_core::TxnEnv<FileHandle>,
            &'tx obj_core::ReaderSnapshot<FileHandle>,
            &str,
        ) = match namespace {
            None => (tx.inner.env(), tx.inner.snapshot(), &name),
            Some(ns) => {
                let ctx = tx
                    .attached
                    .get(ns)
                    .ok_or_else(|| Error::CollectionNamespaceUnknown {
                        namespace: ns.to_owned(),
                    })?;
                (ctx.env.as_ref(), &ctx.snapshot, tail)
            }
        };
        let Some(descriptor) = read_descriptor_via_snapshot_named(env, snapshot, lookup_name)?
        else {
            return Err(Error::CollectionNotFound {
                name: name.into_owned(),
            });
        };
        Ok(Self {
            mode: CollectionMode::Read(ReadRef { snapshot, env }),
            collection_name: name,
            descriptor,
            _phantom: PhantomData,
        })
    }

    /// Construct a deferred-lookup, read-only handle bound to a
    /// runtime collection name. Each method call opens a private
    /// read transaction on `db` and dispatches through
    /// [`Self::open_readonly_named`]. Construction is infallible —
    /// errors (missing collection, unknown namespace, etc.) surface
    /// at the first method call.
    ///
    /// Used by [`crate::Db::collection`].
    pub(crate) fn lazy(db: &'tx crate::Db, name: String) -> Self {
        Self {
            mode: CollectionMode::Lazy(LazyRef { db }),
            collection_name: Cow::Owned(name),
            descriptor: CollectionDescriptor::new(0, 0, 0),
            _phantom: PhantomData,
        }
    }

    /// Cached descriptor (`collection_id`, `primary_root`,
    /// `type_version`, `next_id` at handle-open time).
    #[must_use]
    pub fn descriptor(&self) -> &CollectionDescriptor {
        &self.descriptor
    }

    /// The LIVE primary-tree root for a `Write`-mode handle.
    ///
    /// Prefers the per-txn descriptor cache (which carries every
    /// `primary_root` advance from this txn's prior writes) so a
    /// read-after-write on the same handle inside one transaction
    /// observes its own uncommitted inserts. Falls back to the
    /// handle's open-time `primary_root` if this collection has not
    /// yet been written in the txn (no cache entry).
    fn write_primary_root(&self, write: &WriteRef<'tx>) -> Result<u64> {
        let cache = lock_descriptors(&write.descriptors)?;
        Ok(cache
            .get(self.collection_name.as_ref())
            .map_or(self.descriptor.primary_root, |d| d.primary_root))
    }

    /// The LIVE `root_page_id` for the named `Active` index on a
    /// `Write`-mode handle. Prefers the per-txn cache so a read after
    /// an index-mutating write in the same txn descends the advanced
    /// root; falls back to `fallback` (the handle's open-time
    /// descriptor entry) when the collection has no cache entry yet.
    fn write_index_root(
        &self,
        write: &WriteRef<'tx>,
        index_name: &str,
        fallback: u64,
    ) -> Result<u64> {
        let cache = lock_descriptors(&write.descriptors)?;
        let Some(descriptor) = cache.get(self.collection_name.as_ref()) else {
            return Ok(fallback);
        };
        let live = descriptor
            .indexes
            .iter()
            .find(|d| d.name == index_name && d.status == obj_core::IndexStatus::Active)
            .map_or(fallback, |d| d.root_page_id);
        Ok(live)
    }

    /// Insert `doc`.  Returns the freshly-allocated [`Id`].
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] if the handle is read-only.
    /// - Pager / catalog / codec errors propagated.
    // allow: by-value `doc` is the public insert API — the caller hands over ownership of the document to persist; it is only borrowed internally, but taking it by reference would force every caller to keep the value alive past the call.
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert(&self, doc: T) -> Result<Id>
    where
        T: Schema,
    {
        let write = self.write_or_err("insert")?;
        let name: &str = self.collection_name.as_ref();
        let mut pager = lock_pager(write.env)?;
        let mut catalog = lock_catalog(&write.catalog)?;
        let mut cache = lock_descriptors(&write.descriptors)?;
        let descriptor = cached_descriptor_mut(&mut cache, &mut pager, &catalog, name)?;
        let id = obj_core::id::bump_next_id(&mut descriptor.next_id, || name.to_owned())?;
        persist_current_schema::<T>(&mut catalog, &mut pager, descriptor.collection_id)?;
        let bytes = encode(&doc, descriptor.collection_id)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        crate::index_maint::apply_doc_change::<T>(&mut pager, descriptor, None, Some(&doc), id)?;
        Ok(id)
    }

    /// Fetch the document at `id`.
    ///
    /// On the write side this consults the pager (sees pending
    /// writes in the current txn).  On the read side it consults
    /// the snapshot's frozen view.
    ///
    /// # Lazy migration
    ///
    /// If the on-disk record was written by an older
    /// `Document::VERSION` than the current `T::VERSION`, the codec
    /// walks the stored bytes through the schema registered for
    /// that version (see `T::historical_schemas()`) and dispatches
    /// the resulting structured `Dynamic` through `T::migrate`.
    /// **The migrated bytes are NOT written back to disk.** The
    /// next [`Collection::get`] re-reads the same v(n) bytes and
    /// re-runs migration. Only a subsequent
    /// [`Collection::update`](Self::update) /
    /// [`Collection::upsert`](Self::upsert) writes the document
    /// back, at which point the on-disk header records
    /// `T::VERSION`.
    ///
    /// This contract is what allows mixed-version reads to scale:
    /// a 10⁹-doc collection does not need to be batch-rewritten on
    /// schema upgrade. Every "migration ran" path returns the
    /// migrated `T`; no implicit write-back.
    ///
    /// # Errors
    ///
    /// Pager / B-tree / codec errors propagated. In particular:
    ///
    /// - [`Error::SchemaNotRegistered`](obj_core::Error::SchemaNotRegistered)
    ///   if the stored record carries a `type_version` for which
    ///   `T::historical_schemas()` has no entry.
    /// - [`Error::SchemaMigrationNotImplemented`](obj_core::Error::SchemaMigrationNotImplemented)
    ///   if the registered `T::migrate` returns the default error.
    pub fn get(&self, id: Id) -> Result<Option<T>> {
        if let Some(r) = self.dispatch_lazy(|c| c.get(id)) {
            return r;
        }
        let key = id.to_be_bytes();
        match &self.mode {
            CollectionMode::Write(write) => self.write_get_and_decode(write, &key),
            CollectionMode::Read(read) => self.read_get_and_decode(read, &key),
            CollectionMode::Lazy(_) => Err(Error::ReadOnly {
                operation: "internal: lazy-mode primary read",
            }),
        }
    }

    /// `get` on a write-side handle: descend the LIVE primary root (so a
    /// read-after-write in the same txn sees its own writes) and decode
    /// through the writer's live catalog tree. No reader
    /// snapshot exists inside a `WriteTxn`, so an older-version record
    /// resolves its stored schema via [`Catalog::get_schema_in_txn`].
    fn write_get_and_decode(&self, write: &WriteRef<'tx>, key: &[u8]) -> Result<Option<T>> {
        let primary_root = self.write_primary_root(write)?;
        let mut pager = lock_pager(write.env)?;
        let catalog = lock_catalog(&write.catalog)?;
        let tree = btree_handle(&pager, primary_root)?;
        let Some(bytes) = tree.get(&mut pager, key)? else {
            return Ok(None);
        };
        let collection_id = self.descriptor.collection_id;
        Ok(Some(decode_in_write_txn::<T>(
            &mut pager,
            &catalog,
            collection_id,
            &bytes,
        )?))
    }

    /// `get` on a read-side handle: read the snapshot-pinned primary
    /// root and decode through [`SchemaSource::Snapshot`],
    /// so an older-version record's stored schema is resolved at the
    /// SAME pinned LSN as the document bytes it describes.
    fn read_get_and_decode(&self, read: &ReadRef<'tx>, key: &[u8]) -> Result<Option<T>> {
        let pager = lock_pager(read.env)?;
        let Some(bytes) =
            snapshot_get_locked(&pager, read.snapshot, self.descriptor.primary_root, key)?
        else {
            return Ok(None);
        };
        let collection_id = self.descriptor.collection_id;
        Ok(Some(decode_with::<T, FileHandle>(
            &bytes,
            collection_id,
            SchemaSource::Snapshot {
                pager: &pager,
                snapshot: read.snapshot,
                collection_id,
            },
        )?))
    }

    /// Apply `f` to the document at `id`, writing the mutated value
    /// back.
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] on a read-side handle.
    /// - [`Error::DocumentNotFound`] if `id` is absent.
    /// - Pager / catalog / codec errors propagated.
    pub fn update<F>(&self, id: Id, f: F) -> Result<()>
    where
        F: FnOnce(&mut T),
        T: Schema,
    {
        let write = self.write_or_err("update")?;
        let name: &str = self.collection_name.as_ref();
        let mut pager = lock_pager(write.env)?;
        let mut catalog = lock_catalog(&write.catalog)?;
        let mut cache = lock_descriptors(&write.descriptors)?;
        let descriptor = cached_descriptor_mut(&mut cache, &mut pager, &catalog, name)?;
        let collection_id = descriptor.collection_id;
        persist_current_schema::<T>(&mut catalog, &mut pager, collection_id)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        let existing = tree.get(&mut pager, &key)?.ok_or(Error::DocumentNotFound {
            collection: T::COLLECTION,
            id: id.get(),
        })?;
        let old_value = decode_in_write_txn::<T>(&mut pager, &catalog, collection_id, &existing)?;
        let mut new_value =
            decode_in_write_txn::<T>(&mut pager, &catalog, collection_id, &existing)?;
        f(&mut new_value);
        let bytes = encode(&new_value, descriptor.collection_id)?;
        tree.delete(&mut pager, &key)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        crate::index_maint::apply_doc_change::<T>(
            &mut pager,
            descriptor,
            Some(&old_value),
            Some(&new_value),
            id,
        )?;
        Ok(())
    }

    /// Delete the document at `id`.  Returns `true` if it existed.
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] on a read-side handle.
    /// - Pager / catalog errors propagated.
    pub fn delete(&self, id: Id) -> Result<bool> {
        let write = self.write_or_err("delete")?;
        let name: &str = self.collection_name.as_ref();
        let mut pager = lock_pager(write.env)?;
        let catalog = lock_catalog(&write.catalog)?;
        let mut cache = lock_descriptors(&write.descriptors)?;
        let descriptor = cached_descriptor_mut(&mut cache, &mut pager, &catalog, name)?;
        let collection_id = descriptor.collection_id;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        let old_value = match tree.get(&mut pager, &key)? {
            Some(bytes) => Some(decode_in_write_txn::<T>(
                &mut pager,
                &catalog,
                collection_id,
                &bytes,
            )?),
            None => None,
        };
        let removed = tree.delete(&mut pager, &key)?;
        descriptor.primary_root = tree.root().get();
        crate::index_maint::apply_doc_change::<T>(
            &mut pager,
            descriptor,
            old_value.as_ref(),
            None,
            id,
        )?;
        Ok(removed)
    }

    /// Insert-or-replace `doc` at `id`.
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] on a read-side handle.
    /// - Pager / catalog / codec errors propagated.
    // allow: by-value `doc` is the public upsert API — the caller hands over ownership of the document to persist; it is only borrowed internally, but taking it by reference would force every caller to keep the value alive past the call.
    #[allow(clippy::needless_pass_by_value)]
    pub fn upsert(&self, id: Id, doc: T) -> Result<()>
    where
        T: Schema,
    {
        let write = self.write_or_err("upsert")?;
        let name: &str = self.collection_name.as_ref();
        let mut pager = lock_pager(write.env)?;
        let mut catalog = lock_catalog(&write.catalog)?;
        let mut cache = lock_descriptors(&write.descriptors)?;
        let descriptor = cached_descriptor_mut(&mut cache, &mut pager, &catalog, name)?;
        let collection_id = descriptor.collection_id;
        persist_current_schema::<T>(&mut catalog, &mut pager, collection_id)?;
        let bytes = encode(&doc, descriptor.collection_id)?;
        let key = id.to_be_bytes();
        let mut tree = btree_handle(&pager, descriptor.primary_root)?;
        let old_value = match tree.get(&mut pager, &key)? {
            Some(prior) => Some(decode_in_write_txn::<T>(
                &mut pager,
                &catalog,
                collection_id,
                &prior,
            )?),
            None => None,
        };
        let _ = tree.delete(&mut pager, &key)?;
        tree.insert(&mut pager, &key, &bytes)?;
        descriptor.primary_root = tree.root().get();
        crate::index_maint::apply_doc_change::<T>(
            &mut pager,
            descriptor,
            old_value.as_ref(),
            Some(&doc),
            id,
        )?;
        Ok(())
    }

    /// Look up the single document whose `index_name` key matches
    /// `key` under a `Unique` index.
    ///
    /// Errors with [`Error::IndexNotUnique`] if `index_name` resolves
    /// to a non-unique index — `find_unique` is *only* defined on
    /// `Unique` indexes. For `Standard` / `Each` / `Composite` use
    /// [`Self::lookup`] (which returns an iterator).
    ///
    /// Snapshot-aware: on a write-side handle the lookup sees the
    /// current txn's pending writes; on a read-side handle it sees
    /// the snapshot's frozen view.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - [`Error::IndexNotUnique`] if the index is not `Unique`.
    /// - Pager / B-tree / codec errors propagated.
    pub fn find_unique(
        &self,
        index_name: &str,
        key: impl Into<obj_core::codec::Dynamic>,
    ) -> Result<Option<T>> {
        let key_dyn = key.into();
        if let Some(r) = self.dispatch_lazy(|c| c.find_unique(index_name, key_dyn.clone())) {
            return r;
        }
        let descriptor = self.active_index(index_name)?;
        if descriptor.kind != obj_core::IndexKind::Unique {
            return Err(Error::IndexNotUnique {
                collection: self.collection_name.clone().into_owned(),
                name: index_name.to_owned(),
            });
        }
        let encoded = index_key_for_lookup(descriptor, &[key_dyn])?;
        let id_bytes = self.index_get(descriptor, encoded.as_bytes())?;
        match id_bytes {
            Some(bytes) => match Id::from_be_bytes(&bytes) {
                Some(id) => self.get(id),
                None => Err(Error::Corruption { page_id: 0 }),
            },
            None => Ok(None),
        }
    }

    /// Yield every document whose `index_name` key matches `key`.
    /// Works on `Standard` / `Unique` / `Each` indexes. Returns
    /// `Err(Error::IndexKindMismatch)`-style guidance for
    /// `Composite` (use [`Self::index_range`] for tuple-shaped
    /// keys).
    ///
    /// The same document is yielded at most once even if it owns
    /// multiple matching entries — `Each` indexes can encode the
    /// same `id` under multiple element keys; we de-dup on emit.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - Pager / B-tree / codec errors propagated.
    pub fn lookup(
        &self,
        index_name: &str,
        key: impl Into<obj_core::codec::Dynamic>,
    ) -> Result<IndexIter<'static, T>>
    where
        T: Send + 'static,
    {
        let key_dyn = key.into();
        if let Some(r) = self.dispatch_lazy(|c| c.lookup(index_name, key_dyn.clone())) {
            return r;
        }
        let descriptor = self.active_index(index_name)?;
        let encoded = index_key_for_lookup(descriptor, &[key_dyn])?;
        let ids = match descriptor.kind {
            obj_core::IndexKind::Unique => self.collect_unique(descriptor, encoded.as_bytes())?,
            obj_core::IndexKind::Standard
            | obj_core::IndexKind::Each
            | obj_core::IndexKind::Composite => {
                self.collect_nonunique_equal(descriptor, encoded.as_bytes())?
            }
            _ => return Err(Error::InvalidArgument("unsupported index kind")),
        };
        let resolved = self.resolve_unique_ids(ids)?;
        Ok(Box::new(resolved.into_iter().map(Ok)))
    }

    /// Yield `(user_key, doc)` pairs whose index key falls within
    /// `range`. The bounds may be any scalar that converts into a
    /// [`Dynamic`](obj_core::codec::Dynamic) — see
    /// [`DynamicRange`](crate::DynamicRange) — so `40u64..60` works
    /// without `Dynamic::U64(..)` wrapping, the same ergonomics
    /// [`crate::Query::index_range`] offers. The bounds are encoded
    /// internally through the order-preserving field encoder
    /// ([`obj_core::index::encode_field`]); callers no longer
    /// hand-encode index-key bytes.
    ///
    /// For non-Unique kinds (`Standard` / `Each` / `Composite`) the
    /// bounds are widened internally so a user-facing
    /// `Included(x)..=Included(x)` range matches every entry whose
    /// user-key equals `x` even though the underlying B-tree key
    /// carries an `id_be8` suffix.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - [`obj_core::Error::Codec`] if a `Dynamic::String` bound
    ///   carries an embedded NUL byte (the order-preserving encoder
    ///   rejects those).
    /// - Pager / B-tree / codec errors propagated.
    pub fn index_range<R>(
        &self,
        index_name: &str,
        range: R,
    ) -> Result<IndexIter<'static, (Vec<u8>, T)>>
    where
        R: crate::range::DynamicRange,
        T: Send + 'static,
    {
        let (start, end) = range.into_dynamic_bounds();
        let start = encode_dynamic_bound(start.as_ref())?;
        let end = encode_dynamic_bound(end.as_ref())?;
        self.index_range_encoded(index_name, start, end)
    }

    /// Encoded-bytes variant of [`Self::index_range`]. The bounds are
    /// already the order-preserving field encoding of the user's
    /// `Dynamic` value(s); this keeps the signature general for
    /// `Composite` "starts-with" scans and is the entry point the
    /// query layer / lazy-dispatch recursion call after they have
    /// done their own encoding.
    pub(crate) fn index_range_encoded(
        &self,
        index_name: &str,
        start_bound: std::ops::Bound<Vec<u8>>,
        end_bound: std::ops::Bound<Vec<u8>>,
    ) -> Result<IndexIter<'static, (Vec<u8>, T)>>
    where
        T: Send + 'static,
    {
        if let Some(r) = self.dispatch_lazy(|c| {
            c.index_range_encoded(index_name, start_bound.clone(), end_bound.clone())
        }) {
            return r;
        }
        let descriptor = self.active_index(index_name)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(start_bound, end_bound, descriptor.kind);
        let entries = self.collect_range(descriptor, start, end)?;
        let descriptor_kind = descriptor.kind;
        let mut out: Vec<Result<(Vec<u8>, T)>> = Vec::with_capacity(entries.len());
        let mut emitted_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for (full_key, id_bytes_value) in entries {
            let Some(id) = Id::from_be_bytes(&id_bytes_value) else {
                out.push(Err(Error::Corruption { page_id: 0 }));
                continue;
            };
            if descriptor_kind == obj_core::IndexKind::Each && !emitted_ids.insert(id.get()) {
                continue;
            }
            let user_key = strip_id_suffix(&full_key, descriptor_kind);
            match self.get(id) {
                Ok(Some(doc)) => out.push(Ok((user_key, doc))),
                Ok(None) => {
                    out.push(Err(Error::Corruption { page_id: 0 }));
                }
                Err(e) => out.push(Err(e)),
            }
        }
        Ok(Box::new(out.into_iter()))
    }

    /// Streaming variant of [`Self::index_range`]. Yields
    /// `(user_key, T)` pairs lazily — the
    /// returned [`IterIndexRange`] decodes one `T` per `next()` call
    /// rather than building a `Vec<Result<(_, T)>>` of every match
    /// up front. The iterator borrows `&'a self`, so it must be
    /// consumed inside the lifetime of the enclosing
    /// [`crate::WriteTxn`] / [`crate::ReadTxn`] (or the
    /// [`crate::Db::collection`] handle, in Lazy mode).
    ///
    /// # When to prefer `iter_range` over `index_range`
    ///
    /// - **Memory.** `index_range` allocates `O(matches × sizeof(T))`
    ///   upfront; `iter_range` keeps a fixed-size [`VecDeque`] of
    ///   `(key, id)` pairs (`ITER_INDEX_RANGE_BATCH = 256` entries)
    ///   and decodes one `T` at a time. For a 100k-row range with
    ///   ~500-byte documents that's ~50 MB peak vs. a few KiB.
    /// - **Latency-to-first-row.** `index_range` decodes every
    ///   matching document before returning the iterator;
    ///   `iter_range` returns immediately after the first chunk
    ///   refill, so the first `next()` returns after one index walk
    ///   + one primary-tree `get` (rather than `N`).
    ///
    /// # When `index_range` is still the right answer
    ///
    /// `index_range` returns an `IndexIter<'static, _>` — it can
    /// escape the `read_transaction` / `transaction` closure that
    /// produced it. `iter_range` is bound to `&self`, so the
    /// iterator dies when the [`Collection`] handle dies. If you
    /// need to return the iterator to outer scope, stick with
    /// `index_range`.
    ///
    /// # Per-row `get`-back design choice
    ///
    /// Each `next()` yields `(user_key, T)` by calling
    /// [`Self::get`] under the hood — i.e. a SECOND B+tree descent
    /// per row (the first is the index range walk; the second is
    /// the primary-tree `get(id)`). This is intentional and
    /// inherited from `index_range`: the index leaf stores only
    /// the document `id` (8 bytes), not the document bytes. A
    /// future format-minor bump may add value-in-index storage to
    /// short-circuit the second descent; that work is pinned to
    /// post-1.0.
    ///
    /// # Why each item is a `Result`
    ///
    /// Iteration is *fallible per step*. The iterator walks the
    /// index in bounded chunks and `get`s each document back
    /// lazily, so a pager read, B-tree descent, or codec decode can
    /// fail mid-scan — long after construction already succeeded.
    /// Each `next` therefore yields `Result<(user_key, T)>`; the
    /// canonical loop unwraps the per-step `Result` with `?`, which
    /// is why the `(key, doc)` destructure lives in the loop body
    /// rather than the `for` binding:
    ///
    /// ```ignore
    /// for step in coll.iter_range("placed_at", 10u64..40)? {
    ///     let (key, doc) = step?; // propagate a mid-scan IO error
    ///     // ... use `key` / `doc`
    /// }
    /// ```
    ///
    /// Contrast the eager forms — [`Self::index_range`] (and the
    /// whole-collection [`Self::all`]) materialise their matches and
    /// surface IO failure through a single `?` at the call site
    /// instead of once per element. Choose deliberately: the eager
    /// call reads cleaner when the result set fits in memory; the
    /// lazy iterator keeps peak memory bounded and lets you bail out
    /// after the first row.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::{Db, Document};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "orders_iter_range_doc")]
    /// struct Order {
    ///     #[obj(index)]
    ///     placed_at: u64,
    ///     total_cents: u64,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("iter_range.obj"))?;
    /// for i in 0..10u64 {
    ///     let _ = db.insert(Order { placed_at: i, total_cents: i * 100 })?;
    /// }
    ///
    /// // Stream the [3, 7) window, unwrapping each step with `?`.
    /// let coll = db.collection::<Order>(Order::COLLECTION);
    /// let mut total: u64 = 0;
    /// for step in coll.iter_range("placed_at", 3u64..7)? {
    ///     let (_key, doc) = step?;
    ///     total = total
    ///         .checked_add(doc.total_cents)
    ///         .ok_or(obj::Error::InvalidArgument("overflow"))?;
    /// }
    /// assert_eq!(total, (300 + 400 + 500 + 600));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - Pager / B-tree / codec errors propagated at construction
    ///   and from each `next()` call.
    pub fn iter_range<'a, R>(&'a self, index_name: &str, range: R) -> Result<IterIndexRange<'a, T>>
    where
        R: crate::range::DynamicRange,
        T: Send + 'static,
    {
        let (start, end) = range.into_dynamic_bounds();
        let start_bound = encode_dynamic_bound(start.as_ref())?;
        let end_bound = encode_dynamic_bound(end.as_ref())?;
        self.iter_range_encoded(index_name, start_bound, end_bound)
    }

    /// Encoded-bytes variant of [`Self::iter_range`]. Bounds are the
    /// order-preserving field encoding of the user's `Dynamic`
    /// value(s); used internally by the lazy-mode fallback path.
    fn iter_range_encoded<'a>(
        &'a self,
        index_name: &str,
        start_bound: Bound<Vec<u8>>,
        end_bound: Bound<Vec<u8>>,
    ) -> Result<IterIndexRange<'a, T>>
    where
        T: Send + 'static,
    {
        if matches!(self.mode, CollectionMode::Lazy(_)) {
            return self.iter_range_lazy_fallback(index_name, start_bound, end_bound);
        }
        let descriptor = self.active_index(index_name)?;
        let index_root = match &self.mode {
            CollectionMode::Write(w) => {
                self.write_index_root(w, index_name, descriptor.root_page_id)?
            }
            _ => descriptor.root_page_id,
        };
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(start_bound, end_bound, descriptor.kind);
        let initial_resume = match start {
            Bound::Included(k) => InitialResume::Included(k),
            Bound::Excluded(k) => InitialResume::Excluded(k),
            Bound::Unbounded => InitialResume::Unbounded,
        };
        Ok(IterIndexRange {
            coll: self,
            descriptor_kind: descriptor.kind,
            index_root,
            initial_resume: Some(initial_resume),
            last_full_key: None,
            end_bound: end,
            buffer: VecDeque::new(),
            emitted_ids: HashSet::new(),
            finished: false,
        })
    }

    /// Lazy-mode fallback for [`Self::iter_range`]: delegates to
    /// [`Self::index_range`] (which itself dispatches through a fresh
    /// read txn) and rehouses the eagerly-materialised entries into
    /// the streaming iterator's buffer as
    /// [`StagedEntry::Resolved`]. Kept isolated so the streaming
    /// path's `iter_range` body stays small.
    fn iter_range_lazy_fallback<'a>(
        &'a self,
        index_name: &str,
        start_bound: Bound<Vec<u8>>,
        end_bound: Bound<Vec<u8>>,
    ) -> Result<IterIndexRange<'a, T>>
    where
        T: Send + 'static,
    {
        let materialized = self.index_range_encoded(index_name, start_bound, end_bound)?;
        let mut buffer: VecDeque<Result<StagedEntry<T>>> = VecDeque::new();
        for item in materialized {
            match item {
                Ok((key, doc)) => buffer.push_back(Ok(StagedEntry::Resolved(key, doc))),
                Err(e) => buffer.push_back(Err(e)),
            }
        }
        Ok(IterIndexRange {
            coll: self,
            descriptor_kind: obj_core::IndexKind::Standard,
            index_root: 0,
            initial_resume: None,
            last_full_key: None,
            end_bound: Bound::Unbounded,
            buffer,
            emitted_ids: HashSet::new(),
            finished: true,
        })
    }

    /// Look up the `IndexKind` of an active index by name. Used by
    /// the query layer to dispatch `Query::count` between the
    /// entry-count and distinct-id-count paths.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    pub(crate) fn index_kind(&self, index_name: &str) -> Result<obj_core::IndexKind> {
        Ok(self.active_index(index_name)?.kind)
    }

    /// Resolve `index_name` to an `Active` `IndexDescriptor` on the
    /// collection. Errors with [`Error::IndexNotFound`] if absent
    /// or `DroppedPending`.
    ///
    /// Returns a borrow into `self.descriptor.indexes` — no per-lookup
    /// clone. Every caller uses the descriptor only within the
    /// enclosing `&self` borrow; `iter_range_encoded` copies the two
    /// `Copy` fields it needs (`kind`, `root_page_id`) into the
    /// returned iterator rather than holding the borrow.
    fn active_index(&self, index_name: &str) -> Result<&obj_core::IndexDescriptor> {
        let entry = self
            .descriptor
            .indexes
            .iter()
            .find(|d| d.name == index_name);
        match entry {
            Some(d) if d.status == obj_core::IndexStatus::Active => Ok(d),
            _ => Err(Error::IndexNotFound {
                collection: self.collection_name.clone().into_owned(),
                name: index_name.to_owned(),
            }),
        }
    }

    /// Single-key `get` on an index B-tree. Used by `find_unique`
    /// and by the Unique-kind branch of `lookup`.
    fn index_get(
        &self,
        descriptor: &obj_core::IndexDescriptor,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        match &self.mode {
            CollectionMode::Write(write) => {
                let root_raw =
                    self.write_index_root(write, &descriptor.name, descriptor.root_page_id)?;
                let root = PageId::new(root_raw)
                    .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
                let mut pager = lock_pager(write.env)?;
                let tree = BTree::<FileHandle>::open(&pager, root)?;
                tree.get(&mut pager, key)
            }
            CollectionMode::Read(read) => {
                let root = PageId::new(descriptor.root_page_id)
                    .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
                let pager = lock_pager(read.env)?;
                BTree::<FileHandle>::get_via_snapshot(&pager, read.snapshot, root, key)
            }
            CollectionMode::Lazy(_) => Err(Error::ReadOnly {
                operation: "internal: lazy-mode index_get",
            }),
        }
    }

    /// Collect every `(full_key, value)` entry from an index B-tree
    /// whose key starts with `prefix`. For unique kinds the prefix
    /// is the entire key (one match max); for non-unique kinds we
    /// match every key whose first `prefix.len()` bytes equal
    /// `prefix` (the trailing `id_suffix` varies per doc).
    fn collect_nonunique_equal(
        &self,
        descriptor: &obj_core::IndexDescriptor,
        prefix: &[u8],
    ) -> Result<Vec<u64>> {
        let entries = self.collect_range(
            descriptor,
            std::ops::Bound::Included(prefix.to_vec()),
            std::ops::Bound::Included(append_max_id(prefix)),
        )?;
        let mut ids = Vec::with_capacity(entries.len());
        let mut emitted: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for (_full_key, value) in entries {
            let id = Id::from_be_bytes(&value).ok_or(Error::Corruption { page_id: 0 })?;
            if emitted.insert(id.get()) {
                ids.push(id.get());
            }
        }
        Ok(ids)
    }

    /// Collect a single id from a Unique index B-tree at `key`.
    fn collect_unique(
        &self,
        descriptor: &obj_core::IndexDescriptor,
        key: &[u8],
    ) -> Result<Vec<u64>> {
        match self.index_get(descriptor, key)? {
            Some(bytes) => Id::from_be_bytes(&bytes)
                .map(|id| vec![id.get()])
                .ok_or(Error::Corruption { page_id: 0 }),
            None => Ok(Vec::new()),
        }
    }

    /// Collect every `(full_key, value)` entry from an index B-tree
    /// whose key falls within `(start, end)`.
    fn collect_range(
        &self,
        descriptor: &obj_core::IndexDescriptor,
        start: std::ops::Bound<Vec<u8>>,
        end: std::ops::Bound<Vec<u8>>,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        match &self.mode {
            CollectionMode::Read(r) => {
                let root = PageId::new(descriptor.root_page_id)
                    .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
                let pager = lock_pager(r.env)?;
                let iter = BTree::<FileHandle>::range_via_snapshot(
                    &pager,
                    r.snapshot,
                    root,
                    (start, end),
                )?;
                let mut out = Vec::new();
                for step in iter {
                    out.push(step?);
                }
                Ok(out)
            }
            CollectionMode::Write(w) => {
                let root_raw =
                    self.write_index_root(w, &descriptor.name, descriptor.root_page_id)?;
                let root = PageId::new(root_raw)
                    .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
                let mut pager = lock_pager(w.env)?;
                let tree = BTree::<FileHandle>::open(&pager, root)?;
                let iter = tree.range(&mut pager, (start, end))?;
                let mut out = Vec::new();
                for step in iter {
                    out.push(step?);
                }
                Ok(out)
            }
            CollectionMode::Lazy(_) => Err(Error::ReadOnly {
                operation: "internal: lazy-mode collect_range",
            }),
        }
    }

    /// Resolve a `Vec<u64>` of `Id` integer values into concrete
    /// `T` documents via `self.get`. Preserves order; missing rows
    /// surface as `Error::Corruption` (orphan index entry).
    fn resolve_unique_ids(&self, ids: Vec<u64>) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(ids.len());
        for raw in ids {
            let id =
                Id::from_be_bytes(&raw.to_be_bytes()).ok_or(Error::Corruption { page_id: 0 })?;
            let doc = self.get(id)?.ok_or(Error::Corruption { page_id: 0 })?;
            out.push(doc);
        }
        Ok(out)
    }

    /// Count every entry in the primary tree WITHOUT decoding the
    /// documents. Used by the [`crate::Query::count`] no-decode
    /// fast path; the iterator visits leaf pages and counts entries
    /// rather than running each through postcard.
    ///
    /// Bounded by the B+tree's `MAX_RANGE_NODES` budget (inherited
    /// from `BTree::range`).
    ///
    /// # Errors
    ///
    /// Pager / B-tree errors propagated.
    pub fn count_all(&self) -> Result<u64> {
        // allow: the closure binds `dispatch_lazy`'s transient `&Collection<'_, T>` (a fresh, unnameable lifetime); the suggested `Collection::count_all` method path cannot express that borrow, so the closure stays.
        #[allow(clippy::redundant_closure_for_method_calls)]
        if let Some(r) = self.dispatch_lazy(|c| c.count_all()) {
            return r;
        }
        match &self.mode {
            CollectionMode::Read(r) => {
                let pager = lock_pager(r.env)?;
                let pid = PageId::new(self.descriptor.primary_root)
                    .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
                let iter = BTree::<FileHandle>::range_via_snapshot(&pager, r.snapshot, pid, ..)?;
                count_range_iter(iter)
            }
            CollectionMode::Write(w) => {
                let root = self.write_primary_root(w)?;
                let mut pager = lock_pager(w.env)?;
                let tree = btree_handle(&pager, root)?;
                let iter = tree.range(&mut pager, ..)?;
                count_range_iter(iter)
            }
            CollectionMode::Lazy(_) => Err(Error::ReadOnly {
                operation: "internal: lazy-mode count_all",
            }),
        }
    }

    /// Count every entry whose encoded key falls inside `range` on
    /// the named index's B-tree, WITHOUT decoding any document. Fast
    /// path for [`crate::Query::count`] when the source is an
    /// `index_range`.
    ///
    /// Returns the number of index B-tree entries — for an `Each`
    /// index that may exceed the document count (one doc emits
    /// multiple entries); for other kinds it equals the matching
    /// doc count.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - Pager / B-tree errors propagated.
    pub fn count_index_range<R>(&self, index_name: &str, range: R) -> Result<u64>
    where
        R: crate::range::DynamicRange,
    {
        let (start, end) = range.into_dynamic_bounds();
        let start = encode_dynamic_bound(start.as_ref())?;
        let end = encode_dynamic_bound(end.as_ref())?;
        self.count_index_range_encoded(index_name, start, end)
    }

    /// Encoded-bytes variant of [`Self::count_index_range`]. Bounds
    /// are the order-preserving field encoding of the user's
    /// `Dynamic` value(s); used by the query-layer count fast path.
    pub(crate) fn count_index_range_encoded(
        &self,
        index_name: &str,
        start_bound: std::ops::Bound<Vec<u8>>,
        end_bound: std::ops::Bound<Vec<u8>>,
    ) -> Result<u64> {
        if let Some(r) = self.dispatch_lazy(|c| {
            c.count_index_range_encoded(index_name, start_bound.clone(), end_bound.clone())
        }) {
            return r;
        }
        let descriptor = self.active_index(index_name)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(start_bound, end_bound, descriptor.kind);
        let entries = self.collect_range(descriptor, start, end)?;
        u64::try_from(entries.len()).map_err(|_| Error::BTreeInvariantViolated {
            reason: "index range entry count exceeds u64",
        })
    }

    /// Count distinct document `Id`s whose entries fall inside
    /// `range` on the named index's B-tree, WITHOUT decoding any
    /// document. For `Each` indexes this is the correct shape of
    /// the "how many docs match" question — `count_index_range`
    /// returns the entry count, which overshoots when a single doc
    /// contributes multiple entries.
    ///
    /// Implementation walks the index B-tree, parses the trailing
    /// 8-byte big-endian `Id` suffix from each non-unique key, and
    /// tracks the unique set in a bounded [`std::collections::HashSet`]
    /// capped at [`MAX_DISTINCT_IDS`]. Exceeding the cap surfaces
    /// [`Error::DistinctCountExceeded`] — the caller should narrow
    /// the range.
    ///
    /// # Per-kind semantics
    ///
    /// - `Standard`, `Composite`: equivalent to `count_index_range`
    ///   (one entry per doc by construction; the trailing-id-suffix
    ///   walk still produces the same total).
    /// - `Unique`: keys carry NO id suffix — the entry value is the
    ///   raw 8-byte `Id`; the walk reads the value instead.
    /// - `Each`: the dedup is meaningful — one doc may contribute
    ///   N entries under N distinct element keys.
    ///
    /// # Errors
    ///
    /// - [`Error::IndexNotFound`] if `index_name` is unknown / dropped.
    /// - [`Error::DistinctCountExceeded`] if the distinct set
    ///   exceeds [`MAX_DISTINCT_IDS`].
    /// - [`Error::Corruption`] if an entry's id suffix / value is
    ///   not parseable as an [`obj_core::Id`].
    /// - Pager / B-tree errors propagated.
    pub fn count_distinct_ids_in_range<R>(&self, index_name: &str, range: R) -> Result<u64>
    where
        R: crate::range::DynamicRange,
    {
        let (start, end) = range.into_dynamic_bounds();
        let start = encode_dynamic_bound(start.as_ref())?;
        let end = encode_dynamic_bound(end.as_ref())?;
        self.count_distinct_ids_in_range_encoded(index_name, start, end)
    }

    /// Encoded-bytes variant of [`Self::count_distinct_ids_in_range`].
    /// Bounds are the order-preserving field encoding of the user's
    /// `Dynamic` value(s); used by the query-layer count fast path.
    pub(crate) fn count_distinct_ids_in_range_encoded(
        &self,
        index_name: &str,
        start_bound: std::ops::Bound<Vec<u8>>,
        end_bound: std::ops::Bound<Vec<u8>>,
    ) -> Result<u64> {
        if let Some(r) = self.dispatch_lazy(|c| {
            c.count_distinct_ids_in_range_encoded(
                index_name,
                start_bound.clone(),
                end_bound.clone(),
            )
        }) {
            return r;
        }
        let descriptor = self.active_index(index_name)?;
        let (start, end) =
            crate::index_bound::widen_bounds_for_kind(start_bound, end_bound, descriptor.kind);
        let entries = self.collect_range(descriptor, start, end)?;
        let mut distinct: HashSet<u64> = HashSet::new();
        for (full_key, value) in entries {
            let id = id_from_index_entry(&full_key, &value, descriptor.kind)?;
            if distinct.insert(id) && distinct.len() > MAX_DISTINCT_IDS {
                return Err(Error::DistinctCountExceeded {
                    limit: MAX_DISTINCT_IDS,
                });
            }
        }
        u64::try_from(distinct.len()).map_err(|_| Error::BTreeInvariantViolated {
            reason: "distinct id count exceeds u64",
        })
    }

    /// Materialise every `(Id, T)` pair in the collection.
    ///
    /// Use this when you need the [`Id`] alongside each document
    /// (e.g. to re-`get`, update, or delete by id). If you only want
    /// the documents, [`Self::values`] drops the id for you. The
    /// `Db`-level counterpart, [`crate::Db::all`], is already
    /// id-less — it materialises `Vec<T>` directly — so
    /// `collection.values()` is the per-collection analogue of
    /// `db.all()`, and `collection.all()` is the id-carrying form.
    ///
    /// Implementation note: returns an owned `Vec` rather than a
    /// streaming iterator because the B+tree range API borrows the
    /// pager, and threading that borrow through the mutex guards
    /// in the iterator chain is awkward.  A future revision may
    /// convert to a streaming shape.
    ///
    /// # Errors
    ///
    /// Pager / B-tree / codec errors propagated.
    pub fn all(&self) -> Result<Vec<(Id, T)>> {
        // allow: the closure binds `dispatch_lazy`'s transient `&Collection<'_, T>` (a fresh, unnameable lifetime); the suggested `Collection::all` method path cannot express that borrow, so the closure stays.
        #[allow(clippy::redundant_closure_for_method_calls)]
        if let Some(r) = self.dispatch_lazy(|c| c.all()) {
            return r;
        }
        match &self.mode {
            CollectionMode::Write(write) => {
                let root = self.write_primary_root(write)?;
                let mut pager = lock_pager(write.env)?;
                let catalog = lock_catalog(&write.catalog)?;
                scan_all_in_write_txn::<T>(
                    &mut pager,
                    &catalog,
                    root,
                    self.descriptor.collection_id,
                )
            }
            CollectionMode::Read(read) => snapshot_scan_via_btree::<T>(
                read.snapshot,
                read.env,
                self.descriptor.primary_root,
                self.descriptor.collection_id,
            ),
            CollectionMode::Lazy(_) => Err(Error::ReadOnly {
                operation: "internal: lazy-mode all",
            }),
        }
    }

    /// Materialise every document in the collection, dropping the
    /// [`Id`].
    ///
    /// Convenience over [`Self::all`] for the common case where the
    /// id is not needed: `collection.values()` mirrors
    /// [`crate::Db::all`]'s id-less `Vec<T>` shape, whereas
    /// [`Self::all`] keeps the `(Id, T)` pair when you need to act on
    /// documents by id.
    ///
    /// # Errors
    ///
    /// As [`Self::all`].
    pub fn values(&self) -> Result<Vec<T>> {
        Ok(self.all()?.into_iter().map(|(_id, doc)| doc).collect())
    }

    fn write_or_err(&self, op: &'static str) -> Result<&WriteRef<'tx>> {
        match &self.mode {
            CollectionMode::Write(w) => Ok(w),
            CollectionMode::Read(_) | CollectionMode::Lazy(_) => {
                Err(Error::ReadOnly { operation: op })
            }
        }
    }

    /// If this handle is in `Lazy` mode, dispatch `body` through a
    /// fresh read transaction on the bound [`crate::Db`] — opening a
    /// transient [`Collection`] via [`Self::open_readonly_named`] and
    /// invoking the user-supplied closure on it. Returns `Some(_)`
    /// when the dispatch fires (the closure ran or the underlying
    /// open failed); returns `None` for non-Lazy handles so the
    /// caller can fall through to the existing logic.
    ///
    /// Keeps each public method's body small — the dispatch shim is
    /// one match arm instead of a per-method
    /// `if let CollectionMode::Lazy(_)` ladder.
    fn dispatch_lazy<R, F>(&self, body: F) -> Option<Result<R>>
    where
        F: FnOnce(&Collection<'_, T>) -> Result<R>,
    {
        match &self.mode {
            CollectionMode::Lazy(LazyRef { db }) => {
                let name: Cow<'static, str> = Cow::Owned(self.collection_name.clone().into_owned());
                Some(db.read_transaction(move |tx| {
                    let coll = Collection::<T>::open_readonly_named(tx, name)?;
                    body(&coll)
                }))
            }
            _ => None,
        }
    }
}

fn lock_pager(
    env: &obj_core::TxnEnv<FileHandle>,
) -> Result<std::sync::MutexGuard<'_, Pager<FileHandle>>> {
    env.pager().lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

/// Persist the running-version (`T::VERSION`) schema for `collection_id`
/// into the catalog inside the caller's already-locked `WriteTxn`.
///
/// Delegates to [`Catalog::put_schema`], which is idempotent and
/// drift-guarded: a repeat call with the same shape is a no-op (no WAL
/// churn); a differing shape under the same `(collection_id, version)`
/// key returns [`Error::SchemaShapeChanged`](obj_core::Error::SchemaShapeChanged)
/// and rolls back the whole write txn. Persists ONLY `T::VERSION` — the
/// reader re-derives older shapes from rows older writers persisted.
///
/// `put_schema`'s error propagates via `?`.
fn persist_current_schema<T: Schema + Document>(
    catalog: &mut Catalog<FileHandle>,
    pager: &mut Pager<FileHandle>,
    collection_id: u32,
) -> Result<()> {
    catalog.put_schema(pager, collection_id, T::VERSION, &<T as Schema>::schema())
}

/// Decode a primary-tree record inside a `WriteTxn` (no reader
/// snapshot available) by sourcing any stored-version schema from the
/// writer's own live / pending catalog tree.
///
/// The header is peeked first: an equal-version record (`type_version
/// == T::VERSION`) decodes via the [`decode_with`] fast path with an
/// empty `Resolved` map — `src` is never consulted there. Only the
/// migration branch (`type_version < T::VERSION`) costs a catalog
/// descent: the stored row is fetched via [`Catalog::get_schema_in_txn`]
/// into a one-entry map and handed to `decode_with` as
/// [`SchemaSource::Resolved`]. A missing row is **never** silently
/// dropped — `decode_with` surfaces
/// [`Error::SchemaNotRegistered`](obj_core::Error::SchemaNotRegistered)
/// when the migration branch finds no entry.
///
/// Every fallible step propagates via `?`; no `unwrap` / `expect`.
fn decode_in_write_txn<T: Document>(
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    collection_id: u32,
    bytes: &[u8],
) -> Result<T> {
    let header = DocumentHeader::read_from(bytes)?;
    let mut schemas: BTreeMap<u32, StoredSchema> = BTreeMap::new();
    if header.type_version < T::VERSION {
        if let Some(stored) =
            catalog.get_schema_in_txn(pager, collection_id, header.type_version)?
        {
            schemas.insert(header.type_version, stored);
        }
    }
    decode_with::<T, FileHandle>(
        bytes,
        collection_id,
        SchemaSource::Resolved { schemas: &schemas },
    )
}

/// Ensure `T::COLLECTION` exists in the catalog, lazy-creating an
/// empty primary B-tree on first call.  Used on the write side.
fn ensure_collection<T: Document>(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
) -> Result<CollectionDescriptor> {
    let mut pager = inner.lock_pager()?;
    let mut catalog_guard = lock_catalog(catalog)?;
    if let Some(d) = catalog_guard.get(&mut pager, T::COLLECTION)? {
        return Ok(d);
    }
    let tree = BTree::<FileHandle>::empty(&mut pager)?;
    let descriptor = CollectionDescriptor::new(0, tree.root().get(), T::VERSION);
    let _id = catalog_guard.insert(&mut pager, T::COLLECTION, descriptor)?;
    catalog_guard
        .get(&mut pager, T::COLLECTION)?
        .ok_or(Error::Corruption { page_id: 0 })
}

/// Re-read the descriptor for `T::COLLECTION` after any catalog
/// mutation. Used by [`Collection::open_or_create`] after the
/// reconciler runs.
fn reread_descriptor<T: Document>(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
) -> Result<CollectionDescriptor> {
    let mut pager = inner.lock_pager()?;
    let catalog_guard = lock_catalog(catalog)?;
    catalog_guard
        .get(&mut pager, T::COLLECTION)?
        .ok_or(Error::Corruption { page_id: 0 })
}

/// Reconcile `T::indexes()` against the catalog's stored descriptors
/// on the FIRST call per process per `(collection, version)`.
/// Subsequent calls for the same `(collection, version)` observe the
/// cache hit (in the shared `reconciled` set OR this txn's `staged`
/// set) and skip the catalog walk.
///
/// Reconciliation runs inside the user's WAL transaction so a
/// rolled-back txn leaves the catalog clean. If reconciliation
/// fails (e.g. `Error::IndexKindMismatch`), neither set is populated
/// so the next attempt re-runs the reconciler.
///
/// The skip-check is `shared ∪ staged`: a second handle of the
/// same `(collection, version)` within ONE txn still skips the
/// (idempotent) catalog walk via `staged`, but the key is recorded in
/// the per-txn `staged` set, NOT the shared set.
/// [`crate::WriteTxn::commit`] promotes the staged keys into the shared
/// set only after the WAL commit lands — so a rolled-back lazy-create
/// can never poison the shared cache into skipping reconciliation on a
/// later txn (the original bug).
fn reconcile_indexes_once<T: Document>(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
    reconciled: &Arc<Mutex<HashSet<(String, u32)>>>,
    staged: &mut HashSet<(String, u32)>,
) -> Result<()> {
    reconcile_specs_once(
        inner,
        catalog,
        reconciled,
        staged,
        T::COLLECTION,
        T::VERSION,
        &T::indexes(),
    )
}

/// Non-generic core of index reconciliation, shared by the generic
/// `#[derive(Document)]` path ([`reconcile_indexes_once`]) and the
/// public raw seam ([`crate::WriteTxn::reconcile_indexes_raw`]).
///
/// Reconciles `specs` against the catalog's stored descriptors for
/// `collection` on the FIRST call per process per `(collection,
/// version)`, honoring the same `shared ∪ staged` skip-cache (keyed by
/// `(collection, version)`), per-txn staging, and pre-mutation
/// validation as the generic path — the only difference is that the
/// collection name, version, and spec list are arguments rather than
/// `T::COLLECTION` / `T::VERSION` / `T::indexes()`.
///
/// # Why the cache key includes `version`
///
/// `Catalog::reconcile_indexes` is a FULL reconcile: it declares specs
/// missing from the catalog AND drops `Active` descriptors absent from
/// `specs`. Keying the skip-cache by `collection` ALONE meant that once
/// a process reconciled a collection at one schema version, a LATER
/// version in the same process that ADDED a new index never reconciled
/// — the added index never became `Active` and index-maintaining writes
/// failed with `IndexNotFound`. Keying by `(collection, version)`
/// reconciles each version exactly once: the common single-version case
/// is unchanged (one key, reconciled once), and a cross-version index
/// ADD reconciles the new version's specs on its first write.
///
/// ## Caveat — conflicting index REMOVAL interleaved across versions
///
/// Because each `(collection, version)` reconciles independently and
/// `reconcile_indexes` drops `Active` indexes absent from the version's
/// specs, alternating writes between two live versions of the SAME
/// collection in ONE process — where the versions declare DIFFERENT
/// index sets — can leave the catalog reflecting whichever version
/// reconciled most recently (its specs drive the drop set). Index
/// ADDITION (the common monotonic schema-evolution case) is fully
/// correct. The removal-interleaving edge is a narrow,
/// single-process anti-pattern.
///
/// The caller MUST have ensured `collection` exists in the catalog
/// (the generic path runs [`ensure_collection`] first; the raw seam
/// runs [`crate::txn::ensure_collection_raw`]) — `reconcile_indexes`
/// errors with `CollectionNotFound` otherwise.
pub(crate) fn reconcile_specs_once(
    inner: &obj_core::WriteTxn<'_, FileHandle>,
    catalog: &Arc<Mutex<Catalog<FileHandle>>>,
    reconciled: &Arc<Mutex<HashSet<(String, u32)>>>,
    staged: &mut HashSet<(String, u32)>,
    collection: &str,
    version: u32,
    specs: &[obj_core::IndexSpec],
) -> Result<()> {
    let key = (collection.to_owned(), version);
    if staged.contains(&key) {
        return Ok(());
    }
    {
        let cache = lock_reconciled(reconciled)?;
        if cache.contains(&key) {
            return Ok(());
        }
    }
    for spec in specs {
        spec.validate()?;
    }
    {
        let mut pager = inner.lock_pager()?;
        let mut catalog_guard = lock_catalog(catalog)?;
        let _post = catalog_guard.reconcile_indexes(&mut pager, collection, specs)?;
    }
    staged.insert(key);
    Ok(())
}

fn lock_reconciled(
    reconciled: &Arc<Mutex<HashSet<(String, u32)>>>,
) -> Result<std::sync::MutexGuard<'_, HashSet<(String, u32)>>> {
    reconciled.lock().map_err(|_| Error::Busy {
        kind: obj_core::LockKind::WriterInProcess,
    })
}

/// Read-side descriptor lookup against a caller-supplied
/// collection name. This byte-shape lets the
/// namespace-aware [`Collection::open_readonly`] perform the
/// catalog walk against either the calling Db's snapshot (no
/// namespace) or an attached Db's snapshot (with the namespace
/// prefix stripped).
fn read_descriptor_via_snapshot_named(
    env: &obj_core::TxnEnv<FileHandle>,
    snapshot: &obj_core::ReaderSnapshot<FileHandle>,
    name: &str,
) -> Result<Option<CollectionDescriptor>> {
    let pager = lock_pager(env)?;
    Catalog::<FileHandle>::lookup_via_snapshot(&pager, snapshot, name)
}

/// Fused one-shot point read for [`crate::Db::get`].
///
/// Resolves the collection descriptor, reads the primary-tree value for
/// `id`, and decodes `T` under a SINGLE pager-mutex acquisition (see
/// [`snapshot_resolve_get_decode`]). This collapses the two back-to-back
/// pager locks the equivalent handle path pays (one to open the
/// handle / resolve the descriptor, one for the `get`).
///
/// Observably identical to
/// `tx.collection::<T>()?.get(id)` for the one-shot caller: the
/// descriptor lookup, the value get, and the stored-schema descent
/// all run on the same `ReadTxn` snapshot, and a missing collection
/// still surfaces as [`Error::CollectionNotFound`]. Namespaced reads
/// (`<ns>.<tail>`) fall back to the handle path so the attached-snapshot
/// dispatch is unchanged.
pub(crate) fn fused_point_get<T: Document>(tx: &ReadTxn<'_>, id: Id) -> Result<Option<T>> {
    let (namespace, _tail) = crate::db::split_namespace(T::COLLECTION);
    if namespace.is_some() {
        return Collection::<T>::open_readonly(tx)?.get(id);
    }
    let key = id.to_be_bytes();
    snapshot_resolve_get_decode::<T>(tx.inner.snapshot(), tx.inner.env(), T::COLLECTION, &key)
}

/// Re-read the descriptor inside an already-locked pager + catalog
/// pair.  Surfaces a missing collection as `Error::Corruption`
/// because the caller has already opened a write txn against it.
fn catalog_get_required(
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    name: &str,
) -> Result<CollectionDescriptor> {
    catalog
        .get(pager, name)?
        .ok_or(Error::Corruption { page_id: 0 })
}

fn btree_handle(pager: &Pager<FileHandle>, root: u64) -> Result<BTree<FileHandle>> {
    let root_pid =
        PageId::new(root).ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    BTree::<FileHandle>::open(pager, root_pid)
}

/// Drain a B-tree range iterator, counting entries WITHOUT retaining
/// their bytes. Shared by [`Collection::count_all`]'s live (write) and
/// snapshot-pinned (read) scan arms. The iterator carries its own
/// `MAX_RANGE_NODES` budget; the `u64` overflow check guards the
/// count itself.
fn count_range_iter<I>(iter: I) -> Result<u64>
where
    I: Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>,
{
    let mut n: u64 = 0;
    for step in iter {
        let _ = step?;
        n = n.checked_add(1).ok_or(Error::BTreeInvariantViolated {
            reason: "primary tree entry count exceeds u64",
        })?;
    }
    Ok(n)
}

/// Snapshot-consistent B-tree lookup.
///
/// Walks the primary B+tree rooted at `primary_root` using
/// [`obj_core::btree::BTree::get_via_snapshot`], which descends
/// through [`obj_core::ReaderSnapshot::read_page`] rather than the
/// live `Pager::read_page`. This bypasses the WAL `state.view` /
/// `state.pending` overlays — a concurrent writer's post-snapshot
/// COW commits cannot poison the reader's walk.
///
/// `primary_root` MUST be the descriptor's `primary_root` as-of
/// the snapshot's pinned LSN (i.e. the value read via
/// [`obj_core::Catalog::lookup_via_snapshot`] in
/// [`read_descriptor_via_snapshot`] above). Using the writer's
/// live `primary_root` would defeat the snapshot read.
///
/// Takes an ALREADY-LOCKED pager so the caller can keep the
/// same guard open to feed [`SchemaSource::Snapshot`] into
/// [`decode_with`] — the document bytes and the stored-version schema
/// row are then resolved under one pager-mutex acquisition, at the
/// same pinned LSN.
fn snapshot_get_locked(
    pager: &Pager<FileHandle>,
    snap: &obj_core::ReaderSnapshot<FileHandle>,
    primary_root: u64,
    key: &[u8],
) -> Result<Option<Vec<u8>>> {
    let root_pid = PageId::new(primary_root)
        .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    obj_core::btree::BTree::<FileHandle>::get_via_snapshot(pager, snap, root_pid, key)
}

/// Fused single-lock read. Resolves the descriptor for
/// `name` and performs the primary-tree `get` for `key` under ONE
/// pager-mutex acquisition, against the SAME `snapshot` — collapsing
/// the descriptor-lookup lock (`read_descriptor_via_snapshot_named`)
/// and the value-get lock (`snapshot_get_via_btree`) that the
/// two-call handle path pays back-to-back.
///
/// Resolves the descriptor, reads the value, AND decodes `T` — all
/// under the one pager lock. A missing collection surfaces as
/// `Err(CollectionNotFound)` (matching the handle path's open-time
/// contract for the one-shot caller); a present collection with no
/// entry for `key` surfaces as `Ok(None)`.
///
/// The decode is dispatched through [`SchemaSource::Snapshot`] on
/// the SAME pinned snapshot used to resolve the descriptor and read the
/// value, so an older-version record's stored schema is read at the
/// same LSN as everything else. The pager lock is held across the
/// decode so the schema-row B+tree descent shares it.
///
/// Keeps poison → `Error::Busy` (via `lock_pager`); `debug_assert`s
/// that the snapshot-resolved `primary_root` is the one fed to the
/// get.
fn snapshot_resolve_get_decode<T: Document>(
    snap: &obj_core::ReaderSnapshot<FileHandle>,
    env: &obj_core::TxnEnv<FileHandle>,
    name: &str,
    key: &[u8],
) -> Result<Option<T>> {
    let pager = lock_pager(env)?;
    let Some(descriptor) = Catalog::<FileHandle>::lookup_via_snapshot(&pager, snap, name)? else {
        return Err(Error::CollectionNotFound {
            name: name.to_owned(),
        });
    };
    let root_pid = PageId::new(descriptor.primary_root)
        .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
    debug_assert_eq!(
        root_pid.get(),
        descriptor.primary_root,
        "fused get must descend the snapshot-resolved primary_root",
    );
    let Some(value) =
        obj_core::btree::BTree::<FileHandle>::get_via_snapshot(&pager, snap, root_pid, key)?
    else {
        return Ok(None);
    };
    let collection_id = descriptor.collection_id;
    Ok(Some(decode_with::<T, FileHandle>(
        &value,
        collection_id,
        SchemaSource::Snapshot {
            pager: &pager,
            snapshot: snap,
            collection_id,
        },
    )?))
}

/// Collect every `(Id, raw-bytes)` pair under `primary_root` into an
/// owned `Vec`. The range iterator borrows `&mut pager`, so the bytes
/// are materialized BEFORE decode — that releases the iterator (and its
/// pager borrow) so the second pass can re-borrow the pager / catalog
/// to source stored-version schemas.
///
/// The iterator carries its own bounded `MAX_RANGE_NODES` budget.
fn collect_raw_rows(
    pager: &mut Pager<FileHandle>,
    primary_root: u64,
) -> Result<Vec<(Id, Vec<u8>)>> {
    let tree = btree_handle(pager, primary_root)?;
    let iter = tree.range(pager, ..)?;
    let mut out = Vec::new();
    for entry in iter {
        let (key, value) = entry?;
        let id = Id::from_be_bytes(&key)
            .ok_or(Error::InvalidArgument("primary B-tree key is not an Id"))?;
        out.push((id, value));
    }
    Ok(out)
}

/// Write-side full scan: collect raw rows, then decode each through the
/// writer's live catalog tree. An older-version row sources
/// its stored schema via [`Catalog::get_schema_in_txn`]; an equal-version
/// row never consults the catalog.
fn scan_all_in_write_txn<T: Document>(
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    primary_root: u64,
    collection_id: u32,
) -> Result<Vec<(Id, T)>> {
    let rows = collect_raw_rows(pager, primary_root)?;
    let mut out = Vec::with_capacity(rows.len());
    for (id, bytes) in rows {
        let doc = decode_in_write_txn::<T>(pager, catalog, collection_id, &bytes)?;
        out.push((id, doc));
    }
    Ok(out)
}

/// Read-side full scan: collect raw rows, then decode each through the
/// reader's pinned [`SchemaSource::Snapshot`]. An
/// older-version row's stored schema is resolved at the SAME snapshot
/// LSN as the document bytes; an equal-version row never consults it.
fn snapshot_scan_via_btree<T: Document>(
    snap: &obj_core::ReaderSnapshot<FileHandle>,
    env: &obj_core::TxnEnv<FileHandle>,
    primary_root: u64,
    collection_id: u32,
) -> Result<Vec<(Id, T)>> {
    let mut pager = lock_pager(env)?;
    let rows = collect_raw_rows(&mut pager, primary_root)?;
    let mut out = Vec::with_capacity(rows.len());
    for (id, bytes) in rows {
        let doc = decode_with::<T, FileHandle>(
            &bytes,
            collection_id,
            SchemaSource::Snapshot {
                pager: &pager,
                snapshot: snap,
                collection_id,
            },
        )?;
        out.push((id, doc));
    }
    Ok(out)
}

/// Encode the caller-supplied `Dynamic` value(s) into the bytes a
/// lookup against `descriptor` would use as a B-tree key. For
/// `Unique` indexes the result is the key bytes verbatim; for
/// non-unique kinds the lookup helpers extend with the per-doc
/// id suffix at scan time.
fn index_key_for_lookup(
    descriptor: &obj_core::IndexDescriptor,
    fields: &[obj_core::codec::Dynamic],
) -> Result<obj_core::index::EncodedIndexKey> {
    obj_core::index::encode_index_key_parts(descriptor.kind, &descriptor.key_paths, fields)
}

/// Encode a `Bound<&Dynamic>` into the index-key `Bound<Vec<u8>>` the
/// B-tree scan uses. Shared by the `Dynamic`-taking range methods on
/// [`Collection`].
///
/// A scalar `Dynamic` is encoded with the order-preserving field
/// encoder ([`obj_core::index::encode_field`]) — byte-identical to
/// what [`crate::Query::index_range`] produces, so a query and a
/// direct collection scan over the same scalar bound observe the
/// same entries. A [`Dynamic::Seq`](obj_core::codec::Dynamic::Seq)
/// bound is encoded as a composite key (the
/// [`COMPOSITE_TAG`](obj_core::index::COMPOSITE_TAG)-prefixed
/// concatenation of each element's field encoding) so a `Composite`
/// index can be range-scanned by a full tuple bound.
fn encode_dynamic_bound(
    b: std::ops::Bound<&obj_core::codec::Dynamic>,
) -> Result<std::ops::Bound<Vec<u8>>> {
    match b {
        std::ops::Bound::Included(v) => Ok(std::ops::Bound::Included(encode_bound_value(v)?)),
        std::ops::Bound::Excluded(v) => Ok(std::ops::Bound::Excluded(encode_bound_value(v)?)),
        std::ops::Bound::Unbounded => Ok(std::ops::Bound::Unbounded),
    }
}

/// Encode one `Dynamic` bound value into index-key bytes. Scalars go
/// through [`obj_core::index::encode_field`]; a `Seq` is encoded as a
/// composite tuple key. Kept separate so [`encode_dynamic_bound`]
/// stays a thin three-arm match.
fn encode_bound_value(v: &obj_core::codec::Dynamic) -> Result<Vec<u8>> {
    match v {
        obj_core::codec::Dynamic::Seq(fields) => {
            let mut out = vec![obj_core::index::COMPOSITE_TAG];
            for f in fields {
                out.extend_from_slice(obj_core::index::encode_field(f)?.as_bytes());
            }
            Ok(out)
        }
        _ => Ok(obj_core::index::encode_field(v)?.into_bytes()),
    }
}

/// Append 8 `0xFF` bytes to `prefix`. Used as the exclusive upper
/// bound of an equality lookup against a non-unique index: every
/// key with the same user-prefix is ≤ `prefix || 0xFF..` because
/// the trailing 8 bytes are an `Id` (`u64` BE).
fn append_max_id(prefix: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(prefix.len() + 8);
    out.extend_from_slice(prefix);
    out.extend_from_slice(&u64::MAX.to_be_bytes());
    out
}

/// Trim the trailing 8-byte id suffix off a non-unique index key.
/// For `Unique` keys the suffix is absent, so the full key is the
/// user portion.
fn strip_id_suffix(full_key: &[u8], kind: obj_core::IndexKind) -> Vec<u8> {
    match kind {
        obj_core::IndexKind::Unique => full_key.to_vec(),
        _ if full_key.len() >= 8 => full_key[..full_key.len() - 8].to_vec(),
        _ => full_key.to_vec(),
    }
}

/// Recover the `Id` (as a `u64`) from one index B-tree entry. For
/// non-unique kinds the id is the trailing 8 bytes of the KEY (the
/// suffix appended by the maintenance path); for `Unique` keys the
/// id is the VALUE. Used by
/// [`Collection::count_distinct_ids_in_range`].
fn id_from_index_entry(full_key: &[u8], value: &[u8], kind: obj_core::IndexKind) -> Result<u64> {
    let bytes: &[u8] = if kind == obj_core::IndexKind::Unique {
        value
    } else {
        if full_key.len() < 8 {
            return Err(Error::Corruption { page_id: 0 });
        }
        &full_key[full_key.len() - 8..]
    };
    let id = Id::from_be_bytes(bytes).ok_or(Error::Corruption { page_id: 0 })?;
    Ok(id.get())
}

/// Resumption marker for [`IterIndexRange`]'s first refill. After the
/// first batch the iterator switches to `Excluded(last_emitted_full_key)`
/// for subsequent refills (the same shape `Db::iter_all` uses for the
/// primary tree).
enum InitialResume {
    Included(Vec<u8>),
    Excluded(Vec<u8>),
    Unbounded,
}

/// One entry in [`IterIndexRange`]'s pending buffer. Read/Write
/// modes stage `Pending(key, id)` and resolve the `T` lazily on
/// `next()`; Lazy mode pre-resolves under a single `read_transaction`
/// (to preserve snapshot consistency across the index walk + the
/// per-row primary `get`) and stages `Resolved(key, T)` directly.
enum StagedEntry<T> {
    Pending(Vec<u8>, Id),
    Resolved(Vec<u8>, T),
}

/// Streaming iterator returned by [`Collection::iter_range`]. Yields
/// `Result<(user_key_bytes, T)>` one row at a time; internally
/// refills a fixed-size `(user_key, Id)` buffer in batches of
/// `ITER_INDEX_RANGE_BATCH = 256` so the per-step pager-lock cost
/// amortises. Memory stays bounded at `O(batch × small_bytes +
/// distinct_ids)` regardless of the range's total size.
///
/// Held data: a `&'a Collection<'_, T>` borrow (the iterator is bound
/// to the lifetime of `Collection::iter_range`'s `&self` borrow), the
/// index's root page-id, the dedup set for `Each` indexes, the next-
/// chunk resumption marker, and the staged batch.
///
/// Items are `Result` because iteration is fallible per step — a
/// pager read or codec decode can fail mid-scan, surfacing as
/// `Some(Err(_))` without ending iteration. Unwrap each step with
/// `?`: `for step in iter { let (key, doc) = step?; }`. For the
/// eager, single-`?` alternative that materialises every match up
/// front, see [`Collection::index_range`].
pub struct IterIndexRange<'a, T: Document> {
    coll: &'a Collection<'a, T>,
    descriptor_kind: obj_core::IndexKind,
    index_root: u64,
    /// First-refill marker — `None` after the iterator has emitted
    /// at least one chunk; subsequent refills use `last_full_key`.
    initial_resume: Option<InitialResume>,
    /// Last full B-tree key emitted by the most recent refill. Drives
    /// the `Excluded(_)` resumption bound for the next chunk.
    last_full_key: Option<Vec<u8>>,
    /// User-supplied end bound (already widened per index kind).
    end_bound: Bound<Vec<u8>>,
    /// Pre-staged entries from the most recent refill. `next()`
    /// pops from the front. Each entry is either `Pending(key, id)`
    /// (deferred get-back, the Read/Write streaming path) or
    /// `Resolved(key, T)` (eager get-back inside a single
    /// `read_transaction`, the Lazy-mode fallback).
    buffer: VecDeque<Result<StagedEntry<T>>>,
    /// Persistent de-dup set for `Each` indexes. The set is
    /// intentionally unbounded — if the caller wants
    /// a hard cap they should use
    /// [`Collection::count_distinct_ids_in_range`] (which caps at
    /// [`MAX_DISTINCT_IDS`]); the iterator's correctness contract
    /// is per-row dedup across the whole range.
    emitted_ids: HashSet<u64>,
    finished: bool,
}

impl<T: Document> std::fmt::Debug for IterIndexRange<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterIndexRange")
            .field("descriptor_kind", &self.descriptor_kind)
            .field("index_root", &self.index_root)
            .field("buffer_len", &self.buffer.len())
            .field("emitted_ids_len", &self.emitted_ids.len())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<T: Document> IterIndexRange<'_, T> {
    /// Refill `self.buffer` with up to [`ITER_INDEX_RANGE_BATCH`]
    /// `(user_key, Id)` pairs by walking the index B-tree from the
    /// current resumption marker. Sets `self.finished` when the
    /// underlying range scan yields fewer than the requested batch
    /// (i.e. it ran past the end bound).
    ///
    /// Per-step decode errors are pushed into the buffer as `Err(_)`
    /// so the caller observes them via `next()` rather than aborting
    /// iteration.
    fn refill(&mut self) -> Result<()> {
        let root_pid = PageId::new(self.index_root)
            .ok_or(Error::InvalidArgument("index root_page_id is zero"))?;
        let start = self.next_start_bound();
        let end = clone_bound_ref(&self.end_bound);
        // Read mode walks the snapshot-pinned index root via
        // `range_via_snapshot` so a concurrent writer's post-snapshot
        // COW commits stay invisible — mirroring `collect_range`'s Read
        // arm. Write mode walks the live pager so the txn observes its
        // own uncommitted index mutations. Both refs are `Copy` and
        // outlive the borrow of `self.coll.mode`, so extracting them
        // here releases that borrow before the `&mut self` drain call.
        let (snapshot, env) = match &self.coll.mode {
            CollectionMode::Read(r) => (Some(r.snapshot), r.env),
            CollectionMode::Write(w) => (None, w.env),
            CollectionMode::Lazy(_) => {
                return Err(Error::ReadOnly {
                    operation: "internal: iter_range refill in Lazy mode",
                });
            }
        };
        let mut staged: VecDeque<Result<StagedEntry<T>>> =
            VecDeque::with_capacity(ITER_INDEX_RANGE_BATCH);
        let mut last_full: Option<Vec<u8>> = None;
        let mut pager = lock_pager(env)?;
        let consumed = if let Some(snap) = snapshot {
            let iter = BTree::<FileHandle>::range_via_snapshot(
                &pager, snap, root_pid, (start, end),
            )?;
            self.drain_into_staged(iter, &mut staged, &mut last_full)?
        } else {
            let tree = BTree::<FileHandle>::open(&pager, root_pid)?;
            let iter = tree.range(&mut pager, (start, end))?;
            self.drain_into_staged(iter, &mut staged, &mut last_full)?
        };
        if consumed < ITER_INDEX_RANGE_BATCH {
            self.finished = true;
        }
        drop(pager);
        self.buffer.extend(staged);
        if let Some(k) = last_full {
            self.last_full_key = Some(k);
        }
        Ok(())
    }

    /// Drain up to [`ITER_INDEX_RANGE_BATCH`] entries from a B-tree
    /// range iterator into `staged`, staging each via [`Self::stage_one`]
    /// and returning the number of entries consumed. Generic over the
    /// concrete iterator type so the same bounded loop serves both the
    /// live (`BTree::range`) Write path and the snapshot
    /// (`BTree::range_via_snapshot`) Read path. The loop is bounded by
    /// `ITER_INDEX_RANGE_BATCH` (R2).
    fn drain_into_staged<I>(
        &mut self,
        iter: I,
        staged: &mut VecDeque<Result<StagedEntry<T>>>,
        last_full: &mut Option<Vec<u8>>,
    ) -> Result<usize>
    where
        I: Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>,
    {
        let mut consumed: usize = 0;
        for step in iter {
            if consumed >= ITER_INDEX_RANGE_BATCH {
                break;
            }
            consumed = consumed
                .checked_add(1)
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "iter_range batch counter overflow",
                })?;
            self.stage_one(staged, last_full, step);
        }
        Ok(consumed)
    }

    /// Process one B-tree step into the staged batch. Encapsulates
    /// the `Each`-dedup, the trailing-id-suffix strip, and the
    /// `Id::from_be_bytes` parse. Free helper so the refill body
    /// stays small.
    fn stage_one(
        &mut self,
        staged: &mut VecDeque<Result<StagedEntry<T>>>,
        last_full: &mut Option<Vec<u8>>,
        step: Result<(Vec<u8>, Vec<u8>)>,
    ) {
        let (full_key, id_bytes) = match step {
            Ok(kv) => kv,
            Err(e) => {
                staged.push_back(Err(e));
                return;
            }
        };
        *last_full = Some(full_key.clone());
        let Some(id) = Id::from_be_bytes(&id_bytes) else {
            staged.push_back(Err(Error::Corruption { page_id: 0 }));
            return;
        };
        if self.descriptor_kind == obj_core::IndexKind::Each && !self.emitted_ids.insert(id.get()) {
            return;
        }
        let user_key = strip_id_suffix(&full_key, self.descriptor_kind);
        staged.push_back(Ok(StagedEntry::Pending(user_key, id)));
    }

    /// Compute the start bound for the next refill: use
    /// `initial_resume` on the first call (consuming it), thereafter
    /// use `Excluded(last_full_key)`.
    fn next_start_bound(&mut self) -> Bound<Vec<u8>> {
        if let Some(initial) = self.initial_resume.take() {
            return match initial {
                InitialResume::Included(k) => Bound::Included(k),
                InitialResume::Excluded(k) => Bound::Excluded(k),
                InitialResume::Unbounded => Bound::Unbounded,
            };
        }
        match &self.last_full_key {
            Some(k) => Bound::Excluded(k.clone()),
            None => Bound::Unbounded,
        }
    }
}

impl<T: Document> Iterator for IterIndexRange<'_, T> {
    type Item = Result<(Vec<u8>, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(staged) = self.buffer.pop_front() {
                return Some(self.resolve_one(staged));
            }
            if self.finished {
                return None;
            }
            if let Err(e) = self.refill() {
                self.finished = true;
                return Some(Err(e));
            }
        }
    }
}

impl<T: Document> IterIndexRange<'_, T> {
    /// Resolve one staged entry into a `(user_key, T)` pair. For
    /// `Pending(_, id)` entries (the Read/Write streaming path),
    /// calls [`Collection::get`] to decode `T` on demand; for
    /// `Resolved(_, T)` entries (the Lazy-mode eager path), returns
    /// the already-decoded value. Orphan index entries (id missing
    /// in the primary tree) surface as [`Error::Corruption`],
    /// matching [`Collection::index_range`]'s existing contract.
    fn resolve_one(&self, staged: Result<StagedEntry<T>>) -> Result<(Vec<u8>, T)> {
        match staged? {
            StagedEntry::Pending(user_key, id) => match self.coll.get(id)? {
                Some(doc) => Ok((user_key, doc)),
                None => Err(Error::Corruption { page_id: 0 }),
            },
            StagedEntry::Resolved(user_key, doc) => Ok((user_key, doc)),
        }
    }
}

/// Clone a `&Bound<Vec<u8>>` into an owned `Bound<Vec<u8>>`. Takes a
/// borrowed owned bound (the shape `IterIndexRange::end_bound`
/// stores) and hands back an owned copy for the resumption walk.
fn clone_bound_ref(b: &Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(v.clone()),
        Bound::Excluded(v) => Bound::Excluded(v.clone()),
        Bound::Unbounded => Bound::Unbounded,
    }
}
