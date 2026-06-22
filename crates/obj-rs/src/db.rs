//! `Db` — public entry point.
//!
//! Wraps `obj_core::TxnEnv` + the catalog into the user-facing API.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::ops::Bound;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use obj_core::btree::BTree;
use obj_core::pager::page::PageId;
use obj_core::pager::{lock_path_for, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor, Document, Error, Id, Result, TxnEnv};

use crate::config::Config;
use crate::txn::{AttachedDb, ReadTxn, WriteTxn};

/// Per-batch refill size for [`IterAll`]. The iterator yields one
/// document at a time to the caller, but internally fetches the
/// primary B-tree in `BATCH` chunks so the per-step pager-lock
/// acquisition cost amortises over many `next` calls. The buffer is
/// fixed-size (constant 256 entries — at ~512 bytes/doc that's
/// ~128 KiB peak); the buffer does NOT scale with the collection's
/// total size.
const ITER_ALL_BATCH: usize = 256;

/// The embedded document database.
///
/// `Db` is `Send + Sync`; share across threads via `Arc<Db>` for
/// concurrent reader / single-writer access.
///
/// The public `Db` is hard-typed against `obj_core::FileHandle`.  A
/// future refactor may make it generic over `F: FileBackend` so
/// fault-injection harnesses can build on the same API; today the
/// test helpers reach for the lower-level `obj-core` building blocks
/// instead.
pub struct Db {
    pub(crate) env: Arc<TxnEnv<FileHandle>>,
    pub(crate) catalog: Arc<Mutex<Catalog<FileHandle>>>,
    pub(crate) readonly: bool,
    pub(crate) busy_timeout: Duration,
    /// Per-process cache of `(collection, version)` keys whose
    /// `Document::indexes()` reconciliation has already run.
    /// Reconciliation is idempotent but a catalog walk +
    /// declaration cycle is non-trivial — caching ensures only the
    /// first `WriteTxn::collection::<T>()` call per process per
    /// `(collection, version)` pays the cost. The version is part
    /// of the key so a later schema version that ADDS an index still
    /// reconciles (the name-only key skipped it, leaving the new index
    /// never `Active`).
    pub(crate) reconciled: Arc<Mutex<HashSet<(String, u32)>>>,
    /// Registry of attached read-only databases keyed by
    /// namespace. Populated by [`Db::attach`]; consulted by
    /// [`Db::collection_namespace`] and by
    /// [`Db::read_transaction`] when it pins per-attached
    /// snapshots. `Arc<Mutex<_>>` because `attach` / `detach` take
    /// `&mut self` while live read transactions hold borrows into
    /// the registry; the mutex coordinates the two.
    pub(crate) attached: Arc<Mutex<HashMap<String, AttachedDb>>>,
    /// Published length of `attached`, mutated only while the
    /// `attached` mutex is held. A relaxed load lets the hot read path
    /// ([`Self::pin_attached_snapshots`]) skip the mutex acquire when
    /// no databases are attached (the overwhelmingly common case).
    pub(crate) attached_len: Arc<std::sync::atomic::AtomicUsize>,
}

impl std::fmt::Debug for Db {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Db")
            .field("readonly", &self.readonly)
            .field("busy_timeout", &self.busy_timeout)
            .finish_non_exhaustive()
    }
}

impl Db {
    /// Shared environment Arc — pager + cross-process lock file.
    ///
    /// Used by [`libobj`](../../libobj/index.html) to build owned
    /// transaction handles whose lifetime extends past a single
    /// `Db::transaction` closure call.
    #[doc(hidden)]
    #[must_use]
    pub fn env_arc(&self) -> Arc<TxnEnv<FileHandle>> {
        Arc::clone(&self.env)
    }

    /// Shared catalog Arc.
    ///
    /// Used by [`libobj`](../../libobj/index.html) for the same
    /// reason as [`Self::env_arc`].
    #[doc(hidden)]
    #[must_use]
    pub fn catalog_arc(&self) -> Arc<Mutex<Catalog<FileHandle>>> {
        Arc::clone(&self.catalog)
    }

    /// Shared per-process reconciliation cache Arc.
    ///
    /// Used by [`libobj`](../../libobj/index.html) for the same
    /// reason as [`Self::env_arc`].
    #[doc(hidden)]
    #[must_use]
    pub fn reconciled_arc(&self) -> Arc<Mutex<HashSet<(String, u32)>>> {
        Arc::clone(&self.reconciled)
    }

    /// Busy-lock timeout configured at open time.
    #[doc(hidden)]
    #[must_use]
    pub fn busy_timeout(&self) -> Duration {
        self.busy_timeout
    }
}

impl Db {
    /// Open or create a file-backed database at `path` with default
    /// configuration.
    ///
    /// Creates the file if absent; reopens otherwise. A `Db` is
    /// `Send + Sync` — share across threads via `Arc<Db>` for the
    /// concurrent-reader / single-writer workload.
    ///
    /// For an ephemeral database use [`Db::memory`]; for a
    /// read-only handle that coexists with another process's writer
    /// use [`Db::open_readonly`]; for custom durability /
    /// cache / lock knobs use [`Db::open_with`] + [`Config`].
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    ///
    /// let dir = tempfile::tempdir()?;
    ///
    /// // File-backed. Creates the file if absent; reopens otherwise.
    /// let _db = Db::open(dir.path().join("app.obj"))?;
    ///
    /// // In-memory. No persistence, no file locks. Useful for tests.
    /// let _mem = Db::memory()?;
    ///
    /// // Read-only. Coexists safely with a writer in another process.
    /// // Every mutating call returns `Err(Error::ReadOnly { ... })`.
    /// let _ro = Db::open_readonly(dir.path().join("app.obj"))?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the underlying [`Error`] from
    /// [`obj_core::pager::Pager::open`] on syscall or format
    /// failure.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "db.open",
            level = "info",
            skip_all,
            fields(path = %path.as_ref().display()),
        )
    )]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, Config::default())
    }

    /// Open or create a file-backed database with `config`.
    ///
    /// # Errors
    ///
    /// As [`Db::open`].
    // allow: `config` is consumed by value — `std::mem::take(&mut config.pager)` moves a field out and the rest is read into `from_parts`; a borrow would not let us take ownership of `pager`.
    #[allow(clippy::needless_pass_by_value)]
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "db.open",
            level = "info",
            skip_all,
            fields(path = %path.as_ref().display()),
        )
    )]
    pub fn open_with<P: AsRef<Path>>(path: P, mut config: Config) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let pager_config = std::mem::take(&mut config.pager);
        let pager = Pager::open(&path_buf, pager_config)?;
        let lock_file = if config.cross_process_lock {
            let lock_path = lock_path_for(&path_buf);
            let handle = FileHandle::open_or_create(&lock_path)?;
            handle.set_len(128)?;
            Some(Arc::new(handle))
        } else {
            None
        };
        Self::from_parts(pager, lock_file, &config)
    }

    /// Open a fresh in-memory database.  No persistence, no file
    /// locks.  Useful for unit tests and ephemeral workloads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] only if `config` has
    /// zero cache frames.
    pub fn memory() -> Result<Self> {
        Self::memory_with(Config::default())
    }

    /// As [`Db::memory`] with a caller-supplied [`Config`].
    ///
    /// # Errors
    ///
    /// As [`Db::memory`].
    // allow: `config` is consumed by value — `std::mem::take(&mut config.pager)` moves a field out and the rest is read into `from_parts`; a borrow would not let us take ownership of `pager`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn memory_with(mut config: Config) -> Result<Self> {
        let pager_config = std::mem::take(&mut config.pager);
        let pager = Pager::memory(pager_config)?;
        Self::from_parts(pager, None, &config)
    }

    /// Open the database at `path` in read-only mode.  The
    /// resulting `Db` rejects every mutating operation with
    /// `Err(Error::ReadOnly { ... })`.  Coexists safely with a
    /// writer in another process via the cross-process reader
    /// lock.
    ///
    /// # Errors
    ///
    /// As [`Db::open`].
    pub fn open_readonly<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = Config {
            readonly: true,
            ..Config::default()
        };
        Self::open_with(path, config)
    }

    fn from_parts(
        mut pager: Pager<FileHandle>,
        lock_file: Option<Arc<FileHandle>>,
        config: &Config,
    ) -> Result<Self> {
        pager.begin_txn();
        let init = Catalog::open_or_init(&mut pager);
        let catalog = match init {
            Ok(c) => {
                let commit_result = pager.commit();
                pager.end_txn();
                commit_result?;
                c
            }
            Err(e) => {
                pager.end_txn();
                return Err(e);
            }
        };
        if !config.skip_open_check {
            let report = obj_core::integrity::quick_check(&mut pager)?;
            if let Some(err) = first_failure_as_error(&report) {
                return Err(err);
            }
        }
        Ok(Self {
            env: Arc::new(TxnEnv::new(pager, lock_file)),
            catalog: Arc::new(Mutex::new(catalog)),
            readonly: config.readonly,
            busy_timeout: config.busy_timeout,
            reconciled: Arc::new(Mutex::new(HashSet::new())),
            attached: Arc::new(Mutex::new(HashMap::new())),
            attached_len: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Run a closure inside a write transaction.
    ///
    /// Begins a [`WriteTxn`], runs the closure with `&mut tx`.  If
    /// the closure returns `Ok(r)`, the transaction is committed and
    /// `Ok(r)` is returned.  If the closure returns `Err(e)`, the
    /// transaction is rolled back and `Err(e)` is returned.  A
    /// panic inside the closure unwinds with an implicit rollback
    /// via the `WriteTxn` `Drop` impl.
    ///
    /// # Examples
    ///
    /// Atomic batch — both inserts commit together or not at all:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// struct Order {
    ///     customer_id: u64,
    ///     total_cents: u64,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("txn.obj"))?;
    ///
    /// let (a, b) = db.transaction(|tx| {
    ///     let coll = tx.collection::<Order>()?;
    ///     let a = coll.insert(Order { customer_id: 1, total_cents: 50 })?;
    ///     let b = coll.insert(Order { customer_id: 2, total_cents: 200 })?;
    ///     Ok((a, b))
    /// })?;
    /// assert_ne!(a, b, "freshly-allocated ids are distinct");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Returning `Err(_)` rolls every staged write back; the `Err`
    /// the closure returns is the `Err` the caller sees:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::{Db, Error};
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// struct Order { total_cents: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("rollback.obj"))?;
    /// let id = db.insert(Order { total_cents: 10 })?;
    ///
    /// let outcome: obj::Result<()> = db.transaction(|tx| {
    ///     let coll = tx.collection::<Order>()?;
    ///     coll.update(id, |o| { o.total_cents = 99_999; })?;
    ///     Err(Error::InvalidArgument("synthetic abort"))
    /// });
    /// assert!(matches!(outcome, Err(Error::InvalidArgument(_))));
    ///
    /// let after: Order = db
    ///     .get::<Order>(id)?
    ///     .ok_or(Error::InvalidArgument("just inserted"))?;
    /// assert_eq!(after.total_cents, 10, "rolled-back update is invisible");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] if the database was opened read-only.
    /// - [`Error::Busy`] if a sibling transaction holds the lock(s).
    /// - Any error the closure returns.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "db.transaction", level = "info", skip_all)
    )]
    pub fn transaction<R, F>(&self, body: F) -> Result<R>
    where
        F: FnOnce(&mut WriteTxn<'_>) -> Result<R>,
    {
        if self.readonly {
            return Err(Error::ReadOnly {
                operation: "transaction",
            });
        }
        #[cfg(feature = "tracing")]
        tracing::debug!("begin");
        let inner = obj_core::WriteTxn::begin(&self.env, self.busy_timeout)?;
        let mut tx = WriteTxn::new(
            inner,
            Arc::clone(&self.catalog),
            Arc::clone(&self.reconciled),
        );
        match body(&mut tx) {
            Ok(value) => {
                tx.commit()?;
                #[cfg(feature = "tracing")]
                tracing::debug!("commit");
                Ok(value)
            }
            Err(e) => {
                let _ = tx.rollback();
                let _ = self.refresh_catalog();
                #[cfg(feature = "tracing")]
                tracing::debug!("rollback");
                Err(e)
            }
        }
    }

    /// Re-open the in-memory `Catalog` handle from the pager.  Used
    /// after a transaction rollback to discard the catalog's in-
    /// memory `next_collection_id` / `tree.root` state that may
    /// have advanced during the rolled-back closure.
    fn refresh_catalog(&self) -> Result<()> {
        let mut pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let fresh = obj_core::Catalog::open_or_init(&mut pager)?;
        let mut existing = self.catalog.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        *existing = fresh;
        Ok(())
    }

    /// Run a closure inside a read transaction.  See
    /// [`Self::transaction`] for the closure shape and atomicity
    /// contract; reads inside `body` observe a consistent
    /// snapshot of the database.
    ///
    /// Every read inside the closure observes a single consistent
    /// snapshot — the snapshot is pinned at the moment the closure
    /// begins. Concurrent writers do not affect what the closure
    /// sees.
    ///
    /// # Examples
    ///
    /// Two reads inside one `read_transaction` see the same value
    /// even if a writer commits in between:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
    /// struct Order { total_cents: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("read.obj"))?;
    /// let id = db.insert(Order { total_cents: 10 })?;
    ///
    /// let (a, b) = db.read_transaction(|tx| {
    ///     let coll = tx.collection::<Order>()?;
    ///     let a = coll.get(id)?;
    ///     let b = coll.get(id)?;
    ///     Ok((a, b))
    /// })?;
    /// assert_eq!(a, b);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] if the reader lock could not be acquired.
    /// - Any error the closure returns.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "db.read_transaction", level = "info", skip_all)
    )]
    pub fn read_transaction<R, F>(&self, body: F) -> Result<R>
    where
        F: FnOnce(&ReadTxn<'_>) -> Result<R>,
    {
        #[cfg(feature = "tracing")]
        tracing::debug!("begin");
        let inner = obj_core::ReadTxn::begin_with_timeout(&self.env, self.busy_timeout)?;
        let attached_contexts = self.pin_attached_snapshots()?;
        let tx = ReadTxn::with_attached(inner, attached_contexts);
        let result = body(&tx);
        #[cfg(feature = "tracing")]
        tracing::debug!("commit");
        result
    }

    /// Snapshot every attached database into a per-namespace
    /// [`crate::txn::AttachedReadCtx`]. Each context owns its own
    /// pin; dropping the returned map releases every pin.
    fn pin_attached_snapshots(
        &self,
    ) -> Result<std::collections::HashMap<String, crate::txn::AttachedReadCtx>> {
        if self.attached_len.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return Ok(std::collections::HashMap::new());
        }
        let registry = self.attached.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let mut out: std::collections::HashMap<String, crate::txn::AttachedReadCtx> =
            std::collections::HashMap::with_capacity(registry.len());
        for (namespace, attached) in registry.iter() {
            let env = Arc::clone(&attached.env);
            let snapshot = {
                let mut pager = env.pager().lock().map_err(|_| Error::Busy {
                    kind: obj_core::LockKind::WriterInProcess,
                })?;
                pager.reader_snapshot()?
            };
            out.insert(
                namespace.clone(),
                crate::txn::AttachedReadCtx { env, snapshot },
            );
        }
        Ok(out)
    }

    /// Pin one [`crate::txn::AttachedReadCtx`] for the single attachment
    /// registered under `namespace`. Used by [`Self::dump_raw`]'s
    /// namespaced full-scan path: it needs exactly one attached env +
    /// pinned snapshot, not the whole per-namespace map
    /// [`Self::pin_attached_snapshots`] builds.
    ///
    /// The snapshot is pinned for the returned context's lifetime; the
    /// caller (the `DumpIter`) owns the context, so the pin holds for
    /// the iterator's whole life and a concurrent `detach` cannot
    /// invalidate the in-flight scan.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNamespaceUnknown`] if `namespace` is not
    ///   attached on this handle.
    /// - [`Error::Busy`] if the registry / pager mutex is poisoned.
    pub(crate) fn pin_attached_ctx(&self, namespace: &str) -> Result<crate::txn::AttachedReadCtx> {
        let env = {
            let registry = self.attached.lock().map_err(|_| Error::Busy {
                kind: obj_core::LockKind::WriterInProcess,
            })?;
            let attached =
                registry
                    .get(namespace)
                    .ok_or_else(|| Error::CollectionNamespaceUnknown {
                        namespace: namespace.to_owned(),
                    })?;
            Arc::clone(&attached.env)
        };
        let snapshot = {
            let mut pager = env.pager().lock().map_err(|_| Error::Busy {
                kind: obj_core::LockKind::WriterInProcess,
            })?;
            pager.reader_snapshot()?
        };
        Ok(crate::txn::AttachedReadCtx { env, snapshot })
    }

    /// Attach the database at `path` under `namespace`. The
    /// attached file is opened read-only; collections in the
    /// attached file become visible to subsequent
    /// `Db::collection::<T>()` calls (and the one-shot per-op API
    /// — `Db::get::<T>()`, etc.) when `T::COLLECTION` is of the
    /// form `<namespace>.<collection_name>`.
    ///
    /// Writes against namespaced collections return
    /// [`Error::AttachedDatabaseIsReadOnly`].
    ///
    /// Each attached database gets its own snapshot pinned at
    /// read-transaction begin; [`Db::detach`] removes the registry
    /// entry but in-flight reads complete against their pinned
    /// snapshot.
    ///
    /// # Choosing a receiver
    ///
    /// This `&mut self` form and the `&self`
    /// [`attach_shared`](Self::attach_shared) do the **same work** —
    /// the attachment registry is interior-mutable and mutex-guarded,
    /// so neither receiver is "more correct" at the storage layer. The
    /// `&mut self` here is about *exclusivity*, not necessity: prefer
    /// it when you own the `Db` outright, because an exclusive borrow
    /// is the clearest way to say "nothing else is touching this `Db`
    /// while I attach." Use [`attach_shared`](Self::attach_shared)
    /// when the handle is shared and `&mut self` is unavailable —
    /// most commonly an `Arc<Db>` cloned across threads. The same
    /// split applies to [`detach`](Self::detach) /
    /// [`detach_shared`](Self::detach_shared).
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "orders_attach_doc")]
    /// struct Order { total_cents: u64 }
    ///
    /// // Same struct shape, namespaced collection name. Reads
    /// // against this type route to the attached "archive" db.
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "archive.orders_attach_doc")]
    /// struct ArchivedOrder { total_cents: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let live = dir.path().join("live.obj");
    /// let archive = dir.path().join("archive.obj");
    ///
    /// // Seed the archive (writes go through its own un-namespaced type).
    /// {
    ///     let archive_db = Db::open(&archive)?;
    ///     let _ = archive_db.insert(Order { total_cents: 999 })?;
    /// }
    ///
    /// // Open the live db, attach the archive under "archive".
    /// let mut db = Db::open(&live)?;
    /// let _ = db.insert(Order { total_cents: 100 })?;
    /// db.attach(&archive, "archive")?;
    ///
    /// // One read transaction, two collections.
    /// db.read_transaction(|tx| {
    ///     let live = tx.collection::<Order>()?;
    ///     let arch = tx.collection::<ArchivedOrder>()?;
    ///     assert_eq!(live.all()?.len(), 1);
    ///     assert_eq!(arch.all()?.len(), 1);
    ///     Ok(())
    /// })?;
    ///
    /// db.detach("archive")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::AttachmentAlreadyExists`] if `namespace` is in
    ///   use on this `Db`.
    /// - [`Error::AttachmentNotReadable`] if `path` cannot be
    ///   opened read-only.
    pub fn attach<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        namespace: impl Into<String>,
    ) -> Result<()> {
        self.attach_inner(path.as_ref(), namespace.into())
    }

    /// Shared-reference (`&self`) form of [`Self::attach`]. Attaches
    /// the database at `path` under `namespace` through `&self`, for
    /// callers that hold a shared handle (e.g. an `Arc<Db>`) and
    /// cannot obtain `&mut self`.
    ///
    /// **This is the form you need when the `Db` is behind an `Arc`**
    /// (or any shared reference): an `Arc<Db>` hands out `&Db`, never
    /// `&mut Db`, so [`Self::attach`] is simply not callable. The
    /// receiver is the only difference between the two — see
    /// [`attach`](Self::attach)'s *Choosing a receiver* section for
    /// the full rationale.
    ///
    /// Behaviour is identical to [`Self::attach`]: the attachment
    /// registry is interior-mutable, guarded by the same per-`Db`
    /// mutex, so concurrent `attach_shared` / `detach_shared` calls
    /// from multiple threads serialise on that mutex. The duplicate-
    /// namespace re-check after the read-only open closes the race
    /// between two concurrent attaches of the same namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use std::sync::Arc;
    /// use obj::Db;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let live = dir.path().join("live.obj");
    /// let archive = dir.path().join("archive.obj");
    /// let _ = Db::open(&archive)?;
    ///
    /// // A shared handle cannot call `&mut self` `attach`, but can
    /// // call `attach_shared`.
    /// let db = Arc::new(Db::open(&live)?);
    /// db.attach_shared(&archive, "archive")?;
    /// db.detach_shared("archive")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::AttachmentAlreadyExists`] if `namespace` is in
    ///   use on this `Db`.
    /// - [`Error::AttachmentNotReadable`] if `path` cannot be
    ///   opened read-only.
    /// - [`Error::Busy`] if the registry mutex is poisoned.
    pub fn attach_shared<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        namespace: impl Into<String>,
    ) -> Result<()> {
        self.attach_inner(path.as_ref(), namespace.into())
    }

    /// `&self` core shared by [`Self::attach`] and
    /// [`Self::attach_shared`]. Both signatures are thin wrappers; the
    /// mutation flows through the interior-mutable `attached` registry
    /// regardless of the public method's receiver.
    fn attach_inner(&self, path: &std::path::Path, namespace: String) -> Result<()> {
        let path_buf = path.to_path_buf();
        {
            let registry = self.attached.lock().map_err(|_| Error::Busy {
                kind: obj_core::LockKind::WriterInProcess,
            })?;
            if registry.contains_key(&namespace) {
                return Err(Error::AttachmentAlreadyExists { namespace });
            }
        }
        let attached_db =
            Db::open_readonly(&path_buf).map_err(|source| Error::AttachmentNotReadable {
                path: path_buf.clone(),
                source: Box::new(source),
            })?;
        let env = Arc::clone(&attached_db.env);
        let mut registry = self.attached.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        if registry.contains_key(&namespace) {
            return Err(Error::AttachmentAlreadyExists { namespace });
        }
        registry.insert(
            namespace,
            AttachedDb {
                env,
                _db: attached_db,
            },
        );
        self.attached_len
            .store(registry.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Remove the attachment registered under `namespace`. Returns
    /// [`Error::CollectionNamespaceUnknown`] if the namespace is
    /// not attached.
    ///
    /// In-flight read transactions hold their own snapshot pins on
    /// the attached env; detach removes the registry entry, but the
    /// in-flight read may still complete against its pinned
    /// snapshot.
    ///
    /// This `&mut self` form mirrors [`Self::attach`]; when you hold
    /// only a shared handle (e.g. an `Arc<Db>`), use the `&self`
    /// [`detach_shared`](Self::detach_shared) instead. See
    /// [`attach`](Self::attach)'s *Choosing a receiver* section.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNamespaceUnknown`] if `namespace` is
    ///   not attached.
    /// - [`Error::Busy`] if the registry mutex is poisoned.
    pub fn detach(&mut self, namespace: &str) -> Result<()> {
        self.detach_inner(namespace)
    }

    /// Shared-reference (`&self`) form of [`Self::detach`]. Removes the
    /// attachment registered under `namespace` through `&self`, for
    /// callers that hold a shared handle (e.g. an `Arc<Db>`).
    ///
    /// Behaviour is identical to [`Self::detach`]. In-flight read
    /// transactions hold their own snapshot pins on the attached env;
    /// `detach_shared` removes the registry entry, but the in-flight
    /// read may still complete against its pinned snapshot. Concurrent
    /// `attach_shared` / `detach_shared` calls serialise on the same
    /// per-`Db` registry mutex.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNamespaceUnknown`] if `namespace` is
    ///   not attached.
    /// - [`Error::Busy`] if the registry mutex is poisoned.
    pub fn detach_shared(&self, namespace: &str) -> Result<()> {
        self.detach_inner(namespace)
    }

    /// `&self` core shared by [`Self::detach`] and
    /// [`Self::detach_shared`].
    fn detach_inner(&self, namespace: &str) -> Result<()> {
        let mut registry = self.attached.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        if registry.remove(namespace).is_none() {
            return Err(Error::CollectionNamespaceUnknown {
                namespace: namespace.to_owned(),
            });
        }
        self.attached_len
            .store(registry.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Insert `doc` into its collection.  One-shot transaction;
    /// returns the assigned [`Id`].
    ///
    /// The one-shot API opens, commits, and closes a private
    /// transaction per call. Reach for [`Db::transaction`] when
    /// several mutations must commit or roll back as a single
    /// atomic unit.
    ///
    /// # Examples
    ///
    /// One-shot CRUD against a `Document`-derived type:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// struct Order {
    ///     customer_id: u64,
    ///     total_cents: u64,
    ///     status: String,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("oneshot.obj"))?;
    ///
    /// // insert returns the freshly-allocated Id.
    /// let id = db.insert(Order {
    ///     customer_id: 1,
    ///     total_cents: 100,
    ///     status: "pending".to_owned(),
    /// })?;
    ///
    /// // get returns Option<T>.
    /// let _maybe: Option<Order> = db.get::<Order>(id)?;
    ///
    /// // update applies a closure in place. Annotate the closure
    /// // parameter (`|o: &mut Order|`) so `T` is inferred — no turbofish.
    /// db.update(id, |o: &mut Order| {
    ///     o.status = "shipped".to_owned();
    /// })?;
    ///
    /// // upsert at a caller-supplied id (insert or replace).
    /// // `Id::new` is the literal constructor; use `Id::try_new` for
    /// // ids derived from runtime input.
    /// let id2 = obj::Id::new(42);
    /// db.upsert::<Order>(id2, Order {
    ///     customer_id: 2,
    ///     total_cents: 999,
    ///     status: "new".to_owned(),
    /// })?;
    ///
    /// // delete returns true if the row existed.
    /// let existed = db.delete::<Order>(id)?;
    /// assert!(existed);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::transaction`] plus any error from
    /// [`crate::Collection::insert`].
    pub fn insert<T: Document + obj_core::codec::Schema>(&self, doc: T) -> Result<Id> {
        self.transaction(|tx| tx.collection::<T>()?.insert(doc))
    }

    /// Fetch the document at `id`.  Returns `Ok(None)` if absent.
    ///
    /// # Errors
    ///
    /// As [`Self::read_transaction`] plus any error from
    /// [`crate::Collection::get`].
    pub fn get<T: Document>(&self, id: Id) -> Result<Option<T>> {
        self.read_transaction(|tx| crate::collection::fused_point_get::<T>(tx, id))
    }

    /// Update the document at `id` via the closure.
    ///
    /// # Choosing a call form
    ///
    /// `T` appears only behind the closure's `&mut T` parameter, so
    /// the compiler has nothing in the value arguments to infer it
    /// from. The low-friction form is to **annotate the closure
    /// parameter** — `db.update(id, |o: &mut Order| …)` — and let
    /// inference flow from there. Prefer this. The turbofish
    /// `db.update::<Order, _>(id, …)` works too but is noisier: the
    /// trailing `_` is the closure type you would rather not spell.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// struct Order { total_cents: u64, status: String }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("update.obj"))?;
    /// let id = db.insert(Order { total_cents: 100, status: "pending".to_owned() })?;
    ///
    /// // Recommended: annotate the closure parameter; no turbofish.
    /// db.update(id, |o: &mut Order| o.status = "shipped".to_owned())?;
    ///
    /// let after: Order = db
    ///     .get::<Order>(id)?
    ///     .ok_or(obj::Error::InvalidArgument("just updated"))?;
    /// assert_eq!(after.status, "shipped");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::DocumentNotFound`] if `id` does not exist.
    /// - As [`Self::transaction`].
    pub fn update<T, F>(&self, id: Id, f: F) -> Result<()>
    where
        T: Document + obj_core::codec::Schema,
        F: FnOnce(&mut T),
    {
        self.transaction(|tx| tx.collection::<T>()?.update(id, f))
    }

    /// Delete the document at `id`.  Returns `true` if it existed.
    ///
    /// # Errors
    ///
    /// As [`Self::transaction`].
    pub fn delete<T: Document>(&self, id: Id) -> Result<bool> {
        self.transaction(|tx| tx.collection::<T>()?.delete(id))
    }

    /// Insert or replace the document at `id`.
    ///
    /// # Errors
    ///
    /// As [`Self::transaction`].
    pub fn upsert<T: Document + obj_core::codec::Schema>(&self, id: Id, doc: T) -> Result<()> {
        self.transaction(|tx| tx.collection::<T>()?.upsert(id, doc))
    }

    /// Convenience wrapper around [`crate::Collection::find_unique`]
    /// — `db.find_unique::<Customer>("by_email", "ada@example.com")`.
    /// Runs inside a one-shot read transaction.
    ///
    /// `O(log n)`, no collection scan; the lookup walks the named
    /// index's B-tree directly. Defined only on `Unique` indexes —
    /// for the other kinds use
    /// [`Collection::lookup`](crate::Collection::lookup) or
    /// [`Collection::index_range`](crate::Collection::index_range).
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "customers_find_unique_doc")]
    /// struct Customer {
    ///     #[obj(index = unique)]
    ///     email: String,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("find-unique.obj"))?;
    /// let _ = db.insert(Customer { email: "ada@example.com".to_owned() })?;
    /// let by_email: Option<Customer> = db
    ///     .find_unique::<Customer>("email", "ada@example.com")?;
    /// assert!(by_email.is_some());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::find_unique`].
    pub fn find_unique<T: Document>(
        &self,
        index_name: &str,
        key: impl Into<obj_core::codec::Dynamic>,
    ) -> Result<Option<T>> {
        self.read_transaction(|tx| tx.collection::<T>()?.find_unique(index_name, key))
    }

    /// Construct a fresh [`crate::Query`] builder rooted at this
    /// database. The builder borrows `&self` for the build phase;
    /// the borrow ends when [`crate::Query::fetch`] returns.
    ///
    /// Compose with [`Query::filter`](crate::Query::filter),
    /// [`Query::limit`](crate::Query::limit),
    /// [`Query::sort_by`](crate::Query::sort_by),
    /// [`Query::index_range`](crate::Query::index_range). Terminate
    /// with [`Query::fetch`](crate::Query::fetch) (for the
    /// documents) or [`Query::count`](crate::Query::count) (for the
    /// count alone).
    ///
    /// # Examples
    ///
    /// Top-N matching documents by an indexed field, ascending:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use obj_core::codec::Dynamic;
    /// use serde::{Deserialize, Serialize};
    ///
    /// // `#[obj(schema)]` derives `Schema` for the enum so it can
    /// // nest in `Order`'s auto-emitted `Schema` impl.
    /// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
    /// #[obj(schema)]
    /// enum OrderStatus { Pending, Shipped }
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "orders_query_doc")]
    /// struct Order {
    ///     #[obj(index)]
    ///     customer_id: u64,
    ///     status: OrderStatus,
    ///     #[obj(index)]
    ///     placed_at: u64,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("queries.obj"))?;
    /// for i in 0..20u64 {
    ///     let _ = db.insert(Order {
    ///         customer_id: i % 3,
    ///         status: if i % 2 == 0 { OrderStatus::Pending } else { OrderStatus::Shipped },
    ///         placed_at: i * 1_000,
    ///     })?;
    /// }
    ///
    /// let pending: Vec<Order> = db
    ///     .query::<Order>()
    ///     .filter(|o| o.status == OrderStatus::Pending)
    ///     .sort_by(|o| Dynamic::U64(o.placed_at))
    ///     .limit(5)
    ///     .fetch()?;
    /// assert!(pending.len() <= 5);
    /// assert!(pending.iter().all(|o| o.status == OrderStatus::Pending));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn query<T: Document + Send + 'static>(&self) -> crate::Query<'_, T> {
        crate::Query::new(self)
    }

    /// Open a read-only typed handle to the collection registered
    /// under the runtime `name`, instead of the type's compile-time
    /// `T::COLLECTION`.
    ///
    /// This unlocks the portability use-case for
    /// attached databases: by passing a namespaced name like
    /// `"archive.orders"`, the returned [`crate::Collection`] reads
    /// from the database attached under the `"archive"` namespace
    /// — see [`Db::attach`].
    ///
    /// Construction is **infallible**: errors (missing collection,
    /// unknown namespace, busy lock) surface at the first method
    /// call on the handle, not at the call to `collection(name)`.
    /// Each read-only method on the returned handle opens a private
    /// [`Db::read_transaction`] and dispatches against the
    /// runtime-named collection's catalog row.
    ///
    /// # Read-only
    ///
    /// The returned handle rejects every mutating call —
    /// [`crate::Collection::insert`], `update`, `delete`, `upsert`
    /// all return [`Error::ReadOnly`]. To write into a non-default
    /// collection, override [`obj_core::Document::COLLECTION`] on
    /// the type itself (compile-time-bound) and use the regular
    /// [`Db::transaction`] / [`crate::WriteTxn::collection`] path.
    /// The runtime accessor is intentionally limited to reads — the
    /// write-through-runtime-name path requires engine plumbing that
    /// is deferred.
    ///
    /// # Example
    ///
    /// ```
    /// use obj::{Db, Document, DynamicSchema, Schema};
    /// use serde::{Deserialize, Serialize};
    /// use tempfile::tempdir;
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize)]
    /// struct Order { customer_id: u64, total_cents: u64 }
    ///
    /// impl Document for Order {
    ///     const COLLECTION: &'static str = "orders";
    ///     const VERSION: u32 = 1;
    /// }
    ///
    /// // the insert path persists the current-version schema.
    /// impl Schema for Order {
    ///     fn schema() -> DynamicSchema {
    ///         DynamicSchema::map([
    ///             ("customer_id", DynamicSchema::U64),
    ///             ("total_cents", DynamicSchema::U64),
    ///         ])
    ///     }
    /// }
    ///
    /// fn run() -> obj::Result<()> {
    ///     let dir = tempdir()?;
    ///     let archive_path = dir.path().join("archive.obj");
    ///     // Populate the archive database first.
    ///     {
    ///         let archive_db = Db::open(&archive_path)?;
    ///         archive_db.insert(Order { customer_id: 1, total_cents: 999 })?;
    ///     }
    ///     // Attach it under a namespace and read via the runtime
    ///     // accessor — no need to re-declare `Order` with a
    ///     // namespaced `COLLECTION`.
    ///     let main_path = dir.path().join("main.obj");
    ///     let mut db = Db::open(&main_path)?;
    ///     db.attach(&archive_path, "archive")?;
    ///     let archived: Vec<Order> = db
    ///         .collection::<Order>("archive.orders")
    ///         .all()?
    ///         .into_iter()
    ///         .map(|(_id, doc)| doc)
    ///         .collect();
    ///     assert_eq!(archived.len(), 1);
    ///     Ok(())
    /// }
    /// # run().unwrap();
    /// ```
    #[must_use]
    pub fn collection<T: Document + Send + 'static>(
        &self,
        name: impl Into<String>,
    ) -> crate::Collection<'_, T> {
        crate::Collection::<T>::lazy(self, name.into())
    }

    /// Convenience shim —
    /// `for order in db.all::<Order>()? { ... }`. Returns an owned
    /// `Vec<T>` (materialised). One-line shim over [`Db::iter_all`]
    /// that drives the streaming iterator to exhaustion and
    /// collects; if the collection is large enough that peak
    /// memory matters, prefer [`Db::iter_all`] directly.
    ///
    /// Sorting requires materialisation — the comparator needs
    /// every key up front. Use [`Db::query`] +
    /// [`Query::sort_by`](crate::Query::sort_by) for the
    /// top-N-sorted workload; the iterator side has no streaming
    /// sorted shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// struct Order { total_cents: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("all.obj"))?;
    /// for i in 0..5u64 {
    ///     let _ = db.insert(Order { total_cents: i * 10 })?;
    /// }
    /// let listed: Vec<Order> = db.all::<Order>()?;
    /// assert_eq!(listed.len(), 5);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Db::iter_all`].
    pub fn all<T: Document + Send + 'static>(&self) -> Result<Vec<T>> {
        self.iter_all::<T>()?
            .map(|step| step.map(|(_id, doc)| doc))
            .collect()
    }

    /// Write a self-contained `.obj` file at `dest` carrying this
    /// database's state at the LSN of an internally-taken reader
    /// snapshot.
    ///
    /// Hot backup — writers continue uninterrupted against the
    /// source. Post-snapshot writes are NOT in the destination.
    ///
    /// Algorithm:
    ///
    /// 1. Take a `ReaderSnapshot` against the source pager (pins
    ///    `pinned_lsn`).
    /// 2. `OpenOptions::create_new(true)` on `dest`.
    /// 3. Copy main-file pages `0..page_count` to `dest`.
    /// 4. Overlay every frame in the snapshot's frozen WAL view
    ///    onto `dest` at the frame's page-id offset.
    /// 5. If the snapshot carries a WAL-staged page-0 header,
    ///    overlay it (so `dest`'s page-0 reflects the catalog
    ///    root / freelist head / page count the snapshot would
    ///    have observed).
    /// 6. Patch `dest`'s page-0 header: zero `wal_salt`, recompute
    ///    the header CRC32C.
    /// 7. `sync_data(SyncMode::Full)` on `dest`.
    /// 8. Drop the snapshot (releases the WAL pin).
    ///
    /// On any mid-backup error the destination file is removed
    /// best-effort so a half-written backup does not linger.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "notes_backup_doc")]
    /// struct Note { body: String }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let src = dir.path().join("src.obj");
    /// let dst = dir.path().join("backup.obj");
    ///
    /// let db = Db::open(&src)?;
    /// let _ = db.insert(Note { body: "before backup".to_owned() })?;
    /// db.backup_to(&dst)?;
    ///
    /// // The backup is itself a fully-formed obj file. Open it and read.
    /// let backup = Db::open(&dst)?;
    /// let listed: Vec<Note> = backup.all::<Note>()?;
    /// assert_eq!(listed.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::BackupDestinationExists`] if `dest` already
    ///   exists.
    /// - [`Error::BackupNotSupportedForMemoryPager`] when called on
    ///   a `Db` constructed via [`Db::memory`] / [`Db::memory_with`].
    /// - [`Error::BackupNotSupportedForEncryptedPager`] when called
    ///   on a `Db` opened with an encryption key. The hot-backup
    ///   copy path is plaintext-only; re-encrypting each page for the
    ///   destination is deferred to a future minor.
    /// - [`Error::Io`] on syscall failure during the copy.
    pub fn backup_to<P: AsRef<std::path::Path>>(&self, dest: P) -> Result<()> {
        let guard = obj_core::WriteTxn::begin(&self.env, self.busy_timeout)?;
        let result = self.run_backup_under_guard(dest);
        let unlock = guard.rollback();
        result.and(unlock)
    }

    /// Take the pager `Mutex` (innermost lock), pin a reader
    /// snapshot, and run the backup copy. Factored out of
    /// [`Self::backup_to`] so the cross-process lock guard's lifetime
    /// is unambiguous.
    fn run_backup_under_guard<P: AsRef<std::path::Path>>(&self, dest: P) -> Result<()> {
        let mut pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let snapshot = pager.reader_snapshot()?;
        obj_core::backup::backup_pager_to_path(&pager, &snapshot, dest)?;
        drop(snapshot);
        drop(pager);
        Ok(())
    }

    /// Fold committed WAL pages into the main `.obj` file and reset
    /// the WAL back to its 64-byte header.
    ///
    /// Wraps [`obj_core::pager::Pager::checkpoint`]. Acquires the
    /// pager lock the same way [`Self::backup_to`] does, then calls
    /// the pager checkpoint. After it returns, the committed records
    /// live in the main file rather than the `-wal` sidecar, and the
    /// WAL is truncated to header-only.
    ///
    /// # Deferred / no-op behavior
    ///
    /// - If a live MVCC reader has pinned an LSN below the end of the
    ///   WAL, the checkpoint is **deferred** (a safe no-op) so the
    ///   reader's frames are not reclaimed out from under it.
    /// - If there is nothing to fold (no committed WAL frames), the
    ///   call is a harmless no-op.
    ///
    /// This is the reusable engine entry point for an explicit
    /// checkpoint and a future checkpoint-on-close path.
    ///
    /// # Errors
    ///
    /// - [`Error::ReadOnly`] if the database was opened read-only.
    /// - [`Error::Busy`] if the pager lock is poisoned.
    /// - Any [`Error`] from [`obj_core::pager::Pager::checkpoint`]
    ///   (e.g. [`Error::Io`] on syscall failure).
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "db.checkpoint", level = "info", skip_all)
    )]
    pub fn checkpoint(&self) -> Result<()> {
        if self.readonly {
            return Err(Error::ReadOnly {
                operation: "checkpoint",
            });
        }
        let guard = obj_core::WriteTxn::begin(&self.env, self.busy_timeout)?;
        let result = self.run_checkpoint_under_guard();
        let unlock = guard.rollback();
        result.and(unlock)
    }

    /// Take the pager `Mutex` (innermost lock) and run the
    /// pager checkpoint. Factored out of [`Self::checkpoint`] so the
    /// cross-process lock guard's lifetime is unambiguous.
    fn run_checkpoint_under_guard(&self) -> Result<()> {
        let mut pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        pager.checkpoint()?;
        drop(pager);
        Ok(())
    }

    /// Streaming iterator over every `(Id, T)` pair in the
    /// collection. The returned [`IterAll`] holds a read transaction
    /// — and therefore a pinned reader snapshot — for its entire
    /// lifetime; the borrow on `self` ends when the iterator is
    /// dropped.
    ///
    /// # Why each item is a `Result`
    ///
    /// Iteration is *fallible per step*. The iterator decodes
    /// documents lazily, one bounded batch at a time, so a
    /// mid-iteration page read, B-tree descent, or codec decode can
    /// fail long after construction already succeeded. Each `next`
    /// therefore yields `Result<(Id, T)>` — a failure surfaces as
    /// `Some(Err(_))` rather than ending the iteration, so the
    /// caller decides whether to propagate or continue. The
    /// canonical loop unwraps the per-step `Result` with `?` (which
    /// is why the loop body, not the `for` binding, carries the
    /// `(id, doc)` destructure):
    ///
    /// ```ignore
    /// for step in db.iter_all::<Order>()? {
    ///     let (id, doc) = step?; // propagate a mid-scan IO error
    ///     // ... use `id` / `doc`
    /// }
    /// ```
    ///
    /// Contrast [`Db::all`], which is *eagerly* fallible: it drives
    /// this iterator to exhaustion behind a single `?` and hands
    /// back a plain `Vec<T>` with no per-element `Result` to unwrap.
    /// Choose deliberately — prefer `all` when the collection fits
    /// comfortably in memory and a one-shot fallible call reads
    /// cleaner; prefer `iter_all` when peak memory must stay bounded
    /// and you can handle (or `?`-propagate) a failure mid-scan.
    ///
    /// Peak memory does NOT scale with collection size — the
    /// iterator's internal buffer is fixed at
    /// `ITER_ALL_BATCH = 256` entries (~128 KiB at the
    /// ~512 byte/doc estimate).
    ///
    /// `Query::fetch` is sort-compatible (sort requires
    /// materialisation); `iter_all` is NOT — there is no streaming
    /// shape for sort because the comparator needs every key
    /// up front. Use `Query::sort_by` + `fetch` for the sorted
    /// workload; use `iter_all` for the unsorted large-scan
    /// workload.
    ///
    /// # Examples
    ///
    /// Streaming a small collection and folding into a running sum.
    /// The iterator's peak memory stays bounded regardless of how
    /// many documents the collection holds:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// struct Order { total_cents: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("iter.obj"))?;
    /// for i in 0..5u64 {
    ///     let _ = db.insert(Order { total_cents: i * 10 })?;
    /// }
    ///
    /// let mut total: u64 = 0;
    /// for step in db.iter_all::<Order>()? {
    ///     let (_id, doc) = step?;
    ///     total = total
    ///         .checked_add(doc.total_cents)
    ///         .ok_or(obj::Error::InvalidArgument("overflow"))?;
    /// }
    /// assert_eq!(total, 0 + 10 + 20 + 30 + 40);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - As [`Db::read_transaction`] (construction-time).
    /// - [`Error::CollectionNotFound`] if the collection is not yet
    ///   registered at the snapshot's pinned LSN.
    /// - Per-step iteration may yield `Some(Err(_))` for pager,
    ///   B-tree, or codec failures.
    pub fn iter_all<T: Document + Send + 'static>(&self) -> Result<IterAll<'_, T>> {
        let inner = obj_core::ReadTxn::begin_with_timeout(&self.env, self.busy_timeout)?;
        let txn = ReadTxn::new(inner);
        let descriptor = {
            let coll = txn.collection::<T>()?;
            coll.descriptor().clone()
        };
        Ok(IterAll {
            txn,
            descriptor,
            buffer: VecDeque::new(),
            last_emitted_key: None,
            finished: false,
            _phantom: PhantomData,
        })
    }
}

/// Streaming iterator returned by [`Db::iter_all`].
///
/// Holds a [`ReadTxn`] for its lifetime so every yielded document
/// is consistent with the snapshot pinned at construction. Yields
/// `Result<(Id, T)>` one entry at a time; refills its internal
/// buffer in fixed-size chunks (`ITER_ALL_BATCH = 256` entries) per
/// pager-lock acquisition, so peak memory stays bounded at a small
/// constant regardless of the collection's size.
///
/// Construction errors surface at the [`Db::iter_all`] call site;
/// per-step errors (pager, B-tree, codec) surface as
/// `Some(Err(_))` during iteration and do NOT terminate it — the
/// caller decides whether to continue. Unwrap each step with `?`:
/// `for step in iter { let (id, doc) = step?; }`. For the eager,
/// single-`?` alternative that materialises a `Vec<T>` instead, see
/// [`Db::all`].
pub struct IterAll<'db, T> {
    /// Owns the snapshot pin for the iterator's lifetime.
    txn: ReadTxn<'db>,
    /// Cached primary-tree root + collection id (`Document::decode`
    /// needs the collection id to validate the per-doc header).
    descriptor: CollectionDescriptor,
    /// Pre-decoded buffer of upcoming entries.
    buffer: VecDeque<Result<(Id, T)>>,
    /// Resumption marker — `Excluded(last_emitted_key)` is the
    /// start bound of the next refill.
    last_emitted_key: Option<Vec<u8>>,
    /// `true` once the underlying B-tree iterator returned no more
    /// entries; subsequent `next` calls return `None`.
    finished: bool,
    /// Track the `T` parameter without owning a value.
    _phantom: PhantomData<fn() -> T>,
}

impl<T> std::fmt::Debug for IterAll<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterAll")
            .field("collection_id", &self.descriptor.collection_id)
            .field("buffer_len", &self.buffer.len())
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<T: Document + Send + 'static> IterAll<'_, T> {
    /// Refill the internal buffer with up to [`ITER_ALL_BATCH`]
    /// entries, resuming from `last_emitted_key`. Stores per-doc
    /// decode errors as `Err` in the buffer so the caller can
    /// observe them via `next()` without aborting iteration.
    ///
    /// Every fallible operation is either captured into the buffer
    /// or returned up-front via `?` (which propagates the
    /// lock-acquisition / B-tree-open failures before the buffer is
    /// touched).
    fn refill(&mut self) -> Result<()> {
        let pager_arc: Arc<Mutex<Pager<FileHandle>>> = Arc::clone(self.txn.inner.env().pager());
        let mut pager = pager_arc.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let root_pid = PageId::new(self.descriptor.primary_root)
            .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
        let tree = BTree::<FileHandle>::open(&pager, root_pid)?;
        let start = match &self.last_emitted_key {
            Some(k) => Bound::Excluded(k.clone()),
            None => Bound::Unbounded,
        };
        let collection_id = self.descriptor.collection_id;
        let iter = tree.range(&mut pager, (start, Bound::Unbounded))?;
        let mut yielded: usize = 0;
        let mut last_key: Option<Vec<u8>> = None;
        let mut raw: Vec<RawRow> = Vec::with_capacity(ITER_ALL_BATCH);
        for step in iter {
            if yielded >= ITER_ALL_BATCH {
                break;
            }
            yielded = yielded
                .checked_add(1)
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "iter_all batch counter overflow",
                })?;
            stage_raw_row(&mut raw, &mut last_key, step);
        }
        if yielded < ITER_ALL_BATCH {
            self.finished = true;
        }
        let snapshot = self.txn.inner.snapshot();
        let mut batch: VecDeque<Result<(Id, T)>> = VecDeque::with_capacity(raw.len());
        for row in raw {
            batch.push_back(decode_raw_row::<T>(&pager, snapshot, collection_id, row));
        }
        drop(pager);
        self.buffer.extend(batch);
        if let Some(k) = last_key {
            self.last_emitted_key = Some(k);
        }
        Ok(())
    }
}

/// One raw primary-tree row staged by [`IterAll::refill`] before
/// decode: either a parsed `(Id, payload-bytes)` pair, or a captured
/// error (corruption / a key that is not an `Id`) to be re-emitted
/// verbatim through `next()`.
type RawRow = Result<(Id, Vec<u8>)>;

/// Stage one B-tree iterator step as a [`RawRow`] WITHOUT decoding.
/// Errors and a non-`Id` key are captured as `Err` so they surface via
/// `next()` rather than aborting the refill.
fn stage_raw_row(
    raw: &mut Vec<RawRow>,
    last_key: &mut Option<Vec<u8>>,
    step: Result<(Vec<u8>, Vec<u8>)>,
) {
    let (key, value) = match step {
        Ok(kv) => kv,
        Err(e) => {
            raw.push(Err(e));
            return;
        }
    };
    let Some(id) = Id::from_be_bytes(&key) else {
        raw.push(Err(Error::InvalidArgument(
            "primary B-tree key is not an Id",
        )));
        return;
    };
    *last_key = Some(key);
    raw.push(Ok((id, value)));
}

/// Decode one staged [`RawRow`] into `Result<(Id, T)>`, sourcing any
/// stored-version schema from the iterator's pinned snapshot via
/// [`SchemaSource::Snapshot`]. A captured error is passed
/// through unchanged; a decode error becomes the `Err` entry the caller
/// observes via `next()`.
fn decode_raw_row<T: Document>(
    pager: &Pager<FileHandle>,
    snapshot: &obj_core::ReaderSnapshot<FileHandle>,
    collection_id: u32,
    row: RawRow,
) -> Result<(Id, T)> {
    let (id, value) = row?;
    let doc = obj_core::codec::decode_with::<T, FileHandle>(
        &value,
        collection_id,
        obj_core::codec::SchemaSource::Snapshot {
            pager,
            snapshot,
            collection_id,
        },
    )?;
    Ok((id, doc))
}

impl<T: Document + Send + 'static> Iterator for IterAll<'_, T> {
    type Item = Result<(Id, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.buffer.pop_front() {
            return Some(item);
        }
        if self.finished {
            return None;
        }
        if let Err(e) = self.refill() {
            self.finished = true;
            return Some(Err(e));
        }
        self.buffer.pop_front()
    }
}

/// Split a possibly-namespaced collection name into its
/// `(Some("ns"), "name")` parts. The split is on the FIRST `.`
/// only; downstream collection names may contain further dots.
///
/// `"users"` → `(None, "users")`.
/// `"archive.orders"` → `(Some("archive"), "orders")`.
/// `"archive.orders.legacy"` → `(Some("archive"), "orders.legacy")`.
#[must_use]
pub(crate) fn split_namespace(name: &str) -> (Option<&str>, &str) {
    match name.find('.') {
        Some(idx) => (Some(&name[..idx]), &name[idx + 1..]),
        None => (None, name),
    }
}

/// `Db` is `Send + Sync` so it composes with `Arc<Db>` for
/// concurrent reader / single-writer workloads.  The thread-safety
/// is inherited from the underlying `Arc<TxnEnv>` + `Arc<Mutex<Catalog>>`.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Db>;
};

/// Translate the first failure in `report` (if any) into the
/// strongest `Error` we can synthesise. Used by [`Db::from_parts`]'s
/// open-time fast check so the caller sees `Err(Error::Corruption {
/// page_id })` rather than an opaque `IntegrityReport`. The page-id
/// in the returned error is the locus of the failure when one is
/// available; for failures whose locus is a non-page (e.g. an
/// `OrphanIndexEntry`, which `quick_check` does NOT emit), the
/// catalog root page-id is used as a stand-in.
fn first_failure_as_error(report: &obj_core::IntegrityReport) -> Option<Error> {
    let first = report.failures.first()?;
    let err = match first {
        obj_core::IntegrityFailure::ChecksumMismatch { page_id }
        | obj_core::IntegrityFailure::OrphanPage { page_id }
        | obj_core::IntegrityFailure::BTreeSortViolation { page_id }
        | obj_core::IntegrityFailure::FreelistChainBroken { page_id }
        | obj_core::IntegrityFailure::BTreeSiblingChainBroken { page_id, .. }
        | obj_core::IntegrityFailure::BTreeLevelInvariantViolated { page_id, .. }
        | obj_core::IntegrityFailure::DanglingCatalogPointer { page_id, .. } => {
            Error::Corruption { page_id: *page_id }
        }
        obj_core::IntegrityFailure::BTreeDepthExceeded { limit, .. } => {
            Error::BTreeDepthExceeded { limit: *limit }
        }
        _ => Error::Corruption { page_id: 0 },
    };
    Some(err)
}
