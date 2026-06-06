//! `AsyncDb` — async-facing wrapper over the blocking [`Db`].
//!
//! Each method clones the inner `Arc<Db>` and moves the clone into a
//! [`blocking::unblock`] task; the task runs the corresponding
//! synchronous method to completion. The blocking-task return value is
//! `Send + 'static`, which the existing blocking surface already
//! satisfies (every `Db` method returns `Result<T>` with `T: Send +
//! 'static` for every type we wrap here).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use obj_core::Document;
use obj_core::{Id, Result};

use crate::asynchronous::collection::AsyncCollection;
use crate::asynchronous::query::AsyncQuery;
use crate::{Config, Db, DbStat, IntegrityReport, ReadTxn, WriteTxn};

/// Async-facing wrapper around the blocking [`Db`].
///
/// `AsyncDb` is cheap to clone (one `Arc` bump) so it can be shared
/// across spawned tasks without locking. The blocking engine sits
/// behind a single `Arc<Db>`; every async method hands its body off
/// to the [`blocking`] thread pool.
///
/// Public construction goes through [`AsyncDb::open`] /
/// [`AsyncDb::open_with`] / [`AsyncDb::memory`] /
/// [`AsyncDb::memory_with`] / [`AsyncDb::open_readonly`] /
/// [`AsyncDb::from_blocking`].
#[derive(Clone, Debug)]
pub struct AsyncDb {
    inner: Arc<Db>,
}

impl AsyncDb {
    /// Construct an `AsyncDb` from an already-opened blocking [`Db`].
    ///
    /// Synchronous on purpose — wrapping an in-hand `Db` does no
    /// I/O. Useful when the caller already opened the database from
    /// a blocking context and wants to drive it
    /// async-style from the rest of the program.
    #[must_use]
    pub fn from_blocking(db: Db) -> Self {
        Self {
            inner: Arc::new(db),
        }
    }

    /// Open or create a file-backed database at `path` with default
    /// configuration. See [`Db::open`].
    ///
    /// # Errors
    ///
    /// As [`Db::open`].
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with(path, Config::default()).await
    }

    /// Open or create a file-backed database with `config`. See
    /// [`Db::open_with`].
    ///
    /// # Errors
    ///
    /// As [`Db::open_with`].
    pub async fn open_with<P: AsRef<Path>>(path: P, config: Config) -> Result<Self> {
        let path_buf: PathBuf = path.as_ref().to_path_buf();
        unblock(move || Db::open_with(path_buf, config).map(Self::from_blocking)).await
    }

    /// Open a fresh in-memory database. See [`Db::memory`].
    ///
    /// # Errors
    ///
    /// As [`Db::memory`].
    pub async fn memory() -> Result<Self> {
        Self::memory_with(Config::default()).await
    }

    /// As [`AsyncDb::memory`] with a caller-supplied [`Config`].
    ///
    /// # Errors
    ///
    /// As [`Db::memory_with`].
    pub async fn memory_with(config: Config) -> Result<Self> {
        unblock(move || Db::memory_with(config).map(Self::from_blocking)).await
    }

    /// Open the database at `path` read-only. See [`Db::open_readonly`].
    ///
    /// # Errors
    ///
    /// As [`Db::open_readonly`].
    pub async fn open_readonly<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf: PathBuf = path.as_ref().to_path_buf();
        unblock(move || Db::open_readonly(path_buf).map(Self::from_blocking)).await
    }

    /// Borrow the underlying blocking [`Db`].
    ///
    /// Useful when async code needs to hand a `&Db` to a third-party
    /// API that only takes the blocking type — note the borrow is
    /// scoped to the local task, not the blocking pool.
    #[must_use]
    pub fn as_blocking(&self) -> &Db {
        &self.inner
    }

    /// Insert `doc`. See [`Db::insert`].
    ///
    /// # Errors
    ///
    /// As [`Db::insert`].
    pub async fn insert<T>(&self, doc: T) -> Result<Id>
    where
        T: Document + obj_core::codec::Schema + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.insert(doc)).await
    }

    /// Fetch the document at `id`. See [`Db::get`].
    ///
    /// # Errors
    ///
    /// As [`Db::get`].
    pub async fn get<T>(&self, id: Id) -> Result<Option<T>>
    where
        T: Document + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.get::<T>(id)).await
    }

    /// Update the document at `id`. See [`Db::update`].
    ///
    /// The closure runs **synchronously** inside the blocking task,
    /// so it must be `Send + 'static`. This mirrors the wider
    /// "closure runs on the blocking pool" contract documented on
    /// [`AsyncDb::transaction`].
    ///
    /// # Errors
    ///
    /// As [`Db::update`].
    pub async fn update<T, F>(&self, id: Id, f: F) -> Result<()>
    where
        T: Document + obj_core::codec::Schema + Send + 'static,
        F: FnOnce(&mut T) + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.update::<T, F>(id, f)).await
    }

    /// Delete the document at `id`. See [`Db::delete`].
    ///
    /// # Errors
    ///
    /// As [`Db::delete`].
    pub async fn delete<T>(&self, id: Id) -> Result<bool>
    where
        T: Document + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.delete::<T>(id)).await
    }

    /// Insert-or-replace the document at `id`. See [`Db::upsert`].
    ///
    /// # Errors
    ///
    /// As [`Db::upsert`].
    pub async fn upsert<T>(&self, id: Id, doc: T) -> Result<()>
    where
        T: Document + obj_core::codec::Schema + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.upsert(id, doc)).await
    }

    /// Point lookup on a `Unique` index. See [`Db::find_unique`].
    ///
    /// # Errors
    ///
    /// As [`Db::find_unique`].
    pub async fn find_unique<T, K>(&self, index_name: &str, key: K) -> Result<Option<T>>
    where
        T: Document + Send + 'static,
        K: Into<obj_core::codec::Dynamic> + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        let name = index_name.to_owned();
        unblock(move || inner.find_unique::<T>(&name, key)).await
    }

    /// Materialise the collection's primary B-tree into `Vec<T>`. See
    /// [`Db::all`].
    ///
    /// Streaming async iteration is intentionally **not** wrapped in
    /// this phase — the entire collection is collected inside the
    /// blocking task. For very large collections, drive the blocking
    /// [`Db::iter_all`] from a dedicated `tokio::task::spawn_blocking`
    /// (or equivalent) until the async streaming surface lands.
    ///
    /// # Errors
    ///
    /// As [`Db::all`].
    pub async fn all<T>(&self) -> Result<Vec<T>>
    where
        T: Document + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.all::<T>()).await
    }

    /// Run a closure inside a write transaction. See
    /// [`Db::transaction`].
    ///
    /// # Closure contract
    ///
    /// The closure runs **synchronously** inside the blocking task,
    /// so it must be `Send + 'static`; the return value `R` likewise.
    /// No `async fn` inside the closure — that is a deliberate
    /// "async-over-blocking" restriction, matching the contract used
    /// by sqlx and other async-over-sync database wrappers.
    ///
    /// # Errors
    ///
    /// As [`Db::transaction`].
    pub async fn transaction<R, F>(&self, body: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut WriteTxn<'_>) -> Result<R> + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.transaction(body)).await
    }

    /// Run a closure inside a read transaction. See
    /// [`Db::read_transaction`].
    ///
    /// Same `Send + 'static` closure restriction as
    /// [`AsyncDb::transaction`].
    ///
    /// # Errors
    ///
    /// As [`Db::read_transaction`].
    pub async fn read_transaction<R, F>(&self, body: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&ReadTxn<'_>) -> Result<R> + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.read_transaction(body)).await
    }

    /// Hot-backup the database to `dest`. See [`Db::backup_to`].
    ///
    /// # Errors
    ///
    /// As [`Db::backup_to`].
    pub async fn backup_to<P: AsRef<Path>>(&self, dest: P) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        let dest_buf: PathBuf = dest.as_ref().to_path_buf();
        unblock(move || inner.backup_to(dest_buf)).await
    }

    /// Attach a read-only `.obj` file under `namespace`. See
    /// [`Db::attach`].
    ///
    /// Takes `&mut self` because the blocking [`Db::attach`] takes
    /// `&mut Db`. We temporarily unwrap the `Arc<Db>` (it must be
    /// uniquely owned at this point) so the blocking task can take
    /// the `&mut`. If the `Arc` is shared, attach falls back to
    /// [`Error::Busy`](obj_core::Error::Busy) with the
    /// `WriterInProcess` kind — clone the `AsyncDb` only after all
    /// `attach` / `detach` calls.
    ///
    /// # Errors
    ///
    /// As [`Db::attach`], plus
    /// [`Error::Busy`](obj_core::Error::Busy) when the `Arc<Db>` is
    /// not uniquely owned at call time.
    pub async fn attach<P>(&mut self, path: P, namespace: impl Into<String>) -> Result<()>
    where
        P: AsRef<Path>,
    {
        let path_buf: PathBuf = path.as_ref().to_path_buf();
        let namespace = namespace.into();
        self.with_mut_db(move |db| db.attach(&path_buf, namespace))
            .await
    }

    /// Detach the attachment registered under `namespace`. See
    /// [`Db::detach`].
    ///
    /// Same `&mut self` contract as [`AsyncDb::attach`].
    ///
    /// # Errors
    ///
    /// As [`Db::detach`].
    pub async fn detach(&mut self, namespace: &str) -> Result<()> {
        let namespace = namespace.to_owned();
        self.with_mut_db(move |db| db.detach(&namespace)).await
    }

    /// Run `f` with a `&mut Db`. The `Arc<Db>` inside `self` must be
    /// uniquely owned at call time; otherwise the function returns
    /// [`Error::Busy`](obj_core::Error::Busy) so the caller does not
    /// silently observe stale-state behaviour.
    ///
    /// The `&mut self` receiver guarantees no other `&AsyncDb`
    /// reference borrows `self` for the duration of the call, but
    /// **clones** of `self` (which hold their own `Arc<Db>`) defeat
    /// the `Arc::try_unwrap` path. Reserve `attach` / `detach` for
    /// the bootstrap phase, before the `AsyncDb` is shared across
    /// tasks.
    async fn with_mut_db<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Db) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let sentinel = match Db::memory() {
            Ok(db) => Arc::new(db),
            Err(e) => return Err(e),
        };
        let original = std::mem::replace(&mut self.inner, sentinel);
        let mut db = match Arc::try_unwrap(original) {
            Ok(db) => db,
            Err(arc) => {
                self.inner = arc;
                return Err(obj_core::Error::Busy {
                    kind: obj_core::LockKind::WriterInProcess,
                });
            }
        };
        let (db_back, result) = unblock(move || {
            let result = f(&mut db);
            (db, result)
        })
        .await;
        self.inner = Arc::new(db_back);
        result
    }

    /// Run [`Db::integrity_check`]. See that method.
    ///
    /// # Errors
    ///
    /// As [`Db::integrity_check`].
    pub async fn integrity_check(&self) -> Result<IntegrityReport> {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.integrity_check()).await
    }

    /// Read [`Db::stat`]. See that method.
    ///
    /// # Errors
    ///
    /// As [`Db::stat`].
    pub async fn stat(&self) -> Result<DbStat> {
        let inner = Arc::clone(&self.inner);
        unblock(move || inner.stat()).await
    }

    /// Open a read-only typed handle to a runtime-named collection.
    /// See [`Db::collection`].
    ///
    /// Construction is infallible; the handle dispatches into the
    /// blocking pool on every read-only method call.
    #[must_use]
    pub fn collection<T>(&self, name: impl Into<String>) -> AsyncCollection<T>
    where
        T: Document + Send + 'static,
    {
        AsyncCollection::lazy(Arc::clone(&self.inner), name.into())
    }

    /// Construct a fresh [`AsyncQuery`] builder rooted at this
    /// database. See [`Db::query`].
    #[must_use]
    pub fn query<T>(&self) -> AsyncQuery<T>
    where
        T: Document + Send + 'static,
    {
        AsyncQuery::new(Arc::clone(&self.inner))
    }
}

/// Internal shim — `blocking::unblock` plus optional `tracing` span
/// propagation. Capturing the current span before the hop and
/// re-entering it inside the blocking task is the documented pattern
/// for `tracing` across thread-pool work (see the `tracing` crate's
/// `Span::in_scope` docs).
///
/// The helper centralises the propagation logic so each call site
/// stays a one-liner.
pub(crate) async fn unblock<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    #[cfg(feature = "tracing")]
    let span = tracing::Span::current();
    blocking::unblock(move || {
        #[cfg(feature = "tracing")]
        let _guard = span.enter();
        f()
    })
    .await
}
