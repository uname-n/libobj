//! Transaction layer (L7).
//!
//! Wraps the [`Pager`] + cross-process file locks + reader snapshots
//! into a write- / read-transaction abstraction.  Single-writer
//! model: a [`WriteTxn`] holds (a) the in-process write-
//! serialization gate on the pager-shared [`TxnEnv`] and (b) the
//! cross-process `WRITER_LOCK` byte (when the env was constructed
//! with a lock file).  [`ReadTxn`] holds a shared reader lock byte
//! and a [`ReaderSnapshot`]; readers do not contend with each other
//! and do not block writers.
//!
//! This module exposes the building blocks; the `obj` crate wraps the
//! result as `obj::Db`.
//!
//! # In-process write-serialization gate
//!
//! The gate is an [`AtomicBool`] behind an `Arc`, NOT a `Mutex<()>`.
//! An acquired
//! [`WriteSerialGuard`] OWNS a clone of that `Arc` and, on `Drop`,
//! `store(false, Release)`s the flag.  Because the guard owns the
//! `Arc` (rather than borrowing the env), it is `Send + 'static`,
//! which in turn makes [`WriteTxn`] `Send` — letting an FFI
//! binding move the blocking lock-acquire onto a worker thread.
//!
//! **No poisoning (deliberate, and strictly better).**  A `Mutex<()>`
//! poisons if a thread panics while holding the guard, turning every
//! subsequent `WriteTxn::begin` into a permanent
//! `Busy{WriterInProcess}`.  The `AtomicBool` gate has no such state:
//! if a writer panics mid-transaction, unwinding drops the
//! [`WriteTxn`] (whose `Drop` rolls back — restoring
//! `header_at_begin` so the pager is left at consistent committed
//! state) and then drops the [`WriteSerialGuard`] (which releases the
//! gate).  The next writer proceeds against that consistent state.
//! This replaces a permanent-Busy failure mode with a
//! recover-and-continue one.

#![forbid(unsafe_code)]

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::error::{Error, LockKind, Result};
use crate::pager::page::{Page, PageId};
use crate::pager::{HeaderSnapshot, Pager, ReaderSnapshot};
use crate::platform::{FileBackend, FileHandle, ReaderLock, WriterLock};

/// Default busy timeout for `WriteTxn::begin` and `ReadTxn::begin`
/// when the caller does not pass a per-call deadline.  5 seconds
/// matches `SQLite`'s default.
pub const DEFAULT_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Environment shared by every [`WriteTxn`] / [`ReadTxn`] in a
/// process.  Holds the pager (behind an `Arc<Mutex<_>>`), the in-
/// process write-serialization mutex, and — for file-backed
/// databases — an optional [`FileHandle`] used for cross-process
/// byte-range locking.
///
/// `TxnEnv` is `Send + Sync`; construct one and share via `Arc`
/// across threads (or across an `obj::Db` whose ownership wraps it).
#[derive(Debug)]
pub struct TxnEnv<F: FileBackend = FileHandle> {
    /// The pager.  Behind a Mutex so cache mutations on a cache miss
    /// stay sound under concurrent reader-snapshot reads.  The
    /// scaling is limited by this mutex; lock-free read paths are
    /// future work.
    pager: Arc<Mutex<Pager<F>>>,
    /// In-process writer-serialization gate.  `false` = free,
    /// `true` = held.  A `WriteTxn` holds a [`WriteSerialGuard`] (which
    /// owns a clone of this `Arc`) for its entire lifetime, so at most
    /// one `WriteTxn` is alive per env per process at a time.  Behind
    /// an `Arc` so the guard can OWN it and therefore be `Send`;
    /// an `AtomicBool` rather than a `Mutex<()>` so it
    /// cannot poison (see the module-level "In-process gate" docs).
    write_serialization: Arc<AtomicBool>,
    /// File handle held for cross-process locking.  `None` for in-
    /// memory envs (no file to lock) or for callers that explicitly
    /// disable the cross-process lock (e.g. fault-injection tests
    /// where the file-backend type is a harness rather than the
    /// production `FileHandle`).  The handle owns its own fd so
    /// locking calls do not need the pager's mutex.
    lock_file: Option<Arc<FileHandle>>,
}

impl<F: FileBackend> TxnEnv<F> {
    /// Construct an env wrapping the given pager.  `lock_file` is an
    /// optional dedicated file handle for cross-process locks; pass
    /// `None` for in-memory or fault-injection environments.
    #[must_use]
    pub fn new(pager: Pager<F>, lock_file: Option<Arc<FileHandle>>) -> Self {
        Self {
            pager: Arc::new(Mutex::new(pager)),
            write_serialization: Arc::new(AtomicBool::new(false)),
            lock_file,
        }
    }

    /// Shared access to the pager's `Arc<Mutex<_>>`.  Used by the
    /// `obj::Db` wrapper to dispatch reads inside a
    /// `ReadTxn` closure.  Exposed at the txn boundary so the
    /// stress test can sample state without a full `WriteTxn`.
    #[must_use]
    pub fn pager(&self) -> &Arc<Mutex<Pager<F>>> {
        &self.pager
    }
}

/// RAII guard on the in-process write-serialization gate.
///
/// Owns a clone of the env's `Arc<AtomicBool>` so it is `Send +
/// 'static` (it does NOT borrow the env).  Constructed only by the
/// crate-internal gate acquire, which sets the flag to `true` via a
/// CAS; `Drop` clears it with a `Release` store so a writer that
/// releases on one thread is observed by an acquirer that CASes on
/// another.
///
/// At most one `WriteSerialGuard` exists per env at a time — that is
/// the single-writer invariant the CAS enforces.
#[derive(Debug)]
#[must_use = "the gate is released when the WriteSerialGuard drops"]
pub struct WriteSerialGuard {
    gate: Arc<AtomicBool>,
}

impl Drop for WriteSerialGuard {
    fn drop(&mut self) {
        self.gate.store(false, Ordering::Release);
    }
}

/// A `Send` token holding BOTH blocking-acquired write locks, with no
/// borrow of the env.
///
/// Returned by [`WriteTxn::acquire`] and consumed by
/// [`WriteTxn::from_acquire`].  Splitting the blocking acquisition
/// (which yields this `Send` token) from the cheap, non-blocking txn
/// assembly lets a caller (an FFI binding) run the acquire on a
/// worker thread, then build the `!'static` `WriteTxn` afterward.
///
/// Holds the in-process [`WriteSerialGuard`] and the optional
/// cross-process [`WriterLock`]; both are `Send`, so this token is
/// `Send`.  Generic over `F` only to thread the same backend type
/// through to [`WriteTxn::from_acquire`]; the token itself stores no
/// `F` value, so it carries a `PhantomData<fn() -> F>` to stay
/// covariant-free and `Send` regardless of `F`.
#[derive(Debug)]
#[must_use = "a WriteAcquire holds the write locks until consumed by WriteTxn::from_acquire"]
pub struct WriteAcquire<F: FileBackend> {
    write_guard: WriteSerialGuard,
    writer_lock: Option<WriterLock>,
    _backend: core::marker::PhantomData<fn() -> F>,
}

/// A write transaction.
///
/// Construct via [`WriteTxn::begin`] (or, to release a host lock
/// across the blocking acquire, [`WriteTxn::acquire`] +
/// [`WriteTxn::from_acquire`]).  Holds, for its entire lifetime:
/// 1. A [`WriteSerialGuard`] on the env's in-process write-
///    serialization gate — ensures at most one `WriteTxn` per env per
///    process.
/// 2. An optional cross-process `WRITER_LOCK` byte — ensures at most
///    one `WriteTxn` across the cluster of processes that have
///    opened the same file.
///
/// `WriteTxn::commit` finalises the transaction through the pager's
/// WAL; `WriteTxn::rollback` discards pending writes.  Dropping an
/// uncommitted `WriteTxn` rolls back automatically (and logs a
/// `tracing` debug event, gated on the `tracing` feature, so the
/// caller learns about the silent rollback).
///
/// `Send`: every field is `Send` — `&TxnEnv` is
/// `Send + Sync`, [`WriteSerialGuard`] owns an `Arc<AtomicBool>`, and
/// [`WriterLock`] / [`HeaderSnapshot`] are `Send`.  Soundness rests on
/// the single-writer invariant: at most one `WriteTxn` exists per env
/// at a time (the gate enforces it), and every pager access re-locks
/// the pager `Mutex` per-op, so there is no thread-affine state to
/// violate when the handle moves between threads.
///
/// Generic over `F: FileBackend`.
#[derive(Debug)]
pub struct WriteTxn<'db, F: FileBackend> {
    env: &'db TxnEnv<F>,
    /// In-process write-serialization guard.  Owns an `Arc<AtomicBool>`
    /// (the env's gate), so it is `Send` — this is what makes the
    /// whole `WriteTxn` `Send`.  `Option` because `commit`
    /// and `rollback` consume it before the txn is dropped.
    write_guard: Option<WriteSerialGuard>,
    /// Cross-process `WRITER_LOCK` guard.  `None` for envs without a
    /// `lock_file`.  Released on drop or on explicit
    /// `commit`/`rollback`.
    writer_lock: Option<WriterLock>,
    /// Snapshot of header fields the catalog + freelist code
    /// writes direct-to-disk, plus a clone of the WAL committed
    /// view at txn begin.  Restored on rollback so a rolled-back
    /// txn that mutated the catalog (via the `obj::Db` public
    /// API) does not leak a header that points at unwritten page
    /// bodies, and does not leave the WAL view missing pages that
    /// `free_page` removed.  `Option` because `commit`/`rollback`
    /// take ownership of the snapshot.  See
    /// `Pager::header_snapshot` for the rationale.
    header_at_begin: Option<HeaderSnapshot>,
    /// `true` once `commit` or `rollback` has run.  A `Drop` on a
    /// committed/rolled-back txn is a no-op.
    finished: bool,
}

impl<'db, F: FileBackend> WriteTxn<'db, F> {
    /// Begin a new write transaction against `env`.
    ///
    /// Equivalent to [`Self::from_acquire`]`(env, `[`Self::acquire`]
    /// `(env, timeout)?)`: it performs the two blocking lock acquires
    /// (in-process gate, then cross-process `WRITER_LOCK`) and then
    /// assembles the txn.  Callers that need to release a host runtime
    /// lock across the blocking wait should call
    /// [`Self::acquire`] / [`Self::from_acquire`] directly so the
    /// blocking step runs without the host lock held.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] with `LockKind::Writer` if the cross-
    ///   process lock did not become available within `timeout`.
    /// - [`Error::Busy`] with `LockKind::WriterInProcess` if another
    ///   thread in the same process is mid-write.
    /// - [`Error::Io`] on lock syscall failure.
    pub fn begin(env: &'db TxnEnv<F>, timeout: Duration) -> Result<Self> {
        Self::from_acquire(env, Self::acquire(env, timeout)?)
    }

    /// Perform BOTH blocking write-lock acquires and return a `Send`
    /// token, without touching the pager or borrowing the env's
    /// lifetime beyond the call.
    ///
    /// Acquires the in-process serialization gate FIRST (bounded busy-
    /// poll against `timeout`), THEN the cross-process `WRITER_LOCK`
    /// (if `env.lock_file` is `Some`).  This order is load-bearing:
    /// the cross-process OFD lock is per-fd and the whole process
    /// shares one lock-file fd, so two same-process threads would BOTH
    /// pass the cross-process lock — the in-process gate is the
    /// authoritative same-process serializer and must win first.  On a
    /// cross-process failure the in-process guard is dropped (releasing
    /// the gate) before the error returns, exactly as `begin` did.
    ///
    /// The returned [`WriteAcquire`] owns both guards and is `Send`,
    /// so the caller may move it across threads (e.g. acquire on a
    /// worker thread).
    ///
    /// # Errors
    ///
    /// As [`Self::begin`].
    pub fn acquire(env: &'db TxnEnv<F>, timeout: Duration) -> Result<WriteAcquire<F>> {
        let write_guard = acquire_write_serialization(&env.write_serialization, timeout)?;
        let writer_lock = match env.lock_file.as_ref() {
            Some(handle) => match handle.lock_writer(timeout) {
                Ok(g) => Some(g),
                Err(e) => {
                    drop(write_guard);
                    return Err(e);
                }
            },
            None => None,
        };
        Ok(WriteAcquire {
            write_guard,
            writer_lock,
            _backend: core::marker::PhantomData,
        })
    }

    /// Assemble a `WriteTxn` from a previously-[`acquire`](Self::acquire)d
    /// token.  This is the cheap, NON-blocking half of `begin`: it
    /// briefly locks the pager to flip the txn-depth gauge and take the
    /// header snapshot, then takes ownership of the token's guards.
    ///
    /// Holding the pager mutex here cannot deadlock: the caller already
    /// owns the in-process gate (carried in `acq`), so no other
    /// `WriteTxn` is alive, and the snapshot is cheap.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] with `LockKind::WriterInProcess` if the pager
    ///   mutex is poisoned.
    pub fn from_acquire(env: &'db TxnEnv<F>, acq: WriteAcquire<F>) -> Result<Self> {
        let header_at_begin = {
            let mut pager = env.pager.lock().map_err(|_| Error::Busy {
                kind: LockKind::WriterInProcess,
            })?;
            pager.begin_txn();
            pager.header_snapshot()
        };
        Ok(Self {
            env,
            write_guard: Some(acq.write_guard),
            writer_lock: acq.writer_lock,
            header_at_begin: Some(header_at_begin),
            finished: false,
        })
    }

    /// Write `page` at `id` through the pager.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] on syscall failure.
    pub fn write_page(&self, id: PageId, page: &Page) -> Result<()> {
        let mut pager = self.lock_pager()?;
        pager.write_page(id, page)
    }

    /// Read `id` through the pager (sees pending + committed +
    /// main).  Used inside a write txn that needs to read-modify-
    /// write a page.  Returns an owned [`Page`] because the borrow
    /// chain through the pager's mutex would otherwise tie the
    /// returned reference to the guard.
    ///
    /// # Errors
    ///
    /// As [`Pager::read_page`].
    pub fn read_page(&self, id: PageId) -> Result<Page> {
        let mut pager = self.lock_pager()?;
        let page_ref = pager.read_page(id)?;
        Ok(page_ref.to_owned_page())
    }

    /// Allocate a new page through the pager.
    ///
    /// # Errors
    ///
    /// As [`Pager::alloc_page`].
    pub fn alloc_page(&self) -> Result<PageId> {
        let mut pager = self.lock_pager()?;
        pager.alloc_page()
    }

    /// Acquire the pager mutex.  Bubble a poisoned mutex up as
    /// `WriterInProcess` — every txn method that takes the pager
    /// goes through here so the failure mode is uniform.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Busy`] with `LockKind::WriterInProcess` if
    /// the mutex is poisoned by a previous panic.
    pub fn lock_pager(&self) -> Result<MutexGuard<'_, Pager<F>>> {
        self.env.pager.lock().map_err(|_| Error::Busy {
            kind: LockKind::WriterInProcess,
        })
    }

    /// Access the underlying env.  Used by callers (e.g. `obj`
    /// crate) that compose typed handles over the raw txn.
    #[must_use]
    pub fn env(&self) -> &'db TxnEnv<F> {
        self.env
    }

    /// Commit the transaction.  Calls `Pager::commit` to make the
    /// staged writes durable, then releases both lock layers.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Pager::commit`] returns.
    pub fn commit(mut self) -> Result<()> {
        {
            let mut pager = self.lock_pager()?;
            let _lsn = pager.commit()?;
            pager.end_txn();
        }
        self.finished = true;
        self.write_guard.take();
        self.header_at_begin.take();
        if let Some(lock) = self.writer_lock.take() {
            lock.release()?;
        }
        Ok(())
    }

    /// Roll the transaction back, dropping all pending writes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Busy`] only if the in-process pager mutex
    /// is poisoned; otherwise infallible.
    pub fn rollback(mut self) -> Result<()> {
        let snap = self.header_at_begin.take();
        {
            let mut pager = self.lock_pager()?;
            rollback_pending(&mut pager);
            if let Some(s) = snap {
                pager.restore_header_snapshot(s)?;
            }
            pager.end_txn();
        }
        self.finished = true;
        self.write_guard.take();
        if let Some(lock) = self.writer_lock.take() {
            lock.release()?;
        }
        Ok(())
    }
}

impl<F: FileBackend> Drop for WriteTxn<'_, F> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let snap = self.header_at_begin.take();
        if let Ok(mut pager) = self.env.pager.lock() {
            rollback_pending(&mut pager);
            if let Some(s) = snap {
                let _ = pager.restore_header_snapshot(s);
            }
            pager.end_txn();
        }
        #[cfg(feature = "tracing")]
        tracing::debug!("WriteTxn dropped without commit/rollback; pending writes discarded");
    }
}

/// Discard the pager's pending-transaction buffer.  Idempotent.
///
/// The pager exposes `commit` / `checkpoint` but no explicit
/// rollback — pending writes simply never make it into the WAL.
/// This helper drains the in-memory pending map so a re-entered
/// `WriteTxn` on the same pager observes a clean slate.
fn rollback_pending<F: FileBackend>(pager: &mut Pager<F>) {
    pager.rollback_pending_writes();
}

/// Busy-poll a CAS on the in-process gate (`AtomicBool`) until it is
/// acquired or `timeout` elapses, returning an owning
/// [`WriteSerialGuard`].  The loop is bounded by
/// the deadline AND a defensive iter counter; a return of
/// `Err(Error::Busy{WriterInProcess})` is the surfaced alternative to
/// blocking forever.
///
/// The backoff schedule mirrors the prior `Mutex<()>` polyfill and
/// `platform::lock`'s file-lock retry exactly: 1 ms initial, doubled
/// per miss, capped at 100 ms.  Average overhead under uncontended
/// load is one `compare_exchange_weak`.
///
/// Memory ordering: success uses `Acquire` so the winner observes the
/// previous holder's writes (paired with the `Release` store in
/// [`WriteSerialGuard::drop`]); failure uses `Relaxed` (a failed CAS
/// synchronizes nothing).  `compare_exchange_weak` is used because the
/// retry loop tolerates spurious failures and it is cheaper on some
/// targets.
///
/// Unlike the old `Mutex<()>`, the gate cannot poison — there is no
/// `Poisoned` arm; a panicking writer's guard `Drop` simply frees the
/// gate (see the module-level "In-process gate" docs).
fn acquire_write_serialization(
    gate: &Arc<AtomicBool>,
    timeout: Duration,
) -> Result<WriteSerialGuard> {
    let start = std::time::Instant::now();
    let mut backoff = Duration::from_millis(1);
    let max_backoff = Duration::from_millis(100);
    let timeout_millis = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
    let mut iters: u64 = 0;
    let max_iters = timeout_millis.saturating_add(64);
    loop {
        iters = iters.saturating_add(1);
        if iters > max_iters {
            return Err(Error::Busy {
                kind: LockKind::WriterInProcess,
            });
        }
        if gate
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return Ok(WriteSerialGuard {
                gate: Arc::clone(gate),
            });
        }
        if start.elapsed() >= timeout {
            return Err(Error::Busy {
                kind: LockKind::WriterInProcess,
            });
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(max_backoff);
    }
}

/// A read transaction.
///
/// Construct via [`ReadTxn::begin`].  Holds:
/// 1. A [`ReaderSnapshot`] pinning the WAL end-LSN at construction.
/// 2. An optional cross-process `READER_LOCK` byte (shared with
///    other readers but not exclusive with writers).
///
/// Reads inside a `ReadTxn` see a consistent snapshot of the
/// database; pending writes from a concurrent `WriteTxn` and frames
/// committed AFTER `ReadTxn::begin` are invisible.
#[derive(Debug)]
pub struct ReadTxn<'db, F: FileBackend> {
    env: &'db TxnEnv<F>,
    snapshot: ReaderSnapshot<F>,
    _reader_lock: Option<ReaderLock>,
}

impl<'db, F: FileBackend> ReadTxn<'db, F> {
    /// Begin a new read transaction.
    ///
    /// Acquires a cross-process reader-lock slot (if `env.lock_file`
    /// is `Some`) with the env's default busy timeout, then takes a
    /// [`ReaderSnapshot`] from the pager.  Readers do not contend
    /// with each other (31 shared slots) and do not contend with
    /// writers on the byte-range layer.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] with `LockKind::Reader` on lock timeout.
    /// - [`Error::Io`] on lock or pager syscall failure.
    pub fn begin(env: &'db TxnEnv<F>) -> Result<Self> {
        Self::begin_with_timeout(env, DEFAULT_BUSY_TIMEOUT)
    }

    /// As [`begin`](Self::begin) with a caller-supplied timeout.
    ///
    /// # Errors
    ///
    /// See [`begin`](Self::begin).
    pub fn begin_with_timeout(env: &'db TxnEnv<F>, timeout: Duration) -> Result<Self> {
        let reader_lock = match env.lock_file.as_ref() {
            Some(handle) => Some(handle.lock_reader(timeout)?),
            None => None,
        };
        let snapshot = {
            let mut pager = env.pager.lock().map_err(|_| Error::Busy {
                kind: LockKind::WriterInProcess,
            })?;
            pager.reader_snapshot()?
        };
        Ok(Self {
            env,
            snapshot,
            _reader_lock: reader_lock,
        })
    }

    /// LSN this txn's snapshot pinned.  Diagnostic-only.
    #[must_use]
    pub fn pinned_lsn(&self) -> crate::wal::Lsn {
        self.snapshot.pinned_lsn()
    }

    /// Read page `id` consistent with the txn's snapshot.
    ///
    /// # Errors
    ///
    /// As [`ReaderSnapshot::read_page`].
    pub fn read_page(&self, id: PageId) -> Result<Page> {
        let pager = self.env.pager.lock().map_err(|_| Error::Busy {
            kind: LockKind::WriterInProcess,
        })?;
        Ok(self.snapshot.read_page(&pager, id)?.into_page())
    }

    /// Access the underlying snapshot.  Used by `obj::Db` to
    /// dispatch typed reads on a read txn.
    #[must_use]
    pub fn snapshot(&self) -> &ReaderSnapshot<F> {
        &self.snapshot
    }

    /// Access the env this txn lives in.  Used by `obj::Db` to
    /// access the pager mutex.
    #[must_use]
    pub fn env(&self) -> &TxnEnv<F> {
        self.env
    }

    /// Drop the txn explicitly (releases the snapshot pin and the
    /// reader lock).
    pub fn end(self) {
        drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::page::Page;
    use crate::pager::Config;
    use crate::platform::FileHandle;
    use std::sync::Arc;
    use std::thread;
    use tempfile::TempDir;

    fn build_env(dir: &TempDir) -> (TxnEnv<FileHandle>, PageId) {
        let path = dir.path().join("txn.obj");
        let mut pager = Pager::open(&path, Config::default()).expect("pager");
        pager.begin_txn();
        let a = pager.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0;
        pager.write_page(a, &page).expect("write");
        let _ = pager.commit().expect("commit");
        pager.end_txn();
        let lock_path = crate::pager::lock_path_for(&path);
        let lock_file = Arc::new(FileHandle::open_or_create(&lock_path).expect("lock file"));
        lock_file.set_len(128).expect("lock sidecar len");
        (TxnEnv::new(pager, Some(lock_file)), a)
    }

    #[test]
    fn write_txn_commit_makes_writes_visible() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);
        let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("begin");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x77;
        tx.write_page(a, &page).expect("write");
        tx.commit().expect("commit");

        let rx = ReadTxn::begin(&env).expect("read");
        let observed = rx.read_page(a).expect("read");
        assert_eq!(observed.as_bytes()[0], 0x77);
    }

    #[test]
    fn write_txn_rollback_drops_writes() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);
        let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("begin");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x99;
        tx.write_page(a, &page).expect("write");
        tx.rollback().expect("rollback");

        let rx = ReadTxn::begin(&env).expect("read");
        let observed = rx.read_page(a).expect("read");
        assert_eq!(observed.as_bytes()[0], 0, "rollback must discard writes");
    }

    #[test]
    fn in_process_writers_serialize() {
        let dir = TempDir::new().expect("tmp");
        let (env, _a) = build_env(&dir);
        let tx1 = WriteTxn::begin(&env, Duration::from_millis(50)).expect("tx1");
        let err = WriteTxn::begin(&env, Duration::from_millis(10)).expect_err("tx2 busy");
        assert!(matches!(
            err,
            Error::Busy {
                kind: LockKind::WriterInProcess
            }
        ));
        tx1.commit().expect("commit");
        let _tx3 = WriteTxn::begin(&env, Duration::from_millis(50)).expect("tx3");
    }

    /// `WriteTxn` (and the new `WriteAcquire` /
    /// `WriteSerialGuard`) must be `Send` so an FFI binding can move
    /// the blocking acquire across a worker-thread boundary.  A
    /// compile-time assertion via a generic `fn`; the `let _ =` keeps
    /// it from being dead-code-eliminated.
    #[test]
    fn write_txn_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<WriteTxn<'_, FileHandle>>();
        assert_send::<WriteAcquire<FileHandle>>();
        assert_send::<WriteSerialGuard>();
        assert_send::<ReadTxn<'_, FileHandle>>();
    }

    /// The gate does not poison.  If a writer panics
    /// mid-transaction, unwinding drops the `WriteTxn` (rollback) then
    /// the `WriteSerialGuard` (releases the gate), so the NEXT writer
    /// proceeds — against the rolled-back, consistent state.  The old
    /// `Mutex<()>` would have poisoned and turned every later `begin`
    /// into a permanent `Busy{WriterInProcess}`.
    #[test]
    fn panic_in_writer_releases_gate_and_rolls_back() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);
        let env = Arc::new(env);
        let env_for_panic = Arc::clone(&env);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let tx = WriteTxn::begin(&env_for_panic, Duration::from_millis(50)).expect("begin");
            let mut page = Page::zeroed();
            page.as_bytes_mut()[0] = 0xEE;
            tx.write_page(a, &page).expect("write");
            panic!("simulated mid-write crash");
        }));
        assert!(result.is_err(), "the closure must have panicked");
        let tx2 = WriteTxn::begin(&env, Duration::from_millis(50))
            .expect("gate must be free after a panicking writer");
        tx2.commit().expect("commit");
        let rx = ReadTxn::begin(&env).expect("read");
        let observed = rx.read_page(a).expect("read");
        assert_eq!(
            observed.as_bytes()[0],
            0,
            "panicking writer's staged write must have been rolled back",
        );
    }

    /// 4 writer threads × 1000 iterations each — N writers
    /// serialize via the in-process Mutex + the cross-process
    /// `WRITER_LOCK`.  Every txn must succeed (no deadlock; no
    /// aborted txn).
    #[test]
    fn n_writers_serialize_with_no_deadlock() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("stress.obj");
        let mut pager = Pager::open(&path, Config::default()).expect("pager");
        pager.begin_txn();
        let a = pager.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0;
        pager.write_page(a, &page).expect("write");
        let _ = pager.commit().expect("commit");
        pager.end_txn();
        let lock_path = crate::pager::lock_path_for(&path);
        let lock_file = Arc::new(FileHandle::open_or_create(&lock_path).expect("lock"));
        lock_file.set_len(128).expect("lock sidecar len");
        let env = Arc::new(TxnEnv::new(pager, Some(lock_file)));

        let n_writers = 4usize;
        let iters_per_writer = 250u32;
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_writers);
            for w in 0..n_writers {
                let env = Arc::clone(&env);
                handles.push(scope.spawn(move || {
                    for i in 0..iters_per_writer {
                        let tx = WriteTxn::begin(&env, Duration::from_secs(30))
                            .expect("begin under load");
                        let mut p = Page::zeroed();
                        p.as_bytes_mut()[0] =
                            u8::try_from((w * 1000 + i as usize) % 250 + 1).expect("byte fits");
                        tx.write_page(a, &p).expect("write");
                        tx.commit().expect("commit");
                    }
                }));
            }
            for h in handles {
                h.join().expect("join");
            }
        });

        let rx = ReadTxn::begin(&env).expect("read");
        let p = rx.read_page(a).expect("read");
        assert_ne!(p.as_bytes()[0], 0, "some writer's value must be visible");
    }

    #[test]
    fn drop_without_commit_warns_and_rolls_back() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);
        {
            let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("begin");
            let mut page = Page::zeroed();
            page.as_bytes_mut()[0] = 0xAB;
            tx.write_page(a, &page).expect("write");
        }
        let rx = ReadTxn::begin(&env).expect("read");
        let observed = rx.read_page(a).expect("read");
        assert_eq!(
            observed.as_bytes()[0],
            0,
            "drop-without-commit must roll back",
        );
    }

    #[test]
    fn read_txn_sees_consistent_snapshot() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);

        let rx = ReadTxn::begin(&env).expect("read");

        {
            let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("write");
            let mut p = Page::zeroed();
            p.as_bytes_mut()[0] = 0x55;
            tx.write_page(a, &p).expect("write");
            tx.commit().expect("commit");
        }

        let observed = rx.read_page(a).expect("read");
        assert_eq!(
            observed.as_bytes()[0],
            0,
            "snapshot must isolate reader from concurrent commits",
        );
    }
}
