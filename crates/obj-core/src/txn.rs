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
//!
//! **The pager mutex, too, recovers from poison.**  The serialization
//! gate above is an `AtomicBool`, but the pager itself lives behind an
//! `Arc<Mutex<Pager>>`, and a `std::sync::Mutex` *does* poison if a
//! thread panics while holding its guard.  To keep the guarantee above
//! whole even for a panic that fires *under the pager lock*, every
//! pager acquire in this module goes through `lock_pager_recovering`,
//! which recovers the guard via [`PoisonError::into_inner`] instead of
//! mapping poison to a permanent `Busy{WriterInProcess}`.  Recovery is
//! sound because a poisoned pager mutex reflects only an in-process
//! panic, never on-disk corruption (that is reported separately as
//! [`Error::Corruption`]); the panicking [`WriteTxn`]'s `Drop` still
//! rolls the pager back to its last committed state before the next
//! locker sees it.

#![forbid(unsafe_code)]

use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use crate::error::{Error, LockKind, Result};
use crate::pager::page::{Page, PageId};
use crate::pager::{HeaderSnapshot, Pager, ReaderSnapshot};
use crate::platform::{Clock, FileBackend, FileHandle, ReaderLock, SystemClock, WriterLock};

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
    /// Time source for the write-serialization gate and the cross-
    /// process-lock backoff/timeout loops.  Behind an `Arc<dyn Clock>`
    /// (a cold-path trait object, like [`crate::platform::Entropy`]) so
    /// the DST harness can substitute a deterministic, non-blocking
    /// [`crate::platform::SimClock`] while production uses the real
    /// [`SystemClock`].  Every acquire path reads its time and sleeps
    /// only through this handle.
    clock: Arc<dyn Clock>,
}

impl<F: FileBackend> TxnEnv<F> {
    /// Construct an env wrapping the given pager, using the production
    /// [`SystemClock`].  `lock_file` is an optional dedicated file
    /// handle for cross-process locks; pass `None` for in-memory or
    /// fault-injection environments.
    ///
    /// This is the convenience constructor for callers that do not care
    /// about the clock seam; use [`Self::new_with_clock`] to inject a
    /// deterministic [`crate::platform::SimClock`] under the DST harness.
    #[must_use]
    pub fn new(pager: Pager<F>, lock_file: Option<Arc<FileHandle>>) -> Self {
        Self::new_with_clock(pager, lock_file, Arc::new(SystemClock))
    }

    /// Construct an env wrapping the given pager with an explicit
    /// [`Clock`].  The DST harness passes a
    /// [`crate::platform::SimClock`] so the write-gate and lock
    /// backoff/timeout loops advance virtual time only on `sleep` —
    /// deterministic and non-blocking.
    #[must_use]
    pub fn new_with_clock(
        pager: Pager<F>,
        lock_file: Option<Arc<FileHandle>>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            pager: Arc::new(Mutex::new(pager)),
            write_serialization: Arc::new(AtomicBool::new(false)),
            lock_file,
            clock,
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

    /// The env's injected [`Clock`].  The acquire paths read time and
    /// sleep only through this handle so the DST harness can drive them
    /// deterministically.
    #[must_use]
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
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
        let clock = env.clock.as_ref();
        let write_guard = acquire_write_serialization(&env.write_serialization, timeout, clock)?;
        let writer_lock = match env.lock_file.as_ref() {
            Some(handle) => match handle.lock_writer(timeout, clock) {
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
    /// Infallible today: the pager mutex is acquired with poison
    /// recovery (see `lock_pager_recovering`), so a prior
    /// panic-under-lock no longer surfaces here as `Busy`.
    pub fn from_acquire(env: &'db TxnEnv<F>, acq: WriteAcquire<F>) -> Result<Self> {
        let header_at_begin = {
            let mut pager = lock_pager_recovering(&env.pager);
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

    /// Acquire the pager mutex.  Recovers from a poisoned mutex rather
    /// than wedging the handle — every txn method that takes the pager
    /// goes through here so the failure mode is uniform (see
    /// `lock_pager_recovering` for why recovery is sound).
    ///
    /// # Errors
    ///
    /// Infallible today; retains the `Result` return so callers need
    /// not change if a genuinely fallible acquire is reintroduced.
    pub fn lock_pager(&self) -> Result<MutexGuard<'_, Pager<F>>> {
        Ok(lock_pager_recovering(&self.env.pager))
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
        // Recover a poisoned pager mutex here rather than skipping
        // rollback: if THIS txn is unwinding from a panic that fired
        // under the pager lock, the mutex is poisoned, and a plain
        // `if let Ok(..)` would silently drop the rollback and leave the
        // pager wedged in permanent `Busy`. Recovering lets rollback run
        // (restoring last-committed state), delivering the module's
        // recover-and-continue guarantee. See `lock_pager_recovering`.
        let mut pager = lock_pager_recovering(&self.env.pager);
        rollback_pending(&mut pager);
        if let Some(s) = snap {
            let _ = pager.restore_header_snapshot(s);
        }
        pager.end_txn();
        drop(pager);
        #[cfg(feature = "tracing")]
        tracing::debug!("WriteTxn dropped without commit/rollback; pending writes discarded");
    }
}

/// Acquire the pager mutex, recovering from poison instead of wedging
/// the handle.
///
/// A `std::sync::Mutex` poisons when a thread panics *while holding its
/// guard*.  The pager lives behind `Arc<Mutex<Pager>>`, so a panic under
/// the pager lock would poison it; the previous `.map_err(|_| Busy)`
/// mapping then turned **every** later `begin`/read/write into a
/// permanent `Busy{WriterInProcess}` (the poison persists until reopen),
/// contradicting this module's recover-and-continue contract.
///
/// Recovering via [`PoisonError::into_inner`] is sound here: a poisoned
/// pager mutex signals only an *in-process* panic, never on-disk
/// corruption — real corruption is surfaced separately as
/// [`Error::Corruption`] by the bounds-checked decode paths, so
/// recovering the lock cannot mask it.  In-memory consistency is
/// restored on the next transactional step: a panicking [`WriteTxn`]'s
/// `Drop` rolls back to `header_at_begin` (draining pending writes and
/// restoring the header snapshot), leaving the pager at its last
/// committed state for the next locker.
fn lock_pager_recovering<F: FileBackend>(pager: &Mutex<Pager<F>>) -> MutexGuard<'_, Pager<F>> {
    pager.lock().unwrap_or_else(PoisonError::into_inner)
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
    clock: &dyn Clock,
) -> Result<WriteSerialGuard> {
    let start = clock.now_millis();
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
        if clock.now_millis().saturating_sub(start) >= timeout_millis {
            return Err(Error::Busy {
                kind: LockKind::WriterInProcess,
            });
        }
        clock.sleep(backoff);
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
            Some(handle) => Some(handle.lock_reader(timeout, env.clock.as_ref())?),
            None => None,
        };
        let snapshot = {
            let mut pager = lock_pager_recovering(&env.pager);
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
        let pager = lock_pager_recovering(&self.env.pager);
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
    use crate::platform::{FileHandle, SimClock};
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

    /// Build an env whose clock is an injected [`SimClock`], returning
    /// both so the test can inspect virtual time.  Mirrors `build_env`
    /// but routes through `TxnEnv::new_with_clock`.
    fn build_env_with_sim_clock(dir: &TempDir) -> (TxnEnv<FileHandle>, Arc<SimClock>) {
        let path = dir.path().join("simclock.obj");
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
        let clock = Arc::new(SimClock::new());
        let env = TxnEnv::new_with_clock(pager, Some(lock_file), clock.clone());
        (env, clock)
    }

    /// Acceptance criterion 2: a `SimClock`-driven env drives the write-
    /// serialization gate to `Busy { WriterInProcess }` DETERMINISTICALLY
    /// and WITHOUT real sleeping.  With a real `SystemClock` a 5 s
    /// timeout against a held gate would block ~5 s; the `SimClock`
    /// advances virtual time only on `sleep`, so the loop exhausts the
    /// timeout instantly.
    #[test]
    fn sim_clock_drives_write_gate_to_busy_without_real_sleep() {
        let dir = TempDir::new().expect("tmp");
        let (env, clock) = build_env_with_sim_clock(&dir);

        // First writer holds the in-process gate.
        let tx1 = WriteTxn::begin(&env, Duration::from_millis(50)).expect("tx1");
        assert_eq!(clock.now_millis(), 0, "uncontended begin never sleeps");

        // Second writer contends: a 5 s virtual timeout must resolve to
        // Busy essentially instantly on the real clock.
        let wall_start = std::time::Instant::now();
        let err = WriteTxn::begin(&env, Duration::from_secs(5)).expect_err("tx2 must be busy");
        let wall = wall_start.elapsed();

        assert!(
            matches!(
                err,
                Error::Busy {
                    kind: LockKind::WriterInProcess
                }
            ),
            "contended gate must surface WriterInProcess busy",
        );
        assert!(
            wall < Duration::from_secs(1),
            "SimClock must not really sleep; took {wall:?}",
        );
        assert!(
            clock.now_millis() >= 5000,
            "virtual time must have advanced past the 5 s timeout; got {} ms",
            clock.now_millis(),
        );

        // The gate is still sound: dropping tx1 lets a new writer proceed.
        drop(tx1);
        let tx3 = WriteTxn::begin(&env, Duration::from_millis(50)).expect("tx3 after release");
        tx3.commit().expect("commit");
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

    /// A panic *under the pager lock* poisons the pager `Mutex`, but the
    /// txn layer recovers instead of wedging into permanent `Busy`.
    /// Before the fix, every later `begin`/read re-locked the poisoned
    /// pager and mapped the poison to `Busy{WriterInProcess}` forever.
    #[test]
    fn poisoned_pager_mutex_recovers_and_continues() {
        let dir = TempDir::new().expect("tmp");
        let (env, a) = build_env(&dir);
        let env = Arc::new(env);
        // Poison the pager mutex directly: panic WHILE holding its guard.
        let env_for_panic = Arc::clone(&env);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = env_for_panic.pager.lock().expect("lock");
            panic!("simulated crash under the pager lock");
        }));
        assert!(result.is_err(), "closure must have panicked");
        assert!(env.pager.is_poisoned(), "pager mutex must now be poisoned");

        // Recover-and-continue: a fresh writer still begins, writes, and
        // commits despite the poisoned pager mutex.
        let tx = WriteTxn::begin(&env, Duration::from_millis(50))
            .expect("begin must recover from a poisoned pager mutex");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x5A;
        tx.write_page(a, &page).expect("write");
        tx.commit().expect("commit");

        let rx = ReadTxn::begin(&env).expect("read must recover from poison");
        let observed = rx.read_page(a).expect("read");
        assert_eq!(observed.as_bytes()[0], 0x5A);
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

    /// A `FileBackend` that fails every mutating syscall once `armed`.
    /// Reads always delegate so recovery / snapshot read paths work.
    /// Used to fail a `WriteTxn`'s inline auto-checkpoint AFTER the WAL
    /// commit has made the txn durable (issue #2).
    struct ArmedFailHandle {
        inner: FileHandle,
        armed: Arc<AtomicBool>,
    }

    impl ArmedFailHandle {
        fn fail_if_armed(&self) -> Result<()> {
            if self.armed.load(Ordering::SeqCst) {
                return Err(Error::Io(std::io::Error::other("fault-injected failure")));
            }
            Ok(())
        }
    }

    impl FileBackend for ArmedFailHandle {
        fn len(&self) -> Result<u64> {
            self.inner.len()
        }
        fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
            self.inner.read_exact_at(buf, offset)
        }
        fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
            self.fail_if_armed()?;
            self.inner.write_all_at(buf, offset)
        }
        fn set_len(&self, new_len: u64) -> Result<()> {
            self.fail_if_armed()?;
            self.inner.set_len(new_len)
        }
        fn sync_data(&self, mode: crate::platform::SyncMode) -> Result<()> {
            self.fail_if_armed()?;
            self.inner.sync_data(mode)
        }
        fn sync_all(&self) -> Result<()> {
            self.inner.sync_all()
        }
    }

    /// Issue #2 at the txn layer: a `WriteTxn` whose inline
    /// auto-checkpoint fails AFTER the WAL durability step must still
    /// have `commit()` return Ok, so `WriteTxn::Drop` never rewinds the
    /// page-0 header snapshot over the already-committed transaction.
    /// The committed page must then survive a reopen, and same-process
    /// reads must agree with post-recovery reads.
    #[test]
    fn write_txn_commit_atomic_against_inline_checkpoint_failure() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("txn_atomic.obj");
        let wal_path = crate::pager::wal_path_for(&path);

        let armed_main = Arc::new(AtomicBool::new(false));
        let main = ArmedFailHandle {
            inner: FileHandle::open_or_create(&path).expect("main"),
            armed: Arc::clone(&armed_main),
        };
        let wal = ArmedFailHandle {
            inner: FileHandle::open_or_create(&wal_path).expect("wal"),
            armed: Arc::new(AtomicBool::new(false)),
        };
        // threshold = 1 -> every non-empty commit auto-checkpoints.
        let cfg = Config::default().with_checkpoint_threshold(1);
        let pager = Pager::<ArmedFailHandle>::open_with_backends(
            main,
            wal,
            wal_path,
            cfg,
            std::sync::Arc::new(crate::platform::OsEntropy),
        )
        .expect("open with backends");
        let env = TxnEnv::new(pager, None);

        // Baseline txn (unarmed): commit A; its auto-checkpoint succeeds.
        let a = {
            let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("begin a");
            let a = tx.alloc_page().expect("alloc a");
            let mut a_page = Page::zeroed();
            a_page.as_bytes_mut()[0] = 0xAA;
            tx.write_page(a, &a_page).expect("write a");
            tx.commit().expect("commit a");
            a
        };

        // Arm, then commit B. commit_inner makes B durable in the WAL;
        // the inline auto-checkpoint then fails on the main file.
        // WriteTxn::commit MUST return Ok (pre-fix it returned Err and
        // Drop rewound the header).
        armed_main.store(true, Ordering::SeqCst);
        let b = {
            let tx = WriteTxn::begin(&env, Duration::from_millis(50)).expect("begin b");
            let b = tx.alloc_page().expect("alloc b");
            let mut b_page = Page::zeroed();
            b_page.as_bytes_mut()[0] = 0xBB;
            tx.write_page(b, &b_page).expect("write b");
            tx.commit()
                .expect("WriteTxn::commit must succeed despite a failed auto-checkpoint");
            b
        };
        armed_main.store(false, Ordering::SeqCst);

        // Same-process snapshot read sees committed B.
        let b_marker = {
            let rx = ReadTxn::begin(&env).expect("read");
            rx.read_page(b).expect("same-process read b").as_bytes()[0]
        };
        assert_eq!(b_marker, 0xBB, "committed B visible same-process");

        drop(env);

        // Reopen clean: recovery replays the WAL; B must survive and
        // agree with the same-process read.
        let mut p2 = Pager::open(&path, Config::default()).expect("reopen");
        let b_reopened = p2.read_page(b).expect("recovered b").as_bytes()[0];
        assert_eq!(b_reopened, b_marker, "B agrees same-process vs post-recovery");
        assert_eq!(b_reopened, 0xBB, "committed txn survived reopen");
        let a_reopened = p2.read_page(a).expect("recovered a").as_bytes()[0];
        assert_eq!(a_reopened, 0xAA, "baseline A survived reopen");
    }
}
