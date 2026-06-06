//! Cross-process byte-range file locking.
//!
//! POSIX uses OFD `fcntl` locks (`F_OFD_SETLK` /
//! `F_OFD_SETLKW`) — kernel-tracked per-fd, fork-safe, automatically
//! released on process exit.
//!
//! Locks anchor against a dedicated `<db>.obj-lock` sidecar file
//! created by `Db::open` next to the main database (mirroring the
//! existing `<db>.obj-wal` sidecar convention). Using a sidecar
//! decouples the lock-byte range from any region the pager may
//! read or write, so the lock byte offsets can be the same on
//! every platform and need not be placed past EOF:
//!
//! - [`WRITER_LOCK_OFFSET`] = 96 (exclusive, 1 byte).
//! - [`READER_LOCK_RANGE_OFFSET`] = 97..128 (shared, 31 slots).
//!
//! The lock state lives in the OS kernel's per-fd lock table — the
//! bytes on disk are never read or written by obj.
//!
//! # `unsafe` policy
//!
//! `rustix::fs::fcntl_lock` does whole-file locking with `F_SETLK*`,
//! not OFD locks. We therefore call `libc::fcntl` directly with the
//! OFD command IDs. Every `unsafe` block carries a
//! `// SAFETY:` comment.

// allow: the platform syscall island — fcntl/flock/errno calls live here, the one
// sanctioned home for unsafe in obj-core; every block is documented via // SAFETY:.
#![allow(unsafe_code)]

use std::os::raw::c_int;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::{Error, LockKind, Result};
use crate::platform::FileHandle;

/// Byte offset of the `WRITER_LOCK` (1 byte, exclusive) inside the
/// `<db>.obj-lock` sidecar file.
///
/// The lock anchor lives at the same offset on every platform
/// because the sidecar file is never read or written by the pager
/// — its only purpose is to carry kernel-side lock metadata. On
/// POSIX this byte exists inside a 128-byte sidecar (see
/// `Db::open`'s `set_len(128)` on the sidecar). OFD locks are
/// advisory and would tolerate locks past EOF, but giving the byte
/// a physical existence is the conservative choice across kernels.
pub const WRITER_LOCK_OFFSET: u64 = 96;
/// Byte offset of the first reader-lock slot inside the
/// `<db>.obj-lock` sidecar. See [`WRITER_LOCK_OFFSET`] for the
/// sidecar rationale.
pub const READER_LOCK_RANGE_OFFSET: u64 = 97;
/// Length of the reader-lock byte range. 31 slots.
pub const READER_LOCK_RANGE_LEN: u64 = 31;

/// Initial backoff between busy-loop retries.
/// The retry loop is bounded by `deadline / INITIAL_BACKOFF` so an
/// exhausted budget surfaces deterministically.
const INITIAL_BACKOFF: Duration = Duration::from_millis(1);
/// Cap on the per-retry sleep so a long timeout stays responsive.
const MAX_BACKOFF: Duration = Duration::from_millis(100);

/// RAII guard for a held `WRITER_LOCK` byte. Dropping the guard
/// releases the OS-side lock. The guard is `!Send` only by virtue of
/// the file handle it does NOT own — the underlying lock is per-fd,
/// so as long as the fd survives, releasing from any thread is
/// sound.
#[derive(Debug)]
#[must_use = "WriterLock releases the OS-side lock when dropped"]
pub struct WriterLock {
    fd: c_int,
    released: bool,
}

impl WriterLock {
    /// Explicitly release the lock.  Equivalent to `Drop` but lets
    /// the caller observe a release error (the `Drop` impl silently
    /// swallows errors because panics from `Drop` are toxic).
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` on the unlikely event that the OS
    /// rejects the unlock syscall.
    pub fn release(mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.released = true;
        unlock_range(self.fd, WRITER_LOCK_OFFSET, 1)
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        if !self.released {
            let _ = unlock_range(self.fd, WRITER_LOCK_OFFSET, 1);
        }
    }
}

/// RAII guard for a held reader-lock byte. Dropping the guard
/// releases the OS-side lock.
#[derive(Debug)]
#[must_use = "ReaderLock releases the OS-side lock when dropped"]
pub struct ReaderLock {
    fd: c_int,
    slot: u64,
    released: bool,
}

impl ReaderLock {
    /// Byte offset of the reader-slot this guard holds.  Useful for
    /// diagnostics.
    #[must_use]
    pub fn slot(&self) -> u64 {
        self.slot
    }

    /// Explicitly release the lock.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` on the unlikely event that the OS
    /// rejects the unlock syscall.
    pub fn release(mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        self.released = true;
        unlock_range(self.fd, self.slot, 1)
    }
}

impl Drop for ReaderLock {
    fn drop(&mut self) {
        if !self.released {
            let _ = unlock_range(self.fd, self.slot, 1);
        }
    }
}

impl FileHandle {
    /// Try once, non-blocking, to acquire the `WRITER_LOCK`. Returns
    /// `Ok(Some(guard))` if the lock was acquired, `Ok(None)` if it
    /// is held by someone else, or `Err(Error::Io)` on syscall
    /// failure.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure other than
    /// "would-block / already-locked".
    pub fn try_lock_writer(&self) -> Result<Option<WriterLock>> {
        ensure_ofd_locks_supported()?;
        let fd = self.raw_fd();
        if try_lock_range(fd, WRITER_LOCK_OFFSET, 1, LockMode::Exclusive)? {
            Ok(Some(WriterLock {
                fd,
                released: false,
            }))
        } else {
            Ok(None)
        }
    }

    /// Acquire the `WRITER_LOCK`, retrying with bounded exponential
    /// backoff until either acquired or `timeout` elapses. Returns
    /// `Err(Error::Busy { kind: LockKind::Writer })` on timeout.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] with `LockKind::Writer` on timeout.
    /// - [`Error::Io`] on any non-"would-block" syscall failure.
    pub fn lock_writer(&self, timeout: Duration) -> Result<WriterLock> {
        ensure_ofd_locks_supported()?;
        let fd = self.raw_fd();
        retry_until_acquired(timeout, LockKind::Writer, || {
            try_lock_range(fd, WRITER_LOCK_OFFSET, 1, LockMode::Exclusive)
        })?;
        Ok(WriterLock {
            fd,
            released: false,
        })
    }

    /// Acquire any one of the 31 reader-lock slots in shared mode,
    /// retrying with bounded backoff until either acquired or
    /// `timeout` elapses.
    ///
    /// The slot is chosen with a per-process round-robin counter so
    /// concurrent readers in the same process do not all race for
    /// the same byte.  Shared locks compose, so falling on the same
    /// byte is not a correctness bug — just a hot-spot the spread
    /// avoids in practice.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] with `LockKind::Reader` on timeout (very
    ///   rare — shared locks rarely contend).
    /// - [`Error::Io`] on syscall failure.
    pub fn lock_reader(&self, timeout: Duration) -> Result<ReaderLock> {
        ensure_ofd_locks_supported()?;
        let fd = self.raw_fd();
        let start_slot = next_reader_slot();
        let mut last_err: Option<Error> = None;
        for offset in 0..READER_LOCK_RANGE_LEN {
            let slot = READER_LOCK_RANGE_OFFSET + ((start_slot + offset) % READER_LOCK_RANGE_LEN);
            match try_lock_range(fd, slot, 1, LockMode::Shared) {
                Ok(true) => {
                    return Ok(ReaderLock {
                        fd,
                        slot,
                        released: false,
                    });
                }
                Ok(false) => {}
                Err(e) => last_err = Some(e),
            }
        }
        if let Some(err) = last_err {
            return Err(err);
        }
        let slot = READER_LOCK_RANGE_OFFSET + start_slot;
        retry_until_acquired(timeout, LockKind::Reader, || {
            try_lock_range(fd, slot, 1, LockMode::Shared)
        })?;
        Ok(ReaderLock {
            fd,
            slot,
            released: false,
        })
    }

    /// Raw fd accessor (POSIX). Internal to the platform layer.
    #[cfg(unix)]
    fn raw_fd(&self) -> c_int {
        use std::os::unix::io::AsRawFd;
        self.file_ref().as_raw_fd()
    }
}

/// Per-process round-robin counter so threads in the same process
/// pick different reader-slot bytes by default.  Wraps at
/// `READER_LOCK_RANGE_LEN` — the modulo arithmetic in `lock_reader`
/// handles the actual selection.
static READER_ROUND_ROBIN: AtomicU64 = AtomicU64::new(0);

fn next_reader_slot() -> u64 {
    READER_ROUND_ROBIN.fetch_add(1, Ordering::Relaxed) % READER_LOCK_RANGE_LEN
}

#[derive(Debug, Clone, Copy)]
enum LockMode {
    Exclusive,
    Shared,
}

/// Bounded retry harness shared by `lock_writer` / `lock_reader`.
/// The loop's upper bound is
/// `deadline.elapsed() < timeout`; once `Instant::now() >= deadline`
/// the function returns `Err(Error::Busy)`.
fn retry_until_acquired<F>(timeout: Duration, kind: LockKind, mut once: F) -> Result<()>
where
    F: FnMut() -> Result<bool>,
{
    let start = Instant::now();
    let mut backoff = INITIAL_BACKOFF;
    let timeout_millis = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
    let max_iters: u64 = timeout_millis.saturating_add(2);
    let mut iters: u64 = 0;
    loop {
        iters = iters.saturating_add(1);
        if iters > max_iters.saturating_add(64) {
            return Err(Error::Busy { kind });
        }
        if once()? {
            return Ok(());
        }
        if start.elapsed() >= timeout {
            return Err(Error::Busy { kind });
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
}

/// Build a POSIX `struct flock` for the given byte range.  The
/// numeric type of `l_type` / `l_whence` differs per platform
/// (`i16` on Linux/macOS, `c_short` typedef elsewhere); we use
/// `try_from` rather than `as` so clippy's pedantic
/// `cast_possible_truncation` lint stays clean.
#[cfg(unix)]
fn build_flock(l_type: i32, offset: u64, len: u64) -> Result<libc::flock> {
    let l_type_short =
        libc::c_short::try_from(l_type).map_err(|_| Error::InvalidArgument("lock l_type"))?;
    let l_whence_short = libc::c_short::try_from(libc::SEEK_SET)
        .map_err(|_| Error::InvalidArgument("lock l_whence"))?;
    Ok(libc::flock {
        l_type: l_type_short,
        l_whence: l_whence_short,
        l_start: offset_to_off_t(offset)?,
        l_len: offset_to_off_t(len)?,
        l_pid: 0,
        #[cfg(target_os = "freebsd")]
        l_sysid: 0,
    })
}

#[cfg(unix)]
fn try_lock_range(fd: c_int, offset: u64, len: u64, mode: LockMode) -> Result<bool> {
    // allow: .into() is a no-op where F_WRLCK/F_RDLCK are already i32, but required on
    // targets where the libc constants are a narrower type; keep for cross-platform builds.
    #[allow(clippy::useless_conversion)]
    let l_type: i32 = match mode {
        LockMode::Exclusive => libc::F_WRLCK.into(),
        LockMode::Shared => libc::F_RDLCK.into(),
    };
    let flock = build_flock(l_type, offset, len)?;
    // SAFETY: fd is a live, owned fd from FileHandle::raw_fd valid for this call; flock is a
    // fully-initialized libc::flock on the stack and &raw const flock points to it for the
    // syscall's duration; ofd_setlk_cmd is reachable only after the OFD-support gate passed.
    let ret = unsafe { libc::fcntl(fd, ofd_setlk_cmd(), &raw const flock) };
    if ret == 0 {
        return Ok(true);
    }
    // SAFETY: libc_errno returns libc's valid, non-null per-thread errno location; deref it
    // immediately after the failed fcntl, before any other libc call can overwrite it.
    let errno = unsafe { *libc_errno() };
    if errno == libc::EAGAIN || errno == libc::EACCES {
        return Ok(false);
    }
    Err(Error::Io(std::io::Error::from_raw_os_error(errno)))
}

#[cfg(unix)]
fn unlock_range(fd: c_int, offset: u64, len: u64) -> Result<()> {
    // allow: .into() is a no-op where F_UNLCK is already i32, but required on targets
    // where the libc constant is a narrower type; keep for cross-platform builds.
    #[allow(clippy::useless_conversion)]
    let flock = build_flock(libc::F_UNLCK.into(), offset, len)?;
    // SAFETY: fd is a live, owned fd from FileHandle::raw_fd valid for this call; flock is a
    // fully-initialized libc::flock on the stack and &raw const flock points to it for the
    // syscall's duration; ofd_setlk_cmd is reachable only after the OFD-support gate passed.
    let ret = unsafe { libc::fcntl(fd, ofd_setlk_cmd(), &raw const flock) };
    if ret == 0 {
        return Ok(());
    }
    // SAFETY: libc_errno returns libc's valid, non-null per-thread errno location; deref it
    // immediately after the failed fcntl, before any other libc call can overwrite it.
    let errno = unsafe { *libc_errno() };
    Err(Error::Io(std::io::Error::from_raw_os_error(errno)))
}

/// `true` iff the build target provides OFD (open-file-description)
/// `fcntl` locks — `F_OFD_SETLK` / `F_OFD_SETLKW`. These are the
/// only POSIX lock primitive obj's concurrency model can rely on:
/// they are tracked PER-fd, so two `Db` handles to the same file in
/// one process correctly exclude each other, and they are released
/// on the owning fd's close rather than coalescing across the whole
/// process.
///
/// Classic POSIX `F_SETLK` locks are tracked PER-PROCESS: a second
/// `Db` handle in the same process would silently share (and on the
/// first handle's close, silently drop) the first handle's lock,
/// breaking the single-writer invariant without any error. Rather
/// than fall back to that unsound primitive, obj refuses
/// to lock on a target without OFD locks — see
/// [`ensure_ofd_locks_supported`].
///
/// # Supported-target matrix
///
/// | Target | OFD locks | obj locking |
/// |---|---|---|
/// | Linux ≥ 3.15 / Android | yes (`F_OFD_SETLK` = 37) | supported |
/// | macOS ≥ 10.14 / iOS (Apple) | yes (`F_OFD_SETLK` = 90) | supported |
/// | FreeBSD / other POSIX | not exported by `libc` | **refused at open** |
///
/// This constant is only meaningful on `unix`.
#[cfg(unix)]
const TARGET_HAS_OFD_LOCKS: bool = cfg!(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
));

/// Hard, documented gate for the non-OFD targets. Called at
/// the head of every lock-acquisition entry point so the failure
/// surfaces at `Db::open` time rather than as silent, per-process
/// lock coalescing later.
///
/// On Linux/macOS this is a compile-time-`true` check the optimiser
/// erases — the lock fast path is byte-for-byte unchanged. On a
/// target without OFD locks it returns
/// [`std::io::ErrorKind::Unsupported`] wrapped in [`Error::Io`].
///
/// # Errors
///
/// Returns [`Error::Io`] with `ErrorKind::Unsupported` when the
/// build target lacks OFD `fcntl` locks.
#[cfg(unix)]
fn ensure_ofd_locks_supported() -> Result<()> {
    if TARGET_HAS_OFD_LOCKS {
        return Ok(());
    }
    Err(Error::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "obj requires OFD (open-file-description) fcntl locks, which \
         this target does not provide; classic POSIX F_SETLK locks are \
         per-process and would silently break same-process multi-handle \
         exclusion (see obj-core platform::lock supported-target matrix)",
    )))
}

/// Resolve the `F_OFD_SETLK` command id.  Linux and Apple ship it as
/// a numeric constant in `<fcntl.h>` (`37` on Linux, `90` on macOS
/// 10.14+).  We hard-code the numeric values here because `libc`
/// does not export them on every target.
///
/// This function is only ever reached after
/// [`ensure_ofd_locks_supported`] has returned `Ok` (every public
/// lock entry point gates on it first), so the non-OFD targets never
/// execute the fallback arm below. The `unreachable_target` arm
/// exists solely to keep the function total across `cfg` targets; it
/// returns a deliberately invalid command id (`-1`) so that if a
/// future refactor were ever to call this without the guard, the
/// `fcntl` would fail with `EINVAL` rather than silently install a
/// per-process lock.
#[cfg(unix)]
fn ofd_setlk_cmd() -> c_int {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        37
    }
    #[cfg(target_vendor = "apple")]
    {
        90
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple",)))]
    {
        -1
    }
}

#[cfg(unix)]
fn offset_to_off_t(v: u64) -> Result<libc::off_t> {
    libc::off_t::try_from(v).map_err(|_| Error::InvalidArgument("lock offset overflow"))
}

#[cfg(unix)]
fn libc_errno() -> *mut c_int {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    // SAFETY: glibc/bionic `__errno_location` returns the valid, non-null
    // per-thread errno location pointer; calling it has no preconditions.
    unsafe {
        libc::__errno_location()
    }
    #[cfg(target_vendor = "apple")]
    // SAFETY: libc::__error is Apple libc's documented accessor returning the valid,
    // non-null per-thread errno location pointer; calling it has no preconditions.
    unsafe {
        libc::__error()
    }
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    // SAFETY: FreeBSD/DragonFly libc's `__error` returns the valid, non-null
    // per-thread errno location pointer; calling it has no preconditions.
    unsafe {
        libc::__error()
    }
    #[cfg(any(target_os = "openbsd", target_os = "netbsd"))]
    // SAFETY: OpenBSD/NetBSD libc's `__errno` returns the valid, non-null
    // per-thread errno location pointer; calling it has no preconditions.
    unsafe {
        libc::__errno()
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "netbsd",
    )))]
    // SAFETY: fallback to the POSIX-conventional `__errno_location`, which
    // returns the valid, non-null per-thread errno location pointer with no
    // preconditions on platforms not covered by the cfg branches above.
    unsafe {
        libc::__errno_location()
    }
}

impl FileHandle {
    /// Borrow the inner `std::fs::File`. Internal to the platform
    /// layer; only the lock submodule needs the raw fd / handle.
    fn file_ref(&self) -> &std::fs::File {
        &self.file
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::*;
    #[cfg(unix)]
    use tempfile::TempDir;

    /// Create a file that's at least 4 KiB so the lock byte
    /// offsets at 96 / 97..128 are inside the file. Unix-only
    /// because every caller is gated on `cfg(unix)`.
    #[cfg(unix)]
    fn fresh_handle(dir: &TempDir, name: &str) -> FileHandle {
        let path = dir.path().join(name);
        let h = FileHandle::open_or_create(&path).expect("open");
        h.set_len(4096).expect("extend");
        h
    }

    #[test]
    #[cfg(unix)]
    fn writer_lock_excludes_writers() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("lock.obj");
        FileHandle::open_or_create(&path)
            .expect("init")
            .set_len(4096)
            .expect("len");

        let h1 = FileHandle::open_or_create(&path).expect("h1");
        let h2 = FileHandle::open_or_create(&path).expect("h2");

        let guard = h1
            .try_lock_writer()
            .expect("try lock h1")
            .expect("must acquire");
        let none = h2.try_lock_writer().expect("try lock h2");
        assert!(none.is_none(), "second writer lock must be refused");
        drop(guard);
        let _g2 = h2
            .try_lock_writer()
            .expect("try lock h2 again")
            .expect("now acquires");
    }

    #[test]
    #[cfg(unix)]
    fn writer_busy_timeout_returns_err_busy() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");

        let h1 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h1");
        let h2 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h2");
        let _g1 = h1
            .try_lock_writer()
            .expect("h1 lock")
            .expect("h1 must acquire");
        let start = std::time::Instant::now();
        let err = h2
            .lock_writer(Duration::from_millis(50))
            .expect_err("must time out");
        let elapsed = start.elapsed();
        assert!(matches!(
            err,
            Error::Busy {
                kind: LockKind::Writer
            }
        ));
        assert!(
            elapsed >= Duration::from_millis(45),
            "must wait at least the timeout (~50 ms); got {elapsed:?}",
        );
    }

    #[test]
    #[cfg(unix)]
    fn many_readers_can_coexist() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");
        let h1 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h1");
        let h2 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h2");
        let h3 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h3");
        let g1 = h1.lock_reader(Duration::from_millis(50)).expect("r1");
        let g2 = h2.lock_reader(Duration::from_millis(50)).expect("r2");
        let g3 = h3.lock_reader(Duration::from_millis(50)).expect("r3");
        drop((g1, g2, g3));
    }

    #[test]
    #[cfg(unix)]
    fn reader_and_writer_dont_collide_on_separate_anchors() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");
        let h1 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h1");
        let h2 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h2");
        let _wg = h1.lock_writer(Duration::from_millis(50)).expect("writer");
        let _rg = h2
            .lock_reader(Duration::from_millis(50))
            .expect("reader must not collide");
    }

    #[test]
    #[cfg(unix)]
    fn explicit_release_returns_ok() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");
        let h = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h");
        let g = h.lock_writer(Duration::from_millis(50)).expect("lock");
        g.release().expect("release ok");
        let _g2 = h.lock_writer(Duration::from_millis(50)).expect("relock");
    }

    #[test]
    #[cfg(unix)]
    fn lock_methods_compile_when_dropped() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");
        let h = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h");
        let g = h.lock_reader(Duration::from_millis(10)).expect("rlock");
        drop(g);
    }

    /// The OFD-capability gate must agree with the build target.
    /// On any target obj actually supports (Linux/Android/Apple) the
    /// gate is `Ok` and the per-fd locking primitive is OFD. The
    /// classic-`F_SETLK` fallback that silently broke same-process
    /// multi-fd exclusion on FreeBSD / unknown POSIX is gone: those
    /// targets now hard-error at `ensure_ofd_locks_supported`.
    #[test]
    #[cfg(unix)]
    fn ofd_capability_gate_matches_target() {
        assert_eq!(
            TARGET_HAS_OFD_LOCKS,
            cfg!(any(
                target_os = "linux",
                target_os = "android",
                target_vendor = "apple",
            )),
            "OFD capability constant must track the supported-target set",
        );
        let gate = ensure_ofd_locks_supported();
        if TARGET_HAS_OFD_LOCKS {
            gate.expect("supported targets must pass the gate");
        } else {
            match gate {
                Err(Error::Io(e)) => {
                    assert_eq!(e.kind(), std::io::ErrorKind::Unsupported);
                }
                other => panic!("expected Io(Unsupported), got {other:?}"),
            }
        }
    }

    /// Regression: two `FileHandle`s to the SAME file in the SAME
    /// process must exclude each other for the writer lock. This is
    /// exactly the invariant the classic per-process `F_SETLK`
    /// fallback silently broke (a second open in-process would
    /// "succeed" because the lock coalesces per process). OFD locks
    /// are per-fd, so the second handle is correctly refused.
    #[test]
    #[cfg(unix)]
    fn same_process_multi_fd_writer_exclusion_holds() {
        let dir = TempDir::new().expect("tmp");
        let _h0 = fresh_handle(&dir, "lock.obj");
        let h1 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h1");
        let h2 = FileHandle::open_or_create(dir.path().join("lock.obj")).expect("h2");
        let g1 = h1.try_lock_writer().expect("h1 try").expect("h1 acquires");
        assert!(
            h2.try_lock_writer().expect("h2 try").is_none(),
            "per-fd OFD lock must refuse a second in-process handle; a \
             per-process F_SETLK fallback would wrongly grant this",
        );
        drop(g1);
    }
}
