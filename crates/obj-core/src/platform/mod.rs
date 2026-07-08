//! Platform layer (L0).
//!
//! This module owns the file-system primitives the pager and WAL
//! build on: opening a database file, positioned reads and writes at
//! fixed page boundaries, length queries, truncation, removal, and the
//! durability primitive [`FileHandle::sync_data`].
//!
//! # `unsafe` policy
//!
//! `unsafe` is confined to this submodule (and to
//! `libobj`). All positioned-I/O and durability calls go through the
//! `rustix` crate, which provides audited safe wrappers. The
//! cross-process locking submodule [`lock`] reaches for `libc::fcntl`
//! directly because `rustix` does not expose POSIX
//! OFD-lock variants; every `unsafe` block in that submodule
//! carries a `// SAFETY:` comment. This `mod.rs` itself
//! contains no `unsafe` blocks and is `#![deny(unsafe_code)]`; the
//! lint is scoped to the file rather than the module tree so the
//! `lock` submodule can re-introduce its (audited) `unsafe`
//! blocks.

#![deny(unsafe_code)]

pub mod env;

#[cfg(any(test, feature = "fault-injection"))]
pub mod fault;

pub mod lock;

pub use crate::platform::env::{Clock, Entropy, OsEntropy, SeededEntropy, SimClock, SystemClock};
pub use crate::platform::lock::{ReaderLock, WriterLock};

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::FileExt as _;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;
/// Owner read+write only (`rw-------`). Applied to freshly
/// created database and backup files on unix targets.
#[cfg(unix)]
const OWNER_ONLY_MODE: u32 = 0o600;

use crate::error::{Error, Result};

/// File-backend abstraction the pager and WAL build on.
///
/// `FileBackend` is the common subset of [`FileHandle`] operations
/// that fault-injection harnesses and the production type both expose.
/// Production code never holds `dyn FileBackend`; both
/// [`crate::pager::Pager`] and [`crate::wal::Wal`] are generic over
/// `F: FileBackend` so the dispatch stays monomorphised.
///
/// New methods added to this trait MUST mirror an existing
/// [`FileHandle`] method exactly. Adding a method that does not exist
/// on the production type would let the harness perform syscalls
/// production code cannot — a forbidden divergence (the harness
/// must be a strict superset of legal behaviour, never a separate
/// kingdom).
pub trait FileBackend: Sized {
    /// Length of the file in bytes. See [`FileHandle::len`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn len(&self) -> Result<u64>;

    /// `true` iff the file has zero length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Positioned read. See [`FileHandle::read_exact_at`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure or harness-injected
    /// short read.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()>;

    /// Positioned write. See [`FileHandle::write_all_at`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()>;

    /// Truncate or extend the file. See [`FileHandle::set_len`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn set_len(&self, new_len: u64) -> Result<()>;

    /// See [`FileHandle::sync_data`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn sync_data(&self, mode: SyncMode) -> Result<()>;

    /// See [`FileHandle::sync_all`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    fn sync_all(&self) -> Result<()>;
}

/// Durability mode for [`FileHandle::sync_data`].
///
/// `SyncMode` is the user-visible knob that selects the cross-platform
/// fsync primitive `obj` calls after a WAL commit.
///
/// The default is [`SyncMode::Full`]: a `commit` that returns
/// `Ok(())` is durable across a system-wide power loss. `Normal` is
/// the throughput-tuned middle ground; `Off` skips the syscall and is
/// only safe for tests and benchmarks.
///
/// A three-state enum is far cheaper to audit
/// than three `bool` knobs, and the variants are exhaustive at every
/// `match`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum SyncMode {
    /// Strongest durability. Survives system-wide power loss.
    ///
    /// Maps to `fcntl(F_FULLFSYNC)` on macOS (forces the drive cache
    /// to flush) and `fdatasync` on
    /// Linux / BSDs. macOS's plain `fsync` is **not** sufficient
    /// here — it does not flush the drive cache; `F_FULLFSYNC` does.
    /// This is the standard wisdom for safety-critical macOS storage.
    #[default]
    Full,

    /// Process-crash and kernel-panic durability; may lose data on a
    /// sudden power loss if the drive's write cache has not been
    /// flushed by the time the OS acknowledges the call.
    ///
    /// Maps to `fsync` on Unix.
    Normal,

    /// No durability call. The OS may write the data eventually, but
    /// `obj` does not ask it to. Use only for tests and benchmarks
    /// where data loss is acceptable.
    Off,
}

/// A handle to a database file capable of positioned reads and writes
/// at page boundaries.
///
/// `FileHandle` is intentionally minimal — it exposes only the
/// operations the pager (L1) and WAL (L2) need. Higher layers must
/// never reach past it into `std::fs` directly; routing every syscall
/// through this type is how the project keeps `unsafe` confined to the
/// platform layer.
#[derive(Debug)]
pub struct FileHandle {
    file: File,
}

impl FileHandle {
    /// Open `path` for read-write access, creating it if it does not
    /// exist. The new file is empty; the caller is responsible for
    /// writing the file header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened or created
    /// (permission denied, missing parent directory, etc.).
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        opts.mode(OWNER_ONLY_MODE);
        let file = opts.open(path)?;
        Ok(Self { file })
    }

    /// Open `path` for read-write access, failing if the file
    /// already exists (`O_CREAT | O_EXCL` on POSIX). Used by
    /// hot-backup to guarantee the
    /// destination is never overwritten.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file already exists, the parent
    /// directory does not exist, or any other syscall failure
    /// occurs.
    pub fn create_new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create_new(true);
        #[cfg(unix)]
        opts.mode(OWNER_ONLY_MODE);
        let file = opts.open(path)?;
        Ok(Self { file })
    }

    /// Length of the file in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the metadata syscall fails.
    pub fn len(&self) -> Result<u64> {
        let meta = self.file.metadata()?;
        Ok(meta.len())
    }

    /// `true` if the file is zero-length (i.e. just created).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the metadata syscall fails.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Positioned read. Fills `buf` from byte offset `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure or on short read
    /// (e.g. file shorter than `offset + buf.len()`).
    pub fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        read_exact_at_impl(&self.file, buf, offset).map_err(Error::from)
    }

    /// Positioned write. Writes `buf` to byte offset `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure or on short write.
    pub fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
        write_all_at_impl(&self.file, buf, offset).map_err(Error::from)
    }

    /// Truncate or extend the file to `new_len` bytes.
    ///
    /// Used by the pager when the freelist is exhausted and a fresh
    /// page must be appended.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn set_len(&self, new_len: u64) -> Result<()> {
        self.file.set_len(new_len).map_err(Error::from)
    }

    /// Force file contents and metadata to disk. Used at close.
    ///
    /// The underlying call returns
    /// `io::Result<()>` and is propagated explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn sync_all(&self) -> Result<()> {
        self.file.sync_all().map_err(Error::from)
    }

    /// Force file data (and on `Full`, the drive cache) to persistent
    /// storage according to `mode`. See [`SyncMode`] for the exact
    /// per-variant durability promise.
    ///
    /// On `SyncMode::Off` this call is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn sync_data(&self, mode: SyncMode) -> Result<()> {
        match mode {
            SyncMode::Off => Ok(()),
            SyncMode::Normal => sync_data_normal(&self.file),
            SyncMode::Full => sync_data_full(&self.file),
        }
    }
}

impl FileBackend for FileHandle {
    fn len(&self) -> Result<u64> {
        FileHandle::len(self)
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        FileHandle::read_exact_at(self, buf, offset)
    }
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
        FileHandle::write_all_at(self, buf, offset)
    }
    fn set_len(&self, new_len: u64) -> Result<()> {
        FileHandle::set_len(self, new_len)
    }
    fn sync_data(&self, mode: SyncMode) -> Result<()> {
        FileHandle::sync_data(self, mode)
    }
    fn sync_all(&self) -> Result<()> {
        FileHandle::sync_all(self)
    }
}

#[cfg(unix)]
fn read_exact_at_impl(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    file.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn write_all_at_impl(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    file.write_all_at(buf, offset)
}

/// `Normal` durability — `fsync` on Unix. Survives process / kernel
/// crash but may lose data on a
/// sudden power loss if the drive cache has not been flushed.
fn sync_data_normal(file: &File) -> Result<()> {
    file.sync_all().map_err(Error::from)
}

/// `Full` durability — flush the drive cache where the platform
/// distinguishes it from the OS cache. See [`SyncMode::Full`] for
/// the per-OS mapping.
#[cfg(target_vendor = "apple")]
fn sync_data_full(file: &File) -> Result<()> {
    rustix::fs::fcntl_fullfsync(file).map_err(|e| Error::Io(io::Error::from(e)))
}

/// `Full` durability on non-Apple Unix targets: `fdatasync(2)` is
/// sufficient (the on-disk data is flushed, metadata changes that do
/// not affect the data — like mtime — are not).
#[cfg(all(unix, not(target_vendor = "apple")))]
fn sync_data_full(file: &File) -> Result<()> {
    rustix::fs::fdatasync(file).map_err(|e| Error::Io(io::Error::from(e)))
}

/// Delete the file at `path` if it exists.
///
/// Used by `Pager::close()` to remove the WAL sidecar after a clean
/// shutdown. Missing-file is intentionally **not** an error; the
/// post-condition is "no file at `path`", and that is satisfied either
/// by deletion or by absence.
///
/// # Errors
///
/// Returns [`Error::Io`] on any failure other than `NotFound`.
pub fn remove_file_if_exists<P: AsRef<Path>>(path: P) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::Io(e)),
    }
}

impl From<io::ErrorKind> for Error {
    fn from(kind: io::ErrorKind) -> Self {
        Error::Io(io::Error::from(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::{FileHandle, SyncMode};
    use tempfile::TempDir;

    fn write_and_sync(mode: SyncMode) {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("sync.bin");
        let h = FileHandle::open_or_create(&path).expect("open");
        h.set_len(4096).expect("set_len");
        h.write_all_at(&[0xABu8; 4096], 0).expect("write");
        h.sync_data(mode).expect("sync_data must succeed");
    }

    #[test]
    fn sync_data_full_returns_ok() {
        write_and_sync(SyncMode::Full);
    }

    #[test]
    fn sync_data_normal_returns_ok() {
        write_and_sync(SyncMode::Normal);
    }

    #[test]
    fn sync_data_off_is_noop() {
        write_and_sync(SyncMode::Off);
    }

    #[test]
    fn default_is_full() {
        assert_eq!(SyncMode::default(), SyncMode::Full);
    }
}
