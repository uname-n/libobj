//! Pager (L1) — fixed-size page allocator, freelist, bounded LRU
//! cache, and the WAL-aware write path.
//!
//! The pager is the lowest-level component that handles named pages:
//! [`PageId`]s are non-zero `u64`s, page bodies are exactly
//! [`PAGE_SIZE`] bytes. The pager hides the difference between a
//! file-backed database and an in-memory one (`Pager::memory`) so
//! higher layers see a single uniform interface.

#![forbid(unsafe_code)]

pub mod cache;
pub mod checksum;
pub mod freelist;
pub mod header;
pub mod page;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::pager::cache::{Cache, Evicted};
use crate::pager::checksum::{
    page_trailer_valid, page_trailer_valid_v1, write_page_trailer, write_page_trailer_v1,
};
use crate::pager::freelist::{
    decode as decode_freelist_page, encode as encode_freelist_page, FreeListPage,
};
use crate::pager::header::{
    decode_header, encode_header, FileHeader, FEATURE_FLAG_COMPRESSION, FEATURE_FLAG_ENCRYPTION,
};
use crate::pager::page::{Page, PageId, ENCRYPTION_OVERHEAD, PAGE_SIZE, PAGE_TRAILER_SIZE};
use crate::platform::env::{Entropy, OsEntropy};
use crate::platform::{FileBackend, FileHandle, SyncMode};
use crate::wal::{Lsn, Wal, WalConfig};

pub use crate::pager::page::PAGE_SIZE as PAGER_PAGE_SIZE;

/// A borrowed view of a cached or WAL-staged page.
///
/// `PageRef` is the return type of [`Pager::read_page`]. It carries a
/// shared reference to the page's bytes that lives no longer than
/// the immutable borrow of the pager that produced it — so the
/// borrow checker forbids holding a `PageRef` across any mutating
/// call on the same pager (write, commit, checkpoint, alloc, free).
///
/// # Allocation contract
///
/// Construction of a `PageRef` performs **no heap allocation** on
/// cache hits or WAL-overlay hits: the bytes are already resident in
/// memory and `PageRef` is a thin wrapper around a `&[u8; PAGE_SIZE]`.
/// On a cache miss the read-through path issues one `pread` and
/// inserts the page into the cache, after which the `PageRef`
/// borrows that cache frame.
///
/// # Examples
///
/// ```no_run
/// # use obj_core::pager::{Config, Pager};
/// # use obj_core::pager::page::PageId;
/// # fn id(n: u64) -> PageId { PageId::new(n).unwrap() }
/// # let mut p = Pager::memory(Config::default()).unwrap();
/// # let a = p.alloc_page().unwrap();
/// # let _ = p.commit().unwrap();
/// let view = p.read_page(a)?;
/// let header_byte = view.as_bytes()[0];
/// // `view` must be dropped (or moved) before the next p.write_page(...).
/// # Ok::<(), obj_core::Error>(())
/// ```
#[derive(Debug)]
pub struct PageRef<'a> {
    page_id: PageId,
    bytes: &'a Page,
}

impl<'a> PageRef<'a> {
    fn new(page_id: PageId, bytes: &'a Page) -> Self {
        Self { page_id, bytes }
    }

    /// The page id this view was read for.
    #[must_use]
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// The page's raw bytes. Includes the per-page CRC32C trailer
    /// in its last [`PAGE_TRAILER_SIZE`]
    /// bytes; callers that want only the payload should slice off
    /// the trailer themselves.
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8; PAGE_SIZE] {
        self.bytes.as_bytes()
    }

    /// Clone the underlying page into a fresh owned [`Page`]. This
    /// allocates; only call it when an owned buffer is genuinely
    /// required (e.g. to mutate before a subsequent `write_page`).
    #[must_use]
    pub fn to_owned_page(&self) -> Page {
        self.bytes.clone()
    }
}

/// Snapshot of the page-0 header fields that the catalog +
/// freelist code mutates directly (not through the WAL).
///
/// Returned by [`Pager::header_snapshot`] at txn-begin; passed back
/// to [`Pager::restore_header_snapshot`] on rollback so the on-disk
/// header is rewound alongside the WAL pending buffer. These fields
/// are written direct-to-disk for performance; the transaction
/// layer compensates by snapshotting them whenever a `WriteTxn`
/// might roll back.
#[derive(Debug, Clone)]
pub struct HeaderSnapshot {
    /// Catalog B-tree root page id captured at snapshot time.
    pub root_catalog: u64,
    /// Freelist head page id captured at snapshot time.
    pub freelist_head: u64,
    /// File page count captured at snapshot time.  Used so an
    /// alloc-then-rollback does not leak the appended page.
    pub page_count: u64,
    /// WAL "committed view" snapshot.  See
    /// [`Pager::header_snapshot`] for the rationale — `free_page`
    /// removes per-page entries from the live view, and the
    /// rollback path needs to put them back.
    ///
    /// Pages are held behind `Arc` so cloning this map (in the
    /// rollback snapshot path) is a set of refcount bumps rather than
    /// per-page 4 KiB memcpys, matching the live committed view.
    pub view: HashMap<PageId, Arc<Page>>,
}

/// Opaque identifier for a single live MVCC reader snapshot.
///
/// The id is monotonic per-pager; the value is otherwise meaningless
/// to callers — it only exists so the (private) `SnapshotPin` RAII
/// guard can deregister the right entry from the pager's live-
/// snapshots map when it drops.
///
/// `SnapshotId` is a `#[repr(transparent)]` newtype over `u64`. The
/// serde encoding is `#[serde(transparent)]` so
/// the bytes are identical to the bare `u64`; today `SnapshotId` is
/// purely in-memory, but the transparent encoding preserves wire
/// compatibility for any future diagnostics record that names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SnapshotId(u64);

impl SnapshotId {
    /// Construct a [`SnapshotId`] from a raw `u64`. Total function —
    /// any `u64` (including `0`) is a valid snapshot id.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw id. Exposed for diagnostics.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// RAII handle that keeps a [`ReaderSnapshot`]'s pinned LSN in the
/// pager's live-snapshots map until the snapshot is dropped.
///
/// On drop, the pin removes itself from the map so checkpoint can
/// proceed.  A poisoned mutex on drop is silently ignored (drop
/// must never panic).
#[derive(Debug)]
struct SnapshotPin {
    id: SnapshotId,
    map: Arc<Mutex<HashMap<SnapshotId, Lsn>>>,
}

impl Drop for SnapshotPin {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.map.lock() {
            guard.remove(&self.id);
        }
    }
}

/// Owning handle to a page returned by [`ReaderSnapshot::read_page`].
///
/// `PageHandle` avoids a 4 KiB body clone on the hot path. A
/// point read descends catalog + primary and an index lookup descends
/// two trees, so each op would otherwise pay 3-5 such 4 KiB copies:
///
/// - [`PageHandle::Shared`] — frozen-view hit. Holds an
///   `Arc<Page>` cloned from the snapshot's view (a refcount bump,
///   **no** body copy). Sound because committed pages are immutable:
///   a new version is a fresh `Arc` under the same `PageId`, never an
///   in-place mutation of a shared `Arc` (see `frozen_view`).
/// - [`PageHandle::Owned`] — disk / cache miss. Holds the freshly
///   read, checksum-verified `Page` produced by the existing
///   `read_main_file_page` / `read_cache_or_main` (`read_through`)
///   path; integrity behaviour is unchanged.
///
/// Concrete enum, not `dyn`. Both arms are a single pointer
/// (`Arc<Page>` and `Page`'s `Box<[u8; PAGE_SIZE]>`), so the variants
/// are balanced and `clippy::large_enum_variant` does not fire.
#[derive(Debug, Clone)]
pub enum PageHandle {
    /// Frozen-view hit — shares the snapshot's `Arc<Page>` with no
    /// 4 KiB body clone.
    Shared(Arc<Page>),
    /// Disk / cache miss — owns the checksum-verified page bytes.
    Owned(Page),
}

impl PageHandle {
    /// Borrow the page's raw bytes for decoding, without copying.
    ///
    /// The match over both arms is total — no `Result`, no
    /// `unwrap`/`expect`: every `PageHandle` always holds a
    /// readable page in exactly one of its two arms. Callers that only
    /// need to decode (e.g. `BTree::get_via_snapshot`,
    /// `Catalog::lookup_via_snapshot`) should keep the `PageHandle`
    /// alive for the duration of the borrow and call this.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        match self {
            PageHandle::Shared(page) => page.as_bytes(),
            PageHandle::Owned(page) => page.as_bytes(),
        }
    }

    /// Consume the handle and produce an owned [`Page`].
    ///
    /// The `Shared` arm clones the body exactly once (only when an
    /// owned page is genuinely required, e.g. the public
    /// [`crate::ReadTxn::read_page`] whose pager lock guard cannot
    /// outlive the call); the `Owned` arm moves with no copy. This is
    /// the *only* place the hot-path frozen-view clone is paid, and
    /// only for callers that ask for ownership.
    #[must_use]
    pub fn into_page(self) -> Page {
        match self {
            PageHandle::Shared(page) => (*page).clone(),
            PageHandle::Owned(page) => page,
        }
    }
}

/// A reader-side MVCC snapshot of the database.
///
/// Captures the WAL end-LSN at construction and a clone of the
/// pager's in-memory committed view.  Reads through the snapshot
/// observe `main file ∪ WAL frames with LSN ≤ pinned_lsn`; pending
/// writes from a concurrent `WriteTxn` are NEVER visible.
///
/// `ReaderSnapshot` is `Send` so the user-facing read transaction
/// can run on any thread.  It is NOT `Clone` — each snapshot owns
/// its own pin and cloning would double-register the pin entry.
///
/// Generic over `F: FileBackend` (no `dyn`); the production
/// snapshot is `ReaderSnapshot<FileHandle>`.  Read calls take a
/// `&mut Pager<F>` parameter so the borrow checker can prove that
/// the cache mutation a cache-miss read performs cannot race the
/// snapshot's own view; in practice the `Db` wraps the pager in a
/// `Mutex` and the snapshot's pin is independent of
/// the mutex.
#[derive(Debug)]
pub struct ReaderSnapshot<F: FileBackend> {
    pinned_lsn: Lsn,
    /// Frozen WAL view captured at snapshot creation.  Lookups that
    /// hit this map return the body the writer had committed by
    /// `pinned_lsn`.  Misses fall through to `Pager::read_main_file_page`.
    ///
    /// Each page is an `Arc<Page>` so capturing this view at pin
    /// time (`reader_snapshot`) clones the map as refcount bumps, not
    /// per-page memcpys. MVCC isolation is unaffected: each snapshot
    /// owns its OWN cloned map, and committed pages are immutable —
    /// a new page version is a fresh `Arc` inserted under the same
    /// `PageId`, never an in-place mutation of a shared `Arc`.
    frozen_view: HashMap<PageId, Arc<Page>>,
    /// Optional snapshot of the WAL's committed page-0
    /// (file header) frame at pin time. `None` means "the on-disk
    /// header at offset 0 is authoritative" — no WAL-staged header
    /// update sits in the committed view. Read-side users of the
    /// snapshot do NOT consult this directly; the
    /// [`crate::backup`] module uses it to reconstruct the
    /// snapshot's view of the header bytes when materialising a
    /// hot backup.
    frozen_header: Option<Page>,
    /// Snapshot of the catalog B-tree root page-id at pin
    /// time. Captured from the pager's committed-header view; a
    /// concurrent writer that calls `set_root_catalog` will NOT
    /// mutate this value, so reader threads can pass their pinned
    /// root into a [`crate::Catalog`] opened against the snapshot.
    root_catalog: u64,
    /// Live-map registration; deregistered on drop.
    pin: SnapshotPin,
    _phantom: std::marker::PhantomData<fn() -> F>,
}

impl<F: FileBackend> ReaderSnapshot<F> {
    /// The LSN this snapshot pinned at construction.  Reads through
    /// the snapshot only observe WAL frames with `LSN <= pinned_lsn`.
    #[must_use]
    pub fn pinned_lsn(&self) -> Lsn {
        self.pinned_lsn
    }

    /// Iterate over every `(PageId, &Page)` pair in the snapshot's
    /// frozen WAL view — i.e. every WAL frame at `LSN <= pinned_lsn`
    /// at the moment the snapshot was created. Used by
    /// `Db::backup_to` to overlay the snapshot's view onto a
    /// freshly-copied main file.
    pub fn frozen_pages(&self) -> impl Iterator<Item = (PageId, &Page)> + '_ {
        self.frozen_view
            .iter()
            .map(|(id, page)| (*id, page.as_ref()))
    }

    /// The WAL-staged page-0 (file header) frame at the snapshot's
    /// pinned LSN, or `None` if no header frame sits in the
    /// committed view. Used by `Db::backup_to` to
    /// reconstruct the header bytes the snapshot would observe.
    #[must_use]
    pub fn frozen_header(&self) -> Option<&Page> {
        self.frozen_header.as_ref()
    }

    /// Snapshot id.  Diagnostic-only.
    #[must_use]
    pub fn id(&self) -> SnapshotId {
        self.pin.id
    }

    /// The catalog B-tree root page-id this snapshot
    /// pinned. A concurrent `WriteTxn` that calls
    /// [`crate::pager::Pager::set_root_catalog`] does NOT mutate the
    /// value returned here — the snapshot is frozen at pin time.
    /// Use this when constructing a read-side
    /// [`crate::Catalog`] handle that should observe the catalog at
    /// the snapshot's LSN.
    #[must_use]
    pub fn root_catalog(&self) -> u64 {
        self.root_catalog
    }

    /// Read page `id` consistent with the snapshot's pin.
    ///
    /// Lookup order (file-backed pagers):
    /// 1. Frozen view (WAL frames at LSN ≤ `pinned_lsn` at snapshot
    ///    creation time).  Returns [`PageHandle::Shared`] — an
    ///    `Arc::clone` (refcount bump, no 4 KiB body copy).
    /// 2. Main file via the pager (cache-bypassed; goes through
    ///    `read_through` which verifies the page trailer).  Returns
    ///    [`PageHandle::Owned`].
    ///
    /// On in-memory pagers (`Pager::memory`) there is no WAL and no
    /// MVCC: the snapshot's `frozen_view` is always empty and the
    /// in-memory backend buffer may lag the cache (dirty cache
    /// frames have not yet been written back). For that mode the
    /// snapshot falls through to the LIVE cache (then main backend);
    /// the WAL overlay does not exist. No concurrent writer can
    /// race a reader on a memory pager, so the live read is the
    /// snapshot read.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] on syscall failure during the main-file read.
    /// - [`Error::Corruption`] if the on-disk page trailer fails to
    ///   verify.
    pub fn read_page(&self, pager: &Pager<F>, id: PageId) -> Result<PageHandle> {
        if let Some(page) = self.frozen_view.get(&id) {
            return Ok(PageHandle::Shared(Arc::clone(page)));
        }
        if pager.is_memory_backed() {
            return Ok(PageHandle::Owned(pager.read_cache_or_main(id)?));
        }
        if id.get() >= pager.main_physical_page_count()? {
            return Ok(PageHandle::Owned(Page::zeroed()));
        }
        Ok(PageHandle::Owned(pager.read_main_file_page(id)?))
    }
}

/// Default LRU cache size when [`Config::default`] is used. 64 frames
/// = 256 KiB of cached pages, comfortably within an embedded budget.
pub const DEFAULT_CACHE_FRAMES: usize = 64;

/// Per-pager compression knob. Selects whether
/// newly-created files use the transparent LZ4 page-compression
/// layer (`format_minor = 1`, `feature_flags` bit 0 set) or stay at
/// the original uncompressed `format_minor = 0` layout.
///
/// **No-op against existing files:** when a pager opens an
/// already-initialised database, the file's own header dictates
/// whether compression is in use; this knob only affects file
/// **creation**.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CompressionMode {
    /// Default — newly-created files use `format_minor = 0` with
    /// the full 32-bit CRC32C per-page trailer (no compression).
    #[default]
    Off,
    /// Newly-created files use `format_minor = 1` with LZ4 page
    /// compression. Requires the `compression` Cargo feature on
    /// `obj-core` / `obj-rs`. A build WITHOUT that feature refuses
    /// to open any `format_minor >= 1` file with
    /// [`Error::FormatFeatureUnsupported`].
    Lz4,
}

/// Storage type for the in-memory copy of the caller's
/// 32-byte master key held inside [`Config`].
///
/// Under the `encryption` Cargo feature this is
/// `zeroize::Zeroizing<[u8; 32]>`, which wipes the bytes when the
/// owning `Config` (and therefore this field) is dropped, so the
/// master key does not linger in freed heap/stack. `Zeroizing<T>`
/// is a transparent newtype: it derefs to `[u8; 32]`, implements
/// `Clone` (when the inner type does — `[u8; 32]` is `Clone`), and
/// — with the `zeroize/serde` feature — `Serialize`/`Deserialize`,
/// so the public surface of `Config` is byte-for-byte equivalent to
/// the previous `[u8; 32]` apart from the added drop-glue.
///
/// Without the feature it is a bare `[u8; 32]`: the no-`encryption`
/// build cannot use a key (the open path rejects it with
/// `Error::FormatFeatureUnsupported`), so there is no secret to
/// wipe and the type stays `Copy` to preserve the previous
/// behaviour exactly.
#[cfg(feature = "encryption")]
pub type MasterKeyBytes = zeroize::Zeroizing<[u8; 32]>;

/// See [`MasterKeyBytes`] — no-`encryption` build (bare array, no
/// secret material is ever stored so nothing to wipe).
#[cfg(not(feature = "encryption"))]
pub type MasterKeyBytes = [u8; 32];

/// Wrap raw 32-byte key material into the
/// feature-dependent [`MasterKeyBytes`] storage type.
///
/// Under the `encryption` feature this hands the bytes to
/// `zeroize::Zeroizing` so they wipe on drop; without the feature
/// it is the identity on `[u8; 32]`. Centralising the wrap in one
/// `#[cfg]`-split function keeps the call sites feature-agnostic and
/// avoids a `clippy::useless_conversion` on the no-feature build
/// (where a `From<[u8; 32]> for [u8; 32]` would be a no-op).
#[cfg(feature = "encryption")]
#[inline]
// allow: feature-symmetric helper — both encryption and no-encryption builds
// define `wrap_master_key`, so one variant is unused depending on the feature.
#[allow(dead_code)]
pub(crate) fn wrap_master_key(bytes: [u8; 32]) -> MasterKeyBytes {
    zeroize::Zeroizing::new(bytes)
}

/// See [`wrap_master_key`] — no-`encryption` build (identity).
#[cfg(not(feature = "encryption"))]
#[inline]
// allow: feature-symmetric helper — both encryption and no-encryption builds
// define `wrap_master_key`, so one variant is unused depending on the feature.
#[allow(dead_code)]
pub(crate) fn wrap_master_key(bytes: [u8; 32]) -> MasterKeyBytes {
    bytes
}

/// Pager construction options.
///
/// `Debug` is implemented manually so the `encryption_key` field —
/// if present — never leaks into log output (it redacts to
/// `"<set>"` or `"<not set>"`). The `Serialize`/`Deserialize` impls
/// are auto-derived; serialising a `Config` with a key present
/// will round-trip the bytes (callers must decide for themselves
/// whether persisting that is safe).
///
/// `Copy` is derived only on the no-`encryption` build.
/// Under the `encryption` feature the `encryption_key` field is a
/// `zeroize::Zeroizing` that wipes the key bytes on drop; a type
/// with drop-glue cannot be `Copy` (and `Copy` would defeat
/// zeroization by allowing silent bitwise duplication of the key).
#[cfg_attr(not(feature = "encryption"), derive(Copy))]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Number of cache frames. Must be at least 1.
    pub cache_frames: usize,
    /// Durability mode used by the WAL. See [`SyncMode`].
    pub sync_mode: SyncMode,
    /// Maximum WAL file size in bytes. Default 64 MiB.
    pub wal_size_limit: u64,
    /// Frame count at which the pager auto-checkpoints. Default
    /// 1 000.
    pub checkpoint_threshold: u64,
    /// Page-compression mode. See
    /// [`CompressionMode`]. Default `Off`.
    pub compression_mode: CompressionMode,
    /// Caller-supplied 32-byte master key for
    /// XChaCha20-Poly1305 page encryption. `None` (default) =
    /// unencrypted; `Some(key)` = encrypted (new files stamp
    /// `format_minor = 2` + `feature_flags` bit 1; existing files
    /// must already be `format_minor = 2`).
    ///
    /// Stored as a raw `[u8; 32]`; the per-file page key is
    /// derived via HKDF-SHA256(key, `kdf_salt`) at open time. Not
    /// persisted on disk.
    ///
    /// Available regardless of the `encryption` Cargo feature so
    /// that public APIs (and serde round-trips) remain consistent
    /// across builds. Setting a key in a build without the
    /// `encryption` feature causes `Pager::open` to fail with
    /// `Error::FormatFeatureUnsupported { feature: "encryption" }`
    /// when the file is encryption-capable.
    ///
    /// The inner type is [`MasterKeyBytes`] —
    /// `Zeroizing<[u8; 32]>` under the `encryption` feature so the
    /// bytes are wiped when this `Config` is dropped, or a bare
    /// `[u8; 32]` otherwise. The wrapper derefs to `[u8; 32]`, so
    /// reads via `.as_ref()` / `.as_deref()` are unchanged.
    pub encryption_key: Option<MasterKeyBytes>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("cache_frames", &self.cache_frames)
            .field("sync_mode", &self.sync_mode)
            .field("wal_size_limit", &self.wal_size_limit)
            .field("checkpoint_threshold", &self.checkpoint_threshold)
            .field("compression_mode", &self.compression_mode)
            .field(
                "encryption_key",
                if self.encryption_key.is_some() {
                    &"<set>"
                } else {
                    &"<not set>"
                },
            )
            .finish()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cache_frames: DEFAULT_CACHE_FRAMES,
            sync_mode: SyncMode::Full,
            wal_size_limit: crate::wal::DEFAULT_WAL_SIZE_LIMIT,
            checkpoint_threshold: crate::wal::DEFAULT_CHECKPOINT_THRESHOLD,
            compression_mode: CompressionMode::Off,
            encryption_key: None,
        }
    }
}

impl Config {
    /// Set the cache capacity.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `frames` is zero. The
    /// cache requires at least one frame to make progress.
    pub fn with_cache_frames(self, frames: usize) -> Result<Self> {
        if frames == 0 {
            return Err(Error::InvalidArgument("cache_frames must be >= 1"));
        }
        Ok(Self {
            cache_frames: frames,
            ..self
        })
    }

    /// Set the durability mode the WAL uses for every commit.
    #[must_use]
    pub fn with_sync_mode(self, sync_mode: SyncMode) -> Self {
        Self { sync_mode, ..self }
    }

    /// Set the WAL size cap in bytes.
    #[must_use]
    pub fn with_wal_size_limit(self, limit: u64) -> Self {
        Self {
            wal_size_limit: limit,
            ..self
        }
    }

    /// Set the auto-checkpoint frame threshold.
    #[must_use]
    pub fn with_checkpoint_threshold(self, frames: u64) -> Self {
        Self {
            checkpoint_threshold: frames,
            ..self
        }
    }

    /// Set the per-pager compression mode for
    /// new files. See [`CompressionMode`].
    #[must_use]
    pub fn with_compression_mode(self, mode: CompressionMode) -> Self {
        Self {
            compression_mode: mode,
            ..self
        }
    }

    /// Set the caller's 32-byte master
    /// encryption key. `None` clears any previously-set key. See
    /// [`Config::encryption_key`].
    #[must_use]
    pub fn with_encryption_key(self, key: Option<[u8; 32]>) -> Self {
        Self {
            encryption_key: key.map(wrap_master_key),
            ..self
        }
    }

    fn wal_config(&self) -> WalConfig {
        WalConfig {
            sync_mode: self.sync_mode,
            size_limit: self.wal_size_limit,
            checkpoint_threshold: self.checkpoint_threshold,
        }
    }

    /// Borrow the in-memory master key as a plain
    /// `&[u8; 32]`, hiding the feature-dependent storage type
    /// ([`MasterKeyBytes`]). On the `encryption` build the stored
    /// value is a `Zeroizing<[u8; 32]>`, which derefs to the array;
    /// on the no-feature build it is the array itself. Centralising
    /// the borrow here means the open/derive call sites stay
    /// identical across both builds and never name `Zeroizing`.
    fn master_key(&self) -> Option<&[u8; 32]> {
        self.encryption_key.as_ref().map(|k| {
            let bytes: &[u8; 32] = k;
            bytes
        })
    }
}

/// The pager.
///
/// Owns the storage backend (file or in-memory), a [`FileHeader`]
/// snapshot of page 0, a [`Cache`] of main-file pages, and (for
/// file-backed databases) a [`Wal`] sidecar plus two in-memory
/// overlays: a pending-transaction buffer and a committed-but-not-
/// checkpointed view. All public methods take `&mut self`.
///
/// Generic over `F: FileBackend` (hot-path dispatch is static
/// monomorphisation, never `dyn`). The default is the production
/// [`FileHandle`]; the fault-injection harness substitutes
/// `Pager<FaultyFileHandle>` to drive recovery against torn writes,
/// dropped fsyncs, and bit flips.
#[derive(Debug)]
pub struct Pager<F: FileBackend = FileHandle> {
    backend: Backend<F>,
    header: FileHeader,
    cache: Cache,
    /// WAL state. `None` for the in-memory pager (`Pager::memory`),
    /// `Some` for any file-backed database.
    wal: Option<WalState<F>>,
    config: Config,
    /// Live MVCC reader snapshots, keyed by an opaque snapshot id and
    /// valued by the WAL LSN each one has pinned.  The map is
    /// `Arc<Mutex<_>>` so the `SnapshotPin` RAII guard returned by
    /// `reader_snapshot` can deregister itself from any thread.
    /// Used by `checkpoint` to skip reclamation while a live reader
    /// pins an LSN below end-of-WAL.
    snapshots: Arc<Mutex<HashMap<SnapshotId, Lsn>>>,
    /// Allocator for snapshot ids.  Monotonic; not reset across
    /// pager lifetime.
    next_snapshot_id: Arc<AtomicU64>,
    /// Derived per-file page-encryption key.
    /// `Some` iff the file is encrypted AND the build has the
    /// `encryption` Cargo feature AND the caller supplied a valid
    /// master key. Computed once at open from
    /// `HKDF-SHA256(user_key, header.kdf_salt,
    /// b"obj-page-encryption-v1")` so the read/write hot path
    /// doesn't re-derive per page. Redacted in `Debug`.
    derived_key: Option<PageEncryptionKey>,
    /// High-water mark — the number of pages the **main file**
    /// is physically extended to cover (page 0 plus `main_high_water - 1`
    /// data slots). Only meaningful on the `Backend::File` arm; the
    /// in-memory backend keeps a `Vec` sized exactly to `page_count`
    /// and never consults this field.
    ///
    /// This is decoupled from `page_count` entirely: fresh allocations
    /// ride the WAL and do NOT extend the main file, so between a
    /// committed growing transaction and its next checkpoint the file is
    /// physically SHORTER than `page_count` — `main_high_water <
    /// page_count` is the normal state in that window. The slots in
    /// `[main_high_water, page_count)` live only in the WAL view; every
    /// read path resolves them from `pending`/`view` ahead of the main
    /// file (`read_page` priority). [`Self::apply_checkpoint_view`] is
    /// the sole place that grows the file (one bounded `set_len` to cover
    /// the max drained `PageId`) and re-seeds this mark from the grown
    /// length. The mark is in-memory only — it is NEVER written to disk.
    ///
    /// Its invariant: `file_length_for(mark - 1)` (for `mark >= 1`)
    /// equals the true on-disk length, so a slot at index `< mark` is
    /// guaranteed to physically exist. Seeded at `open` from the real
    /// file length via [`Self::main_pages_for_len`].
    main_high_water: u64,
    /// Injected entropy source for the KDF salt (at file creation) and,
    /// on encryption builds, each page's AEAD nonce. `Arc<dyn Entropy>`
    /// so the DST harness can substitute a seeded, reproducible source;
    /// production carries [`OsEntropy`]. Deliberately NOT part of
    /// [`Config`] — that type is `Copy + Serialize` on no-encryption
    /// builds and an `Arc<dyn>` would break both — so it is threaded as
    /// a constructor parameter instead. Cold path — untouched on reads.
    // allow: read only on the encryption build (page-nonce generation in
    // `encrypt_logical`). On a no-encryption build the value is still
    // threaded through to the WAL for salt generation, but this Pager
    // field itself is write-only, so it looks dead there.
    #[allow(dead_code)]
    entropy: Arc<dyn Entropy>,
}

/// Newtype wrapper around the derived 32-byte
/// page-encryption key. Manual `Debug` impl redacts the bytes so
/// the key never appears in log output even if the caller dumps a
/// `Pager`. The bytes are still accessible internally via
/// [`PageEncryptionKey::as_bytes`].
///
/// On a no-`encryption`-feature build the type still exists (the
/// `derived_key` field on `Pager` carries an `Option<_>` regardless
/// of the feature) but is never read; the `#[allow(dead_code)]`
/// reflects that the no-feature build sees only `None`.
///
/// The inner field is [`MasterKeyBytes`], so under the
/// `encryption` feature the derived per-file page key is wiped from
/// memory when the owning `Pager` (and therefore this value) is
/// dropped. `Copy` is derived only on the no-`encryption` build,
/// where the field is a bare `[u8; 32]` and never holds a real key.
#[cfg_attr(not(feature = "encryption"), derive(Copy))]
#[derive(Clone)]
// allow: held only on the encryption build; the no-encryption build constructs
// the pager without ever reading this key, so the field/methods look dead there.
#[allow(dead_code)]
struct PageEncryptionKey(MasterKeyBytes);

// allow: see PageEncryptionKey — methods are exercised only on the encryption build.
#[allow(dead_code)]
impl PageEncryptionKey {
    #[inline]
    fn as_bytes(&self) -> &[u8; 32] {
        let bytes: &[u8; 32] = &self.0;
        bytes
    }
}

impl std::fmt::Debug for PageEncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PageEncryptionKey(<redacted>)")
    }
}

#[derive(Debug)]
struct WalState<F: FileBackend> {
    wal: Wal<F>,
    /// Pages staged for the current uncommitted transaction. Drained
    /// into `view` on `commit`.
    pending: HashMap<PageId, Page>,
    /// Pages committed to the WAL but not yet checkpointed into the
    /// main file. Populated by recovery and by every successful
    /// `commit`; drained by `checkpoint` and by `flush` (a
    /// backward-compatible alias for "make data
    /// durable on the main file").
    ///
    /// Values are `Arc<Page>` so `reader_snapshot` / `header_
    /// snapshot` capture the view as cheap refcount bumps instead of
    /// 4 KiB-per-page memcpys. Each entry is replaced wholesale by a
    /// fresh `Arc` on the next commit of that id (never mutated in
    /// place), and the map is only touched via wholesale insert /
    /// drain / full-replace — there is deliberately NO `get_mut` /
    /// `entry` / index-assign on `view`, so an `Arc<Page>` shared
    /// into a pinned snapshot can never be mutated under it.
    view: HashMap<PageId, Arc<Page>>,
    /// Dirty flag for the file-header `root_catalog` slot.
    /// Set by [`Pager::set_root_catalog`]; cleared on commit /
    /// rollback. When set at commit time, the pager appends a
    /// page-0 frame to the WAL carrying the current in-memory
    /// encoded header so reader snapshots taken AFTER the commit
    /// observe the new value AND a crash before checkpoint can
    /// re-apply the header via replay.
    header_dirty: bool,
    /// Committed-view of the header at offset 0. Holds the
    /// encoded page-0 of the most-recent WAL frame that touched
    /// the header (whether from recovery or from a runtime commit).
    /// Drained into the main file by [`Pager::checkpoint`]. `None`
    /// means "the on-disk header at offset 0 is authoritative" —
    /// either no header frame has ever been committed, or the
    /// most-recent checkpoint already wrote the staged copy out.
    view_header: Option<Page>,
    /// Snapshot of the committed `root_catalog`. Mirrors
    /// `Pager.header.root_catalog` but lags it by one txn — it
    /// only advances on a successful [`Pager::commit`]. Reader
    /// snapshots capture THIS value (not the live one) so a writer
    /// mid-txn cannot leak its in-flight catalog root to readers
    /// pinned at an earlier LSN.
    committed_root_catalog: u64,
    /// Transaction depth — incremented by
    /// [`Pager::begin_txn`] (called from
    /// [`crate::txn::WriteTxn::begin`]) and decremented by
    /// [`Pager::end_txn`]. Drives the [`Pager::in_txn`] helper that
    /// the catalog debug-asserts at its mutation boundaries.
    txn_depth: u32,
}

/// Storage backend.
#[derive(Debug)]
enum Backend<F: FileBackend> {
    /// File-backed database. `pread`/`pwrite` go through the
    /// generic `F` (`FileHandle` in production, `FaultyFileHandle`
    /// in fault-injection tests).
    File(F),
    /// Memory-backed database: one `Vec<u8>` of `page_count * PAGE_SIZE`.
    Memory(Vec<u8>),
}

impl Pager<FileHandle> {
    /// Open or create a database file at `path`. A new file is
    /// initialised with a default [`FileHeader`] and no allocated
    /// pages beyond page 0.
    ///
    /// Cache capacity is taken from `config`. The cache is allocated
    /// before any read or write; subsequent operations never call
    /// the global allocator on the cache hot path.
    ///
    /// Opening a file-backed database also opens (or creates)
    /// the WAL sidecar at `<path>-wal` and replays any committed-but-
    /// not-checkpointed frames before any read can succeed. If no
    /// WAL exists, or the existing WAL belongs to a previous
    /// generation (salt mismatch), the database opens as if the WAL
    /// were empty.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `config.cache_frames == 0`.
    /// - [`Error::Io`] if the file cannot be opened or initialised.
    /// - [`Error::InvalidFormat`] if an existing main file does not
    ///   look like an obj database, or if an existing WAL has a
    ///   header that disagrees with the main file's format.
    pub fn open<P: AsRef<Path>>(path: P, config: Config) -> Result<Self> {
        Self::open_with_env(path, config, Arc::new(OsEntropy))
    }

    /// Open or create a database file at `path`, drawing every salt
    /// and nonce from the caller-supplied `entropy` source rather than
    /// the OS default. This is the deterministic-simulation-testing
    /// (DST) entry point: passing a
    /// [`SeededEntropy`](crate::platform::SeededEntropy) makes two
    /// runs of the same seed produce byte-identical on-disk salts and
    /// nonces. [`Pager::open`] is the production convenience that
    /// defaults `entropy` to [`OsEntropy`].
    ///
    /// # Errors
    ///
    /// Identical to [`Pager::open`].
    pub fn open_with_env<P: AsRef<Path>>(
        path: P,
        config: Config,
        entropy: Arc<dyn Entropy>,
    ) -> Result<Self> {
        let main_path = path.as_ref().to_path_buf();
        let main = FileHandle::open_or_create(&main_path)?;
        let wal_path = wal_path_for(&main_path);
        let wal = FileHandle::open_or_create(&wal_path)?;
        Self::open_with_backends(main, wal, wal_path, config, entropy)
    }

    /// Construct a fresh in-memory pager. Cache capacity is taken from
    /// `config`; the backing store starts at one page (the header).
    /// The in-memory pager has no WAL — all writes go straight to the
    /// in-memory buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `config.cache_frames` is
    /// zero.
    pub fn memory(config: Config) -> Result<Self> {
        if config.cache_frames == 0 {
            return Err(Error::InvalidArgument("cache_frames must be >= 1"));
        }
        refuse_compression_without_feature(config.compression_mode)?;
        refuse_encryption_without_feature(config.encryption_key.is_some())?;
        let entropy: Arc<dyn Entropy> = Arc::new(OsEntropy);
        let header =
            build_new_file_header(config.compression_mode, config.master_key(), entropy.as_ref())?;
        let mut bytes = vec![0u8; PAGE_SIZE];
        let mut p = Page::zeroed();
        encode_header(&header, &mut p);
        bytes[..PAGE_SIZE].copy_from_slice(p.as_bytes());
        let derived_key = derive_key_for_open(&config, &header)?;
        Ok(Self {
            backend: Backend::Memory(bytes),
            header,
            cache: Cache::new(config.cache_frames),
            wal: None,
            config,
            snapshots: Arc::new(Mutex::new(HashMap::new())),
            next_snapshot_id: Arc::new(AtomicU64::new(1)),
            derived_key,
            main_high_water: 0,
            entropy,
        })
    }
}

impl<F: FileBackend> Pager<F> {
    /// Open a file-backed pager on top of caller-supplied backends.
    ///
    /// `main` is the database file; `wal` is the WAL sidecar at
    /// `wal_path`. Both must already be open and writable. The WAL is
    /// walked for recovery before any user-visible read
    /// can succeed.
    ///
    /// Production callers SHOULD use [`Pager::open`]; the
    /// fault-injection harness uses this entry point to drop a
    /// `FaultyFileHandle` into the pager and a *separate* one into
    /// the WAL.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `config.cache_frames == 0`.
    /// - [`Error::Io`] on any syscall failure.
    /// - [`Error::InvalidFormat`] if the existing main file does not
    ///   look like an obj database, or if the WAL header disagrees
    ///   with the main file's format.
    /// - [`Error::WalCorruption`] if the WAL contains a CRC-invalid
    ///   frame before its last commit marker.
    pub fn open_with_backends(
        main: F,
        wal: F,
        wal_path: std::path::PathBuf,
        config: Config,
        entropy: Arc<dyn Entropy>,
    ) -> Result<Self> {
        if config.cache_frames == 0 {
            return Err(Error::InvalidArgument("cache_frames must be >= 1"));
        }
        refuse_compression_without_feature(config.compression_mode)?;
        refuse_encryption_without_feature(config.encryption_key.is_some())?;
        let mut header = if main.is_empty()? {
            initialise_file(
                &main,
                config.compression_mode,
                config.master_key(),
                entropy.as_ref(),
            )?
        } else {
            load_header(&main)?
        };
        refuse_unsupported_features(&header)?;
        let derived_key = derive_key_for_open(&config, &header)?;
        let (wal_state, recovered_view, view_header) = recover_or_create_wal(
            &main,
            wal,
            wal_path,
            &mut header,
            &config,
            derived_key.as_ref(),
            Arc::clone(&entropy),
        )?;
        let view: HashMap<PageId, Arc<Page>> = recovered_view
            .into_iter()
            .map(|(id, page)| (id, Arc::new(page)))
            .collect();
        let committed_root_catalog = header.root_catalog;
        let file_len = main.len()?;
        let mut pager = Self {
            backend: Backend::File(main),
            header,
            cache: Cache::new(config.cache_frames),
            wal: Some(WalState {
                wal: wal_state,
                pending: HashMap::new(),
                view,
                header_dirty: false,
                view_header,
                committed_root_catalog,
                txn_depth: 0,
            }),
            config,
            snapshots: Arc::new(Mutex::new(HashMap::new())),
            next_snapshot_id: Arc::new(AtomicU64::new(1)),
            derived_key,
            main_high_water: 0,
            entropy,
        };
        pager.main_high_water = pager.main_pages_for_len(file_len);
        pager.debug_assert_recovered_pages_covered();
        Ok(pager)
    }

    /// Assert the post-recovery reachability invariant —
    /// every page id the recovered header claims (`1..page_count`) is
    /// resolvable, either because the main file physically covers it
    /// (`id < main_high_water`) or because the recovered WAL `view`
    /// carries its body. A page in `[main_high_water, page_count)` that
    /// is absent from the view would be genuinely lost (the header names
    /// a page neither the file nor the WAL can produce); the recovery
    /// contract guarantees this never happens, since the same WAL frame
    /// that advanced `page_count` also staged the page body. Bounded by
    /// `page_count`. Elided in release builds.
    fn debug_assert_recovered_pages_covered(&self) {
        #[cfg(debug_assertions)]
        {
            let Some(state) = self.wal.as_ref() else {
                return;
            };
            let mut id_raw = self.main_high_water.max(1);
            while id_raw < self.header.page_count {
                if let Some(pid) = PageId::new(id_raw) {
                    debug_assert!(
                        state.view.contains_key(&pid) || state.pending.contains_key(&pid),
                        "#91: recovered page {id_raw} beyond the physical \
                         high-water must be resident in the WAL view",
                    );
                }
                id_raw += 1;
            }
        }
    }

    /// Total number of pages in the database, including page 0.
    #[must_use]
    pub fn page_count(&self) -> u64 {
        self.header.page_count
    }

    /// Number of pages the main file is PHYSICALLY long enough to
    /// hold (page 0 plus `result - 1` data slots) — the real on-disk
    /// high-water. On the file backend this is computed from the live
    /// file length, NOT from `page_count`: a committed growing
    /// transaction advances `page_count` while its fresh pages still
    /// live only in the WAL view, so the file is SHORTER than
    /// `page_count` until the next checkpoint. The backup path
    /// ([`crate::backup`]) gates its main-file copy by THIS value and
    /// relies on `overlay_frozen_view` to fill the WAL-resident fresh
    /// pages — reading them off the (too-short) main file would
    /// `UnexpectedEof`. On the in-memory backend the `Vec` is sized
    /// exactly to `page_count`, so this returns `page_count`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file length cannot be queried.
    pub fn main_physical_page_count(&self) -> Result<u64> {
        match &self.backend {
            Backend::File(handle) => {
                let len = handle.len()?;
                Ok(self.main_pages_for_len(len))
            }
            Backend::Memory(_) => Ok(self.header.page_count),
        }
    }

    /// On-disk page size in bytes (4096 at format major 0). Surfaced
    /// for diagnostics; callers who want the
    /// compile-time constant should reach for
    /// [`crate::pager::page::PAGE_SIZE`] directly.
    #[must_use]
    pub fn page_size(&self) -> u16 {
        self.header.page_size
    }

    /// `(format_major, format_minor)` from the on-disk header.
    /// Surfaced for diagnostics so a forensic
    /// tool can confirm the file's format vintage without re-reading
    /// page 0.
    #[must_use]
    pub fn format_version(&self) -> (u16, u16) {
        (self.header.format_major, self.header.format_minor)
    }

    /// The current freelist head (`0` = empty). Useful for tests.
    #[must_use]
    pub fn freelist_head(&self) -> u64 {
        self.header.freelist_head
    }

    /// The catalog B-tree root page-id, or `0` if no catalog has yet
    /// been installed. The catalog uses this field to bootstrap
    /// on first open; older `format_minor = 0` databases
    /// always carry zero here.
    #[must_use]
    pub fn root_catalog(&self) -> u64 {
        self.header.root_catalog
    }

    /// Update the catalog B-tree root page-id and persist the change
    /// in the file header.
    ///
    /// The catalog calls this exactly once per
    /// `open_or_init` when it allocates a fresh empty catalog root,
    /// and on every catalog mutation that produces a new root via
    /// the B+tree's copy-on-write contract.
    ///
    /// The header update is **WAL-staged** on
    /// file-backed pagers: the call records the new value in the
    /// in-memory header (so subsequent reads from this writer see
    /// it) AND stages the encoded page-0 into the current WAL
    /// transaction. The on-disk header at offset 0 is NOT touched
    /// until checkpoint; reader snapshots therefore see the
    /// pre-commit value of `root_catalog` (whichever value the
    /// committed WAL view held at snapshot time). For in-memory
    /// pagers the call still writes the header into the in-memory
    /// backend buffer immediately (no WAL exists).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure writing the header
    /// (in-memory backend only — the file-backed path no longer
    /// performs an immediate write).
    pub fn set_root_catalog(&mut self, root: u64) -> Result<()> {
        self.header.root_catalog = root;
        self.stage_or_write_header()
    }

    /// Allocate a new page. If the freelist is non-empty, recycles its
    /// head; otherwise appends a brand-new page to the file.
    ///
    /// `alloc_page` is **transactional**: the
    /// freelist-page mutation it performs is staged in the current
    /// WAL transaction (the freelist link page is written through
    /// the WAL just like a regular user page write). The file-header
    /// update (`freelist_head` / `page_count`)
    /// also rides the WAL (via the same private
    /// `stage_or_write_header` pathway installed for
    /// [`Pager::set_root_catalog`]) — a crash between the WAL frame
    /// durability and the header write can no longer leave the
    /// on-disk header pointing at a not-yet-durable freelist link
    /// page. Callers SHOULD call [`Pager::commit`] before relying
    /// on the allocation being durable; pending allocations are
    /// lost on `Pager::open` after a crash, exactly like
    /// uncommitted user writes.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on syscall failure when extending the file.
    /// - [`Error::Corruption`] if the freelist head fails to decode
    ///   (indicates a previously-written freelist page has been
    ///   damaged).
    /// - [`Error::InvalidArgument`] in the unrealistic case that
    ///   `page_count` would overflow `u64` or the resulting file size
    ///   would overflow.
    pub fn alloc_page(&mut self) -> Result<PageId> {
        debug_assert!(
            self.in_txn(),
            "alloc_page must be inside a Pager txn (begin_txn/end_txn)"
        );
        if let Some(head) = PageId::new(self.header.freelist_head) {
            self.alloc_from_freelist(head)
        } else {
            self.alloc_fresh()
        }
    }

    /// Read page `id`. Returns a borrow-shaped [`PageRef`] that
    /// references bytes resident in one of (a) the in-flight
    /// transaction buffer, (b) the committed-but-not-checkpointed
    /// WAL view, (c) the LRU cache, or (d) — on a cache miss — a
    /// freshly-inserted cache frame populated by a single `pread`.
    ///
    /// Read priority: in-flight transaction buffer → committed
    /// (WAL) view → cache → main file. The first three are
    /// in-memory hash-map / cache lookups; the last is a `pread`.
    ///
    /// # Allocation contract
    ///
    /// `read_page` performs **no heap allocation on cache hits or
    /// WAL-overlay hits**. On a cache miss, a single `pread` is
    /// issued and the page is inserted into the cache; the returned
    /// `PageRef` then borrows that cache frame.
    ///
    /// # Lifetime contract
    ///
    /// The returned `PageRef<'_>` borrows `self`. The borrow checker
    /// forbids any mutating call on the same pager (`write_page`,
    /// `commit`, `checkpoint`, `alloc_page`, `free_page`, `close`,
    /// `flush`) while a `PageRef` is alive. Callers that need an
    /// owned page across mutating calls can use
    /// [`PageRef::to_owned_page`].
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] if a cache-miss read from disk fails.
    /// - [`Error::Corruption`] if the page trailer fails to verify
    ///   on a cache-miss path.
    pub fn read_page(&mut self, id: PageId) -> Result<PageRef<'_>> {
        debug_assert!(id.get() > 0, "PageId is non-zero by construction");
        debug_assert!(
            id.get() < self.header.page_count,
            "read_page called with out-of-range id",
        );
        if id.get() >= self.header.page_count {
            return Err(Error::InvalidArgument("page id out of range"));
        }
        if self.wal_lookup_some(id) {
            return self.lookup_in_wal(id);
        }
        if self.cache.get(id).is_some() {
            return self.lookup_in_cache(id);
        }
        let buf = self.read_through(id)?;
        let evicted = self.cache.insert(id, buf, false);
        self.handle_eviction(evicted)?;
        self.lookup_in_cache(id)
    }

    /// `true` iff the WAL overlay carries an entry for `id`.
    fn wal_lookup_some(&self, id: PageId) -> bool {
        let Some(state) = self.wal.as_ref() else {
            return false;
        };
        state.pending.contains_key(&id) || state.view.contains_key(&id)
    }

    /// Return a `PageRef` borrowing from the WAL overlay. Assumes
    /// the caller verified the entry exists via [`Self::wal_lookup_some`].
    fn lookup_in_wal(&self, id: PageId) -> Result<PageRef<'_>> {
        let state = self
            .wal
            .as_ref()
            .ok_or(Error::InvalidArgument("internal: wal overlay missing"))?;
        let page = state
            .pending
            .get(&id)
            .or_else(|| state.view.get(&id).map(Arc::as_ref))
            .ok_or(Error::InvalidArgument("internal: wal lookup race"))?;
        Ok(PageRef::new(id, page))
    }

    /// Return a `PageRef` borrowing from the cache. The caller must
    /// have just touched the cache (so the LRU is already updated).
    fn lookup_in_cache(&mut self, id: PageId) -> Result<PageRef<'_>> {
        let page = self
            .cache
            .get(id)
            .ok_or(Error::InvalidArgument("internal: cache miss after insert"))?;
        Ok(PageRef::new(id, page))
    }

    /// Write `page` back to `id`. For file-backed databases, the write
    /// is staged in the WAL transaction buffer; for in-memory
    /// databases, the write goes straight to the cache.
    ///
    /// To make the write durable, call [`Pager::commit`].
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] if a dirty eviction triggered by this insert
    ///   fails to write its predecessor to disk (memory pager only).
    pub fn write_page(&mut self, id: PageId, page: &Page) -> Result<()> {
        debug_assert!(id.get() < self.header.page_count);
        if id.get() >= self.header.page_count {
            return Err(Error::InvalidArgument("page id out of range"));
        }
        if let Some(state) = self.wal.as_mut() {
            state.pending.insert(id, page.clone());
            return Ok(());
        }
        if let Some(slot) = self.cache.get_mut(id) {
            *slot = page.clone();
            return Ok(());
        }
        let evicted = self.cache.insert(id, page.clone(), true);
        self.handle_eviction(evicted)
    }

    /// Free a previously-allocated page, returning it to the freelist.
    /// `id` must refer to a currently-allocated page; freeing the same
    /// page twice is a caller bug.
    ///
    /// The freelist link page is staged in the
    /// current WAL transaction (file-backed pagers) rather than
    /// written directly to the main file. The
    /// header `freelist_head` update also rides the WAL (via the
    /// same `stage_or_write_header` pathway used
    /// for [`Pager::set_root_catalog`]); a crash mid-txn no longer
    /// leaves the on-disk header pointing at a freelist link that is
    /// only durable in the WAL view. Call [`Pager::commit`] before
    /// relying on the free being durable.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] if the freelist record or header write fails.
    pub fn free_page(&mut self, id: PageId) -> Result<()> {
        debug_assert!(id.get() > 0);
        debug_assert!(id.get() < self.header.page_count);
        debug_assert!(
            self.in_txn(),
            "free_page must be inside a Pager txn (begin_txn/end_txn)"
        );
        if id.get() >= self.header.page_count {
            return Err(Error::InvalidArgument("page id out of range"));
        }
        let _ = self.cache.evict(id);
        let next = self.header.freelist_head;
        let mut buf = Page::zeroed();
        encode_freelist_page(FreeListPage::new(next), &mut buf);
        write_page_trailer(&mut buf);
        if let Some(state) = self.wal.as_mut() {
            state.pending.insert(id, buf);
        } else {
            self.write_back_page(id, &buf)?;
        }
        self.header.freelist_head = id.get();
        self.stage_or_write_header()?;
        Ok(())
    }

    /// Commit the in-flight transaction. Writes every staged frame to
    /// the WAL with a single `sync_data` at the end (group commit).
    /// Returns the LSN of the last committed frame, or `0` if the
    /// transaction was empty.
    ///
    /// If the WAL's committed-frame count exceeds
    /// `Config::checkpoint_threshold` after the commit, the pager
    /// inlines a [`Pager::checkpoint`] call. Auto-checkpoint amortises
    /// recovery time across writers without surfacing as a separate
    /// API call to the caller.
    ///
    /// The inline auto-checkpoint is **non-fatal to the commit**: it
    /// runs only after the WAL durability step has succeeded, so if it
    /// fails (e.g. ENOSPC/EIO on the main file) the commit still
    /// returns Ok — the frames are durable in the WAL, the in-memory
    /// view is left intact, and the next commit re-tries the
    /// checkpoint. This is the failure-atomicity guarantee: a commit
    /// that returns Ok is committed; a commit's durably-written header
    /// frame is never rewound by a later checkpoint failure.
    ///
    /// For in-memory pagers (no WAL) this is a no-op returning `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] only if the WAL durability step fails; a
    /// failed auto-checkpoint is swallowed (deferred), not surfaced.
    pub fn commit(&mut self) -> Result<Lsn> {
        // WAL durability step. Once this returns Ok the transaction is
        // committed: its frames (pending pages + the page-0 header
        // frame) are durable in the WAL and will be replayed on the
        // next open.
        let lsn = self.commit_inner()?;
        // Auto-checkpoint is best-effort cleanup that runs AFTER
        // durability. A failure here (e.g. ENOSPC/EIO writing the main
        // file) must NOT fail the commit: the frames are already
        // durable, `checkpoint` leaves the in-memory view intact on
        // failure, and the next commit re-tries (committed_frames is
        // still over threshold). Propagating the error would let
        // `WriteTxn::Drop` rewind the header snapshot over a
        // transaction the WAL has already committed — resurrecting it
        // on reopen and diverging same-process reads from post-recovery
        // reads. Treat it as a deferred checkpoint instead.
        let needs_checkpoint = match self.wal.as_ref() {
            Some(state) => state.wal.committed_frames() >= self.config.checkpoint_threshold,
            None => false,
        };
        if needs_checkpoint {
            #[cfg(feature = "tracing")]
            if let Err(err) = self.checkpoint() {
                tracing::warn!(
                    error = ?err,
                    "auto-checkpoint deferred after failure; committed frames remain durable in the WAL"
                );
            }
            #[cfg(not(feature = "tracing"))]
            let _ = self.checkpoint();
        }
        Ok(lsn)
    }

    fn commit_inner(&mut self) -> Result<Lsn> {
        let mut header_page: Page = Page::zeroed();
        encode_header(&self.header, &mut header_page);

        let Some(state) = self.wal.as_mut() else {
            return Ok(Lsn::ZERO);
        };
        let header_dirty = state.header_dirty;
        if state.pending.is_empty() && !header_dirty {
            return Ok(Lsn::ZERO);
        }
        let mut txn = state.wal.begin_txn();
        let mut ids: Vec<PageId> = state.pending.keys().copied().collect();
        ids.sort_unstable();
        for id in &ids {
            if let Some(page) = state.pending.get(id) {
                txn.append(*id, page)?;
            }
        }
        if header_dirty {
            txn.append_header(&header_page)?;
        }
        let lsn = txn.commit()?;
        for id in ids {
            if let Some(page) = state.pending.remove(&id) {
                let fresh = Arc::new(page);
                debug_assert_eq!(
                    Arc::strong_count(&fresh),
                    1,
                    "#53: a freshly-committed page version must be \
                     uniquely owned when published into the view",
                );
                state.view.insert(id, fresh);
            }
            let _ = self.cache.evict(id);
        }
        if header_dirty {
            state.view_header = Some(header_page);
            state.header_dirty = false;
            state.committed_root_catalog = self.header.root_catalog;
        }
        Ok(lsn)
    }

    /// Perform a final checkpoint and remove the WAL sidecar.
    ///
    /// `close` does NOT auto-commit a pending transaction —
    /// `write_page` calls without a matching `commit()` are dropped
    /// silently, matching the "uncommitted writes are not durable"
    /// half of the ACID contract. If you want the pending
    /// txn to land on disk, call `commit()` before `close()`.
    ///
    /// After `close()` returns, a fresh `Pager::open` on the same
    /// path observes a database with no WAL — the
    /// "no sidecar files left behind after a clean shutdown"
    /// invariant.
    ///
    /// For in-memory pagers `close` is a no-op (no WAL to remove).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn close(mut self) -> Result<()> {
        if let Some(state) = self.wal.as_mut() {
            state.pending.clear();
        }
        self.checkpoint()?;
        let path = self
            .wal
            .as_ref()
            .map(|state| state.wal.path().to_path_buf());
        drop(self);
        if let Some(p) = path {
            crate::wal::remove_wal(&p)?;
        }
        Ok(())
    }

    /// Backward-compatible flush. This is `commit() +
    /// checkpoint() + fsync(main)` for file-backed databases. For
    /// the in-memory backend it preserves the "drain dirty
    /// cache + fsync" semantics. Kept as a stable alias —
    /// new code SHOULD call [`Pager::commit`] +
    /// [`Pager::checkpoint`] / [`Pager::close`] directly.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn flush(&mut self) -> Result<()> {
        let _ = self.commit()?;
        self.checkpoint()?;
        let cap = self.cache.capacity();
        let pending: Vec<(PageId, Page)> = self.cache.drain_dirty().take(cap).collect();
        for (id, page) in pending {
            self.write_back_page(id, &page)?;
        }
        self.write_header()?;
        match &self.backend {
            Backend::File(handle) => handle.sync_data(self.config.sync_mode)?,
            Backend::Memory(_) => {}
        }
        Ok(())
    }

    /// Roll every committed WAL frame forward into the main file.
    ///
    /// Protocol:
    /// 1. For every page-id in the WAL view, write the page (with
    ///    its CRC32C trailer) into the main file.
    /// 2. `sync_data(SyncMode::Full)` on the main file. Only after
    ///    this returns Ok are the main-file writes durable.
    /// 3. Rotate the WAL salt via `Wal::reset_after_checkpoint` and
    ///    truncate the WAL to header-only with the new salt.
    /// 4. Stamp the new salt into the main file's `wal_salt` header
    ///    field and `sync_data` the main file again.
    ///
    /// Idempotent: a second invocation on an empty view is a no-op.
    ///
    /// Crash-recovery model: a crash before step 3 leaves the old
    /// WAL with the old salt; the next `Pager::open` recovers it
    /// (idempotent — re-applying writes the same bytes step 1
    /// already wrote). A crash after step 3 but before step 4
    /// leaves the WAL with the new salt and the main file with the
    /// old; the next open reads the OLD salt from the main header,
    /// fails to match, treats the WAL as empty, and proceeds —
    /// recovery loses no data because step 2 made the main file
    /// authoritative before the salt rotated.
    ///
    /// In-memory pagers have no WAL; `checkpoint` is a no-op for
    /// them.
    ///
    /// Failure-atomic: the in-memory committed view (and staged page-0
    /// header) is only dropped after the WHOLE checkpoint succeeds. If
    /// any step returns `Err`, the overlay is left intact so the frames
    /// — still durable in the WAL — remain visible to same-process
    /// reads and to a later retry.
    ///
    /// # Deferral while a reader is pinned (writer-starvation hazard)
    ///
    /// If any live MVCC reader (`reader_snapshot`) has pinned an LSN
    /// below end-of-WAL, this returns `Ok(())` immediately and reclaims
    /// **nothing** (`checkpoint_deferred_for_pinned_reader`) — folding
    /// the WAL would discard frames the reader still needs. This is a
    /// deliberate all-or-nothing deferral: there is no partial reclaim
    /// up to `min_pinned_lsn`, so while the reader lives the WAL only
    /// grows. Because `commit`'s auto-checkpoint uses this same path, a
    /// single long-held [`ReaderSnapshot`] starves writers — the WAL
    /// climbs to `WalConfig::size_limit` and `append_raw` then fails all
    /// writes with `"wal size limit exceeded"`. See [`Self::reader_snapshot`]
    /// for the full interaction and the caller's "keep reads short"
    /// obligation.
    ///
    /// # Single-process scope (cross-process readers are NOT protected)
    ///
    /// The pinned-reader check consults **this process's** snapshots map
    /// only. A reader in another process is invisible here, and its
    /// shared cross-process `READER_LOCK` byte does not conflict with
    /// the writer lock this checkpoint runs under. Consequently a
    /// checkpoint here can fold WAL frames and rotate the salt out from
    /// under a reader in a *different* process. Snapshot isolation is a
    /// single-process guarantee; the cross-process contract is
    /// single-writer exclusion only. See [`Self::reader_snapshot`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "pager.checkpoint", level = "debug", skip_all)
    )]
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.checkpoint_deferred_for_pinned_reader() {
            #[cfg(feature = "tracing")]
            tracing::debug!(reason = "reader_pin", "deferred");
            return Ok(());
        }
        // Clone (do NOT drain) the committed view. A mid-checkpoint
        // I/O failure must leave the in-memory overlay intact: the
        // frames are still durable in the WAL, so same-process reads
        // and any later retry must continue to observe them. The
        // overlay is dropped only after the WHOLE checkpoint (main
        // write-back + salt rotation) has succeeded — at which point
        // the main file is authoritative and the WAL has been
        // truncated. The Arc clones are pointer-only; this is the
        // amortised once-per-threshold path, never a read hot path.
        let (view_pages, staged_header): (Vec<(PageId, Arc<Page>)>, Option<Page>) =
            if let Some(state) = self.wal.as_ref() {
                let mut pages: Vec<(PageId, Arc<Page>)> = state
                    .view
                    .iter()
                    .map(|(id, page)| (*id, Arc::clone(page)))
                    .collect();
                // Sort by PageId so the write-back (and, for encrypted
                // DBs, the per-page nonce draw from the injected Entropy
                // stream) happens in a deterministic order regardless of
                // the source HashMap's iteration order — mirroring the
                // WAL-commit sort in `commit_inner`. Without this, two
                // same-seed runs pair pages with different nonces and the
                // encrypted main file diverges byte-for-byte.
                pages.sort_unstable_by_key(|(id, _)| id.get());
                (pages, state.view_header.clone())
            } else {
                return Ok(());
            };
        let nothing_to_do = view_pages.is_empty() && staged_header.is_none();
        self.apply_checkpoint_view(&view_pages, staged_header.as_ref())?;
        if nothing_to_do {
            return Ok(());
        }
        self.rotate_wal_salt_and_persist()?;
        if let Some(state) = self.wal.as_mut() {
            state.view.clear();
            state.view_header = None;
        }
        Ok(())
    }

    /// Returns `true` when a live MVCC reader has pinned an LSN below
    /// the current end-of-WAL — in which case [`Self::checkpoint`]
    /// must defer rather than reclaim frames the reader still needs.
    ///
    /// Deferral is all-or-nothing: a single pinned LSN below end-of-WAL
    /// blocks the entire checkpoint (no partial reclaim up to
    /// `min_pinned_lsn`), so the WAL grows until the reader drops — the
    /// writer-starvation hazard documented on [`Self::reader_snapshot`].
    /// Only **this process's** snapshots map is consulted; a reader in
    /// another process cannot be seen and is not protected. See
    /// [`Self::checkpoint`] for the cross-process scope note.
    fn checkpoint_deferred_for_pinned_reader(&self) -> bool {
        let Some(min_lsn) = self.min_pinned_lsn() else {
            return false;
        };
        let end_lsn = self
            .wal
            .as_ref()
            .map_or(Lsn::ZERO, |s| s.wal.next_lsn().prev_saturating());
        min_lsn < end_lsn
    }

    /// Phase 2 of [`Self::checkpoint`]: write each WAL-view page back
    /// to the main file (evicting the matching cache slot so a
    /// subsequent read re-fetches the durable copy), apply any staged
    /// page-0 header, and `sync_data` the main backend so the writes
    /// are durable before the salt rotation.
    fn apply_checkpoint_view(
        &mut self,
        view_pages: &[(PageId, Arc<Page>)],
        staged_header: Option<&Page>,
    ) -> Result<()> {
        self.grow_main_to_cover(view_pages)?;
        for (id, page) in view_pages {
            self.write_back_page(*id, page.as_ref())?;
            let _ = self.cache.evict(*id);
        }
        if let Some(hp) = staged_header {
            match &mut self.backend {
                Backend::File(handle) => handle.write_all_at(hp.as_bytes(), 0)?,
                Backend::Memory(bytes) => {
                    if bytes.len() < PAGE_SIZE {
                        bytes.resize(PAGE_SIZE, 0);
                    }
                    bytes[..PAGE_SIZE].copy_from_slice(hp.as_bytes());
                }
            }
        }
        match &self.backend {
            Backend::File(handle) => handle.sync_data(self.config.sync_mode)?,
            Backend::Memory(_) => {}
        }
        Ok(())
    }

    /// Guardrail: grow the main file so it physically covers the
    /// maximum `PageId` in `view_pages`, in a SINGLE bounded `set_len`
    /// (the target length is `file_length_for(max_id)`, a closed
    /// form — no per-page syscalls). Re-seed [`Self::main_high_water`]
    /// from the grown length. A no-op when the file already covers every
    /// drained id (e.g. an all-overwrite checkpoint with no fresh pages)
    /// or on the in-memory backend (whose `Vec` was grown per-alloc and
    /// will be resized by `write_back_page` if needed).
    fn grow_main_to_cover(&mut self, view_pages: &[(PageId, Arc<Page>)]) -> Result<()> {
        if !matches!(self.backend, Backend::File(_)) {
            return Ok(());
        }
        let Some(max_id) = view_pages.iter().map(|(id, _)| id.get()).max() else {
            return Ok(());
        };
        if max_id < self.main_high_water {
            return Ok(());
        }
        let new_len = self.file_length_for(max_id)?;
        if let Backend::File(handle) = &mut self.backend {
            handle.set_len(new_len)?;
        }
        self.main_high_water = self.main_pages_for_len(new_len);
        debug_assert!(
            self.main_high_water > max_id,
            "#91: grown file must physically cover the max drained id",
        );
        Ok(())
    }

    /// Phase 3 of [`Self::checkpoint`]: rotate the WAL salt (which
    /// also truncates the WAL to header-only with the new salt),
    /// stamp the new salt into the main file header, and `sync_data`
    /// the main backend.
    fn rotate_wal_salt_and_persist(&mut self) -> Result<()> {
        if let Some(state) = self.wal.as_mut() {
            state.wal.reset_after_checkpoint()?;
            stamp_salt_into_header(&mut self.header, state.wal.salt());
        }
        self.write_header()?;
        match &self.backend {
            Backend::File(handle) => handle.sync_data(self.config.sync_mode)?,
            Backend::Memory(_) => {}
        }
        Ok(())
    }

    /// Open a new MVCC reader snapshot at the current WAL end-LSN.
    ///
    /// The snapshot captures (1) the LSN of the most-recent committed
    /// frame in the WAL at the moment of the call, and (2) a clone of
    /// the pager's in-memory committed view (`WalState.view`).
    /// Reads through the snapshot use the cloned view + the main
    /// file; pending writes from a concurrent `WriteTxn` and frames
    /// committed AFTER the snapshot was taken are invisible.
    ///
    /// On in-memory pagers (no WAL) the snapshot captures `pinned_lsn
    /// = 0` and an empty frozen view — every read falls through to
    /// the main backend.
    ///
    /// The returned [`ReaderSnapshot`] registers a pin in the pager's
    /// live-snapshots map and removes the pin on drop.  Checkpoint
    /// consults `snapshots.values().min()` when deciding whether it
    /// is safe to reclaim WAL frames.
    ///
    /// # Writer-starvation interaction (single long-lived reader)
    ///
    /// The pin is what keeps a reader's frames durable, but it is also
    /// a *liveness hazard for writers*. While a snapshot pins an LSN
    /// below end-of-WAL, [`Self::checkpoint`] defers and reclaims
    /// **nothing** (see `checkpoint_deferred_for_pinned_reader`). The
    /// auto-checkpoint that `commit` triggers takes the same deferred
    /// path, so every subsequent commit keeps *appending* frames while
    /// none are folded back into the main file. The WAL therefore grows
    /// monotonically toward `WalConfig::size_limit` (default 64 MiB) for
    /// as long as the snapshot is held. Once the cap is reached,
    /// `WalTxn::append_raw` returns
    /// `Error::InvalidArgument("wal size limit exceeded")` and **every
    /// writer fails** until the snapshot is dropped and a checkpoint can
    /// reclaim the WAL.
    ///
    /// A single long-held `ReaderSnapshot` can thus wedge all writers.
    /// Callers MUST treat reader snapshots (and the `ReadTxn` that wraps
    /// one) as short-lived: take a snapshot, read what you need, and let
    /// it drop. Do not hold one open across an unbounded amount of write
    /// traffic. (Partial reclamation up to `min_pinned_lsn` — folding
    /// only frames the reader no longer needs — is **not** implemented:
    /// the WAL has no per-LSN partial-truncation primitive.
    /// `Wal::reset_after_checkpoint` resets the whole file with a fresh
    /// salt and `state.view` is a collapsed `PageId -> latest-version`
    /// map that retains no superseded versions, so there is no safe way
    /// to fold a post-snapshot overwrite without corrupting the pinned
    /// reader's view.)
    ///
    /// # Single-process isolation only
    ///
    /// The live-snapshots map is **this process's** `Pager` state. A
    /// reader in a *different* process has its own `Pager`, its own
    /// snapshots map, and its own pinned LSN — none of which are visible
    /// here. The cross-process `READER_LOCK` byte is shared among
    /// readers but is **not exclusive with writers**, so it does not
    /// defer another process's checkpoint either. Snapshot isolation
    /// (a reader observing a stable, consistent view across concurrent
    /// writes) is therefore guaranteed **only within a single process**.
    /// Across processes the contract is single-writer exclusion only: a
    /// writer/checkpoint in process A can fold WAL frames and rotate the
    /// salt while process B holds a reader, and process B's snapshot is
    /// not protected from that. See [`Self::checkpoint`].
    ///
    /// No `dyn`. The snapshot is generic over
    /// `F: FileBackend`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` only via underlying syscalls; the in-
    /// memory portion of this call cannot fail.
    pub fn reader_snapshot(&mut self) -> Result<ReaderSnapshot<F>> {
        let pinned_lsn = self
            .wal
            .as_ref()
            .map_or(Lsn::ZERO, |s| s.wal.next_lsn().prev_saturating());
        let frozen_view = self
            .wal
            .as_ref()
            .map(|s| s.view.clone())
            .unwrap_or_default();
        let frozen_header = self.wal.as_ref().and_then(|s| s.view_header.clone());
        let root_catalog = match self.wal.as_ref() {
            Some(state) => state.committed_root_catalog,
            None => self.header.root_catalog,
        };
        let snapshot_id = SnapshotId::new(self.next_snapshot_id.fetch_add(1, Ordering::Relaxed));
        let mut guard = self
            .snapshots
            .lock()
            .map_err(|_| Error::InvalidArgument("snapshot map poisoned"))?;
        debug_assert!(
            !guard.contains_key(&snapshot_id),
            "next_snapshot_id is monotonic; collisions are impossible",
        );
        guard.insert(snapshot_id, pinned_lsn);
        drop(guard);
        Ok(ReaderSnapshot {
            pinned_lsn,
            frozen_view,
            frozen_header,
            root_catalog,
            pin: SnapshotPin {
                id: snapshot_id,
                map: Arc::clone(&self.snapshots),
            },
            _phantom: std::marker::PhantomData,
        })
    }

    /// Snapshot the pager's in-memory header AND WAL committed
    /// view for txn-rollback purposes.  Returned to the caller and
    /// passed back into [`Self::restore_header_snapshot`] on
    /// rollback.
    ///
    /// The view is captured because [`Self::free_page`] removes
    /// per-page entries from the WAL view immediately (the page's
    /// committed content becomes stale once the id is back on the
    /// freelist).  Without snapshotting the view, a rolled-back txn
    /// that freed a page would leave readers no way to find the
    /// page's committed content — it sits below `state.view` (now
    /// missing the entry) and below the on-disk main file (never
    /// checkpointed).  Snapshot/restore closes that gap.
    ///
    /// Header fields (`root_catalog`, `freelist_head`,
    /// `page_count`) are written direct to disk (not through the
    /// WAL) so a pure pending-buffer discard leaves the header
    /// inconsistent with the rolled-back page bodies.  The
    /// snapshot/restore pair closes that gap for the
    /// `Db::transaction` rollback path.
    #[must_use]
    pub fn header_snapshot(&self) -> HeaderSnapshot {
        HeaderSnapshot {
            root_catalog: self.header.root_catalog,
            freelist_head: self.header.freelist_head,
            page_count: self.header.page_count,
            view: self
                .wal
                .as_ref()
                .map(|s| s.view.clone())
                .unwrap_or_default(),
        }
    }

    /// Restore the in-memory header AND WAL view from a
    /// previously-captured snapshot, then write the restored header
    /// to disk.  Used by [`crate::txn::WriteTxn::rollback`] to undo
    /// direct header writes + view mutations that happened during
    /// the rolled-back txn.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure when writing the
    /// restored header to disk.
    pub fn restore_header_snapshot(&mut self, snap: HeaderSnapshot) -> Result<()> {
        self.header.root_catalog = snap.root_catalog;
        self.header.freelist_head = snap.freelist_head;
        self.header.page_count = snap.page_count;
        if let Some(state) = self.wal.as_mut() {
            state.view = snap.view;
        }
        self.write_header()
    }

    /// Discard every page in the in-flight transaction buffer.
    /// Used by [`crate::txn::WriteTxn::rollback`].  Idempotent —
    /// calling on an in-memory pager or on a file pager with an
    /// empty pending buffer is a no-op.
    ///
    /// Also clears the header-dirty flag so a rolled-back
    /// `set_root_catalog` does not emit a stray page-0 frame on the
    /// next commit. The in-memory `self.header.root_catalog`
    /// restoration is the caller's job — `WriteTxn` uses
    /// [`Self::header_snapshot`] + [`Self::restore_header_snapshot`].
    pub fn rollback_pending_writes(&mut self) {
        if let Some(state) = self.wal.as_mut() {
            state.pending.clear();
            state.header_dirty = false;
        }
    }

    /// Number of live reader snapshots.  For diagnostics and tests.
    #[must_use]
    pub fn live_snapshot_count(&self) -> usize {
        self.snapshots.lock().map(|g| g.len()).unwrap_or_default()
    }

    /// Lowest LSN any live reader has pinned, or `None` if no
    /// snapshots are live.
    pub fn min_pinned_lsn(&self) -> Option<Lsn> {
        let guard = self.snapshots.lock().ok()?;
        guard.values().copied().min()
    }

    /// `true` iff this pager has no WAL — i.e. it was constructed
    /// via [`Pager::memory`]. In-memory pagers have no MVCC surface;
    /// a [`ReaderSnapshot`] against one reads the live cache rather
    /// than the (absent) WAL frozen view. Public so callers in
    /// peer crates (e.g. `Db::backup_to`) can dispatch on the
    /// in-memory case without reaching across the privacy boundary.
    #[must_use]
    pub fn is_memory_backed(&self) -> bool {
        self.wal.is_none()
    }

    /// Read page `id` consulting the cache first, then the main
    /// backend (no WAL overlay). Used by
    /// [`ReaderSnapshot::read_page`] on memory pagers, where the
    /// cache may be ahead of the in-memory backend buffer because
    /// memory pagers write to cache and only flush on eviction.
    /// The caller takes `&Pager`; no cache mutation occurs (a miss
    /// here is a `read_through`, not an insert).
    pub(crate) fn read_cache_or_main(&self, id: PageId) -> Result<Page> {
        debug_assert!(id.get() > 0);
        debug_assert!(id.get() < self.header.page_count);
        if id.get() >= self.header.page_count {
            return Err(Error::InvalidArgument("page id out of range"));
        }
        if let Some(page) = self.cache.peek(id) {
            return Ok(page.clone());
        }
        self.read_through(id)
    }

    /// Read the first [`PAGE_SIZE`] bytes from the main backend
    /// into `buf`, bypassing the cache and the WAL overlay.
    /// Used by [`crate::backup`] to capture page 0 (the file
    /// header) for inclusion in a backup. Errors with
    /// [`Error::BackupNotSupportedForMemoryPager`] on an in-memory
    /// pager (which has no on-disk file to read from).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure or
    /// [`Error::BackupNotSupportedForMemoryPager`] when the pager
    /// has no file backend.
    pub fn read_main_file_page_zero(&self, buf: &mut [u8; PAGE_SIZE]) -> Result<()> {
        match &self.backend {
            Backend::File(handle) => handle.read_exact_at(buf, 0),
            Backend::Memory(_) => Err(Error::BackupNotSupportedForMemoryPager),
        }
    }

    /// Read page `id` consulting ONLY the main backend (no WAL
    /// overlay, no cache).  Used by [`ReaderSnapshot::read_page`]
    /// when the frozen view does not contain the page.
    ///
    /// Internal to the snapshot path; the on-disk page's CRC32C
    /// trailer is verified before the page is returned.
    /// Read page `id` consulting ONLY the main backend (no WAL
    /// overlay, no cache). Verifies the on-disk page trailer
    /// before returning the bytes.
    ///
    /// Used by [`ReaderSnapshot::read_page`] when the frozen view
    /// does not contain the page, and by the [`crate::backup`]
    /// module to materialise the source's main-file pages into a
    /// destination backup file.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `id` is out of range.
    /// - [`Error::Io`] on syscall failure.
    /// - [`Error::Corruption`] if the on-disk trailer fails to
    ///   verify.
    pub fn read_main_file_page(&self, id: PageId) -> Result<Page> {
        debug_assert!(id.get() > 0, "PageId is non-zero by construction");
        debug_assert!(id.get() < self.header.page_count);
        if id.get() >= self.header.page_count {
            return Err(Error::InvalidArgument("page id out of range"));
        }
        self.read_through(id)
    }

    /// Read a freelist link page using the same WAL → cache → main
    /// priority chain that [`Self::read_page`] uses. Required
    /// because [`Self::free_page`] stages freelist pages in
    /// the WAL transaction buffer; a subsequent [`Self::alloc_page`]
    /// must observe the most-recent (possibly uncommitted) freelist
    /// link.
    fn read_freelist_page(&self, id: PageId) -> Result<Page> {
        if let Some(state) = self.wal.as_ref() {
            if let Some(p) = state
                .pending
                .get(&id)
                .or_else(|| state.view.get(&id).map(Arc::as_ref))
            {
                return Ok(p.clone());
            }
        }
        self.read_through(id)
    }

    fn alloc_from_freelist(&mut self, head: PageId) -> Result<PageId> {
        let head_page = self.read_freelist_page(head)?;
        let entry = decode_freelist_page(&head_page).ok_or(Error::Corruption {
            page_id: head.get(),
        })?;
        if entry.next != 0 && (entry.next == head.get() || entry.next >= self.header.page_count) {
            return Err(Error::Corruption {
                page_id: head.get(),
            });
        }
        self.header.freelist_head = entry.next;
        let _ = self.cache.evict(head);
        if let Some(state) = self.wal.as_mut() {
            state.pending.remove(&head);
        }
        self.stage_or_write_header()?;
        Ok(head)
    }

    fn alloc_fresh(&mut self) -> Result<PageId> {
        debug_assert!(
            self.in_txn(),
            "alloc_fresh must be inside a Pager txn (begin_txn/end_txn)"
        );
        let new_id_raw = self.header.page_count;
        let new_id =
            PageId::new(new_id_raw).ok_or(Error::InvalidArgument("page_count overflow"))?;
        if self.wal.is_some() {
            self.alloc_fresh_wal(new_id, new_id_raw)
        } else {
            self.alloc_fresh_memory(new_id, new_id_raw)
        }
    }

    /// File-backend fresh allocation. Stage a zeroed, trailer-
    /// stamped page into `state.pending` under `new_id` so it rides the
    /// SAME `WalTxn` as the page-0/`page_count` frame — one group-commit
    /// fsync covers both. The main file is NOT extended here and the body
    /// is NOT written to it; `read_page` resolves the slot from `pending`
    /// (and, post-commit, `view`) ahead of the main file, and
    /// `apply_checkpoint_view` grows the file + writes the body out at
    /// checkpoint. This deletes the past-EOF hazard at the root: the
    /// only durable record of a fresh page before checkpoint is its WAL
    /// frame, never an un-WAL'd main-file extension.
    fn alloc_fresh_wal(&mut self, new_id: PageId, new_id_raw: u64) -> Result<PageId> {
        let mut blank = Page::zeroed();
        write_page_trailer(&mut blank);
        self.header.page_count = new_id_raw
            .checked_add(1)
            .ok_or(Error::InvalidArgument("page_count overflow"))?;
        let state = self
            .wal
            .as_mut()
            .ok_or(Error::InvalidArgument("internal: wal overlay missing"))?;
        state.pending.insert(new_id, blank);
        debug_assert!(
            state.pending.contains_key(&new_id),
            "#91: fresh page must be staged in pending before commit",
        );
        self.stage_or_write_header()?;
        Ok(new_id)
    }

    /// In-memory fresh allocation.
    /// `extend_main_for` resizes the backing `Vec` per alloc, the cache
    /// is seeded dirty so the zeroed body lands on the next
    /// flush-eviction, and the blank body is written through (its trailer
    /// is stamped inside `write_back_page`). The in-memory backend has no
    /// durability surface, so none of the WAL/file-grow machinery
    /// applies.
    fn alloc_fresh_memory(&mut self, new_id: PageId, new_id_raw: u64) -> Result<PageId> {
        self.extend_main_for(new_id_raw)?;
        self.header.page_count = new_id_raw
            .checked_add(1)
            .ok_or(Error::InvalidArgument("page_count overflow"))?;
        let evicted = self.cache.insert(new_id, Page::zeroed(), true);
        self.handle_eviction(evicted)?;
        self.write_back_page(new_id, &Page::zeroed())?;
        self.stage_or_write_header()?;
        Ok(new_id)
    }

    /// In-memory fresh-alloc helper: grow the backing `Vec` by one
    /// physical stride so the slot `alloc_fresh_memory` is about to hand
    /// out exists. Memory-only by construction — the file backend routes
    /// fresh pages through the WAL and grows the file lazily in
    /// [`Self::apply_checkpoint_view`], so it never calls this. The
    /// in-memory `Vec` must stay sized exactly to `page_count` because
    /// reads index straight into it.
    ///
    /// A `debug_assert` confirms this is never
    /// reached on the file backend.
    fn extend_main_for(&mut self, new_id_raw: u64) -> Result<()> {
        let _ = new_id_raw;
        let stride = self.physical_stride();
        match &mut self.backend {
            Backend::Memory(bytes) => {
                bytes.resize(bytes.len() + stride, 0);
                Ok(())
            }
            Backend::File(_) => {
                debug_assert!(
                    false,
                    "#91: file backend extends the main file at checkpoint, not at alloc",
                );
                Err(Error::InvalidArgument(
                    "internal: extend_main_for on file backend",
                ))
            }
        }
    }

    /// Compute the on-disk file size for a main file whose top data
    /// slot index is `top_id_raw` — i.e. a file holding `top_id_raw + 1`
    /// pages total (page 0 plus slots `1..=top_id_raw`). Uses the
    /// encrypted physical stride (4136) when the file is
    /// encryption-capable; otherwise the legacy 4096-byte stride.
    /// Page 0 always contributes 4096 bytes. Used by
    /// [`Self::apply_checkpoint_view`] to size the single
    /// checkpoint `set_len`.
    fn file_length_for(&self, new_id_raw: u64) -> Result<u64> {
        let stride = self.physical_stride() as u64;
        let data_pages = new_id_raw;
        let data_bytes = data_pages
            .checked_mul(stride)
            .ok_or(Error::InvalidArgument("file too large"))?;
        (PAGE_SIZE as u64)
            .checked_add(data_bytes)
            .ok_or(Error::InvalidArgument("file too large"))
    }

    /// Inverse of [`Self::file_length_for`] — the number of pages
    /// (page 0 plus the data slots) physically present in a main file
    /// of `file_len` bytes. Used at open to seed
    /// [`Self::main_high_water`] from the real on-disk size so an
    /// already-extended file is not re-grown. Returns `0` for a file
    /// shorter than the header page (a brand-new / empty file), letting
    /// the first `alloc_fresh` grow from scratch. A partial trailing
    /// stride (torn extension) is floored — it does not count as a
    /// usable slot, so `alloc_fresh` will rewrite it cleanly.
    fn main_pages_for_len(&self, file_len: u64) -> u64 {
        let stride = self.physical_stride() as u64;
        if file_len < PAGE_SIZE as u64 {
            return 0;
        }
        let data_bytes = file_len - PAGE_SIZE as u64;
        1 + data_bytes / stride
    }

    /// `true` iff this pager was opened against
    /// a `format_minor >= 1` file (i.e. one whose per-page trailer
    /// uses the v1 interpretation and whose pages MAY be
    /// LZ4-compressed on disk). Cached at open time from the
    /// file header so the read/write hot path doesn't re-decode
    /// it per page.
    #[must_use]
    pub fn is_compression_capable(&self) -> bool {
        (self.header.feature_flags & FEATURE_FLAG_COMPRESSION) != 0
    }

    /// `true` iff this pager was opened against
    /// an encryption-capable file (`format_minor = 2` +
    /// `feature_flags` bit 1). Cached from the header at open so
    /// hot-path callers don't re-decode per page.
    #[must_use]
    pub fn is_encryption_capable(&self) -> bool {
        (self.header.feature_flags & FEATURE_FLAG_ENCRYPTION) != 0
    }

    /// Byte offset of page `id`'s on-disk slot
    /// in the main file. Uses the encrypted physical stride
    /// (4136 bytes) when the file is encryption-capable; otherwise
    /// the legacy 4096-byte stride. Page 0 is always at offset 0.
    #[must_use]
    fn physical_offset(&self, id: PageId) -> u64 {
        crate::pager::page::physical_offset_for(id.get(), self.header.feature_flags)
    }

    /// On-disk size of a single non-header page.
    /// Returns 4096 on unencrypted files, 4136 on encrypted ones.
    #[must_use]
    fn physical_stride(&self) -> usize {
        crate::pager::page::physical_page_stride(self.header.feature_flags)
    }

    /// Read straight from the backend without consulting the cache.
    /// Verifies the page trailer; page 0 is the caller's
    /// responsibility (the header carries its own checksum).
    ///
    /// On `format_minor >= 1` (compression-
    /// capable) files the on-disk trailer is v1 (1-bit flag +
    /// 31-bit CRC). CRC verification is **always** performed
    /// BEFORE any LZ4 decompression: malicious / corrupt input must not
    /// reach the decompressor without integrity first.
    fn read_through(&self, id: PageId) -> Result<Page> {
        let mut p = Page::zeroed();
        let off = self.physical_offset(id);
        if self.is_encryption_capable() {
            self.read_encrypted_into(id, off, &mut p)?;
        } else {
            self.read_plain_into(id, off, &mut p)?;
        }
        if self.is_compression_capable() {
            decode_page_v1(&p, id.get())
        } else {
            if !page_trailer_valid(&p) {
                return Err(Error::Corruption { page_id: id.get() });
            }
            Ok(p)
        }
    }

    /// Read a plaintext page (4096 bytes) into
    /// `p`. The non-encrypted physical path.
    fn read_plain_into(&self, id: PageId, off: u64, p: &mut Page) -> Result<()> {
        match &self.backend {
            Backend::File(handle) => handle.read_exact_at(p.as_bytes_mut(), off)?,
            Backend::Memory(bytes) => {
                let start =
                    usize::try_from(off).map_err(|_| Error::InvalidArgument("offset overflow"))?;
                let end = start
                    .checked_add(PAGE_SIZE)
                    .ok_or(Error::InvalidArgument("offset overflow"))?;
                if end > bytes.len() {
                    return Err(Error::Corruption { page_id: id.get() });
                }
                p.as_bytes_mut().copy_from_slice(&bytes[start..end]);
            }
        }
        Ok(())
    }

    /// Read an encrypted physical page (4136
    /// bytes) and decrypt it into `p` (4096 bytes of plaintext).
    /// Returns [`Error::EncryptionKeyInvalid`] if Poly1305 fails.
    fn read_encrypted_into(&self, id: PageId, off: u64, p: &mut Page) -> Result<()> {
        let stride = self.physical_stride();
        let mut phys = [0u8; PAGE_SIZE + ENCRYPTION_OVERHEAD];
        debug_assert_eq!(stride, phys.len(), "stride must match encrypted buffer");
        match &self.backend {
            Backend::File(handle) => handle.read_exact_at(&mut phys, off)?,
            Backend::Memory(bytes) => {
                let start =
                    usize::try_from(off).map_err(|_| Error::InvalidArgument("offset overflow"))?;
                let end = start
                    .checked_add(stride)
                    .ok_or(Error::InvalidArgument("offset overflow"))?;
                if end > bytes.len() {
                    return Err(Error::Corruption { page_id: id.get() });
                }
                phys.copy_from_slice(&bytes[start..end]);
            }
        }
        self.decrypt_physical(id, &phys, p)
    }

    /// Decrypt `phys` (4136 bytes) into `out`
    /// (4096 bytes). The cipher key must already be derived; if
    /// `self.derived_key` is `None` on an encryption-capable file
    /// we surface `EncryptionKeyRequired` rather than panic
    /// (this branch is unreachable at runtime — `derive_key_for_open`
    /// rejects that combination at open).
    fn decrypt_physical(
        &self,
        id: PageId,
        phys: &[u8; PAGE_SIZE + ENCRYPTION_OVERHEAD],
        out: &mut Page,
    ) -> Result<()> {
        let Some(key) = self.derived_key.as_ref() else {
            return Err(Error::EncryptionKeyRequired);
        };
        #[cfg(feature = "encryption")]
        {
            crate::crypto::decrypt_page(key.as_bytes(), id.get(), phys, out.as_bytes_mut())
        }
        #[cfg(not(feature = "encryption"))]
        {
            let _ = (id, phys, out, key);
            Err(Error::FormatFeatureUnsupported {
                feature: "encryption",
            })
        }
    }

    /// Write straight to the backend without going through the cache.
    /// Computes and stamps the page trailer before the write so that a
    /// read-back will always verify.
    ///
    /// On compression-capable files, the
    /// on-disk representation is produced by
    /// [`Self::encode_page_for_disk`] — LZ4 if it helps, raw
    /// otherwise. The in-memory `page` argument is always the
    /// 4092-byte raw body the encoders produced; compression is
    /// fully transparent at this layer.
    fn write_back_page(&mut self, id: PageId, page: &Page) -> Result<()> {
        let off = self.physical_offset(id);
        let stamped = self.encode_page_for_disk(page)?;
        if self.is_encryption_capable() {
            let phys = self.encrypt_logical(id, &stamped)?;
            self.write_phys_encrypted(off, &phys)?;
        } else {
            self.write_phys_4096(off, stamped.as_bytes())?;
        }
        Ok(())
    }

    /// Encrypt a stamped 4096-byte logical
    /// page into a 4136-byte physical block (ciphertext || nonce
    /// || tag) suitable for direct write-out.
    fn encrypt_logical(
        &self,
        id: PageId,
        page: &Page,
    ) -> Result<[u8; PAGE_SIZE + ENCRYPTION_OVERHEAD]> {
        let Some(key) = self.derived_key.as_ref() else {
            return Err(Error::EncryptionKeyRequired);
        };
        #[cfg(feature = "encryption")]
        {
            let mut out = [0u8; PAGE_SIZE + ENCRYPTION_OVERHEAD];
            crate::crypto::encrypt_page(
                key.as_bytes(),
                id.get(),
                page.as_bytes(),
                &mut out,
                self.entropy.as_ref(),
            )?;
            Ok(out)
        }
        #[cfg(not(feature = "encryption"))]
        {
            let _ = (id, page, key);
            Err(Error::FormatFeatureUnsupported {
                feature: "encryption",
            })
        }
    }

    /// Write a 4136-byte encrypted physical
    /// page at `off`.
    fn write_phys_encrypted(
        &mut self,
        off: u64,
        phys: &[u8; PAGE_SIZE + ENCRYPTION_OVERHEAD],
    ) -> Result<()> {
        let stride = PAGE_SIZE + ENCRYPTION_OVERHEAD;
        match &mut self.backend {
            Backend::File(handle) => handle.write_all_at(phys, off)?,
            Backend::Memory(bytes) => {
                let start =
                    usize::try_from(off).map_err(|_| Error::InvalidArgument("offset overflow"))?;
                let end = start
                    .checked_add(stride)
                    .ok_or(Error::InvalidArgument("offset overflow"))?;
                if end > bytes.len() {
                    bytes.resize(end, 0);
                }
                bytes[start..end].copy_from_slice(phys);
            }
        }
        Ok(())
    }

    /// Write a 4096-byte plain physical page at `off`.
    fn write_phys_4096(&mut self, off: u64, page_bytes: &[u8; PAGE_SIZE]) -> Result<()> {
        match &mut self.backend {
            Backend::File(handle) => handle.write_all_at(page_bytes, off)?,
            Backend::Memory(bytes) => {
                let start =
                    usize::try_from(off).map_err(|_| Error::InvalidArgument("offset overflow"))?;
                let end = start
                    .checked_add(PAGE_SIZE)
                    .ok_or(Error::InvalidArgument("offset overflow"))?;
                if end > bytes.len() {
                    bytes.resize(end, 0);
                }
                bytes[start..end].copy_from_slice(page_bytes);
            }
        }
        Ok(())
    }

    /// Produce the on-disk representation of a
    /// raw 4092-byte page body.
    ///
    /// - On `format_minor = 0` files (uncompressed): stamp the v0
    ///   32-bit CRC32C trailer.
    /// - On `format_minor = 1` files (compression-capable): try
    ///   LZ4 compress. If the compressed body fits in
    ///   `PAGE_SIZE - PAGE_TRAILER_SIZE - 2` bytes (= 4090), emit
    ///   the compressed layout (`u16 LE compressed_len` + LZ4
    ///   bytes + zero padding + v1 trailer with flag = 1).
    ///   Otherwise emit the raw body + v1 trailer with flag = 0.
    fn encode_page_for_disk(&self, page: &Page) -> Result<Page> {
        let mut stamped = page.clone();
        if !self.is_compression_capable() {
            write_page_trailer(&mut stamped);
            return Ok(stamped);
        }
        encode_page_v1(page)
    }

    fn handle_eviction(&mut self, evicted: Option<Evicted>) -> Result<()> {
        if let Some(ev) = evicted {
            if ev.dirty {
                self.write_back_page(ev.page_id, &ev.buffer)?;
            }
        }
        Ok(())
    }

    /// Re-encode and write page 0.
    fn write_header(&mut self) -> Result<()> {
        let mut p = Page::zeroed();
        encode_header(&self.header, &mut p);
        match &mut self.backend {
            Backend::File(handle) => handle.write_all_at(p.as_bytes(), 0)?,
            Backend::Memory(bytes) => {
                if bytes.len() < PAGE_SIZE {
                    bytes.resize(PAGE_SIZE, 0);
                }
                bytes[..PAGE_SIZE].copy_from_slice(p.as_bytes());
            }
        }
        Ok(())
    }

    /// Route a [`Self::set_root_catalog`] header update through
    /// the WAL on file-backed pagers; preserve the direct-write
    /// behavior on the in-memory pager (no WAL exists). Marks the
    /// header dirty so [`Self::commit`] knows to append a page-0
    /// frame.
    fn stage_or_write_header(&mut self) -> Result<()> {
        match self.wal.as_mut() {
            Some(state) => {
                state.header_dirty = true;
                Ok(())
            }
            None => self.write_header(),
        }
    }

    /// Mark the start of a WAL transaction. Called by
    /// [`crate::txn::WriteTxn::begin`]. The pager tracks a depth
    /// counter so a future nested-txn API can bump/decrement
    /// without breaking the Catalog's debug-assert. For the in-
    /// memory pager this is a no-op (no WAL transactional surface).
    pub fn begin_txn(&mut self) {
        if let Some(state) = self.wal.as_mut() {
            state.txn_depth = state.txn_depth.saturating_add(1);
        }
    }

    /// Mark the end of a WAL transaction. Symmetric with
    /// [`Self::begin_txn`]; called by [`crate::txn::WriteTxn::commit`]
    /// and [`crate::txn::WriteTxn::rollback`] (and the implicit
    /// `Drop` rollback). Saturating-decrement so a stray end without
    /// a matching begin does not underflow.
    pub fn end_txn(&mut self) {
        if let Some(state) = self.wal.as_mut() {
            state.txn_depth = state.txn_depth.saturating_sub(1);
        }
    }

    /// `true` if the pager is currently inside a WAL
    /// transaction (file-backed) or is an in-memory pager (no WAL
    /// transactional surface — every mutation is immediately
    /// visible). Catalog mutations debug-assert this at their entry
    /// points so the direct-write bug class cannot regress.
    #[must_use]
    pub fn in_txn(&self) -> bool {
        match self.wal.as_ref() {
            Some(state) => state.txn_depth > 0,
            None => true,
        }
    }
}

/// Construct the in-memory header for a brand-new file. When
/// `mode == CompressionMode::Lz4` the new file
/// gets `format_minor = 1` and `feature_flags` bit 0 set;
/// otherwise it stays at the `format_minor = 0` layout.
/// Initialise a freshly-created file: write the default header at
/// offset 0 and `fsync`. The
/// [`CompressionMode`] knob picks the new file's `format_minor` +
/// `feature_flags`.
///
/// When `encryption_key` is `Some`, a fresh
/// 32-byte `kdf_salt` is drawn from the injected `Entropy` source
/// (production defaults to `OsEntropy`/OS CSPRNG) and stamped
/// into the header at offset 72..104; `format_minor` is bumped to
/// `2` and `feature_flags` bit 1 is set. Page 0 itself remains
/// plaintext (the salt MUST be readable by tooling that does not
/// have the key).
fn initialise_file<F: FileBackend>(
    handle: &F,
    compression_mode: CompressionMode,
    encryption_key: Option<&[u8; 32]>,
    entropy: &dyn Entropy,
) -> Result<FileHeader> {
    let header = build_new_file_header(compression_mode, encryption_key, entropy)?;
    let mut p = Page::zeroed();
    encode_header(&header, &mut p);
    handle.set_len(PAGE_SIZE as u64)?;
    handle.write_all_at(p.as_bytes(), 0)?;
    handle.sync_all()?;
    Ok(header)
}

/// Pick the right header for a brand-new file
/// given the (`compression_mode`, `encryption_key`) tuple.
fn build_new_file_header(
    compression_mode: CompressionMode,
    encryption_key: Option<&[u8; 32]>,
    entropy: &dyn Entropy,
) -> Result<FileHeader> {
    match (compression_mode, encryption_key) {
        (CompressionMode::Off, None) => Ok(FileHeader::new_empty()),
        (CompressionMode::Lz4, None) => Ok(FileHeader::new_empty_with_compression()),
        (CompressionMode::Off, Some(_)) => {
            let salt = fresh_kdf_salt(entropy)?;
            Ok(FileHeader::new_empty_with_encryption(salt))
        }
        (CompressionMode::Lz4, Some(_)) => {
            let salt = fresh_kdf_salt(entropy)?;
            Ok(FileHeader::new_empty_with_encryption_and_compression(salt))
        }
    }
}

/// Pull 32 bytes from the injected [`Entropy`] into a fresh KDF salt.
/// The source itself handles the OS `getrandom` / `rand` split (see
/// [`OsEntropy`]); this function stays feature-agnostic so the open
/// path can pattern-match on `encryption_key.is_some()` without a
/// `#[cfg]` at every call site. The `Result` is retained (always
/// `Ok`) to keep [`build_new_file_header`]'s signature stable.
// allow: entropy fill is infallible, but the Ok-only wrap keeps the
// build_new_file_header / initialise_file call chain uniformly fallible.
#[allow(clippy::unnecessary_wraps)]
fn fresh_kdf_salt(entropy: &dyn Entropy) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    entropy.fill_bytes(&mut out);
    Ok(out)
}

/// Given the persisted on-disk header and the
/// caller's `Config`, derive the page-encryption key (or fall
/// back to `None`). This is the single function that maps the
/// open-time error matrix:
///
/// | File state | Build feature | Key | Result |
/// |---|---|---|---|
/// | minor < 2 | any | None | Ok(None) |
/// | minor < 2 | any | Some | Err(EncryptionKeyMismatch) |
/// | minor = 2 (bit 1) | OFF | any | Err(FormatFeatureUnsupported) |
/// | minor = 2 (bit 1) | ON | None | Err(EncryptionKeyRequired) |
/// | minor = 2 (bit 1) | ON | Some | Ok(Some(derive(key, kdf_salt))) |
fn derive_key_for_open(config: &Config, header: &FileHeader) -> Result<Option<PageEncryptionKey>> {
    let file_is_encrypted = (header.feature_flags & FEATURE_FLAG_ENCRYPTION) != 0;
    let has_feature = cfg!(feature = "encryption");
    match (file_is_encrypted, has_feature, config.master_key()) {
        (false, _, None) => Ok(None),
        (false, _, Some(_)) => Err(Error::EncryptionKeyMismatch),
        (true, false, _) => Err(Error::FormatFeatureUnsupported {
            feature: "encryption",
        }),
        (true, true, None) => Err(Error::EncryptionKeyRequired),
        // allow: `user_key` is unused on the no-encryption build, where this arm's body discards it; it is consumed only under `cfg(feature = "encryption")`.
        #[allow(unused_variables)]
        (true, true, Some(user_key)) => {
            #[cfg(feature = "encryption")]
            {
                let derived = crate::crypto::derive_page_key(user_key, &header.kdf_salt)?;
                Ok(Some(PageEncryptionKey(wrap_master_key(derived))))
            }
            #[cfg(not(feature = "encryption"))]
            {
                let _ = user_key;
                Err(Error::FormatFeatureUnsupported {
                    feature: "encryption",
                })
            }
        }
    }
}

/// Reject opens that require features the
/// running binary was not compiled with. Today there is exactly
/// one such gate (`compression`); the function is structured so
/// future features (encryption, alternate codecs) plug in beside
/// it without touching the call site.
fn refuse_unsupported_features(header: &FileHeader) -> Result<()> {
    let uses_compression = (header.feature_flags & FEATURE_FLAG_COMPRESSION) != 0;
    if uses_compression && !cfg!(feature = "compression") {
        return Err(Error::FormatFeatureUnsupported {
            feature: "compression",
        });
    }
    let uses_encryption = (header.feature_flags & FEATURE_FLAG_ENCRYPTION) != 0;
    if uses_encryption && !cfg!(feature = "encryption") {
        return Err(Error::FormatFeatureUnsupported {
            feature: "encryption",
        });
    }
    Ok(())
}

/// Refuse to create new compression-capable
/// files when the build lacks the `compression` Cargo feature.
/// Without this guard, the open would happily produce a
/// `format_minor = 1` file the same build cannot reopen.
fn refuse_compression_without_feature(mode: CompressionMode) -> Result<()> {
    if matches!(mode, CompressionMode::Lz4) && !cfg!(feature = "compression") {
        return Err(Error::FormatFeatureUnsupported {
            feature: "compression",
        });
    }
    Ok(())
}

/// Refuse to create new encryption-capable
/// files when the build lacks the `encryption` Cargo feature.
fn refuse_encryption_without_feature(has_key: bool) -> Result<()> {
    if has_key && !cfg!(feature = "encryption") {
        return Err(Error::FormatFeatureUnsupported {
            feature: "encryption",
        });
    }
    Ok(())
}

/// `true` iff this build was compiled with the
/// `encryption` Cargo feature. Exposed for diagnostic tooling and
/// integration tests that need to dispatch on the feature without
/// embedding a `cfg!` of their own.
#[must_use]
pub const fn encryption_feature_compiled_in() -> bool {
    cfg!(feature = "encryption")
}

/// Extracted from [`Pager::open_with_backends`] so that function
/// stays within its line budget. Either
/// rolls a recovered WAL forward into a `WalState` + view, or
/// creates a fresh WAL and stamps its salt into the main file
/// header.
///
/// Rationale for the `clippy::type_complexity` allow: the tuple
/// return shape is local to this single call site and mirrors
/// the immediate `(wal, view, view_header)` destructuring in
/// [`Pager::open_with_backends`]. Naming the tuple would be more
/// machinery than it deserves for a private helper that exists
/// solely to keep that function within its line budget.
#[allow(clippy::type_complexity)]
fn recover_or_create_wal<F: FileBackend>(
    main: &F,
    wal: F,
    wal_path: std::path::PathBuf,
    header: &mut FileHeader,
    config: &Config,
    derived_key: Option<&PageEncryptionKey>,
    entropy: Arc<dyn Entropy>,
) -> Result<(Wal<F>, HashMap<PageId, Page>, Option<Page>)> {
    let expected_salt = salt_from_header(header);
    let wal_key_bytes = derived_key.map(|k| *k.as_bytes());
    let recovered = Wal::<F>::open_for_recovery_with_key(
        &wal,
        expected_salt,
        config.wal_size_limit,
        wal_key_bytes,
    )?;
    if recovered.committed_frames > 0 {
        let mut w = Wal::<F>::from_recovered_meta(
            wal,
            wal_path,
            recovered.salt,
            recovered.next_lsn,
            recovered.end_offset,
            recovered.committed_frames,
            config.wal_config(),
            entropy,
        );
        w.set_key(wal_key_bytes);
        let recovered_header = recovered.header.clone();
        if let Some(hp) = &recovered_header {
            let decoded = decode_header(hp)?;
            header.root_catalog = decoded.root_catalog;
            header.freelist_head = decoded.freelist_head;
            header.page_count = decoded.page_count;
        }
        Ok((w, recovered.into_view(), recovered_header))
    } else {
        let mut w = Wal::<F>::create_fresh_with(wal, wal_path, config.wal_config(), entropy)?;
        w.set_key(wal_key_bytes);
        stamp_salt_into_header(header, w.salt());
        write_header_to_backend(main, header)?;
        main.sync_data(config.sync_mode)?;
        Ok((w, HashMap::new(), None))
    }
}

/// Byte offset of the trailer within a page.
/// Equal to `PAGE_SIZE - PAGE_TRAILER_SIZE = 4092` and matches the
/// constant of the same name in `pager::checksum`. Declared here
/// too so the encode/decode helpers in this module can stay
/// independent of the `checksum` module's private constants.
const V1_BODY_END: usize = PAGE_SIZE - PAGE_TRAILER_SIZE;

/// Max LZ4 compressed-payload size that fits
/// inside the page body alongside the 2-byte length prefix.
/// Equal to `V1_BODY_END - 2 = 4090`.
const V1_MAX_COMPRESSED_LEN: usize = V1_BODY_END - 2;

/// Encode a raw 4092-byte page body for
/// on-disk storage in a `format_minor = 1` (compression-capable)
/// file.
///
/// Tries LZ4 compression. If the compressed payload fits in
/// [`V1_MAX_COMPRESSED_LEN`] (= 4090) bytes the layout is
/// `[u16 LE compressed_len][LZ4 bytes][zero pad to 4092][v1
/// trailer flag=1, 31-bit CRC]`. Otherwise the layout is
/// `[raw 4092-byte body][v1 trailer flag=0, 31-bit CRC]`.
///
/// Without the `compression` Cargo feature this function emits
/// the uncompressed layout unconditionally — the `Pager::open`
/// refusal in [`refuse_unsupported_features`] guarantees we
/// never reach this code path against a `format_minor >= 1`
/// file in that build configuration, but defending in depth is
/// cheap.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] if the LZ4-compressed
/// length somehow exceeds the 2-byte length prefix's range, rather
/// than writing a self-corrupting length-0 page. The
/// `V1_MAX_COMPRESSED_LEN` guard makes this unreachable today.
#[cfg_attr(not(feature = "compression"), allow(clippy::unnecessary_wraps))]
fn encode_page_v1(page: &Page) -> Result<Page> {
    #[cfg(feature = "compression")]
    {
        let max_out = lz4_flex::block::get_maximum_output_size(V1_BODY_END);
        let mut scratch = [0u8; 8192];
        if max_out > scratch.len() {
        } else {
            let raw = &page.as_bytes()[..V1_BODY_END];
            if let Ok(compressed_len) = lz4_flex::block::compress_into(raw, &mut scratch[..max_out])
            {
                if compressed_len > 0 && compressed_len <= V1_MAX_COMPRESSED_LEN {
                    let mut out = Page::zeroed();
                    let buf = out.as_bytes_mut();
                    let len_u16 = u16::try_from(compressed_len).map_err(|_| {
                        Error::InvalidArgument(
                            "encode_page_v1: compressed length exceeds u16 length prefix",
                        )
                    })?;
                    buf[0..2].copy_from_slice(&len_u16.to_le_bytes());
                    buf[2..2 + compressed_len].copy_from_slice(&scratch[..compressed_len]);
                    write_page_trailer_v1(&mut out, true);
                    return Ok(out);
                }
            }
        }
    }
    let mut out = page.clone();
    write_page_trailer_v1(&mut out, false);
    Ok(out)
}

/// Decode a 4096-byte on-disk page from a
/// `format_minor = 1` file into its raw 4092-byte body
/// representation (with a v0 trailer re-stamped so downstream
/// consumers that call [`page_trailer_valid`] continue to work).
///
/// Verifies the v1 trailer FIRST (CRC before
/// decompress); a CRC mismatch returns
/// `Error::Corruption { page_id }` without invoking LZ4. The
/// LZ4 decompress is bounded to a fixed 4092-byte output buffer;
/// any size mismatch is `Error::Corruption`.
///
/// Public for fuzz / differential testing of the decode path.
/// Application code SHOULD reach for [`Pager::read_page`] instead.
///
/// # Errors
///
/// - [`Error::Corruption`] with the supplied `page_id` if the v1
///   trailer's 31-bit CRC does not match the recomputed CRC, if
///   the compressed-length prefix is out of range, or if the
///   LZ4 output is not exactly 4092 bytes.
/// - [`Error::FormatFeatureUnsupported`] (when the `compression`
///   Cargo feature is OFF and the trailer's compression flag is
///   set — unreachable on a well-formed open path, which refuses
///   such files at `Db::open`).
pub fn decode_page_v1(disk: &Page, page_id: u64) -> Result<Page> {
    if !page_trailer_valid_v1(disk) {
        return Err(Error::Corruption { page_id });
    }
    if !crate::pager::checksum::page_trailer_flag_v1(disk) {
        let mut out = Page::zeroed();
        out.as_bytes_mut()[..V1_BODY_END].copy_from_slice(&disk.as_bytes()[..V1_BODY_END]);
        write_page_trailer(&mut out);
        return Ok(out);
    }
    decode_compressed_page_v1(disk, page_id)
}

/// Decompress a `format_minor = 1` page
/// whose flag bit is set. Split out so the gated `compression`
/// feature is localised to a single function; without the
/// feature the only way to reach this code is a malformed file
/// the open-time refusal already rejected, but we keep the
/// shape consistent.
fn decode_compressed_page_v1(disk: &Page, page_id: u64) -> Result<Page> {
    let body = &disk.as_bytes()[..V1_BODY_END];
    let mut len_buf = [0u8; 2];
    len_buf.copy_from_slice(&body[0..2]);
    let compressed_len = usize::from(u16::from_le_bytes(len_buf));
    if compressed_len == 0 || compressed_len > V1_MAX_COMPRESSED_LEN {
        return Err(Error::Corruption { page_id });
    }
    #[cfg(feature = "compression")]
    {
        let input = &body[2..2 + compressed_len];
        let mut out = Page::zeroed();
        let decompressed = {
            let dest = &mut out.as_bytes_mut()[..V1_BODY_END];
            lz4_flex::block::decompress_into(input, dest)
                .map_err(|_| Error::Corruption { page_id })?
        };
        if decompressed != V1_BODY_END {
            return Err(Error::Corruption { page_id });
        }
        write_page_trailer(&mut out);
        Ok(out)
    }
    #[cfg(not(feature = "compression"))]
    {
        let _ = compressed_len;
        let _ = body;
        Err(Error::FormatFeatureUnsupported {
            feature: "compression",
        })
    }
}

/// Read and decode page 0 from an existing file.
fn load_header<F: FileBackend>(handle: &F) -> Result<FileHeader> {
    let len = handle.len()?;
    if len < PAGE_SIZE as u64 {
        return Err(Error::InvalidFormat {
            reason: "file is shorter than one page",
        });
    }
    let mut p = Page::zeroed();
    handle.read_exact_at(p.as_bytes_mut(), 0)?;
    decode_header(&p)
}

/// Read the WAL generation salt from the main file's `wal_salt`
/// field. The first four bytes carry the current generation salt;
/// the remaining 12 are reserved (zero in format-major 0).
fn salt_from_header(header: &FileHeader) -> u32 {
    u32::from_le_bytes([
        header.wal_salt[0],
        header.wal_salt[1],
        header.wal_salt[2],
        header.wal_salt[3],
    ])
}

/// Stamp `salt` into the first four bytes of `header.wal_salt`. The
/// remaining 12 bytes are zeroed (format-major 0 reserves them).
fn stamp_salt_into_header(header: &mut FileHeader, salt: u32) {
    let bytes = salt.to_le_bytes();
    header.wal_salt = [0u8; 16];
    header.wal_salt[0..4].copy_from_slice(&bytes);
}

/// Re-encode and write the header to any file backend. Used during
/// open when we cannot yet borrow `&mut self` (the pager is still
/// being constructed).
fn write_header_to_backend<F: FileBackend>(handle: &F, header: &FileHeader) -> Result<()> {
    let mut p = Page::zeroed();
    encode_header(header, &mut p);
    handle.write_all_at(p.as_bytes(), 0)
}

/// Construct the WAL sidecar path for a given main-file path. We
/// append `-wal` to the file name (mirroring `SQLite`'s convention);
/// the WAL lives next to the main file.
///
/// Exposed for integration tests and tooling that need to inspect or
/// manipulate the sidecar directly (e.g. the crash-cycle fault
/// harness).
#[must_use]
pub fn wal_path_for(main: &Path) -> PathBuf {
    let mut buf = main.as_os_str().to_os_string();
    buf.push("-wal");
    PathBuf::from(buf)
}

/// Construct the cross-process lock sidecar path for a given
/// main-file path. We append `-lock` to the file name (mirroring
/// the `<db>-wal` sidecar convention); the lock file lives next
/// to the main DB and is the byte-range target for `WRITER_LOCK`
/// / `READER_LOCK_RANGE` (see `platform::lock`).
///
/// Keeping the lock byte in a dedicated file prevents
/// pager I/O on the main DB from ever overlapping the locked
/// byte range. With the sidecar, the lock handle and the pager
/// handle target different files, so the failure mode cannot recur.
#[must_use]
pub fn lock_path_for(main: &Path) -> PathBuf {
    let mut buf = main.as_os_str().to_os_string();
    buf.push("-lock");
    PathBuf::from(buf)
}

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "fault-injection"))]
#[cfg(test)]
mod tests_fault;
