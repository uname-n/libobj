//! `Db` open-time configuration.
//!
//! A thin wrapper around `obj_core::pager::Config` plus the
//! `busy_timeout` knob.

use std::time::Duration;

use obj_core::pager::CompressionMode;
use obj_core::pager::Config as PagerConfig;
use obj_core::SyncMode;

/// Upper bound on the LRU cache size, expressed in 4 KiB frames.
/// [`Config::cache_size`] clamps any request above this ceiling down
/// to it rather than erroring. `4_194_304` frames × 4 KiB = 16 GiB —
/// far above any realistic working set, but bounded so a bogus
/// `usize::MAX` byte count cannot ask the pager to pre-size an
/// absurd cache.
pub const MAX_CACHE_FRAMES: usize = 4_194_304;

/// `Db` open-time configuration.  Construct via [`Config::default`]
/// and modify with the builder methods.
///
/// `Debug` is implemented manually so the embedded
/// `pager.encryption_key` field never leaks key material — the
/// derived `Debug` on the pager's `Config` already redacts it, but
/// implementing it manually here keeps the redaction story local
/// for the obj-rs crate as well.
///
/// # Examples
///
/// Chain the setters from [`Config::default`] and hand the result
/// to [`Db::open_with`](crate::Db::open_with):
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::{Config, Db, SyncMode};
/// use std::time::Duration;
///
/// let dir = tempfile::tempdir()?;
///
/// let cfg = Config::default()
///     // Cache size in bytes. Rounded down to whole 4 KiB pages and
///     // clamped into range. Default: 256 KiB (64 frames).
///     .cache_size(64 * 1024 * 1024)
///     // Durability mode used by the WAL on every commit.
///     // Default: SyncMode::Full (survives system-wide power loss).
///     .sync_mode(SyncMode::Full)
///     // Maximum wait when acquiring the writer / reader lock.
///     // Default: 5 seconds. Beyond the budget, the txn returns
///     // `Err(Error::Busy)` rather than blocking indefinitely.
///     .busy_timeout(Duration::from_secs(2))
///     // Skip the open-time catalog walk. Default: false. Production
///     // callers should leave this alone.
///     .skip_open_check(false)
///     // Cross-process file locking. Default: true.
///     .cross_process_lock(true);
///
/// let _db = Db::open_with(dir.path().join("configured.obj"), cfg)?;
/// # Ok(())
/// # }
/// ```
///
/// Quick reference for when to change each knob:
///
/// - [`Config::cache_size`] — bigger cache for read-heavy
///   workloads on large databases; tiny cache on
///   memory-constrained targets.
/// - [`Config::sync_mode`] — [`SyncMode::Normal`] if you accept
///   losing the last few milliseconds of writes on a power loss;
///   [`SyncMode::Off`] only for tests and benchmarks.
/// - [`Config::busy_timeout`] — shorter when the caller prefers a
///   fast `Error::Busy` to a long wait; longer when contention is
///   rare and you would rather block than retry.
/// - [`Config::skip_open_check`] — leave on in production. The
///   narrow use-cases are fault-injection harnesses, hot-reload
///   tooling that opens the same file many times per second, and
///   developer workflows that have just run a full
///   `integrity_check`.
/// - [`Config::cross_process_lock`] — leave on for any real
///   deployment. The off path is for in-process stress tests where
///   one shared `Db` serves many threads on a single fd.
///
/// # Performance tuning
///
/// The defaults favour durability and a small memory footprint, not
/// throughput. Three levers move the needle, in rough order of impact:
///
/// **1. Batch writes into one transaction.** Every committed write
/// transaction costs one WAL durability sync (under the default
/// [`SyncMode::Full`]). A single-document insert is therefore
/// dominated by that one sync — inserting N documents in N separate
/// transactions pays the sync N times. Inserting the same N documents
/// inside one [`Db::transaction`](crate::Db::transaction) closure pays
/// it **once**, which is why batched inserts are dramatically faster
/// per document. Prefer one transaction per logical unit of work, not
/// one per row. (Keep a single batch under the WAL size limit — very
/// large batches should be split into chunks of, say, a few thousand
/// documents.)
///
/// ```
/// # fn main() -> obj::Result<()> {
/// # use serde::{Serialize, Deserialize};
/// # #[derive(Serialize, Deserialize, obj::Document)]
/// # struct Row { n: u64 }
/// # let dir = tempfile::tempdir()?;
/// let db = obj::Db::open(dir.path().join("batched.obj"))?;
///
/// // One transaction, one sync — not 1_000 of them.
/// db.transaction(|tx| {
///     let coll = tx.collection::<Row>()?;
///     for n in 0..1_000 {
///         coll.insert(Row { n })?;
///     }
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// **2. Size the cache for the working set.** The default LRU cache is
/// intentionally small — 64 frames (256 KiB). That is fine for
/// write-mostly or memory-constrained workloads, but a read-heavy
/// service over a large database will repeatedly evict and re-read hot
/// pages. Raise [`Config::cache_size`] so the hot set (inner B-tree
/// nodes plus frequently-read leaves) stays resident; tens of MiB is a
/// reasonable starting point, scaling up toward your working-set size.
///
/// **3. Relax the sync mode only if your durability budget allows.**
/// [`SyncMode`] trades durability for commit latency:
///
/// - [`SyncMode::Full`] (default) — survives system-wide power loss;
///   the safe choice and the right default for production data.
/// - [`SyncMode::Normal`] — survives process crash and kernel panic,
///   but a sudden power loss can lose the last few committed
///   transactions if the drive's write cache has not flushed. Faster
///   commits; choose it only when that window is acceptable.
/// - [`SyncMode::Off`] — issues no durability call at all. Use **only**
///   for tests, benchmarks, and rebuildable scratch data where loss is
///   acceptable. Never for data you cannot recreate.
///
/// A throughput-tuned (but still process-crash-durable) config:
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::{Config, Db, SyncMode};
///
/// # let dir = tempfile::tempdir()?;
/// let cfg = Config::default()
///     .cache_size(64 * 1024 * 1024) // 64 MiB hot set
///     .sync_mode(SyncMode::Normal);  // crash-durable, faster commits
/// let _db = Db::open_with(dir.path().join("tuned.obj"), cfg)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config {
    pub(crate) pager: PagerConfig,
    pub(crate) busy_timeout: Duration,
    pub(crate) readonly: bool,
    pub(crate) cross_process_lock: bool,
    pub(crate) skip_open_check: bool,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("pager", &self.pager)
            .field("busy_timeout", &self.busy_timeout)
            .field("readonly", &self.readonly)
            .field("cross_process_lock", &self.cross_process_lock)
            .field("skip_open_check", &self.skip_open_check)
            .finish()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pager: PagerConfig::default(),
            busy_timeout: obj_core::DEFAULT_BUSY_TIMEOUT,
            readonly: false,
            cross_process_lock: true,
            skip_open_check: false,
        }
    }
}

impl Config {
    /// Set the pager's LRU cache size, in bytes.  Rounded down to
    /// the nearest 4 KiB page and clamped into the supported range.
    ///
    /// A `bytes` value smaller than one page is clamped UP to a
    /// single frame (the pager requires at least one); a value large
    /// enough to exceed `MAX_CACHE_FRAMES` is clamped DOWN to that
    /// ceiling. This keeps the builder infallible so it chains like
    /// every other setter — there is no out-of-range error to
    /// surface.
    #[must_use]
    pub fn cache_size(self, bytes: usize) -> Self {
        let frames = (bytes / obj_core::pager::PAGER_PAGE_SIZE).clamp(1, MAX_CACHE_FRAMES);
        let mut pager = self.pager;
        pager.cache_frames = frames;
        Self { pager, ..self }
    }

    /// Set the durability mode the WAL uses for every commit.
    #[must_use]
    pub fn sync_mode(self, mode: SyncMode) -> Self {
        Self {
            pager: self.pager.with_sync_mode(mode),
            ..self
        }
    }

    /// Set the cross-process / in-process busy-lock timeout.
    /// `WriteTxn::begin` and `ReadTxn::begin` return
    /// `Err(Error::Busy)` if the relevant lock cannot be acquired
    /// within this budget.
    #[must_use]
    pub fn busy_timeout(self, timeout: Duration) -> Self {
        Self {
            busy_timeout: timeout,
            ..self
        }
    }

    /// Skip the lightweight open-time integrity check.
    ///
    /// By default (`false`), [`crate::Db::open`] / [`crate::Db::open_with`]
    /// run a fast subset of [`crate::Db::integrity_check`] before
    /// returning: file-header CRC, catalog root sanity, catalog
    /// B-tree CRC + invariants, and per-collection pointer-range
    /// validation. The walk is bounded to the catalog tree only —
    /// no per-collection deep walk — so the cost is essentially
    /// independent of the database's total size.
    ///
    /// Set to `true` to opt out. The knob exists for narrow use
    /// cases — fault-injection harnesses that deliberately open a
    /// corrupted DB to exercise downstream error paths, hot-reload
    /// tooling that re-opens the same file many times per second,
    /// or developer workflows that have just run a full
    /// `Db::integrity_check` and don't want to repeat the catalog
    /// portion. Production callers SHOULD leave it on.
    ///
    /// Skipping the open check does NOT bypass detection: a
    /// corrupted page surfaces on the first operation that touches
    /// it. Note also that the obj `Db` constructor performs an
    /// implicit `Catalog::open_or_init` that reads the catalog
    /// B-tree's reserved row; a DB whose catalog tree is so
    /// corrupted that descend fails will still error out of the
    /// open path even with `skip_open_check(true)`. The knob's
    /// guarantee is "no EXTRA walk beyond what was already
    /// required to construct the Db handle."
    #[must_use]
    pub fn skip_open_check(self, skip: bool) -> Self {
        Self {
            skip_open_check: skip,
            ..self
        }
    }

    /// Select per-page compression for new files.
    ///
    /// `mode = CompressionMode::Lz4` causes [`crate::Db::open_with`]
    /// to create a brand-new database file at `format_minor = 2`
    /// (the v1.0 feature-complete minor; the LZ4 layer is signalled
    /// by a per-page flag bit, not by the minor version) with the
    /// LZ4 page-compression layer engaged. Pages are
    /// compressed at the pager layer only — every higher-level
    /// encoder (B-tree, freelist, catalog, document) still
    /// operates on the 4092-byte logical body. Compression is
    /// fully transparent to user code.
    ///
    /// **No-op against existing files.** When the database file
    /// already exists, its on-disk header dictates whether
    /// compression is in use; this knob is consulted only on
    /// file creation. Opening an existing `format_minor = 0`
    /// (uncompressed) file with
    /// `Config::compression(CompressionMode::Lz4)` does NOT
    /// upgrade the file; reads and writes continue to use the
    /// uncompressed layout. Migrating an existing database to
    /// compression is deferred to a future tool.
    ///
    /// **Build-time requirement.** Compression requires the
    /// `compression` Cargo feature on the `obj-rs` crate (which
    /// in turn enables `obj-core/compression`). A build WITHOUT
    /// the feature that calls `Config::compression(Lz4)` will
    /// return `Error::FormatFeatureUnsupported { feature:
    /// "compression" }` from `Db::open_with` at the moment a
    /// new file would otherwise be created.
    #[must_use]
    pub fn compression(self, mode: CompressionMode) -> Self {
        Self {
            pager: self.pager.with_compression_mode(mode),
            ..self
        }
    }

    /// Supply a 32-byte master encryption key.
    ///
    /// When set on a **new** database file, the file is created at
    /// `format_minor = 2` with `feature_flags` bit 1 set and a fresh
    /// CSPRNG-generated `kdf_salt` stored plaintext in the page-0
    /// header. Every non-header page is encrypted with
    /// XChaCha20-Poly1305: 4096-byte logical page → 4136-byte
    /// physical page (24-byte nonce + 16-byte tag).
    ///
    /// When set on an **existing** database file:
    /// - If the file is `format_minor = 2` (encryption-capable):
    ///   the key is used to derive the per-file page key via
    ///   `HKDF-SHA256(key, kdf_salt, b"obj-page-encryption-v1")`.
    ///   A wrong key surfaces as
    ///   [`Error::EncryptionKeyInvalid`](obj_core::Error::EncryptionKeyInvalid)
    ///   on the first encrypted page read.
    /// - If the file is `format_minor < 2`: open returns
    ///   [`Error::EncryptionKeyMismatch`](obj_core::Error::EncryptionKeyMismatch).
    ///
    /// **Build-time requirement.** Encryption requires the
    /// `encryption` Cargo feature on the `obj-rs` crate (which
    /// propagates to `obj-core/encryption`). A build WITHOUT the
    /// feature that sets a key returns
    /// [`Error::FormatFeatureUnsupported`](obj_core::Error::FormatFeatureUnsupported)
    /// from `Db::open_with` (`feature = "encryption"`).
    ///
    /// The key is held in memory inside the obj-core pager
    /// [`Config`](obj_core::pager::Config); the `Debug` impl
    /// redacts it (`encryption_key: "<set>"`). The key is NOT
    /// persisted to disk. Callers are responsible for the master
    /// key's lifecycle (storage, rotation, derivation from a
    /// passphrase via Argon2/scrypt, etc.).
    #[must_use]
    pub fn encryption_key(self, key: [u8; 32]) -> Self {
        Self {
            pager: self.pager.with_encryption_key(Some(key)),
            ..self
        }
    }

    /// Enable / disable the cross-process file lock layer.
    ///
    /// When `false`, the [`Db`](crate::Db) opened with this config
    /// does NOT acquire OS-level byte-range locks on the database
    /// file.  Used by the concurrent stress test where every
    /// thread shares one `Db` (and therefore one file descriptor):
    /// POSIX OFD locks are per-fd, so multiple threads on the same
    /// fd cannot use the lock to enforce inter-thread exclusion —
    /// that's what the in-process write-serialization mutex is for.
    ///
    /// Default: `true` (cross-process locking enabled).
    #[must_use]
    pub fn cross_process_lock(self, enabled: bool) -> Self {
        Self {
            cross_process_lock: enabled,
            ..self
        }
    }
}
