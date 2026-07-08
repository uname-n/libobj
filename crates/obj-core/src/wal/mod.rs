//! Write-ahead log (L2).
//!
//! The WAL is the durability layer that sits between the pager and the
//! main file. Writes go to an append-only sidecar (`<main>-wal`) first;
//! a checkpoint later rolls them into the main file.
//! Recovery / replay on open is implemented by
//! [`Wal::open_for_recovery`].

#![forbid(unsafe_code)]

pub mod frame;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::pager::page::{Page, PageId, PAGE_SIZE};
use crate::platform::{remove_file_if_exists, FileBackend, FileHandle, SyncMode};
use crate::wal::frame::{
    decode_frame_header_classified, encode_frame_header, frame_size_for, FrameDecode, FrameHeader,
    FRAME_HEADER_SIZE, FRAME_SIZE, WAL_HEADER_SIZE, WAL_MAGIC,
};
#[cfg(feature = "encryption")]
use crate::wal::frame::{FRAME_AEAD_SUFFIX_SIZE, FRAME_SIZE_ENCRYPTED};

/// Log sequence number.
///
/// Monotonically increasing within a single WAL generation; reset to
/// zero across checkpoints (the salt rotation disambiguates). The
/// sentinel value [`Lsn::ZERO`] represents "no LSN" — returned by
/// [`crate::pager::Pager::commit`] for an empty transaction and by
/// [`crate::pager::Pager::reader_snapshot`] for in-memory pagers
/// that have no WAL.
///
/// `Lsn` is a `#[repr(transparent)]` newtype over `u64` so the
/// type-system rejects implicit confusion with page counts, byte
/// offsets, or page ids. The serde encoding is
/// `#[serde(transparent)]` — an `Lsn` round-trips byte-identically to
/// the bare `u64` it wraps, which preserves wire compatibility with
/// any future on-disk record that names it directly.
///
/// `Lsn` deliberately does NOT implement `Add<u64>` / `AddAssign<u64>`
/// or any other arithmetic trait. Step it through the explicit
/// [`Lsn::checked_next`] / [`Lsn::prev_saturating`] helpers so every
/// mutation is auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Lsn(u64);

impl Lsn {
    /// The sentinel "no LSN" value. Returned by
    /// [`crate::pager::Pager::commit`] when the transaction was empty
    /// and by [`crate::pager::Pager::reader_snapshot`] on in-memory
    /// pagers (no WAL exists).
    pub const ZERO: Self = Self(0);

    /// The LSN handed out for the first frame of a fresh WAL
    /// generation.
    pub const ONE: Self = Self(1);

    /// Construct an [`Lsn`] from a raw `u64`. The underlying `u64`
    /// has no invariants — any value (including `0`) is valid —
    /// so this is a total function.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw `u64` LSN value. Use this only when crossing into
    /// hand-rolled byte serialization (see
    /// [`crate::wal::frame::FrameHeader::lsn`]) or when emitting
    /// diagnostics; arithmetic should go through the explicit step
    /// helpers below.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Monotonic step: return the next LSN, or [`Error::InvalidArgument`]
    /// on `u64` overflow.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the underlying counter
    /// would wrap past `u64::MAX`. At 10⁶ commits/sec this is ~584 000
    /// years; the check is defensive and extremely cheap.
    pub fn checked_next(self) -> Result<Self> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(Error::InvalidArgument("LSN overflow"))
    }

    /// Predecessor LSN, saturating at [`Lsn::ZERO`].
    ///
    /// Used by [`crate::pager::Pager::commit`] / `reader_snapshot`
    /// to report the LSN of the *last* committed frame as
    /// `next_lsn - 1`, with the special case `next_lsn == ZERO`
    /// mapping back to `ZERO` rather than wrapping.
    #[must_use]
    pub const fn prev_saturating(self) -> Self {
        Self(self.0.saturating_sub(1))
    }
}

/// Default size cap on the WAL file, in bytes. The cap exists so that
/// a runaway "write without ever committing or checkpointing"
/// workload cannot make recovery walk unboundedly many frames.
///
/// 64 MiB / 4160 bytes/frame ≈ 16 145 frames — the recovery walk
/// length we have to ship a bound for.
pub const DEFAULT_WAL_SIZE_LIMIT: u64 = 64 * 1024 * 1024;

/// Default automatic-checkpoint threshold, in frames. When the WAL
/// has more than this many frames committed, the pager will call its
/// checkpoint routine inline.
pub const DEFAULT_CHECKPOINT_THRESHOLD: u64 = 1_000;

/// WAL construction options.
#[derive(Debug, Clone, Copy)]
pub struct WalConfig {
    /// Per-commit durability primitive.
    pub sync_mode: SyncMode,
    /// Maximum WAL file size in bytes. Exceeding this returns
    /// `Error::InvalidArgument("wal size limit exceeded")`.
    pub size_limit: u64,
    /// Auto-checkpoint threshold (in frames).
    pub checkpoint_threshold: u64,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            sync_mode: SyncMode::Full,
            size_limit: DEFAULT_WAL_SIZE_LIMIT,
            checkpoint_threshold: DEFAULT_CHECKPOINT_THRESHOLD,
        }
    }
}

/// Result of walking an on-disk WAL during recovery.
///
/// `view` is the per-page-id last-committed payload, ready to be
/// merged into the pager's in-memory view. `next_lsn` and
/// `end_offset` are the seekpoints the resulting [`Wal`] uses for
/// subsequent appends; `salt` and `committed_frames` carry over
/// from the WAL header.
///
/// `header` carries the page-0 file-header bytes from the
/// most-recent committed frame whose `page_id` was `0`. The pager
/// applies these on adoption so the in-memory header reflects
/// WAL-staged catalog-root updates that the on-disk header at offset
/// 0 does not yet carry (until checkpoint).
#[derive(Debug)]
pub struct Recovered {
    /// Per-page-id, the body of the most-recent committed frame.
    pub view: HashMap<PageId, Page>,
    /// Header page-0 bytes recovered from a WAL frame with
    /// `page_id = 0`, if any.
    pub header: Option<Page>,
    /// LSN that the next [`WalTxn::commit`] will assign.
    pub next_lsn: Lsn,
    /// WAL generation salt (as read from the WAL header on disk).
    pub salt: u32,
    /// Number of committed frames on disk (torn-tail not counted).
    pub committed_frames: u64,
    /// Byte length where the next frame will be appended. Equals the
    /// position just past the last committed frame; torn tail (if
    /// any) sits between this offset and the file length on disk.
    pub end_offset: u64,
}

impl Recovered {
    /// Consume the [`Recovered`] and return ownership of the per-
    /// page recovered view. Used by the pager when it adopts the
    /// recovered state.
    #[must_use]
    pub fn into_view(self) -> HashMap<PageId, Page> {
        self.view
    }
}

/// Newtype wrapper around the derived 32-byte
/// WAL page-encryption key. Manual `Debug` impl redacts the bytes so
/// the key never appears in log output.
///
/// The inner field is [`crate::pager::MasterKeyBytes`], so
/// under the `encryption` feature the WAL's copy of the per-file page
/// key is wiped from memory when the owning [`Wal`] is dropped.
/// `Copy` is derived only on the no-`encryption` build, where the
/// field is a bare `[u8; 32]` and never holds a real key.
#[cfg_attr(not(feature = "encryption"), derive(Copy))]
#[derive(Clone)]
// allow: the type and its methods are exercised only on the `encryption` build; the whole newtype is dead on the no-encryption build.
#[allow(dead_code)]
pub(crate) struct WalKey(crate::pager::MasterKeyBytes);

impl WalKey {
    #[must_use]
    // allow: constructor is only used on the `encryption` build; dead on the no-encryption build where WalKey holds a bare key.
    #[allow(dead_code)]
    pub(crate) fn new(bytes: [u8; 32]) -> Self {
        Self(crate::pager::wrap_master_key(bytes))
    }

    #[inline]
    // allow: accessor is only used by the `encryption` page-key path; dead on the no-encryption build.
    #[allow(dead_code)]
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        let bytes: &[u8; 32] = &self.0;
        bytes
    }
}

impl std::fmt::Debug for WalKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WalKey(<redacted>)")
    }
}

/// The write-ahead log.
///
/// Owns the on-disk WAL file, the current generation salt, and the LSN
/// counter. The pager talks to a `Wal` via [`Wal::begin_txn`], staging
/// per-page writes in a [`WalTxn`] and then calling [`WalTxn::commit`]
/// to make them durable.
///
/// Generic over `F: FileBackend` (static dispatch on the hot
/// path). Production code uses `Wal<FileHandle>`; the fault-injection
/// harness substitutes `Wal<FaultyFileHandle>` to drive recovery
/// against torn writes, dropped fsyncs, and bit flips.
///
/// When the parent pager opens an
/// encryption-capable file with the right key, the WAL also
/// encrypts each frame body with `XChaCha20-Poly1305`. The frame
/// layout gains a 40-byte suffix (`nonce || tag`), the on-disk
/// per-frame stride becomes 4200 bytes, and the frame's existing
/// CRC32C is computed over (`header_sans_crc` + PLAINTEXT body) —
/// the CRC catches in-memory bit-flips on the post-decryption
/// representation rather than running on attacker-controlled
/// ciphertext.
#[derive(Debug)]
pub struct Wal<F: FileBackend = FileHandle> {
    file: F,
    path: PathBuf,
    salt: u32,
    next_lsn: Lsn,
    /// Byte offset where the next frame will be written.
    end_offset: u64,
    /// Frames-on-disk count (committed; torn-tail not counted). Used
    /// by the pager to decide when to auto-checkpoint.
    committed_frames: u64,
    config: WalConfig,
    /// Per-file page-encryption key, derived
    /// once at open from `HKDF-SHA256(user_key, kdf_salt,
    /// b"obj-page-encryption-v1")` by the pager. `None` =
    /// plaintext WAL. The key is the SAME as the
    /// pager's `derived_key` — the design specifically calls out
    /// that the WAL and the main file share one key.
    key: Option<WalKey>,
}

/// An in-progress WAL transaction.
///
/// Buffers `(page_id, page_body)` pairs in memory; the actual disk
/// writes happen at [`WalTxn::commit`]. This is how group commit
/// works: many calls to [`WalTxn::append`] amortise one `sync_data`.
#[derive(Debug)]
pub struct WalTxn<'a, F: FileBackend = FileHandle> {
    wal: &'a mut Wal<F>,
    /// LIFO of staged frames; iterated in order on commit.
    staged: Vec<(PageId, Page)>,
    /// Per-staged-frame "is this a page-0 file-header
    /// update?" flag, index-aligned with `staged`. On commit a `true`
    /// entry is emitted with `page_id == 0` in the on-disk frame
    /// header; the `staged` tuple carries a stand-in `PageId::new(1)`
    /// because `PageId` cannot represent zero. Carried as a parallel
    /// `Vec<bool>` (not folded into the tuple) so `drain_staged` can
    /// still hand back `staged` verbatim, and allocated once per txn
    /// rather than rebuilt into a `HashSet` per commit.
    is_header: Vec<bool>,
}

impl Wal<FileHandle> {
    /// Create or truncate the WAL sidecar at `path` to a fresh,
    /// empty WAL backed by a [`FileHandle`]. Convenience for
    /// production callers; see [`Wal::create_fresh_with`] when the
    /// caller already holds a backend instance (e.g. a fault-injection
    /// harness).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn create_fresh(path: &Path, config: WalConfig) -> Result<Self> {
        let file = FileHandle::open_or_create(path)?;
        Self::create_fresh_with(file, path.to_path_buf(), config)
    }

    /// Walk the on-disk WAL at `path` and produce a [`Recovered`]
    /// snapshot, opening the WAL with a production [`FileHandle`].
    ///
    /// See [`Wal::open_for_recovery_with`] for the documented
    /// algorithm; see [`Wal::create_fresh`] for the file-handle
    /// rationale.
    ///
    /// # Errors
    ///
    /// See [`Wal::open_for_recovery_with`].
    pub fn open_for_recovery(
        path: &Path,
        expected_salt: u32,
        size_limit: u64,
    ) -> Result<Recovered> {
        if !path.exists() {
            return Ok(empty_recovered(expected_salt));
        }
        let file = FileHandle::open_or_create(path)?;
        Self::open_for_recovery_with(&file, expected_salt, size_limit)
    }
}

impl<F: FileBackend> Wal<F> {
    /// Create or truncate the WAL sidecar at `path` to a fresh,
    /// empty WAL on top of an already-opened backend `file`. Any
    /// existing content is overwritten with a new WAL header carrying
    /// a freshly-sampled generation salt.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn create_fresh_with(file: F, path: PathBuf, config: WalConfig) -> Result<Self> {
        file.set_len(0)?;
        let salt = fresh_salt();
        write_wal_header(&file, salt)?;
        file.sync_data(config.sync_mode)?;
        Ok(Self {
            file,
            path,
            salt,
            next_lsn: Lsn::ONE,
            end_offset: WAL_HEADER_SIZE as u64,
            committed_frames: 0,
            config,
            key: None,
        })
    }

    /// Set the WAL's page-encryption key. Called
    /// by the pager immediately after open / create on
    /// encryption-capable files. `None` clears the key (no-op for
    /// callers that already opened a plaintext WAL).
    ///
    /// Must be called BEFORE any `append` or recovery — the WAL
    /// records its frame size at write/read time from
    /// `self.key.is_some()`, so toggling the key mid-stream would
    /// produce frames of mixed sizes that recovery cannot walk.
    pub(crate) fn set_key(&mut self, key: Option<[u8; 32]>) {
        self.key = key.map(WalKey::new);
    }

    /// Adopt an already-walked WAL handle. Used by `Pager::open`
    /// after [`Wal::open_for_recovery`] has returned a [`Recovered`].
    /// `salt`, `next_lsn`, `committed_frames`, and `end_offset` are
    /// taken from `recovered`; the caller separately merges
    /// `recovered.view` into the pager's in-memory state.
    #[must_use]
    pub fn from_recovered_meta(
        file: F,
        path: PathBuf,
        salt: u32,
        next_lsn: Lsn,
        end_offset: u64,
        committed_frames: u64,
        config: WalConfig,
    ) -> Self {
        Self {
            file,
            path,
            salt,
            next_lsn,
            end_offset,
            committed_frames,
            config,
            key: None,
        }
    }

    /// Walk an already-open WAL file and produce a [`Recovered`]
    /// snapshot.
    ///
    /// Algorithm:
    ///
    /// 1. If `path` does not exist, or is shorter than a WAL header,
    ///    return an empty `Recovered` carrying `expected_salt`.
    /// 2. Read the WAL header. If magic / format-major / page-size
    ///    disagree with the build, fail with
    ///    [`Error::InvalidFormat`].
    /// 3. If the header's salt does not equal `expected_salt`, the
    ///    WAL is from a previous generation; return an empty
    ///    `Recovered`.
    /// 4. **Pass 1**: scan every aligned frame in the WAL and record
    ///    the byte offset of the *last* frame whose salt matches and
    ///    whose CRC validates AND whose commit-marker bit is set.
    ///    Frames whose CRC fails (or whose salt does not match) in
    ///    pass 1 are silently skipped — they might be torn-tail noise
    ///    that precedes a later valid commit marker.
    /// 5. **Pass 2**: walk frames from offset [`WAL_HEADER_SIZE`] up
    ///    to (but not past) the last-commit-end offset from pass 1.
    ///    Any frame in this range whose salt matches MUST have a
    ///    valid CRC; otherwise return [`Error::WalCorruption`] — the
    ///    bad frame sits between two intact commit markers and
    ///    recovery cannot determine if data was lost.
    /// 6. Salt-mismatched frames inside pass 2's range are skipped
    ///    (they are stale-generation noise, not corruption). Frames
    ///    *past* the last commit marker are torn tail and are
    ///    silently discarded.
    ///
    /// # Errors
    ///
    /// - [`Error::Io`] on syscall failure.
    /// - [`Error::InvalidFormat`] when the WAL header is malformed
    ///   in a way that indicates a config mismatch rather than torn
    ///   tail.
    /// - [`Error::WalCorruption`] when a CRC-invalid frame sits
    ///   before the last committed frame in the current generation.
    /// - [`Error::InvalidArgument`] if `size_limit` would be
    ///   exceeded during the walk (a runaway WAL caps recovery).
    pub fn open_for_recovery_with(
        file: &F,
        expected_salt: u32,
        size_limit: u64,
    ) -> Result<Recovered> {
        Self::open_for_recovery_with_key(file, expected_salt, size_limit, None)
    }

    /// Same as
    /// [`Self::open_for_recovery_with`] but takes an optional
    /// per-file page-encryption key. On encrypted WALs each frame
    /// body is decrypted with the supplied key BEFORE the frame's
    /// CRC32C is validated; the recovery walker therefore needs the
    /// key at construction. The pager calls this entry point.
    ///
    /// # Errors
    ///
    /// As [`Self::open_for_recovery_with`], plus
    /// [`Error::EncryptionKeyInvalid`] when a salt-matching frame
    /// in the WAL fails Poly1305 verification AND no other frame
    /// decrypted successfully — the smoking-gun wrong-key signal. A
    /// decrypt failure alongside at least one frame that DID decrypt is
    /// classified by position instead (torn tail vs. `WalCorruption`).
    pub fn open_for_recovery_with_key(
        file: &F,
        expected_salt: u32,
        size_limit: u64,
        key: Option<[u8; 32]>,
    ) -> Result<Recovered> {
        let len = file.len()?;
        if len < WAL_HEADER_SIZE as u64 {
            return Ok(empty_recovered(expected_salt));
        }
        let header_salt = read_wal_header(file)?;
        if header_salt != expected_salt {
            return Ok(empty_recovered(expected_salt));
        }
        let key = key.map(WalKey::new);
        walk_frames(file, header_salt, len, size_limit, key.as_ref())
    }

    /// Path the WAL was opened at. Used by the pager to remove the
    /// sidecar on clean shutdown.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// On-disk per-frame stride in bytes. Equal
    /// to [`FRAME_SIZE`] (4160) on plaintext WALs, [`FRAME_SIZE_ENCRYPTED`]
    /// (4200) on encrypted ones. Read at every site that walks the
    /// WAL — the constant `FRAME_SIZE` is no longer authoritative
    /// across all builds.
    #[must_use]
    fn frame_size_bytes(&self) -> usize {
        frame_size_for(self.key.is_some())
    }

    /// Current WAL generation salt.
    #[must_use]
    pub fn salt(&self) -> u32 {
        self.salt
    }

    /// LSN the next appended frame will carry.
    #[must_use]
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }

    /// Frames currently on disk (committed; torn-tail not counted).
    #[must_use]
    pub fn committed_frames(&self) -> u64 {
        self.committed_frames
    }

    /// Configured auto-checkpoint threshold.
    #[must_use]
    pub fn checkpoint_threshold(&self) -> u64 {
        self.config.checkpoint_threshold
    }

    /// Begin a new transaction. The returned [`WalTxn`] holds a
    /// mutable borrow of the WAL; only one transaction can be open at
    /// a time.
    pub fn begin_txn(&mut self) -> WalTxn<'_, F> {
        WalTxn {
            wal: self,
            staged: Vec::new(),
            is_header: Vec::new(),
        }
    }

    /// Reset the WAL after a successful checkpoint: rotate the salt,
    /// write the new header, fsync, and truncate to header-only.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn reset_after_checkpoint(&mut self) -> Result<()> {
        let new_salt = next_salt(self.salt);
        write_wal_header(&self.file, new_salt)?;
        self.file.sync_data(self.config.sync_mode)?;
        self.file.set_len(WAL_HEADER_SIZE as u64)?;
        self.file.sync_data(self.config.sync_mode)?;
        self.salt = new_salt;
        self.next_lsn = Lsn::ONE;
        self.end_offset = WAL_HEADER_SIZE as u64;
        self.committed_frames = 0;
        Ok(())
    }
}

impl<F: FileBackend> WalTxn<'_, F> {
    /// Append `(page_id, page)` to the transaction. The frame is held
    /// in memory until [`WalTxn::commit`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the resulting WAL size
    /// would exceed `Config::wal_size_limit`.
    pub fn append(&mut self, page_id: PageId, page: &Page) -> Result<()> {
        self.append_raw(page_id.get(), page)
    }

    /// Append a file-header (page-0) frame to the
    /// transaction. The WAL frame carries `page_id = 0`; recovery's
    /// `WalkState::absorb` routes it into a dedicated header slot.
    /// Used by [`crate::pager::Pager::commit`] when
    /// [`crate::pager::Pager::set_root_catalog`] dirtied the
    /// in-memory header.
    ///
    /// # Errors
    ///
    /// As [`Self::append`].
    pub fn append_header(&mut self, page: &Page) -> Result<()> {
        self.append_raw(0, page)
    }

    /// Internal: stage a frame with the given raw page-id (zero for
    /// header updates, non-zero for regular page writes). Centralises
    /// the size-cap check so both [`Self::append`] and
    /// [`Self::append_header`] share one bound.
    fn append_raw(&mut self, page_id: u64, page: &Page) -> Result<()> {
        let prospective_size = self
            .wal
            .end_offset
            .checked_add(
                (self
                    .staged
                    .len()
                    .checked_add(1)
                    .ok_or(Error::InvalidArgument("txn frame count overflow"))?
                    as u64)
                    .checked_mul(self.wal.frame_size_bytes() as u64)
                    .ok_or(Error::InvalidArgument("wal frame offset overflow"))?,
            )
            .ok_or(Error::InvalidArgument("wal offset overflow"))?;
        if prospective_size > self.wal.config.size_limit {
            return Err(Error::InvalidArgument("wal size limit exceeded"));
        }
        let staged_id = PageId::new(if page_id == 0 { 1 } else { page_id }).ok_or(
            Error::InvalidArgument("internal: PageId::new returned None on a non-zero input"),
        )?;
        self.staged.push((staged_id, page.clone()));
        self.is_header.push(page_id == 0);
        debug_assert_eq!(
            self.staged.len(),
            self.is_header.len(),
            "is_header must stay index-aligned with staged"
        );
        Ok(())
    }

    /// Number of frames currently staged in this transaction.
    #[must_use]
    pub fn staged_frame_count(&self) -> usize {
        self.staged.len()
    }

    /// Commit the transaction. Writes every staged frame to disk,
    /// stamps the last one as the commit marker, performs one
    /// `sync_data(sync_mode)`, and returns the LSN of the last
    /// frame.
    ///
    /// An empty transaction is a no-op and returns the current
    /// `next_lsn - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn commit(self) -> Result<Lsn> {
        if self.staged.is_empty() {
            return Ok(self.wal.next_lsn.prev_saturating());
        }
        let last_index = self.staged.len() - 1;
        let mut last_lsn: Lsn = Lsn::ZERO;
        let mut offset = self.wal.end_offset;
        let bound = self.staged.len();
        debug_assert_eq!(
            self.staged.len(),
            self.is_header.len(),
            "is_header must stay index-aligned with staged"
        );
        let mut scratch = [0u8; FRAME_SIZE];
        for (index, (page_id, page)) in self.staged.iter().enumerate().take(bound) {
            let lsn = self.wal.next_lsn;
            self.wal.next_lsn = self.wal.next_lsn.checked_next()?;
            let is_commit = index == last_index;
            let wire_page_id = if self.is_header[index] {
                0
            } else {
                page_id.get()
            };
            let header = FrameHeader {
                page_id: wire_page_id,
                lsn: lsn.get(),
                salt: self.wal.salt,
                commit: is_commit,
            };
            write_frame(
                &self.wal.file,
                offset,
                &header,
                page,
                self.wal.key.as_ref(),
                &mut scratch,
            )?;
            last_lsn = lsn;
            offset = offset
                .checked_add(self.wal.frame_size_bytes() as u64)
                .ok_or(Error::InvalidArgument("wal offset overflow"))?;
        }
        self.wal.file.sync_data(self.wal.config.sync_mode)?;
        self.wal.end_offset = offset;
        let count_u64 = u64::try_from(self.staged.len())
            .map_err(|_| Error::InvalidArgument("txn frame count overflow"))?;
        self.wal.committed_frames = self
            .wal
            .committed_frames
            .checked_add(count_u64)
            .ok_or(Error::InvalidArgument("committed-frame count overflow"))?;
        Ok(last_lsn)
    }

    /// Drain the staged frames into an owned `Vec` so the pager can
    /// merge them into its in-memory view after a successful commit.
    /// Called by `WalTxn::commit_returning_view` (see
    /// `pager::commit`).
    #[must_use]
    pub fn drain_staged(self) -> Vec<(PageId, Page)> {
        self.staged
    }
}

fn empty_recovered(salt: u32) -> Recovered {
    Recovered {
        view: HashMap::new(),
        header: None,
        next_lsn: Lsn::ONE,
        salt,
        committed_frames: 0,
        end_offset: WAL_HEADER_SIZE as u64,
    }
}

fn read_wal_header<F: FileBackend>(file: &F) -> Result<u32> {
    let mut buf = [0u8; WAL_HEADER_SIZE];
    file.read_exact_at(&mut buf, 0)?;
    if buf[0..4] != WAL_MAGIC {
        return Err(Error::InvalidFormat {
            reason: "WAL magic does not match",
        });
    }
    let major = u16::from_le_bytes([buf[4], buf[5]]);
    if !crate::pager::header::is_supported_format_major(major) {
        return Err(Error::InvalidFormat {
            reason: "WAL format-major does not match",
        });
    }
    let minor = u16::from_le_bytes([buf[6], buf[7]]);
    if !crate::pager::header::is_supported_minor(major, minor) {
        return Err(Error::InvalidFormat {
            reason: "WAL format-minor is not supported",
        });
    }
    let page_size = u16::from_le_bytes([buf[8], buf[9]]);
    if usize::from(page_size) != PAGE_SIZE {
        return Err(Error::InvalidFormat {
            reason: "WAL page-size does not match this build",
        });
    }
    Ok(u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]))
}

/// Two-pass WAL recovery walk.
///
/// **Pass 1** finds the byte offset of the last frame in the current
/// generation that satisfies (salt matches AND CRC valid AND commit
/// flag set). Frames that fail decoding are silently skipped in pass
/// 1 — they may be torn tail.
///
/// **Pass 2** walks from `WAL_HEADER_SIZE` up to (but not past) the
/// end of the last-commit frame found in pass 1. Any frame in that
/// range whose salt matches `salt` MUST have a valid CRC32C; a
/// mismatch is `Error::WalCorruption`. Salt-mismatched frames inside
/// the range are silently skipped (treated like stale-generation
/// noise that happens to sit before a later commit).
///
/// If pass 1 finds no commit marker, the WAL contains no recoverable
/// state and we return an empty `Recovered`. In that case any bad CRC
/// past the WAL header is treated as torn tail — the standard
/// "WAL exists but no transaction ever committed" path.
fn walk_frames<F: FileBackend>(
    file: &F,
    salt: u32,
    file_len: u64,
    size_limit: u64,
    key: Option<&WalKey>,
) -> Result<Recovered> {
    let frame_size = frame_size_for(key.is_some());
    let frame_limit = bounded_frame_limit(size_limit, frame_size);
    let scan_end = scan_aligned_end(file_len, frame_size);
    let scan = find_last_commit_end(file, salt, scan_end, frame_limit, key, frame_size)?;
    if let Some(fail_off) = scan.first_decrypt_failure_offset {
        // A salt-matching frame that failed to decrypt is only
        // wrong-key / corruption evidence when it sits inside the
        // authoritative committed prefix. Once a real commit marker is
        // found (`last_commit_end > WAL_HEADER_SIZE`), that marker
        // proves the key decrypts and CRC-validates committed frames —
        // so a decrypt failure AT or PAST it is an ordinary torn tail
        // (plaintext header persisted, ciphertext body partial or
        // bit-flipped) and is discarded exactly as the plaintext path
        // discards a bad-CRC tail.
        //
        // A failure *before* the commit marker is handled by the marker
        // itself: because that marker decrypted and CRC-validated under
        // the same key, the key is provably correct, so the failure is
        // **corruption of a committed frame** (`WalCorruption`), not a
        // wrong key.
        //
        // With NO commit marker at all, the inference is subtler. A
        // no-marker generation is NOT unique to a wrong key: it recurs
        // after every checkpoint, because `reset_after_checkpoint`
        // rotates the salt and truncates the WAL to header-only, so the
        // first transaction re-enters the "committed frames = 0, no
        // marker yet" window. If some salt-matching frame decrypted AND
        // CRC-validated (`any_frame_decrypted`), the key is provably
        // correct, so a no-marker generation is a torn UNCOMMITTED tail:
        // discard it and open cleanly (`Ok(empty_recovered)` below),
        // exactly as the plaintext path discards a bad-CRC tail. We
        // escalate to `EncryptionKeyInvalid` ONLY when nothing decrypted
        // and a salt-matching decrypt failure exists — the genuine
        // wrong-key signature (whole generation undecryptable). Either
        // way the open is refused or discarded fail-closed; no committed
        // data is ever silently dropped. `first_decrypt_failure_offset`
        // is the earliest such offset, so checking it covers every later
        // failure too.
        let has_commit_marker = scan.last_commit_end > WAL_HEADER_SIZE as u64;
        let torn_tail_past_commit = has_commit_marker && fail_off >= scan.last_commit_end;
        if !torn_tail_past_commit {
            if has_commit_marker {
                return Err(Error::WalCorruption {
                    frame_offset: fail_off,
                });
            }
            if !scan.any_frame_decrypted {
                return Err(Error::EncryptionKeyInvalid);
            }
            // Key proven correct, no commit marker → torn uncommitted
            // tail. Fall through to the `last_commit_end <=
            // WAL_HEADER_SIZE` discard below.
        }
    }
    if scan.last_commit_end <= WAL_HEADER_SIZE as u64 {
        return Ok(empty_recovered(salt));
    }
    replay_up_to_commit(
        file,
        salt,
        scan.last_commit_end,
        frame_limit,
        key,
        frame_size,
    )
}

/// Result of pass 1 of the WAL walk. Carries
/// the last-commit-end byte offset, the earliest byte offset
/// at which a salt-matching frame failed to decrypt (if any), and
/// whether ANY salt-matching frame decrypted-and-CRC-validated.
///
/// The offset — rather than a bare flag — lets the caller distinguish
/// a decrypt failure inside the committed prefix (wrong key /
/// corruption) from one past the last commit marker (an ordinary torn
/// tail, discarded like the plaintext path). Because pass 1 walks
/// frames in ascending offset order, the first failure recorded is the
/// smallest offset, so it represents every later failure too.
///
/// `any_frame_decrypted` proves the key correct: a fresh or
/// post-checkpoint generation legitimately has no commit marker yet, so
/// a decrypt failure there is the wrong-key smoking gun ONLY when
/// nothing decrypted. If even one salt-matching frame decrypted and
/// CRC-validated, the key is provably correct and a no-marker generation
/// is a torn UNCOMMITTED tail to be discarded, not a wrong-key open.
#[derive(Debug, Clone, Copy)]
struct ScanResult {
    last_commit_end: u64,
    first_decrypt_failure_offset: Option<u64>,
    any_frame_decrypted: bool,
}

/// Byte offset just past the last full-frame boundary that fits in
/// `file_len`. Any bytes after this are torn tail (less than a frame
/// worth) and never inspected.
///
/// `file_len` is OS-supplied and effectively caller-
/// controlled in a fault-injection harness; saturate the arithmetic
/// at `u64::MAX` rather than relying on the `payload / FRAME_SIZE`
/// reduction to bound the final product. The saturation is benign:
/// the recovery walker's `walked > frame_limit` check is the actual
/// termination guarantee.
fn scan_aligned_end(file_len: u64, frame_size: usize) -> u64 {
    if file_len < WAL_HEADER_SIZE as u64 {
        return WAL_HEADER_SIZE as u64;
    }
    let payload = file_len - WAL_HEADER_SIZE as u64;
    let aligned_frames = payload / frame_size as u64;
    aligned_frames
        .checked_mul(frame_size as u64)
        .and_then(|product| product.checked_add(WAL_HEADER_SIZE as u64))
        .unwrap_or(u64::MAX)
}

/// Pass 1: walk every aligned frame between [`WAL_HEADER_SIZE`] and
/// `scan_end`. Record the byte offset just past the *last* frame in
/// the current generation whose salt matches and whose CRC validates
/// AND whose commit flag is set. Returns `WAL_HEADER_SIZE` if no such
/// frame exists.
fn find_last_commit_end<F: FileBackend>(
    file: &F,
    salt: u32,
    scan_end: u64,
    frame_limit: u64,
    key: Option<&WalKey>,
    frame_size: usize,
) -> Result<ScanResult> {
    let mut offset = WAL_HEADER_SIZE as u64;
    let mut last_commit_end = WAL_HEADER_SIZE as u64;
    let mut first_decrypt_failure_offset: Option<u64> = None;
    let mut any_frame_decrypted = false;
    let mut walked: u64 = 0;
    while let Some(frame_end) = offset.checked_add(frame_size as u64) {
        if frame_end > scan_end {
            break;
        }
        if walked > frame_limit {
            return Err(Error::InvalidArgument(
                "WAL exceeds size limit during recovery",
            ));
        }
        walked = walked.saturating_add(1);
        let frame = read_plaintext_frame_diag(file, offset, key, frame_size, salt)?;
        if let FrameDecode::Ok(header) = decode_frame_header_classified(&frame.buf, salt) {
            // Salt matched AND CRC validated on the decrypted body — this
            // proves the key is correct, whether or not the frame is a
            // commit marker.
            any_frame_decrypted = true;
            if header.commit {
                last_commit_end = offset
                    .checked_add(frame_size as u64)
                    .ok_or(Error::InvalidArgument("wal offset overflow"))?;
            }
        } else if frame.salt_matched_but_decrypt_failed && first_decrypt_failure_offset.is_none() {
            first_decrypt_failure_offset = Some(offset);
        }
        offset = offset
            .checked_add(frame_size as u64)
            .ok_or(Error::InvalidArgument("wal offset overflow"))?;
    }
    Ok(ScanResult {
        last_commit_end,
        first_decrypt_failure_offset,
        any_frame_decrypted,
    })
}

/// On-disk frame reader that ALSO returns a
/// diagnostic flag set to `true` when the on-disk frame's header
/// salt matched but the body decrypt failed. Used in pass 1 to
/// distinguish "wrong-key open" from "torn tail" / "stale
/// generation".
struct PlaintextFrame {
    buf: Vec<u8>,
    salt_matched_but_decrypt_failed: bool,
}

fn read_plaintext_frame_diag<F: FileBackend>(
    file: &F,
    offset: u64,
    key: Option<&WalKey>,
    frame_size: usize,
    expected_salt: u32,
) -> Result<PlaintextFrame> {
    let raw = read_frame_bytes(file, offset, frame_size)?;
    let Some(key) = key else {
        let _ = expected_salt;
        return Ok(PlaintextFrame {
            buf: raw,
            salt_matched_but_decrypt_failed: false,
        });
    };
    #[cfg(feature = "encryption")]
    {
        let mut out = vec![0u8; FRAME_SIZE];
        out[..FRAME_HEADER_SIZE].copy_from_slice(&raw[..FRAME_HEADER_SIZE]);
        let mut ad = [0u8; 16];
        ad.copy_from_slice(&raw[..16]);
        let mut ct = [0u8; PAGE_SIZE + FRAME_AEAD_SUFFIX_SIZE];
        ct.copy_from_slice(&raw[FRAME_HEADER_SIZE..]);
        let mut pt = [0u8; PAGE_SIZE];
        let salt_matched_but_decrypt_failed = if wal_decrypt(key, &ad, &ct, &mut pt).is_ok() {
            out[FRAME_HEADER_SIZE..].copy_from_slice(&pt);
            false
        } else {
            let frame_salt = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]);
            out[FRAME_HEADER_SIZE..].copy_from_slice(&ct[..PAGE_SIZE]);
            frame_salt == expected_salt
        };
        Ok(PlaintextFrame {
            buf: out,
            salt_matched_but_decrypt_failed,
        })
    }
    #[cfg(not(feature = "encryption"))]
    {
        let _ = (key, expected_salt);
        Ok(PlaintextFrame {
            buf: raw,
            salt_matched_but_decrypt_failed: false,
        })
    }
}

/// Pass 2: replay frames from the WAL header up to (but not past)
/// `commit_end`. Frames whose salt matches MUST have a valid CRC;
/// salt-mismatched frames are skipped (treated like stale-generation
/// noise that pre-dates the current run). Returns the recovered view
/// with the merged committed state.
fn replay_up_to_commit<F: FileBackend>(
    file: &F,
    salt: u32,
    commit_end: u64,
    frame_limit: u64,
    key: Option<&WalKey>,
    frame_size: usize,
) -> Result<Recovered> {
    let mut state = WalkState::new();
    let mut walked: u64 = 0;
    while state.offset < commit_end {
        if walked > frame_limit {
            return Err(Error::InvalidArgument(
                "WAL exceeds size limit during recovery",
            ));
        }
        walked = walked.saturating_add(1);
        let buf = read_plaintext_frame(file, state.offset, key, frame_size)?;
        match decode_frame_header_classified(&buf, salt) {
            FrameDecode::Ok(header) => {
                let mut page = Page::zeroed();
                page.as_bytes_mut()
                    .copy_from_slice(&buf[FRAME_HEADER_SIZE..]);
                state.absorb(header, page, frame_size)?;
            }
            FrameDecode::CrcInvalid => {
                return Err(Error::WalCorruption {
                    frame_offset: state.offset,
                });
            }
            FrameDecode::SaltMismatch | FrameDecode::Malformed => {}
        }
        state.offset = state
            .offset
            .checked_add(frame_size as u64)
            .ok_or(Error::InvalidArgument("wal offset overflow"))?;
    }
    Ok(state.into_recovered(salt))
}

struct WalkState {
    view: HashMap<PageId, Page>,
    pending: HashMap<PageId, Page>,
    pending_count: u64,
    /// A frame with `page_id == 0` carries an updated page-0
    /// file header. Accumulate the most-recent uncommitted one here
    /// and promote on commit (alongside the regular `pending` map).
    pending_header: Option<Page>,
    /// Most-recent COMMITTED page-0 frame body.
    view_header: Option<Page>,
    offset: u64,
    next_lsn: Lsn,
    committed_frames: u64,
    last_committed_offset: u64,
}

impl WalkState {
    fn new() -> Self {
        Self {
            view: HashMap::new(),
            pending: HashMap::new(),
            pending_count: 0,
            pending_header: None,
            view_header: None,
            offset: WAL_HEADER_SIZE as u64,
            next_lsn: Lsn::ONE,
            committed_frames: 0,
            last_committed_offset: WAL_HEADER_SIZE as u64,
        }
    }

    /// Absorb one decoded frame. A frame with `page_id == 0`
    /// is a file-header (page-0) update; route it into a dedicated
    /// slot. Frames with non-zero `page_id` are regular page writes.
    /// Returns `Ok(false)` only on the malformed case where a frame
    /// is neither (today: never — kept as a forward-compat hook).
    ///
    /// `frame_size` is the on-disk per-frame
    /// stride (4160 plaintext / 4200 encrypted) so the
    /// `last_committed_offset` computation can step the right number
    /// of bytes.
    fn absorb(&mut self, header: FrameHeader, page: Page, frame_size: usize) -> Result<bool> {
        if header.page_id == 0 {
            self.pending_header = Some(page);
        } else {
            let Some(page_id) = PageId::new(header.page_id) else {
                return Ok(false);
            };
            self.pending.insert(page_id, page);
        }
        self.pending_count = self
            .pending_count
            .checked_add(1)
            .ok_or(Error::InvalidArgument("pending frame count overflow"))?;
        if header.commit {
            promote_pending(&mut self.pending, &mut self.view);
            if let Some(hp) = self.pending_header.take() {
                self.view_header = Some(hp);
            }
            self.committed_frames = self
                .committed_frames
                .checked_add(self.pending_count)
                .ok_or(Error::InvalidArgument("committed frame count overflow"))?;
            self.pending_count = 0;
            self.last_committed_offset = self
                .offset
                .checked_add(frame_size as u64)
                .ok_or(Error::InvalidArgument("wal offset overflow"))?;
        }
        self.next_lsn = Lsn::new(header.lsn.saturating_add(1));
        Ok(true)
    }

    fn into_recovered(self, salt: u32) -> Recovered {
        Recovered {
            view: self.view,
            header: self.view_header,
            next_lsn: self.next_lsn,
            salt,
            committed_frames: self.committed_frames,
            end_offset: self.last_committed_offset,
        }
    }
}

fn promote_pending(pending: &mut HashMap<PageId, Page>, view: &mut HashMap<PageId, Page>) {
    for (id, page) in pending.drain() {
        view.insert(id, page);
    }
}

fn read_frame_bytes<F: FileBackend>(file: &F, offset: u64, frame_size: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; frame_size];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

/// Read the on-disk physical frame at `offset`
/// and return its **plaintext** representation (always `FRAME_SIZE`
/// = 4160 bytes). On plaintext WALs (`key` is `None`) this is
/// exactly the on-disk bytes. On encrypted WALs we read 4200
/// bytes, copy the 64-byte plaintext header verbatim, and
/// AEAD-decrypt the
/// remaining body. A decryption failure surfaces as a plaintext
/// buffer carrying the original ciphertext — `decode_frame_header_
/// classified` will then return `FrameDecode::CrcInvalid` (the
/// caller treats that as torn tail in pass 1 and as
/// `Error::WalCorruption` in pass 2).
fn read_plaintext_frame<F: FileBackend>(
    file: &F,
    offset: u64,
    key: Option<&WalKey>,
    frame_size: usize,
) -> Result<Vec<u8>> {
    let raw = read_frame_bytes(file, offset, frame_size)?;
    let Some(key) = key else {
        return Ok(raw);
    };
    #[cfg(feature = "encryption")]
    {
        let mut out = vec![0u8; FRAME_SIZE];
        out[..FRAME_HEADER_SIZE].copy_from_slice(&raw[..FRAME_HEADER_SIZE]);
        let mut ad = [0u8; 16];
        ad.copy_from_slice(&raw[..16]);
        let mut ct = [0u8; PAGE_SIZE + FRAME_AEAD_SUFFIX_SIZE];
        ct.copy_from_slice(&raw[FRAME_HEADER_SIZE..]);
        let mut pt = [0u8; PAGE_SIZE];
        match wal_decrypt(key, &ad, &ct, &mut pt) {
            Ok(()) => {
                out[FRAME_HEADER_SIZE..].copy_from_slice(&pt);
            }
            Err(_) => {
                out[FRAME_HEADER_SIZE..].copy_from_slice(&ct[..PAGE_SIZE]);
            }
        }
        Ok(out)
    }
    #[cfg(not(feature = "encryption"))]
    {
        let _ = key;
        Ok(raw)
    }
}

fn bounded_frame_limit(size_limit: u64, frame_size: usize) -> u64 {
    size_limit / frame_size as u64 + 1
}

/// Generate a fresh 32-bit salt from the OS RNG. Used at first WAL
/// open and at every checkpoint rotation.
fn fresh_salt() -> u32 {
    let mut rng = rand::rng();
    rng.next_u32()
}

/// Generate the next-generation salt. Guarantees `next != current`
/// even if the OS RNG returns the same value back-to-back (rare but
/// theoretically possible with a constant-output mock RNG).
fn next_salt(current: u32) -> u32 {
    let mut candidate = fresh_salt();
    if candidate == current {
        candidate = current.wrapping_add(1);
    }
    candidate
}

fn write_wal_header<F: FileBackend>(file: &F, salt: u32) -> Result<()> {
    let mut buf = [0u8; WAL_HEADER_SIZE];
    buf[0..4].copy_from_slice(&WAL_MAGIC);
    buf[4..6].copy_from_slice(&crate::pager::header::FORMAT_MAJOR.to_le_bytes());
    buf[6..8].copy_from_slice(&crate::pager::header::FORMAT_MINOR.to_le_bytes());
    let page_size_u16 =
        u16::try_from(PAGE_SIZE).map_err(|_| Error::InvalidArgument("page size > u16"))?;
    buf[8..10].copy_from_slice(&page_size_u16.to_le_bytes());
    buf[12..16].copy_from_slice(&salt.to_le_bytes());
    file.write_all_at(&buf, 0)
}

fn write_frame<F: FileBackend>(
    file: &F,
    offset: u64,
    header: &FrameHeader,
    page: &Page,
    key: Option<&WalKey>,
    scratch: &mut [u8],
) -> Result<()> {
    debug_assert_eq!(
        scratch.len(),
        FRAME_SIZE,
        "frame scratch must be FRAME_SIZE"
    );
    let frame_buf = scratch;
    frame_buf[FRAME_HEADER_SIZE..].copy_from_slice(page.as_bytes());
    encode_frame_header(header, frame_buf);
    let Some(key) = key else {
        return file.write_all_at(frame_buf, offset);
    };
    encrypt_frame_body(key, frame_buf, offset, file)
}

/// Encrypt a stamped 4160-byte plaintext frame
/// buffer (`[header][plaintext_body]`) into a 4200-byte encrypted
/// physical frame (`[header][ciphertext_body][nonce][tag]`) and
/// write it to `file` at `offset`.
fn encrypt_frame_body<F: FileBackend>(
    key: &WalKey,
    plain_frame: &[u8],
    offset: u64,
    file: &F,
) -> Result<()> {
    debug_assert_eq!(plain_frame.len(), FRAME_SIZE);
    #[cfg(feature = "encryption")]
    {
        let mut out = [0u8; FRAME_SIZE_ENCRYPTED];
        out[..FRAME_HEADER_SIZE].copy_from_slice(&plain_frame[..FRAME_HEADER_SIZE]);
        let mut body_pt = [0u8; PAGE_SIZE];
        body_pt.copy_from_slice(&plain_frame[FRAME_HEADER_SIZE..]);
        let mut body_phys = [0u8; PAGE_SIZE + FRAME_AEAD_SUFFIX_SIZE];
        let mut ad = [0u8; 16];
        ad.copy_from_slice(&plain_frame[..16]);
        wal_encrypt(key, &ad, &body_pt, &mut body_phys)?;
        out[FRAME_HEADER_SIZE..].copy_from_slice(&body_phys);
        file.write_all_at(&out, offset)
    }
    #[cfg(not(feature = "encryption"))]
    {
        let _ = (key, plain_frame, offset, file);
        Err(Error::FormatFeatureUnsupported {
            feature: "encryption",
        })
    }
}

/// XChaCha20-Poly1305 encrypt a 4096-byte body
/// into a 4136-byte (body || nonce || tag) buffer, with AD = `ad`.
/// The nonce is `XChaCha20`'s 24-byte (192-bit) extended nonce.
/// Returns [`Error::Io`] on CSPRNG failure or
/// [`Error::EncryptionKeyInvalid`] on the structurally-unreachable
/// AEAD error.
#[cfg(feature = "encryption")]
fn wal_encrypt(
    key: &WalKey,
    ad: &[u8; 16],
    plaintext: &[u8; PAGE_SIZE],
    out: &mut [u8; PAGE_SIZE + FRAME_AEAD_SUFFIX_SIZE],
) -> Result<()> {
    use chacha20poly1305::aead::{AeadInPlace, KeyInit};
    use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
    let mut nonce_bytes = [0u8; 24];
    getrandom::getrandom(&mut nonce_bytes)
        .map_err(|e| Error::Io(std::io::Error::other(format!("getrandom failure: {e}"))))?;
    let nonce = XNonce::from_slice(&nonce_bytes);
    out[..PAGE_SIZE].copy_from_slice(plaintext);
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    let tag = cipher
        .encrypt_in_place_detached(nonce, ad, &mut out[..PAGE_SIZE])
        .map_err(|_| Error::EncryptionKeyInvalid)?;
    out[PAGE_SIZE..PAGE_SIZE + 24].copy_from_slice(&nonce_bytes);
    out[PAGE_SIZE + 24..].copy_from_slice(&tag);
    Ok(())
}

/// XChaCha20-Poly1305 decrypt a 4136-byte
/// (ciphertext || nonce || tag) buffer into a 4096-byte body.
#[cfg(feature = "encryption")]
fn wal_decrypt(
    key: &WalKey,
    ad: &[u8; 16],
    ciphertext: &[u8; PAGE_SIZE + FRAME_AEAD_SUFFIX_SIZE],
    out: &mut [u8; PAGE_SIZE],
) -> Result<()> {
    use chacha20poly1305::aead::{AeadInPlace, KeyInit};
    use chacha20poly1305::{Key, Tag, XChaCha20Poly1305, XNonce};
    let mut nonce_bytes = [0u8; 24];
    nonce_bytes.copy_from_slice(&ciphertext[PAGE_SIZE..PAGE_SIZE + 24]);
    let nonce = XNonce::from_slice(&nonce_bytes);
    let mut tag_bytes = [0u8; 16];
    tag_bytes.copy_from_slice(&ciphertext[PAGE_SIZE + 24..]);
    let tag = Tag::from_slice(&tag_bytes);
    out.copy_from_slice(&ciphertext[..PAGE_SIZE]);
    let cipher = XChaCha20Poly1305::new(Key::from_slice(key.as_bytes()));
    cipher
        .decrypt_in_place_detached(nonce, ad, out, tag)
        .map_err(|_| Error::EncryptionKeyInvalid)?;
    Ok(())
}

/// Remove the WAL file at `path`. Idempotent — missing-file is OK.
///
/// # Errors
///
/// Returns [`Error::Io`] on any failure other than `NotFound`.
pub fn remove_wal(path: &Path) -> Result<()> {
    remove_file_if_exists(path)
}

#[cfg(test)]
mod tests;
