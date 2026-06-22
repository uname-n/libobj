//! Hot backup primitives.
//!
//! [`backup_pager_to_path`] is the obj-core entry point the obj
//! crate's `Db::backup_to` dispatches through. The function takes a
//! `&Pager<F>` (the caller has already pinned a [`ReaderSnapshot`])
//! and writes a self-contained `.obj` file at `dest`. Writers may
//! continue against the source pager throughout; their post-snapshot
//! commits do not appear in the destination.

#![forbid(unsafe_code)]

use std::path::Path;

use crate::error::{Error, Result};
use crate::pager::header::{decode_header, encode_header, FileHeader};
use crate::pager::page::{Page, PageId, PAGE_SIZE};
use crate::pager::{Pager, ReaderSnapshot};
use crate::platform::{remove_file_if_exists, FileBackend, FileHandle, SyncMode};

/// Build a self-contained `.obj` file at `dest` carrying the state
/// of `source` as of `snapshot.pinned_lsn()`.
///
/// `snapshot` MUST have been taken against `source` and not yet
/// dropped; the pin keeps the source's WAL frames at-or-below the
/// pinned LSN from being reclaimed while the backup runs.
///
/// # Algorithm
///
/// 1. Refuse to overwrite an existing `dest` (`create_new`).
/// 2. Copy main-file pages `0..source.page_count()` byte-for-byte.
/// 3. Overlay every frame in the snapshot's frozen WAL view onto
///    the destination at its page-id offset.
/// 4. If the snapshot's frozen view carries a page-0 header frame,
///    overlay that on top of the main-file copy of page 0.
/// 5. Patch the destination header: zero `wal_salt`, recompute the
///    header CRC32C.
/// 6. `sync_data(SyncMode::Full)` on the destination.
///
/// On any mid-backup error the destination file is removed best-
/// effort so a half-written backup does not linger.
///
/// # Errors
///
/// - [`Error::BackupDestinationExists`] if `dest` already exists.
/// - [`Error::BackupNotSupportedForMemoryPager`] if `source` is an
///   in-memory pager.
/// - [`Error::BackupNotSupportedForEncryptedPager`] if `source` is
///   an encryption-capable pager. The copy path reads decrypted
///   plaintext bodies; writing them under the source's encrypted
///   header would yield an unrecoverable file, so the backup is
///   refused up front.
/// - [`Error::Io`] on any syscall failure during the copy.
/// - [`Error::InvalidFormat`] / [`Error::Corruption`] propagated
///   from the source header decode (the source's header bytes are
///   re-encoded with the WAL-staged values applied).
pub fn backup_pager_to_path<F: FileBackend>(
    source: &Pager<F>,
    snapshot: &ReaderSnapshot<F>,
    dest: impl AsRef<Path>,
) -> Result<()> {
    let dest_path = dest.as_ref().to_path_buf();
    if source.is_memory_backed() {
        return Err(Error::BackupNotSupportedForMemoryPager);
    }
    if source.is_encryption_capable() {
        return Err(Error::BackupNotSupportedForEncryptedPager);
    }
    if dest_path.exists() {
        return Err(Error::BackupDestinationExists { path: dest_path });
    }
    let result = run_backup(source, snapshot, &dest_path);
    if result.is_err() {
        let _ = remove_file_if_exists(&dest_path);
    }
    result
}

fn run_backup<F: FileBackend>(
    source: &Pager<F>,
    snapshot: &ReaderSnapshot<F>,
    dest_path: &Path,
) -> Result<()> {
    let dest_handle = FileHandle::create_new(dest_path)?;
    let page_count = source.page_count();
    dest_handle.set_len(
        page_count
            .checked_mul(PAGE_SIZE as u64)
            .ok_or(Error::InvalidArgument("backup: file size overflow"))?,
    )?;
    let physical_page_count = source.main_physical_page_count()?;
    copy_main_file(source, &dest_handle, physical_page_count)?;
    overlay_frozen_view(snapshot, &dest_handle, page_count)?;
    overlay_frozen_header(snapshot, &dest_handle)?;
    patch_destination_header(&dest_handle)?;
    dest_handle.sync_data(SyncMode::Full)?;
    Ok(())
}

/// Copy every page in `0..physical_page_count` from the source pager's
/// main file to `dest`. Reads bypass the WAL overlay — that's the
/// snapshot's job in [`overlay_frozen_view`].
///
/// The bound is the source's PHYSICAL high-water, not its
/// `page_count`. Pages in `[physical_page_count, page_count)` exist
/// only in the WAL view and are filled by [`overlay_frozen_view`];
/// reading them off the (too-short) main file here would
/// `UnexpectedEof`.
///
/// Bounded by `physical_page_count` (itself bounded by `page_count`);
/// a single page-sized scratch buffer is reused across the loop.
fn copy_main_file<F: FileBackend>(
    source: &Pager<F>,
    dest: &FileHandle,
    physical_page_count: u64,
) -> Result<()> {
    let mut buf = Page::zeroed();
    let page_size_u64 = PAGE_SIZE as u64;
    let mut id_raw: u64 = 0;
    while id_raw < physical_page_count {
        let off = id_raw
            .checked_mul(page_size_u64)
            .ok_or(Error::InvalidArgument("backup: byte-offset overflow"))?;
        if id_raw == 0 {
            source.read_main_file_page_zero(buf.as_bytes_mut())?;
        } else {
            let pid = PageId::new(id_raw)
                .ok_or(Error::InvalidArgument("backup: zero page id (impossible)"))?;
            let page = source.read_main_file_page(pid)?;
            buf.as_bytes_mut().copy_from_slice(page.as_bytes());
        }
        dest.write_all_at(buf.as_bytes(), off)?;
        id_raw = id_raw
            .checked_add(1)
            .ok_or(Error::InvalidArgument("backup: page id overflow"))?;
    }
    Ok(())
}

/// Overlay the snapshot's frozen WAL view onto `dest`. After this
/// returns, every page-id `<= page_count` whose body the snapshot
/// would observe via [`ReaderSnapshot::read_page`] carries that
/// observed body in `dest`.
fn overlay_frozen_view<F: FileBackend>(
    snapshot: &ReaderSnapshot<F>,
    dest: &FileHandle,
    page_count: u64,
) -> Result<()> {
    let page_size_u64 = PAGE_SIZE as u64;
    for (pid, page) in snapshot.frozen_pages() {
        if pid.get() >= page_count {
            continue;
        }
        let off = pid
            .get()
            .checked_mul(page_size_u64)
            .ok_or(Error::InvalidArgument("backup: byte-offset overflow"))?;
        dest.write_all_at(page.as_bytes(), off)?;
    }
    Ok(())
}

/// Overlay the snapshot's frozen page-0 header (if any) on top of
/// `dest`. This places the WAL-staged catalog root / freelist head /
/// page count into the destination's header BEFORE
/// [`patch_destination_header`] zeros the WAL salt.
fn overlay_frozen_header<F: FileBackend>(
    snapshot: &ReaderSnapshot<F>,
    dest: &FileHandle,
) -> Result<()> {
    if let Some(header_page) = snapshot.frozen_header() {
        dest.write_all_at(header_page.as_bytes(), 0)?;
    }
    Ok(())
}

/// Re-encode the destination's page-0 header with `wal_salt` zeroed
/// and the header CRC recomputed. After this the destination is
/// self-consistent: a `Db::open(dest)` will see no WAL salt and
/// will create a fresh empty WAL on first open.
fn patch_destination_header(dest: &FileHandle) -> Result<()> {
    let mut page = Page::zeroed();
    dest.read_exact_at(page.as_bytes_mut(), 0)?;
    let mut header: FileHeader = decode_header(&page)?;
    header.wal_salt = [0u8; 16];
    encode_header(&header, &mut page);
    dest.write_all_at(page.as_bytes(), 0)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::checksum::write_page_trailer;
    use crate::pager::{Config, Pager};
    use tempfile::TempDir;

    fn pid(n: u64) -> PageId {
        PageId::new(n).expect("non-zero")
    }

    /// A page body with a marker byte and a valid CRC32C trailer — the
    /// shape every real caller (B-tree / catalog) passes to `write_page`.
    fn stamped(marker: u8) -> Page {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[0] = marker;
        write_page_trailer(&mut p);
        p
    }

    /// A backup taken in the window BETWEEN a growing
    /// commit and its next checkpoint must round-trip the fresh pages —
    /// even though the source main file is physically SHORTER than its
    /// `page_count` (the fresh bodies live only in the WAL view). The
    /// `copy_main_file` loop is gated by the source's PHYSICAL high-water
    /// (so it never reads past the short file's EOF) and
    /// `overlay_frozen_view` fills the WAL-resident fresh pages.
    #[test]
    fn backup_between_growing_commit_and_checkpoint_round_trips() {
        let dir = TempDir::new().expect("tmp");
        let src = dir.path().join("src.obj");
        let dst = dir.path().join("backup.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);

        let (a, b) = {
            let mut p = Pager::open(&src, cfg).expect("open source");
            p.begin_txn();
            let a = p.alloc_page().expect("alloc a");
            p.write_page(a, &stamped(0xA1)).expect("write a");
            let _ = p.commit().expect("commit a");
            p.checkpoint().expect("checkpoint a");
            let b = p.alloc_page().expect("alloc b");
            p.write_page(b, &stamped(0xB2)).expect("write b");
            let _ = p.commit().expect("commit b");

            let physical = p.main_physical_page_count().expect("physical");
            assert!(
                b.get() >= physical,
                "test premise: fresh page `b` must be beyond the physical \
                 high-water (b={}, physical={physical})",
                b.get(),
            );

            let snap = p.reader_snapshot().expect("snap");
            backup_pager_to_path(&p, &snap, &dst).expect("backup");
            (a.get(), b.get())
        };

        let mut bp = Pager::open(&dst, Config::default()).expect("open backup");
        let ra = bp.read_page(pid(a)).expect("read a from backup");
        assert_eq!(ra.as_bytes()[0], 0xA1, "checkpointed page survives backup");
        let rb = bp.read_page(pid(b)).expect("read b from backup");
        assert_eq!(
            rb.as_bytes()[0],
            0xB2,
            "WAL-resident fresh page survives backup via overlay_frozen_view",
        );
    }
}
