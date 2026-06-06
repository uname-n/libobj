//! Page primitives: the fixed page size, the strongly-typed page id,
//! and the owned page buffer.
//!
//! All multi-byte integers in the on-disk page format are little-endian;
//! this module is the single place that knows that fact for the
//! pager. Encoders and decoders for individual page bodies live next
//! to their owners (header in [`super::header`], freelist in
//! [`super::freelist`]).

#![forbid(unsafe_code)]

use core::num::NonZeroU64;

/// Page size in bytes. Fixed at 4 KiB for format version 0.
/// Parameterising this is deferred; the header reserves the byte for
/// it.
pub const PAGE_SIZE: usize = 4096;

/// Number of bytes a CRC32C page trailer occupies. The trailer is
/// written by [`super::Pager::write_page`].
pub const PAGE_TRAILER_SIZE: usize = 4;

/// Per-page on-disk overhead added by the
/// encryption layer (24-byte XChaCha20-Poly1305 nonce + 16-byte
/// Poly1305 tag = 40 bytes). Defined as a public constant in
/// `pager::page` so callers that don't link the `encryption` Cargo
/// feature can still reason about the encrypted physical stride
/// (e.g. recovery tools that need to compute file offsets without
/// invoking any crypto). Kept in lock-step with
/// `crypto::ENCRYPTION_OVERHEAD` (24-byte nonce + 16-byte tag).
pub const ENCRYPTION_OVERHEAD: usize = 24 + 16;

/// Physical-stride helper. Returns the on-disk
/// size of a single non-header page given the file's
/// `feature_flags` bit-set:
///
/// - When encryption bit 1 is set: `PAGE_SIZE + ENCRYPTION_OVERHEAD`
///   = 4136 bytes (4096-byte ciphertext + 24-byte nonce + 16-byte
///   tag).
/// - Otherwise: `PAGE_SIZE` (4096 bytes).
///
/// Page 0 (the file header) is ALWAYS 4096 bytes regardless of
/// `feature_flags` — the header carries the plaintext signal that
/// tells a reader how to compute offsets for pages 1..N.
#[must_use]
pub const fn physical_page_stride(feature_flags: u32) -> usize {
    if feature_flags & (1u32 << 1) != 0 {
        PAGE_SIZE + ENCRYPTION_OVERHEAD
    } else {
        PAGE_SIZE
    }
}

/// Byte offset of page `raw_id`'s on-disk slot
/// for a file with the given `feature_flags`.
///
/// Page 0 is always at offset 0 (4096 bytes, plaintext). Pages 1..N
/// stride by [`physical_page_stride`] starting at byte 4096.
///
/// Total function: arithmetic uses `u64`, and a realistic file is
/// orders of magnitude below `u64::MAX / stride`. Returns
/// `u64::MAX` only on a contrived overflow input — the pager's
/// `alloc_fresh` checks the resulting file length against a real
/// `set_len` call which surfaces an `EINVAL`-style I/O error well
/// below this saturation point.
#[must_use]
pub fn physical_offset_for(raw_id: u64, feature_flags: u32) -> u64 {
    if raw_id == 0 {
        return 0;
    }
    let stride = physical_page_stride(feature_flags) as u64;
    let after_header = PAGE_SIZE as u64;
    let Some(rel) = raw_id.checked_sub(1).and_then(|n| n.checked_mul(stride)) else {
        return u64::MAX;
    };
    after_header.saturating_add(rel)
}

/// Identifier of a page in a database file.
///
/// `PageId` is `NonZeroU64` so the on-disk value `0` can be used as a
/// sentinel "no page" marker (e.g. the freelist-empty case). Encoding
/// the invariant in the type means neither the compiler nor a reviewer
/// has to remember it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(NonZeroU64);

impl PageId {
    /// Construct a `PageId` from a raw `u64`. Returns `None` if `raw`
    /// is `0`. Prefer this over the `unsafe` `NonZeroU64::new_unchecked`
    /// (no `unsafe` outside the platform layer).
    #[inline]
    #[must_use]
    pub const fn new(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(nz) => Some(Self(nz)),
            None => None,
        }
    }

    /// Construct a `PageId` from a `NonZeroU64`. Total function.
    #[must_use]
    pub const fn from_nonzero(nz: NonZeroU64) -> Self {
        Self(nz)
    }

    /// Get the underlying `u64`. Always non-zero.
    #[inline]
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Byte offset at which this page begins in the database file,
    /// for a file with the given `feature_flags`.
    ///
    /// The offset is stride-aware. Encrypted files (encryption
    /// bit set in `feature_flags`) use a 4136-byte physical stride
    /// for pages 1..N, so a fixed `id * PAGE_SIZE` computation is
    /// wrong for them. This delegates to [`physical_offset_for`],
    /// the same helper the pager uses for its real reads/writes
    /// (`Pager::physical_offset`), so the two never diverge. Page 0
    /// is always at offset 0. Pass `0` for plaintext (unencrypted)
    /// files.
    #[must_use]
    pub fn byte_offset(self, feature_flags: u32) -> u64 {
        physical_offset_for(self.0.get(), feature_flags)
    }
}

/// An owned, page-sized buffer. Lives in the cache's `Vec<Frame>` and
/// is reused across loads — the pager never allocates a new `Page` on
/// the read hot path.
#[derive(Debug, Clone)]
pub struct Page {
    bytes: Box<[u8; PAGE_SIZE]>,
}

impl Page {
    /// Allocate a new zeroed page. Called only during cache
    /// initialisation; never on the read hot path.
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            bytes: Box::new([0u8; PAGE_SIZE]),
        }
    }

    /// Get the page as a byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.bytes
    }

    /// Get the page as a mutable byte slice.
    pub fn as_bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        &mut self.bytes
    }

    /// Zero the entire page in place. Used when the cache evicts a
    /// frame and prepares it for reuse with new contents.
    pub fn zero(&mut self) {
        self.bytes.fill(0);
    }
}

impl Default for Page {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::{Page, PageId, ENCRYPTION_OVERHEAD, PAGE_SIZE};

    #[test]
    fn page_id_rejects_zero() {
        assert!(PageId::new(0).is_none());
        assert_eq!(PageId::new(1).map(PageId::get), Some(1));
    }

    #[test]
    fn page_id_byte_offset() {
        let id = PageId::new(3).expect("non-zero");
        assert_eq!(id.byte_offset(0), 3 * PAGE_SIZE as u64);
        let enc_flags = 1u32 << 1;
        assert_eq!(
            id.byte_offset(enc_flags),
            super::physical_offset_for(3, enc_flags)
        );
        assert_eq!(
            id.byte_offset(enc_flags),
            PAGE_SIZE as u64 + 2 * (PAGE_SIZE + ENCRYPTION_OVERHEAD) as u64
        );
    }

    #[test]
    fn page_zeroed_and_zero() {
        let mut p = Page::zeroed();
        assert!(p.as_bytes().iter().all(|&b| b == 0));
        p.as_bytes_mut()[0] = 0xAB;
        p.zero();
        assert!(p.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn physical_page_stride_picks_4096_or_4136() {
        assert_eq!(super::physical_page_stride(0), super::PAGE_SIZE);
        assert_eq!(super::physical_page_stride(0b01), super::PAGE_SIZE);
        assert_eq!(
            super::physical_page_stride(0b10),
            super::PAGE_SIZE + ENCRYPTION_OVERHEAD
        );
        assert_eq!(
            super::physical_page_stride(0b11),
            super::PAGE_SIZE + ENCRYPTION_OVERHEAD
        );
    }

    #[test]
    fn physical_offset_for_page_zero_is_zero() {
        assert_eq!(super::physical_offset_for(0, 0), 0);
        assert_eq!(super::physical_offset_for(0, 0b10), 0);
    }

    #[test]
    fn physical_offset_for_unencrypted_matches_legacy() {
        for raw in 1..32u64 {
            assert_eq!(
                super::physical_offset_for(raw, 0),
                raw * super::PAGE_SIZE as u64
            );
        }
    }

    #[test]
    fn physical_offset_for_encrypted_uses_4136_stride() {
        assert_eq!(super::physical_offset_for(1, 0b10), super::PAGE_SIZE as u64);
        assert_eq!(
            super::physical_offset_for(2, 0b10),
            (super::PAGE_SIZE + super::PAGE_SIZE + super::ENCRYPTION_OVERHEAD) as u64
        );
        assert_eq!(
            super::physical_offset_for(3, 0b10),
            (super::PAGE_SIZE + 2 * (super::PAGE_SIZE + super::ENCRYPTION_OVERHEAD)) as u64
        );
    }
}
