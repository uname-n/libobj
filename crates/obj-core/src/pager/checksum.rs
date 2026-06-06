//! CRC32C helpers used by both the page-0 header and the page trailer.
//!
//! The choice of CRC32C (Castagnoli) is fixed at format-major 0. This
//! module is the only place that calls into the `crc32c` crate so the
//! algorithm can be revisited in one edit.
//!
//! # Trailer formats
//!
//! Two trailer interpretations live side by side:
//!
//! - **v0** ([`write_page_trailer`] / [`page_trailer_valid`]) — the
//!   full 32-bit CRC32C of bytes `[0..PAGE_SIZE - PAGE_TRAILER_SIZE]`.
//!   Used by every non-header page in a `format_minor = 0` file, and
//!   by the in-memory representation of every page in every file
//!   (the pager re-stamps the on-disk trailer on `write_back_page`).
//! - **v1** ([`write_page_trailer_v1`] / [`page_trailer_valid_v1`] /
//!   [`page_trailer_flag_v1`]) — bit 31 of the trailer is the
//!   per-page compression flag; bits 0..30 are the 31-bit CRC32C of
//!   the on-disk bytes.

#![forbid(unsafe_code)]

use crate::pager::page::{Page, PAGE_SIZE, PAGE_TRAILER_SIZE};

/// Compute the CRC32C of `bytes` using the Castagnoli polynomial.
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes)
}

/// Continue a CRC32C computation: fold `bytes` into a running CRC
/// produced by a prior [`crc32c()`] / [`crc32c_append`] call. The
/// result is byte-identical to `crc32c(prefix ++ bytes)` where
/// `prefix` was the input that produced `crc`. This is the only
/// place that calls into the `crc32c` crate's incremental API, so
/// the WAL frame CRC can be computed over discontiguous segments
/// (header sans-CRC, zeroed CRC field, page body) without allocating
/// a contiguous scratch buffer.
#[must_use]
pub fn crc32c_append(crc: u32, bytes: &[u8]) -> u32 {
    crc32c::crc32c_append(crc, bytes)
}

/// Byte offset of the page trailer inside any non-header page.
pub const TRAILER_OFFSET: usize = PAGE_SIZE - PAGE_TRAILER_SIZE;

/// Mask isolating the lower 31 bits of the v1 trailer (CRC region).
pub const V1_CRC_MASK: u32 = 0x7FFF_FFFF;
/// Mask isolating bit 31 of the v1 trailer (compression flag).
pub const V1_FLAG_MASK: u32 = 0x8000_0000;

/// Compute the page trailer for `page` and write it into the last
/// [`PAGE_TRAILER_SIZE`] bytes (v0 interpretation: full 32-bit CRC).
pub fn write_page_trailer(page: &mut Page) {
    let buf = page.as_bytes_mut();
    let crc = crc32c(&buf[..TRAILER_OFFSET]);
    buf[TRAILER_OFFSET..].copy_from_slice(&crc.to_le_bytes());
}

/// `true` iff the page trailer matches the recomputed CRC32C of the
/// body (v0 interpretation). Caller decides what error to surface
/// on mismatch.
#[must_use]
pub fn page_trailer_valid(page: &Page) -> bool {
    let buf = page.as_bytes();
    let stored = read_trailer_u32(buf);
    let computed = crc32c(&buf[..TRAILER_OFFSET]);
    stored == computed
}

/// v1 trailer writer. Stamps the per-page
/// `compressed` flag into bit 31 and the 31-bit CRC32C of bytes
/// `[0..TRAILER_OFFSET]` into bits 0..30.
pub fn write_page_trailer_v1(page: &mut Page, compressed: bool) {
    let buf = page.as_bytes_mut();
    let crc = crc32c(&buf[..TRAILER_OFFSET]) & V1_CRC_MASK;
    let flag = if compressed { V1_FLAG_MASK } else { 0 };
    let trailer = crc | flag;
    buf[TRAILER_OFFSET..].copy_from_slice(&trailer.to_le_bytes());
}

/// v1 trailer verifier. Compares the lower 31
/// bits of the trailer against the recomputed 31-bit CRC32C of
/// bytes `[0..TRAILER_OFFSET]`. The flag bit (bit 31) is NOT part
/// of the checksum.
///
/// Residual risk (intentionally NOT fixed): masking bit 31
/// out of the CRC means the per-page compression flag is not
/// covered by page integrity. On a PLAINTEXT (`format_minor < 2`
/// path or non-encrypted) file, a bit-rot flip of bit 31 alone is
/// NOT detected here — the CRC over `[0..TRAILER_OFFSET]` still
/// matches, yet the flipped flag changes whether the body is
/// decoded as LZ4-compressed or raw, which decompression will
/// usually (but not always) reject downstream. On ENCRYPTED files
/// the body's AEAD tag covers the post-CRC physical bytes, so a
/// flipped flag fails Poly1305 verification and IS detected.
/// Including bit 31 in the CRC is deliberately avoided because it
/// would change the on-disk trailer of every existing v1 page and
/// break reading already-written files.
#[must_use]
pub fn page_trailer_valid_v1(page: &Page) -> bool {
    let buf = page.as_bytes();
    let stored = read_trailer_u32(buf) & V1_CRC_MASK;
    let computed = crc32c(&buf[..TRAILER_OFFSET]) & V1_CRC_MASK;
    stored == computed
}

/// Read the v1 compression flag (bit 31) from
/// the trailer. The caller is responsible for verifying the
/// trailer with [`page_trailer_valid_v1`] BEFORE consulting the
/// flag — an unverified buffer's flag is meaningless.
#[must_use]
pub fn page_trailer_flag_v1(page: &Page) -> bool {
    let buf = page.as_bytes();
    (read_trailer_u32(buf) & V1_FLAG_MASK) != 0
}

/// Internal helper: read the 4-byte trailer as a little-endian `u32`.
fn read_trailer_u32(buf: &[u8; PAGE_SIZE]) -> u32 {
    let mut t = [0u8; PAGE_TRAILER_SIZE];
    t.copy_from_slice(&buf[TRAILER_OFFSET..]);
    u32::from_le_bytes(t)
}

#[cfg(test)]
mod tests {
    use super::{
        page_trailer_flag_v1, page_trailer_valid, page_trailer_valid_v1, write_page_trailer,
        write_page_trailer_v1, V1_FLAG_MASK,
    };
    use crate::pager::page::Page;

    #[test]
    fn trailer_round_trip() {
        let mut p = Page::zeroed();
        for (i, b) in p.as_bytes_mut().iter_mut().enumerate().take(64) {
            *b = u8::try_from(i).expect("i < 64");
        }
        write_page_trailer(&mut p);
        assert!(page_trailer_valid(&p));
    }

    #[test]
    fn flipping_any_body_byte_invalidates_trailer() {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[100] = 0xAA;
        write_page_trailer(&mut p);
        assert!(page_trailer_valid(&p));
        p.as_bytes_mut()[42] ^= 0x01;
        assert!(!page_trailer_valid(&p));
    }

    #[test]
    fn flipping_trailer_byte_invalidates_trailer() {
        let mut p = Page::zeroed();
        write_page_trailer(&mut p);
        assert!(page_trailer_valid(&p));
        let len = p.as_bytes().len();
        p.as_bytes_mut()[len - 1] ^= 0x80;
        assert!(!page_trailer_valid(&p));
    }

    #[test]
    fn v1_trailer_round_trip_uncompressed() {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[10] = 0xAB;
        write_page_trailer_v1(&mut p, false);
        assert!(page_trailer_valid_v1(&p));
        assert!(!page_trailer_flag_v1(&p));
    }

    #[test]
    fn v1_trailer_round_trip_compressed() {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[10] = 0xCD;
        write_page_trailer_v1(&mut p, true);
        assert!(page_trailer_valid_v1(&p));
        assert!(page_trailer_flag_v1(&p));
    }

    #[test]
    fn v1_trailer_flag_independent_of_crc() {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[10] = 0xAB;
        write_page_trailer_v1(&mut p, false);
        assert!(page_trailer_valid_v1(&p));
        assert!(!page_trailer_flag_v1(&p));
        let len = p.as_bytes().len();
        let trailer_off = len - 4;
        let mut t = [0u8; 4];
        t.copy_from_slice(&p.as_bytes()[trailer_off..]);
        let mut trailer = u32::from_le_bytes(t);
        trailer ^= V1_FLAG_MASK;
        p.as_bytes_mut()[trailer_off..].copy_from_slice(&trailer.to_le_bytes());
        assert!(page_trailer_valid_v1(&p));
        assert!(page_trailer_flag_v1(&p));
    }

    #[test]
    fn v1_trailer_flipping_body_invalidates() {
        let mut p = Page::zeroed();
        write_page_trailer_v1(&mut p, true);
        assert!(page_trailer_valid_v1(&p));
        p.as_bytes_mut()[42] ^= 0x01;
        assert!(!page_trailer_valid_v1(&p));
    }
}
