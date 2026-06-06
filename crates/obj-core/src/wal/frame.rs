//! WAL frame encode / decode helpers.
//!
//! This module is the single place that knows the byte layout; the
//! parent [`super::Wal`] only calls into the helpers exposed here.

#![forbid(unsafe_code)]

use crate::pager::checksum::crc32c_append;
use crate::pager::page::PAGE_SIZE;

/// File magic for the WAL header. ASCII `OBJW`.
pub const WAL_MAGIC: [u8; 4] = *b"OBJW";

/// Fixed size of the WAL file header.
pub const WAL_HEADER_SIZE: usize = 64;

/// Fixed size of a per-frame header (preceding the page body).
pub const FRAME_HEADER_SIZE: usize = 64;

/// Per-frame size on disk for a **plaintext** WAL: header + page body.
/// Equal to 4160 bytes.
pub const FRAME_SIZE: usize = FRAME_HEADER_SIZE + PAGE_SIZE;

/// Per-frame AEAD suffix length for encrypted
/// WALs (24-byte XChaCha20-Poly1305 nonce + 16-byte Poly1305 tag = 40
/// bytes). Frames in encrypted WALs are
/// `FRAME_SIZE + FRAME_AEAD_SUFFIX_SIZE = 4200` bytes on disk; the
/// extra bytes sit AFTER the ciphertext body.
pub const FRAME_AEAD_SUFFIX_SIZE: usize = 24 + 16;

/// Per-frame size on disk for an **encrypted** WAL.
pub const FRAME_SIZE_ENCRYPTED: usize = FRAME_SIZE + FRAME_AEAD_SUFFIX_SIZE;

/// Pick the right per-frame size given the `encrypted` flag.
#[must_use]
pub const fn frame_size_for(encrypted: bool) -> usize {
    if encrypted {
        FRAME_SIZE_ENCRYPTED
    } else {
        FRAME_SIZE
    }
}

const OFF_PAGE_ID: usize = 0;
const OFF_LSN: usize = 8;
const OFF_SALT: usize = 16;
const OFF_FLAGS: usize = 20;
const OFF_CRC: usize = 60;

const FLAG_COMMIT: u8 = 0x01;

/// In-memory representation of a WAL frame header.
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Page-id whose payload this frame replaces.
    pub page_id: u64,
    /// Monotonic per-WAL-generation log sequence number.
    pub lsn: u64,
    /// WAL generation salt; must match the WAL header.
    pub salt: u32,
    /// `true` iff this frame is the last in its transaction (commit
    /// marker).
    pub commit: bool,
}

/// Encode `header` into the first [`FRAME_HEADER_SIZE`] bytes of
/// `buf` AND compute the per-frame CRC32C covering the frame header
/// (with the CRC field zeroed) plus the page body that follows.
///
/// `buf.len()` must equal [`FRAME_SIZE`]; the page body must already
/// have been written into `buf[FRAME_HEADER_SIZE..]`.
///
/// On encrypted WALs the CRC is computed
/// over (`header_sans_crc` + PLAINTEXT body); the caller MUST pass
/// a buffer whose `[FRAME_HEADER_SIZE..FRAME_SIZE]` slice carries
/// the plaintext body before invoking this helper. Encryption of
/// the body happens AFTER `encode_frame_header` returns (see
/// `Wal::write_frame`).
pub fn encode_frame_header(header: &FrameHeader, buf: &mut [u8]) {
    debug_assert_eq!(buf.len(), FRAME_SIZE, "frame buffer must be FRAME_SIZE");
    for b in buf.iter_mut().take(FRAME_HEADER_SIZE) {
        *b = 0;
    }
    buf[OFF_PAGE_ID..OFF_PAGE_ID + 8].copy_from_slice(&header.page_id.to_le_bytes());
    buf[OFF_LSN..OFF_LSN + 8].copy_from_slice(&header.lsn.to_le_bytes());
    buf[OFF_SALT..OFF_SALT + 4].copy_from_slice(&header.salt.to_le_bytes());
    buf[OFF_FLAGS] = if header.commit { FLAG_COMMIT } else { 0 };
    let crc = compute_frame_crc(buf);
    buf[OFF_CRC..OFF_CRC + 4].copy_from_slice(&crc.to_le_bytes());
}

/// Decode and validate a frame from `buf`. Returns `None` if the
/// CRC does not validate, if the salt does not match
/// `expected_salt`, or if reserved bytes are non-zero.
///
/// Caller-side: `None` means "tail" — recovery stops here.
#[must_use]
pub fn decode_frame_header(buf: &[u8], expected_salt: u32) -> Option<FrameHeader> {
    match decode_frame_header_classified(buf, expected_salt) {
        FrameDecode::Ok(header) => Some(header),
        FrameDecode::SaltMismatch | FrameDecode::CrcInvalid | FrameDecode::Malformed => None,
    }
}

/// Classified outcome of decoding a single WAL frame. Pass-2 of recovery
/// distinguishes "torn tail / stale generation" (silently discarded)
/// from "CRC mismatch in a frame that should have been valid" (which
/// surfaces as `Error::WalCorruption`).
#[derive(Debug)]
pub enum FrameDecode {
    /// Salt matches and CRC validates — a usable frame.
    Ok(FrameHeader),
    /// The frame's salt does not match `expected_salt`. The frame may
    /// be torn-tail bytes from a previous generation or an in-progress
    /// torn write; in either case it is **not** corruption.
    SaltMismatch,
    /// Salt matches but the CRC32C does not validate. In pass 2 (frames
    /// before the last commit marker) this is `Error::WalCorruption`;
    /// in pass 1 / past the last commit it is torn tail.
    CrcInvalid,
    /// Reserved-flag bits or buffer-length problem. Treated as torn
    /// tail by recovery (forward-compat boundary).
    Malformed,
}

/// Classify a single frame buffer. The buffer length must equal
/// [`FRAME_SIZE`]; otherwise [`FrameDecode::Malformed`] is returned.
#[must_use]
pub fn decode_frame_header_classified(buf: &[u8], expected_salt: u32) -> FrameDecode {
    if buf.len() != FRAME_SIZE {
        return FrameDecode::Malformed;
    }
    let salt = u32::from_le_bytes([
        buf[OFF_SALT],
        buf[OFF_SALT + 1],
        buf[OFF_SALT + 2],
        buf[OFF_SALT + 3],
    ]);
    if salt != expected_salt {
        return FrameDecode::SaltMismatch;
    }
    let stored_crc = u32::from_le_bytes([
        buf[OFF_CRC],
        buf[OFF_CRC + 1],
        buf[OFF_CRC + 2],
        buf[OFF_CRC + 3],
    ]);
    let computed = compute_frame_crc(buf);
    if stored_crc != computed {
        return FrameDecode::CrcInvalid;
    }
    let flags = buf[OFF_FLAGS];
    if flags & !FLAG_COMMIT != 0 {
        return FrameDecode::Malformed;
    }
    let page_id = u64::from_le_bytes([
        buf[OFF_PAGE_ID],
        buf[OFF_PAGE_ID + 1],
        buf[OFF_PAGE_ID + 2],
        buf[OFF_PAGE_ID + 3],
        buf[OFF_PAGE_ID + 4],
        buf[OFF_PAGE_ID + 5],
        buf[OFF_PAGE_ID + 6],
        buf[OFF_PAGE_ID + 7],
    ]);
    let lsn = u64::from_le_bytes([
        buf[OFF_LSN],
        buf[OFF_LSN + 1],
        buf[OFF_LSN + 2],
        buf[OFF_LSN + 3],
        buf[OFF_LSN + 4],
        buf[OFF_LSN + 5],
        buf[OFF_LSN + 6],
        buf[OFF_LSN + 7],
    ]);
    FrameDecode::Ok(FrameHeader {
        page_id,
        lsn,
        salt,
        commit: flags & FLAG_COMMIT != 0,
    })
}

/// Byte offset of the frame at index `frame_index` (0-based, where
/// 0 is the first frame after the WAL header).
///
/// `frame_index` is caller-controlled (this is
/// a `pub` helper used by tests and by external WAL inspection
/// utilities). Saturate at `u64::MAX` on overflow rather than
/// panicking — every production caller passes an index bounded by
/// `committed_frames`, but a fuzz / forensic caller could pass an
/// arbitrarily-large index and `overflow-checks = true`
/// would otherwise turn the multiply into a panic.
///
/// `frame_size` is the on-disk per-frame stride
/// (4160 for unencrypted WALs, 4188 for encrypted ones). Use
/// [`frame_size_for`] to pick the right value.
#[must_use]
pub fn frame_offset(frame_index: u64, frame_size: usize) -> u64 {
    frame_index
        .checked_mul(frame_size as u64)
        .and_then(|product| product.checked_add(WAL_HEADER_SIZE as u64))
        .unwrap_or(u64::MAX)
}

/// Compute the CRC32C over (frame header with its CRC field zeroed)
/// ++ (page body), folding the three contiguous segments directly
/// into the running CRC via [`crc32c_append`] — no scratch buffer,
/// no memcpy.
///
/// The segments, in order, are exactly the bytes the previous
/// memcpy-based implementation laid out in its linear scratch:
///   1. `buf[0..OFF_CRC]`          — header bytes before the CRC field
///   2. four zero bytes            — the zeroed CRC field (`[OFF_CRC, 64)`)
///   3. `buf[OFF_CRC + 4..]`       — the full ~4096-byte page body
///
/// Because `OFF_CRC + 4 == FRAME_HEADER_SIZE` (the CRC field is the
/// final 4 bytes of the 64-byte header), segment 3 is precisely the
/// page body that followed the header in the old linear scratch —
/// there are no reserved header bytes between the CRC field and the
/// body. The `debug_assert!` below pins that invariant.
///
/// `crc32c` is `crc32c_append(0, ..)` and CRC32C is computed strictly
/// left-to-right over the byte stream, so folding the segments
/// incrementally is byte-identical to one `crc32c` call over their
/// concatenation. The output therefore matches the prior
/// scratch-buffer implementation bit-for-bit, which is mandatory:
/// this function is shared by writer encode AND reader/recovery
/// decode, and any divergence would silently corrupt the on-disk
/// format.
fn compute_frame_crc(buf: &[u8]) -> u32 {
    debug_assert_eq!(buf.len(), FRAME_SIZE);
    debug_assert_eq!(OFF_CRC + 4, FRAME_HEADER_SIZE, "CRC field ends the header");
    let crc = crc32c_append(0, &buf[..OFF_CRC]);
    let crc = crc32c_append(crc, &[0u8; 4]);
    crc32c_append(crc, &buf[OFF_CRC + 4..])
}

#[cfg(test)]
mod tests {
    use super::{
        compute_frame_crc, decode_frame_header, encode_frame_header, FrameHeader,
        FRAME_HEADER_SIZE, FRAME_SIZE, OFF_CRC,
    };
    use crate::pager::checksum::crc32c;

    /// Reference implementation: the original 3-step, single-`crc32c`
    /// algorithm (build a contiguous `[u8; FRAME_SIZE]` scratch with
    /// the full header, zero the CRC field, copy in the body, hash
    /// once). The production `compute_frame_crc` MUST agree with this
    /// for every input — it is the on-disk format contract shared by
    /// the writer and by recovery.
    fn compute_frame_crc_reference(buf: &[u8]) -> u32 {
        assert_eq!(buf.len(), FRAME_SIZE);
        let mut linear = [0u8; FRAME_SIZE];
        linear[..FRAME_HEADER_SIZE].copy_from_slice(&buf[..FRAME_HEADER_SIZE]);
        for b in &mut linear[OFF_CRC..OFF_CRC + 4] {
            *b = 0;
        }
        linear[FRAME_HEADER_SIZE..].copy_from_slice(&buf[FRAME_HEADER_SIZE..]);
        crc32c(&linear)
    }

    /// THE load-bearing test: the `crc32c_append`-based
    /// `compute_frame_crc` must be byte-identical to the original
    /// memcpy reference for a spread of pseudo-random `(header, body)`
    /// inputs, INCLUDING non-zero CRC-field bytes (which both
    /// implementations must ignore by treating that region as zero).
    #[test]
    fn crc_byte_identical_to_memcpy_reference() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..256u32 {
            let mut buf = [0u8; FRAME_SIZE];
            for b in &mut buf {
                *b = u8::try_from(next() & 0xFF).expect("masked to a byte");
            }
            let got = compute_frame_crc(&buf);
            let want = compute_frame_crc_reference(&buf);
            assert_eq!(
                got, want,
                "case {case}: crc32c_append result diverged from memcpy reference"
            );
        }
        let zero = [0u8; FRAME_SIZE];
        assert_eq!(compute_frame_crc(&zero), compute_frame_crc_reference(&zero));
        let ones = [0xFFu8; FRAME_SIZE];
        assert_eq!(compute_frame_crc(&ones), compute_frame_crc_reference(&ones));
    }

    /// The CRC must be independent of whatever bytes already sit in
    /// the CRC field — both the new impl and the reference treat that
    /// 4-byte region as zero.
    #[test]
    fn crc_ignores_stale_crc_field_bytes() {
        let mut a = [0xABu8; FRAME_SIZE];
        for (i, b) in a.iter_mut().enumerate() {
            *b = u8::try_from(i & 0xFF).expect("masked");
        }
        let mut b = a;
        a[OFF_CRC..OFF_CRC + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        b[OFF_CRC..OFF_CRC + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(compute_frame_crc(&a), compute_frame_crc(&b));
        assert_eq!(compute_frame_crc(&a), compute_frame_crc_reference(&a));
    }

    #[test]
    fn round_trip_basic_frame() {
        let header = FrameHeader {
            page_id: 7,
            lsn: 42,
            salt: 0xDEAD_BEEF,
            commit: true,
        };
        let mut buf = vec![0u8; FRAME_SIZE];
        for (i, b) in buf.iter_mut().enumerate().skip(64).take(128) {
            *b = u8::try_from(i & 0xFF).expect("masked");
        }
        encode_frame_header(&header, &mut buf);
        let decoded = decode_frame_header(&buf, 0xDEAD_BEEF).expect("decode");
        assert_eq!(decoded.page_id, 7);
        assert_eq!(decoded.lsn, 42);
        assert_eq!(decoded.salt, 0xDEAD_BEEF);
        assert!(decoded.commit);
    }

    #[test]
    fn salt_mismatch_yields_tail() {
        let header = FrameHeader {
            page_id: 1,
            lsn: 1,
            salt: 1,
            commit: true,
        };
        let mut buf = vec![0u8; FRAME_SIZE];
        encode_frame_header(&header, &mut buf);
        assert!(decode_frame_header(&buf, 2).is_none());
    }

    #[test]
    fn flipped_body_invalidates_crc() {
        let header = FrameHeader {
            page_id: 1,
            lsn: 1,
            salt: 1,
            commit: false,
        };
        let mut buf = vec![0u8; FRAME_SIZE];
        buf[64 + 50] = 0xAA;
        encode_frame_header(&header, &mut buf);
        assert!(decode_frame_header(&buf, 1).is_some());
        buf[64 + 50] ^= 0x01;
        assert!(decode_frame_header(&buf, 1).is_none());
    }

    #[test]
    fn unknown_flag_bits_are_tail() {
        let header = FrameHeader {
            page_id: 1,
            lsn: 1,
            salt: 1,
            commit: true,
        };
        let mut buf = vec![0u8; FRAME_SIZE];
        encode_frame_header(&header, &mut buf);
        buf[20] = 0x80;
        let crc = super::compute_frame_crc(&buf);
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
        assert!(decode_frame_header(&buf, 1).is_none());
    }
}
