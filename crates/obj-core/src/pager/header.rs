//! Page-0 file header — encode / decode.
//!
//! Field offsets and
//! sizes are encoded as `const` items rather than magic numbers in the
//! function bodies so a reviewer can audit the layout against the spec
//! at a glance.

#![forbid(unsafe_code)]

use crate::error::{Error, Result};
use crate::pager::checksum::crc32c;
use crate::pager::page::{Page, PAGE_SIZE};

/// File magic. ASCII `OBJF`.
pub const MAGIC: [u8; 4] = *b"OBJF";

/// Format major version implemented by this build.
///
/// Every new database the v1.0.0 writer creates stamps
/// `format_major = 1` on page 0. Readers accept `format_major ∈
/// {0, 1}` so pre-1.0 (0.x-era) databases continue to open
/// without a migration tool; see `SUPPORTED_FORMAT_MAJORS`
/// (crate-private).
pub const FORMAT_MAJOR: u16 = 1;
/// The set of `format_major` values this build's reader accepts.
///
/// - `0` — pre-1.0 (0.x) databases. Read-compatible only; v1.0
///   writers never produce new `format_major = 0` files.
/// - `1` — v1.0 frozen wire format (this build's writer).
///
/// A future v2.0 build will reject `format_major = 0` and `1` in
/// favour of `2`.
///
/// Private to the crate so the v1.0 public API surface stays
/// pinned; the WAL recovery path
/// re-exports the helper [`is_supported_format_major`].
pub(crate) const SUPPORTED_FORMAT_MAJORS: &[u16] = &[0, 1];

/// Crate-private predicate over [`SUPPORTED_FORMAT_MAJORS`].
///
/// Exposed for the WAL header reader, which validates that a
/// `-wal` sidecar's `format_major` field is in the same set the
/// page-0 decoder accepts. Public on `pub(crate)` to avoid
/// duplicating the slice in the WAL module while keeping the v1.0
/// public-API surface unchanged.
#[must_use]
pub(crate) fn is_supported_format_major(format_major: u16) -> bool {
    SUPPORTED_FORMAT_MAJORS.contains(&format_major)
}
/// Format minor version implemented by this build.
///
/// - `0` — pre-1.0 baseline (`format_major = 0` only). Full-32-bit
///   CRC32C per-page trailer; no compression, no encryption.
/// - `1` — pre-1.0 compression-capable layout (`format_major = 0`
///   only). When `feature_flags` bit 0 is set, every non-header
///   page on disk uses the v1 trailer interpretation (bit 31 =
///   "page is LZ4-compressed", bits 0..30 = 31-bit CRC32C).
/// - `2` — feature-complete encryption-capable layout. The ONLY
///   valid `format_minor` for `format_major = 1` files. Page 0
///   carries a 32-byte `kdf_salt` field at offset 72..104, and
///   when `feature_flags` bit 1 is set every non-header page on
///   disk is encrypted with XChaCha20-Poly1305 (4136-byte
///   physical stride: 4096 ciphertext + 24-byte nonce + 16-byte
///   tag). Compression (bit 0) and encryption (bit 1) compose:
///   compress first, encrypt second.
///
/// `FORMAT_MINOR` is frozen at `2` for the
/// indefinite v1.x series — no further minor bumps without a
/// `format_major` 2.0 release.
pub const FORMAT_MINOR: u16 = 2;

/// `feature_flags` bit indicating the file uses per-page LZ4
/// compression. Set on creation by a `Pager`
/// opened with `Config::compression_mode = CompressionMode::Lz4`.
pub const FEATURE_FLAG_COMPRESSION: u32 = 1 << 0;

/// `feature_flags` bit indicating the file uses per-page
/// `XChaCha20-Poly1305` encryption. Set on
/// creation by a `Pager` whose `Config::encryption_key` is
/// `Some(_)`. When this bit is set, every non-header page on
/// disk is `4096 + 24 + 16 = 4136` bytes physical (ciphertext
/// plus nonce plus Poly1305 tag), and the page-0 `kdf_salt`
/// field (offset 72..104) carries the 32-byte HKDF-SHA256 salt
/// used to derive the per-file page encryption key from the
/// caller's user key.
pub const FEATURE_FLAG_ENCRYPTION: u32 = 1 << 1;

/// Mask of all `feature_flags` bits this build understands. Any
/// bit set in the on-disk header but NOT in this mask is rejected
/// at open time with [`Error::InvalidFormat`]: an unknown flag
/// might change how subsequent bytes are interpreted, and a reader
/// that does not know what it means MUST refuse to guess.
pub const FEATURE_FLAGS_KNOWN: u32 = FEATURE_FLAG_COMPRESSION | FEATURE_FLAG_ENCRYPTION;

/// `PAGE_SIZE` (4096) expressed as a `u16` for the on-disk header
/// field. A `const` assertion below pins the value so the cast is
/// audit-grade rather than a magic literal.
const PAGE_SIZE_U16: u16 = 4096;
const _: () = assert!(PAGE_SIZE_U16 as usize == PAGE_SIZE);

const OFF_MAGIC: usize = 0;
const OFF_FORMAT_MAJOR: usize = 4;
const OFF_FORMAT_MINOR: usize = 6;
const OFF_PAGE_SIZE: usize = 8;
const OFF_FEATURE_FLAGS: usize = 10;
const OFF_PAGE_COUNT: usize = 16;
const OFF_ROOT_CATALOG: usize = 24;
const OFF_FREELIST_HEAD: usize = 32;
const OFF_WAL_SALT: usize = 40;
const OFF_FILE_UUID: usize = 56;
const OFF_KDF_SALT: usize = 72;
const OFF_HEADER_CRC: usize = PAGE_SIZE - 4;

/// In-memory representation of the page-0 file header.
///
/// Constructed by [`decode_header`] or by the pager when initialising
/// a new file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// Format major version. Must equal [`FORMAT_MAJOR`].
    pub format_major: u16,
    /// Format minor version. Must satisfy `<=` [`FORMAT_MINOR`] for
    /// write access; readers tolerate higher minors.
    pub format_minor: u16,
    /// On-disk page size. Must equal [`PAGE_SIZE`] at format major 0.
    pub page_size: u16,
    /// Per-file feature-bit mask. Bit 0 =
    /// "uses LZ4 page compression"; other bits reserved (MUST be
    /// zero — readers reject unknown bits as
    /// [`Error::InvalidFormat`]).
    pub feature_flags: u32,
    /// Number of pages in the file, including page 0.
    pub page_count: u64,
    /// Root catalog page-id, or `0` if the catalog is empty.
    pub root_catalog: u64,
    /// First page on the freelist, or `0` if the freelist is empty.
    pub freelist_head: u64,
    /// Salt for WAL frame hashes.
    pub wal_salt: [u8; 16],
    /// Stable file UUID.
    pub file_uuid: [u8; 16],
    /// 32-byte salt for the HKDF-SHA256
    /// per-file page-key derivation. Plaintext on disk (page 0
    /// is never encrypted); the file's actual page-encryption
    /// key is `HKDF-SHA256(ikm=user_key, salt=kdf_salt,
    /// info=b"obj-page-encryption-v1")`. Always zero on
    /// `format_minor < 2` files; CSPRNG-generated on creation
    /// of `format_minor = 2` files with `feature_flags` bit 1
    /// set.
    ///
    /// Integrity posture: the `kdf_salt` lives in the
    /// plaintext page-0 header and is protected ONLY by the
    /// header's own CRC. It is NOT bound into any page's AEAD
    /// associated data (page AD is just `page_id`; see
    /// `crypto.rs`), so the AEAD tag does not authenticate it.
    /// Its integrity therefore rests on two independent layers:
    /// (1) the page-0 header CRC detects accidental corruption,
    /// and (2) any tampering that survives the CRC changes the
    /// derived page key, which surfaces as
    /// `Error::EncryptionKeyInvalid` (wrong-key detection) on the
    /// first page decrypt rather than as silent plaintext
    /// disclosure. Binding the salt into page AD is deliberately
    /// NOT done — it would be a format-affecting change.
    pub kdf_salt: [u8; 32],
}

impl FileHeader {
    /// Header for a freshly-initialised database: just page 0, no
    /// catalog, empty freelist, zero WAL salt and UUID.
    ///
    /// Every v1.0 writer stamps
    /// `format_major = 1, format_minor = 2` — the feature-complete
    /// frozen baseline. `feature_flags = 0` because this constructor
    /// produces a plain (no-compression, no-encryption) file; the
    /// other `new_empty_*` constructors set the corresponding
    /// `feature_flags` bits.
    #[must_use]
    pub const fn new_empty() -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: PAGE_SIZE_U16,
            feature_flags: 0,
            page_count: 1,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: [0; 32],
        }
    }

    /// Header for a freshly-initialised
    /// compression-capable database. `feature_flags` bit 0 set;
    /// `format_minor` is the frozen v1.0 feature-complete value
    /// ([`FORMAT_MINOR`] = 2). Everything else matches
    /// [`FileHeader::new_empty`].
    #[must_use]
    pub const fn new_empty_with_compression() -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: PAGE_SIZE_U16,
            feature_flags: FEATURE_FLAG_COMPRESSION,
            page_count: 1,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: [0; 32],
        }
    }

    /// Header for a freshly-initialised
    /// encryption-capable database. `format_minor = 2`,
    /// `feature_flags` bit 1 set, `kdf_salt` populated from the
    /// caller-supplied CSPRNG bytes. Compression (bit 0) is
    /// left OFF; the higher-level
    /// [`FileHeader::new_empty_with_encryption_and_compression`]
    /// constructor sets both bits.
    #[must_use]
    pub const fn new_empty_with_encryption(kdf_salt: [u8; 32]) -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: PAGE_SIZE_U16,
            feature_flags: FEATURE_FLAG_ENCRYPTION,
            page_count: 1,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt,
        }
    }

    /// Header for a freshly-initialised
    /// database that uses BOTH compression AND encryption. The
    /// layering order is compress-then-encrypt: the 4092-byte
    /// raw body is compressed, the resulting
    /// 4096-byte logical page is encrypted, and
    /// the encrypted ciphertext (+ nonce + tag) lands on disk
    /// as a 4136-byte physical page.
    #[must_use]
    pub const fn new_empty_with_encryption_and_compression(kdf_salt: [u8; 32]) -> Self {
        Self {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: PAGE_SIZE_U16,
            feature_flags: FEATURE_FLAG_COMPRESSION | FEATURE_FLAG_ENCRYPTION,
            page_count: 1,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt,
        }
    }
}

/// Encode `header` into `page`. Invariants the
/// caller is supposed to uphold are documented via `debug_assert!`.
pub fn encode_header(header: &FileHeader, page: &mut Page) {
    debug_assert_eq!(
        header.page_size as usize, PAGE_SIZE,
        "every supported format major fixes PAGE_SIZE at 4096",
    );
    debug_assert!(
        SUPPORTED_FORMAT_MAJORS.contains(&header.format_major),
        "encoder only writes a format major this build supports",
    );

    let buf = page.as_bytes_mut();
    buf.fill(0);
    buf[OFF_MAGIC..OFF_MAGIC + 4].copy_from_slice(&MAGIC);
    buf[OFF_FORMAT_MAJOR..OFF_FORMAT_MAJOR + 2].copy_from_slice(&header.format_major.to_le_bytes());
    buf[OFF_FORMAT_MINOR..OFF_FORMAT_MINOR + 2].copy_from_slice(&header.format_minor.to_le_bytes());
    buf[OFF_PAGE_SIZE..OFF_PAGE_SIZE + 2].copy_from_slice(&header.page_size.to_le_bytes());
    buf[OFF_FEATURE_FLAGS..OFF_FEATURE_FLAGS + 4]
        .copy_from_slice(&header.feature_flags.to_le_bytes());
    buf[OFF_PAGE_COUNT..OFF_PAGE_COUNT + 8].copy_from_slice(&header.page_count.to_le_bytes());
    buf[OFF_ROOT_CATALOG..OFF_ROOT_CATALOG + 8].copy_from_slice(&header.root_catalog.to_le_bytes());
    buf[OFF_FREELIST_HEAD..OFF_FREELIST_HEAD + 8]
        .copy_from_slice(&header.freelist_head.to_le_bytes());
    buf[OFF_WAL_SALT..OFF_WAL_SALT + 16].copy_from_slice(&header.wal_salt);
    buf[OFF_FILE_UUID..OFF_FILE_UUID + 16].copy_from_slice(&header.file_uuid);
    buf[OFF_KDF_SALT..OFF_KDF_SALT + 32].copy_from_slice(&header.kdf_salt);

    let crc = crc32c(&buf[..OFF_HEADER_CRC]);
    buf[OFF_HEADER_CRC..OFF_HEADER_CRC + 4].copy_from_slice(&crc.to_le_bytes());
}

/// Decode the page-0 header from the given `page` buffer. Validates
/// magic, page-size, major version, the per-major `format_minor`
/// constraint, and the `header_crc32c` field.
///
/// Readers accept any
/// `SUPPORTED_FORMAT_MAJORS` value (`0` for pre-1.0 databases,
/// `1` for v1.0+). The per-major `format_minor` constraint is:
///
/// - `format_major = 0` → `format_minor ∈ {0, 1, 2}` (the pre-1.0
///   incremental rollout: baseline, compression-capable,
///   encryption-capable).
/// - `format_major = 1` → `format_minor = 2` (the v1.0 frozen
///   feature-complete value; the only valid minor inside the v1.x
///   series).
///
/// # Errors
///
/// - [`Error::InvalidFormat`] if the magic bytes do not match, if
///   `format_major` is unsupported by this build, if
///   `format_minor` is not valid for the file's `format_major`,
///   or if `page_size` disagrees with [`PAGE_SIZE`].
/// - [`Error::Corruption`] with `page_id = 0` if the stored
///   `header_crc32c` does not match the CRC32C of the rest of the
///   page.
pub fn decode_header(page: &Page) -> Result<FileHeader> {
    let buf = page.as_bytes();
    if buf[OFF_MAGIC..OFF_MAGIC + 4] != MAGIC {
        return Err(Error::InvalidFormat {
            reason: "magic bytes are not OBJF",
        });
    }
    let format_major = u16::from_le_bytes(read_array(buf, OFF_FORMAT_MAJOR));
    if !SUPPORTED_FORMAT_MAJORS.contains(&format_major) {
        return Err(Error::InvalidFormat {
            reason: "format-major version not supported",
        });
    }
    let format_minor = u16::from_le_bytes(read_array::<2>(buf, OFF_FORMAT_MINOR));
    if !is_supported_minor(format_major, format_minor) {
        return Err(Error::InvalidFormat {
            reason: "format-minor not valid for the file's format-major",
        });
    }
    let page_size = u16::from_le_bytes(read_array(buf, OFF_PAGE_SIZE));
    if usize::from(page_size) != PAGE_SIZE {
        return Err(Error::InvalidFormat {
            reason: "page size does not match this build",
        });
    }
    let stored_crc = u32::from_le_bytes(read_array::<4>(buf, OFF_HEADER_CRC));
    let computed_crc = crc32c(&buf[..OFF_HEADER_CRC]);
    if stored_crc != computed_crc {
        return Err(Error::Corruption { page_id: 0 });
    }
    let feature_flags = u32::from_le_bytes(read_array::<4>(buf, OFF_FEATURE_FLAGS));
    if feature_flags & !FEATURE_FLAGS_KNOWN != 0 {
        return Err(Error::InvalidFormat {
            reason: "unknown feature_flags bit set on page-0 header",
        });
    }
    let reserved_after_flags =
        u16::from_le_bytes([buf[OFF_FEATURE_FLAGS + 4], buf[OFF_FEATURE_FLAGS + 5]]);
    if reserved_after_flags != 0 {
        return Err(Error::InvalidFormat {
            reason: "reserved bytes after feature_flags must be zero",
        });
    }
    Ok(FileHeader {
        format_major,
        format_minor,
        page_size,
        feature_flags,
        page_count: u64::from_le_bytes(read_array(buf, OFF_PAGE_COUNT)),
        root_catalog: u64::from_le_bytes(read_array(buf, OFF_ROOT_CATALOG)),
        freelist_head: u64::from_le_bytes(read_array(buf, OFF_FREELIST_HEAD)),
        wal_salt: read_array(buf, OFF_WAL_SALT),
        file_uuid: read_array(buf, OFF_FILE_UUID),
        kdf_salt: read_array(buf, OFF_KDF_SALT),
    })
}

/// Read a fixed-size array out of the page buffer. Used by
/// [`decode_header`] to avoid `unwrap` on `try_into`.
fn read_array<const N: usize>(buf: &[u8; PAGE_SIZE], off: usize) -> [u8; N] {
    debug_assert!(off + N <= PAGE_SIZE, "header field out of bounds");
    let mut out = [0u8; N];
    out.copy_from_slice(&buf[off..off + N]);
    out
}

/// Per-major `format_minor` enforcement.
///
/// v1.0 freezes `format_minor = 2` as the only minor for
/// `format_major = 1`; pre-1.0 (`format_major = 0`) files keep
/// their historical `format_minor ∈ {0, 1, 2}` range. Any other
/// pairing (including any major outside [`SUPPORTED_FORMAT_MAJORS`])
/// returns `false`; the caller surfaces that as
/// [`Error::InvalidFormat`].
pub(crate) fn is_supported_minor(format_major: u16, format_minor: u16) -> bool {
    match format_major {
        0 => (0..=2).contains(&format_minor),
        1 => format_minor == FORMAT_MINOR,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_header, encode_header, FileHeader, FORMAT_MAJOR, FORMAT_MINOR};
    use crate::pager::page::Page;

    #[test]
    fn round_trip_default_header() {
        let h = FileHeader::new_empty();
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        let decoded = decode_header(&p).expect("encode/decode round-trip");
        assert_eq!(decoded, h);
    }

    #[test]
    fn round_trip_non_default_header() {
        let h = FileHeader {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: 4096,
            feature_flags: 0,
            page_count: 17,
            root_catalog: 2,
            freelist_head: 3,
            wal_salt: [0xAA; 16],
            file_uuid: [0xCC; 16],
            kdf_salt: [0; 32],
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        assert_eq!(decode_header(&p).expect("round-trip"), h);
    }

    #[test]
    fn round_trip_compression_header() {
        let h = FileHeader {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: 4096,
            feature_flags: super::FEATURE_FLAG_COMPRESSION,
            page_count: 5,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: [0; 32],
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        assert_eq!(decode_header(&p).expect("round-trip"), h);
    }

    #[test]
    fn round_trip_encryption_header() {
        let mut salt = [0u8; 32];
        for (i, b) in salt.iter_mut().enumerate() {
            *b = u8::try_from(i & 0xFF).unwrap_or(0);
        }
        let h = FileHeader {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: 4096,
            feature_flags: super::FEATURE_FLAG_ENCRYPTION,
            page_count: 5,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: salt,
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        assert_eq!(decode_header(&p).expect("round-trip"), h);
    }

    #[test]
    fn round_trip_encryption_and_compression_header() {
        let h = FileHeader {
            format_major: FORMAT_MAJOR,
            format_minor: FORMAT_MINOR,
            page_size: 4096,
            feature_flags: super::FEATURE_FLAG_COMPRESSION | super::FEATURE_FLAG_ENCRYPTION,
            page_count: 5,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: [0x77; 32],
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        assert_eq!(decode_header(&p).expect("round-trip"), h);
    }

    #[test]
    fn rejects_unknown_feature_flag() {
        let mut h = FileHeader::new_empty();
        h.feature_flags = 0b100;
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        let err = decode_header(&p).expect_err("unknown flag must fail");
        assert!(matches!(err, crate::error::Error::InvalidFormat { .. }));
    }

    /// Backward-compat reader contract. A
    /// pre-1.0 (`format_major = 0`) file with the baseline
    /// `format_minor = 0` MUST open under the v1.0 reader. We
    /// synthesize one by hand (no v1.0 writer can produce a
    /// `format_major = 0` file) and confirm `decode_header`
    /// accepts it.
    #[test]
    fn decodes_legacy_format_major_zero_minor_zero() {
        let h = FileHeader {
            format_major: 0,
            format_minor: 0,
            page_size: 4096,
            feature_flags: 0,
            page_count: 7,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0x11; 16],
            file_uuid: [0x22; 16],
            kdf_salt: [0; 32],
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        let decoded = decode_header(&p).expect("legacy 0.x file must decode");
        assert_eq!(decoded, h);
        assert_eq!(decoded.format_major, 0);
        assert_eq!(decoded.format_minor, 0);
    }

    /// A pre-1.0 compression-capable file
    /// (`format_major = 0, format_minor = 1`) MUST open under
    /// the v1.0 reader.
    #[test]
    fn decodes_legacy_format_major_zero_minor_one() {
        let h = FileHeader {
            format_major: 0,
            format_minor: 1,
            page_size: 4096,
            feature_flags: super::FEATURE_FLAG_COMPRESSION,
            page_count: 3,
            root_catalog: 0,
            freelist_head: 0,
            wal_salt: [0; 16],
            file_uuid: [0; 16],
            kdf_salt: [0; 32],
        };
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        let decoded = decode_header(&p).expect("0.x compression-capable must decode");
        assert_eq!(decoded, h);
    }

    /// `format_major = 2` (a hypothetical
    /// future-major file) MUST be rejected with `InvalidFormat`.
    /// The reader never silently misinterprets bytes from an
    /// unknown major. Synthesised by hand because the encoder
    /// `debug_assert` forbids constructing a `FileHeader` with
    /// `format_major = 2`.
    #[test]
    fn rejects_unsupported_format_major_two() {
        let h = FileHeader::new_empty();
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        p.as_bytes_mut()[super::OFF_FORMAT_MAJOR..super::OFF_FORMAT_MAJOR + 2]
            .copy_from_slice(&2u16.to_le_bytes());
        let crc = super::crc32c(&p.as_bytes()[..super::OFF_HEADER_CRC]);
        p.as_bytes_mut()[super::OFF_HEADER_CRC..super::OFF_HEADER_CRC + 4]
            .copy_from_slice(&crc.to_le_bytes());
        let err = decode_header(&p).expect_err("format_major = 2 must be rejected");
        assert!(
            matches!(err, crate::error::Error::InvalidFormat { reason }
                if reason.contains("format-major")),
            "expected InvalidFormat reason mentioning format-major; got {err:?}",
        );
    }

    /// A `format_major = 1` file with
    /// `format_minor = 0` or `1` must be rejected. The v1.0
    /// freeze locks the only valid minor for major 1 at 2.
    #[test]
    fn rejects_format_major_one_with_legacy_minor() {
        for bad_minor in [0u16, 1u16] {
            let h = FileHeader::new_empty();
            let mut p = Page::zeroed();
            encode_header(&h, &mut p);
            p.as_bytes_mut()[super::OFF_FORMAT_MINOR..super::OFF_FORMAT_MINOR + 2]
                .copy_from_slice(&bad_minor.to_le_bytes());
            let crc = super::crc32c(&p.as_bytes()[..super::OFF_HEADER_CRC]);
            p.as_bytes_mut()[super::OFF_HEADER_CRC..super::OFF_HEADER_CRC + 4]
                .copy_from_slice(&crc.to_le_bytes());
            let err = decode_header(&p).expect_err("format_major = 1 + legacy minor must fail");
            assert!(
                matches!(err, crate::error::Error::InvalidFormat { reason }
                    if reason.contains("format-minor")),
                "expected InvalidFormat reason mentioning format-minor; got {err:?}",
            );
        }
    }

    #[test]
    fn rejects_nonzero_reserved_after_feature_flags() {
        let h = FileHeader::new_empty();
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        p.as_bytes_mut()[14] = 0xFF;
        let crc = super::crc32c(&p.as_bytes()[..super::OFF_HEADER_CRC]);
        p.as_bytes_mut()[super::OFF_HEADER_CRC..super::OFF_HEADER_CRC + 4]
            .copy_from_slice(&crc.to_le_bytes());
        let err = decode_header(&p).expect_err("nonzero reserved must fail");
        assert!(matches!(err, crate::error::Error::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_bad_magic() {
        let h = FileHeader::new_empty();
        let mut p = Page::zeroed();
        encode_header(&h, &mut p);
        p.as_bytes_mut()[0] = b'X';
        let err = decode_header(&p).expect_err("bad magic must fail");
        assert!(matches!(err, crate::error::Error::InvalidFormat { .. }));
    }
}
