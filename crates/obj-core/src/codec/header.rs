//! Per-document record header — encode / decode.
//!
//! The header is 16 bytes laid out as four little-endian `u32` fields,
//! immediately followed by the `postcard` payload. The page-level
//! CRC32C trailer (on the B+tree leaf containing this record) covers
//! everything around the record; the header's own `payload_crc32c`
//! covers only the payload bytes so that a forensic tool can verify
//! a record in isolation.

#![forbid(unsafe_code)]

use crate::btree::{max_inline_value, max_key_len};
use crate::error::{Error, Result};

/// Size of the per-document header in bytes.
pub const DOC_HEADER_SIZE: usize = 16;

const OFF_COLLECTION_ID: usize = 0;
const OFF_TYPE_VERSION: usize = 4;
const OFF_PAYLOAD_LEN: usize = 8;
const OFF_PAYLOAD_CRC32C: usize = 12;

/// Maximum on-disk record length that still fits inline in a B+tree
/// leaf alongside at least one slot.
///
/// Equals the codec's slice of the leaf's `max_inline_value` budget
/// at the worst-case key length (`max_key_len()`). Records exceeding
/// this bound return [`Error::DocumentTooLarge`].
///
/// The value is computed at compile time from
/// [`crate::btree::max_inline_value`] so the codec and the B+tree
/// agree on the bound: any record the codec accepts will also fit
/// in a leaf — there is no second runtime check at insert time.
pub const MAX_INLINE_DOC: usize = max_inline_value(max_key_len());

/// In-memory representation of the per-document record header.
///
/// Constructed by [`DocumentHeader::read_from`] (on decode) or by
/// the codec on encode. All four fields are stored on disk as
/// little-endian `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentHeader {
    /// The catalog-assigned id of the collection this record belongs
    /// to. Decode rejects mismatches with
    /// [`Error::CollectionIdMismatch`].
    pub collection_id: u32,
    /// `Document::VERSION` of the type that wrote this record.
    pub type_version: u32,
    /// Number of payload bytes that follow this header.
    pub payload_len: u32,
    /// CRC32C of the payload bytes only — the page-trailer CRC32C
    /// on the containing leaf covers everything else.
    pub payload_crc32c: u32,
}

impl DocumentHeader {
    /// Write the header into `dst`. Appends exactly
    /// [`DOC_HEADER_SIZE`] bytes.
    ///
    /// Used by [`crate::codec::encode`]; the format is the
    /// canonical disk shape so the buffer can be handed straight to
    /// `BTree::insert` without further wrapping.
    pub fn write_to(&self, dst: &mut Vec<u8>) {
        debug_assert_eq!(
            OFF_PAYLOAD_CRC32C + 4,
            DOC_HEADER_SIZE,
            "header offsets must cover exactly DOC_HEADER_SIZE bytes"
        );
        dst.reserve(DOC_HEADER_SIZE);
        dst.extend_from_slice(&self.collection_id.to_le_bytes());
        dst.extend_from_slice(&self.type_version.to_le_bytes());
        dst.extend_from_slice(&self.payload_len.to_le_bytes());
        dst.extend_from_slice(&self.payload_crc32c.to_le_bytes());
    }

    /// Decode a header from `bytes`. Validates only the layout
    /// (length >= [`DOC_HEADER_SIZE`]); semantic checks (CRC,
    /// collection-id, version range) are the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Corruption`] with `page_id = 0` if `bytes`
    /// is shorter than [`DOC_HEADER_SIZE`].
    pub fn read_from(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < DOC_HEADER_SIZE {
            return Err(Error::Corruption { page_id: 0 });
        }
        let collection_id = u32::from_le_bytes(read_array(bytes, OFF_COLLECTION_ID));
        let type_version = u32::from_le_bytes(read_array(bytes, OFF_TYPE_VERSION));
        let payload_len = u32::from_le_bytes(read_array(bytes, OFF_PAYLOAD_LEN));
        let payload_crc32c = u32::from_le_bytes(read_array(bytes, OFF_PAYLOAD_CRC32C));
        Ok(Self {
            collection_id,
            type_version,
            payload_len,
            payload_crc32c,
        })
    }
}

/// Read a fixed-size field out of the input slice. Mirrors the
/// helper in `pager::header` so the codec does not need to import
/// pager internals. `off + N <= bytes.len()` is the caller's
/// invariant (every call-site reads a field whose offset is checked
/// against `DOC_HEADER_SIZE` upstream).
fn read_array<const N: usize>(bytes: &[u8], off: usize) -> [u8; N] {
    debug_assert!(off + N <= bytes.len(), "header field out of bounds");
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes[off..off + N]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = DocumentHeader {
            collection_id: 0x1122_3344,
            type_version: 5,
            payload_len: 99,
            payload_crc32c: 0xDEAD_BEEF,
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf);
        assert_eq!(buf.len(), DOC_HEADER_SIZE);
        let decoded = DocumentHeader::read_from(&buf).expect("decode");
        assert_eq!(decoded, h);
    }

    #[test]
    fn header_layout_little_endian() {
        let h = DocumentHeader {
            collection_id: 0x0403_0201,
            type_version: 0x0807_0605,
            payload_len: 0x0C0B_0A09,
            payload_crc32c: 0x100F_0E0D,
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf);
        assert_eq!(
            &buf[..],
            &[
                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
                0x0F, 0x10,
            ]
        );
    }

    #[test]
    fn header_short_input_errors() {
        let err = DocumentHeader::read_from(&[0u8; DOC_HEADER_SIZE - 1])
            .expect_err("short input rejected");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn max_inline_doc_is_positive() {
        const {
            assert!(
                MAX_INLINE_DOC > DOC_HEADER_SIZE,
                "MAX_INLINE_DOC must leave room for at least one payload byte",
            );
        }
    }
}
