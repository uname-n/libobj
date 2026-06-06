//! Freelist representation.
//!
//! Format version 0 uses an in-page linked list: the file header's
//! `freelist_head` points at the most-recently-freed page, and each
//! freelist page stores the `PageId` of the next one (or `0` for the
//! end of the list).
//!
//! The freelist holds at most one id per page (the head of the
//! list).
//!
//! This module exposes pure encode / decode helpers. The pager
//! (`super::Pager`) is responsible for sequencing the writes; keeping
//! the codec separate makes it easy to unit-test in isolation.

#![forbid(unsafe_code)]

use crate::pager::page::{Page, PAGE_SIZE, PAGE_TRAILER_SIZE};

/// On-disk type tag for a freelist page.
pub const TYPE_FREE_LIST: u8 = 0x05;

/// Offset of the `next` link inside a freelist page (in bytes).
const OFF_NEXT: usize = 8;

const _: () = assert!(PAGE_SIZE >= PAGE_TRAILER_SIZE + OFF_NEXT + 8);

/// In-memory view of a freelist page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeListPage {
    /// Next page on the freelist, or `0` if this is the last entry.
    pub next: u64,
}

impl FreeListPage {
    /// Construct a freelist page that points at `next`. Pass `0` to
    /// mark the tail of the list.
    #[must_use]
    pub const fn new(next: u64) -> Self {
        Self { next }
    }
}

/// Encode `entry` into `page`. The page-trailer region (last
/// [`PAGE_TRAILER_SIZE`] bytes) is left zero; the pager writes the
/// trailer.
pub fn encode(entry: FreeListPage, page: &mut Page) {
    let buf = page.as_bytes_mut();
    buf.fill(0);
    buf[0] = TYPE_FREE_LIST;
    buf[OFF_NEXT..OFF_NEXT + 8].copy_from_slice(&entry.next.to_le_bytes());
}

/// Decode a freelist page. Caller is responsible for checking the
/// page trailer.
///
/// Returns `None` if the type tag is wrong; in that case the caller
/// should surface `Error::Corruption`.
#[must_use]
pub fn decode(page: &Page) -> Option<FreeListPage> {
    let buf = page.as_bytes();
    if buf[0] != TYPE_FREE_LIST {
        return None;
    }
    let mut next_bytes = [0u8; 8];
    next_bytes.copy_from_slice(&buf[OFF_NEXT..OFF_NEXT + 8]);
    Some(FreeListPage {
        next: u64::from_le_bytes(next_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, FreeListPage};
    use crate::pager::page::Page;

    #[test]
    fn round_trip() {
        for &next in &[0u64, 1, 42, u64::MAX] {
            let mut p = Page::zeroed();
            encode(FreeListPage::new(next), &mut p);
            assert_eq!(decode(&p), Some(FreeListPage { next }));
        }
    }

    #[test]
    fn decode_rejects_bad_tag() {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[0] = 0x03;
        assert!(decode(&p).is_none());
    }
}
