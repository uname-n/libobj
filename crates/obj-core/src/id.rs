//! Per-collection document identifier — `Id(NonZeroU64)`.
//!
//! `Id` is the public identifier type every collection hands out for
//! its documents. The newtype encoding pins three invariants:
//!
//! 1. Zero is reserved as the sentinel "no id" value — `Id::try_new(0)`
//!    returns `None`. This makes `Option<Id>` the same size as `Id`
//!    and gives the freelist / catalog primitives an unambiguous
//!    "missing" marker.
//! 2. Allocation is per-collection monotonic. Two collections can
//!    independently hand out `Id(1), Id(2), ...` without interference;
//!    the catalog row for each collection carries its own `next_id`
//!    watermark.
//! 3. The serde representation of `Id` is its inner `NonZeroU64`, so
//!    an `Id` can appear in user `Document` types verbatim and round-
//!    trip through postcard.

#![forbid(unsafe_code)]

use core::num::NonZeroU64;

use postcard::experimental::max_size::MaxSize;
use serde::{Deserialize, Serialize};

use crate::codec::schema::{DynamicSchema, Schema};
use crate::error::{Error, Result};

/// Per-collection document identifier.
///
/// Wraps a [`NonZeroU64`] so the on-disk value `0` is unambiguously
/// "no id". Allocated by the catalog via
/// `Catalog::next_id` — see that method for the full transactional
/// contract.
///
/// `Id` implements `serde::Serialize + Deserialize + MaxSize` so it
/// can appear in user `Document` types directly, including inside a
/// `Vec<Id>` or a nested struct. The serde encoding is the inner
/// `NonZeroU64` — postcard varint-encodes it, and the deserializer
/// rejects the on-the-wire value `0` because `NonZeroU64`'s `serde`
/// impl already does the validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Id(NonZeroU64);

impl Id {
    /// Construct an [`Id`] from a raw `u64`. Returns `None` if `raw`
    /// is `0` (per the sentinel-zero contract).
    #[must_use]
    pub const fn try_new(raw: u64) -> Option<Self> {
        match NonZeroU64::new(raw) {
            Some(nz) => Some(Self(nz)),
            None => None,
        }
    }

    /// Total-function constructor: builds an [`Id`] from a
    /// [`NonZeroU64`] that the caller already proved is non-zero.
    /// Use [`try_new`](Self::try_new) at runtime boundaries.
    #[must_use]
    pub const fn from_nonzero(nz: NonZeroU64) -> Self {
        Self(nz)
    }

    /// The underlying `u64`. Always non-zero.
    #[inline]
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// The underlying [`NonZeroU64`].
    #[must_use]
    pub const fn as_nonzero(self) -> NonZeroU64 {
        self.0
    }

    /// Big-endian byte encoding used as a key in a collection's
    /// primary B-tree. The big-endian shape makes lexicographic
    /// byte comparison agree with numeric `<` on the underlying
    /// `u64`.
    #[inline]
    #[must_use]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.0.get().to_be_bytes()
    }

    /// Decode a big-endian `Id` from `bytes`. Returns `None` if the
    /// byte slice is the wrong length or names the sentinel zero.
    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        let arr: [u8; 8] = bytes.try_into().ok()?;
        Self::try_new(u64::from_be_bytes(arr))
    }
}

impl MaxSize for Id {
    const POSTCARD_MAX_SIZE: usize = 10;
}

impl Schema for Id {
    /// `Id` is `#[serde(transparent)]` over `NonZeroU64`, so its
    /// postcard wire shape is exactly an unsigned varint — the same
    /// bytes as a plain `u64`. The schema therefore reports
    /// [`DynamicSchema::U64`] so an `Id` field nests transparently in
    /// a `#[derive(Document)]` type's schema (the derive maps `u64`
    /// to the same variant).
    fn schema() -> DynamicSchema {
        DynamicSchema::U64
    }
}

/// Allocator shim used by the catalog to mint the next `Id` in a
/// collection.
///
/// The full allocator lives in `Catalog::next_id`.
/// This module provides only the **arithmetic step** — incrementing
/// a `u64` watermark and rejecting wraparound — so the catalog can
/// be tested in isolation.
///
/// The `collection` argument is a closure that builds the owned
/// collection name on demand, so the happy path never allocates a
/// `String`. The closure is only invoked on the wraparound /
/// zero-watermark error path.
///
/// # Errors
///
/// Returns [`Error::IdSpaceExhausted`] when the increment would
/// overflow `u64::MAX`. At 10⁹ inserts/sec this is ~584 years; the
/// check is defensive and cheap.
pub fn bump_next_id<F>(next_id: &mut u64, collection: F) -> Result<Id>
where
    F: FnOnce() -> String,
{
    if *next_id == 0 {
        return Err(Error::IdSpaceExhausted {
            collection: collection(),
        });
    }
    let issued = *next_id;
    *next_id = next_id.checked_add(1).unwrap_or(0);
    Id::try_new(issued).ok_or_else(|| Error::IdSpaceExhausted {
        collection: collection(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_rejects_zero() {
        assert!(Id::try_new(0).is_none());
        assert_eq!(Id::try_new(1).map(Id::get), Some(1));
        assert_eq!(Id::try_new(u64::MAX).map(Id::get), Some(u64::MAX));
    }

    #[test]
    fn be_bytes_round_trip() {
        let id = Id::try_new(0x0102_0304_0506_0708).expect("non-zero");
        assert_eq!(
            id.to_be_bytes(),
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(Id::from_be_bytes(&id.to_be_bytes()), Some(id));
        assert_eq!(Id::from_be_bytes(&[0u8; 8]), None);
        assert_eq!(Id::from_be_bytes(&[0u8; 7]), None);
    }

    #[test]
    fn serde_round_trip_via_postcard() {
        let id = Id::try_new(42).expect("non-zero");
        let bytes = postcard::to_allocvec(&id).expect("encode");
        let back: Id = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, id);
    }

    #[test]
    fn serde_rejects_zero_on_decode() {
        let bytes = [0u8];
        let result = postcard::from_bytes::<Id>(&bytes);
        assert!(result.is_err(), "zero must be rejected; got {result:?}");
    }

    #[test]
    fn schema_is_u64() {
        assert_eq!(<Id as Schema>::schema(), DynamicSchema::U64);
    }

    #[test]
    fn postcard_max_size_constant() {
        assert_eq!(Id::POSTCARD_MAX_SIZE, 10);
    }

    #[test]
    fn bump_allocator_advances() {
        let mut next = 1u64;
        let id1 = bump_next_id(&mut next, || "test".to_owned()).expect("bump 1");
        assert_eq!(id1.get(), 1);
        assert_eq!(next, 2);
        let id2 = bump_next_id(&mut next, || "test".to_owned()).expect("bump 2");
        assert_eq!(id2.get(), 2);
        assert_eq!(next, 3);
    }

    #[test]
    fn bump_allocator_detects_wraparound() {
        let mut next = u64::MAX;
        let id = bump_next_id(&mut next, || "wrap".to_owned()).expect("last id");
        assert_eq!(id.get(), u64::MAX);
        let err = bump_next_id(&mut next, || "wrap".to_owned()).expect_err("wraparound");
        match err {
            Error::IdSpaceExhausted { collection } => assert_eq!(collection, "wrap"),
            other => panic!("expected IdSpaceExhausted, got {other:?}"),
        }
    }

    #[test]
    fn bump_allocator_detects_zero_watermark() {
        let mut next = 0u64;
        let err = bump_next_id(&mut next, || "zerowm".to_owned()).expect_err("zero watermark");
        match err {
            Error::IdSpaceExhausted { collection } => assert_eq!(collection, "zerowm"),
            other => panic!("expected IdSpaceExhausted, got {other:?}"),
        }
    }

    #[test]
    fn bump_allocator_error_field_preserves_user_supplied_name() {
        let mut next = 0u64;
        let user_input = String::from("dynamically built name");
        let err = bump_next_id(&mut next, || user_input.clone()).expect_err("error");
        match err {
            Error::IdSpaceExhausted { collection } => {
                assert_eq!(collection, "dynamically built name");
            }
            other => panic!("expected IdSpaceExhausted, got {other:?}"),
        }
    }
}
