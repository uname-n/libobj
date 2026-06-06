//! Bound-widening for non-Unique index range scans.
//!
//! Non-Unique index B-tree keys are encoded as
//! `encode(user_value) || id_be8` (the uniqueness trick). A
//! user-facing range like `Included(x)..=Included(x)` therefore
//! matches zero B-tree entries unless we widen the encoded upper
//! bound to cover every `id_be8` suffix the maintenance path may
//! have appended.
//!
//! This module turns user-facing `(Bound<Vec<u8>>, Bound<Vec<u8>>)`
//! pairs (where each `Vec<u8>` is the order-preserving encoding of
//! the user's `Dynamic` value) into the internal B-tree bounds that
//! actually match the user's intent. The widening is kind-aware:
//! `Unique` keys carry no `id_be8` suffix so they pass through
//! unchanged; the other three kinds (`Standard` / `Each` /
//! `Composite`) get the eight-byte widening applied.

use std::ops::Bound;

use obj_core::IndexKind;

/// Eight `0xFF` bytes — the maximum possible value of the `id_be8`
/// suffix the maintenance path appends to a non-unique index key.
/// Used to widen `Included` upper bounds and `Excluded` lower bounds
/// so a user-facing `Included(x)..=Included(x)` range matches every
/// entry with user-key `x` regardless of the trailing suffix.
pub(crate) const SUFFIX_HIGH: [u8; 8] = [0xFF; 8];

/// Translate user-facing encoded bounds into the internal B-tree
/// bounds that match the user's intent under `kind`.
///
/// The widening table:
///
/// | User bound       | Standard / Each / Composite          | Unique           |
/// |------------------|--------------------------------------|------------------|
/// | `Included(x)` LO | `Included(encode(x))`                | `Included(...)`  |
/// | `Excluded(x)` LO | `Excluded(encode(x) ++ [0xFF; 8])`   | `Excluded(...)`  |
/// | `Included(x)` HI | `Included(encode(x) ++ [0xFF; 8])`   | `Included(...)`  |
/// | `Excluded(x)` HI | `Excluded(encode(x))`                | `Excluded(...)`  |
/// | `Unbounded`      | `Unbounded`                          | `Unbounded`      |
///
/// Rationale: a non-Unique B-tree key is
/// `encode(x) || id_be8 ∈ [encode(x) || 0x00;8, encode(x) || 0xFF;8]`.
///
/// - `Included(x)` LO: `encode(x) ≤ encode(x) || 0x00;8`, so the raw
///   `Included(encode(x))` already covers every entry with user-key
///   `≥ x`.
/// - `Included(x)` HI: `encode(x) || 0xFF;8 ≥ encode(x) || id_be8`
///   for any `id`, so `Included(encode(x) || 0xFF;8)` covers every
///   entry with user-key `≤ x`.
/// - `Excluded(x)` LO: the largest entry with user-key `x` is
///   `encode(x) || 0xFF;8`; `Excluded(encode(x) || 0xFF;8)` skips
///   every such entry.
/// - `Excluded(x)` HI: the smallest entry with user-key `x` is
///   `encode(x) || 0x00;8 > encode(x)`; `Excluded(encode(x))` skips
///   every entry with user-key `x`.
///
/// `Unique` keys carry no `id_be8` suffix, so the user-facing and
/// internal bounds are identical.
pub(crate) fn widen_bounds_for_kind(
    start: Bound<Vec<u8>>,
    end: Bound<Vec<u8>>,
    kind: IndexKind,
) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
    if kind == IndexKind::Unique {
        return (start, end);
    }
    (widen_lower(start), widen_upper(end))
}

/// Widen the lower bound under the non-Unique table above. See
/// [`widen_bounds_for_kind`] for the full contract.
fn widen_lower(b: Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(v),
        Bound::Excluded(v) => Bound::Excluded(append_suffix_high(v)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// Widen the upper bound under the non-Unique table above. See
/// [`widen_bounds_for_kind`] for the full contract.
fn widen_upper(b: Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(append_suffix_high(v)),
        Bound::Excluded(v) => Bound::Excluded(v),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// Append [`SUFFIX_HIGH`] (eight `0xFF` bytes) to `v` so the result
/// is `≥ v || id_be8` for every possible `id_be8`.
fn append_suffix_high(mut v: Vec<u8>) -> Vec<u8> {
    v.extend_from_slice(&SUFFIX_HIGH);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_pass_through() {
        let (s, e) = widen_bounds_for_kind(
            Bound::Included(b"abc".to_vec()),
            Bound::Included(b"abc".to_vec()),
            IndexKind::Unique,
        );
        assert_eq!(s, Bound::Included(b"abc".to_vec()));
        assert_eq!(e, Bound::Included(b"abc".to_vec()));
    }

    #[test]
    fn standard_included_upper_widens() {
        let (_s, e) = widen_bounds_for_kind(
            Bound::Included(b"a".to_vec()),
            Bound::Included(b"a".to_vec()),
            IndexKind::Standard,
        );
        let mut expected = b"a".to_vec();
        expected.extend_from_slice(&[0xFF; 8]);
        assert_eq!(e, Bound::Included(expected));
    }

    #[test]
    fn standard_excluded_lower_widens() {
        let (s, _e) = widen_bounds_for_kind(
            Bound::Excluded(b"a".to_vec()),
            Bound::Unbounded,
            IndexKind::Standard,
        );
        let mut expected = b"a".to_vec();
        expected.extend_from_slice(&[0xFF; 8]);
        assert_eq!(s, Bound::Excluded(expected));
    }

    #[test]
    fn excluded_upper_and_included_lower_unchanged_for_non_unique() {
        let (s, e) = widen_bounds_for_kind(
            Bound::Included(b"lo".to_vec()),
            Bound::Excluded(b"hi".to_vec()),
            IndexKind::Each,
        );
        assert_eq!(s, Bound::Included(b"lo".to_vec()));
        assert_eq!(e, Bound::Excluded(b"hi".to_vec()));
    }

    #[test]
    fn unbounded_pass_through_for_all_kinds() {
        for kind in [
            IndexKind::Standard,
            IndexKind::Unique,
            IndexKind::Each,
            IndexKind::Composite,
        ] {
            let (s, e) = widen_bounds_for_kind(Bound::Unbounded, Bound::Unbounded, kind);
            assert_eq!(s, Bound::Unbounded);
            assert_eq!(e, Bound::Unbounded);
        }
    }
}
