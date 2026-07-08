//! Order-preserving byte encoding for index keys.
//!
//! This module is the reference implementation.
//!
//! # Encoding contract
//!
//! For two `Dynamic` values `a` and `b`,
//! `encode_field(a) < encode_field(b)` (lexicographic byte
//! comparison) **iff** `a < b` (semantic comparison) within their
//! shared type. Cross-type comparisons follow the tag ordering:
//! `NULL < false < true < numeric < string < bytes`.
//!
//! The format is **distinct from** the tagged-Dynamic wire format in
//! [`crate::codec::dynamic`]: that format is round-trip-correct but
//! NOT order-preserving (negative integers, in particular, sort
//! wrong under varint encoding). The two encodings serve different
//! purposes — migration uses tagged-Dynamic for reflective access,
//! indexes use this module for lexicographic ordering.
//!
//! # Non-unique key disambiguation
//!
//! `Standard`, `Each`, and `Composite` indexes are non-unique: two
//! documents may share the same encoded user key. The B+tree
//! rejects duplicate keys (`Error::BTreeKeyExists`); the index
//! maintenance path side-steps this by **appending the document's
//! `Id` (8 bytes big-endian)** to the encoded user key before
//! writing into the B+tree. `Unique` indexes do **not** append
//! the suffix — a collision is the whole point.
//!
//! `encode_index_key` returns the user key portion only; the
//! caller (the maintenance path) is responsible for
//! appending the `Id` suffix for non-unique kinds. The
//! [`encoded_id_suffix_len`] constant is exported so range-scan
//! readers can trim the suffix back off.

#![forbid(unsafe_code)]

use crate::codec::Dynamic;
use crate::error::{Error, Result};
use crate::index::spec::{IndexKind, IndexSpec};

/// Length in bytes of the `Id` big-endian suffix that the
/// maintenance path appends to non-unique encoded keys. The suffix
/// is the document's `Id` as 8 bytes big-endian.
pub const ENCODED_ID_SUFFIX_LEN: usize = 8;

/// Tag byte distinguishing a composite key envelope from a single
/// primitive key.
pub const COMPOSITE_TAG: u8 = 0x80;

/// `Null`.
const TAG_NULL: u8 = 0x00;
/// `Bool(false)`.
const TAG_BOOL_FALSE: u8 = 0x01;
/// `Bool(true)`.
const TAG_BOOL_TRUE: u8 = 0x02;
/// `I64(negative)`.
const TAG_I64_NEG: u8 = 0x10;
/// `I64(zero)`.
const TAG_I64_ZERO: u8 = 0x11;
/// `I64(positive)`.
const TAG_I64_POS: u8 = 0x12;
/// `U64`.
const TAG_U64: u8 = 0x20;
/// `F64`.
const TAG_F64: u8 = 0x30;
/// `String`.
const TAG_STRING: u8 = 0x40;
/// `Bytes`.
const TAG_BYTES: u8 = 0x41;

/// Terminator byte for the order-preserving `String` encoding. UTF-8
/// strings cannot contain an interior `0x00`, so this byte is
/// unambiguous; an `extract_index_keys` impl that observes one
/// returns `Err(Error::Codec)` rather than silently splitting the
/// key.
const STRING_TERMINATOR: u8 = 0x00;

/// The byte representation of a single field encoded under the
/// order-preserving format.
///
/// Strongly-typed wrapper around `Vec<u8>` so the caller cannot
/// confuse an encoded key with a raw user value. The inner bytes
/// can be borrowed via `as_bytes` and consumed via `into_bytes`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EncodedIndexKey(Vec<u8>);

impl EncodedIndexKey {
    /// View as a byte slice.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Consume the wrapper and return the inner `Vec<u8>`.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Construct from raw bytes. Used by the maintenance path when
    /// it composes an encoded user key with the trailing `Id`
    /// suffix.
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Length of the encoded bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` if the encoded bytes are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<EncodedIndexKey> for Vec<u8> {
    fn from(k: EncodedIndexKey) -> Self {
        k.0
    }
}

impl AsRef<[u8]> for EncodedIndexKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Length of the trailing `Id` suffix appended to non-unique keys
/// by the maintenance path. See module docs.
#[must_use]
pub const fn encoded_id_suffix_len() -> usize {
    ENCODED_ID_SUFFIX_LEN
}

/// Encode `fields` into a single index key under the kind-specific
/// rules.
///
/// - `Standard`, `Unique`, `Each`: `fields.len()` must be 1; the
///   output is the order-preserving encoding of that field, with
///   no envelope tag. The caller appends the `Id` suffix for
///   non-unique kinds.
/// - `Composite`: `fields.len()` must equal `spec.key_paths.len()`
///   (≥ 2 by `IndexSpec::validate`); the output is the
///   composite-envelope tag (`0x80`) followed by the concatenated
///   per-field encodings.
///
/// # Errors
///
/// - [`Error::InvalidArgument`] if `fields.len()` disagrees with the
///   kind's per-spec contract.
/// - [`Error::InvalidArgument`] if a `String` field contains an
///   embedded `0x00` byte.
pub fn encode_index_key(spec: &IndexSpec, fields: &[Dynamic]) -> Result<EncodedIndexKey> {
    encode_index_key_parts(spec.kind, &spec.key_paths, fields)
}

/// Reference-based index-key encode. Takes the index `kind`
/// (a `Copy` discriminator) and `key_paths` BY REFERENCE so a caller
/// holding an [`crate::catalog::IndexDescriptor`] can encode a lookup
/// key WITHOUT cloning `name` / `key_paths` into a transient
/// [`IndexSpec`] (and without re-running [`IndexSpec::validate`]).
///
/// The produced [`EncodedIndexKey`] is **byte-identical** to
/// [`encode_index_key`] for the same `(kind, key_paths, fields)` — it
/// shares the same `encode_scalar` / `encode_composite` bodies and
/// keeps `encode_index_key`'s field-count caller-contract check.
/// Only the redundant `IndexSpec::validate` (re-checking
/// non-empty name/path + path-count-vs-kind on an already-validated,
/// on-disk descriptor) is dropped — the field-count check below is
/// the load-bearing one for the key bytes.
///
/// # Errors
///
/// - [`Error::InvalidArgument`] if `fields.len()` disagrees with
///   `key_paths.len()` (the kind's per-spec contract).
/// - [`Error::InvalidArgument`] if a `String` field contains an
///   embedded `0x00` byte.
pub fn encode_index_key_parts(
    kind: IndexKind,
    key_paths: &[String],
    fields: &[Dynamic],
) -> Result<EncodedIndexKey> {
    validate_key_part_shape(kind, key_paths, fields)?;
    match kind {
        IndexKind::Standard | IndexKind::Unique | IndexKind::Each => encode_scalar(&fields[0]),
        IndexKind::Composite => encode_composite(fields),
    }
}

fn validate_key_part_shape(
    kind: IndexKind,
    key_paths: &[String],
    fields: &[Dynamic],
) -> Result<()> {
    if fields.len() != key_paths.len() {
        return Err(Error::InvalidArgument(
            "encode_index_key: field count disagrees with spec",
        ));
    }
    match kind {
        IndexKind::Standard | IndexKind::Unique | IndexKind::Each if fields.len() != 1 => Err(
            Error::InvalidArgument("encode_index_key: scalar indexes require exactly one field"),
        ),
        IndexKind::Composite if fields.len() < 2 => Err(Error::InvalidArgument(
            "encode_index_key: composite indexes require at least two fields",
        )),
        IndexKind::Standard | IndexKind::Unique | IndexKind::Each | IndexKind::Composite => Ok(()),
    }
}

/// Encode a single `Dynamic` value to its order-preserving byte
/// representation. Used both by the public `encode_index_key` and
/// by the composite encoder.
///
/// # Errors
///
/// - [`Error::InvalidArgument`] if `value` is a `String` containing an
///   embedded `0x00` byte.
pub fn encode_field(value: &Dynamic) -> Result<EncodedIndexKey> {
    encode_scalar(value)
}

/// Encode a single non-composite field. Bounded-width or
/// self-delimiting per the per-tag rules.
fn encode_scalar(value: &Dynamic) -> Result<EncodedIndexKey> {
    let mut out = Vec::with_capacity(9);
    write_one_into(value, &mut out)?;
    Ok(EncodedIndexKey(out))
}

/// Encode an N-field composite key.
fn encode_composite(fields: &[Dynamic]) -> Result<EncodedIndexKey> {
    debug_assert!(fields.len() >= 2, "composite requires ≥ 2 fields");
    let mut out = Vec::with_capacity(fields.len() * 9 + 1);
    out.push(COMPOSITE_TAG);
    for f in fields {
        write_one_into(f, &mut out)?;
    }
    Ok(EncodedIndexKey(out))
}

/// Append `value` in its order-preserving form to `out`.
fn write_one_into(value: &Dynamic, out: &mut Vec<u8>) -> Result<()> {
    match value {
        Dynamic::Null => out.push(TAG_NULL),
        Dynamic::Bool(false) => out.push(TAG_BOOL_FALSE),
        Dynamic::Bool(true) => out.push(TAG_BOOL_TRUE),
        Dynamic::I64(n) => write_i64(*n, out),
        Dynamic::U64(n) => write_u64(*n, out),
        Dynamic::F64(f) => write_f64(*f, out),
        Dynamic::String(s) => write_string(s, out)?,
        Dynamic::Bytes(b) => write_bytes(b, out)?,
        Dynamic::Seq(_) | Dynamic::Map(_) | Dynamic::Enum { .. } => {
            return Err(Error::InvalidArgument(
                "index key field must be a primitive Dynamic value (Null/Bool/I64/U64/F64/String/Bytes)",
            ));
        }
    }
    Ok(())
}

/// Order-preserving `i64` encoding: tag distinguishes the sign, body
/// is the 8-byte BE representation with the sign bit flipped.
///
/// Bit-flipping the sign bit on a two's-complement big-endian `i64`
/// produces bytes whose unsigned lexicographic order matches the
/// signed numeric order. Splitting by sign with three tags
/// (`neg < zero < pos`) is functionally equivalent and slightly
/// nicer to debug — the leading byte tells the human reader the
/// sign at a glance.
fn write_i64(n: i64, out: &mut Vec<u8>) {
    match n.cmp(&0) {
        std::cmp::Ordering::Less => out.push(TAG_I64_NEG),
        std::cmp::Ordering::Equal => {
            out.push(TAG_I64_ZERO);
            return;
        }
        std::cmp::Ordering::Greater => out.push(TAG_I64_POS),
    }
    let flipped = n.cast_unsigned() ^ 0x8000_0000_0000_0000;
    out.extend_from_slice(&flipped.to_be_bytes());
}

/// Order-preserving `u64` encoding: tag plus 8-byte BE.
fn write_u64(n: u64, out: &mut Vec<u8>) {
    out.push(TAG_U64);
    out.extend_from_slice(&n.to_be_bytes());
}

/// Order-preserving `f64` encoding: tag plus 8-byte BE with the
/// IEEE-754 total-order transform applied. See module docs.
///
/// Transform: if the sign bit is 0, flip the sign bit only;
/// otherwise, flip every bit. The result has the property that
/// `total_order_bytes(a) < total_order_bytes(b)` (unsigned BE
/// lexicographic) iff `a < b` under IEEE-754 total order
/// (`-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN`). NaN
/// values are ordered by their bit pattern at the extremes; obj
/// makes no promise about NaN semantics beyond
/// "the ordering is total and deterministic".
fn write_f64(f: f64, out: &mut Vec<u8>) {
    out.push(TAG_F64);
    let bits = f.to_bits();
    let transformed = if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        bits ^ 0xFFFF_FFFF_FFFF_FFFF
    };
    out.extend_from_slice(&transformed.to_be_bytes());
}

/// Order-preserving `String` encoding: tag plus UTF-8 bytes plus a
/// trailing `0x00` terminator. Rejects strings containing an
/// embedded `0x00` so the terminator stays unambiguous.
fn write_string(s: &str, out: &mut Vec<u8>) -> Result<()> {
    if s.as_bytes().contains(&STRING_TERMINATOR) {
        return Err(Error::InvalidArgument(
            "index key: String value contains embedded NUL (0x00)",
        ));
    }
    out.push(TAG_STRING);
    out.extend_from_slice(s.as_bytes());
    out.push(STRING_TERMINATOR);
    Ok(())
}

/// Order-preserving raw-bytes encoding: tag plus 4-byte BE length
/// plus the bytes themselves. The 4-byte length prefix bounds
/// individual byte fields to 4 GiB; `Error::BTreeKeyTooLarge` will
/// fire well before that.
fn write_bytes(b: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let len_u32 = u32::try_from(b.len())
        .map_err(|_| Error::InvalidArgument("index key: Bytes field exceeds 4 GiB length limit"))?;
    out.push(TAG_BYTES);
    out.extend_from_slice(&len_u32.to_be_bytes());
    out.extend_from_slice(b);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::spec::IndexSpec;
    use std::cmp::Ordering;

    /// Helper: encode a single Dynamic to its byte form. Panics on
    /// error — only used for ordering tests against well-formed
    /// inputs.
    fn enc(d: &Dynamic) -> Vec<u8> {
        encode_scalar(d).expect("encode").into_bytes()
    }

    /// Assert `encode(left) < encode(right)`.
    fn assert_lt(left: &Dynamic, right: &Dynamic) {
        let a = enc(left);
        let b = enc(right);
        assert!(
            a < b,
            "encode({left:?}) NOT < encode({right:?}): a={a:?} b={b:?}",
        );
    }

    #[test]
    fn null_sorts_before_everything() {
        assert_lt(&Dynamic::Null, &Dynamic::Bool(false));
        assert_lt(&Dynamic::Null, &Dynamic::I64(i64::MIN));
        assert_lt(&Dynamic::Null, &Dynamic::U64(0));
    }

    #[test]
    fn bool_false_before_true() {
        assert_lt(&Dynamic::Bool(false), &Dynamic::Bool(true));
        assert_lt(&Dynamic::Bool(true), &Dynamic::I64(i64::MIN));
    }

    #[test]
    fn signed_negative_before_zero_before_positive() {
        let neg_big = Dynamic::I64(i64::MIN);
        let neg_small = Dynamic::I64(-1);
        let zero = Dynamic::I64(0);
        let pos_small = Dynamic::I64(1);
        let pos_big = Dynamic::I64(i64::MAX);
        assert_lt(&neg_big, &neg_small);
        assert_lt(&neg_small, &zero);
        assert_lt(&zero, &pos_small);
        assert_lt(&pos_small, &pos_big);
        assert_lt(&neg_big, &Dynamic::I64(i64::MAX));
    }

    #[test]
    fn i64_ordering_full_sweep() {
        let samples = [i64::MIN, -1_000_000_000, -1, 0, 1, 1_000_000_000, i64::MAX];
        for window in samples.windows(2) {
            assert_lt(&Dynamic::I64(window[0]), &Dynamic::I64(window[1]));
        }
    }

    #[test]
    fn u64_ordering_monotone() {
        let samples = [0u64, 1, u64::from(u32::MAX), u64::MAX];
        for window in samples.windows(2) {
            assert_lt(&Dynamic::U64(window[0]), &Dynamic::U64(window[1]));
        }
    }

    #[test]
    fn i64_sorts_before_u64_by_tag() {
        assert_lt(&Dynamic::I64(i64::MAX), &Dynamic::U64(0));
    }

    #[test]
    fn f64_total_order() {
        let samples = [
            f64::NEG_INFINITY,
            -1.5,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.5,
            f64::INFINITY,
        ];
        for window in samples.windows(2) {
            let a = enc(&Dynamic::F64(window[0]));
            let b = enc(&Dynamic::F64(window[1]));
            assert_ne!(a.cmp(&b), Ordering::Greater, "f64 order: {window:?}");
        }
    }

    #[test]
    fn string_lexicographic_order() {
        let samples = ["", "a", "ab", "abc", "b", "ba", "bb", "z"];
        for window in samples.windows(2) {
            assert_lt(
                &Dynamic::String(window[0].to_owned()),
                &Dynamic::String(window[1].to_owned()),
            );
        }
    }

    #[test]
    fn string_with_embedded_nul_rejected() {
        let err = encode_scalar(&Dynamic::String("a\0b".to_owned())).expect_err("nul");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn string_terminator_disambiguates_prefix() {
        let a = enc(&Dynamic::String("a".to_owned()));
        let b = enc(&Dynamic::String("ab".to_owned()));
        assert!(a < b);
        let c = enc(&Dynamic::String("az".to_owned()));
        assert!(a < c);
    }

    #[test]
    fn bytes_with_embedded_nul_round_trip_ok() {
        let a = enc(&Dynamic::Bytes(vec![0x00, 0x01]));
        let b = enc(&Dynamic::Bytes(vec![0x00, 0x02]));
        assert!(a < b);
    }

    #[test]
    fn string_sorts_before_bytes() {
        assert_lt(
            &Dynamic::String("zzz".to_owned()),
            &Dynamic::Bytes(vec![0x00]),
        );
    }

    #[test]
    fn composite_envelope_starts_with_tag() {
        let spec = IndexSpec::composite("by_ct", &["c", "t"]).expect("spec");
        let key = encode_index_key(&spec, &[Dynamic::U64(7), Dynamic::String("a".to_owned())])
            .expect("encode");
        assert_eq!(key.as_bytes()[0], COMPOSITE_TAG);
    }

    #[test]
    fn composite_orders_first_field_then_second() {
        let spec = IndexSpec::composite("by_ct", &["c", "t"]).expect("spec");
        let pairs = [
            (1u64, "a"),
            (1u64, "b"),
            (2u64, "a"),
            (2u64, "b"),
            (3u64, "a"),
        ];
        let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(pairs.len());
        for (c, t) in pairs {
            let k = encode_index_key(&spec, &[Dynamic::U64(c), Dynamic::String((*t).to_owned())])
                .expect("encode")
                .into_bytes();
            encoded.push(k);
        }
        let mut sorted = encoded.clone();
        sorted.sort();
        assert_eq!(encoded, sorted);
    }

    #[test]
    fn field_count_mismatch_rejected() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = encode_index_key(&spec, &[Dynamic::U64(1), Dynamic::U64(2)])
            .expect_err("count mismatch");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn map_field_rejected_as_unindexable() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let m = Dynamic::Map(std::collections::BTreeMap::new());
        let err = encode_index_key(&spec, &[m]).expect_err("map unindexable");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn seq_field_rejected_for_non_each_kinds() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let s = Dynamic::Seq(vec![Dynamic::U64(1)]);
        let err = encode_index_key(&spec, &[s]).expect_err("seq must be unfolded");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    /// The ref-based [`encode_index_key_parts`] entry point MUST
    /// produce bytes byte-identical to the `IndexSpec` + `from_parts`
    /// path (`encode_index_key`) for every index kind. The on-disk
    /// index-key format MUST NOT change when a lookup goes through the
    /// new clone-free path.
    #[test]
    fn parts_encode_byte_identical_to_spec_path() {
        let cases: Vec<(IndexKind, Vec<String>, Vec<Dynamic>)> = vec![
            (
                IndexKind::Standard,
                vec!["x".to_owned()],
                vec![Dynamic::I64(-42)],
            ),
            (
                IndexKind::Standard,
                vec!["x".to_owned()],
                vec![Dynamic::Null],
            ),
            (
                IndexKind::Unique,
                vec!["email".to_owned()],
                vec![Dynamic::String("a@b.c".to_owned())],
            ),
            (
                IndexKind::Unique,
                vec!["flag".to_owned()],
                vec![Dynamic::Bool(true)],
            ),
            (
                IndexKind::Each,
                vec!["tags".to_owned()],
                vec![Dynamic::Bytes(vec![0x01, 0xFF, 0x00, 0x7F])],
            ),
            (
                IndexKind::Each,
                vec!["scores".to_owned()],
                vec![Dynamic::F64(-0.0)],
            ),
            (
                IndexKind::Composite,
                vec!["c".to_owned(), "t".to_owned()],
                vec![Dynamic::U64(7), Dynamic::String("z".to_owned())],
            ),
            (
                IndexKind::Composite,
                vec!["a".to_owned(), "b".to_owned(), "d".to_owned()],
                vec![Dynamic::I64(i64::MIN), Dynamic::U64(0), Dynamic::F64(1.5)],
            ),
        ];
        for (kind, key_paths, fields) in cases {
            let spec = IndexSpec::from_parts("idx", kind, key_paths.clone()).expect("valid spec");
            let via_spec = encode_index_key(&spec, &fields).expect("spec-path encode");
            let via_parts =
                encode_index_key_parts(kind, &key_paths, &fields).expect("parts-path encode");
            assert_eq!(
                via_spec, via_parts,
                "byte mismatch for kind={kind:?} paths={key_paths:?} fields={fields:?}"
            );
        }
    }

    /// The ref-based path keeps `encode_index_key`'s field-count
    /// check — a count mismatch must still error, not panic
    /// or silently encode the wrong key.
    #[test]
    fn parts_encode_keeps_field_count_check() {
        let err = encode_index_key_parts(
            IndexKind::Standard,
            &["x".to_owned()],
            &[Dynamic::U64(1), Dynamic::U64(2)],
        )
        .expect_err("count mismatch");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn parts_encode_rejects_empty_scalar_shape() {
        let err = encode_index_key_parts(IndexKind::Standard, &[], &[])
            .expect_err("empty scalar shape must error");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn parts_encode_rejects_under_width_composite_shape() {
        let err =
            encode_index_key_parts(IndexKind::Composite, &["x".to_owned()], &[Dynamic::U64(1)])
                .expect_err("under-width composite must error");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
