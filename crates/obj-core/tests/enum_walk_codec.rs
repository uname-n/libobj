//! Black-box coverage of the `DynamicSchema::Enum`
//! migration walk in [`obj_core::codec::Dynamic::from_postcard_bytes`].
//!
//! The in-crate unit tests (`codec::dynamic` `tests` module) already
//! round-trip unit / struct / newtype variants and assert the
//! unknown-discriminant error. This sibling exercises the same walk
//! **through the crate's public surface** and adds the case those
//! tests do not cover directly: an enum variant whose payload carries
//! an **integer** (a bare-`i64`/`u64` newtype variant and a struct
//! variant nesting an integer field). postcard encodes an enum as a
//! varint `u32` discriminant followed by the matched variant's
//! payload; the walk must read the discriminant, dispatch to the
//! matching `EnumVariantSchema` by `discriminant`, and recurse into
//! that variant's payload schema — including its integer slots.

#![forbid(unsafe_code)]
#![allow(clippy::missing_panics_doc)]

use obj_core::codec::{Dynamic, DynamicSchema, EnumVariantSchema};
use obj_core::Error;
use serde::{Deserialize, Serialize};

/// An enum whose variants carry integer payloads — the slot kind the
/// in-crate enum tests (`Null` / `String` / `Map<String>`) do not
/// exercise through the schema walk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Measurement {
    /// Unit variant — discriminant only, zero payload bytes.
    Missing,
    /// Newtype variant carrying a single unsigned integer.
    Count(u64),
    /// Struct variant nesting a signed integer alongside a label.
    Reading { label: String, value: i64 },
}

/// Hand-built schema for [`Measurement`]. Variants MUST be declared in
/// strictly-ascending discriminant order so the walker's binary search
/// holds. Integer slots use the live (un-normalized) signedness tags —
/// `U64` for `Count`, `I64` for `Reading.value` — exactly as the Rust
/// derive would emit them.
fn measurement_schema() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Missing", DynamicSchema::Null),
        EnumVariantSchema::new(1, "Count", DynamicSchema::U64),
        EnumVariantSchema::new(
            2,
            "Reading",
            DynamicSchema::map([
                ("label", DynamicSchema::String),
                ("value", DynamicSchema::I64),
            ]),
        ),
    ])
}

#[test]
fn walk_decodes_unit_variant_with_no_payload_bytes() {
    let bytes = postcard::to_allocvec(&Measurement::Missing).expect("encode");
    let view = Dynamic::from_postcard_bytes(&bytes, &measurement_schema()).expect("walk");
    assert_eq!(view.enum_variant(), Some("Missing"));
    assert_eq!(view.enum_payload(), Some(&Dynamic::Null));
}

#[test]
fn walk_decodes_newtype_variant_carrying_unsigned_integer() {
    let bytes = postcard::to_allocvec(&Measurement::Count(4_000_000_000)).expect("encode");
    let view = Dynamic::from_postcard_bytes(&bytes, &measurement_schema()).expect("walk");
    assert_eq!(view.enum_variant(), Some("Count"));
    assert_eq!(view.enum_payload(), Some(&Dynamic::U64(4_000_000_000)));
}

#[test]
fn walk_decodes_struct_variant_nesting_signed_integer() {
    let value = Measurement::Reading {
        label: "temp".to_owned(),
        value: -273,
    };
    let bytes = postcard::to_allocvec(&value).expect("encode");
    let view = Dynamic::from_postcard_bytes(&bytes, &measurement_schema()).expect("walk");
    assert_eq!(view.enum_variant(), Some("Reading"));
    let payload = view.enum_payload().expect("payload");
    assert_eq!(
        payload.get("label"),
        Some(&Dynamic::String("temp".to_owned()))
    );
    assert_eq!(payload.get("value"), Some(&Dynamic::I64(-273)));
}

#[test]
fn walk_errors_on_unknown_discriminant_without_panicking() {
    let forged = [7u8];
    let err =
        Dynamic::from_postcard_bytes(&forged, &measurement_schema()).expect_err("unknown variant");
    assert!(
        matches!(
            err,
            Error::SchemaTypeMismatch {
                expected: "known variant",
                ..
            }
        ),
        "expected SchemaTypeMismatch{{expected: \"known variant\"}}, got {err:?}",
    );
}
