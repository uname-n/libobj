//! GOLDEN normalized-schema bytes.
//!
//! The **normalized** `StoredSchema.schema` is signedness-canonical: a
//! `u64` field and an `i64` field for the same logical column normalize
//! to the same canonical integer tag and therefore produce
//! byte-identical stored bytes. This makes the on-disk schema
//! language-neutral — any binding that lowers the equivalent shape
//! lands on the same bytes.
//!
//! This test asserts that
//! `normalize_schema_to_postcard(<canonical shape>)` equals a frozen
//! GOLDEN hex constant; if normalization drifts, the golden fails
//! loudly.
//!
//! Canonical shape (declaration order is load-bearing — postcard is
//! positional):
//!
//! ```text
//! { customer_id: u64,  // -> normalized integer tag
//!   total:       f64,  // -> normalized float tag
//!   region:      str,
//!   tags:        list[str] }
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::missing_panics_doc)]

use std::fmt::Write as _;

use obj_core::codec::{normalize_schema_to_postcard, DynamicSchema};

/// The frozen normalized-schema contract. Decoding (variant ordinals
/// `Null=0,Bool=1,U64=2,I64=3,F64=4,String=5,Bytes=6,Seq=7,Map=8`):
///
/// - `08` Map, `04` four fields,
/// - `0b "customer_id"` `02` U64,
/// - `05 "total"`       `04` F64,
/// - `06 "region"`      `05` String,
/// - `04 "tags"`        `07` Seq `05` String.
const GOLDEN_NORMALIZED_SCHEMA_HEX: &str =
    "08040b637573746f6d65725f69640205746f74616c0406726567696f6e0504746167730705";

/// The canonical shape with unsigned integers (`u64`, `f64`, `String`,
/// `Vec<String>` — every integer is `U64`).
fn canonical_shape_unsigned() -> DynamicSchema {
    DynamicSchema::map([
        ("customer_id", DynamicSchema::U64),
        ("total", DynamicSchema::F64),
        ("region", DynamicSchema::String),
        ("tags", DynamicSchema::seq(DynamicSchema::String)),
    ])
}

/// The same logical shape with a signed `i64` integer instead of the
/// `u64` above. Normalization must collapse `I64` and `U64` to the SAME
/// tag, so this and the unsigned shape MUST produce identical golden
/// bytes.
fn canonical_shape_signed() -> DynamicSchema {
    DynamicSchema::map([
        ("customer_id", DynamicSchema::I64),
        ("total", DynamicSchema::F64),
        ("region", DynamicSchema::String),
        ("tags", DynamicSchema::seq(DynamicSchema::String)),
    ])
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(out, "{b:02x}").expect("write to String is infallible");
    }
    out
}

#[test]
fn normalized_bytes_match_golden() {
    let bytes = normalize_schema_to_postcard(&canonical_shape_unsigned()).expect("encode");
    assert_eq!(
        to_hex(&bytes),
        GOLDEN_NORMALIZED_SCHEMA_HEX,
        "normalized schema bytes drifted from the golden",
    );
}

#[test]
fn signed_int_normalizes_to_same_golden() {
    let unsigned =
        normalize_schema_to_postcard(&canonical_shape_unsigned()).expect("encode unsigned");
    let signed = normalize_schema_to_postcard(&canonical_shape_signed()).expect("encode signed");
    assert_eq!(
        unsigned, signed,
        "u64 and i64 fields must normalize to identical bytes",
    );
    assert_eq!(to_hex(&signed), GOLDEN_NORMALIZED_SCHEMA_HEX);
}
