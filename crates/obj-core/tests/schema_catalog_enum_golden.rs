//! GOLDEN normalized bytes for an enum-bearing schema shape.
//!
//! `schema_catalog_golden.rs` pins the golden for a `Map` shape. This
//! file is the ENUM analogue: it freezes the normalized
//! `StoredSchema.schema` bytes for a `DynamicSchema::Enum` whose variant
//! payloads carry an integer (the signedness the normalization pass
//! collapses) and a struct.
//!
//! Canonical enum shape (variant declaration order is FROZEN — postcard
//! assigns discriminants positionally and the catalog stores them by
//! discriminant):
//!
//! ```text
//! enum Status {
//!     Active,                            // disc 0, payload Null
//!     Retries(u64),                      // disc 1, payload integer
//!     Failed { code: u64, reason: str }, // disc 2, payload Map
//! }
//! ```
//!
//! The `Retries` integer and `Failed.code` integer carry the signedness
//! the normalization neutralizes: `U64` and `I64` collapse to the same
//! canonical integer tag, so a signed and an unsigned variant of this
//! shape MUST land on the same golden.

#![forbid(unsafe_code)]
#![allow(clippy::missing_panics_doc)]

use std::fmt::Write as _;

use obj_core::codec::{normalize_schema_to_postcard, DynamicSchema, EnumVariantSchema};

/// The frozen normalized-enum contract. Decoding (variant ordinals
/// `Null=0,Bool=1,U64=2,I64=3,F64=4,String=5,Bytes=6,Seq=7,`
/// `Map=8,Enum=9`):
///
/// - `09` Enum, `03` three variants,
/// - `00` disc 0 `06 "Active"` `00` Null,
/// - `01` disc 1 `07 "Retries"` `02` U64,
/// - `02` disc 2 `06 "Failed"` `08` Map `02` two fields,
///   `04 "code"` `02` U64, `06 "reason"` `05` String.
///
/// Both `U64` and `I64` land on the same `02` integer tag after
/// normalization, INSIDE the enum payloads.
const GOLDEN_ENUM_NORMALIZED_HEX: &str =
    "09030006416374697665000107526574726965730202064661696c6564080204636f64650206726561736f6e05";

/// The canonical enum with unsigned integers (`u64` → `U64`).
fn canonical_enum_unsigned() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Active", DynamicSchema::Null),
        EnumVariantSchema::new(1, "Retries", DynamicSchema::U64),
        EnumVariantSchema::new(
            2,
            "Failed",
            DynamicSchema::map([
                ("code", DynamicSchema::U64),
                ("reason", DynamicSchema::String),
            ]),
        ),
    ])
}

/// The same logical enum with signed `i64` payloads. Normalization must
/// collapse `I64` and `U64` to the SAME tag, so this and the unsigned
/// enum MUST produce identical golden bytes.
fn canonical_enum_signed() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Active", DynamicSchema::Null),
        EnumVariantSchema::new(1, "Retries", DynamicSchema::I64),
        EnumVariantSchema::new(
            2,
            "Failed",
            DynamicSchema::map([
                ("code", DynamicSchema::I64),
                ("reason", DynamicSchema::String),
            ]),
        ),
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
fn enum_normalized_bytes_match_golden() {
    let bytes = normalize_schema_to_postcard(&canonical_enum_unsigned()).expect("encode");
    assert_eq!(
        to_hex(&bytes),
        GOLDEN_ENUM_NORMALIZED_HEX,
        "normalized enum schema bytes drifted from the golden",
    );
}

#[test]
fn enum_signed_int_normalizes_to_same_golden() {
    let unsigned =
        normalize_schema_to_postcard(&canonical_enum_unsigned()).expect("encode unsigned");
    let signed = normalize_schema_to_postcard(&canonical_enum_signed()).expect("encode signed");
    assert_eq!(
        unsigned, signed,
        "u64 and i64 inside enum payloads must normalize identically",
    );
    assert_eq!(to_hex(&signed), GOLDEN_ENUM_NORMALIZED_HEX);
}
