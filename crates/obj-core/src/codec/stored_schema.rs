//! [`StoredSchema`] — the disk-backed, cross-language-normalized form
//! of a [`DynamicSchema`], plus the normalization pass that produces
//! it.
//!
//! # Why a separate stored form?
//!
//! A live [`DynamicSchema`] describes the *writer's* wire shape, which
//! differs by the field's declared integer type: a `u64` field maps to
//! [`DynamicSchema::U64`] while an `i64` field maps to
//! [`DynamicSchema::I64`]. Two writers describing the *same* logical
//! column with different signedness therefore emit *different* schema
//! bytes.
//!
//! The disk catalog must store **one** canonical shape so a drift
//! guard does not turn a benign, wire-compatible signedness divergence
//! into a hard write failure. [`StoredSchema`] is that canonical form:
//!
//! - [`StoredSchema::schema`] is the **normalized** shape:
//!   [`normalize_schema`] collapses every integer width/signedness
//!   variant to a single canonical integer tag and every float
//!   variant to a single canonical float tag, recursing through
//!   `Seq` / `Map` / `Enum`. A `u64` field and an `i64` field for the
//!   same column produce **identical** normalized bytes.
//! - [`StoredSchema::int_signed`] is a per-integer-field **signedness
//!   hint** that is preserved verbatim but is *not* part of the shape:
//!   the drift guard compares
//!   [`StoredSchema::schema`] only, never the hint. The hint exists so
//!   the migration decoder can re-pick plain-varint vs
//!   zigzag without re-deriving the reader's signedness for fields
//!   that no longer exist.
//!
//! # Bootstrap invariant
//!
//! **System rows like [`StoredSchema`] are decoded by compiled-in
//! concrete types, never by a stored [`DynamicSchema`].** The schema
//! catalog decodes *documents*; documents never decode the catalog.
//! `postcard::from_bytes::<StoredSchema>` is owned by this binary, so
//! there is no infinite regress: the thing that describes how to read
//! old documents is itself read by a hard-coded Rust type, not by a
//! schema that would need a schema to read.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::codec::schema::{DynamicSchema, EnumVariantSchema, MAX_SCHEMA_DEPTH};
use crate::error::{Error, Result};

/// The wire-format discriminator this build of obj writes and
/// understands for [`StoredSchema`]. The decoder rejects any other
/// value with [`Error::UnsupportedSchemaFormat`].
pub const STORED_SCHEMA_FORMAT_V1: u8 = 1;

/// Upper bound on the number of schema nodes [`normalize_schema`] and
/// the hint collector will visit before surfacing an error. Far above
/// any plausible real schema; a guard against an adversarial tree that
/// is wide rather than deep.
const MAX_SCHEMA_NODES: usize = 1 << 20;

/// The disk-backed, cross-language-normalized form of a
/// [`DynamicSchema`].
///
/// `format` is the **first** field so postcard places its single byte
/// at offset 0 of the encoded row; [`StoredSchema::from_postcard_bytes`]
/// reads it first and rejects unknown discriminators. See the module
/// docs for the bootstrap invariant and the normalization
/// rationale.
///
/// # Hint vs. shape
///
/// [`schema`](StoredSchema::schema) is the drift-comparable shape.
/// [`int_signed`](StoredSchema::int_signed) is *metadata*: it
/// round-trips through serde but is deliberately excluded from the
/// equality the drift guard uses — the guard compares the
/// normalized `schema` only. Two `StoredSchema` values whose `schema`
/// fields are equal describe the same wire shape even if their
/// `int_signed` hints differ (e.g. a `u64` writer vs. an `i64` writer
/// for the same column).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSchema {
    /// Wire-format discriminator. MUST be the first decoded field so
    /// postcard places it at offset 0. Starts at
    /// [`STORED_SCHEMA_FORMAT_V1`] (`1`); `0` is never written.
    pub format: u8,
    /// The canonical, cross-language-normalized field shape. Produced
    /// by [`normalize_schema`]; this is what the drift guard compares.
    pub schema: DynamicSchema,
    /// Per-integer-field signedness hint, in the **pre-order traversal
    /// order** of the normalized [`schema`](StoredSchema::schema) (the
    /// same order [`collect_int_signedness`] visits integer slots).
    /// `true` = the source field was signed (zigzag varint), `false` =
    /// unsigned (plain varint).
    ///
    /// This is NOT part of the shape: the drift guard ignores it (it
    /// compares `schema` only). The decode-side re-specialization
    /// walks the normalized schema's integer slots in the
    /// same pre-order and consumes this list element-by-element to pick
    /// plain-varint vs zigzag for fields the reader's live type no
    /// longer describes.
    pub int_signed: Vec<bool>,
}

impl StoredSchema {
    /// Build a v1 [`StoredSchema`] from a *live* (un-normalized)
    /// schema: normalize the shape and record the per-integer
    /// signedness hint from the original.
    ///
    /// # Errors
    ///
    /// [`Error::SchemaDepthExceeded`] if the schema nests deeper than
    /// [`MAX_SCHEMA_DEPTH`].
    pub fn from_live(schema: &DynamicSchema) -> Result<Self> {
        let normalized = normalize_schema(schema)?;
        let int_signed = collect_int_signedness(schema)?;
        debug_assert_eq!(
            collect_int_signedness(&normalized)?.len(),
            int_signed.len(),
            "normalization must preserve the integer-slot count",
        );
        Ok(Self {
            format: STORED_SCHEMA_FORMAT_V1,
            schema: normalized,
            int_signed,
        })
    }

    /// Encode this row to postcard bytes. `format` lands at offset 0.
    ///
    /// # Errors
    ///
    /// [`Error::Codec`] if postcard encoding fails.
    pub fn to_postcard_bytes(&self) -> Result<Vec<u8>> {
        debug_assert_ne!(self.format, 0, "format 0 is never a valid stored row");
        postcard::to_allocvec(self).map_err(Error::from)
    }

    /// Decode a [`StoredSchema`] from postcard bytes, reading `format`
    /// first (offset 0) and rejecting any value this build does not
    /// understand.
    ///
    /// # Errors
    ///
    /// - [`Error::Codec`] if the bytes are not a well-formed
    ///   [`StoredSchema`].
    /// - [`Error::UnsupportedSchemaFormat`] if `format` is not
    ///   [`STORED_SCHEMA_FORMAT_V1`]. The error carries the offending
    ///   discriminator; no panic.
    pub fn from_postcard_bytes(bytes: &[u8]) -> Result<Self> {
        let stored: StoredSchema = postcard::from_bytes(bytes).map_err(Error::from)?;
        if stored.format != STORED_SCHEMA_FORMAT_V1 {
            return Err(Error::UnsupportedSchemaFormat {
                format: stored.format,
            });
        }
        Ok(stored)
    }
}

/// Normalize a live [`DynamicSchema`] into its canonical,
/// cross-language form.
///
/// The transform is shape-only:
///
/// - every integer variant ([`DynamicSchema::U64`],
///   [`DynamicSchema::I64`], and any future integer width) collapses to
///   the single canonical integer tag [`DynamicSchema::U64`];
/// - every float variant ([`DynamicSchema::F64`], and any future float
///   width) collapses to the single canonical float tag
///   [`DynamicSchema::F64`];
/// - all other variants pass through structurally unchanged, recursing
///   into `Seq` / `Map` / `Enum` payloads.
///
/// Choosing existing variants as the canonical tags (rather than
/// adding a new `Int` / `Float` variant) keeps the frozen
/// [`DynamicSchema`] variant set append-only.
///
/// The signedness lost by collapsing `I64` → `U64` is preserved
/// separately in [`StoredSchema::int_signed`] via
/// [`collect_int_signedness`] — call both against the *same* live
/// schema (as [`StoredSchema::from_live`] does).
///
/// # Errors
///
/// [`Error::SchemaDepthExceeded`] if the schema nests deeper than
/// [`MAX_SCHEMA_DEPTH`].
pub fn normalize_schema(schema: &DynamicSchema) -> Result<DynamicSchema> {
    normalize_at(schema, 0)
}

/// Normalize a live [`DynamicSchema`] and return the **postcard
/// encoding of the normalized shape** — i.e. exactly the bytes stored
/// in [`StoredSchema::schema`] on disk.
///
/// This is the canonical normalized-schema encoding: equivalent
/// logical shapes produce byte-identical output here (a `u64` and an
/// `i64` field both normalize to the canonical integer tag, so they
/// agree). A golden-bytes test routes through this single function, so
/// the normalization cannot drift without the golden failing. Any
/// second-language writer that lowers the equivalent shape must
/// reproduce the same bytes.
///
/// # Errors
///
/// - [`Error::SchemaDepthExceeded`] if the schema nests deeper than
///   [`MAX_SCHEMA_DEPTH`].
/// - [`Error::Codec`] if postcard encoding fails.
pub fn normalize_schema_to_postcard(schema: &DynamicSchema) -> Result<Vec<u8>> {
    let normalized = normalize_schema(schema)?;
    postcard::to_allocvec(&normalized).map_err(Error::from)
}

/// Recursion-free worker for [`normalize_schema`].
///
/// Uses Rust-language recursion bounded by an explicit `depth` counter:
/// the tree is at most [`MAX_SCHEMA_DEPTH`] deep, so the call
/// chain is statically bounded and cannot exhaust the stack. Each level
/// rebuilds its node from already-normalized children.
fn normalize_at(schema: &DynamicSchema, depth: usize) -> Result<DynamicSchema> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        });
    }
    let next = depth + 1;
    match schema {
        DynamicSchema::U64 | DynamicSchema::I64 => Ok(DynamicSchema::U64),
        DynamicSchema::F64 => Ok(DynamicSchema::F64),
        DynamicSchema::Null => Ok(DynamicSchema::Null),
        DynamicSchema::Bool => Ok(DynamicSchema::Bool),
        DynamicSchema::String => Ok(DynamicSchema::String),
        DynamicSchema::Bytes => Ok(DynamicSchema::Bytes),
        DynamicSchema::Seq(inner) => Ok(DynamicSchema::Seq(Box::new(normalize_at(inner, next)?))),
        DynamicSchema::Map(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for (name, field) in fields {
                out.push((name.clone(), normalize_at(field, next)?));
            }
            Ok(DynamicSchema::Map(out))
        }
        DynamicSchema::Enum(variants) => {
            let mut out = Vec::with_capacity(variants.len());
            for v in variants {
                out.push(EnumVariantSchema::new(
                    v.discriminant,
                    v.name.clone(),
                    normalize_at(&v.payload, next)?,
                ));
            }
            Ok(DynamicSchema::Enum(out))
        }
    }
}

/// Collect the per-integer-field signedness hint for a *live* schema,
/// in pre-order traversal (the same order `normalize_at` visits
/// nodes, so the hint aligns slot-for-slot with the normalized
/// integer tags).
///
/// `true` for [`DynamicSchema::I64`] (signed, zigzag varint), `false`
/// for [`DynamicSchema::U64`] (unsigned, plain varint). Non-integer
/// nodes contribute nothing. The resulting `Vec<bool>` is stored in
/// [`StoredSchema::int_signed`].
///
/// # Errors
///
/// [`Error::SchemaDepthExceeded`] if the schema nests deeper than
/// [`MAX_SCHEMA_DEPTH`]; [`Error::SchemaTypeMismatch`] never — this is
/// a pure structural walk.
pub fn collect_int_signedness(schema: &DynamicSchema) -> Result<Vec<bool>> {
    let mut out: Vec<bool> = Vec::new();
    let mut stack: Vec<(&DynamicSchema, usize)> = vec![(schema, 0)];
    let mut visited = 0usize;
    while let Some((node, depth)) = stack.pop() {
        visited += 1;
        if visited > MAX_SCHEMA_NODES {
            return Err(Error::SchemaDepthExceeded {
                depth: MAX_SCHEMA_DEPTH,
            });
        }
        if depth > MAX_SCHEMA_DEPTH {
            return Err(Error::SchemaDepthExceeded {
                depth: MAX_SCHEMA_DEPTH,
            });
        }
        push_children(node, depth, &mut out, &mut stack);
    }
    Ok(out)
}

/// Record `node`'s signedness if it is an integer leaf, otherwise push
/// its children for the [`collect_int_signedness`] walk. Children are
/// pushed in reverse so the stack pops them in declaration order,
/// keeping the hint pre-order and stable.
fn push_children<'s>(
    node: &'s DynamicSchema,
    depth: usize,
    out: &mut Vec<bool>,
    stack: &mut Vec<(&'s DynamicSchema, usize)>,
) {
    let child_depth = depth + 1;
    match node {
        DynamicSchema::U64 => out.push(false),
        DynamicSchema::I64 => out.push(true),
        DynamicSchema::Seq(inner) => stack.push((inner, child_depth)),
        DynamicSchema::Map(fields) => {
            for (_name, field) in fields.iter().rev() {
                stack.push((field, child_depth));
            }
        }
        DynamicSchema::Enum(variants) => {
            for v in variants.iter().rev() {
                stack.push((&v.payload, child_depth));
            }
        }
        DynamicSchema::Null
        | DynamicSchema::Bool
        | DynamicSchema::F64
        | DynamicSchema::String
        | DynamicSchema::Bytes => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::schema::EnumVariantSchema;

    fn nested_schema() -> DynamicSchema {
        DynamicSchema::map([
            ("id", DynamicSchema::U64),
            ("name", DynamicSchema::String),
            ("tags", DynamicSchema::seq(DynamicSchema::String)),
            (
                "inner",
                DynamicSchema::map([("a", DynamicSchema::I64), ("b", DynamicSchema::F64)]),
            ),
            (
                "choice",
                DynamicSchema::enumeration([
                    EnumVariantSchema::new(0, "None", DynamicSchema::Null),
                    EnumVariantSchema::new(1, "Some", DynamicSchema::I64),
                ]),
            ),
        ])
    }

    #[test]
    fn dynamic_schema_round_trips_through_postcard() {
        let schema = nested_schema();
        let bytes = postcard::to_allocvec(&schema).expect("encode");
        let back: DynamicSchema = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, schema);
    }

    #[test]
    fn enum_variant_schema_round_trips() {
        let v = EnumVariantSchema::new(7, "Variant", DynamicSchema::Bytes);
        let bytes = postcard::to_allocvec(&v).expect("encode");
        let back: EnumVariantSchema = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, v);
    }

    #[test]
    fn stored_schema_round_trips() {
        let stored = StoredSchema::from_live(&nested_schema()).expect("from_live");
        let bytes = stored.to_postcard_bytes().expect("encode");
        let back = StoredSchema::from_postcard_bytes(&bytes).expect("decode");
        assert_eq!(back, stored);
        assert_eq!(back.format, STORED_SCHEMA_FORMAT_V1);
    }

    #[test]
    fn format_zero_lands_at_offset_zero() {
        let stored = StoredSchema::from_live(&DynamicSchema::U64).expect("from_live");
        let bytes = stored.to_postcard_bytes().expect("encode");
        assert_eq!(bytes.first().copied(), Some(STORED_SCHEMA_FORMAT_V1));
    }

    #[test]
    fn unknown_format_yields_unsupported_error_no_panic() {
        let rogue = StoredSchema {
            format: 2,
            schema: DynamicSchema::U64,
            int_signed: vec![false],
        };
        let bytes = postcard::to_allocvec(&rogue).expect("encode");
        let err = StoredSchema::from_postcard_bytes(&bytes).expect_err("unknown format");
        assert!(matches!(err, Error::UnsupportedSchemaFormat { format: 2 }));
    }

    #[test]
    fn normalize_collapses_u64_and_i64_identically() {
        let unsigned = DynamicSchema::map([("v", DynamicSchema::U64)]);
        let signed = DynamicSchema::map([("v", DynamicSchema::I64)]);
        let n_unsigned = normalize_schema(&unsigned).expect("normalize");
        let n_signed = normalize_schema(&signed).expect("normalize");
        assert_eq!(n_unsigned, n_signed);
        let b_unsigned = postcard::to_allocvec(&n_unsigned).expect("encode");
        let b_signed = postcard::to_allocvec(&n_signed).expect("encode");
        assert_eq!(b_unsigned, b_signed);
    }

    #[test]
    fn signedness_hint_distinguishes_u64_from_i64() {
        let unsigned = DynamicSchema::map([("v", DynamicSchema::U64)]);
        let signed = DynamicSchema::map([("v", DynamicSchema::I64)]);
        let s_unsigned = StoredSchema::from_live(&unsigned).expect("from_live");
        let s_signed = StoredSchema::from_live(&signed).expect("from_live");
        assert_eq!(s_unsigned.schema, s_signed.schema);
        assert_eq!(s_unsigned.int_signed, vec![false]);
        assert_eq!(s_signed.int_signed, vec![true]);
        assert_ne!(s_unsigned.int_signed, s_signed.int_signed);
    }

    #[test]
    fn signedness_hint_is_preorder_and_complete() {
        let hint = collect_int_signedness(&nested_schema()).expect("hint");
        assert_eq!(hint, vec![false, true, true]);
    }

    #[test]
    fn normalize_preserves_non_numeric_variants() {
        let schema = DynamicSchema::map([
            ("s", DynamicSchema::String),
            ("b", DynamicSchema::Bytes),
            ("flag", DynamicSchema::Bool),
            ("unit", DynamicSchema::Null),
            ("list", DynamicSchema::seq(DynamicSchema::String)),
        ]);
        let normalized = normalize_schema(&schema).expect("normalize");
        assert_eq!(normalized, schema);
    }

    #[test]
    fn float_widths_normalize_to_canonical_float() {
        let schema = DynamicSchema::map([("f", DynamicSchema::F64)]);
        let normalized = normalize_schema(&schema).expect("normalize");
        assert_eq!(normalized, schema);
        let hint = collect_int_signedness(&schema).expect("hint");
        assert!(hint.is_empty());
    }

    #[test]
    fn normalize_recurses_through_enum_and_seq() {
        let schema = DynamicSchema::seq(DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "U", DynamicSchema::U64),
            EnumVariantSchema::new(1, "S", DynamicSchema::I64),
        ]));
        let normalized = normalize_schema(&schema).expect("normalize");
        let expected = DynamicSchema::seq(DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "U", DynamicSchema::U64),
            EnumVariantSchema::new(1, "S", DynamicSchema::U64),
        ]));
        assert_eq!(normalized, expected);
    }
}
