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
    /// Before postcard touches the bytes, `ensure_schema_depth_bounded`
    /// pre-scans them iteratively and rejects a row whose `schema` field
    /// nests deeper than [`MAX_SCHEMA_DEPTH`]. The serde-derive
    /// `Deserialize` for the self-referential [`DynamicSchema`] recurses
    /// once per nesting level on the native stack and postcard imposes no
    /// depth limit, so an adversarial row (e.g. a long run of the `Seq`
    /// tag byte `0x07`) would otherwise overflow the stack — an
    /// uncatchable abort, not a recoverable error. The obj-side
    /// [`MAX_SCHEMA_DEPTH`] guards run only *after* the tree is
    /// materialized, so the pre-scan is what protects this step.
    ///
    /// # Errors
    ///
    /// - [`Error::SchemaDepthExceeded`] if the `schema` field nests
    ///   deeper than [`MAX_SCHEMA_DEPTH`] (caught before postcard
    ///   recurses).
    /// - [`Error::Codec`] if the bytes are not a well-formed
    ///   [`StoredSchema`].
    /// - [`Error::UnsupportedSchemaFormat`] if `format` is not
    ///   [`STORED_SCHEMA_FORMAT_V1`]. The error carries the offending
    ///   discriminator; no panic.
    pub fn from_postcard_bytes(bytes: &[u8]) -> Result<Self> {
        ensure_schema_depth_bounded(bytes)?;
        let stored: StoredSchema = postcard::from_bytes(bytes).map_err(Error::from)?;
        if stored.format != STORED_SCHEMA_FORMAT_V1 {
            return Err(Error::UnsupportedSchemaFormat {
                format: stored.format,
            });
        }
        Ok(stored)
    }
}

/// A minimal forward byte cursor over a postcard row, used only by the
/// pre-decode depth guard. Every read returns `None` on a short read so
/// the guard can defer the precise diagnosis to postcard rather than
/// inventing one.
struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read one byte and advance. `None` past end of input.
    fn read_u8(&mut self) -> Option<u8> {
        let byte = self.bytes.get(self.pos).copied()?;
        self.pos += 1;
        Some(byte)
    }

    /// Decode a postcard LEB128 varint as `u64`. The loop is bounded at
    /// the 10 bytes a `u64` needs (R2), returning `None` on a truncated
    /// or over-long varint.
    fn read_varint(&mut self) -> Option<u64> {
        let mut value: u64 = 0;
        for shift in (0..70u32).step_by(7) {
            let byte = self.read_u8()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    /// Advance past `n` bytes. `None` if fewer than `n` remain.
    fn skip(&mut self, n: usize) -> Option<()> {
        let end = self.pos.checked_add(n)?;
        if end > self.bytes.len() {
            return None;
        }
        self.pos = end;
        Some(())
    }
}

/// Pending parse work for the iterative depth guard. The variants mirror
/// postcard's field order so the guard tracks nesting depth exactly
/// without itself recursing on the native stack.
///
/// `Copy` because every field is a small integer; the guard pops items
/// off its work stack by value and never needs to retain the original.
#[derive(Clone, Copy)]
enum SchemaWork {
    /// Parse one [`DynamicSchema`] node header at the cursor, sitting at
    /// tree depth `depth`.
    Node { depth: usize },
    /// `remaining` more `(name, schema)` [`DynamicSchema::Map`] entries,
    /// whose schemas sit at `depth`.
    MapEntry { remaining: usize, depth: usize },
    /// `remaining` more [`DynamicSchema::Enum`] variants
    /// (`u32` discriminant, name, payload), whose payloads sit at
    /// `depth`.
    EnumVariant { remaining: usize, depth: usize },
}

/// Reject a stored-schema row whose `schema` field nests deeper than
/// [`MAX_SCHEMA_DEPTH`] *before* postcard deserializes it.
///
/// Walks the postcard byte stream with an explicit work stack (never the
/// native call stack), tracking the depth postcard's serde-derive
/// `Deserialize` would recurse to. The guard matches the engine's own
/// bound (`depth > MAX_SCHEMA_DEPTH`, as in [`normalize_at`]), so it
/// never rejects a schema this build could have written. Any
/// malformed/truncated row is left for postcard to diagnose — the guard
/// only fails on a *proven* over-deep nesting.
///
/// # Errors
///
/// [`Error::SchemaDepthExceeded`] if the `schema` field nests deeper than
/// [`MAX_SCHEMA_DEPTH`].
fn ensure_schema_depth_bounded(bytes: &[u8]) -> Result<()> {
    let mut cur = ByteCursor::new(bytes);
    // Skip the leading `format: u8` (postcard encodes u8 as one byte at
    // offset 0). A missing byte means a truncated row — defer to postcard.
    if cur.read_u8().is_none() {
        return Ok(());
    }
    let mut stack: Vec<SchemaWork> = vec![SchemaWork::Node { depth: 0 }];
    // Every iteration that makes progress consumes at least one input
    // byte, so the row length is a hard upper bound on the work (R2).
    let max_iters = bytes.len().saturating_add(1);
    for _ in 0..max_iters {
        let Some(item) = stack.pop() else {
            return Ok(());
        };
        if !step_schema_guard(&mut cur, item, &mut stack)? {
            return Ok(());
        }
    }
    Ok(())
}

/// Advance the depth guard by one work item. Returns `Ok(true)` to keep
/// going, `Ok(false)` to stop early on a short read (deferring to
/// postcard), or [`Error::SchemaDepthExceeded`] when the bound trips.
fn step_schema_guard(
    cur: &mut ByteCursor,
    item: SchemaWork,
    stack: &mut Vec<SchemaWork>,
) -> Result<bool> {
    match item {
        SchemaWork::Node { depth } => parse_node(cur, depth, stack),
        SchemaWork::MapEntry { remaining, depth } => {
            // Entry = (String name, DynamicSchema). Skip the name, queue
            // the continuation, then the node (LIFO → node first).
            let Some(name_len) = cur.read_varint() else {
                return Ok(false);
            };
            let Ok(n) = usize::try_from(name_len) else {
                return Ok(false);
            };
            if cur.skip(n).is_none() {
                return Ok(false);
            }
            if remaining > 1 {
                stack.push(SchemaWork::MapEntry {
                    remaining: remaining - 1,
                    depth,
                });
            }
            stack.push(SchemaWork::Node { depth });
            Ok(true)
        }
        SchemaWork::EnumVariant { remaining, depth } => {
            // Variant = (u32 discriminant, String name, payload node).
            if cur.read_varint().is_none() {
                return Ok(false);
            }
            let Some(name_len) = cur.read_varint() else {
                return Ok(false);
            };
            let Ok(n) = usize::try_from(name_len) else {
                return Ok(false);
            };
            if cur.skip(n).is_none() {
                return Ok(false);
            }
            if remaining > 1 {
                stack.push(SchemaWork::EnumVariant {
                    remaining: remaining - 1,
                    depth,
                });
            }
            stack.push(SchemaWork::Node { depth });
            Ok(true)
        }
    }
}

/// Parse one [`DynamicSchema`] node header at `depth`, queueing any
/// children one level deeper. Returns `Ok(false)` on a short/unknown
/// read (defer to postcard) and [`Error::SchemaDepthExceeded`] past the
/// bound.
fn parse_node(cur: &mut ByteCursor, depth: usize, stack: &mut Vec<SchemaWork>) -> Result<bool> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        });
    }
    let Some(disc) = cur.read_varint() else {
        return Ok(false);
    };
    let child_depth = depth + 1;
    match disc {
        // Null, Bool, U64, I64, F64, String, Bytes — leaf variants, no
        // payload in the schema encoding.
        0..=6 => Ok(true),
        // Seq(Box<DynamicSchema>) — exactly one child, one level down.
        7 => {
            stack.push(SchemaWork::Node { depth: child_depth });
            Ok(true)
        }
        8 => Ok(queue_collection(cur, child_depth, stack, false)),
        9 => Ok(queue_collection(cur, child_depth, stack, true)),
        // Unknown discriminant: let postcard report the codec error.
        _ => Ok(false),
    }
}

/// Read a `Map`/`Enum` length prefix and queue its `count` children at
/// `child_depth`. `is_enum` selects the variant layout (enum entries
/// carry a leading `u32` discriminant). Returns `false` on a short read
/// (deferring to postcard), `true` otherwise. Never fails the depth
/// bound itself — that is checked when each queued child node is parsed.
fn queue_collection(
    cur: &mut ByteCursor,
    child_depth: usize,
    stack: &mut Vec<SchemaWork>,
    is_enum: bool,
) -> bool {
    let Some(len) = cur.read_varint() else {
        return false;
    };
    let Ok(count) = usize::try_from(len) else {
        return false;
    };
    if count == 0 {
        return true;
    }
    stack.push(if is_enum {
        SchemaWork::EnumVariant {
            remaining: count,
            depth: child_depth,
        }
    } else {
        SchemaWork::MapEntry {
            remaining: count,
            depth: child_depth,
        }
    });
    true
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

    /// Hand-build a postcard `StoredSchema` row whose `schema` field is
    /// `seq_levels` nested `Seq` nodes wrapping a `Null` leaf, WITHOUT
    /// materializing the (potentially stack-blowing) `DynamicSchema` tree
    /// in Rust. Layout: `format` (1 byte) + `seq_levels` copies of the
    /// `Seq` discriminant `0x07` + the leaf `Null` discriminant `0x00` +
    /// the empty `int_signed` Vec length prefix `0x00`.
    fn nested_seq_row(seq_levels: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(seq_levels + 3);
        bytes.push(STORED_SCHEMA_FORMAT_V1); // format: u8 at offset 0
        bytes.extend(std::iter::repeat_n(7u8, seq_levels)); // Seq * N
        bytes.push(0u8); // Null leaf (innermost element)
        bytes.push(0u8); // int_signed: Vec<bool> length = 0
        bytes
    }

    #[test]
    fn deeply_nested_seq_row_errors_not_aborts() {
        // ~100k levels of Seq nesting — serde-derive would recurse ~100k
        // native stack frames and abort the process inside
        // `postcard::from_bytes` if the pre-scan did not reject it first.
        let bytes = nested_seq_row(100_000);
        let err = StoredSchema::from_postcard_bytes(&bytes)
            .expect_err("deeply nested schema must be rejected, not crash");
        assert!(
            matches!(err, Error::SchemaDepthExceeded { .. }),
            "expected SchemaDepthExceeded, got {err:?}",
        );
    }

    #[test]
    fn seq_row_at_max_depth_is_accepted_by_guard() {
        // A row exactly at the bound the guard tolerates (the deepest
        // node sits at `MAX_SCHEMA_DEPTH`) must pass the pre-scan and
        // decode through postcard unchanged — the guard must not reject a
        // schema this build could legitimately have written.
        let bytes = nested_seq_row(MAX_SCHEMA_DEPTH);
        let back = StoredSchema::from_postcard_bytes(&bytes)
            .expect("schema at MAX_SCHEMA_DEPTH must decode");
        // Confirm the decoded shape is the expected nested-Seq tree.
        let mut node = &back.schema;
        for _ in 0..MAX_SCHEMA_DEPTH {
            match node {
                DynamicSchema::Seq(inner) => node = inner,
                other => panic!("expected Seq at this level, got {other:?}"),
            }
        }
        assert_eq!(*node, DynamicSchema::Null);
    }

    #[test]
    fn seq_row_one_past_max_depth_is_rejected_by_guard() {
        // One level deeper than the guard tolerates must be rejected with
        // a clean error (this is shallow enough that postcard would not
        // overflow, so it isolates the guard's boundary).
        let bytes = nested_seq_row(MAX_SCHEMA_DEPTH + 2);
        let err = StoredSchema::from_postcard_bytes(&bytes)
            .expect_err("past MAX_SCHEMA_DEPTH must be rejected");
        assert!(
            matches!(err, Error::SchemaDepthExceeded { .. }),
            "expected SchemaDepthExceeded, got {err:?}",
        );
    }

    #[test]
    fn shallow_schema_still_round_trips_through_guard() {
        // The pre-scan must be transparent to every valid schema: a real
        // nested (Map/Seq/Enum/scalar) schema round-trips unchanged.
        let stored = StoredSchema::from_live(&nested_schema()).expect("from_live");
        let bytes = stored.to_postcard_bytes().expect("encode");
        let back = StoredSchema::from_postcard_bytes(&bytes).expect("decode");
        assert_eq!(back, stored);
    }
}
