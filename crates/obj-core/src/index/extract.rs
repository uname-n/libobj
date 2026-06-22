//! Field extraction: `Document` → encoded index keys.
//!
//! [`extract_index_keys`] is the bridge between a user `Document`
//! and the per-index B-trees of the catalog layer. For each
//! declared index it walks the document's field path(s) and hands
//! the resolved [`Dynamic`] values to `encode_index_key`.
//!
//! # postcard is not self-describing
//!
//! The [`Dynamic`] type ships a `from_postcard_bytes` decoder
//! that ONLY accepts the tagged-Dynamic wire format — NOT raw
//! native-postcard payloads. Field extraction cannot use that
//! decoder for user documents (which are always native postcard).
//!
//! The workaround is a dedicated serde-driven reflection: a
//! `DynamicSerializer` walks the document's `serde::Serialize`
//! impl and emits a `Dynamic` tree with `Dynamic::Map` for every
//! struct and `Dynamic::Seq` for every sequence. The result is
//! the same shape `Dynamic::get` is built for, so the top-level
//! field-path walk in `extract_index_keys` is one `Dynamic::get`
//! call per `IndexSpec::key_paths` entry.
//!
//! # Path limitations
//!
//! Only **top-level field paths** are supported — a single field name
//! within the document's struct. Dotted paths (`"address.city"`),
//! array indexing (`"tags[0]"`), and `JSONPath` syntax are out of
//! scope. The `IndexSpec::key_paths` vector is a list of top-level
//! field names; each entry is one `Dynamic::get` lookup.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde::{ser, Serialize};

use crate::codec::{Document, Dynamic};
use crate::error::{Error, Result};
use crate::index::key::{encode_field, encode_index_key, EncodedIndexKey};
use crate::index::spec::{IndexKind, IndexSpec};

/// Maximum depth of the document-reflection walk. Mirrors the
/// `MAX_DYNAMIC_DEPTH` bound in [`crate::codec::dynamic`] —
/// defensive against pathological nested-struct inputs that would
/// otherwise grow `Dynamic` unboundedly.
const MAX_REFLECT_DEPTH: usize = 32;

/// Maximum number of entries a single `Each` extraction may emit.
/// 16 384 is the same ceiling we use for the per-document key set;
/// exceeding it suggests a runaway data shape rather than a real
/// indexable field.
pub const MAX_EACH_ENTRIES: usize = 16_384;

/// Extract the set of encoded index keys for `doc` under `spec`.
///
/// - For `Standard`, `Unique`, `Composite`: returns exactly one
///   [`EncodedIndexKey`].
/// - For `Each`: returns one entry per element of the sequence at
///   the configured path. Empty sequence → empty `Vec` (no index
///   work for this doc on this index).
///
/// `collection` is plumbed through purely so error variants carry
/// the collection name in their context (the catalog reconciler
/// calls this per-collection).
///
/// # Errors
///
/// - [`Error::IndexFieldMissing`] if the configured path is absent.
/// - [`Error::IndexFieldTypeMismatch`] for `Each` on a non-sequence,
///   `Composite` field on a `Map`, etc.
/// - [`Error::InvalidArgument`] if `Each` would emit more than
///   [`MAX_EACH_ENTRIES`].
/// - Propagates encoding errors from [`encode_index_key`].
pub fn extract_index_keys<T: Document>(
    collection: &str,
    spec: &IndexSpec,
    doc: &T,
) -> Result<Vec<EncodedIndexKey>> {
    let fields = project_fields(collection, spec, doc)?;
    match spec.kind {
        IndexKind::Standard | IndexKind::Unique => extract_scalar(spec, &fields).map(|k| vec![k]),
        IndexKind::Each => extract_each(collection, spec, &fields),
        IndexKind::Composite => extract_composite(spec, &fields).map(|k| vec![k]),
    }
}

/// Encode the single resolved scalar field. Used by `Standard` and
/// `Unique` kinds. `fields` carries exactly one entry — the value at
/// `spec.key_paths[0]` resolved by [`project_fields`].
fn extract_scalar(spec: &IndexSpec, fields: &[Dynamic]) -> Result<EncodedIndexKey> {
    debug_assert_eq!(fields.len(), 1, "scalar kind projects exactly one field");
    encode_index_key(spec, fields)
}

/// Encode one key per element of the resolved sequence field.
fn extract_each(
    collection: &str,
    spec: &IndexSpec,
    fields: &[Dynamic],
) -> Result<Vec<EncodedIndexKey>> {
    debug_assert_eq!(fields.len(), 1, "Each kind projects exactly one field");
    let value = &fields[0];
    let Dynamic::Seq(items) = value else {
        return Err(Error::IndexFieldTypeMismatch {
            collection: collection.to_owned(),
            index: spec.name.clone(),
            path: spec.key_paths[0].clone(),
            expected: "Seq",
            found: dynamic_kind_name(value),
        });
    };
    if items.len() > MAX_EACH_ENTRIES {
        return Err(Error::EachIndexTooLarge {
            collection: collection.to_owned(),
            index: spec.name.clone(),
            len: items.len(),
            max: MAX_EACH_ENTRIES,
        });
    }
    let mut out = Vec::with_capacity(items.len());
    for element in items {
        out.push(encode_field(element)?);
    }
    Ok(out)
}

/// Encode the resolved composite fields into the envelope.
fn extract_composite(spec: &IndexSpec, fields: &[Dynamic]) -> Result<EncodedIndexKey> {
    debug_assert!(fields.len() >= 2, "composite projects ≥ 2 fields");
    encode_index_key(spec, fields)
}

/// Project the `Dynamic` value at each `spec.key_paths` entry out of
/// `doc`, visiting ONLY those fields (every other field is discarded
/// by a no-op serializer — see [`NullSerializer`]). Returns one
/// resolved [`Dynamic`] per path, in `key_paths` order.
///
/// # Errors
///
/// - [`Error::IndexFieldMissing`] if a path is absent from `doc`'s
///   top-level struct (parity with the old `lookup_field` →
///   `Dynamic::get` → `None` path).
/// - [`Error::InvalidArgument`] if `doc` rejects reflection (a
///   non-struct/map top-level, depth overflow, or a `Serialize` impl
///   error) — collapsed from [`DynamicSerError`].
fn project_fields<T: Document>(
    collection: &str,
    spec: &IndexSpec,
    doc: &T,
) -> Result<Vec<Dynamic>> {
    debug_assert!(!spec.key_paths.is_empty(), "spec has ≥ 1 key path");
    let projector = FieldProjector::new(&spec.key_paths);
    let mut resolved = doc.serialize(projector).map_err(Error::from)?;
    let mut out = Vec::with_capacity(spec.key_paths.len());
    for path in &spec.key_paths {
        let value = resolved
            .remove(path)
            .ok_or_else(|| Error::IndexFieldMissing {
                collection: collection.to_owned(),
                index: spec.name.clone(),
                path: path.clone(),
            })?;
        out.push(value);
    }
    Ok(out)
}

/// Diagnostic name of a `Dynamic` variant, used in
/// [`Error::IndexFieldTypeMismatch::found`].
fn dynamic_kind_name(value: &Dynamic) -> &'static str {
    match value {
        Dynamic::Null => "Null",
        Dynamic::Bool(_) => "Bool",
        Dynamic::U64(_) => "U64",
        Dynamic::I64(_) => "I64",
        Dynamic::F64(_) => "F64",
        Dynamic::String(_) => "String",
        Dynamic::Bytes(_) => "Bytes",
        Dynamic::Seq(_) => "Seq",
        Dynamic::Map(_) => "Map",
        Dynamic::Enum { .. } => "Enum",
    }
}

/// Convert a `T: Serialize` into a full `Dynamic` tree by driving
/// serde through the [`DynamicSerializer`].
///
/// This is **not** a postcard round-trip — postcard is not self-
/// describing and cannot reconstruct field names. The serializer
/// emits a `Dynamic::Map` for every struct, keyed by serde's field
/// names; the field values are nested `Dynamic` trees following the
/// same rules recursively (bounded by [`MAX_REFLECT_DEPTH`]).
///
/// # Status
///
/// The index-extraction hot path NO LONGER calls this — it projects
/// only the indexed field(s) via [`FieldProjector`] so a large
/// unindexed `payload` is never cloned into a `Dynamic`. The
/// full-document walk is retained (test-gated) as the byte-identity
/// oracle the projecting path is validated against, and as the
/// reference shape for any future full-doc reflection caller
/// (reconcile / migration); `DynamicSerializer` itself remains live
/// on the production path because [`FieldProjector`] reuses it to
/// reflect each matched field's value.
#[cfg(test)]
fn to_dynamic<T: Serialize>(value: &T) -> Result<Dynamic> {
    let ser = DynamicSerializer { depth: 0 };
    value.serialize(ser).map_err(Error::from)
}

/// Top-level field-projecting `Serializer`. Collects the values of the
/// `wanted` field names into `collected`; discards every other field.
///
/// Only the struct / struct-variant / map top-level shapes collect
/// anything — a top-level scalar / seq / enum has no named fields, so
/// `collected` stays empty and [`project_fields`] surfaces the same
/// [`Error::IndexFieldMissing`] the old `Dynamic::get`-on-a-non-Map
/// path produced (error parity).
struct FieldProjector<'w> {
    wanted: &'w [String],
}

impl<'w> FieldProjector<'w> {
    fn new(wanted: &'w [String]) -> Self {
        Self { wanted }
    }
}

/// The accumulator a `FieldProjector` hands back: the resolved
/// `Dynamic` for each matched field name. A `BTreeMap` (not a `Vec`)
/// so `project_fields` can pull paths out in `key_paths` order and so
/// a duplicate field name (impossible for a real struct) cannot grow
/// the result unboundedly.
type ProjectedFields = BTreeMap<String, Dynamic>;

/// Builder shared by `SerializeStruct` / `SerializeStructVariant` /
/// `SerializeMap` for the projecting walk. Holds the wanted-name set
/// and the fields matched so far.
struct ProjectBuilder<'w> {
    wanted: &'w [String],
    collected: ProjectedFields,
    pending_key: Option<String>,
}

impl<'w> ProjectBuilder<'w> {
    fn new(wanted: &'w [String]) -> Self {
        Self {
            wanted,
            collected: BTreeMap::new(),
            pending_key: None,
        }
    }

    /// Reflect `value` into a `Dynamic` only if `key` is wanted;
    /// otherwise discard it through [`NullSerializer`] (no allocation).
    fn take_field<T: ?Sized + Serialize>(&mut self, key: &str, value: &T) -> DynRes<()> {
        if self.wanted.iter().any(|w| w == key) {
            let val = value.serialize(DynamicSerializer { depth: 1 })?;
            self.collected.insert(key.to_owned(), val);
        } else {
            value.serialize(NullSerializer)?;
        }
        Ok(())
    }
}

// allow: the serde `Serializer` trait dictates `self`-by-value methods; several take no field from `self`, so unused_self is unavoidable here.
#[allow(clippy::unused_self)]
impl<'w> ser::Serializer for FieldProjector<'w> {
    type Ok = ProjectedFields;
    type Error = DynamicSerError;
    type SerializeSeq = ser::Impossible<ProjectedFields, DynamicSerError>;
    type SerializeTuple = ser::Impossible<ProjectedFields, DynamicSerError>;
    type SerializeTupleStruct = ser::Impossible<ProjectedFields, DynamicSerError>;
    type SerializeTupleVariant = ser::Impossible<ProjectedFields, DynamicSerError>;
    type SerializeMap = ProjectBuilder<'w>;
    type SerializeStruct = ProjectBuilder<'w>;
    type SerializeStructVariant = ProjectBuilder<'w>;

    fn serialize_struct(self, _name: &'static str, _len: usize) -> DynRes<ProjectBuilder<'w>> {
        Ok(ProjectBuilder::new(self.wanted))
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> DynRes<ProjectBuilder<'w>> {
        Ok(ProjectBuilder::new(self.wanted))
    }
    fn serialize_map(self, _len: Option<usize>) -> DynRes<ProjectBuilder<'w>> {
        Ok(ProjectBuilder::new(self.wanted))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        v: &T,
    ) -> DynRes<ProjectedFields> {
        v.serialize(self)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> DynRes<ProjectedFields> {
        v.serialize(self)
    }
    fn serialize_i128(self, _v: i128) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_u128(self, _v: u128) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_bool(self, _v: bool) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_i8(self, _v: i8) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_i16(self, _v: i16) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_i32(self, _v: i32) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_i64(self, _v: i64) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_u8(self, _v: u8) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_u16(self, _v: u16) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_u32(self, _v: u32) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_u64(self, _v: u64) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_f32(self, _v: f32) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_f64(self, _v: f64) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_char(self, _v: char) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_str(self, _v: &str) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_bytes(self, _v: &[u8]) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_none(self) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_unit(self) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _v: &T,
    ) -> DynRes<ProjectedFields> {
        Ok(BTreeMap::new())
    }
    fn serialize_seq(self, _len: Option<usize>) -> DynRes<Self::SerializeSeq> {
        Err(seq_unsupported())
    }
    fn serialize_tuple(self, _len: usize) -> DynRes<Self::SerializeTuple> {
        Err(seq_unsupported())
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> DynRes<Self::SerializeTupleStruct> {
        Err(seq_unsupported())
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> DynRes<Self::SerializeTupleVariant> {
        Err(seq_unsupported())
    }
}

/// A top-level tuple / tuple-struct / tuple-variant has positional
/// (not named) fields and so contributes no addressable index field —
/// the same outcome as `Dynamic::get` on a non-Map. Surface it as the
/// reflection-rejected error rather than silently succeeding with no
/// fields, so a mis-shaped spec is not masked.
fn seq_unsupported() -> DynamicSerError {
    DynamicSerError("index extraction: top-level tuple has no named fields to project".to_owned())
}

impl ser::SerializeStruct for ProjectBuilder<'_> {
    type Ok = ProjectedFields;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        self.take_field(key, value)
    }
    fn end(self) -> DynRes<ProjectedFields> {
        Ok(self.collected)
    }
}

impl ser::SerializeStructVariant for ProjectBuilder<'_> {
    type Ok = ProjectedFields;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        self.take_field(key, value)
    }
    fn end(self) -> DynRes<ProjectedFields> {
        Ok(self.collected)
    }
}

impl ser::SerializeMap for ProjectBuilder<'_> {
    type Ok = ProjectedFields;
    type Error = DynamicSerError;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> DynRes<()> {
        let key_dyn = key.serialize(DynamicSerializer { depth: 1 })?;
        let key_string = match key_dyn {
            Dynamic::String(s) => s,
            Dynamic::U64(n) => n.to_string(),
            Dynamic::I64(n) => n.to_string(),
            Dynamic::Bool(b) => b.to_string(),
            other => {
                return Err(DynamicSerError(format!(
                    "map key must be stringable (got {other:?})"
                )));
            }
        };
        self.pending_key = Some(key_string);
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        let key = self
            .pending_key
            .take()
            .ok_or_else(|| DynamicSerError("map value without preceding key".to_owned()))?;
        self.take_field(&key, value)
    }
    fn end(self) -> DynRes<ProjectedFields> {
        Ok(self.collected)
    }
}

/// No-op serializer: visits and discards, allocates nothing.
struct NullSerializer;

// allow: the serde `Serializer` trait dictates `self`-by-value methods; this no-op serializer ignores `self` in every one, so unused_self is unavoidable.
#[allow(clippy::unused_self)]
impl ser::Serializer for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, _v: bool) -> DynRes<()> {
        Ok(())
    }
    fn serialize_i8(self, _v: i8) -> DynRes<()> {
        Ok(())
    }
    fn serialize_i16(self, _v: i16) -> DynRes<()> {
        Ok(())
    }
    fn serialize_i32(self, _v: i32) -> DynRes<()> {
        Ok(())
    }
    fn serialize_i64(self, _v: i64) -> DynRes<()> {
        Ok(())
    }
    fn serialize_u8(self, _v: u8) -> DynRes<()> {
        Ok(())
    }
    fn serialize_u16(self, _v: u16) -> DynRes<()> {
        Ok(())
    }
    fn serialize_u32(self, _v: u32) -> DynRes<()> {
        Ok(())
    }
    fn serialize_u64(self, _v: u64) -> DynRes<()> {
        Ok(())
    }
    fn serialize_f32(self, _v: f32) -> DynRes<()> {
        Ok(())
    }
    fn serialize_f64(self, _v: f64) -> DynRes<()> {
        Ok(())
    }
    fn serialize_char(self, _v: char) -> DynRes<()> {
        Ok(())
    }
    fn serialize_str(self, _v: &str) -> DynRes<()> {
        Ok(())
    }
    fn serialize_bytes(self, _v: &[u8]) -> DynRes<()> {
        Ok(())
    }
    fn serialize_none(self) -> DynRes<()> {
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> DynRes<()> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> DynRes<()> {
        Ok(())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> DynRes<()> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> DynRes<()> {
        Ok(())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        v: &T,
    ) -> DynRes<()> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        v: &T,
    ) -> DynRes<()> {
        v.serialize(self)
    }
    fn serialize_seq(self, _len: Option<usize>) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_tuple(self, _len: usize) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_map(self, _len: Option<usize>) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> DynRes<Self> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> DynRes<Self> {
        Ok(self)
    }
}

impl ser::SerializeSeq for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeTuple for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeMap for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> DynRes<()> {
        key.serialize(NullSerializer)
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeStruct for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for NullSerializer {
    type Ok = ();
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        value.serialize(NullSerializer)
    }
    fn end(self) -> DynRes<()> {
        Ok(())
    }
}

/// Errors surfaced from the [`DynamicSerializer`].
///
/// Lives next to the serializer because every serde method returns
/// a `Result<_, Self::Error>`. The error carries an owned message
/// so a wrapped `serde::ser::Error::custom` does not leak `'static`.
#[derive(Debug)]
struct DynamicSerError(String);

impl std::fmt::Display for DynamicSerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DynamicSerError {}

impl ser::Error for DynamicSerError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self(msg.to_string())
    }
}

impl From<DynamicSerError> for Error {
    fn from(_e: DynamicSerError) -> Self {
        Error::InvalidArgument(
            "index extraction: Document Serialize impl rejected reflection \
             (see DynamicSerError for the detail)",
        )
    }
}

/// Serde `Serializer` that converts the visited value into a
/// `Dynamic` tree. Bounded-depth via the `depth` counter.
struct DynamicSerializer {
    depth: usize,
}

impl DynamicSerializer {
    fn deeper(&self) -> Result<Self> {
        let next = self
            .depth
            .checked_add(1)
            .ok_or(Error::InvalidArgument("index extraction: depth overflow"))?;
        if next >= MAX_REFLECT_DEPTH {
            return Err(Error::InvalidArgument(
                "index extraction: max reflection depth exceeded",
            ));
        }
        Ok(Self { depth: next })
    }
}

/// Local helper for serde plumbing: a `Result<Dynamic,
/// DynamicSerError>` alias.
type DynRes<T = Dynamic> = std::result::Result<T, DynamicSerError>;

// allow: truncation is intentional — i128/u128 inputs are deliberately narrowed into Dynamic's 64-bit I64/U64 variants (the model has no 128-bit type).
#[allow(clippy::cast_possible_truncation)]
// allow: sign loss is intentional — the u128 input is narrowed via `as u64` into Dynamic::U64, which is the unsigned target by design.
#[allow(clippy::cast_sign_loss)]
impl ser::Serializer for DynamicSerializer {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    type SerializeSeq = SeqBuilder;
    type SerializeTuple = SeqBuilder;
    type SerializeTupleStruct = SeqBuilder;
    type SerializeTupleVariant = SeqBuilder;
    type SerializeMap = MapBuilder;
    type SerializeStruct = MapBuilder;
    type SerializeStructVariant = MapBuilder;

    fn serialize_bool(self, v: bool) -> DynRes {
        Ok(Dynamic::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> DynRes {
        Ok(Dynamic::I64(i64::from(v)))
    }
    fn serialize_i16(self, v: i16) -> DynRes {
        Ok(Dynamic::I64(i64::from(v)))
    }
    fn serialize_i32(self, v: i32) -> DynRes {
        Ok(Dynamic::I64(i64::from(v)))
    }
    fn serialize_i64(self, v: i64) -> DynRes {
        Ok(Dynamic::I64(v))
    }
    fn serialize_i128(self, v: i128) -> DynRes {
        Ok(Dynamic::I64(v as i64))
    }
    fn serialize_u8(self, v: u8) -> DynRes {
        Ok(Dynamic::U64(u64::from(v)))
    }
    fn serialize_u16(self, v: u16) -> DynRes {
        Ok(Dynamic::U64(u64::from(v)))
    }
    fn serialize_u32(self, v: u32) -> DynRes {
        Ok(Dynamic::U64(u64::from(v)))
    }
    fn serialize_u64(self, v: u64) -> DynRes {
        Ok(Dynamic::U64(v))
    }
    fn serialize_u128(self, v: u128) -> DynRes {
        Ok(Dynamic::U64(v as u64))
    }
    fn serialize_f32(self, v: f32) -> DynRes {
        Ok(Dynamic::F64(f64::from(v)))
    }
    fn serialize_f64(self, v: f64) -> DynRes {
        Ok(Dynamic::F64(v))
    }
    fn serialize_char(self, v: char) -> DynRes {
        Ok(Dynamic::String(v.to_string()))
    }
    fn serialize_str(self, v: &str) -> DynRes {
        Ok(Dynamic::String(v.to_owned()))
    }
    fn serialize_bytes(self, v: &[u8]) -> DynRes {
        Ok(Dynamic::Bytes(v.to_vec()))
    }
    fn serialize_none(self) -> DynRes {
        Ok(Dynamic::Null)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> DynRes {
        v.serialize(self)
    }
    fn serialize_unit(self) -> DynRes {
        Ok(Dynamic::Null)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> DynRes {
        Ok(Dynamic::Null)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> DynRes {
        Ok(Dynamic::String(variant.to_owned()))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, v: &T) -> DynRes {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        v: &T,
    ) -> DynRes {
        let inner = v.serialize(deeper(&self)?)?;
        let mut m = BTreeMap::new();
        m.insert(variant.to_owned(), inner);
        Ok(Dynamic::Map(m))
    }
    fn serialize_seq(self, len: Option<usize>) -> DynRes<SeqBuilder> {
        let cap = len.unwrap_or(0).min(MAX_EACH_ENTRIES);
        Ok(SeqBuilder {
            depth: self.depth,
            items: Vec::with_capacity(cap),
        })
    }
    fn serialize_tuple(self, len: usize) -> DynRes<SeqBuilder> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> DynRes<SeqBuilder> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        len: usize,
    ) -> DynRes<SeqBuilder> {
        self.serialize_seq(Some(len))
    }
    fn serialize_map(self, _len: Option<usize>) -> DynRes<MapBuilder> {
        Ok(MapBuilder {
            depth: self.depth,
            map: BTreeMap::new(),
            pending_key: None,
        })
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> DynRes<MapBuilder> {
        self.serialize_map(None)
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> DynRes<MapBuilder> {
        self.serialize_map(None)
    }
}

/// Construct a deeper [`DynamicSerializer`] for nested entries; the
/// result mirrors `self.depth + 1` and trips
/// [`MAX_REFLECT_DEPTH`].
fn deeper(s: &DynamicSerializer) -> DynRes<DynamicSerializer> {
    s.deeper().map_err(|e| {
        DynamicSerError(match e {
            Error::InvalidArgument(msg) => msg.to_owned(),
            other => other.to_string(),
        })
    })
}

/// Builder for sequence / tuple / tuple-struct / tuple-variant
/// shapes. Each `serialize_element` recurses into a deeper
/// `DynamicSerializer`.
struct SeqBuilder {
    depth: usize,
    items: Vec<Dynamic>,
}

impl ser::SerializeSeq for SeqBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        if self.items.len() >= MAX_EACH_ENTRIES {
            return Err(DynamicSerError(
                "index extraction: sequence exceeds MAX_EACH_ENTRIES".to_owned(),
            ));
        }
        let item = value.serialize(deeper(&DynamicSerializer { depth: self.depth })?)?;
        self.items.push(item);
        Ok(())
    }
    fn end(self) -> DynRes {
        Ok(Dynamic::Seq(self.items))
    }
}

impl ser::SerializeTuple for SeqBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        <Self as ser::SerializeSeq>::serialize_element(self, value)
    }
    fn end(self) -> DynRes {
        <Self as ser::SerializeSeq>::end(self)
    }
}

impl ser::SerializeTupleStruct for SeqBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        <Self as ser::SerializeSeq>::serialize_element(self, value)
    }
    fn end(self) -> DynRes {
        <Self as ser::SerializeSeq>::end(self)
    }
}

impl ser::SerializeTupleVariant for SeqBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        <Self as ser::SerializeSeq>::serialize_element(self, value)
    }
    fn end(self) -> DynRes {
        <Self as ser::SerializeSeq>::end(self)
    }
}

/// Builder for map / struct / struct-variant shapes. Uses
/// `pending_key` to handle the `serialize_key` / `serialize_value`
/// pairing required by `SerializeMap`.
struct MapBuilder {
    depth: usize,
    map: BTreeMap<String, Dynamic>,
    pending_key: Option<String>,
}

impl MapBuilder {
    fn deeper_serializer(&self) -> DynRes<DynamicSerializer> {
        deeper(&DynamicSerializer { depth: self.depth })
    }
}

impl ser::SerializeMap for MapBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> DynRes<()> {
        let key_dyn = key.serialize(self.deeper_serializer()?)?;
        let key_string = match key_dyn {
            Dynamic::String(s) => s,
            Dynamic::U64(n) => n.to_string(),
            Dynamic::I64(n) => n.to_string(),
            Dynamic::Bool(b) => b.to_string(),
            other => {
                return Err(DynamicSerError(format!(
                    "map key must be stringable (got {other:?})"
                )));
            }
        };
        self.pending_key = Some(key_string);
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> DynRes<()> {
        let key = self
            .pending_key
            .take()
            .ok_or_else(|| DynamicSerError("map value without preceding key".to_owned()))?;
        let val = value.serialize(self.deeper_serializer()?)?;
        self.map.insert(key, val);
        Ok(())
    }
    fn end(self) -> DynRes {
        Ok(Dynamic::Map(self.map))
    }
}

impl ser::SerializeStruct for MapBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        let val = value.serialize(self.deeper_serializer()?)?;
        self.map.insert(key.to_owned(), val);
        Ok(())
    }
    fn end(self) -> DynRes {
        Ok(Dynamic::Map(self.map))
    }
}

impl ser::SerializeStructVariant for MapBuilder {
    type Ok = Dynamic;
    type Error = DynamicSerError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> DynRes<()> {
        <Self as ser::SerializeStruct>::serialize_field(self, key, value)
    }
    fn end(self) -> DynRes {
        <Self as ser::SerializeStruct>::end(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct Customer {
        email: String,
        score: i64,
        tags: Vec<String>,
    }

    impl Document for Customer {
        const COLLECTION: &'static str = "customers";
        const VERSION: u32 = 1;
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Order {
        customer_id: u64,
        placed_at: u64,
        amount_cents: i64,
    }

    impl Document for Order {
        const COLLECTION: &'static str = "orders";
        const VERSION: u32 = 1;
    }

    #[test]
    fn dynamic_reflection_of_simple_struct() {
        let c = Customer {
            email: "ada@example.com".to_owned(),
            score: 42,
            tags: vec!["alpha".to_owned(), "beta".to_owned()],
        };
        let d = to_dynamic(&c).expect("reflect");
        let Dynamic::Map(map) = &d else {
            panic!("expected Map, got {d:?}");
        };
        assert_eq!(
            map.get("email"),
            Some(&Dynamic::String("ada@example.com".to_owned())),
        );
        assert_eq!(map.get("score"), Some(&Dynamic::I64(42)));
        match map.get("tags") {
            Some(Dynamic::Seq(items)) => assert_eq!(items.len(), 2),
            other => panic!("expected Seq, got {other:?}"),
        }
    }

    #[test]
    fn standard_extract_returns_one_key() {
        let c = Customer {
            email: "ada@example.com".to_owned(),
            score: 7,
            tags: vec![],
        };
        let spec = IndexSpec::standard("by_email", "email").expect("spec");
        let keys = extract_index_keys(Customer::COLLECTION, &spec, &c).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::String("ada@example.com".to_owned())).expect("enc");
        assert_eq!(keys[0], expected);
    }

    #[test]
    fn unique_extract_returns_one_key() {
        let c = Customer {
            email: "u@e.com".to_owned(),
            score: 1,
            tags: vec![],
        };
        let spec = IndexSpec::unique("by_email", "email").expect("spec");
        let keys = extract_index_keys(Customer::COLLECTION, &spec, &c).expect("extract");
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn each_extract_returns_n_keys() {
        let c = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        };
        let spec = IndexSpec::each("by_tag", "tags").expect("spec");
        let keys = extract_index_keys(Customer::COLLECTION, &spec, &c).expect("extract");
        assert_eq!(keys.len(), 3);
        let want_a = encode_field(&Dynamic::String("a".to_owned())).expect("enc");
        assert_eq!(keys[0], want_a);
    }

    #[test]
    fn each_extract_on_empty_seq_returns_empty_vec() {
        let c = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec![],
        };
        let spec = IndexSpec::each("by_tag", "tags").expect("spec");
        let keys = extract_index_keys(Customer::COLLECTION, &spec, &c).expect("extract");
        assert!(keys.is_empty());
    }

    #[test]
    fn composite_extract_returns_one_envelope_key() {
        let o = Order {
            customer_id: 7,
            placed_at: 12_345,
            amount_cents: 100,
        };
        let spec = IndexSpec::composite("by_ct", &["customer_id", "placed_at"]).expect("spec");
        let keys = extract_index_keys(Order::COLLECTION, &spec, &o).expect("extract");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].as_bytes()[0], crate::index::key::COMPOSITE_TAG);
    }

    #[test]
    fn missing_field_path_errors() {
        let c = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec![],
        };
        let spec = IndexSpec::standard("by_nope", "nope").expect("spec");
        let err = extract_index_keys(Customer::COLLECTION, &spec, &c).expect_err("missing");
        match err {
            Error::IndexFieldMissing {
                collection,
                index,
                path,
            } => {
                assert_eq!(collection, "customers");
                assert_eq!(index, "by_nope");
                assert_eq!(path, "nope");
            }
            other => panic!("expected IndexFieldMissing, got {other:?}"),
        }
    }

    #[test]
    fn each_on_non_seq_field_errors() {
        let c = Customer {
            email: "x".to_owned(),
            score: 99,
            tags: vec![],
        };
        let spec = IndexSpec::each("by_score", "score").expect("spec");
        let err = extract_index_keys(Customer::COLLECTION, &spec, &c).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch {
                collection,
                index,
                path,
                expected,
                found,
            } => {
                assert_eq!(collection, "customers");
                assert_eq!(index, "by_score");
                assert_eq!(path, "score");
                assert_eq!(expected, "Seq");
                assert_eq!(found, "I64");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn composite_decode_round_trip_matches_direct_encoding() {
        let o = Order {
            customer_id: 7,
            placed_at: 12_345,
            amount_cents: 100,
        };
        let spec = IndexSpec::composite("by_ct", &["customer_id", "placed_at"]).expect("spec");
        let extracted = extract_index_keys(Order::COLLECTION, &spec, &o).expect("extract");
        let direct = encode_index_key(
            &spec,
            &[Dynamic::U64(o.customer_id), Dynamic::U64(o.placed_at)],
        )
        .expect("direct");
        assert_eq!(extracted[0], direct);
    }

    /// Legacy scalar/each extraction: full-doc reflect → `get` →
    /// encode.
    fn legacy_extract<T: Document>(
        collection: &str,
        spec: &IndexSpec,
        doc: &T,
    ) -> Result<Vec<EncodedIndexKey>> {
        let dynamic = to_dynamic(doc)?;
        let lookup = |path: &str| -> Result<Dynamic> {
            dynamic
                .get(path)
                .cloned()
                .ok_or_else(|| Error::IndexFieldMissing {
                    collection: collection.to_owned(),
                    index: spec.name.clone(),
                    path: path.to_owned(),
                })
        };
        match spec.kind {
            IndexKind::Standard | IndexKind::Unique => {
                let v = lookup(&spec.key_paths[0])?;
                Ok(vec![encode_index_key(spec, std::slice::from_ref(&v))?])
            }
            IndexKind::Each => {
                let v = lookup(&spec.key_paths[0])?;
                let Dynamic::Seq(items) = v else {
                    panic!("legacy each on non-seq");
                };
                items.iter().map(encode_field).collect()
            }
            IndexKind::Composite => {
                let mut fields = Vec::new();
                for p in &spec.key_paths {
                    fields.push(lookup(p)?);
                }
                Ok(vec![encode_index_key(spec, &fields)?])
            }
        }
    }

    /// Every indexable field shape, in a single doc, so one struct
    /// exercises u64 / i64 (sign split) / String / bool / Option
    /// (Some + None) / newtype / enum / f64 / bytes for Standard,
    /// Unique, and Composite.
    #[derive(Debug, Serialize, Deserialize)]
    struct Shapes {
        u: u64,
        i_pos: i64,
        i_neg: i64,
        i_zero: i64,
        s: String,
        flag: bool,
        opt_some: Option<u64>,
        opt_none: Option<u64>,
        nt: Newtype,
        en: Color,
        f: f64,
        b: Bytes,
        payload: Vec<u8>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Newtype(i64);

    #[derive(Debug, Serialize, Deserialize)]
    enum Color {
        Red,
        Green,
    }

    /// A bytes-valued field. Hand-written `Serialize` so it drives
    /// `serialize_bytes` (→ `Dynamic::Bytes`) rather than a per-element
    /// seq — the shape the encoder's `TAG_BYTES` arm consumes.
    #[derive(Debug)]
    struct Bytes(Vec<u8>);

    impl Serialize for Bytes {
        fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
            ser.serialize_bytes(&self.0)
        }
    }

    impl<'de> Deserialize<'de> for Bytes {
        fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
            let v = <Vec<u8>>::deserialize(de)?;
            Ok(Bytes(v))
        }
    }

    impl Document for Shapes {
        const COLLECTION: &'static str = "shapes";
        const VERSION: u32 = 1;
    }

    fn sample_shapes() -> Shapes {
        Shapes {
            u: 7,
            i_pos: 42,
            i_neg: -42,
            i_zero: 0,
            s: "hello".to_owned(),
            flag: true,
            opt_some: Some(99),
            opt_none: None,
            nt: Newtype(-1),
            en: Color::Green,
            f: -1.5,
            b: Bytes(vec![0x00, 0x01, 0xFF]),
            payload: vec![0xAB; 480],
        }
    }

    /// Assert the projecting path is byte-identical to the legacy
    /// full-doc path for `spec` against `doc`.
    fn assert_byte_identical<T: Document>(spec: &IndexSpec, doc: &T) {
        let new = extract_index_keys(T::COLLECTION, spec, doc).expect("project extract");
        let old = legacy_extract(T::COLLECTION, spec, doc).expect("legacy extract");
        assert_eq!(
            new, old,
            "byte mismatch for spec={spec:?}: projecting path diverged from full-doc path"
        );
    }

    #[test]
    fn project_standard_unique_byte_identical_to_full_doc() {
        let doc = sample_shapes();
        for field in [
            "u", "i_pos", "i_neg", "i_zero", "s", "flag", "opt_some", "opt_none", "nt", "en", "f",
            "b",
        ] {
            let std = IndexSpec::standard("ix", field).expect("standard spec");
            assert_byte_identical(&std, &doc);
            let uniq = IndexSpec::unique("ix", field).expect("unique spec");
            assert_byte_identical(&uniq, &doc);
        }
    }

    #[test]
    fn project_composite_byte_identical_to_full_doc() {
        let doc = sample_shapes();
        let cases: &[&[&str]] = &[
            &["u", "i_neg"],
            &["s", "flag"],
            &["i_pos", "i_zero", "u"],
            &["opt_some", "opt_none"],
            &["nt", "en", "f"],
            &["b", "u", "s"],
        ];
        for paths in cases {
            let spec = IndexSpec::composite("ix", paths).expect("composite spec");
            assert_byte_identical(&spec, &doc);
        }
    }

    #[test]
    fn project_each_byte_identical_to_full_doc() {
        let c = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec!["a".to_owned(), "bb".to_owned(), "ccc".to_owned()],
        };
        let spec = IndexSpec::each("by_tag", "tags").expect("each spec");
        assert_byte_identical(&spec, &c);
        let empty = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec![],
        };
        assert_byte_identical(&spec, &empty);
    }

    #[test]
    fn project_missing_field_errors_identically() {
        let doc = sample_shapes();
        let spec = IndexSpec::standard("by_nope", "nope").expect("spec");
        let new = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("missing");
        let old = legacy_extract(Shapes::COLLECTION, &spec, &doc).expect_err("legacy missing");
        assert!(matches!(new, Error::IndexFieldMissing { .. }));
        assert_eq!(format!("{new:?}"), format!("{old:?}"));
    }

    #[test]
    fn project_each_on_non_seq_errors_identically() {
        let doc = sample_shapes();
        let spec = IndexSpec::each("by_u", "u").expect("spec");
        let new = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("type");
        match new {
            Error::IndexFieldTypeMismatch {
                ref path,
                expected,
                found,
                ..
            } => {
                assert_eq!(path, "u");
                assert_eq!(expected, "Seq");
                assert_eq!(found, "U64");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    // ── FieldProjector on struct-variant top-level ────────────────────────

    /// A document whose serde representation is a struct variant
    /// (enum with named fields). The `FieldProjector` goes through
    /// `serialize_struct_variant`, which opens a `ProjectBuilder`
    /// identical to the struct path.
    #[derive(Debug, Serialize, Deserialize)]
    enum Event {
        Created { user_id: u64, label: String },
        Deleted { user_id: u64 },
    }

    impl Document for Event {
        const COLLECTION: &'static str = "events";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_struct_variant_extracts_field() {
        let ev = Event::Created {
            user_id: 42,
            label: "hello".to_owned(),
        };
        let spec = IndexSpec::standard("by_user", "user_id").expect("spec");
        let keys = extract_index_keys(Event::COLLECTION, &spec, &ev).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(42)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    #[test]
    fn projector_struct_variant_missing_field_errors() {
        // Deleted variant only has user_id; asking for "label" → missing.
        let ev = Event::Deleted { user_id: 7 };
        let spec = IndexSpec::standard("by_label", "label").expect("spec");
        let err = extract_index_keys(Event::COLLECTION, &spec, &ev).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    // ── FieldProjector on map top-level ─────────────────────────────────

    /// A document whose serde representation is a `BTreeMap<String, u64>`.
    /// The `FieldProjector` takes the `serialize_map` path and
    /// `ProjectBuilder::serialize_key` / `serialize_value`.
    #[derive(Debug, Serialize, Deserialize)]
    struct MapDoc(std::collections::BTreeMap<String, u64>);

    impl Document for MapDoc {
        const COLLECTION: &'static str = "mapdocs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_map_doc_extracts_string_key() {
        let mut m = std::collections::BTreeMap::new();
        m.insert("score".to_owned(), 99u64);
        m.insert("rank".to_owned(), 1u64);
        let doc = MapDoc(m);
        let spec = IndexSpec::standard("by_score", "score").expect("spec");
        let keys = extract_index_keys(MapDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(99)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    #[test]
    fn projector_map_doc_missing_key_errors() {
        let mut m = std::collections::BTreeMap::new();
        m.insert("rank".to_owned(), 1u64);
        let doc = MapDoc(m);
        let spec = IndexSpec::standard("by_score", "score").expect("spec");
        let err = extract_index_keys(MapDoc::COLLECTION, &spec, &doc).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    // ── Scalar / seq / enum top-level shapes → IndexFieldMissing ────────

    /// A document that serializes as a plain scalar (u64). The
    /// `FieldProjector` returns an empty `BTreeMap` from
    /// `serialize_u64`, so every field path is absent.
    #[derive(Debug, Serialize, Deserialize)]
    struct ScalarDoc(u64);

    impl Document for ScalarDoc {
        const COLLECTION: &'static str = "scalars";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_scalar_top_level_errors_missing() {
        let doc = ScalarDoc(7);
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(ScalarDoc::COLLECTION, &spec, &doc).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    /// A document that serializes as a sequence. The `FieldProjector`
    /// returns an error from `serialize_seq` (via `seq_unsupported`),
    /// which is then converted to `Error::InvalidArgument`.
    #[derive(Debug, Serialize, Deserialize)]
    struct SeqDoc(Vec<u64>);

    impl Document for SeqDoc {
        const COLLECTION: &'static str = "seqdocs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_seq_top_level_errors_invalid_argument() {
        let doc = SeqDoc(vec![1, 2, 3]);
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(SeqDoc::COLLECTION, &spec, &doc).expect_err("tuple");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    /// A document that serializes as a tuple struct. The
    /// `FieldProjector` takes the `serialize_tuple_struct` path →
    /// `seq_unsupported` → `Error::InvalidArgument`.
    #[derive(Debug, Serialize, Deserialize)]
    struct TupleStructDoc(u64, String);

    impl Document for TupleStructDoc {
        const COLLECTION: &'static str = "tuples";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_tuple_struct_top_level_errors_invalid_argument() {
        let doc = TupleStructDoc(1, "x".to_owned());
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(TupleStructDoc::COLLECTION, &spec, &doc).expect_err("tuple");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    /// A document that serializes as a tuple variant. The
    /// `FieldProjector` takes the `serialize_tuple_variant` path →
    /// `seq_unsupported` → `Error::InvalidArgument`.
    #[derive(Debug, Serialize, Deserialize)]
    enum TupleVariantDoc {
        Point(u64, u64),
    }

    impl Document for TupleVariantDoc {
        const COLLECTION: &'static str = "tuplevar";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_tuple_variant_top_level_errors_invalid_argument() {
        let doc = TupleVariantDoc::Point(3, 4);
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err =
            extract_index_keys(TupleVariantDoc::COLLECTION, &spec, &doc).expect_err("tuple");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── FieldProjector: newtype-struct forwarding ───────────────────────

    /// A newtype wrapper around a struct. The `FieldProjector`
    /// receives `serialize_newtype_struct`, which delegates to the
    /// inner `Serialize` — the inner struct is then reflected as usual.
    #[derive(Debug, Serialize, Deserialize)]
    struct Inner {
        val: u64,
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct NewtypeWrapper(Inner);

    impl Document for NewtypeWrapper {
        const COLLECTION: &'static str = "wrappers";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_newtype_struct_delegates_to_inner() {
        let doc = NewtypeWrapper(Inner { val: 55 });
        let spec = IndexSpec::standard("by_val", "val").expect("spec");
        let keys = extract_index_keys(NewtypeWrapper::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(55)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    // ── FieldProjector: Some/None forwarding ───────────────────────────

    /// A document whose top-level serde form is `Option<Inner>`.
    /// `serialize_some` should forward into `Inner`'s struct fields;
    /// `serialize_none` returns an empty map → `IndexFieldMissing`.
    #[derive(Debug, Serialize, Deserialize)]
    struct OptDoc(Option<Inner>);

    impl Document for OptDoc {
        const COLLECTION: &'static str = "optdocs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_some_delegates_to_inner() {
        let doc = OptDoc(Some(Inner { val: 7 }));
        let spec = IndexSpec::standard("by_val", "val").expect("spec");
        let keys = extract_index_keys(OptDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(7)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    #[test]
    fn projector_none_top_level_errors_missing() {
        let doc = OptDoc(None);
        let spec = IndexSpec::standard("by_val", "val").expect("spec");
        let err = extract_index_keys(OptDoc::COLLECTION, &spec, &doc).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    // ── FieldProjector: i128 / u128 top-level → empty map ─────────────

    /// Hand-written `Serialize` impls that call `serialize_i128` /
    /// `serialize_u128` at the top level. Both paths on `FieldProjector`
    /// return `Ok(BTreeMap::new())`, which leads to `IndexFieldMissing`.
    struct I128Doc(i128);

    impl serde::Serialize for I128Doc {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            s.serialize_i128(self.0)
        }
    }

    impl<'de> serde::Deserialize<'de> for I128Doc {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            Ok(I128Doc(i128::deserialize(d)?))
        }
    }

    impl Document for I128Doc {
        const COLLECTION: &'static str = "i128docs";
        const VERSION: u32 = 1;
    }

    struct U128Doc(u128);

    impl serde::Serialize for U128Doc {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            s.serialize_u128(self.0)
        }
    }

    impl<'de> serde::Deserialize<'de> for U128Doc {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            Ok(U128Doc(u128::deserialize(d)?))
        }
    }

    impl Document for U128Doc {
        const COLLECTION: &'static str = "u128docs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_i128_top_level_errors_missing() {
        let doc = I128Doc(42);
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(I128Doc::COLLECTION, &spec, &doc).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    #[test]
    fn projector_u128_top_level_errors_missing() {
        let doc = U128Doc(42);
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(U128Doc::COLLECTION, &spec, &doc).expect_err("missing");
        assert!(matches!(err, Error::IndexFieldMissing { .. }));
    }

    // ── Each boundary: at / near / over MAX_EACH_ENTRIES ──────────────

    /// A document with a sequence field that can be sized at will for
    /// boundary tests.
    #[derive(Debug, Serialize, Deserialize)]
    struct BigSeqDoc {
        items: Vec<u64>,
    }

    impl Document for BigSeqDoc {
        const COLLECTION: &'static str = "bigseq";
        const VERSION: u32 = 1;
    }

    #[test]
    fn each_at_max_entries_succeeds() {
        // A Vec with exactly MAX_EACH_ENTRIES elements must succeed:
        // SeqBuilder only errors when it would push the
        // (MAX_EACH_ENTRIES + 1)th element.
        let doc = BigSeqDoc {
            items: (0..MAX_EACH_ENTRIES as u64).collect(),
        };
        let spec = IndexSpec::each("by_item", "items").expect("spec");
        let keys = extract_index_keys(BigSeqDoc::COLLECTION, &spec, &doc).expect("at-max");
        assert_eq!(keys.len(), MAX_EACH_ENTRIES);
    }

    #[test]
    fn each_seq_overflow_during_reflection_errors_invalid_argument() {
        // A Vec with MAX_EACH_ENTRIES + 1 elements will cause the
        // SeqBuilder inside DynamicSerializer to error when
        // serialize_element is called for the (MAX_EACH_ENTRIES + 1)th
        // element. That DynamicSerError propagates as InvalidArgument.
        let doc = BigSeqDoc {
            items: (0..=MAX_EACH_ENTRIES as u64).collect(),
        };
        let spec = IndexSpec::each("by_item", "items").expect("spec");
        let err =
            extract_index_keys(BigSeqDoc::COLLECTION, &spec, &doc).expect_err("overflow");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── Each: duplicate elements ────────────────────────────────────────

    #[test]
    fn each_with_duplicate_elements_emits_duplicate_keys() {
        // A sequence with repeated values must yield one key per element,
        // including duplicates. The caller (index maintenance) handles
        // deduplication if needed; extraction itself does not filter.
        let c = Customer {
            email: "x".to_owned(),
            score: 0,
            tags: vec!["dup".to_owned(), "dup".to_owned(), "unique".to_owned()],
        };
        let spec = IndexSpec::each("by_tag", "tags").expect("spec");
        let keys = extract_index_keys(Customer::COLLECTION, &spec, &c).expect("extract");
        assert_eq!(keys.len(), 3);
        // First two keys are the same encoded value.
        assert_eq!(keys[0], keys[1]);
        // Third key is different.
        assert_ne!(keys[0], keys[2]);
    }

    // ── dynamic_kind_name covers all Dynamic variants ──────────────────

    #[test]
    fn dynamic_kind_name_covers_all_variants() {
        assert_eq!(dynamic_kind_name(&Dynamic::Null), "Null");
        assert_eq!(dynamic_kind_name(&Dynamic::Bool(true)), "Bool");
        assert_eq!(dynamic_kind_name(&Dynamic::U64(0)), "U64");
        assert_eq!(dynamic_kind_name(&Dynamic::I64(0)), "I64");
        assert_eq!(dynamic_kind_name(&Dynamic::F64(0.0)), "F64");
        assert_eq!(
            dynamic_kind_name(&Dynamic::String(String::new())),
            "String"
        );
        assert_eq!(dynamic_kind_name(&Dynamic::Bytes(vec![])), "Bytes");
        assert_eq!(dynamic_kind_name(&Dynamic::Seq(vec![])), "Seq");
        assert_eq!(
            dynamic_kind_name(&Dynamic::Map(std::collections::BTreeMap::new())),
            "Map"
        );
        assert_eq!(
            dynamic_kind_name(&Dynamic::Enum {
                variant: "X".to_owned(),
                payload: Box::new(Dynamic::Null),
            }),
            "Enum"
        );
    }

    // ── each on Map / Null / Bool / F64 / String / Bytes / Enum ────────

    /// Exercises `dynamic_kind_name` indirectly via `extract_each`:
    /// the `found` field in `IndexFieldTypeMismatch` is produced by
    /// `dynamic_kind_name`.
    #[derive(Debug, Serialize, Deserialize)]
    struct NullFieldDoc {
        x: Option<u64>,
    }

    impl Document for NullFieldDoc {
        const COLLECTION: &'static str = "nulldocs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn each_on_null_field_reports_null_in_type_mismatch() {
        let doc = NullFieldDoc { x: None };
        let spec = IndexSpec::each("by_x", "x").expect("spec");
        let err = extract_index_keys(NullFieldDoc::COLLECTION, &spec, &doc).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch { found, .. } => {
                assert_eq!(found, "Null");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn each_on_bool_field_reports_bool_in_type_mismatch() {
        let doc = Shapes {
            flag: true,
            ..sample_shapes()
        };
        let spec = IndexSpec::each("by_flag", "flag").expect("spec");
        let err = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch { found, .. } => {
                assert_eq!(found, "Bool");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn each_on_f64_field_reports_f64_in_type_mismatch() {
        let doc = sample_shapes();
        let spec = IndexSpec::each("by_f", "f").expect("spec");
        let err = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch { found, .. } => {
                assert_eq!(found, "F64");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn each_on_string_field_reports_string_in_type_mismatch() {
        let doc = sample_shapes();
        let spec = IndexSpec::each("by_s", "s").expect("spec");
        let err = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch { found, .. } => {
                assert_eq!(found, "String");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn each_on_bytes_field_reports_bytes_in_type_mismatch() {
        let doc = sample_shapes();
        let spec = IndexSpec::each("by_b", "b").expect("spec");
        let err = extract_index_keys(Shapes::COLLECTION, &spec, &doc).expect_err("type");
        match err {
            Error::IndexFieldTypeMismatch { found, .. } => {
                assert_eq!(found, "Bytes");
            }
            other => panic!("expected IndexFieldTypeMismatch, got {other:?}"),
        }
    }

    // ── ProjectBuilder map key serialization paths ─────────────────────

    /// A map whose keys are `u64`, exercising the `Dynamic::U64(n) →
    /// n.to_string()` arm of `ProjectBuilder::serialize_key`.
    #[derive(Debug, Serialize, Deserialize)]
    struct U64KeyMapDoc(std::collections::BTreeMap<u64, String>);

    impl Document for U64KeyMapDoc {
        const COLLECTION: &'static str = "u64keymaps";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_map_u64_key_stringified_for_lookup() {
        let mut m = std::collections::BTreeMap::new();
        m.insert(42u64, "hello".to_owned());
        let doc = U64KeyMapDoc(m);
        // The map key 42 is stringified to "42"; look it up under that name.
        let spec = IndexSpec::standard("by_42", "42").expect("spec");
        let keys = extract_index_keys(U64KeyMapDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::String("hello".to_owned())).expect("enc");
        assert_eq!(keys[0], expected);
    }

    // ── DynamicSerializer depth overflow ──────────────────────────────

    /// A hand-written `Serialize` impl that recurses exactly
    /// `MAX_REFLECT_DEPTH` levels deep inside a field value, tripping
    /// the depth guard in `DynamicSerializer::deeper`. This tests the
    /// `Error::InvalidArgument("index extraction: max reflection depth exceeded")`
    /// path that `deeper()` converts into `DynamicSerError`, which
    /// `project_fields` maps to `Error::InvalidArgument`.
    struct DeepDoc {
        depth: usize,
    }

    impl<'de> serde::Deserialize<'de> for DeepDoc {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            Ok(DeepDoc {
                depth: usize::deserialize(d)?,
            })
        }
    }

    impl serde::Serialize for DeepDoc {
        fn serialize<S: serde::Serializer>(
            &self,
            s: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeStruct;
            // Serialize as a struct with one field "x" whose value is
            // another DeepDoc (one level shallower).  When depth == 0
            // the "x" field value is a plain u64.
            let mut st = s.serialize_struct("DeepDoc", 1)?;
            if self.depth == 0 {
                st.serialize_field("x", &0u64)?;
            } else {
                st.serialize_field("x", &DeepDoc { depth: self.depth - 1 })?;
            }
            st.end()
        }
    }

    impl Document for DeepDoc {
        const COLLECTION: &'static str = "deepdocs";
        const VERSION: u32 = 1;
    }

    #[test]
    fn dynamic_serializer_depth_overflow_errors() {
        // MAX_REFLECT_DEPTH is 32; FieldProjector starts at depth 1 for
        // the field value, so 32 levels of nesting overflows.
        let doc = DeepDoc {
            depth: MAX_REFLECT_DEPTH,
        };
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(DeepDoc::COLLECTION, &spec, &doc).expect_err("overflow");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    /// A recursively-nested **sequence** value: serializes as a
    /// one-element seq whose element is a shallower `DeepSeq` (or a
    /// scalar at depth 0). Exercises the `SeqBuilder::serialize_element`
    /// recursion path, which must be bounded by `MAX_REFLECT_DEPTH`.
    struct DeepSeq {
        depth: usize,
    }

    impl serde::Serialize for DeepSeq {
        fn serialize<S: serde::Serializer>(
            &self,
            s: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut sq = s.serialize_seq(Some(1))?;
            if self.depth == 0 {
                sq.serialize_element(&0u64)?;
            } else {
                sq.serialize_element(&DeepSeq { depth: self.depth - 1 })?;
            }
            sq.end()
        }
    }

    /// A recursively-nested **map** value: serializes as a one-entry map
    /// whose value is a shallower `DeepMap` (or a scalar at depth 0).
    /// Exercises the `MapBuilder::deeper_serializer` recursion path
    /// (`SerializeMap`), which must be bounded by `MAX_REFLECT_DEPTH`.
    struct DeepMap {
        depth: usize,
    }

    impl serde::Serialize for DeepMap {
        fn serialize<S: serde::Serializer>(
            &self,
            s: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            let mut mp = s.serialize_map(Some(1))?;
            mp.serialize_key("x")?;
            if self.depth == 0 {
                mp.serialize_value(&0u64)?;
            } else {
                mp.serialize_value(&DeepMap { depth: self.depth - 1 })?;
            }
            mp.end()
        }
    }

    /// Wraps a recursively-nested compound value in the single indexed
    /// field `"x"` of a top-level struct, so `extract_index_keys` drives
    /// the reflection walk over `inner`. `Deserialize` is a trivial stub
    /// (never exercised) to satisfy the `Document` bound.
    struct DeepFieldDoc<V> {
        inner: V,
    }

    impl<'de, V> serde::Deserialize<'de> for DeepFieldDoc<V> {
        fn deserialize<D: serde::Deserializer<'de>>(
            _d: D,
        ) -> std::result::Result<Self, D::Error> {
            Err(serde::de::Error::custom("DeepFieldDoc is serialize-only"))
        }
    }

    impl<V: serde::Serialize> serde::Serialize for DeepFieldDoc<V> {
        fn serialize<S: serde::Serializer>(
            &self,
            s: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeStruct;
            let mut st = s.serialize_struct("DeepFieldDoc", 1)?;
            st.serialize_field("x", &self.inner)?;
            st.end()
        }
    }

    impl Document for DeepFieldDoc<DeepSeq> {
        const COLLECTION: &'static str = "deepseqdocs";
        const VERSION: u32 = 1;
    }

    impl Document for DeepFieldDoc<DeepMap> {
        const COLLECTION: &'static str = "deepmapdocs";
        const VERSION: u32 = 1;
    }

    /// Regression for the seq recursion path: before the fix
    /// `SeqBuilder::serialize_element` incremented `self.depth + 1`
    /// without consulting `MAX_REFLECT_DEPTH`, so a deeply nested seq
    /// aborted the process via stack overflow. It must now return a
    /// graceful `Error::InvalidArgument` — and a depth far past the
    /// bound must still error gracefully, not crash.
    #[test]
    fn deep_seq_nesting_errors_not_aborts() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        for depth in [MAX_REFLECT_DEPTH, 5_000] {
            let doc = DeepFieldDoc {
                inner: DeepSeq { depth },
            };
            let err = extract_index_keys(
                <DeepFieldDoc<DeepSeq>>::COLLECTION,
                &spec,
                &doc,
            )
            .expect_err("deep seq overflow");
            assert!(matches!(err, Error::InvalidArgument(_)));
        }
    }

    /// Regression for the map/struct recursion path: before the fix
    /// `MapBuilder::deeper_serializer` incremented `self.depth + 1`
    /// without consulting `MAX_REFLECT_DEPTH`. A deeply nested map must
    /// now return a graceful `Error::InvalidArgument`, even far past the
    /// bound.
    #[test]
    fn deep_map_nesting_errors_not_aborts() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        for depth in [MAX_REFLECT_DEPTH, 5_000] {
            let doc = DeepFieldDoc {
                inner: DeepMap { depth },
            };
            let err = extract_index_keys(
                <DeepFieldDoc<DeepMap>>::COLLECTION,
                &spec,
                &doc,
            )
            .expect_err("deep map overflow");
            assert!(matches!(err, Error::InvalidArgument(_)));
        }
    }

    /// Regression for the struct recursion path at a depth far past the
    /// bound: complements `dynamic_serializer_depth_overflow_errors`
    /// (which tests exactly `MAX_REFLECT_DEPTH`) by confirming a
    /// pathological ~5000-deep struct chain returns a graceful `Err`
    /// rather than aborting the process.
    #[test]
    fn deep_struct_nesting_errors_not_aborts() {
        let doc = DeepDoc { depth: 5_000 };
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(DeepDoc::COLLECTION, &spec, &doc)
            .expect_err("deep struct overflow");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── Composite error paths ─────────────────────────────────────────

    /// A document whose composite spec names a field that does not
    /// exist. `project_fields` returns `IndexFieldMissing` for the
    /// absent path.
    #[derive(Debug, Serialize, Deserialize)]
    struct TwoFieldDoc {
        a: u64,
        b: u64,
    }

    impl Document for TwoFieldDoc {
        const COLLECTION: &'static str = "twofield";
        const VERSION: u32 = 1;
    }

    #[test]
    fn composite_missing_component_errors_index_field_missing() {
        let doc = TwoFieldDoc { a: 1, b: 2 };
        // "c" does not exist in TwoFieldDoc.
        let spec = IndexSpec::composite("by_abc", &["a", "c"]).expect("spec");
        let err = extract_index_keys(TwoFieldDoc::COLLECTION, &spec, &doc).expect_err("missing");
        match err {
            Error::IndexFieldMissing {
                collection,
                index,
                path,
            } => {
                assert_eq!(collection, "twofield");
                assert_eq!(index, "by_abc");
                assert_eq!(path, "c");
            }
            other => panic!("expected IndexFieldMissing, got {other:?}"),
        }
    }

    /// A document where one composite component field holds a `Vec`
    /// (which reflects as `Dynamic::Seq`). `encode_composite` →
    /// `write_one_into` rejects `Seq` with `Error::InvalidArgument`.
    #[derive(Debug, Serialize, Deserialize)]
    struct SeqComponentDoc {
        id: u64,
        tags: Vec<String>,
    }

    impl Document for SeqComponentDoc {
        const COLLECTION: &'static str = "seqcomp";
        const VERSION: u32 = 1;
    }

    #[test]
    fn composite_seq_component_errors_invalid_argument() {
        let doc = SeqComponentDoc {
            id: 42,
            tags: vec!["a".to_owned()],
        };
        // "tags" resolves to Dynamic::Seq, which is not a primitive
        // indexable type; encode_composite must reject it.
        let spec = IndexSpec::composite("by_id_tags", &["id", "tags"]).expect("spec");
        let err =
            extract_index_keys(SeqComponentDoc::COLLECTION, &spec, &doc).expect_err("seq in composite");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── Composite: three-field byte identity ──────────────────────────

    #[test]
    fn composite_three_field_byte_identical_to_direct_encoding() {
        // Verify that the extraction path produces the same bytes as
        // constructing the key directly, for a 3-field composite.
        let o = Order {
            customer_id: 5,
            placed_at: 999,
            amount_cents: -50,
        };
        let spec =
            IndexSpec::composite("by_all", &["customer_id", "placed_at", "amount_cents"])
                .expect("spec");
        let extracted = extract_index_keys(Order::COLLECTION, &spec, &o).expect("extract");
        let direct = encode_index_key(
            &spec,
            &[
                Dynamic::U64(o.customer_id),
                Dynamic::U64(o.placed_at),
                Dynamic::I64(o.amount_cents),
            ],
        )
        .expect("direct");
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0], direct);
        assert_eq!(extracted[0].as_bytes()[0], crate::index::key::COMPOSITE_TAG);
    }

    // ── FieldProjector: every remaining top-level scalar stub ──────────

    /// Selects which `Serializer` entry point [`TopScalarDoc`] drives.
    /// One variant per `FieldProjector` stub that returns
    /// `Ok(BTreeMap::new())` and is not covered by a derived doc shape.
    #[derive(Debug, Clone, Copy)]
    enum ScalarMode {
        Bool,
        I8,
        I16,
        I32,
        I64,
        U8,
        U16,
        U32,
        F32,
        F64,
        Char,
        Str,
        Bytes,
        Unit,
        UnitStruct,
        UnitVariant,
        NewtypeVariant,
    }

    /// Hand-written `Serialize` that hits exactly one top-level
    /// `FieldProjector` method per [`ScalarMode`]. All of them yield an
    /// empty projection, so extraction reports `IndexFieldMissing`.
    struct TopScalarDoc(ScalarMode);

    impl Serialize for TopScalarDoc {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            match self.0 {
                ScalarMode::Bool => s.serialize_bool(true),
                ScalarMode::I8 => s.serialize_i8(-1),
                ScalarMode::I16 => s.serialize_i16(-2),
                ScalarMode::I32 => s.serialize_i32(-3),
                ScalarMode::I64 => s.serialize_i64(-4),
                ScalarMode::U8 => s.serialize_u8(1),
                ScalarMode::U16 => s.serialize_u16(2),
                ScalarMode::U32 => s.serialize_u32(3),
                ScalarMode::F32 => s.serialize_f32(1.5),
                ScalarMode::F64 => s.serialize_f64(2.5),
                ScalarMode::Char => s.serialize_char('q'),
                ScalarMode::Str => s.serialize_str("top"),
                ScalarMode::Bytes => s.serialize_bytes(&[0xAB]),
                ScalarMode::Unit => s.serialize_unit(),
                ScalarMode::UnitStruct => s.serialize_unit_struct("Marker"),
                ScalarMode::UnitVariant => s.serialize_unit_variant("E", 0, "V"),
                ScalarMode::NewtypeVariant => s.serialize_newtype_variant("E", 0, "V", &7u64),
            }
        }
    }

    impl<'de> serde::Deserialize<'de> for TopScalarDoc {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            serde::de::IgnoredAny::deserialize(d)?;
            Ok(Self(ScalarMode::Unit))
        }
    }

    impl Document for TopScalarDoc {
        const COLLECTION: &'static str = "topscalars";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_scalar_top_levels_error_missing() {
        let modes = [
            ScalarMode::Bool,
            ScalarMode::I8,
            ScalarMode::I16,
            ScalarMode::I32,
            ScalarMode::I64,
            ScalarMode::U8,
            ScalarMode::U16,
            ScalarMode::U32,
            ScalarMode::F32,
            ScalarMode::F64,
            ScalarMode::Char,
            ScalarMode::Str,
            ScalarMode::Bytes,
            ScalarMode::Unit,
            ScalarMode::UnitStruct,
            ScalarMode::UnitVariant,
            ScalarMode::NewtypeVariant,
        ];
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        for mode in modes {
            let doc = TopScalarDoc(mode);
            let err = extract_index_keys(TopScalarDoc::COLLECTION, &spec, &doc)
                .expect_err("empty projection must report the path as missing");
            assert!(
                matches!(err, Error::IndexFieldMissing { .. }),
                "mode {mode:?}: expected IndexFieldMissing, got {err:?}"
            );
        }
    }

    // ── FieldProjector: top-level tuple → seq_unsupported ──────────────

    /// Hand-written `Serialize` driving `serialize_tuple` at the top
    /// level — the one `seq_unsupported` arm no derived doc reaches
    /// (derives produce tuple *structs* / *variants*, not bare tuples).
    struct TopTupleDoc;

    impl Serialize for TopTupleDoc {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeTuple;
            let mut t = s.serialize_tuple(2)?;
            t.serialize_element(&1u64)?;
            t.serialize_element(&2u64)?;
            t.end()
        }
    }

    impl<'de> serde::Deserialize<'de> for TopTupleDoc {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            serde::de::IgnoredAny::deserialize(d)?;
            Ok(Self)
        }
    }

    impl Document for TopTupleDoc {
        const COLLECTION: &'static str = "toptuples";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_bare_tuple_top_level_errors_invalid_argument() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(TopTupleDoc::COLLECTION, &spec, &TopTupleDoc)
            .expect_err("tuple top-level");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── ProjectBuilder: i64 / bool map keys + key error paths ──────────

    /// Map keyed by `i64`, exercising the `Dynamic::I64(n) →
    /// n.to_string()` arm of `ProjectBuilder::serialize_key`.
    #[derive(Debug, Serialize, Deserialize)]
    struct I64KeyMapDoc(std::collections::BTreeMap<i64, u64>);

    impl Document for I64KeyMapDoc {
        const COLLECTION: &'static str = "i64keymaps";
        const VERSION: u32 = 1;
    }

    /// Map keyed by `bool`, exercising the `Dynamic::Bool(b) →
    /// b.to_string()` arm of `ProjectBuilder::serialize_key`.
    #[derive(Debug, Serialize, Deserialize)]
    struct BoolKeyMapDoc(std::collections::BTreeMap<bool, u64>);

    impl Document for BoolKeyMapDoc {
        const COLLECTION: &'static str = "boolkeymaps";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_map_i64_key_stringified_for_lookup() {
        let mut m = std::collections::BTreeMap::new();
        m.insert(-5i64, 7u64);
        let doc = I64KeyMapDoc(m);
        let spec = IndexSpec::standard("by_neg5", "-5").expect("spec");
        let keys = extract_index_keys(I64KeyMapDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(7)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    #[test]
    fn projector_map_bool_key_stringified_for_lookup() {
        let mut m = std::collections::BTreeMap::new();
        m.insert(true, 3u64);
        let doc = BoolKeyMapDoc(m);
        let spec = IndexSpec::standard("by_true", "true").expect("spec");
        let keys = extract_index_keys(BoolKeyMapDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(3)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    /// A map whose key reflects to `Dynamic::F64` — not coercible to a
    /// field name. Drives the non-stringable-key error arm of BOTH
    /// `ProjectBuilder::serialize_key` (top-level extract) and
    /// `MapBuilder::serialize_key` (full reflection via `to_dynamic`).
    struct BadKeyMap;

    impl Serialize for BadKeyMap {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            let mut m = s.serialize_map(Some(1))?;
            m.serialize_key(&1.5f64)?;
            m.serialize_value(&1u64)?;
            m.end()
        }
    }

    impl<'de> serde::Deserialize<'de> for BadKeyMap {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            serde::de::IgnoredAny::deserialize(d)?;
            Ok(Self)
        }
    }

    impl Document for BadKeyMap {
        const COLLECTION: &'static str = "badkeymaps";
        const VERSION: u32 = 1;
    }

    /// A map that emits a value with no preceding key. Drives the
    /// `pending_key == None` error arm of both `ProjectBuilder` and
    /// `MapBuilder` `serialize_value`.
    struct ValuelessMap;

    impl Serialize for ValuelessMap {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            let mut m = s.serialize_map(Some(1))?;
            m.serialize_value(&1u64)?;
            m.end()
        }
    }

    impl<'de> serde::Deserialize<'de> for ValuelessMap {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            serde::de::IgnoredAny::deserialize(d)?;
            Ok(Self)
        }
    }

    impl Document for ValuelessMap {
        const COLLECTION: &'static str = "valuelessmaps";
        const VERSION: u32 = 1;
    }

    #[test]
    fn projector_map_non_stringable_key_errors() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(BadKeyMap::COLLECTION, &spec, &BadKeyMap)
            .expect_err("f64 map key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn projector_map_value_without_key_errors() {
        let spec = IndexSpec::standard("by_x", "x").expect("spec");
        let err = extract_index_keys(ValuelessMap::COLLECTION, &spec, &ValuelessMap)
            .expect_err("value without key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    // ── NullSerializer: discard every unwanted field shape ─────────────

    /// One variant per compound enum shape, so a single unwanted-field
    /// doc walks `NullSerializer`'s unit-variant, newtype-variant,
    /// tuple-variant, and struct-variant paths.
    #[derive(Debug, Serialize, Deserialize)]
    enum Mixed {
        Unit,
        New(u64),
        Tup(u64, bool),
        Struct { a: u64 },
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct PairTuple(u64, i64);

    #[derive(Debug, Serialize, Deserialize)]
    struct UnitMarker;

    /// A doc with one wanted field and one unwanted field of every
    /// shape `NullSerializer` must visit-and-discard: narrow ints,
    /// f32, char, unit, unit struct, bare tuple, tuple struct, map
    /// (key + value), nested struct, and all four enum variant kinds.
    #[derive(Debug, Serialize, Deserialize)]
    struct KitchenDoc {
        wanted: u64,
        skip_i8: i8,
        skip_i16: i16,
        skip_i32: i32,
        skip_u16: u16,
        skip_u32: u32,
        skip_f32: f32,
        skip_char: char,
        skip_unit: (),
        skip_unit_struct: UnitMarker,
        skip_tuple: (u64, bool),
        skip_tuple_struct: PairTuple,
        skip_map: std::collections::BTreeMap<String, u64>,
        skip_struct: Inner,
        skip_enum_unit: Mixed,
        skip_enum_newtype: Mixed,
        skip_enum_tuple: Mixed,
        skip_enum_struct: Mixed,
    }

    impl Document for KitchenDoc {
        const COLLECTION: &'static str = "kitchen";
        const VERSION: u32 = 1;
    }

    #[test]
    fn null_serializer_discards_every_unwanted_shape() {
        let mut m = std::collections::BTreeMap::new();
        m.insert("k".to_owned(), 1u64);
        let doc = KitchenDoc {
            wanted: 9,
            skip_i8: -1,
            skip_i16: -2,
            skip_i32: -3,
            skip_u16: 2,
            skip_u32: 3,
            skip_f32: 1.5,
            skip_char: 'z',
            skip_unit: (),
            skip_unit_struct: UnitMarker,
            skip_tuple: (4, true),
            skip_tuple_struct: PairTuple(5, -6),
            skip_map: m,
            skip_struct: Inner { val: 8 },
            skip_enum_unit: Mixed::Unit,
            skip_enum_newtype: Mixed::New(10),
            skip_enum_tuple: Mixed::Tup(11, false),
            skip_enum_struct: Mixed::Struct { a: 12 },
        };
        let spec = IndexSpec::standard("by_wanted", "wanted").expect("spec");
        let keys = extract_index_keys(KitchenDoc::COLLECTION, &spec, &doc).expect("extract");
        assert_eq!(keys.len(), 1);
        let expected = encode_field(&Dynamic::U64(9)).expect("enc");
        assert_eq!(keys[0], expected);
    }

    // ── DynamicSerializer: scalar widths, 128-bit narrowing, shapes ────

    #[test]
    fn dynamic_scalar_widths_reflect() {
        assert_eq!(to_dynamic(&-1i8).expect("i8"), Dynamic::I64(-1));
        assert_eq!(to_dynamic(&-2i16).expect("i16"), Dynamic::I64(-2));
        assert_eq!(to_dynamic(&-3i32).expect("i32"), Dynamic::I64(-3));
        assert_eq!(to_dynamic(&1u8).expect("u8"), Dynamic::U64(1));
        assert_eq!(to_dynamic(&2u16).expect("u16"), Dynamic::U64(2));
        assert_eq!(to_dynamic(&3u32).expect("u32"), Dynamic::U64(3));
        assert_eq!(to_dynamic(&1.5f32).expect("f32"), Dynamic::F64(1.5));
        assert_eq!(to_dynamic(&'q').expect("char"), Dynamic::String("q".to_owned()));
        assert_eq!(to_dynamic(&()).expect("unit"), Dynamic::Null);
        assert_eq!(to_dynamic(&UnitMarker).expect("unit struct"), Dynamic::Null);
    }

    #[test]
    fn dynamic_i128_u128_narrow_to_64_bit() {
        assert_eq!(to_dynamic(&-5i128).expect("i128"), Dynamic::I64(-5));
        assert_eq!(to_dynamic(&5u128).expect("u128"), Dynamic::U64(5));
    }

    #[test]
    fn dynamic_compound_shapes_reflect() {
        assert_eq!(
            to_dynamic(&(1u64, true)).expect("tuple"),
            Dynamic::Seq(vec![Dynamic::U64(1), Dynamic::Bool(true)]),
        );
        assert_eq!(
            to_dynamic(&PairTuple(1, -2)).expect("tuple struct"),
            Dynamic::Seq(vec![Dynamic::U64(1), Dynamic::I64(-2)]),
        );
        assert_eq!(
            to_dynamic(&Mixed::Unit).expect("unit variant"),
            Dynamic::String("Unit".to_owned()),
        );
        assert_eq!(
            to_dynamic(&Mixed::New(7)).expect("newtype variant"),
            Dynamic::Map(BTreeMap::from([("New".to_owned(), Dynamic::U64(7))])),
        );
        assert_eq!(
            to_dynamic(&Mixed::Tup(1, false)).expect("tuple variant"),
            Dynamic::Seq(vec![Dynamic::U64(1), Dynamic::Bool(false)]),
        );
        assert_eq!(
            to_dynamic(&Mixed::Struct { a: 3 }).expect("struct variant"),
            Dynamic::Map(BTreeMap::from([("a".to_owned(), Dynamic::U64(3))])),
        );
    }

    // ── MapBuilder: key coercion + key error paths ─────────────────────

    #[test]
    fn dynamic_map_numeric_and_bool_keys_stringified() {
        let mut mu = BTreeMap::new();
        mu.insert(7u64, 1u64);
        assert_eq!(
            to_dynamic(&mu).expect("u64 keys"),
            Dynamic::Map(BTreeMap::from([("7".to_owned(), Dynamic::U64(1))])),
        );
        let mut mi = BTreeMap::new();
        mi.insert(-7i64, 2u64);
        assert_eq!(
            to_dynamic(&mi).expect("i64 keys"),
            Dynamic::Map(BTreeMap::from([("-7".to_owned(), Dynamic::U64(2))])),
        );
        let mut mb = BTreeMap::new();
        mb.insert(false, 3u64);
        assert_eq!(
            to_dynamic(&mb).expect("bool keys"),
            Dynamic::Map(BTreeMap::from([("false".to_owned(), Dynamic::U64(3))])),
        );
    }

    #[test]
    fn dynamic_map_non_stringable_key_errors() {
        let err = to_dynamic(&BadKeyMap).expect_err("f64 map key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn dynamic_map_value_without_key_errors() {
        let err = to_dynamic(&ValuelessMap).expect_err("value without key");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }
}
