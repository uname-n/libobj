//! [`Dynamic`] — reflective value tree for schema migration.
//!
//! `Dynamic` is the bridge that lets a [`Migrate`](crate::codec::Migrate)
//! impl read fields out of an older stored record without round-tripping
//! through every intermediate type. It is intentionally heavyweight (one
//! allocation per node); migration is a cold path and we trade speed for
//! reflective access.
//!
//! # Wire format — design choice
//!
//! `postcard` is **not self-describing** — there is no way to reconstruct
//! field names or even the structural shape of a previously-written
//! postcard payload from the bytes alone. Two paths are available:
//!
//! 1. Pair every `Document` with a hand-written
//!    `postcard::experimental::schema::Schema` description and use
//!    that to walk the bytes.
//! 2. Define an obj-internal "tagged Dynamic" wire format and round-trip
//!    via that.
//!
//! obj picks (2). Rationale: option (1) couples obj to postcard's
//! experimental-schema API which has neither stability nor MSRV
//! guarantees; option (2) is fully under our control, is auditable in
//! 100 lines of code, and never leaks onto the disk image of normal
//! documents (only migration code paths see it).
//!
//! `Dynamic` is *not* an on-disk encoding for application documents.
//! Application code that writes documents through
//! [`codec::encode`](crate::codec::encode) writes **native postcard**, NOT
//! the tagged format. `Dynamic` only appears in flight during a migration
//! and is discarded as soon as the migrated record reaches its
//! caller-visible type.
//!
//! # Schema-driven decode
//!
//! [`Dynamic::from_postcard_bytes`] decodes a **native-postcard** payload
//! into a structured `Dynamic` using a caller-supplied
//! [`DynamicSchema`]. This is the
//! path the codec takes when reading a v1 record through a v2
//! reader: it consults `Document::historical_schemas()`
//! for the v1 schema, walks the bytes per that schema, and hands
//! the resulting `Dynamic` to `Migrate::migrate` so the v2 author
//! can read fields by name. The walker is iterative
//! (`Vec<Frame>` stack) and bounded by
//! [`MAX_SCHEMA_DEPTH`].
//!
//! The obj-internal tagged-Dynamic format remains available via
//! [`Dynamic::from_tagged_bytes`] / [`Dynamic::to_postcard_bytes`]
//! for forensic logging of an in-flight `Dynamic`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::codec::schema::{DynamicSchema, EnumVariantSchema, MAX_SCHEMA_DEPTH};
use crate::error::{Error, Result};

/// Maximum depth of a [`Dynamic`] tree. Defense-in-depth bound that
/// stops a forged payload from triggering unbounded growth.
pub const MAX_DYNAMIC_DEPTH: usize = 32;

/// Maximum total node count in a [`Dynamic`] tree. Defense-in-depth
/// bound on the worst-case allocation a malformed payload could
/// trigger.
///
/// The container node (Seq or Map) itself consumes one slot, so the
/// effective maximum *element count* for a single Seq or Map is
/// `MAX_DYNAMIC_NODES - 1` (65 535). A Seq with 65 535 elements
/// uses exactly 65 536 total nodes — the full budget.
pub const MAX_DYNAMIC_NODES: usize = 65_536;

/// Upper bound on the capacity any single composite decode frame
/// reserves *up front* from its declared (attacker-controlled) length.
///
/// A `Seq` frame's wire length only sets `remaining`; the backing
/// `Vec` is grown on push as elements are actually decoded. Capping
/// the initial `Vec::with_capacity` here keeps the worst-case
/// eager reservation across all simultaneously-open frames bounded by
/// `max(MAX_SCHEMA_DEPTH, MAX_DYNAMIC_DEPTH) * SEQ_PREALLOC *
/// size_of::<Dynamic>()` (a few KB) across both the schema-walker
/// (`from_postcard_bytes`) and tagged (`from_tagged_bytes`) decode paths,
/// rather than `max(MAX_SCHEMA_DEPTH, MAX_DYNAMIC_DEPTH) *
/// MAX_DYNAMIC_NODES * size_of::<Dynamic>()` (tens of MB) that a forged
/// deeply-nested `Seq(Seq(..))` payload would otherwise amplify a
/// sub-100-byte input into. The actual memory used then tracks the
/// genuinely decoded element count, which each walker's iteration cap
/// already bounds. `16` is large enough that typical small sequences
/// never reallocate, small enough that the amplification ceiling is
/// negligible.
const SEQ_PREALLOC: usize = 16;

const TAG_NULL: u8 = 0x00;
const TAG_BOOL: u8 = 0x01;
const TAG_U64: u8 = 0x02;
const TAG_I64: u8 = 0x03;
const TAG_F64: u8 = 0x04;
const TAG_STRING: u8 = 0x05;
const TAG_BYTES: u8 = 0x06;
const TAG_SEQ: u8 = 0x07;
const TAG_MAP: u8 = 0x08;
const TAG_ENUM: u8 = 0x09;

/// A postcard-decoded view of a stored record.
///
/// `Dynamic` is intentionally simple. It is the input shape every
/// `Migrate::migrate` impl receives; the migration code reads the
/// fields it cares about via [`Dynamic::get`] or
/// [`Dynamic::deserialize`] and constructs the target type by hand.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Dynamic {
    /// JSON-like null / Rust-like `()`.
    Null,
    /// Boolean.
    Bool(bool),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit floating-point. NaN values do not compare equal to
    /// themselves; equality for `Dynamic::F64` follows the IEEE-754
    /// definition (so `Dynamic::F64(f64::NAN) != Dynamic::F64(f64::NAN)`).
    F64(f64),
    /// UTF-8 string.
    String(String),
    /// Raw byte sequence.
    Bytes(Vec<u8>),
    /// Ordered sequence of nested `Dynamic` values.
    Seq(Vec<Dynamic>),
    /// String-keyed map of nested `Dynamic` values. Iteration order
    /// is the `BTreeMap`'s sorted-by-key order.
    Map(BTreeMap<String, Dynamic>),
    /// Tagged-enum value. `variant` carries the Rust variant name
    /// (verbatim from
    /// [`EnumVariantSchema::name`](crate::codec::schema::EnumVariantSchema));
    /// `payload` carries the decoded inner value (a
    /// [`Dynamic::Null`] for unit variants, a [`Dynamic::Map`] for
    /// tuple / struct variants, or the inner type's `Dynamic` for
    /// newtype variants). Migration code distinguishes variants by
    /// `variant` and pulls fields out of `payload`.
    Enum {
        /// The Rust variant name.
        variant: String,
        /// The decoded inner payload.
        payload: Box<Dynamic>,
    },
}

impl Dynamic {
    /// Decode `bytes` as a tagged-`Dynamic` payload produced by
    /// [`Dynamic::to_postcard_bytes`].
    ///
    /// This is **not** the inverse of `postcard::to_allocvec(&doc)`
    /// for an arbitrary `Document` — see the module-level docs.
    /// [`Dynamic::deserialize`] reconstructs a typed value
    /// shape-faithfully from an in-memory `Dynamic` (see
    /// [`dynamic_de`](crate::codec::dynamic_de)).
    ///
    /// # Errors
    ///
    /// - [`Error::Codec`] if the underlying postcard decode fails.
    /// - [`Error::Corruption`] on an unknown tag, a length field that
    ///   exceeds the input, or a tree depth or node count beyond the
    ///   defense-in-depth bounds.
    pub fn from_tagged_bytes(bytes: &[u8]) -> Result<Self> {
        let (value, _rest) = Self::decode_value(bytes, 0, &mut 0)?;
        Ok(value)
    }

    /// Decode a **native-postcard** payload according to `schema`.
    ///
    /// This is the migration entry point: given the on-disk
    /// bytes of an older record + a [`DynamicSchema`] describing
    /// that older type's shape, the walker produces a structured
    /// `Dynamic` view that a [`Migrate::migrate`](crate::codec::Migrate)
    /// impl can read fields out of.
    ///
    /// Walks the byte stream iteratively with an explicit
    /// `Vec<Frame>` stack. Depth is bounded
    /// by [`MAX_SCHEMA_DEPTH`].
    ///
    /// # Errors
    ///
    /// - [`Error::SchemaDepthExceeded`] if the schema is deeper than
    ///   [`MAX_SCHEMA_DEPTH`].
    /// - [`Error::SchemaTypeMismatch`] if the bytes do not match the
    ///   schema (truncation, non-UTF-8 string, non-`0|1` bool, etc.).
    /// - [`Error::Corruption`] on a varint that overflows `u64`.
    pub fn from_postcard_bytes(bytes: &[u8], schema: &DynamicSchema) -> Result<Self> {
        let (value, rest) = walk_schema(bytes, schema)?;
        debug_assert!(rest.len() <= bytes.len(), "walker consumed more than input");
        if !rest.is_empty() {
            return Err(Error::SchemaTypeMismatch {
                expected: "exact",
                found: "trailing-bytes",
                path: String::new(),
            });
        }
        Ok(value)
    }

    /// Encode `self` as a tagged-`Dynamic` payload.
    ///
    /// # Errors
    ///
    /// - [`Error::Codec`] if the underlying postcard encode fails.
    pub fn to_postcard_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.encode_into(&mut buf)?;
        Ok(buf)
    }

    /// Look up `field` in a [`Dynamic::Map`]. Returns `None` for any
    /// non-map variant, mirroring JSON's "missing field" semantics.
    #[must_use]
    pub fn get(&self, field: &str) -> Option<&Dynamic> {
        if let Dynamic::Map(m) = self {
            m.get(field)
        } else {
            None
        }
    }

    /// Typed accessor for `String` fields. Errors if `field` is
    /// missing OR carries a non-string value — distinguishes
    /// "absent" from "wrong shape" so migration code can fail
    /// loudly on a schema mismatch.
    ///
    /// # Errors
    ///
    /// - [`Error::SchemaTypeMismatch`] with `expected = "String"`
    ///   when the field is absent OR the field's variant is not
    ///   [`Dynamic::String`].
    pub fn get_str(&self, field: &str) -> Result<&str> {
        match self.get(field) {
            Some(Dynamic::String(s)) => Ok(s.as_str()),
            Some(other) => Err(Error::SchemaTypeMismatch {
                expected: "String",
                found: variant_name(other),
                path: field.to_owned(),
            }),
            None => Err(Error::SchemaTypeMismatch {
                expected: "String",
                found: "absent",
                path: field.to_owned(),
            }),
        }
    }

    /// Set `field` to `value` in a [`Dynamic::Map`].
    ///
    /// On a non-map variant the method replaces `self` with a fresh
    /// `Map` containing only `(field, value)`. This matches the
    /// "best-effort upgrade" shape `Migrate` impls typically need
    /// when adding a brand-new field to a previously-scalar payload.
    ///
    /// Accepts any `impl Into<Dynamic>` — call sites can write
    /// `doc.set("tier", "gold")` or `doc.set("count", 42u64)`
    /// without explicitly wrapping the value.
    pub fn set<S, V>(&mut self, field: S, value: V)
    where
        S: Into<String>,
        V: Into<Dynamic>,
    {
        let key = field.into();
        let val = value.into();
        if let Dynamic::Map(m) = self {
            m.insert(key, val);
        } else {
            let mut m = BTreeMap::new();
            m.insert(key, val);
            *self = Dynamic::Map(m);
        }
    }

    /// Remove `field` from a [`Dynamic::Map`].
    ///
    /// Map-only. On a non-Map variant returns
    /// [`Error::DynamicPathNotMap`]. On a Map, returns the removed
    /// value (or `None` if absent) — distinguishes "field absent"
    /// from "called remove on the wrong shape".
    ///
    /// Typical use is inside a `Migrate::migrate` body when the
    /// new-version type has dropped a field that the old-version
    /// payload carried:
    ///
    /// ```ignore
    /// fn migrate(mut doc: Dynamic, _from: u32) -> Result<Self> {
    ///     doc.remove("deprecated_field")?; // drop, do not roundtrip
    ///     doc.set("new_field", "default");
    ///     doc.deserialize()
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`Error::DynamicPathNotMap`] if `self` is not a `Map`.
    pub fn remove(&mut self, field: &str) -> Result<Option<Dynamic>> {
        match self {
            Dynamic::Map(m) => Ok(m.remove(field)),
            _ => Err(Error::DynamicPathNotMap {
                path: field.to_owned(),
            }),
        }
    }

    /// If `self` is a [`Dynamic::Enum`], return the variant name;
    /// otherwise `None`. Mirrors [`Dynamic::get`]'s "missing field"
    /// semantics: a non-Enum value silently returns `None` rather
    /// than erroring.
    #[must_use]
    pub fn enum_variant(&self) -> Option<&str> {
        match self {
            Dynamic::Enum { variant, .. } => Some(variant.as_str()),
            _ => None,
        }
    }

    /// If `self` is a [`Dynamic::Enum`], return the decoded payload;
    /// otherwise `None`. Pairs with [`Dynamic::enum_variant`] in a
    /// `Migrate::migrate` body — match on the variant name, then
    /// pull fields out of the payload.
    #[must_use]
    pub fn enum_payload(&self) -> Option<&Dynamic> {
        match self {
            Dynamic::Enum { payload, .. } => Some(payload.as_ref()),
            _ => None,
        }
    }

    /// Deserialise `self` into the target type `T` **shape-faithfully**.
    ///
    /// Drives a [`serde::Deserializer`] directly over the in-memory
    /// `Dynamic` tree (see
    /// [`dynamic_de`](crate::codec::dynamic_de)), the same pattern
    /// `serde_json::Value`'s deserializer uses. Every `Dynamic`
    /// variant maps to its natural serde shape — crucially, a
    /// [`Dynamic::Enum`] becomes a serde enum (variant matched by
    /// name), so `Option<T>` and user-enum fields reconstruct
    /// correctly. This is the path `#[obj(auto_migrate)]`
    /// reconstructs fields through.
    ///
    /// # Errors
    ///
    /// - [`Error::Corruption`] if the `Dynamic` shape disagrees with
    ///   `T` (wrong variant, missing struct field, out-of-range
    ///   integer, or a tree deeper than
    ///   [`MAX_SCHEMA_DEPTH`]).
    ///   Never panics on a malformed `Dynamic`.
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        crate::codec::dynamic_de::from_dynamic::<T>(self).map_err(Error::from)
    }

    /// Walk-driven decode. `depth` is the current tree depth;
    /// `nodes` is the running count of decoded nodes.
    fn decode_value<'a>(
        bytes: &'a [u8],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(Self, &'a [u8])> {
        if depth >= MAX_DYNAMIC_DEPTH {
            return Err(Error::Corruption { page_id: 0 });
        }
        *nodes = nodes
            .checked_add(1)
            .ok_or(Error::Corruption { page_id: 0 })?;
        if *nodes > MAX_DYNAMIC_NODES {
            return Err(Error::Corruption { page_id: 0 });
        }
        let (tag, rest) = split_first(bytes)?;
        Self::decode_body(tag, rest, depth, nodes)
    }

    fn decode_body<'a>(
        tag: u8,
        rest: &'a [u8],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(Self, &'a [u8])> {
        match tag {
            TAG_NULL => Ok((Dynamic::Null, rest)),
            TAG_BOOL => decode_bool(rest),
            TAG_U64 => decode_u64(rest),
            TAG_I64 => decode_i64(rest),
            TAG_F64 => decode_f64(rest),
            TAG_STRING => decode_string(rest),
            TAG_BYTES => decode_bytes(rest),
            TAG_SEQ => Self::decode_seq(rest, depth + 1, nodes),
            TAG_MAP => Self::decode_map(rest, depth + 1, nodes),
            TAG_ENUM => Self::decode_enum(rest, depth + 1, nodes),
            _ => Err(Error::Corruption { page_id: 0 }),
        }
    }

    /// Decode a tagged-enum body: varint length + UTF-8 variant name,
    /// then a nested `Dynamic` payload. Mirrors [`Self::decode_map`]
    /// in its depth-bound accounting so a forged payload cannot
    /// trigger unbounded recursion via Enum-of-Enum-of-Enum… chains.
    fn decode_enum<'a>(
        bytes: &'a [u8],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(Self, &'a [u8])> {
        let (name_bytes, after_name) = take_len_prefixed(bytes)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| Error::Corruption { page_id: 0 })?
            .to_owned();
        let (payload, after_payload) = Self::decode_value(after_name, depth, nodes)?;
        Ok((
            Dynamic::Enum {
                variant: name,
                payload: Box::new(payload),
            },
            after_payload,
        ))
    }

    fn decode_seq<'a>(
        bytes: &'a [u8],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(Self, &'a [u8])> {
        let (len, mut rest) = take_varint_usize(bytes)?;
        // The container node itself already consumed one slot in `nodes`
        // (incremented by `decode_value` before dispatching here), so
        // allowing `len == MAX_DYNAMIC_NODES` elements would push the total
        // to `MAX_DYNAMIC_NODES + 1`. Reject at `>=` to cap element count
        // at `MAX_DYNAMIC_NODES - 1`, keeping the total within budget.
        if len >= MAX_DYNAMIC_NODES {
            return Err(Error::Corruption { page_id: 0 });
        }
        let mut items = Vec::with_capacity(len.min(SEQ_PREALLOC));
        for _ in 0..len {
            let (item, next) = Self::decode_value(rest, depth, nodes)?;
            items.push(item);
            rest = next;
        }
        Ok((Dynamic::Seq(items), rest))
    }

    fn decode_map<'a>(
        bytes: &'a [u8],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(Self, &'a [u8])> {
        let (len, mut rest) = take_varint_usize(bytes)?;
        // Same accounting as `decode_seq`: the Map container node itself
        // has already consumed one slot, so cap element count at
        // `MAX_DYNAMIC_NODES - 1` via a `>=` guard.
        if len >= MAX_DYNAMIC_NODES {
            return Err(Error::Corruption { page_id: 0 });
        }
        let mut map = BTreeMap::new();
        for _ in 0..len {
            let (key_bytes, after_key) = take_len_prefixed(rest)?;
            let key = std::str::from_utf8(key_bytes)
                .map_err(|_| Error::Corruption { page_id: 0 })?
                .to_owned();
            let (value, after_value) = Self::decode_value(after_key, depth, nodes)?;
            if map.insert(key, value).is_some() {
                return Err(Error::Corruption { page_id: 0 });
            }
            rest = after_value;
        }
        Ok((Dynamic::Map(map), rest))
    }

    fn encode_into(&self, dst: &mut Vec<u8>) -> Result<()> {
        encode_one_bounded(self, 0, dst)
    }
}

/// Recursive encoder bounded by [`MAX_DYNAMIC_DEPTH`]: every
/// recursive call increments `depth`; exceeding the
/// bound returns [`Error::Corruption`] rather than risking a stack
/// overflow on a forged input.
fn encode_one_bounded(value: &Dynamic, depth: usize, dst: &mut Vec<u8>) -> Result<()> {
    if depth >= MAX_DYNAMIC_DEPTH {
        return Err(Error::Corruption { page_id: 0 });
    }
    match value {
        Dynamic::Null => dst.push(TAG_NULL),
        Dynamic::Bool(b) => {
            dst.push(TAG_BOOL);
            dst.push(u8::from(*b));
        }
        Dynamic::U64(n) => {
            dst.push(TAG_U64);
            write_varint_u64(*n, dst);
        }
        Dynamic::I64(n) => {
            dst.push(TAG_I64);
            write_varint_i64(*n, dst);
        }
        Dynamic::F64(f) => {
            dst.push(TAG_F64);
            dst.extend_from_slice(&f.to_le_bytes());
        }
        Dynamic::String(s) => {
            dst.push(TAG_STRING);
            write_varint_u64(s.len() as u64, dst);
            dst.extend_from_slice(s.as_bytes());
        }
        Dynamic::Bytes(b) => {
            dst.push(TAG_BYTES);
            write_varint_u64(b.len() as u64, dst);
            dst.extend_from_slice(b);
        }
        Dynamic::Seq(items) => {
            dst.push(TAG_SEQ);
            write_varint_u64(items.len() as u64, dst);
            for item in items {
                encode_one_bounded(item, depth + 1, dst)?;
            }
        }
        Dynamic::Map(map) => {
            dst.push(TAG_MAP);
            write_varint_u64(map.len() as u64, dst);
            for (k, v) in map {
                write_varint_u64(k.len() as u64, dst);
                dst.extend_from_slice(k.as_bytes());
                encode_one_bounded(v, depth + 1, dst)?;
            }
        }
        Dynamic::Enum { variant, payload } => {
            dst.push(TAG_ENUM);
            write_varint_u64(variant.len() as u64, dst);
            dst.extend_from_slice(variant.as_bytes());
            encode_one_bounded(payload, depth + 1, dst)?;
        }
    }
    Ok(())
}

fn split_first(bytes: &[u8]) -> Result<(u8, &[u8])> {
    let (first, rest) = bytes
        .split_first()
        .ok_or(Error::Corruption { page_id: 0 })?;
    Ok((*first, rest))
}

fn take_n(bytes: &[u8], n: usize) -> Result<(&[u8], &[u8])> {
    if bytes.len() < n {
        return Err(Error::Corruption { page_id: 0 });
    }
    Ok(bytes.split_at(n))
}

fn take_len_prefixed(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let (len, rest) = take_varint_usize(bytes)?;
    let (data, after) = take_n(rest, len)?;
    Ok((data, after))
}

/// Read a LEB128-style unsigned varint (the postcard / protobuf
/// shape: 7 data bits per byte, MSB = continuation).
fn take_varint_usize(bytes: &[u8]) -> Result<(usize, &[u8])> {
    let (v, rest) = take_varint_u64(bytes)?;
    let v = usize::try_from(v).map_err(|_| Error::Corruption { page_id: 0 })?;
    Ok((v, rest))
}

const MAX_VARINT_BYTES: usize = 10;

fn take_varint_u64(bytes: &[u8]) -> Result<(u64, &[u8])> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut i: usize = 0;
    while i < bytes.len() && i < MAX_VARINT_BYTES {
        let b = bytes[i];
        let part = u64::from(b & 0x7F);
        let shifted = part
            .checked_shl(shift)
            .ok_or(Error::Corruption { page_id: 0 })?;
        value |= shifted;
        i += 1;
        if (b & 0x80) == 0 {
            return Ok((value, &bytes[i..]));
        }
        shift = shift
            .checked_add(7)
            .ok_or(Error::Corruption { page_id: 0 })?;
    }
    Err(Error::Corruption { page_id: 0 })
}

fn write_varint_u64(mut value: u64, dst: &mut Vec<u8>) {
    while value >= 0x80 {
        // allow: truncation is intentional — value is masked to 7 bits (& 0x7F) before the cast, so only the low byte is kept.
        #[allow(clippy::cast_possible_truncation)]
        let lo = (value & 0x7F) as u8;
        dst.push(lo | 0x80);
        value >>= 7;
    }
    // allow: truncation is intentional — the loop guarantees value < 0x80 here, so the cast to the final varint byte is lossless.
    #[allow(clippy::cast_possible_truncation)]
    let last = value as u8;
    dst.push(last);
}

fn write_varint_i64(value: i64, dst: &mut Vec<u8>) {
    let zz = ((value << 1) ^ (value >> 63)).cast_unsigned();
    write_varint_u64(zz, dst);
}

fn decode_bool(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (b, rest) = split_first(bytes)?;
    match b {
        0 => Ok((Dynamic::Bool(false), rest)),
        1 => Ok((Dynamic::Bool(true), rest)),
        _ => Err(Error::Corruption { page_id: 0 }),
    }
}

fn decode_u64(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (v, rest) = take_varint_u64(bytes)?;
    Ok((Dynamic::U64(v), rest))
}

fn decode_i64(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (zz, rest) = take_varint_u64(bytes)?;
    let high = (zz >> 1).cast_signed();
    let low = (zz & 1).cast_signed();
    let v = high ^ -low;
    Ok((Dynamic::I64(v), rest))
}

fn decode_f64(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (data, rest) = take_n(bytes, 8)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(data);
    Ok((Dynamic::F64(f64::from_le_bytes(buf)), rest))
}

fn decode_string(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (data, rest) = take_len_prefixed(bytes)?;
    let s = std::str::from_utf8(data)
        .map_err(|_| Error::Corruption { page_id: 0 })?
        .to_owned();
    Ok((Dynamic::String(s), rest))
}

fn decode_bytes(bytes: &[u8]) -> Result<(Dynamic, &[u8])> {
    let (data, rest) = take_len_prefixed(bytes)?;
    Ok((Dynamic::Bytes(data.to_vec()), rest))
}

impl Serialize for Dynamic {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            Dynamic::Null => ser.serialize_unit(),
            Dynamic::Bool(b) => ser.serialize_bool(*b),
            Dynamic::U64(n) => ser.serialize_u64(*n),
            Dynamic::I64(n) => ser.serialize_i64(*n),
            Dynamic::F64(f) => ser.serialize_f64(*f),
            Dynamic::String(s) => ser.serialize_str(s),
            Dynamic::Bytes(b) => ser.serialize_bytes(b),
            Dynamic::Seq(items) => {
                let mut s = ser.serialize_seq(Some(items.len()))?;
                for item in items {
                    s.serialize_element(item)?;
                }
                s.end()
            }
            Dynamic::Map(map) => {
                let mut m = ser.serialize_map(Some(map.len()))?;
                for (k, v) in map {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
            Dynamic::Enum { variant, payload } => {
                let mut m = ser.serialize_map(Some(1))?;
                m.serialize_entry(variant, payload.as_ref())?;
                m.end()
            }
        }
    }
}

impl From<bool> for Dynamic {
    fn from(b: bool) -> Self {
        Dynamic::Bool(b)
    }
}

impl From<u64> for Dynamic {
    fn from(n: u64) -> Self {
        Dynamic::U64(n)
    }
}

impl From<u32> for Dynamic {
    fn from(n: u32) -> Self {
        Dynamic::U64(u64::from(n))
    }
}

impl From<i64> for Dynamic {
    fn from(n: i64) -> Self {
        Dynamic::I64(n)
    }
}

impl From<i32> for Dynamic {
    fn from(n: i32) -> Self {
        Dynamic::I64(i64::from(n))
    }
}

impl From<f64> for Dynamic {
    fn from(f: f64) -> Self {
        Dynamic::F64(f)
    }
}

impl From<String> for Dynamic {
    fn from(s: String) -> Self {
        Dynamic::String(s)
    }
}

impl From<&str> for Dynamic {
    fn from(s: &str) -> Self {
        Dynamic::String(s.to_owned())
    }
}

impl From<Vec<u8>> for Dynamic {
    fn from(b: Vec<u8>) -> Self {
        Dynamic::Bytes(b)
    }
}

impl From<&[u8]> for Dynamic {
    fn from(b: &[u8]) -> Self {
        Dynamic::Bytes(b.to_vec())
    }
}

/// One pending composite frame in the schema walker's explicit
/// stack. Each frame describes a `Seq` or `Map` whose children are
/// still being decoded; once enough children are emitted the frame
/// is folded into the parent (or returned as the root value).
enum Frame {
    /// `Seq` decode in progress: `remaining` children to read,
    /// each shaped like `elem`; `acc` collects the decoded children
    /// in order.
    Seq {
        elem: DynamicSchema,
        remaining: usize,
        acc: Vec<Dynamic>,
    },
    /// `Map` decode in progress.
    ///
    /// `pending` is the FIFO of `(name, schema)` pairs whose values
    /// have NOT yet been decoded.  `current_name` is the name of
    /// the field whose value the walker is decoding RIGHT NOW —
    /// it has already been popped off `pending` by
    /// [`request_next_child`] but its value is not yet in `acc`.
    /// `acc` holds the already-completed fields. `path_prefix` is
    /// the dotted path leading up to (but not including) this map,
    /// reused for diagnostic mismatch errors.
    Map {
        pending: std::collections::VecDeque<(String, DynamicSchema)>,
        current_name: Option<String>,
        acc: BTreeMap<String, Dynamic>,
        path_prefix: String,
    },
    /// `Enum` decode in progress — the walker has read the
    /// discriminant + resolved the variant and is now waiting for
    /// the payload's `Dynamic` to fold back. Single-child shape:
    /// `pending_payload` carries the payload schema until
    /// [`request_next_child`] takes it; once the payload completes
    /// `fold_into` moves the decoded value into `acc` and the frame
    /// is popped + replaced with a [`Dynamic::Enum`] carrying the
    /// remembered `variant` name and `acc`'s payload.
    Enum {
        /// Variant name decoded from the schema; carried verbatim
        /// into the resulting [`Dynamic::Enum`].
        variant: String,
        /// Payload schema; `None` once
        /// [`request_next_child`] has handed it to the outer loop.
        pending_payload: Option<DynamicSchema>,
        /// The decoded payload `Dynamic`, populated by `fold_into`
        /// when the single child completes. `None` until then.
        acc: Option<Dynamic>,
        /// Dotted path leading up to (but not including) this enum,
        /// reused for diagnostic mismatch errors.
        #[allow(dead_code)]
        path_prefix: String,
    },
}

/// Walk `bytes` per `schema`. Returns the decoded `Dynamic` and any
/// trailing bytes the walker did not consume (typically empty for a
/// matched schema / payload pair).
fn walk_schema<'a>(bytes: &'a [u8], schema: &DynamicSchema) -> Result<(Dynamic, &'a [u8])> {
    let mut stack: Vec<Frame> = Vec::new();
    let mut rest = bytes;
    let mut next_schema: DynamicSchema = schema.clone();
    let mut next_path: String = String::new();
    let mut iters: usize = 0;
    let iter_cap = (1 + MAX_SCHEMA_DEPTH) * (1 + MAX_DYNAMIC_NODES);
    loop {
        iters = iters.checked_add(1).ok_or(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        })?;
        check_walk_schema_bounds(iters, iter_cap, stack.len())?;
        let outcome = decode_slot(rest, &next_schema, &next_path, &mut stack)?;
        rest = outcome.rest;
        match advance_after_slot(outcome.value, &mut stack, next_path)? {
            WalkStep::Complete(root) => return Ok((root, rest)),
            WalkStep::Continue {
                next_schema: ns,
                next_path: np,
            } => {
                next_schema = ns;
                next_path = np;
            }
        }
    }
}

/// Enforce the per-iteration bounds for [`walk_schema`]: the overall
/// iteration cap and the in-flight frame-stack depth limit
/// (mirrored from the tagged-format walker).
fn check_walk_schema_bounds(iters: usize, iter_cap: usize, stack_len: usize) -> Result<()> {
    if iters > iter_cap {
        return Err(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        });
    }
    if stack_len >= MAX_SCHEMA_DEPTH {
        return Err(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        });
    }
    Ok(())
}

/// Outcome of [`advance_after_slot`]: either the walker has produced
/// the root `Dynamic` and is done, or it has identified the next
/// schema slot to drive `decode_slot` against.
enum WalkStep {
    /// The fold collapsed every open frame; the carried value is the
    /// fully-decoded root.
    Complete(Dynamic),
    /// More frames remain; carry the next schema slot + diagnostic
    /// path back to the driver loop.
    Continue {
        next_schema: DynamicSchema,
        next_path: String,
    },
}

/// Dispatch on a `decode_slot` outcome: either continue with the next
/// child's schema/path, fold a completed value (possibly all the way
/// back to the root), or surface a corruption-shaped error if the
/// frame stack and the slot value disagree about whether more children
/// remain to be decoded.
fn advance_after_slot(
    slot_value: Option<Dynamic>,
    stack: &mut Vec<Frame>,
    current_path: String,
) -> Result<WalkStep> {
    match slot_value {
        None => {
            if let Some((next_schema, next_path)) = request_next_child(stack) {
                Ok(WalkStep::Continue {
                    next_schema,
                    next_path,
                })
            } else {
                Err(Error::SchemaTypeMismatch {
                    expected: "non-empty",
                    found: "empty-frame",
                    path: current_path,
                })
            }
        }
        Some(value) => {
            if let Some(root) = fold_value(value, stack) {
                return Ok(WalkStep::Complete(root));
            }
            let Some((next_schema, next_path)) = request_next_child(stack) else {
                return Err(Error::SchemaTypeMismatch {
                    expected: "next-child",
                    found: "exhausted-frame",
                    path: current_path,
                });
            };
            Ok(WalkStep::Continue {
                next_schema,
                next_path,
            })
        }
    }
}

/// Result of decoding one schema slot.
struct SlotOutcome<'a> {
    /// `Some(value)` if the slot produced a complete `Dynamic`
    /// inline (scalars + empty composites); `None` if a new frame
    /// was pushed onto the stack and the outer driver should
    /// continue with the new frame's first child.
    value: Option<Dynamic>,
    /// Remaining input bytes after the slot's bytes were consumed.
    rest: &'a [u8],
}

/// Decode a single schema slot.
fn decode_slot<'a>(
    bytes: &'a [u8],
    schema: &DynamicSchema,
    path: &str,
    stack: &mut Vec<Frame>,
) -> Result<SlotOutcome<'a>> {
    match schema {
        DynamicSchema::Null => Ok(SlotOutcome {
            value: Some(Dynamic::Null),
            rest: bytes,
        }),
        DynamicSchema::Bool => {
            let (v, rest) = decode_bool_schema(bytes, path)?;
            Ok(SlotOutcome {
                value: Some(v),
                rest,
            })
        }
        DynamicSchema::U64 => {
            let (n, rest) = take_varint_u64(bytes)?;
            Ok(SlotOutcome {
                value: Some(Dynamic::U64(n)),
                rest,
            })
        }
        DynamicSchema::I64 => {
            let (zz, rest) = take_varint_u64(bytes)?;
            Ok(SlotOutcome {
                value: Some(Dynamic::I64(zigzag_decode_i64(zz))),
                rest,
            })
        }
        DynamicSchema::F64 => {
            let (v, rest) = decode_f64_schema(bytes, path)?;
            Ok(SlotOutcome {
                value: Some(v),
                rest,
            })
        }
        DynamicSchema::String => {
            let (v, rest) = decode_string_schema(bytes, path)?;
            Ok(SlotOutcome {
                value: Some(v),
                rest,
            })
        }
        DynamicSchema::Bytes => {
            let (v, rest) = decode_bytes_schema(bytes, path)?;
            Ok(SlotOutcome {
                value: Some(v),
                rest,
            })
        }
        DynamicSchema::Seq(elem) => decode_seq_slot(bytes, elem, stack),
        DynamicSchema::Map(fields) => Ok(decode_map_slot(bytes, fields, path, stack)),
        DynamicSchema::Enum(variants) => decode_enum_slot(bytes, variants, path, stack),
    }
}

/// Decode a `Bool` schema slot.
fn decode_bool_schema<'a>(bytes: &'a [u8], path: &str) -> Result<(Dynamic, &'a [u8])> {
    let (b, rest) = bytes
        .split_first()
        .ok_or_else(|| Error::SchemaTypeMismatch {
            expected: "Bool",
            found: "truncated",
            path: path.to_owned(),
        })?;
    match *b {
        0 => Ok((Dynamic::Bool(false), rest)),
        1 => Ok((Dynamic::Bool(true), rest)),
        _ => Err(Error::SchemaTypeMismatch {
            expected: "Bool",
            found: "non-bool-byte",
            path: path.to_owned(),
        }),
    }
}

/// Decode an `F64` schema slot (8 LE bytes).
fn decode_f64_schema<'a>(bytes: &'a [u8], path: &str) -> Result<(Dynamic, &'a [u8])> {
    if bytes.len() < 8 {
        return Err(Error::SchemaTypeMismatch {
            expected: "F64",
            found: "truncated",
            path: path.to_owned(),
        });
    }
    let (data, rest) = bytes.split_at(8);
    let mut buf = [0u8; 8];
    buf.copy_from_slice(data);
    Ok((Dynamic::F64(f64::from_le_bytes(buf)), rest))
}

/// Decode a `String` schema slot (varint length + UTF-8 bytes).
fn decode_string_schema<'a>(bytes: &'a [u8], path: &str) -> Result<(Dynamic, &'a [u8])> {
    let (len, rest) = take_varint_usize(bytes)?;
    if rest.len() < len {
        return Err(Error::SchemaTypeMismatch {
            expected: "String",
            found: "truncated",
            path: path.to_owned(),
        });
    }
    let (data, after) = rest.split_at(len);
    let s = std::str::from_utf8(data)
        .map_err(|_| Error::SchemaTypeMismatch {
            expected: "String",
            found: "non-utf8",
            path: path.to_owned(),
        })?
        .to_owned();
    Ok((Dynamic::String(s), after))
}

/// Decode a `Bytes` schema slot (varint length + raw bytes).
fn decode_bytes_schema<'a>(bytes: &'a [u8], path: &str) -> Result<(Dynamic, &'a [u8])> {
    let (len, rest) = take_varint_usize(bytes)?;
    if rest.len() < len {
        return Err(Error::SchemaTypeMismatch {
            expected: "Bytes",
            found: "truncated",
            path: path.to_owned(),
        });
    }
    let (data, after) = rest.split_at(len);
    Ok((Dynamic::Bytes(data.to_vec()), after))
}

/// Begin a `Seq` schema slot. Reads the varint length; if zero,
/// returns `Some(Dynamic::Seq(vec![]))` directly. Otherwise pushes
/// a `Frame::Seq` and returns `None` so the outer loop walks the
/// first element. The frame's accumulator is pre-sized to at most
/// [`SEQ_PREALLOC`] (not the declared `len`) so a forged nested-`Seq`
/// payload cannot amplify a tiny input into a multi-megabyte
/// reservation; the `Vec` grows on push as elements are decoded.
fn decode_seq_slot<'a>(
    bytes: &'a [u8],
    elem: &DynamicSchema,
    stack: &mut Vec<Frame>,
) -> Result<SlotOutcome<'a>> {
    let (len, rest) = take_varint_usize(bytes)?;
    if len == 0 {
        return Ok(SlotOutcome {
            value: Some(Dynamic::Seq(Vec::new())),
            rest,
        });
    }
    // Use `>=` for consistency with the tagged-format walker: the schema
    // walker's iteration cap already bounds total nodes, but keeping the
    // same threshold avoids accepting lengths that would be rejected by
    // the tagged-format round-trip path.
    if len >= MAX_DYNAMIC_NODES {
        return Err(Error::SchemaTypeMismatch {
            expected: "Seq",
            found: "length-exceeds-bound",
            path: String::new(),
        });
    }
    // Reserve only a small constant up front, not the declared `len`:
    // `len` is attacker-controlled and a deeply-nested `Seq(Seq(..))`
    // schema would otherwise open `MAX_SCHEMA_DEPTH` frames each
    // eagerly reserving `MAX_DYNAMIC_NODES` slots (~80 MB) before any
    // element is decoded. The `acc` `Vec` grows on push, so real memory
    // tracks the genuinely decoded element count. See [`SEQ_PREALLOC`].
    stack.push(Frame::Seq {
        elem: elem.clone(),
        remaining: len,
        acc: Vec::with_capacity(len.min(SEQ_PREALLOC)),
    });
    Ok(SlotOutcome { value: None, rest })
}

/// Begin a `Map` schema slot. Postcard emits **no length prefix**
/// for a struct's fields — the schema names every field in order.
/// Empty `Map` short-circuits with an empty `Dynamic::Map`;
/// otherwise a `Frame::Map` is pushed.
fn decode_map_slot<'a>(
    bytes: &'a [u8],
    fields: &[(String, DynamicSchema)],
    path: &str,
    stack: &mut Vec<Frame>,
) -> SlotOutcome<'a> {
    if fields.is_empty() {
        return SlotOutcome {
            value: Some(Dynamic::Map(BTreeMap::new())),
            rest: bytes,
        };
    }
    let pending: std::collections::VecDeque<(String, DynamicSchema)> =
        fields.iter().cloned().collect();
    stack.push(Frame::Map {
        pending,
        current_name: None,
        acc: BTreeMap::new(),
        path_prefix: path.to_owned(),
    });
    SlotOutcome {
        value: None,
        rest: bytes,
    }
}

/// Begin an `Enum` schema slot. Reads the varint `u32` discriminant,
/// binary-searches `variants` for the match, and pushes a
/// `Frame::Enum` carrying the variant name + payload schema. The
/// schema's `variants` MUST be sorted ascending by discriminant —
/// debug-asserted on first decode. A missing discriminant returns
/// [`Error::SchemaTypeMismatch`].
fn decode_enum_slot<'a>(
    bytes: &'a [u8],
    variants: &[EnumVariantSchema],
    path: &str,
    stack: &mut Vec<Frame>,
) -> Result<SlotOutcome<'a>> {
    debug_assert!(
        variants
            .windows(2)
            .all(|w| w[0].discriminant < w[1].discriminant),
        "Enum schema variants must be strictly ascending by discriminant",
    );
    debug_assert!(
        !variants.is_empty(),
        "Enum schema must declare at least one variant",
    );
    let (disc_u64, rest) = take_varint_u64(bytes)?;
    let disc = u32::try_from(disc_u64).map_err(|_| Error::SchemaTypeMismatch {
        expected: "u32-discriminant",
        found: "varint-overflow",
        path: path.to_owned(),
    })?;
    let idx = variants
        .binary_search_by(|v| v.discriminant.cmp(&disc))
        .map_err(|_| Error::SchemaTypeMismatch {
            expected: "known variant",
            found: "unknown-discriminant",
            path: path.to_owned(),
        })?;
    let variant = &variants[idx];
    stack.push(Frame::Enum {
        variant: variant.name.clone(),
        pending_payload: Some(variant.payload.as_ref().clone()),
        acc: None,
        path_prefix: path.to_owned(),
    });
    Ok(SlotOutcome { value: None, rest })
}

/// Fold a freshly-decoded `value` into the top frame's accumulator.
/// May cascade pops up the stack if `value` completes one or more
/// composite frames. Returns `Some(root)` when the stack empties
/// (the walk is finished), `None` otherwise.
fn fold_value(mut value: Dynamic, stack: &mut Vec<Frame>) -> Option<Dynamic> {
    let mut iters: usize = 0;
    loop {
        iters = iters.saturating_add(1);
        debug_assert!(
            iters <= MAX_SCHEMA_DEPTH + 1,
            "fold cascade exceeded MAX_SCHEMA_DEPTH",
        );
        let Some(top) = stack.last_mut() else {
            return Some(value);
        };
        if !fold_into(top, &mut value) {
            return None;
        }
        let completed = match stack.pop() {
            Some(Frame::Seq { acc, .. }) => Dynamic::Seq(acc),
            Some(Frame::Map { acc, .. }) => Dynamic::Map(acc),
            Some(Frame::Enum {
                variant,
                acc: payload,
                ..
            }) => {
                debug_assert!(
                    payload.is_some(),
                    "Enum frame popped before fold_into populated acc",
                );
                Dynamic::Enum {
                    variant,
                    payload: Box::new(payload.unwrap_or(Dynamic::Null)),
                }
            }
            None => return Some(value),
        };
        value = completed;
    }
}

/// Push `value` into `top`. Returns `true` if `top` is now complete
/// (caller must pop + cascade); `false` if the frame still has
/// children pending.
fn fold_into(top: &mut Frame, value: &mut Dynamic) -> bool {
    match top {
        Frame::Seq { remaining, acc, .. } => {
            acc.push(std::mem::replace(value, Dynamic::Null));
            *remaining = remaining.saturating_sub(1);
            *remaining == 0
        }
        Frame::Map {
            pending,
            current_name,
            acc,
            ..
        } => {
            let name = current_name
                .take()
                .unwrap_or_else(|| String::from("<bug:missing-current-name>"));
            debug_assert!(
                !name.starts_with("<bug:"),
                "Map frame missing current_name in fold",
            );
            acc.insert(name, std::mem::replace(value, Dynamic::Null));
            pending.is_empty()
        }
        Frame::Enum { acc, .. } => {
            debug_assert!(acc.is_none(), "Enum frame folded twice");
            *acc = Some(std::mem::replace(value, Dynamic::Null));
            true
        }
    }
}

/// Ask the top frame for the schema (and dotted path) of the next
/// child slot to decode.  `Seq` frames return the element schema
/// unchanged; `Map` frames pop the next pending field off and
/// remember its name in `current_name` for the eventual fold.
fn request_next_child(stack: &mut [Frame]) -> Option<(DynamicSchema, String)> {
    let top = stack.last_mut()?;
    match top {
        Frame::Seq { elem, .. } => Some((elem.clone(), String::new())),
        Frame::Map {
            pending,
            current_name,
            path_prefix,
            ..
        } => {
            let (name, schema) = pending.pop_front()?;
            let path = if path_prefix.is_empty() {
                name.clone()
            } else {
                format!("{path_prefix}.{name}")
            };
            *current_name = Some(name);
            Some((schema, path))
        }
        Frame::Enum {
            variant,
            pending_payload,
            path_prefix,
            ..
        } => {
            let schema = pending_payload.take()?;
            let path = if path_prefix.is_empty() {
                variant.clone()
            } else {
                format!("{path_prefix}.{variant}")
            };
            Some((schema, path))
        }
    }
}

/// Static name of `Dynamic`'s variant. Used by [`Dynamic::get_str`]
/// for diagnostic mismatch errors.
fn variant_name(d: &Dynamic) -> &'static str {
    match d {
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

/// Zigzag-decode a u64 into i64. Mirrors the encoding postcard uses
/// for signed varints.
fn zigzag_decode_i64(zz: u64) -> i64 {
    let high = (zz >> 1).cast_signed();
    let low = (zz & 1).cast_signed();
    high ^ -low
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[test]
    fn round_trip_scalar() {
        for value in [
            Dynamic::Null,
            Dynamic::Bool(true),
            Dynamic::Bool(false),
            Dynamic::U64(0),
            Dynamic::U64(u64::MAX),
            Dynamic::I64(0),
            Dynamic::I64(-1),
            Dynamic::I64(i64::MIN),
            Dynamic::I64(i64::MAX),
            Dynamic::F64(1.5),
            Dynamic::String("hello".to_owned()),
            Dynamic::Bytes(vec![1, 2, 3]),
        ] {
            let bytes = value.to_postcard_bytes().expect("encode");
            let decoded = Dynamic::from_tagged_bytes(&bytes).expect("decode");
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn round_trip_nested_map() {
        let mut inner = BTreeMap::new();
        inner.insert("a".to_owned(), Dynamic::U64(1));
        inner.insert("b".to_owned(), Dynamic::String("two".to_owned()));
        let mut outer = BTreeMap::new();
        outer.insert("inner".to_owned(), Dynamic::Map(inner));
        outer.insert(
            "list".to_owned(),
            Dynamic::Seq(vec![
                Dynamic::U64(10),
                Dynamic::U64(20),
                Dynamic::Bool(false),
            ]),
        );
        let value = Dynamic::Map(outer);
        let bytes = value.to_postcard_bytes().expect("encode");
        let decoded = Dynamic::from_tagged_bytes(&bytes).expect("decode");
        assert_eq!(decoded, value);
    }

    #[test]
    fn unknown_tag_is_corruption() {
        let err = Dynamic::from_tagged_bytes(&[0xFF]).expect_err("unknown tag");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn truncated_tagged_string_is_corruption() {
        let bytes = [TAG_STRING, 100, b'x'];
        let err = Dynamic::from_tagged_bytes(&bytes).expect_err("truncated");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn get_and_set_on_map() {
        let mut m = BTreeMap::new();
        m.insert("a".to_owned(), Dynamic::U64(1));
        let mut value = Dynamic::Map(m);
        assert_eq!(value.get("a"), Some(&Dynamic::U64(1)));
        assert_eq!(value.get("missing"), None);
        value.set("b", Dynamic::Bool(true));
        assert_eq!(value.get("b"), Some(&Dynamic::Bool(true)));
    }

    #[test]
    fn remove_from_map_returns_removed() {
        let mut m = BTreeMap::new();
        m.insert("a".to_owned(), Dynamic::U64(1));
        m.insert("b".to_owned(), Dynamic::String("two".to_owned()));
        let mut value = Dynamic::Map(m);
        let removed = value.remove("a").expect("ok");
        assert_eq!(removed, Some(Dynamic::U64(1)));
        assert_eq!(value.get("a"), None);
        assert_eq!(value.get("b"), Some(&Dynamic::String("two".to_owned())));
        let again = value.remove("a").expect("ok");
        assert_eq!(again, None);
    }

    #[test]
    fn remove_on_non_map_errors() {
        let mut value = Dynamic::U64(5);
        let err = value.remove("k").expect_err("non-map");
        assert!(matches!(err, Error::DynamicPathNotMap { .. }));
    }

    #[test]
    fn set_on_non_map_replaces() {
        let mut value = Dynamic::U64(5);
        value.set("k", Dynamic::String("v".to_owned()));
        match value {
            Dynamic::Map(m) => {
                assert_eq!(m.len(), 1);
                assert_eq!(m.get("k"), Some(&Dynamic::String("v".to_owned())));
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }

    /// Reproduces the migration shape: encode a v1 doc through
    /// postcard, build a `Dynamic` view of its fields, set the new
    /// v2 field with a default, then construct the v2 struct from
    /// the populated `Dynamic`. This is the pattern every
    /// hand-written `Migrate::migrate` impl will follow.
    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct V1 {
        a: u32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct V2 {
        a: u32,
        b: String,
    }

    #[test]
    fn dynamic_migration_shape() {
        let v1 = V1 { a: 42 };
        let v1_postcard = postcard::to_allocvec(&v1).expect("encode v1");

        let recovered_v1: V1 = postcard::from_bytes(&v1_postcard).expect("decode v1");
        let mut dynamic = Dynamic::Map(BTreeMap::new());
        dynamic.set("a", Dynamic::U64(u64::from(recovered_v1.a)));

        dynamic.set("b", Dynamic::String("default-b".to_owned()));

        let a = match dynamic.get("a") {
            Some(Dynamic::U64(n)) => u32::try_from(*n).expect("u32 range"),
            other => panic!("missing or wrong-type a: {other:?}"),
        };
        let b = match dynamic.get("b") {
            Some(Dynamic::String(s)) => s.clone(),
            other => panic!("missing or wrong-type b: {other:?}"),
        };
        let v2 = V2 { a, b };
        assert_eq!(
            v2,
            V2 {
                a: 42,
                b: "default-b".to_owned(),
            }
        );
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct SchemaPair {
        name: String,
        age: u32,
    }

    #[test]
    fn schema_walker_decodes_simple_struct() {
        let s = SchemaPair {
            name: "ada".to_owned(),
            age: 36,
        };
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let schema =
            DynamicSchema::map([("name", DynamicSchema::String), ("age", DynamicSchema::U64)]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(
            dyn_view.get("name"),
            Some(&Dynamic::String("ada".to_owned())),
        );
        assert_eq!(dyn_view.get("age"), Some(&Dynamic::U64(36)));
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct SchemaInner {
        x: u32,
        y: u32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct SchemaOuter {
        tag: String,
        inner: SchemaInner,
        tail: bool,
    }

    #[test]
    fn schema_walker_decodes_nested_struct() {
        let s = SchemaOuter {
            tag: "hello".to_owned(),
            inner: SchemaInner { x: 1, y: 2 },
            tail: true,
        };
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let schema = DynamicSchema::map([
            ("tag", DynamicSchema::String),
            (
                "inner",
                DynamicSchema::map([("x", DynamicSchema::U64), ("y", DynamicSchema::U64)]),
            ),
            ("tail", DynamicSchema::Bool),
        ]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(
            dyn_view.get("tag"),
            Some(&Dynamic::String("hello".to_owned())),
        );
        let inner = dyn_view.get("inner").expect("inner present");
        assert_eq!(inner.get("x"), Some(&Dynamic::U64(1)));
        assert_eq!(inner.get("y"), Some(&Dynamic::U64(2)));
        assert_eq!(dyn_view.get("tail"), Some(&Dynamic::Bool(true)));
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct SchemaWithSeq {
        items: Vec<u32>,
        name: String,
    }

    #[test]
    fn schema_walker_decodes_seq() {
        let s = SchemaWithSeq {
            items: vec![10, 20, 30],
            name: "vec".to_owned(),
        };
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let schema = DynamicSchema::map([
            ("items", DynamicSchema::seq(DynamicSchema::U64)),
            ("name", DynamicSchema::String),
        ]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        match dyn_view.get("items") {
            Some(Dynamic::Seq(items)) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Dynamic::U64(10));
                assert_eq!(items[1], Dynamic::U64(20));
                assert_eq!(items[2], Dynamic::U64(30));
            }
            other => panic!("expected Seq, got {other:?}"),
        }
        assert_eq!(
            dyn_view.get("name"),
            Some(&Dynamic::String("vec".to_owned())),
        );
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct SchemaSigned {
        neg: i32,
        pos: i32,
    }

    #[test]
    fn schema_walker_decodes_signed_ints() {
        let s = SchemaSigned { neg: -5, pos: 7 };
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let schema = DynamicSchema::map([("neg", DynamicSchema::I64), ("pos", DynamicSchema::I64)]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(dyn_view.get("neg"), Some(&Dynamic::I64(-5)));
        assert_eq!(dyn_view.get("pos"), Some(&Dynamic::I64(7)));
    }

    #[test]
    fn schema_walker_decodes_bytes_and_unit() {
        #[derive(Serialize, Deserialize)]
        struct B {
            blob: Vec<u8>,
            nothing: (),
            present: bool,
        }
        let v = B {
            blob: vec![1, 2, 3, 4],
            nothing: (),
            present: false,
        };
        let bytes = postcard::to_allocvec(&v).expect("encode");
        let schema = DynamicSchema::map([
            ("blob", DynamicSchema::Bytes),
            ("nothing", DynamicSchema::Null),
            ("present", DynamicSchema::Bool),
        ]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        match dyn_view.get("blob") {
            Some(Dynamic::Bytes(b)) => assert_eq!(b, &vec![1, 2, 3, 4]),
            other => panic!("expected Bytes, got {other:?}"),
        }
        assert_eq!(dyn_view.get("nothing"), Some(&Dynamic::Null));
        assert_eq!(dyn_view.get("present"), Some(&Dynamic::Bool(false)));
    }

    #[test]
    fn schema_walker_round_trip_to_native_deserialize() {
        let s = SchemaPair {
            name: "ada".to_owned(),
            age: 36,
        };
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let direct: SchemaPair = postcard::from_bytes(&bytes).expect("postcard");
        let schema =
            DynamicSchema::map([("name", DynamicSchema::String), ("age", DynamicSchema::U64)]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(
            dyn_view.get("name"),
            Some(&Dynamic::String(direct.name.clone())),
        );
        assert_eq!(
            dyn_view.get("age"),
            Some(&Dynamic::U64(u64::from(direct.age)))
        );
    }

    #[test]
    fn schema_walker_rejects_truncated_payload() {
        let s = SchemaPair {
            name: "ada".to_owned(),
            age: 36,
        };
        let mut bytes = postcard::to_allocvec(&s).expect("encode");
        bytes.truncate(2);
        let schema =
            DynamicSchema::map([("name", DynamicSchema::String), ("age", DynamicSchema::U64)]);
        let err = Dynamic::from_postcard_bytes(&bytes, &schema).expect_err("truncated");
        assert!(matches!(err, Error::SchemaTypeMismatch { .. }));
    }

    #[test]
    fn schema_walker_rejects_excess_depth() {
        let mut s = DynamicSchema::U64;
        for _ in 0..(MAX_SCHEMA_DEPTH + 2) {
            s = DynamicSchema::seq(s);
        }
        let bytes = vec![1u8; MAX_SCHEMA_DEPTH + 16];
        let err = Dynamic::from_postcard_bytes(&bytes, &s).expect_err("depth");
        assert!(matches!(err, Error::SchemaDepthExceeded { .. }));
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    enum SchemaEnumProbe {
        Pending,
        Shipped { tracking: String },
        Cancelled(String),
    }

    fn schema_enum_probe_schema() -> DynamicSchema {
        DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "Pending", DynamicSchema::Null),
            EnumVariantSchema::new(
                1,
                "Shipped",
                DynamicSchema::map([("tracking", DynamicSchema::String)]),
            ),
            EnumVariantSchema::new(2, "Cancelled", DynamicSchema::String),
        ])
    }

    #[test]
    fn schema_walker_decodes_unit_variant() {
        let bytes = postcard::to_allocvec(&SchemaEnumProbe::Pending).expect("encode");
        let schema = schema_enum_probe_schema();
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(dyn_view.enum_variant(), Some("Pending"));
        assert_eq!(dyn_view.enum_payload(), Some(&Dynamic::Null));
    }

    #[test]
    fn schema_walker_decodes_struct_variant() {
        let v = SchemaEnumProbe::Shipped {
            tracking: "ABC".to_owned(),
        };
        let bytes = postcard::to_allocvec(&v).expect("encode");
        let dyn_view =
            Dynamic::from_postcard_bytes(&bytes, &schema_enum_probe_schema()).expect("walk");
        assert_eq!(dyn_view.enum_variant(), Some("Shipped"));
        let payload = dyn_view.enum_payload().expect("payload");
        assert_eq!(
            payload.get("tracking"),
            Some(&Dynamic::String("ABC".to_owned())),
        );
    }

    #[test]
    fn schema_walker_decodes_newtype_variant() {
        let v = SchemaEnumProbe::Cancelled("late".to_owned());
        let bytes = postcard::to_allocvec(&v).expect("encode");
        let dyn_view =
            Dynamic::from_postcard_bytes(&bytes, &schema_enum_probe_schema()).expect("walk");
        assert_eq!(dyn_view.enum_variant(), Some("Cancelled"));
        assert_eq!(
            dyn_view.enum_payload(),
            Some(&Dynamic::String("late".to_owned())),
        );
    }

    #[test]
    fn schema_walker_unknown_discriminant_errors() {
        let bytes = [99u8];
        let err =
            Dynamic::from_postcard_bytes(&bytes, &schema_enum_probe_schema()).expect_err("unknown");
        assert!(matches!(
            err,
            Error::SchemaTypeMismatch {
                expected: "known variant",
                ..
            }
        ));
    }

    #[test]
    fn schema_walker_decodes_enum_inside_map() {
        #[derive(Serialize, Deserialize)]
        struct Wrap {
            label: String,
            status: SchemaEnumProbe,
        }
        let v = Wrap {
            label: "order".to_owned(),
            status: SchemaEnumProbe::Shipped {
                tracking: "XYZ".to_owned(),
            },
        };
        let bytes = postcard::to_allocvec(&v).expect("encode");
        let schema = DynamicSchema::map([
            ("label", DynamicSchema::String),
            ("status", schema_enum_probe_schema()),
        ]);
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        assert_eq!(
            dyn_view.get("label"),
            Some(&Dynamic::String("order".to_owned())),
        );
        let status = dyn_view.get("status").expect("status");
        assert_eq!(status.enum_variant(), Some("Shipped"));
        let payload = status.enum_payload().expect("payload");
        assert_eq!(
            payload.get("tracking"),
            Some(&Dynamic::String("XYZ".to_owned())),
        );
    }

    #[test]
    fn enum_tagged_round_trip() {
        let value = Dynamic::Enum {
            variant: "Shipped".to_owned(),
            payload: Box::new(Dynamic::Map({
                let mut m = BTreeMap::new();
                m.insert("tracking".to_owned(), Dynamic::String("ABC".to_owned()));
                m
            })),
        };
        let bytes = value.to_postcard_bytes().expect("encode");
        let back = Dynamic::from_tagged_bytes(&bytes).expect("decode");
        assert_eq!(back, value);
    }

    #[test]
    fn dynamic_tagged_round_trip_matches_intent() {
        let value = Dynamic::Seq(vec![
            Dynamic::Null,
            Dynamic::Bool(true),
            Dynamic::String("nested".to_owned()),
            Dynamic::Map({
                let mut m = BTreeMap::new();
                m.insert("k".to_owned(), Dynamic::I64(-7));
                m
            }),
        ]);
        let bytes = value.to_postcard_bytes().expect("encode");
        let back = Dynamic::from_tagged_bytes(&bytes).expect("decode");
        assert_eq!(back, value);
    }
}
