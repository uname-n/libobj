//! [`DynamicSchema`] — declarative description of a postcard-encoded
//! payload's shape.
//!
//! Postcard is **not self-describing**: a postcard payload is a flat
//! byte stream whose meaning depends entirely on the type that
//! produced it. To migrate a v1 document into a v2 type at decode
//! time the codec needs **some** description of the v1 wire shape;
//! `DynamicSchema` is that description.
//!
//! A `DynamicSchema` is pure data — a tree of enum variants, no
//! generics, no lifetimes beyond `'static`, no allocations beyond
//! the `Vec`/`Box` literal data carries. The schema describes the
//! field order + per-field type of a postcard payload; the walker
//! [`Dynamic::from_postcard_bytes`](crate::codec::Dynamic::from_postcard_bytes)
//! consumes both schema and bytes and produces a structured
//! [`Dynamic`](crate::codec::Dynamic) view.
//!
//! # Wire-format dependency
//!
//! `DynamicSchema` only describes postcard payloads. The mapping of
//! schema variants → postcard byte-stream operations is the **only**
//! postcard-specific knowledge in obj's migration path:
//!
//! | Schema variant       | Postcard wire shape                                        |
//! |----------------------|------------------------------------------------------------|
//! | `Null`               | (zero bytes — `()` / unit struct)                          |
//! | `Bool`               | 1 byte (`0` = false, `1` = true)                           |
//! | `U64`                | unsigned LEB128 varint                                     |
//! | `I64`                | zigzag-encoded varint                                      |
//! | `F64`                | 8 little-endian bytes                                      |
//! | `String`             | varint length + UTF-8 bytes                                |
//! | `Bytes`              | varint length + raw bytes                                  |
//! | `Seq(elem)`          | varint length, then N elements of the inner schema         |
//! | `Map(fields)`        | each `(name, schema)` decoded in order — *no* field names  |
//! | `Enum(variants)`     | varint `u32` discriminant, then the matched variant's payload |
//!
//! Postcard treats a Rust struct as a **field-ordered tuple** with
//! no names, so `Map` in this schema does NOT correspond to
//! postcard's `serialize_map`; it corresponds to a sequence of
//! per-field bytes. Field names are an obj-side convention used to
//! tag fields in the resulting `Dynamic` for `get` / `set` / `remove`
//! addressing.
//!
//! For enums, postcard writes a varint `u32` discriminant followed
//! by the matched variant's payload. Unit variants carry **only**
//! the discriminant; newtype / tuple / struct variants follow with
//! the inner type's field-ordered bytes (no length prefix — the
//! schema names every field for `Dynamic::get` addressing the same
//! way `Map` does). Tuple variants are represented by `Map` with
//! synthetic numeric field names (`"0"`, `"1"`, …) — there is no
//! `Tuple` schema variant.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Maximum nesting depth of a [`DynamicSchema`] tree that the walker
/// will traverse. Exceeding this bound returns
/// [`Error::SchemaDepthExceeded`](crate::error::Error::SchemaDepthExceeded).
///
/// 32 matches [`crate::codec::dynamic::MAX_DYNAMIC_DEPTH`] so the
/// schema walker cannot produce a deeper `Dynamic` than the tagged-
/// format walker can re-encode.
pub const MAX_SCHEMA_DEPTH: usize = 32;

/// Describes the byte-stream shape of a postcard-encoded payload at
/// one version. See module docs for the variant ↔ wire-format
/// mapping.
///
/// `DynamicSchema` is `'static`-friendly: literal schemas can live in
/// const-fn-adjacent contexts (e.g. a `LazyLock` registry) without
/// borrowing.
///
/// `Box`ing the inner schema in `Seq` keeps the enum's `size_of`
/// reasonable (the alternative — `Vec<DynamicSchema>` of length 1 —
/// is overkill for the single-element case and reads worse).
///
/// # Serde / disk-format invariant — variant order is FROZEN
///
/// `DynamicSchema` derives `Serialize`/`Deserialize` so it can be
/// stored verbatim inside a
/// [`StoredSchema`](crate::codec::stored_schema::StoredSchema) row in
/// the on-disk schema catalog. postcard
/// encodes an enum as a **varint discriminant equal to the variant's
/// declaration index**, so the order of the variants below IS the
/// wire format.
///
/// **The variant declaration order is therefore APPEND-ONLY.** New
/// variants may only be added at the END of the list; existing
/// variants must never be reordered, removed, or have a new variant
/// inserted between them. Violating this silently re-numbers every
/// later variant's discriminant and mis-decodes every catalog row
/// written by an older binary. `#[non_exhaustive]` keeps adding a
/// trailing variant a non-breaking change for in-tree matchers; it
/// does not relax the order freeze.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DynamicSchema {
    /// Unit / zero-byte field. Postcard emits no bytes for `()`.
    Null,
    /// Boolean — single byte `0` / `1`.
    Bool,
    /// Unsigned 64-bit integer (varint).
    U64,
    /// Signed 64-bit integer (zigzag varint).
    I64,
    /// 64-bit IEEE-754 float (8 LE bytes).
    F64,
    /// UTF-8 string (varint length + bytes).
    String,
    /// Raw byte sequence (varint length + bytes).
    Bytes,
    /// Variable-length sequence of `elem`-shaped values.
    Seq(Box<DynamicSchema>),
    /// Postcard-encoded struct described as an **ordered** list of
    /// `(field_name, field_schema)` pairs. Order MUST match the
    /// Rust declaration order of the struct that wrote the bytes —
    /// postcard reads fields positionally, so a transposed schema
    /// will mis-decode the payload as silently as any out-of-order
    /// tuple destructure.
    Map(Vec<(String, DynamicSchema)>),
    /// Postcard-encoded enum: a varint `u32` discriminant followed by
    /// the matched variant's payload bytes. `variants` MUST be sorted
    /// strictly ascending by [`EnumVariantSchema::discriminant`]; the
    /// walker binary-searches the list and a missing discriminant
    /// surfaces as [`Error::SchemaTypeMismatch`](crate::error::Error::SchemaTypeMismatch).
    ///
    /// Unit variants set `payload = DynamicSchema::Null`. Newtype
    /// variants set `payload` to the inner type's schema. Tuple and
    /// struct variants set `payload = DynamicSchema::Map(...)` with
    /// the variant's fields in declaration order; for tuple variants
    /// the field names are the synthetic strings `"0"`, `"1"`, …
    /// (postcard writes tuple variants positionally — the same wire
    /// shape as a Rust struct's bytes).
    Enum(Vec<EnumVariantSchema>),
}

/// One variant of a [`DynamicSchema::Enum`] description.
///
/// `discriminant` is the varint `u32` postcard writes for this
/// variant (postcard uses the Rust enum's declaration order, so the
/// first variant has discriminant `0`, the second `1`, and so on —
/// `#[serde(other)]` / explicit discriminants change this). `name`
/// is the Rust variant identifier carried through to the decoded
/// [`Dynamic::Enum`](crate::codec::Dynamic) so a `Migrate::migrate`
/// impl can distinguish variants by name. `payload` is the variant's
/// inner shape, `Box`ed because `DynamicSchema` is self-referential.
///
/// Derives `Serialize`/`Deserialize` so it round-trips inside a
/// stored [`DynamicSchema::Enum`] (see the disk-format invariant on
/// [`DynamicSchema`]). `#[non_exhaustive]` does not block the derive —
/// the struct is pure data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EnumVariantSchema {
    /// Postcard's varint `u32` discriminant for this variant.
    pub discriminant: u32,
    /// The Rust variant identifier, carried into
    /// [`Dynamic::Enum`](crate::codec::Dynamic) verbatim.
    pub name: String,
    /// Wire shape of the variant's payload. Use
    /// [`DynamicSchema::Null`] for unit variants;
    /// [`DynamicSchema::Map`] for tuple and struct variants
    /// (synthetic numeric names for tuple variants); the inner
    /// type's schema for newtype variants.
    pub payload: Box<DynamicSchema>,
}

impl EnumVariantSchema {
    /// Convenience constructor: lift a `(discriminant, name, payload)`
    /// triple into an [`EnumVariantSchema`]. The `Box` around the
    /// payload is added internally.
    #[must_use]
    pub fn new<S: Into<String>>(discriminant: u32, name: S, payload: DynamicSchema) -> Self {
        Self {
            discriminant,
            name: name.into(),
            payload: Box::new(payload),
        }
    }
}

impl DynamicSchema {
    /// Convenience constructor for a `Seq` schema with `elem` as
    /// the element shape.
    #[must_use]
    pub fn seq(elem: DynamicSchema) -> Self {
        DynamicSchema::Seq(Box::new(elem))
    }

    /// Convenience constructor for a `Map` schema. `fields` is the
    /// `(name, schema)` list in **declaration order**.
    #[must_use]
    pub fn map<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = (S, DynamicSchema)>,
        S: Into<String>,
    {
        DynamicSchema::Map(fields.into_iter().map(|(n, s)| (n.into(), s)).collect())
    }

    /// Convenience constructor for an `Enum` schema. `variants` is
    /// the list of variants; the caller is responsible for keeping
    /// them sorted ascending by discriminant — the walker
    /// debug-asserts the invariant on the first decode of each
    /// schema. Use [`EnumVariantSchema::new`] for each entry.
    #[must_use]
    pub fn enumeration<I>(variants: I) -> Self
    where
        I: IntoIterator<Item = EnumVariantSchema>,
    {
        let mut v: Vec<EnumVariantSchema> = variants.into_iter().collect();
        v.sort_by_key(|e| e.discriminant);
        DynamicSchema::Enum(v)
    }
}

/// A type whose postcard wire shape is describable by a
/// [`DynamicSchema`].
///
/// `Schema` is implemented for every `T: Document` (via the derive)
/// and **also** for hand-written
/// "historical" types that describe the wire shape of an older
/// version of a Document. Historical types never need to be
/// `Document`s themselves — they exist purely to describe bytes the
/// current reader still has on disk.
///
/// Separating `Schema` from `Document` keeps the migration registry
/// honest: a `historical_schemas()` entry refers to a
/// `Schema`-only type that owns no collection name and has no
/// `Document::VERSION` of its own (the `(u32, ...)` pair on the
/// outside carries the version).
pub trait Schema {
    /// Return the postcard wire-shape description for `Self`.
    ///
    /// Implementations build the schema bottom-up from the variants
    /// in [`DynamicSchema`].  The derive emits a `Schema`
    /// impl for every `#[derive(Document)]` type alongside the
    /// `Document` impl; hand-impls follow the same shape.
    #[must_use]
    fn schema() -> DynamicSchema;
}

/// `Option<T>` describes postcard's two-variant enum encoding: a
/// varint `u32` discriminant (`0` = `None`, `1` = `Some`) followed by
/// the matched variant's payload. `None` carries [`DynamicSchema::Null`]
/// (discriminant only, zero payload bytes); `Some` carries `T`'s own
/// schema.
///
/// This blanket impl lets an `Option<T>` field nest in a
/// `#[derive(Document)]` type's schema without the derive needing to
/// special-case `Option` syntactically — the derive's fallthrough
/// (`<FieldTy as Schema>::schema()`) resolves to this impl as long as
/// `T: Schema`.
impl<T: Schema> Schema for Option<T> {
    fn schema() -> DynamicSchema {
        DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "None", DynamicSchema::Null),
            EnumVariantSchema::new(1, "Some", T::schema()),
        ])
    }
}

#[doc(hidden)]
pub mod __private {
    //! Internal helpers used by `obj_derive` and by hand-written
    //! `historical_schemas()` implementations. Not part of the
    //! public API surface — the module exists only so the derive
    //! can refer to a stable path.

    use super::{DynamicSchema, Schema};

    /// Build a `historical_schemas()` return value from a list of
    /// `(version, fn() -> DynamicSchema)` pairs. Useful when a
    /// hand-impl wants the same look-and-feel as the derive output.
    #[must_use]
    pub fn schemas_from<const N: usize>(
        entries: [(u32, fn() -> DynamicSchema); N],
    ) -> Vec<(u32, DynamicSchema)> {
        entries.into_iter().map(|(v, f)| (v, f())).collect()
    }

    /// Helper that materialises `<T as Schema>::schema()` — used by
    /// the derive macro's `historical_schemas()` body so the
    /// generated code stays short.
    #[must_use]
    pub fn schema_of<T: Schema>() -> DynamicSchema {
        T::schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal `Schema`-implementing stand-in. Scalars deliberately
    /// do NOT implement `Schema` (the derive maps them syntactically),
    /// so the `Option<T>` tests need a real `T: Schema` to nest.
    struct Leaf;
    impl Schema for Leaf {
        fn schema() -> DynamicSchema {
            DynamicSchema::U64
        }
    }

    #[test]
    fn seq_constructor_boxes_inner() {
        let s = DynamicSchema::seq(DynamicSchema::U64);
        match s {
            DynamicSchema::Seq(inner) => assert_eq!(*inner, DynamicSchema::U64),
            other => panic!("expected Seq, got {other:?}"),
        }
    }

    #[test]
    fn enumeration_constructor_sorts_by_discriminant() {
        let s = DynamicSchema::enumeration([
            EnumVariantSchema::new(2, "B", DynamicSchema::Null),
            EnumVariantSchema::new(0, "Z", DynamicSchema::Null),
            EnumVariantSchema::new(1, "M", DynamicSchema::Null),
        ]);
        match s {
            DynamicSchema::Enum(v) => {
                assert_eq!(v.len(), 3);
                assert_eq!(v[0].discriminant, 0);
                assert_eq!(v[0].name, "Z");
                assert_eq!(v[1].discriminant, 1);
                assert_eq!(v[1].name, "M");
                assert_eq!(v[2].discriminant, 2);
                assert_eq!(v[2].name, "B");
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[test]
    fn option_schema_is_postcard_two_variant_enum() {
        let s = <Option<Leaf> as Schema>::schema();
        match s {
            DynamicSchema::Enum(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].discriminant, 0);
                assert_eq!(v[0].name, "None");
                assert_eq!(*v[0].payload, DynamicSchema::Null);
                assert_eq!(v[1].discriminant, 1);
                assert_eq!(v[1].name, "Some");
                assert_eq!(*v[1].payload, DynamicSchema::U64);
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[test]
    fn option_schema_nests_inner_schema() {
        let s = <Option<Option<Leaf>> as Schema>::schema();
        match s {
            DynamicSchema::Enum(v) => {
                assert_eq!(*v[1].payload, <Option<Leaf> as Schema>::schema());
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[test]
    fn map_constructor_preserves_order() {
        let s = DynamicSchema::map([
            ("a", DynamicSchema::U64),
            ("b", DynamicSchema::String),
            ("c", DynamicSchema::Bool),
        ]);
        match s {
            DynamicSchema::Map(fields) => {
                assert_eq!(fields.len(), 3);
                assert_eq!(fields[0].0, "a");
                assert_eq!(fields[1].0, "b");
                assert_eq!(fields[2].0, "c");
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }
}
