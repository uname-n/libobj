//! Shape-faithful [`Dynamic`] → `T` deserializer.
//!
//! This module implements a [`serde::Deserializer`] over a borrowed
//! [`Dynamic`] tree — the standard "deserialize from an in-memory
//! value" pattern, the same shape `serde_json::Value`'s deserializer
//! uses.
//!
//! # serde data-model ↔ `Dynamic` mapping
//!
//! | serde call            | `Dynamic`                          | visitor                |
//! |-----------------------|------------------------------------|------------------------|
//! | `deserialize_bool`    | [`Dynamic::Bool`]                  | `visit_bool`           |
//! | integer methods       | [`Dynamic::U64`] / [`Dynamic::I64`]| `visit_u*` / `visit_i*`|
//! | `deserialize_f*`      | [`Dynamic::F64`]                   | `visit_f64`            |
//! | `deserialize_str`     | [`Dynamic::String`]                | `visit_str`            |
//! | `deserialize_bytes`   | [`Dynamic::Bytes`]                 | `visit_bytes`          |
//! | `deserialize_seq`     | [`Dynamic::Seq`]                   | `visit_seq`            |
//! | `deserialize_map` /   | [`Dynamic::Map`]                   | `visit_map`            |
//! | `deserialize_struct`  |                                    |                        |
//! | `deserialize_enum`    | [`Dynamic::Enum`]                  | `visit_enum`           |
//! | `deserialize_option`  | `Enum{None}`→`visit_none`;         |                        |
//! |                       | `Enum{Some}`→`visit_some(payload)` |                        |
//! | `deserialize_unit`    | [`Dynamic::Null`]                  | `visit_unit`           |
//!
//! `Enum` variants are matched by **name** (`variant`), so no
//! discriminant is needed: serde looks the variant up in the target
//! type's variant list. Unit variants carry a [`Dynamic::Null`]
//! payload; newtype variants carry the inner type's `Dynamic`; tuple /
//! struct variants carry a [`Dynamic::Map`].

#![forbid(unsafe_code)]

use std::collections::btree_map;
use std::fmt;

use serde::de::{
    self, DeserializeOwned, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};
use serde::Deserializer;

use crate::codec::dynamic::Dynamic;
use crate::codec::schema::MAX_SCHEMA_DEPTH;
use crate::error::Error;

/// Error raised while deserializing a value out of a [`Dynamic`] tree.
///
/// Carries an owned, human-readable message. serde's `de::Error`
/// contract requires a `custom` constructor that takes any
/// `fmt::Display`; we capture it as a `String` so the message survives
/// across the error boundary. Converted into the crate's
/// [`Error::Corruption`] when surfaced through [`Dynamic::deserialize`]
/// (a shape mismatch in a migration `Dynamic` is, by construction, a
/// corrupt / disagreeing stored record).
#[derive(Debug)]
pub struct DynamicDeError {
    msg: String,
}

impl fmt::Display for DynamicDeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for DynamicDeError {}

impl de::Error for DynamicDeError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        DynamicDeError {
            msg: msg.to_string(),
        }
    }
}

impl From<DynamicDeError> for Error {
    fn from(_e: DynamicDeError) -> Self {
        Error::Corruption { page_id: 0 }
    }
}

type DeResult<T> = std::result::Result<T, DynamicDeError>;

/// Deserialize `value` into `T` via the shape-faithful in-memory path.
///
/// # Errors
///
/// - [`DynamicDeError`] if the `Dynamic` shape disagrees with `T`
///   (wrong variant, missing struct field, out-of-range integer, tree
///   deeper than [`MAX_SCHEMA_DEPTH`], …). Never panics.
pub fn from_dynamic<T: DeserializeOwned>(value: &Dynamic) -> DeResult<T> {
    T::deserialize(DynamicDeserializer::new(value, 0))
}

/// A [`Deserializer`] borrowing one [`Dynamic`] node + its tree depth.
struct DynamicDeserializer<'a> {
    value: &'a Dynamic,
    depth: usize,
}

impl<'a> DynamicDeserializer<'a> {
    fn new(value: &'a Dynamic, depth: usize) -> Self {
        DynamicDeserializer { value, depth }
    }

    /// Bound the tree depth. Each composite descent calls this
    /// before building the child deserializer.
    fn check_depth(&self) -> DeResult<usize> {
        if self.depth >= MAX_SCHEMA_DEPTH {
            return Err(de::Error::custom("Dynamic tree exceeded MAX_SCHEMA_DEPTH"));
        }
        Ok(self.depth + 1)
    }

    fn type_err(&self, expected: &str) -> DynamicDeError {
        de::Error::custom(format!(
            "expected {expected}, found Dynamic::{}",
            variant_label(self.value)
        ))
    }
}

/// Static label for a `Dynamic` variant, used in mismatch messages.
fn variant_label(d: &Dynamic) -> &'static str {
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

impl<'de> Deserializer<'de> for DynamicDeserializer<'_> {
    type Error = DynamicDeError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Null => visitor.visit_unit(),
            Dynamic::Bool(b) => visitor.visit_bool(*b),
            Dynamic::U64(n) => visitor.visit_u64(*n),
            Dynamic::I64(n) => visitor.visit_i64(*n),
            Dynamic::F64(f) => visitor.visit_f64(*f),
            Dynamic::String(s) => visitor.visit_str(s),
            Dynamic::Bytes(b) => visitor.visit_bytes(b),
            Dynamic::Seq(_) => self.deserialize_seq(visitor),
            Dynamic::Map(_) => self.deserialize_map(visitor),
            Dynamic::Enum { .. } => self.deserialize_enum("", &[], visitor),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Bool(b) => visitor.visit_bool(*b),
            _ => Err(self.type_err("bool")),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_i8(i8::try_from(self.as_i64()?).map_err(|_| self.type_err("i8"))?)
    }

    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_i16(i16::try_from(self.as_i64()?).map_err(|_| self.type_err("i16"))?)
    }

    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_i32(i32::try_from(self.as_i64()?).map_err(|_| self.type_err("i32"))?)
    }

    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_i64(self.as_i64()?)
    }

    fn deserialize_i128<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_i128(i128::from(self.as_i64()?))
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_u8(u8::try_from(self.as_u64()?).map_err(|_| self.type_err("u8"))?)
    }

    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_u16(u16::try_from(self.as_u64()?).map_err(|_| self.type_err("u16"))?)
    }

    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_u32(u32::try_from(self.as_u64()?).map_err(|_| self.type_err("u32"))?)
    }

    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_u64(self.as_u64()?)
    }

    fn deserialize_u128<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_u128(u128::from(self.as_u64()?))
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            // allow: narrowing is intentional — caller asked for f32, so the stored f64 is deliberately rounded to f32 precision.
            #[allow(clippy::cast_possible_truncation)]
            Dynamic::F64(f) => visitor.visit_f32(*f as f32),
            _ => Err(self.type_err("f32")),
        }
    }

    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::F64(f) => visitor.visit_f64(*f),
            _ => Err(self.type_err("f64")),
        }
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::String(s) if s.chars().count() == 1 => {
                let c = s.chars().next().ok_or_else(|| self.type_err("char"))?;
                visitor.visit_char(c)
            }
            _ => Err(self.type_err("char")),
        }
    }

    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::String(s) => visitor.visit_str(s),
            _ => Err(self.type_err("str")),
        }
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Bytes(b) => visitor.visit_bytes(b),
            Dynamic::Seq(_) => self.deserialize_seq(visitor),
            _ => Err(self.type_err("bytes")),
        }
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        self.deserialize_bytes(visitor)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Enum { variant, payload } => match variant.as_str() {
                "None" => visitor.visit_none(),
                "Some" => {
                    let next = self.check_depth()?;
                    visitor.visit_some(DynamicDeserializer::new(payload, next))
                }
                other => Err(de::Error::custom(format!(
                    "expected Option (None|Some), found Enum variant {other:?}"
                ))),
            },
            Dynamic::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Null => visitor.visit_unit(),
            _ => Err(self.type_err("unit")),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> DeResult<V::Value> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> DeResult<V::Value> {
        let next = self.check_depth()?;
        visitor.visit_newtype_struct(DynamicDeserializer::new(self.value, next))
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Seq(items) => {
                let next = self.check_depth()?;
                visitor.visit_seq(SeqDe {
                    iter: items.iter(),
                    depth: next,
                })
            }
            Dynamic::Bytes(bytes) => {
                let next = self.check_depth()?;
                visitor.visit_seq(BytesSeqDe {
                    iter: bytes.iter(),
                    depth: next,
                })
            }
            _ => Err(self.type_err("seq")),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> DeResult<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> DeResult<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Map(map) => {
                let next = self.check_depth()?;
                visitor.visit_map(MapDe {
                    iter: map.iter(),
                    value: None,
                    depth: next,
                })
            }
            _ => Err(self.type_err("map")),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> DeResult<V::Value> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Enum { variant, payload } => {
                let next = self.check_depth()?;
                visitor.visit_enum(EnumDe {
                    variant,
                    payload,
                    depth: next,
                })
            }
            _ => Err(self.type_err("enum")),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        visitor.visit_unit()
    }
}

impl DynamicDeserializer<'_> {
    /// Read the node as a `u64` (only [`Dynamic::U64`] / a non-negative
    /// [`Dynamic::I64`]); error on any other shape.
    fn as_u64(&self) -> DeResult<u64> {
        match self.value {
            Dynamic::U64(n) => Ok(*n),
            Dynamic::I64(n) if *n >= 0 => Ok(n.unsigned_abs()),
            _ => Err(self.type_err("unsigned integer")),
        }
    }

    /// Read the node as an `i64` (only [`Dynamic::I64`] / a
    /// representable [`Dynamic::U64`]); error on any other shape.
    fn as_i64(&self) -> DeResult<i64> {
        match self.value {
            Dynamic::I64(n) => Ok(*n),
            Dynamic::U64(n) => i64::try_from(*n).map_err(|_| self.type_err("signed integer")),
            _ => Err(self.type_err("signed integer")),
        }
    }
}

/// `SeqAccess` over a [`Dynamic::Seq`]'s elements.
struct SeqDe<'a> {
    iter: std::slice::Iter<'a, Dynamic>,
    depth: usize,
}

impl<'de> SeqAccess<'de> for SeqDe<'_> {
    type Error = DynamicDeError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> DeResult<Option<T::Value>> {
        match self.iter.next() {
            Some(item) => seed
                .deserialize(DynamicDeserializer::new(item, self.depth))
                .map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

/// `SeqAccess` over a [`Dynamic::Bytes`] node, yielding each byte as a
/// `u8`-shaped element so a `Vec<u8>` / `[u8; N]` target decodes.
struct BytesSeqDe<'a> {
    iter: std::slice::Iter<'a, u8>,
    depth: usize,
}

impl<'de> SeqAccess<'de> for BytesSeqDe<'_> {
    type Error = DynamicDeError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> DeResult<Option<T::Value>> {
        match self.iter.next() {
            Some(b) => {
                let node = Dynamic::U64(u64::from(*b));
                seed.deserialize(DynamicDeserializer::new(&node, self.depth))
                    .map(Some)
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

/// `MapAccess` over a [`Dynamic::Map`]'s entries. `value` holds the
/// pending value between the `next_key_seed` / `next_value_seed` pair.
struct MapDe<'a> {
    iter: btree_map::Iter<'a, String, Dynamic>,
    value: Option<&'a Dynamic>,
    depth: usize,
}

impl<'de> MapAccess<'de> for MapDe<'_> {
    type Error = DynamicDeError;

    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> DeResult<Option<K::Value>> {
        match self.iter.next() {
            Some((k, v)) => {
                self.value = Some(v);
                seed.deserialize(k.as_str().into_deserializer()).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> DeResult<V::Value> {
        let value = self
            .value
            .take()
            .ok_or_else(|| de::Error::custom("MapAccess: value requested before key"))?;
        seed.deserialize(DynamicDeserializer::new(value, self.depth))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

/// `EnumAccess` over a [`Dynamic::Enum`]: hands serde the variant name,
/// then a [`VariantDe`] that decodes the payload by variant kind.
struct EnumDe<'a> {
    variant: &'a str,
    payload: &'a Dynamic,
    depth: usize,
}

impl<'a, 'de> EnumAccess<'de> for EnumDe<'a> {
    type Error = DynamicDeError;
    type Variant = VariantDe<'a>;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> DeResult<(V::Value, Self::Variant)> {
        let variant = seed.deserialize(self.variant.into_deserializer())?;
        Ok((
            variant,
            VariantDe {
                payload: self.payload,
                depth: self.depth,
            },
        ))
    }
}

/// `VariantAccess` decoding the payload of one [`Dynamic::Enum`]
/// variant. The kind (unit / newtype / tuple / struct) is dictated by
/// the target type via the method serde calls; the payload `Dynamic`
/// must match (a [`Dynamic::Null`] for a unit variant).
struct VariantDe<'a> {
    payload: &'a Dynamic,
    depth: usize,
}

impl<'de> VariantAccess<'de> for VariantDe<'_> {
    type Error = DynamicDeError;

    fn unit_variant(self) -> DeResult<()> {
        match self.payload {
            Dynamic::Null => Ok(()),
            other => Err(de::Error::custom(format!(
                "unit variant expected Null payload, found Dynamic::{}",
                variant_label(other)
            ))),
        }
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> DeResult<T::Value> {
        seed.deserialize(DynamicDeserializer::new(self.payload, self.depth))
    }

    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> DeResult<V::Value> {
        DynamicDeserializer::new(self.payload, self.depth).deserialize_tuple_payload(visitor)
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> DeResult<V::Value> {
        DynamicDeserializer::new(self.payload, self.depth).deserialize_map(visitor)
    }
}

impl DynamicDeserializer<'_> {
    /// Decode a tuple-variant payload. The walker stores tuple variants
    /// as a [`Dynamic::Map`] with synthetic numeric keys; serde wants a
    /// positional seq, so yield the map's values in key order. A
    /// non-map payload (e.g. a 1-tuple stored as the bare inner value)
    /// is wrapped as a single-element seq.
    fn deserialize_tuple_payload<'de, V: Visitor<'de>>(self, visitor: V) -> DeResult<V::Value> {
        match self.value {
            Dynamic::Map(map) => {
                // The walker stores tuple-variant fields under synthetic
                // decimal keys "0", "1", … "10", "11". A `BTreeMap`
                // orders those keys LEXICOGRAPHICALLY ("10" < "2"), so
                // iterating `values()` reorders fields once the arity
                // reaches 10. Sort the entries by the NUMERIC value of
                // each key to recover positional order. Keys that aren't
                // decimal integers sort after the numeric ones (stably,
                // in original key order) so a malformed payload still
                // decodes deterministically rather than panicking.
                let mut entries: Vec<(Option<u64>, &Dynamic)> =
                    map.iter().map(|(k, v)| (k.parse::<u64>().ok(), v)).collect();
                entries.sort_by_key(|&(key, _)| (key.is_none(), key));
                let values: Vec<&Dynamic> = entries.into_iter().map(|(_, v)| v).collect();
                visitor.visit_seq(MapValuesSeqDe {
                    iter: values.into_iter(),
                    depth: self.depth,
                })
            }
            Dynamic::Seq(_) => self.deserialize_seq(visitor),
            _ => Err(self.type_err("tuple-variant payload (Map or Seq)")),
        }
    }
}

/// `SeqAccess` over a [`Dynamic::Map`]'s VALUES in NUMERIC key order —
/// used for tuple-variant payloads whose synthetic keys are "0", "1", …,
/// "10". The values are pre-sorted by parsed key in
/// [`DynamicDeserializer::deserialize_tuple_payload`] so positional
/// fields survive arities of 10 or more (lexicographic order would put
/// "10" before "2").
struct MapValuesSeqDe<'a> {
    iter: std::vec::IntoIter<&'a Dynamic>,
    depth: usize,
}

impl<'de> SeqAccess<'de> for MapValuesSeqDe<'_> {
    type Error = DynamicDeError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> DeResult<Option<T::Value>> {
        match self.iter.next() {
            Some(item) => seed
                .deserialize(DynamicDeserializer::new(item, self.depth))
                .map(Some),
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    fn enum_node(variant: &str, payload: Dynamic) -> Dynamic {
        Dynamic::Enum {
            variant: variant.to_owned(),
            payload: Box::new(payload),
        }
    }

    #[test]
    fn option_some_round_trips() {
        let node = enum_node("Some", Dynamic::U64(42));
        let got: Option<u64> = from_dynamic(&node).expect("decode");
        assert_eq!(got, Some(42));
    }

    #[test]
    fn option_none_round_trips() {
        let node = enum_node("None", Dynamic::Null);
        let got: Option<u64> = from_dynamic(&node).expect("decode");
        assert_eq!(got, None);
    }

    #[test]
    fn old_broken_behavior_is_gone() {
        let node = enum_node("Some", Dynamic::U64(42));
        let got: Option<u64> = from_dynamic(&node).expect("decode");
        assert_eq!(got, Some(42), "Some(42) must NOT mis-decode to Some(4)");
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    enum Tier {
        Free,
        Pro(u32),
        Team { seats: u32 },
    }

    #[test]
    fn enum_unit_variant() {
        let node = enum_node("Free", Dynamic::Null);
        let got: Tier = from_dynamic(&node).expect("decode");
        assert_eq!(got, Tier::Free);
    }

    #[test]
    fn enum_newtype_variant_with_int() {
        let node = enum_node("Pro", Dynamic::U64(7));
        let got: Tier = from_dynamic(&node).expect("decode");
        assert_eq!(got, Tier::Pro(7));
    }

    #[test]
    fn enum_struct_variant() {
        let mut payload = BTreeMap::new();
        payload.insert("seats".to_owned(), Dynamic::U64(12));
        let node = enum_node("Team", Dynamic::Map(payload));
        let got: Tier = from_dynamic(&node).expect("decode");
        assert_eq!(got, Tier::Team { seats: 12 });
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    enum Shape {
        Point,
        Pair(i32, i32),
    }

    #[test]
    fn enum_tuple_variant() {
        let mut payload = BTreeMap::new();
        payload.insert("0".to_owned(), Dynamic::I64(-3));
        payload.insert("1".to_owned(), Dynamic::I64(5));
        let node = enum_node("Pair", Dynamic::Map(payload));
        let got: Shape = from_dynamic(&node).expect("decode");
        assert_eq!(got, Shape::Pair(-3, 5));
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct WithOptionAndEnum {
        name: String,
        badge: Option<u64>,
        tier: Tier,
        count: u32,
    }

    #[test]
    fn struct_with_option_and_enum_fields() {
        let mut map = BTreeMap::new();
        map.insert("name".to_owned(), Dynamic::String("ada".to_owned()));
        map.insert("badge".to_owned(), enum_node("Some", Dynamic::U64(99)));
        map.insert("tier".to_owned(), enum_node("Pro", Dynamic::U64(3)));
        map.insert("count".to_owned(), Dynamic::U64(5));
        let node = Dynamic::Map(map);
        let got: WithOptionAndEnum = from_dynamic(&node).expect("decode");
        assert_eq!(
            got,
            WithOptionAndEnum {
                name: "ada".to_owned(),
                badge: Some(99),
                tier: Tier::Pro(3),
                count: 5,
            }
        );
    }

    #[test]
    fn vec_of_options() {
        let node = Dynamic::Seq(vec![
            enum_node("Some", Dynamic::U64(1)),
            enum_node("None", Dynamic::Null),
            enum_node("Some", Dynamic::U64(3)),
        ]);
        let got: Vec<Option<u64>> = from_dynamic(&node).expect("decode");
        assert_eq!(got, vec![Some(1), None, Some(3)]);
    }

    #[test]
    fn scalar_round_trips() {
        assert!(from_dynamic::<bool>(&Dynamic::Bool(true)).expect("b"));
        assert_eq!(from_dynamic::<u8>(&Dynamic::U64(255)).expect("u8"), 255u8);
        assert_eq!(
            from_dynamic::<u16>(&Dynamic::U64(1000)).expect("u16"),
            1000u16
        );
        assert_eq!(
            from_dynamic::<u32>(&Dynamic::U64(70_000)).expect("u32"),
            70_000u32
        );
        assert_eq!(from_dynamic::<i32>(&Dynamic::I64(-5)).expect("i32"), -5i32);
        assert_eq!(from_dynamic::<i64>(&Dynamic::I64(-9)).expect("i64"), -9i64);
        let f = from_dynamic::<f64>(&Dynamic::F64(1.5)).expect("f64");
        assert!((f - 1.5f64).abs() < f64::EPSILON);
        assert_eq!(
            from_dynamic::<String>(&Dynamic::String("hi".to_owned())).expect("s"),
            "hi".to_owned()
        );
    }

    #[test]
    fn malformed_shape_is_err_not_panic() {
        let err = from_dynamic::<u64>(&Dynamic::String("x".to_owned()));
        assert!(err.is_err(), "string→u64 must error, not panic");

        let bad = enum_node("Maybe", Dynamic::Null);
        let err2 = from_dynamic::<Option<u64>>(&bad);
        assert!(err2.is_err(), "unknown Option variant must error");

        let err3 = from_dynamic::<u32>(&Dynamic::U64(u64::from(u32::MAX) + 1));
        assert!(err3.is_err(), "out-of-range u32 must error");
    }

    #[test]
    fn deep_tree_is_bounded() {
        let mut node = Dynamic::U64(1);
        for _ in 0..(MAX_SCHEMA_DEPTH + 4) {
            node = Dynamic::Seq(vec![node]);
        }
        let err = from_dynamic::<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<u64>>>>>>>>>>>(&node);
        let _ = err;

        let inner = Dynamic::Seq(vec![Dynamic::U64(1)]);
        let de = DynamicDeserializer::new(&inner, MAX_SCHEMA_DEPTH);
        assert!(de.check_depth().is_err(), "depth bound must trip");
    }

    // --- malformed shapes for bool ---

    #[test]
    fn bool_wrong_shape_errors() {
        let err = from_dynamic::<bool>(&Dynamic::U64(1));
        assert!(err.is_err(), "U64→bool must error");
    }

    // --- numeric boundary: i8 / i16 / i32 overflow ---

    #[test]
    fn i8_in_range_round_trips() {
        assert_eq!(from_dynamic::<i8>(&Dynamic::I64(-128)).expect("min"), -128i8);
        assert_eq!(from_dynamic::<i8>(&Dynamic::I64(127)).expect("max"), 127i8);
    }

    #[test]
    fn i8_overflow_errors() {
        assert!(from_dynamic::<i8>(&Dynamic::I64(128)).is_err(), "128→i8");
        assert!(from_dynamic::<i8>(&Dynamic::I64(-129)).is_err(), "-129→i8");
    }

    #[test]
    fn i8_wrong_shape_errors() {
        assert!(
            from_dynamic::<i8>(&Dynamic::String("x".to_owned())).is_err(),
            "string→i8"
        );
    }

    #[test]
    fn i16_overflow_errors() {
        assert!(
            from_dynamic::<i16>(&Dynamic::I64(32_768)).is_err(),
            "32768→i16"
        );
        assert!(
            from_dynamic::<i16>(&Dynamic::I64(-32_769)).is_err(),
            "-32769→i16"
        );
    }

    #[test]
    fn i16_wrong_shape_errors() {
        assert!(
            from_dynamic::<i16>(&Dynamic::Bool(false)).is_err(),
            "bool→i16"
        );
    }

    #[test]
    fn i32_overflow_errors() {
        let too_big = i64::from(i32::MAX) + 1;
        assert!(
            from_dynamic::<i32>(&Dynamic::I64(too_big)).is_err(),
            "i32 overflow positive"
        );
        let too_small = i64::from(i32::MIN) - 1;
        assert!(
            from_dynamic::<i32>(&Dynamic::I64(too_small)).is_err(),
            "i32 overflow negative"
        );
    }

    #[test]
    fn i32_wrong_shape_errors() {
        assert!(
            from_dynamic::<i32>(&Dynamic::String("x".to_owned())).is_err(),
            "string→i32"
        );
    }

    // --- i64 wrong shape ---

    #[test]
    fn i64_wrong_shape_errors() {
        assert!(
            from_dynamic::<i64>(&Dynamic::Bool(true)).is_err(),
            "bool→i64"
        );
    }

    // --- i128 / u128 round trips ---

    #[test]
    fn i128_round_trips() {
        let got = from_dynamic::<i128>(&Dynamic::I64(-7)).expect("i128");
        assert_eq!(got, -7i128);
    }

    #[test]
    fn i128_wrong_shape_errors() {
        assert!(
            from_dynamic::<i128>(&Dynamic::Bool(false)).is_err(),
            "bool→i128"
        );
    }

    #[test]
    fn u128_round_trips() {
        let got = from_dynamic::<u128>(&Dynamic::U64(99)).expect("u128");
        assert_eq!(got, 99u128);
    }

    #[test]
    fn u128_wrong_shape_errors() {
        assert!(
            from_dynamic::<u128>(&Dynamic::Bool(false)).is_err(),
            "bool→u128"
        );
    }

    // --- numeric boundary: u8 / u16 / u32 overflow ---

    #[test]
    fn u8_overflow_errors() {
        assert!(
            from_dynamic::<u8>(&Dynamic::U64(256)).is_err(),
            "256→u8 overflow"
        );
    }

    #[test]
    fn u8_wrong_shape_errors() {
        assert!(
            from_dynamic::<u8>(&Dynamic::String("x".to_owned())).is_err(),
            "string→u8"
        );
    }

    #[test]
    fn u16_overflow_errors() {
        assert!(
            from_dynamic::<u16>(&Dynamic::U64(65_536)).is_err(),
            "65536→u16 overflow"
        );
    }

    #[test]
    fn u16_wrong_shape_errors() {
        assert!(
            from_dynamic::<u16>(&Dynamic::Bool(true)).is_err(),
            "bool→u16"
        );
    }

    #[test]
    fn u64_wrong_shape_errors() {
        assert!(
            from_dynamic::<u64>(&Dynamic::Bool(false)).is_err(),
            "bool→u64"
        );
    }

    // --- wrong signedness: negative I64 → unsigned ---

    #[test]
    fn negative_i64_as_unsigned_errors() {
        // as_u64 only allows I64 values >= 0
        assert!(
            from_dynamic::<u64>(&Dynamic::I64(-1)).is_err(),
            "negative I64→u64"
        );
        assert!(
            from_dynamic::<u8>(&Dynamic::I64(-1)).is_err(),
            "negative I64→u8"
        );
    }

    // --- U64 too large for i64 ---

    #[test]
    fn u64_too_large_for_i64_errors() {
        let too_big = u64::MAX;
        assert!(
            from_dynamic::<i64>(&Dynamic::U64(too_big)).is_err(),
            "u64::MAX→i64"
        );
    }

    // --- f32 round trip and wrong shape ---

    #[test]
    fn f32_round_trips() {
        let got = from_dynamic::<f32>(&Dynamic::F64(2.5)).expect("f32");
        assert!((got - 2.5f32).abs() < f32::EPSILON);
    }

    #[test]
    fn f32_wrong_shape_errors() {
        assert!(
            from_dynamic::<f32>(&Dynamic::U64(1)).is_err(),
            "U64→f32"
        );
    }

    #[test]
    fn f64_wrong_shape_errors() {
        assert!(
            from_dynamic::<f64>(&Dynamic::U64(1)).is_err(),
            "U64→f64"
        );
    }

    // --- char round trips and wrong shape ---

    #[test]
    fn char_round_trips() {
        let got = from_dynamic::<char>(&Dynamic::String("A".to_owned())).expect("char");
        assert_eq!(got, 'A');
    }

    #[test]
    fn char_multi_char_string_errors() {
        assert!(
            from_dynamic::<char>(&Dynamic::String("AB".to_owned())).is_err(),
            "multi-char→char"
        );
    }

    #[test]
    fn char_wrong_shape_errors() {
        assert!(
            from_dynamic::<char>(&Dynamic::U64(65)).is_err(),
            "U64→char"
        );
    }

    // --- str / string wrong shape ---

    #[test]
    fn str_wrong_shape_errors() {
        assert!(
            from_dynamic::<String>(&Dynamic::U64(1)).is_err(),
            "U64→String"
        );
    }

    // --- bytes with Seq path, and wrong shape ---

    #[test]
    fn bytes_from_seq_round_trips() {
        // deserialize_bytes also accepts Seq as a fallback
        let node = Dynamic::Seq(vec![Dynamic::U64(1), Dynamic::U64(2), Dynamic::U64(3)]);
        let got: Vec<u8> = from_dynamic(&node).expect("bytes from seq");
        assert_eq!(got, vec![1u8, 2, 3]);
    }

    #[test]
    fn bytes_from_bytes_node_round_trips() {
        // Vec<u8> decodes from Dynamic::Bytes via deserialize_seq →
        // BytesSeqDe, which yields each byte as a U64 element.
        let node = Dynamic::Bytes(vec![10, 20, 30]);
        let got: Vec<u8> = from_dynamic(&node).expect("bytes buf");
        assert_eq!(got, vec![10u8, 20, 30]);
    }

    #[test]
    fn bytes_wrong_shape_errors() {
        // A Bool node is not accepted as bytes.
        assert!(
            from_dynamic::<Vec<u8>>(&Dynamic::Bool(false)).is_err(),
            "Bool→bytes"
        );
    }

    // --- unit / unit-struct wrong shape ---

    #[test]
    fn unit_wrong_shape_errors() {
        assert!(
            from_dynamic::<()>(&Dynamic::U64(0)).is_err(),
            "U64→unit"
        );
    }

    #[test]
    fn unit_struct_round_trips() {
        #[derive(Debug, PartialEq, Eq, serde::Deserialize)]
        struct Empty;
        let got: Empty = from_dynamic(&Dynamic::Null).expect("unit-struct");
        assert_eq!(got, Empty);
    }

    #[test]
    fn unit_struct_wrong_shape_errors() {
        #[derive(Debug, serde::Deserialize)]
        struct Empty;
        assert!(
            from_dynamic::<Empty>(&Dynamic::U64(0)).is_err(),
            "U64→unit-struct"
        );
    }

    // --- newtype struct ---

    #[test]
    fn newtype_struct_round_trips() {
        #[derive(Debug, PartialEq, Eq, serde::Deserialize)]
        struct Wrapper(u64);
        let got: Wrapper = from_dynamic(&Dynamic::U64(42)).expect("newtype");
        assert_eq!(got, Wrapper(42));
    }

    // --- tuple and tuple-struct ---

    #[test]
    fn tuple_round_trips() {
        let node = Dynamic::Seq(vec![Dynamic::U64(1), Dynamic::U64(2)]);
        let got: (u64, u64) = from_dynamic(&node).expect("tuple");
        assert_eq!(got, (1, 2));
    }

    #[test]
    fn tuple_wrong_shape_errors() {
        assert!(
            from_dynamic::<(u64, u64)>(&Dynamic::U64(1)).is_err(),
            "U64→tuple"
        );
    }

    #[test]
    fn tuple_struct_round_trips() {
        #[derive(Debug, PartialEq, Eq, serde::Deserialize)]
        struct Pair(u64, u64);
        let node = Dynamic::Seq(vec![Dynamic::U64(3), Dynamic::U64(4)]);
        let got: Pair = from_dynamic(&node).expect("tuple-struct");
        assert_eq!(got, Pair(3, 4));
    }

    // --- map wrong shape ---

    #[test]
    fn map_wrong_shape_errors() {
        assert!(
            from_dynamic::<BTreeMap<String, u64>>(&Dynamic::U64(1)).is_err(),
            "U64→map"
        );
    }

    #[test]
    fn map_round_trips() {
        let mut m = BTreeMap::new();
        m.insert("x".to_owned(), Dynamic::U64(1));
        m.insert("y".to_owned(), Dynamic::U64(2));
        let node = Dynamic::Map(m);
        let got: BTreeMap<String, u64> = from_dynamic(&node).expect("map");
        assert_eq!(got.get("x"), Some(&1u64));
        assert_eq!(got.get("y"), Some(&2u64));
    }

    // --- struct wrong shape (not a map) ---

    #[test]
    fn struct_wrong_shape_errors() {
        // Use WithOptionAndEnum (already defined) as the struct target;
        // a bare U64 cannot deserialize as a struct.
        assert!(
            from_dynamic::<WithOptionAndEnum>(&Dynamic::U64(0)).is_err(),
            "U64→struct"
        );
    }

    // --- enum wrong shape (not Dynamic::Enum) ---

    #[test]
    fn enum_wrong_shape_errors() {
        assert!(
            from_dynamic::<Tier>(&Dynamic::U64(0)).is_err(),
            "U64→enum"
        );
        assert!(
            from_dynamic::<Tier>(&Dynamic::String("Free".to_owned())).is_err(),
            "String→enum"
        );
    }

    // --- VariantDe::unit_variant with non-Null payload errors ---

    #[test]
    fn unit_variant_non_null_payload_errors() {
        // Tier::Free is a unit variant; its payload must be Null.
        // A non-Null payload (here U64) must error, not panic.
        let bad = Dynamic::Enum {
            variant: "Free".to_owned(),
            payload: Box::new(Dynamic::U64(1)),
        };
        let err = from_dynamic::<Tier>(&bad);
        assert!(err.is_err(), "unit variant with non-Null payload must error");
    }

    // --- tuple-variant payload stored as a Seq (not Map) ---

    #[test]
    fn tuple_variant_seq_payload_round_trips() {
        // Shape::Pair encoded with a Seq payload rather than a Map.
        let node = Dynamic::Enum {
            variant: "Pair".to_owned(),
            payload: Box::new(Dynamic::Seq(vec![Dynamic::I64(10), Dynamic::I64(20)])),
        };
        let got: Shape = from_dynamic(&node).expect("tuple variant via Seq");
        assert_eq!(got, Shape::Pair(10, 20));
    }

    // --- tuple variant of arity >= 10: synthetic keys "0".."10" must
    //     decode in NUMERIC order, not BTreeMap lexicographic order
    //     ("10" sorts before "2"). Regression for issue #5. ---

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    enum Wide {
        #[allow(clippy::many_single_char_names)] // allow: an 11-field tuple variant is the point of this regression test.
        Eleven(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64),
    }

    #[test]
    fn tuple_variant_arity_eleven_map_payload_round_trips() {
        // Build the Map payload the schema walker produces: synthetic
        // decimal keys "0".."10" with distinct, position-revealing values.
        let mut payload = BTreeMap::new();
        for i in 0..11i64 {
            // value = 100 + position, so a reorder is detectable per slot.
            payload.insert(i.to_string(), Dynamic::I64(100 + i));
        }
        let node = enum_node("Eleven", Dynamic::Map(payload));
        let got: Wide = from_dynamic(&node).expect("decode 11-field tuple variant");
        assert_eq!(
            got,
            Wide::Eleven(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
            "fields must come back in positional (numeric-key) order, not lexicographic"
        );
    }

    #[test]
    fn tuple_variant_arity_eleven_encode_round_trips() {
        use crate::codec::schema::{DynamicSchema, EnumVariantSchema};

        // Full encode + Dynamic::deserialize round-trip through the real
        // postcard encoder and the schema walker, which produces exactly
        // the synthetic-key Map ("0".."10") that this fix re-sorts.
        let original = Wide::Eleven(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let bytes = postcard::to_allocvec(&original).expect("encode");

        // Schema for the single tuple variant: a Map of decimal keys.
        let fields: Vec<(String, DynamicSchema)> =
            (0..11).map(|i: u64| (i.to_string(), DynamicSchema::I64)).collect();
        let schema =
            DynamicSchema::enumeration([EnumVariantSchema::new(0, "Eleven", DynamicSchema::map(fields))]);

        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        let got: Wide = from_dynamic(&dyn_view).expect("from_dynamic");
        assert_eq!(got, original, "encode + deserialize must preserve field order");
    }

    #[test]
    fn tuple_variant_wrong_payload_errors() {
        // A Bool payload for a tuple variant is neither Map nor Seq.
        let bad = Dynamic::Enum {
            variant: "Pair".to_owned(),
            payload: Box::new(Dynamic::Bool(false)),
        };
        let err = from_dynamic::<Shape>(&bad);
        assert!(err.is_err(), "wrong tuple-variant payload must error");
    }

    // --- option with non-Enum, non-Null node (bare value) ---

    #[test]
    fn option_bare_value_is_some() {
        // A bare Dynamic::U64 (not wrapped in Enum{Some}) should decode
        // as Some via the fallback `_ => visitor.visit_some(self)` branch.
        let node = Dynamic::U64(7);
        let got: Option<u64> = from_dynamic(&node).expect("bare→some");
        assert_eq!(got, Some(7));
    }

    #[test]
    fn option_null_is_none() {
        let got: Option<u64> = from_dynamic(&Dynamic::Null).expect("null→none");
        assert_eq!(got, None);
    }

    // --- deserialize_any paths ---
    //
    // `deserialize_any` is called when the target is a type like
    // `Dynamic` itself (round-trip through a generic serde visitor).
    // We drive it via a simple collecting visitor defined inline.

    /// A minimal visitor that accepts any serde value and records what
    /// method was called, so we can assert which `deserialize_any`
    /// branch fired without pulling in `serde_json`.
    #[derive(Debug, PartialEq, Eq)]
    enum AnyTag {
        Unit,
        Bool,
        U64,
        I64,
        F64,
        Str,
        Bytes,
        Seq,
        Map,
        Enum,
    }

    struct TagVisitor;

    impl<'de> serde::de::Visitor<'de> for TagVisitor {
        type Value = AnyTag;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("any")
        }

        fn visit_unit<E: de::Error>(self) -> Result<AnyTag, E> { Ok(AnyTag::Unit) }
        fn visit_bool<E: de::Error>(self, _: bool) -> Result<AnyTag, E> { Ok(AnyTag::Bool) }
        fn visit_u64<E: de::Error>(self, _: u64) -> Result<AnyTag, E> { Ok(AnyTag::U64) }
        fn visit_i64<E: de::Error>(self, _: i64) -> Result<AnyTag, E> { Ok(AnyTag::I64) }
        fn visit_f64<E: de::Error>(self, _: f64) -> Result<AnyTag, E> { Ok(AnyTag::F64) }
        fn visit_str<E: de::Error>(self, _: &str) -> Result<AnyTag, E> { Ok(AnyTag::Str) }
        fn visit_bytes<E: de::Error>(self, _: &[u8]) -> Result<AnyTag, E> { Ok(AnyTag::Bytes) }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut a: A) -> Result<AnyTag, A::Error> {
            while a.next_element::<de::IgnoredAny>()?.is_some() {}
            Ok(AnyTag::Seq)
        }

        fn visit_map<A: de::MapAccess<'de>>(self, mut a: A) -> Result<AnyTag, A::Error> {
            while a.next_entry::<de::IgnoredAny, de::IgnoredAny>()?.is_some() {}
            Ok(AnyTag::Map)
        }

        fn visit_enum<A: de::EnumAccess<'de>>(self, a: A) -> Result<AnyTag, A::Error> {
            use de::VariantAccess;
            let (_, v) = a.variant::<de::IgnoredAny>()?;
            v.unit_variant()?;
            Ok(AnyTag::Enum)
        }
    }

    fn any_tag(node: &Dynamic) -> AnyTag {
        DynamicDeserializer::new(node, 0)
            .deserialize_any(TagVisitor)
            .expect("any tag")
    }

    #[test]
    fn deserialize_any_scalars() {
        assert_eq!(any_tag(&Dynamic::Null), AnyTag::Unit);
        assert_eq!(any_tag(&Dynamic::Bool(true)), AnyTag::Bool);
        assert_eq!(any_tag(&Dynamic::U64(5)), AnyTag::U64);
        assert_eq!(any_tag(&Dynamic::I64(-3)), AnyTag::I64);
        assert_eq!(any_tag(&Dynamic::F64(1.5)), AnyTag::F64);
        assert_eq!(any_tag(&Dynamic::String("hi".to_owned())), AnyTag::Str);
        assert_eq!(any_tag(&Dynamic::Bytes(vec![1])), AnyTag::Bytes);
    }

    #[test]
    fn deserialize_any_seq_and_map() {
        let seq = Dynamic::Seq(vec![Dynamic::U64(1)]);
        assert_eq!(any_tag(&seq), AnyTag::Seq);

        let mut m = BTreeMap::new();
        m.insert("k".to_owned(), Dynamic::U64(9));
        let map_node = Dynamic::Map(m);
        assert_eq!(any_tag(&map_node), AnyTag::Map);
    }

    #[test]
    fn deserialize_any_enum() {
        let node = enum_node("Free", Dynamic::Null);
        assert_eq!(any_tag(&node), AnyTag::Enum);
    }

    // --- bytes-as-seq: BytesSeqDe size_hint ---

    #[test]
    fn bytes_seq_size_hint_matches_length() {
        // Decode as Vec<u8> via the Seq path to exercise BytesSeqDe.
        let node = Dynamic::Bytes(vec![0xAA, 0xBB, 0xCC]);
        let got: Vec<u8> = from_dynamic(&node).expect("bytes→vec");
        assert_eq!(got, vec![0xAAu8, 0xBB, 0xCC]);
    }

    // --- SeqDe and MapDe size_hint via vec/btreemap round-trips ---

    #[test]
    fn seq_size_hint_exercised() {
        let node = Dynamic::Seq(vec![Dynamic::U64(10), Dynamic::U64(20), Dynamic::U64(30)]);
        let got: Vec<u64> = from_dynamic(&node).expect("seq");
        assert_eq!(got, vec![10u64, 20, 30]);
    }

    #[test]
    fn map_size_hint_exercised() {
        let mut m = BTreeMap::new();
        m.insert("a".to_owned(), Dynamic::U64(1));
        m.insert("b".to_owned(), Dynamic::U64(2));
        let node = Dynamic::Map(m);
        let got: BTreeMap<String, u64> = from_dynamic(&node).expect("map");
        assert_eq!(got.len(), 2);
    }

    // --- i64 from non-negative U64 (as_i64 U64 branch) ---

    #[test]
    fn i64_from_small_u64_round_trips() {
        let got = from_dynamic::<i64>(&Dynamic::U64(100)).expect("i64 from U64");
        assert_eq!(got, 100i64);
    }

    // --- u64 from non-negative I64 (as_u64 I64 branch) ---

    #[test]
    fn u64_from_nonneg_i64_round_trips() {
        let got = from_dynamic::<u64>(&Dynamic::I64(50)).expect("u64 from I64");
        assert_eq!(got, 50u64);
    }

    // --- deserialize_ignored_any ---

    #[test]
    fn ignored_any_always_succeeds() {
        // serde's IgnoredAny type uses deserialize_ignored_any.
        use serde::de::IgnoredAny;
        for node in [
            Dynamic::Null,
            Dynamic::Bool(true),
            Dynamic::U64(1),
            Dynamic::I64(-1),
            Dynamic::F64(0.0),
            Dynamic::String("s".to_owned()),
            Dynamic::Bytes(vec![1]),
        ] {
            from_dynamic::<IgnoredAny>(&node).expect("ignored_any");
        }
    }
}
