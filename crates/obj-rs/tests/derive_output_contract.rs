//! `#[derive(obj::Document)]` generated-code contract.
//!
//! `obj-derive` is a frozen 1.0 surface, but `public-api/obj-derive.txt`
//! is only two lines (the macro entry point); the freeze gate cannot
//! see the GENERATED code, which is itself a format contract. The
//! generated `Document` / `Schema` impls decide:
//!
//! - the `COLLECTION` name and schema `VERSION`,
//! - the exact `indexes()` list (each `IndexSpec`'s name / kind /
//!   key-paths, and their order),
//! - the derived `impl Schema` field layout,
//! - and — transitively, via the `serde::Serialize` impl the derive
//!   relies on — the on-disk byte encoding of a stored record.
//!
//! This test pins all of those for one representative struct that
//! exercises every index kind (standard + unique + each + composite)
//! and a non-default `#[obj(version = N)]`. A future derive change
//! that alters index emission, the schema version, or the encoded
//! bytes will break this test even when the macro's public signature
//! is unchanged.
//!
//! Only the public `obj` API is used, plus `obj_core::codec::{encode,
//! decode}` (a normal dependency of `obj`, already reachable from
//! these tests) to assert the BYTE-LEVEL on-disk record — the one
//! contract a self-consistent encode/decode round-trip alone cannot
//! pin.

#![forbid(unsafe_code)]

use obj::{Document, DynamicSchema, EnumVariantSchema, IndexKind, Schema};
use obj_core::codec::{decode, encode, DOC_HEADER_SIZE};
use obj_core::pager::checksum::crc32c;
use serde::{Deserialize, Serialize};

// allow: the field set is a fixed golden-bytes fixture exercising every index kind;
// the shared suffixes are intentional and renaming them would re-pin the encoding.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "accounts")]
#[obj(index = ("owner", "opened_at"))]
struct Account {
    #[obj(index = unique)]
    owner: String,

    #[obj(index)]
    region: u32,

    #[obj(index = each)]
    tags: Vec<String>,

    opened_at: u64,
    balance_cents: u64,
}

/// A fixed sample value. Every byte of its encoding is pinned below,
/// so this MUST stay constant — changing it requires re-pinning the
/// expected bytes.
fn sample() -> Account {
    Account {
        owner: "ada".to_owned(),
        region: 7,
        tags: vec!["gold".to_owned(), "beta".to_owned()],
        opened_at: 1_700_000_000,
        balance_cents: 4_242,
    }
}

/// The `collection_id` the codec stamps into the record header. The
/// catalog assigns this at runtime; for a byte-level pin we choose a
/// fixed value and assert against it.
const SAMPLE_COLLECTION_ID: u32 = 3;

#[test]
fn collection_and_version_constants_are_pinned() {
    assert_eq!(<Account as Document>::COLLECTION, "accounts");
    assert_eq!(<Account as Document>::VERSION, 2);
}

#[test]
fn indexes_list_is_pinned_exactly() {
    let specs = <Account as Document>::indexes();
    assert_eq!(specs.len(), 4, "owner(unique) + region + tags + composite");

    assert_eq!(specs[0].name, "owner");
    assert_eq!(specs[0].kind, IndexKind::Unique);
    assert_eq!(specs[0].key_paths, vec!["owner".to_owned()]);

    assert_eq!(specs[1].name, "region");
    assert_eq!(specs[1].kind, IndexKind::Standard);
    assert_eq!(specs[1].key_paths, vec!["region".to_owned()]);

    assert_eq!(specs[2].name, "tags");
    assert_eq!(specs[2].kind, IndexKind::Each);
    assert_eq!(specs[2].key_paths, vec!["tags".to_owned()]);

    assert_eq!(specs[3].name, "owner__opened_at");
    assert_eq!(specs[3].kind, IndexKind::Composite);
    assert_eq!(
        specs[3].key_paths,
        vec!["owner".to_owned(), "opened_at".to_owned()],
    );
}

#[test]
fn historical_schemas_uses_trait_default() {
    let history = <Account as Document>::historical_schemas();
    assert!(
        history.is_empty(),
        "derive must NOT emit a historical_schemas() override; got {history:?}",
    );
}

#[test]
fn current_schema_matches_field_layout() {
    let schema = <Account as Schema>::schema();
    assert_eq!(
        schema,
        DynamicSchema::map([
            ("owner", DynamicSchema::String),
            ("region", DynamicSchema::U64),
            ("tags", DynamicSchema::seq(DynamicSchema::String)),
            ("opened_at", DynamicSchema::U64),
            ("balance_cents", DynamicSchema::U64),
        ]),
    );
}

#[test]
fn encoded_record_round_trips() {
    let value = sample();
    let bytes = encode(&value, SAMPLE_COLLECTION_ID).expect("encode");
    let back: Account = decode(&bytes, SAMPLE_COLLECTION_ID).expect("decode");
    assert_eq!(back, value, "decode(encode(x)) must equal x");
}

#[test]
fn encoded_record_header_fields_are_pinned() {
    let value = sample();
    let bytes = encode(&value, SAMPLE_COLLECTION_ID).expect("encode");
    assert!(bytes.len() > DOC_HEADER_SIZE);
    let collection_id = u32::from_le_bytes(bytes[0..4].try_into().expect("4"));
    let type_version = u32::from_le_bytes(bytes[4..8].try_into().expect("4"));
    let payload_len = u32::from_le_bytes(bytes[8..12].try_into().expect("4"));
    let payload_crc32c = u32::from_le_bytes(bytes[12..16].try_into().expect("4"));
    let payload = &bytes[DOC_HEADER_SIZE..];

    assert_eq!(collection_id, SAMPLE_COLLECTION_ID);
    assert_eq!(type_version, 2, "header pins Account::VERSION = 2");
    assert_eq!(payload_len as usize, payload.len());
    assert_eq!(payload_crc32c, crc32c(payload), "header CRC pins payload");
}

#[test]
fn encoded_payload_bytes_are_pinned() {
    let value = sample();
    let bytes = encode(&value, SAMPLE_COLLECTION_ID).expect("encode");
    let payload = &bytes[DOC_HEADER_SIZE..];
    assert_eq!(
        payload, EXPECTED_PAYLOAD,
        "encoded payload diverged from the frozen golden bytes",
    );
}

/// A nested struct used as `Option<Nested>` to confirm the
/// nested-struct Option path is byte-identical (it takes the syntactic
/// path, but the Some payload still delegates to
/// `<Nested as Schema>::schema()`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct Nested {
    label: String,
    count: u64,
}

/// Derives `Document` (and thus `Schema`) with `Option<scalar>` fields
/// plus an `Option<Nested>` to pin the nested-struct path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct OptionalFields {
    badge: Option<u64>,
    nickname: Option<String>,
    parent: Option<Nested>,
}

/// The two-variant enum schema for `Option<inner>` — postcard's
/// `None`=0 (Null payload) / `Some`=1 (inner schema). This is exactly
/// the structure `impl<T: Schema> Schema for Option<T>` produces.
fn optional_schema(inner: DynamicSchema) -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "None", DynamicSchema::Null),
        EnumVariantSchema::new(1, "Some", inner),
    ])
}

#[test]
fn derived_option_scalar_schema_matches_two_variant_enum() {
    let schema = <OptionalFields as Schema>::schema();
    assert_eq!(
        schema,
        DynamicSchema::map([
            ("badge", optional_schema(DynamicSchema::U64)),
            ("nickname", optional_schema(DynamicSchema::String)),
            ("parent", optional_schema(<Nested as Schema>::schema())),
        ]),
    );
}

#[test]
fn derived_option_nested_struct_equals_blanket() {
    let derived = <OptionalFields as Schema>::schema();
    let parent_field = match derived {
        DynamicSchema::Map(fields) => fields
            .into_iter()
            .find(|(name, _)| name == "parent")
            .map(|(_, schema)| schema)
            .expect("parent field present"),
        other => panic!("expected Map, got {other:?}"),
    };
    assert_eq!(
        parent_field,
        <Option<Nested> as Schema>::schema(),
        "derived Option<Nested> must equal the Option<T>: Schema blanket",
    );
}

#[test]
fn derived_option_scalar_round_trips() {
    let some = OptionalFields {
        badge: Some(7),
        nickname: Some("ada".to_owned()),
        parent: Some(Nested {
            label: "root".to_owned(),
            count: 3,
        }),
    };
    let none = OptionalFields {
        badge: None,
        nickname: None,
        parent: None,
    };
    for value in [some, none] {
        let bytes = encode(&value, SAMPLE_COLLECTION_ID).expect("encode");
        let back: OptionalFields = decode(&bytes, SAMPLE_COLLECTION_ID).expect("decode");
        assert_eq!(back, value, "Option<scalar> round-trips");
    }
}

/// Frozen postcard payload of `sample()`. Captured from the current
/// build; a change here means the on-disk encoding moved.
///
/// Breakdown (postcard, declaration order):
///   `[3]`                         owner len = 3
///   `[97, 100, 97]`               b"ada"
///   `[7]`                         region = 7u32 (varint)
///   `[2]`                         tags len = 2
///   `[4, 103, 111, 108, 100]`     len(4) + b"gold"
///   `[4, 98, 101, 116, 97]`       len(4) + b"beta"
///   `[128, 226, 207, 170, 6]`     `opened_at` = `1_700_000_000u64` (varint)
///   `[146, 33]`                   `balance_cents` = `4_242u64` (varint)
const EXPECTED_PAYLOAD: &[u8] = &[
    3, 97, 100, 97, 7, 2, 4, 103, 111, 108, 100, 4, 98, 101, 116, 97, 128, 226, 207, 170, 6, 146,
    33,
];
