//! Postcard wire-shape fixtures.
//!
//! This test pins the postcard wire shape Rust's
//! `#[derive(obj::Document)]` produces for a known logical schema, so
//! any drift in the on-disk byte sequence has a single,
//! machine-verifiable anchor.
//!
//! # Schema
//!
//! `Person { name: String, age: u64 }` — the simplest two-field
//! shape that exercises one variable-length type (`String`) and
//! one varint-encoded scalar (`u64`).
//!
//! For `Person { name: "Ada", age: 36 }` the postcard payload is:
//!
//! - `name` (String, varint length + UTF-8 bytes):
//!     - length 3 → `0x03`
//!     - bytes `b"Ada"` → `0x41 0x64 0x61`
//! - `age` (U64, unsigned LEB128 varint):
//!     - 36 → `0x24` (fits in 1 byte)
//!
//! Total payload: `[0x03, 0x41, 0x64, 0x61, 0x24]` (5 bytes).
//!
//! This sequence is the load-bearing byte-identity fixture. If the
//! derived postcard bytes drift, this fixture catches it.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use obj::Document;
use obj_core::codec::{encode, DocumentHeader, DOC_HEADER_SIZE};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
// allow: `Person` is the fixed wire-format fixture name this test pins; it is local
// to the test and the lint's repetition concern does not apply to test fixtures.
#[allow(clippy::module_name_repetitions)]
struct Person {
    name: String,
    age: u64,
}

impl Document for Person {
    const COLLECTION: &'static str = "people";
    const VERSION: u32 = 1;
}

impl obj::Schema for Person {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("age", obj::DynamicSchema::U64),
        ])
    }
}

/// The byte-identity fixture for `Person { name: "Ada", age: 36 }`
/// where `age` is a `u64` field.
///
/// Bytes: `[0x03, 0x41, 0x64, 0x61, 0x24]` — varint length 3 +
/// "Ada" + unsigned varint 36.
pub const PERSON_ADA_36_POSTCARD_U64: &[u8] = &[0x03, 0x41, 0x64, 0x61, 0x24];

/// Byte-identity fixture for a signed `i64` field, which postcard
/// encodes as a zigzag varint.
///
/// Bytes: `[0x03, 0x41, 0x64, 0x61, 0x48]` — varint length 3 +
/// "Ada" + zigzag varint 36 (`zigzag(36) = 72 = 0x48`).
pub const PERSON_ADA_36_POSTCARD_I64: &[u8] = &[0x03, 0x41, 0x64, 0x61, 0x48];

/// Version-2 schema used by the header-level byte-identity test.
/// Mirrors `Person`'s field layout so the postcard payload bytes are
/// identical to the v1 U64 fixture; only the `type_version` field in
/// the record header differs.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PersonV2 {
    name: String,
    age: u64,
}

impl Document for PersonV2 {
    /// Distinct collection name so the v2 fixture does not collide
    /// with the v1 `Person` collection used elsewhere in this test
    /// file.
    const COLLECTION: &'static str = "people_v2";
    const VERSION: u32 = 2;
}

impl obj::Schema for PersonV2 {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("age", obj::DynamicSchema::U64),
        ])
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PersonI64 {
    name: String,
    age: i64,
}

impl Document for PersonI64 {
    const COLLECTION: &'static str = "people_i64";
    const VERSION: u32 = 1;
}

impl obj::Schema for PersonI64 {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("age", obj::DynamicSchema::I64),
        ])
    }
}

#[test]
fn rust_derive_emits_known_postcard_bytes_u64() {
    let person = Person {
        name: "Ada".to_owned(),
        age: 36,
    };
    let bytes = postcard::to_allocvec(&person).expect("postcard encode");
    assert_eq!(
        bytes, PERSON_ADA_36_POSTCARD_U64,
        "u64 postcard payload drifted from the pinned wire-format fixture"
    );
}

#[test]
fn rust_derive_emits_known_postcard_bytes_i64() {
    let person = PersonI64 {
        name: "Ada".to_owned(),
        age: 36,
    };
    let bytes = postcard::to_allocvec(&person).expect("postcard encode");
    assert_eq!(
        bytes, PERSON_ADA_36_POSTCARD_I64,
        "i64 postcard payload drifted from the pinned wire-format fixture"
    );
}

#[test]
fn rust_derive_round_trips() {
    let person = Person {
        name: "Ada".to_owned(),
        age: 36,
    };
    let bytes = postcard::to_allocvec(&person).expect("encode");
    let back: Person = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(back, person);
}

#[test]
fn db_write_then_read_round_trip() {
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let path = tmpdir.path().join("interop.obj");
    let db = obj::Db::open(&path).expect("open");
    let id = db
        .insert(Person {
            name: "Ada".to_owned(),
            age: 36,
        })
        .expect("insert");
    let got: Person = db.get(id).expect("get").expect("present");
    assert_eq!(
        got,
        Person {
            name: "Ada".to_owned(),
            age: 36,
        }
    );
}

/// Compute the expected on-disk record bytes (header + payload) for
/// `PersonV2 { name: "Ada", age: 36 }` written into a fresh
/// collection.
///
/// `collection_id` is `1` because the catalog allocates ids from
/// `1` upwards and the test inserts into a single fresh collection
/// per DB.
fn person_v2_expected_record_bytes() -> Vec<u8> {
    let doc = PersonV2 {
        name: "Ada".to_owned(),
        age: 36,
    };
    encode(&doc, 1).expect("codec encode")
}

#[test]
fn rust_v2_record_header_bytes_are_pinned() {
    let bytes = person_v2_expected_record_bytes();
    assert_eq!(
        bytes.len(),
        DOC_HEADER_SIZE + PERSON_ADA_36_POSTCARD_U64.len(),
        "expected record = 16-byte header + 5-byte payload"
    );
    assert_eq!(&bytes[0..4], &[0x01, 0x00, 0x00, 0x00]);
    assert_eq!(
        &bytes[4..8],
        &[0x02, 0x00, 0x00, 0x00],
        "type_version field must carry T::VERSION (=2) in LE u32"
    );
    assert_eq!(&bytes[8..12], &[0x05, 0x00, 0x00, 0x00]);
    assert_eq!(&bytes[DOC_HEADER_SIZE..], PERSON_ADA_36_POSTCARD_U64);
    let header = DocumentHeader::read_from(&bytes).expect("header decode");
    assert_eq!(header.collection_id, 1);
    assert_eq!(header.type_version, 2);
    assert_eq!(header.payload_len, 5);
    eprintln!(
        "PersonV2 v2 record bytes (hex): {}",
        bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

#[test]
fn rust_v2_engine_write_matches_codec_encode() {
    let tmpdir = tempfile::tempdir().expect("tempdir");
    let path = tmpdir.path().join("v2_record.obj");
    let db = obj::Db::open(&path).expect("open");
    let _ = db
        .insert(PersonV2 {
            name: "Ada".to_owned(),
            age: 36,
        })
        .expect("insert");
    let dump = db.dump_raw("people_v2", 0).expect("dump_raw");
    let records: Vec<_> = dump.map(|step| step.expect("dump step")).collect();
    assert_eq!(records.len(), 1, "expected exactly one record");
    let record = &records[0];
    assert_eq!(record.header.collection_id, 1);
    assert_eq!(record.header.type_version, 2);
    assert_eq!(record.header.payload_len, 5);
    assert_eq!(record.payload.as_slice(), PERSON_ADA_36_POSTCARD_U64);
    let mut reassembled = Vec::with_capacity(DOC_HEADER_SIZE + record.payload.len());
    record.header.write_to(&mut reassembled);
    reassembled.extend_from_slice(&record.payload);
    let expected = person_v2_expected_record_bytes();
    assert_eq!(
        reassembled, expected,
        "Rust-derive on-disk record drifted from codec::encode output"
    );
}
