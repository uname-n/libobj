//! Iterator + `find_unique` + count tests.
//!
//! Tests focus on the `obj_iter_*` + `obj_count_all` paths. The
//! `obj_iter_index_range` / `obj_find_unique` / `obj_count_index_range`
//! happy paths require a populated index, which is built by the
//! typed Rust API; the raw-bytes C ABI doesn't expose index
//! declaration. We seed a small fixture via the obj public Rust
//! crate, then exercise the C ABI.

// allow: this test crate exercises the unsafe C ABI directly, so every FFI call site is `unsafe`.
#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use obj_engine::{Db, Document, IndexSpec};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use obj::{
    obj_buf_free, obj_close, obj_count_all, obj_count_index_range, obj_db_t, obj_doc_insert_raw,
    obj_find_unique, obj_iter_all, obj_iter_free, obj_iter_index_range, obj_iter_next, obj_iter_t,
    obj_open, obj_read_txn_t, obj_txn_begin_read, obj_txn_begin_write, obj_txn_commit,
    obj_txn_end_read, obj_write_txn_t, ObjBound, OBJ_ERR_NOT_FOUND, OBJ_OK,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Customer {
    email: String,
    age: u32,
}

impl obj_engine::Schema for Customer {
    fn schema() -> obj_engine::DynamicSchema {
        obj_engine::DynamicSchema::map([
            ("email", obj_engine::DynamicSchema::String),
            ("age", obj_engine::DynamicSchema::U64),
        ])
    }
}

impl Document for Customer {
    const COLLECTION: &'static str = "customers";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        let mut specs = Vec::new();
        if let Ok(spec) = IndexSpec::unique("by_email", "email") {
            specs.push(spec);
        }
        if let Ok(spec) = IndexSpec::standard("by_age", "age") {
            specs.push(spec);
        }
        specs
    }
}

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn open_db_c(path: &Path) -> *mut obj_db_t {
    let cs = path_cstring(path);
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK);
    db
}

fn begin_read(db: *mut obj_db_t) -> *mut obj_read_txn_t {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_read(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

fn begin_write(db: *mut obj_db_t) -> *mut obj_write_txn_t {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

/// Drive the iterator to exhaustion, collecting `(id, payload)`.
fn drain(iter: *mut obj_iter_t) -> Vec<(u64, Vec<u8>)> {
    let mut out = Vec::new();
    loop {
        let mut id: u64 = 0;
        let mut payload: *mut u8 = ptr::null_mut();
        let mut len: usize = 0;
        let code = unsafe { obj_iter_next(iter, &raw mut id, &raw mut payload, &raw mut len) };
        match code {
            OBJ_OK => {
                let bytes = unsafe { std::slice::from_raw_parts(payload, len) }.to_vec();
                unsafe { obj_buf_free(payload) };
                out.push((id, bytes));
            }
            OBJ_ERR_NOT_FOUND => break,
            other => panic!("obj_iter_next returned {other}"),
        }
    }
    out
}

#[test]
fn iter_all_visits_every_inserted_doc() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("iter.obj");

    let db = open_db_c(&path);
    let txn = begin_write(db);
    let collection = CString::new("things").expect("non-NUL");
    let mut ids: Vec<u64> = Vec::new();
    for i in 0..5u8 {
        let payload = [i];
        let mut id: u64 = 0;
        let code = unsafe {
            obj_doc_insert_raw(
                txn,
                collection.as_ptr(),
                payload.as_ptr(),
                payload.len(),
                &raw mut id,
            )
        };
        assert_eq!(code, OBJ_OK);
        ids.push(id);
    }
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    let code = unsafe { obj_iter_all(rtxn, collection.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);
    let entries = drain(iter);
    unsafe { obj_iter_free(iter) };
    assert_eq!(entries.len(), 5);
    for (i, (id, payload)) in entries.iter().enumerate() {
        assert_eq!(*id, ids[i]);
        let expected_byte = u8::try_from(i).expect("0..5 fits in u8");
        assert_eq!(payload.as_slice(), [expected_byte].as_slice());
    }
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn count_all_matches_inserted() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("count.obj");
    let db = open_db_c(&path);
    let txn = begin_write(db);
    let collection = CString::new("c").expect("non-NUL");
    for _ in 0..7 {
        let mut id: u64 = 0;
        let code =
            unsafe { obj_doc_insert_raw(txn, collection.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
        assert_eq!(code, OBJ_OK);
    }
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let mut count: u64 = 0;
    let code = unsafe { obj_count_all(rtxn, collection.as_ptr(), &raw mut count) };
    assert_eq!(code, OBJ_OK);
    assert_eq!(count, 7);
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
// allow: end-to-end C-API scenario test; splitting the round-trip would obscure what it documents.
#[allow(clippy::too_many_lines)]
fn find_unique_locates_doc_via_typed_index() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("find.obj");

    let (target_id, target_email) = {
        let db = Db::open(&path).expect("open");
        let id_a = db
            .insert(Customer {
                email: "ada@example.com".to_string(),
                age: 36,
            })
            .expect("insert");
        let _id_b = db
            .insert(Customer {
                email: "grace@example.com".to_string(),
                age: 85,
            })
            .expect("insert");
        (id_a, "ada@example.com".to_string())
    };

    let key_bytes = encode_unique_string_key(&target_email);

    let db = open_db_c(&path);
    let rtxn = begin_read(db);
    let collection = CString::new("customers").expect("non-NUL");
    let index = CString::new("by_email").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let code = unsafe {
        obj_find_unique(
            rtxn,
            collection.as_ptr(),
            index.as_ptr(),
            key_bytes.as_ptr(),
            key_bytes.len(),
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_OK);
    assert_eq!(id, target_id.get());
    let bytes = unsafe { std::slice::from_raw_parts(payload, len) }.to_vec();
    unsafe { obj_buf_free(payload) };
    assert!(!bytes.is_empty());

    let other_key = encode_unique_string_key("nobody@example.com");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let code = unsafe {
        obj_find_unique(
            rtxn,
            collection.as_ptr(),
            index.as_ptr(),
            other_key.as_ptr(),
            other_key.len(),
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    assert!(payload.is_null());

    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_walks_typed_index() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("range.obj");

    let _ids: Vec<obj_engine::Id> = {
        let db = Db::open(&path).expect("open");
        let mut ids = Vec::new();
        for (email, age) in [
            ("a@x.com", 20u32),
            ("b@x.com", 30u32),
            ("c@x.com", 40u32),
            ("d@x.com", 50u32),
        ] {
            ids.push(
                db.insert(Customer {
                    email: email.to_string(),
                    age,
                })
                .expect("insert"),
            );
        }
        ids
    };

    let db = open_db_c(&path);
    let rtxn = begin_read(db);
    let collection = CString::new("customers").expect("non-NUL");
    let index = CString::new("by_age").expect("non-NUL");

    let lower = encode_standard_u32_key(25);
    let upper = encode_standard_u32_key(45);

    let mut iter: *mut obj_iter_t = ptr::null_mut();
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            collection.as_ptr(),
            index.as_ptr(),
            ObjBound { ptr: lower.as_ptr(), len: lower.len(), inclusive: true },
            ObjBound { ptr: upper.as_ptr(), len: upper.len(), inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_OK);
    let entries = drain(iter);
    unsafe { obj_iter_free(iter) };

    assert_eq!(entries.len(), 2, "got {entries:?}");

    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_matches_iter_count() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("count-range.obj");
    {
        let db = Db::open(&path).expect("open");
        for (email, age) in [
            ("a@x.com", 10u32),
            ("b@x.com", 20u32),
            ("c@x.com", 30u32),
            ("d@x.com", 40u32),
        ] {
            db.insert(Customer {
                email: email.to_string(),
                age,
            })
            .expect("insert");
        }
    }
    let db = open_db_c(&path);
    let rtxn = begin_read(db);
    let collection = CString::new("customers").expect("non-NUL");
    let index = CString::new("by_age").expect("non-NUL");
    let lower = encode_standard_u32_key(15);
    let upper = encode_standard_u32_key(35);
    let mut count: u64 = 0;
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            collection.as_ptr(),
            index.as_ptr(),
            ObjBound { ptr: lower.as_ptr(), len: lower.len(), inclusive: true },
            ObjBound { ptr: upper.as_ptr(), len: upper.len(), inclusive: true },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_OK);
    assert_eq!(count, 2, "ages 20 + 30 should match the range");
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn iter_free_is_null_tolerant() {
    unsafe { obj_iter_free(ptr::null_mut()) };
}

/// Encode a `Dynamic::String(s)` value as a Unique index key
/// would land on disk. The function reaches into `obj_core`'s
/// `encode_index_key` helper through a transient `IndexSpec`.
fn encode_unique_string_key(s: &str) -> Vec<u8> {
    let spec = IndexSpec::unique("key", "x").expect("valid unique spec");
    let encoded = obj_core::index::encode_index_key(
        &spec,
        &[obj_core::codec::Dynamic::String(s.to_string())],
    )
    .expect("encode_index_key");
    encoded.as_bytes().to_vec()
}

/// Encode a `Dynamic::U32(n)` as a Standard index key prefix
/// (no id suffix). The libobj range path widens bounds via the
/// non-Unique kind rules, so we just supply the user-key portion.
fn encode_standard_u32_key(n: u32) -> Vec<u8> {
    let encoded = obj_core::index::encode_field(&obj_core::codec::Dynamic::U64(u64::from(n)))
        .expect("encode_field");
    encoded.as_bytes().to_vec()
}
