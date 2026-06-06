//! Raw-bytes CRUD round-trip tests.
//!
//! Exercises the C ABI directly via the rlib import path. Each
//! test opens a fresh temp DB, runs a sequence of writes /
//! reads / commits, then closes everything.

#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_close, obj_db_t, obj_doc_delete, obj_doc_get, obj_doc_insert, obj_doc_update,
    obj_doc_upsert, obj_free_buffer, obj_open, obj_read_txn_t, obj_txn_begin_read,
    obj_txn_begin_write, obj_txn_commit, obj_txn_end_read, obj_txn_rollback, obj_write_txn_t,
    OBJ_ERR_INVALID_ARG, OBJ_ERR_NOT_FOUND, OBJ_OK,
};

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

/// Open a fresh temp DB and return (`TempDir`, db pointer).
fn open_db(name: &str) -> (TempDir, *mut obj_db_t) {
    let dir = TempDir::new().expect("tmp");
    let cs = path_cstring(&dir.path().join(name));
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK, "obj_open returned {code}");
    assert!(!db.is_null());
    (dir, db)
}

/// Begin a write txn against `db`, asserting success.
fn begin_write(db: *mut obj_db_t) -> *mut obj_write_txn_t {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK, "obj_txn_begin_write returned {code}");
    assert!(!txn.is_null());
    txn
}

/// Begin a read txn against `db`, asserting success.
fn begin_read(db: *mut obj_db_t) -> *mut obj_read_txn_t {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_read(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK, "obj_txn_begin_read returned {code}");
    assert!(!txn.is_null());
    txn
}

/// Insert a doc inside `txn`. Returns the assigned id.
fn insert(txn: *mut obj_write_txn_t, collection: &str, payload: &[u8]) -> u64 {
    let cs = CString::new(collection).expect("non-NUL");
    let mut id: u64 = 0;
    let code = unsafe {
        obj_doc_insert(
            txn,
            cs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_OK, "obj_doc_insert returned {code}");
    assert_ne!(id, 0);
    id
}

/// Get a doc inside `txn`. Returns None on `OBJ_ERR_NOT_FOUND`.
fn get(txn: *mut obj_read_txn_t, collection: &str, id: u64) -> Option<Vec<u8>> {
    let cs = CString::new(collection).expect("non-NUL");
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let code = unsafe { obj_doc_get(txn, cs.as_ptr(), id, &raw mut payload, &raw mut len) };
    match code {
        OBJ_OK => {
            let bytes = unsafe { std::slice::from_raw_parts(payload, len) }.to_vec();
            unsafe { obj_free_buffer(payload, len) };
            Some(bytes)
        }
        OBJ_ERR_NOT_FOUND => None,
        other => panic!("unexpected obj_doc_get code {other}"),
    }
}

#[test]
fn insert_get_round_trip() {
    let (_dir, db) = open_db("crud.obj");
    let txn = begin_write(db);
    let id = insert(txn, "things", b"hello");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let got = get(rtxn, "things", id);
    assert_eq!(got.as_deref(), Some(b"hello".as_slice()));
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn update_changes_payload() {
    let (_dir, db) = open_db("update.obj");
    let txn = begin_write(db);
    let id = insert(txn, "things", b"first");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let txn = begin_write(db);
    let cs = CString::new("things").expect("non-NUL");
    let new_payload = b"second";
    let code = unsafe {
        obj_doc_update(
            txn,
            cs.as_ptr(),
            id,
            new_payload.as_ptr(),
            new_payload.len(),
        )
    };
    assert_eq!(code, OBJ_OK);
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let got = get(rtxn, "things", id);
    assert_eq!(got.as_deref(), Some(b"second".as_slice()));
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn delete_removes_document() {
    let (_dir, db) = open_db("delete.obj");
    let txn = begin_write(db);
    let id = insert(txn, "things", b"goodbye");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let txn = begin_write(db);
    let cs = CString::new("things").expect("non-NUL");
    let code = unsafe { obj_doc_delete(txn, cs.as_ptr(), id) };
    assert_eq!(code, OBJ_OK);
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let got = get(rtxn, "things", id);
    assert_eq!(got, None);
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn update_missing_returns_not_found() {
    let (_dir, db) = open_db("update-missing.obj");
    let txn = begin_write(db);
    let _id = insert(txn, "things", b"present");
    let cs = CString::new("things").expect("non-NUL");
    let code = unsafe { obj_doc_update(txn, cs.as_ptr(), 999, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND, "expected NOT_FOUND, got {code}");
    unsafe { obj_txn_rollback(txn) };
    unsafe { obj_close(db) };
}

#[test]
fn delete_missing_returns_not_found() {
    let (_dir, db) = open_db("delete-missing.obj");
    let txn = begin_write(db);
    let _id = insert(txn, "things", b"present");
    let cs = CString::new("things").expect("non-NUL");
    let code = unsafe { obj_doc_delete(txn, cs.as_ptr(), 999) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    unsafe { obj_txn_rollback(txn) };
    unsafe { obj_close(db) };
}

#[test]
fn upsert_at_specific_id_creates_then_replaces() {
    let (_dir, db) = open_db("upsert.obj");
    let txn = begin_write(db);
    let cs = CString::new("things").expect("non-NUL");
    let code = unsafe { obj_doc_upsert(txn, cs.as_ptr(), 42, b"first".as_ptr(), 5) };
    assert_eq!(code, OBJ_OK);
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let txn = begin_write(db);
    let code = unsafe { obj_doc_upsert(txn, cs.as_ptr(), 42, b"second".as_ptr(), 6) };
    assert_eq!(code, OBJ_OK);
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let got = get(rtxn, "things", 42);
    assert_eq!(got.as_deref(), Some(b"second".as_slice()));
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn insert_with_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("invalid-arg.obj");
    let txn = begin_write(db);
    let mut id: u64 = 0;
    let code = unsafe { obj_doc_insert(txn, ptr::null(), b"x".as_ptr(), 1, &raw mut id) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    unsafe { obj_txn_rollback(txn) };
    unsafe { obj_close(db) };
}

#[test]
fn null_payload_with_zero_len_is_accepted() {
    let (_dir, db) = open_db("zero-len.obj");
    let txn = begin_write(db);
    let cs = CString::new("things").expect("non-NUL");
    let mut id: u64 = 0;
    let code = unsafe { obj_doc_insert(txn, cs.as_ptr(), ptr::null(), 0, &raw mut id) };
    assert_eq!(code, OBJ_OK);
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);
    let got = get(rtxn, "things", id);
    assert_eq!(got.as_deref(), Some(b"".as_slice()));
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn free_buffer_is_null_tolerant() {
    unsafe { obj_free_buffer(ptr::null_mut(), 0) };
}

#[test]
fn data_persists_across_reopen() {
    let dir = TempDir::new().expect("tmp");
    let cs = path_cstring(&dir.path().join("persist.obj"));
    let (id_a, id_b) = {
        let mut db: *mut obj_db_t = ptr::null_mut();
        let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
        assert_eq!(code, OBJ_OK);
        let txn = begin_write(db);
        let id_a = insert(txn, "p", b"A");
        let id_b = insert(txn, "p", b"B");
        let code = unsafe { obj_txn_commit(txn) };
        assert_eq!(code, OBJ_OK);
        unsafe { obj_close(db) };
        (id_a, id_b)
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK);
    let rtxn = begin_read(db);
    assert_eq!(get(rtxn, "p", id_a).as_deref(), Some(b"A".as_slice()));
    assert_eq!(get(rtxn, "p", id_b).as_deref(), Some(b"B".as_slice()));
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}
