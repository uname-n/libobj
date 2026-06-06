//! Negative-path FFI tests for `txn.rs`.
//!
//! Covers:
//!
//! - Null handle / null out-pointer arguments to `obj_txn_begin_write`,
//!   `obj_txn_begin_read`, `obj_txn_commit`, and `obj_txn_end_read`.
//! - Null handle / null collection / null out-pointer to every raw-bytes
//!   CRUD function (`obj_doc_insert`, `obj_doc_get`, `obj_doc_update`,
//!   `obj_doc_delete`, `obj_doc_upsert`).
//! - Invalid `(payload, payload_len)` combinations (null pointer with
//!   non-zero length).
//! - Invalid id `0` on `obj_doc_get`, `obj_doc_update`, `obj_doc_delete`,
//!   `obj_doc_upsert`.
//! - Invalid UTF-8 collection names.
//! - Null handle / null collection / null out-pointer to the indexed
//!   write family (`obj_doc_insert_indexed`, `obj_doc_update_indexed`,
//!   `obj_doc_delete_indexed`).
//! - Invalid id `0` on indexed operations.
//! - Invalid `obj_index_entry_t` combinations (non-null count with null
//!   entries pointer, null key with non-zero `key_len`).

// allow: this test crate exercises the unsafe C ABI directly.
#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_close, obj_db_t, obj_doc_delete, obj_doc_delete_indexed, obj_doc_get, obj_doc_insert,
    obj_doc_insert_indexed, obj_doc_update, obj_doc_update_indexed, obj_doc_upsert,
    obj_index_entry_t, obj_open, obj_read_txn_t, obj_txn_begin_read, obj_txn_begin_write,
    obj_txn_commit, obj_txn_end_read, obj_txn_rollback, obj_write_txn_t, OBJ_ERR_INVALID_ARG,
    OBJ_ERR_UTF8, OBJ_OK,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn open_db(name: &str) -> (TempDir, *mut obj_db_t) {
    let dir = TempDir::new().expect("tmp dir");
    let path = dir.path().join(name);
    let cs = path_cstr(&path);
    let mut db: *mut obj_db_t = ptr::null_mut();
    // SAFETY: cs is a non-null NUL-terminated path string; db is a writable out-pointer.
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK, "obj_open failed with {code}");
    assert!(!db.is_null());
    (dir, db)
}

fn path_cstr(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn begin_write(db: *mut obj_db_t) -> *mut obj_write_txn_t {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    // SAFETY: db is valid; txn is a writable out-pointer.
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK, "obj_txn_begin_write failed with {code}");
    assert!(!txn.is_null());
    txn
}

fn begin_read(db: *mut obj_db_t) -> *mut obj_read_txn_t {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    // SAFETY: db is valid; txn is a writable out-pointer.
    let code = unsafe { obj_txn_begin_read(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK, "obj_txn_begin_read failed with {code}");
    assert!(!txn.is_null());
    txn
}

/// Insert one document and return its id, committing the transaction.
fn insert_one(db: *mut obj_db_t, coll: &str, payload: &[u8]) -> u64 {
    let txn = begin_write(db);
    let cs = CString::new(coll).expect("non-NUL collection");
    let mut id: u64 = 0;
    // SAFETY: txn is valid; cs is NUL-terminated; payload/len are consistent; id is writable.
    let code = unsafe {
        obj_doc_insert(
            txn,
            cs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_OK, "insert_one: obj_doc_insert returned {code}");
    // SAFETY: txn is valid and was not yet committed.
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK, "insert_one: obj_txn_commit returned {code}");
    id
}

// ── obj_txn_begin_write ───────────────────────────────────────────────────────

#[test]
fn begin_write_null_db_returns_invalid_arg() {
    // SAFETY: deliberately passing null db; out_txn is a valid writable pointer so
    // the implementation can zero it before returning.
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(ptr::null_mut(), &raw mut txn) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn begin_write_null_out_txn_returns_invalid_arg() {
    let (_dir, db) = open_db("bw-null-out.obj");
    // SAFETY: db is valid; out_txn is deliberately null.
    let code = unsafe { obj_txn_begin_write(db, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: db still open; close it cleanly.
    unsafe { obj_close(db) };
}

// ── obj_txn_commit ────────────────────────────────────────────────────────────

#[test]
fn commit_null_txn_returns_invalid_arg() {
    // SAFETY: deliberately passing null; the function checks for null before dereferencing.
    let code = unsafe { obj_txn_commit(ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

// ── obj_txn_begin_read ────────────────────────────────────────────────────────

#[test]
fn begin_read_null_db_returns_invalid_arg() {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    // SAFETY: deliberately passing null db.
    let code = unsafe { obj_txn_begin_read(ptr::null_mut(), &raw mut txn) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn begin_read_null_out_txn_returns_invalid_arg() {
    let (_dir, db) = open_db("br-null-out.obj");
    // SAFETY: db is valid; out_txn is deliberately null.
    let code = unsafe { obj_txn_begin_read(db, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_txn_end_read (already tested for null-tolerance in txn_isolation) ────

// ── obj_doc_insert ────────────────────────────────────────────────────────────

#[test]
fn insert_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: txn deliberately null; collection, payload, out_id all valid.
    let code =
        unsafe { obj_doc_insert(ptr::null_mut(), cs.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn insert_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-null-col.obj");
    let txn = begin_write(db);
    let mut id: u64 = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe { obj_doc_insert(txn, ptr::null(), b"x".as_ptr(), 1, &raw mut id) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn is valid but unused; roll back cleanly.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_null_out_id_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-null-out.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: out_id deliberately null; other args valid.
    let code = unsafe { obj_doc_insert(txn, cs.as_ptr(), b"x".as_ptr(), 1, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn is valid but unused.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_null_payload_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-bad-payload.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: payload=null but payload_len=1 — the invalid combination the fn guards against.
    let code = unsafe { obj_doc_insert(txn, cs.as_ptr(), ptr::null(), 1, &raw mut id) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("ins-bad-utf8.obj");
    let txn = begin_write(db);
    // Invalid UTF-8 bytes: 0xFF cannot appear in a valid UTF-8 sequence.
    let bad: &[u8] = b"\xff\xfe invalid\0";
    let mut id: u64 = 0;
    // SAFETY: bad is a non-null NUL-terminated byte string (not valid UTF-8); other args valid.
    let code = unsafe {
        obj_doc_insert(
            txn,
            bad.as_ptr().cast(),
            b"x".as_ptr(),
            1,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_get ───────────────────────────────────────────────────────────────

#[test]
fn get_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let mut p: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: txn deliberately null; other args valid.
    let code =
        unsafe { obj_doc_get(ptr::null_mut(), cs.as_ptr(), 1, &raw mut p, &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn get_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("get-null-col.obj");
    let rtxn = begin_read(db);
    let mut p: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code =
        unsafe { obj_doc_get(rtxn, ptr::null(), 1, &raw mut p, &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn get_null_out_payload_returns_invalid_arg() {
    let (_dir, db) = open_db("get-null-payload-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut len: usize = 0;
    // SAFETY: out_payload deliberately null; other args valid.
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), 1, ptr::null_mut(), &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn get_null_out_len_returns_invalid_arg() {
    let (_dir, db) = open_db("get-null-len-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut p: *mut u8 = ptr::null_mut();
    // SAFETY: out_payload_len deliberately null; other args valid.
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), 1, &raw mut p, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn get_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("get-id-zero.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut p: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: id=0 is the specific sentinel the fn rejects; all pointer args valid.
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), 0, &raw mut p, &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn get_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("get-bad-utf8.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let mut p: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 (valid), pointers valid.
    let code = unsafe {
        obj_doc_get(rtxn, bad.as_ptr().cast(), 1, &raw mut p, &raw mut len)
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_update ────────────────────────────────────────────────────────────

#[test]
fn update_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: txn deliberately null.
    let code = unsafe { obj_doc_update(ptr::null_mut(), cs.as_ptr(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn update_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-null-col.obj");
    let txn = begin_write(db);
    // SAFETY: collection deliberately null.
    let code = unsafe { obj_doc_update(txn, ptr::null(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-id-zero.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: id=0 is the sentinel the fn rejects; other args valid.
    let code = unsafe { obj_doc_update(txn, cs.as_ptr(), 0, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_null_payload_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-bad-payload.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: payload=null but len=1 — the specific invalid combo the fn guards against.
    let code = unsafe { obj_doc_update(txn, cs.as_ptr(), 1, ptr::null(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("upd-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 valid; payload valid.
    let code =
        unsafe { obj_doc_update(txn, bad.as_ptr().cast(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_delete ────────────────────────────────────────────────────────────

#[test]
fn delete_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: txn deliberately null.
    let code = unsafe { obj_doc_delete(ptr::null_mut(), cs.as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn delete_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("del-null-col.obj");
    let txn = begin_write(db);
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe { obj_doc_delete(txn, ptr::null(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn delete_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("del-id-zero.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: id=0 is the sentinel the fn rejects; other args valid.
    let code = unsafe { obj_doc_delete(txn, cs.as_ptr(), 0) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn delete_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("del-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 valid.
    let code = unsafe { obj_doc_delete(txn, bad.as_ptr().cast(), 1) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_upsert ────────────────────────────────────────────────────────────

#[test]
fn upsert_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: txn deliberately null.
    let code = unsafe { obj_doc_upsert(ptr::null_mut(), cs.as_ptr(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn upsert_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("ups-null-col.obj");
    let txn = begin_write(db);
    // SAFETY: collection deliberately null.
    let code = unsafe { obj_doc_upsert(txn, ptr::null(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn upsert_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("ups-id-zero.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: id=0 is the sentinel the fn rejects; other args valid.
    let code = unsafe { obj_doc_upsert(txn, cs.as_ptr(), 0, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn upsert_null_payload_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("ups-bad-payload.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: payload=null but len=1 — the invalid combo the fn guards against.
    let code = unsafe { obj_doc_upsert(txn, cs.as_ptr(), 1, ptr::null(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn upsert_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("ups-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 valid; payload valid.
    let code =
        unsafe { obj_doc_upsert(txn, bad.as_ptr().cast(), 1, b"x".as_ptr(), 1) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_insert_indexed ────────────────────────────────────────────────────

#[test]
fn insert_indexed_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: txn deliberately null; no entries (null ptr + count 0 is allowed).
    let code = unsafe {
        obj_doc_insert_indexed(
            ptr::null_mut(),
            cs.as_ptr(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn insert_indexed_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-idx-null-col.obj");
    let txn = begin_write(db);
    let mut id: u64 = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            ptr::null(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_indexed_null_out_id_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-idx-null-out.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: out_id deliberately null; other args valid.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_indexed_null_payload_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-idx-bad-payload.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: payload=null but len=3 — the invalid combo read_indexed_inputs guards against.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            ptr::null(),
            3,
            ptr::null(),
            0,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_indexed_nonzero_entry_count_null_entries_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-idx-null-entries.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: entry_count=1 but entries=null — entries_from_c rejects this.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            1,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_indexed_entry_null_key_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("ins-idx-entry-bad-key.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    let index_name = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    // One entry: index_name valid, but key=null with key_len=4 — entries_from_c rejects this.
    let entry = obj_index_entry_t {
        index_name: index_name.as_ptr(),
        key: ptr::null(),
        key_len: 4,
    };
    // SAFETY: entry is a valid obj_index_entry_t on the stack; entry_count=1 matches.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            b"x".as_ptr(),
            1,
            &raw const entry,
            1,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn insert_indexed_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("ins-idx-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let mut id: u64 = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            bad.as_ptr().cast(),
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_update_indexed ────────────────────────────────────────────────────

#[test]
fn update_indexed_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: txn deliberately null; no entries.
    let code = unsafe {
        obj_doc_update_indexed(
            ptr::null_mut(),
            cs.as_ptr(),
            1,
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null(),
            0,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn update_indexed_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-idx-null-col.obj");
    let txn = begin_write(db);
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe {
        obj_doc_update_indexed(
            txn,
            ptr::null(),
            1,
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null(),
            0,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_indexed_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-idx-id-zero.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: id=0 is the sentinel the fn rejects; other args valid.
    let code = unsafe {
        obj_doc_update_indexed(
            txn,
            cs.as_ptr(),
            0,
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null(),
            0,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_indexed_null_payload_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("upd-idx-bad-payload.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: payload=null but len=2 — read_indexed_inputs rejects this.
    let code = unsafe {
        obj_doc_update_indexed(
            txn,
            cs.as_ptr(),
            1,
            ptr::null(),
            2,
            ptr::null(),
            0,
            ptr::null(),
            0,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn update_indexed_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("upd-idx-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 valid; payload valid.
    let code = unsafe {
        obj_doc_update_indexed(
            txn,
            bad.as_ptr().cast(),
            1,
            b"x".as_ptr(),
            1,
            ptr::null(),
            0,
            ptr::null(),
            0,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_doc_delete_indexed ────────────────────────────────────────────────────

#[test]
fn delete_indexed_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: txn deliberately null.
    let code =
        unsafe { obj_doc_delete_indexed(ptr::null_mut(), cs.as_ptr(), 1, ptr::null(), 0) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn delete_indexed_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("del-idx-null-col.obj");
    let txn = begin_write(db);
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe { obj_doc_delete_indexed(txn, ptr::null(), 1, ptr::null(), 0) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn delete_indexed_id_zero_returns_invalid_arg() {
    let (_dir, db) = open_db("del-idx-id-zero.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: id=0 is the sentinel the fn rejects; other args valid.
    let code = unsafe { obj_doc_delete_indexed(txn, cs.as_ptr(), 0, ptr::null(), 0) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn delete_indexed_nonzero_remove_count_null_entries_returns_invalid_arg() {
    let (_dir, db) = open_db("del-idx-null-entries.obj");
    let txn = begin_write(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: remove_count=1 but entries=null — entries_from_c rejects this.
    let code = unsafe { obj_doc_delete_indexed(txn, cs.as_ptr(), 1, ptr::null(), 1) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn delete_indexed_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("del-idx-bad-utf8.obj");
    let txn = begin_write(db);
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); id=1 valid; no entries.
    let code = unsafe { obj_doc_delete_indexed(txn, bad.as_ptr().cast(), 1, ptr::null(), 0) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: txn still valid.
    unsafe { obj_txn_rollback(txn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── cross-check: valid id from one collection is NOT found in another ─────────

#[test]
fn get_known_id_in_wrong_collection_returns_not_found() {
    use obj::OBJ_ERR_NOT_FOUND;

    let (_dir, db) = open_db("cross-coll.obj");
    let id = insert_one(db, "alpha", b"doc");

    let rtxn = begin_read(db);
    let cs = CString::new("beta").expect("non-NUL");
    let mut p: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: rtxn valid; cs is NUL-terminated UTF-8; id>0; pointers valid.
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), id, &raw mut p, &raw mut len) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}
