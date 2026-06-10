//! Negative-path FFI tests for `queries.rs`.
//!
//! Covers:
//!
//! - `obj_iter_all`: null txn, null collection, null `out_iter`.
//! - `obj_iter_index_range`: null `txn`/`collection`/`index_name`/`out_iter`,
//!   invalid `(ptr=null, len>0)` lower/upper bounds, invalid UTF-8.
//! - `obj_iter_next`: null iter, null `out_id`, null `out_payload`, null
//!   `out_payload_len`, iterator exhaustion.
//! - `obj_iter_free`: null-tolerance.
//! - `obj_find_unique`: null `txn`/`collection`/`index_name`, null out-pointers,
//!   `(key=null, key_len>0)`, invalid UTF-8 collection/index names, and
//!   unknown collection / unknown index (returns `OBJ_ERR_NOT_FOUND`).
//! - `obj_count_all`: null txn, null collection, null `out_count`,
//!   invalid UTF-8, unknown collection.
//! - `obj_count_index_range`: null `txn`/`collection`/`index_name`/`out_count`,
//!   invalid `(ptr=null, len>0)` bounds, invalid UTF-8.
//! - `obj_stat`: null db, null `out_stat`.
//! - `obj_integrity_check`: null db, null `out_report`.
//! - `obj_integrity_report_failure_at`: null report, null `out_string`,
//!   null `out_string_len`, out-of-range index.
//! - `obj_backup_to`: null db, null dest, invalid UTF-8 dest,
//!   destination that already exists.

// allow: this test crate exercises the unsafe C ABI directly.
#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_backup_to, obj_close, obj_count_all, obj_count_index_range, obj_db_t, obj_doc_insert_raw,
    obj_find_unique, obj_free_buffer, obj_integrity_check, obj_integrity_report_failure_at,
    obj_integrity_report_free, obj_integrity_report_t, obj_iter_all, obj_iter_free,
    obj_iter_index_range, obj_iter_next, obj_iter_t, obj_open, obj_read_txn_t, obj_stat,
    obj_stat_t, obj_txn_begin_read, obj_txn_begin_write, obj_txn_commit, obj_txn_end_read,
    obj_txn_rollback, obj_write_txn_t, ObjBound, OBJ_ERR_INVALID_ARG, OBJ_ERR_NOT_FOUND,
    OBJ_ERR_UTF8, OBJ_OK,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn open_db(name: &str) -> (TempDir, *mut obj_db_t) {
    let dir = TempDir::new().expect("tmp dir");
    let path = dir.path().join(name);
    let cs = path_cstr(&path);
    let mut db: *mut obj_db_t = ptr::null_mut();
    // SAFETY: cs is non-null NUL-terminated; db is a writable out-pointer.
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

/// Insert one doc and commit; returns an empty read txn and the inserted id.
fn seed_one_doc(db: *mut obj_db_t, coll: &str) -> u64 {
    let txn = begin_write(db);
    let cs = CString::new(coll).expect("non-NUL");
    let mut id: u64 = 0;
    // SAFETY: all args valid.
    let code = unsafe { obj_doc_insert_raw(txn, cs.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
    assert_eq!(code, OBJ_OK);
    // SAFETY: txn valid and not yet committed.
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);
    id
}

// ── obj_iter_all ──────────────────────────────────────────────────────────────

#[test]
fn iter_all_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: txn deliberately null; collection and out_iter are valid.
    let code = unsafe { obj_iter_all(ptr::null_mut(), cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn iter_all_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-all-null-col.obj");
    let rtxn = begin_read(db);
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: collection deliberately null; txn and out_iter are valid.
    let code = unsafe { obj_iter_all(rtxn, ptr::null(), &raw mut iter) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_all_null_out_iter_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-all-null-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: out_iter deliberately null; txn and collection are valid.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_all_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("iter-all-bad-utf8.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe { obj_iter_all(rtxn, bad.as_ptr().cast(), &raw mut iter) };
    assert_eq!(code, OBJ_ERR_UTF8);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_all_unknown_collection_returns_not_found() {
    let (_dir, db) = open_db("iter-all-unknown-coll.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("no_such_collection").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: rtxn valid; cs is NUL-terminated UTF-8; out_iter is writable.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_iter_index_range ──────────────────────────────────────────────────────

#[test]
fn iter_index_range_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: txn deliberately null; other args valid.
    let code = unsafe {
        obj_iter_index_range(
            ptr::null_mut(),
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn iter_index_range_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-range-null-col.obj");
    let rtxn = begin_read(db);
    let idx = CString::new("by_x").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            ptr::null(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_null_index_name_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-range-null-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: index_name deliberately null; other args valid.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs.as_ptr(),
            ptr::null(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_null_out_iter_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-range-null-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    // SAFETY: out_iter deliberately null; other args valid.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ptr::null_mut(),
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_null_lower_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-range-bad-lower.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: lower.ptr=null but lower.len=1 — invalid combo bytes_to_bound guards against.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 1, inclusive: true },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_null_upper_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-range-bad-upper.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let lower_key = b"\x01";
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: upper.ptr=null but upper.len=1 — invalid combo bytes_to_bound guards against.
    // lower is a valid non-null 1-byte slice.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: lower_key.as_ptr(), len: lower_key.len(), inclusive: true },
            ObjBound { ptr: ptr::null(), len: 1, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("iter-range-bad-utf8-col.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let idx = CString::new("by_x").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            bad.as_ptr().cast(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_index_range_invalid_utf8_index_name_returns_utf8_error() {
    let (_dir, db) = open_db("iter-range-bad-utf8-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let bad: &[u8] = b"\xff\xfe\0";
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8) for index_name; other args valid.
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs.as_ptr(),
            bad.as_ptr().cast(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    assert!(iter.is_null());
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_iter_next ─────────────────────────────────────────────────────────────

#[test]
fn iter_next_null_iter_returns_invalid_arg() {
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: iter deliberately null; other out-pointers valid.
    let code = unsafe {
        obj_iter_next(
            ptr::null_mut(),
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn iter_next_null_out_id_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-next-null-id.obj");
    seed_one_doc(db, "c");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: rtxn valid; cs NUL-terminated; out_iter writable.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: iter valid from obj_iter_all; out_id deliberately null.
    let code = unsafe { obj_iter_next(iter, ptr::null_mut(), &raw mut payload, &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: iter valid; free it.
    unsafe { obj_iter_free(iter) };
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_next_null_out_payload_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-next-null-payload.obj");
    seed_one_doc(db, "c");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: rtxn valid; cs NUL-terminated; out_iter writable.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);
    let mut id: u64 = 0;
    let mut len: usize = 0;
    // SAFETY: iter valid; out_payload deliberately null.
    let code = unsafe { obj_iter_next(iter, &raw mut id, ptr::null_mut(), &raw mut len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: iter valid.
    unsafe { obj_iter_free(iter) };
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_next_null_out_len_returns_invalid_arg() {
    let (_dir, db) = open_db("iter-next-null-len.obj");
    seed_one_doc(db, "c");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: rtxn valid; cs NUL-terminated; out_iter writable.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    // SAFETY: iter valid; out_payload_len deliberately null.
    let code = unsafe { obj_iter_next(iter, &raw mut id, &raw mut payload, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: iter valid.
    unsafe { obj_iter_free(iter) };
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_next_exhaustion_returns_not_found() {
    let (_dir, db) = open_db("iter-exhaust.obj");
    seed_one_doc(db, "c");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    // SAFETY: rtxn valid; cs NUL-terminated; out_iter writable.
    let code = unsafe { obj_iter_all(rtxn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);

    // Drain the one doc.
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: iter valid; all out-pointers writable.
    let code = unsafe { obj_iter_next(iter, &raw mut id, &raw mut payload, &raw mut len) };
    assert_eq!(code, OBJ_OK);
    assert!(!payload.is_null());
    // SAFETY: payload/len from a successful obj_iter_next; must free with obj_free_buffer.
    unsafe { obj_free_buffer(payload, len) };

    // Next call must signal exhaustion.
    let mut id2: u64 = 0;
    let mut payload2: *mut u8 = ptr::null_mut();
    let mut len2: usize = 0;
    // SAFETY: iter valid; all out-pointers writable.
    let code = unsafe { obj_iter_next(iter, &raw mut id2, &raw mut payload2, &raw mut len2) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    assert_eq!(id2, 0);
    assert!(payload2.is_null());
    assert_eq!(len2, 0);

    // SAFETY: iter valid.
    unsafe { obj_iter_free(iter) };
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn iter_free_null_is_noop() {
    // SAFETY: deliberately passing null; obj_iter_free must be null-tolerant.
    unsafe { obj_iter_free(ptr::null_mut()) };
}

// ── obj_find_unique ───────────────────────────────────────────────────────────

#[test]
fn find_unique_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: txn deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            ptr::null_mut(),
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn find_unique_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-col.obj");
    let rtxn = begin_read(db);
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            ptr::null(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_null_index_name_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: index_name deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_null_out_id_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-out-id.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: out_id deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            0,
            ptr::null_mut(),
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_null_out_payload_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-out-payload.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut len: usize = 0;
    // SAFETY: out_payload deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            ptr::null_mut(),
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_null_out_len_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-out-len.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    // SAFETY: out_payload_len deliberately null; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            ptr::null_mut(),
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_null_key_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("fu-null-key.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: key=null but key_len=4 — the invalid combo obj_find_unique guards against.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            4,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("fu-bad-utf8-col.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            bad.as_ptr().cast(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_invalid_utf8_index_name_returns_utf8_error() {
    let (_dir, db) = open_db("fu-bad-utf8-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let bad: &[u8] = b"\xff\xfe\0";
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8) for index_name; other args valid.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            bad.as_ptr().cast(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn find_unique_unknown_collection_returns_not_found() {
    let (_dir, db) = open_db("fu-unknown-coll.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("no_such_collection").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    // SAFETY: all args valid; collection and index simply don't exist in this db.
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ptr::null(),
            0,
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_count_all ─────────────────────────────────────────────────────────────

#[test]
fn count_all_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: txn deliberately null; other args valid.
    let code = unsafe { obj_count_all(ptr::null_mut(), cs.as_ptr(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn count_all_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("ca-null-col.obj");
    let rtxn = begin_read(db);
    let mut count: u64 = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe { obj_count_all(rtxn, ptr::null(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_all_null_out_count_returns_invalid_arg() {
    let (_dir, db) = open_db("ca-null-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    // SAFETY: out_count deliberately null; other args valid.
    let code = unsafe { obj_count_all(rtxn, cs.as_ptr(), ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_all_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("ca-bad-utf8.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let mut count: u64 = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe { obj_count_all(rtxn, bad.as_ptr().cast(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_all_unknown_collection_returns_not_found() {
    let (_dir, db) = open_db("ca-unknown-coll.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("ghost").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: rtxn valid; cs NUL-terminated UTF-8; out_count writable — collection does not exist.
    let code = unsafe { obj_count_all(rtxn, cs.as_ptr(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_count_index_range ─────────────────────────────────────────────────────

#[test]
fn count_index_range_null_txn_returns_invalid_arg() {
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: txn deliberately null; other args valid.
    let code = unsafe {
        obj_count_index_range(
            ptr::null_mut(),
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn count_index_range_null_collection_returns_invalid_arg() {
    let (_dir, db) = open_db("cir-null-col.obj");
    let rtxn = begin_read(db);
    let idx = CString::new("by_x").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: collection deliberately null; other args valid.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            ptr::null(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_null_index_name_returns_invalid_arg() {
    let (_dir, db) = open_db("cir-null-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: index_name deliberately null; other args valid.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            cs.as_ptr(),
            ptr::null(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_null_out_count_returns_invalid_arg() {
    let (_dir, db) = open_db("cir-null-out.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    // SAFETY: out_count deliberately null; other args valid.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ptr::null_mut(),
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_null_lower_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("cir-bad-lower.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: lower.ptr=null but lower.len=1 — invalid combo bytes_to_bound guards against.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 1, inclusive: true },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_null_upper_nonzero_len_returns_invalid_arg() {
    let (_dir, db) = open_db("cir-bad-upper.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let idx = CString::new("by_x").expect("non-NUL");
    let lower_key = b"\x01";
    let mut count: u64 = 0;
    // SAFETY: upper.ptr=null but upper.len=2 — invalid combo bytes_to_bound guards against;
    // lower is a valid non-null 1-byte slice.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            cs.as_ptr(),
            idx.as_ptr(),
            ObjBound { ptr: lower_key.as_ptr(), len: lower_key.len(), inclusive: true },
            ObjBound { ptr: ptr::null(), len: 2, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_invalid_utf8_collection_returns_utf8_error() {
    let (_dir, db) = open_db("cir-bad-utf8-col.obj");
    let rtxn = begin_read(db);
    let bad: &[u8] = b"\xff\xfe\0";
    let idx = CString::new("by_x").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8); other args valid.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            bad.as_ptr().cast(),
            idx.as_ptr(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn count_index_range_invalid_utf8_index_name_returns_utf8_error() {
    let (_dir, db) = open_db("cir-bad-utf8-idx.obj");
    let rtxn = begin_read(db);
    let cs = CString::new("c").expect("non-NUL");
    let bad: &[u8] = b"\xff\xfe\0";
    let mut count: u64 = 0;
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8) for index_name; other args valid.
    let code = unsafe {
        obj_count_index_range(
            rtxn,
            cs.as_ptr(),
            bad.as_ptr().cast(),
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            ObjBound { ptr: ptr::null(), len: 0, inclusive: false },
            &raw mut count,
        )
    };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: rtxn valid.
    unsafe { obj_txn_end_read(rtxn) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_stat ──────────────────────────────────────────────────────────────────

#[test]
fn stat_null_db_returns_invalid_arg() {
    let mut stat = obj_stat_t {
        format_major: 0,
        format_minor: 0,
        page_size: 0,
        reserved: 0,
        page_count: 0,
        file_size_bytes: 0,
        collection_count: 0,
    };
    // SAFETY: db deliberately null; out_stat is a valid writable pointer.
    let code = unsafe { obj_stat(ptr::null_mut(), &raw mut stat) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn stat_null_out_stat_returns_invalid_arg() {
    let (_dir, db) = open_db("stat-null-out.obj");
    // SAFETY: db valid; out_stat deliberately null.
    let code = unsafe { obj_stat(db, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_integrity_check ───────────────────────────────────────────────────────

#[test]
fn integrity_check_null_db_returns_invalid_arg() {
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    // SAFETY: db deliberately null; out_report is valid.
    let code = unsafe { obj_integrity_check(ptr::null_mut(), &raw mut report) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn integrity_check_null_out_report_returns_invalid_arg() {
    let (_dir, db) = open_db("ic-null-out.obj");
    // SAFETY: db valid; out_report deliberately null.
    let code = unsafe { obj_integrity_check(db, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_integrity_report_failure_at ──────────────────────────────────────────

#[test]
fn failure_at_null_report_returns_invalid_arg() {
    let mut out_str: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: report deliberately null; out_string and out_string_len are valid.
    let code = unsafe {
        obj_integrity_report_failure_at(ptr::null(), 0, &raw mut out_str, &raw mut out_len)
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn failure_at_null_out_string_returns_invalid_arg() {
    let (_dir, db) = open_db("fa-null-out-str.obj");
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    // SAFETY: db valid; out_report writable.
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    assert_eq!(code, OBJ_OK);
    assert!(!report.is_null());
    let mut out_len: usize = 0;
    // SAFETY: report valid from integrity_check; out_string deliberately null.
    let code = unsafe { obj_integrity_report_failure_at(report, 0, ptr::null_mut(), &raw mut out_len) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: report valid.
    unsafe { obj_integrity_report_free(report) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn failure_at_null_out_len_returns_invalid_arg() {
    let (_dir, db) = open_db("fa-null-out-len.obj");
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    // SAFETY: db valid; out_report writable.
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    assert_eq!(code, OBJ_OK);
    assert!(!report.is_null());
    let mut out_str: *mut u8 = ptr::null_mut();
    // SAFETY: report valid; out_string_len deliberately null.
    let code = unsafe { obj_integrity_report_failure_at(report, 0, &raw mut out_str, ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: report valid.
    unsafe { obj_integrity_report_free(report) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn failure_at_out_of_range_index_returns_not_found() {
    let (_dir, db) = open_db("fa-oob-idx.obj");
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    // SAFETY: db valid; out_report writable.
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    assert_eq!(code, OBJ_OK);
    assert!(!report.is_null());
    let mut out_str: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: report valid; index=999 is far beyond any actual failure list (clean db has 0).
    let code = unsafe {
        obj_integrity_report_failure_at(report, 999, &raw mut out_str, &raw mut out_len)
    };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    assert!(out_str.is_null());
    assert_eq!(out_len, 0);
    // SAFETY: report valid.
    unsafe { obj_integrity_report_free(report) };
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_backup_to ─────────────────────────────────────────────────────────────

#[test]
fn backup_null_db_returns_invalid_arg() {
    let dir = TempDir::new().expect("tmp");
    let dst = dir.path().join("dst.obj");
    let cs = path_cstr(&dst);
    // SAFETY: db deliberately null; dest is a valid NUL-terminated path.
    let code = unsafe { obj_backup_to(ptr::null_mut(), cs.as_ptr()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn backup_null_dest_returns_invalid_arg() {
    let (_dir, db) = open_db("bk-null-dest.obj");
    // SAFETY: db valid; dest deliberately null.
    let code = unsafe { obj_backup_to(db, ptr::null()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn backup_invalid_utf8_dest_returns_utf8_error() {
    let (_dir, db) = open_db("bk-bad-utf8.obj");
    let bad: &[u8] = b"\xff\xfe\0";
    // SAFETY: bad is non-null NUL-terminated (not valid UTF-8) for dest; db is valid.
    let code = unsafe { obj_backup_to(db, bad.as_ptr().cast()) };
    assert_eq!(code, OBJ_ERR_UTF8);
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn backup_to_existing_dest_returns_error() {
    let (_src_dir, db) = open_db("bk-existing-src.obj");
    // Create the destination file so it already exists.
    let dst_dir = TempDir::new().expect("tmp dst dir");
    let dst = dst_dir.path().join("existing.obj");
    std::fs::write(&dst, b"").expect("create dst");
    let cs = path_cstr(&dst);
    // SAFETY: db valid; dest is a valid NUL-terminated path pointing to an existing file.
    let code = unsafe { obj_backup_to(db, cs.as_ptr()) };
    assert_ne!(code, OBJ_OK, "expected error when dest already exists");
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

// ── obj_txn_rollback on read txn (cleanup helper coverage) ───────────────────

#[test]
fn rollback_null_write_txn_is_noop() {
    // SAFETY: deliberately passing null; obj_txn_rollback must be null-tolerant.
    unsafe { obj_txn_rollback(ptr::null_mut()) };
}
