//! Transaction isolation tests.
//!
//! A rolled-back write txn must be invisible. A reader pinned
//! before a commit sees only the pre-commit state; a fresh reader
//! after the commit sees both.

#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_close, obj_db_t, obj_doc_get, obj_doc_insert, obj_free_buffer, obj_iter_all, obj_iter_free,
    obj_iter_next, obj_iter_t, obj_open, obj_read_txn_t, obj_txn_begin_read, obj_txn_begin_write,
    obj_txn_commit, obj_txn_end_read, obj_txn_rollback, obj_write_txn_t, OBJ_ERR_NOT_FOUND, OBJ_OK,
};

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn open_db(name: &str) -> (TempDir, *mut obj_db_t) {
    let dir = TempDir::new().expect("tmp");
    let cs = path_cstring(&dir.path().join(name));
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK);
    (dir, db)
}

fn begin_write(db: *mut obj_db_t) -> *mut obj_write_txn_t {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

fn begin_read(db: *mut obj_db_t) -> *mut obj_read_txn_t {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_read(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

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
    assert_eq!(code, OBJ_OK);
    id
}

/// Returns the doc count for the named ids (any present count as
/// 1). Used by isolation tests to count visible rows.
fn lookup_count(txn: *mut obj_read_txn_t, collection: &str, ids: &[u64]) -> usize {
    let cs = CString::new(collection).expect("non-NUL");
    let mut present = 0;
    for &id in ids {
        let mut payload: *mut u8 = ptr::null_mut();
        let mut len: usize = 0;
        let code = unsafe { obj_doc_get(txn, cs.as_ptr(), id, &raw mut payload, &raw mut len) };
        match code {
            OBJ_OK => {
                present += 1;
                if !payload.is_null() {
                    unsafe { obj_free_buffer(payload, len) };
                }
            }
            OBJ_ERR_NOT_FOUND => {}
            other => panic!("obj_doc_get returned {other}"),
        }
    }
    present
}

fn iter_all_ids(txn: *mut obj_read_txn_t, collection: &str) -> Vec<u64> {
    let cs = CString::new(collection).expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    let code = unsafe { obj_iter_all(txn, cs.as_ptr(), &raw mut iter) };
    assert_eq!(code, OBJ_OK);
    let mut ids = Vec::new();
    loop {
        let mut id: u64 = 0;
        let mut payload: *mut u8 = ptr::null_mut();
        let mut len: usize = 0;
        let code = unsafe { obj_iter_next(iter, &raw mut id, &raw mut payload, &raw mut len) };
        if code == OBJ_ERR_NOT_FOUND {
            break;
        }
        assert_eq!(code, OBJ_OK);
        unsafe { obj_free_buffer(payload, len) };
        ids.push(id);
    }
    unsafe { obj_iter_free(iter) };
    ids
}

#[test]
fn rollback_makes_writes_invisible_to_subsequent_readers() {
    let (_dir, db) = open_db("rollback.obj");

    let txn = begin_write(db);
    let id_committed = insert(txn, "t", b"committed");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let txn = begin_write(db);
    let _id_rolled_back = insert(txn, "t", b"rolled-back");
    unsafe { obj_txn_rollback(txn) };

    let rtxn = begin_read(db);
    assert_eq!(lookup_count(rtxn, "t", &[id_committed]), 1);
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn snapshot_pinned_reader_does_not_observe_later_commit() {
    let (_dir, db) = open_db("snapshot.obj");

    let txn = begin_write(db);
    let id_initial = insert(txn, "t", b"initial");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn_a = begin_read(db);

    let txn = begin_write(db);
    let id_post = insert(txn, "t", b"post-snapshot");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    assert_eq!(lookup_count(rtxn_a, "t", &[id_initial]), 1);
    assert_eq!(
        lookup_count(rtxn_a, "t", &[id_post]),
        0,
        "snapshot reader must not see post-commit writes"
    );

    let rtxn_b = begin_read(db);
    assert_eq!(lookup_count(rtxn_b, "t", &[id_initial, id_post]), 2);

    unsafe { obj_txn_end_read(rtxn_a) };
    unsafe { obj_txn_end_read(rtxn_b) };
    unsafe { obj_close(db) };
}

#[test]
fn iter_all_uses_supplied_read_transaction_snapshot() {
    let (_dir, db) = open_db("iter-all-snapshot.obj");

    let txn = begin_write(db);
    let id_initial = insert(txn, "t", b"initial");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let rtxn = begin_read(db);

    let txn = begin_write(db);
    let _id_post = insert(txn, "t", b"post-snapshot");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let ids = iter_all_ids(rtxn, "t");
    assert_eq!(ids, vec![id_initial]);

    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn rollback_is_null_tolerant() {
    unsafe { obj_txn_rollback(ptr::null_mut()) };
}

#[test]
fn end_read_is_null_tolerant() {
    unsafe { obj_txn_end_read(ptr::null_mut()) };
}
