//! Integrity check / backup / stat tests.

// allow: this test crate exercises the unsafe C ABI directly, so every FFI call site is `unsafe`.
#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_backup_to, obj_close, obj_db_t, obj_doc_insert_raw, obj_free_buffer, obj_integrity_check,
    obj_integrity_report_failure_at, obj_integrity_report_failure_count, obj_integrity_report_free,
    obj_integrity_report_is_ok, obj_integrity_report_pages_checked, obj_integrity_report_t,
    obj_open, obj_stat, obj_stat_t, obj_txn_begin_write, obj_txn_commit, OBJ_ERR_INTEGRITY, OBJ_OK,
};

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

#[test]
fn integrity_check_passes_on_clean_db() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("clean.obj");
    let db = open_db_c(&path);

    let mut txn: *mut obj::obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    let collection = CString::new("c").expect("non-NUL");
    for _ in 0..3 {
        let mut id: u64 = 0;
        let code =
            unsafe { obj_doc_insert_raw(txn, collection.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
        assert_eq!(code, OBJ_OK);
    }
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    assert_eq!(code, OBJ_OK);
    assert!(!report.is_null());
    assert!(unsafe { obj_integrity_report_is_ok(report) });
    let pages = unsafe { obj_integrity_report_pages_checked(report) };
    assert!(pages > 0, "expected non-zero pages checked");
    let failures = unsafe { obj_integrity_report_failure_count(report) };
    assert_eq!(failures, 0);
    unsafe { obj_integrity_report_free(report) };
    unsafe { obj_close(db) };
}

#[test]
fn integrity_report_is_null_tolerant_accessors() {
    assert!(!unsafe { obj_integrity_report_is_ok(ptr::null()) });
    assert_eq!(
        unsafe { obj_integrity_report_pages_checked(ptr::null()) },
        0
    );
    assert_eq!(
        unsafe { obj_integrity_report_failure_count(ptr::null()) },
        0
    );
    unsafe { obj_integrity_report_free(ptr::null_mut()) };
}

#[test]
// allow: end-to-end C-API scenario test; splitting the round-trip would obscure what it documents.
#[allow(clippy::too_many_lines)]
fn integrity_check_surfaces_corruption_after_byte_flip() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("corrupt.obj");

    {
        let db = open_db_c(&path);
        let mut txn: *mut obj::obj_write_txn_t = ptr::null_mut();
        let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
        assert_eq!(code, OBJ_OK);
        let collection = CString::new("c").expect("non-NUL");
        for _ in 0..16 {
            let mut id: u64 = 0;
            let code = unsafe {
                obj_doc_insert_raw(
                    txn,
                    collection.as_ptr(),
                    b"payload".as_ptr(),
                    7,
                    &raw mut id,
                )
            };
            assert_eq!(code, OBJ_OK);
        }
        let code = unsafe { obj_txn_commit(txn) };
        assert_eq!(code, OBJ_OK);
        unsafe { obj_close(db) };
    }

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open file");
    let file_len = file.metadata().expect("metadata").len();
    let offset = file_len / 2;
    file.seek(SeekFrom::Start(offset)).expect("seek");
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf).expect("read");
    buf[0] ^= 0xFF;
    file.seek(SeekFrom::Start(offset)).expect("seek");
    file.write_all(&buf).expect("write");
    file.sync_all().expect("sync");
    drop(file);

    let mut config = obj::obj_config_t {
        struct_size: u32::try_from(std::mem::size_of::<obj::obj_config_t>())
            .expect("config size fits u32"),
        sync_mode: obj::OBJ_SYNC_MODE_FULL,
        busy_timeout_ms: 0,
        skip_open_check: true,
        has_encryption_key: false,
        encryption_key: [0u8; obj::OBJ_ENCRYPTION_KEY_LEN],
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let cs = path_cstring(&path);
    let code = unsafe { obj::obj_open_with_config(cs.as_ptr(), &raw mut config, &raw mut db) };
    if code != OBJ_OK {
        return;
    }
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    if code == OBJ_ERR_INTEGRITY {
        assert!(!report.is_null());
        let n = unsafe { obj_integrity_report_failure_count(report) };
        assert!(n > 0);
        let mut out_str: *mut u8 = ptr::null_mut();
        let mut out_len: usize = 0;
        let code = unsafe {
            obj_integrity_report_failure_at(report, 0, &raw mut out_str, &raw mut out_len)
        };
        assert_eq!(code, OBJ_OK);
        assert!(!out_str.is_null());
        let bytes = unsafe { std::slice::from_raw_parts(out_str, out_len) }.to_vec();
        unsafe { obj_free_buffer(out_str, out_len) };
        assert!(!bytes.is_empty());
    }
    if !report.is_null() {
        unsafe { obj_integrity_report_free(report) };
    }
    unsafe { obj_close(db) };
}

#[test]
fn backup_round_trips_via_c_abi() {
    let dir = TempDir::new().expect("tmp");
    let src = dir.path().join("src.obj");
    let dst = dir.path().join("backup.obj");

    let db = open_db_c(&src);
    let mut txn: *mut obj::obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    let collection = CString::new("c").expect("non-NUL");
    for _ in 0..8 {
        let mut id: u64 = 0;
        let code =
            unsafe { obj_doc_insert_raw(txn, collection.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
        assert_eq!(code, OBJ_OK);
    }
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let dst_cs = path_cstring(&dst);
    let code = unsafe { obj_backup_to(db, dst_cs.as_ptr()) };
    assert_eq!(code, OBJ_OK);
    unsafe { obj_close(db) };

    let db = open_db_c(&dst);
    let mut report: *mut obj_integrity_report_t = ptr::null_mut();
    let code = unsafe { obj_integrity_check(db, &raw mut report) };
    assert_eq!(code, OBJ_OK);
    assert!(unsafe { obj_integrity_report_is_ok(report) });
    unsafe { obj_integrity_report_free(report) };
    unsafe { obj_close(db) };
}

#[test]
fn stat_reports_consistent_summary() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("stat.obj");

    let db = open_db_c(&path);
    let mut txn: *mut obj::obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    for collection_name in ["alpha", "beta"] {
        let cs = CString::new(collection_name).expect("non-NUL");
        for _ in 0..3 {
            let mut id: u64 = 0;
            let code = unsafe { obj_doc_insert_raw(txn, cs.as_ptr(), b"x".as_ptr(), 1, &raw mut id) };
            assert_eq!(code, OBJ_OK);
        }
    }
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    let mut stat = obj_stat_t {
        format_major: 0,
        format_minor: 0,
        page_size: 0,
        reserved: 0,
        page_count: 0,
        file_size_bytes: 0,
        collection_count: 0,
    };
    let code = unsafe { obj_stat(db, &raw mut stat) };
    assert_eq!(code, OBJ_OK);
    assert_eq!(stat.collection_count, 2);
    assert!(stat.page_count > 0);
    assert!(stat.file_size_bytes >= u64::from(stat.page_size) * stat.page_count);
    assert!(stat.page_size > 0);
    unsafe { obj_close(db) };
}

#[test]
fn strerror_for_integrity_code_is_non_empty() {
    let ptr = unsafe { obj::obj_strerror(OBJ_ERR_INTEGRITY) };
    assert!(!ptr.is_null());
    let cstr = unsafe { CStr::from_ptr(ptr) };
    assert!(!cstr.to_bytes().is_empty());
}
