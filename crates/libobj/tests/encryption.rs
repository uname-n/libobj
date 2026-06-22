//! Encryption-key C ABI tests.
//!
//! libobj is built with the `encryption` (and `compression`) feature
//! on its `obj_engine` dependency, so `obj_open_with_config` honours
//! a 32-byte `encryption_key` supplied via `obj_config_t`. These
//! tests drive the whole loop over the C ABI:
//!
//! 1. Create an encrypted DB, write + commit a document.
//! 2. Reopen with the SAME key and read the document back (round-trip).
//! 3. Reopen with the WRONG key and confirm the data cannot be read
//!    (Poly1305 authentication fails → `OBJ_ERR_CORRUPTION`).
//!
//! The wrong-key failure surfaces either at open time (the open-time
//! integrity walk reads an encrypted page) or on the first data read;
//! both map through `error_to_code` to `OBJ_ERR_CORRUPTION`, so the
//! test accepts a failure at either point.

#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_backup_to, obj_buf_free, obj_close, obj_config_t, obj_db_t, obj_doc_get,
    obj_doc_insert_raw, obj_open, obj_open_with_config, obj_read_txn_t, obj_txn_begin_read,
    obj_txn_begin_write, obj_txn_commit, obj_txn_end_read, obj_write_txn_t, OBJ_ENCRYPTION_KEY_LEN,
    OBJ_ERR_CORRUPTION, OBJ_ERR_UNSUPPORTED, OBJ_OK, OBJ_SYNC_MODE_FULL,
};

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn config_size() -> u32 {
    u32::try_from(std::mem::size_of::<obj_config_t>()).expect("config size fits u32")
}

/// Build an `obj_config_t` that opens with `key` (or unencrypted
/// when `key` is `None`).
fn config_with_key(key: Option<[u8; OBJ_ENCRYPTION_KEY_LEN]>) -> obj_config_t {
    obj_config_t {
        struct_size: config_size(),
        sync_mode: OBJ_SYNC_MODE_FULL,
        busy_timeout_ms: 0,
        skip_open_check: false,
        has_encryption_key: key.is_some(),
        encryption_key: key.unwrap_or([0u8; OBJ_ENCRYPTION_KEY_LEN]),
    }
}

fn key_a() -> [u8; OBJ_ENCRYPTION_KEY_LEN] {
    let mut k = [0u8; OBJ_ENCRYPTION_KEY_LEN];
    for (i, b) in k.iter_mut().enumerate() {
        *b = u8::try_from(i & 0xFF).expect("byte");
    }
    k
}

fn key_b() -> [u8; OBJ_ENCRYPTION_KEY_LEN] {
    let mut k = [0u8; OBJ_ENCRYPTION_KEY_LEN];
    for (i, b) in k.iter_mut().enumerate() {
        *b = u8::try_from((i ^ 0x55) & 0xFF).expect("byte");
    }
    k
}

fn open_with_key(path: &Path, key: Option<[u8; OBJ_ENCRYPTION_KEY_LEN]>) -> (i32, *mut obj_db_t) {
    let p = path_cstring(path);
    let config = config_with_key(key);
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    (code, db)
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

/// Insert a doc inside `txn`, returning the assigned id.
fn insert(txn: *mut obj_write_txn_t, collection: &str, payload: &[u8]) -> u64 {
    let cs = CString::new(collection).expect("non-NUL");
    let mut id: u64 = 0;
    let code = unsafe {
        obj_doc_insert_raw(
            txn,
            cs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            &raw mut id,
        )
    };
    assert_eq!(code, OBJ_OK, "obj_doc_insert_raw returned {code}");
    id
}

#[test]
fn encrypted_open_round_trip_via_c_abi() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc.obj");
    let collection = "secrets";
    let payload = b"top-secret-bytes";

    let id = {
        let (code, db) = open_with_key(&path, Some(key_a()));
        assert_eq!(code, OBJ_OK, "encrypted create returned {code}");
        assert!(!db.is_null());
        let txn = begin_write(db);
        let id = insert(txn, collection, payload);
        let code = unsafe { obj_txn_commit(txn) };
        assert_eq!(code, OBJ_OK);
        unsafe { obj_close(db) };
        id
    };

    let (code, db) = open_with_key(&path, Some(key_a()));
    assert_eq!(code, OBJ_OK, "reopen-with-correct-key returned {code}");
    let rtxn = begin_read(db);
    let cs = CString::new(collection).expect("non-NUL");
    let mut out_payload: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    let code = unsafe {
        obj_doc_get(
            rtxn,
            cs.as_ptr(),
            id,
            &raw mut out_payload,
            &raw mut out_len,
        )
    };
    assert_eq!(code, OBJ_OK, "obj_doc_get on encrypted db returned {code}");
    let got = unsafe { std::slice::from_raw_parts(out_payload, out_len) }.to_vec();
    unsafe { obj_buf_free(out_payload) };
    assert_eq!(got.as_slice(), payload.as_slice());
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn wrong_key_cannot_read_encrypted_data_via_c_abi() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc-wrong.obj");
    let collection = "secrets";

    {
        let (code, db) = open_with_key(&path, Some(key_a()));
        assert_eq!(code, OBJ_OK);
        let txn = begin_write(db);
        for i in 0..16u8 {
            let _ = insert(txn, collection, &[i; 64]);
        }
        let code = unsafe { obj_txn_commit(txn) };
        assert_eq!(code, OBJ_OK);
        unsafe { obj_close(db) };
    }

    let (open_code, db) = open_with_key(&path, Some(key_b()));
    if open_code != OBJ_OK {
        assert_eq!(
            open_code, OBJ_ERR_CORRUPTION,
            "wrong-key open should surface OBJ_ERR_CORRUPTION, got {open_code}"
        );
        assert!(db.is_null(), "out_db must be NULL on a failed open");
        return;
    }
    let rtxn = begin_read(db);
    let cs = CString::new(collection).expect("non-NUL");
    let mut out_payload: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), 1, &raw mut out_payload, &raw mut out_len) };
    assert_eq!(
        code, OBJ_ERR_CORRUPTION,
        "wrong-key read should surface OBJ_ERR_CORRUPTION, got {code}"
    );
    assert!(out_payload.is_null());
    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

#[test]
fn backup_on_encrypted_db_returns_unsupported() {
    // obj-core refuses hot backups of an encrypted pager
    // (`Error::BackupNotSupportedForEncryptedPager`). That refusal must
    // surface through `error_to_code` as `OBJ_ERR_UNSUPPORTED` — the same
    // code its sibling memory-pager refusal maps to — rather than the
    // fail-closed `OBJ_ERR_CORRUPTION` wildcard, which would tell the
    // caller a healthy encrypted database is corrupt.
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc-backup.obj");

    let (code, db) = open_with_key(&path, Some(key_a()));
    assert_eq!(code, OBJ_OK, "encrypted create returned {code}");
    assert!(!db.is_null());
    let txn = begin_write(db);
    let _ = insert(txn, "secrets", b"payload");
    let code = unsafe { obj_txn_commit(txn) };
    assert_eq!(code, OBJ_OK);

    // Destination must not already exist so the encrypted-pager refusal
    // is what fails the call, not a pre-existing-destination error.
    let dst_dir = TempDir::new().expect("tmp dst dir");
    let dst = dst_dir.path().join("backup.obj");
    let dst_cs = path_cstring(&dst);
    let code = unsafe { obj_backup_to(db, dst_cs.as_ptr()) };
    assert_eq!(
        code, OBJ_ERR_UNSUPPORTED,
        "backup of an encrypted db should map to OBJ_ERR_UNSUPPORTED, got {code}"
    );

    unsafe { obj_close(db) };
}

#[test]
fn key_on_plaintext_db_is_rejected() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("plain.obj");

    {
        let p = path_cstring(&path);
        let mut db: *mut obj_db_t = ptr::null_mut();
        let code = unsafe { obj_open(p.as_ptr(), &raw mut db) };
        assert_eq!(code, OBJ_OK);
        unsafe { obj_close(db) };
    }

    let (code, db) = open_with_key(&path, Some(key_a()));
    assert_ne!(
        code, OBJ_OK,
        "opening a plaintext file with a key must fail"
    );
    assert!(
        code == obj::OBJ_ERR_INVALID_ARG || code == OBJ_ERR_UNSUPPORTED,
        "expected INVALID_ARG or UNSUPPORTED, got {code}"
    );
    assert!(db.is_null());
}
