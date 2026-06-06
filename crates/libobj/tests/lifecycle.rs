//! Lifecycle tests driven over the C ABI.
//!
//! Calls `obj_open` / `obj_open_with_config` / `obj_close` via
//! the `libobj` crate's `pub extern "C"` exports. The crate's
//! `rlib` crate-type lets the test harness consume them as
//! regular Rust items; the same symbols are exported by the
//! `cdylib` / `staticlib` artifacts for C consumers.

#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_close, obj_config_t, obj_db_t, obj_open, obj_open_with_config, obj_strerror,
    OBJ_ENCRYPTION_KEY_LEN, OBJ_ERR_INVALID_ARG, OBJ_ERR_UTF8, OBJ_OK, OBJ_SYNC_MODE_FULL,
    OBJ_SYNC_MODE_NORMAL, OBJ_SYNC_MODE_OFF,
};

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

/// Build a default-shaped `obj_config_t` with `struct_size` set to
/// the current layout — the idiom every C caller is expected to use.
fn default_config() -> obj_config_t {
    obj_config_t {
        struct_size: u32::try_from(std::mem::size_of::<obj_config_t>())
            .expect("config size fits u32"),
        sync_mode: OBJ_SYNC_MODE_FULL,
        busy_timeout_ms: 0,
        skip_open_check: false,
        has_encryption_key: false,
        encryption_key: [0u8; OBJ_ENCRYPTION_KEY_LEN],
    }
}

#[test]
fn open_default_then_close() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("life.obj"));
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(p.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK, "obj_open returned {code}");
    assert!(!db.is_null(), "expected non-null db handle");
    unsafe { obj_close(db) };
}

#[test]
fn open_with_config_full_durability() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("cfg-full.obj"));
    let config = obj_config_t {
        sync_mode: OBJ_SYNC_MODE_FULL,
        busy_timeout_ms: 1_000,
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_OK);
    assert!(!db.is_null());
    unsafe { obj_close(db) };
}

#[test]
fn open_with_config_off_sync_skips_check() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("cfg-off.obj"));
    let config = obj_config_t {
        sync_mode: OBJ_SYNC_MODE_OFF,
        busy_timeout_ms: 0,
        skip_open_check: true,
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_OK);
    unsafe { obj_close(db) };
}

#[test]
fn open_with_config_normal_durability() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("cfg-norm.obj"));
    let config = obj_config_t {
        sync_mode: OBJ_SYNC_MODE_NORMAL,
        busy_timeout_ms: 250,
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_OK);
    unsafe { obj_close(db) };
}

#[test]
fn close_is_null_tolerant() {
    unsafe { obj_close(ptr::null_mut()) };
}

#[test]
fn open_rejects_null_path() {
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(ptr::null(), &raw mut db) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(db.is_null(), "out_db must be NULL on failure");
}

#[test]
fn open_rejects_null_out_db() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("ignored.obj"));
    let code = unsafe { obj_open(p.as_ptr(), ptr::null_mut()) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
}

#[test]
fn open_with_config_rejects_null_config() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("ignored.obj"));
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), ptr::null(), &raw mut db) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(db.is_null());
}

#[test]
fn open_invalid_utf8_path_reports_utf8_error() {
    let invalid: Vec<u8> = vec![0x80, 0x80, 0x80, 0x00];
    let cs = unsafe { CString::from_vec_with_nul_unchecked(invalid) };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_ERR_UTF8);
    assert!(db.is_null());
}

#[test]
fn strerror_returns_non_null_for_every_known_code() {
    for code in [OBJ_OK, OBJ_ERR_INVALID_ARG, OBJ_ERR_UTF8, 2, 3, 4, 5, 6, 7] {
        let ptr = unsafe { obj_strerror(code) };
        assert!(!ptr.is_null(), "obj_strerror({code}) returned NULL");
    }
}

#[test]
fn strerror_returns_unknown_for_out_of_range_code() {
    let ptr = unsafe { obj_strerror(9_999_999) };
    assert!(!ptr.is_null());
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let msg = cstr.to_string_lossy();
    assert!(
        msg.starts_with("unknown"),
        "expected 'unknown' prefix, got {msg:?}"
    );
}

#[test]
fn open_with_config_struct_size_zero_is_legacy_compat() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("size-zero.obj"));
    let config = obj_config_t {
        struct_size: 0,
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_OK);
    assert!(!db.is_null());
    unsafe { obj_close(db) };
}

#[test]
fn open_with_config_rejects_too_small_struct_size() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("size-tiny.obj"));
    let config = obj_config_t {
        struct_size: 4,
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(db.is_null(), "out_db must be NULL on a rejected config");
}

#[test]
fn open_with_config_rejects_oversized_struct_size() {
    let dir = TempDir::new().expect("tmp");
    let p = path_cstring(&dir.path().join("size-big.obj"));
    let config = obj_config_t {
        struct_size: u32::try_from(std::mem::size_of::<obj_config_t>() + 8).expect("fits"),
        ..default_config()
    };
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open_with_config(p.as_ptr(), &raw const config, &raw mut db) };
    assert_eq!(code, OBJ_ERR_INVALID_ARG);
    assert!(db.is_null());
}
