//! `obj_errmsg` per-handle diagnostic-string tests, driven over the C ABI.
//!
//! Covers the `sqlite3_errmsg`-style contract added by issue #10:
//!
//! - a freshly-opened handle reports the static `"(no error)"`;
//! - a NULL handle reports the static `"(null db handle)"`;
//! - a failing call made through a transaction reaches the db handle's
//!   shared diagnostic slot (the txn keepalive path) and records the
//!   engine's SPECIFIC reason, not just the generic code label;
//! - a failing db-direct call (`obj_backup_to`) records a message too.

// allow: this test crate exercises the unsafe C ABI directly.
#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tempfile::TempDir;

use obj::{
    obj_backup_to, obj_close, obj_count_all, obj_db_t, obj_errmsg, obj_open, obj_read_txn_t,
    obj_txn_begin_read, obj_txn_end_read, OBJ_ERR_NOT_FOUND, OBJ_OK,
};

use core::ffi::c_char;

fn path_cstr(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn open_db(name: &str) -> (TempDir, *mut obj_db_t) {
    let dir = TempDir::new().expect("tmp dir");
    let cs = path_cstr(&dir.path().join(name));
    let mut db: *mut obj_db_t = ptr::null_mut();
    // SAFETY: cs is non-null NUL-terminated; db is a writable out-pointer.
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK, "obj_open failed with {code}");
    assert!(!db.is_null());
    (dir, db)
}

/// Read the (never-NULL) `obj_errmsg` pointer into an owned String.
fn errmsg(db: *mut obj_db_t) -> String {
    // SAFETY: obj_errmsg never returns NULL; the returned pointer is a
    // valid NUL-terminated C string owned by `db` (or a static).
    let ptr = unsafe { obj_errmsg(db) };
    assert!(!ptr.is_null(), "obj_errmsg must never return NULL");
    // SAFETY: ptr is a non-null NUL-terminated C string per the contract.
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

#[test]
fn fresh_handle_reports_no_error() {
    let (_dir, db) = open_db("errmsg-fresh.obj");
    assert_eq!(errmsg(db), "(no error)");
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

#[test]
fn null_handle_reports_null_db_handle() {
    assert_eq!(errmsg(ptr::null_mut()), "(null db handle)");
}

#[test]
fn failing_txn_call_records_specific_message() {
    let (_dir, db) = open_db("errmsg-txn.obj");

    let mut rtxn: *mut obj_read_txn_t = ptr::null_mut();
    // SAFETY: db valid; rtxn is a writable out-pointer.
    let code = unsafe { obj_txn_begin_read(db, &raw mut rtxn) };
    assert_eq!(code, OBJ_OK);

    // Count over a collection that does not exist — a real engine Err
    // routed through the txn keepalive to the db's diagnostic slot.
    let coll = CString::new("ghost_collection").expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: rtxn valid; coll NUL-terminated UTF-8; out_count writable.
    let code = unsafe { obj_count_all(rtxn, coll.as_ptr(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);

    let msg = errmsg(db);
    assert!(!msg.is_empty(), "errmsg must be non-empty after a failure");
    assert_ne!(msg, "(no error)", "a recorded error must replace the default");
    assert_ne!(msg, "(null db handle)");
    assert!(
        msg.contains("ghost_collection"),
        "expected the specific reason naming the collection, got {msg:?}"
    );

    // SAFETY: rtxn valid; db still open.
    unsafe { obj_txn_end_read(rtxn) };
    // The message outlives the txn — it is owned by the db handle.
    assert!(errmsg(db).contains("ghost_collection"));
    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

/// Drive one failing call through a fresh read txn so a SPECIFIC
/// error message naming `coll` is recorded against `db`'s diagnostic
/// slot (the same keepalive path `obj_count_all` uses in production).
fn record_ghost_error(db: *mut obj_db_t, coll: &str) {
    let mut rtxn: *mut obj_read_txn_t = ptr::null_mut();
    // SAFETY: db valid; rtxn is a writable out-pointer.
    let code = unsafe { obj_txn_begin_read(db, &raw mut rtxn) };
    assert_eq!(code, OBJ_OK);
    let cs = CString::new(coll).expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: rtxn valid; cs NUL-terminated UTF-8; out_count writable.
    let code = unsafe { obj_count_all(rtxn, cs.as_ptr(), &raw mut count) };
    assert_eq!(code, OBJ_ERR_NOT_FOUND);
    // SAFETY: rtxn valid and not yet ended.
    unsafe { obj_txn_end_read(rtxn) };
}

/// Read a raw `obj_errmsg` pointer's bytes into an owned String.
fn read_ptr(p: *const c_char) -> String {
    assert!(!p.is_null());
    // SAFETY: p is a non-null NUL-terminated C string the handle owns.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// A pointer obtained from `obj_errmsg` must stay valid and unchanged
/// after subsequent errors are recorded on the same handle. Under the
/// old free-on-swap behaviour the first follow-up error freed the
/// string this pointer aliases — a use-after-free on the next read.
#[test]
fn obtained_pointer_survives_later_error_recording() {
    let (_dir, db) = open_db("errmsg-survive.obj");

    record_ghost_error(db, "alpha_ghost");
    // SAFETY: db valid; obj_errmsg never returns NULL.
    let p = unsafe { obj_errmsg(db) };
    let original = read_ptr(p);
    assert!(original.contains("alpha_ghost"), "got {original:?}");

    // Several more errors (still within the 16-deep ring grace window).
    for i in 0..8 {
        record_ghost_error(db, &format!("beta_ghost_{i}"));
    }

    // The earlier pointer is still valid and its bytes are untouched.
    assert_eq!(
        read_ptr(p),
        original,
        "earlier obj_errmsg pointer was freed or mutated"
    );

    // SAFETY: db still open.
    unsafe { obj_close(db) };
}

/// `*mut obj_db_t` wrapper that crosses a thread boundary in the
/// concurrent test. `obj_db_t` itself asserts `Send + Sync` in
/// `lifecycle.rs`, and its FFI surface is internally synchronised, so
/// sharing the raw handle across threads is sound; the raw pointer's
/// own `!Send` is what this newtype overrides.
struct SendDb(*mut obj_db_t);
// SAFETY: obj_db_t is Send + Sync (asserted in lifecycle.rs); its FFI
// surface is internally synchronised, so the raw handle may cross the
// thread boundary.
unsafe impl Send for SendDb {}

/// The cross-thread shape from the report: thread A obtains a pointer,
/// thread B records errors on the SAME shared handle, A's pointer must
/// remain valid. `obj_db_t` asserts `Send + Sync`, so the raw handle
/// is legitimately shared here.
#[test]
fn obtained_pointer_survives_concurrent_set_error() {
    let (_dir, db) = open_db("errmsg-concurrent.obj");

    record_ghost_error(db, "main_ghost");
    // SAFETY: db valid; obj_errmsg never returns NULL.
    let p = unsafe { obj_errmsg(db) };
    let original = read_ptr(p);
    assert!(original.contains("main_ghost"), "got {original:?}");

    // Share the handle with a worker thread that records more errors.
    let shared = SendDb(db);
    let worker = std::thread::spawn(move || {
        // Bind the whole `SendDb` first so the closure captures the
        // Send newtype as a unit (disjoint closure captures would
        // otherwise grab the inner `*mut obj_db_t` field directly,
        // which is not Send) before destructuring it.
        let shared = shared;
        let db = shared.0;
        for i in 0..4 {
            record_ghost_error(db, &format!("other_ghost_{i}"));
        }
    });
    worker.join().expect("worker thread joined");

    // A's pointer survived B's concurrent error recording.
    assert_eq!(
        read_ptr(p),
        original,
        "concurrent set_error freed a pointer obj_errmsg had returned"
    );

    // SAFETY: db still open (worker has finished and joined).
    unsafe { obj_close(db) };
}

/// Drive one failing count through a fresh read txn like
/// [`record_ghost_error`], but tolerate the transaction machinery
/// returning early under heavy concurrent contention instead of
/// asserting exact codes: the goal here is to fire `set_error` bursts,
/// not to pin down per-call return values.
fn burst_ghost_error(db: *mut obj_db_t, coll: &str) {
    let mut rtxn: *mut obj_read_txn_t = ptr::null_mut();
    // SAFETY: db valid; rtxn is a writable out-pointer.
    let code = unsafe { obj_txn_begin_read(db, &raw mut rtxn) };
    if code != OBJ_OK || rtxn.is_null() {
        return;
    }
    let cs = CString::new(coll).expect("non-NUL");
    let mut count: u64 = 0;
    // SAFETY: rtxn valid; cs NUL-terminated UTF-8; out_count writable.
    let _ = unsafe { obj_count_all(rtxn, cs.as_ptr(), &raw mut count) };
    // SAFETY: rtxn valid and not yet ended.
    unsafe { obj_txn_end_read(rtxn) };
}

/// Stress the error ring's seq-ordered reclamation: many writer threads
/// record errors on one shared handle (bursts far wider than the
/// 16-slot ring) while a reader thread spins on `obj_errmsg` and
/// validates the bytes it hands back. Under the pre-fix out-of-order
/// free, a preempted writer could republish a freed pointer as
/// `last_error`, so the reader would eventually read freed / non-UTF-8
/// storage (or crash under a sanitizer). With reclamation serialised,
/// `last_error` always points at a live ring string.
#[test]
fn concurrent_error_bursts_keep_errmsg_valid() {
    const WRITERS: usize = 32;
    const ITERS: usize = 64;

    let (_dir, db) = open_db("errmsg-stress.obj");
    // Seed so `last_error` is non-null before the reader starts.
    record_ghost_error(db, "seed_ghost");

    let stop = Arc::new(AtomicBool::new(false));

    let reader_stop = Arc::clone(&stop);
    let reader_db = SendDb(db);
    let reader = std::thread::spawn(move || {
        let reader_db = reader_db;
        let db = reader_db.0;
        let mut reads = 0u64;
        while !reader_stop.load(Ordering::Relaxed) {
            // SAFETY: db is live for the whole test; obj_errmsg never
            // returns NULL.
            let p = unsafe { obj_errmsg(db) };
            assert!(!p.is_null(), "obj_errmsg must never return NULL");
            // SAFETY: p is a non-null NUL-terminated C string owned by db.
            let cstr = unsafe { CStr::from_ptr(p) };
            assert!(
                cstr.to_str().is_ok(),
                "obj_errmsg handed back non-UTF-8 / freed bytes"
            );
            reads += 1;
        }
        assert!(reads > 0, "reader never observed a message");
    });

    let mut writers = Vec::with_capacity(WRITERS);
    for w in 0..WRITERS {
        let writer_db = SendDb(db);
        writers.push(std::thread::spawn(move || {
            let writer_db = writer_db;
            let db = writer_db.0;
            for i in 0..ITERS {
                burst_ghost_error(db, &format!("burst_{w}_{i}"));
            }
        }));
    }
    for h in writers {
        h.join().expect("writer thread joined");
    }
    stop.store(true, Ordering::Relaxed);
    reader.join().expect("reader thread joined");

    // The final published message is still a valid, readable string.
    let final_msg = errmsg(db);
    assert!(!final_msg.is_empty());

    // SAFETY: db still open (all workers finished and joined).
    unsafe { obj_close(db) };
}

#[test]
fn failing_db_direct_call_records_message() {
    let (_dir, db) = open_db("errmsg-direct.obj");

    // Back up onto a path that already exists — a db-direct Err path.
    let dst_dir = TempDir::new().expect("tmp dst");
    let dst = dst_dir.path().join("exists.obj");
    std::fs::write(&dst, b"").expect("create dst");
    let cs = path_cstr(&dst);
    // SAFETY: db valid; cs is a NUL-terminated path to an existing file.
    let code = unsafe { obj_backup_to(db, cs.as_ptr()) };
    assert_ne!(code, OBJ_OK, "expected error when dest already exists");

    let msg = errmsg(db);
    assert!(!msg.is_empty());
    assert_ne!(msg, "(no error)");

    // SAFETY: db still open.
    unsafe { obj_close(db) };
}
