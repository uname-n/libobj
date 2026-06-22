//! Queries, iteration, `find_unique`, counts, and diagnostic
//! entry points.
//!
//! Wraps the obj engine read-side
//! `ReadTxn::{all_raw, index_range_raw, find_unique_raw, count_*}`
//! helpers, and the `Db::{stat, integrity_check, backup_to}`
//! diagnostic surface.

use core::ffi::c_char;
use core::ptr;
use core::slice;
use std::collections::VecDeque;
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::{
    catch_ffi, catch_ffi_or, catch_ffi_void, obj_error_t, OBJ_ERR_INTEGRITY, OBJ_ERR_INVALID_ARG,
    OBJ_ERR_NOT_FOUND, OBJ_ERR_PANIC, OBJ_ERR_UTF8, OBJ_OK,
};
use crate::lifecycle::{db_error_code, db_handle_error_code, obj_db_t, DbInner};
use crate::txn::obj_read_txn_t;

/// A range bound passed by value across the C ABI.
///
/// `ptr == NULL` (paired with `len == 0`) means unbounded on that side.
/// A non-null `ptr` points to `len` order-preserving encoded key bytes;
/// `inclusive` selects [`Bound::Included`] vs [`Bound::Excluded`].
/// `inclusive` is ignored when `ptr` is `NULL`.
///
/// `inclusive` is a `u8` flag, NOT a C `_Bool`: any non-zero byte means
/// inclusive, `0` means exclusive. It is deliberately typed `u8` rather
/// than `bool` because this struct is passed BY VALUE across the ABI, so
/// a C `_Bool` byte that is not exactly 0 or 1 would be undefined
/// behaviour the moment it materialised as a Rust `bool` — before any
/// accessor could reinterpret it. A `u8` is valid for every bit pattern,
/// so the `!= 0` normalisation is the only possible read. Mirrors the
/// reasoning behind the `obj_config_t` bool-field reads in
/// `crates/libobj/src/lifecycle.rs`.
///
/// cbindgen emits this as `obj_bound_t` in the generated header.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ObjBound {
    /// Key bytes pointer. `NULL` = unbounded.
    pub ptr: *const u8,
    /// Length of the key in bytes. Must be `0` when `ptr` is `NULL`.
    pub len: usize,
    /// Non-zero → `Bound::Included`; `0` → `Bound::Excluded`. Typed `u8`
    /// (not `bool`) so any C-supplied byte is a valid value, never UB.
    pub inclusive: u8,
}

/// Iterator concrete variants. No `dyn` — the C side sees
/// one opaque pointer; the Rust side dispatches via this enum.
enum IterImpl {
    /// `obj_iter_all` — pre-materialised pairs from the supplied
    /// read transaction's pinned snapshot.
    All {
        /// Materialised `(id, payload)` pairs, drained by `next`.
        pending: VecDeque<(u64, Vec<u8>)>,
        /// Arc keepalive on the parent Db so any engine resources
        /// referenced by the snapshot-derived rows remain valid for
        /// the C iterator handle's lifetime.
        // allow: dead_code — held purely to pin the parent Db's lifetime; never read in normal control flow, only released on drop.
        #[allow(dead_code)]
        db: Arc<DbInner>,
    },
    /// `obj_iter_index_range` — pre-materialised pairs.
    IndexRange {
        /// Materialised `(id, payload)` pairs, drained by `next`.
        pending: VecDeque<(u64, Vec<u8>)>,
        /// Arc keepalive on the parent Db. Iteration is done by the
        /// time we construct this variant, but we keep the Arc to
        /// match the `IterImpl::All` lifecycle and to keep the
        /// snapshot pinned in case future variants stream lazily.
        // allow: dead_code — held purely to pin the parent Db's snapshot; never read in normal control flow, only released on drop.
        #[allow(dead_code)]
        db: Arc<DbInner>,
    },
}

/// Opaque iterator handle returned by `obj_iter_all` /
/// `obj_iter_index_range`. Drives one element at a time via
/// `obj_iter_next` and is freed by `obj_iter_free`.
pub struct obj_iter_t {
    /// Wrapped concrete enum. `ManuallyDrop` is NOT needed here —
    /// dropping the variant chain releases everything; the Arc is
    /// declared in each variant so Rust's natural drop order
    /// applies.
    inner: IterImpl,
}

impl obj_iter_t {
    /// Drive one step of the underlying iterator. Returns
    /// `Some((id, payload))` on a step and `None` on end-of-
    /// iteration.
    fn step(&mut self) -> Option<(u64, Vec<u8>)> {
        match &mut self.inner {
            IterImpl::All { pending, .. } | IterImpl::IndexRange { pending, .. } => {
                pending.pop_front()
            }
        }
    }
}

/// Opaque integrity report handle. Buffers the failure list so
/// per-failure access by index is cheap.
pub struct obj_integrity_report_t {
    /// `true` iff the report carries no failures.
    is_ok: bool,
    /// Total pages the walker visited.
    pages_checked: u64,
    /// Pre-formatted failure messages, indexed by position.
    failures: Vec<String>,
}

/// Flat C struct returned by [`obj_stat`].
#[repr(C)]
pub struct obj_stat_t {
    /// Format major version from page 0.
    pub format_major: u16,
    /// Format minor version from page 0.
    pub format_minor: u16,
    /// On-disk page size in bytes.
    pub page_size: u16,
    /// Reserved padding to keep the struct's layout deterministic
    /// across cbindgen versions / compilers. Always zero.
    pub reserved: u16,
    /// Total number of pages.
    pub page_count: u64,
    /// Logical file size = `page_count * page_size`.
    pub file_size_bytes: u64,
    /// Number of registered collections.
    pub collection_count: u64,
}

/// Construct an iterator over every doc in `collection`, snapshot-
/// consistent against `txn`. Caller pairs with [`obj_iter_next`]
/// and [`obj_iter_free`].
///
/// # Safety
///
/// - `txn` must be a live read-txn handle.
/// - `collection` must be a NUL-terminated UTF-8 C string.
/// - `out_iter` must be a writable `obj_iter_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_iter_all(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    out_iter: *mut *mut obj_iter_t,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() || out_iter.is_null() {
            if !out_iter.is_null() {
                // SAFETY: out_iter is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_iter = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => {
                // SAFETY: out_iter is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_iter = ptr::null_mut() };
                return code;
            }
        };
        // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
        let (db_arc, pairs_result) = unsafe {
            let txn_ref = (*txn).inner_ref();
            ((*txn).db_arc(), txn_ref.all_raw(collection_str))
        };
        let pairs = match pairs_result {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: out_iter is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_iter = ptr::null_mut() };
                return db_error_code(&db_arc, &e);
            }
        };
        let pending: VecDeque<(u64, Vec<u8>)> =
            pairs.into_iter().map(|(id, p)| (id.get(), p)).collect();
        let handle = Box::new(obj_iter_t {
            inner: IterImpl::All {
                pending,
                db: db_arc,
            },
        });
        // SAFETY: out_iter is non-null (checked above) and writable; handle is a fresh Box transferred to the caller.
        unsafe { *out_iter = Box::into_raw(handle) };
        OBJ_OK
    })
}

/// Construct an iterator over the index range described by `lower` and
/// `upper` on `index_name` in `collection`. A bound with `ptr == NULL`
/// and `len == 0` means unbounded on that side; `inclusive` selects
/// `Included` vs `Excluded`.
///
/// # Safety
///
/// - `txn`, `collection`, `index_name`, `out_iter` must follow the
///   usual non-null + NUL-terminated conventions.
/// - `lower.ptr` / `upper.ptr` may be NULL (paired with `len = 0`)
///   or point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn obj_iter_index_range(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    index_name: *const c_char,
    lower: ObjBound,
    upper: ObjBound,
    out_iter: *mut *mut obj_iter_t,
) -> obj_error_t {
    // SAFETY: all pointer args are forwarded unchanged to iter_index_range_inner, which upholds the same # Safety contract as this fn.
    catch_ffi(OBJ_ERR_PANIC, || unsafe {
        iter_index_range_inner(txn, collection, index_name, lower, upper, out_iter)
    })
}

/// Body of [`obj_iter_index_range`], factored out so the `extern "C"`
/// entry point stays within the 60-line budget.
///
/// # Safety
///
/// Same pointer / lifetime contract as [`obj_iter_index_range`].
unsafe fn iter_index_range_inner(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    index_name: *const c_char,
    lower: ObjBound,
    upper: ObjBound,
    out_iter: *mut *mut obj_iter_t,
) -> obj_error_t {
    if txn.is_null() || collection.is_null() || index_name.is_null() || out_iter.is_null() {
        if !out_iter.is_null() {
            // SAFETY: out_iter is non-null (just checked) and writable per the # Safety contract.
            unsafe { *out_iter = ptr::null_mut() };
        }
        return OBJ_ERR_INVALID_ARG;
    }
    // SAFETY: collection and index_name are non-null (checked above) and NUL-terminated C strings per the # Safety contract.
    let (collection_str, index_str) = match (unsafe { cstr_to_str(collection) }, unsafe {
        cstr_to_str(index_name)
    }) {
        (Ok(c), Ok(i)) => (c, i),
        (Err(code), _) | (_, Err(code)) => {
            // SAFETY: out_iter is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_iter = ptr::null_mut() };
            return code;
        }
    };
    // SAFETY: lower/upper bounds are either NULL (treated as unbounded) or point to their respective len readable bytes per the # Safety contract.
    let bounds = unsafe { decode_bound_pair(lower, upper) };
    let (lower_bound, upper_bound) = match bounds {
        Ok(pair) => pair,
        Err(code) => {
            // SAFETY: out_iter is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_iter = ptr::null_mut() };
            return code;
        }
    };
    // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract; inner_ref/db_arc derive from it.
    let (db_arc, pairs_result) = unsafe {
        let txn_ref = (*txn).inner_ref();
        (
            (*txn).db_arc(),
            txn_ref.index_range_raw(collection_str, index_str, lower_bound, upper_bound),
        )
    };
    let pairs = match pairs_result {
        Ok(p) => p,
        Err(e) => {
            // SAFETY: out_iter is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_iter = ptr::null_mut() };
            return db_error_code(&db_arc, &e);
        }
    };
    let pending: VecDeque<(u64, Vec<u8>)> =
        pairs.into_iter().map(|(id, p)| (id.get(), p)).collect();
    let handle = Box::new(obj_iter_t {
        inner: IterImpl::IndexRange {
            pending,
            db: db_arc,
        },
    });
    // SAFETY: out_iter is non-null (checked above) and writable; handle is a fresh Box transferred to the caller.
    unsafe { *out_iter = Box::into_raw(handle) };
    OBJ_OK
}

/// Step the iterator. On `OBJ_OK` the caller owns `*out_payload`
/// (length `*out_payload_len`) and MUST free it with
/// `obj_buf_free`. Returns `OBJ_ERR_NOT_FOUND` at end-of-
/// iteration; a real engine failure surfaces as the corresponding
/// error code.
///
/// # Safety
///
/// - `iter` must be a live iterator handle.
/// - `out_id`, `out_payload`, `out_payload_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn obj_iter_next(
    iter: *mut obj_iter_t,
    out_id: *mut u64,
    out_payload: *mut *mut u8,
    out_payload_len: *mut usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if iter.is_null() || out_id.is_null() || out_payload.is_null() || out_payload_len.is_null()
        {
            if !out_payload.is_null() {
                // SAFETY: out_payload is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_payload = ptr::null_mut() };
            }
            if !out_payload_len.is_null() {
                // SAFETY: out_payload_len is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_payload_len = 0 };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: iter is non-null (checked above) and a live iterator handle per the # Safety contract.
        let step = unsafe { (*iter).step() };
        if let Some((id, payload)) = step {
            let len = payload.len();
            let ptr = crate::txn::alloc_bytes(&payload);
            // SAFETY: out_id, out_payload, out_payload_len are non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_id = id };
            // SAFETY: out_payload is non-null (checked above) and writable; ptr is a fresh alloc_bytes buffer owned by the caller.
            unsafe { *out_payload = ptr };
            // SAFETY: out_payload_len is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_payload_len = len };
            return OBJ_OK;
        }
        // SAFETY: out_id is non-null (checked above) and writable per the # Safety contract.
        unsafe { *out_id = 0 };
        // SAFETY: out_payload is non-null (checked above) and writable per the # Safety contract.
        unsafe { *out_payload = ptr::null_mut() };
        // SAFETY: out_payload_len is non-null (checked above) and writable per the # Safety contract.
        unsafe { *out_payload_len = 0 };
        OBJ_ERR_NOT_FOUND
    })
}

/// Free an iterator handle. Null-tolerant.
///
/// # Safety
///
/// If non-null, `iter` must be a handle returned by
/// [`obj_iter_all`] / [`obj_iter_index_range`].
#[no_mangle]
pub unsafe extern "C" fn obj_iter_free(iter: *mut obj_iter_t) {
    if iter.is_null() {
        return;
    }
    catch_ffi_void(|| {
        // SAFETY: iter is non-null (checked above) and came from Box::into_raw in obj_iter_all/obj_iter_index_range; reclaimed exactly once here.
        let _ = unsafe { Box::from_raw(iter) };
    });
}

/// Look up a single doc by index key on a `Unique` index. The
/// caller pre-encodes `key_bytes` per the order-preserving
/// scheme. Returns `OBJ_ERR_NOT_FOUND` when no doc matches.
///
/// # Safety
///
/// - `txn` must be live.
/// - `collection`, `index_name`, `key` follow the usual pointer
///   rules; `key` may be NULL when `key_len == 0`.
/// - `out_id`, `out_payload`, `out_payload_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn obj_find_unique(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    index_name: *const c_char,
    key: *const u8,
    key_len: usize,
    out_id: *mut u64,
    out_payload: *mut *mut u8,
    out_payload_len: *mut usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null()
            || collection.is_null()
            || index_name.is_null()
            || out_id.is_null()
            || out_payload.is_null()
            || out_payload_len.is_null()
        {
            clear_payload_outs(out_id, out_payload, out_payload_len);
            return OBJ_ERR_INVALID_ARG;
        }
        if key.is_null() && key_len != 0 {
            clear_payload_outs(out_id, out_payload, out_payload_len);
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => {
                clear_payload_outs(out_id, out_payload, out_payload_len);
                return code;
            }
        };
        // SAFETY: index_name is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let index_str = match unsafe { cstr_to_str(index_name) } {
            Ok(s) => s,
            Err(code) => {
                clear_payload_outs(out_id, out_payload, out_payload_len);
                return code;
            }
        };
        let key_slice: &[u8] = if key_len == 0 {
            &[]
        } else {
            // SAFETY: key_len != 0 here implies key is non-null (the null+nonzero case returned above), and points to key_len readable bytes per the # Safety contract.
            unsafe { slice::from_raw_parts(key, key_len) }
        };
        // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
        let txn_ref = unsafe { (*txn).inner_ref() };
        match txn_ref.find_unique_raw(collection_str, index_str, key_slice) {
            Ok(Some((id, payload))) => {
                let len = payload.len();
                let ptr = crate::txn::alloc_bytes(&payload);
                // SAFETY: out_id is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_id = id.get() };
                // SAFETY: out_payload is non-null (checked above) and writable; ptr is a fresh alloc_bytes buffer owned by the caller.
                unsafe { *out_payload = ptr };
                // SAFETY: out_payload_len is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_payload_len = len };
                OBJ_OK
            }
            Ok(None) => {
                clear_payload_outs(out_id, out_payload, out_payload_len);
                OBJ_ERR_NOT_FOUND
            }
            Err(e) => {
                clear_payload_outs(out_id, out_payload, out_payload_len);
                // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
                db_error_code(unsafe { (*txn).db_inner() }, &e)
            }
        }
    })
}

/// Count every doc in `collection`.
///
/// # Safety
///
/// - `txn` must be live; `collection` NUL-terminated UTF-8;
///   `out_count` writable.
#[no_mangle]
pub unsafe extern "C" fn obj_count_all(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    out_count: *mut u64,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() || out_count.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
        let txn_ref = unsafe { (*txn).inner_ref() };
        match txn_ref.count_all_raw(collection_str) {
            Ok(n) => {
                // SAFETY: out_count is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_count = n };
                OBJ_OK
            }
            // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
            Err(e) => db_error_code(unsafe { (*txn).db_inner() }, &e),
        }
    })
}

/// Count index B-tree entries inside the range described by `lower` and
/// `upper`. Bound semantics match [`obj_iter_index_range`].
///
/// # Safety
///
/// As [`obj_iter_index_range`] plus `out_count` writable.
#[no_mangle]
pub unsafe extern "C" fn obj_count_index_range(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    index_name: *const c_char,
    lower: ObjBound,
    upper: ObjBound,
    out_count: *mut u64,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() || index_name.is_null() || out_count.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        // SAFETY: index_name is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let index_str = match unsafe { cstr_to_str(index_name) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        // SAFETY: lower.ptr is either NULL (treated as unbounded) or points to lower.len readable bytes per the # Safety contract.
        let lower_bound = match unsafe { bytes_to_bound(lower.ptr, lower.len, lower.inclusive != 0) }
        {
            Ok(b) => b,
            Err(code) => return code,
        };
        // SAFETY: upper.ptr is either NULL (treated as unbounded) or points to upper.len readable bytes per the # Safety contract.
        let upper_bound = match unsafe { bytes_to_bound(upper.ptr, upper.len, upper.inclusive != 0) }
        {
            Ok(b) => b,
            Err(code) => return code,
        };
        // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
        let txn_ref = unsafe { (*txn).inner_ref() };
        match txn_ref.count_index_range_raw(collection_str, index_str, lower_bound, upper_bound) {
            Ok(n) => {
                // SAFETY: out_count is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_count = n };
                OBJ_OK
            }
            // SAFETY: txn is non-null (checked above) and a live read-txn handle per the # Safety contract.
            Err(e) => db_error_code(unsafe { (*txn).db_inner() }, &e),
        }
    })
}

/// Run the full bidirectional integrity walk against `db`. Returns
/// the report through `*out_report`. On `OBJ_OK` the report holds
/// the walk's outcome (the caller checks `is_ok` via
/// [`obj_integrity_report_is_ok`]); on `OBJ_ERR_INTEGRITY` the
/// walk completed but found at least one failure. Any I/O / engine
/// error during the walk surfaces as a different error code with
/// `*out_report` set to NULL.
///
/// # Safety
///
/// - `db` must be a live db handle.
/// - `out_report` must be a writable `obj_integrity_report_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_check(
    db: *mut obj_db_t,
    out_report: *mut *mut obj_integrity_report_t,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if db.is_null() || out_report.is_null() {
            if !out_report.is_null() {
                // SAFETY: out_report is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_report = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
        let db_arc = unsafe { (*db).db_arc() };
        match db_arc.integrity_check() {
            Ok(report) => {
                let failures: Vec<String> =
                    report.failures.iter().map(|f| format!("{f:?}")).collect();
                let is_ok = report.is_ok();
                let pages_checked = report.pages_checked;
                let handle = Box::new(obj_integrity_report_t {
                    is_ok,
                    pages_checked,
                    failures,
                });
                // SAFETY: out_report is non-null (checked above) and writable; handle is a fresh Box transferred to the caller.
                unsafe { *out_report = Box::into_raw(handle) };
                if is_ok {
                    OBJ_OK
                } else {
                    OBJ_ERR_INTEGRITY
                }
            }
            Err(e) => {
                // SAFETY: out_report is non-null (checked above) and writable per the # Safety contract.
                unsafe { *out_report = ptr::null_mut() };
                // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
                db_handle_error_code(unsafe { &*db }, &e)
            }
        }
    })
}

/// `true` iff the report carries no failures. Null-tolerant —
/// returns `false` for a NULL report (a NULL report is by
/// definition not a clean one).
///
/// # Safety
///
/// If non-null, `report` must be a handle from
/// [`obj_integrity_check`] not yet freed.
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_report_is_ok(report: *const obj_integrity_report_t) -> bool {
    if report.is_null() {
        return false;
    }
    // SAFETY: report is non-null (checked above) and a live obj_integrity_report_t handle per the # Safety contract.
    catch_ffi_or(false, || unsafe { (*report).is_ok })
}

/// Number of pages the walker visited.
///
/// # Safety
///
/// If non-null, `report` must be a handle from
/// [`obj_integrity_check`] not yet freed.
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_report_pages_checked(
    report: *const obj_integrity_report_t,
) -> u64 {
    if report.is_null() {
        return 0;
    }
    // SAFETY: report is non-null (checked above) and a live obj_integrity_report_t handle per the # Safety contract.
    catch_ffi_or(0, || unsafe { (*report).pages_checked })
}

/// Number of failure entries in the report.
///
/// # Safety
///
/// As [`obj_integrity_report_is_ok`].
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_report_failure_count(
    report: *const obj_integrity_report_t,
) -> usize {
    if report.is_null() {
        return 0;
    }
    // SAFETY: report is non-null (checked above) and a live obj_integrity_report_t handle per the # Safety contract.
    catch_ffi_or(0, || unsafe { (*report).failures.len() })
}

/// Copy the Debug-formatted text of failure `index` into a fresh
/// libobj-owned buffer. Caller pairs with `obj_buf_free`.
/// Returns `OBJ_ERR_NOT_FOUND` for an out-of-range index.
///
/// # Safety
///
/// As [`obj_integrity_report_is_ok`] plus `out_string` /
/// `out_string_len` writable.
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_report_failure_at(
    report: *const obj_integrity_report_t,
    index: usize,
    out_string: *mut *mut u8,
    out_string_len: *mut usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if report.is_null() || out_string.is_null() || out_string_len.is_null() {
            if !out_string.is_null() {
                // SAFETY: out_string is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_string = ptr::null_mut() };
            }
            if !out_string_len.is_null() {
                // SAFETY: out_string_len is non-null (just checked) and writable per the # Safety contract.
                unsafe { *out_string_len = 0 };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: report is non-null (checked above) and a live obj_integrity_report_t handle per the # Safety contract.
        let failures_ref: &Vec<String> = unsafe { &(*report).failures };
        if let Some(s) = failures_ref.get(index) {
            let len = s.len();
            let ptr = crate::txn::alloc_bytes(s.as_bytes());
            // SAFETY: out_string is non-null (checked above) and writable; ptr is a fresh alloc_bytes buffer owned by the caller.
            unsafe { *out_string = ptr };
            // SAFETY: out_string_len is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_string_len = len };
            OBJ_OK
        } else {
            // SAFETY: out_string is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_string = ptr::null_mut() };
            // SAFETY: out_string_len is non-null (checked above) and writable per the # Safety contract.
            unsafe { *out_string_len = 0 };
            OBJ_ERR_NOT_FOUND
        }
    })
}

/// Free an integrity report. Null-tolerant.
///
/// # Safety
///
/// If non-null, `report` must be a handle from
/// [`obj_integrity_check`] not yet freed.
#[no_mangle]
pub unsafe extern "C" fn obj_integrity_report_free(report: *mut obj_integrity_report_t) {
    if report.is_null() {
        return;
    }
    catch_ffi_void(|| {
        // SAFETY: report is non-null (checked above) and came from Box::into_raw in obj_integrity_check; reclaimed exactly once here.
        let _ = unsafe { Box::from_raw(report) };
    });
}

/// Take a hot backup of `db` to `dest`. The destination MUST NOT
/// already exist.
///
/// # Safety
///
/// - `db` must be a live db handle.
/// - `dest` must be a NUL-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn obj_backup_to(db: *mut obj_db_t, dest: *const c_char) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if db.is_null() || dest.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: dest is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let dest_path = match unsafe { cstr_to_path(dest) } {
            Ok(p) => p,
            Err(code) => return code,
        };
        // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
        let db_arc = unsafe { (*db).db_arc() };
        match db_arc.backup_to(&dest_path) {
            Ok(()) => OBJ_OK,
            // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
            Err(e) => db_handle_error_code(unsafe { &*db }, &e),
        }
    })
}

/// Populate `out_stat` with a snapshot of the db's header +
/// collection-count summary.
///
/// # Safety
///
/// - `db` must be live.
/// - `out_stat` must be a writable `obj_stat_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_stat(db: *mut obj_db_t, out_stat: *mut obj_stat_t) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if db.is_null() || out_stat.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
        let db_arc = unsafe { (*db).db_arc() };
        let stat = match db_arc.stat() {
            Ok(s) => s,
            // SAFETY: db is non-null (checked above) and a live db handle per the # Safety contract.
            Err(e) => return db_handle_error_code(unsafe { &*db }, &e),
        };
        let collection_count = u64::try_from(stat.collections.len()).unwrap_or(u64::MAX);
        let out = obj_stat_t {
            format_major: stat.format_major,
            format_minor: stat.format_minor,
            page_size: stat.page_size,
            reserved: 0,
            page_count: stat.page_count,
            file_size_bytes: stat.file_size_bytes,
            collection_count,
        };
        // SAFETY: out_stat is non-null (checked above) and a writable obj_stat_t* per the # Safety contract; out is moved into it.
        unsafe { ptr::write(out_stat, out) };
        OBJ_OK
    })
}

/// A decoded `(lower, upper)` index-range bound pair. Aliased so
/// the [`decode_bound_pair`] signature stays readable (clippy
/// `type_complexity`).
type BoundPair = (Bound<Vec<u8>>, Bound<Vec<u8>>);

/// Decode a (ptr, len, inclusive) tuple into a
/// [`Bound<Vec<u8>>`]. A NULL ptr with `len == 0` is treated as
/// `Unbounded`; a NULL ptr with non-zero len is an error.
///
/// # Safety
///
/// If `ptr` is non-null, `len` bytes at `ptr` must be readable.
unsafe fn bytes_to_bound(
    ptr: *const u8,
    len: usize,
    inclusive: bool,
) -> Result<Bound<Vec<u8>>, obj_error_t> {
    if ptr.is_null() {
        if len != 0 {
            return Err(OBJ_ERR_INVALID_ARG);
        }
        return Ok(Bound::Unbounded);
    }
    // SAFETY: ptr is non-null here (the null case returned above) and points to len readable bytes per the # Safety contract.
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    let v = slice.to_vec();
    Ok(if inclusive {
        Bound::Included(v)
    } else {
        Bound::Excluded(v)
    })
}

/// Decode a `(lower, upper)` [`ObjBound`] pair into a [`BoundPair`].
/// Returns the first decode error encountered.
///
/// # Safety
///
/// As [`bytes_to_bound`] applied to each bound.
unsafe fn decode_bound_pair(lower: ObjBound, upper: ObjBound) -> Result<BoundPair, obj_error_t> {
    // SAFETY: lower.ptr is either NULL (treated as unbounded) or points to lower.len readable bytes per the # Safety contract.
    let lower_bound = unsafe { bytes_to_bound(lower.ptr, lower.len, lower.inclusive != 0) }?;
    // SAFETY: upper.ptr is either NULL (treated as unbounded) or points to upper.len readable bytes per the # Safety contract.
    let upper_bound = unsafe { bytes_to_bound(upper.ptr, upper.len, upper.inclusive != 0) }?;
    Ok((lower_bound, upper_bound))
}

/// Convert a NUL-terminated C string to `&str`.
///
/// # Safety
///
/// `s` must be non-null + NUL-terminated.
unsafe fn cstr_to_str<'a>(s: *const c_char) -> Result<&'a str, obj_error_t> {
    // SAFETY: s is non-null and NUL-terminated per the # Safety contract, so CStr::from_ptr can scan to the terminator.
    let cstr = unsafe { std::ffi::CStr::from_ptr(s) };
    cstr.to_str().map_err(|_| OBJ_ERR_UTF8)
}

/// Convert a NUL-terminated C string to `PathBuf`.
///
/// # Safety
///
/// `s` must be non-null + NUL-terminated.
unsafe fn cstr_to_path(s: *const c_char) -> Result<PathBuf, obj_error_t> {
    // SAFETY: s is non-null and NUL-terminated per the # Safety contract, satisfying cstr_to_str's contract.
    let str_ref = unsafe { cstr_to_str(s) }?;
    Ok(PathBuf::from(str_ref))
}

/// Helper to zero the three out-pointers when a call fails. Each
/// pointer was already null-checked by the caller.
fn clear_payload_outs(out_id: *mut u64, out_payload: *mut *mut u8, out_payload_len: *mut usize) {
    if !out_id.is_null() {
        // SAFETY: out_id is non-null (just checked) and writable per the caller's # Safety contract.
        unsafe { *out_id = 0 };
    }
    if !out_payload.is_null() {
        // SAFETY: out_payload is non-null (just checked) and writable per the caller's # Safety contract.
        unsafe { *out_payload = ptr::null_mut() };
    }
    if !out_payload_len.is_null() {
        // SAFETY: out_payload_len is non-null (just checked) and writable per the caller's # Safety contract.
        unsafe { *out_payload_len = 0 };
    }
}
