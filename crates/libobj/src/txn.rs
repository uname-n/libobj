//! Transaction handles + raw-bytes CRUD entry points.
//!
//! This module owns the FFI surface for opaque
//! `obj_write_txn_t` / `obj_read_txn_t` handles and the
//! `obj_doc_*` operations that operate on raw payload bytes.
//!
//! # Two write families: primary-only vs index-maintaining
//!
//! The plain `obj_doc_insert_raw` / `obj_doc_update_raw` / `obj_doc_upsert_raw` /
//! `obj_doc_delete_raw` entry points write the **primary** record only —
//! they deliberately do NOT touch any secondary index. They remain
//! the right choice for a collection with no secondary indexes, or
//! when the caller does not need its writes reflected in the
//! index-read functions.
//!
//! The index-MAINTAINING family — [`obj_doc_insert_indexed`],
//! [`obj_doc_update_indexed`], [`obj_doc_delete_indexed`] — also
//! maintains the named secondary indexes. Because the C ABI carries
//! an opaque payload with **no schema** (index-key extraction
//! [`obj_core::index::extract_index_keys`] needs a serde-reflectable
//! typed `Document`, which C cannot supply), the CALLER supplies the
//! order-preserving FIELD key for each index value — built with
//! [`obj_index_key_encode`](crate::obj_index_key_encode) — via an
//! [`obj_index_entry_t`] array. obj does the kind-specific
//! STORAGE-key composition + `Unique` enforcement, matching the typed
//! Rust path byte-for-byte (it routes through `obj-rs`'s
//! `WriteTxn::insert_raw_indexed` / `update_raw_indexed` /
//! `delete_raw_indexed`, which share the non-generic
//! `index_maint::maintain_index_from_keys` seam with the typed
//! `Collection::<T>` path).
//!
//! **Consequence for the index-READ entry points.** The read-side
//! index functions in `queries.rs`
//! ([`obj_find_unique`](crate::obj_find_unique),
//! [`obj_iter_index_range`](crate::obj_iter_index_range),
//! [`obj_count_index_range`](crate::obj_count_index_range)) observe
//! index entries written by the typed API AND by the
//! `obj_doc_*_indexed` family. A document written through the
//! primary-only family above is still **invisible** to its secondary
//! indexes (it has no entry); a consumer that needs the write
//! reflected in an index must use the `_indexed` variant.
//!
//! # Lifetime gotcha + the Arc-handle pattern
//!
//! `obj_engine::WriteTxn<'db>` (and the wrapped
//! `obj_core::WriteTxn<'db, FileHandle>`) carry a `'db` lifetime
//! tied to the `&TxnEnv` they were begun against. On the C side
//! the txn handle lives independently of the `obj_db_t` pointer —
//! the consumer may even `obj_close` the db before they finish
//! using the txn (UB, but the contract should NOT cause a Rust-
//! level lifetime fight at construction time).
//!
//! The fix is the standard self-referential FFI trick: we keep an
//! `Arc<obj::Db>` keepalive INSIDE the txn handle, and erase the
//! lifetime on the wrapped txn to `'static` via `core::mem::transmute`.
//! Soundness rests on three invariants:
//!
//! 1. The Arc keepalive outlives the wrapped txn — they are
//!    co-located in the same `Box`, and the `Drop` for the Box
//!    drops the txn BEFORE the Arc (struct fields drop in
//!    declaration order; we put `inner` first, `_db` second).
//! 2. The wrapped txn never escapes the handle as `'static` — the
//!    `as_inner_mut` accessor erases back to `'_` at the call
//!    site so user code never observes the lie.
//! 3. The transmute is the only `unsafe` in this module; every
//!    `unsafe` op carries an explicit SAFETY block.

use core::ffi::c_char;
use core::mem::{align_of, size_of, ManuallyDrop};
use core::ptr;
use core::slice;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::ffi::CStr;
use std::sync::Arc;

use obj_engine::{Db, Id, WriteTxn};

use crate::error::{
    catch_ffi, catch_ffi_void, error_to_code, obj_error_t, OBJ_ERR_INVALID_ARG, OBJ_ERR_NOT_FOUND,
    OBJ_ERR_PANIC, OBJ_ERR_UTF8, OBJ_OK,
};
use crate::lifecycle::obj_db_t;

/// Opaque write-transaction handle.
///
/// Holds:
///
/// - `inner`: a `'static`-erased [`WriteTxn`] borrowing from the
///   shared env. Real lifetime is bound to the `_db` keepalive
///   below; the erasure is sound because the two fields drop in
///   the order declared (Rust guarantees in-declaration-order
///   field drop) and the txn never escapes as `'static`.
/// - `_db`: an Arc keepalive on the parent [`obj::Db`]; the env
///   referenced by `inner` lives at least as long as this Arc.
///
/// `ManuallyDrop` on `inner` lets us explicitly drop the txn (via
/// `obj_txn_commit` / `obj_txn_rollback` / the handle's `Drop`)
/// before the Arc is released — order matters for soundness.
pub struct obj_write_txn_t {
    /// SAFETY-load-bearing: declaration order is INNER then DB so
    /// `Drop` drops the txn first, releasing the borrow on the
    /// env, before the Arc is decremented.
    inner: ManuallyDrop<WriteTxn<'static>>,
    /// Arc keepalive. Suppresses a dead-code warning because the
    /// field is read only via Drop.
    // allow: dead_code — `db` is never read by name; it exists purely as the keepalive that
    // outlives `inner` and is released on Drop, so the lint would otherwise fire.
    #[allow(dead_code)]
    db: Arc<Db>,
}

impl Drop for obj_write_txn_t {
    fn drop(&mut self) {
        // SAFETY: `inner` is dropped exactly once here (Drop runs once) and is never taken
        // elsewhere except via the audited `into_parts` teardown (used by commit), which frees
        // the box by reinterpreting it as Box<ManuallyDrop<obj_write_txn_t>> so this Drop does
        // not also run; the txn drops before `db`, releasing the env borrow first.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}

/// Opaque read-transaction handle. Same lifetime erasure pattern
/// as [`obj_write_txn_t`].
pub struct obj_read_txn_t {
    /// `'static`-erased read txn; real lifetime tied to `db`.
    inner: ManuallyDrop<obj_engine::ReadTxn<'static>>,
    /// Arc keepalive. Read via Drop ordering and via
    /// [`Self::db_arc`] when the queries module spins out an
    /// iterator whose lifetime extends past the read txn.
    db: Arc<Db>,
}

impl Drop for obj_read_txn_t {
    fn drop(&mut self) {
        // SAFETY: `inner` is dropped exactly once here (Drop runs once) and is never taken
        // elsewhere; the txn drops before `db`, releasing the env borrow before the Arc decrements.
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}

/// Begin a write transaction against `db`. Acquires the writer
/// slot (subject to the configured busy timeout).
///
/// # Safety
///
/// - `db` must be a valid handle returned by `obj_open*` and not
///   yet closed.
/// - `out_txn` must be a writable `obj_write_txn_t *`. On success
///   it is set to a fresh handle; on failure it is set to NULL.
#[no_mangle]
pub unsafe extern "C" fn obj_txn_begin_write(
    db: *mut obj_db_t,
    out_txn: *mut *mut obj_write_txn_t,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if db.is_null() || out_txn.is_null() {
            if !out_txn.is_null() {
                // SAFETY: out_txn is non-null (just checked) and points to a writable
                // *mut obj_write_txn_t per the # Safety contract.
                unsafe { *out_txn = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: db is non-null (checked above) and a valid obj_db_t from obj_open* per the
        // # Safety contract, so dereferencing to call db_arc() is sound.
        let db_arc = unsafe { (*db).db_arc() };
        let env = db_arc.env_arc();
        let inner_core = match obj_core::WriteTxn::begin(&env, db_arc.busy_timeout()) {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: out_txn is non-null (checked above) and points to a writable
                // *mut obj_write_txn_t per the # Safety contract.
                unsafe { *out_txn = ptr::null_mut() };
                return error_to_code(&e);
            }
        };
        // SAFETY: lifetime-only transmute erasing the env borrow to 'static; the borrowed env is
        // kept alive by the `db` Arc co-located in the handle, which outlives `inner` (module docs).
        let inner_core: obj_core::WriteTxn<'static, _> =
            unsafe { core::mem::transmute(inner_core) };
        let wrapped =
            WriteTxn::from_parts(inner_core, db_arc.catalog_arc(), db_arc.reconciled_arc());
        let handle = Box::new(obj_write_txn_t {
            inner: ManuallyDrop::new(wrapped),
            db: db_arc,
        });
        // SAFETY: out_txn is non-null (checked above) and points to a writable
        // *mut obj_write_txn_t per the # Safety contract; the boxed handle is transferred to it.
        unsafe { *out_txn = Box::into_raw(handle) };
        OBJ_OK
    })
}

/// Commit a write transaction. Consumes the handle — the C
/// caller MUST NOT touch `txn` again after this returns.
///
/// # Safety
///
/// `txn` must be a non-null handle returned by
/// [`obj_txn_begin_write`] and not yet committed / rolled back.
#[no_mangle]
pub unsafe extern "C" fn obj_txn_commit(txn: *mut obj_write_txn_t) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: txn is non-null (checked above) and a handle from obj_txn_begin_write not yet
        // committed/rolled back per the # Safety contract, so it is reclaimed exactly once here.
        let handle = unsafe { Box::from_raw(txn) };
        // All raw teardown — taking `inner`, reading the `db` keepalive, and freeing the box
        // without running Drop — is localized in the audited `into_parts` primitive below, so this
        // entry point holds no unsafe reclaim that a future edit could turn back into a leaking
        // `mem::forget` (issue #1).
        let (wrapped, db) = handle.into_parts();
        let result = match wrapped.commit() {
            Ok(()) => OBJ_OK,
            Err(e) => error_to_code(&e),
        };
        // Release the `db` keepalive only AFTER the txn is finalized, preserving the
        // inner-before-db drop order the lifetime erasure relies on (module docs).
        drop(db);
        result
    })
}

/// Roll back a write transaction. Null-tolerant.
///
/// # Safety
///
/// If non-null, `txn` must be a handle returned by
/// [`obj_txn_begin_write`] and not yet committed / rolled back.
#[no_mangle]
pub unsafe extern "C" fn obj_txn_rollback(txn: *mut obj_write_txn_t) {
    if txn.is_null() {
        return;
    }
    catch_ffi_void(|| {
        // SAFETY: txn is non-null (checked above) and, per the # Safety contract, a handle from
        // obj_txn_begin_write not yet committed/rolled back, so it is reclaimed exactly once here.
        let _ = unsafe { Box::from_raw(txn) };
    });
}

/// Begin a read transaction against `db`. Pins a snapshot at the
/// current writer LSN.
///
/// # Safety
///
/// - `db` must be a valid handle returned by `obj_open*`.
/// - `out_txn` must be a writable `obj_read_txn_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_txn_begin_read(
    db: *mut obj_db_t,
    out_txn: *mut *mut obj_read_txn_t,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if db.is_null() || out_txn.is_null() {
            if !out_txn.is_null() {
                // SAFETY: out_txn is non-null (just checked) and points to a writable
                // *mut obj_read_txn_t per the # Safety contract.
                unsafe { *out_txn = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: db is non-null (checked above) and a valid obj_db_t from obj_open* per the
        // # Safety contract, so dereferencing to call db_arc() is sound.
        let db_arc = unsafe { (*db).db_arc() };
        let env = db_arc.env_arc();
        let inner_core = match obj_core::ReadTxn::begin_with_timeout(&env, db_arc.busy_timeout()) {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: out_txn is non-null (checked above) and points to a writable
                // *mut obj_read_txn_t per the # Safety contract.
                unsafe { *out_txn = ptr::null_mut() };
                return error_to_code(&e);
            }
        };
        // SAFETY: lifetime-only transmute erasing the env borrow to 'static; the borrowed env is
        // kept alive by the `db` Arc co-located in the handle, which outlives `inner` (module docs).
        let inner_core: obj_core::ReadTxn<'static, _> = unsafe { core::mem::transmute(inner_core) };
        let wrapped = obj_engine::ReadTxn::from_parts(inner_core);
        let handle = Box::new(obj_read_txn_t {
            inner: ManuallyDrop::new(wrapped),
            db: db_arc,
        });
        // SAFETY: out_txn is non-null (checked above) and points to a writable
        // *mut obj_read_txn_t per the # Safety contract; the boxed handle is transferred to it.
        unsafe { *out_txn = Box::into_raw(handle) };
        OBJ_OK
    })
}

/// End a read transaction. Releases the reader slot + the
/// pinned-snapshot WAL anchor. Null-tolerant.
///
/// # Safety
///
/// If non-null, `txn` must be a handle returned by
/// [`obj_txn_begin_read`] and not yet ended.
#[no_mangle]
pub unsafe extern "C" fn obj_txn_end_read(txn: *mut obj_read_txn_t) {
    if txn.is_null() {
        return;
    }
    catch_ffi_void(|| {
        // SAFETY: txn is non-null (checked above) and, per the # Safety contract, a handle from
        // obj_txn_begin_read not yet ended, so it is reclaimed exactly once here.
        let _ = unsafe { Box::from_raw(txn) };
    });
}

/// `type_version` stamped on documents written through the C ABI
/// raw-bytes path. Mirrors `obj-rs`'s crate-private
/// `RAW_BYTES_TYPE_VERSION`; the plain `obj_doc_insert_raw` family and
/// the index-maintaining `obj_doc_*_indexed` family both stamp this
/// so a doc inserted one way and read the other observes the same
/// header version.
const C_RAW_BYTES_TYPE_VERSION: u32 = 1;

/// One secondary-index maintenance entry: the index `index_name` is
/// to gain (or lose) the document under the field key in
/// `key[0..key_len]`.
///
/// `key` is the ORDER-PRESERVING field encoding of the indexed
/// value, produced by
/// [`obj_index_key_encode`](crate::obj_index_key_encode). obj
/// composes the kind-specific storage key (it appends the document
/// id for `Standard` / `Each` / `Composite`, uses the key as-is with
/// the id as value for `Unique`); the caller supplies only the field
/// encoding.
#[repr(C)]
pub struct obj_index_entry_t {
    /// NUL-terminated UTF-8 name of the index this entry maintains.
    /// Must name an `Active` index on the collection or the call
    /// returns `OBJ_ERR_NOT_FOUND`.
    pub index_name: *const c_char,
    /// Pointer to the order-preserving field-key bytes. May be NULL
    /// IFF `key_len == 0`.
    pub key: *const u8,
    /// Length of the field-key bytes at `key`.
    pub key_len: usize,
}

/// Insert a raw-bytes document into `collection` AND maintain the
/// named secondary indexes from the caller-supplied entries. On
/// success `*out_id` is set to the freshly-allocated id.
///
/// Unlike [`obj_doc_insert_raw`] (primary record only), this is the
/// index-maintaining write: each `entries[i]` names an
/// index and supplies the order-preserving field key (build it with
/// [`obj_index_key_encode`](crate::obj_index_key_encode)) the document
/// indexes under. obj does the kind-specific storage-key composition
/// and enforces `Unique` constraints, so the written doc is
/// immediately visible to [`obj_find_unique`](crate::obj_find_unique)
/// / [`obj_iter_index_range`](crate::obj_iter_index_range).
///
/// Transaction contract: the primary insert + every index update
/// share one WAL transaction. An unknown index name returns
/// `OBJ_ERR_NOT_FOUND`; a `Unique` collision returns
/// `OBJ_ERR_INVALID_ARG`. On either error, the transaction is still
/// uncommitted but may be dirty; call [`obj_txn_rollback`] (or avoid
/// committing the handle) so no half-written state commits.
///
/// # Safety
///
/// - `txn` must be a live write-txn handle.
/// - `collection` must be a non-null NUL-terminated UTF-8 string.
/// - `payload` may be NULL IFF `payload_len == 0`.
/// - `entries` may be NULL IFF `entry_count == 0`; otherwise it must
///   point to `entry_count` readable [`obj_index_entry_t`], each with
///   a valid `index_name` + (`key`, `key_len`) per that struct's
///   contract.
/// - `out_id` must be a writable `uint64_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_doc_insert_indexed(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    payload: *const u8,
    payload_len: usize,
    entries: *const obj_index_entry_t,
    entry_count: usize,
    out_id: *mut u64,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() || out_id.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) NUL-terminated UTF-8 and (payload,
        // payload_len) satisfies the read_indexed_inputs contract per this fn's # Safety contract.
        let inputs = match unsafe { read_indexed_inputs(collection, payload, payload_len) } {
            Ok(v) => v,
            Err(code) => return code,
        };
        // SAFETY: (entries, entry_count) follows the entries_from_c contract per this fn's
        // # Safety contract (NULL allowed only when entry_count == 0).
        let add = match unsafe { entries_from_c(entries, entry_count) } {
            Ok(v) => v,
            Err(code) => return code,
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.insert_raw_indexed(
            &inputs.collection,
            &inputs.payload,
            C_RAW_BYTES_TYPE_VERSION,
            &add,
        ) {
            Ok(id) => {
                // SAFETY: out_id is non-null (checked above) and points to a writable u64 per
                // the # Safety contract.
                unsafe { *out_id = id.get() };
                OBJ_OK
            }
            Err(e) => error_to_code(&e),
        }
    })
}

/// Update the document at `id` in `collection` to `payload` AND move
/// its secondary-index entries from the `remove` field keys to the
/// `add` field keys.
///
/// obj cannot re-derive the OLD index keys from the stored opaque
/// payload, so the caller supplies BOTH arrays: `remove` is what the
/// document indexed under before, `add` is what it indexes under
/// after. Returns `OBJ_ERR_NOT_FOUND` if `id` is absent.
///
/// Transaction contract: if index maintenance fails after the primary
/// update is staged, the transaction remains uncommitted but may be
/// dirty; call [`obj_txn_rollback`] rather than committing it to
/// preserve atomicity.
///
/// # Safety
///
/// As [`obj_doc_insert_indexed`] plus: `remove` / `add` each follow
/// the `entries` / `entry_count` contract; `id` must come from a
/// prior insert on the same collection.
#[no_mangle]
// allow: C-ABI shape — these args map 1:1 to the public header signature; grouping them into a
// struct would break the stable C ABI.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn obj_doc_update_indexed(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    id: u64,
    payload: *const u8,
    payload_len: usize,
    remove: *const obj_index_entry_t,
    remove_count: usize,
    add: *const obj_index_entry_t,
    add_count: usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (checked above) NUL-terminated UTF-8 and (payload,
        // payload_len) satisfies the read_indexed_inputs contract per this fn's # Safety contract.
        let inputs = match unsafe { read_indexed_inputs(collection, payload, payload_len) } {
            Ok(v) => v,
            Err(code) => return code,
        };
        // SAFETY: each (remove, remove_count) and (add, add_count) pair follows the entries
        // contract per this fn's # Safety contract.
        let (remove_v, add_v) =
            match unsafe { read_remove_add(remove, remove_count, add, add_count) } {
                Ok(v) => v,
                Err(code) => return code,
            };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.update_raw_indexed(
            &inputs.collection,
            id,
            &inputs.payload,
            C_RAW_BYTES_TYPE_VERSION,
            &remove_v,
            &add_v,
        ) {
            Ok(()) => OBJ_OK,
            Err(obj_engine::Error::CollectionNotFound { .. }) => OBJ_ERR_NOT_FOUND,
            Err(e) => error_to_code(&e),
        }
    })
}

/// Delete the document at `id` in `collection` AND remove its
/// secondary-index entries given the caller-supplied `remove` field
/// keys. Returns `OBJ_ERR_NOT_FOUND` if `id` is absent.
///
/// As with [`obj_doc_update_indexed`], obj cannot re-derive the
/// index keys from stored bytes, so the caller supplies the field
/// keys the document indexed under.
///
/// Transaction contract: if index maintenance fails after the primary
/// delete is staged, the transaction remains uncommitted but may be
/// dirty; call [`obj_txn_rollback`] rather than committing it to
/// preserve atomicity.
///
/// # Safety
///
/// As [`obj_doc_insert_indexed`]; `remove` / `remove_count` follow
/// the `entries` contract.
#[no_mangle]
pub unsafe extern "C" fn obj_doc_delete_indexed(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    id: u64,
    remove: *const obj_index_entry_t,
    remove_count: usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (checked above) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        // SAFETY: (remove, remove_count) follows the entries_from_c contract per this fn's
        // # Safety contract (NULL allowed only when remove_count == 0).
        let remove_v = match unsafe { entries_from_c(remove, remove_count) } {
            Ok(v) => v,
            Err(code) => return code,
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.delete_raw_indexed(collection_str, id, &remove_v) {
            Ok(true) => OBJ_OK,
            Ok(false) => OBJ_ERR_NOT_FOUND,
            Err(e) => error_to_code(&e),
        }
    })
}

/// Owned, validated inputs shared by the indexed write entry points:
/// the collection name + the payload bytes copied out of the C
/// pointers. Returned by [`read_indexed_inputs`].
struct IndexedInputs {
    collection: String,
    payload: Vec<u8>,
}

/// Validate + copy the `collection` C string and `(payload,
/// payload_len)` buffer into owned Rust values. Centralises the
/// null / UTF-8 / payload-null checks the indexed entry points share.
///
/// # Safety
///
/// `collection` must be non-null NUL-terminated UTF-8; `payload` may
/// be NULL only when `payload_len == 0`, else it points to
/// `payload_len` readable bytes.
unsafe fn read_indexed_inputs(
    collection: *const c_char,
    payload: *const u8,
    payload_len: usize,
) -> Result<IndexedInputs, obj_error_t> {
    if payload.is_null() && payload_len != 0 {
        return Err(OBJ_ERR_INVALID_ARG);
    }
    // SAFETY: collection is a non-null NUL-terminated UTF-8 string per the # Safety contract.
    let collection = unsafe { cstr_to_str(collection) }?.to_owned();
    let payload = if payload_len == 0 {
        Vec::new()
    } else {
        // SAFETY: payload_len != 0 here and the null-with-nonzero-len case was rejected above, so
        // payload points to payload_len readable bytes per the # Safety contract.
        unsafe { slice::from_raw_parts(payload, payload_len) }.to_vec()
    };
    Ok(IndexedInputs {
        collection,
        payload,
    })
}

/// Validate + materialise the `remove` and `add` entry arrays for
/// [`obj_doc_update_indexed`]. Split out so the entry point stays
/// within the 60-line budget.
///
/// # Safety
///
/// Each `(ptr, count)` pair follows the `entries` / `entry_count`
/// contract documented on [`obj_doc_insert_indexed`].
// allow: the returned tuple-of-Vec-of-tuples is the exact owned shape the obj-rs raw-indexed API
// consumes; introducing a named type would only add indirection for this one internal helper.
#[allow(clippy::type_complexity)]
unsafe fn read_remove_add(
    remove: *const obj_index_entry_t,
    remove_count: usize,
    add: *const obj_index_entry_t,
    add_count: usize,
) -> Result<(Vec<(String, Vec<u8>)>, Vec<(String, Vec<u8>)>), obj_error_t> {
    // SAFETY: (remove, remove_count) follows the entries_from_c contract per this fn's # Safety contract.
    let remove_v = unsafe { entries_from_c(remove, remove_count) }?;
    // SAFETY: (add, add_count) follows the entries_from_c contract per this fn's # Safety contract.
    let add_v = unsafe { entries_from_c(add, add_count) }?;
    Ok((remove_v, add_v))
}

/// Convert a C `obj_index_entry_t` array into the owned
/// `Vec<(index_name, field_key)>` the obj-rs raw-indexed API takes.
///
/// Copies each entry's NUL-terminated UTF-8 `index_name` + its
/// `key[0..key_len]` bytes into owned values, so the result outlives
/// the borrowed C pointers. A non-UTF-8 name becomes
/// [`OBJ_ERR_UTF8`]; a NULL `key` with non-zero `key_len` becomes
/// [`OBJ_ERR_INVALID_ARG`].
///
/// The loop is bounded by `count`, the caller-
/// supplied entry count.
///
/// # Safety
///
/// `entries` may be NULL IFF `count == 0`; otherwise it must point to
/// `count` readable [`obj_index_entry_t`], each with a valid
/// `index_name` pointer and a `(key, key_len)` pair per the struct
/// contract.
unsafe fn entries_from_c(
    entries: *const obj_index_entry_t,
    count: usize,
) -> Result<Vec<(String, Vec<u8>)>, obj_error_t> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if entries.is_null() {
        return Err(OBJ_ERR_INVALID_ARG);
    }
    // SAFETY: count != 0 here and entries is non-null (checked above), so per the # Safety
    // contract it points to `count` readable obj_index_entry_t.
    let slice = unsafe { slice::from_raw_parts(entries, count) };
    let mut out: Vec<(String, Vec<u8>)> = Vec::with_capacity(count);
    for entry in slice {
        if entry.index_name.is_null() || (entry.key.is_null() && entry.key_len != 0) {
            return Err(OBJ_ERR_INVALID_ARG);
        }
        // SAFETY: entry.index_name is non-null (just checked) and a NUL-terminated UTF-8 name
        // per the struct contract.
        let name = unsafe { cstr_to_str(entry.index_name) }?.to_owned();
        let key = if entry.key_len == 0 {
            Vec::new()
        } else {
            // SAFETY: key_len != 0 here and the null-with-nonzero-len case was rejected just above,
            // so entry.key points to key_len readable bytes per the struct contract.
            unsafe { slice::from_raw_parts(entry.key, entry.key_len) }.to_vec()
        };
        out.push((name, key));
    }
    Ok(out)
}

/// Insert a raw-bytes document into `collection`. The collection
/// is lazy-created if absent. On success `*out_id` is set to the
/// freshly-allocated id.
///
/// **Secondary indexes are NOT maintained**: this writes
/// the primary record only, so the doc is invisible to
/// `obj_find_unique` / `obj_iter_index_range` /
/// `obj_count_index_range`. Use [`obj_doc_insert_indexed`] to
/// maintain secondary indexes on insert.
///
/// # Safety
///
/// - `txn` must be a live write txn handle.
/// - `collection` must be a non-null NUL-terminated UTF-8 string.
/// - `payload` may be null IFF `payload_len == 0`; otherwise it
///   must point to `payload_len` readable bytes.
/// - `out_id` must be a writable `uint64_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_doc_insert_raw(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    payload: *const u8,
    payload_len: usize,
    out_id: *mut u64,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() || out_id.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        if payload.is_null() && payload_len != 0 {
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: collection is non-null (checked above) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        let payload_slice: &[u8] = if payload_len == 0 {
            &[]
        } else {
            // SAFETY: payload_len != 0 here and the null-with-nonzero-len case was rejected above,
            // so payload points to payload_len readable bytes per the # Safety contract.
            unsafe { slice::from_raw_parts(payload, payload_len) }
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.insert_raw_bytes(collection_str, payload_slice) {
            Ok(id) => {
                // SAFETY: out_id is non-null (checked above) and points to a writable u64 per
                // the # Safety contract.
                unsafe { *out_id = id.get() };
                OBJ_OK
            }
            Err(e) => error_to_code(&e),
        }
    })
}

/// Fetch the document at `id` in `collection`. On `OBJ_OK` the
/// caller owns `*out_payload` (length `*out_payload_len`) and
/// MUST free it via `obj_buf_free`. Returns
/// `OBJ_ERR_NOT_FOUND` if the document is absent.
///
/// # Safety
///
/// - `txn` must be a live read txn handle.
/// - `collection` must be a non-null NUL-terminated UTF-8 string.
/// - `out_payload` and `out_payload_len` must be writable pointers.
#[no_mangle]
pub unsafe extern "C" fn obj_doc_get(
    txn: *mut obj_read_txn_t,
    collection: *const c_char,
    id: u64,
    out_payload: *mut *mut u8,
    out_payload_len: *mut usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null()
            || collection.is_null()
            || out_payload.is_null()
            || out_payload_len.is_null()
        {
            if !out_payload.is_null() {
                // SAFETY: out_payload is non-null (just checked) and a writable *mut *mut u8 per
                // the # Safety contract.
                unsafe { *out_payload = ptr::null_mut() };
            }
            if !out_payload_len.is_null() {
                // SAFETY: out_payload_len is non-null (just checked) and a writable *mut usize per
                // the # Safety contract.
                unsafe { *out_payload_len = 0 };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            // SAFETY: out_payload is non-null (top guard passed) and a writable *mut *mut u8 per
            // the # Safety contract.
            unsafe { *out_payload = ptr::null_mut() };
            // SAFETY: out_payload_len is non-null (top guard passed) and a writable *mut usize per
            // the # Safety contract.
            unsafe { *out_payload_len = 0 };
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (top guard passed) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => {
                // SAFETY: out_payload is non-null (top guard passed) and a writable *mut *mut u8
                // per the # Safety contract.
                unsafe { *out_payload = ptr::null_mut() };
                // SAFETY: out_payload_len is non-null (top guard passed) and a writable *mut usize
                // per the # Safety contract.
                unsafe { *out_payload_len = 0 };
                return code;
            }
        };
        // SAFETY: txn is non-null (top guard passed) and a live read-txn handle per the # Safety
        // contract; inner_ref borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_ref() };
        match txn_ref.get_raw_bytes(collection_str, id) {
            Ok(Some(bytes)) => {
                let len = bytes.len();
                let ptr = alloc_bytes(&bytes);
                // SAFETY: out_payload is non-null (top guard passed) and writable per the
                // # Safety contract; ptr is the alloc_bytes allocation the caller frees via obj_buf_free.
                unsafe { *out_payload = ptr };
                // SAFETY: out_payload_len is non-null (top guard passed) and a writable *mut usize
                // per the # Safety contract.
                unsafe { *out_payload_len = len };
                OBJ_OK
            }
            Ok(None) => {
                // SAFETY: out_payload is non-null (top guard passed) and a writable *mut *mut u8
                // per the # Safety contract.
                unsafe { *out_payload = ptr::null_mut() };
                // SAFETY: out_payload_len is non-null (top guard passed) and a writable *mut usize
                // per the # Safety contract.
                unsafe { *out_payload_len = 0 };
                OBJ_ERR_NOT_FOUND
            }
            Err(e) => {
                // SAFETY: out_payload is non-null (top guard passed) and a writable *mut *mut u8
                // per the # Safety contract.
                unsafe { *out_payload = ptr::null_mut() };
                // SAFETY: out_payload_len is non-null (top guard passed) and a writable *mut usize
                // per the # Safety contract.
                unsafe { *out_payload_len = 0 };
                error_to_code(&e)
            }
        }
    })
}

/// Update the document at `id` in `collection` with `payload`.
/// Returns `OBJ_ERR_NOT_FOUND` if the id is absent.
///
/// **Secondary indexes are NOT maintained**: use
/// [`obj_doc_update_indexed`] to move a document's index entries on
/// update.
///
/// # Safety
///
/// As [`obj_doc_insert_raw`] plus: `id` must have been returned by a
/// prior `obj_doc_insert_raw` against the same collection.
#[no_mangle]
pub unsafe extern "C" fn obj_doc_update_raw(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    id: u64,
    payload: *const u8,
    payload_len: usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        if payload.is_null() && payload_len != 0 {
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (checked above) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        let payload_slice: &[u8] = if payload_len == 0 {
            &[]
        } else {
            // SAFETY: payload_len != 0 here and the null-with-nonzero-len case was rejected above,
            // so payload points to payload_len readable bytes per the # Safety contract.
            unsafe { slice::from_raw_parts(payload, payload_len) }
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.update_raw_bytes(collection_str, id, payload_slice) {
            Ok(()) => OBJ_OK,
            Err(obj_engine::Error::CollectionNotFound { .. }) => OBJ_ERR_NOT_FOUND,
            Err(e) => error_to_code(&e),
        }
    })
}

/// Delete the document at `id` in `collection`. Returns
/// `OBJ_ERR_NOT_FOUND` if the id is absent.
///
/// **Secondary indexes are NOT maintained**: a delete
/// through this path leaves any secondary-index entry behind. Use
/// [`obj_doc_delete_indexed`] to remove the index entries too.
///
/// # Safety
///
/// As [`obj_doc_update_raw`].
#[no_mangle]
pub unsafe extern "C" fn obj_doc_delete_raw(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    id: u64,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (checked above) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.delete_raw_bytes(collection_str, id) {
            Ok(true) => OBJ_OK,
            Ok(false) => OBJ_ERR_NOT_FOUND,
            Err(e) => error_to_code(&e),
        }
    })
}

/// Insert-or-replace the document at `id` in `collection` with
/// `payload`. The collection is lazy-created if absent.
///
/// **Secondary indexes are NOT maintained**. There is no
/// `_indexed` upsert variant: an upsert that must maintain indexes
/// has to supply the OLD field keys to remove, which the
/// [`obj_doc_update_indexed`] (when the id exists) /
/// [`obj_doc_insert_indexed`] (when it does not) pair expresses
/// directly.
///
/// # Safety
///
/// As [`obj_doc_insert_raw`].
#[no_mangle]
pub unsafe extern "C" fn obj_doc_upsert_raw(
    txn: *mut obj_write_txn_t,
    collection: *const c_char,
    id: u64,
    payload: *const u8,
    payload_len: usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if txn.is_null() || collection.is_null() {
            return OBJ_ERR_INVALID_ARG;
        }
        if payload.is_null() && payload_len != 0 {
            return OBJ_ERR_INVALID_ARG;
        }
        let Some(id) = Id::try_new(id) else {
            return OBJ_ERR_INVALID_ARG;
        };
        // SAFETY: collection is non-null (checked above) and NUL-terminated UTF-8 per the
        // # Safety contract.
        let collection_str = match unsafe { cstr_to_str(collection) } {
            Ok(s) => s,
            Err(code) => return code,
        };
        let payload_slice: &[u8] = if payload_len == 0 {
            &[]
        } else {
            // SAFETY: payload_len != 0 here and the null-with-nonzero-len case was rejected above,
            // so payload points to payload_len readable bytes per the # Safety contract.
            unsafe { slice::from_raw_parts(payload, payload_len) }
        };
        // SAFETY: txn is non-null (checked above) and a live write-txn handle per the # Safety
        // contract; inner_mut borrows it for this expression only.
        let txn_ref = unsafe { (*txn).inner_mut() };
        match txn_ref.upsert_raw_bytes(collection_str, id, payload_slice) {
            Ok(()) => OBJ_OK,
            Err(e) => error_to_code(&e),
        }
    })
}

/// Free a buffer returned by an `obj_doc_get` / iteration call.
/// Pairs with the [`alloc_bytes`] allocation. Null-tolerant;
/// passing a pointer that did NOT come from libobj is undefined
/// behaviour.
///
/// # Safety
///
/// - `ptr` may be NULL (no-op).
/// - If non-null, `ptr` MUST have been produced by a prior
///   libobj call that documents this buffer-ownership convention.
#[no_mangle]
pub unsafe extern "C" fn obj_buf_free(ptr: *mut u8) {
    // SAFETY: ptr is either null (tolerated) or a valid alloc_bytes allocation
    // per the # Safety contract.
    unsafe { dealloc_bytes(ptr) };
}

/// Allocate a length-prefixed buffer: write `data.len()` as a `usize`
/// header, copy `data` after it, and return a pointer to the data region.
/// The caller MUST free via [`dealloc_bytes`] / [`obj_buf_free`].
pub(crate) fn alloc_bytes(data: &[u8]) -> *mut u8 {
    let Some(total) = size_of::<usize>().checked_add(data.len()) else {
        handle_alloc_error(Layout::new::<u8>())
    };
    // unreachable error: align_of::<usize>() is always a valid power-of-two alignment
    let Ok(layout) = Layout::from_size_align(total, align_of::<usize>()) else {
        handle_alloc_error(Layout::new::<u8>())
    };
    // SAFETY: layout has non-zero size (total >= size_of::<usize>() >= 1).
    let base = unsafe { alloc(layout) };
    if base.is_null() {
        handle_alloc_error(layout);
    }
    // SAFETY: base is valid for `total` bytes; write_unaligned avoids requiring usize-alignment
    // on the raw pointer (the layout already aligns to align_of::<usize>(), but the cast would
    // trigger cast_ptr_alignment; write_unaligned is correct here regardless).
    unsafe { ptr::write_unaligned(base.cast::<usize>(), data.len()) };
    if !data.is_empty() {
        // SAFETY: base + size_of::<usize>() is within the allocation; data is data.len() bytes.
        unsafe {
            base.add(size_of::<usize>())
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
    }
    // SAFETY: base + size_of::<usize>() is within the allocation (total >= size_of::<usize>()).
    unsafe { base.add(size_of::<usize>()) }
}

/// Deallocate a buffer produced by [`alloc_bytes`]. Null-tolerant.
///
/// # Safety
///
/// If non-null, `ptr` must have been returned by [`alloc_bytes`] and
/// not yet freed.
pub(crate) unsafe fn dealloc_bytes(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: ptr was returned by alloc_bytes as base + size_of::<usize>(); subtracting recovers base.
    let base = unsafe { ptr.sub(size_of::<usize>()) };
    // SAFETY: base points to the usize written by alloc_bytes via write_unaligned; read_unaligned matches.
    let len = unsafe { ptr::read_unaligned(base.cast::<usize>()) };
    // cannot overflow: alloc_bytes checked the sum before allocating
    let total = size_of::<usize>() + len;
    // unreachable error: same layout as alloc_bytes constructed
    let Ok(layout) = Layout::from_size_align(total, align_of::<usize>()) else {
        handle_alloc_error(Layout::new::<u8>())
    };
    // SAFETY: base and layout exactly match those used by alloc_bytes.
    unsafe { dealloc(base, layout) };
}

/// Convert a NUL-terminated C string to a `&str`. Returns
/// [`OBJ_ERR_UTF8`] for non-UTF-8 bytes.
///
/// # Safety
///
/// `s` must be a non-null pointer to a NUL-terminated byte
/// sequence.
unsafe fn cstr_to_str<'a>(s: *const c_char) -> Result<&'a str, obj_error_t> {
    // SAFETY: s is a non-null pointer to a NUL-terminated byte sequence per the # Safety contract.
    let cstr = unsafe { CStr::from_ptr(s) };
    cstr.to_str().map_err(|_| OBJ_ERR_UTF8)
}

impl obj_write_txn_t {
    /// Mutable access to the wrapped `WriteTxn`. The returned
    /// reference's lifetime is bound to `&mut self`; the C-side
    /// caller observes this as "the txn handle must outlive the
    /// operation".
    fn inner_mut(&mut self) -> &mut WriteTxn<'static> {
        &mut self.inner
    }

    /// Consume the boxed handle into its two owned parts — the wrapped
    /// [`WriteTxn`] and the `Db` keepalive Arc — freeing the box
    /// allocation WITHOUT running `Drop for obj_write_txn_t`.
    ///
    /// This is the SINGLE audited teardown primitive for a write-txn
    /// handle; [`obj_txn_commit`] is its only caller. Localizing the raw
    /// `ManuallyDrop::take` + `ptr::read` + drop-suppressing box reclaim
    /// here (rather than open-coding it at the FFI boundary) means the
    /// entry point contains no raw teardown, so a future edit cannot
    /// reintroduce a leaking `core::mem::forget` at the call site — the
    /// regression that was issue #1.
    ///
    /// The caller MUST finalize the returned `WriteTxn` (commit or drop)
    /// BEFORE releasing the returned Arc, preserving the inner-before-db
    /// drop order the `'static` lifetime erasure relies on (module docs).
    fn into_parts(mut self: Box<Self>) -> (WriteTxn<'static>, Arc<Db>) {
        // SAFETY: `self` is a uniquely-owned `Box` consumed here, built by obj_txn_begin_write, so
        // `inner` has never been taken and `db` has never been read. Within the one block:
        //   - `ManuallyDrop::take` moves `inner` out exactly once (it is live, taken nowhere else);
        //   - `ptr::read` copies `db` out exactly once (a valid, aligned, initialized Arc<Db>);
        //   - `Box::into_raw` then reinterprets the SAME allocation as
        //     `Box<ManuallyDrop<obj_write_txn_t>>` (identical layout) and drops it, freeing the
        //     heap slot while running NO field destructor — so neither the already-taken `inner`
        //     nor the already-read `db` is double-dropped. (`mem::forget` would skip the
        //     deallocation and leak the slot; that was the issue #1 bug.)
        unsafe {
            let wrapped = ManuallyDrop::take(&mut self.inner);
            let db = core::ptr::read(&raw const self.db);
            let raw = Box::into_raw(self).cast::<ManuallyDrop<obj_write_txn_t>>();
            drop(Box::from_raw(raw));
            (wrapped, db)
        }
    }
}

impl obj_read_txn_t {
    /// Shared access to the wrapped `ReadTxn`. The returned
    /// reference's lifetime is bound to `&self`.
    pub(crate) fn inner_ref(&self) -> &obj_engine::ReadTxn<'static> {
        &self.inner
    }

    /// Cloneable Arc handle on the parent Db. Used by the queries
    /// module to construct iterators whose lifetime extends past
    /// the read txn (the iterator owns its own snapshot).
    pub(crate) fn db_arc(&self) -> Arc<Db> {
        Arc::clone(&self.db)
    }
}
