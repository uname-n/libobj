//! `libobj` — C ABI surface for the obj embedded document database.
//!
//! This crate is the **`unsafe` boundary** of obj. Every C entry
//! point is `unsafe extern "C"`; internally each shim is a short
//! piece of glue that validates pointers, calls into the safe-Rust
//! [`obj`] API, and converts the [`obj_engine::Error`] result into an
//! [`obj_error_t`] code + out-pointer.
//!
//! # Pointer / lifetime conventions
//!
//! - All opaque handle pointers (`obj_db_t*`, `obj_write_txn_t*`,
//!   etc.) returned by this library are obtained by leaking a
//!   `Box<T>`. The corresponding `obj_*_close` / `_free` /
//!   `_rollback` calls reclaim ownership via [`Box::from_raw`].
//! - All byte buffers returned from this library are allocated as
//!   `Box<[u8]>` and MUST be freed via `obj_free_buffer`.
//!   Mixing Rust's allocator with C's `malloc` is undefined.
//! - Null handles are treated specifically per function: close /
//!   free / rollback are null-tolerant (no-op); every other entry
//!   point returns [`OBJ_ERR_INVALID_ARG`] when given a null
//!   handle / out-pointer.

// allow: this crate is the C-ABI boundary; every entry point is `unsafe extern
// "C"`. unsafe is confined here and audited per-block via `// SAFETY:` (Rule 8).
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
// allow: C-ABI types use snake_case (`obj_db_t`, ...) to match the public header.
#![allow(non_camel_case_types)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
// Rules 7 & 8 — shipping code never reaches for unwrap/expect/panic-family
// constructs, and every `unsafe` block carries a `// SAFETY:` contract. Gated on
// `not(test)` so unit tests (cfg test) and integration tests (separate crates)
// keep using unwrap/expect/panic without scattered #[allow]s; only production
// code paths are held to the bound.
#![cfg_attr(not(test), deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_in_result,
    clippy::get_unwrap,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::undocumented_unsafe_blocks
))]

mod error;
mod lifecycle;
mod queries;
mod txn;

pub use error::{
    obj_error_t, obj_strerror, OBJ_ERR_BUSY, OBJ_ERR_CORRUPTION, OBJ_ERR_INTEGRITY,
    OBJ_ERR_INVALID_ARG, OBJ_ERR_IO, OBJ_ERR_NOT_FOUND, OBJ_ERR_RESERVED_MAX, OBJ_ERR_UNSUPPORTED,
    OBJ_ERR_UTF8, OBJ_OK,
};
pub use lifecycle::{
    obj_close, obj_config_t, obj_db_t, obj_index_key_encode, obj_index_value_kind_t, obj_open,
    obj_open_with_config, obj_sync_mode_t, OBJ_ENCRYPTION_KEY_LEN, OBJ_INDEX_VALUE_BOOL,
    OBJ_INDEX_VALUE_BYTES, OBJ_INDEX_VALUE_F64, OBJ_INDEX_VALUE_I64, OBJ_INDEX_VALUE_STRING,
    OBJ_INDEX_VALUE_U64, OBJ_SYNC_MODE_FULL, OBJ_SYNC_MODE_NORMAL, OBJ_SYNC_MODE_OFF,
};
pub use queries::{
    obj_backup_to, obj_count_all, obj_count_index_range, obj_find_unique, obj_integrity_check,
    obj_integrity_report_failure_at, obj_integrity_report_failure_count, obj_integrity_report_free,
    obj_integrity_report_is_ok, obj_integrity_report_pages_checked, obj_integrity_report_t,
    obj_iter_all, obj_iter_free, obj_iter_index_range, obj_iter_next, obj_iter_t, obj_stat,
    obj_stat_t, ObjBound,
};
pub use txn::{
    obj_doc_delete_raw, obj_doc_delete_indexed, obj_doc_get, obj_doc_insert_raw,
    obj_doc_insert_indexed, obj_doc_update_raw, obj_doc_update_indexed, obj_doc_upsert_raw,
    obj_free_buffer, obj_index_entry_t,
    obj_read_txn_t, obj_txn_begin_read, obj_txn_begin_write, obj_txn_commit, obj_txn_end_read,
    obj_txn_rollback, obj_write_txn_t,
};
