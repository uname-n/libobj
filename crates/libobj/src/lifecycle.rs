//! `obj_open` / `obj_open_with_config` / `obj_close` plus the
//! `obj_db_t` opaque handle and `obj_config_t` flat C struct.
//!
//! The Db handle is wrapped in an [`Arc`] so future txn handles
//! can each hold an `Arc<obj_engine::Db>` and outlive the
//! original `obj_open` return point on the C side without
//! introducing borrow checker grief on the Rust side. The Arc
//! shape is the minimal new public surface added to the `obj`
//! crate for FFI: see `crates/obj-rs/src/db.rs::Db::new_arc`.

use core::ffi::c_char;
use core::ptr;
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{
    catch_ffi, catch_ffi_void, error_to_code, obj_error_t, OBJ_ERR_INVALID_ARG, OBJ_ERR_PANIC,
    OBJ_ERR_UTF8, OBJ_OK,
};

/// Smallest `struct_size` byte count that fully covers the
/// encryption-key fields (`has_encryption_key` + the trailing
/// 32-byte `encryption_key`). A caller whose declared `struct_size`
/// is at least this value participates in the encryption-key
/// contract; a shorter, older caller layout is opened unencrypted
/// regardless of the bytes after its allocation. Computed at
/// compile time from the real type so it can never drift from the
/// struct definition.
const KEY_FIELDS_END_OFFSET: usize =
    core::mem::offset_of!(obj_config_t, encryption_key) + OBJ_ENCRYPTION_KEY_LEN;

/// Minimum `struct_size` the library understands: enough bytes to
/// cover the v1 knobs (`struct_size` + `sync_mode` +
/// `busy_timeout_ms` + `skip_open_check`). Computed as the
/// unpadded end of `skip_open_check`, a sound lower bound — any
/// caller declaring at least this many bytes definitely has all
/// three v1 fields fully readable. An explicit, non-zero
/// `struct_size` below this is rejected: the caller's view of
/// `obj_config_t` predates this library's oldest understood shape,
/// so the bytes cannot be interpreted safely.
const MIN_CONFIG_SIZE: usize =
    core::mem::offset_of!(obj_config_t, skip_open_check) + core::mem::size_of::<bool>();

/// Opaque database handle.
///
/// Created by [`obj_open`] / [`obj_open_with_config`]; freed by
/// [`obj_close`]. The Rust side holds an [`Arc<obj_engine::Db>`] so
/// future transaction handles can each hold an
/// `Arc` to outlive the original [`obj_open`] return point.
///
/// The struct is **opaque to C**: cbindgen emits a forward
/// declaration only (`typedef struct obj_db_t obj_db_t;`). The
/// Rust-side layout is NOT `#[repr(C)]` because the Arc field is
/// not C-compatible — C callers only ever see pointer-shaped
/// access via the API below.
pub struct obj_db_t {
    /// The Arc-wrapped [`obj_engine::Db`]. Boxed-on-the-heap inside
    /// `obj_open*`; the wrapping `Box<obj_db_t>` is freed by
    /// `obj_close` via [`Box::from_raw`], which drops the field
    /// (releasing the final Arc strong reference and the
    /// underlying file locks). Read via [`Self::db_arc`] by the
    /// sibling FFI modules (txn / queries).
    inner: Arc<obj_engine::Db>,
}

impl obj_db_t {
    /// Wrap an [`obj_engine::Db`] into a heap-allocated handle the C side
    /// holds as `obj_db_t *`. The wrapped value is stored behind
    /// an [`Arc`] so transaction-handle modules can `Arc::clone`
    /// to outlive the original `obj_open` return point on the C
    /// side.
    pub(crate) fn new(db: obj_engine::Db) -> Self {
        Self {
            inner: Arc::new(db),
        }
    }

    /// Cloneable [`Arc`] handle on the wrapped Db. Used by the FFI
    /// txn / iter modules to keep the Db alive past the original
    /// `obj_open` return point.
    pub(crate) fn db_arc(&self) -> Arc<obj_engine::Db> {
        Arc::clone(&self.inner)
    }
}

/// `SyncMode` selector for [`obj_config_t::sync_mode`].
///
/// Three variants, mapping 1:1 to [`obj_engine::SyncMode`]. Values are
/// chosen for cbindgen-stable layout: `int32_t` repr keeps the
/// type word-aligned for the surrounding `obj_config_t`.
pub type obj_sync_mode_t = i32;

/// Strongest durability — survives system-wide power loss.
pub const OBJ_SYNC_MODE_FULL: obj_sync_mode_t = 0;
/// Process-crash / kernel-panic durability.
pub const OBJ_SYNC_MODE_NORMAL: obj_sync_mode_t = 1;
/// No durability call. Tests and benchmarks only.
pub const OBJ_SYNC_MODE_OFF: obj_sync_mode_t = 2;

/// Length in bytes of the [`obj_config_t::encryption_key`] field.
/// The master encryption key is always exactly 32 bytes
/// (XChaCha20-Poly1305 / HKDF master key width); the C contract
/// encodes the length in the type rather than a separate argument.
pub const OBJ_ENCRYPTION_KEY_LEN: usize = 32;

/// Flat C struct mirroring [`obj_engine::Config`]'s public knobs.
///
/// # Forward-compatibility convention
///
/// The **first** field is `struct_size`, the byte size of the
/// `obj_config_t` the *caller* was compiled against
/// (`sizeof(obj_config_t)`). This is the standard C-ABI struct-
/// versioning trick: it lets the library grow the struct with new
/// trailing fields in future minor releases WITHOUT breaking
/// callers compiled against an older, shorter layout, and without
/// minting a fresh `obj_open_with_config2` entry point.
///
/// Callers MUST set `struct_size = sizeof(obj_config_t)` before
/// calling [`obj_open_with_config`]. The library:
///
/// - reads `struct_size` first and rejects a value it cannot make
///   sense of (smaller than the v1 layout, or absurdly large) with
///   [`OBJ_ERR_INVALID_ARG`];
/// - reads ONLY the fields the caller's declared `struct_size`
///   covers, so an older caller's shorter struct never causes the
///   library to read past the caller's allocation;
/// - treats `struct_size == 0` as "legacy / unspecified" and falls
///   back to reading the full current layout — convenient for the
///   common `obj_config_t cfg = {0}; cfg.struct_size = sizeof cfg;`
///   idiom while still tolerating a forgotten assignment defensively.
///
/// New fields added in future releases are appended AFTER the
/// existing fields and documented as "honoured only when
/// `struct_size` covers this field; otherwise the engine default
/// applies".
///
/// # Field defaults
///
/// Field-by-field defaults mirror [`obj_engine::Config::default`]:
///
/// | Field               | Default               |
/// |---------------------|-----------------------|
/// | `struct_size`       | `sizeof(obj_config_t)`|
/// | `sync_mode`         | `OBJ_SYNC_MODE_FULL`  |
/// | `busy_timeout_ms`   | `5000`                |
/// | `skip_open_check`   | `false` (0)           |
/// | `has_encryption_key`| `false` (0)           |
/// | `encryption_key`    | all-zero (ignored)    |
#[repr(C)]
pub struct obj_config_t {
    /// `sizeof(obj_config_t)` as seen by the caller. See the struct
    /// doc-comment for the forward-compatibility contract. `0` is
    /// tolerated as "legacy / unspecified" and read as the full
    /// current layout.
    pub struct_size: u32,
    /// fsync primitive used after every WAL commit.
    pub sync_mode: obj_sync_mode_t,
    /// Busy-lock timeout in milliseconds. `0` is treated as the
    /// engine's default (5 s).
    pub busy_timeout_ms: u64,
    /// Skip the open-time integrity check. Non-zero = skip.
    pub skip_open_check: bool,
    /// When non-zero, [`Self::encryption_key`] holds a 32-byte master
    /// key the database is opened with. When zero, the
    /// key bytes are ignored and the database is opened unencrypted.
    ///
    /// Honoured only when `struct_size` covers this field; an older
    /// (shorter) caller layout is opened unencrypted regardless of
    /// the bytes that happen to follow its allocation.
    pub has_encryption_key: bool,
    /// 32-byte master encryption key, consulted IFF
    /// [`Self::has_encryption_key`] is non-zero. Supplying a key
    /// requires a libobj built with the `encryption` feature on its
    /// `obj_engine` dependency; otherwise the engine returns
    /// `OBJ_ERR_UNSUPPORTED` from [`obj_open_with_config`].
    pub encryption_key: [u8; OBJ_ENCRYPTION_KEY_LEN],
}

/// Map an [`obj_sync_mode_t`] to [`obj_engine::SyncMode`]. Unknown
/// values fall back to [`obj_engine::SyncMode::Full`] — the strongest
/// durability shape — so a misencoded byte cannot silently
/// downgrade safety.
fn sync_mode_from_c(raw: obj_sync_mode_t) -> obj_engine::SyncMode {
    match raw {
        OBJ_SYNC_MODE_NORMAL => obj_engine::SyncMode::Normal,
        OBJ_SYNC_MODE_OFF => obj_engine::SyncMode::Off,
        _ => obj_engine::SyncMode::Full,
    }
}

/// Open or create a file-backed database at `path`. Default
/// configuration ([`obj_engine::Config::default`]).
///
/// On success: `*out_db` is set to a newly-allocated [`obj_db_t`]
/// (caller must free via [`obj_close`]) and `OBJ_OK` is returned.
/// On failure: `*out_db` is set to NULL and an error code is
/// returned.
///
/// # Safety
///
/// - `path` must be a non-null pointer to a NUL-terminated UTF-8
///   string. The pointee is read but not retained.
/// - `out_db` must be a non-null pointer to a writable
///   `obj_db_t*`. The pointee is unconditionally written before
///   return (NULL on failure).
#[no_mangle]
pub unsafe extern "C" fn obj_open(path: *const c_char, out_db: *mut *mut obj_db_t) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if path.is_null() || out_db.is_null() {
            if !out_db.is_null() {
                // SAFETY: out_db is non-null in this branch (just checked) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: path is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let path_buf = match unsafe { cstr_to_path(path) } {
            Ok(p) => p,
            Err(code) => {
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
                return code;
            }
        };
        match obj_engine::Db::open(&path_buf) {
            Ok(db) => {
                let handle = Box::new(obj_db_t::new(db));
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract; handle is a fresh Box for the caller to free via obj_close.
                unsafe { *out_db = Box::into_raw(handle) };
                OBJ_OK
            }
            Err(e) => {
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
                error_to_code(&e)
            }
        }
    })
}

/// As [`obj_open`] but with caller-supplied configuration.
///
/// `config->struct_size` MUST be set to `sizeof(obj_config_t)` (or
/// left `0` for the legacy / full-layout reading). See the
/// [`obj_config_t`] doc-comment for the forward-compatibility
/// contract. An explicit `struct_size` that is smaller than the
/// oldest understood layout, or larger than this library's current
/// layout, is rejected with [`OBJ_ERR_INVALID_ARG`].
///
/// # Safety
///
/// - `path` must be a non-null pointer to a NUL-terminated UTF-8
///   string.
/// - `config` must be a non-null pointer to an [`obj_config_t`]
///   whose first `config->struct_size` bytes (or, when
///   `struct_size == 0`, whose full current-layout size) are
///   readable and initialised. The pointee is read but not
///   retained.
/// - `out_db` must be a non-null pointer to a writable
///   `obj_db_t*`.
#[no_mangle]
pub unsafe extern "C" fn obj_open_with_config(
    path: *const c_char,
    config: *const obj_config_t,
    out_db: *mut *mut obj_db_t,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if path.is_null() || config.is_null() || out_db.is_null() {
            if !out_db.is_null() {
                // SAFETY: out_db is non-null in this branch (just checked) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
            }
            return OBJ_ERR_INVALID_ARG;
        }
        // SAFETY: path is non-null (checked above) and a NUL-terminated C string per the # Safety contract.
        let path_buf = match unsafe { cstr_to_path(path) } {
            Ok(p) => p,
            Err(code) => {
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
                return code;
            }
        };
        // SAFETY: config is non-null (checked above) and a readable obj_config_t prefix per the # Safety contract; build_config_from_c only reads struct_size-covered fields.
        let cfg = match unsafe { build_config_from_c(config) } {
            Ok(c) => c,
            Err(code) => {
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
                return code;
            }
        };
        match obj_engine::Db::open_with(&path_buf, cfg) {
            Ok(db) => {
                let handle = Box::new(obj_db_t::new(db));
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract; handle is a fresh Box for the caller to free via obj_close.
                unsafe { *out_db = Box::into_raw(handle) };
                OBJ_OK
            }
            Err(e) => {
                // SAFETY: out_db is non-null (checked above) and points to a writable obj_db_t* per the # Safety contract.
                unsafe { *out_db = ptr::null_mut() };
                error_to_code(&e)
            }
        }
    })
}

/// Validate `config->struct_size` and translate the caller's
/// [`obj_config_t`] prefix into an [`obj_engine::Config`].
///
/// Honours the forward-compatibility contract documented on
/// [`obj_config_t`]: only fields covered by the caller's declared
/// `struct_size` are read, so an older (shorter) caller layout is
/// never read past its allocation. The encryption-key field is
/// applied only when (a) `struct_size` covers `has_encryption_key`
/// and (b) that flag is non-zero.
///
/// # Safety
///
/// `config` must be non-null and its first `struct_size` bytes (or,
/// when `struct_size == 0`, its full current-layout size) must be a
/// readable, initialised prefix of an [`obj_config_t`].
unsafe fn build_config_from_c(
    config: *const obj_config_t,
) -> Result<obj_engine::Config, obj_error_t> {
    let current_size = core::mem::size_of::<obj_config_t>();
    // SAFETY: config is non-null per the # Safety contract; struct_size is the first field, always within the readable prefix.
    let declared = unsafe { (*config).struct_size } as usize;
    let effective = if declared == 0 {
        current_size
    } else {
        declared
    };
    if effective < MIN_CONFIG_SIZE || effective > current_size {
        return Err(OBJ_ERR_INVALID_ARG);
    }
    // SAFETY: config is non-null per the # Safety contract; effective >= MIN_CONFIG_SIZE (checked above) guarantees sync_mode is within the readable prefix.
    let sync_mode = unsafe { (*config).sync_mode };
    // SAFETY: config is non-null per the # Safety contract; effective >= MIN_CONFIG_SIZE (checked above) guarantees busy_timeout_ms is within the readable prefix.
    let busy_timeout_ms = unsafe { (*config).busy_timeout_ms };
    // SAFETY: config is non-null per the # Safety contract; effective >= MIN_CONFIG_SIZE (checked above) guarantees skip_open_check is within the readable prefix.
    let skip_open_check = unsafe { (*config).skip_open_check };
    let mut cfg = obj_engine::Config::default().sync_mode(sync_mode_from_c(sync_mode));
    if busy_timeout_ms > 0 {
        cfg = cfg.busy_timeout(Duration::from_millis(busy_timeout_ms));
    }
    cfg = cfg.skip_open_check(skip_open_check);
    // SAFETY: config is non-null per the # Safety contract; the effective >= KEY_FIELDS_END_OFFSET short-circuit guarantees has_encryption_key is within the readable prefix before it is read.
    if effective >= KEY_FIELDS_END_OFFSET && unsafe { (*config).has_encryption_key } {
        // SAFETY: config is non-null per the # Safety contract; effective >= KEY_FIELDS_END_OFFSET (checked in the enclosing if) guarantees the 32-byte encryption_key is within the readable prefix.
        let key = unsafe { (*config).encryption_key };
        cfg = cfg.encryption_key(key);
    }
    Ok(cfg)
}

/// Close a database handle. Null-tolerant.
///
/// # Safety
///
/// - If non-null, `db` must have been returned by [`obj_open`] /
///   [`obj_open_with_config`] and not yet freed. After this call
///   returns the pointer is dangling — the caller must not use it
///   again.
#[no_mangle]
pub unsafe extern "C" fn obj_close(db: *mut obj_db_t) {
    if db.is_null() {
        return;
    }
    catch_ffi_void(|| {
        // SAFETY: db is non-null (checked above) and came from Box::into_raw in obj_open* per the # Safety contract; reclaimed exactly once here.
        let _ = unsafe { Box::from_raw(db) };
    });
}

/// Convert a NUL-terminated C string to a [`PathBuf`]. The shim
/// enforces UTF-8 because obj's path layer is UTF-8 only — a non-
/// UTF-8 path becomes [`OBJ_ERR_UTF8`] rather than an opaque I/O
/// error later.
///
/// # Safety
///
/// `s` must be a non-null pointer to a NUL-terminated byte
/// sequence; the pointee is read for one strlen + length bytes.
unsafe fn cstr_to_path(s: *const c_char) -> Result<PathBuf, obj_error_t> {
    // SAFETY: s is a non-null pointer to a NUL-terminated byte sequence per the # Safety contract, read for one strlen + length bytes.
    let cstr = unsafe { CStr::from_ptr(s) };
    match cstr.to_str() {
        Ok(s) => Ok(PathBuf::from(s)),
        Err(_) => Err(OBJ_ERR_UTF8),
    }
}

const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<obj_db_t>;
};

/// Scalar value-kind selector for [`obj_index_key_encode`]. Maps a
/// C-side primitive onto the matching `obj_core::codec::Dynamic`
/// scalar so the produced bytes are obj's order-preserving field
/// encoding — the SAME encoding the typed Rust write path produces
/// via `extract_index_keys`.
///
/// `int32_t` repr (like [`obj_sync_mode_t`]) for cbindgen-stable
/// layout: cbindgen emits a `typedef int32_t obj_index_value_kind_t;`
/// plus `#define`-style constants.
pub type obj_index_value_kind_t = i32;

/// `bool` field. The value buffer is exactly 1 byte; `0` ⇒ `false`,
/// any non-zero ⇒ `true`.
pub const OBJ_INDEX_VALUE_BOOL: obj_index_value_kind_t = 0;
/// `u64` field. The value buffer is exactly 8 bytes interpreted in
/// HOST byte order (read straight from a C `uint64_t`).
pub const OBJ_INDEX_VALUE_U64: obj_index_value_kind_t = 1;
/// `i64` field. The value buffer is exactly 8 bytes interpreted in
/// HOST byte order (read straight from a C `int64_t`).
pub const OBJ_INDEX_VALUE_I64: obj_index_value_kind_t = 2;
/// `f64` field. The value buffer is exactly 8 bytes interpreted in
/// HOST byte order (read straight from a C `double`).
pub const OBJ_INDEX_VALUE_F64: obj_index_value_kind_t = 3;
/// UTF-8 string field. The value buffer is `value_len` bytes of
/// UTF-8 (NOT NUL-terminated; the length is explicit). A buffer
/// containing an embedded NUL byte is rejected — the order-
/// preserving string encoding reserves `0x00` as its terminator.
pub const OBJ_INDEX_VALUE_STRING: obj_index_value_kind_t = 4;
/// Raw bytes field. The value buffer is `value_len` arbitrary bytes
/// (embedded NULs permitted — the bytes encoding is length-prefixed).
pub const OBJ_INDEX_VALUE_BYTES: obj_index_value_kind_t = 5;

/// Encode a single index field `value` into obj's order-preserving
/// field-key bytes, writing the result into the caller's `out`
/// buffer.
///
/// This is the encoder a C consumer uses to build the `key` bytes
/// of an [`crate::obj_index_entry_t`] handed to
/// [`obj_doc_insert_indexed`](crate::obj_doc_insert_indexed) /
/// `_update_indexed` / `_delete_indexed`. The bytes it produces are
/// byte-identical to what the typed Rust write path stores, so the
/// `obj_find_unique` / `obj_iter_index_range` readers find the doc.
///
/// # Buffer contract (size query)
///
/// On success the encoded length is written to `*out_len` and the
/// bytes to `out[0..*out_len]`, and the call returns [`OBJ_OK`].
///
/// If `out_cap` is too small (including `out` being NULL with
/// `out_cap == 0`), the call writes the REQUIRED length to `*out_len`
/// and returns [`OBJ_ERR_INVALID_ARG`] WITHOUT touching `out`. So the
/// standard two-call idiom works: call once with `out_cap = 0` to
/// learn the size, allocate, then call again. (An encoded field is at
/// most `value_len + 5` bytes, so `value_len + 5` is always a safe
/// up-front capacity.)
///
/// # Safety
///
/// - `value` may be NULL IFF `value_len == 0`; otherwise it must
///   point to `value_len` readable bytes.
/// - For fixed-width kinds (`BOOL` ⇒ 1 byte, `U64`/`I64`/`F64` ⇒
///   8 bytes) `value_len` MUST equal the exact width or the call
///   returns [`OBJ_ERR_INVALID_ARG`].
/// - `out` may be NULL only when `out_cap == 0` (size-query form);
///   otherwise it must point to `out_cap` writable bytes.
/// - `out_len` must be a writable `size_t *`.
#[no_mangle]
pub unsafe extern "C" fn obj_index_key_encode(
    kind: obj_index_value_kind_t,
    value: *const u8,
    value_len: usize,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> obj_error_t {
    catch_ffi(OBJ_ERR_PANIC, || {
        if out_len.is_null()
            || (value.is_null() && value_len != 0)
            || (out.is_null() && out_cap != 0)
        {
            return OBJ_ERR_INVALID_ARG;
        }
        let value_slice: &[u8] = if value_len == 0 {
            &[]
        } else {
            // SAFETY: value_len != 0 here, so the (value.is_null() && value_len != 0) check above guarantees value is non-null and points to value_len readable bytes per the # Safety contract.
            unsafe { core::slice::from_raw_parts(value, value_len) }
        };
        let dynamic = match dynamic_from_kind(kind, value_slice) {
            Ok(d) => d,
            Err(code) => return code,
        };
        let encoded = match obj_core::index::encode_field(&dynamic) {
            Ok(e) => e,
            Err(e) => return error_to_code(&e),
        };
        // SAFETY: out_len is non-null (checked above) and out points to out_cap writable bytes (or is null with out_cap == 0, checked above), satisfying write_encoded_out's # Safety contract.
        unsafe { write_encoded_out(encoded.as_bytes(), out, out_cap, out_len) }
    })
}

/// Map a [`obj_index_value_kind_t`] + raw value bytes to the matching
/// `Dynamic` scalar. Fixed-width kinds enforce their exact byte
/// width; variable-width kinds (`STRING` / `BYTES`) take the buffer
/// verbatim. An unknown kind, a width mismatch, or non-UTF-8 string
/// bytes becomes [`OBJ_ERR_INVALID_ARG`].
fn dynamic_from_kind(
    kind: obj_index_value_kind_t,
    value: &[u8],
) -> Result<obj_core::codec::Dynamic, obj_error_t> {
    use obj_core::codec::Dynamic;
    match kind {
        OBJ_INDEX_VALUE_BOOL => {
            let [b] = value else {
                return Err(OBJ_ERR_INVALID_ARG);
            };
            Ok(Dynamic::Bool(*b != 0))
        }
        OBJ_INDEX_VALUE_U64 => fixed8(value).map(u64::from_ne_bytes).map(Dynamic::U64),
        OBJ_INDEX_VALUE_I64 => fixed8(value).map(i64::from_ne_bytes).map(Dynamic::I64),
        OBJ_INDEX_VALUE_F64 => fixed8(value).map(f64::from_ne_bytes).map(Dynamic::F64),
        OBJ_INDEX_VALUE_STRING => match core::str::from_utf8(value) {
            Ok(s) => Ok(Dynamic::String(s.to_owned())),
            Err(_) => Err(OBJ_ERR_UTF8),
        },
        OBJ_INDEX_VALUE_BYTES => Ok(Dynamic::Bytes(value.to_vec())),
        _ => Err(OBJ_ERR_INVALID_ARG),
    }
}

/// Interpret an exactly-8-byte value buffer as a `[u8; 8]`. A
/// length other than 8 is a caller contract violation.
fn fixed8(value: &[u8]) -> Result<[u8; 8], obj_error_t> {
    <[u8; 8]>::try_from(value).map_err(|_| OBJ_ERR_INVALID_ARG)
}

/// Write `encoded` into the caller's `(out, out_cap, out_len)`
/// triple per the size-query contract documented on
/// [`obj_index_key_encode`].
///
/// # Safety
///
/// `out_len` must be writable. `out` must point to `out_cap`
/// writable bytes when `out_cap >= encoded.len()`.
unsafe fn write_encoded_out(
    encoded: &[u8],
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
) -> obj_error_t {
    let needed = encoded.len();
    // SAFETY: out_len is writable per the # Safety contract.
    unsafe { *out_len = needed };
    if out_cap < needed {
        return OBJ_ERR_INVALID_ARG;
    }
    // SAFETY: out_cap >= needed here (the out_cap < needed early return is past), so out points to out_cap >= encoded.len() writable bytes per the # Safety contract; encoded.as_ptr() is a distinct readable source of needed bytes.
    unsafe { core::ptr::copy_nonoverlapping(encoded.as_ptr(), out, needed) };
    OBJ_OK
}
