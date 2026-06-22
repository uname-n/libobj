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
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{
    catch_ffi, catch_ffi_or, catch_ffi_void, error_to_code, obj_error_t, OBJ_ERR_INVALID_ARG,
    OBJ_ERR_PANIC, OBJ_ERR_UTF8, OBJ_OK,
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

/// Number of recent error strings a handle retains before the
/// oldest is reclaimed. Bounds per-handle diagnostic memory at this
/// many [`CString`]s (never grows without limit, however many errors
/// a long-lived handle records) while giving a pointer returned by
/// [`obj_errmsg`] a grace window of this many *subsequent*
/// error-recording calls — on any thread — before its backing
/// storage can be freed. See the [`obj_errmsg`] doc for the caller
/// contract this window implies. Keep this value in sync with the
/// "16" quoted in that doc-comment (and thus the generated header).
const ERROR_RING_LEN: usize = 16;

/// Arc-shared interior of an [`obj_db_t`].
///
/// Bundles the [`obj_engine::Db`] with a per-handle `last_error`
/// diagnostic slot. It lives behind an [`Arc`] so the txn / iter
/// FFI modules can keep it alive past the original [`obj_open`]
/// return point — and, crucially, reach the SAME `last_error` slot
/// the owning [`obj_db_t`] exposes through [`obj_errmsg`]. A
/// transaction that fails records its specific reason here, and the
/// C caller retrieves it via `obj_errmsg(db)` on the originating db
/// handle.
///
/// [`Deref`](core::ops::Deref) to [`obj_engine::Db`] keeps every
/// existing `db_arc.<engine method>()` call site working unchanged.
pub(crate) struct DbInner {
    /// The wrapped storage engine handle.
    db: obj_engine::Db,
    /// Pointer to the most recently recorded error message — exactly
    /// what [`obj_errmsg`] returns. `null` means "no error recorded
    /// yet". This is a NON-owning view: the heap [`CString`] it
    /// points at is owned by `error_ring`, so a concurrent
    /// [`Self::set_error`] never frees the string a just-returned
    /// `obj_errmsg` pointer aliases (it lives until displaced from
    /// the ring, [`ERROR_RING_LEN`] writes later).
    last_error: AtomicPtr<c_char>,
    /// Ring buffer owning the last [`ERROR_RING_LEN`] error
    /// [`CString`]s (each as [`CString::into_raw`]). [`Self::set_error`]
    /// writes the new string into the next slot and frees ONLY the
    /// string that slot displaced — which is [`ERROR_RING_LEN`]
    /// errors old, so any reader that observed it has had a full
    /// grace window. The sole owner of these heap strings; freed once
    /// in [`Drop`].
    error_ring: [AtomicPtr<c_char>; ERROR_RING_LEN],
    /// Monotonic slot cursor; `% ERROR_RING_LEN` selects the
    /// `error_ring` slot the next [`Self::set_error`] claims.
    error_seq: AtomicUsize,
}

impl DbInner {
    /// Record `msg` as this handle's most recent error. Builds a
    /// [`CString`] from `msg` (falling back to a static
    /// `"(non-UTF-8 error)"` text if `msg` contains an interior NUL),
    /// parks it in the next `error_ring` slot, and publishes it as
    /// `last_error`. The only pointer this frees is the one the
    /// claimed slot displaces — [`ERROR_RING_LEN`] errors old — so a
    /// pointer a concurrent [`obj_errmsg`] just returned is never
    /// freed under the reader (it survives until [`ERROR_RING_LEN`]
    /// further errors recycle its slot). Memory stays bounded at
    /// [`ERROR_RING_LEN`] strings however many errors are recorded.
    pub(crate) fn set_error(&self, msg: &str) {
        let cstring = match CString::new(msg) {
            Ok(c) => c,
            Err(_) => match CString::new("(non-UTF-8 error)") {
                Ok(c) => c,
                // The literal contains no interior NUL, so this arm is
                // unreachable; bail without recording rather than panic.
                Err(_) => return,
            },
        };
        let raw = cstring.into_raw();
        // Claim a ring slot; the string it displaces is ERROR_RING_LEN
        // errors old, so any reader that observed it has had a full
        // grace window and we can reclaim it now.
        let slot = self.error_seq.fetch_add(1, Ordering::AcqRel) % ERROR_RING_LEN;
        let displaced = self.error_ring[slot].swap(raw, Ordering::AcqRel);
        // Publish as the latest message any obj_errmsg reader sees;
        // `raw` lives in the ring, so this view never dangles early.
        self.last_error.store(raw, Ordering::Release);
        if !displaced.is_null() {
            // SAFETY: a non-null `displaced` was produced by a prior CString::into_raw into this ring slot (set_error is the only writer of error_ring) and has not been freed, so reclaiming it via from_raw exactly once is sound.
            drop(unsafe { CString::from_raw(displaced) });
        }
    }

    /// Load the stored error pointer (`null` if none recorded).
    fn last_error_ptr(&self) -> *const c_char {
        self.last_error.load(Ordering::Acquire).cast_const()
    }
}

impl core::ops::Deref for DbInner {
    type Target = obj_engine::Db;

    fn deref(&self) -> &Self::Target {
        &self.db
    }
}

impl Drop for DbInner {
    fn drop(&mut self) {
        // Free every retained ring string. `last_error` is a
        // non-owning view into `error_ring`, so it is NOT freed here —
        // doing so would double-free the slot that still owns it. The
        // loop is bounded by the fixed ERROR_RING_LEN (R2).
        for slot in &mut self.error_ring {
            let ptr = *slot.get_mut();
            if !ptr.is_null() {
                // SAFETY: a non-null pointer here was produced by CString::into_raw in set_error and not yet freed (Drop runs once, after the final Arc strong ref is released, and each ring slot is reclaimed exactly once by this loop), so reclaiming it frees the backing CString.
                drop(unsafe { CString::from_raw(ptr) });
            }
        }
    }
}

/// Opaque database handle.
///
/// Created by [`obj_open`] / [`obj_open_with_config`]; freed by
/// [`obj_close`]. The Rust side holds an [`Arc<DbInner>`] (the
/// engine handle plus a `last_error` diagnostic slot) so future
/// transaction handles can each hold an `Arc` to outlive the
/// original [`obj_open`] return point AND reach the shared
/// `last_error` slot for [`obj_errmsg`].
///
/// The struct is **opaque to C**: cbindgen emits a forward
/// declaration only (`typedef struct obj_db_t obj_db_t;`). The
/// Rust-side layout is NOT `#[repr(C)]` because the Arc field is
/// not C-compatible — C callers only ever see pointer-shaped
/// access via the API below.
pub struct obj_db_t {
    /// The Arc-wrapped [`DbInner`]. Boxed-on-the-heap inside
    /// `obj_open*`; the wrapping `Box<obj_db_t>` is freed by
    /// `obj_close` via [`Box::from_raw`], which drops this field
    /// (decrementing the Arc; the last strong reference releases the
    /// Db's file locks and frees any stored `last_error`). Read via
    /// [`Self::db_arc`] / [`Self::inner_ref`] by the sibling FFI
    /// modules (txn / queries).
    inner: Arc<DbInner>,
}

impl obj_db_t {
    /// Wrap an [`obj_engine::Db`] into a heap-allocated handle the C side
    /// holds as `obj_db_t *`. The wrapped value is stored behind
    /// an [`Arc<DbInner>`] so transaction-handle modules can
    /// `Arc::clone` to outlive the original `obj_open` return point
    /// on the C side. The `last_error` slot starts empty.
    pub(crate) fn new(db: obj_engine::Db) -> Self {
        Self {
            inner: Arc::new(DbInner {
                db,
                last_error: AtomicPtr::new(ptr::null_mut()),
                error_ring: core::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
                error_seq: AtomicUsize::new(0),
            }),
        }
    }

    /// Cloneable [`Arc`] handle on the wrapped [`DbInner`]. Used by
    /// the FFI txn / iter modules to keep the Db (and its shared
    /// `last_error` slot) alive past the original `obj_open` return
    /// point.
    pub(crate) fn db_arc(&self) -> Arc<DbInner> {
        Arc::clone(&self.inner)
    }

    /// Borrow the shared [`DbInner`] — used to record an error
    /// against this handle from the db-direct entry points.
    pub(crate) fn inner_ref(&self) -> &DbInner {
        &self.inner
    }
}

/// Record `msg` as `db`'s most recent error. Crate-internal helper
/// matching the diagnostic contract documented on [`obj_errmsg`]:
/// the db-handle entry point that the db-direct FFI calls use to
/// stash a specific reason against the originating handle.
pub(crate) fn set_db_error(db: &obj_db_t, msg: &str) {
    db.inner.set_error(msg);
}

/// Record `e`'s specific message against the db-handle `db` and
/// return its [`obj_error_t`] code. The one-call shape the db-direct
/// `Err(e)` paths (stat / backup / integrity) use so the C caller can
/// retrieve the reason via [`obj_errmsg`].
pub(crate) fn db_handle_error_code(db: &obj_db_t, e: &obj_engine::Error) -> obj_error_t {
    set_db_error(db, &e.to_string());
    error_to_code(e)
}

/// Record `e`'s specific message against the shared [`DbInner`]
/// reached through a txn / iter Arc keepalive, and return its
/// [`obj_error_t`] code. The keepalive-side analog of
/// [`db_handle_error_code`] used by the txn-scoped `Err(e)` paths.
pub(crate) fn db_error_code(inner: &DbInner, e: &obj_engine::Error) -> obj_error_t {
    inner.set_error(&e.to_string());
    error_to_code(e)
}

/// Static `"(no error)"` text returned by [`obj_errmsg`] when no
/// error has been recorded on the handle.
const NO_ERROR_MSG: &[u8] = b"(no error)\0";

/// Static `"(null db handle)"` text returned by [`obj_errmsg`] when
/// `db` is NULL.
const NULL_DB_MSG: &[u8] = b"(null db handle)\0";

/// Return the most recent error message recorded against `db`, as a
/// NUL-terminated C string. Mirrors `sqlite3_errmsg`: where
/// [`obj_strerror`](crate::obj_strerror) maps a code to a generic
/// label, this returns the SPECIFIC reason the last failing call on
/// `db` produced (e.g. `"index 'email_idx' not found"`).
///
/// Never returns NULL:
/// - a NULL `db` yields a static `"(null db handle)"`;
/// - a handle with no recorded error yields a static `"(no error)"`.
///
/// The returned pointer is owned by `db`; the caller MUST NOT free
/// it. `db` retains its most recent error strings in a small
/// fixed-size ring (the last 16), so a returned pointer stays valid
/// until 16 *further* error-recording calls on `db` — or on any
/// txn / iter derived from it, including from other threads — have
/// recycled its ring slot, or until [`obj_close`] frees the handle.
/// In other words there is a bounded grace window, not single-call
/// invalidation: a pointer you just obtained is NOT freed out from
/// under you by one concurrent failing call. Memory stays bounded —
/// only the most recent 16 strings are kept, however many errors the
/// handle records over its lifetime.
///
/// Still, treat the result like `sqlite3_errmsg`: copy the string
/// promptly (before driving many more failing calls on the handle)
/// rather than holding the raw pointer indefinitely.
///
/// # Safety
///
/// If non-null, `db` must be a live handle returned by
/// [`obj_open`] / [`obj_open_with_config`] and not yet closed.
#[no_mangle]
pub unsafe extern "C" fn obj_errmsg(db: *mut obj_db_t) -> *const c_char {
    catch_ffi_or(NO_ERROR_MSG.as_ptr().cast::<c_char>(), || {
        if db.is_null() {
            return NULL_DB_MSG.as_ptr().cast::<c_char>();
        }
        // SAFETY: db is non-null (checked above) and a live obj_db_t handle per the # Safety contract.
        let stored = unsafe { (*db).inner_ref().last_error_ptr() };
        if stored.is_null() {
            NO_ERROR_MSG.as_ptr().cast::<c_char>()
        } else {
            stored
        }
    })
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
