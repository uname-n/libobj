//! Error-code enum + `obj_strerror`.
//!
//! Every fallible C entry point returns [`obj_error_t`] — a signed
//! `int32_t` because C99 has no `uint32_t`-typed enum constants in
//! the language proper (cbindgen emits a typedef + a set of
//! `#define`-style constants).
//!
//! The mapping from [`obj_engine::Error`] to one of these codes lives in
//! [`error_to_code`]. Every [`obj_engine::Error`] variant that exists
//! today is enumerated explicitly so each one maps to its most
//! accurate [`obj_error_t`] code. Because
//! [`obj_engine::Error`] is `#[non_exhaustive]` and lives in a
//! different crate, Rust requires a trailing wildcard arm even when
//! every current variant is listed; that wildcard is reachable ONLY
//! by a variant added to obj-core in the future. Adding such a
//! variant does NOT break this build — the wildcard catches it — so
//! a maintainer must consult the obj-core changelog when bumping the
//! engine dependency to give the new variant a deliberate mapping.
//! The wildcard collapses any not-yet-mapped variant into the most
//! defensive code ([`OBJ_ERR_CORRUPTION`]).

use core::ffi::c_char;

/// Numeric type carried by `obj_error_t`. Aliased so the cbindgen
/// header emits `typedef int32_t obj_error_t;` rather than a bare
/// `int32_t` everywhere.
pub type obj_error_t = i32;

/// Operation succeeded.
pub const OBJ_OK: obj_error_t = 0;

/// Caller passed an invalid argument: a null handle / out-pointer
/// where one is required, a non-UTF-8 path or collection name, a
/// length that exceeds the documented maximum, etc.
pub const OBJ_ERR_INVALID_ARG: obj_error_t = 1;

/// Underlying I/O / syscall failure: file open, read, write, fsync,
/// directory enumeration, lock acquisition syscall.
pub const OBJ_ERR_IO: obj_error_t = 2;

/// Content-level corruption: a page CRC mismatch, malformed
/// catalog row, schema-version-from-future, B-tree invariant
/// violated, etc.
pub const OBJ_ERR_CORRUPTION: obj_error_t = 3;

/// A lock could not be acquired inside the configured timeout —
/// another transaction holds the writer mutex / cross-process
/// writer byte, or the in-process pager mutex was poisoned.
pub const OBJ_ERR_BUSY: obj_error_t = 4;

/// The requested record / index entry / collection was not found.
/// Iterators also return this when exhausted (see
/// `obj_iter_next` docs).
pub const OBJ_ERR_NOT_FOUND: obj_error_t = 5;

/// An integrity check completed and found violations. Distinct
/// from [`OBJ_ERR_CORRUPTION`] which is the hard fail-closed shape
/// emitted by individual operations; [`OBJ_ERR_INTEGRITY`] is the
/// soft "the report has failures" shape returned by aggregate
/// diagnostic calls.
pub const OBJ_ERR_INTEGRITY: obj_error_t = 6;

/// Operation is not supported in this build / for this database
/// (e.g. `obj_backup_to` on an in-memory database, an attempt to
/// mutate an attached read-only attachment, an unimplemented
/// schema migration).
pub const OBJ_ERR_UNSUPPORTED: obj_error_t = 7;

/// A C string handed across the FFI boundary was not valid UTF-8.
/// Distinct from [`OBJ_ERR_INVALID_ARG`] because UTF-8 issues are
/// frequent enough on user paths / index names to merit their own
/// signal.
pub const OBJ_ERR_UTF8: obj_error_t = 8;

/// Highest reserved code in the obj-defined range. Future error
/// codes must stay `<= OBJ_ERR_RESERVED_MAX`; this allows
/// downstream callers to reliably distinguish obj-emitted codes
/// from any they may layer on top.
pub const OBJ_ERR_RESERVED_MAX: obj_error_t = 1023;

/// Static null-terminated message used by [`obj_strerror`]. The
/// strings carry an embedded `\0` terminator so they can be
/// returned as `*const c_char` directly.
struct ErrorEntry {
    /// The associated code.
    code: obj_error_t,
    /// NUL-terminated UTF-8 bytes.
    msg: &'static [u8],
}

const ERROR_TABLE: &[ErrorEntry] = &[
    ErrorEntry {
        code: OBJ_OK,
        msg: b"OK\0",
    },
    ErrorEntry {
        code: OBJ_ERR_INVALID_ARG,
        msg: b"invalid argument\0",
    },
    ErrorEntry {
        code: OBJ_ERR_IO,
        msg: b"i/o error\0",
    },
    ErrorEntry {
        code: OBJ_ERR_CORRUPTION,
        msg: b"corruption detected\0",
    },
    ErrorEntry {
        code: OBJ_ERR_BUSY,
        msg: b"lock busy\0",
    },
    ErrorEntry {
        code: OBJ_ERR_NOT_FOUND,
        msg: b"not found\0",
    },
    ErrorEntry {
        code: OBJ_ERR_INTEGRITY,
        msg: b"integrity check failed\0",
    },
    ErrorEntry {
        code: OBJ_ERR_UNSUPPORTED,
        msg: b"operation not supported\0",
    },
    ErrorEntry {
        code: OBJ_ERR_UTF8,
        msg: b"invalid UTF-8\0",
    },
];

/// Default fallback when [`obj_strerror`] is called with an
/// unknown code. The leading `unknown:` prefix is stable enough
/// for grep-driven diagnostics.
const UNKNOWN_ERR: &[u8] = b"unknown obj_error_t\0";

/// Map an [`obj_engine::Error`] variant onto an [`obj_error_t`] code.
///
/// Every variant that `obj_engine::Error` carries today is listed
/// explicitly so each maps to its most accurate C-ABI code rather
/// than collapsing silently into one bucket.
///
/// `obj_engine::Error` is `#[non_exhaustive]` and defined in a
/// different crate, so Rust *requires* a trailing `_` arm here even
/// though every present variant is enumerated above it. That arm is
/// unreachable for any current variant; it exists solely to catch a
/// variant added to obj-core in a future release. A new obj-core
/// variant therefore does NOT fail this build — it falls through to
/// the wildcard — so when bumping the engine dependency a maintainer
/// must check the obj-core changelog and slot any new variant into
/// the taxonomy above. The wildcard's [`OBJ_ERR_CORRUPTION`] is the
/// most defensive choice: a C caller treating unknown errors as
/// "stop and investigate" stays safe until the mapping is updated.
#[must_use]
// allow: several engine error variants intentionally map to the same OBJ_ERR_*
// code yet are kept as distinct arms so each variant is explicitly classified
// (see the taxonomy doc above); merging them would erase that mapping intent.
#[allow(clippy::match_same_arms)]
pub(crate) fn error_to_code(err: &obj_engine::Error) -> obj_error_t {
    use obj_engine::Error;
    match err {
        Error::Io(_) => OBJ_ERR_IO,
        Error::Corruption { .. }
        | Error::WalCorruption { .. }
        | Error::InvalidFormat { .. }
        | Error::BTreeDepthExceeded { .. }
        | Error::BTreeInvariantViolated { .. }
        | Error::CollectionIdMismatch { .. } => OBJ_ERR_CORRUPTION,
        Error::InvalidArgument(_)
        | Error::BTreeKeyTooLarge { .. }
        | Error::BTreeValueTooLarge { .. }
        | Error::BTreeKeyExists
        | Error::BTreeScanLimitExceeded { .. }
        | Error::DocumentTooLarge { .. }
        | Error::SortBufferExceeded { .. }
        | Error::SortKeyEncode { .. }
        | Error::DistinctCountExceeded { .. }
        | Error::IndexFieldMissing { .. }
        | Error::IndexFieldTypeMismatch { .. }
        | Error::IndexKindMismatch { .. }
        | Error::IndexKeyPathsMismatch { .. }
        | Error::UniqueConstraintViolation { .. }
        | Error::EachIndexTooLarge { .. }
        | Error::SchemaTypeMismatch { .. }
        | Error::SchemaDepthExceeded { .. }
        | Error::DynamicPathNotMap { .. }
        | Error::EncryptionKeyRequired
        | Error::EncryptionKeyMismatch => OBJ_ERR_INVALID_ARG,
        Error::Codec(_) | Error::EncryptionKeyInvalid => OBJ_ERR_CORRUPTION,
        Error::Busy { .. } => OBJ_ERR_BUSY,
        Error::CollectionNotFound { .. }
        | Error::DocumentNotFound { .. }
        | Error::IndexNotFound { .. }
        | Error::IndexNotUnique { .. }
        | Error::CollectionNamespaceUnknown { .. } => OBJ_ERR_NOT_FOUND,
        Error::CollectionAlreadyExists { .. }
        | Error::IdSpaceExhausted { .. }
        | Error::ReadOnly { .. }
        | Error::SchemaMigrationNotImplemented { .. }
        | Error::SchemaVersionFromFuture { .. }
        | Error::SchemaNotRegistered { .. }
        | Error::BackupNotSupportedForMemoryPager
        | Error::BackupNotSupportedForEncryptedPager
        | Error::FormatFeatureUnsupported { .. }
        | Error::AttachedDatabaseIsReadOnly { .. } => OBJ_ERR_UNSUPPORTED,
        Error::BackupDestinationExists { .. }
        | Error::AttachmentAlreadyExists { .. }
        | Error::AttachmentNotReadable { .. } => OBJ_ERR_IO,
        _ => OBJ_ERR_CORRUPTION,
    }
}

/// Error code returned when a Rust `panic!` is caught at the C
/// boundary by [`catch_ffi`]. A panic is an internal invariant
/// violation the C ABI has no dedicated code for, so it reuses the
/// most defensive existing code ([`OBJ_ERR_CORRUPTION`]) — the same
/// fail-closed shape the [`error_to_code`] wildcard uses. "Stop and
/// investigate" is the only safe interpretation of an engine panic.
pub(crate) const OBJ_ERR_PANIC: obj_error_t = OBJ_ERR_CORRUPTION;

/// Run `body` inside [`std::panic::catch_unwind`] and translate a
/// caught panic into an error code instead of letting it unwind
/// across the `extern "C"` boundary.
///
/// Since Rust 1.81 a panic that reaches an `extern "C"` frame
/// aborts the whole host process (defined behaviour, but a hard
/// abort with no error code). Wrapping each entry-point body here
/// converts that abort into a recoverable [`obj_error_t`] the C
/// caller can branch on: no failure
/// mode escapes as an uncontrolled abort.
///
/// `body` must perform its own out-pointer bookkeeping for its
/// success and `Err` paths exactly as before; on a *caught panic*
/// the body returns this `fallback` code and any out-pointers it
/// had not yet written stay as the caller left them. A panic is an
/// abnormal path, so this is an accepted degradation versus
/// aborting the entire process.
///
/// [`AssertUnwindSafe`](std::panic::AssertUnwindSafe) is required
/// because the closures capture raw FFI pointers (which are not
/// `UnwindSafe`); the boundary is sound because a caught panic
/// returns immediately without the C caller observing any
/// half-mutated Rust value — only its own out-pointers, which it
/// must treat as undefined on a non-`OBJ_OK` return per the C
/// contract.
pub(crate) fn catch_ffi<F>(fallback: obj_error_t, body: F) -> obj_error_t
where
    F: FnOnce() -> obj_error_t,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => fallback,
    }
}

/// As [`catch_ffi`] but for entry points that return no error code
/// (the null-tolerant `*_close` / `*_free` / `*_rollback` calls).
/// A panic inside a teardown closure is swallowed rather than
/// allowed to abort the host: there is no code to return, and the
/// resource is being released regardless.
pub(crate) fn catch_ffi_void<F>(body: F)
where
    F: FnOnce(),
{
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
}

/// As [`catch_ffi`] but for entry points whose return type is not
/// [`obj_error_t`] (the integrity-report accessors that return
/// `bool` / `u64` / `usize`). On a caught panic the supplied
/// `fallback` value is returned instead of unwinding across the C
/// boundary. The fallback mirrors each accessor's documented
/// null-handle answer (`false` / `0`), so a panic degrades to the
/// same "treat as absent/empty" shape the C contract already
/// defines for a NULL handle.
pub(crate) fn catch_ffi_or<T, F>(fallback: T, body: F) -> T
where
    F: FnOnce() -> T,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => fallback,
    }
}

/// Return a static C string describing `code`. Never returns NULL.
///
/// # Safety
///
/// The returned pointer points into the read-only data segment of
/// the loaded library; it has `'static` lifetime and MUST NOT be
/// freed. It is safe to read until the dynamic library is
/// unloaded.
#[no_mangle]
pub unsafe extern "C" fn obj_strerror(code: obj_error_t) -> *const c_char {
    // Wrapped in catch_ffi_or for consistency with every other extern "C"
    // entry point: a future edit adding a panicking operation here would
    // otherwise unwind across the boundary and abort the host. The fallback
    // is the same static UNKNOWN_ERR pointer the unknown-code path returns,
    // preserving the "never returns NULL" contract.
    catch_ffi_or(UNKNOWN_ERR.as_ptr().cast::<c_char>(), || {
        for entry in ERROR_TABLE {
            if entry.code == code {
                return entry.msg.as_ptr().cast::<c_char>();
            }
        }
        UNKNOWN_ERR.as_ptr().cast::<c_char>()
    })
}
