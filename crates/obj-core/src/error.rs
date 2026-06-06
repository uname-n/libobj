//! Crate-level error type.
//!
//! Every fallible operation in `obj-core` returns
//! [`Result<T, Error>`](Result). `Error` is intentionally small. The
//! variants are non-exhaustive so additions do not count as breaking
//! changes for in-tree callers.
//!
//! Every `Result` and `Option` is handled explicitly. There are no
//! `.unwrap()` or `.expect()` calls in this crate's production code
//! paths.

#![forbid(unsafe_code)]

use std::io;
use thiserror::Error;

use crate::index::IndexKind;

/// The pager-level error type.
///
/// Construct variants directly when synthesising an error; downstream
/// callers should match exhaustively or use the `#[non_exhaustive]`
/// catch-all so that future variants are not source-breaking additions.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error from the platform layer (file open, read, write,
    /// flush). Wraps the underlying [`std::io::Error`] for inspection
    /// and never silently discards it.
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),

    /// The on-disk image is malformed or its checksum does not match.
    /// `page_id` is the page where corruption was detected; the value
    /// `0` refers to the file header. The error carries no `source` —
    /// corruption is the *direct* failure mode.
    #[error("corruption detected on page {page_id}")]
    Corruption {
        /// Page index where the corruption was detected. `0` = header.
        page_id: u64,
    },

    /// A WAL frame whose CRC32C does not validate sits **before** the
    /// last commit marker in the current generation. Unlike a torn
    /// tail (silently discarded), mid-WAL corruption is a recovery
    /// error: replaying past it would drop or alias durable data, and
    /// recovery refuses to guess. `frame_offset` is the byte offset
    /// of the bad frame in the WAL file.
    #[error("WAL corruption detected at frame offset {frame_offset}")]
    WalCorruption {
        /// Byte offset of the corrupted frame inside the WAL sidecar.
        frame_offset: u64,
    },

    /// The file's magic, page-size, or major version is not what this
    /// build of the library understands. Distinct from `Corruption`
    /// because the file may be perfectly valid for a different reader.
    #[error("not an obj database (or unsupported format): {reason}")]
    InvalidFormat {
        /// Human-readable explanation; intended for logging, not for
        /// programmatic dispatch.
        reason: &'static str,
    },

    /// Caller passed an out-of-range `PageId`, capacity, or similar
    /// numeric input that the type system could not statically
    /// rule out. Always indicates a caller bug.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),

    /// A B+tree traversal exceeded its statically-bounded depth limit
    /// (`MAX_BTREE_DEPTH = 32`). Every recursive shape is bounded; this
    /// is the surfaced error when the bound trips, not a panic.
    #[error("B+tree depth exceeded the {limit}-level bound")]
    BTreeDepthExceeded {
        /// The bound that was exceeded.
        limit: usize,
    },

    /// A B+tree insert was given a key longer than the format spec
    /// permits (`PAGE_SIZE / 4`).
    #[error("B+tree key length {key_len} exceeds max {max}")]
    BTreeKeyTooLarge {
        /// The offending key's length in bytes.
        key_len: usize,
        /// The maximum permitted key length in bytes.
        max: usize,
    },

    /// A B+tree insert was given a value too large to fit inline in a
    /// leaf alongside at least one slot. Overflow chains are deferred
    /// to a future minor format version.
    #[error("B+tree value length {value_len} exceeds inline max {max}")]
    BTreeValueTooLarge {
        /// The offending value's length in bytes.
        value_len: usize,
        /// The maximum value length that still fits inline.
        max: usize,
    },

    /// A B+tree insert was given a key that already exists in the
    /// tree. Trees do not allow duplicates.
    #[error("B+tree key already exists")]
    BTreeKeyExists,

    /// A B+tree range scan exceeded the per-scan node budget
    /// (`MAX_RANGE_NODES = 1_000_000`).
    #[error("B+tree range scan exceeded the {limit}-node budget")]
    BTreeScanLimitExceeded {
        /// The bound that was exceeded.
        limit: usize,
    },

    /// A B+tree invariant that `debug_assert!` would normally catch
    /// has tripped in a release build. Surfaced as an `Error` rather
    /// than a panic.
    #[error("B+tree invariant violated: {reason}")]
    BTreeInvariantViolated {
        /// Human-readable description of the violated invariant.
        reason: &'static str,
    },

    /// A document encode call produced a record larger than the
    /// B+tree leaf can hold inline. Overflow chains for oversize
    /// documents are deferred to a later format-minor. Documents must
    /// fit inline.
    #[error("document record length {len} exceeds inline max {max}")]
    DocumentTooLarge {
        /// Total record length (per-doc header + postcard payload).
        len: usize,
        /// Maximum length that still fits inline in a leaf.
        max: usize,
    },

    /// A decode call observed a per-document header whose
    /// `collection_id` does not match the catalog row for the
    /// `Document` type being decoded. Indicates a programming bug
    /// (wrong type at a key) or a forensic mishap (cross-collection
    /// byte forgery) — never a transient I/O issue.
    #[error("collection id mismatch: header says {found}, expected {expected}")]
    CollectionIdMismatch {
        /// The collection id the caller said this record should belong to.
        expected: u32,
        /// The collection id the on-disk record actually carries.
        found: u32,
    },

    /// The on-disk record's `type_version` is older than the
    /// `Document::VERSION` of the Rust type, and that type's
    /// `Migrate` impl is the default-erroring body. Real
    /// `Document` types override `Migrate::migrate` to handle this.
    #[error(
        "schema migration not implemented for collection '{collection}': from v{from_version} to v{to_version}"
    )]
    SchemaMigrationNotImplemented {
        /// The collection whose record could not be migrated.
        collection: &'static str,
        /// The stored `type_version`.
        from_version: u32,
        /// The reader's `Document::VERSION`.
        to_version: u32,
    },

    /// The on-disk record's `type_version` is **newer** than the
    /// reader's `Document::VERSION`. The reader refuses to guess
    /// what the unknown fields mean — better a hard error than a
    /// silent loss of data.
    #[error(
        "schema version from future for collection '{collection}': stored v{from}, reader v{to}"
    )]
    SchemaVersionFromFuture {
        /// The collection whose record was rejected.
        collection: &'static str,
        /// The stored `type_version` (larger than reader's).
        from: u32,
        /// The reader's `Document::VERSION`.
        to: u32,
    },

    /// A disk-backed schema-catalog row (a
    /// [`StoredSchema`](crate::codec::stored_schema::StoredSchema))
    /// carried a wire-format discriminator (`format`) this build does
    /// not understand. `format` is the first decoded field (offset 0),
    /// so the decoder rejects the row before trusting the rest of the
    /// bytes — better a hard error than a misinterpreted shape. This
    /// build understands `format = 1`. Distinct from
    /// [`Error::SchemaVersionFromFuture`] (a *document* newer than the
    /// reader) — this is the *catalog row encoding* itself being from a
    /// newer obj.
    #[error("unsupported stored-schema format: {format}")]
    UnsupportedSchemaFormat {
        /// The on-disk `format` discriminator the reader does not
        /// understand.
        format: u8,
    },

    /// `Catalog::put_schema` observed a normalized schema
    /// **shape** for `(collection_id, version)` that differs from the
    /// shape already persisted under the same key — i.e. the type's
    /// wire layout changed without a corresponding `VERSION` bump. Two
    /// incompatible layouts sharing one `type_version` would silently
    /// corrupt reads (postcard is positional), so the drift guard
    /// refuses the write and rolls back the whole transaction before any
    /// corrupt bytes persist.
    ///
    /// Note: only the normalized *shape* is compared. A benign
    /// signedness difference (e.g. a `u64` field expressed elsewhere as
    /// a signed `i64`) does NOT trip this guard — see
    /// [`StoredSchema`](crate::codec::stored_schema::StoredSchema).
    #[error(
        "schema shape for collection {collection_id} v{version} changed without a version bump; \
         bump VERSION and add a migrate for the old version"
    )]
    SchemaShapeChanged {
        /// The collection id whose stored schema shape changed.
        collection_id: u32,
        /// The `type_version` under which the conflicting shapes share
        /// a key.
        version: u32,
    },

    /// `Catalog::insert` was called for a name already present in the
    /// catalog. The user should call `Catalog::update` instead, or
    /// pick a different name.
    #[error("collection '{name}' already exists in the catalog")]
    CollectionAlreadyExists {
        /// The collection name that was already registered.
        name: String,
    },

    /// Per-collection `Id` allocator exhausted its `u64` space, or
    /// the catalog's per-collection-id `u32` space. At 10⁹/sec the
    /// `u64` case takes ~584 years; this is a defense-in-depth
    /// check, not a likely runtime event.
    ///
    /// The `collection` field is an owned `String` so user-supplied
    /// `&str` names can be reported verbatim without leaking a
    /// `'static` slot. The `#[non_exhaustive]` attribute on `Error`
    /// keeps this widening backward-compatible for downstream
    /// pattern-matchers.
    #[error("id space exhausted for collection '{collection}'")]
    IdSpaceExhausted {
        /// The collection (or `"<catalog>"`) whose id allocator
        /// was exhausted.
        collection: String,
    },

    /// A postcard encode or decode operation failed at the codec
    /// boundary. The wrapped error carries postcard's diagnostic;
    /// obj does not further interpret it because postcard is treated
    /// as a black-box codec.
    #[error("codec (postcard) error: {0}")]
    Codec(#[from] postcard::Error),

    /// A cross-process or in-process lock was contended for the
    /// caller-supplied timeout (`Config::busy_timeout` or the
    /// explicit deadline passed to `FileHandle::lock_writer` /
    /// `FileHandle::lock_reader`).  Every wait has an explicit budget;
    /// exhausting it surfaces here rather than blocking the caller
    /// forever.
    #[error("lock busy ({kind:?})")]
    Busy {
        /// Which lock category was contended.
        kind: LockKind,
    },

    /// The `Db` was opened with `Db::open_readonly` and the caller
    /// invoked an operation that would mutate the database.
    /// Surfaced eagerly so callers never have to inspect the
    /// underlying lock state to know the call was illegal.
    #[error("operation '{operation}' is not allowed on a read-only database")]
    ReadOnly {
        /// Short label naming the call that was rejected (e.g.
        /// `"insert"`, `"transaction"`).
        operation: &'static str,
    },

    /// A typed-API caller asked for a collection that the catalog
    /// does not have a row for. Distinct from
    /// [`Error::CollectionAlreadyExists`] (the insert-side dual);
    /// distinct from [`Error::Corruption`] (which would indicate a
    /// catalog row is malformed, not absent).
    #[error("collection '{name}' is not registered")]
    CollectionNotFound {
        /// The collection name the caller asked for.
        name: String,
    },

    /// A `Collection::update` or `Collection::delete` was given an
    /// id that does not exist in the collection's primary B-tree.
    #[error("document {id} not found in collection '{collection}'")]
    DocumentNotFound {
        /// The collection name.
        collection: &'static str,
        /// The id that was not found.
        id: u64,
    },

    /// An index extraction call was unable to resolve the
    /// configured field path against the document's `Dynamic` view.
    /// The document does not carry a value at the path the index
    /// names — typically a schema-evolution gap: the type used to
    /// have the field, an older document did not, and reconciliation
    /// has not rewritten it yet.
    #[error("index '{index}' on collection '{collection}': field path '{path}' is absent")]
    IndexFieldMissing {
        /// The collection the index belongs to.
        collection: String,
        /// The index's name.
        index: String,
        /// The dotted path the index is keyed on.
        path: String,
    },

    /// Lookup of a named index against a collection's descriptor
    /// failed: the catalog has no `IndexDescriptor` with that name,
    /// or the descriptor exists but is in `DroppedPending` status.
    #[error("index '{name}' not found on collection '{collection}'")]
    IndexNotFound {
        /// The collection the lookup was scoped to.
        collection: String,
        /// The index name the caller asked for.
        name: String,
    },

    /// `Collection::find_unique` was called on an index that is not
    /// `Unique` — `find_unique` is only defined for unique indexes.
    /// Callers that want non-unique lookups should use
    /// `Collection::lookup`.
    #[error("index '{name}' on collection '{collection}' is not a Unique index")]
    IndexNotUnique {
        /// The collection the lookup was scoped to.
        collection: String,
        /// The index name the caller asked for.
        name: String,
    },

    /// The reconciler observed a runtime [`crate::index::IndexSpec`] whose
    /// `kind` disagrees with the on-disk descriptor of the same
    /// name. To change an index's kind the application must
    /// drop-then-redeclare under a different name (or rebuild from
    /// scratch); silent in-place rewrites would invalidate any
    /// extant entries.
    #[error(
        "index '{name}' kind mismatch: existing descriptor is {found:?}, runtime spec is {expected:?}"
    )]
    IndexKindMismatch {
        /// The index name.
        name: String,
        /// The runtime spec's kind.
        expected: IndexKind,
        /// The on-disk descriptor's kind.
        found: IndexKind,
    },

    /// The reconciler observed a runtime [`crate::index::IndexSpec`] whose
    /// `key_paths` disagree with the on-disk descriptor of the
    /// same name and kind. Like [`Error::IndexKindMismatch`], the
    /// only safe response is for the application to drop-and-
    /// redeclare under a different name.
    #[error("index '{name}' key_paths mismatch with stored descriptor")]
    IndexKeyPathsMismatch {
        /// The index name.
        name: String,
    },

    /// A `Unique` index extraction observed a key value that
    /// already exists on a different document in the same
    /// collection. The maintenance path surfaces this rather than
    /// silently overwriting; the WAL transaction rolls back so no
    /// partial state remains.
    #[error(
        "unique constraint violation on index '{index}' \
         in collection '{collection}'"
    )]
    UniqueConstraintViolation {
        /// The collection the index belongs to.
        collection: String,
        /// The index name.
        index: String,
        /// The encoded key bytes that collided.
        key: Vec<u8>,
    },

    /// A single document's `Each` extraction emitted more than the
    /// per-doc ceiling number of index entries — almost certainly
    /// the application supplied a runaway sequence rather than a
    /// real indexable field.
    #[error(
        "each-index '{index}' on collection '{collection}': sequence \
         of {len} entries exceeds per-doc cap {max}"
    )]
    EachIndexTooLarge {
        /// The collection the index belongs to.
        collection: String,
        /// The index name.
        index: String,
        /// The observed sequence length.
        len: usize,
        /// The per-doc cap.
        max: usize,
    },

    /// An index extraction call resolved a field path but the
    /// value's `Dynamic` shape disagrees with the index kind's
    /// contract — e.g. an `Each` index whose target field is not a
    /// sequence, or a `Composite` field that resolved to a `Map`
    /// (only primitive `Dynamic` values are indexable).
    #[error(
        "index '{index}' on collection '{collection}': field '{path}' \
         has type '{found}', expected '{expected}'"
    )]
    IndexFieldTypeMismatch {
        /// The collection the index belongs to.
        collection: String,
        /// The index's name.
        index: String,
        /// The dotted path the index is keyed on.
        path: String,
        /// The expected `Dynamic` shape (e.g. `"Seq"` for `Each`).
        expected: &'static str,
        /// The shape the document actually carries at the path.
        found: &'static str,
    },

    /// A query's `sort_by` extension collected more than its
    /// `sort_buffer_limit` (default 100 000) candidate documents
    /// before the sort+truncate step could narrow them down.
    /// The in-memory sort is bounded; the user should add a `.filter`
    /// / `.index_range` / `.limit` that bounds the survivors, or raise
    /// the limit explicitly via `.sort_buffer_limit(N)` if the
    /// workload genuinely needs it.
    ///
    /// These sorts are designed for "screen-of-results" workloads, not
    /// "sort a million rows" workloads; a disk-spill sort is a
    /// follow-up if this turns out to be too restrictive.
    #[error("query sort buffer exceeded the {limit}-document budget")]
    SortBufferExceeded {
        /// The bound that was exceeded.
        limit: usize,
    },

    /// A query's `sort_by` extractor produced a [`Dynamic`] whose
    /// `obj_core::index::encode_field` representation could not be
    /// computed (e.g. a `Dynamic::String` carrying an embedded NUL
    /// byte, which the order-preserving encoder rejects). The
    /// underlying error is propagated rather than silently collapsed
    /// into an empty key. Callers who want to
    /// control the encoding themselves should use
    /// `Query::sort_by_bytes`, which never touches `encode_field`.
    ///
    /// [`Dynamic`]: crate::codec::Dynamic
    #[error("sort_by key encoding failed: {source}")]
    SortKeyEncode {
        /// The underlying encoder failure. Boxed to keep `Error`
        /// (which contains this variant) `Sized`.
        #[source]
        source: Box<Error>,
    },

    /// `Collection::count_distinct_ids_in_range`
    /// observed more than the caller-configurable per-call cap of
    /// distinct `Id`s while walking the index B-tree. The in-memory
    /// `HashSet<Id>` is bounded; the user
    /// should narrow the range via `.index_range(...)` so fewer
    /// distinct docs fall inside the window. The fast path on the
    /// query layer dispatches to this routine ONLY for `Each`-kind
    /// indexes — other kinds count entries via the cheaper
    /// `count_index_range` path that has no distinct-tracking cost.
    #[error("distinct-id count exceeded the {limit}-id budget")]
    DistinctCountExceeded {
        /// The bound that was exceeded.
        limit: usize,
    },

    /// `codec::decode` saw an on-disk record whose `type_version` is
    /// older than the reader's `Document::VERSION`, but the reader's
    /// `Document::historical_schemas()` has no entry for that version.
    /// The codec refuses to invent a `Dynamic` view —
    /// silent fallback hides schema-evolution bugs.
    #[error("schema not registered for collection '{collection}': stored type_version v{version}")]
    SchemaNotRegistered {
        /// The collection whose record could not be migrated.
        collection: &'static str,
        /// The stored `type_version` for which no schema was found.
        version: u32,
    },

    /// The postcard byte-stream walker driven by
    /// [`Dynamic::from_postcard_bytes`](crate::codec::Dynamic::from_postcard_bytes)
    /// exceeded the depth bound
    /// [`MAX_SCHEMA_DEPTH`](crate::codec::schema::MAX_SCHEMA_DEPTH).
    /// The walker uses an explicit stack, but the stack depth is
    /// itself bounded to avoid a pathological schema triggering
    /// unbounded growth.
    #[error("schema walker exceeded the {depth}-level depth bound")]
    SchemaDepthExceeded {
        /// The bound that was exceeded.
        depth: usize,
    },

    /// The postcard byte-stream walker observed a payload that
    /// disagrees with the supplied schema in a way the schema
    /// itself cannot recover from (e.g. a `Bool` slot carrying a
    /// byte other than `0` / `1`, a `String` slot whose bytes are
    /// not UTF-8, or a `Seq` length that overflows the input).
    #[error("schema mismatch at path '{path}': expected {expected}, found {found}")]
    SchemaTypeMismatch {
        /// The schema-side variant the walker was decoding when the
        /// mismatch surfaced.
        expected: &'static str,
        /// The shape the bytes actually carried (e.g. `"non-utf8"`,
        /// `"truncated"`, `"non-bool-byte"`).
        found: &'static str,
        /// Dotted path naming the schema slot where the mismatch
        /// surfaced — useful for forensic logging on a real
        /// migration failure.
        path: String,
    },

    /// `Dynamic::remove` was called on a `Dynamic`
    /// whose root value (or any intermediate path segment) is not a
    /// [`Map`](crate::codec::Dynamic::Map). Map-only by construction —
    /// callers needing scalar removal should replace the value
    /// outright via `Dynamic::set`.
    #[error("Dynamic::remove on a non-Map at path '{path}'")]
    DynamicPathNotMap {
        /// The dotted path that resolved to a non-Map value.
        path: String,
    },

    /// `Db::backup_to` was called with a destination path
    /// that already exists. The backup never overwrites existing
    /// files — the operator must remove the destination explicitly
    /// or pick a fresh path.
    #[error("backup destination already exists: {path}")]
    BackupDestinationExists {
        /// The destination path the call refused to write.
        path: std::path::PathBuf,
    },

    /// `Db::backup_to` was called on an in-memory database.
    /// Memory pagers have no file backend; serialising the in-memory
    /// state to a fresh `.obj` file is deferred to a future minor.
    #[error("backup_to is not supported for in-memory pagers")]
    BackupNotSupportedForMemoryPager,

    /// `Db::attach` was called with a namespace already
    /// registered on the calling `Db`. Detach the existing
    /// attachment first or pick a different namespace.
    #[error("attachment namespace '{namespace}' is already in use")]
    AttachmentAlreadyExists {
        /// The namespace the caller tried to register.
        namespace: String,
    },

    /// `Db::attach` was unable to open the attached file.
    /// The wrapped error carries the underlying cause (typically
    /// [`Error::Io`] or [`Error::Corruption`]).
    #[error("attachment at {path} could not be opened: {source}")]
    AttachmentNotReadable {
        /// The path the caller tried to attach.
        path: std::path::PathBuf,
        /// Underlying open failure.
        #[source]
        source: Box<Error>,
    },

    /// `Db::insert` / `Db::update` / `Db::delete` / `Db::upsert` (and
    /// the `WriteTxn::collection<T>()` equivalents) refused to
    /// mutate a collection whose name resolves to an attached
    /// database. Attached databases are read-only through the
    /// calling `Db`.
    #[error(
        "collection '{collection}' lives in attached database '{namespace}' \
         which is read-only"
    )]
    AttachedDatabaseIsReadOnly {
        /// Namespace prefix that resolved to an attached db.
        namespace: String,
        /// The unqualified collection name within the attached db.
        collection: String,
    },

    /// A namespaced collection name was opened against a calling
    /// `Db` that has no attachment registered under the namespace.
    #[error("collection namespace '{namespace}' is not attached")]
    CollectionNamespaceUnknown {
        /// The namespace prefix that did not resolve.
        namespace: String,
    },

    /// The file's `format_minor` (or one of its `feature_flags`
    /// bits) requires a build-time feature that this build was
    /// compiled without.  The reader refuses to open the file
    /// rather than risk a partial / nonsensical decode.
    ///
    /// Issued by [`crate::pager::Pager::open`] when the on-disk
    /// header signals a feature the running binary does not link
    /// against (`format_minor >= 1` without
    /// `cargo build --features obj-rs/compression`).
    #[error("on-disk format requires unsupported feature: {feature}")]
    FormatFeatureUnsupported {
        /// Static name of the missing feature (e.g. `"compression"`).
        feature: &'static str,
    },

    /// The on-disk file declares encryption
    /// (`format_minor = 2`, `feature_flags` bit 1 set) but the
    /// caller's [`crate::pager::Config`] (and the obj-rs
    /// `Config::encryption_key`) supplied no key. The pager
    /// refuses to open the file rather than fail later on the
    /// first ciphertext read.
    #[error("encryption key required: file is encrypted but no key was provided")]
    EncryptionKeyRequired,

    /// The caller-supplied encryption key
    /// does NOT decrypt the first ciphertext page (Poly1305
    /// verification failed). Surfaced on the first encrypted
    /// page read after open — the file header is plaintext, so
    /// a wrong key is only caught when the pager touches a real
    /// data page.
    #[error("encryption key is invalid: ciphertext failed authentication")]
    EncryptionKeyInvalid,

    /// The caller set
    /// [`Config::encryption_key`](crate::pager::Config) on a
    /// file whose on-disk header reports
    /// `format_minor < 2` (i.e. NOT an encrypted file). The
    /// caller's intent ("decrypt this file with this key") does
    /// not match the file's reality ("I am not encrypted") — a
    /// silent open would either succeed with the key ignored
    /// (surprising) or corrupt later writes if the pager
    /// believed the file should be encrypted. Refuse loudly.
    #[error("encryption key mismatch: file is not encrypted but a key was supplied")]
    EncryptionKeyMismatch,
}

/// Lock category for [`Error::Busy`]. Three variants because the
/// three categories of contention produce different operator
/// guidance: a contended cross-process `WRITER_LOCK` means another
/// process is writing; a contended `WriterInProcess` means another
/// thread of the same process is writing; a contended reader lock
/// is unusual (31 slots, shared) and indicates either a saturated
/// 31+-process workload or a stale lock left by a frozen process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LockKind {
    /// Cross-process `WRITER_LOCK` at byte 96 of the
    /// `<db>.obj-lock` sidecar file.
    Writer,
    /// Cross-process `READER_LOCK` byte (any slot in 97..128 of
    /// the `<db>.obj-lock` sidecar file).
    Reader,
    /// In-process write mutex (a sibling thread is mid-write).
    WriterInProcess,
}

/// Crate-local `Result` alias. Use this in new code unless an explicit
/// `std::result::Result` is required for trait-impl reasons.
pub type Result<T> = std::result::Result<T, Error>;
