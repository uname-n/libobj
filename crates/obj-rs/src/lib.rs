//! `obj` — embedded document database (public crate).
//!
//! This crate is the user-facing surface of the `obj` storage
//! engine. It wraps the `obj-core` building blocks (pager, WAL,
//! B+tree, codec, catalog, transaction layer) into the typed
//! [`Db`] / [`Collection<T>`] API.
//!
//! Worked examples for every topic live next to the relevant item
//! in this crate's rustdoc:
//!
//! - Opening / CRUD: see [`Db::open`], [`Db::insert`], [`Db::get`],
//!   [`Db::update`], [`Db::delete`], [`Db::upsert`].
//! - Transactions: see [`Db::transaction`] and
//!   [`Db::read_transaction`].
//! - Iteration: see [`Db::iter_all`] and [`Db::all`].
//! - Queries: see [`Db::query`], [`Query::sort_by`],
//!   [`Query::index_range`], [`Query::count`].
//! - Attach / backup / integrity: see [`Db::attach`],
//!   [`Db::backup_to`], [`Db::integrity_check`].
//! - Configuration: see [`Config`].
//!
//! # Quick start
//!
//! ```no_run
//! use obj::Db;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize, obj::Document)]
//! struct Order { customer_id: u64, total_cents: u64 }
//!
//! fn run() -> obj::Result<()> {
//!     let db = Db::open("app.obj")?;
//!     let id = db.insert(Order { customer_id: 1, total_cents: 100 })?;
//!     let back: Option<Order> = db.get(id)?;
//!     assert!(back.is_some());
//!     Ok(())
//! }
//! ```
//!
//! # Core CRUD and the `Document` derive
//!
//! Open a database with one of three constructors:
//!
//! - [`Db::open`] / [`Db::open_with`] — file-backed; creates if
//!   absent, reopens otherwise.
//! - [`Db::memory`] / [`Db::memory_with`] — in-memory, ephemeral.
//!   No persistence, no file locks. Useful for unit tests.
//! - [`Db::open_readonly`] — read-only against an existing file.
//!   Every mutating call returns
//!   [`Err(Error::ReadOnly { .. })`](Error::ReadOnly).
//!
//! Each `Db` is `Send + Sync`. Share across threads via `Arc<Db>`
//! for the concurrent-reader / single-writer workload.
//!
//! Implement the [`Document`] trait on every type you want to
//! persist. The [`obj::Document`](crate::Document) re-export is a
//! `proc-macro` that fills in the trait's associated constants
//! from optional `#[obj(...)]` attributes:
//!
//! - `#[obj(collection = "...")]` — sets [`Document::COLLECTION`].
//!   Default: the type name.
//! - `#[obj(version = N)]` — sets [`Document::VERSION`]. Default: 1.
//! - `#[obj(index)]`, `#[obj(index = unique)]`,
//!   `#[obj(index = each)]` on a field — declare secondary indexes
//!   (see § "Queries and indexes" below).
//! - `#[obj(index_composite(fields = ("a", "b")))]` at struct
//!   level — declare a composite index.
//!
//! The one-shot API runs each call inside a private transaction
//! and is the typical entry point for ad-hoc work:
//!
//! - [`Db::insert`] — allocate an `Id`, write the doc.
//! - [`Db::get`] — fetch by `Id`. Returns `Option<T>`.
//! - [`Db::update`] — apply a closure in place. Errors with
//!   [`Error::DocumentNotFound`] if the id is absent.
//! - [`Db::delete`] — remove by `Id`. Returns `true` if it existed.
//! - [`Db::upsert`] — insert-or-replace at a caller-supplied `Id`.
//! - [`Db::find_unique`] — point lookup on a `Unique` index.
//!   `O(log n)`, no collection scan.
//!
//! # Transactions and iteration
//!
//! For multi-document atomicity, [`Db::transaction`] runs a closure
//! with a `&mut WriteTxn`. The closure returns `Result<R>`; commit
//! on `Ok`, rollback on `Err`, rollback-via-`Drop` on panic. Inside
//! the closure, [`WriteTxn::collection`] yields a typed
//! [`Collection<T>`] handle whose methods compose with the parent
//! txn — every write rides one WAL transaction.
//!
//! For read-only consistency across multiple reads,
//! [`Db::read_transaction`] runs a closure with a `&ReadTxn`. The
//! closure observes one consistent snapshot of the database;
//! concurrent writers do not affect what it sees.
//!
//! For full-collection iteration there are two shapes:
//!
//! - [`Db::iter_all`] — streaming iterator over `Result<(Id, T)>`.
//!   Peak memory is bounded at a small constant (256 entries per
//!   refill) regardless of collection size.
//! - [`Db::all`] — one-line shim that drives `iter_all` to
//!   exhaustion and collects into `Vec<T>`. Pays memory
//!   proportional to the collection.
//!
//! See [`Db::transaction`] / [`Db::read_transaction`] for worked
//! examples of the closure shape.
//!
//! # Queries and indexes
//!
//! [`Db::query::<T>()`](Db::query) constructs a [`Query`] builder.
//! Compose with [`Query::filter`], [`Query::limit`],
//! [`Query::sort_by`], [`Query::index_range`]; terminate with
//! [`Query::fetch`] (materialised `Vec<T>`) or [`Query::count`]
//! (count alone, without decoding documents on the fast path).
//!
//! The query layer has two sources: a full primary-tree scan
//! (default) or an index-range slice ([`Query::index_range`]). No
//! cost-based planner — the caller picks. Source order is by
//! primary `Id` for the full scan and by encoded index-key bytes
//! for the index range.
//!
//! [`Query::sort_by`] materialises every surviving candidate into
//! a sort buffer before applying [`Query::limit`]. The buffer is
//! capped at [`MAX_SORT_BUFFER`] (100 000 documents); overflowing
//! the cap surfaces [`Error::SortBufferExceeded`]. Override the
//! cap with [`Query::sort_buffer_limit`] when the workload
//! genuinely needs more.
//!
//! Indexes are declared on the document type via
//! [`Document::indexes`] (or the derive's `#[obj(index ...)]`
//! attributes). The catalog reconciler runs on the first
//! [`WriteTxn::collection::<T>()`](WriteTxn::collection) call per
//! process per collection: it declares missing specs, marks
//! stale active descriptors `DroppedPending`, and is idempotent.
//! Reconciliation rides the caller's WAL transaction — a rolled-
//! back insert leaves no half-created index behind.
//!
//! Four [`IndexKind`]s are exposed: `Standard`, `Unique`, `Each`,
//! `Composite`. Construct typed [`IndexSpec`]s via
//! `IndexSpec::standard` / `::unique` / `::each` / `::composite`
//! when hand-implementing [`Document::indexes`].
//!
//! # Schema evolution
//!
//! Bump [`Document::VERSION`] on every breaking change. Register a
//! [`DynamicSchema`] for each prior version in
//! [`Document::historical_schemas`], and provide a
//! [`Document::migrate`] body that lifts the structured
//! [`obj_core::codec::Dynamic`] view into the current `Self`.
//!
//! Migration is lazy: a stored record whose `type_version` is
//! older than `Self::VERSION` is migrated on read but the on-disk
//! bytes are NOT rewritten until the next
//! [`Collection::update`] / [`Collection::upsert`] for that id.
//! The collection therefore scales to billions of documents
//! without a stop-the-world rebuild on schema bumps.
//!
//! - [`Error::SchemaNotRegistered`] surfaces when a stored
//!   `type_version` has no entry in `historical_schemas()`.
//! - [`Error::SchemaVersionFromFuture`] surfaces when the stored
//!   `type_version` is newer than `Self::VERSION` (downgrade
//!   attempt).
//!
//! Worked recipes for the four common patterns — single-version
//! migration, multi-version chains, tombstoned fields, enum-variant
//! migration — live on [`Document::migrate`] and in the
//! integration tests `historical_schemas.rs`, `tombstone_migration.rs`,
//! `enum_migration.rs`, and `lazy_migration.rs`. The lazy-rewrite
//! cycle itself is documented on [`Collection::get`].
//!
//! # Attach, backup, integrity
//!
//! [`Db::attach`] registers a read-only second `.obj` file under a
//! caller-chosen namespace. Any [`Document`] whose `COLLECTION` is
//! of the form `<namespace>.<name>` dispatches reads against the
//! attached file; writes against a namespaced collection return
//! [`Error::AttachedDatabaseIsReadOnly`]. Each attached database
//! gets its own snapshot pinned at read-transaction begin;
//! [`Db::detach`] removes the registry entry but in-flight reads
//! complete against their pinned snapshot.
//!
//! [`Db::backup_to`] writes a self-contained `.obj` file at the
//! LSN of an internally-taken reader snapshot. Writers continue
//! against the source; post-snapshot writes are NOT in the
//! destination. Two failure modes:
//! [`Error::BackupDestinationExists`] (refuses to overwrite) and
//! [`Error::BackupNotSupportedForMemoryPager`] (in-memory dbs have
//! no file backend to copy from).
//!
//! [`Db::integrity_check`] runs a full bidirectional walk: every
//! active collection's primary + index B-trees, freelist sweep,
//! orphan-page detection, primary↔index cross-reference. Returns
//! [`IntegrityReport`] with a `failures` list and a
//! `pages_checked` count. The lightweight subset that
//! [`Db::open`] runs at open time is
//! `obj_core::integrity::quick_check`; opt out of the open-time
//! walk via [`Config::skip_open_check`].
//!
//! # Configuration
//!
//! [`Config`] is a `Clone` builder. Defaults match a
//! "production-safe" posture:
//!
//! - [`Config::cache_size`] — bytes for the pager's LRU. Default
//!   256 KiB (64 frames). Larger for read-heavy workloads on
//!   large databases; smaller on memory-constrained targets.
//! - [`Config::sync_mode`] — durability mode for every WAL
//!   commit. Default [`SyncMode::Full`] (system-wide power loss
//!   survivable). [`SyncMode::Normal`] for `fsync`-only
//!   durability; [`SyncMode::Off`] only for tests and benchmarks.
//! - [`Config::busy_timeout`] — max wait when acquiring the
//!   reader / writer lock. Default 5 seconds. Beyond the budget,
//!   the txn returns [`Err(Error::Busy)`](Error::Busy) rather
//!   than blocking indefinitely.
//! - [`Config::skip_open_check`] — opt out of the open-time
//!   catalog walk. Default `false` (run the walk). Production
//!   callers should leave it on.
//! - [`Config::cross_process_lock`] — toggle OS-level byte-range
//!   locking. Default `true` (on). Off only when every accessor
//!   shares one `Db` inside one process (in-process stress tests).
//!
//! # Cargo features
//!
//! - `serde` (off by default) — derive `serde::Serialize` and
//!   `serde::Deserialize` on the public types in this crate
//!   (`Config`, `DbStat`, `CollectionStat`, `DumpRecord`,
//!   `IntegrityReport`, `IntegrityFailure`, plus the obj-core
//!   re-exports `Id`, `SyncMode`, `LockKind`, `IndexKind`,
//!   `IndexSpec`). When the feature is on, `Serialize` and
//!   `Deserialize` are also re-exported from the crate root, so
//!   downstream callers do not need a separate `serde` dependency.
//!   Pure additive surface — no on-disk format byte changes.
//! - `tracing` (off by default) — emit structured spans around the
//!   observability surface: `db.open`, `db.transaction`,
//!   `db.read_transaction`, `db.integrity_check`, `query.execute`,
//!   and the obj-core `pager.checkpoint` span (propagated via the
//!   `obj-core/tracing` sub-feature). The feature gates the
//!   optional `tracing` dependency on both crates so the default
//!   build has zero new transitive deps and zero span overhead.
//!   `tracing` is intentionally NOT re-exported from this crate —
//!   downstream subscribers add `tracing-subscriber` (or another
//!   subscriber crate) directly, mirroring the idiom used by
//!   `tokio` and `axum`.
//! - `compression` (off by default) — LZ4 per-page compression at
//!   the pager layer. Propagates to obj-core.
//!   Every v1.0 writer stamps `format_minor = 2` regardless of which
//!   codecs are enabled; whether a file *uses* compression is
//!   recorded by `feature_flags` bit 0, not by the minor. A build
//!   WITHOUT this feature opens any file whose bit 0 is clear, and
//!   refuses (with `Error::FormatFeatureUnsupported`) only a file
//!   that actually has the compression flag set.
//! - `encryption` (off by default) — XChaCha20-Poly1305 per-page
//!   at-rest encryption. Propagates to
//!   obj-core. As with compression, the file's minor is always 2;
//!   `feature_flags` bit 1 records whether the file is encrypted. A
//!   build WITHOUT this feature opens any file whose bit 1 is clear,
//!   and refuses (with `Error::FormatFeatureUnsupported`) a file
//!   whose bit 1 is set — the refusal keys off the feature flag, not
//!   the minor version.
//! - `async` (off by default) — runtime-agnostic async surface
//!   mirroring the blocking [`Db`] / [`Collection`] / [`Query`]
//!   API behind a new `obj::asynchronous` module. Work is routed
//!   through the
//!   `blocking` crate's process-wide
//!   thread pool, so the wrapper composes with Tokio, smol, and any
//!   other async runtime — no per-runtime
//!   sub-features. With the feature off the baseline build adds
//!   no new transitive dependencies and no async overhead.
//!
//! # Observability
//!
//! Enable the `tracing` feature to emit spans around database
//! operations; spans are gated and free when the feature is off.
//! The span set is small and stable: one `info`-level span at every
//! transaction boundary, one `debug`-level span at every query
//! execution and pager checkpoint. No span field captures user
//! payload bytes — the only string-ish field is `path` on
//! `db.open`, which is a filesystem path rather than user content.
//!
//! # `unsafe` policy
//!
//! This crate is `#![forbid(unsafe_code)]`. All `unsafe` lives in
//! `obj-core::platform` and carries a documented safety contract.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
// Rule 7 — shipping code never reaches for unwrap/expect/panic-family constructs.
// Gated on `not(test)` so unit and integration tests keep using them freely; only
// production code paths are held to the bound. (`unsafe` is forbidden crate-wide,
// so Rule 8's undocumented_unsafe_blocks is not applicable here.)
#![cfg_attr(not(test), deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_in_result,
    clippy::get_unwrap,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
))]

#[cfg(feature = "async")]
pub mod asynchronous;

mod cli;
mod collection;
mod config;
mod db;
mod index_bound;
mod index_maint;
mod integrity;
mod query;
mod range;
mod txn;

pub use crate::cli::{CollectionStat, DbStat, DumpIter, DumpRecord};
pub use crate::collection::{Collection, IterIndexRange, MAX_DISTINCT_IDS};
pub use crate::config::Config;
pub use crate::db::{Db, IterAll};
pub use crate::query::{Query, MAX_SORT_BUFFER};
pub use crate::range::DynamicRange;
pub use crate::txn::{ReadTxn, WriteTxn};

pub use obj_core::codec::{Dynamic, DynamicSchema, EnumVariantSchema, Schema};
pub use obj_core::integrity::{IntegrityFailure, IntegrityReport};
pub use obj_core::{
    CompressionMode, Document, Error, Id, IndexKind, IndexSpec, LockKind, Result, SyncMode,
};

/// Re-export of `serde::Serialize` + `serde::Deserialize` under the
/// opt-in `serde` feature. Lets downstream code write
/// `use obj::{Serialize, Deserialize}` without a separate `serde`
/// dependency — the same convention `tokio` and `axum` use.
#[cfg(feature = "serde")]
pub use serde::{Deserialize, Serialize};

/// `#[derive(obj::Document)]` proc-macro re-export.
///
/// Lives in the sibling `obj-derive` crate; re-exported here so
/// users only have to depend on `obj` to use the derive. The trait
/// itself is still `obj_core::Document` re-exported above —
/// proc-macros and traits share a single name namespace and Rust
/// resolves the two by use-site (`#[derive(Document)]` vs `impl
/// Document for ...`).
///
/// The derive fills in [`Document::COLLECTION`] (default: the type
/// name) and [`Document::VERSION`] (default: `1`). The struct still
/// needs serde derives — the macro intentionally does not emit them
/// so you stay in control of serde-level attributes
/// (`#[serde(rename = ...)]`, etc.).
///
/// # Examples
///
/// Derive with defaults:
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::Db;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, obj::Document)]
/// struct Order {
///     customer_id: u64,
///     total_cents: u64,
/// }
///
/// let dir = tempfile::tempdir()?;
/// let db = Db::open(dir.path().join("orders.obj"))?;
///
/// // `Document::COLLECTION` defaulted to "Order".
/// assert_eq!(<Order as obj::Document>::COLLECTION, "Order");
/// assert_eq!(<Order as obj::Document>::VERSION, 1);
///
/// let id = db.insert(Order { customer_id: 1, total_cents: 4_200 })?;
/// let back: Option<Order> = db.get::<Order>(id)?;
/// assert_eq!(back.map(|o| o.total_cents), Some(4_200));
/// # Ok(())
/// # }
/// ```
///
/// Override the defaults with `#[obj(...)]`:
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::Db;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, obj::Document)]
/// #[obj(collection = "people", version = 2)]
/// struct Customer {
///     name: String,
/// }
///
/// assert_eq!(<Customer as obj::Document>::COLLECTION, "people");
/// assert_eq!(<Customer as obj::Document>::VERSION, 2);
///
/// let dir = tempfile::tempdir()?;
/// let db = Db::open(dir.path().join("people.obj"))?;
/// let id = db.insert(Customer { name: "Ada".to_owned() })?;
/// let back: Customer = db
///     .get::<Customer>(id)?
///     .ok_or(obj::Error::InvalidArgument("just inserted"))?;
/// assert_eq!(back.name, "Ada");
/// # Ok(())
/// # }
/// ```
///
/// Multiple `#[obj(...)]` attributes compose, and key=value pairs
/// may share a single attribute. Both shapes produce the same impl.
///
/// # Declaring indexes
///
/// Four kinds map to the same `IndexSpec` shape:
///
/// | Kind      | Attribute                                                | Behaviour                                  |
/// |-----------|----------------------------------------------------------|--------------------------------------------|
/// | Standard  | `#[obj(index)]`                                          | B-tree index; duplicates allowed.          |
/// | Unique    | `#[obj(index = unique)]`                                 | Uniqueness enforced at write time.         |
/// | Each      | `#[obj(index = each)]`                                   | Indexes every element of a `Vec<T>` field. |
/// | Composite | `#[obj(index_composite(fields = ("a", "b")))]`           | One index over a tuple of fields.          |
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::Db;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Clone, Serialize, Deserialize, obj::Document)]
/// #[obj(collection = "customers_idx_doc")]
/// #[obj(index_composite(fields = ("region", "tier"), name = "by_region_tier"))]
/// struct Customer {
///     #[obj(index)]
///     customer_id: u64,
///     #[obj(index = unique)]
///     email: String,
///     #[obj(index = each)]
///     tags: Vec<String>,
///     region: String,
///     tier: String,
/// }
///
/// let dir = tempfile::tempdir()?;
/// let db = Db::open(dir.path().join("indexes.obj"))?;
/// let _id = db.insert(Customer {
///     customer_id: 1,
///     email: "ada@example.com".to_owned(),
///     tags: vec!["red".to_owned(), "blue".to_owned()],
///     region: "us-east".to_owned(),
///     tier: "gold".to_owned(),
/// })?;
///
/// // Unique-index point lookup. O(log n), no collection scan.
/// let by_email: Option<Customer> = db
///     .find_unique::<Customer>("email", "ada@example.com")?;
/// assert!(by_email.is_some());
/// # Ok(())
/// # }
/// ```
///
/// # Hand-implementing `Document`
///
/// The derive is sugar over a trait. Implement the trait directly
/// when you need full control — for example to share a
/// `historical_schemas()` body across many types, or to compute the
/// `indexes()` list at runtime:
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::{Db, Document, IndexSpec};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize)]
/// struct Customer { email: String }
///
/// impl Document for Customer {
///     const COLLECTION: &'static str = "customers_hand_doc";
///     const VERSION: u32 = 1;
///
///     fn indexes() -> Vec<IndexSpec> {
///         vec![IndexSpec::unique("email", "email").expect("static spec")]
///     }
/// }
///
/// let dir = tempfile::tempdir()?;
/// let _db = Db::open(dir.path().join("hand-idx.obj"))?;
/// # Ok(())
/// # }
/// ```
///
/// The reconciler runs on the first
/// [`WriteTxn::collection::<T>()`](WriteTxn::collection) call per
/// process per collection: it declares specs absent from the
/// catalog, flips active descriptors absent from `indexes()` to
/// `DroppedPending`, and leaves matches alone. Reconciliation
/// rides the user's WAL transaction — a rolled-back insert leaves
/// no half-created index behind.
///
/// # Schema evolution
///
/// Schema evolution is `(version bump) + (migrate)`. The historical
/// *wire shape* needed to walk old bytes is **not** something you keep
/// alive in code — it is recovered from the database file itself.
///
/// **How the shape gets onto disk.** Every typed write persists the
/// running version's schema into a reserved row of the catalog B+tree,
/// in the same WAL transaction as the document. The disk catalog holds
/// a schema row for version `K` **iff** some process running version-`K`
/// code wrote a version-`K` document to this file. At read time the
/// decoder sources the old shape from that row, not from a live old
/// class.
///
/// **Load-bearing invariant.** *A document stamped version `N` is
/// decodable iff a `(collection, N)` schema row exists on disk.* The
/// write path guarantees the row commits before (or with) the first
/// version-`N` document, so for any file written by this engine the
/// invariant always holds. There is no compiled-in read fallback.
///
/// Old records read through the new type are migrated in memory; their
/// on-disk bytes are not rewritten until the next `update` / `upsert`.
/// The collection therefore scales to billions of docs without a
/// stop-the-world rebuild on every schema change.
///
/// **Authoring `migrate`.** A type that needs a real migration
/// hand-writes its [`Document`] impl and **overrides
/// [`Document::migrate`]**. Do *not* write `impl Migrate for T` — the
/// [`Migrate`](obj_core::codec::Migrate) trait has a blanket impl over
/// every `Document`, so a direct impl conflicts and fails to compile.
/// Read the old fields out of the `Dynamic` map per field; deserializing
/// a `Dynamic::Map` as a struct misreads the map length as field 0.
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::{Db, Document, Schema};
/// use obj_core::codec::{Dynamic, DynamicSchema};
/// use serde::{Deserialize, Serialize};
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("evo.obj");
///
/// // --- v1: a plain derived Document. Inserting it persists the
/// //         derived v1 schema row in the same write txn. ---
/// let id = {
///     #[derive(Debug, Serialize, Deserialize, Document)]
///     #[obj(version = 1, collection = "customers_evo_doc")]
///     struct Customer {
///         name: String,
///         email: String,
///     }
///
///     let db = Db::open(&path)?;
///     db.insert(Customer {
///         name: "Ada".to_owned(),
///         email: "ada@example.com".to_owned(),
///     })?
/// };
///
/// // --- v2: same collection, higher version. v2 adds `tier` and
/// //         hand-writes `Document` to override `migrate`. It carries
/// //         NO historical_schemas() — the v1 wire shape comes only
/// //         from the catalog row the v1 writer persisted. ---
/// #[derive(Debug, Serialize, Deserialize)]
/// struct Customer {
///     name: String,
///     email: String,
///     tier: String,
/// }
///
/// // The insert/write path persists the current-version schema, so a
/// // hand-impl `Document` used as a writer also needs a `Schema` impl.
/// impl Schema for Customer {
///     fn schema() -> DynamicSchema {
///         DynamicSchema::map([
///             ("name", DynamicSchema::String),
///             ("email", DynamicSchema::String),
///             ("tier", DynamicSchema::String),
///         ])
///     }
/// }
///
/// impl Document for Customer {
///     const COLLECTION: &'static str = "customers_evo_doc";
///     const VERSION: u32 = 2;
///
///     // Override `Document::migrate` — NOT `impl Migrate for Customer`.
///     fn migrate(dynamic: Dynamic, from_version: u32) -> obj::Result<Self> {
///         if from_version != 1 {
///             return Err(obj::Error::SchemaMigrationNotImplemented {
///                 collection: Self::COLLECTION,
///                 from_version,
///                 to_version: Self::VERSION,
///             });
///         }
///         // Per-field extraction; the v1 shape is read from disk.
///         Ok(Customer {
///             name: dynamic.get_str("name")?.to_owned(),
///             email: dynamic.get_str("email")?.to_owned(),
///             tier: "standard".to_owned(), // default for the new field
///         })
///     }
/// }
///
/// // Reopen as v2 and read the v1 doc back: the migration walk sources
/// // the v1 wire shape entirely from the on-disk catalog row.
/// let db = Db::open(&path)?;
/// let back: Customer = db
///     .get::<Customer>(id)?
///     .ok_or(obj::Error::InvalidArgument("just inserted"))?;
/// assert_eq!(back.tier, "standard");
/// # Ok(())
/// # }
/// ```
///
/// The rules are mechanical:
///
/// 1. Bump `VERSION` on every breaking change.
/// 2. Override [`Document::migrate`] to transform the old `Dynamic`
///    view into the current shape. The codec walks the on-disk postcard
///    payload through the **stored** schema for the record's version
///    (sourced from disk) and hands your `migrate` body the resulting
///    `Dynamic`.
/// 3. `migrate` returns `Self`. Default values for new fields are
///    the migration's responsibility — there is no implicit
///    default.
///
/// ## Pure-additive shortcut: `#[obj(auto_migrate)]`
///
/// When a version bump **only adds fields** (no removals, renames, or
/// type changes), you do not need to hand-write `migrate` at all. Add
/// `#[obj(auto_migrate)]` to the struct and the derive generates the
/// `migrate` body for you: every current field is read from the older
/// record's `Dynamic` map by name; pre-existing fields carry over and
/// fields added in this version backfill with `Default::default()`.
///
/// ```
/// # fn main() -> obj::Result<()> {
/// use obj::{Db, Document};
/// use serde::{Deserialize, Serialize};
///
/// let dir = tempfile::tempdir()?;
/// let path = dir.path().join("auto_evo.obj");
///
/// // v1: a plain derived Document.
/// let id = {
///     #[derive(Debug, Serialize, Deserialize, Document)]
///     #[obj(version = 1, collection = "auto_customers")]
///     struct Customer { name: String, email: String }
///     let db = Db::open(&path)?;
///     db.insert(Customer { name: "Ada".into(), email: "ada@x.io".into() })?
/// };
///
/// // v2: adds `tier`; the derive generates `migrate`. No hand-impl.
/// #[derive(Debug, Serialize, Deserialize, Document)]
/// #[obj(version = 2, collection = "auto_customers", auto_migrate)]
/// struct Customer {
///     name: String,
///     email: String,
///     // Custom backfill for the new field (defaults to "" otherwise).
///     #[obj(default = "standard".to_owned())]
///     tier: String,
/// }
///
/// let db = Db::open(&path)?;
/// let back: Customer = db.get(id)?.ok_or(obj::Error::InvalidArgument("missing"))?;
/// assert_eq!(back.name, "Ada");            // carried over
/// assert_eq!(back.tier, "standard");       // backfilled
/// # Ok(())
/// # }
/// ```
///
/// `auto_migrate` and a hand-written `migrate` are mutually exclusive
/// (both define the same method). Use `auto_migrate` for additive
/// changes; hand-write the full `impl Document` for a removal with side
/// effects, a rename, a type change, or any backfill that varies by
/// `from_version`. A field whose type is not `Default` (and carries no
/// `#[obj(default = <expr>)]`) makes the generated body fail to compile
/// — a loud, intentional signal that the change is not purely additive.
///
/// A stored record whose `type_version` is newer than
/// `Self::VERSION` surfaces [`Error::SchemaVersionFromFuture`]; an
/// older `type_version` with no schema row on disk surfaces
/// [`Error::SchemaNotRegistered`]. For multi-version chains,
/// tombstoned fields, and enum-variant migration recipes, see the
/// integration tests `disk_schema_migration.rs`, `schema_evolution.rs`,
/// `tombstone_migration.rs`, `enum_migration.rs`, and
/// `lazy_migration.rs`.
pub use obj_derive::Document;
