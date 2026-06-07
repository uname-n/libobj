//! `obj-core` — internal storage engine for the `obj` embedded document
//! database.
//!
//! # ⚠️ UNSTABLE — not a stable public API
//!
//! `obj-core` is an **implementation detail of `obj-rs`**. It is published
//! to `crates.io` only because `obj-rs` depends on it; its public API carries
//! **no `SemVer` guarantee** and may change in any release, including patch
//! releases. Do not depend on `obj-core` directly — depend on `obj-rs` and
//! use the `obj` crate's API. Only `obj-rs`'s public surface is the
//! supported, `SemVer`-governed API; `obj-core` is deliberately excluded
//! from the public-api stability gate so the engine can evolve freely.
//!
//! This crate hosts the layered storage engine: the platform syscall
//! wrappers (L0), the pager (L1), the WAL (L2), the B-tree (L3), the
//! document codec (L4), the catalog (L5), and the transaction manager
//! (L7). Layers are built bottom-up.
//!
//! # On-disk format
//!
//! This crate is the reference implementation of the `.obj` file
//! format: the page-0 file header, the per-page CRC32C trailer scheme,
//! and the page-type tag enumeration.
//!
//! # `unsafe` policy
//!
//! This crate does **not** carry a crate-level `#![forbid(unsafe_code)]`
//! because the `platform` submodule holds the project's syscall
//! wrappers. Every other submodule should be safe; new submodules
//! SHOULD include `#![forbid(unsafe_code)]` of their own where
//! appropriate.

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

pub mod backup;
pub mod btree;
pub mod catalog;
pub mod codec;
#[cfg(feature = "encryption")]
pub mod crypto;
pub mod error;
pub mod id;
pub mod index;
pub mod integrity;
pub mod pager;
pub mod platform;
pub mod txn;
pub mod wal;

pub use crate::catalog::{
    lookup_schema_via_snapshot, schema_key, Catalog, CollectionDescriptor, IndexDescriptor,
    IndexStatus,
};
pub use crate::codec::Document;
pub use crate::error::{Error, LockKind, Result};
pub use crate::id::Id;
pub use crate::index::{IndexKind, IndexSpec};
pub use crate::integrity::{IntegrityFailure, IntegrityReport};
pub use crate::pager::{CompressionMode, PageHandle, ReaderSnapshot, SnapshotId};
pub use crate::platform::{FileBackend, SyncMode};
pub use crate::txn::{ReadTxn, TxnEnv, WriteTxn, DEFAULT_BUSY_TIMEOUT};
pub use crate::wal::Lsn;
