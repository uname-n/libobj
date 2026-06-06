//! Schema-evolution trait — `Migrate`.
//!
//! `Migrate` is the API by which a [`Document`] type
//! transforms older stored records into the current shape. The codec
//! invokes [`migrate`](Migrate::migrate) whenever a stored record's
//! `type_version` is less than the reader's `Document::VERSION`.
//!
//! Every `T: Document` is implicitly `Migrate` via the blanket impl
//! at the bottom of this file. The default body of the inherent
//! method [`Document::migrate`] returns
//! [`Error::SchemaMigrationNotImplemented`].
//! Real types override [`Document::migrate`]
//! to handle older versions; `Migrate::migrate` forwards to it.
//!
//! The on-disk contract pinned here is permanent:
//! `type_version > T::VERSION` is always
//! [`Error::SchemaVersionFromFuture`];
//! `type_version < T::VERSION` always routes through `migrate`.

#![forbid(unsafe_code)]

use crate::codec::{Document, Dynamic};
#[cfg(doc)]
use crate::error::Error;
use crate::error::Result;

/// Static-dispatch shim trait so [`crate::codec::decode`] can write
/// `<T as Migrate>::migrate(...)` without taking `T: Migrate` as an
/// explicit bound on top of `T: Document`.
///
/// `Migrate` is implemented for every `T: Document` via the blanket
/// impl below, which forwards to the inherent method
/// [`Document::migrate`]. Concrete types
/// customise migration by overriding `Document::migrate` (which has
/// a default erroring body); they do NOT need to touch `Migrate`
/// directly.
///
/// The separation exists for two reasons:
///
/// 1. Rust does not have stable specialisation, so a blanket `impl
///    Migrate for T` cannot coexist with concrete `impl Migrate for
///    MyType` overrides. Putting the override on the `Document`
///    trait (a single, non-blanket impl per type) sidesteps the
///    conflict.
/// 2. Keeping `Migrate` as a thin shim trait preserves the
///    surface area (the codec calls `<T as
///    Migrate>::migrate(...)`) without forcing a separate
///    `impl Migrate for ...` block at every `Document` site.
///
/// # Errors
///
/// Propagates the override's errors. The default body returns
/// [`Error::SchemaMigrationNotImplemented`].
pub trait Migrate: Document {
    /// Migrate an older stored record into the current `Self` type.
    ///
    /// Forwards to [`Document::migrate`];
    /// see that method for the contract.
    ///
    /// # Errors
    ///
    /// Propagates [`Document::migrate`]'s
    /// errors verbatim. The default body returns
    /// [`Error::SchemaMigrationNotImplemented`].
    fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
        <Self as Document>::migrate(dynamic, from_version)
    }
}

impl<T: Document> Migrate for T {}
