//! Secondary indexes — index B+trees layered over the B+tree.
//!
//! Per-collection B+trees keyed by an order-preserving encoding of
//! one or more document fields, valued by the document's
//! [`Id`](crate::Id).
//!
//! Each declared index lives in its own B+tree (no new on-disk
//! page types). The index's root page-id is persisted inside the
//! owning [`CollectionDescriptor`](crate::catalog::CollectionDescriptor)
//! as an [`IndexDescriptor`](crate::catalog::IndexDescriptor) entry.
//!
//! # Module split
//!
//! - [`spec`] — runtime declaration shapes: [`IndexKind`],
//!   [`IndexSpec`], constructors. The descriptor on-disk shape lives
//!   in [`crate::catalog`] because the catalog owns the wire row.
//! - [`key`] — order-preserving byte encoding for `Dynamic` values
//!   plus the composite envelope.

#![forbid(unsafe_code)]

pub mod extract;
pub mod key;
pub mod spec;

pub use crate::index::extract::{extract_index_keys, MAX_EACH_ENTRIES};
pub use crate::index::key::{
    encode_field, encode_index_key, encode_index_key_parts, encoded_id_suffix_len, EncodedIndexKey,
    COMPOSITE_TAG,
};
pub use crate::index::spec::{IndexKind, IndexSpec};
