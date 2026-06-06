//! [`IndexKind`] + [`IndexSpec`] — runtime declaration of a secondary
//! index.
//!
//! `IndexSpec` is the *runtime* declaration a `Document` type emits
//! from `Document::indexes()`; the on-disk reflection is the
//! `IndexDescriptor` in [`crate::catalog`]. The spec is small and
//! pure-data so it composes cleanly with the derive macro.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// What kind of secondary index a given [`IndexSpec`] declares.
///
/// The on-disk numeric discriminants are pinned. The `#[repr(u8)]`
/// attribute mirrors the spec so a future format-version that
/// streams the kind as a single byte (instead of postcard's
/// variant-index varint) can do so without a migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[non_exhaustive]
pub enum IndexKind {
    /// Non-unique scalar index. One B-tree entry per document — the
    /// encoded key carries the document's `Id` as an 8-byte
    /// big-endian suffix to keep the B-tree key globally unique
    /// (the B+tree rejects duplicate keys).
    Standard = 0,
    /// Unique scalar index. Encoded key is the user value alone (no
    /// `Id` suffix); collisions are surfaced as
    /// [`crate::Error::UniqueConstraintViolation`].
    Unique = 1,
    /// Multi-value index over a `Vec<T>` field. Emits one entry per
    /// element of the indexed sequence. Like `Standard`, the
    /// document's `Id` is appended to each encoded user key.
    Each = 2,
    /// Multi-field composite index. The B-tree key is the
    /// concatenation of every encoded field in `key_paths` order,
    /// prefixed by a single envelope tag byte. Composite indexes
    /// also append the `Id` suffix (composite + unique is not a
    /// supported combination — Composite is non-unique by
    /// construction).
    Composite = 3,
}

/// A runtime index declaration.
///
/// `Document::indexes()` returns a `Vec<IndexSpec>`; the catalog
/// reconciler compares this list against the catalog's stored
/// [`crate::catalog::IndexDescriptor`] rows and declares or drops
/// the difference.
///
/// # Construction
///
/// Prefer the kind-specific constructors over building the struct
/// literal — they enforce the per-kind path-count invariants:
///
/// ```
/// use obj_core::index::IndexSpec;
///
/// // Standard / Unique / Each take exactly one field path.
/// let by_email_unique = IndexSpec::unique("by_email", "email").expect("valid");
/// let by_status = IndexSpec::standard("by_status", "status").expect("valid");
/// let by_tag = IndexSpec::each("by_tag", "tags").expect("valid");
///
/// // Composite requires two or more.
/// let by_customer_time = IndexSpec::composite(
///     "by_customer_time",
///     &["customer_id", "placed_at"],
/// ).expect("valid");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IndexSpec {
    /// User-visible name. Stable across reopens; the catalog uses it
    /// to match a runtime spec against an on-disk descriptor.
    pub name: String,
    /// Discriminator. See [`IndexKind`].
    pub kind: IndexKind,
    /// Field path(s) within the document. Single-element for
    /// `Standard` / `Unique` / `Each`; ≥ 2 for `Composite`.
    pub key_paths: Vec<String>,
}

impl IndexSpec {
    /// Construct a [`IndexKind::Standard`] spec.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `name` or `path` is empty.
    pub fn standard<N: Into<String>, P: Into<String>>(name: N, path: P) -> Result<Self> {
        Self::scalar(IndexKind::Standard, name.into(), path.into())
    }

    /// Construct a [`IndexKind::Unique`] spec.
    ///
    /// # Errors
    ///
    /// As [`IndexSpec::standard`].
    pub fn unique<N: Into<String>, P: Into<String>>(name: N, path: P) -> Result<Self> {
        Self::scalar(IndexKind::Unique, name.into(), path.into())
    }

    /// Construct a [`IndexKind::Each`] spec. The indexed field MUST
    /// be a sequence-valued field at extract time; if it is not,
    /// `extract_index_keys` errors with
    /// [`Error::IndexFieldTypeMismatch`].
    ///
    /// # Errors
    ///
    /// As [`IndexSpec::standard`].
    pub fn each<N: Into<String>, P: Into<String>>(name: N, path: P) -> Result<Self> {
        Self::scalar(IndexKind::Each, name.into(), path.into())
    }

    /// Construct a [`IndexKind::Composite`] spec.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `name` is empty, if
    /// `paths` has fewer than two entries, or if any path is empty.
    pub fn composite<N: Into<String>>(name: N, paths: &[&str]) -> Result<Self> {
        let owned: Vec<String> = paths.iter().map(|s| (*s).to_owned()).collect();
        let spec = Self {
            name: name.into(),
            kind: IndexKind::Composite,
            key_paths: owned,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Construct a spec from its individual parts.
    ///
    /// Unlike the kind-specific constructors, this accepts an
    /// arbitrary [`IndexKind`] plus a `key_paths` vector and is the
    /// general entry point callers reach for when reconstructing a
    /// spec from an on-disk [`crate::catalog::IndexDescriptor`] (where
    /// the kind is data, not a compile-time choice). The result is
    /// [`validate`](IndexSpec::validate)d, so a malformed
    /// descriptor surfaces as an error rather than a silently-wrong
    /// spec.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `name` is empty, any path
    /// is empty, or the path count disagrees with the kind.
    pub fn from_parts<N: Into<String>>(
        name: N,
        kind: IndexKind,
        key_paths: Vec<String>,
    ) -> Result<Self> {
        let spec = Self {
            name: name.into(),
            kind,
            key_paths,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Validate the spec's shape against the per-kind invariants.
    ///
    /// Called by every constructor; safe to call again on a
    /// round-tripped (postcard-decoded) spec as a defense-in-depth
    /// check before the catalog stamps the descriptor.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `name` is empty, any path is
    ///   empty, or the path count disagrees with the kind.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::InvalidArgument("index name must be non-empty"));
        }
        if self.key_paths.iter().any(String::is_empty) {
            return Err(Error::InvalidArgument("index key path must be non-empty"));
        }
        match self.kind {
            IndexKind::Standard | IndexKind::Unique | IndexKind::Each => {
                if self.key_paths.len() != 1 {
                    return Err(Error::InvalidArgument(
                        "Standard/Unique/Each indexes require exactly one key path",
                    ));
                }
            }
            IndexKind::Composite => {
                if self.key_paths.len() < 2 {
                    return Err(Error::InvalidArgument(
                        "Composite indexes require at least two key paths",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Helper that constructs one of the three scalar-shaped specs
    /// (`Standard`, `Unique`, `Each`) and validates.
    fn scalar(kind: IndexKind, name: String, path: String) -> Result<Self> {
        let spec = Self {
            name,
            kind,
            key_paths: vec![path],
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_constructors_set_kind_and_single_path() {
        let s = IndexSpec::standard("by_x", "x").expect("standard");
        assert_eq!(s.kind, IndexKind::Standard);
        assert_eq!(s.key_paths, vec!["x".to_owned()]);

        let u = IndexSpec::unique("by_email", "email").expect("unique");
        assert_eq!(u.kind, IndexKind::Unique);
        assert_eq!(u.key_paths, vec!["email".to_owned()]);

        let e = IndexSpec::each("by_tag", "tags").expect("each");
        assert_eq!(e.kind, IndexKind::Each);
        assert_eq!(e.key_paths, vec!["tags".to_owned()]);
    }

    #[test]
    fn composite_requires_two_or_more_paths() {
        let ok = IndexSpec::composite("by_ct", &["c", "t"]).expect("ok");
        assert_eq!(ok.kind, IndexKind::Composite);
        assert_eq!(ok.key_paths, vec!["c".to_owned(), "t".to_owned()]);

        let err = IndexSpec::composite("by_one", &["only"]).expect_err("too few");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn empty_name_or_path_rejected() {
        let err = IndexSpec::standard("", "x").expect_err("empty name");
        assert!(matches!(err, Error::InvalidArgument(_)));
        let err = IndexSpec::standard("by_x", "").expect_err("empty path");
        assert!(matches!(err, Error::InvalidArgument(_)));
        let err = IndexSpec::composite("c", &["", "y"]).expect_err("empty middle path");
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn validate_idempotent() {
        let s = IndexSpec::standard("by_x", "x").expect("ok");
        s.validate().expect("re-validate");
        s.validate().expect("re-validate again");
    }

    #[test]
    fn postcard_round_trip() {
        let s = IndexSpec::composite("by_ct", &["c", "t"]).expect("ok");
        let bytes = postcard::to_allocvec(&s).expect("encode");
        let back: IndexSpec = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(s, back);
        back.validate().expect("post-decode validate");
    }
}
