//! Document codec (L4) — per-document header + postcard payload.
//!
//! The codec sits between the typed `Document` API (this module) and
//! the B+tree (L3): every stored value in a collection's primary
//! B-tree is the byte string produced by [`encode`], and every
//! `Db::get` decodes through [`decode`]. The catalog (L5) supplies
//! the `collection_id` that the per-document header pins; the
//! catalog itself stores `CollectionDescriptor`s through this same
//! codec.

#![forbid(unsafe_code)]

pub mod dynamic;
pub mod dynamic_de;
pub mod header;
pub mod migrate;
pub mod schema;
pub mod stored_schema;

pub use crate::codec::dynamic::Dynamic;
pub use crate::codec::header::{DocumentHeader, DOC_HEADER_SIZE, MAX_INLINE_DOC};
pub use crate::codec::migrate::Migrate;
pub use crate::codec::schema::{DynamicSchema, EnumVariantSchema, Schema, MAX_SCHEMA_DEPTH};
pub use crate::codec::stored_schema::{
    collect_int_signedness, normalize_schema, normalize_schema_to_postcard, StoredSchema,
    STORED_SCHEMA_FORMAT_V1,
};

use crate::catalog::lookup_schema_via_snapshot;
use crate::error::{Error, Result};
use crate::pager::checksum::crc32c;
use crate::pager::{Pager, ReaderSnapshot};
use crate::platform::FileBackend;

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;

/// The trait every user document type implements.
///
/// `Document` types are `serde::Serialize + DeserializeOwned` so
/// they round-trip through postcard. Each implementation provides
/// two associated constants:
///
/// - [`COLLECTION`](Document::COLLECTION) — the collection name
///   under which records of this type are stored. The catalog
///   resolves it to a numeric `collection_id` at registration time;
///   the codec takes that id as an argument to [`encode`]/[`decode`].
/// - [`VERSION`](Document::VERSION) — the type's schema version.
///   Stored in every record's header; the decoder routes a stored-
///   version mismatch through [`Document::migrate`].
///
/// `Document` is `'static` so type-erased catalog rows can carry the
/// collection name as `&'static str`.
pub trait Document: Serialize + DeserializeOwned + 'static {
    /// The collection name this document type stores into.
    ///
    /// Must be a stable, application-chosen identifier. The
    /// catalog resolves it to a `collection_id` on first
    /// registration; subsequent opens reuse the existing id.
    const COLLECTION: &'static str;

    /// The schema version of this `Document` implementation.
    ///
    /// Bump on any breaking change (added/removed/renamed fields,
    /// changed semantics). The decoder enforces:
    ///
    /// - `header.type_version < VERSION` → dispatch through
    ///   [`Document::migrate`].
    /// - `header.type_version == VERSION` → decode directly.
    /// - `header.type_version > VERSION` →
    ///   [`Error::SchemaVersionFromFuture`].
    const VERSION: u32;

    /// Transform an older stored record into `Self`.
    ///
    /// The codec calls [`migrate`](Document::migrate) when a stored
    /// record's `type_version` is strictly less than [`VERSION`](Document::VERSION).
    /// `dynamic` is a structured [`Dynamic`] view of the older
    /// record — the codec walks the on-disk payload through the
    /// schema registered for `from_version` (see
    /// [`historical_schemas`](Document::historical_schemas)) and
    /// hands the resulting map-shaped `Dynamic` to this method.
    /// Concrete overrides read the fields they care about with
    /// [`Dynamic::get`](crate::codec::Dynamic::get) /
    /// [`Dynamic::get_str`](crate::codec::Dynamic::get_str) /
    /// [`Dynamic::deserialize`](crate::codec::Dynamic::deserialize)
    /// and construct the target `Self`.
    ///
    /// `from_version` is the on-disk `type_version` — always `<
    /// Self::VERSION` when this is invoked by the codec.
    ///
    /// # Default body
    ///
    /// Returns [`Error::SchemaMigrationNotImplemented`]. Real types
    /// override this method to handle older versions.
    ///
    /// # Errors
    ///
    /// User overrides MAY return any [`Error`] variant. The default
    /// returns [`Error::SchemaMigrationNotImplemented`].
    fn migrate(_dynamic: crate::codec::Dynamic, from_version: u32) -> Result<Self> {
        Err(Error::SchemaMigrationNotImplemented {
            collection: Self::COLLECTION,
            from_version,
            to_version: Self::VERSION,
        })
    }

    /// Schemas for stored records of OLDER `type_version`s than
    /// [`VERSION`](Document::VERSION).
    ///
    /// Returns a list of `(version, schema)` pairs sorted strictly
    /// ascending by version.  The codec consults this list whenever
    /// it observes `header.type_version < Self::VERSION`:
    ///
    /// 1. Look up the matching `version`.
    /// 2. Walk the on-disk payload bytes through
    ///    [`Dynamic::from_postcard_bytes`](crate::codec::Dynamic::from_postcard_bytes)
    ///    using the registered `schema`.
    /// 3. Hand the resulting structured `Dynamic` to
    ///    [`migrate`](Document::migrate) along with the stored
    ///    version.
    ///
    /// A miss (no entry for the stored version) surfaces as
    /// [`Error::SchemaNotRegistered`].
    /// The default body returns an empty list — a `Document` with
    /// no `historical_schemas()` cannot migrate any older payload.
    ///
    /// # Ordering
    ///
    /// The returned slice MUST be sorted strictly ascending by
    /// version. The codec debug-asserts on read; out-of-order
    /// entries are a programming bug.
    #[must_use]
    fn historical_schemas() -> Vec<(u32, crate::codec::schema::DynamicSchema)> {
        Vec::new()
    }

    /// Declared secondary indexes for this `Document` type.
    ///
    /// Default body returns the empty vector — no indexes. Override
    /// to declare per-collection indexes; the catalog reconciler
    /// compares this list against the catalog's stored
    /// descriptors on the FIRST `WriteTxn::collection::<T>()` call
    /// per process per collection and:
    ///
    /// - declares specs absent from the catalog,
    /// - flips active descriptors absent from this list to
    ///   `DroppedPending`,
    /// - leaves unchanged matches alone (idempotent).
    ///
    /// The reconciler runs inside the user's WAL transaction so a
    /// rolled-back txn leaves the catalog clean.
    ///
    /// `&self` is intentionally **not** taken — indexes are a
    /// type-level property of the `Document`, not a per-instance
    /// one.
    #[must_use]
    fn indexes() -> Vec<crate::index::IndexSpec> {
        Vec::new()
    }
}

/// Encode `doc` into the on-disk record format.
///
/// Layout: `DocumentHeader` (16 bytes) followed by
/// `postcard::to_allocvec(doc)`. Returns the assembled bytes in a
/// fresh `Vec<u8>`; allocation is unavoidable here because the
/// payload length is not known until postcard runs.
///
/// # Errors
///
/// - [`Error::Codec`] if postcard encoding fails.
/// - [`Error::DocumentTooLarge`] if the resulting record exceeds
///   [`MAX_INLINE_DOC`] — overflow chains for oversize records are
///   deferred to a later format-minor.
pub fn encode<T: Document>(doc: &T, collection_id: u32) -> Result<Vec<u8>> {
    let payload = postcard::to_allocvec(doc)?;
    let payload_len = u32::try_from(payload.len()).map_err(|_| Error::DocumentTooLarge {
        len: payload.len(),
        max: MAX_INLINE_DOC,
    })?;
    let payload_crc32c = crc32c(&payload);
    let header = DocumentHeader {
        collection_id,
        type_version: T::VERSION,
        payload_len,
        payload_crc32c,
    };
    let total = DOC_HEADER_SIZE
        .checked_add(payload.len())
        .ok_or(Error::DocumentTooLarge {
            len: usize::MAX,
            max: MAX_INLINE_DOC,
        })?;
    if total > MAX_INLINE_DOC {
        return Err(Error::DocumentTooLarge {
            len: total,
            max: MAX_INLINE_DOC,
        });
    }
    let mut out = Vec::with_capacity(total);
    header.write_to(&mut out);
    out.extend_from_slice(&payload);
    debug_assert_eq!(out.len(), total, "encode: assembled size mismatch");
    Ok(out)
}

/// Validate the per-document header at the runtime boundary
/// and return the parsed [`DocumentHeader`]
/// together with the payload slice.
///
/// This is the shared front-half of both [`decode`] and
/// [`decode_with`]: the two decode entry points must apply **exactly
/// the same** five validation layers in the same order, so they live
/// here once rather than being copied (and risking drift):
///
/// 1. `bytes.len() >= DOC_HEADER_SIZE` (via [`DocumentHeader::read_from`]).
/// 2. `header.collection_id == expected_collection_id`.
/// 3. `bytes.len() == DOC_HEADER_SIZE + header.payload_len`.
/// 4. CRC32C of the payload matches `header.payload_crc32c`.
/// 5. `header.type_version <= T::VERSION` (no future versions).
///
/// The version-routing decision (layer 6 — equal-version fast path vs
/// migration branch) is deliberately left to the caller: the two
/// decode functions diverge only there.
///
/// # Errors
///
/// - [`Error::Corruption`] (`page_id = 0`) on a malformed/truncated
///   header, a length mismatch, or a CRC mismatch.
/// - [`Error::CollectionIdMismatch`] on a collection-id mismatch.
/// - [`Error::SchemaVersionFromFuture`] on a stored record newer than
///   `T::VERSION`.
fn validate_record<T: Document>(
    bytes: &[u8],
    expected_collection_id: u32,
) -> Result<(DocumentHeader, &[u8])> {
    let header = DocumentHeader::read_from(bytes)?;
    if header.collection_id != expected_collection_id {
        return Err(Error::CollectionIdMismatch {
            expected: expected_collection_id,
            found: header.collection_id,
        });
    }
    let payload_len =
        usize::try_from(header.payload_len).map_err(|_| Error::Corruption { page_id: 0 })?;
    let total = DOC_HEADER_SIZE
        .checked_add(payload_len)
        .ok_or(Error::Corruption { page_id: 0 })?;
    if bytes.len() != total {
        return Err(Error::Corruption { page_id: 0 });
    }
    let payload = &bytes[DOC_HEADER_SIZE..total];
    if crc32c(payload) != header.payload_crc32c {
        return Err(Error::Corruption { page_id: 0 });
    }
    if header.type_version > T::VERSION {
        return Err(Error::SchemaVersionFromFuture {
            collection: T::COLLECTION,
            from: header.type_version,
            to: T::VERSION,
        });
    }
    Ok((header, payload))
}

/// Decode an on-disk record into a `T: Document` instance.
///
/// Validates the per-document header at the runtime boundary:
///
/// 1. `bytes.len() >= DOC_HEADER_SIZE`.
/// 2. `header.collection_id == expected_collection_id`.
/// 3. `bytes.len() == DOC_HEADER_SIZE + header.payload_len`.
/// 4. CRC32C of the payload matches `header.payload_crc32c`.
/// 5. `header.type_version <= T::VERSION` (no future versions).
/// 6. If `header.type_version < T::VERSION`, the codec consults
///    `T::historical_schemas()` for the matching version, walks
///    the payload through that schema via
///    [`Dynamic::from_postcard_bytes`](crate::codec::Dynamic::from_postcard_bytes),
///    and dispatches into [`Migrate::migrate`]
///    with the structured `Dynamic`. If no schema is
///    registered for `header.type_version`, the codec returns
///    [`Error::SchemaNotRegistered`] without invoking `migrate` —
///    silent fallback hides schema-evolution bugs.
///    Otherwise (versions match) the payload is decoded directly
///    via `postcard::from_bytes::<T>(payload)`.
///
/// # Errors
///
/// - [`Error::Corruption`] with `page_id = 0` on a malformed header
///   or CRC mismatch (the codec does not know the page id; callers
///   that need a specific id should wrap or re-emit).
/// - [`Error::CollectionIdMismatch`] on a collection-id mismatch.
/// - [`Error::SchemaVersionFromFuture`] on a stored record newer
///   than `T::VERSION`.
/// - [`Error::SchemaNotRegistered`] when the stored
///   `type_version` is older than `T::VERSION` and
///   `T::historical_schemas()` has no entry for it.
/// - [`Error::SchemaMigrationNotImplemented`] when the registered
///   `Migrate::migrate` body returns the default error.
/// - [`Error::SchemaTypeMismatch`] / [`Error::SchemaDepthExceeded`]
///   on a schema / payload disagreement.
/// - [`Error::Codec`] on postcard decode failures.
pub fn decode<T: Document>(bytes: &[u8], expected_collection_id: u32) -> Result<T> {
    let (header, payload) = validate_record::<T>(bytes, expected_collection_id)?;
    if header.type_version < T::VERSION {
        let history = <T as Document>::historical_schemas();
        debug_assert!(
            history.windows(2).all(|w| w[0].0 < w[1].0),
            "historical_schemas() must be strictly ascending by version",
        );
        let schema = history
            .iter()
            .find(|(v, _)| *v == header.type_version)
            .map(|(_, s)| s)
            .ok_or(Error::SchemaNotRegistered {
                collection: T::COLLECTION,
                version: header.type_version,
            })?;
        let dynamic = Dynamic::from_postcard_bytes(payload, schema)?;
        return <T as Migrate>::migrate(dynamic, header.type_version);
    }
    postcard::from_bytes::<T>(payload).map_err(Error::from)
}

/// Where [`decode_with`] sources the stored-version wire shape needed
/// to migrate an older record.
///
/// Both variants resolve a [`StoredSchema`] persisted on disk by an
/// older writer. There is
/// deliberately **no** compiled-in (`Type`) variant: the existing
/// [`decode`] already *is* the compiled-in / equal-version path, and a
/// `Type` variant here would force an otherwise-unused `F` generic on
/// callers that have no pager.
///
/// `Copy` (manual impls below) because every field is `Copy` (two
/// shared references plus a `u32`, or a single shared reference):
/// passing a `SchemaSource` by value is a trivial register copy, so
/// callers need not borrow it. The impls are hand-written rather than
/// derived so they do **not** carry a spurious `F: Copy` bound — `F`
/// is only ever held behind a shared reference, never by value.
pub enum SchemaSource<'a, F: FileBackend> {
    /// Snapshot-isolated catalog read, for point reads. Resolves the
    /// stored-version row via [`lookup_schema_via_snapshot`] at the
    /// reader's pinned LSN, so the schema row is read at the **same**
    /// snapshot as the document bytes it describes.
    Snapshot {
        /// The pager the catalog B+tree lives in.
        pager: &'a Pager<F>,
        /// The reader's pinned snapshot; its `root_catalog` is the
        /// catalog root descended.
        snapshot: &'a ReaderSnapshot<F>,
        /// The collection whose schema rows are looked up.
        collection_id: u32,
    },
    /// Pre-resolved per-(stored)version schemas, built once per scan
    /// (hot paths). Keyed by stored `type_version`. Avoids a B+tree
    /// descent per row.
    Resolved {
        /// `type_version → StoredSchema` for the collection being
        /// scanned, built once from the scan's snapshot.
        schemas: &'a BTreeMap<u32, StoredSchema>,
    },
}

impl<F: FileBackend> Clone for SchemaSource<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: FileBackend> Copy for SchemaSource<'_, F> {}

/// Decode an on-disk record into a `T: Document`, sourcing the
/// stored-version wire shape from `src` rather than from the reader's
/// compiled-in `T::historical_schemas()`.
///
/// Mirrors [`decode`]'s five validation layers verbatim (they share
/// `validate_record`) **and** its equal-version fast path verbatim:
/// when `header.type_version == T::VERSION` the payload is decoded
/// directly via `postcard::from_bytes::<T>` and `src` is never
/// consulted. Only the migration branch differs — instead of the
/// compiled-in history list, the stored-version [`StoredSchema`] is
/// resolved from `src`, [`respecialize`]d to recover the original
/// per-integer varint encoding, walked into a [`Dynamic`], and handed
/// to `<T as Migrate>::migrate` exactly once with
/// `from_version = header.type_version` (never chained).
///
/// # Errors
///
/// - Everything `validate_record` can return (corruption,
///   collection-id mismatch, future version).
/// - [`Error::SchemaNotRegistered`] when the migration branch is
///   reached but `src` has no row for `header.type_version`. A missing
///   row is **never** a silent `None`.
/// - [`Error::UnsupportedSchemaFormat`] / [`Error::Codec`] from the
///   catalog read (`Snapshot` source).
/// - [`Error::SchemaTypeMismatch`] / [`Error::SchemaDepthExceeded`] /
///   [`Error::Codec`] from re-specialization and the schema walk.
/// - Whatever the user's `migrate` override returns.
pub fn decode_with<T: Document, F: FileBackend>(
    bytes: &[u8],
    expected_collection_id: u32,
    src: SchemaSource<'_, F>,
) -> Result<T> {
    let (header, payload) = validate_record::<T>(bytes, expected_collection_id)?;
    if header.type_version < T::VERSION {
        let stored = resolve_stored_schema::<T, F>(src, header.type_version)?;
        let schema = respecialize(&stored)?;
        let dynamic = Dynamic::from_postcard_bytes(payload, &schema)?;
        return <T as Migrate>::migrate(dynamic, header.type_version);
    }
    postcard::from_bytes::<T>(payload).map_err(Error::from)
}

/// Resolve the stored-version [`StoredSchema`] for `version` from a
/// [`SchemaSource`]. A missing row maps to
/// [`Error::SchemaNotRegistered`] (the existing variant — never a
/// silent `None`), carrying `T::COLLECTION` and the stored version.
///
/// Takes `src` by value: it is only ever resolved once per record (the
/// migration branch is mutually exclusive with the equal-version fast
/// path), so there is nothing to keep it alive for afterwards.
fn resolve_stored_schema<T: Document, F: FileBackend>(
    src: SchemaSource<'_, F>,
    version: u32,
) -> Result<StoredSchema> {
    let found = match src {
        SchemaSource::Snapshot {
            pager,
            snapshot,
            collection_id,
        } => lookup_schema_via_snapshot(pager, snapshot, collection_id, version)?,
        SchemaSource::Resolved { schemas } => schemas.get(&version).cloned(),
    };
    found.ok_or(Error::SchemaNotRegistered {
        collection: T::COLLECTION,
        version,
    })
}

/// Re-specialize a normalized [`StoredSchema`] back into a
/// [`DynamicSchema`] whose integer slots carry the writer's *original*
/// signedness, so the schema walk decodes each integer with the right
/// varint encoding (plain for `U64`, zigzag for `I64`).
///
/// `stored.schema` was normalized at write time: every integer is
/// `U64`. The writer's true signedness was recorded slot-for-slot in
/// `stored.int_signed`, in the **same pre-order** that
/// [`collect_int_signedness`] visits integer slots (a node, then `Map`
/// fields in declaration order, `Seq` inner, then `Enum` variant
/// payloads in discriminant order). This walk rebuilds the tree in
/// that identical order, emitting [`DynamicSchema::I64`] for the i-th
/// integer slot when `int_signed[i]` is `true`, else
/// [`DynamicSchema::U64`].
///
/// The hint is **authoritative** for the stored bytes — the reader's
/// own type `T` is NOT used as the signedness oracle here (it is used
/// only later by `migrate`). A `U64` slot whose hint says `true` is
/// decoded as zigzag because that is how the bytes were written.
///
/// # Errors
///
/// - [`Error::SchemaDepthExceeded`] if the schema nests deeper than
///   [`MAX_SCHEMA_DEPTH`].
/// - [`Error::Codec`] if `int_signed` is exhausted before every
///   integer slot is filled, or has leftover entries after the walk —
///   a corrupt / mismatched row. This is a hard error, never a panic.
pub fn respecialize(stored: &StoredSchema) -> Result<DynamicSchema> {
    let mut hint = stored.int_signed.iter().copied();
    let out = respecialize_at(&stored.schema, 0, &mut hint)?;
    if hint.next().is_some() {
        return Err(Error::Codec(postcard::Error::DeserializeUnexpectedEnd));
    }
    debug_assert_eq!(
        collect_int_signedness(&out)?.len(),
        stored.int_signed.len(),
        "respecialize must consume exactly int_signed.len() hints",
    );
    Ok(out)
}

/// Recursion-free-by-bound worker for [`respecialize`].
///
/// Bounded by an explicit `depth` counter: the tree is at
/// most [`MAX_SCHEMA_DEPTH`] deep, so the call chain cannot exhaust the
/// stack. Each integer leaf consumes one entry from `hint`; running
/// out mid-walk is an error, never a panic.
fn respecialize_at(
    schema: &DynamicSchema,
    depth: usize,
    hint: &mut impl Iterator<Item = bool>,
) -> Result<DynamicSchema> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(Error::SchemaDepthExceeded {
            depth: MAX_SCHEMA_DEPTH,
        });
    }
    let next = depth + 1;
    match schema {
        DynamicSchema::U64 | DynamicSchema::I64 => match hint.next() {
            Some(true) => Ok(DynamicSchema::I64),
            Some(false) => Ok(DynamicSchema::U64),
            None => Err(Error::Codec(postcard::Error::DeserializeUnexpectedEnd)),
        },
        DynamicSchema::Null => Ok(DynamicSchema::Null),
        DynamicSchema::Bool => Ok(DynamicSchema::Bool),
        DynamicSchema::F64 => Ok(DynamicSchema::F64),
        DynamicSchema::String => Ok(DynamicSchema::String),
        DynamicSchema::Bytes => Ok(DynamicSchema::Bytes),
        DynamicSchema::Seq(inner) => Ok(DynamicSchema::Seq(Box::new(respecialize_at(
            inner, next, hint,
        )?))),
        DynamicSchema::Map(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for (name, field) in fields {
                out.push((name.clone(), respecialize_at(field, next, hint)?));
            }
            Ok(DynamicSchema::Map(out))
        }
        DynamicSchema::Enum(variants) => {
            let mut out = Vec::with_capacity(variants.len());
            for v in variants {
                out.push(EnumVariantSchema::new(
                    v.discriminant,
                    v.name.clone(),
                    respecialize_at(&v.payload, next, hint)?,
                ));
            }
            Ok(DynamicSchema::Enum(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TinyDoc {
        a: u32,
        b: String,
    }

    impl Document for TinyDoc {
        const COLLECTION: &'static str = "tiny";
        const VERSION: u32 = 1;
    }

    #[test]
    fn round_trip_small_document() {
        let d = TinyDoc {
            a: 42,
            b: "hello".to_owned(),
        };
        let bytes = encode(&d, 7).expect("encode");
        let back: TinyDoc = decode(&bytes, 7).expect("decode");
        assert_eq!(back, d);
    }

    #[test]
    fn collection_id_mismatch_errors() {
        let d = TinyDoc {
            a: 1,
            b: "x".to_owned(),
        };
        let bytes = encode(&d, 7).expect("encode");
        let err = decode::<TinyDoc>(&bytes, 9).expect_err("mismatched id");
        assert!(matches!(
            err,
            Error::CollectionIdMismatch {
                expected: 9,
                found: 7
            }
        ));
    }

    #[test]
    fn crc_mismatch_errors() {
        let d = TinyDoc {
            a: 1,
            b: "x".to_owned(),
        };
        let mut bytes = encode(&d, 7).expect("encode");
        bytes[DOC_HEADER_SIZE] ^= 0xFF;
        let err = decode::<TinyDoc>(&bytes, 7).expect_err("crc mismatch");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn truncated_payload_errors() {
        let d = TinyDoc {
            a: 1,
            b: "x".to_owned(),
        };
        let bytes = encode(&d, 7).expect("encode");
        let truncated = &bytes[..bytes.len() - 1];
        let err = decode::<TinyDoc>(truncated, 7).expect_err("truncated");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn header_too_short_errors() {
        let bytes = [0u8; DOC_HEADER_SIZE - 1];
        let err = decode::<TinyDoc>(&bytes, 0).expect_err("short header");
        assert!(matches!(err, Error::Corruption { page_id: 0 }));
    }

    #[test]
    fn oversize_document_errors() {
        #[derive(Serialize, Deserialize)]
        struct Big {
            blob: Vec<u8>,
        }
        impl Document for Big {
            const COLLECTION: &'static str = "big";
            const VERSION: u32 = 1;
        }
        let huge: Vec<u8> = vec![0xAB; MAX_INLINE_DOC + 64];
        let big = Big { blob: huge };
        let err = encode(&big, 1).expect_err("oversize");
        match err {
            Error::DocumentTooLarge { len, max } => {
                assert!(len > max, "len {len} should exceed max {max}");
                assert_eq!(max, MAX_INLINE_DOC);
            }
            other => panic!("expected DocumentTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn future_version_errors() {
        let d = TinyDoc {
            a: 1,
            b: "x".to_owned(),
        };
        let mut bytes = encode(&d, 7).expect("encode");
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = decode::<TinyDoc>(&bytes, 7).expect_err("future");
        assert!(matches!(
            err,
            Error::SchemaVersionFromFuture {
                collection: "tiny",
                from: 99,
                to: 1
            }
        ));
    }

    #[test]
    fn missing_schema_errors_schema_not_registered() {
        let payload = postcard::to_allocvec(&TinyDoc {
            a: 1,
            b: "x".to_owned(),
        })
        .expect("postcard");
        let payload_crc = crc32c(&payload);
        let header = DocumentHeader {
            collection_id: 7,
            type_version: 0,
            payload_len: u32::try_from(payload.len()).expect("fits u32"),
            payload_crc32c: payload_crc,
        };
        let mut bytes = Vec::with_capacity(DOC_HEADER_SIZE + payload.len());
        header.write_to(&mut bytes);
        bytes.extend_from_slice(&payload);
        let err = decode::<TinyDoc>(&bytes, 7).expect_err("v0 stored, v1 reader");
        assert!(matches!(
            err,
            Error::SchemaNotRegistered {
                collection: "tiny",
                version: 0,
            }
        ));
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct EvolvingV1 {
        a: u32,
    }

    impl Document for EvolvingV1 {
        const COLLECTION: &'static str = "evolving";
        const VERSION: u32 = 1;
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct EvolvingV2 {
        a: u32,
        b: String,
    }

    impl Document for EvolvingV2 {
        const COLLECTION: &'static str = "evolving";
        const VERSION: u32 = 2;

        fn historical_schemas() -> Vec<(u32, crate::codec::schema::DynamicSchema)> {
            vec![(
                1,
                crate::codec::schema::DynamicSchema::map([(
                    "a",
                    crate::codec::schema::DynamicSchema::U64,
                )]),
            )]
        }

        fn migrate(dynamic: crate::codec::Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let a = match dynamic.get("a") {
                Some(crate::codec::Dynamic::U64(n)) => {
                    u32::try_from(*n).map_err(|_| Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    })?
                }
                _ => {
                    return Err(Error::SchemaMigrationNotImplemented {
                        collection: Self::COLLECTION,
                        from_version,
                        to_version: Self::VERSION,
                    });
                }
            };
            Ok(EvolvingV2 {
                a,
                b: "<default>".to_owned(),
            })
        }
    }

    #[test]
    fn migrate_override_lifts_v1_to_v2() {
        let v1 = EvolvingV1 { a: 99 };
        let payload = postcard::to_allocvec(&v1).expect("postcard");
        let header = DocumentHeader {
            collection_id: 13,
            type_version: 1,
            payload_len: u32::try_from(payload.len()).expect("fits u32"),
            payload_crc32c: crc32c(&payload),
        };
        let mut record = Vec::with_capacity(DOC_HEADER_SIZE + payload.len());
        header.write_to(&mut record);
        record.extend_from_slice(&payload);

        let decoded: EvolvingV2 = decode(&record, 13).expect("migrate succeeds");
        assert_eq!(
            decoded,
            EvolvingV2 {
                a: 99,
                b: "<default>".to_owned(),
            }
        );
    }

    #[test]
    fn current_version_does_not_route_through_migrate() {
        let v2 = EvolvingV2 {
            a: 7,
            b: "in-band".to_owned(),
        };
        let bytes = encode(&v2, 13).expect("encode");
        let back: EvolvingV2 = decode(&bytes, 13).expect("decode");
        assert_eq!(back, v2);
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct UnregisteredV2 {
        a: u32,
    }

    impl Document for UnregisteredV2 {
        const COLLECTION: &'static str = "unregistered";
        const VERSION: u32 = 2;

        fn migrate(_dynamic: crate::codec::Dynamic, _from_version: u32) -> Result<Self> {
            unimplemented!("not reached when no schema is registered")
        }
    }

    #[test]
    fn missing_history_entry_errors_schema_not_registered() {
        let payload = postcard::to_allocvec(&UnregisteredV2 { a: 1 }).expect("postcard");
        let header = DocumentHeader {
            collection_id: 17,
            type_version: 1,
            payload_len: u32::try_from(payload.len()).expect("fits u32"),
            payload_crc32c: crc32c(&payload),
        };
        let mut record = Vec::with_capacity(DOC_HEADER_SIZE + payload.len());
        header.write_to(&mut record);
        record.extend_from_slice(&payload);
        let err = decode::<UnregisteredV2>(&record, 17).expect_err("unregistered");
        assert!(matches!(
            err,
            Error::SchemaNotRegistered {
                collection: "unregistered",
                version: 1,
            }
        ));
    }

    use crate::codec::schema::EnumVariantSchema;
    use crate::platform::FileHandle;
    use std::collections::BTreeMap;

    /// Build a v1 `EvolvingV1` record at `type_version` 1 under
    /// collection id `cid`, the same hand-built-header trick the
    /// migration tests above use (`EvolvingV2` owns the "evolving"
    /// collection at compile time, so we cannot `encode::<EvolvingV1>`).
    fn v1_record(cid: u32) -> Vec<u8> {
        let v1 = EvolvingV1 { a: 99 };
        let payload = postcard::to_allocvec(&v1).expect("postcard");
        let header = DocumentHeader {
            collection_id: cid,
            type_version: 1,
            payload_len: u32::try_from(payload.len()).expect("fits u32"),
            payload_crc32c: crc32c(&payload),
        };
        let mut record = Vec::with_capacity(DOC_HEADER_SIZE + payload.len());
        header.write_to(&mut record);
        record.extend_from_slice(&payload);
        record
    }

    #[test]
    fn decode_with_resolved_migrates_v1_to_v2() {
        let mut schemas: BTreeMap<u32, StoredSchema> = BTreeMap::new();
        let v1_schema = DynamicSchema::map([("a", DynamicSchema::U64)]);
        schemas.insert(1, StoredSchema::from_live(&v1_schema).expect("from_live"));

        let record = v1_record(13);
        let decoded: EvolvingV2 = decode_with::<EvolvingV2, FileHandle>(
            &record,
            13,
            SchemaSource::Resolved { schemas: &schemas },
        )
        .expect("migrate via Resolved source");
        assert_eq!(
            decoded,
            EvolvingV2 {
                a: 99,
                b: "<default>".to_owned(),
            }
        );
    }

    #[test]
    fn decode_with_missing_version_errors_schema_not_registered() {
        let schemas: BTreeMap<u32, StoredSchema> = BTreeMap::new();
        let record = v1_record(13);
        let err = decode_with::<EvolvingV2, FileHandle>(
            &record,
            13,
            SchemaSource::Resolved { schemas: &schemas },
        )
        .expect_err("missing v1 row");
        assert!(matches!(
            err,
            Error::SchemaNotRegistered {
                collection: "evolving",
                version: 1,
            }
        ));
    }

    #[test]
    fn decode_with_equal_version_matches_decode_fast_path() {
        let v2 = EvolvingV2 {
            a: 7,
            b: "in-band".to_owned(),
        };
        let bytes = encode(&v2, 13).expect("encode");
        let empty: BTreeMap<u32, StoredSchema> = BTreeMap::new();
        let via_with: EvolvingV2 = decode_with::<EvolvingV2, FileHandle>(
            &bytes,
            13,
            SchemaSource::Resolved { schemas: &empty },
        )
        .expect("equal-version fast path");
        let via_plain: EvolvingV2 = decode(&bytes, 13).expect("decode");
        assert_eq!(via_with, via_plain);
        assert_eq!(via_with, v2);
    }

    #[derive(Serialize, Deserialize)]
    struct OneSigned {
        v: i64,
    }

    #[derive(Serialize, Deserialize)]
    struct OneUnsigned {
        v: u64,
    }

    #[test]
    fn respecialize_signed_field_decodes_zigzag() {
        let bytes = postcard::to_allocvec(&OneSigned { v: -12_345 }).expect("encode");
        let real = DynamicSchema::map([("v", DynamicSchema::I64)]);
        let stored = StoredSchema::from_live(&real).expect("from_live");
        assert_eq!(
            stored.schema,
            DynamicSchema::map([("v", DynamicSchema::U64)])
        );
        assert_eq!(stored.int_signed, vec![true]);

        let respec = respecialize(&stored).expect("respecialize");
        let walked = Dynamic::from_postcard_bytes(&bytes, &respec).expect("walk");
        assert_eq!(walked.get("v"), Some(&Dynamic::I64(-12_345)));
    }

    #[test]
    fn respecialize_unsigned_field_decodes_plain_varint() {
        let bytes = postcard::to_allocvec(&OneUnsigned { v: 9_000_000_000 }).expect("encode");
        let real = DynamicSchema::map([("v", DynamicSchema::U64)]);
        let stored = StoredSchema::from_live(&real).expect("from_live");
        assert_eq!(stored.int_signed, vec![false]);

        let respec = respecialize(&stored).expect("respecialize");
        let walked = Dynamic::from_postcard_bytes(&bytes, &respec).expect("walk");
        assert_eq!(walked.get("v"), Some(&Dynamic::U64(9_000_000_000)));
    }

    #[test]
    fn respecialize_wrong_hint_misdecodes_proving_hint_is_load_bearing() {
        let bytes = postcard::to_allocvec(&OneSigned { v: -1 }).expect("encode");
        let normalized = DynamicSchema::map([("v", DynamicSchema::U64)]);
        let walked = Dynamic::from_postcard_bytes(&bytes, &normalized).expect("walk");
        assert_eq!(walked.get("v"), Some(&Dynamic::U64(1)));
        assert_ne!(walked.get("v"), Some(&Dynamic::I64(-1)));
    }

    #[test]
    fn respecialize_consumes_hint_in_preorder() {
        let real = DynamicSchema::map([
            ("id", DynamicSchema::U64),
            ("inner", DynamicSchema::map([("a", DynamicSchema::I64)])),
            (
                "choice",
                DynamicSchema::enumeration([
                    EnumVariantSchema::new(0, "None", DynamicSchema::Null),
                    EnumVariantSchema::new(1, "Some", DynamicSchema::I64),
                ]),
            ),
        ]);
        let stored = StoredSchema::from_live(&real).expect("from_live");
        let respec = respecialize(&stored).expect("respecialize");
        assert_eq!(respec, real);
    }

    #[test]
    fn respecialize_too_few_hints_errors_no_panic() {
        let rogue = StoredSchema {
            format: STORED_SCHEMA_FORMAT_V1,
            schema: DynamicSchema::map([("a", DynamicSchema::U64), ("b", DynamicSchema::U64)]),
            int_signed: vec![false],
        };
        let err = respecialize(&rogue).expect_err("too few hints");
        assert!(matches!(err, Error::Codec(_)));
    }

    #[test]
    fn respecialize_too_many_hints_errors_no_panic() {
        let rogue = StoredSchema {
            format: STORED_SCHEMA_FORMAT_V1,
            schema: DynamicSchema::map([("a", DynamicSchema::U64)]),
            int_signed: vec![false, true],
        };
        let err = respecialize(&rogue).expect_err("too many hints");
        assert!(matches!(err, Error::Codec(_)));
    }

    #[test]
    fn decode_with_snapshot_migrates_via_catalog() {
        use crate::catalog::Catalog;
        use crate::pager::{Config, Pager};
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("decode-with-snap.obj");
        let mut pager = Pager::<FileHandle>::open(&path, Config::default()).expect("open");

        pager.begin_txn();
        let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init");
        let v1_schema = DynamicSchema::map([("a", DynamicSchema::U64)]);
        catalog
            .put_schema(&mut pager, 13, 1, &v1_schema)
            .expect("put_schema v1");
        let _ = pager.commit().expect("commit");
        let snap = pager.reader_snapshot().expect("snapshot");

        let record = v1_record(13);
        let decoded: EvolvingV2 = decode_with(
            &record,
            13,
            SchemaSource::Snapshot {
                pager: &pager,
                snapshot: &snap,
                collection_id: 13,
            },
        )
        .expect("migrate via Snapshot source");
        assert_eq!(
            decoded,
            EvolvingV2 {
                a: 99,
                b: "<default>".to_owned(),
            }
        );

        let err = decode_with::<EvolvingV2, FileHandle>(
            &v1_record(99),
            99,
            SchemaSource::Snapshot {
                pager: &pager,
                snapshot: &snap,
                collection_id: 99,
            },
        )
        .expect_err("no schema row for collection 99");
        assert!(matches!(
            err,
            Error::SchemaNotRegistered {
                collection: "evolving",
                version: 1,
            }
        ));
    }
}
