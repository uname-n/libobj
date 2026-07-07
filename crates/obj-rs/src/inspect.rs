//! Diagnostic and inspection hooks on [`Db`].
//!
//! Type-erased introspection surface that does not fit cleanly into
//! the typed read/write transaction API:
//!
//! - [`Db::stat`] — collects a one-shot snapshot of header + catalog
//!   summary.
//! - [`Db::dump_raw`] — type-erased streaming walk of a named
//!   collection's primary B-tree.
//!
//! These methods are marked **may move pre-1.0** in rustdoc — the
//! typed `Db::iter_all` / `Db::collection::<T>()` API is the
//! long-term shape for user code; the helpers here exist for
//! inspecting files whose `Document` types are not statically known
//! (e.g. through the C ABI).

use std::collections::VecDeque;
use std::ops::Bound;
use std::sync::Arc;

use obj_core::btree::BTree;
use obj_core::codec::{DocumentHeader, DOC_HEADER_SIZE};
use obj_core::pager::page::PageId;
use obj_core::pager::Pager;
use obj_core::platform::FileHandle;
use obj_core::{CollectionDescriptor, Error, Id, Result};

use crate::txn::{AttachedReadCtx, ReadTxn};
use crate::Db;

/// Refill batch for [`DumpIter`]. Same rationale as
/// `ITER_ALL_BATCH` — a small fixed cap so peak memory is bounded
/// regardless of the collection's size.
const DUMP_BATCH: usize = 256;

/// One drained [`DumpIter`] refill batch: the staged records, the last
/// key seen (the next resumption marker), and the count drained (a
/// count `< DUMP_BATCH` signals end-of-tree). Aliased to keep the
/// refill-helper signatures within clippy's `type_complexity` budget.
type DumpBatch = (VecDeque<Result<DumpRecord>>, Option<Vec<u8>>, usize);

/// One-shot snapshot of a database's header + catalog summary.
///
/// Returned by [`Db::stat`]. May evolve pre-1.0; user
/// code should reach for the typed [`Db::iter_all`] /
/// `Db::read_transaction(|tx| tx.collection::<T>())` APIs instead.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbStat {
    /// Format major version from page 0.
    pub format_major: u16,
    /// Format minor version from page 0.
    pub format_minor: u16,
    /// On-disk page size in bytes.
    pub page_size: u16,
    /// Total number of pages in the database, including page 0.
    pub page_count: u64,
    /// Logical file size in bytes (`page_count * page_size`). For
    /// file-backed pagers this equals the on-disk size; for memory
    /// pagers it is the in-memory backing buffer length.
    pub file_size_bytes: u64,
    /// One entry per registered collection. Order matches the
    /// catalog B-tree's natural sort (by name).
    pub collections: Vec<CollectionStat>,
}

/// Per-collection summary inside [`DbStat`].
///
/// `doc_count` and `total_payload_bytes` are computed by walking
/// the collection's primary B-tree once; the cost is O(n) in the
/// number of documents in the collection. Computed once per
/// [`Db::stat`] call; tools that need realtime telemetry should NOT
/// use this surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CollectionStat {
    /// User-visible name.
    pub name: String,
    /// Catalog-assigned numeric id.
    pub collection_id: u32,
    /// `Document::VERSION` of the type that last wrote to the
    /// collection.
    pub type_version: u32,
    /// Number of documents in the primary B-tree.
    pub doc_count: u64,
    /// Approximate total payload bytes — sum of every doc header's
    /// `payload_len`. Does NOT include the 16-byte per-doc header
    /// itself; the in-file footprint is
    /// `payload_len + DOC_HEADER_SIZE` per doc, plus B+tree
    /// per-page overhead.
    pub total_payload_bytes: u64,
    /// Number of `Active` secondary indexes. `DroppedPending`
    /// descriptors are NOT counted.
    pub active_index_count: usize,
    /// Number of `DroppedPending` index descriptors (kept so the
    /// `index_id` is never reused; pages reclaimed on the next
    /// checkpoint).
    pub dropped_index_count: usize,
    /// Full secondary-index descriptors for this collection, in
    /// catalog order. Includes both `Active` and `DroppedPending`
    /// entries — `indexes.len() == active_index_count +
    /// dropped_index_count`. Surfaced for introspection tooling
    /// (e.g. a binding's descriptor view); the count fields above
    /// remain the cheap summary.
    pub indexes: Vec<obj_core::IndexDescriptor>,
}

/// One raw record yielded by [`DumpIter`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DumpRecord {
    /// Primary id, decoded from the B-tree key.
    pub id: Id,
    /// Per-document header (16 bytes on disk, decoded).
    pub header: DocumentHeader,
    /// Payload bytes following the header. Type-erased; no
    /// schema-aware decode is attempted.
    pub payload: Vec<u8>,
}

impl Db {
    /// One-shot snapshot of header + catalog summary.
    ///
    /// May move pre-1.0; user code
    /// should prefer the typed [`Db::iter_all`] /
    /// `Db::read_transaction(|tx| tx.collection::<T>())` APIs.
    ///
    /// # Errors
    ///
    /// - [`Error::Busy`] if the pager mutex is poisoned or contested.
    /// - Pager / B-tree / postcard errors propagated from the
    ///   catalog walk + per-collection primary-tree walks.
    pub fn stat(&self) -> Result<DbStat> {
        let descriptors = self.collect_descriptors()?;
        let mut collections: Vec<CollectionStat> = Vec::with_capacity(descriptors.len());
        for (name, descriptor) in descriptors {
            collections.push(self.stat_one_collection(&name, &descriptor)?);
        }
        let pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let (format_major, format_minor) = pager.format_version();
        let page_size = pager.page_size();
        let page_count = pager.page_count();
        let file_size_bytes = u64::from(page_size).saturating_mul(page_count);
        Ok(DbStat {
            format_major,
            format_minor,
            page_size,
            page_count,
            file_size_bytes,
            collections,
        })
    }

    /// Streaming type-erased walk of a named collection's primary
    /// B-tree. Each step yields a [`DumpRecord`]: the primary id,
    /// the per-doc header (decoded), and the raw payload bytes.
    ///
    /// `limit == 0` is treated as unbounded; the caller's iteration
    /// loop is the implicit bound. Callers who want a hard cap must
    /// impose it on their `take(...)`.
    ///
    /// The `Document` trait is
    /// NOT consulted; schema-aware decode requires a registered
    /// type and is the caller's responsibility above this layer.
    ///
    /// # Namespace dispatch + snapshot isolation
    ///
    /// A bare `collection` resolves against the calling Db's own
    /// env, but BOTH the catalog lookup and the primary-tree walk go
    /// through the read txn's pinned [`obj_core::ReaderSnapshot`]
    /// — the scan is snapshot-isolated against concurrent
    /// writers exactly as the point reads (`get_via_snapshot`) and the
    /// attached path are; a writer's post-snapshot node splits / page
    /// reuse cannot surface as a spurious `Corruption`. A
    /// `"<namespace>.<tail>"` name resolves against the read-only
    /// database attached under `<namespace>`: the iterator pins one
    /// [`obj_core::ReaderSnapshot`] on the attached env for its whole
    /// lifetime and walks that env's primary tree as-of the pinned LSN
    /// — mirroring the namespace dispatch the point-read shims
    /// (`get_with_version` etc.) use, so `all()` /
    /// `query.fetch()` over an attachment see the same documents the
    /// namespaced point reads do.
    ///
    /// # Errors
    ///
    /// - [`Error::CollectionNotFound`] if `collection` is not
    ///   registered AT THE SNAPSHOT'S PINNED LSN.
    /// - [`Error::CollectionNamespaceUnknown`] if `collection` carries
    ///   a namespace prefix that is not attached on this handle.
    /// - As [`Db::read_transaction`] (construction-time).
    pub fn dump_raw(&self, collection: &str, limit: usize) -> Result<DumpIter<'_>> {
        let inner = obj_core::ReadTxn::begin_with_timeout(&self.env, self.busy_timeout)?;
        let txn = ReadTxn::new(inner);
        let (descriptor, attached) = self.resolve_dump_target(&txn, collection)?;
        Ok(DumpIter {
            txn,
            attached,
            descriptor,
            buffer: VecDeque::new(),
            last_emitted_key: None,
            finished: false,
            limit,
            emitted: 0,
        })
    }

    /// Resolve `collection` to the `(descriptor, attached)` pair the
    /// [`DumpIter`] should walk. A bare name resolves against `txn`'s
    /// pinned [`obj_core::ReaderSnapshot`] via
    /// [`obj_core::Catalog::lookup_via_snapshot`] (`attached == None`,
    /// walked locally) — this yields a `primary_root` consistent
    /// with the snapshot the scan reads pages through, so a concurrent
    /// writer's freelist-recycled catalog page cannot surface as a
    /// spurious `Corruption`. A `"<ns>.<tail>"` name pins a snapshot on
    /// the attached env and resolves `<tail>` against THAT snapshot's
    /// catalog, so the iterator is consistent with the namespaced
    /// point-read shims.
    fn resolve_dump_target(
        &self,
        txn: &ReadTxn<'_>,
        collection: &str,
    ) -> Result<(CollectionDescriptor, Option<AttachedReadCtx>)> {
        let (namespace, tail) = crate::db::try_split_namespace(collection)?;
        let Some(ns) = namespace else {
            let pager = self.env.pager().lock().map_err(|_| Error::Busy {
                kind: obj_core::LockKind::WriterInProcess,
            })?;
            let descriptor = obj_core::Catalog::<FileHandle>::lookup_via_snapshot(
                &pager,
                txn.inner.snapshot(),
                collection,
            )?
            .ok_or_else(|| Error::CollectionNotFound {
                name: collection.to_owned(),
            })?;
            return Ok((descriptor, None));
        };
        let ctx = self.pin_attached_ctx(ns)?;
        let descriptor = {
            let pager = ctx.env.pager().lock().map_err(|_| Error::Busy {
                kind: obj_core::LockKind::WriterInProcess,
            })?;
            obj_core::Catalog::<FileHandle>::lookup_via_snapshot(&pager, &ctx.snapshot, tail)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: collection.to_owned(),
                })?
        };
        Ok((descriptor, Some(ctx)))
    }

    /// Walk the catalog and return `(name, descriptor)` pairs. Held
    /// across the per-collection stats walk so the catalog mutex is
    /// not held simultaneously with the pager mutex inside
    /// [`Self::stat_one_collection`].
    fn collect_descriptors(&self) -> Result<Vec<(String, CollectionDescriptor)>> {
        let mut pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let catalog = self.catalog.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        catalog.list_collections(&mut pager)
    }

    /// Walk one collection's primary B-tree, counting docs and
    /// summing payload bytes. Returns the populated
    /// [`CollectionStat`].
    fn stat_one_collection(
        &self,
        name: &str,
        descriptor: &CollectionDescriptor,
    ) -> Result<CollectionStat> {
        let (doc_count, total_payload_bytes) = self.walk_primary_for_stat(name)?;
        let active_index_count = descriptor
            .indexes
            .iter()
            .filter(|d| d.status == obj_core::IndexStatus::Active)
            .count();
        let dropped_index_count = descriptor
            .indexes
            .iter()
            .filter(|d| d.status == obj_core::IndexStatus::DroppedPending)
            .count();
        Ok(CollectionStat {
            name: name.to_owned(),
            collection_id: descriptor.collection_id,
            type_version: descriptor.type_version,
            doc_count,
            total_payload_bytes,
            active_index_count,
            dropped_index_count,
            indexes: descriptor.indexes.clone(),
        })
    }

    /// Streaming walk of one primary tree. Returns `(doc_count,
    /// total_payload_bytes)`. Bounded by
    /// `obj_core::catalog::MAX_COLLECTIONS * <doc cap>` — the B-tree
    /// range iterator does NOT carry a cap of its own, but the
    /// loop count is bounded by the on-disk doc count and so is
    /// itself bounded by the (finite) file size. We add an
    /// explicit overflow check on the running counters.
    ///
    /// The walk pins a [`obj_core::ReaderSnapshot`], re-resolves the
    /// collection's `primary_root` against THAT snapshot's catalog
    /// (via [`obj_core::Catalog::lookup_via_snapshot`]), and iterates
    /// the tree via [`BTree::range_via_snapshot`] so `Db::stat`
    /// is isolated from concurrent writers' node splits/merges and page
    /// reuse. Pinning a snapshot but walking the live root (or the live
    /// tree) would spuriously decode-error mid-scan. The snapshot is
    /// pinned at or after the live-catalog read that produced `name`, so
    /// the collection is guaranteed present at the pinned LSN.
    fn walk_primary_for_stat(&self, name: &str) -> Result<(u64, u64)> {
        let mut pager = self.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let snapshot = pager.reader_snapshot()?;
        let descriptor =
            obj_core::Catalog::<FileHandle>::lookup_via_snapshot(&pager, &snapshot, name)?
                .ok_or_else(|| Error::CollectionNotFound {
                    name: name.to_owned(),
                })?;
        let root_pid = PageId::new(descriptor.primary_root)
            .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
        let iter = BTree::<FileHandle>::range_via_snapshot(&pager, &snapshot, root_pid, ..)?;
        let mut doc_count: u64 = 0;
        let mut total: u64 = 0;
        for step in iter {
            let (_key, value) = step?;
            let header = DocumentHeader::read_from(&value)?;
            doc_count = doc_count
                .checked_add(1)
                .ok_or(Error::BTreeInvariantViolated {
                    reason: "stat doc-count overflow",
                })?;
            total = total.checked_add(u64::from(header.payload_len)).ok_or(
                Error::BTreeInvariantViolated {
                    reason: "stat payload-byte sum overflow",
                },
            )?;
        }
        Ok((doc_count, total))
    }
}

/// Streaming iterator returned by [`Db::dump_raw`].
///
/// Holds a [`ReadTxn`] for its lifetime so the snapshot pin keeps
/// the catalog and primary B-tree stable across refills. Yields
/// `Result<DumpRecord>` one entry at a time; per-step errors do
/// NOT terminate iteration — the caller decides whether to
/// continue. Construction errors surface at the [`Db::dump_raw`]
/// call site.
pub struct DumpIter<'db> {
    txn: ReadTxn<'db>,
    /// `Some(_)` for a namespaced (`"<ns>.<tail>"`) scan: the iterator
    /// walks this attached env's primary tree as-of the pinned snapshot.
    /// `None` for a bare-name scan, which walks the calling
    /// Db's primary tree as-of `txn`'s pinned reader snapshot —
    /// snapshot-isolated against concurrent writers, exactly as the
    /// attached path is.
    attached: Option<AttachedReadCtx>,
    descriptor: CollectionDescriptor,
    buffer: VecDeque<Result<DumpRecord>>,
    last_emitted_key: Option<Vec<u8>>,
    finished: bool,
    limit: usize,
    emitted: usize,
}

impl std::fmt::Debug for DumpIter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DumpIter")
            .field("collection_id", &self.descriptor.collection_id)
            .field("buffer_len", &self.buffer.len())
            .field("finished", &self.finished)
            .field("limit", &self.limit)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl DumpIter<'_> {
    /// Refill the internal buffer with up to [`DUMP_BATCH`] entries.
    /// Resumption marker is `Excluded(last_emitted_key)` — identical
    /// to the [`crate::IterAll`] refill pattern.
    ///
    /// Dispatches on [`Self::attached`]: a namespaced scan walks the
    /// attached env's primary tree as-of its pinned snapshot
    /// ([`Self::refill_attached`]); a bare-name scan walks the
    /// calling Db's primary tree as-of the dump txn's pinned
    /// [`obj_core::ReaderSnapshot`] ([`Self::refill_local`]) —
    /// the same snapshot-isolation guarantee the attached path and the
    /// point reads (`get_via_snapshot`) already enjoy, so concurrent
    /// writes cannot mutate the tree out from under the scan.
    fn refill(&mut self) -> Result<()> {
        let start = match &self.last_emitted_key {
            Some(k) => Bound::Excluded(k.clone()),
            None => Bound::Unbounded,
        };
        let primary_root = self.descriptor.primary_root;
        let (batch, last_key, drained) = match &self.attached {
            None => Self::refill_local(
                self.txn.inner.env(),
                self.txn.inner.snapshot(),
                primary_root,
                start,
            ),
            Some(ctx) => Self::refill_attached(ctx, primary_root, start),
        }?;
        if drained < DUMP_BATCH {
            self.finished = true;
        }
        self.buffer.extend(batch);
        if let Some(k) = last_key {
            self.last_emitted_key = Some(k);
        }
        Ok(())
    }

    /// Drain up to [`DUMP_BATCH`] entries from the calling Db's primary
    /// tree (bare-name scan), pinned to the dump txn's
    /// [`obj_core::ReaderSnapshot`] via
    /// [`BTree::range_via_snapshot`]. Walking the snapshot rather
    /// than the live tree (`BTree::range`) keeps the scan isolated from
    /// concurrent writers' node splits/merges and page reuse — the same
    /// guarantee as [`Self::refill_attached`] and the point reads.
    fn refill_local(
        env: &obj_core::TxnEnv<FileHandle>,
        snapshot: &obj_core::ReaderSnapshot<FileHandle>,
        primary_root: u64,
        start: Bound<Vec<u8>>,
    ) -> Result<DumpBatch> {
        let pager_arc: Arc<std::sync::Mutex<Pager<FileHandle>>> = Arc::clone(env.pager());
        let pager = pager_arc.lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let root_pid = PageId::new(primary_root)
            .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
        let iter = BTree::<FileHandle>::range_via_snapshot(
            &pager,
            snapshot,
            root_pid,
            (start, Bound::Unbounded),
        )?;
        drain_dump_batch(iter)
    }

    /// Drain up to [`DUMP_BATCH`] entries from an attached env's primary
    /// tree, pinned to the attachment's snapshot (namespaced scan).
    fn refill_attached(
        ctx: &AttachedReadCtx,
        primary_root: u64,
        start: Bound<Vec<u8>>,
    ) -> Result<DumpBatch> {
        let pager = ctx.env.pager().lock().map_err(|_| Error::Busy {
            kind: obj_core::LockKind::WriterInProcess,
        })?;
        let root_pid = PageId::new(primary_root)
            .ok_or(Error::InvalidArgument("collection primary_root is zero"))?;
        let iter = BTree::<FileHandle>::range_via_snapshot(
            &pager,
            &ctx.snapshot,
            root_pid,
            (start, Bound::Unbounded),
        )?;
        drain_dump_batch(iter)
    }
}

/// Drain at most [`DUMP_BATCH`] entries from a primary-tree range
/// iterator into a staged batch. Returns `(batch, last_key, drained)`
/// where `drained < DUMP_BATCH` signals end-of-tree to the caller.
///
/// Generic over the concrete range-iterator type (a single
/// static bound, no `dyn`) so the local and attached snapshot walks
/// (both `BTree::range_via_snapshot`) share one body.
/// Bounded by `DUMP_BATCH`; the overflow check guards the
/// running counter.
fn drain_dump_batch<I>(iter: I) -> Result<DumpBatch>
where
    I: Iterator<Item = Result<(Vec<u8>, Vec<u8>)>>,
{
    let mut yielded: usize = 0;
    let mut last_key: Option<Vec<u8>> = None;
    let mut batch: VecDeque<Result<DumpRecord>> = VecDeque::with_capacity(DUMP_BATCH);
    for step in iter {
        if yielded >= DUMP_BATCH {
            break;
        }
        yielded = yielded
            .checked_add(1)
            .ok_or(Error::BTreeInvariantViolated {
                reason: "dump_raw batch counter overflow",
            })?;
        buffer_one_dump_entry(&mut batch, &mut last_key, step);
    }
    Ok((batch, last_key, yielded))
}

/// Process one B-tree iterator step into the staged `batch` for
/// [`DumpIter::refill`]. Errors are captured as `Err` entries so
/// they surface via `next()` rather than aborting the refill.
fn buffer_one_dump_entry(
    batch: &mut VecDeque<Result<DumpRecord>>,
    last_key: &mut Option<Vec<u8>>,
    step: Result<(Vec<u8>, Vec<u8>)>,
) {
    let (key, value) = match step {
        Ok(kv) => kv,
        Err(e) => {
            batch.push_back(Err(e));
            return;
        }
    };
    let Some(id) = Id::from_be_bytes(&key) else {
        batch.push_back(Err(Error::InvalidArgument(
            "primary B-tree key is not an Id",
        )));
        return;
    };
    *last_key = Some(key);
    let header = match DocumentHeader::read_from(&value) {
        Ok(h) => h,
        Err(e) => {
            batch.push_back(Err(e));
            return;
        }
    };
    let payload = value
        .get(DOC_HEADER_SIZE..)
        .map(<[u8]>::to_vec)
        .unwrap_or_default();
    batch.push_back(Ok(DumpRecord {
        id,
        header,
        payload,
    }));
}

impl Iterator for DumpIter<'_> {
    type Item = Result<DumpRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.limit != 0 && self.emitted >= self.limit {
            return None;
        }
        if let Some(item) = self.buffer.pop_front() {
            self.emitted = self.emitted.saturating_add(1);
            return Some(item);
        }
        if self.finished {
            return None;
        }
        if let Err(e) = self.refill() {
            self.finished = true;
            return Some(Err(e));
        }
        let item = self.buffer.pop_front()?;
        self.emitted = self.emitted.saturating_add(1);
        Some(item)
    }
}
