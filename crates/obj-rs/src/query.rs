//! `Query<T>` — the query builder.
//!
//! `Query` is a thin builder over the `Collection` API. It
//! borrows a `&Db` for the duration of the build phase; `.fetch()`
//! and `.count()` consume / inspect the builder and open a fresh
//! `read_transaction` for the actual scan.
//!
//! ## Sources
//!
//! - [`Source::Full`] — full collection scan via
//!   [`crate::Collection::all`]. Order is by primary `Id`.
//! - [`Source::IndexRange`] — index-range scan via
//!   [`crate::Collection::index_range`]. Order is by encoded index
//!   key bytes.
//!
//! No cost-based planner: the caller picks the source.
//!
//! ## Filters
//!
//! Filters are arbitrary `Fn(&T) -> bool + 'static` closures. They
//! are evaluated on the decoded document, so they pay the per-doc
//! decode cost; an `.index_range(...)` source keeps the work bounded
//! by walking only the index slice.
//!
//! Multiple `.filter(...)` calls AND together — every predicate must
//! return `true` for a doc to be emitted. The closures are stored as
//! `Box<dyn Fn(&T) -> bool + 'static>`; `'static` lets the closure
//! outlive the temporary builder.

use std::ops::Bound;

use obj_core::codec::Dynamic;
use obj_core::{Error, Result};

use crate::Db;
use crate::Document;

/// Boxed filter predicate. `'static` so the closure can outlive the
/// `Query` builder.
type FilterFn<T> = Box<dyn Fn(&T) -> bool + 'static>;

/// Boxed sort-key extractor. The closure may fail at encode time —
/// e.g. a `sort_by(|t| Dynamic::String(...))` whose string carries
/// an embedded NUL byte that the order-preserving encoder rejects.
/// `sort_by_bytes` callers wrap an infallible byte-producing closure
/// in `Ok(_)`; `sort_by` callers run their `Dynamic` output through
/// `encode_field` here and propagate the failure rather than
/// collapsing it to an empty key. See [`Query::sort_by`].
type SortKeyFn<T> = Box<dyn Fn(&T) -> Result<Vec<u8>> + 'static>;

/// Default cap on the in-memory sort buffer. The query layer reads
/// at most this many surviving documents into RAM before sorting; a
/// scan that produces more candidates surfaces
/// [`Error::SortBufferExceeded`].
pub const MAX_SORT_BUFFER: usize = 100_000;

/// Where a [`Query`] reads its candidate documents from.
///
/// Construct via the [`Query`] builder methods; the source is full-
/// scan by default and switches to `IndexRange` on
/// [`Query::index_range`].
#[derive(Debug, Clone)]
enum Source {
    /// Full primary-tree scan via [`crate::Collection::all`].
    Full,
    /// Walk the named index's B-tree slice via
    /// [`crate::Collection::index_range`]. Bounds are stored as
    /// already-encoded byte ranges — the builder runs
    /// `encode_field` at the boundary so the borrow-checker does not
    /// need to track `Dynamic` across the read txn.
    IndexRange {
        /// Index name.
        name: String,
        /// Encoded lower bound.
        start: Bound<Vec<u8>>,
        /// Encoded upper bound.
        end: Bound<Vec<u8>>,
    },
}

/// The query builder.
///
/// Obtain via [`Db::query`]. Compose with [`Query::filter`],
/// [`Query::limit`], and [`Query::index_range`]; terminate with
/// [`Query::fetch`].
///
/// `Query` borrows `&'db Db` so multiple builders can coexist; the
/// borrow ends when `.fetch()` returns. The actual scan runs inside
/// a fresh `read_transaction` opened by `.fetch()` — the builder
/// itself holds no locks.
pub struct Query<'db, T: Document> {
    /// Borrowed `Db` so the builder is cheap to construct and to drop
    /// without committing to the scan up-front.
    db: &'db Db,
    /// Where to draw candidate documents from.
    source: Source,
    /// User-supplied predicates. Applied in declaration order; every
    /// predicate must return `true` for a doc to be emitted.
    filters: Vec<FilterFn<T>>,
    /// Optional caller-supplied cap on the result count.
    limit: Option<usize>,
    /// Optional sort-key extractor. When set, the fetch collects up
    /// to `sort_buffer_limit` filtered candidates into RAM, sorts by
    /// the extractor's byte output, then applies `limit`. Last-call-
    /// wins if `sort_by` is invoked multiple times.
    sort_key: Option<SortKeyFn<T>>,
    /// Per-query override for the sort buffer ceiling. Defaults to
    /// [`MAX_SORT_BUFFER`]. Only consulted when `sort_key` is set.
    sort_buffer_limit: Option<usize>,
}

impl<'db, T: Document> Query<'db, T>
where
    T: Send + 'static,
{
    /// Construct a fresh full-scan query. Crate-internal — public
    /// callers go through [`Db::query`].
    pub(crate) fn new(db: &'db Db) -> Self {
        Self {
            db,
            source: Source::Full,
            filters: Vec::new(),
            limit: None,
            sort_key: None,
            sort_buffer_limit: None,
        }
    }

    /// Append a filter predicate. Filters compose with AND — every
    /// predicate must return `true` for a doc to be emitted.
    ///
    /// `'static` is required so the closure can outlive the
    /// temporary builder; capture by value if you need to borrow a
    /// stack-local value.
    #[must_use]
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&T) -> bool + 'static,
    {
        self.filters.push(Box::new(predicate));
        self
    }

    /// Cap the result set at `n` documents. Order is the source
    /// order (primary `Id` for full-scan; index-key bytes for an
    /// index range).
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Switch the query source from full-scan to the named index's
    /// range. The bounds may be any scalar that converts into a
    /// [`Dynamic`] (`u64`, `i64`, `&str`, …) — see
    /// [`DynamicRange`](crate::DynamicRange) —
    /// so a bare `40u64..60` works without wrapping each end in
    /// `Dynamic::U64(..)`. The builder encodes them through
    /// `obj_core::index::encode_field` at call time so the actual
    /// range arithmetic sees byte-ordered keys.
    ///
    /// Order is by the index key bytes, not by primary `Id`. The
    /// scan is bounded to the slice of the index B-tree the range
    /// covers — no full-collection walk.
    ///
    /// # Examples
    ///
    /// Range query on an indexed `u64` field — scalar bounds, no
    /// `Dynamic::` wrapping:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "orders_index_range_doc")]
    /// struct Order {
    ///     #[obj(index)]
    ///     placed_at: u64,
    /// }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("range.obj"))?;
    /// for i in 0..100u64 {
    ///     let _ = db.insert(Order { placed_at: i * 1_000 })?;
    /// }
    /// let recent: Vec<Order> = db
    ///     .query::<Order>()
    ///     .index_range("placed_at", 30_000u64..60_000)?
    ///     .fetch()?;
    /// assert_eq!(recent.len(), 30);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - [`obj_core::Error::Codec`] if a `Dynamic::String` bound
    ///   carries an embedded NUL byte (the order-preserving encoder
    ///   rejects those — see `obj_core::index::encode_field`).
    pub fn index_range<R>(mut self, name: &str, range: R) -> Result<Self>
    where
        R: crate::range::DynamicRange,
    {
        let (start, end) = range.into_dynamic_bounds();
        let start = encode_bound(start.as_ref())?;
        let end = encode_bound(end.as_ref())?;
        self.source = Source::IndexRange {
            name: name.to_owned(),
            start,
            end,
        };
        Ok(self)
    }

    /// Sort the result by `key`'s output in ascending key order.
    ///
    /// `key` returns an [`obj_core::codec::Dynamic`] for each
    /// document; the builder runs each value through
    /// [`obj_core::index::encode_field`] so the comparator
    /// is a byte comparison whose ordering matches the value's
    /// natural `Ord`. This reuses the same order-preserving encoder
    /// the index layer uses, so a `.sort_by(|o| o.placed_at.into())`
    /// produces the same ordering an `index_range("placed_at", ...)`
    /// scan would visit.
    ///
    /// Last-call-wins: a second `.sort_by` (or `.sort_by_bytes`)
    /// overwrites the first; the two extractors share the same
    /// internal slot because they are mutually exclusive.
    ///
    /// The sort runs BEFORE [`Query::limit`] truncation, so
    /// `.sort_by(...).limit(N)` returns the N smallest by the
    /// extractor's key — the "top-N sorted" workload.
    ///
    /// # Errors at fetch time
    ///
    /// `sort_by` itself is infallible — it stores the closure. The
    /// encoder runs during [`Query::fetch`]; a `Dynamic` whose
    /// `encode_field` representation cannot be computed (e.g. a
    /// `Dynamic::String` carrying an embedded `0x00` byte that the
    /// order-preserving encoder rejects) surfaces as
    /// [`Error::SortKeyEncode`]. Callers who want to bypass
    /// `encode_field` entirely should use [`Query::sort_by_bytes`].
    ///
    /// # Sort-buffer bound
    ///
    /// The pre-sort buffer is capped at
    /// [`Query::sort_buffer_limit`] (default
    /// [`MAX_SORT_BUFFER`] = 100 000). A scan that produces more
    /// candidates surfaces [`Error::SortBufferExceeded`]; the user
    /// should narrow the source via `.filter` / `.index_range` /
    /// `.limit`, or raise the cap with `.sort_buffer_limit(N)`.
    #[must_use]
    pub fn sort_by<F>(mut self, key: F) -> Self
    where
        F: Fn(&T) -> Dynamic + 'static,
    {
        let encoded: SortKeyFn<T> = Box::new(move |doc: &T| {
            let dynamic = key(doc);
            obj_core::index::encode_field(&dynamic)
                .map(obj_core::index::EncodedIndexKey::into_bytes)
                .map_err(|e| Error::SortKeyEncode {
                    source: Box::new(e),
                })
        });
        self.sort_key = Some(encoded);
        self
    }

    /// Sort the result by `key`'s raw byte output in ascending order.
    ///
    /// Companion to [`Query::sort_by`] that lets callers supply the
    /// already-encoded sort bytes. The byte-order = sort-order
    /// invariant is the caller's responsibility: two documents whose
    /// key bytes compare `Less` MUST also be in the desired sort
    /// order, otherwise the produced ordering is unspecified.
    ///
    /// Bypassing [`obj_core::index::encode_field`] means
    /// [`Error::SortKeyEncode`] cannot fire — `sort_by_bytes` is the
    /// right shape for callers who already have an order-preserving
    /// byte form (e.g. a precomputed `i64::to_be_bytes` of a signed
    /// counter that they have already biased to the unsigned range).
    ///
    /// Last-call-wins: `sort_by_bytes` overwrites a prior
    /// [`Query::sort_by`] (and vice versa).
    ///
    /// # Sort-buffer bound
    ///
    /// Same bound as [`Query::sort_by`] — see that method's docs.
    #[must_use]
    pub fn sort_by_bytes<F>(mut self, key: F) -> Self
    where
        F: Fn(&T) -> Vec<u8> + 'static,
    {
        let encoded: SortKeyFn<T> = Box::new(move |doc: &T| Ok(key(doc)));
        self.sort_key = Some(encoded);
        self
    }

    /// Override the per-query sort-buffer ceiling. Only consulted
    /// when [`Query::sort_by`] is set. Defaults to
    /// [`MAX_SORT_BUFFER`] (100 000).
    ///
    /// A scan that overshoots the cap surfaces
    /// [`Error::SortBufferExceeded`]; narrow the candidate set
    /// with [`Query::filter`] / [`Query::index_range`] /
    /// [`Query::limit`], or raise the cap with this method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use obj_core::codec::Dynamic;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "ticks_sort_buffer_doc")]
    /// struct Tick { value: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("sort.obj"))?;
    /// for v in 0..10u64 {
    ///     let _ = db.insert(Tick { value: v })?;
    /// }
    /// let top: Vec<Tick> = db
    ///     .query::<Tick>()
    ///     .sort_by(|t| Dynamic::U64(t.value))
    ///     .sort_buffer_limit(1_000) // narrower than the default
    ///     .limit(3)
    ///     .fetch()?;
    /// assert_eq!(top.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn sort_buffer_limit(mut self, n: usize) -> Self {
        self.sort_buffer_limit = Some(n);
        self
    }

    /// Execute the query and materialise the matching documents.
    ///
    /// Opens a fresh `read_transaction`, drives the configured
    /// source iterator, applies every filter in declaration order,
    /// and truncates to `limit` (if set). Returns the documents in
    /// source order.
    ///
    /// # Errors
    ///
    /// - As [`Db::read_transaction`].
    /// - Any error from the underlying [`crate::Collection`] scan.
    pub fn fetch(self) -> Result<Vec<T>> {
        #[cfg(feature = "tracing")]
        let span = tracing::debug_span!("query.execute", kind = tracing::field::Empty);
        #[cfg(feature = "tracing")]
        let _guard = span.enter();
        #[cfg(feature = "tracing")]
        span.record("kind", query_kind(&self.source));
        self.db.read_transaction(|tx| {
            let coll = tx.collection::<T>()?;
            if self.sort_key.is_some() {
                fetch_sorted(&coll, &self)
            } else {
                fetch_unsorted(&coll, &self)
            }
        })
    }

    /// Count the documents this query would return, ignoring
    /// [`Query::sort_by`] (sort does not change the count) but
    /// honouring [`Query::limit`] (returns `min(total, limit)`).
    ///
    /// Takes `&self` rather than consuming the builder so callers
    /// can chain a follow-up `.fetch()` on the same predicate set.
    ///
    /// # Fast path (no filter set)
    ///
    /// When no filter is set, `count` walks the source B-tree
    /// without decoding any document. The exact shape of the walk
    /// depends on the source's index kind so the answer matches
    /// what `fetch` would return:
    ///
    /// - **Full scan** — walks the primary B-tree counting entries
    ///   (`Collection::count_all`). One entry per doc; the count is
    ///   exact.
    /// - **`Standard` / `Unique` / `Composite` `index_range`** — walks
    ///   the named index's B-tree counting entries
    ///   (`Collection::count_index_range`). One entry per doc;
    ///   the count is exact.
    /// - **`Each` `index_range`** — walks the named index's B-tree
    ///   tracking distinct trailing-`id` suffixes via
    ///   `Collection::count_distinct_ids_in_range`. A single doc may
    ///   emit multiple entries under
    ///   different element keys; the entry count would overshoot.
    ///   The distinct set is bounded by
    ///   [`crate::MAX_DISTINCT_IDS`] (100 000); exceeding it
    ///   surfaces [`obj_core::Error::DistinctCountExceeded`].
    ///
    /// # Slow path (filter set)
    ///
    /// When at least one filter is set, the predicate has to see a
    /// decoded `T`, so the slow path pays the per-doc decode cost
    /// (same as `fetch`). Sort, if set, is ignored — the total
    /// count is the same.
    ///
    /// # Examples
    ///
    /// Count without materialising documents:
    ///
    /// ```
    /// # fn main() -> obj::Result<()> {
    /// use obj::Db;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, Serialize, Deserialize, obj::Document)]
    /// #[obj(collection = "counts_doc")]
    /// struct Order { customer_id: u64 }
    ///
    /// let dir = tempfile::tempdir()?;
    /// let db = Db::open(dir.path().join("count.obj"))?;
    /// for i in 0..30u64 {
    ///     let _ = db.insert(Order { customer_id: i % 3 })?;
    /// }
    /// let n: u64 = db.query::<Order>()
    ///     .filter(|o| o.customer_id == 1)
    ///     .count()?;
    /// assert_eq!(n, 10);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// - As [`Db::read_transaction`].
    /// - Any error from the underlying [`crate::Collection`] scan.
    /// - [`obj_core::Error::DistinctCountExceeded`] on the `Each`
    ///   fast path.
    pub fn count(&self) -> Result<u64> {
        #[cfg(feature = "tracing")]
        let span = tracing::debug_span!("query.execute", kind = tracing::field::Empty);
        #[cfg(feature = "tracing")]
        let _guard = span.enter();
        #[cfg(feature = "tracing")]
        span.record("kind", query_kind(&self.source));
        self.db.read_transaction(|tx| {
            let coll = tx.collection::<T>()?;
            let total = if self.filters.is_empty() {
                count_fast(&coll, &self.source)?
            } else {
                count_slow(&coll, self)?
            };
            Ok(apply_count_limit(total, self.limit))
        })
    }
}

/// Map the [`Source`] variant to the `kind` value recorded on the
/// `query.execute` span.
///
/// `Source::Full` → `"filter"` (a full primary-tree walk plus
/// in-memory predicates is the "filter scan" path); `Source::IndexRange`
/// → `"index"` (the work is bounded by the named index's slice).
#[cfg(feature = "tracing")]
fn query_kind(source: &Source) -> &'static str {
    match source {
        Source::Full => "filter",
        Source::IndexRange { .. } => "index",
    }
}

/// Apply the optional `.limit(N)` cap to a raw count. `min(total, N)`
/// when `limit` is set; the raw total otherwise.
fn apply_count_limit(total: u64, limit: Option<usize>) -> u64 {
    match limit {
        Some(n) => total.min(u64::try_from(n).unwrap_or(u64::MAX)),
        None => total,
    }
}

/// Drain the configured source with no `sort_by` set.
///
/// The loop is bounded by the source iterator's length (the primary
/// B-tree's `MAX_RANGE_NODES` budget). The `.limit(N)` cap further
/// shortens the loop.
fn fetch_unsorted<T>(coll: &crate::Collection<'_, T>, q: &Query<'_, T>) -> Result<Vec<T>>
where
    T: Document + Send + 'static,
{
    if q.limit == Some(0) {
        return Ok(Vec::new());
    }
    let mut out: Vec<T> = Vec::new();
    for_each_candidate(coll, q, |doc| {
        if !q.filters.iter().all(|f| f(&doc)) {
            return Ok(true);
        }
        out.push(doc);
        if let Some(n) = q.limit {
            if out.len() >= n {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    Ok(out)
}

/// Drain the configured source with `sort_by` set.
///
/// Collects up to `sort_buffer_limit` filtered candidates into RAM,
/// sorts ascending by the extractor's byte output, then truncates
/// to `limit`. The buffer is bounded — exceeding it surfaces
/// [`Error::SortBufferExceeded`] rather than chewing arbitrary
/// memory.
fn fetch_sorted<T>(coll: &crate::Collection<'_, T>, q: &Query<'_, T>) -> Result<Vec<T>>
where
    T: Document + Send + 'static,
{
    let cap = q.sort_buffer_limit.unwrap_or(MAX_SORT_BUFFER);
    let sort_key = q
        .sort_key
        .as_ref()
        .ok_or(Error::InvalidArgument("fetch_sorted without sort_key"))?;
    let mut buf: Vec<(Vec<u8>, T)> = Vec::new();
    for_each_candidate(coll, q, |doc| {
        if !q.filters.iter().all(|f| f(&doc)) {
            return Ok(true);
        }
        if buf.len() >= cap {
            return Err(Error::SortBufferExceeded { limit: cap });
        }
        let key_bytes = sort_key(&doc)?;
        buf.push((key_bytes, doc));
        Ok(true)
    })?;
    buf.sort_by(|a, b| a.0.cmp(&b.0));
    let truncated_len = match q.limit {
        Some(n) => buf.len().min(n),
        None => buf.len(),
    };
    let mut out: Vec<T> = Vec::with_capacity(truncated_len);
    for (_k, d) in buf.into_iter().take(truncated_len) {
        out.push(d);
    }
    Ok(out)
}

/// Walk the configured source and call `f(doc)` for each decoded
/// document. `f` returns `Ok(true)` to continue, `Ok(false)` to stop
/// early (e.g. the limit is reached), or `Err(_)` to abort the scan.
///
/// Factors the source-dispatch + iteration boilerplate so the
/// sorted / unsorted entry points stay readable.
fn for_each_candidate<T, F>(
    coll: &crate::Collection<'_, T>,
    q: &Query<'_, T>,
    mut f: F,
) -> Result<()>
where
    T: Document + Send + 'static,
    F: FnMut(T) -> Result<bool>,
{
    match &q.source {
        Source::Full => {
            let docs = coll.all()?;
            for (_id, doc) in docs {
                if !f(doc)? {
                    return Ok(());
                }
            }
        }
        Source::IndexRange { name, start, end } => {
            let iter = coll.index_range_encoded(name, clone_bound(start), clone_bound(end))?;
            for step in iter {
                let (_key, doc) = step?;
                if !f(doc)? {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// No-filter fast path for [`Query::count`].
///
/// Walks the underlying B-tree (primary or index) without decoding
/// any document. For an `Each`-kind `index_range` source, dispatches
/// to [`crate::Collection::count_distinct_ids_in_range`] so the count
/// matches what `fetch` would return (which de-duplicates per doc).
/// Other kinds use the cheaper entry-count
/// [`crate::Collection::count_index_range`] path. See the rustdoc on
/// `Query::count` for the full per-kind table.
fn count_fast<T>(coll: &crate::Collection<'_, T>, source: &Source) -> Result<u64>
where
    T: Document + Send + 'static,
{
    match source {
        Source::Full => coll.count_all(),
        Source::IndexRange { name, start, end } => {
            let kind = coll.index_kind(name)?;
            if kind == obj_core::IndexKind::Each {
                coll.count_distinct_ids_in_range_encoded(name, clone_bound(start), clone_bound(end))
            } else {
                coll.count_index_range_encoded(name, clone_bound(start), clone_bound(end))
            }
        }
    }
}

/// Filter-applied slow path for [`Query::count`].
///
/// Must decode each candidate to evaluate the predicate. The decode
/// cost is unavoidable: there is no inverted-index data structure
/// that would let us count filter matches without touching the
/// document.
fn count_slow<T>(coll: &crate::Collection<'_, T>, q: &Query<'_, T>) -> Result<u64>
where
    T: Document + Send + 'static,
{
    let mut n: u64 = 0;
    for_each_candidate(coll, q, |doc| {
        if q.filters.iter().all(|f| f(&doc)) {
            n = n.checked_add(1).ok_or(Error::BTreeInvariantViolated {
                reason: "slow-path count exceeds u64",
            })?;
        }
        Ok(true)
    })?;
    Ok(n)
}

/// Encode a `Bound<Dynamic>` into a `Bound<Vec<u8>>` using the
/// order-preserving field encoder. Used by [`Query::index_range`].
fn encode_bound(b: Bound<&Dynamic>) -> Result<Bound<Vec<u8>>> {
    match b {
        Bound::Included(v) => Ok(Bound::Included(
            obj_core::index::encode_field(v)?.into_bytes(),
        )),
        Bound::Excluded(v) => Ok(Bound::Excluded(
            obj_core::index::encode_field(v)?.into_bytes(),
        )),
        Bound::Unbounded => Ok(Bound::Unbounded),
    }
}

/// Clone a borrowed `Bound<Vec<u8>>` into an owned `Bound<Vec<u8>>`.
/// Used by [`fetch_index_range`] to hand the bounds to the
/// `Collection::index_range` API (which takes ownership).
fn clone_bound(b: &Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(v.clone()),
        Bound::Excluded(v) => Bound::Excluded(v.clone()),
        Bound::Unbounded => Bound::Unbounded,
    }
}
