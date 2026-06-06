//! `AsyncQuery` — async-facing wrapper over the [`crate::Query`]
//! builder.
//!
//! The blocking [`crate::Query`] borrows `&'db Db`; the async builder
//! cannot keep that borrow alive across an `.await`. Instead,
//! `AsyncQuery` stores the *configuration* (source, filters, sort key,
//! limit) and re-materialises a `Query<'db, T>` on the inner `Db`
//! inside the [`blocking`] task at terminal-method time.
//!
//! Builder setters (`filter`, `sort_by`, `sort_by_bytes`, `limit`,
//! `index_range`, `sort_buffer_limit`) are synchronous because they
//! manipulate in-memory state only. Terminal methods (`fetch`,
//! `count`) are `async fn` — they hand the blocking scan off to the
//! `blocking` pool.

use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::Arc;

use obj_core::codec::Dynamic;
use obj_core::{Document, Result};

use crate::asynchronous::db::unblock;
use crate::Db;

/// Boxed filter predicate — same shape as the blocking
/// [`crate::Query`]'s internal `FilterFn`, with the extra `Send`
/// bound that the blocking-task hop requires.
type FilterFn<T> = Box<dyn Fn(&T) -> bool + Send + 'static>;

/// Boxed sort-key extractor producing the structured `Dynamic` shape.
/// Forwarded onto [`crate::Query::sort_by`] in the blocking task so
/// the existing `Error::SortKeyEncode` propagation lights up at fetch
/// time — no error swallowing.
type SortDynamicFn<T> = Box<dyn Fn(&T) -> Dynamic + Send + 'static>;

/// Boxed sort-key extractor producing the raw byte shape. Forwarded
/// onto [`crate::Query::sort_by_bytes`] in the blocking task; the
/// caller's contract is infallible.
type SortBytesFn<T> = Box<dyn Fn(&T) -> Vec<u8> + Send + 'static>;

/// Mutually-exclusive sort-key state. `sort_by` and `sort_by_bytes`
/// overwrite each other on the blocking [`crate::Query`] surface;
/// mirroring that here keeps the build step a one-for-one mapping.
enum SortKey<T> {
    Dynamic(SortDynamicFn<T>),
    Bytes(SortBytesFn<T>),
}

#[derive(Debug, Clone)]
enum AsyncSource {
    Full,
    IndexRange {
        name: String,
        start: Bound<Dynamic>,
        end: Bound<Dynamic>,
    },
}

/// Async-facing query builder. See [`crate::Query`] for the surface
/// semantics; this wrapper only changes the terminal methods to
/// `async fn` and adds `Send` to every stored closure.
pub struct AsyncQuery<T> {
    db: Arc<Db>,
    source: AsyncSource,
    filters: Vec<FilterFn<T>>,
    limit: Option<usize>,
    sort_key: Option<SortKey<T>>,
    sort_buffer_limit: Option<usize>,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> std::fmt::Debug for AsyncQuery<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncQuery")
            .field("source", &self.source)
            .field("filters", &self.filters.len())
            .field("limit", &self.limit)
            .field("sort_key", &self.sort_key.is_some())
            .field("sort_buffer_limit", &self.sort_buffer_limit)
            .finish_non_exhaustive()
    }
}

impl<T> AsyncQuery<T>
where
    T: Document + Send + 'static,
{
    pub(crate) fn new(db: Arc<Db>) -> Self {
        Self {
            db,
            source: AsyncSource::Full,
            filters: Vec::new(),
            limit: None,
            sort_key: None,
            sort_buffer_limit: None,
            _phantom: PhantomData,
        }
    }

    /// Append a filter predicate. Async sibling of
    /// [`crate::Query::filter`] — adds a `Send` bound so the closure
    /// can ride the blocking-task hop.
    #[must_use]
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&T) -> bool + Send + 'static,
    {
        self.filters.push(Box::new(predicate));
        self
    }

    /// Cap the result set at `n`. See [`crate::Query::limit`].
    #[must_use]
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Switch the source to a named index's range. See
    /// [`crate::Query::index_range`].
    ///
    /// # Deferred errors
    ///
    /// Infallible — like the blocking [`crate::Query::index_range`],
    /// the order-preserving encoder fires at `fetch` / `count` time,
    /// not here: the builder stores the structured `Dynamic` bounds
    /// and forwards them onto the blocking `Query` inside the blocking
    /// task. An unencodable bound surfaces from `fetch` / `count`.
    #[must_use]
    pub fn index_range<R>(mut self, name: &str, range: R) -> Self
    where
        R: crate::range::DynamicRange,
    {
        let (start, end) = range.into_dynamic_bounds();
        self.source = AsyncSource::IndexRange {
            name: name.to_owned(),
            start,
            end,
        };
        self
    }

    /// Sort the result by `key`'s output. See [`crate::Query::sort_by`].
    /// Adds a `Send` bound to the closure so it can ride the
    /// blocking-task hop.
    #[must_use]
    pub fn sort_by<F>(mut self, key: F) -> Self
    where
        F: Fn(&T) -> Dynamic + Send + 'static,
    {
        self.sort_key = Some(SortKey::Dynamic(Box::new(key)));
        self
    }

    /// Sort by an extracted key, mapping it through [`Into<Dynamic>`].
    /// See [`crate::Query::sort_by_key`]. Adds a `Send` bound to the
    /// closure so it can ride the blocking-task hop.
    #[must_use]
    pub fn sort_by_key<K, F>(self, key: F) -> Self
    where
        F: Fn(&T) -> K + Send + 'static,
        K: Into<Dynamic>,
    {
        self.sort_by(move |doc| key(doc).into())
    }

    /// Sort by raw bytes. See [`crate::Query::sort_by_bytes`].
    #[must_use]
    pub fn sort_by_bytes<F>(mut self, key: F) -> Self
    where
        F: Fn(&T) -> Vec<u8> + Send + 'static,
    {
        self.sort_key = Some(SortKey::Bytes(Box::new(key)));
        self
    }

    /// Override the per-query sort-buffer ceiling. See
    /// [`crate::Query::sort_buffer_limit`].
    #[must_use]
    pub fn sort_buffer_limit(mut self, n: usize) -> Self {
        self.sort_buffer_limit = Some(n);
        self
    }

    /// Materialise the matching documents. See [`crate::Query::fetch`].
    ///
    /// # Errors
    ///
    /// As [`crate::Query::fetch`].
    pub async fn fetch(self) -> Result<Vec<T>> {
        let AsyncQuery {
            db,
            source,
            filters,
            limit,
            sort_key,
            sort_buffer_limit,
            _phantom,
        } = self;
        unblock(move || {
            let q = build_blocking_query::<T>(
                &db,
                source,
                filters,
                limit,
                sort_key,
                sort_buffer_limit,
            );
            q.fetch()
        })
        .await
    }

    /// Return the first matching document, or `None` when nothing
    /// matches. Async mirror of [`crate::Query::first`].
    ///
    /// Forces an internal `.limit(1)` and reuses the async
    /// [`AsyncQuery::fetch`] path — no second decode route. Filters,
    /// `index_range`, and `sort_by` already configured are honoured;
    /// with a sort key set "first" is the smallest by the sort key.
    ///
    /// # Errors
    ///
    /// As [`AsyncQuery::fetch`].
    pub async fn first(self) -> Result<Option<T>> {
        Ok(self.limit(1).fetch().await?.into_iter().next())
    }

    /// Count matching documents. See [`crate::Query::count`].
    ///
    /// Takes `self` by value rather than by reference because the
    /// async-builder's stored closures are `!Sync` in the general
    /// case; consuming the builder avoids cloning the closures and
    /// keeps the surface honest about ownership.
    ///
    /// # Errors
    ///
    /// As [`crate::Query::count`].
    pub async fn count(self) -> Result<u64> {
        let AsyncQuery {
            db,
            source,
            filters,
            limit,
            sort_key,
            sort_buffer_limit,
            _phantom,
        } = self;
        unblock(move || {
            let q = build_blocking_query::<T>(
                &db,
                source,
                filters,
                limit,
                sort_key,
                sort_buffer_limit,
            );
            q.count()
        })
        .await
    }
}

/// Construct a blocking [`crate::Query<'db, T>`] from the async
/// builder's stored configuration. The lifetime `'db` is whatever the
/// caller's blocking task sees — borrowing `&'db Db` for the duration
/// of `.fetch()` / `.count()` is the standard blocking-query contract.
fn build_blocking_query<T>(
    db: &Db,
    source: AsyncSource,
    filters: Vec<FilterFn<T>>,
    limit: Option<usize>,
    sort_key: Option<SortKey<T>>,
    sort_buffer_limit: Option<usize>,
) -> crate::Query<'_, T>
where
    T: Document + Send + 'static,
{
    let mut q = db.query::<T>();
    match source {
        AsyncSource::Full => {}
        AsyncSource::IndexRange { name, start, end } => {
            q = q.index_range(&name, (start, end));
        }
    }
    for predicate in filters {
        q = q.filter(move |doc| predicate(doc));
    }
    match sort_key {
        Some(SortKey::Dynamic(f)) => {
            q = q.sort_by(move |doc| f(doc));
        }
        Some(SortKey::Bytes(f)) => {
            q = q.sort_by_bytes(move |doc| f(doc));
        }
        None => {}
    }
    if let Some(n) = limit {
        q = q.limit(n);
    }
    if let Some(n) = sort_buffer_limit {
        q = q.sort_buffer_limit(n);
    }
    q
}
