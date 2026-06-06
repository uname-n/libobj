//! `AsyncCollection` — async-facing wrapper over a runtime-named
//! read-only [`crate::Collection`] handle.
//!
//! Construction is infallible (see [`crate::Db::collection`]); each
//! async method opens a fresh blocking [`crate::Db::collection`]
//! handle inside the [`blocking`] pool and dispatches one of its
//! read-only methods.
//!
//! Writes via a runtime-named handle are intentionally out of scope —
//! the runtime accessor is limited to reads. Use
//! [`crate::asynchronous::AsyncDb::transaction`] +
//! [`crate::WriteTxn::collection`] for the typed-write path.

use std::marker::PhantomData;
use std::sync::Arc;

use obj_core::{Document, Id, Result};

use crate::asynchronous::db::unblock;
use crate::Db;

/// Async-facing wrapper over a runtime-named read-only collection
/// handle.
///
/// Cheap to clone — holds one `Arc<Db>` + the collection name.
#[derive(Debug)]
pub struct AsyncCollection<T> {
    db: Arc<Db>,
    name: String,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> Clone for AsyncCollection<T> {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            name: self.name.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T> AsyncCollection<T>
where
    T: Document + Send + 'static,
{
    pub(crate) fn lazy(db: Arc<Db>, name: String) -> Self {
        Self {
            db,
            name,
            _phantom: PhantomData,
        }
    }

    /// Async sibling of [`crate::Collection::get`]. Opens a private
    /// read transaction inside the blocking task.
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::get`].
    pub async fn get(&self, id: Id) -> Result<Option<T>> {
        let db = Arc::clone(&self.db);
        let name = self.name.clone();
        unblock(move || db.collection::<T>(name).get(id)).await
    }

    /// Async sibling of [`crate::Collection::all`].
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::all`].
    pub async fn all(&self) -> Result<Vec<(Id, T)>> {
        let db = Arc::clone(&self.db);
        let name = self.name.clone();
        unblock(move || db.collection::<T>(name).all()).await
    }

    /// Async sibling of [`crate::Collection::values`].
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::values`].
    pub async fn values(&self) -> Result<Vec<T>> {
        let db = Arc::clone(&self.db);
        let name = self.name.clone();
        unblock(move || db.collection::<T>(name).values()).await
    }

    /// Async sibling of [`crate::Collection::count_all`].
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::count_all`].
    pub async fn count_all(&self) -> Result<u64> {
        let db = Arc::clone(&self.db);
        let name = self.name.clone();
        unblock(move || db.collection::<T>(name).count_all()).await
    }

    /// Async sibling of [`crate::Collection::find_unique`].
    ///
    /// # Errors
    ///
    /// As [`crate::Collection::find_unique`].
    pub async fn find_unique<K>(&self, index_name: &str, key: K) -> Result<Option<T>>
    where
        K: Into<obj_core::codec::Dynamic> + Send + 'static,
    {
        let db = Arc::clone(&self.db);
        let name = self.name.clone();
        let index_name = index_name.to_owned();
        unblock(move || db.collection::<T>(name).find_unique(&index_name, key)).await
    }
}
