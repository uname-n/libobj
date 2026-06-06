//! Runtime-agnostic async surface.
//!
//! `obj::asynchronous` mirrors the blocking [`crate::Db`] /
//! [`crate::Collection`] / [`crate::Query`] API. Every async method
//! hands its synchronous body to the [`blocking`] crate's process-wide
//! thread pool via `blocking::unblock(...).await`; the engine itself
//! stays blocking. This composes with **any** async runtime: Tokio,
//! smol, glommio, and friends — no per-runtime sub-features.
//!
//! The module is gated under the `async` Cargo feature. With the
//! feature off, the baseline build adds no new transitive dependencies
//! and no async overhead — the entire module is `#[cfg(...)]`-excised.
//!
//! # Design
//!
//! - **Wrapping pattern.** [`AsyncDb`] wraps `Arc<Db>` internally.
//!   Each async method clones the `Arc` and moves the clone into the
//!   blocking task. The public blocking [`crate::Db`] does **not**
//!   derive `Clone` — async opt-in does not impose a public-API
//!   change on blocking users.
//! - **Closure-bodied methods.**
//!   [`AsyncDb::transaction`](AsyncDb::transaction) and
//!   [`AsyncDb::read_transaction`](AsyncDb::read_transaction) take a
//!   closure that runs **synchronously** inside the blocking task.
//!   The closure must be `Send + 'static` so it can move across the
//!   thread boundary. No `async fn` inside the transaction body —
//!   this is the standard "async-over-blocking" contract (sqlx,
//!   rusqlite/tokio-rusqlite, etc.).
//! - **Iterators.** Streaming iteration (`Db::iter_all` /
//!   `Stream<Item = Result<T>>`) is **not** wrapped in this phase.
//!   [`AsyncDb::all`] collects the entire collection in the blocking
//!   task and returns `Vec<T>`. A future `AsyncDb::stream_all` would
//!   add a `Stream` adapter — left as a follow-up.
//! - **Tracing.** When the `tracing` feature is also on, every async
//!   method captures `tracing::Span::current()` before
//!   `blocking::unblock` and re-enters it inside the closure via a
//!   guard, so spans propagate across the thread-pool hop. With
//!   `tracing` off, the wrapper is a pure pass-through.
//!
//! # Example (Tokio)
//!
//! ```no_run
//! use obj::asynchronous::AsyncDb;
//! use serde::{Deserialize, Serialize};
//! use tempfile::tempdir;
//!
//! #[derive(Debug, Serialize, Deserialize, obj::Document)]
//! struct Order {
//!     customer_id: u64,
//!     total_cents: u64,
//! }
//!
//! #[tokio::main(flavor = "multi_thread")]
//! async fn main() -> obj::Result<()> {
//!     let dir = tempdir()?;
//!     let path = dir.path().join("app.obj");
//!     let db = AsyncDb::open(path).await?;
//!     let id = db.insert(Order { customer_id: 1, total_cents: 999 }).await?;
//!     let back: Option<Order> = db.get(id).await?;
//!     assert!(back.is_some());
//!     Ok(())
//! }
//! ```
//!
//! The same `AsyncDb` value drives identically from Tokio, smol, or any
//! other async runtime.

mod collection;
mod db;
mod query;

pub use collection::AsyncCollection;
pub use db::AsyncDb;
pub use query::AsyncQuery;
