//! Exercises the opt-in `tracing` feature.
//!
//! Builds only when the feature is enabled; the entire file is
//! gated at the module level so `cargo test` without
//! `--features tracing` skips it cleanly.
//!
//! The test is a wire-up smoke check, not a tracing-semantics
//! freeze: it attaches a capturing [`tracing_subscriber::Layer`]
//! that records every `on_new_span` event into a shared
//! `Vec<String>`, exercises the four obj-rs entry points
//! (`Db::open` → `Db::memory`, `Db::transaction`, `Db::query::fetch`,
//! `Db::integrity_check`), and asserts the expected span names
//! show up. The exact event count + ordering is intentionally NOT
//! frozen — future revisions are free to add events inside a span
//! without breaking the test.

#![cfg(feature = "tracing")]
#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use obj::{Db, Document};
use serde::{Deserialize, Serialize};
use tracing::subscriber::with_default;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
use tracing_subscriber::Registry;

/// Toy document with no indexes — keeps the test focused on the
/// tracing wire-up rather than the index machinery.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tick {
    /// The single payload byte. Not captured by any span.
    n: u64,
}

impl Document for Tick {
    const COLLECTION: &'static str = "ticks";
    const VERSION: u32 = 1;
}

impl obj::Schema for Tick {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("n", obj::DynamicSchema::U64)])
    }
}

/// `Layer` that captures every span's `name()` into a shared `Vec`.
/// The `Arc<Mutex<Vec<String>>>` is the minimal invariant boundary —
/// the only mutation is `push`, and every lock acquisition is checked
/// for poisoning.
struct CaptureLayer {
    spans: Arc<Mutex<Vec<String>>>,
}

impl<S> Layer<S> for CaptureLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::Id,
        _ctx: Context<'_, S>,
    ) {
        if let Ok(mut guard) = self.spans.lock() {
            guard.push(attrs.metadata().name().to_owned());
        }
    }
}

#[test]
fn tracing_feature_emits_expected_spans() -> obj::Result<()> {
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let layer = CaptureLayer {
        spans: Arc::clone(&captured),
    };
    let subscriber = Registry::default().with(layer);

    with_default(subscriber, || -> obj::Result<()> {
        let db = Db::memory()?;
        db.transaction(|tx| {
            tx.collection::<Tick>()?.insert(Tick { n: 1 })?;
            Ok(())
        })?;
        let _ = db.query::<Tick>().fetch()?;
        let _ = db.integrity_check()?;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("trace.obj");
        let _ondisk = Db::open(&path)?;
        Ok(())
    })?;

    let names = captured.lock().expect("capture mutex not poisoned");
    let snapshot: Vec<&str> = names.iter().map(String::as_str).collect();
    for expected in [
        "db.open",
        "db.transaction",
        "db.read_transaction",
        "db.integrity_check",
        "query.execute",
    ] {
        assert!(
            snapshot.contains(&expected),
            "expected span `{expected}` in capture; saw {snapshot:?}",
        );
    }
    Ok(())
}
