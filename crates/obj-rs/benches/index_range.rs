//! Index-range vs full-scan benchmark.
//!
//! Populates a database with 100 000 documents carrying one
//! `Standard` index over a `u64` timestamp-like field, then measures
//! both:
//!
//! - `Collection::index_range(low..high)` — the supposedly fast
//!   selective scan that walks only the index B-tree pages whose
//!   keys fall in `[low, high)`.
//! - `Collection::all().filter(..)` — the materialise-and-filter
//!   equivalent that walks the entire primary B-tree.
//!
//! The bench prints both medians plus the `full_scan / index_range`
//! ratio. The acceptance target is a ≥ 10× speedup; the bench reports
//! that ratio against the target but never fails on a miss — it is
//! informational only.
//!
//! Populate runs in 1 000-doc batches per `Db::transaction` to stay
//! inside the default `wal_size_limit` (64 MiB).
//!
//! Run with:
//! ```text
//! cargo bench --bench index_range
//! ```

use std::ops::Bound;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use obj::{Db, Document, IndexSpec};
use obj_core::codec::Dynamic;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{fresh_db, populate_batched, report_gate, Gate};

const POPULATE_COUNT: usize = 100_000;
const POPULATE_BATCH: usize = 1_000;
/// Approximate number of docs the range scan should return. Picked
/// at 100 so the index walk is dominated by leaf reads rather than
/// per-element overhead.
const TARGET_RANGE_DOCS: usize = 100;
/// The factor by which `index_range` should beat `Collection::all()
/// .filter(...)`.
const TARGET_RATIO: f64 = 10.0;
/// `placed_at` field's value range — the bench draws timestamp values
/// uniformly within `[0, RANGE_MAX)`.
const RANGE_MAX: u64 = 1_000_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RangeDoc {
    /// Timestamp-like field — uniformly distributed in `[0, RANGE_MAX)`.
    placed_at: u64,
    /// Filler so the encoded record has realistic shape.
    label: String,
}

impl Document for RangeDoc {
    const COLLECTION: &'static str = "range_docs";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::standard("by_placed_at", "placed_at").expect("standard spec")]
    }
}

impl obj::Schema for RangeDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("placed_at", obj::DynamicSchema::U64),
            ("label", obj::DynamicSchema::String),
        ])
    }
}

fn populate(db: &Db) {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FF_EE61_4434_4444);
    populate_batched(db, POPULATE_COUNT, POPULATE_BATCH, |_| {
        let placed_at = rng.random_range(0..RANGE_MAX);
        RangeDoc {
            placed_at,
            label: format!("p-{placed_at}"),
        }
    });
}

/// Run `index_range(low..high)` once and return the number of docs it
/// yielded (so the optimiser cannot elide the work).
fn run_index_range(db: &Db, low: u64, high: u64) -> usize {
    db.read_transaction(|tx| {
        let coll = tx.collection::<RangeDoc>()?;
        let iter = coll.index_range(
            "by_placed_at",
            (
                Bound::Included(Dynamic::U64(low)),
                Bound::Excluded(Dynamic::U64(high)),
            ),
        )?;
        let mut n = 0usize;
        for step in iter {
            let _ = step?;
            n += 1;
        }
        Ok(n)
    })
    .expect("index_range read txn")
}

/// Same shape as `run_index_range` but uses the new streaming
/// [`obj::Collection::iter_range`] API. The
/// per-iteration work is identical (drain the iterator, count rows);
/// the difference is the memory profile (no upfront materialisation)
/// and the latency-to-first-row (see [`measure_first_row_ns`]).
fn run_iter_range(db: &Db, low: u64, high: u64) -> usize {
    db.read_transaction(|tx| {
        let coll = tx.collection::<RangeDoc>()?;
        let iter = coll.iter_range(
            "by_placed_at",
            (
                Bound::Included(Dynamic::U64(low)),
                Bound::Excluded(Dynamic::U64(high)),
            ),
        )?;
        let mut n = 0usize;
        for step in iter {
            let _ = step?;
            n += 1;
        }
        Ok(n)
    })
    .expect("iter_range read txn")
}

/// Latency-to-first-row probe for the eager `index_range`
/// path: opens the iterator (which internally pre-decodes every
/// matching `T`), then measures the wall time until the first
/// `next()` returns. The construction itself dominates — by the
/// time the `Iterator` exists, all the work has been done.
fn first_row_index_range_ns(db: &Db, low: u64, high: u64) -> u128 {
    db.read_transaction(|tx| {
        let coll = tx.collection::<RangeDoc>()?;
        let start = Instant::now();
        let mut iter = coll.index_range(
            "by_placed_at",
            (
                Bound::Included(Dynamic::U64(low)),
                Bound::Excluded(Dynamic::U64(high)),
            ),
        )?;
        let _first = iter.next().expect("at least one row").expect("step");
        Ok(start.elapsed().as_nanos())
    })
    .expect("index_range first-row read txn")
}

/// Latency-to-first-row probe for the streaming
/// [`obj::Collection::iter_range`] path. Construction runs a
/// chunk-sized index walk + one primary `get`; subsequent rows pay
/// for one `get` each. On wide ranges this number should be
/// orders-of-magnitude smaller than `first_row_index_range_ns`.
fn first_row_iter_range_ns(db: &Db, low: u64, high: u64) -> u128 {
    db.read_transaction(|tx| {
        let coll = tx.collection::<RangeDoc>()?;
        let start = Instant::now();
        let mut iter = coll.iter_range(
            "by_placed_at",
            (
                Bound::Included(Dynamic::U64(low)),
                Bound::Excluded(Dynamic::U64(high)),
            ),
        )?;
        let _first = iter.next().expect("at least one row").expect("step");
        Ok(start.elapsed().as_nanos())
    })
    .expect("iter_range first-row read txn")
}

/// Run `Collection::all().filter(...)` once and return the number of
/// matching docs.
fn run_full_scan(db: &Db, low: u64, high: u64) -> usize {
    db.read_transaction(|tx| {
        let coll = tx.collection::<RangeDoc>()?;
        let docs = coll.all()?;
        let n = docs
            .into_iter()
            .filter(|(_id, d)| d.placed_at >= low && d.placed_at < high)
            .count();
        Ok(n)
    })
    .expect("full_scan read txn")
}

/// Pick `(low, high)` so the half-open range covers roughly
/// `TARGET_RANGE_DOCS` documents. With 100k docs uniformly in
/// `[0, RANGE_MAX)`, that's a window of `RANGE_MAX *
/// TARGET_RANGE_DOCS / POPULATE_COUNT`.
fn pick_range(rng: &mut ChaCha8Rng) -> (u64, u64) {
    let window = (RANGE_MAX
        .checked_mul(TARGET_RANGE_DOCS as u64)
        .expect("range overflow"))
        / (POPULATE_COUNT as u64);
    let low = rng.random_range(0..(RANGE_MAX - window));
    (low, low + window)
}

fn bench_index_range(c: &mut Criterion) {
    let bench_db = fresh_db("idx_range");
    let db = &bench_db.db;
    populate(db);
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF_CAFE_2026_0524);

    for _ in 0..8u32 {
        let (low, high) = pick_range(&mut rng);
        let _ = run_index_range(db, low, high);
        let _ = run_full_scan(db, low, high);
    }

    run_criterion_group(c, db, &mut rng);
    print_ratio(db, &mut rng);
    print_first_row_ratio(db, &mut rng);
}

/// Run criterion's `bench_function` calls in a group. Split out so
/// `bench_index_range` stays under the 60-line ceiling.
///
/// Adds a third row, `iter_range_stream`, that drives the new
/// streaming iterator over the same window. Its
/// throughput should be within noise of `index_range_seek` — both
/// pay for the same B-tree walk + per-row `get`. The win is memory
/// and latency-to-first-row, not aggregate throughput; the latter
/// is measured separately by `print_first_row_ratio` below.
fn run_criterion_group(c: &mut Criterion, db: &Db, rng: &mut ChaCha8Rng) {
    let mut group = c.benchmark_group("index_range");
    group.throughput(Throughput::Elements(TARGET_RANGE_DOCS as u64));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);
    group.bench_function("index_range_seek", |b| {
        b.iter(|| {
            let (low, high) = pick_range(rng);
            let n = run_index_range(db, low, high);
            std::hint::black_box(n);
        });
    });
    group.bench_function("iter_range_stream", |b| {
        b.iter(|| {
            let (low, high) = pick_range(rng);
            let n = run_iter_range(db, low, high);
            std::hint::black_box(n);
        });
    });
    group.bench_function("full_scan_filter", |b| {
        b.iter(|| {
            let (low, high) = pick_range(rng);
            let n = run_full_scan(db, low, high);
            std::hint::black_box(n);
        });
    });
    group.finish();
}

/// Side-by-side median printout. Reuses a single, deterministic
/// sequence of `(low, high)` for both timings so neither variant
/// gets handed a lucky workload.
fn print_ratio(db: &Db, rng: &mut ChaCha8Rng) {
    let mut range_ns: Vec<u128> = Vec::with_capacity(20);
    let mut scan_ns: Vec<u128> = Vec::with_capacity(20);
    for _ in 0..20u32 {
        let (low, high) = pick_range(rng);
        let t0 = Instant::now();
        let n_r = run_index_range(db, low, high);
        let dt_r = t0.elapsed().as_nanos();
        std::hint::black_box(n_r);
        let t1 = Instant::now();
        let n_s = run_full_scan(db, low, high);
        let dt_s = t1.elapsed().as_nanos();
        std::hint::black_box(n_s);
        range_ns.push(dt_r);
        scan_ns.push(dt_s);
    }
    range_ns.sort_unstable();
    scan_ns.sort_unstable();
    let range_median = range_ns[range_ns.len() / 2];
    let scan_median = scan_ns[scan_ns.len() / 2];
    // allow: precision loss is harmless — the medians are ns durations cast to f64
    // only to print an informational full-scan/index-range speedup ratio.
    #[allow(clippy::cast_precision_loss)]
    let ratio = (scan_median as f64) / (range_median.max(1) as f64);
    report_gate(
        "index_range",
        Gate::Ratio {
            value: ratio,
            target: TARGET_RATIO,
            higher_is_better: true,
        },
        false,
    );
}

/// Side-by-side latency-to-first-row print for `index_range`
/// (eager) vs `iter_range` (streaming) over a
/// WIDE range covering ~10 % of the populated docs. With 100k docs
/// that's ~10 000 matches; the eager path must decode all 10k
/// before its first `next()` can yield, while the streaming path
/// yields after one chunk refill (~256 keys + 1 `T` decode).
///
/// Informational only — the absolute numbers vary by host. The
/// streaming/eager ratio is the meaningful signal; it should be
/// well above 1× on any host.
fn print_first_row_ratio(db: &Db, rng: &mut ChaCha8Rng) {
    let wide = RANGE_MAX / 10;
    let mut eager_ns: Vec<u128> = Vec::with_capacity(20);
    let mut stream_ns: Vec<u128> = Vec::with_capacity(20);
    for _ in 0..20u32 {
        let low = rng.random_range(0..(RANGE_MAX - wide));
        let high = low + wide;
        eager_ns.push(first_row_index_range_ns(db, low, high));
        stream_ns.push(first_row_iter_range_ns(db, low, high));
    }
    eager_ns.sort_unstable();
    stream_ns.sort_unstable();
    let eager = eager_ns[eager_ns.len() / 2];
    let stream = stream_ns[stream_ns.len() / 2];
    // allow: precision loss is harmless — the medians are ns durations cast to f64
    // only to print an informational eager/streaming latency ratio.
    #[allow(clippy::cast_precision_loss)]
    let ratio = (eager as f64) / (stream.max(1) as f64);
    println!(
        "[iter_range] first-row median: eager (index_range) = {eager} ns;  \
         streaming (iter_range) = {stream} ns;  \
         eager/streaming = {ratio:.2}x  \
         (informational only)"
    );
}

criterion_group!(benches, bench_index_range);
criterion_main!(benches);
