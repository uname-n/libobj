//! 8-row perf-table reproduction harness.
//!
//! Runs each of the eight operations under criterion and emits a
//! markdown table with the measured medians, the per-row gate
//! baseline, and a notes column. The table lands on stdout AND at
//! `target/criterion/perf_table/perf_table.md` (atomic write via a
//! tempfile + rename).
//!
//! Each row's `target_ns` is a **gate baseline** — the per-op wall
//! time the bench is expected to clear, with headroom for
//! run-to-run noise. It is not the aspirational number, and
//! improvements move the gate baseline downward over time.
//!
//! # Fail-gate
//!
//! Default `cargo bench --bench perf_table` is informational: the
//! table prints and the bench exits 0 regardless of how far the
//! measured medians drift. Set `OBJ_BENCH_ENFORCE=1` to turn the
//! informational target into a hard failure (e.g. in your own CI):
//! after the table emits, every row whose `target_ns` is finite is
//! checked against `≤ 10 × target_ns`; the first miss prints the
//! `target/measured` ratio and exits 1.
//!
//! The 10× allowance covers run-to-run noise plus cross-platform
//! drift (Apple Silicon laptops report 2-3× the Linux `NVMe`
//! numbers, and noisy schedules can drift another 2-3×). Anything
//! beyond that range is a real signal.
//!
//! # Config used for these numbers
//!
//! Every row opens its database with `Db::open` / `fresh_db`, i.e.
//! the **durable default config**: `SyncMode::Full` (one WAL sync
//! per committed transaction) and the default 256 KiB / 64-frame page
//! cache. No throughput-tuned config (larger `cache_size`,
//! `SyncMode::Normal`/`Off`) is used anywhere in this harness.
//!
//! This is why the "Single insert" row is comparatively slow: under
//! `SyncMode::Full` it pays one durability sync per inserted document,
//! whereas the "Batch insert" rows amortize a single sync across the
//! whole transaction. The single-insert number is the cost of one
//! durable commit, not the per-document cost of bulk loading — for
//! bulk loading, batch into one transaction (see the `Config` docs'
//! "Performance tuning" section). Re-tuning the config would change
//! these numbers and is deliberately out of scope here: the gates
//! track the engine under its shipped defaults.
//!
//! # Reproduction
//!
//! ```text
//! cargo bench --bench perf_table                # informational
//! OBJ_BENCH_ENFORCE=1 cargo bench --bench perf_table  # gated
//! cargo bench --bench perf_table -- --quick     # < 5 min sanity
//! ```

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use obj::{Db, Document, Id, IndexSpec};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    bench_enforce_enabled, fresh_db, measure_median_ns, measure_median_ns_with_setup, reopen,
};

/// Document count for the index-lookup / collection-scan / concurrent
/// rows. Matches the populate scale in `index_lookup.rs` /
/// `collection_scan.rs` / `concurrent_reads.rs`.
const POPULATE_COUNT: usize = 100_000;
/// Documents per populate transaction. A 100k single-tx commit
/// exceeds the default `wal_size_limit` (64 MiB).
const POPULATE_BATCH: usize = 1_000;
/// Document count for the small "warm cache" / "cold page" /
/// "single insert" rows. Big enough that a B-tree is in play, small
/// enough that populate finishes in well under a second.
const SMALL_POPULATE_COUNT: usize = 1_000;
/// Approximate payload size, matched to a "~512 bytes" reference
/// document.
const PAYLOAD_BYTES: usize = 480;
/// Concurrent-reader thread count.
const CONCURRENT_THREADS: usize = 8;
/// Number of independent samples gathered for each measured row's
/// median. Independent of criterion's own sample count; small enough
/// that the harness completes in well under 5 minutes total.
const GATE_SAMPLES: usize = 11;
/// Order-of-magnitude tolerance: a row's `measured_ns` must stay at
/// or below `10 × target_ns` under `OBJ_BENCH_ENFORCE=1`.
const ORDER_OF_MAGNITUDE_NS: u128 = 10;

/// Bench document with ~512-byte encoded size and a unique index on
/// `customer_id` so the harness can exercise both `Collection::get`
/// and `Db::find_unique` against the same on-disk shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerfDoc {
    customer_id: u64,
    payload: Vec<u8>,
}

impl Document for PerfDoc {
    const COLLECTION: &'static str = "perf_docs";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_customer_id", "customer_id").expect("unique spec")]
    }
}

impl obj::Schema for PerfDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

/// A single row of the emitted markdown table. `target_ns == None`
/// means the operation is reported as a string (e.g. "linear
/// scaling") and is exempt from the order-of-magnitude gate.
#[derive(Debug, Clone)]
struct PerfRow {
    operation: &'static str,
    measured_us: f64,
    target_display: &'static str,
    target_ns: Option<u128>,
    notes: &'static str,
}

/// Outcome of a populate call: every `customer_id` used for inserts
/// (shuffled) and the database-assigned `Id` for each one. The id
/// vector aligns with `customer_ids` by index — `ids[k]` is the
/// `Id` returned by the insert call whose payload carried
/// `customer_ids[k]`.
struct PopulatedDb {
    customer_ids: Vec<u64>,
    ids: Vec<Id>,
}

/// Populate `db` with `count` docs in `POPULATE_BATCH`-sized
/// transactions. The shuffled customer-id sequence means two
/// sequential ids do not share a B-tree leaf neighbourhood — a hot-
/// path lookup walking a uniformly-distributed key set measures
/// the page-walk depth, not the leaf-cache hit rate.
fn populate(db: &Db, count: usize) -> PopulatedDb {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE_FACE_1234_5678);
    let mut customer_ids: Vec<u64> = (1..=(count as u64)).collect();
    customer_ids.shuffle(&mut rng);
    let mut ids: Vec<Id> = Vec::with_capacity(count);
    let mut inserted = 0usize;
    while inserted < count {
        let batch_end = (inserted + POPULATE_BATCH).min(count);
        let slice: Vec<u64> = customer_ids[inserted..batch_end].to_vec();
        let batch_ids: Vec<Id> = db
            .transaction(|tx| {
                let coll = tx.collection::<PerfDoc>()?;
                let mut out = Vec::with_capacity(slice.len());
                for cid in &slice {
                    let mut payload = vec![0u8; PAYLOAD_BYTES];
                    rng.fill(&mut payload[..]);
                    let id = coll.insert(PerfDoc {
                        customer_id: *cid,
                        payload,
                    })?;
                    out.push(id);
                }
                Ok(out)
            })
            .expect("populate batch");
        ids.extend(batch_ids);
        inserted = batch_end;
    }
    PopulatedDb { customer_ids, ids }
}

/// Row 1 — point read, warm cache. The same `Id` is fetched in a
/// tight loop so the leaf page stays in the pager cache.
fn row_point_read_warm(c: &mut Criterion) -> PerfRow {
    let bench_db = fresh_db("perf_warm");
    let populated = populate(&bench_db.db, SMALL_POPULATE_COUNT);
    let hot_id = populated.ids[0];
    let mut group = c.benchmark_group("perf_table");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("point_read_warm", |b| {
        b.iter(|| {
            let got: Option<PerfDoc> = bench_db
                .db
                .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
                .expect("read");
            black_box(got);
        });
    });
    group.finish();
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let got: Option<PerfDoc> = bench_db
            .db
            .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
            .expect("read");
        black_box(got);
    });
    PerfRow {
        operation: "Point read, warm cache",
        measured_us: ns_to_us(median_ns),
        target_display: "~100 µs",
        target_ns: Some(100_000),
        notes: "Memory-mapped, zero-copy",
    }
}

/// Row 2 — point read, cold page. Per-iteration re-open forces the
/// pager cache to start from empty; the OS page cache may still be
/// warm but the obj-side pager state is cold each time.
fn row_point_read_cold(c: &mut Criterion) -> PerfRow {
    let bench_db = fresh_db("perf_cold");
    let populated = populate(&bench_db.db, SMALL_POPULATE_COUNT);
    let hot_id = populated.ids[0];
    let path = bench_db.path.clone();
    let tempdir = bench_db.tempdir;
    drop(bench_db.db);
    let mut group = c.benchmark_group("perf_table");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("point_read_cold", |b| {
        b.iter_batched(
            || Db::open(&path).expect("re-open cold"),
            |db| {
                let got: Option<PerfDoc> = db
                    .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
                    .expect("read");
                black_box(got);
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
    let median_ns = measure_median_ns_with_setup(
        GATE_SAMPLES,
        || Db::open(&path).expect("re-open cold"),
        |db| {
            let got: Option<PerfDoc> = db
                .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
                .expect("read");
            black_box(got);
        },
    );
    drop(tempdir);
    PerfRow {
        operation: "Point read, cold page",
        measured_us: ns_to_us(median_ns),
        target_display: "~100 µs",
        target_ns: Some(100_000),
        notes: "Single page from NVMe",
    }
}

/// Row 3 — single insert, fresh transaction per call. Measures the
/// round-trip including WAL append + commit fsync. Under the durable
/// default config (`SyncMode::Full`) this row is dominated by that one
/// per-transaction sync; it is the cost of one durable commit, not the
/// amortized per-document cost of a bulk load (see rows 4/5).
fn row_single_insert(c: &mut Criterion) -> PerfRow {
    let bench_db = fresh_db("perf_insert1");
    let mut rng = ChaCha8Rng::seed_from_u64(0xA5A5_DECA_5555_1111);
    let mut seq = 0u64;
    let mut group = c.benchmark_group("perf_table");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("single_insert", |b| {
        b.iter(|| {
            seq += 1;
            let mut payload = vec![0u8; PAYLOAD_BYTES];
            rng.fill(&mut payload[..]);
            bench_db
                .db
                .transaction(|tx| {
                    tx.collection::<PerfDoc>()?.insert(PerfDoc {
                        customer_id: seq,
                        payload: payload.clone(),
                    })
                })
                .expect("insert");
        });
    });
    group.finish();
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        seq += 1;
        let mut payload = vec![0u8; PAYLOAD_BYTES];
        rng.fill(&mut payload[..]);
        bench_db
            .db
            .transaction(|tx| {
                tx.collection::<PerfDoc>()?.insert(PerfDoc {
                    customer_id: seq,
                    payload: payload.clone(),
                })
            })
            .expect("insert");
    });
    PerfRow {
        operation: "Single insert",
        measured_us: ns_to_us(median_ns),
        target_display: "~750 µs",
        target_ns: Some(750_000),
        notes: "WAL append + one Full sync",
    }
}

/// Row 4 — batch insert, 1 000 docs in one transaction. Measured by
/// resetting to a fresh DB per criterion iteration (the population
/// state mustn't carry between iters or the encoded sizes drift).
fn row_batch_insert_1k(c: &mut Criterion) -> PerfRow {
    let mut group = c.benchmark_group("perf_table");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("batch_insert_1k", |b| {
        b.iter_batched(
            || (fresh_db("perf_batch1k"), build_payloads(1_000, 0xB1B1)),
            |(bench_db, batch)| {
                bench_db
                    .db
                    .transaction(|tx| {
                        let coll = tx.collection::<PerfDoc>()?;
                        for doc in batch {
                            let _ = coll.insert(doc)?;
                        }
                        Ok(())
                    })
                    .expect("batch insert");
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
    let median_ns = measure_median_ns_with_setup(
        GATE_SAMPLES,
        || (fresh_db("perf_batch1k"), build_payloads(1_000, 0xB1B2)),
        |(bench_db, batch)| {
            bench_db
                .db
                .transaction(|tx| {
                    let coll = tx.collection::<PerfDoc>()?;
                    for doc in batch {
                        let _ = coll.insert(doc)?;
                    }
                    Ok(())
                })
                .expect("batch insert");
        },
    );
    PerfRow {
        operation: "Batch insert, 1 000 docs, one tx",
        measured_us: ns_to_us(median_ns),
        target_display: "~110 ms",
        target_ns: Some(110_000_000),
        notes: "Single transaction, one Full sync amortized",
    }
}

/// Row 5 — batch insert, 10 000 docs in one transaction. Heavier;
/// `sample_size(10)` per the AC.
fn row_batch_insert_10k(c: &mut Criterion) -> PerfRow {
    let mut group = c.benchmark_group("perf_table");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("batch_insert_10k", |b| {
        b.iter_batched(
            || (fresh_db("perf_batch10k"), build_payloads(10_000, 0xB10B)),
            |(bench_db, batch)| {
                bench_db
                    .db
                    .transaction(|tx| {
                        let coll = tx.collection::<PerfDoc>()?;
                        for doc in batch {
                            let _ = coll.insert(doc)?;
                        }
                        Ok(())
                    })
                    .expect("batch insert");
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
    let median_ns = measure_median_ns_with_setup(
        3,
        || (fresh_db("perf_batch10k"), build_payloads(10_000, 0xB10C)),
        |(bench_db, batch)| {
            bench_db
                .db
                .transaction(|tx| {
                    let coll = tx.collection::<PerfDoc>()?;
                    for doc in batch {
                        let _ = coll.insert(doc)?;
                    }
                    Ok(())
                })
                .expect("batch insert");
        },
    );
    PerfRow {
        operation: "Batch insert, 10 000 docs, one tx",
        measured_us: ns_to_us(median_ns),
        target_display: "~1.3 s",
        target_ns: Some(1_300_000_000),
        notes: "Single transaction, one Full sync amortized",
    }
}

/// Row 6 — `find_unique` index lookup against 100k populated docs.
fn row_index_lookup() -> PerfRow {
    let bench_db = fresh_db("perf_idx");
    let populated = populate(&bench_db.db, POPULATE_COUNT);
    let customer_ids = populated.customer_ids;
    let mut rng = ChaCha8Rng::seed_from_u64(0x1DEC_DEAD_C0DE_4242);
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let idx = rng.random_range(0..customer_ids.len());
        let cid = customer_ids[idx];
        let got: Option<PerfDoc> = bench_db
            .db
            .find_unique::<PerfDoc>("by_customer_id", cid)
            .expect("find_unique");
        black_box(got);
    });
    PerfRow {
        operation: "Index lookup",
        measured_us: ns_to_us(median_ns),
        target_display: "~250 µs",
        target_ns: Some(250_000),
        notes: "B-tree traversal",
    }
}

/// Row 7 — full collection scan, 100k docs.
fn row_collection_scan() -> PerfRow {
    let bench_db = fresh_db("perf_scan");
    let _populated = populate(&bench_db.db, POPULATE_COUNT);
    let bench_db = reopen(bench_db);
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let docs: Vec<PerfDoc> = bench_db.db.all::<PerfDoc>().expect("all");
        debug_assert_eq!(docs.len(), POPULATE_COUNT, "gate scan length");
        black_box(docs);
    });
    PerfRow {
        operation: "Collection scan, 100 000 docs",
        measured_us: ns_to_us(median_ns),
        target_display: "~750 ms",
        target_ns: Some(750_000_000),
        notes: "Sequential read",
    }
}

/// Row 8 — 8 concurrent readers, each doing a tight `iter_all`-style
/// `db.all` loop. Reports per-thread throughput as a ratio to the
/// single-thread baseline; the table's `measured_us` carries the
/// 8-thread per-op median for the gate.
fn row_concurrent_readers() -> PerfRow {
    let bench_db = fresh_db("perf_conc");
    let _populated = populate(&bench_db.db, POPULATE_COUNT);
    let bench_db = reopen(bench_db);
    let db = Arc::new(bench_db.db);
    let (per_thread_ns, ratio_pct) = measure_concurrent(&db);
    drop(db);
    drop(bench_db.tempdir);
    let pass = if ratio_pct >= 80 { "PASS" } else { "INFO" };
    println!(
        "[perf_table] concurrent_readers: scaling ratio = {}.{:02}× (target ≥ 0.80) [{pass}]",
        ratio_pct / 100,
        ratio_pct % 100,
    );
    PerfRow {
        operation: "Concurrent readers, 8 threads",
        measured_us: ns_to_us(per_thread_ns),
        target_display: "linear scaling",
        target_ns: None,
        notes: "No reader contention",
    }
}

/// Run a single-thread baseline, then an 8-thread parallel run, both
/// for a short fixed window. Returns the 8-thread per-thread median
/// scan time (ns) and the scaling ratio in hundredths.
fn measure_concurrent(db: &Arc<Db>) -> (u128, u128) {
    let one = measure_concurrent_pass(db, 1);
    let eight = measure_concurrent_pass(db, CONCURRENT_THREADS);
    let ratio_hundredths = if one == 0 {
        0
    } else {
        (eight * 100) / (one * (CONCURRENT_THREADS as u128))
    };
    (eight, ratio_hundredths)
}

/// Median over `GATE_SAMPLES` passes: spawn `n` threads, each does
/// one `db.all` scan, return the median wall-clock per pass (ns).
fn measure_concurrent_pass(db: &Arc<Db>, n: usize) -> u128 {
    let mut samples: Vec<u128> = Vec::with_capacity(GATE_SAMPLES);
    for _ in 0..GATE_SAMPLES {
        let barrier = Arc::new(Barrier::new(n + 1));
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let db = Arc::clone(db);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let docs: Vec<PerfDoc> = db.all::<PerfDoc>().expect("all");
                black_box(docs);
            }));
        }
        barrier.wait();
        let start = Instant::now();
        for h in handles {
            h.join().expect("join");
        }
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[GATE_SAMPLES / 2]
}

/// Convert ns → microseconds as `f64`. Bench durations on every
/// row stay well below 2^53 ns so f64 mantissa loss is not a
/// concern; the cast is documented and clippy-suppressed.
// allow: precision loss is harmless — bench durations stay well below 2^53 ns, so the
// f64 mantissa holds them exactly; the cast only feeds human-readable table output.
#[allow(clippy::cast_precision_loss)]
fn ns_to_us(ns: u128) -> f64 {
    (ns as f64) / 1_000.0
}

/// Build a `Vec<PerfDoc>` for the batch-insert rows. Pre-generates
/// the payloads outside the criterion measurement loop so the timed
/// path only measures the transaction cost.
fn build_payloads(n: usize, seed: u64) -> Vec<PerfDoc> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut payload = vec![0u8; PAYLOAD_BYTES];
        rng.fill(&mut payload[..]);
        out.push(PerfDoc {
            customer_id: (i + 1) as u64,
            payload,
        });
    }
    out
}

/// Format the eight rows as a GitHub-flavoured markdown table.
fn format_table(rows: &[PerfRow]) -> String {
    let mut out = String::new();
    out.push_str("# obj perf table\n\n");
    out.push_str("| Operation | Measured (median) | Gate baseline | Notes |\n");
    out.push_str("|---|---|---|---|\n");
    for r in rows {
        let measured = format_measured(r.measured_us);
        writeln!(
            out,
            "| {} | {} | {} | {} |",
            r.operation, measured, r.target_display, r.notes,
        )
        .expect("write to String");
    }
    out.push_str("\n*Local hardware. Each row's baseline is the wall time the bench is expected to clear; `OBJ_BENCH_ENFORCE=1` enforces a 10× headroom over each baseline.*\n");
    out
}

/// Pick a human-friendly unit for the measured median: ns < 1 µs,
/// µs < 1 ms, ms otherwise.
fn format_measured(us: f64) -> String {
    if us < 1.0 {
        format!("{:.0} ns", us * 1_000.0)
    } else if us < 1_000.0 {
        format!("{us:.2} µs")
    } else {
        format!("{:.2} ms", us / 1_000.0)
    }
}

/// Atomically write `contents` to `dst`: write to `dst.tmp` then
/// rename. Every `Result` is propagated; bench setup is allowed to
/// panic on filesystem failure with a clear diagnostic.
fn atomic_write(dst: &PathBuf, contents: &str) {
    let tmp = dst.with_extension("md.tmp");
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).expect("create_dir_all");
    }
    fs::write(&tmp, contents).expect("write tmp");
    fs::rename(&tmp, dst).expect("rename");
}

/// Resolve the markdown output path. `CARGO_TARGET_DIR` overrides the
/// default `target/` location (Cargo runs the bench with the package
/// directory as CWD; that is where the relative path resolves).
fn output_path() -> PathBuf {
    let root =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    root.join("criterion/perf_table/perf_table.md")
}

/// Run all eight rows, emit the table, and (if enforced) gate.
fn bench_perf_table(c: &mut Criterion) {
    let rows: Vec<PerfRow> = vec![
        row_point_read_warm(c),
        row_point_read_cold(c),
        row_single_insert(c),
        row_batch_insert_1k(c),
        row_batch_insert_10k(c),
        row_index_lookup(),
        row_collection_scan(),
        row_concurrent_readers(),
    ];
    let table = format_table(&rows);
    println!("\n{table}");
    let dst = output_path();
    atomic_write(&dst, &table);
    println!("[perf_table] markdown table written to {}", dst.display());
    if bench_enforce_enabled() {
        enforce_targets(&rows);
    }
}

/// Walk the rows; the first whose measured median exceeds
/// `ORDER_OF_MAGNITUDE_NS × target_ns` triggers a panic with the
/// ratio printed. Rows whose `target_ns` is `None` (e.g. the
/// concurrent-readers "linear scaling" row) are skipped — the
/// per-row print already surfaces the scaling ratio.
fn enforce_targets(rows: &[PerfRow]) {
    for r in rows {
        let Some(target_ns) = r.target_ns else {
            continue;
        };
        // allow: truncation/sign-loss are safe — `measured_us` is a positive bench
        // median well within u128 range; flooring to whole ns is the intended rounding.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let measured_ns = (r.measured_us * 1_000.0) as u128;
        let ceiling = target_ns.saturating_mul(ORDER_OF_MAGNITUDE_NS);
        if measured_ns > ceiling {
            let ratio_hundredths = (measured_ns * 100) / target_ns.max(1);
            panic!(
                "OBJ_BENCH_ENFORCE=1: row {:?} measured {} ns > 10× target ({} ns); ratio = {}.{:02}×",
                r.operation,
                measured_ns,
                ceiling,
                ratio_hundredths / 100,
                ratio_hundredths % 100,
            );
        }
    }
}

criterion_group!(benches, bench_perf_table);
criterion_main!(benches);
