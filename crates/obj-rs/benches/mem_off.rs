//! 6-row in-memory + `SyncMode::Off` perf-table reproduction harness.
//!
//! Runs each of six operations under criterion and emits a markdown
//! table with the measured medians, the per-row gate baseline, and a
//! notes column. The table lands on stdout AND at
//! `target/criterion/mem_off/mem_off.md` (atomic write via a tempfile
//! + rename).
//!
//! # Config used for these numbers
//!
//! Every row opens its database with
//! `Db::memory_with(Config::default().sync_mode(SyncMode::Off))`, i.e.
//! an **in-memory, ephemeral** pager with **no per-commit durability
//! sync**. This is the relevant config for scratch / test / ephemeral
//! workloads and is meaningfully faster than the durable default —
//! single-insert in particular, which under this config drops the
//! per-commit fsync entirely.
//!
//! This is the deliberate counterpart to `perf_table.rs`, which
//! measures the **durable default config**: on-disk `Db::open`,
//! `SyncMode::Full` (one WAL sync per committed transaction), and the
//! default 256 KiB / 64-frame page cache. The two harnesses share the
//! same `PerfDoc` shape, RNG seeding style, and gate/markdown
//! machinery so the numbers are directly comparable; only the open
//! config differs.
//!
//! # Rows measured (and rows deliberately omitted)
//!
//! Six rows are measured, mirroring `perf_table` minus its two
//! on-disk-only rows:
//!
//! 1. Point read, warm cache
//! 2. Single insert (fresh tx per call)
//! 3. Batch insert, 1 000 docs, one tx
//! 4. Batch insert, 10 000 docs, one tx
//! 5. Index lookup (`find_unique`)
//! 6. Collection scan, 100 000 docs
//!
//! `perf_table`'s **"Point read, cold page"** row is intentionally
//! absent: an in-memory pager has no on-disk page to go cold, and the
//! cold-cache measurement re-opens from a path — there is no path for
//! an ephemeral memory DB. `perf_table`'s **"Concurrent readers"** row
//! is also intentionally absent — out of scope here and can be a
//! follow-up.
//!
//! # Fail-gate
//!
//! Default `cargo bench --bench mem_off` is informational: the table
//! prints and the bench exits 0 regardless of how far the measured
//! medians drift. Set `OBJ_BENCH_ENFORCE=1` to turn the informational
//! target into a hard failure (e.g. in your own CI): after the table
//! emits, every row whose `target_ns` is finite is checked against
//! `≤ 10 × target_ns`; the first miss prints the `target/measured`
//! ratio and exits 1.
//!
//! The 10× allowance covers run-to-run noise plus cross-platform drift
//! (Apple Silicon laptops report 2-3× the Linux `NVMe` numbers, and
//! noisy schedules can drift another 2-3×). Anything beyond that range
//! is a real signal.
//!
//! Each row's `target_ns` is a **gate baseline** — the per-op wall
//! time the bench is expected to clear, with headroom for run-to-run
//! noise. It is not a promise or an aspirational number, and
//! improvements move the gate baseline downward over time.
//!
//! # Reproduction
//!
//! ```text
//! cargo bench --bench mem_off                       # informational
//! OBJ_BENCH_ENFORCE=1 cargo bench --bench mem_off   # gated
//! cargo bench --bench mem_off -- --quick            # quick sanity
//! ```

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use obj::{Db, Document, Id, IndexSpec};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    bench_enforce_enabled, fresh_mem_db, measure_median_ns, measure_median_ns_with_setup,
};

/// Document count for the index-lookup / collection-scan rows.
const POPULATE_COUNT: usize = 100_000;
/// Documents per populate transaction, matching `perf_table`.
const POPULATE_BATCH: usize = 1_000;
/// Document count for the small "warm cache" / "single insert" rows.
const SMALL_POPULATE_COUNT: usize = 1_000;
/// Approximate payload size, matched to a "~512 bytes" reference
/// document.
const PAYLOAD_BYTES: usize = 480;
/// Number of independent samples gathered for each measured row's
/// median.
const GATE_SAMPLES: usize = 11;
/// Order-of-magnitude tolerance: a row's `measured_ns` must stay at or
/// below `10 × target_ns` under `OBJ_BENCH_ENFORCE=1`.
const ORDER_OF_MAGNITUDE_NS: u128 = 10;

/// Bench document with ~512-byte encoded size and a unique index on
/// `customer_id` so the harness can exercise both `Collection::get`
/// and `Db::find_unique` against the same in-memory shape.
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
/// means the operation is reported as a string and is exempt from the
/// order-of-magnitude gate.
#[derive(Debug, Clone)]
struct PerfRow {
    operation: &'static str,
    measured_us: f64,
    target_display: &'static str,
    target_ns: Option<u128>,
    notes: &'static str,
}

/// Outcome of a populate call: every `customer_id` used for inserts
/// (shuffled) and the database-assigned `Id` for each one. `ids[k]`
/// aligns with `customer_ids[k]`.
struct PopulatedDb {
    customer_ids: Vec<u64>,
    ids: Vec<Id>,
}

/// Populate `db` with `count` docs in `POPULATE_BATCH`-sized
/// transactions. The shuffled customer-id sequence means two
/// sequential ids do not share a B-tree leaf neighbourhood — a hot-
/// path lookup walking a uniformly-distributed key set measures the
/// page-walk depth, not the leaf-cache hit rate.
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

/// Row 1 — point read, warm cache. The same `Id` is fetched in a tight
/// loop so the leaf page stays in the pager cache.
fn row_point_read_warm(c: &mut Criterion) -> PerfRow {
    let db = fresh_mem_db();
    let populated = populate(&db, SMALL_POPULATE_COUNT);
    let hot_id = populated.ids[0];
    let mut group = c.benchmark_group("mem_off");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("point_read_warm", |b| {
        b.iter(|| {
            let got: Option<PerfDoc> = db
                .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
                .expect("read");
            black_box(got);
        });
    });
    group.finish();
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let got: Option<PerfDoc> = db
            .read_transaction(|tx| tx.collection::<PerfDoc>()?.get(hot_id))
            .expect("read");
        black_box(got);
    });
    PerfRow {
        operation: "Point read, warm cache",
        measured_us: ns_to_us(median_ns),
        target_display: "~50 µs",
        target_ns: Some(50_000),
        notes: "In-memory pager, zero-copy",
    }
}

/// Row 2 — single insert, fresh transaction per call. Under
/// `SyncMode::Off` there is no per-commit durability sync, so this row
/// measures the in-memory WAL append + commit bookkeeping only — far
/// cheaper than the durable-default single insert in `perf_table`.
fn row_single_insert(c: &mut Criterion) -> PerfRow {
    let db = fresh_mem_db();
    let mut rng = ChaCha8Rng::seed_from_u64(0xA5A5_DECA_5555_1111);
    let mut seq = 0u64;
    let mut group = c.benchmark_group("mem_off");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("single_insert", |b| {
        b.iter(|| {
            seq += 1;
            let mut payload = vec![0u8; PAYLOAD_BYTES];
            rng.fill(&mut payload[..]);
            db.transaction(|tx| {
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
        db.transaction(|tx| {
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
        target_display: "~75 µs",
        target_ns: Some(75_000),
        notes: "In-memory WAL append, no sync",
    }
}

/// Row 3 — batch insert, 1 000 docs in one transaction. Reset to a
/// fresh in-memory DB per criterion iteration so population state does
/// not carry between iters.
fn row_batch_insert_1k(c: &mut Criterion) -> PerfRow {
    let mut group = c.benchmark_group("mem_off");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("batch_insert_1k", |b| {
        b.iter_batched(
            || (fresh_mem_db(), build_payloads(1_000, 0xB1B1)),
            |(db, batch)| {
                db.transaction(|tx| {
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
        || (fresh_mem_db(), build_payloads(1_000, 0xB1B2)),
        |(db, batch)| {
            db.transaction(|tx| {
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
        target_display: "~90 ms",
        target_ns: Some(90_000_000),
        notes: "Single transaction, no sync",
    }
}

/// Row 4 — batch insert, 10 000 docs in one transaction. Heavier;
/// `sample_size(10)` and a wider measurement window.
fn row_batch_insert_10k(c: &mut Criterion) -> PerfRow {
    let mut group = c.benchmark_group("mem_off");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("batch_insert_10k", |b| {
        b.iter_batched(
            || (fresh_mem_db(), build_payloads(10_000, 0xB10B)),
            |(db, batch)| {
                db.transaction(|tx| {
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
        || (fresh_mem_db(), build_payloads(10_000, 0xB10C)),
        |(db, batch)| {
            db.transaction(|tx| {
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
        target_display: "~750 ms",
        target_ns: Some(750_000_000),
        notes: "Single transaction, no sync",
    }
}

/// Row 5 — `find_unique` index lookup against 100k populated docs.
fn row_index_lookup() -> PerfRow {
    let db = fresh_mem_db();
    let populated = populate(&db, POPULATE_COUNT);
    let customer_ids = populated.customer_ids;
    let mut rng = ChaCha8Rng::seed_from_u64(0x1DEC_DEAD_C0DE_4242);
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let idx = rng.random_range(0..customer_ids.len());
        let cid = customer_ids[idx];
        let got: Option<PerfDoc> = db
            .find_unique::<PerfDoc>("by_customer_id", cid)
            .expect("find_unique");
        black_box(got);
    });
    PerfRow {
        operation: "Index lookup",
        measured_us: ns_to_us(median_ns),
        target_display: "~75 µs",
        target_ns: Some(75_000),
        notes: "B-tree traversal",
    }
}

/// Row 6 — full collection scan, 100k docs. No reopen step (an
/// in-memory DB has no path to re-open from); the scan runs against the
/// populated DB directly.
fn row_collection_scan() -> PerfRow {
    let db = fresh_mem_db();
    let _populated = populate(&db, POPULATE_COUNT);
    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let docs: Vec<PerfDoc> = db.all::<PerfDoc>().expect("all");
        debug_assert_eq!(docs.len(), POPULATE_COUNT, "gate scan length");
        black_box(docs);
    });
    PerfRow {
        operation: "Collection scan, 100 000 docs",
        measured_us: ns_to_us(median_ns),
        target_display: "~600 ms",
        target_ns: Some(600_000_000),
        notes: "Sequential read",
    }
}

/// Convert ns → microseconds as `f64`. Bench durations on every row
/// stay well below 2^53 ns so f64 mantissa loss is not a concern.
// allow: precision loss is harmless — bench durations stay well below 2^53 ns, so the
// f64 mantissa holds them exactly; the cast only feeds human-readable table output.
#[allow(clippy::cast_precision_loss)]
fn ns_to_us(ns: u128) -> f64 {
    (ns as f64) / 1_000.0
}

/// Build a `Vec<PerfDoc>` for the batch-insert rows. Pre-generates the
/// payloads outside the criterion measurement loop so the timed path
/// only measures the transaction cost.
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

/// Format the six rows as a GitHub-flavoured markdown table.
fn format_table(rows: &[PerfRow]) -> String {
    let mut out = String::new();
    out.push_str("# obj mem_off perf table (in-memory, SyncMode::Off)\n\n");
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
    out.push_str("\n*In-memory pager, `SyncMode::Off`. Local hardware. Each row's baseline is the wall time the bench is expected to clear; `OBJ_BENCH_ENFORCE=1` enforces a 10× headroom over each baseline.*\n");
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
/// default `target/` location.
fn output_path() -> PathBuf {
    let root =
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    root.join("criterion/mem_off/mem_off.md")
}

/// Run all six rows, emit the table, and (if enforced) gate.
fn bench_mem_off(c: &mut Criterion) {
    let rows: Vec<PerfRow> = vec![
        row_point_read_warm(c),
        row_single_insert(c),
        row_batch_insert_1k(c),
        row_batch_insert_10k(c),
        row_index_lookup(),
        row_collection_scan(),
    ];
    let table = format_table(&rows);
    println!("\n{table}");
    let dst = output_path();
    atomic_write(&dst, &table);
    println!("[mem_off] markdown table written to {}", dst.display());
    if bench_enforce_enabled() {
        enforce_targets(&rows);
    }
}

/// Walk the rows; the first whose measured median exceeds
/// `ORDER_OF_MAGNITUDE_NS × target_ns` triggers a panic with the ratio
/// printed. Rows whose `target_ns` is `None` are skipped.
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

criterion_group!(benches, bench_mem_off);
criterion_main!(benches);
