//! Shared bench helpers for the obj-rs benchmark binaries.
//!
//! Each `benches/*.rs` is its own binary crate, so this module is
//! compiled once per consumer; `#![allow(dead_code)]` keeps
//! unused-function warnings quiet on the consumers that use only a
//! subset.
//!
//! # Gate convention
//!
//! A bench measures a median and compares it against a local gate via
//! [`report_gate`], which prints one consistent line tagged `[PASS]`,
//! `[MISS]` (a hard gate over budget), or `[INFO]` (a soft gate over
//! budget). By default the run exits `0` either way — the gates are an
//! informational performance bar, not CI wiring.
//!
//! Set `OBJ_BENCH_ENFORCE=1` to turn a **hard** gate's miss into a
//! panic (non-zero exit) for use in your own CI; soft gates never fail
//! the run. Expect 2–3× wall-time drift across platforms (Apple
//! Silicon laptops vs Linux `NVMe`), so the baselines carry headroom.

// allow: this module is compiled into every bench binary, so any helper unused by a
// given bench would warn; dead_code is expected for a shared, partially-used toolbox.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Instant;

use obj::{Config, Db, Document, Id, Schema, SyncMode};
use tempfile::TempDir;

/// RAII wrapper around a `(TempDir, Db)` pair. Field order matters:
/// `Db` drops before `TempDir` so the file handles release their
/// advisory locks before the directory is removed (Rust drops fields
/// in declaration order).
pub struct BenchDb {
    pub db: Db,
    pub path: PathBuf,
    pub tempdir: TempDir,
}

/// Open a fresh on-disk `Db` inside a tempdir. `name` is the database
/// file basename so concurrent bench runs do not collide. Dropping the
/// returned [`BenchDb`] deletes the underlying file.
///
/// Uses [`Db::open`], i.e. the **durable default config**:
/// `SyncMode::Full` (one WAL sync per committed transaction) and the
/// default 256 KiB / 64-frame page cache. Every bench in this crate
/// measures the engine under those shipped defaults, not under a
/// throughput-tuned config.
///
/// # Panics
///
/// Panics on tempdir creation or open failure — bench setup is not a
/// production path.
#[must_use]
pub fn fresh_db(name: &str) -> BenchDb {
    let tempdir = TempDir::new().expect("tempdir");
    let path = tempdir.path().join(format!("{name}.obj"));
    let db = Db::open(&path).expect("open fresh db");
    BenchDb { db, path, tempdir }
}

/// Open a fresh **in-memory** `Db` configured with [`SyncMode::Off`] —
/// the ephemeral, throughput-tuned scratch config: no persistence, no
/// file locks, and no per-commit durability sync. This is the config
/// the `mem_off` bench measures; the on-disk [`fresh_db`] (durable
/// default `SyncMode::Full`) is what `perf_table` measures. Unlike
/// [`fresh_db`] there is no path or `TempDir`, so this returns a bare
/// [`Db`] rather than a [`BenchDb`].
///
/// # Panics
///
/// Panics if the in-memory open fails — bench setup is not a
/// production path.
#[must_use]
pub fn fresh_mem_db() -> Db {
    let cfg = Config::default().sync_mode(SyncMode::Off);
    Db::memory_with(cfg).expect("open in-memory db")
}

/// Drop the supplied `Db` and re-open the same path, flushing the WAL
/// via the open-time recovery checkpoint so the returned `Db` sees a
/// quiescent file with no in-flight frames. The `TempDir` is carried
/// across so the path survives the drop/reopen cycle.
///
/// # Panics
///
/// Panics if the re-open fails.
#[must_use]
pub fn reopen(bench_db: BenchDb) -> BenchDb {
    let BenchDb { db, path, tempdir } = bench_db;
    drop(db);
    let db = Db::open(&path).expect("re-open db");
    BenchDb { db, path, tempdir }
}

/// `OBJ_BENCH_ENFORCE=1` (or any non-empty, non-`"0"` value) turns a
/// **hard** gate's miss into a panic. Off by default; soft gates ignore
/// it entirely. Absence and an empty value both leave it disabled.
#[must_use]
pub fn bench_enforce_enabled() -> bool {
    std::env::var("OBJ_BENCH_ENFORCE")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
}

/// Median elapsed nanoseconds over `samples` timed runs of `op`.
///
/// # Panics
///
/// Panics if `samples == 0`.
pub fn measure_median_ns<F: FnMut()>(samples: usize, mut op: F) -> u128 {
    assert!(samples > 0, "samples must be > 0");
    let mut elapsed: Vec<u128> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        op();
        elapsed.push(start.elapsed().as_nanos());
    }
    elapsed.sort_unstable();
    elapsed[samples / 2]
}

/// As [`measure_median_ns`] but `setup` runs **untimed** before each
/// timed `op` (e.g. re-opening a `Db` for a cold-cache measurement).
///
/// # Panics
///
/// Panics if `samples == 0`.
pub fn measure_median_ns_with_setup<S, T, F>(samples: usize, mut setup: S, mut op: F) -> u128
where
    S: FnMut() -> T,
    F: FnMut(T),
{
    assert!(samples > 0, "samples must be > 0");
    let mut elapsed: Vec<u128> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let input = setup();
        let start = Instant::now();
        op(input);
        elapsed.push(start.elapsed().as_nanos());
    }
    elapsed.sort_unstable();
    elapsed[samples / 2]
}

/// How a measured number is compared against its local gate.
#[derive(Clone, Copy)]
pub enum Gate {
    /// `measured_ns` must stay at or below `baseline_ns × mult`.
    Ceiling {
        measured_ns: u128,
        baseline_ns: u128,
        mult: u128,
    },
    /// `value` is compared against `target`; `higher_is_better` flips
    /// the direction (throughput / speedup ratios want higher).
    Ratio {
        value: f64,
        target: f64,
        higher_is_better: bool,
    },
}

/// Print one consistent gate line for `label` and return whether it
/// passed. A **hard** gate (`hard == true`) that fails while
/// `OBJ_BENCH_ENFORCE=1` panics (non-zero exit); soft gates only print.
///
/// # Panics
///
/// Panics when `hard` is set, the gate failed, and
/// `OBJ_BENCH_ENFORCE=1` — this is the intended enforce behavior.
pub fn report_gate(label: &str, gate: Gate, hard: bool) -> bool {
    let (pass, line) = match gate {
        Gate::Ceiling {
            measured_ns,
            baseline_ns,
            mult,
        } => {
            let ceiling = baseline_ns.saturating_mul(mult);
            let pass = measured_ns <= ceiling;
            let gate_desc = if mult == 1 {
                fmt_ns(baseline_ns)
            } else {
                format!("{mult}× {} = {}", fmt_ns(baseline_ns), fmt_ns(ceiling))
            };
            let line = format!(
                "[{label}] median = {}; gate = {gate_desc} [{}]",
                fmt_ns(measured_ns),
                status(pass, hard),
            );
            (pass, line)
        }
        Gate::Ratio {
            value,
            target,
            higher_is_better,
        } => {
            let pass = if higher_is_better {
                value >= target
            } else {
                value <= target
            };
            let dir = if higher_is_better { "≥" } else { "≤" };
            let line = format!(
                "[{label}] ratio = {value:.2} (target {dir} {target:.2}) [{}]",
                status(pass, hard),
            );
            (pass, line)
        }
    };
    println!("{line}");
    assert!(
        !(hard && !pass && bench_enforce_enabled()),
        "[{label}] OBJ_BENCH_ENFORCE=1: gate missed — {line}"
    );
    pass
}

/// `PASS` when within budget; `MISS` for an over-budget hard gate;
/// `INFO` for an over-budget soft gate.
fn status(pass: bool, hard: bool) -> &'static str {
    if pass {
        "PASS"
    } else if hard {
        "MISS"
    } else {
        "INFO"
    }
}

/// Human-friendly duration formatting for a nanosecond count, picking
/// ns / µs / ms / s by magnitude.
// allow: precision loss is harmless here — `ns` is a duration for human-readable
// display formatting only, never compared back against an exact integer.
#[allow(clippy::cast_precision_loss)]
fn fmt_ns(ns: u128) -> String {
    let n = ns as f64;
    if n < 1_000.0 {
        format!("{n:.0} ns")
    } else if n < 1_000_000.0 {
        format!("{:.2} µs", n / 1_000.0)
    } else if n < 1_000_000_000.0 {
        format!("{:.2} ms", n / 1_000_000.0)
    } else {
        format!("{:.2} s", n / 1_000_000_000.0)
    }
}

/// Insert `count` documents built by `build`, committing in
/// `batch`-sized transactions — a single mega-transaction would exceed
/// the default 64 MiB `wal_size_limit`. `build(i)` returns the `i`-th
/// document (`i` in `0..count`); the returned `Id`s are in insertion
/// order. Each bench keeps its own document type, index shape, payload
/// size, and RNG seed — only the batch loop is shared.
///
/// # Panics
///
/// Panics if a populate transaction fails.
pub fn populate_batched<T, F>(db: &Db, count: usize, batch: usize, mut build: F) -> Vec<Id>
where
    T: Document + Schema,
    F: FnMut(usize) -> T,
{
    assert!(batch > 0, "batch must be > 0");
    let mut ids: Vec<Id> = Vec::with_capacity(count);
    let mut inserted = 0usize;
    while inserted < count {
        let end = (inserted + batch).min(count);
        let batch_ids = db
            .transaction(|tx| {
                let coll = tx.collection::<T>()?;
                let mut out = Vec::with_capacity(end - inserted);
                for i in inserted..end {
                    out.push(coll.insert(build(i))?);
                }
                Ok(out)
            })
            .expect("populate batch");
        ids.extend(batch_ids);
        inserted = end;
    }
    ids
}
