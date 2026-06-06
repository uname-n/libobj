//! `Db::all::<T>()` collection-scan benchmark.
//!
//! A 100 000-document collection scan on Linux x86-64 `NVMe` storage
//! takes ~38 ms. The bench's `TARGET_MS` constant is the **gate
//! baseline** — the wall time the bench is expected to clear, with
//! headroom for run-to-run noise. It is not the aspirational number,
//! and improvements move the gate baseline downward over time.
//!
//! Where the raw-range bench measures `BTree::range` throughput, this
//! bench measures end-to-end `Db::all::<T>()` — which adds
//! postcard-decode + per-doc header validation on top of the
//! B+tree walk. The two numbers together let a reviewer see whether
//! a perf regression is in the storage layer or the codec layer.
//!
//! # Fail-gate
//!
//! Default `cargo bench` is informational — the bench prints
//! `TARGET_MS` alongside criterion's median but never exits non-zero
//! on a miss. Set `OBJ_BENCH_ENFORCE=1` to turn the informational
//! target into a hard failure (e.g. in your own CI).
//!
//! # Populate strategy
//!
//! Populate runs in 1 000-doc batches per `Db::transaction` —
//! single mega-txns would blow past the default `wal_size_limit`
//! (64 MiB) on a 100k populate.
//!
//! Run with:
//! ```text
//! cargo bench --bench collection_scan
//! ```

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use obj::{Db, Document};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{fresh_db, measure_median_ns, populate_batched, report_gate, Gate};

const POPULATE_COUNT: usize = 100_000;
const POPULATE_BATCH: usize = 1_000;
/// Gate baseline for end-to-end `Db::all::<T>()`. Set above the
/// observed median with headroom for run-to-run noise; improvements
/// drive it down, regressions push it past the gate.
const TARGET_MS: u128 = 800;
/// Independent-median sample count for the `OBJ_BENCH_ENFORCE` gate.
const GATE_SAMPLES: usize = 11;
/// Approximate postcard-encoded payload size. The bench is meant to
/// match a "~512 bytes" target document; a `Vec<u8>` of this length
/// plus a small handful of scalar fields is close enough for the
/// measurement.
const PAYLOAD_BYTES: usize = 480;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanDoc {
    /// Sequential index so the bench can assert it observed every
    /// doc on the way out.
    seq: u64,
    /// Filler payload — random bytes so postcard does no length-
    /// encoding tricks.
    payload: Vec<u8>,
}

impl Document for ScanDoc {
    const COLLECTION: &'static str = "scan_docs";
    const VERSION: u32 = 1;
}

impl obj::Schema for ScanDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("seq", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

/// Populate `db` with `POPULATE_COUNT` documents in `POPULATE_BATCH`-sized
/// transactions via the shared batch loop. The RNG is seeded once and
/// drained in insertion order so payloads stay deterministic.
fn populate(db: &Db) {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5CAD_DEC0_DEAD_BEEF);
    let _ = populate_batched(db, POPULATE_COUNT, POPULATE_BATCH, |i| {
        let mut payload = vec![0u8; PAYLOAD_BYTES];
        rng.fill(&mut payload[..]);
        ScanDoc {
            seq: i as u64,
            payload,
        }
    });
}

fn bench_collection_scan(c: &mut Criterion) {
    let bench_db = fresh_db("scan");
    let db = &bench_db.db;
    populate(db);

    let mut group = c.benchmark_group("collection_scan");
    group.throughput(Throughput::Elements(POPULATE_COUNT as u64));
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("db_all_100k_x_512b", |b| {
        b.iter(|| {
            let docs: Vec<ScanDoc> = db.all::<ScanDoc>().expect("all");
            assert_eq!(docs.len(), POPULATE_COUNT, "scan must return every doc");
            black_box(docs);
        });
    });
    group.finish();

    let median_ns = measure_median_ns(GATE_SAMPLES, || {
        let docs: Vec<ScanDoc> = db.all::<ScanDoc>().expect("all");
        assert_eq!(docs.len(), POPULATE_COUNT, "gate scan must return every doc");
        black_box(docs);
    });
    report_gate(
        "collection_scan",
        Gate::Ceiling {
            measured_ns: median_ns,
            baseline_ns: TARGET_MS * 1_000_000,
            mult: 1,
        },
        true,
    );
}

criterion_group!(benches, bench_collection_scan);
criterion_main!(benches);
