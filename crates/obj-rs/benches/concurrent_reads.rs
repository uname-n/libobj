//! Concurrent-reader scaling benchmark.
//!
//! Populates a database with 100 000 documents (~512-byte payloads)
//! then runs a tight `Db::read_transaction(|tx| tx.collection::<Doc>()
//! ?.get(random_id))` loop with N ∈ {1, 2, 4, 8} reader threads.
//! For each N the bench measures aggregate reads/sec and prints the
//! 8-thread scaling ratio relative to the 1-thread baseline.
//!
//! `Scaling ratio = throughput(8) / (8 * throughput(1))`.  A ratio
//! ≥ 0.80 indicates near-linear scaling (within 80% of the target
//! of fully-parallel readers).  Informational only — the bench does
//! not panic on a miss.  On a development laptop with ≥ 8 cores
//! expect a ratio close to 1.0 once the page cache is warm; on a
//! 2-core machine the ratio is bounded by the hardware.
//!
//! Set `OBJ_BENCH_ENFORCE=1` to turn the informational target into a
//! hard failure (e.g. in your own CI).
//!
//! Run with:
//! ```text
//! cargo bench --bench concurrent_reads
//! ```

#![forbid(unsafe_code)]

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use obj::{Db, Document, Id};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{fresh_db, populate_batched, report_gate, Gate};

const POPULATE_COUNT: usize = 100_000;
const PAYLOAD_BYTES: usize = 400;
/// Documents per populate transaction. 100k documents in a single
/// transaction blows past the default `wal_size_limit` (64 MiB);
/// splitting into 1 000-doc batches keeps each commit's WAL footprint
/// at ~500 KiB of payload plus B-tree overhead — comfortably inside
/// the limit. Batches of 1 000 also align with the default
/// `checkpoint_threshold` (1 000 committed frames), so the populate
/// loop drains the WAL naturally and ends with the file in a
/// quiescent state.
const POPULATE_BATCH: usize = 1_000;
const WARMUP_SECS: u64 = 5;
const MEASURE_SECS: u64 = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchDoc {
    customer_id: u64,
    payload: Vec<u8>,
}

impl Document for BenchDoc {
    const COLLECTION: &'static str = "bench_docs";
    const VERSION: u32 = 1;
}

impl obj::Schema for BenchDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

fn populate(db: &Db) -> Vec<Id> {
    let mut rng = ChaCha8Rng::seed_from_u64(0x0B12_2026_C0EC_040D);
    populate_batched(db, POPULATE_COUNT, POPULATE_BATCH, |i| {
        let mut payload = vec![0u8; PAYLOAD_BYTES];
        rng.fill(payload.as_mut_slice());
        BenchDoc {
            customer_id: i as u64,
            payload,
        }
    })
}

/// Run N reader threads doing tight `Db::read_transaction(...)`
/// loops for `duration`.  Returns aggregate reads completed.
fn run_readers(db: &Arc<Db>, ids: &Arc<Vec<Id>>, n: usize, duration: Duration) -> u64 {
    let barrier = Arc::new(Barrier::new(n + 1));
    let mut handles = Vec::with_capacity(n);
    for t in 0..n {
        let db = Arc::clone(db);
        let ids = Arc::clone(ids);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || -> u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(0xBE5C_0000 ^ (t as u64));
            barrier.wait();
            let start = Instant::now();
            let mut reads = 0u64;
            while start.elapsed() < duration {
                let idx = rng.random_range(0..ids.len());
                let id = ids[idx];
                let _: Option<BenchDoc> = db
                    .read_transaction(|tx| tx.collection::<BenchDoc>()?.get(id))
                    .expect("read");
                reads += 1;
            }
            reads
        }));
    }
    barrier.wait();
    handles.into_iter().map(|h| h.join().expect("join")).sum()
}

/// Run a single read-thread scaling configuration and return the
/// reads-per-second.
fn measure(db: &Arc<Db>, ids: &Arc<Vec<Id>>, n: usize) -> f64 {
    let _ = run_readers(db, ids, n, Duration::from_secs(WARMUP_SECS));
    let measure_duration = Duration::from_secs(MEASURE_SECS);
    let reads = run_readers(db, ids, n, measure_duration);
    // allow: precision loss is harmless — `reads` is a throughput count converted to
    // f64 only to compute an approximate reads/sec rate, not an exact value.
    #[allow(clippy::cast_precision_loss)]
    let reads_f = reads as f64;
    reads_f / measure_duration.as_secs_f64()
}

fn bench_concurrent_reads(_c: &mut Criterion) {
    let bench_db = fresh_db("concurrent_reads");
    let ids = Arc::new(populate(&bench_db.db));
    let db = Arc::new(bench_db.db);

    let mut throughputs: Vec<(usize, f64)> = Vec::new();
    for &n in &[1usize, 2, 4, 8] {
        let tput = measure(&db, &ids, n);
        throughputs.push((n, tput));
        println!("[concurrent_reads] N={n:>2} threads: {tput:.0} reads/sec");
    }

    let throughput_at = |n: usize| {
        throughputs
            .iter()
            .find(|(k, _)| *k == n)
            .map_or(0.0, |(_, v)| *v)
    };
    let one = throughput_at(1);
    let eight = throughput_at(8);
    if one > 0.0 {
        let ratio = eight / (8.0 * one);
        report_gate(
            "concurrent_reads",
            Gate::Ratio {
                value: ratio,
                target: 0.80,
                higher_is_better: true,
            },
            false,
        );
    }
}

criterion_group!(benches, bench_concurrent_reads);
criterion_main!(benches);
