//! Lean, human-facing perf-iteration bench.
//!
//! This is the FAST ITERATION loop — *not* a gate bench and *not*
//! part of CI. It has no `OBJ_BENCH_ENFORCE` machinery, no target
//! constants, and emits no markdown. The only signal is criterion's
//! own baseline comparison: save a baseline, make your change, and
//! re-run against that baseline to see the delta.
//!
//! ```text
//! cargo bench -p obj-rs --bench quick -- --save-baseline before
//! # ...make your change...
//! cargo bench -p obj-rs --bench quick -- --baseline before
//! ```
//!
//! It populates ~256 docs once, then measures three short operations
//! — `point_read_warm`, `index_lookup`, and `batch_insert_64` — all
//! under one criterion group so a single run reports every line. Each
//! window is deliberately small so the whole run finishes in a few
//! seconds.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use obj::{Db, Document, Id, IndexSpec};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::fresh_db;

/// Approximate per-doc payload size (matches the perf-table reference
/// document's ~512-byte encoded shape).
const PAYLOAD_BYTES: usize = 480;
/// Docs populated once before the read / lookup benches run. Big
/// enough that a B-tree is in play, small enough to populate in well
/// under a second.
const POPULATE_COUNT: usize = 256;
/// Docs per populate transaction. A single large commit would push
/// past the default WAL size limit; batching keeps each commit small.
const POPULATE_BATCH: usize = 1_000;
/// Docs inserted per `batch_insert_64` iteration.
const BATCH_INSERT_COUNT: usize = 64;
/// Fixed RNG seed for the populate payloads.
const POPULATE_SEED: u64 = 0x0B_0078_5EED_0001;
/// Fixed RNG seed for the index-lookup key sampling.
const LOOKUP_SEED: u64 = 0x0B_0078_5EED_0002;
/// Fixed RNG seed for the batch-insert payload builder.
const BATCH_SEED: u64 = 0x0B_0078_5EED_0003;

/// Quick-bench document: a unique `key` index plus a ~480-byte
/// payload so reads and index lookups exercise a realistic on-disk
/// shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuickDoc {
    key: u64,
    payload: Vec<u8>,
}

impl Document for QuickDoc {
    const COLLECTION: &'static str = "quick_docs";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_key", "key").expect("unique spec")]
    }
}

impl obj::Schema for QuickDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("key", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

/// Populate `db` with `count` docs in `POPULATE_BATCH`-sized
/// transactions. Returns the inserted keys (so `index_lookup` can
/// sample a real one) and the database-assigned ids (so
/// `point_read_warm` can fix a hot id); `keys[k]` and `ids[k]` refer
/// to the same document.
fn populate(db: &Db, count: usize) -> (Vec<u64>, Vec<Id>) {
    let mut rng = ChaCha8Rng::seed_from_u64(POPULATE_SEED);
    let mut keys: Vec<u64> = Vec::with_capacity(count);
    let mut ids: Vec<Id> = Vec::with_capacity(count);
    let mut inserted = 0usize;
    while inserted < count {
        let batch_end = (inserted + POPULATE_BATCH).min(count);
        let batch_ids: Vec<Id> = db
            .transaction(|tx| {
                let coll = tx.collection::<QuickDoc>()?;
                let mut out = Vec::with_capacity(batch_end - inserted);
                for k in inserted..batch_end {
                    let key = (k as u64) + 1;
                    let mut payload = vec![0u8; PAYLOAD_BYTES];
                    rng.fill(&mut payload[..]);
                    out.push(coll.insert(QuickDoc { key, payload })?);
                    keys.push(key);
                }
                Ok(out)
            })
            .expect("populate batch");
        ids.extend(batch_ids);
        inserted = batch_end;
    }
    (keys, ids)
}

/// Build `n` `QuickDoc`s outside the timed routine so the
/// `batch_insert_64` measurement only captures the transaction cost.
fn build_payloads(n: usize, seed: u64) -> Vec<QuickDoc> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut payload = vec![0u8; PAYLOAD_BYTES];
        rng.fill(&mut payload[..]);
        out.push(QuickDoc {
            key: (i as u64) + 1,
            payload,
        });
    }
    out
}

/// Point read of a fixed hot id — the leaf page stays in the pager
/// cache across iterations.
fn point_read_warm(c: &mut Criterion, db: &Db, hot_id: Id) {
    let mut group = c.benchmark_group("quick");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(700));
    group.bench_function("point_read_warm", |b| {
        b.iter(|| {
            let got: Option<QuickDoc> = db
                .read_transaction(|tx| tx.collection::<QuickDoc>()?.get(hot_id))
                .expect("read");
            black_box(got);
        });
    });
    group.finish();
}

/// `find_unique` lookup against a random existing key each iteration.
fn index_lookup(c: &mut Criterion, db: &Db, keys: &[u64]) {
    let mut rng = ChaCha8Rng::seed_from_u64(LOOKUP_SEED);
    let mut group = c.benchmark_group("quick");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(700));
    group.bench_function("index_lookup", |b| {
        b.iter(|| {
            let key = keys[rng.random_range(0..keys.len())];
            let got: Option<QuickDoc> = db
                .find_unique::<QuickDoc>("by_key", key)
                .expect("find_unique");
            black_box(got);
        });
    });
    group.finish();
}

/// Insert `BATCH_INSERT_COUNT` docs in one transaction against a
/// fresh DB per iteration so populate state never carries between
/// iters.
fn batch_insert_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.bench_function("batch_insert_64", |b| {
        b.iter_batched(
            || {
                (
                    fresh_db("quick_batch"),
                    build_payloads(BATCH_INSERT_COUNT, BATCH_SEED),
                )
            },
            |(bench_db, batch)| {
                bench_db
                    .db
                    .transaction(|tx| {
                        let coll = tx.collection::<QuickDoc>()?;
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
}

/// Populate once, then run the three quick-iteration benches.
fn bench_quick(c: &mut Criterion) {
    let bench_db = fresh_db("quick");
    let (keys, ids) = populate(&bench_db.db, POPULATE_COUNT);
    let hot_id = ids[0];
    point_read_warm(c, &bench_db.db, hot_id);
    index_lookup(c, &bench_db.db, &keys);
    batch_insert_64(c);
}

criterion_group!(benches, bench_quick);
criterion_main!(benches);
