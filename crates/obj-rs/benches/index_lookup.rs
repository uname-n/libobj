//! Index `find_unique` warm-cache lookup benchmark.
//!
//! Populates a database with 100 000 documents carrying one `Unique`
//! index over a randomly-shuffled `customer_id` field, then measures
//! `Collection::find_unique` median latency under criterion.
//!
//! The gate baseline is a 400 ns warm index lookup; the bench prints
//! the measured median side-by-side with that baseline. This bench is
//! informational only — nothing fails the run on a miss.
//!
//! Populate runs in 1 000-document batches per
//! [`obj::Db::transaction`] call to stay inside the default
//! `wal_size_limit` (64 MiB).
//!
//! Run with:
//! ```text
//! cargo bench --bench index_lookup
//! ```

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use obj::{Document, IndexSpec};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

mod common;
use common::{fresh_db, measure_median_ns, populate_batched, report_gate, Gate};

const POPULATE_COUNT: usize = 100_000;
/// Documents per populate transaction.
const POPULATE_BATCH: usize = 1_000;
/// Gate baseline for a warm index lookup.
const TARGET_NS: u128 = 400;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdxDoc {
    /// Randomly-shuffled value used as the unique-index key.
    customer_id: u64,
    /// Filler payload so the document record looks realistic
    /// (~ several dozen bytes encoded).
    label: String,
}

impl Document for IdxDoc {
    const COLLECTION: &'static str = "idx_docs";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_customer_id", "customer_id").expect("unique spec")]
    }
}

impl obj::Schema for IdxDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("label", obj::DynamicSchema::String),
        ])
    }
}

fn bench_find_unique(c: &mut Criterion) {
    let bench_db = fresh_db("idx_lookup");
    let db = &bench_db.db;

    // Randomly-shuffled customer-id values, used both as the inserted
    // unique-index keys and as the keys the bench queries (so each
    // lookup hits a document it knows is present).
    let mut populate_rng = ChaCha8Rng::seed_from_u64(0xF1AD_4D17_C0EC_0001);
    let mut ids: Vec<u64> = (1..=(POPULATE_COUNT as u64)).collect();
    ids.shuffle(&mut populate_rng);
    let _: u8 = populate_rng.random();

    let _ = populate_batched(db, POPULATE_COUNT, POPULATE_BATCH, |i| {
        let cid = ids[i];
        IdxDoc {
            customer_id: cid,
            label: format!("c-{cid}"),
        }
    });

    let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE_F00D_0BBE_5004);

    let mut group = c.benchmark_group("index_lookup");
    group.throughput(Throughput::Elements(1));
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    group.bench_function("find_unique_warm", |b| {
        b.iter(|| {
            let idx = rng.random_range(0..ids.len());
            let cid = ids[idx];
            let got: Option<IdxDoc> = db
                .find_unique::<IdxDoc>("by_customer_id", cid)
                .expect("find_unique");
            debug_assert!(got.is_some(), "populated id must be present");
            std::hint::black_box(got);
        });
    });
    group.finish();

    let median = measure_median_ns(200, || {
        let idx = rng.random_range(0..ids.len());
        let cid = ids[idx];
        let got: Option<IdxDoc> = db
            .find_unique::<IdxDoc>("by_customer_id", cid)
            .expect("find_unique");
        std::hint::black_box(got);
    });
    report_gate(
        "index_lookup",
        Gate::Ceiling {
            measured_ns: median,
            baseline_ns: TARGET_NS,
            mult: 1,
        },
        false,
    );
}

criterion_group!(benches, bench_find_unique);
criterion_main!(benches);
