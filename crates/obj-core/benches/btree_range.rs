//! Range-scan throughput over a B+tree
//! populated with 100 000 random `(key=8 bytes, value=512 bytes)`
//! pairs.
//!
//! The bench's
//! `TARGET_MS` constant is the **gate baseline** — the wall time
//! the bench is expected to clear, with
//! enough headroom to absorb run-to-run noise. Improvements move the
//! gate baseline downward over time. The benchmark prints both the
//! measured median and the gate so a glance at the output tells
//! the developer whether the commit moves the bar.
//!
//! # Fail-gate
//!
//! By default the bench is informational — it prints the
//! target alongside criterion's median and never fails the run.
//! Set `OBJ_BENCH_ENFORCE=1` to turn the informational target into a
//! hard failure (e.g. in your own CI).
//!
//! The bench is intentionally allocation-tolerant: the iterator
//! yields owned `(Vec<u8>, Vec<u8>)` pairs because the
//! `pager.read_page` lifetime is per-step. MVCC reader snapshots
//! may revisit this trade-off; until then the benchmark is the
//! reference measurement for the current design.
//!
//! # Running
//!
//! ```text
//! cargo bench --bench btree_range
//! ```
//!
//! Performance is hardware-dependent; expect wall-time drift across
//! platforms (Apple Silicon laptops vs Linux `NVMe`), so the baseline
//! carries headroom. A miss on faster or slower hardware is a platform
//! difference, not a regression.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use obj_core::btree::BTree;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_ENTRIES: usize = 100_000;
const KEY_BYTES: usize = 8;
const VALUE_BYTES: usize = 512;
/// Seed for the random key/value generator. Fixed so that the
/// bench's tree shape is reproducible across runs.
const SEED: u64 = 0xDEAD_BEEF;
/// Gate baseline wall time for the 100k-doc scan. Set above the
/// observed median with headroom for run-to-run noise; "we work from
/// here" — improvements drive this number down, regressions push it up
/// past the gate.
const TARGET_MS: u128 = 600;
/// Sample count for the `OBJ_BENCH_ENFORCE` gate. Independent of
/// criterion's own sample count — small enough that the extra
/// measurement does not meaningfully extend bench time, large enough
/// that the median is stable.
const GATE_SAMPLES: usize = 11;

fn populate_tree() -> (Pager<FileHandle>, BTree<FileHandle>) {
    let mut pager = Pager::<FileHandle>::memory(Config::default()).expect("pager init");
    let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty tree");
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut inserted = 0usize;
    let mut value = [0u8; VALUE_BYTES];
    while inserted < NUM_ENTRIES {
        let key = random_key(&mut rng);
        rng.fill(&mut value[..]);
        match tree.insert(&mut pager, &key, &value) {
            Ok(()) => inserted += 1,
            Err(obj_core::Error::BTreeKeyExists) => (),
            Err(e) => panic!("populate: insert failed: {e:?}"),
        }
    }
    pager.commit().expect("commit");
    (pager, tree)
}

fn random_key(rng: &mut ChaCha8Rng) -> [u8; KEY_BYTES] {
    let mut buf = [0u8; KEY_BYTES];
    rng.fill(&mut buf);
    buf
}

fn bench_full_scan(c: &mut Criterion) {
    let (mut pager, tree) = populate_tree();
    let mut group = c.benchmark_group("btree_range");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("full_scan_100k_x_512b", |b| {
        b.iter(|| {
            let iter = tree.iter(&mut pager).expect("iter");
            let mut count = 0usize;
            for step in iter {
                let (k, v) = step.expect("iter step");
                black_box(&k);
                black_box(&v);
                count += 1;
            }
            assert_eq!(count, NUM_ENTRIES, "scan must return every entry");
            black_box(count);
        });
    });
    group.finish();
    let median_ms = measure_median_ms(&mut pager, &tree);
    let status = if median_ms <= TARGET_MS { "PASS" } else { "MISS" };
    eprintln!("[btree_range] median = {median_ms} ms; gate = {TARGET_MS} ms [{status}]");
    assert!(
        !(bench_enforce_enabled() && median_ms > TARGET_MS),
        "btree_range: OBJ_BENCH_ENFORCE=1 and measured median {median_ms} ms > target {TARGET_MS} ms"
    );
}

/// Measure `GATE_SAMPLES` full-scan wall times and return the median
/// in whole milliseconds. The sample-count loop is bounded by
/// `GATE_SAMPLES`.
fn measure_median_ms(pager: &mut Pager<FileHandle>, tree: &BTree<FileHandle>) -> u128 {
    let mut samples_ms: Vec<u128> = Vec::with_capacity(GATE_SAMPLES);
    for _ in 0..GATE_SAMPLES {
        let start = Instant::now();
        let iter = tree.iter(pager).expect("iter");
        let mut count = 0usize;
        for step in iter {
            let (k, v) = step.expect("iter step");
            black_box(&k);
            black_box(&v);
            count += 1;
        }
        let elapsed = start.elapsed();
        assert_eq!(
            count, NUM_ENTRIES,
            "gate scan must return every entry (got {count})"
        );
        samples_ms.push(elapsed.as_millis());
    }
    samples_ms.sort_unstable();
    samples_ms[GATE_SAMPLES / 2]
}

/// Deliberate local copy - obj-core benches cannot import the obj-rs
/// bench harness.
fn bench_enforce_enabled() -> bool {
    std::env::var("OBJ_BENCH_ENFORCE")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
}

criterion_group!(benches, bench_full_scan);
criterion_main!(benches);
