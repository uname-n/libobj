//! Borrowed-decode vs owned-decode B+tree scan comparison.
//!
//! This bench measures the read-path win from the borrowed B+tree leaf
//! decode (`BorrowedLeaf` / `read_leaf_slot`) introduced for the
//! snapshot range scan. It contrasts two scans over the *same* 100 000-
//! entry tree:
//!
//! - `owned_decode_iter` — [`BTree::iter`] / [`RangeIter`], which
//!   materializes every leaf into a `DecodedNode` (2N inner `Vec<u8>`
//!   allocations per leaf: one per key, one per value).
//! - `borrowed_decode_snapshot` — [`BTree::range_via_snapshot`] /
//!   [`SnapshotRangeIter`], which holds the leaf's page handle and
//!   reads each entry as borrowed `&[u8]` slices, allocating only the
//!   single `(Vec, Vec)` pair it yields.
//!
//! Both iterators yield identical owned `(Vec<u8>, Vec<u8>)` items, so
//! the per-yield allocation is constant across the two; the difference
//! is the per-leaf decode allocation that the borrowed path eliminates.
//! On a 100k-entry tree the borrowed path saves ~2N `Vec<u8>`
//! allocations per leaf scanned (the entries that are decoded-but-
//! never-touched in the owned path are the bulk of the win).
//!
//! # Running
//!
//! ```text
//! cargo bench --bench btree_borrowed_scan
//! ```
//!
//! Informational by default. Wall time is hardware-dependent; the
//! point of interest is the *ratio* between the two functions on the
//! same machine, not the absolute numbers.

#![forbid(unsafe_code)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use obj_core::btree::BTree;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const NUM_ENTRIES: usize = 100_000;
const KEY_BYTES: usize = 8;
const VALUE_BYTES: usize = 512;
const SEED: u64 = 0xDEAD_BEEF;

fn populate_tree() -> (Pager<FileHandle>, BTree<FileHandle>) {
    let mut pager = Pager::<FileHandle>::memory(Config::default()).expect("pager init");
    let mut tree = BTree::<FileHandle>::empty(&mut pager).expect("empty tree");
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut inserted = 0usize;
    let mut value = [0u8; VALUE_BYTES];
    while inserted < NUM_ENTRIES {
        let mut key = [0u8; KEY_BYTES];
        rng.fill(&mut key);
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

fn bench_scans(c: &mut Criterion) {
    let (mut pager, tree) = populate_tree();
    let root = tree.root();
    let mut group = c.benchmark_group("btree_borrowed_scan");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(12));

    group.bench_function("owned_decode_iter", |b| {
        b.iter(|| {
            let iter = tree.iter(&mut pager).expect("iter");
            let mut count = 0usize;
            for step in iter {
                let (k, v) = step.expect("iter step");
                black_box(&k);
                black_box(&v);
                count += 1;
            }
            assert_eq!(count, NUM_ENTRIES, "owned scan must return every entry");
            black_box(count);
        });
    });

    group.bench_function("borrowed_decode_snapshot", |b| {
        b.iter(|| {
            let snapshot = pager.reader_snapshot().expect("snapshot");
            let iter =
                BTree::<FileHandle>::range_via_snapshot(&pager, &snapshot, root, ..).expect("range");
            let mut count = 0usize;
            for step in iter {
                let (k, v) = step.expect("iter step");
                black_box(&k);
                black_box(&v);
                count += 1;
            }
            assert_eq!(
                count, NUM_ENTRIES,
                "borrowed snapshot scan must return every entry"
            );
            black_box(count);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_scans);
criterion_main!(benches);
