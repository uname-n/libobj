//! Regression test: a bare-name full scan
//! ([`Db::dump_raw`], which backs `db.all` / `query.fetch` /
//! `ReadTxn::iter_all`) must be snapshot-isolated against concurrent
//! writers.
//!
//! Before the fix, `DumpIter::refill_local` walked the LIVE primary
//! B-tree (`BTree::range`) instead of the dump txn's pinned
//! `ReaderSnapshot`. Concurrent writers mutate the tree mid-iteration
//! (node splits/merges, page reuse), so the scan reads a freed /
//! interior page and surfaces a spurious `CorruptionError` /
//! `InvalidArgumentError` — ~all reads fail under load. The fix routes
//! the local scan through `BTree::range_via_snapshot` against the
//! txn's pinned snapshot, exactly like the attached path and
//! the point reads (`get_via_snapshot`).
//!
//! This test spawns writer threads inserting into a collection on a
//! shared `Arc<Db>` while the main thread loops `db.dump_raw(coll, 0)`
//! draining the FULL scan many times. It asserts ZERO per-scan errors
//! and that each completed scan's count is monotonic (snapshot
//! isolation: a scan never sees fewer docs than an earlier scan, since
//! the writers only insert). On the pre-fix `tree.range` code this
//! fails almost immediately with a corruption/invalid-argument error;
//! with `range_via_snapshot` it completes 0 errors over all N scans.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use obj::{Config, Db, Document};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Number of concurrent writer threads.
const N_WRITERS: u64 = 4;
/// Documents each writer thread inserts.
const INSERTS_PER_WRITER: u64 = 500;
/// Minimum number of full scans the reader must complete (N >= 200).
const N_SCANS: u64 = 200;

/// Test document — a small fixed-shape payload. Field order is stable
/// so the postcard layout the dump path decodes the header of is
/// deterministic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ScanDoc {
    /// Which writer produced this doc (diagnostic only).
    writer: u64,
    /// Per-writer monotonic sequence (diagnostic only).
    seq: u64,
    /// A little payload so docs span more than one B-tree leaf.
    payload: Vec<u8>,
}

impl Document for ScanDoc {
    const COLLECTION: &'static str = "scan";
    const VERSION: u32 = 1;
}

impl obj::Schema for ScanDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("writer", obj::DynamicSchema::U64),
            ("seq", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

#[test]
fn full_scan_is_snapshot_isolated_against_concurrent_writes() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("scan.obj");
    let config = Config::default()
        .cross_process_lock(false)
        .busy_timeout(Duration::from_mins(2));
    let db = Arc::new(Db::open_with(&path, config).expect("open"));

    db.insert(ScanDoc {
        writer: u64::MAX,
        seq: 0,
        payload: vec![0u8; 32],
    })
    .expect("seed insert");

    let scan_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let total_completed_scans = Arc::new(AtomicU64::new(0));

    thread::scope(|scope| {
        for w in 0..N_WRITERS {
            let db = Arc::clone(&db);
            scope.spawn(move || writer_loop(&db, w));
        }
        run_scan_loop(&db, &scan_errors, &total_completed_scans);
    });

    let errors = scan_errors.lock().expect("scan_errors lock");
    assert!(
        errors.is_empty(),
        "full scan surfaced {} error(s) under concurrent writes (expected 0); first few: {:?}",
        errors.len(),
        &errors[..errors.len().min(5)],
    );
    assert!(
        total_completed_scans.load(Ordering::SeqCst) >= N_SCANS,
        "reader did not complete the required {N_SCANS} scans",
    );

    let expected = N_WRITERS * INSERTS_PER_WRITER + 1;
    let final_count = drain_full_scan(&db).expect("final scan must be error-free");
    assert_eq!(
        final_count, expected,
        "post-quiesce full scan count {final_count} != expected {expected}",
    );
}

/// Each writer inserts `INSERTS_PER_WRITER` docs into the shared
/// collection, one per committed transaction (maximising tree churn —
/// splits/merges/page reuse — which is exactly what broke the live
/// scan).
fn writer_loop(db: &Arc<Db>, writer: u64) {
    let mut seq: u64 = 0;
    while seq < INSERTS_PER_WRITER {
        seq = seq.saturating_add(1);
        let fill = u8::try_from((writer ^ seq) & 0xFF).unwrap_or(0);
        let doc = ScanDoc {
            writer,
            seq,
            payload: vec![fill; 48],
        };
        match db.insert(doc) {
            Ok(_) => {}
            Err(obj::Error::Busy { .. }) => {
                thread::yield_now();
                seq = seq.saturating_sub(1);
            }
            Err(e) => panic!("writer {writer} insert failed: {e:?}"),
        }
    }
}

/// Reader loop: drain the full bare-name scan `N_SCANS` times while
/// the writers churn the tree. Records any per-step error, and asserts
/// per-scan count monotonicity (inserts only ⇒ a later snapshot can
/// only see >= an earlier one).
fn run_scan_loop(
    db: &Arc<Db>,
    scan_errors: &Arc<Mutex<Vec<String>>>,
    total_completed_scans: &Arc<AtomicU64>,
) {
    let mut last_count: u64 = 0;
    let mut scans: u64 = 0;
    while scans < N_SCANS {
        scans = scans.saturating_add(1);
        match drain_full_scan(db) {
            Ok(count) => {
                if count < last_count {
                    if let Ok(mut log) = scan_errors.lock() {
                        log.push(format!(
                            "non-monotonic snapshot: scan {scans} saw {count} < prior {last_count}",
                        ));
                    }
                }
                last_count = count;
                total_completed_scans.fetch_add(1, Ordering::SeqCst);
            }
            Err(e) => {
                if let Ok(mut log) = scan_errors.lock() {
                    log.push(format!("scan {scans}: {e:?}"));
                }
            }
        }
    }
}

/// Drain the full bare-name primary-tree scan (`limit == 0` means no
/// cap) and return the document count. The FIRST per-step error
/// short-circuits to `Err`.
fn drain_full_scan(db: &Db) -> obj::Result<u64> {
    let iter = db.dump_raw(ScanDoc::COLLECTION, 0)?;
    let mut count: u64 = 0;
    for step in iter {
        step?;
        count = count.saturating_add(1);
    }
    Ok(count)
}
