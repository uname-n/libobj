//! Read-mode full-scan snapshot isolation for `Collection::all` and
//! `Db::iter_all` / `Db::all`.
//!
//! `read_transaction` and `Db::iter_all` are documented as
//! snapshot-isolated: every yielded document is consistent with the
//! snapshot pinned at construction, regardless of what a concurrent
//! writer commits afterwards. Before the fix (incomplete #9), the two
//! PRIMARY full-scan enumerators still walked the LIVE primary B-tree:
//!
//!  * `Collection::all()` Read arm → `snapshot_scan_via_btree` →
//!    `collect_raw_rows` did `tree.range(pager, ..)` (the live walk),
//!    consulting the snapshot only for the decode schema source.
//!  * `IterAll::refill` opened `BTree::open` + `tree.range(&mut pager, ..)`
//!    (live) across every batch refill.
//!
//! A concurrent writer that frees a primary-tree page which the
//! freelist later recycles makes the live read of the recycled page
//! surface a spurious `Some(Err(Error::Corruption { page_id: 0 }))`
//! mid-scan — the precise anomaly the snapshot pin exists to hide — and
//! lets post-snapshot rows leak so `all()` and `count_all()` disagree
//! on the same handle. After the fix both routes walk
//! `BTree::range_via_snapshot`, so the scan stays frozen at the pinned
//! LSN and agrees with `count_all()`.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use obj::{Config, Db, Document, Id};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Rows in the never-mutated baseline set the reader pins. 300 > the
/// `ITER_ALL_BATCH = 256` refill size, so `iter_all` MUST refill at
/// least once AFTER the concurrent writer commits.
const BASELINE: u64 = 300;

/// Test document. `wave == 0` marks a baseline row; the concurrent
/// writer only ever commits rows with `wave >= 1`, so any `wave != 0`
/// a pinned reader observes is a snapshot-isolation leak. The payload
/// makes each doc span enough bytes that the baseline occupies several
/// B-tree leaves (so a writer's frees/recycles actually touch pages the
/// reader's snapshot references).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ScanDoc {
    /// `0` for baseline rows, `>= 1` for churn rows.
    wave: u64,
    /// Per-wave sequence (diagnostic only).
    seq: u64,
    /// Padding so docs span more than one leaf.
    payload: Vec<u8>,
}

impl Document for ScanDoc {
    const COLLECTION: &'static str = "scan";
    const VERSION: u32 = 1;
}

impl obj::Schema for ScanDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("wave", obj::DynamicSchema::U64),
            ("seq", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

/// Hermetic file-backed `Db` — the MVCC / WAL surface only exists for a
/// file pager (a memory pager's snapshot read falls through to the live
/// cache, so the bug can't be exercised there).
fn fresh_db() -> (Arc<Db>, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("read-full-scan.obj");
    let config = Config::default()
        .cross_process_lock(false)
        .busy_timeout(Duration::from_mins(2));
    let db = Arc::new(Db::open_with(&path, config).expect("open"));
    (db, dir)
}

/// Seed the `BASELINE` never-mutated rows, one per committed txn.
fn seed_baseline(db: &Db) {
    for seq in 0..BASELINE {
        let fill = u8::try_from(seq & 0xFF).unwrap_or(0);
        db.insert(ScanDoc {
            wave: 0,
            seq,
            payload: vec![fill; 48],
        })
        .expect("seed insert");
    }
}

/// Insert `n` churn rows (`wave` tag), returning their ids.
fn insert_wave(db: &Db, wave: u64, n: u64) -> Vec<Id> {
    let mut ids = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
    for seq in 0..n {
        let fill = u8::try_from((wave ^ seq) & 0xFF).unwrap_or(0);
        let id = db
            .insert(ScanDoc {
                wave,
                seq,
                payload: vec![fill; 48],
            })
            .expect("churn insert");
        ids.push(id);
    }
    ids
}

/// One churn cycle: insert a wave, delete it (freeing primary-tree
/// pages), then insert another wave (recycling the freed pages with new
/// content) and delete it too. This is the page free+recycle the pinned
/// reader must not observe.
fn churn_once(db: &Db, wave: u64) {
    let first = insert_wave(db, wave, 200);
    for id in first {
        db.delete::<ScanDoc>(id).expect("churn delete");
    }
    let second = insert_wave(db, wave.saturating_add(1), 200);
    for id in second {
        db.delete::<ScanDoc>(id).expect("churn delete");
    }
}

/// Read-txn `all()` and `count_all()` on the SAME pinned handle must
/// both stay frozen at the `BASELINE` snapshot — returning exactly the
/// baseline rows and agreeing with each other — while a concurrent
/// writer commits a full insert/delete/recycle churn. A leak surfaces
/// as a `wave != 0` row, a count mismatch, or a spurious corruption.
#[test]
fn read_collection_all_and_count_frozen_under_concurrent_churn() {
    let (db, _dir) = fresh_db();
    seed_baseline(&db);

    let (snap_tx, snap_rx) = mpsc::channel::<()>();
    let (commit_tx, commit_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let db_r = Arc::clone(&db);
        let db_w = Arc::clone(&db);

        s.spawn(move || {
            db_r.read_transaction(|tx| {
                let coll = tx.collection::<ScanDoc>()?;
                snap_tx.send(()).expect("snap_tx");
                commit_rx.recv().expect("commit_rx");
                assert_frozen_at_baseline(&coll)
            })
            .expect("read_transaction must not surface Error::Corruption");
        });

        s.spawn(move || {
            snap_rx.recv().expect("snap_rx");
            churn_once(&db_w, 1);
            commit_tx.send(()).expect("commit_tx");
        });
    });
}

/// Assert a pinned Read-mode handle sees exactly the baseline: `all()`
/// yields `BASELINE` rows all tagged `wave == 0`, `count_all()` equals
/// `BASELINE`, and the two agree.
fn assert_frozen_at_baseline(coll: &obj::Collection<'_, ScanDoc>) -> obj::Result<()> {
    let rows = coll.all()?;
    assert_eq!(
        u64::try_from(rows.len()).unwrap_or(u64::MAX),
        BASELINE,
        "Collection::all leaked/lost rows vs the pinned snapshot \
         (#2 — all() walked the live B-tree, not the snapshot)",
    );
    assert!(
        rows.iter().all(|(_id, d)| d.wave == 0),
        "Collection::all surfaced a post-snapshot churn row (wave != 0)",
    );
    let count = coll.count_all()?;
    assert_eq!(
        count,
        u64::try_from(rows.len()).unwrap_or(u64::MAX),
        "Collection::all and count_all disagree on the same handle (#2)",
    );
    assert_eq!(count, BASELINE, "count_all leaked post-snapshot rows");
    Ok(())
}

/// `Db::iter_all` pins its snapshot at construction and must keep every
/// batch refill consistent with it. Construct the iterator (pins the
/// snapshot), let a concurrent writer commit a full churn, THEN drain —
/// the drain crosses the `ITER_ALL_BATCH` boundary AFTER the commit, the
/// window the live-refill bug manifested in. The drain must yield
/// exactly the `BASELINE` baseline rows with no `Error::Corruption`.
#[test]
fn db_iter_all_snapshot_isolated_across_refills_under_churn() {
    let (db, _dir) = fresh_db();
    seed_baseline(&db);

    let (snap_tx, snap_rx) = mpsc::channel::<()>();
    let (commit_tx, commit_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let db_r = Arc::clone(&db);
        let db_w = Arc::clone(&db);

        s.spawn(move || {
            // Construction pins the snapshot before the writer commits.
            let iter = db_r.iter_all::<ScanDoc>().expect("iter_all");
            snap_tx.send(()).expect("snap_tx");
            commit_rx.recv().expect("commit_rx");

            let mut seen: u64 = 0;
            for step in iter {
                // Pre-fix this `?` propagated Error::Corruption when a
                // refill read a freed/recycled page live.
                let (_id, doc) = step.expect("iter_all step must not error");
                assert_eq!(doc.wave, 0, "iter_all surfaced a churn row (#2)");
                seen = seen.saturating_add(1);
            }
            assert_eq!(
                seen, BASELINE,
                "iter_all leaked/lost rows across refills under a \
                 concurrent writer (#2 — refill walked the live pager)",
            );
        });

        s.spawn(move || {
            snap_rx.recv().expect("snap_rx");
            churn_once(&db_w, 1);
            commit_tx.send(()).expect("commit_tx");
        });
    });
}

/// Number of concurrent writer threads for the stress variant.
const N_WRITERS: u64 = 4;
/// Churn cycles each writer thread runs.
const CYCLES_PER_WRITER: u64 = 40;
/// Minimum full scans the reader must complete error-free.
const N_SCANS: u64 = 200;

/// Stress: while writers hammer the tree with insert/delete/recycle
/// churn, a reader repeatedly runs `db.all()` (eager) and `db.iter_all()`
/// (streaming). Each scan is snapshot-isolated to its own pin, so it
/// must ALWAYS observe at least the `BASELINE` baseline rows and must
/// NEVER surface an error. On the pre-fix live walk a recycled page
/// read fails almost immediately with a spurious corruption.
#[test]
fn read_full_scans_never_corrupt_under_concurrent_churn() {
    let (db, _dir) = fresh_db();
    seed_baseline(&db);

    let scan_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let completed = Arc::new(AtomicU64::new(0));

    thread::scope(|scope| {
        for w in 0..N_WRITERS {
            let db = Arc::clone(&db);
            scope.spawn(move || churn_writer_loop(&db, w));
        }
        run_read_scan_loop(&db, &scan_errors, &completed);
    });

    let errors = scan_errors.lock().expect("scan_errors lock");
    assert!(
        errors.is_empty(),
        "read full scan surfaced {} error(s) under concurrent churn \
         (expected 0); first few: {:?}",
        errors.len(),
        &errors[..errors.len().min(5)],
    );
    assert!(
        completed.load(Ordering::SeqCst) >= N_SCANS,
        "reader did not complete the required {N_SCANS} scans",
    );
    // Post-quiesce, the eager and streaming enumerators must agree, and
    // every baseline row must survive (churn rows may linger only if a
    // Busy interrupted a cycle mid-delete — those carry wave != 0).
    let eager = db.all::<ScanDoc>().expect("final all");
    let streamed = drain_iter_all(&db).expect("final iter_all");
    assert_eq!(
        u64::try_from(eager.len()).unwrap_or(u64::MAX),
        streamed,
        "post-quiesce all() and iter_all() disagree",
    );
    let baseline_seen = eager.iter().filter(|d| d.wave == 0).count();
    assert_eq!(
        u64::try_from(baseline_seen).unwrap_or(u64::MAX),
        BASELINE,
        "post-quiesce all() dropped baseline rows",
    );
}

/// Each writer runs `CYCLES_PER_WRITER` churn cycles on its own wave
/// tag-space (offset by the writer index so waves never collide),
/// retrying on `Busy`.
fn churn_writer_loop(db: &Arc<Db>, writer: u64) {
    let base = writer.saturating_mul(1000).saturating_add(1);
    let mut cycle: u64 = 0;
    while cycle < CYCLES_PER_WRITER {
        cycle = cycle.saturating_add(1);
        match churn_cycle_checked(db, base.saturating_add(cycle.saturating_mul(2))) {
            Ok(()) => {}
            Err(obj::Error::Busy { .. }) => {
                thread::yield_now();
                cycle = cycle.saturating_sub(1);
            }
            Err(e) => panic!("writer {writer} churn failed: {e:?}"),
        }
    }
}

/// A `Busy`-propagating churn cycle: insert a wave, delete it, insert
/// the next wave, delete it. A `Busy` mid-cycle bubbles up so the caller
/// retries the whole cycle (leftover rows carry `wave != 0`, so a reader
/// still never mistakes them for baseline).
fn churn_cycle_checked(db: &Db, wave: u64) -> obj::Result<()> {
    let mut ids = Vec::new();
    for seq in 0..50u64 {
        let fill = u8::try_from((wave ^ seq) & 0xFF).unwrap_or(0);
        ids.push(db.insert(ScanDoc {
            wave,
            seq,
            payload: vec![fill; 48],
        })?);
    }
    for id in ids {
        db.delete::<ScanDoc>(id)?;
    }
    Ok(())
}

/// Reader loop: alternate eager `db.all()` and streaming `db.iter_all()`
/// full scans `N_SCANS` times, recording any per-scan error and
/// asserting each scan sees at least the baseline rows.
fn run_read_scan_loop(
    db: &Arc<Db>,
    scan_errors: &Arc<Mutex<Vec<String>>>,
    completed: &Arc<AtomicU64>,
) {
    let mut scans: u64 = 0;
    while scans < N_SCANS {
        scans = scans.saturating_add(1);
        let result = if scans.is_multiple_of(2) {
            drain_eager_all(db)
        } else {
            drain_iter_all(db)
        };
        match result {
            Ok(count) if count < BASELINE => record(scan_errors, format!(
                "scan {scans} saw {count} < baseline {BASELINE} (lost a pinned row)",
            )),
            Ok(_) => {
                completed.fetch_add(1, Ordering::SeqCst);
            }
            Err(e) => record(scan_errors, format!("scan {scans}: {e:?}")),
        }
    }
}

/// Push a failure message into the shared error log.
fn record(scan_errors: &Arc<Mutex<Vec<String>>>, msg: String) {
    if let Ok(mut log) = scan_errors.lock() {
        log.push(msg);
    }
}

/// Drain `db.all::<ScanDoc>()` and return the row count; the eager
/// collect short-circuits to `Err` on the first per-step error.
fn drain_eager_all(db: &Db) -> obj::Result<u64> {
    let rows = db.all::<ScanDoc>()?;
    Ok(u64::try_from(rows.len()).unwrap_or(u64::MAX))
}

/// Drain `db.iter_all::<ScanDoc>()` and return the row count; the FIRST
/// per-step error short-circuits to `Err`.
fn drain_iter_all(db: &Db) -> obj::Result<u64> {
    let iter = db.iter_all::<ScanDoc>()?;
    let mut count: u64 = 0;
    for step in iter {
        step?;
        count = count.saturating_add(1);
    }
    Ok(count)
}
