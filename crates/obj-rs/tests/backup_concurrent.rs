//! Hot-backup integrity under concurrent writes.
//!
//! "Backup of a live, actively-written
//! DB produces a file that itself passes integrity check."
//!
//! Workload: `std::thread::scope` with two threads —
//!
//! 1. A **writer** that continuously runs `Db::transaction` with
//!    a 60% insert / 25% update / 15% delete mix. Uses a
//!    deterministic seeded `ChaCha8Rng` so
//!    a failing seed is reproducible via `OBJ_BACKUP_SEED=<N>`.
//! 2. A **backup driver** that waits a brief warm-up period (so
//!    the writer commits some data), then calls
//!    `Db::backup_to(backup_path)`. After backup completes, the
//!    driver flips `stop = true` so the writer winds down.
//!
//! After the threads join, the test:
//!
//! - Opens `Db::open(backup_path)` and runs `integrity_check` —
//!   asserts `report.is_ok()`. If this ever fails for any seed,
//!   a categorical bug has surfaced.
//! - Runs `integrity_check` on the source DB at end-of-workload —
//!   also asserts `report.is_ok()`.
//! - Records doc counts from both source and backup as
//!   diagnostics. We do NOT assert a tight
//!   doc-count bound because the snapshot LSN's view is internal
//!   to `backup_to`; the contract the test enforces is the
//!   stronger "the backup passes `integrity_check`," which implies
//!   any present doc carries a self-consistent index / primary
//!   pair regardless of the snapshot's exact pin point.
//!
//! Duration is parameterised via `OBJ_BACKUP_DURATION_SECS`
//! (default 10 s for local runs; CI runs the 30 s gate).
//!
//! Heartbeat watchdog: the writer bumps an `AtomicU64` heartbeat
//! every 100 ops; a watchdog thread panics if it fails to advance
//! for 5 s.
//!
//! On failure the run prints `OBJ_BACKUP_SEED=<N>` to stderr and
//! writes the captured op log to
//! `target/backup_concurrent/seed-<N>.log`.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use obj::{Config, Db, Document, Id};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Default duration for a local run when `OBJ_BACKUP_DURATION_SECS`
/// is unset. CI uses 30 s; the human-driven release-validation
/// soak may run longer.
const DEFAULT_DURATION_SECS: u64 = 10;
/// Ops the writer commits between heartbeat bumps.
const HEARTBEAT_OPS_GRANULARITY: u64 = 100;
/// Watchdog tolerance for a stalled heartbeat. 5 s, matching the
/// "writer's heartbeat doesn't advance for 5 s" spec.
const HEARTBEAT_STALL_SECS: u64 = 5;
/// Default seed when `OBJ_BACKUP_SEED` is unset.
const DEFAULT_SEED: u64 = 0xBACC_BACC_DEAD_BEEF;
/// Warm-up delay before the backup driver fires. Gives the writer
/// time to commit some data so the backup is not against an empty
/// pager view.
const BACKUP_WARMUP: Duration = Duration::from_millis(500);

/// Test document. Carries an `id_echo` field so the integrity
/// check's primary-vs-index cross-walk has something to validate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct BackupDoc {
    /// Echo of the document's `Id` value at insert / update time.
    id_echo: u64,
    /// Monotonic-ish version (advances on update).
    version: u32,
    /// Random payload.
    payload: Vec<u8>,
}

impl Document for BackupDoc {
    const COLLECTION: &'static str = "backup_docs";
    const VERSION: u32 = 1;
}

impl obj::Schema for BackupDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("id_echo", obj::DynamicSchema::U64),
            ("version", obj::DynamicSchema::U64),
            ("payload", obj::DynamicSchema::seq(obj::DynamicSchema::U64)),
        ])
    }
}

/// Writer operation tag.
#[derive(Debug, Clone, Copy)]
enum Op {
    Insert,
    Update,
    Delete,
}

#[ignore = "M11 exit gate: concurrent-backup integrity test; run via --ignored"]
#[test]
fn backup_concurrent() {
    let duration_secs = env::var("OBJ_BACKUP_DURATION_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_DURATION_SECS);
    let seed = env::var("OBJ_BACKUP_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SEED);
    eprintln!("OBJ_BACKUP_SEED={seed} OBJ_BACKUP_DURATION_SECS={duration_secs}");

    let dir = TempDir::new().expect("tempdir");
    let src_path = dir.path().join("src.obj");
    let backup_path = dir.path().join("backup.obj");
    let outcome = run_backup_workload(&src_path, &backup_path, seed, duration_secs);
    report(&outcome, seed);
}

/// Aggregated outcome of one concurrent-backup run. `Some(msg)`
/// is a failure; `None` is success.
struct Outcome {
    failure: Option<String>,
    writer_ops: u64,
    writer_busy: u64,
    /// Source-DB doc count at end-of-workload.
    source_count: usize,
    /// Backup-DB doc count.
    backup_count: usize,
    op_log: Vec<String>,
}

fn report(outcome: &Outcome, seed: u64) {
    eprintln!(
        "backup_concurrent run complete: writer_ops={} writer_busy={} \
         source_count={} backup_count={}",
        outcome.writer_ops, outcome.writer_busy, outcome.source_count, outcome.backup_count,
    );
    if let Some(msg) = outcome.failure.as_ref() {
        let log_dir = PathBuf::from("target").join("backup_concurrent");
        let _ = fs::create_dir_all(&log_dir);
        let log_path = log_dir.join(format!("seed-{seed}.log"));
        if let Ok(mut f) = fs::File::create(&log_path) {
            for line in &outcome.op_log {
                let _ = writeln!(f, "{line}");
            }
        }
        panic!(
            "OBJ_BACKUP_SEED={seed} FAIL: {msg}\nop log: {}",
            log_path.display(),
        );
    }
}

/// Spawn the writer + backup driver + watchdog and wait for them
/// to finish. After the threads join, integrity-check both source
/// and backup.
fn run_backup_workload(
    src_path: &Path,
    backup_path: &Path,
    seed: u64,
    duration_secs: u64,
) -> Outcome {
    let config = Config::default()
        .cross_process_lock(false)
        .busy_timeout(Duration::from_mins(2));
    let db = Arc::new(Db::open_with(src_path, config).expect("open source"));
    let duration = Duration::from_secs(duration_secs);
    let shared = SharedState::new();
    let backup_outcome = Arc::new(Mutex::new(None));

    thread::scope(|scope| {
        let writer_db = Arc::clone(&db);
        let writer_shared = shared.clone();
        scope.spawn(move || {
            writer_loop(&writer_db, seed, duration, &writer_shared);
        });

        let backup_driver_db = Arc::clone(&db);
        let backup_driver_shared = shared.clone();
        let backup_path_buf: PathBuf = backup_path.to_path_buf();
        let backup_outcome_for_driver = Arc::clone(&backup_outcome);
        scope.spawn(move || {
            let res = backup_driver(&backup_driver_db, &backup_path_buf, &backup_driver_shared);
            if let Ok(mut slot) = backup_outcome_for_driver.lock() {
                *slot = Some(res);
            }
            backup_driver_shared.stop.store(true, Ordering::SeqCst);
        });

        run_watchdog(&shared, duration);
    });

    finalize_outcome(&db, backup_path, shared, &backup_outcome)
}

/// Per-run shared state pulled out of `run_backup_workload`.
#[derive(Clone)]
struct SharedState {
    heartbeat: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    expected: Arc<Mutex<HashMap<u64, ExpectedState>>>,
    op_log: Arc<Mutex<Vec<String>>>,
    writer_ops: Arc<AtomicU64>,
    writer_busy: Arc<AtomicU64>,
    writer_failure: Arc<Mutex<Option<String>>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            heartbeat: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(AtomicBool::new(false)),
            expected: Arc::new(Mutex::new(HashMap::new())),
            op_log: Arc::new(Mutex::new(Vec::new())),
            writer_ops: Arc::new(AtomicU64::new(0)),
            writer_busy: Arc::new(AtomicU64::new(0)),
            writer_failure: Arc::new(Mutex::new(None)),
        }
    }
}

/// Last known committed state for an id — same contract as the
/// stress test. Used by the writer to skip ids it knows are
/// deleted.
#[derive(Debug, Clone, Copy)]
enum ExpectedState {
    Present,
    Deleted,
}

fn writer_loop(db: &Arc<Db>, seed: u64, duration: Duration, shared: &SharedState) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = Instant::now();
    let mut iter: u64 = 0;
    while !shared.stop.load(Ordering::Relaxed) && start.elapsed() < duration {
        iter = iter.saturating_add(1);
        if iter.is_multiple_of(HEARTBEAT_OPS_GRANULARITY) {
            shared.heartbeat.store(iter, Ordering::Relaxed);
        }
        let op = choose_op(&mut rng);
        match perform_writer_op(db, op, &mut rng, &shared.expected) {
            Ok(()) => {
                shared.writer_ops.fetch_add(1, Ordering::Relaxed);
            }
            Err(obj::Error::Busy { .. }) => {
                shared.writer_busy.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                if let Ok(mut log) = shared.op_log.lock() {
                    log.push(format!("writer iter {iter}: op {op:?} err: {e:?}"));
                }
                if let Ok(mut slot) = shared.writer_failure.lock() {
                    *slot = Some(format!("writer iter {iter}: op {op:?} err: {e:?}"));
                }
                shared.stop.store(true, Ordering::SeqCst);
                return;
            }
        }
    }
    shared.heartbeat.store(iter, Ordering::Relaxed);
}

/// 60% Insert / 25% Update / 15% Delete — same distribution as
/// the stress test.
fn choose_op(rng: &mut ChaCha8Rng) -> Op {
    let n: u32 = rng.random_range(0..100);
    match n {
        0..60 => Op::Insert,
        60..85 => Op::Update,
        _ => Op::Delete,
    }
}

fn perform_writer_op(
    db: &Db,
    op: Op,
    rng: &mut ChaCha8Rng,
    expected: &Arc<Mutex<HashMap<u64, ExpectedState>>>,
) -> obj::Result<()> {
    match op {
        Op::Insert => writer_insert(db, rng, expected),
        Op::Update => writer_update(db, rng, expected),
        Op::Delete => writer_delete(db, rng, expected),
    }
}

fn writer_insert(
    db: &Db,
    rng: &mut ChaCha8Rng,
    expected: &Arc<Mutex<HashMap<u64, ExpectedState>>>,
) -> obj::Result<()> {
    let payload = random_payload(rng);
    let inserted = db.transaction(|tx| {
        let coll = tx.collection::<BackupDoc>()?;
        let id = coll.insert(BackupDoc {
            id_echo: 0,
            version: 1,
            payload,
        })?;
        coll.update(id, |d: &mut BackupDoc| {
            d.id_echo = id.get();
        })?;
        Ok(id)
    })?;
    if let Ok(mut map) = expected.lock() {
        map.insert(inserted.get(), ExpectedState::Present);
    }
    Ok(())
}

fn writer_update(
    db: &Db,
    rng: &mut ChaCha8Rng,
    expected: &Arc<Mutex<HashMap<u64, ExpectedState>>>,
) -> obj::Result<()> {
    let Some((id, new_version)) = pick_existing_id(rng, expected) else {
        return Ok(());
    };
    let payload = random_payload(rng);
    let r = db.update::<BackupDoc, _>(
        Id::try_new(id).expect("nonzero by construction"),
        |d: &mut BackupDoc| {
            d.payload.clone_from(&payload);
            d.version = new_version;
        },
    );
    match r {
        Ok(()) => {
            if let Ok(mut map) = expected.lock() {
                map.insert(id, ExpectedState::Present);
            }
            Ok(())
        }
        Err(obj::Error::DocumentNotFound { .. }) => Ok(()),
        Err(e) => Err(e),
    }
}

fn writer_delete(
    db: &Db,
    rng: &mut ChaCha8Rng,
    expected: &Arc<Mutex<HashMap<u64, ExpectedState>>>,
) -> obj::Result<()> {
    let Some((id, _)) = pick_existing_id(rng, expected) else {
        return Ok(());
    };
    let existed = db.delete::<BackupDoc>(Id::try_new(id).expect("nonzero"))?;
    if existed {
        if let Ok(mut map) = expected.lock() {
            map.insert(id, ExpectedState::Deleted);
        }
    }
    Ok(())
}

fn random_payload(rng: &mut ChaCha8Rng) -> Vec<u8> {
    let len: usize = rng.random_range(8..256);
    let mut buf = vec![0u8; len];
    rng.fill(buf.as_mut_slice());
    buf
}

fn pick_existing_id(
    rng: &mut ChaCha8Rng,
    expected: &Arc<Mutex<HashMap<u64, ExpectedState>>>,
) -> Option<(u64, u32)> {
    let Ok(map) = expected.lock() else {
        return None;
    };
    let present: Vec<u64> = map
        .iter()
        .filter_map(|(k, v)| matches!(v, ExpectedState::Present).then_some(*k))
        .collect();
    if present.is_empty() {
        return None;
    }
    let idx = rng.random_range(0..present.len());
    let id = present[idx];
    let next_version: u32 = rng.random_range(2..u32::MAX);
    Some((id, next_version))
}

/// Backup driver: wait for the warm-up window, then take a
/// backup. Returns the `backup_to` result so the test can surface
/// it as a hard failure if it errors.
fn backup_driver(db: &Arc<Db>, backup_path: &Path, shared: &SharedState) -> obj::Result<()> {
    let warmup_start = Instant::now();
    while warmup_start.elapsed() < BACKUP_WARMUP && !shared.stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(50));
    }
    db.backup_to(backup_path)
}

/// Watchdog: panics if the writer's heartbeat fails to advance
/// for `HEARTBEAT_STALL_SECS`. Returns when the duration elapses
/// or the `stop` flag is set.
fn run_watchdog(shared: &SharedState, duration: Duration) {
    let start = Instant::now();
    let mut last_seen = shared.heartbeat.load(Ordering::Relaxed);
    let mut last_change = Instant::now();
    while !shared.stop.load(Ordering::Relaxed) && start.elapsed() < duration {
        thread::sleep(Duration::from_millis(250));
        let current = shared.heartbeat.load(Ordering::Relaxed);
        if current != last_seen {
            last_seen = current;
            last_change = Instant::now();
        } else if last_change.elapsed() > Duration::from_secs(HEARTBEAT_STALL_SECS) {
            shared.stop.store(true, Ordering::SeqCst);
            panic!(
                "watchdog: writer heartbeat stalled at {current} for \
                 {HEARTBEAT_STALL_SECS}s; rerun with OBJ_BACKUP_SEED=<seed> \
                 to repro",
            );
        }
    }
    shared.stop.store(true, Ordering::SeqCst);
}

/// After threads join: open the backup, integrity-check it, count
/// docs in both source + backup, integrity-check the source.
fn finalize_outcome(
    source_db: &Arc<Db>,
    backup_path: &Path,
    shared: SharedState,
    backup_outcome: &Arc<Mutex<Option<obj::Result<()>>>>,
) -> Outcome {
    let writer_ops = shared.writer_ops.load(Ordering::SeqCst);
    let writer_busy = shared.writer_busy.load(Ordering::SeqCst);
    let op_log: Vec<String> = Arc::try_unwrap(shared.op_log)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    let mut failure: Option<String> = Arc::try_unwrap(shared.writer_failure)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .flatten();

    let backup_result: Option<obj::Result<()>> =
        backup_outcome.lock().ok().and_then(|mut g| g.take());
    if failure.is_none() {
        match &backup_result {
            Some(Ok(())) => {}
            Some(Err(e)) => failure = Some(format!("backup_to: {e:?}")),
            None => failure = Some("backup driver did not run".to_owned()),
        }
    }

    let (source_count, source_check_err) = check_source(source_db);
    if failure.is_none() {
        failure = source_check_err;
    }

    let (backup_count, backup_check_err) = check_backup(backup_path);
    if failure.is_none() {
        failure = backup_check_err;
    }

    Outcome {
        failure,
        writer_ops,
        writer_busy,
        source_count,
        backup_count,
        op_log,
    }
}

/// Run `integrity_check` on the source DB and count its docs.
/// Returns `(doc_count, Some(failure_message))` on a problem,
/// `(doc_count, None)` on success.
fn check_source(db: &Arc<Db>) -> (usize, Option<String>) {
    let count = db.all::<BackupDoc>().map_or(0, |v| v.len());
    let report = match db.integrity_check() {
        Ok(r) => r,
        Err(e) => return (count, Some(format!("source integrity_check: {e:?}"))),
    };
    if !report.is_ok() {
        return (
            count,
            Some(format!(
                "source DB failed integrity_check at end-of-workload: \
                 failures = {:?}",
                report.failures,
            )),
        );
    }
    (count, None)
}

/// Re-open the backup file and run `integrity_check`. This is the
/// strongest invariant — a failure here is a
/// categorical bug.
fn check_backup(backup_path: &Path) -> (usize, Option<String>) {
    let db = match Db::open(backup_path) {
        Ok(d) => d,
        Err(e) => return (0, Some(format!("Db::open(backup): {e:?}"))),
    };
    let count = db.all::<BackupDoc>().map_or(0, |v| v.len());
    let report = match db.integrity_check() {
        Ok(r) => r,
        Err(e) => return (count, Some(format!("backup integrity_check: {e:?}"))),
    };
    if !report.is_ok() {
        return (
            count,
            Some(format!(
                "M11 EXIT GATE FAIL: backup did NOT pass integrity_check; \
                 failures = {:?}",
                report.failures,
            )),
        );
    }
    (count, None)
}

/// Compile-time `Send + Sync` check on the harness's shared state.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<SharedState>;
};

/// A backup that races an external-process writer must be
/// excluded by the cross-process `WRITER_LOCK`.
#[test]
fn backup_blocked_by_cross_process_writer_lock() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("xproc_backup.obj");
    let backup_path = dir.path().join("xproc_backup_out.obj");

    let cfg = || Config::default().busy_timeout(Duration::from_millis(300));
    let db_writer = Db::open_with(&path, cfg()).expect("open writer");
    let db_backup = Db::open_with(&path, cfg()).expect("open backup handle");

    db_writer
        .insert(BackupDoc {
            id_echo: 1,
            version: 1,
            payload: vec![0xAB; 64],
        })
        .expect("seed insert");

    db_writer
        .transaction(|_tx| {
            let err = db_backup
                .backup_to(&backup_path)
                .expect_err("backup must be refused while a writer holds the lock");
            assert!(
                matches!(err, obj::Error::Busy { .. }),
                "expected Error::Busy while external writer holds WRITER_LOCK, got {err:?}",
            );
            assert!(
                !backup_path.exists(),
                "a refused backup must not leave a destination file behind",
            );
            Ok::<(), obj::Error>(())
        })
        .expect("writer txn commits");

    db_backup
        .backup_to(&backup_path)
        .expect("backup succeeds once the writer lock is free");
    let restored = Db::open(&backup_path).expect("reopen backup");
    let report = restored.integrity_check().expect("integrity check runs");
    assert!(
        report.is_ok(),
        "cross-process backup must pass integrity_check; failures = {:?}",
        report.failures,
    );
}

/// `checkpoint` must likewise hold the cross-process
/// `WRITER_LOCK`, so a checkpoint racing an external-process writer is
/// serialized rather than interleaving main-file writes / salt
/// rotation.
#[test]
fn checkpoint_blocked_by_cross_process_writer_lock() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("xproc_ckpt.obj");

    let cfg = || Config::default().busy_timeout(Duration::from_millis(300));
    let db_writer = Db::open_with(&path, cfg()).expect("open writer");
    let db_ckpt = Db::open_with(&path, cfg()).expect("open checkpoint handle");

    db_writer
        .insert(BackupDoc {
            id_echo: 2,
            version: 1,
            payload: vec![0xCD; 32],
        })
        .expect("seed insert");

    db_writer
        .transaction(|_tx| {
            let err = db_ckpt
                .checkpoint()
                .expect_err("checkpoint must be refused while a writer holds the lock");
            assert!(
                matches!(err, obj::Error::Busy { .. }),
                "expected Error::Busy while external writer holds WRITER_LOCK, got {err:?}",
            );
            Ok::<(), obj::Error>(())
        })
        .expect("writer txn commits");

    db_ckpt
        .checkpoint()
        .expect("checkpoint succeeds once the writer lock is free");
}
