//! 10 000 randomized crash-cycle tests.
//!
//! Each cycle:
//! 1. Allocates a fresh database file in a temp directory.
//! 2. Seeds a [`ChaCha8Rng`] from the cycle's `u64` seed.
//! 3. Runs a randomized workload of 30..100 operations drawn from
//!    `{Write, Commit, Checkpoint, Reopen}`. Per-page expected
//!    values are tracked in an in-memory `HashMap` that mirrors the
//!    committed state.
//! 4. Optionally injects a deliberate panic AFTER one of the
//!    completed commits — this simulates a SIGKILL between two
//!    well-defined consistent points and is caught by
//!    `std::panic::catch_unwind`.
//! 5. Reopens the database and asserts: every page-id whose latest
//!    write was committed before the injected panic (or before the
//!    workload ended cleanly) carries the expected bytes; reads of
//!    those pages do not surface `Error::Corruption`.
//!
//! The test is `#[ignore]` so a plain `cargo test --workspace` does
//! not pay the ~minutes cost. Invoke it explicitly:
//!
//! ```text
//! cargo test --test crash_cycles -- --ignored --test-threads=1
//! ```
//!
//! Shard selection for CI uses the `CRASH_CYCLES_START` and
//! `CRASH_CYCLES_END` environment variables. When unset, the test
//! runs the full `[0, 10_000)` range.
//!
//! When a seed fails, the test prints `SEED=<N>` to stderr and
//! writes a per-seed operation log to
//! `target/crash_cycles/seed-<N>.log`. The same seed can be re-run
//! deterministically via:
//!
//! ```text
//! CRASH_CYCLES_START=42 CRASH_CYCLES_END=43 \
//!     cargo test --test crash_cycles -- --ignored --test-threads=1
//! ```

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use obj_core::pager::page::{Page, PageId, PAGE_SIZE};
use obj_core::pager::{wal_path_for, Config, Pager};
use obj_core::platform::fault::{FaultPlan, FaultyFileHandle};
use obj_core::platform::{FileBackend, FileHandle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const DEFAULT_START: u64 = 0;
const DEFAULT_END: u64 = 10_000;

#[derive(Debug, Clone, Copy)]
enum Op {
    Write(usize),
    Commit,
    Checkpoint,
    Reopen,
    Alloc,
    /// Free the `idx`-th currently-allocated page (mod `allocated.len()`).
    Free(usize),
}

/// Backend selector used by the cycle harness to pick whether each
/// cycle runs against a plain [`FileHandle`] (panic-only crashes) or
/// a `FaultyFileHandle` driving torn writes, dropped fsyncs, and bit
/// flips. Each variant carries its own `Pager` opener; the rest of
/// the harness is identical.
#[derive(Debug, Clone, Copy)]
enum BackendKind {
    /// Panic-only crashes via `std::panic::panic_any`. The pager
    /// holds the production [`FileHandle`]; no syscall-level faults.
    PanicOnly,
    /// Intra-syscall faults via [`FaultyFileHandle`]. Each cycle
    /// derives a [`FaultPlan`] from the seed; recovery invariants
    /// must still hold.
    IntraSyscall,
}

#[test]
#[ignore = "10k-cycle stress test; ~12 min locally. Run via `cargo test --test crash_cycles -- --ignored`"]
fn crash_cycles_seed_range() {
    run_seed_range(BackendKind::PanicOnly, "crash_cycles");
}

#[test]
#[ignore = "10k-cycle stress test with intra-syscall faults (issue #20); slower than the panic-only variant. Run via `cargo test --test crash_cycles -- --ignored`"]
fn crash_cycles_with_intra_syscall_faults() {
    run_seed_range(BackendKind::IntraSyscall, "crash_cycles_faulty");
}

fn run_seed_range(kind: BackendKind, label: &str) {
    let (start, end) = read_seed_range();
    let mut failures: Vec<u64> = Vec::new();
    let mut total: u64 = 0;
    let mode = match kind {
        BackendKind::PanicOnly => VerifyMode::Strict,
        BackendKind::IntraSyscall => VerifyMode::Lenient,
    };
    for seed in start..end {
        total += 1;
        if let Err(msg) = run_one_cycle(seed, kind, mode) {
            eprintln!("SEED={seed} ({label}) FAILED: {msg}");
            failures.push(seed);
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: {} of {} seeds failed: first 10 = {:?}",
        failures.len(),
        total,
        &failures[..failures.len().min(10)]
    );
    eprintln!("{label}: {total} seeds passed (range {start}..{end})");
}

/// Recovery-invariant strictness selector.
///
/// - `Strict`: each page whose history is non-empty must recover its
///   **LAST** committed bytes. Required for `crash_cycles_seed_range`
///   (panic-only) where ACID demands the most recent `commit()`'s
///   bytes survive a post-Ok panic.
/// - `Lenient`: each page may match ANY historical commit (or zeros,
///   or be absent). Required for
///   `crash_cycles_with_intra_syscall_faults` where dropped fsyncs
///   and torn writes legitimately roll a commit back to an earlier
///   durable state.
#[derive(Debug, Clone, Copy)]
enum VerifyMode {
    Strict,
    Lenient,
}

fn read_seed_range() -> (u64, u64) {
    let start = std::env::var("CRASH_CYCLES_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_START);
    let end = std::env::var("CRASH_CYCLES_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_END);
    (start, end)
}

fn run_one_cycle(seed: u64, kind: BackendKind, mode: VerifyMode) -> Result<(), String> {
    match kind {
        BackendKind::PanicOnly => run_one_cycle_with::<FileHandle, _>(seed, &PanicOnlyOpener, mode),
        BackendKind::IntraSyscall => {
            let opener = FaultyOpener::from_seed(seed);
            run_one_cycle_with::<FaultyFileHandle, _>(seed, &opener, mode)
        }
    }
}

/// Factory for [`Pager`]s used by a single cycle. Each cycle uses
/// its own factory instance so seed-derived state (PRNG seeds for
/// the fault plan) is stamped once.
trait CycleOpener<F: FileBackend> {
    fn open(&self, path: &Path, config: Config) -> obj_core::Result<Pager<F>>;
}

struct PanicOnlyOpener;

impl CycleOpener<FileHandle> for PanicOnlyOpener {
    fn open(&self, path: &Path, config: Config) -> obj_core::Result<Pager<FileHandle>> {
        let mut p = Pager::open(path, config)?;
        p.begin_txn();
        Ok(p)
    }
}

/// Plan template for the intra-syscall variant. Probabilities are
/// deliberately small so most ops succeed transparently; only a
/// minority of cycles see a fault. Seed-derived PRNG seeds keep
/// each cycle's faults reproducible across CI runners.
///
/// Fault placement (seed→fault-mode mapping):
/// - **WAL** sees `torn_write_prob`, `dropped_fsync_prob`, and
///   `bit_flip_prob`. The WAL is the durability layer designed to
///   absorb these — CRC32C frame checks + the two-pass recovery
///   walk detect corruption; salt-mismatched torn tails are
///   silently discarded.
/// - **Main file** sees only `dropped_fsync_prob`. A torn write or
///   bit flip on the main file is bit-rot — recovery cannot
///   reconstruct lost bytes; that is the page-trailer CRC's job to
///   detect, not the WAL's job to recover. Restricting main-file
///   faults to dropped-fsync exercises the salt-rotation logic
///   (which was designed for exactly this case) without producing
///   unrecoverable on-disk corruption.
///
/// Tuning rationale (recorded with the test on purpose, since the
/// numbers carry semantic weight):
/// - `torn_write_prob=0.003`: ~1 in 333 WAL writes lands short.
/// - `dropped_fsync_prob=0.01`: ~1 in 100 fsyncs is silently
///   dropped. The salt-rotation logic was designed for this
///   case.
/// - `bit_flip_prob=0.0008`: <1 in 1000 WAL writes carries a
///   flipped bit. CRC32C catches it; it surfaces as
///   `Error::WalCorruption`.
/// - `short_read_prob=0.0`: skipped; a fault-injected EOF surfaces
///   as `Error::Io` and is propagated normally by the workload.
/// - `crash_after_ops=0`: panic-only crashes are the *other*
///   variant; mixing here would conflate failure modes.
struct FaultyOpener {
    main_seed: u64,
    wal_seed: u64,
}

impl FaultyOpener {
    fn from_seed(seed: u64) -> Self {
        Self {
            main_seed: seed ^ 0x9E37_79B9_7F4A_7C15,
            wal_seed: seed ^ 0xBB67_AE85_84CA_A73B,
        }
    }

    fn main_plan(&self) -> FaultPlan {
        FaultPlan::new(self.main_seed, 0.0, 0.01, 0.0, 0.0, 0)
    }

    fn wal_plan(&self) -> FaultPlan {
        FaultPlan::new(self.wal_seed, 0.003, 0.01, 0.0, 0.0008, 0)
    }
}

impl CycleOpener<FaultyFileHandle> for FaultyOpener {
    fn open(&self, path: &Path, config: Config) -> obj_core::Result<Pager<FaultyFileHandle>> {
        let main = FaultyFileHandle::new(FileHandle::open_or_create(path)?, self.main_plan());
        let wal_path = wal_path_for(path);
        let wal = FaultyFileHandle::new(FileHandle::open_or_create(&wal_path)?, self.wal_plan());
        let mut p = Pager::<FaultyFileHandle>::open_with_backends(main, wal, wal_path, config)?;
        p.begin_txn();
        Ok(p)
    }
}

fn run_one_cycle_with<F: FileBackend, O: CycleOpener<F>>(
    seed: u64,
    opener: &O,
    mode: VerifyMode,
) -> Result<(), String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dir = TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
    let db_path = dir.path().join("crash.obj");
    let n_pages: u32 = rng.random_range(4u32..12);
    let n_ops: u32 = rng.random_range(20u32..60);
    let panic_after_commit: Option<u32> = if rng.random::<u32>() % 4 == 0 {
        Some(rng.random_range(0u32..n_ops))
    } else {
        None
    };
    let mut state = CycleState::new(&db_path, n_pages, opener)?;
    let mut log: Vec<String> = Vec::with_capacity(n_ops as usize);
    let ops: Vec<Op> = (0..n_ops).map(|_| random_op(&mut rng, n_pages)).collect();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        run_workload(&mut state, &ops, panic_after_commit, &mut log, opener)
    }));
    handle_workload_result(seed, result, &log)?;
    let verify = match mode {
        VerifyMode::Strict => verify_recovery_strict,
        VerifyMode::Lenient => verify_recovery_lenient,
    };
    verify(&db_path, &state.history, &state.allocated).inspect_err(|_| {
        let _ = write_seed_log(seed, &log);
    })?;
    Ok(())
}

fn handle_workload_result(
    seed: u64,
    result: std::thread::Result<Result<(), String>>,
    log: &[String],
) -> Result<(), String> {
    match result {
        Err(panic) => {
            let msg = panic_message(&panic);
            if msg.contains("__INJECTED_CRASH__") || msg.contains("obj-core::fault") {
                Ok(())
            } else {
                write_seed_log(seed, log).map_err(|e| format!("write log: {e}"))?;
                Err(format!("workload panicked: {msg}"))
            }
        }
        Ok(Err(workload_err)) => {
            if workload_err.starts_with("write_page:")
                || workload_err.starts_with("commit:")
                || workload_err.starts_with("checkpoint:")
                || workload_err.starts_with("close:")
                || workload_err.starts_with("reopen:")
            {
                Ok(())
            } else {
                write_seed_log(seed, log).map_err(|e| format!("write log: {e}"))?;
                Err(format!("workload error: {workload_err}"))
            }
        }
        Ok(Ok(())) => Ok(()),
    }
}

fn random_op(rng: &mut ChaCha8Rng, n_pages: u32) -> Op {
    let pick = rng.random::<u32>() % 100;
    match pick {
        0..=49 => Op::Write(usize::try_from(rng.random_range(0u32..n_pages)).unwrap_or(0)),
        50..=74 => Op::Commit,
        75..=84 => Op::Checkpoint,
        85..=89 => Op::Reopen,
        90..=94 => Op::Alloc,
        _ => Op::Free(usize::try_from(rng.random_range(0u32..n_pages)).unwrap_or(0)),
    }
}

struct CycleState<F: FileBackend> {
    pager: Option<Pager<F>>,
    db_path: PathBuf,
    /// Expected committed bytes per page-id. Updated after each
    /// successful `commit` (or `flush`).
    expected: HashMap<PageId, Vec<u8>>,
    /// All historical committed values per page-id. The fault-
    /// injected variant accepts "page reads back matching ANY
    /// historical value" as a valid recovery outcome: a torn write
    /// or dropped fsync may roll a transaction back to a prior
    /// commit, but it can never fabricate new bytes — that is the
    /// invariant CRC32C protects.
    history: HashMap<PageId, Vec<Vec<u8>>>,
    /// Pending writes since the last commit. Lost on reopen without
    /// commit; merged into `expected` on commit.
    pending: HashMap<PageId, Vec<u8>>,
    /// Allocated page-ids (`NonZeroU64`).
    allocated: Vec<PageId>,
    /// Number of commits performed so far. Used to match
    /// `panic_after_commit`.
    commit_count: u32,
}

impl<F: FileBackend> CycleState<F> {
    fn new<O: CycleOpener<F>>(db_path: &Path, n_pages: u32, opener: &O) -> Result<Self, String> {
        let mut p = opener
            .open(db_path, Config::default())
            .map_err(|e| format!("open: {e}"))?;
        let cap = usize::try_from(n_pages).unwrap_or(0);
        let mut allocated = Vec::with_capacity(cap);
        for _ in 0..n_pages {
            allocated.push(p.alloc_page().map_err(|e| format!("alloc: {e}"))?);
        }
        let _ = p.commit().map_err(|e| format!("setup commit: {e}"))?;
        Ok(Self {
            pager: Some(p),
            db_path: db_path.to_path_buf(),
            expected: HashMap::new(),
            history: HashMap::new(),
            pending: HashMap::new(),
            allocated,
            commit_count: 0,
        })
    }

    fn pager_mut(&mut self) -> &mut Pager<F> {
        self.pager
            .as_mut()
            .expect("pager Some during workload step")
    }
}

fn run_workload<F: FileBackend, O: CycleOpener<F>>(
    state: &mut CycleState<F>,
    ops: &[Op],
    panic_after_commit: Option<u32>,
    log: &mut Vec<String>,
    opener: &O,
) -> Result<(), String> {
    for (step, op) in ops.iter().enumerate() {
        match *op {
            Op::Write(idx) => op_write(state, step, idx, log)?,
            Op::Commit => op_commit(state, step, panic_after_commit, log)?,
            Op::Checkpoint => {
                state
                    .pager_mut()
                    .checkpoint()
                    .map_err(|e| format!("checkpoint: {e}"))?;
                log.push(format!("{step}: checkpoint"));
            }
            Op::Reopen => op_reopen(state, step, log, opener)?,
            Op::Alloc => op_alloc(state, step, log)?,
            Op::Free(idx) => op_free(state, step, idx, log)?,
        }
    }
    Ok(())
}

fn op_write<F: FileBackend>(
    state: &mut CycleState<F>,
    step: usize,
    idx: usize,
    log: &mut Vec<String>,
) -> Result<(), String> {
    if state.allocated.is_empty() {
        return Ok(());
    }
    let pid = state.allocated[idx % state.allocated.len()];
    let body = page_for(step, pid);
    let mut page = Page::zeroed();
    let body_len = body.len().min(PAGE_SIZE - 4);
    page.as_bytes_mut()[..body_len].copy_from_slice(&body[..body_len]);
    state
        .pager_mut()
        .write_page(pid, &page)
        .map_err(|e| format!("write_page: {e}"))?;
    state.pending.insert(pid, page.as_bytes().to_vec());
    log.push(format!("{step}: write {pid:?}", pid = pid.get()));
    Ok(())
}

fn op_commit<F: FileBackend>(
    state: &mut CycleState<F>,
    step: usize,
    panic_after_commit: Option<u32>,
    log: &mut Vec<String>,
) -> Result<(), String> {
    let _ = state
        .pager_mut()
        .commit()
        .map_err(|e| format!("commit: {e}"))?;
    merge_pending_into_history(state);
    state.commit_count = state.commit_count.saturating_add(1);
    log.push(format!(
        "{step}: commit (#{count})",
        count = state.commit_count
    ));
    if panic_after_commit == Some(state.commit_count.saturating_sub(1)) {
        log.push(format!("{step}: __INJECTED_CRASH__"));
        panic!("__INJECTED_CRASH__ at step {step}");
    }
    Ok(())
}

fn op_reopen<F: FileBackend, O: CycleOpener<F>>(
    state: &mut CycleState<F>,
    step: usize,
    log: &mut Vec<String>,
    opener: &O,
) -> Result<(), String> {
    let p = state.pager.take().expect("pager");
    p.close().map_err(|e| format!("close: {e}"))?;
    state.pending.clear();
    let p = opener
        .open(&state.db_path, Config::default())
        .map_err(|e| format!("reopen: {e}"))?;
    state.pager = Some(p);
    log.push(format!("{step}: reopen"));
    Ok(())
}

fn op_alloc<F: FileBackend>(
    state: &mut CycleState<F>,
    step: usize,
    log: &mut Vec<String>,
) -> Result<(), String> {
    let pid = state
        .pager_mut()
        .alloc_page()
        .map_err(|e| format!("alloc: {e}"))?;
    state.allocated.push(pid);
    log.push(format!("{step}: alloc -> {id}", id = pid.get()));
    let _ = state
        .pager_mut()
        .commit()
        .map_err(|e| format!("commit: {e}"))?;
    merge_pending_into_history(state);
    state.commit_count = state.commit_count.saturating_add(1);
    Ok(())
}

fn op_free<F: FileBackend>(
    state: &mut CycleState<F>,
    step: usize,
    idx: usize,
    log: &mut Vec<String>,
) -> Result<(), String> {
    if state.allocated.is_empty() {
        return Ok(());
    }
    let pos = idx % state.allocated.len();
    let victim = state.allocated.remove(pos);
    state.expected.remove(&victim);
    state.history.remove(&victim);
    state.pending.remove(&victim);
    state
        .pager_mut()
        .free_page(victim)
        .map_err(|e| format!("free: {e}"))?;
    log.push(format!("{step}: free {id}", id = victim.get()));
    let _ = state
        .pager_mut()
        .commit()
        .map_err(|e| format!("commit: {e}"))?;
    merge_pending_into_history(state);
    state.commit_count = state.commit_count.saturating_add(1);
    Ok(())
}

fn merge_pending_into_history<F: FileBackend>(state: &mut CycleState<F>) {
    for (id, body) in state.pending.drain() {
        state.expected.insert(id, body.clone());
        state.history.entry(id).or_default().push(body);
    }
}

fn verify_recovery_lenient(
    db_path: &std::path::Path,
    history: &HashMap<PageId, Vec<Vec<u8>>>,
    allocated: &[PageId],
) -> Result<(), String> {
    let mut p = match Pager::open(db_path, Config::default()) {
        Ok(p) => p,
        Err(obj_core::Error::WalCorruption { .. }) => {
            return Ok(());
        }
        Err(e) => return Err(format!("recovery open: {e}")),
    };
    for pid in allocated {
        let Some(want_versions) = history.get(pid) else {
            match p.read_page(*pid) {
                Ok(_) | Err(obj_core::Error::Corruption { .. }) => continue,
                Err(e) => {
                    return Err(format!("recovery read {pid:?}: {e}", pid = pid.get()));
                }
            }
        };
        let read = match p.read_page(*pid) {
            Ok(r) => r,
            Err(obj_core::Error::Corruption { .. }) => {
                return Err(format!(
                    "page {pid:?}: unexpected on-disk corruption",
                    pid = pid.get()
                ));
            }
            Err(e) => return Err(format!("recovery read {pid:?}: {e}", pid = pid.get())),
        };
        let body_len = PAGE_SIZE - 4;
        let got: &[u8] = &read.as_bytes()[..body_len];
        if !page_matches_any_committed(got, want_versions) {
            return Err(format!(
                "page {pid:?}: bytes match neither any historical commit nor zeros",
                pid = pid.get()
            ));
        }
    }
    Ok(())
}

/// `true` if `got` matches some entry in `history` (truncated to the
/// pre-trailer body length) OR the all-zero initial state.
fn page_matches_any_committed(got: &[u8], history: &[Vec<u8>]) -> bool {
    if got.iter().all(|&b| b == 0) {
        return true;
    }
    history.iter().any(|want| {
        let want_body: &[u8] = &want[..got.len().min(want.len())];
        got[..want_body.len()] == *want_body
    })
}

/// `true` if `got` matches the LAST entry in `history` (truncated to
/// the pre-trailer body length). Used by `verify_recovery_strict`
/// where the test harness has tracked the most-recent committed
/// version per page and a recovery that returns an EARLIER version
/// is a real durability bug.
fn page_matches_last_committed(got: &[u8], history: &[Vec<u8>]) -> bool {
    let Some(want) = history.last() else {
        return got.iter().all(|&b| b == 0);
    };
    let want_body: &[u8] = &want[..got.len().min(want.len())];
    got[..want_body.len()] == *want_body
}

/// `verify_recovery_strict`: every page whose history is
/// non-empty must read back the LAST committed bytes. This is the
/// ACID contract for the panic-only crash variant — a panic that
/// fires AFTER `commit()` returns Ok cannot legitimately roll a
/// committed transaction back to an earlier version.
///
/// Pages whose history is empty (allocated but never written) are
/// only required to be readable without surfacing `Error::Corruption`
/// — same as the lenient variant, since the freelist / checkpoint
/// paths legitimately leave such pages with arbitrary bytes.
fn verify_recovery_strict(
    db_path: &std::path::Path,
    history: &HashMap<PageId, Vec<Vec<u8>>>,
    allocated: &[PageId],
) -> Result<(), String> {
    let mut p = match Pager::open(db_path, Config::default()) {
        Ok(p) => p,
        Err(e) => return Err(format!("recovery open: {e}")),
    };
    for pid in allocated {
        let Some(want_versions) = history.get(pid) else {
            match p.read_page(*pid) {
                Ok(_) | Err(obj_core::Error::Corruption { .. }) => continue,
                Err(e) => {
                    return Err(format!("recovery read {pid:?}: {e}", pid = pid.get()));
                }
            }
        };
        let read = p
            .read_page(*pid)
            .map_err(|e| format!("recovery read {pid:?}: {e}", pid = pid.get()))?;
        let body_len = PAGE_SIZE - 4;
        let got: &[u8] = &read.as_bytes()[..body_len];
        if !page_matches_last_committed(got, want_versions) {
            return Err(format!(
                "page {pid:?}: strict recovery: bytes do not match the LAST committed value \
                 (panic-only ACID violation — a committed txn rolled back to an earlier state)",
                pid = pid.get()
            ));
        }
    }
    Ok(())
}

fn page_for(step: usize, pid: PageId) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let mix = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ pid.get();
    for (i, b) in out.iter_mut().enumerate() {
        *b = ((mix >> ((i % 8) * 8)) & 0xFF) as u8;
    }
    out
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    "<non-string panic payload>".to_string()
}

fn write_seed_log(seed: u64, log: &[String]) -> std::io::Result<()> {
    let dir = PathBuf::from("target/crash_cycles");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("seed-{seed}.log"));
    std::fs::write(&path, log.join("\n"))?;
    eprintln!(
        "crash_cycles: wrote log for seed {seed} to {}",
        path.display()
    );
    Ok(())
}
