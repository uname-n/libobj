//! DST determinism campaign — same seed ⇒ byte-identical on-disk state.
//!
//! This is the load-bearing proof of the "full determinism seam" (the
//! [`Entropy`](obj_core::platform::Entropy) source, issue #6). With a
//! [`SeededEntropy`](obj_core::platform::SeededEntropy) injected via
//! [`Pager::open_with_env`], every salt and nonce the engine samples is
//! a pure function of the seed. Two runs of the same seeded workload,
//! against two fresh temp directories, must therefore produce
//! byte-for-byte identical main files AND WAL sidecars — a property that
//! was impossible before salt/nonce sampling was funnelled through the
//! injectable source.
//!
//! Each seed:
//! 1. Derives a workload (page count, op count, op sequence) from a
//!    [`ChaCha8Rng`] seeded by the `u64` seed — identical to the
//!    `crash_cycles.rs` op grammar (Write/Commit/Checkpoint/Reopen/
//!    Alloc/Free).
//! 2. Runs that workload TWICE, each in its own temp dir, each opening
//!    the pager with a fresh `SeededEntropy::new(seed)`.
//! 3. Asserts the two runs agree on: op outcomes (Ok/Err shape), the
//!    final main-file bytes, and the final WAL-file bytes.
//!
//! The `encryption`-gated variant proves the same identity holds with
//! page nonces in play (nonces are now reproducible).
//! `different_seeds_produce_different_bytes` proves the entropy seam is
//! actually load-bearing: distinct seeds must yield distinct bytes, or
//! the RNG is being ignored entirely.
//!
//! The tests are `#[ignore]` so a plain `cargo test --workspace` pays
//! nothing. Invoke explicitly:
//!
//! ```text
//! cargo test --test determinism -- --ignored --test-threads=1
//! cargo test --features encryption --test determinism -- --ignored --test-threads=1
//! ```
//!
//! Shard selection for CI uses the `DETERMINISM_START` and
//! `DETERMINISM_END` environment variables; when unset the range is
//! `[0, 1000)`. A failing seed prints `SEED=<N>` to stderr and writes a
//! diff summary to `target/determinism/seed-<N>.log`. Re-run one seed:
//!
//! ```text
//! DETERMINISM_START=42 DETERMINISM_END=43 \
//!     cargo test --test determinism -- --ignored --test-threads=1
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use obj_core::pager::page::{Page, PageId, PAGE_SIZE};
use obj_core::pager::{wal_path_for, Config, Pager};
use obj_core::platform::{FileHandle, SeededEntropy};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const DEFAULT_START: u64 = 0;
const DEFAULT_END: u64 = 1000;

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

#[test]
#[ignore = "DST determinism campaign; run via `cargo test --test determinism -- --ignored`"]
fn determinism_seed_range_plain() {
    run_seed_range(plain_config, "determinism");
}

#[cfg(feature = "encryption")]
#[test]
#[ignore = "DST determinism campaign (encryption); run via `cargo test --features encryption --test determinism -- --ignored`"]
fn determinism_seed_range_encrypted() {
    run_seed_range(encrypted_config, "determinism_encrypted");
}

/// Guard against a regression where the [`Entropy`] seam is ignored and
/// bytes become seed-independent: distinct seeds must diverge on disk.
#[test]
#[ignore = "DST entropy-seam guard; run via `cargo test --test determinism -- --ignored`"]
fn different_seeds_produce_different_bytes() {
    assert_seeds_diverge(plain_config, "plain");
}

#[cfg(feature = "encryption")]
#[test]
#[ignore = "DST entropy-seam guard (encryption); run via `cargo test --features encryption --test determinism -- --ignored`"]
fn different_seeds_produce_different_bytes_encrypted() {
    assert_seeds_diverge(encrypted_config, "encrypted");
}

fn plain_config() -> Config {
    Config::default()
}

#[cfg(feature = "encryption")]
fn encrypted_config() -> Config {
    Config::default().with_encryption_key(Some([0x42u8; 32]))
}

fn run_seed_range(make_config: fn() -> Config, label: &str) {
    let (start, end) = read_seed_range();
    let mut failures: Vec<u64> = Vec::new();
    for seed in start..end {
        if let Err(msg) = run_one_seed(seed, make_config) {
            eprintln!("SEED={seed} ({label}) FAILED: {msg}");
            let _ = write_seed_log(seed, &msg);
            failures.push(seed);
        }
    }
    assert!(
        failures.is_empty(),
        "{label}: {} of {} seeds failed: first 10 = {:?}",
        failures.len(),
        end.saturating_sub(start),
        &failures[..failures.len().min(10)]
    );
    eprintln!(
        "{label}: {} seeds passed (range {start}..{end})",
        end.saturating_sub(start)
    );
}

/// Run seeds `0..8` once each and assert they do not all collapse to the
/// same bytes. A single WAL-salt collision (1 in 2^32) cannot make all
/// eight identical, so `distinct > 1` cleanly separates "entropy is
/// consumed" from "entropy is ignored".
fn assert_seeds_diverge(make_config: fn() -> Config, label: &str) {
    let mut mains: Vec<Vec<u8>> = Vec::new();
    for seed in 0u64..8 {
        let opener = Opener { seed, make_config };
        let run = run_workload(&opener).expect("workload should complete");
        mains.push(run.main);
    }
    let distinct = mains
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 1,
        "{label}: all {} seeds produced identical main bytes — the entropy seam is being ignored",
        mains.len()
    );
}

fn run_one_seed(seed: u64, make_config: fn() -> Config) -> Result<(), String> {
    let opener = Opener { seed, make_config };
    let a = run_workload(&opener)?;
    let b = run_workload(&opener)?;
    if a.outcomes != b.outcomes {
        return Err(format!(
            "op outcomes diverged: run A = {:?}, run B = {:?}",
            a.outcomes, b.outcomes
        ));
    }
    if a.main != b.main {
        return Err(first_diff("main", &a.main, &b.main));
    }
    if a.wal != b.wal {
        return Err(first_diff("wal", &a.wal, &b.wal));
    }
    Ok(())
}

fn read_seed_range() -> (u64, u64) {
    let start = std::env::var("DETERMINISM_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_START);
    let end = std::env::var("DETERMINISM_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_END);
    (start, end)
}

/// The observable result of one workload run: the final on-disk bytes of
/// both files plus the per-op outcome tags. Two same-seed runs must
/// match on all three.
struct RunResult {
    main: Vec<u8>,
    wal: Vec<u8>,
    outcomes: Vec<String>,
}

/// Factory for the pager used by a single run. Each open (initial and
/// every reopen) draws from a fresh `SeededEntropy::new(seed)`, mirroring
/// production where every `Pager::open` constructs a fresh `OsEntropy`.
struct Opener {
    seed: u64,
    make_config: fn() -> Config,
}

impl Opener {
    fn open(&self, path: &Path) -> Result<Pager<FileHandle>, String> {
        let entropy = Arc::new(SeededEntropy::new(self.seed));
        let mut p = Pager::open_with_env(path, (self.make_config)(), entropy)
            .map_err(|e| format!("open: {e}"))?;
        p.begin_txn();
        Ok(p)
    }
}

fn run_workload(opener: &Opener) -> Result<RunResult, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(opener.seed);
    let n_pages: u32 = rng.random_range(4u32..12);
    let n_ops: u32 = rng.random_range(20u32..60);
    let ops: Vec<Op> = (0..n_ops).map(|_| random_op(&mut rng, n_pages)).collect();
    let dir = TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
    let db_path = dir.path().join("determ.obj");
    let mut state = WorkState::open(opener, &db_path, n_pages)?;
    let mut outcomes: Vec<String> = Vec::with_capacity(ops.len());
    for (step, op) in ops.iter().enumerate() {
        let tag = state.apply_op(step, *op);
        let stop = tag.ends_with(":err");
        outcomes.push(tag);
        if stop {
            break;
        }
    }
    state.close()?;
    let main = std::fs::read(&db_path).map_err(|e| format!("read main: {e}"))?;
    let wal = std::fs::read(wal_path_for(&db_path)).unwrap_or_default();
    Ok(RunResult {
        main,
        wal,
        outcomes,
    })
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

/// In-flight state for a single run. Only `allocated` is tracked (to pick
/// page-ids for write/free) — correctness is proven by byte comparison
/// against the twin run, so no expected-value bookkeeping is needed.
struct WorkState<'o> {
    pager: Option<Pager<FileHandle>>,
    db_path: PathBuf,
    allocated: Vec<PageId>,
    opener: &'o Opener,
}

impl<'o> WorkState<'o> {
    fn open(opener: &'o Opener, db_path: &Path, n_pages: u32) -> Result<Self, String> {
        let mut p = opener.open(db_path)?;
        let cap = usize::try_from(n_pages).unwrap_or(0);
        let mut allocated = Vec::with_capacity(cap);
        for _ in 0..n_pages {
            allocated.push(p.alloc_page().map_err(|e| format!("alloc: {e}"))?);
        }
        let _ = p.commit().map_err(|e| format!("setup commit: {e}"))?;
        Ok(Self {
            pager: Some(p),
            db_path: db_path.to_path_buf(),
            allocated,
            opener,
        })
    }

    fn pager_mut(&mut self) -> &mut Pager<FileHandle> {
        self.pager.as_mut().expect("pager present during workload")
    }

    fn close(&mut self) -> Result<(), String> {
        if let Some(p) = self.pager.take() {
            p.close().map_err(|e| format!("close: {e}"))?;
        }
        Ok(())
    }

    fn apply_op(&mut self, step: usize, op: Op) -> String {
        match op {
            Op::Write(idx) => self.op_write(step, idx),
            Op::Commit => outcome_tag("commit", &self.pager_mut().commit().map(|_| ())),
            Op::Checkpoint => outcome_tag("checkpoint", &self.pager_mut().checkpoint()),
            Op::Reopen => self.op_reopen(),
            Op::Alloc => self.op_alloc(),
            Op::Free(idx) => self.op_free(idx),
        }
    }

    fn op_write(&mut self, step: usize, idx: usize) -> String {
        if self.allocated.is_empty() {
            return "write:skip".to_string();
        }
        let pid = self.allocated[idx % self.allocated.len()];
        let body = page_for(step, pid);
        let mut page = Page::zeroed();
        let body_len = body.len().min(PAGE_SIZE - 4);
        page.as_bytes_mut()[..body_len].copy_from_slice(&body[..body_len]);
        outcome_tag("write", &self.pager_mut().write_page(pid, &page))
    }

    fn op_reopen(&mut self) -> String {
        let Some(p) = self.pager.take() else {
            return "reopen:err".to_string();
        };
        if p.close().is_err() {
            return "reopen:err".to_string();
        }
        match self.opener.open(&self.db_path) {
            Ok(p) => {
                self.pager = Some(p);
                "reopen:ok".to_string()
            }
            Err(_) => "reopen:err".to_string(),
        }
    }

    fn op_alloc(&mut self) -> String {
        match self.pager_mut().alloc_page() {
            Ok(pid) => {
                self.allocated.push(pid);
                outcome_tag("alloc", &self.pager_mut().commit().map(|_| ()))
            }
            Err(_) => "alloc:err".to_string(),
        }
    }

    fn op_free(&mut self, idx: usize) -> String {
        if self.allocated.is_empty() {
            return "free:skip".to_string();
        }
        let pos = idx % self.allocated.len();
        let victim = self.allocated.remove(pos);
        if self.pager_mut().free_page(victim).is_err() {
            return "free:err".to_string();
        }
        outcome_tag("free", &self.pager_mut().commit().map(|_| ()))
    }
}

/// Map an op's result to a stable, run-independent tag. Only the Ok/Err
/// shape is recorded — the error's `Display` is deliberately excluded
/// because it can carry the (per-run, distinct) temp-dir path, which
/// would spuriously diverge between the two runs of the same seed.
fn outcome_tag(name: &str, r: &obj_core::Result<()>) -> String {
    if r.is_ok() {
        format!("{name}:ok")
    } else {
        format!("{name}:err")
    }
}

fn page_for(step: usize, pid: PageId) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let mix = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ pid.get();
    for (i, b) in out.iter_mut().enumerate() {
        *b = ((mix >> ((i % 8) * 8)) & 0xFF) as u8;
    }
    out
}

/// Human-readable first point of divergence between two byte buffers, for
/// the failure message / seed log.
fn first_diff(which: &str, a: &[u8], b: &[u8]) -> String {
    if a.len() != b.len() {
        return format!("{which}: length differs (A={}, B={})", a.len(), b.len());
    }
    match a.iter().zip(b.iter()).position(|(x, y)| x != y) {
        Some(i) => format!(
            "{which}: bytes differ at offset {i} (A={a:#04x}, B={b:#04x})",
            a = a[i],
            b = b[i]
        ),
        None => format!("{which}: reported unequal but no byte mismatch found"),
    }
}

fn write_seed_log(seed: u64, msg: &str) -> std::io::Result<()> {
    let dir = PathBuf::from("target/determinism");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("seed-{seed}.log"));
    std::fs::write(&path, msg)?;
    eprintln!("determinism: wrote log for seed {seed} to {}", path.display());
    Ok(())
}
