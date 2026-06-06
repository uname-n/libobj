//! 1 000 000-operation `BTreeMap` oracle property test.
//!
//! The plan: drive the [`obj_core::btree::BTree`] and an oracle
//! [`std::collections::BTreeMap`] through one million randomised
//! operations drawn from `{insert, delete, get, range, range_unbounded}`.
//! After each operation, the externally-observable state of the two
//! data structures is compared:
//!
//! - `get(key)` returns the same `Option<Vec<u8>>` from both.
//! - `range(start..end)` returns the same `Vec<(Vec<u8>, Vec<u8>)>`
//!   from both.
//! - `range_unbounded()` returns the same complete `Vec<(Vec<u8>,
//!   Vec<u8>)>` from both.
//!
//! The first divergence at any operation aborts the cycle with a
//! readable mismatch report and writes the full operation log to
//! `target/btree_oracle/seed-<N>.log` for post-mortem replay.
//!
//! The test is `#[ignore]` (same convention as `crash_cycles`) so a
//! plain `cargo test --workspace` does not pay the ~5-15 minute cost
//! of a 1M-op cycle. Invoke it explicitly:
//!
//! ```text
//! cargo test --release --test btree_oracle -- --ignored --test-threads=1
//! ```
//!
//! Shard selection for CI uses the `OBJ_BTREE_ORACLE_START` and
//! `OBJ_BTREE_ORACLE_END` environment variables. When unset, the test
//! runs one full cycle (`[0, 1)`).
//!
//! ```text
//! OBJ_BTREE_ORACLE_START=42 OBJ_BTREE_ORACLE_END=43 \
//!     cargo test --release --test btree_oracle -- --ignored --test-threads=1
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Bound;
use std::path::PathBuf;

use obj_core::btree::BTree;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Number of operations driven per seed.
/// Fixed at 1 000 000; do not lower without a clear rationale.
const OPS_PER_CYCLE: usize = 1_000_000;

const DEFAULT_START: u64 = 0;
const DEFAULT_END: u64 = 1;

/// Weights (out of 100) for each operation. Inserts dominate so the
/// tree grows; deletes target present keys 80% of the time so the
/// tree size stays in a sensible range (low tens of thousands of
/// entries on average across a 1M-op run). `range_unbounded` is
/// expensive (drains the whole iterator), so it is rare; bounded
/// `range` queries are cheap because they're bounded by two random
/// pivots and skip when ill-ordered.
const OP_WEIGHTS: OpWeights = OpWeights {
    insert: 35,
    delete: 35,
    get: 15,
    range: 14,
    range_unbounded: 1,
};

struct OpWeights {
    insert: u32,
    delete: u32,
    get: u32,
    range: u32,
    range_unbounded: u32,
}

#[derive(Debug, Clone)]
enum Op {
    Insert {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        key: Vec<u8>,
    },
    Get {
        key: Vec<u8>,
    },
    Range {
        start: Bound<Vec<u8>>,
        end: Bound<Vec<u8>>,
    },
    RangeUnbounded,
}

#[test]
#[ignore = "1M-op oracle; ~5-15 min in release. Run via `cargo test --release --test btree_oracle -- --ignored`"]
fn btree_oracle_seed_range() {
    let (start, end) = read_seed_range();
    let mut failures: Vec<u64> = Vec::new();
    let mut total: u64 = 0;
    for seed in start..end {
        total += 1;
        if let Err(msg) = run_one_cycle(seed) {
            eprintln!("OBJ_BTREE_ORACLE_SEED={seed} FAILED: {msg}");
            failures.push(seed);
        } else {
            eprintln!("btree_oracle: seed {seed} passed ({OPS_PER_CYCLE} ops)");
        }
    }
    assert!(
        failures.is_empty(),
        "btree_oracle: {} of {} seeds failed: first 10 = {:?}",
        failures.len(),
        total,
        &failures[..failures.len().min(10)]
    );
    eprintln!("btree_oracle: {total} seed(s) passed (range {start}..{end})");
}

fn read_seed_range() -> (u64, u64) {
    let start = std::env::var("OBJ_BTREE_ORACLE_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_START);
    let end = std::env::var("OBJ_BTREE_ORACLE_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_END);
    (start, end)
}

fn run_one_cycle(seed: u64) -> Result<(), String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pager =
        Pager::<FileHandle>::memory(Config::default()).map_err(|e| format!("pager: {e}"))?;
    let mut tree =
        BTree::<FileHandle>::empty(&mut pager).map_err(|e| format!("tree empty: {e}"))?;
    let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut log_tail: Vec<String> = Vec::with_capacity(256);
    for step in 0..OPS_PER_CYCLE {
        let op = draw_op(&mut rng, &oracle);
        log_op(&mut log_tail, step, &op);
        if let Err(msg) = run_op(&mut pager, &mut tree, &mut oracle, &op) {
            write_seed_log(seed, &log_tail).ok();
            return Err(format!("step {step}: {msg}"));
        }
    }
    Ok(())
}

fn run_op(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    op: &Op,
) -> Result<(), String> {
    match op {
        Op::Insert { key, value } => check_insert(pager, tree, oracle, key, value),
        Op::Delete { key } => check_delete(pager, tree, oracle, key),
        Op::Get { key } => check_get(pager, tree, oracle, key),
        Op::Range { start, end } => check_range(pager, tree, oracle, start, end),
        Op::RangeUnbounded => check_range_unbounded(pager, tree, oracle),
    }
}

fn check_insert(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    key: &[u8],
    value: &[u8],
) -> Result<(), String> {
    let key_present = oracle.contains_key(key);
    let res = tree.insert(pager, key, value);
    if key_present {
        if !matches!(res, Err(obj_core::Error::BTreeKeyExists)) {
            return Err(format!("insert dup: expected BTreeKeyExists, got {res:?}"));
        }
    } else {
        res.map_err(|e| format!("insert: {e}"))?;
        oracle.insert(key.to_vec(), value.to_vec());
    }
    let got = tree
        .get(pager, key)
        .map_err(|e| format!("insert post-get: {e}"))?;
    let want = oracle.get(key).cloned();
    if got != want {
        return Err(format!(
            "insert post-get mismatch: got {got:?} want {want:?}"
        ));
    }
    Ok(())
}

fn check_delete(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    oracle: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    key: &[u8],
) -> Result<(), String> {
    let want = oracle.remove(key).is_some();
    let got = tree
        .delete(pager, key)
        .map_err(|e| format!("delete: {e}"))?;
    if got != want {
        return Err(format!("delete presence mismatch: got {got} want {want}"));
    }
    let after = tree
        .get(pager, key)
        .map_err(|e| format!("delete post-get: {e}"))?;
    if after.is_some() {
        return Err(format!("delete post-get returned Some: {after:?}"));
    }
    Ok(())
}

fn check_get(
    pager: &mut Pager<FileHandle>,
    tree: &BTree<FileHandle>,
    oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
    key: &[u8],
) -> Result<(), String> {
    let got = tree.get(pager, key).map_err(|e| format!("get: {e}"))?;
    let want = oracle.get(key).cloned();
    if got != want {
        return Err(format!("get mismatch: got {got:?} want {want:?}"));
    }
    Ok(())
}

fn check_range(
    pager: &mut Pager<FileHandle>,
    tree: &BTree<FileHandle>,
    oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
    start: &Bound<Vec<u8>>,
    end: &Bound<Vec<u8>>,
) -> Result<(), String> {
    if !bounds_well_ordered(start, end) {
        return Ok(());
    }
    let iter = tree
        .range(pager, (start.clone(), end.clone()))
        .map_err(|e| format!("range: {e}"))?;
    let mut got: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for step in iter {
        let item = step.map_err(|e| format!("range step: {e}"))?;
        got.push(item);
    }
    let want: Vec<(Vec<u8>, Vec<u8>)> = oracle
        .range::<Vec<u8>, _>((start.as_ref(), end.as_ref()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if got != want {
        return Err(format!(
            "range mismatch: bounds=({start:?}, {end:?}) got_len={} want_len={}",
            got.len(),
            want.len()
        ));
    }
    Ok(())
}

fn check_range_unbounded(
    pager: &mut Pager<FileHandle>,
    tree: &BTree<FileHandle>,
    oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> Result<(), String> {
    let iter = tree.iter(pager).map_err(|e| format!("iter: {e}"))?;
    let mut got: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(oracle.len());
    for step in iter {
        let item = step.map_err(|e| format!("iter step: {e}"))?;
        got.push(item);
    }
    if got.len() != oracle.len() {
        return Err(format!(
            "iter length mismatch: got {} want {}",
            got.len(),
            oracle.len()
        ));
    }
    for ((gk, gv), (wk, wv)) in got.iter().zip(oracle.iter()) {
        if gk != wk || gv != wv {
            return Err(format!(
                "iter mismatch at key {gk:?}: got value {gv:?} want value {wv:?}"
            ));
        }
    }
    Ok(())
}

fn bounds_well_ordered(start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> bool {
    match (start, end) {
        (Bound::Unbounded, _) | (_, Bound::Unbounded) => true,
        (Bound::Excluded(s), Bound::Excluded(e)) => s.as_slice() < e.as_slice(),
        (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e)) => {
            s.as_slice() <= e.as_slice()
        }
    }
}

fn draw_op(rng: &mut ChaCha8Rng, oracle: &BTreeMap<Vec<u8>, Vec<u8>>) -> Op {
    let total = OP_WEIGHTS.insert
        + OP_WEIGHTS.delete
        + OP_WEIGHTS.get
        + OP_WEIGHTS.range
        + OP_WEIGHTS.range_unbounded;
    let pick = rng.random_range(0u32..total);
    let mut cumulative = 0u32;
    cumulative += OP_WEIGHTS.insert;
    if pick < cumulative {
        return Op::Insert {
            key: random_key(rng),
            value: random_value(rng),
        };
    }
    cumulative += OP_WEIGHTS.delete;
    if pick < cumulative {
        return Op::Delete {
            key: pick_delete_key(rng, oracle),
        };
    }
    cumulative += OP_WEIGHTS.get;
    if pick < cumulative {
        return Op::Get {
            key: pick_get_key(rng, oracle),
        };
    }
    cumulative += OP_WEIGHTS.range;
    if pick < cumulative {
        let (start, end) = random_bounds(rng, oracle);
        return Op::Range { start, end };
    }
    Op::RangeUnbounded
}

/// 80% chance: pick a key currently in the oracle (exercises the
/// "delete present key" path). 20% chance: random key (likely a
/// no-op delete). This matches the in-module 10k oracle in
/// `delete.rs`.
fn pick_delete_key(rng: &mut ChaCha8Rng, oracle: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<u8> {
    if !oracle.is_empty() && rng.random_range(0u32..5) > 0 {
        let n = oracle.len();
        let pick = rng.random_range(0..n);
        oracle.keys().nth(pick).cloned().unwrap_or_default()
    } else {
        random_key(rng)
    }
}

/// 50% chance: pick a present key (exercises hit path); 50% random.
fn pick_get_key(rng: &mut ChaCha8Rng, oracle: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<u8> {
    if !oracle.is_empty() && rng.random_range(0u32..2) == 0 {
        let n = oracle.len();
        let pick = rng.random_range(0..n);
        oracle.keys().nth(pick).cloned().unwrap_or_default()
    } else {
        random_key(rng)
    }
}

fn random_bounds(
    rng: &mut ChaCha8Rng,
    oracle: &BTreeMap<Vec<u8>, Vec<u8>>,
) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
    let s = match rng.random_range(0u32..4) {
        0 => Bound::Unbounded,
        1 => Bound::Included(pick_get_key(rng, oracle)),
        2 => Bound::Excluded(pick_get_key(rng, oracle)),
        _ => Bound::Included(random_key(rng)),
    };
    let e = match rng.random_range(0u32..4) {
        0 => Bound::Unbounded,
        1 => Bound::Included(pick_get_key(rng, oracle)),
        2 => Bound::Excluded(pick_get_key(rng, oracle)),
        _ => Bound::Excluded(random_key(rng)),
    };
    (s, e)
}

/// Keys: random byte strings of length 1..=64 — exercises the full
/// inline-key range without ever hitting `Error::BTreeKeyTooLarge`.
/// The alphabet is `[a..z]` so that random draws collide often
/// enough that delete-present targeting actually finds keys; pure
/// 0..=255 random bytes essentially never collide.
fn random_key(rng: &mut ChaCha8Rng) -> Vec<u8> {
    let len = rng.random_range(1u32..=64);
    (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect()
}

/// Values: random byte strings of length 0..=256. Exercises the
/// inline-value path including the empty-value edge case.
fn random_value(rng: &mut ChaCha8Rng) -> Vec<u8> {
    let len = rng.random_range(0u32..=256);
    (0..len).map(|_| rng.random()).collect()
}

/// Append a one-line description of `op` to the log tail buffer.
/// Caps the buffer at 256 entries: B+tree bugs that take 1M ops to
/// surface are almost always caught within the last few ops, and
/// the seed alone is sufficient to replay deterministically.
fn log_op(log: &mut Vec<String>, step: usize, op: &Op) {
    if log.len() >= 256 {
        log.remove(0);
    }
    log.push(format!("{step}: {}", describe_op(op)));
}

fn describe_op(op: &Op) -> String {
    match op {
        Op::Insert { key, value } => {
            format!("insert key={} value_len={}", hex(key), value.len())
        }
        Op::Delete { key } => format!("delete key={}", hex(key)),
        Op::Get { key } => format!("get key={}", hex(key)),
        Op::Range { start, end } => format!(
            "range start={} end={}",
            describe_bound(start),
            describe_bound(end)
        ),
        Op::RangeUnbounded => String::from("range_unbounded"),
    }
}

fn describe_bound(b: &Bound<Vec<u8>>) -> String {
    match b {
        Bound::Included(k) => format!("Included({})", hex(k)),
        Bound::Excluded(k) => format!("Excluded({})", hex(k)),
        Bound::Unbounded => String::from("Unbounded"),
    }
}

fn hex(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len() * 2);
    for &byte in b.iter().take(32) {
        let _ = write!(out, "{byte:02x}");
    }
    if b.len() > 32 {
        out.push_str("...");
    }
    out
}

fn write_seed_log(seed: u64, log: &[String]) -> std::io::Result<()> {
    let dir = PathBuf::from("target/btree_oracle");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("seed-{seed}.log"));
    std::fs::write(&path, log.join("\n"))?;
    eprintln!(
        "btree_oracle: wrote log (last {} ops) for seed {seed} to {}",
        log.len(),
        path.display()
    );
    Ok(())
}
