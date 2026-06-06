//! Deterministic, seed-controlled fault-injection harness.
//!
//! `FaultyFileHandle` is a test-only wrapper around [`FileHandle`] that
//! injects the kinds of failure modes real hardware exhibits — torn
//! writes, dropped fsyncs, short reads, bit-flips, sudden crashes — at
//! deterministic, seed-controlled points. The 10k crash-cycle test
//! drives this to exercise the WAL's recovery contract.
//!
//! Determinism is the load-bearing property: the same `FaultPlan` seed,
//! applied to the same sequence of operations, produces the same fault
//! sequence on every machine, every Rust toolchain, every CI run. The
//! PRNG is `rand_chacha::ChaCha8Rng`, chosen because its output is
//! specified at the algorithm level (not at the implementation level)
//! — so a future bump of `rand_chacha` cannot silently change the
//! seeds-to-faults mapping.
//!
//! # `unsafe` policy
//!
//! This module inherits `platform`'s `#![forbid(unsafe_code)]`. The
//! faults it injects are byte-level manipulations of the underlying
//! file via the existing [`FileHandle`] safe API; no syscalls of
//! `FaultyFileHandle`'s own.

use std::cell::RefCell;
use std::path::Path;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::error::{Error, Result};
use crate::platform::{FileBackend, FileHandle, SyncMode};

/// Panic message stamped onto deliberate-crash panics. The 10k cycle
/// test driver matches on this string at the [`catch_unwind`][cu]
/// boundary to distinguish injected crashes from genuine bugs.
///
/// [cu]: std::panic::catch_unwind
pub const FAULT_CRASH_MARKER: &str = "obj-core::fault::deliberate-crash";

/// Per-fault probabilities. All values are `f64` in `[0.0, 1.0]` and
/// clamped at construction.
#[derive(Debug, Clone, Copy)]
pub struct FaultPlan {
    /// Probability that a `write_all_at` call writes only a prefix of
    /// the bytes it was handed and silently returns Ok.
    pub torn_write_prob: f64,
    /// Probability that a `sync_data` / `sync_all` call is a silent
    /// no-op (i.e. the data is lost across a power loss).
    ///
    /// # Coverage limitation
    ///
    /// This models a *dropped fsync* by skipping the syscall, but it
    /// CANNOT model the resulting **data loss**. The bytes handed to a
    /// prior [`FileBackend::write_all_at`] have already been written
    /// through the real [`FileHandle`] into the kernel page cache.
    /// Dropping the subsequent fsync does not evict them, so a later
    /// read — in the *same process*, against the *same live kernel* —
    /// still observes the unsynced bytes. A true "fsync dropped, then
    /// power loss discards the page cache" sequence is unreachable
    /// from an in-process harness: only a real reboot, a forced cache
    /// drop, or a separate-machine fault injector can evict cached-but-
    /// unsynced pages.
    ///
    /// The harness therefore exercises the *control-flow* of the
    /// dropped-fsync path (the pager issues no real sync; recovery must
    /// still cope) but NOT the *durability* consequence. Coarser-grain
    /// power-loss durability is covered instead by the crash-cycle
    /// process-kill model in `obj-core/tests/crash_cycles.rs`, which
    /// treats an injected panic as a crash between two consistent
    /// commit points and asserts the reopen invariant. See the note on
    /// [`FaultyFileHandle::sync_data`] and the
    /// `dropped_fsync_on_checkpointed_main_file_recovers_via_wal_salt_match`
    /// test for the precise boundary of what is and is not verified.
    pub dropped_fsync_prob: f64,
    /// Probability that a `read_exact_at` call short-reads (the
    /// underlying FS would surface this as a [`std::io::Error`] of
    /// kind `UnexpectedEof`; we surface it as `Error::Io` with the
    /// same kind).
    pub short_read_prob: f64,
    /// Probability that a single bit in the buffer being written
    /// gets flipped before reaching the disk.
    pub bit_flip_prob: f64,
    /// If non-zero, the harness deliberately panics on the Nth
    /// `write_all_at` / `sync_data` / `sync_all` operation. `0`
    /// disables the trigger.
    pub crash_after_ops: u64,
    /// Seed for the deterministic PRNG. The seed alone uniquely
    /// determines every fault decision the plan makes.
    pub seed: u64,
}

impl Default for FaultPlan {
    fn default() -> Self {
        Self::noop(0)
    }
}

impl FaultPlan {
    /// Construct a plan that never injects any fault. Useful as a
    /// baseline against which to verify the harness is otherwise
    /// transparent.
    #[must_use]
    pub const fn noop(seed: u64) -> Self {
        Self {
            torn_write_prob: 0.0,
            dropped_fsync_prob: 0.0,
            short_read_prob: 0.0,
            bit_flip_prob: 0.0,
            crash_after_ops: 0,
            seed,
        }
    }

    /// Helper for the `[0.0, 1.0]` clamp used at probability sites.
    fn clamp01(v: f64) -> f64 {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 1.0)
        }
    }

    /// Construct a plan with caller-specified probabilities, each
    /// clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(
        seed: u64,
        torn_write_prob: f64,
        dropped_fsync_prob: f64,
        short_read_prob: f64,
        bit_flip_prob: f64,
        crash_after_ops: u64,
    ) -> Self {
        Self {
            torn_write_prob: Self::clamp01(torn_write_prob),
            dropped_fsync_prob: Self::clamp01(dropped_fsync_prob),
            short_read_prob: Self::clamp01(short_read_prob),
            bit_flip_prob: Self::clamp01(bit_flip_prob),
            crash_after_ops,
            seed,
        }
    }
}

/// Fault-injecting wrapper around [`FileHandle`].
///
/// The PRNG and operation counter live behind a `RefCell` so the
/// outer API stays `&self` (matching [`FileHandle`]); multi-threaded
/// access is not in scope.
pub struct FaultyFileHandle {
    inner: FileHandle,
    plan: FaultPlan,
    state: RefCell<FaultState>,
}

struct FaultState {
    rng: ChaCha8Rng,
    op_count: u64,
}

impl std::fmt::Debug for FaultyFileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaultyFileHandle")
            .field("inner", &self.inner)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl FaultyFileHandle {
    /// Wrap an existing [`FileHandle`] with the given `plan`. The
    /// PRNG is reseeded from `plan.seed`; the operation counter
    /// starts at zero.
    #[must_use]
    pub fn new(inner: FileHandle, plan: FaultPlan) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(plan.seed);
        Self {
            inner,
            plan,
            state: RefCell::new(FaultState { rng, op_count: 0 }),
        }
    }

    /// Convenience: open a file at `path` and wrap it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on syscall failure.
    pub fn open_or_create<P: AsRef<Path>>(path: P, plan: FaultPlan) -> Result<Self> {
        let inner = FileHandle::open_or_create(path)?;
        Ok(Self::new(inner, plan))
    }

    /// Borrow the wrapped [`FileHandle`]. Useful for tests that need
    /// to check the on-disk state past the harness.
    #[must_use]
    pub fn inner(&self) -> &FileHandle {
        &self.inner
    }

    /// Advance the operation counter and, if it reaches
    /// `plan.crash_after_ops`, panic with [`FAULT_CRASH_MARKER`].
    // allow: this panic IS the fault — a deliberate crash injector, not a bug; the crate-level deny(clippy::panic) must not reject it in --features fault-injection builds.
    #[allow(clippy::panic)]
    fn maybe_crash(&self, kind: &str) {
        let crash_at = self.plan.crash_after_ops;
        let mut state = self.state.borrow_mut();
        state.op_count = state.op_count.saturating_add(1);
        if crash_at != 0 && state.op_count == crash_at {
            drop(state);
            panic!("{FAULT_CRASH_MARKER}: {kind}");
        }
    }

    fn roll(&self, prob: f64) -> bool {
        if prob <= 0.0 {
            return false;
        }
        if prob >= 1.0 {
            return true;
        }
        let mut state = self.state.borrow_mut();
        state.rng.random::<f64>() < prob
    }

    fn rand_split(&self, len: usize) -> usize {
        if len <= 1 {
            return 0;
        }
        let mut state = self.state.borrow_mut();
        let r: u64 = state.rng.next_u64();
        let len_u64 = u64::try_from(len).unwrap_or(u64::MAX);
        let kept = r % len_u64;
        usize::try_from(kept).unwrap_or(len - 1)
    }
}

impl FileBackend for FaultyFileHandle {
    fn len(&self) -> Result<u64> {
        self.inner.len()
    }

    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        if self.roll(self.plan.short_read_prob) {
            let kind = std::io::ErrorKind::UnexpectedEof;
            return Err(Error::Io(std::io::Error::new(
                kind,
                "fault-injected short read",
            )));
        }
        self.inner.read_exact_at(buf, offset)
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
        self.maybe_crash("write_all_at");
        if self.roll(self.plan.torn_write_prob) {
            let kept = self.rand_split(buf.len());
            if kept == 0 {
                return Ok(());
            }
            return self.inner.write_all_at(&buf[..kept], offset);
        }
        if self.roll(self.plan.bit_flip_prob) && !buf.is_empty() {
            let mut buf_copy = buf.to_vec();
            let byte_idx: usize = {
                let mut state = self.state.borrow_mut();
                let r = state.rng.next_u64();
                let len_u64 = u64::try_from(buf_copy.len()).unwrap_or(u64::MAX);
                usize::try_from(r % len_u64).unwrap_or(0)
            };
            let bit_idx: u8 = {
                let mut state = self.state.borrow_mut();
                u8::try_from(state.rng.next_u32() & 0x7).unwrap_or(0)
            };
            buf_copy[byte_idx] ^= 1u8 << bit_idx;
            return self.inner.write_all_at(&buf_copy, offset);
        }
        self.inner.write_all_at(buf, offset)
    }

    fn set_len(&self, new_len: u64) -> Result<()> {
        self.inner.set_len(new_len)
    }

    fn sync_data(&self, mode: SyncMode) -> Result<()> {
        self.maybe_crash("sync_data");
        if self.roll(self.plan.dropped_fsync_prob) {
            return Ok(());
        }
        self.inner.sync_data(mode)
    }

    fn sync_all(&self) -> Result<()> {
        self.maybe_crash("sync_all");
        if self.roll(self.plan.dropped_fsync_prob) {
            return Ok(());
        }
        self.inner.sync_all()
    }
}

/// Test-support helpers for the `tests` submodule below. Module-level
/// so the path is short; private so external crates can't depend on
/// the panic-message format.
#[cfg(test)]
struct FaultBackendTestSupport;

#[cfg(test)]
impl FaultBackendTestSupport {
    // allow: payload type is fixed by catch_unwind, which yields exactly Box<dyn Any + Send>; we borrow it to downcast without taking ownership, so &Box is the natural shape here.
    #[allow(clippy::borrowed_box)]
    fn extract_panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<String>() {
            return s.clone();
        }
        if let Some(s) = payload.downcast_ref::<&'static str>() {
            return (*s).to_string();
        }
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{FaultBackendTestSupport, FaultPlan, FaultyFileHandle, FAULT_CRASH_MARKER};
    use crate::platform::{FileBackend, SyncMode};
    use tempfile::TempDir;

    fn make(dir: &TempDir, name: &str, plan: FaultPlan) -> FaultyFileHandle {
        let path = dir.path().join(name);
        FaultyFileHandle::open_or_create(&path, plan).expect("open faulty")
    }

    #[test]
    fn noop_plan_is_transparent() {
        let dir = TempDir::new().expect("tempdir");
        let h = make(&dir, "noop.bin", FaultPlan::noop(0));
        h.set_len(4096).expect("set_len");
        h.write_all_at(&[0xAAu8; 4096], 0).expect("write");
        let mut out = [0u8; 4096];
        h.read_exact_at(&mut out, 0).expect("read");
        assert_eq!(out[0], 0xAA);
        h.sync_data(SyncMode::Full).expect("sync");
    }

    #[test]
    fn torn_write_truncates_on_disk() {
        let dir = TempDir::new().expect("tempdir");
        let plan = FaultPlan::new(123, 1.0, 0.0, 0.0, 0.0, 0);
        let h = make(&dir, "torn.bin", plan);
        h.set_len(4096).expect("set_len");
        let inner_path = dir.path().join("torn.bin");
        std::fs::write(&inner_path, vec![0xFFu8; 4096]).expect("prefill");
        let buf = vec![0x00u8; 4096];
        h.write_all_at(&buf, 0).expect("torn write returns Ok");
        let on_disk = std::fs::read(&inner_path).expect("readback");
        let zeros = on_disk.iter().take_while(|&&b| b == 0).count();
        assert!(
            zeros < 4096,
            "torn write must NOT write the whole buffer; got {zeros} zero bytes",
        );
    }

    #[test]
    fn dropped_fsync_is_silent() {
        let dir = TempDir::new().expect("tempdir");
        let plan = FaultPlan::new(7, 0.0, 1.0, 0.0, 0.0, 0);
        let h = make(&dir, "df.bin", plan);
        h.sync_data(SyncMode::Full)
            .expect("dropped fsync surfaces Ok");
        h.sync_all().expect("dropped sync_all surfaces Ok");
    }

    #[test]
    fn short_read_returns_unexpected_eof() {
        let dir = TempDir::new().expect("tempdir");
        let plan = FaultPlan::new(9, 0.0, 0.0, 1.0, 0.0, 0);
        let h = make(&dir, "sr.bin", plan);
        h.set_len(4096).expect("set_len");
        h.inner()
            .write_all_at(&[0x55u8; 4096], 0)
            .expect("ground truth write");
        let mut out = [0u8; 4096];
        let err = h.read_exact_at(&mut out, 0).expect_err("short read");
        let crate::error::Error::Io(io) = err else {
            panic!("expected Error::Io");
        };
        assert_eq!(io.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn crash_after_ops_panics_with_marker() {
        let dir = TempDir::new().expect("tempdir");
        let plan = FaultPlan::new(0, 0.0, 0.0, 0.0, 0.0, 1);
        let path = dir.path().join("crash.bin");
        let h = FaultyFileHandle::open_or_create(&path, plan).expect("open");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = h.write_all_at(&[0u8; 16], 0);
        }));
        let panic_payload = result.expect_err("deliberate panic");
        let msg = FaultBackendTestSupport::extract_panic_message(&panic_payload);
        assert!(
            msg.contains(FAULT_CRASH_MARKER),
            "panic must carry the crash marker; got {msg}",
        );
    }

    #[test]
    fn bit_flip_changes_one_byte() {
        let dir = TempDir::new().expect("tempdir");
        let plan = FaultPlan::new(42, 0.0, 0.0, 0.0, 1.0, 0);
        let path = dir.path().join("bf.bin");
        let h = FaultyFileHandle::open_or_create(&path, plan).expect("open");
        h.set_len(4096).expect("set_len");
        let buf = vec![0u8; 256];
        h.write_all_at(&buf, 0).expect("write");
        let on_disk = std::fs::read(&path).expect("readback");
        let diffs: Vec<(usize, u8)> = on_disk
            .iter()
            .take(256)
            .enumerate()
            .filter(|(_, &b)| b != 0)
            .map(|(i, &b)| (i, b))
            .collect();
        assert_eq!(diffs.len(), 1, "exactly one byte must differ");
        let (_, b) = diffs[0];
        assert_eq!(b.count_ones(), 1, "exactly one bit must be flipped");
    }
}
