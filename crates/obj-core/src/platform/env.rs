//! Injectable environment seams (L0): entropy and clock.
//!
//! Salt and nonce generation used to draw directly from the OS —
//! `rand::rng()` for WAL / KDF salts, `getrandom` for AEAD nonces.
//! That made two runs of the *same* seed produce different on-disk
//! bytes, which blocks byte-for-byte determinism under the
//! deterministic-simulation-testing (DST) harness. This module
//! introduces the [`Entropy`] seam: a cold-path trait object the
//! pager and WAL draw every salt and nonce from, so a test can
//! substitute a seeded, reproducible source while production keeps
//! the OS CSPRNG.
//!
//! # Design
//!
//! - [`Entropy`] is a trait object (`Arc<dyn Entropy>`), **not** a
//!   generic parameter. Salt / nonce generation is a cold path; a
//!   generic would force `Pager<F, E>` / `Wal<F, E>` across the whole
//!   crate for no hot-path benefit.
//! - [`OsEntropy`] is the production source. It mirrors the pre-seam
//!   `cfg` split: `getrandom` under the `encryption` feature, the
//!   `rand` crate's OS-seeded CSPRNG otherwise.
//! - [`SeededEntropy`] is the DST source. It wraps a
//!   [`ChaCha8Rng`] behind a [`Mutex`] (it must be `Sync` — the
//!   pager and WAL cross threads), reseeded from a `u64`. `ChaCha8` is
//!   chosen for the same reason as `platform::fault`: its output is
//!   specified at the algorithm level, so a future `rand_chacha` bump
//!   cannot silently change the seed-to-bytes mapping.
//!
//! # Clock
//!
//! The [`Clock`] seam is the time counterpart to [`Entropy`]. The
//! write-serialization gate and the cross-process-lock retry loops
//! measure elapsed time and back off with `std::thread::sleep`; both
//! block real wall-clock time and read a real monotonic clock, which
//! makes the deterministic-simulation harness slow and non-reproducible.
//!
//! - [`Clock`] is a trait object (`Arc<dyn Clock>`) for the same
//!   cold-path reason as [`Entropy`]: the backoff/timeout paths are not
//!   hot, so a generic parameter across the txn/lock layers would buy
//!   nothing. Time is a plain `u64` millis counter rather than an opaque
//!   [`Instant`] so the trait is dyn-friendly (no associated types).
//! - [`SystemClock`] is the production source: `now_millis` reads a
//!   process-start [`Instant`] baseline; `sleep` is `std::thread::sleep`.
//! - [`SimClock`] is the DST source: virtual time advances ONLY on
//!   `sleep`, which bumps an [`AtomicU64`]; `now_millis` reads that
//!   counter without blocking. Every `elapsed >= timeout` decision then
//!   becomes deterministic and instant.
//!
//! This module inherits `platform`'s no-`unsafe` posture; it performs
//! no syscalls of its own.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use rand::{RngCore as _, SeedableRng as _};
use rand_chacha::ChaCha8Rng;

/// A source of entropy for WAL / KDF salts and page / WAL nonces.
///
/// Implementations MUST be `Send + Sync`: the pager and WAL that hold
/// an `Arc<dyn Entropy>` are moved and shared across threads. `Debug`
/// is required so the owning `Pager` / `Wal` can keep deriving it.
pub trait Entropy: Send + Sync + core::fmt::Debug {
    /// Fill `buf` with entropy. Infallible — a source that cannot
    /// produce bytes must fall back rather than error, because the
    /// salt / nonce call sites have no meaningful recovery.
    fn fill_bytes(&self, buf: &mut [u8]);

    /// Draw a `u32`. Default implementation pulls four bytes through
    /// [`Entropy::fill_bytes`] and reads them little-endian, so a
    /// seeded source yields a reproducible `u32` stream.
    fn next_u32(&self) -> u32 {
        let mut b = [0u8; 4];
        self.fill_bytes(&mut b);
        u32::from_le_bytes(b)
    }
}

/// Production entropy source backed by the operating system CSPRNG.
///
/// Zero-sized and cheap to clone; construct with [`OsEntropy`] or
/// [`OsEntropy::default`].
#[derive(Debug, Default, Clone, Copy)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill_bytes(&self, buf: &mut [u8]) {
        os_fill(buf);
    }
}

/// Fill `buf` from the OS CSPRNG under the `encryption` feature,
/// preferring `getrandom` (the pre-seam nonce source). If `getrandom`
/// is unavailable, fall back to `rand`'s OS-seeded generator so the
/// infallible [`Entropy::fill_bytes`] contract holds without an
/// `unwrap` / `panic` in this crate.
#[cfg(feature = "encryption")]
fn os_fill(buf: &mut [u8]) {
    if getrandom::getrandom(buf).is_ok() {
        return;
    }
    rand::rng().fill_bytes(buf);
}

/// Fill `buf` from `rand`'s OS-seeded CSPRNG (the pre-seam salt
/// source) on builds without the `encryption` feature.
#[cfg(not(feature = "encryption"))]
fn os_fill(buf: &mut [u8]) {
    rand::rng().fill_bytes(buf);
}

/// Deterministic entropy source for the DST harness.
///
/// Wraps a [`ChaCha8Rng`] behind a [`Mutex`] so the type is `Sync`
/// (the pager and WAL that hold it cross threads). Two instances
/// constructed from the same seed produce identical byte streams for
/// the same sequence of draws.
#[derive(Debug)]
pub struct SeededEntropy {
    rng: Mutex<ChaCha8Rng>,
}

impl SeededEntropy {
    /// Construct a source seeded from `seed`. The seed alone
    /// determines every byte the source subsequently produces.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(seed)),
        }
    }
}

impl Entropy for SeededEntropy {
    fn fill_bytes(&self, buf: &mut [u8]) {
        // A prior holder that panicked mid-fill would poison the lock,
        // but the RNG state is still well-defined; recover the guard
        // rather than propagate a panic through this infallible call.
        let mut guard = match self.rng.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.fill_bytes(buf);
    }
}

/// A source of monotonic time and blocking sleeps for the write-
/// serialization gate and the cross-process-lock retry loops.
///
/// Implementations MUST be `Send + Sync`: the [`Clock`] is held behind
/// an `Arc<dyn Clock>` on the txn environment, which is shared and moved
/// across threads. `Debug` is required so the owning env can keep
/// deriving it.
///
/// Time is exposed as a `u64` millisecond counter rather than an opaque
/// [`Instant`] so the trait carries no associated types and stays
/// dyn-safe. Callers measure elapsed time as
/// `clock.now_millis().saturating_sub(start)` and compare against a
/// timeout expressed in milliseconds.
pub trait Clock: Send + Sync + core::fmt::Debug {
    /// Current time as a millisecond counter. Monotonic (never runs
    /// backward across calls on the same clock); the absolute origin is
    /// unspecified — only differences are meaningful.
    fn now_millis(&self) -> u64;

    /// Block for approximately `d`. Production sleeps the calling
    /// thread; the simulation clock instead advances virtual time (see
    /// [`SimClock`]).
    fn sleep(&self, d: Duration);
}

/// Process-start baseline for [`SystemClock::now_millis`]. Initialised
/// once, lazily, on first use so every [`SystemClock`] in the process
/// shares one monotonic origin and the type stays zero-sized.
fn process_start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Production clock backed by the OS monotonic clock and
/// `std::thread::sleep`.
///
/// Zero-sized and cheap to clone; construct with [`SystemClock`] or
/// [`SystemClock::default`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_millis(&self) -> u64 {
        // Saturate rather than panic: the process would have to run for
        // ~584 million years to overflow a u64 of milliseconds.
        u64::try_from(process_start().elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// Deterministic virtual clock for the DST harness.
///
/// Virtual time starts at 0 and advances ONLY when [`Clock::sleep`] is
/// called, by the sleep duration (in whole milliseconds). Because
/// [`Clock::now_millis`] simply reads the counter and `sleep` never
/// touches the real clock, every backoff/timeout loop that draws its
/// time from a `SimClock` terminates instantly and identically on every
/// run — no wall-clock wait.
///
/// The counter lives in an [`AtomicU64`] so the clock is `Sync` (the txn
/// env that holds it crosses threads).
#[derive(Debug, Default)]
pub struct SimClock {
    now_millis: AtomicU64,
}

impl SimClock {
    /// Construct a virtual clock whose time starts at 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            now_millis: AtomicU64::new(0),
        }
    }
}

impl Clock for SimClock {
    fn now_millis(&self) -> u64 {
        self.now_millis.load(Ordering::Relaxed)
    }

    fn sleep(&self, d: Duration) {
        // Saturate the per-sleep advance; the accumulator itself
        // saturates on add so virtual time can never wrap.
        let ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
        let mut cur = self.now_millis.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_add(ms);
            match self.now_millis.compare_exchange_weak(
                cur,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Clock, Entropy, OsEntropy, SeededEntropy, SimClock, SystemClock};
    use std::time::Duration;

    #[test]
    fn same_seed_same_fill_stream() {
        let a = SeededEntropy::new(0xDEAD_BEEF);
        let b = SeededEntropy::new(0xDEAD_BEEF);
        let mut ba = [0u8; 64];
        let mut bb = [0u8; 64];
        a.fill_bytes(&mut ba);
        b.fill_bytes(&mut bb);
        assert_eq!(ba, bb, "same seed must yield identical bytes");
    }

    #[test]
    fn same_seed_same_u32_stream() {
        let a = SeededEntropy::new(7);
        let b = SeededEntropy::new(7);
        let seq_a: Vec<u32> = (0..8).map(|_| a.next_u32()).collect();
        let seq_b: Vec<u32> = (0..8).map(|_| b.next_u32()).collect();
        assert_eq!(seq_a, seq_b, "same seed must yield identical u32 stream");
    }

    #[test]
    fn different_seeds_differ() {
        let a = SeededEntropy::new(1);
        let b = SeededEntropy::new(2);
        let mut ba = [0u8; 32];
        let mut bb = [0u8; 32];
        a.fill_bytes(&mut ba);
        b.fill_bytes(&mut bb);
        assert_ne!(ba, bb, "distinct seeds must diverge");
    }

    #[test]
    fn interleaved_salt_then_nonce_is_reproducible() {
        // Mirror the pager's real draw order: a 4-byte salt (next_u32)
        // followed by a 24-byte nonce. Same seed => same pair.
        let draw = |seed: u64| {
            let e = SeededEntropy::new(seed);
            let salt = e.next_u32();
            let mut nonce = [0u8; 24];
            e.fill_bytes(&mut nonce);
            (salt, nonce)
        };
        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42).0, draw(43).0);
    }

    #[test]
    fn os_entropy_fills_and_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OsEntropy>();
        assert_send_sync::<SeededEntropy>();
        let os = OsEntropy;
        let mut buf = [0u8; 16];
        os.fill_bytes(&mut buf);
        // Overwhelmingly unlikely to be all-zero from a real CSPRNG.
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn clocks_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SystemClock>();
        assert_send_sync::<SimClock>();
    }

    #[test]
    fn sim_clock_advances_only_on_sleep() {
        let c = SimClock::new();
        assert_eq!(c.now_millis(), 0, "virtual time starts at 0");
        // now_millis is a pure read: repeated calls do not advance time.
        assert_eq!(c.now_millis(), 0);
        c.sleep(Duration::from_millis(5));
        assert_eq!(c.now_millis(), 5, "sleep advances virtual time");
        c.sleep(Duration::from_millis(95));
        assert_eq!(c.now_millis(), 100, "advances are cumulative");
    }

    #[test]
    fn sim_clock_sleep_does_not_block() {
        // A multi-second virtual sleep must return effectively instantly.
        let c = SimClock::new();
        let wall = std::time::Instant::now();
        c.sleep(Duration::from_secs(30));
        assert!(
            wall.elapsed() < Duration::from_millis(500),
            "SimClock::sleep must not block on the real clock",
        );
        assert_eq!(c.now_millis(), 30_000);
    }

    #[test]
    fn system_clock_time_is_monotonic() {
        let c = SystemClock;
        let a = c.now_millis();
        c.sleep(Duration::from_millis(2));
        let b = c.now_millis();
        assert!(b >= a, "system clock must not run backward");
    }
}
