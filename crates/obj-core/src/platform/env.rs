//! Injectable entropy source (L0).
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
//! This module inherits `platform`'s no-`unsafe` posture; it performs
//! no syscalls of its own.

#![forbid(unsafe_code)]

use std::sync::Mutex;

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

#[cfg(test)]
mod tests {
    use super::{Entropy, OsEntropy, SeededEntropy};

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
}
