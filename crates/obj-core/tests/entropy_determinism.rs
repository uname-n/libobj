//! Integration coverage for the injectable [`Entropy`] seam (issue #6).
//!
//! Verifies the load-bearing determinism property: two freshly-created
//! databases opened with a [`SeededEntropy`] carrying the *same* seed
//! stamp byte-identical WAL generation salts, while distinct seeds
//! diverge. The WAL salt is the first on-disk byte-stream the seam
//! controls, so pinning it here guards the whole determinism contract
//! at the public `Pager::open_with_env` boundary.

use std::path::Path;
use std::sync::Arc;

use obj_core::pager::{wal_path_for, Config, Pager};
use obj_core::platform::{Entropy, SeededEntropy};
use tempfile::TempDir;

/// Byte offset of the 32-bit generation salt inside a WAL header,
/// matching `wal::write_wal_header` (`buf[12..16]`).
const WAL_SALT_OFFSET: usize = 12;

/// Create a fresh database at `path` drawing all salts/nonces from
/// `entropy`, then read the 4-byte WAL generation salt back off disk.
fn fresh_db_wal_salt(path: &Path, entropy: Arc<dyn Entropy>) -> u32 {
    {
        let _p = Pager::open_with_env(path, Config::default(), entropy).expect("open fresh db");
        // Drop closes the pager; the WAL sidecar keeps its header salt.
    }
    let wal_bytes = std::fs::read(wal_path_for(path)).expect("read wal sidecar");
    let mut salt = [0u8; 4];
    salt.copy_from_slice(&wal_bytes[WAL_SALT_OFFSET..WAL_SALT_OFFSET + 4]);
    u32::from_le_bytes(salt)
}

#[test]
fn same_seed_two_fresh_dbs_get_identical_wal_salt() {
    let dir = TempDir::new().expect("tempdir");
    let a = fresh_db_wal_salt(&dir.path().join("a.obj"), Arc::new(SeededEntropy::new(0x51E5)));
    let b = fresh_db_wal_salt(&dir.path().join("b.obj"), Arc::new(SeededEntropy::new(0x51E5)));
    assert_eq!(a, b, "same seed must yield identical WAL salt");
}

#[test]
fn different_seeds_get_different_wal_salt() {
    let dir = TempDir::new().expect("tempdir");
    let a = fresh_db_wal_salt(&dir.path().join("a.obj"), Arc::new(SeededEntropy::new(1)));
    let b = fresh_db_wal_salt(&dir.path().join("b.obj"), Arc::new(SeededEntropy::new(2)));
    assert_ne!(a, b, "distinct seeds must diverge");
}
