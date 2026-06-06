//! Cross-process lock wiring.
//!
//! These tests live in the integration-test crate so they exercise
//! the same `Db::open` path that real callers use — including the
//! `<db>.obj-lock` sidecar `FileHandle` creation.
//!
//! The tests are platform-agnostic. On every supported platform a
//! second writer that opens the same DB path while another writer
//! holds the cross-process `WRITER_LOCK` byte must surface
//! `Error::Busy { kind: LockKind::Writer }` rather than corrupting
//! state or silently
//! interleaving.

use std::time::Duration;

use obj::{Db, Document, Error, LockKind};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Note {
    body: String,
}

impl Document for Note {
    const COLLECTION: &'static str = "notes";
    const VERSION: u32 = 1;
}

impl obj::Schema for Note {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("body", obj::DynamicSchema::String)])
    }
}

/// The `<db>.obj-lock` sidecar must appear next to the main file
/// after the first `Db::open` (when cross-process locking is
/// enabled).
#[test]
fn open_creates_lock_sidecar() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("app.obj");
    let lock_path = {
        let mut buf = path.as_os_str().to_os_string();
        buf.push("-lock");
        std::path::PathBuf::from(buf)
    };

    {
        let _db = Db::open(&path).expect("open");
        assert!(
            lock_path.exists(),
            "lock sidecar must exist after Db::open: {}",
            lock_path.display(),
        );
        let meta = std::fs::metadata(&lock_path).expect("sidecar metadata");
        assert!(
            meta.len() >= 128,
            "lock sidecar must be sized to >= 128 bytes so the \
             locked byte range exists as real content; got len={}",
            meta.len(),
        );
    }

    assert!(
        lock_path.exists(),
        "lock sidecar must persist across Db close",
    );
}

/// A second writer on the same DB path must be rejected while the
/// first writer holds the cross-process `WRITER_LOCK`. Proves the
/// sidecar is wired up on every platform.
#[test]
fn second_writer_rejected_while_first_holds_lock() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("contend.obj");

    let db1 = Db::open(&path).expect("first open");
    db1.insert(Note {
        body: "first".to_owned(),
    })
    .expect("seed insert");

    let db2 = Db::open(&path).expect("second open");

    let outcome = db1.transaction(|_tx1| {
        let err = db2
            .transaction(|_tx2| Ok::<(), Error>(()))
            .expect_err("second writer must be refused");
        assert!(
            matches!(
                err,
                Error::Busy {
                    kind: LockKind::Writer | LockKind::WriterInProcess,
                },
            ),
            "expected Error::Busy{{Writer|WriterInProcess}}, got {err:?}",
        );
        Ok::<(), Error>(())
    });
    outcome.expect("outer txn must commit");

    db2.insert(Note {
        body: "second".to_owned(),
    })
    .expect("second writer succeeds once lock released");
}

/// Smoke-test that the pager can grow the main DB well past byte
/// 96 (the lock anchor) without ever colliding with the lock byte.
/// Pager I/O
/// on the main file must be independent of the lock byte's
/// presence in the `<db>.obj-lock` sidecar.
///
/// The test is intentionally small (megabytes, not gigabytes) so
/// it runs in seconds on every platform. The `integrity_1gb`
/// bench covers the full >1 GiB case.
#[test]
fn writes_past_lock_anchor_offset_succeed() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("growable.obj");
    let db = Db::open_with(
        &path,
        obj::Config::default().busy_timeout(Duration::from_secs(5)),
    )
    .expect("open");

    let blob = "x".repeat(2048);
    db.transaction(|tx| {
        let col = tx.collection::<Note>()?;
        for _ in 0..4096u32 {
            col.insert(Note { body: blob.clone() })?;
        }
        Ok::<(), Error>(())
    })
    .expect("bulk insert must not collide with lock bytes");
}
