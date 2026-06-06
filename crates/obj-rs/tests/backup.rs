//! `Db::backup_to` acceptance tests.
//!
//! Clean DB → backup → reopen backup → all docs present. Cross-
//! platform; uses `tempfile::TempDir` to host both source and
//! destination so the test does not depend on the host filesystem
//! layout.
//!
//! The deeper concurrent-backup-vs-writer test
//! lives in a separate file.

use obj::{Db, Document, Error, IntegrityReport};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Note {
    title: String,
    body: String,
}

impl Document for Note {
    const COLLECTION: &'static str = "notes";
    const VERSION: u32 = 1;
}

impl obj::Schema for Note {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("title", obj::DynamicSchema::String),
            ("body", obj::DynamicSchema::String),
        ])
    }
}

#[test]
fn backup_round_trips_a_clean_db() {
    let dir = TempDir::new().expect("tmp");
    let src_path = dir.path().join("src.obj");
    let dst_path = dir.path().join("backup.obj");

    let mut ids: Vec<(obj::Id, Note)> = Vec::new();
    {
        let db = Db::open(&src_path).expect("open source");
        for i in 0..32u32 {
            let note = Note {
                title: format!("note-{i}"),
                body: format!("body-{i}"),
            };
            let id = db.insert(note.clone()).expect("insert");
            ids.push((id, note));
        }
        db.backup_to(&dst_path).expect("backup_to");
    }

    let backup_db = Db::open(&dst_path).expect("open backup");
    for (id, expected) in &ids {
        let got: Option<Note> = backup_db.get(*id).expect("get");
        assert_eq!(
            got.as_ref(),
            Some(expected),
            "doc {} survived backup",
            id.get()
        );
    }

    let report: IntegrityReport = backup_db.integrity_check().expect("integrity");
    assert!(
        report.is_ok(),
        "backup must pass integrity_check; failures = {:?}",
        report.failures,
    );
}

#[test]
fn backup_rejects_existing_destination() {
    let dir = TempDir::new().expect("tmp");
    let src_path = dir.path().join("src.obj");
    let dst_path = dir.path().join("existing.obj");
    std::fs::write(&dst_path, b"already here").expect("create existing");
    let db = Db::open(&src_path).expect("open");
    db.insert(Note {
        title: "t".into(),
        body: "b".into(),
    })
    .expect("insert");
    let err = db.backup_to(&dst_path).expect_err("must refuse");
    assert!(
        matches!(err, Error::BackupDestinationExists { .. }),
        "expected BackupDestinationExists; got {err:?}",
    );
}

#[test]
fn backup_rejected_on_memory_pager() {
    let db = Db::memory().expect("memory db");
    db.insert(Note {
        title: "t".into(),
        body: "b".into(),
    })
    .expect("insert");
    let dir = TempDir::new().expect("tmp");
    let dst_path = dir.path().join("never-written.obj");
    let err = db.backup_to(&dst_path).expect_err("memory backup rejected");
    assert!(
        matches!(err, Error::BackupNotSupportedForMemoryPager),
        "expected BackupNotSupportedForMemoryPager; got {err:?}",
    );
    assert!(
        !dst_path.exists(),
        "destination must NOT be created on the rejected backup",
    );
}

#[test]
fn backup_isolates_from_post_snapshot_writes() {
    let dir = TempDir::new().expect("tmp");
    let src_path = dir.path().join("src.obj");
    let dst_path = dir.path().join("backup.obj");
    let db = Db::open(&src_path).expect("open source");
    let pre_id = db
        .insert(Note {
            title: "pre".into(),
            body: "before-backup".into(),
        })
        .expect("pre-insert");
    db.backup_to(&dst_path).expect("backup");
    let post_id = db
        .insert(Note {
            title: "post".into(),
            body: "after-backup".into(),
        })
        .expect("post-insert");

    let backup_db = Db::open(&dst_path).expect("open backup");
    assert!(
        backup_db.get::<Note>(pre_id).expect("get pre").is_some(),
        "pre-backup doc must be present",
    );
    assert!(
        backup_db.get::<Note>(post_id).expect("get post").is_none(),
        "post-backup doc must NOT be present",
    );
}
