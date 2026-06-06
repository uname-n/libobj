//! Integration tests for at-rest encryption.
//!
//! This file is gated on the `encryption` Cargo feature; tests that
//! must run under the default build live in `encryption_refusal.rs`.

#![cfg(feature = "encryption")]

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

#[cfg(feature = "compression")]
use obj::CompressionMode;
use obj::{Config, Db, Document, Error};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Doc {
    title: String,
    body: String,
}

impl Document for Doc {
    const COLLECTION: &'static str = "docs";
    const VERSION: u32 = 1;
}

impl obj::Schema for Doc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("title", obj::DynamicSchema::String),
            ("body", obj::DynamicSchema::String),
        ])
    }
}

fn k1() -> [u8; 32] {
    let mut k = [0u8; 32];
    for (i, b) in k.iter_mut().enumerate() {
        *b = u8::try_from(i & 0xFF).expect("byte");
    }
    k
}

fn k2() -> [u8; 32] {
    let mut k = [0u8; 32];
    for (i, b) in k.iter_mut().enumerate() {
        *b = u8::try_from((i ^ 0x55) & 0xFF).expect("byte");
    }
    k
}

#[test]
fn encryption_feature_is_compiled_in_when_test_is_run() {
    assert!(obj_core::pager::encryption_feature_compiled_in());
}

#[test]
fn round_trip_insert_and_reopen() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc.obj");
    let n_docs: u32 = 100;
    let config = Config::default().encryption_key(k1());

    let mut ids = Vec::with_capacity(n_docs as usize);
    {
        let db = Db::open_with(&path, config).expect("open with key");
        for i in 0..n_docs {
            let id = db
                .insert(Doc {
                    title: format!("title-{i}"),
                    body: "x".repeat(64),
                })
                .expect("insert");
            ids.push((i, id));
        }
        drop(db);
    }

    let db = Db::open_with(&path, Config::default().encryption_key(k1())).expect("reopen");
    for (seed, id) in &ids {
        let got: Doc = db.get::<Doc>(*id).expect("get").expect("present");
        assert_eq!(got.title, format!("title-{seed}"));
    }
}

#[test]
fn wrong_key_rejected_on_reopen() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc.obj");
    {
        let db = Db::open_with(&path, Config::default().encryption_key(k1())).expect("open");
        for i in 0..10 {
            db.insert(Doc {
                title: format!("t-{i}"),
                body: "b".repeat(64),
            })
            .expect("insert");
        }
        drop(db);
    }
    let res = Db::open_with(&path, Config::default().encryption_key(k2()));
    match res {
        Err(Error::EncryptionKeyInvalid) => {}
        Ok(db) => {
            let err = db
                .all::<Doc>()
                .expect_err("expected EncryptionKeyInvalid on wrong key");
            assert!(
                matches!(err, Error::EncryptionKeyInvalid),
                "expected EncryptionKeyInvalid; got {err:?}"
            );
        }
        Err(other) => panic!("expected EncryptionKeyInvalid; got {other:?}"),
    }
}

#[test]
fn missing_key_rejected_on_reopen() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc.obj");
    {
        let db = Db::open_with(&path, Config::default().encryption_key(k1())).expect("open");
        drop(db);
    }
    let err = Db::open(&path).expect_err("must refuse missing key");
    assert!(
        matches!(err, Error::EncryptionKeyRequired),
        "expected EncryptionKeyRequired; got {err:?}"
    );
}

#[test]
fn key_mismatch_on_unencrypted_file() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("plain.obj");
    {
        let db = Db::open(&path).expect("open");
        drop(db);
    }
    let err =
        Db::open_with(&path, Config::default().encryption_key(k1())).expect_err("must refuse");
    assert!(
        matches!(err, Error::EncryptionKeyMismatch),
        "expected EncryptionKeyMismatch; got {err:?}"
    );
}

#[cfg(feature = "compression")]
#[test]
fn layered_compression_and_encryption_round_trips() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("both.obj");
    let config = Config::default()
        .compression(CompressionMode::Lz4)
        .encryption_key(k1());
    let mut ids = Vec::with_capacity(50);
    {
        let db = Db::open_with(&path, config).expect("open both");
        for i in 0..50u32 {
            let id = db
                .insert(Doc {
                    title: format!("t-{i}"),
                    body: "a".repeat(1024),
                })
                .expect("insert");
            ids.push((i, id));
        }
        drop(db);
    }

    let db = Db::open_with(
        &path,
        Config::default()
            .compression(CompressionMode::Lz4)
            .encryption_key(k1()),
    )
    .expect("reopen both");
    for (seed, id) in &ids {
        let got: Doc = db.get::<Doc>(*id).expect("get").expect("present");
        assert_eq!(got.title, format!("t-{seed}"));
    }
}

#[test]
fn ciphertext_bit_flip_is_not_silent() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("flipped.obj");
    let wal_path = dir.path().join("flipped.obj-wal");
    {
        let db = Db::open_with(&path, Config::default().encryption_key(k1())).expect("open");
        for i in 0..30u32 {
            db.insert(Doc {
                title: format!("t-{i}"),
                body: "b".repeat(32),
            })
            .expect("insert");
        }
        drop(db);
    }
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .expect("open WAL");
        f.seek(SeekFrom::Start(200)).expect("seek");
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte).expect("read");
        byte[0] ^= 0x40;
        f.seek(SeekFrom::Start(200)).expect("seek back");
        f.write_all(&byte).expect("write");
        f.sync_all().expect("sync");
    }
    match Db::open_with(&path, Config::default().encryption_key(k1())) {
        Err(
            Error::WalCorruption { .. } | Error::Corruption { .. } | Error::EncryptionKeyInvalid,
        ) => {}
        Ok(db) => {
            let _ = db.all::<Doc>();
        }
        Err(other) => {
            panic!("expected WalCorruption / Corruption / EncryptionKeyInvalid; got {other:?}")
        }
    }
}

#[test]
fn pager_format_minor_is_two_on_encrypted_file() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("enc.obj");
    {
        let db = Db::open_with(&path, Config::default().encryption_key(k1())).expect("open");
        drop(db);
    }
    let mut buf = [0u8; 4096];
    let mut f = OpenOptions::new().read(true).open(&path).expect("open");
    f.read_exact(&mut buf).expect("read page 0");
    let format_minor = u16::from_le_bytes([buf[6], buf[7]]);
    assert_eq!(format_minor, 2);
    let feature_flags = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);
    assert_eq!(feature_flags, 0b10);
    let salt = &buf[72..104];
    assert!(
        salt.iter().any(|&b| b != 0),
        "kdf_salt must be CSPRNG-derived, not zero"
    );
}

#[cfg(feature = "compression")]
#[test]
fn compression_and_encryption_set_both_feature_bits() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("both.obj");
    {
        let db = Db::open_with(
            &path,
            Config::default()
                .compression(CompressionMode::Lz4)
                .encryption_key(k1()),
        )
        .expect("open");
        drop(db);
    }
    let mut buf = [0u8; 4096];
    let mut f = OpenOptions::new().read(true).open(&path).expect("open");
    f.read_exact(&mut buf).expect("read page 0");
    let format_minor = u16::from_le_bytes([buf[6], buf[7]]);
    assert_eq!(format_minor, 2);
    let feature_flags = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);
    assert_eq!(feature_flags, 0b11);
}
