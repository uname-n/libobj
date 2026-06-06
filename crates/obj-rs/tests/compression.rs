//! Integration tests for LZ4 page compression.
//!
//! Exercised only when the `compression` Cargo feature is enabled.
//! Covers:
//!
//! - round-trip of 1000 documents under
//!   `Config::compression(CompressionMode::Lz4)`,
//! - file-size sanity: a compressible workload yields a smaller
//!   `.obj` file with compression on than the same workload
//!   without,
//! - mixed-page case: one big compressible doc + one tiny
//!   incompressible doc, both decode after reopen.
//!
//! All tests use TempDir-backed file paths.

#![cfg(feature = "compression")]

use std::fmt::Write as _;

use obj::{CompressionMode, Config, Db, Document};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Blob {
    payload: String,
}

impl Document for Blob {
    const COLLECTION: &'static str = "blobs";
    const VERSION: u32 = 1;
}

impl obj::Schema for Blob {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([("payload", obj::DynamicSchema::String)])
    }
}

/// Build a roughly 500-byte highly-compressible payload that varies
/// with `seed` so each document is distinct but every page still
/// compresses well (`u8`-only ASCII runs).
fn compressible_payload(seed: u32) -> String {
    let mut s = String::with_capacity(512);
    for _ in 0..32 {
        s.push_str("compression_phase3_lz4_obj_");
    }
    write!(&mut s, "seed_{seed}").expect("write to String never fails");
    s
}

#[test]
fn round_trip_1000_compressed_docs() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("compressed.obj");
    let cfg = Config::default().compression(CompressionMode::Lz4);

    let mut ids = Vec::with_capacity(1000);
    {
        let db = Db::open_with(&path, cfg.clone()).expect("open lz4");
        for seed in 0u32..1000 {
            let id = db
                .insert(Blob {
                    payload: compressible_payload(seed),
                })
                .expect("insert");
            ids.push((seed, id));
        }
        drop(db);
    }

    {
        let db = Db::open_with(&path, cfg).expect("reopen lz4");
        for (seed, id) in &ids {
            let got: Blob = db.get::<Blob>(*id).expect("get").expect("doc present");
            assert_eq!(got.payload, compressible_payload(*seed), "seed {seed}");
        }
    }
}

#[test]
fn compressed_pages_use_less_disk_content_than_uncompressed() {
    let dir = TempDir::new().expect("tmp");
    let uncompressed_path = dir.path().join("plain.obj");
    let compressed_path = dir.path().join("lz4.obj");

    let workload: Vec<String> = (0u32..1500)
        .map(|seed| {
            let mut s = String::with_capacity(1024);
            for _ in 0..64 {
                s.push_str("aaaaaaaaaaaaaaaa");
            }
            write!(&mut s, "{seed:08}").expect("write to String never fails");
            s
        })
        .collect();

    let plain_cfg = Config::default();
    let lz4_cfg = Config::default().compression(CompressionMode::Lz4);

    {
        let db = Db::open_with(&uncompressed_path, plain_cfg).expect("open plain");
        for payload in &workload {
            db.insert(Blob {
                payload: payload.clone(),
            })
            .expect("insert");
        }
    }
    {
        let db = Db::open_with(&compressed_path, lz4_cfg).expect("open lz4");
        for payload in &workload {
            db.insert(Blob {
                payload: payload.clone(),
            })
            .expect("insert");
        }
    }

    let plain_bytes = std::fs::read(&uncompressed_path).expect("plain bytes");
    let lz4_bytes = std::fs::read(&compressed_path).expect("lz4 bytes");

    let plain_nonzero = plain_bytes.iter().filter(|&&b| b != 0).count();
    let lz4_nonzero = lz4_bytes.iter().filter(|&&b| b != 0).count();
    assert!(
        lz4_nonzero < plain_nonzero,
        "compressed pages must contain fewer non-zero bytes than \
         uncompressed pages for a compressible workload \
         (lz4_nonzero = {lz4_nonzero}, plain_nonzero = {plain_nonzero})",
    );
    assert!(
        lz4_nonzero * 4 < plain_nonzero * 3,
        "compressed non-zero byte count ({lz4_nonzero}) should be \
         well under 75% of uncompressed ({plain_nonzero}) for an LZ4-friendly workload",
    );
}

#[test]
fn mixed_compressible_and_incompressible_docs_round_trip() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("mixed.obj");
    let cfg = Config::default().compression(CompressionMode::Lz4);

    let compressible = Blob {
        payload: compressible_payload(0),
    };
    let incompressible = Blob {
        payload: (0u8..=255)
            .cycle()
            .take(128)
            .map(|b| char::from(b.wrapping_mul(37).wrapping_add(11) | 0x20))
            .collect::<String>(),
    };

    let (id_c, id_i) = {
        let db = Db::open_with(&path, cfg.clone()).expect("open");
        let id_c = db.insert(compressible.clone()).expect("insert compr");
        let id_i = db.insert(incompressible.clone()).expect("insert incompr");
        (id_c, id_i)
    };

    {
        let db = Db::open_with(&path, cfg).expect("reopen");
        let got_c: Blob = db.get::<Blob>(id_c).expect("get compr").expect("present");
        let got_i: Blob = db.get::<Blob>(id_i).expect("get incompr").expect("present");
        assert_eq!(got_c, compressible);
        assert_eq!(got_i, incompressible);
    }
}
