//! Contract test for the no-compression-feature
//! refusal path.
//!
//! Constructs a synthetic 4 KiB page-0 header with
//! `format_minor = 1` and `feature_flags = 0x0000_0001` (the
//! compression bit), writes it as a fresh `.obj` file, and asserts
//! that a `Db::open_with` against a default-features build returns
//! `Error::FormatFeatureUnsupported { feature: "compression" }`
//! BEFORE any page-level operation. This is the user-facing
//! contract for "this build can't open compressed files".
//!
//! The test deliberately does NOT carry the `#![cfg(feature =
//! "compression")]` gate — it MUST compile and pass under the
//! default-features build so a regression in the refusal path
//! surfaces in baseline CI.

use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

use obj::Db;
#[cfg(not(feature = "compression"))]
use obj::Error;

const PAGE_SIZE: usize = 4096;
const HEADER_CRC_OFFSET: usize = PAGE_SIZE - 4;

/// Synthesise a minimum-viable page-0 header that opens to a
/// compression-capable file: magic `OBJF`, `format_major = 0`,
/// `format_minor = 1`, `page_size = 4096`, `feature_flags = 0x01`,
/// `page_count = 1`, everything else zero. The header CRC is
/// recomputed last so `decode_header` accepts it; the open path
/// then trips the feature-refusal check.
fn synth_compression_capable_header() -> [u8; PAGE_SIZE] {
    let mut buf = [0u8; PAGE_SIZE];
    buf[0..4].copy_from_slice(b"OBJF");
    buf[4..6].copy_from_slice(&0u16.to_le_bytes());
    buf[6..8].copy_from_slice(&1u16.to_le_bytes());
    buf[8..10].copy_from_slice(
        &u16::try_from(PAGE_SIZE)
            .expect("PAGE_SIZE fits in u16")
            .to_le_bytes(),
    );
    buf[10..14].copy_from_slice(&1u32.to_le_bytes());
    buf[16..24].copy_from_slice(&1u64.to_le_bytes());

    let crc = obj_core::pager::checksum::crc32c(&buf[..HEADER_CRC_OFFSET]);
    buf[HEADER_CRC_OFFSET..HEADER_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

#[cfg(not(feature = "compression"))]
#[test]
fn open_refuses_format_minor_one_without_compression_feature() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("compressed_capable.obj");

    let header = synth_compression_capable_header();
    {
        let mut f = File::create(&path).expect("create");
        f.write_all(&header).expect("write header");
        f.sync_all().expect("sync");
    }

    let err = Db::open(&path).expect_err("default build must refuse");

    match err {
        Error::FormatFeatureUnsupported { feature } => {
            assert_eq!(feature, "compression");
        }
        other => panic!(
            "expected Error::FormatFeatureUnsupported {{ feature: \"compression\" }}; \
             got {other:?}",
        ),
    }
}

/// Compile-time test: with the `compression` feature on, the
/// `Error::FormatFeatureUnsupported` variant still exists. The
/// behavioral side of the refusal contract has nothing to test
/// in this build configuration (compressed files open
/// successfully), but compiling this no-op against the variant
/// catches accidental removal under `--all-features`.
#[cfg(feature = "compression")]
#[test]
fn format_feature_unsupported_variant_exists_under_compression() {
    let err = obj::Error::FormatFeatureUnsupported {
        feature: "compression",
    };
    assert!(matches!(
        err,
        obj::Error::FormatFeatureUnsupported { feature } if feature == "compression"
    ));
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("c.obj");
    let header = synth_compression_capable_header();
    {
        let mut f = File::create(&path).expect("create");
        f.write_all(&header).expect("write header");
        f.sync_all().expect("sync");
    }
    let _ok = Db::open(&path).expect("compression feature build opens fine");
}
