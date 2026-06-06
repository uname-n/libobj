//! Regenerate `include/libobj.h` to a temp directory and byte-
//! compare against the committed copy. Fails if a Rust signature
//! change forgot to update the committed header.
//!
//! This test is the canonical CI guard against C-ABI drift.

use std::fs;
use std::path::PathBuf;

#[test]
fn committed_header_matches_freshly_generated() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let committed = crate_dir.join("include").join("libobj.h");
    assert!(
        committed.exists(),
        "expected committed header at {}; run `cargo build -p libobj`",
        committed.display(),
    );

    let config =
        cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")).expect("cbindgen.toml parses");
    let bindings = cbindgen::Builder::new()
        .with_config(config)
        .with_crate(&crate_dir)
        .generate()
        .expect("cbindgen generates");

    let tmp = tempfile::TempDir::new().expect("tmp");
    let regen = tmp.path().join("libobj.h");
    bindings.write_to_file(&regen);

    let committed_bytes = fs::read(&committed).expect("read committed header");
    let regen_bytes = fs::read(&regen).expect("read regen header");

    if committed_bytes != regen_bytes {
        let committed_str = String::from_utf8_lossy(&committed_bytes);
        let regen_str = String::from_utf8_lossy(&regen_bytes);
        eprintln!(
            "header drift: committed = {} bytes, regenerated = {} bytes",
            committed_bytes.len(),
            regen_bytes.len(),
        );
        eprintln!("--- committed (first 2 KiB) ---");
        let prefix = committed_str.chars().take(2048).collect::<String>();
        eprintln!("{prefix}");
        eprintln!("--- regenerated (first 2 KiB) ---");
        let prefix = regen_str.chars().take(2048).collect::<String>();
        eprintln!("{prefix}");
        panic!(
            "crates/libobj/include/libobj.h is out of date — run `cargo build -p libobj` \
             and commit the regenerated header",
        );
    }
}
