//! C smoke harness driver.
//!
//! Shells out to `bash crates/libobj/examples/build-smoke.sh`, which
//! builds + runs the C smoke and asserts the marker line
//! `OBJ_C_SMOKE_OK`. This Rust test exists so `cargo test --workspace
//! --tests` exercises the same path locally.
//!
//! The test is `#[ignore]` by default because the build step takes
//! several seconds on a cold cache; CI sets the gate via a separate
//! step in `.github/workflows/ci.yml`. Run locally via:
//!
//!     cargo test -p libobj --test c_smoke -- --ignored

#![allow(unsafe_code)]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(std::path::Path::parent)
        .expect("repo root above crates/libobj")
        .to_path_buf()
}

#[test]
#[ignore = "builds libobj in release mode and links a C program; opt in via --ignored"]
fn c_smoke_links_and_runs() {
    let root = repo_root();
    let script = root
        .join("crates")
        .join("libobj")
        .join("examples")
        .join("build-smoke.sh");
    let status = Command::new("bash")
        .arg(&script)
        .current_dir(&root)
        .status()
        .expect("spawn bash");
    assert!(status.success(), "build-smoke exited {status}");
}
