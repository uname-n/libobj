//! Assert the README's quickstart `rust` block actually compiles
//! against the live `obj` API.
//!
//! Mechanism:
//!
//! 1. `include_str!("../../../README.md")` pulls the workspace-root
//!    README into the test binary at compile time.
//! 2. A tiny parser finds the **first** triple-back-tick `rust`
//!    fenced block and returns the body verbatim.
//! 3. The body is written into a `*.rs` file in a tempdir alongside
//!    a hand-curated `Cargo.toml` that depends on the same `obj`,
//!    `serde`, and `tempfile` paths the rest of the test suite uses.
//! 4. `cargo build` against the generated crate must exit `0`. Any
//!    breakage in the README snippet — wrong type name, removed
//!    method, missing import — fails the test long before docs
//!    review notices.

#![cfg(unix)]
// allow: test-support code — `expect`/`?` panics and error returns are the test's
// own failure signal, not a documented public-API contract worth a doc section.
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const README: &str = include_str!("../../../README.md");

/// Find the first triple-backtick `rust` fenced block in the README
/// and return its body. Returns `None` if no such block exists.
fn first_rust_block(src: &str) -> Option<String> {
    let mut lines = src.lines();
    let mut body = String::new();
    let mut inside = false;
    for line in &mut lines {
        if !inside {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("```") {
                let lang = rest.split(|c: char| c.is_whitespace() || c == ',').next()?;
                if lang == "rust" {
                    inside = true;
                }
            }
        } else if line.trim_start().starts_with("```") {
            return Some(body);
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    None
}

/// The first fenced rust block is the quickstart. We check it
/// compiles by writing it to a fresh tempdir crate and invoking
/// `cargo build`.
#[test]
#[cfg(unix)]
fn readme_quickstart_compiles() {
    let snippet = first_rust_block(README).expect("README must contain a ```rust block");
    assert!(
        snippet.contains("fn main"),
        "quickstart snippet must declare its own fn main; got: {snippet}"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let crate_dir: PathBuf = tmp.path().to_path_buf();
    fs::create_dir_all(crate_dir.join("src")).expect("mkdir src");

    let obj_crate_path = workspace_relative("crates/obj-rs");
    let manifest = format!(
        r#"[package]
name = "obj_readme_compile_test"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
obj-rs = {{ path = "{obj}" }}
serde = {{ version = "1", features = ["derive"] }}
tempfile = "3"

[workspace]
"#,
        obj = obj_crate_path.display(),
    );
    fs::write(crate_dir.join("Cargo.toml"), manifest).expect("write manifest");
    fs::write(crate_dir.join("src").join("main.rs"), &snippet).expect("write snippet");

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(&cargo)
        .arg("build")
        .arg("--quiet")
        .current_dir(&crate_dir)
        .env_remove("RUSTFLAGS")
        .status()
        .expect("invoke cargo build");
    assert!(
        status.success(),
        "README quickstart failed to compile; snippet was:\n---\n{snippet}\n---"
    );
}

/// Walk up from `CARGO_MANIFEST_DIR` until we hit the workspace
/// root (the directory that contains `Cargo.toml` with
/// `[workspace]`), then resolve `child` against it.
fn workspace_relative(child: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut cur: PathBuf = manifest_dir;
    for _ in 0..8 {
        let candidate = cur.join("Cargo.toml");
        if let Ok(text) = fs::read_to_string(&candidate) {
            if text.contains("[workspace]") {
                return cur.join(child);
            }
        }
        if !cur.pop() {
            break;
        }
    }
    panic!(
        "could not locate workspace root from {}",
        env!("CARGO_MANIFEST_DIR")
    );
}
