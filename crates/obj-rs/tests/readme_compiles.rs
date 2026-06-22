//! Assert that **every** `rust` fenced block in the README actually
//! compiles against the live `obj` API.
//!
//! Mechanism:
//!
//! 1. `include_str!("../../../README.md")` pulls the workspace-root
//!    README into the test binary at compile time.
//! 2. A tiny parser collects the body of every triple-back-tick `rust`
//!    fenced block, in document order.
//! 3. Each body is written into its own `*.rs` file in a tempdir crate
//!    alongside a hand-curated `Cargo.toml` that depends on the same
//!    `obj`, `serde`, and `tempfile` paths the rest of the suite uses.
//!    A block that already declares `fn main` is written verbatim;
//!    every other block (item-only definitions, or statements that use
//!    `?`) is wrapped in `fn main() -> obj::Result<()> { … Ok(()) }`,
//!    mirroring how rustdoc auto-wraps a doctest without its own `main`.
//! 4. `cargo build` against each generated crate must exit `0`. Any
//!    breakage in a README snippet — wrong type name, removed method,
//!    a changed `default_with` signature, a renamed `Config` knob —
//!    fails the test long before docs review notices.

#![cfg(unix)]
// allow: test-support code — `expect`/`?` panics and error returns are the test's
// own failure signal, not a documented public-API contract worth a doc section.
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const README: &str = include_str!("../../../README.md");

/// Hard cap on the number of fenced blocks we will extract — a bounded
/// loop (Power-of-Ten R2). The README has a handful today; this leaves
/// generous head-room while still terminating on a malformed file.
const MAX_BLOCKS: usize = 64;

/// Collect the body of every triple-backtick `rust` fenced block in the
/// README, in document order. A block's body is the text between its
/// opening ```` ```rust ```` line and the next ```` ``` ```` line.
fn all_rust_blocks(src: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut inside = false;
    for line in src.lines() {
        if blocks.len() >= MAX_BLOCKS {
            break;
        }
        if inside {
            if line.trim_start().starts_with("```") {
                blocks.push(std::mem::take(&mut body));
                inside = false;
            } else {
                body.push_str(line);
                body.push('\n');
            }
        } else if let Some(rest) = line.trim_start().strip_prefix("```") {
            let lang = rest
                .split(|c: char| c.is_whitespace() || c == ',')
                .next()
                .unwrap_or("");
            if lang == "rust" {
                inside = true;
            }
        }
    }
    blocks
}

/// Wrap a snippet so it is a complete program. Blocks that declare
/// their own `fn main` (the quickstart) are returned verbatim; the rest
/// — item-only definitions and `?`-using statement blocks — are wrapped
/// in a `fn main() -> obj::Result<()>` so `?` resolves and any declared
/// items live in a valid scope. A crate-level `#![allow(dead_code,
/// unused)]` keeps illustrative-but-unused snippets warning-free without
/// touching the README prose.
fn as_program(snippet: &str) -> String {
    if snippet.contains("fn main") {
        format!("#![allow(dead_code, unused)]\n{snippet}")
    } else {
        format!(
            "#![allow(dead_code, unused)]\nfn main() -> obj::Result<()> {{\n{snippet}\nOk(())\n}}\n"
        )
    }
}

/// Write `program` into a fresh single-binary crate under `crate_dir`
/// and `cargo build` it. Returns the build's exit status alongside the
/// program text so callers can report the offending source on failure.
fn compile_program(crate_dir: &Path, program: &str) -> bool {
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
    fs::write(crate_dir.join("src").join("main.rs"), program).expect("write program");

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    Command::new(&cargo)
        .arg("build")
        .arg("--quiet")
        .current_dir(crate_dir)
        .env_remove("RUSTFLAGS")
        .status()
        .expect("invoke cargo build")
        .success()
}

/// Every fenced rust block in the README must compile against the live
/// `obj` API. Each block is compiled in its own throwaway crate.
#[test]
#[cfg(unix)]
fn readme_blocks_compile() {
    let blocks = all_rust_blocks(README);
    assert!(
        !blocks.is_empty(),
        "README must contain at least one ```rust block"
    );

    // The first block is the quickstart and is expected to be a
    // complete program with its own `fn main`.
    assert!(
        blocks[0].contains("fn main"),
        "quickstart snippet must declare its own fn main; got: {}",
        blocks[0]
    );

    for (i, snippet) in blocks.iter().enumerate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let program = as_program(snippet);
        assert!(
            compile_program(tmp.path(), &program),
            "README rust block #{} failed to compile; program was:\n---\n{program}\n---",
            i + 1,
        );
    }
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
