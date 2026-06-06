//! Bench harness compile + `--quick` smoke gate.
//!
//! The full `cargo bench --bench perf_table` run takes ~30 minutes
//! on macOS Apple Silicon (10 minutes on Linux x86-64 `NVMe`). The
//! per-PR CI matrix can't afford that latency. This test validates
//! the harness *shape* by running `cargo bench --bench perf_table --
//! --quick`, which exercises a single criterion sample per row in
//! under five minutes total.
//!
//! The test is marked `#[ignore]` so `cargo test` skips it by
//! default — it still builds (catching API drift) and a developer
//! can opt in with `cargo test --release -- --ignored
//! perf_table_compiles`.
//!
//! The test asserts:
//! 1. The bench compiles and runs to completion (exit 0).
//! 2. The markdown table lands at the expected path or the
//!    stdout transcript contains the header line.

#![forbid(unsafe_code)]

use std::process::Command;

#[test]
#[ignore = "runs the perf_table bench; ~5 minutes on Apple Silicon"]
fn perf_table_quick_runs() {
    let output = Command::new(env!("CARGO"))
        .args(["bench", "--bench", "perf_table", "--", "--quick"])
        .output()
        .expect("invoke cargo bench");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "perf_table bench failed: status = {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );
    assert!(
        stdout.contains("# obj perf table (M14 #119)"),
        "expected markdown header in stdout:\n{stdout}",
    );
    assert!(
        stdout.contains("| Operation | Measured (median) | Target (design.md) | Notes |"),
        "expected markdown table header in stdout:\n{stdout}",
    );
}
