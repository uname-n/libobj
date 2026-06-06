//! `trybuild` compile-fail suite for `#[derive(obj::Document)]`.
//!
//! Each file under `tests/derive_compile_fail/` provokes one specific
//! error from the proc-macro; the `.stderr` next to it pins the
//! expected diagnostic. `trybuild` will fail this test if either the
//! file compiles unexpectedly or the produced diagnostic drifts from
//! the recorded `.stderr`.
//!
//! # Regenerating `.stderr` files
//!
//! `trybuild` `.stderr` snapshots are toolchain-pinned. When the
//! diagnostic changes (because rustc was upgraded, the macro's error
//! string was reworded, etc.), regenerate via:
//!
//! ```bash
//! TRYBUILD=overwrite cargo test --test derive_compile_fail
//! ```
//!
//! The repository's `rust-toolchain.toml` pins the stable channel so
//! committed `.stderr` files stay valid as long as contributors use
//! the project toolchain.

#[test]
fn derive_compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/derive_compile_fail/*.rs");
}
