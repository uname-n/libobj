//! Build script: run cbindgen to regenerate `include/libobj.h`.
//!
//! The generated header is committed to the repo (see
//! `crates/libobj/include/libobj.h`). A drift test in
//! `tests/header_drift.rs` regenerates the header to a tempdir
//! and byte-compares against the committed file.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let crate_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo");
    let crate_path = PathBuf::from(&crate_dir);
    let header_path = crate_path.join("include").join("libobj.h");

    if env::var_os("OBJ_SKIP_CBINDGEN").is_some() {
        return;
    }

    let config = match cbindgen::Config::from_file(crate_path.join("cbindgen.toml")) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=cbindgen config load failed: {e}");
            return;
        }
    };

    let generated = match cbindgen::Builder::new()
        .with_config(config)
        .with_crate(&crate_dir)
        .generate()
    {
        Ok(g) => g,
        Err(e) => {
            println!("cargo:warning=cbindgen generation failed: {e}");
            return;
        }
    };

    if let Some(parent) = header_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            println!("cargo:warning=failed to create include dir: {e}");
            return;
        }
    }

    let _changed = generated.write_to_file(&header_path);
}
