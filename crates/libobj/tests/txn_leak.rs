//! Regression guard for issue #1: `obj_txn_commit` must not leak the
//! `obj_write_txn_t` heap allocation.
//!
//! ## The bug
//!
//! The original `obj_txn_commit` took the wrapped txn out with
//! `ManuallyDrop::take(&mut handle.inner)`, read the `db` Arc out with
//! `ptr::read`, then called `core::mem::forget(handle)` to suppress the
//! struct's `Drop` (so `inner` would not be double-dropped). But
//! `mem::forget` on a `Box<T>` skips **both** the destructor AND the
//! deallocation — so `size_of::<obj_write_txn_t>()` bytes (a `WriteTxn`
//! slot + an `Arc<Db>` slot) leaked on **every** commit. A long-running
//! process that commits in a loop leaks unboundedly.
//!
//! The structural fix localizes the whole teardown into the audited
//! `obj_write_txn_t::into_parts` primitive (one SAFETY proof), so the
//! entry point holds no raw reclaim a future edit could turn back into a
//! leaking `mem::forget`.
//!
//! ## How this test catches a reintroduced leak
//!
//! This is a plain begin-write / commit loop over a handful of empty
//! transactions — no global allocator, no allocation counter, no
//! tolerance assertion (the alloc-counting approach was removed: it is
//! repo-banned as "too fragile" in CLAUDE.md, and a tolerance window
//! gives false confidence by silently passing a slow 1-in-N leak).
//!
//! - Under stable `cargo test` the loop simply **exercises** the commit
//!   reclaim path. It passes trivially and is not flaky; its value here
//!   is keeping the path compiled and run.
//! - Under **Miri** (`cargo +nightly miri test`) the assertion is
//!   Miri's own **exit-time leak report**: Miri tracks every live
//!   allocation and fails the run if any outlive the program. A
//!   reintroduced `mem::forget` (or any reclaim that frees the box
//!   without deallocating) leaks one `obj_write_txn_t` per iteration,
//!   which Miri reports deterministically — no counter or threshold
//!   needed.
//!
//! ## Making the guard actually run
//!
//! The Miri assertion only fires when Miri runs this test. That is
//! tracked as **issue #2** (add a CI Miri job); until that lands, run it
//! locally with:
//!
//! ```text
//! MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test -p libobj --test txn_leak
//! ```
//!
//! ## Miri / file-IO limitation
//!
//! The C ABI exposes only file-backed open paths (`obj_open` /
//! `obj_open_with_config`) — there is no in-memory `obj_*` open entry
//! point — so this test must open a real file via a `TempDir`. The
//! file-backed pager performs real filesystem I/O, which Miri only
//! permits with `-Zmiri-disable-isolation` (hence the flag above), and
//! even then memory-mapped or platform-specific syscalls in the pager
//! may be unsupported under Miri. If Miri cannot execute the open path
//! in a given toolchain, this test is expected to error out at `obj_open`
//! under Miri rather than silently pass; the leak coverage it provides is
//! therefore best-effort until either Miri's file support covers the
//! pager or an in-memory C open path is added. Under stable `cargo test`
//! it always runs.

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use tempfile::TempDir;

use obj::{
    obj_close, obj_db_t, obj_open, obj_txn_begin_write, obj_txn_commit, obj_write_txn_t, OBJ_OK,
};

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

/// Begin then immediately commit one empty write transaction. Each call
/// boxes exactly one `obj_write_txn_t` in `obj_txn_begin_write` that a
/// correct `obj_txn_commit` must free via `into_parts`.
fn begin_and_commit(db: *mut obj_db_t) {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    // SAFETY: `db` is a live handle from obj_open and `&raw mut txn` is a
    // writable out-pointer, satisfying obj_txn_begin_write's contract.
    let begin = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(begin, OBJ_OK, "begin_write should succeed");
    // SAFETY: `txn` is the live, not-yet-committed handle just produced
    // by the successful begin above, satisfying obj_txn_commit's contract.
    let commit = unsafe { obj_txn_commit(txn) };
    assert_eq!(commit, OBJ_OK, "commit should succeed");
}

/// A small iteration count: enough to exercise the reclaim path and to
/// give Miri's exit-time leak report several allocations to account for,
/// while keeping the (slow) Miri run tractable. The leak signal does not
/// depend on the count — one leaked handle is enough for Miri to fail —
/// so this is deliberately tiny, not a statistical sample.
const ITERS: usize = 16;

#[test]
fn commit_does_not_leak_write_txn_handle() {
    let dir = TempDir::new().expect("tmp");
    let cs = path_cstring(&dir.path().join("leak.obj"));
    let mut db: *mut obj_db_t = ptr::null_mut();
    // SAFETY: `cs` is a live NUL-terminated path and `&raw mut db` is a
    // writable out-pointer, satisfying obj_open's contract.
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK);

    for _ in 0..ITERS {
        begin_and_commit(db);
    }

    // SAFETY: `db` is the live handle from obj_open above, not yet freed.
    unsafe { obj_close(db) };
}
