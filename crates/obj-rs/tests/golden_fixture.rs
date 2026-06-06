//! Golden on-disk fixture: backward-compatibility anchor.
//!
//! `crates/obj-rs/tests/fixtures/golden_v1.obj` is a REAL `.obj` file
//! produced by the obj writer (`format_major=1`, `format_minor=2`,
//! plaintext) and COMMITTED to the repo as static bytes. This test
//! opens it READ-ONLY and asserts the current build still reads the
//! exact same bytes: the file decodes, passes `integrity_check`, and
//! every committed document round-trips to the expected value.
//!
//! Unlike `encryption_refusal.rs` (synthetic header bytes) and the
//! codec's lockstep encode/decode round-trips, this test pins a whole
//! real file. For a format about to freeze it is the single highest-
//! value backward-compat anchor: a future change that silently alters
//! the on-disk layout, the page-CRC scheme, the catalog encoding, the
//! index key encoding, or the document codec will break THIS test even
//! if every encode/decode round-trip still agrees with itself.
//!
//! # How the fixture was generated
//!
//! A throwaway generator (`zz_generate_golden.rs`, since removed)
//! wrote the file via the public `obj::Db` API:
//!
//! - collection `users` (derive `User`): five docs at pinned ids 1..=5
//!   with a `unique` index on `email`, a standard index on `age`, and
//!   an `each` index on `roles`;
//! - collection `orders` (derive `Order`): four docs at pinned ids
//!   1..=4 with a composite index on `(customer_id, placed_at)`.
//!
//! After the writes it checkpointed the WAL into the main file and
//! removed the sidecar, producing a single self-contained `.obj`. The
//! bytes were then committed; this test only ever READS them.
//!
//! To regenerate after an intentional, reviewed format change: restore
//! the generator, run `cargo test -p obj-rs --test zz_generate_golden
//! -- --ignored`, confirm the diff is expected, and re-commit the new
//! bytes alongside a CHANGELOG note.
//!
//! # Legacy `format_major=0`
//!
//! A 0.x writer no longer exists in this codebase, so there is no way
//! to regenerate a real major-0 file from the current build. Major-0
//! READ-compat is covered by the header-decode unit tests in
//! `crates/obj-core/src/pager/header.rs` (which decode synthetic
//! major-0 header bytes) and by `crates/obj-rs/tests/encryption_refusal.rs`.
//!
//! TODO(format-freeze): if a hand-crafted minimal valid major-0 file
//! becomes cleanly feasible (a tiny catalog-only file with a single
//! collection), add it here as `golden_v0.obj` to anchor major-0 read
//! compatibility with a real file rather than synthetic header bytes.

use std::path::PathBuf;

use obj::Db;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(collection = "users")]
struct User {
    #[obj(index = unique)]
    email: String,
    #[obj(index)]
    age: u64,
    #[obj(index = each)]
    roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(collection = "orders")]
#[obj(index = ("customer_id", "placed_at"))]
struct Order {
    customer_id: u64,
    placed_at: u64,
    total_cents: u64,
}

/// Absolute path to the committed fixture. `CARGO_MANIFEST_DIR` is the
/// `crates/obj-rs` directory at compile time, so this resolves regardless
/// of the working directory the test runner uses.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_v1.obj")
}

/// Copy the committed fixture into a fresh tempdir and return the
/// guard + the copy's path.
///
/// `Db::open` / `open_readonly` touch a `-lock` (and may create a
/// `-wal`) sidecar next to the file. Opening the fixture in place
/// would litter the source tree and could let a stale WAL mask the
/// committed main-file bytes on a re-run. Copying first keeps the
/// committed bytes pristine and the test hermetic. The copy is the
/// EXACT committed bytes, so the read path is still validated against
/// the frozen on-disk format.
fn open_fixture_copy() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tmp");
    let dst = dir.path().join("golden_v1.obj");
    std::fs::copy(fixture_path(), &dst).expect("copy fixture into tempdir");
    (dir, dst)
}

#[test]
fn golden_fixture_exists_and_is_nonempty() {
    let path = fixture_path();
    let meta = std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("golden fixture missing at {}: {e}", path.display()));
    assert!(meta.len() > 0, "golden fixture must be non-empty");
    assert_eq!(
        meta.len() % 4096,
        0,
        "fixture length {} is not a whole number of 4 KiB pages",
        meta.len()
    );
}

#[test]
fn golden_fixture_header_is_format_major_1_minor_2() {
    let (_dir, path) = open_fixture_copy();
    let db = Db::open_readonly(&path).expect("open_readonly golden fixture");
    let stat = db.stat().expect("stat");
    assert_eq!(stat.format_major, 1, "format_major must be 1");
    assert_eq!(stat.format_minor, 2, "format_minor must be 2");
    assert_eq!(stat.page_size, 4096, "page_size must be 4096");
    assert!(stat.page_count >= 1, "must have at least the header page");
}

#[test]
fn golden_fixture_passes_integrity_check() {
    let (_dir, path) = open_fixture_copy();
    let db = Db::open_readonly(&path).expect("open_readonly golden fixture");
    let report = db.integrity_check().expect("integrity_check completes");
    assert!(
        report.is_ok(),
        "committed golden fixture must pass integrity_check; got: {:?}",
        report.failures
    );
    assert!(report.pages_checked > 0, "must have inspected pages");
}

#[test]
fn golden_fixture_users_round_trip_to_expected_values() {
    let (_dir, path) = open_fixture_copy();
    let db = Db::open_readonly(&path).expect("open_readonly golden fixture");
    let mut users = db.all::<User>().expect("all users");
    users.sort_by(|a, b| a.email.cmp(&b.email));
    assert_eq!(users.len(), 5, "expected five users");
    for (i, user) in (1u64..=5).zip(users.iter()) {
        assert_eq!(user.email, format!("user{i}@example.com"));
        assert_eq!(user.age, 20 + i);
        assert_eq!(
            user.roles,
            vec!["reader".to_owned(), format!("tier{}", i % 3)],
            "roles for user{i} must match the committed bytes",
        );
    }
}

#[test]
fn golden_fixture_orders_round_trip_to_expected_values() {
    let (_dir, path) = open_fixture_copy();
    let db = Db::open_readonly(&path).expect("open_readonly golden fixture");
    let mut orders = db.all::<Order>().expect("all orders");
    orders.sort_by_key(|o| o.placed_at);
    assert_eq!(orders.len(), 4, "expected four orders");
    for (i, order) in (1u64..=4).zip(orders.iter()) {
        assert_eq!(order.customer_id, i % 3 + 1);
        assert_eq!(order.placed_at, 1_700_000_000 + i * 86_400);
        assert_eq!(order.total_cents, i * 1_000);
    }
}

#[test]
fn golden_fixture_unique_index_lookup_still_resolves() {
    let (_dir, path) = open_fixture_copy();
    let db = Db::open_readonly(&path).expect("open_readonly golden fixture");
    let found: Option<User> = db
        .find_unique::<User>("email", "user3@example.com".to_owned())
        .expect("find_unique");
    let user = found.expect("user3 present via unique index");
    assert_eq!(user.age, 23);
    assert_eq!(user.roles, vec!["reader".to_owned(), "tier0".to_owned()]);
}
