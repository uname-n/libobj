//! Exercises the opt-in `serde` feature.
//!
//! Builds only when the feature is enabled; the entire file is
//! gated at the module level so `cargo test` without
//! `--features serde` skips it cleanly.
//!
//! The round-trips below cover one type per category:
//!
//! - `Config` — obj-rs local struct with no public field set.
//! - `IntegrityReport` — re-exported from obj-core; contains a
//!   `Vec<IntegrityFailure>` so a non-empty report exercises both.
//! - `DbStat` — composite type that drags `CollectionStat` in
//!   transitively, and `SyncMode` is reachable through `Config` /
//!   `obj-core::pager::Config`.
//! - `Id` — obj-core re-export already derived
//!   `Serialize`/`Deserialize` unconditionally; this round-trip is
//!   the regression guard against accidentally hiding it behind a
//!   feature flag in the future.

#![cfg(feature = "serde")]
#![forbid(unsafe_code)]

use obj::{Config, Db, DbStat, Id, IntegrityReport};

/// Round-trip a value through `serde_json` and assert structural
/// equality via `Debug` formatting. `Debug` is used because not
/// every type in the public surface implements `PartialEq` (e.g.
/// `Config` does not); a `Debug` repr match is sufficient for
/// structural equality and avoids requiring extra trait bounds at
/// the call site.
fn roundtrip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let encoded = serde_json::to_string(value).expect("serialize via serde_json");
    let decoded: T = serde_json::from_str(&encoded).expect("deserialize via serde_json");
    assert_eq!(format!("{value:?}"), format!("{decoded:?}"));
}

#[test]
fn config_roundtrip() {
    let cfg = Config::default();
    roundtrip(&cfg);
}

#[test]
fn id_roundtrip() {
    let id = Id::try_new(42).expect("Id::try_new(42)");
    roundtrip(&id);
}

#[test]
fn integrity_report_roundtrip() {
    let db = Db::memory().expect("Db::memory");
    let report: IntegrityReport = db.integrity_check().expect("integrity_check");
    roundtrip(&report);
}

#[test]
fn db_stat_roundtrip() {
    let db = Db::memory().expect("Db::memory");
    let stat: DbStat = db.stat().expect("Db::stat");
    roundtrip(&stat);
}
