//! Regression tests for the `Dynamic` node-budget boundary (pit issue #8).
//!
//! `decode_seq` and `decode_map` previously used a strict `>` guard against
//! `MAX_DYNAMIC_NODES`, which allowed declaring `len == MAX_DYNAMIC_NODES`
//! elements. Because `decode_value` increments the shared `nodes` counter for
//! *every* node — including the container itself — a Seq with 65 536 elements
//! would push `nodes` from 1 (the Seq node) to 65 537, tripping the budget
//! check on the very last element and returning `Err(Corruption)` even though
//! the payload is structurally valid.
//!
//! The fix changes both guards to `>=`, capping element count at
//! `MAX_DYNAMIC_NODES - 1` (65 535). A Seq of 65 535 elements then uses
//! exactly 65 536 total nodes — the full budget.
//!
//! These tests verify:
//!   1. A `Seq` / `Map` with `MAX_DYNAMIC_NODES - 1` elements round-trips
//!      cleanly through `to_postcard_bytes` + `from_tagged_bytes`.
//!   2. A `Seq` / `Map` with exactly `MAX_DYNAMIC_NODES` elements is
//!      correctly rejected at decode time (element count >= budget).

use std::collections::BTreeMap;

use obj_core::codec::dynamic::MAX_DYNAMIC_NODES;
use obj_core::codec::Dynamic;
use obj_core::Error;

/// Number of elements at the effective maximum: one below the node budget so
/// that the container itself fits within `MAX_DYNAMIC_NODES` total nodes.
const MAX_ELEMENTS: usize = MAX_DYNAMIC_NODES - 1;

// ── Seq round-trip at effective maximum ─────────────────────────────────────

#[test]
fn seq_at_max_elements_round_trips() {
    let seq = Dynamic::Seq(vec![Dynamic::Null; MAX_ELEMENTS]);
    let bytes = seq.to_postcard_bytes().expect("encode");
    let back = Dynamic::from_tagged_bytes(&bytes).expect("decode");
    assert_eq!(back, seq, "Seq({MAX_ELEMENTS} × Null) must round-trip");
}

// ── Seq one-over-budget is rejected ─────────────────────────────────────────

#[test]
fn seq_over_budget_is_rejected() {
    // Encode a Seq with MAX_DYNAMIC_NODES elements directly by hand:
    // the encoder has no budget check, so this produces valid bytes
    // for a Seq whose declared len == MAX_DYNAMIC_NODES.
    let oversize = Dynamic::Seq(vec![Dynamic::Null; MAX_DYNAMIC_NODES]);
    let bytes = oversize.to_postcard_bytes().expect("encode oversized seq");
    let err = Dynamic::from_tagged_bytes(&bytes).expect_err("should be rejected");
    assert!(
        matches!(err, Error::Corruption { page_id: 0 }),
        "expected Corruption, got {err:?}",
    );
}

// ── Map round-trip at effective maximum ─────────────────────────────────────

/// Build a `Map` with `n` string-keyed `Null` entries. Keys are
/// zero-padded decimal strings ("00000", "00001", …) so they are
/// unique and sort stably.
fn null_map(n: usize) -> Dynamic {
    let mut m = BTreeMap::new();
    for i in 0..n {
        m.insert(format!("{i:05}"), Dynamic::Null);
    }
    Dynamic::Map(m)
}

#[test]
fn map_at_max_elements_round_trips() {
    let map = null_map(MAX_ELEMENTS);
    let bytes = map.to_postcard_bytes().expect("encode");
    let back = Dynamic::from_tagged_bytes(&bytes).expect("decode");
    assert_eq!(back, map, "Map({MAX_ELEMENTS} entries) must round-trip");
}

// ── Map one-over-budget is rejected ─────────────────────────────────────────

#[test]
fn map_over_budget_is_rejected() {
    let oversize = null_map(MAX_DYNAMIC_NODES);
    let bytes = oversize.to_postcard_bytes().expect("encode oversized map");
    let err = Dynamic::from_tagged_bytes(&bytes).expect_err("should be rejected");
    assert!(
        matches!(err, Error::Corruption { page_id: 0 }),
        "expected Corruption, got {err:?}",
    );
}
