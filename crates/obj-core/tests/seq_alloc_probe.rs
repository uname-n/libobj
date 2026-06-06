//! Regression test for the schema-walker `Seq` allocation-amplification bug
//! (pit issue #3).
//!
//! A schema of `Seq(Seq(...Seq(U64)...))` nested `MAX_SCHEMA_DEPTH` deep, whose
//! wire bytes declare `len = MAX_DYNAMIC_NODES` (65536) at every level, used to
//! make `decode_seq_slot` eagerly `Vec::with_capacity(65536)` for each of the
//! ~32 simultaneously-open frames — on the order of ~80 MB reserved from a
//! payload of under 100 bytes, before any element was actually decoded.
//!
//! The decode itself errors either way (the frame stack hits `MAX_SCHEMA_DEPTH`),
//! so the bug is invisible to a value/`Result` assertion — the only observable
//! is the heap reserved while the frames are open. We therefore account
//! allocation deterministically with a counting global allocator and assert the
//! peak stays bounded by the *actually decoded* element count, not the declared
//! (attacker-controlled) lengths. This is deterministic byte accounting, not a
//! flaky RSS probe.

mod alloc_probe_common;
use alloc_probe_common::{CEILING, CUR, NEST, PEAK, VARINT_65536};

use std::sync::atomic::Ordering;

use obj_core::codec::schema::DynamicSchema;
use obj_core::codec::Dynamic;

#[global_allocator]
static ALLOCATOR: alloc_probe_common::Counting = alloc_probe_common::Counting;

#[test]
fn nested_seq_does_not_amplify_allocation() {
    // Schema: Seq(Seq(...Seq(U64)...)) nested NEST deep.
    let mut schema = DynamicSchema::U64;
    for _ in 0..NEST {
        schema = DynamicSchema::seq(schema);
    }

    // Wire bytes: varint(65536) repeated NEST times — under 100 bytes total.
    // Note: no tag bytes appear in the schema-walk (postcard) path, so a
    // change to TAG_SEQ in dynamic.rs cannot cause a vacuous pass here.
    let mut payload = Vec::with_capacity(NEST * VARINT_65536.len());
    for _ in 0..NEST {
        payload.extend_from_slice(&VARINT_65536);
    }
    assert!(payload.len() < 100, "payload should be tiny");

    // Measure only the heap delta caused by the decode.
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
    let result = Dynamic::from_postcard_bytes(&payload, &schema);
    let peak_delta = PEAK.load(Ordering::Relaxed).saturating_sub(CUR.load(Ordering::Relaxed));

    // The decode must not panic — malformed/oversized input returns an Err.
    assert!(result.is_err(), "deeply nested oversized Seq should be rejected");

    // The fix caps each open frame's reservation to a small constant, so the
    // total reserved is bounded by (frames * SMALL_PREALLOC * size_of::<Dynamic>())
    // — a few KB — not (frames * 65536 * size_of::<Dynamic>()) ~ 80 MB. A
    // generous 1 MiB ceiling cleanly separates the two regimes.
    assert!(
        peak_delta < CEILING,
        "decode reserved {peak_delta} bytes from a {}-byte payload (ceiling {CEILING}); \
         per-frame Vec::with_capacity is amplifying allocation",
        payload.len(),
    );
}
