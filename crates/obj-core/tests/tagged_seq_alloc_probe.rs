//! Regression test for the tagged-format `decode_seq` allocation-amplification
//! bug (pit issue #6).
//!
//! A tagged `Seq(Seq(...Seq...))` payload nested `MAX_DYNAMIC_DEPTH` (32) deep,
//! whose wire bytes declare `len = MAX_DYNAMIC_NODES` (65 536) at every level,
//! used to make `decode_seq` eagerly `Vec::with_capacity(65536)` for each of
//! the 32 simultaneously-open frames — on the order of ~80 MB reserved from a
//! 128-byte payload, before the depth check tripped on the innermost frame.
//!
//! The decode fails either way (depth limit), so the bug is invisible to a
//! value/`Result` assertion. We account allocations deterministically with a
//! counting global allocator and assert that the peak delta stays bounded by
//! the *actually decoded* element count (zero elements decoded before the
//! depth-limit error), not the declared lengths. This is deterministic byte
//! accounting, not a flaky RSS probe.

mod alloc_probe_common;
use alloc_probe_common::{CEILING, CUR, NEST, PEAK, VARINT_65536};

use std::sync::atomic::Ordering;

use obj_core::codec::Dynamic;

#[global_allocator]
static ALLOCATOR: alloc_probe_common::Counting = alloc_probe_common::Counting;

/// Tagged-format Seq tag byte.
const TAG_SEQ: u8 = 0x07;

#[test]
fn tagged_nested_seq_does_not_amplify_allocation() {
    // Payload: TAG_SEQ + varint(65536) repeated NEST times.
    // Each decode_seq frame reads one varint and allocates its Vec before
    // the next frame opens, so NEST frames are simultaneously live at peak.
    // The decode fails at the depth limit (depth=NEST); the bug was visible
    // only in the heap reserved while those frames were open.
    let mut payload = Vec::with_capacity(NEST * (1 + VARINT_65536.len()));
    for _ in 0..NEST {
        payload.push(TAG_SEQ);
        payload.extend_from_slice(&VARINT_65536);
    }
    assert!(payload.len() < 200, "payload should be tiny");

    // Canary: verify TAG_SEQ is the correct tag byte — if it drifts, the
    // rest of this test exercises the wrong code path (the decoder hits the
    // `_ => Err(Corruption)` arm at depth 0 with zero Vec allocations, and
    // both assertions below pass vacuously without testing the fix).
    let tiny = Dynamic::Seq(vec![Dynamic::Null]);
    let mut tiny_bytes = tiny.to_postcard_bytes().expect("canary encode");
    assert_eq!(
        tiny_bytes[0],
        TAG_SEQ,
        "TAG_SEQ constant is out of sync with dynamic.rs"
    );
    // Truncate so the canary allocation is released before the probe window.
    tiny_bytes.truncate(0);
    drop(tiny_bytes);

    // Measure only the heap delta caused by the decode.
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
    let result = Dynamic::from_tagged_bytes(&payload);
    let peak_delta = PEAK.load(Ordering::Relaxed).saturating_sub(CUR.load(Ordering::Relaxed));

    // The decode must not panic — a depth-exceeded payload returns an Err.
    assert!(result.is_err(), "deeply nested oversized tagged Seq should be rejected");

    // The fix caps each open frame's reservation to SEQ_PREALLOC (16), so the
    // total reserved is bounded by (NEST * SEQ_PREALLOC * size_of::<Dynamic>())
    // — a few KB — not (NEST * 65536 * size_of::<Dynamic>()) ~ 80 MB. A
    // generous 1 MiB ceiling cleanly separates the two regimes.
    assert!(
        peak_delta < CEILING,
        "tagged decode reserved {peak_delta} bytes from a {}-byte payload (ceiling {CEILING}); \
         per-frame Vec::with_capacity is amplifying allocation",
        payload.len(),
    );
}
