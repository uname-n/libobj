//! Manual allocation probe for the schema-walker `Seq` amplification bug
//! (pit issue #3). Run with:
//!
//! ```text
//! cargo run --example seq_alloc_probe -p obj-core
//! ```
//!
//! Decodes a sub-100-byte `Seq(Seq(...Seq(U64)...))` payload whose every level
//! declares `len = 65536` and prints the peak heap reserved during the decode.
//! Before the fix this printed ~67 MB; after the per-frame `SEQ_PREALLOC` cap it
//! prints a few KB. The automated regression guard lives in
//! `tests/seq_alloc_probe.rs`; this example exists only for manual inspection.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use obj_core::codec::schema::DynamicSchema;
use obj_core::codec::Dynamic;

static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call forwards verbatim to the system allocator; the only added
// behaviour is relaxed atomic accounting, which cannot affect memory safety.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's layout unchanged to the system allocator.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let now = CUR.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CUR.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: forwarding the same (ptr, layout) pair `alloc` returned.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const NEST: usize = 32;

fn main() {
    // Schema: Seq(Seq(...Seq(U64)...)) nested NEST deep.
    let mut schema = DynamicSchema::U64;
    for _ in 0..NEST {
        schema = DynamicSchema::seq(schema);
    }

    // Wire bytes: varint(65536) == [0x80, 0x80, 0x04], repeated NEST times.
    let mut payload = Vec::new();
    for _ in 0..NEST {
        payload.extend_from_slice(&[0x80, 0x80, 0x04]);
    }

    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
    let result = Dynamic::from_postcard_bytes(&payload, &schema);
    let peak_delta = PEAK.load(Ordering::Relaxed).saturating_sub(CUR.load(Ordering::Relaxed));

    println!("size_of::<Dynamic>() = {}", std::mem::size_of::<Dynamic>());
    println!("payload bytes        = {}", payload.len());
    println!("decode is_err        = {}", result.is_err());
    println!("peak reserved bytes  = {peak_delta}");
}
