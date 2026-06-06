//! Shared counting-allocator infrastructure for allocation-probe regression
//! tests (`seq_alloc_probe` and `tagged_seq_alloc_probe`).
//!
//! Each integration test file includes this module via `mod alloc_probe_common`
//! and registers the allocator with `#[global_allocator]` in its own binary:
//!
//! ```ignore
//! mod alloc_probe_common;
//! #[global_allocator]
//! static ALLOCATOR: alloc_probe_common::Counting = alloc_probe_common::Counting;
//! ```
//!
//! Because each integration test is its own binary, every binary gets its own
//! copy of `CUR` and `PEAK` — so the statics are not shared across binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub static CUR: AtomicUsize = AtomicUsize::new(0);
pub static PEAK: AtomicUsize = AtomicUsize::new(0);

/// System allocator wrapper that tracks current + peak live bytes.
pub struct Counting;

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

/// LEB128 encoding of 65 536 (`MAX_DYNAMIC_NODES`).
pub const VARINT_65536: [u8; 3] = [0x80, 0x80, 0x04];

/// Depth of the crafted nesting — matches `MAX_SCHEMA_DEPTH` / `MAX_DYNAMIC_DEPTH`.
pub const NEST: usize = 32;

/// Peak-reservation ceiling. The fixed walker reserves a few KB for the probe
/// payloads; the buggy version reserved ~67–80 MB. 1 MiB cleanly separates the
/// two regimes.
pub const CEILING: usize = 1024 * 1024;
