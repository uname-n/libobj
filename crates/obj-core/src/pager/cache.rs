//! Bounded LRU page cache.
//!
//! Frames are allocated up front at [`Cache::new`] and reused for the
//! lifetime of the pager — no heap allocation occurs on the read hot
//! path. The LRU ordering is maintained as a
//! doubly-linked list embedded in the same `Vec<Frame>` that holds the
//! buffers; the index-typed pointers make every traversal a small
//! bounded integer hop.
//!
//! This cache is not thread-safe. The pager wraps it in `&mut self`
//! methods.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use crate::pager::page::{Page, PageId};

/// `usize::MAX` used as the "null index" sentinel inside the LRU
/// linked list. No frame can have this index because the cache size
/// is bounded well below `usize::MAX / 2` in any realistic build.
const NIL: usize = usize::MAX;

/// A cache frame: a pre-allocated page buffer plus metadata for the
/// LRU bookkeeping. `page_id == None` means the frame is empty (its
/// buffer contents are irrelevant).
#[derive(Debug)]
struct Frame {
    page_id: Option<PageId>,
    buffer: Page,
    dirty: bool,
    prev: usize,
    next: usize,
}

/// Outcome of [`Cache::insert`].
#[derive(Debug)]
pub struct Evicted {
    /// Page-id that was evicted, if any.
    pub page_id: PageId,
    /// `true` if the evicted frame was dirty and must be flushed.
    pub dirty: bool,
    /// Snapshot of the evicted page's bytes (the caller writes this
    /// to disk when `dirty` is set).
    pub buffer: Page,
}

/// Fixed-capacity LRU cache keyed by [`PageId`].
#[derive(Debug)]
pub struct Cache {
    frames: Vec<Frame>,
    /// `PageId` → index into `frames`. The map's capacity is set at
    /// construction; `HashMap::insert` only allocates when the
    /// capacity is exceeded, which cannot happen because `frames.len()`
    /// is fixed and every key in the map corresponds to a frame.
    index: HashMap<PageId, usize>,
    /// Head of the LRU list (most-recently-used frame).
    head: usize,
    /// Tail of the LRU list (least-recently-used frame).
    tail: usize,
    /// Frames whose `page_id` is `None`, available without eviction.
    free_head: usize,
}

impl Cache {
    /// Construct a cache with `capacity` pre-allocated frames.
    ///
    /// `capacity` must be at least 1. Eviction kicks in once all
    /// frames are bound to a page.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        debug_assert!(capacity >= 1, "cache must have at least 1 frame");
        let cap = capacity.max(1);
        let mut frames = Vec::with_capacity(cap);
        for i in 0..cap {
            frames.push(Frame {
                page_id: None,
                buffer: Page::zeroed(),
                dirty: false,
                prev: NIL,
                next: if i + 1 == cap { NIL } else { i + 1 },
            });
        }
        Self {
            frames,
            index: HashMap::with_capacity(cap),
            head: NIL,
            tail: NIL,
            free_head: 0,
        }
    }

    /// Number of frames in the cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.frames.len()
    }

    /// Get a read-only reference to a cached page, if present, and
    /// mark it as most-recently-used. No allocation occurs.
    pub fn get(&mut self, id: PageId) -> Option<&Page> {
        let idx = *self.index.get(&id)?;
        self.unlink_lru(idx);
        self.push_front(idx);
        Some(&self.frames[idx].buffer)
    }

    /// Read-only peek that does NOT touch the LRU order. Used by
    /// `Pager::read_cache_or_main` (which holds a shared `&Pager`
    /// borrow on the snapshot-read path); a mutating `get` would
    /// be a borrow-checker error there.
    #[must_use]
    pub fn peek(&self, id: PageId) -> Option<&Page> {
        let idx = *self.index.get(&id)?;
        Some(&self.frames[idx].buffer)
    }

    /// Get a mutable reference to a cached page, marking it dirty and
    /// most-recently-used.
    pub fn get_mut(&mut self, id: PageId) -> Option<&mut Page> {
        let idx = *self.index.get(&id)?;
        self.unlink_lru(idx);
        self.push_front(idx);
        self.frames[idx].dirty = true;
        Some(&mut self.frames[idx].buffer)
    }

    /// Insert `(id, buffer)` into the cache. If the cache is full,
    /// evicts the LRU frame; the returned [`Evicted`] is the caller's
    /// responsibility to write back if `dirty` is set.
    ///
    /// `dirty` controls whether the newly-inserted frame is considered
    /// modified. Pages loaded from disk should pass `false`; pages
    /// allocated fresh by `alloc_page` should pass `true`.
    pub fn insert(&mut self, id: PageId, buffer: Page, dirty: bool) -> Option<Evicted> {
        debug_assert!(
            !self.index.contains_key(&id),
            "caller must remove an existing key before reinserting",
        );
        let (idx, evicted) = self.acquire_frame();
        self.frames[idx].page_id = Some(id);
        self.frames[idx].buffer = buffer;
        self.frames[idx].dirty = dirty;
        self.push_front(idx);
        self.index.insert(id, idx);
        evicted
    }

    /// Acquire a frame for a fresh insert. Prefers a free frame;
    /// otherwise evicts the LRU.
    fn acquire_frame(&mut self) -> (usize, Option<Evicted>) {
        if self.free_head != NIL {
            let idx = self.free_head;
            self.free_head = self.frames[idx].next;
            self.frames[idx].next = NIL;
            self.frames[idx].prev = NIL;
            return (idx, None);
        }
        let tail = self.tail;
        debug_assert!(tail != NIL, "tail must exist when no free frames");
        self.unlink_lru(tail);
        let dirty = std::mem::replace(&mut self.frames[tail].dirty, false);
        let buffer = std::mem::take(&mut self.frames[tail].buffer);
        if let Some(evicted_id) = self.frames[tail].page_id.take() {
            self.index.remove(&evicted_id);
            (
                tail,
                Some(Evicted {
                    page_id: evicted_id,
                    dirty,
                    buffer,
                }),
            )
        } else {
            debug_assert!(false, "tail frame on LRU list must be bound");
            (tail, None)
        }
    }

    /// Force-evict `id` if present, returning its buffer + dirty flag.
    /// Used by `free_page` to invalidate the cache entry.
    pub fn evict(&mut self, id: PageId) -> Option<Evicted> {
        let idx = self.index.remove(&id)?;
        self.unlink_lru(idx);
        let dirty = std::mem::replace(&mut self.frames[idx].dirty, false);
        let buffer = std::mem::take(&mut self.frames[idx].buffer);
        let page_id = self.frames[idx].page_id.take().unwrap_or(id);
        debug_assert_eq!(
            page_id, id,
            "frame referenced by index must carry the same id",
        );
        self.frames[idx].next = self.free_head;
        self.free_head = idx;
        Some(Evicted {
            page_id,
            dirty,
            buffer,
        })
    }

    /// Move `idx` to the head of the LRU list. Assumes the frame is
    /// already detached (or about to be).
    fn push_front(&mut self, idx: usize) {
        debug_assert_ne!(idx, NIL);
        self.frames[idx].prev = NIL;
        self.frames[idx].next = self.head;
        if self.head != NIL {
            self.frames[self.head].prev = idx;
        }
        self.head = idx;
        if self.tail == NIL {
            self.tail = idx;
        }
    }

    /// Detach `idx` from its current position in the LRU list. The
    /// frame remains in `self.frames` and `self.index`.
    fn unlink_lru(&mut self, idx: usize) {
        let (prev, next) = (self.frames[idx].prev, self.frames[idx].next);
        if prev != NIL {
            self.frames[prev].next = next;
        } else if self.head == idx {
            self.head = next;
        }
        if next != NIL {
            self.frames[next].prev = prev;
        } else if self.tail == idx {
            self.tail = prev;
        }
        self.frames[idx].prev = NIL;
        self.frames[idx].next = NIL;
    }

    /// Iterate over `(page_id, buffer)` for every dirty bound frame,
    /// returning ownership of each buffer. Used at close to flush dirty
    /// pages. The iterator is bounded by `self.capacity()`.
    pub fn drain_dirty(&mut self) -> impl Iterator<Item = (PageId, Page)> + '_ {
        let cap = self.frames.len();
        (0..cap).filter_map(|idx| {
            if !self.frames[idx].dirty {
                return None;
            }
            self.frames[idx].dirty = false;
            let id = self.frames[idx].page_id?;
            let buf = std::mem::take(&mut self.frames[idx].buffer);
            self.frames[idx].page_id = None;
            self.index.remove(&id);
            self.unlink_lru(idx);
            self.frames[idx].next = self.free_head;
            self.free_head = idx;
            Some((id, buf))
        })
    }

    /// Capacity-checked LRU order, for tests. Returns `PageId`s in
    /// MRU → LRU order.
    #[must_use]
    #[cfg(test)]
    pub fn lru_order(&self) -> Vec<PageId> {
        let mut out = Vec::with_capacity(self.frames.len());
        let mut idx = self.head;
        let mut bound = self.frames.len() + 1;
        while idx != NIL && bound > 0 {
            if let Some(id) = self.frames[idx].page_id {
                out.push(id);
            }
            idx = self.frames[idx].next;
            bound -= 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;
    use crate::pager::page::{Page, PageId};

    fn id(n: u64) -> PageId {
        PageId::new(n).expect("non-zero")
    }

    fn page_with(byte: u8) -> Page {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[0] = byte;
        p
    }

    #[test]
    fn empty_cache_misses() {
        let mut c = Cache::new(4);
        assert!(c.get(id(1)).is_none());
    }

    #[test]
    fn insert_then_get() {
        let mut c = Cache::new(2);
        assert!(c.insert(id(1), page_with(0xAA), false).is_none());
        assert_eq!(c.get(id(1)).map(|p| p.as_bytes()[0]), Some(0xAA));
    }

    #[test]
    fn lru_eviction_is_deterministic() {
        let mut c = Cache::new(3);
        for n in 1u8..=3 {
            assert!(c.insert(id(u64::from(n)), page_with(n), false).is_none());
        }
        let _ = c.get(id(1));
        let ev = c.insert(id(4), page_with(4), false).expect("eviction");
        assert_eq!(ev.page_id, id(2));
        assert!(!ev.dirty);
        assert_eq!(c.lru_order(), vec![id(4), id(1), id(3)]);
    }

    #[test]
    fn dirty_eviction_returns_buffer() {
        let mut c = Cache::new(1);
        let _ = c.insert(id(1), page_with(0xAB), true);
        let ev = c.insert(id(2), page_with(0xCD), false).expect("evict");
        assert_eq!(ev.page_id, id(1));
        assert!(ev.dirty);
        assert_eq!(ev.buffer.as_bytes()[0], 0xAB);
    }

    #[test]
    fn evict_releases_frame() {
        let mut c = Cache::new(2);
        let _ = c.insert(id(1), page_with(1), true);
        let _ = c.insert(id(2), page_with(2), false);
        let ev = c.evict(id(1)).expect("evict");
        assert!(ev.dirty);
        assert!(c.insert(id(3), page_with(3), false).is_none());
        assert!(c.get(id(2)).is_some());
    }

    #[test]
    fn drain_dirty_yields_each_dirty_page_once() {
        let mut c = Cache::new(4);
        let _ = c.insert(id(1), page_with(1), true);
        let _ = c.insert(id(2), page_with(2), false);
        let _ = c.insert(id(3), page_with(3), true);
        let mut drained: Vec<u64> = c.drain_dirty().map(|(id, _)| id.get()).collect();
        drained.sort_unstable();
        assert_eq!(drained, vec![1, 3]);
        let again: Vec<_> = c.drain_dirty().collect();
        assert!(again.is_empty());
    }
}
