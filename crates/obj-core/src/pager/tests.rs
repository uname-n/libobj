//! Pager tests — unit + property.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

use proptest::prelude::*;
use tempfile::TempDir;

use crate::error::Error;
use crate::pager::page::{Page, PageId, PAGE_SIZE};
use crate::pager::{Config, Pager};
use crate::wal::Lsn;

fn id(n: u64) -> PageId {
    PageId::new(n).expect("non-zero")
}

/// Test helper — open a file-backed pager and enter a WAL txn so the
/// tests below can call `alloc_page` / `free_page` without tripping
/// the `in_txn` debug-assert. The header is routed
/// through the WAL via `stage_or_write_header`, which requires an
/// open txn. The helper preserves the tests' original flow
/// (open → mutate → maybe-commit → drop) without adding a `begin_txn`
/// at every call site.
fn open_file(path: &std::path::Path, config: Config) -> Pager<crate::platform::FileHandle> {
    let mut p = Pager::open(path, config).expect("open");
    p.begin_txn();
    p
}

#[test]
fn memory_pager_alloc_round_trip() {
    let mut p = Pager::memory(Config::default()).expect("construct");
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0xAB;
    p.write_page(a, &page).expect("write");
    let read = p.read_page(a).expect("read");
    assert_eq!(read.as_bytes()[0], 0xAB);
}

#[test]
fn alloc_and_free_recycles_id() {
    let mut p = Pager::memory(Config::default()).expect("construct");
    let a = p.alloc_page().expect("alloc");
    let b = p.alloc_page().expect("alloc");
    assert_ne!(a, b);
    p.free_page(a).expect("free");
    let c = p.alloc_page().expect("realloc");
    assert_eq!(c, a, "freelist must recycle the most recently freed id");
}

#[test]
fn free_then_alloc_lifo_order() {
    let mut p = Pager::memory(Config::default()).expect("construct");
    let ids: Vec<PageId> = (0..4).map(|_| p.alloc_page().expect("alloc")).collect();
    for &i in &ids {
        p.free_page(i).expect("free");
    }
    let recycled: Vec<PageId> = (0..4).map(|_| p.alloc_page().expect("realloc")).collect();
    let expected: Vec<PageId> = ids.iter().rev().copied().collect();
    assert_eq!(recycled, expected);
}

#[test]
fn file_backend_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("test.obj");
    {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0..4].copy_from_slice(b"WXYZ");
        p.write_page(a, &page).expect("write");
        p.flush().expect("flush");
    }
    let mut p = open_file(&path, Config::default());
    let a = id(1);
    let read = p.read_page(a).expect("read");
    assert_eq!(&read.as_bytes()[0..4], b"WXYZ");
}

#[test]
fn close_reopen_preserves_freelist_head() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("test.obj");
    let (a, b) = {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let b = p.alloc_page().expect("alloc");
        p.free_page(a).expect("free a");
        p.flush().expect("flush");
        let head = p.freelist_head();
        assert_eq!(head, a.get());
        (a, b)
    };
    let mut p = open_file(&path, Config::default());
    assert_eq!(p.freelist_head(), a.get());
    let recycled = p.alloc_page().expect("realloc");
    assert_eq!(recycled, a);
    assert_ne!(recycled, b);
}

#[test]
fn cache_evicts_under_pressure() {
    let cfg = Config::default().with_cache_frames(2).expect("cap");
    let mut p = Pager::memory(cfg).expect("construct");
    let a = p.alloc_page().expect("alloc");
    let b = p.alloc_page().expect("alloc");
    let c = p.alloc_page().expect("alloc");
    for &x in &[a, b, c] {
        let _ = p.read_page(x).expect("read");
    }
    assert!(p.read_page(a).is_ok());
    assert!(p.read_page(b).is_ok());
    assert!(p.read_page(c).is_ok());
}

#[test]
fn dirty_eviction_writes_back() {
    let cfg = Config::default().with_cache_frames(1).expect("cap");
    let mut p = Pager::memory(cfg).expect("construct");
    let a = p.alloc_page().expect("alloc");
    let b = p.alloc_page().expect("alloc");
    let mut data = Page::zeroed();
    data.as_bytes_mut()[100] = 0x77;
    p.write_page(a, &data).expect("write");
    let _ = p.read_page(b).expect("read");
    let back = p.read_page(a).expect("read");
    assert_eq!(back.as_bytes()[100], 0x77);
}

#[test]
fn page_count_grows_only_when_freelist_empty() {
    let mut p = Pager::memory(Config::default()).expect("construct");
    let before = p.page_count();
    let a = p.alloc_page().expect("alloc");
    assert_eq!(p.page_count(), before + 1);
    p.free_page(a).expect("free");
    let _ = p.alloc_page().expect("realloc");
    assert_eq!(p.page_count(), before + 1);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn alloc_free_sequence_is_consistent(
        ops in proptest::collection::vec(any::<bool>(), 0..32),
    ) {
        let mut p = Pager::memory(Config::default()).expect("construct");
        let mut live: HashSet<u64> = HashSet::new();
        let mut freed: Vec<u64> = Vec::new();

        for op in ops {
            if op || live.is_empty() {
                let id = p.alloc_page().expect("alloc").get();
                prop_assert!(live.insert(id), "double-issued id {id}");
                if let Some(expected) = freed.last().copied() {
                    prop_assert_eq!(id, expected, "freelist must be LIFO");
                    freed.pop();
                }
            } else {
                let &victim = live.iter().next().expect("non-empty");
                prop_assert!(live.remove(&victim));
                freed.push(victim);
                p.free_page(PageId::new(victim).expect("non-zero")).expect("free");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn close_reopen_freelist_head_round_trips(
        n_alloc in 1usize..16,
        seed in any::<u8>(),
    ) {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("rt.obj");
        let expected_head = {
            let mut p = open_file(&path, Config::default());
            let mut ids = Vec::with_capacity(n_alloc);
            for _ in 0..n_alloc {
                ids.push(p.alloc_page().expect("alloc"));
            }
            let mut to_free: Vec<PageId> = ids
                .iter()
                .enumerate()
                .filter(|(i, _)| ((seed as usize).wrapping_add(*i)) & 1 == 0)
                .map(|(_, &x)| x)
                .collect();
            if to_free.is_empty() {
                to_free.push(ids[0]);
            }
            let mut last_freed = 0;
            for fid in &to_free {
                p.free_page(*fid).expect("free");
                last_freed = fid.get();
            }
            p.flush().expect("flush");
            prop_assert_eq!(p.freelist_head(), last_freed);
            last_freed
        };
        let p = open_file(&path, Config::default());
        prop_assert_eq!(p.freelist_head(), expected_head);
    }
}

/// Build a file with `n_pages` written pages, each filled with a
/// distinctive byte pattern. Returns the path and the list of
/// `PageId`s written.
fn build_corruption_fixture(dir: &TempDir, n_pages: u64) -> (std::path::PathBuf, Vec<PageId>) {
    let path = dir.path().join("corruption.obj");
    let mut p = open_file(&path, Config::default());
    let cap = usize::try_from(n_pages).expect("n_pages fits in usize");
    let mut ids = Vec::with_capacity(cap);
    for i in 0..n_pages {
        let id = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        for (j, b) in page.as_bytes_mut().iter_mut().enumerate().take(64) {
            let j = u64::try_from(j).expect("j < 64");
            let mixed = i.wrapping_mul(13).wrapping_add(j) & 0xFF;
            *b = u8::try_from(mixed).expect("masked to 0..256");
        }
        p.write_page(id, &page).expect("write");
        ids.push(id);
    }
    p.flush().expect("flush");
    drop(p);
    (path, ids)
}

#[test]
fn flipping_a_data_byte_is_detected_as_corruption() {
    let dir = TempDir::new().expect("tempdir");
    let (path, ids) = build_corruption_fixture(&dir, 1);
    let victim = ids[0];
    let offset = victim.byte_offset(0) + 100;
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("reopen-rw");
        f.seek(SeekFrom::Start(offset)).expect("seek");
        let mut b = [0u8; 1];
        f.read_exact(&mut b).expect("read");
        b[0] ^= 0x55;
        f.seek(SeekFrom::Start(offset)).expect("seek-back");
        f.write_all(&b).expect("write");
        f.sync_all().expect("sync");
    }
    let mut p = open_file(&path, Config::default());
    match p.read_page(victim) {
        Err(Error::Corruption { page_id }) => assert_eq!(page_id, victim.get()),
        other => panic!("expected Corruption, got {other:?}"),
    }
}

#[test]
fn flipping_the_header_is_detected_at_open() {
    let dir = TempDir::new().expect("tempdir");
    let (path, _) = build_corruption_fixture(&dir, 1);
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("reopen-rw");
        f.seek(SeekFrom::Start(80)).expect("seek");
        let mut b = [0u8; 1];
        f.read_exact(&mut b).expect("read");
        b[0] ^= 0xFF;
        f.seek(SeekFrom::Start(80)).expect("seek-back");
        f.write_all(&b).expect("write");
        f.sync_all().expect("sync");
    }
    match Pager::open(&path, Config::default()) {
        Err(Error::Corruption { page_id }) => assert_eq!(page_id, 0),
        other => panic!("expected Corruption on header CRC mismatch, got {other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1_000,
        max_shrink_iters: 4,
        .. ProptestConfig::default()
    })]

    /// For any randomly chosen (page, byte-offset), flipping that byte
    /// on disk must cause `read_page` to surface `Error::Corruption`.
    /// 1000 iterations per the issue acceptance criteria.
    #[test]
    fn byte_flip_anywhere_in_a_data_page_is_detected(
        n_pages in 1u64..4,
        page_idx in 0usize..4,
        byte_offset in 0u64..(PAGE_SIZE as u64),
    ) {
        let dir = TempDir::new().expect("tempdir");
        let (path, ids) = build_corruption_fixture(&dir, n_pages);
        let victim = ids[page_idx % ids.len()];
        let file_offset = victim.byte_offset(0) + byte_offset;
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("reopen-rw");
            f.seek(SeekFrom::Start(file_offset)).expect("seek");
            let mut b = [0u8; 1];
            f.read_exact(&mut b).expect("read");
            b[0] ^= 0x01;
            f.seek(SeekFrom::Start(file_offset)).expect("seek-back");
            f.write_all(&b).expect("write");
            f.sync_all().expect("sync");
        }
        let mut p = open_file(&path, Config::default());
        match p.read_page(victim) {
            Err(Error::Corruption { page_id }) => prop_assert_eq!(page_id, victim.get()),
            other => prop_assert!(false, "expected Corruption, got {other:?}"),
        }
    }
}

const _: usize = PAGE_SIZE;

#[test]
fn write_then_read_within_same_session_sees_uncommitted_data() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal.obj");
    let mut p = open_file(&path, Config::default());
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0x77;
    p.write_page(a, &page).expect("write");
    let read = p.read_page(a).expect("read");
    assert_eq!(read.as_bytes()[0], 0x77);
}

#[test]
fn commit_drains_pending_into_view() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal.obj");
    let mut p = open_file(&path, Config::default());
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[100] = 0xAB;
    p.write_page(a, &page).expect("write");
    let lsn = p.commit().expect("commit");
    assert!(lsn >= Lsn::ONE, "commit must assign a positive LSN");
    let read = p.read_page(a).expect("read");
    assert_eq!(read.as_bytes()[100], 0xAB);
}

#[test]
fn empty_commit_returns_zero_lsn() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal.obj");
    let mut p = open_file(&path, Config::default());
    let lsn = p.commit().expect("commit");
    assert_eq!(lsn, Lsn::ZERO);
}

#[test]
fn memory_pager_commit_is_noop() {
    let mut p = Pager::memory(Config::default()).expect("memory");
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0x42;
    p.write_page(a, &page).expect("write");
    let lsn = p.commit().expect("memory commit");
    assert_eq!(lsn, Lsn::ZERO);
    let read = p.read_page(a).expect("read");
    assert_eq!(read.as_bytes()[0], 0x42);
}

#[test]
fn group_commit_assigns_consecutive_lsns() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal.obj");
    let mut p = open_file(&path, Config::default());
    let ids: Vec<PageId> = (0..4).map(|_| p.alloc_page().expect("alloc")).collect();
    for (i, &pid) in ids.iter().enumerate() {
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = u8::try_from(i & 0xFF).expect("masked");
        p.write_page(pid, &page).expect("write");
    }
    let lsn = p.commit().expect("commit");
    assert_eq!(
        lsn,
        Lsn::new(5),
        "four user frames + one header frame must produce LSNs 1..=5"
    );
    for (i, &pid) in ids.iter().enumerate() {
        let r = p.read_page(pid).expect("read");
        assert_eq!(r.as_bytes()[0], u8::try_from(i & 0xFF).expect("masked"));
    }
}

#[test]
fn commit_then_reopen_recovers_data_without_flush() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("recover.obj");
    let a_id = {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0..4].copy_from_slice(b"PQRS");
        p.write_page(a, &page).expect("write");
        let _ = p.commit().expect("commit");
        a.get()
    };
    let mut p = open_file(&path, Config::default());
    let a = id(a_id);
    let read = p.read_page(a).expect("read");
    assert_eq!(
        &read.as_bytes()[0..4],
        b"PQRS",
        "recovery must replay the committed write"
    );
}

#[test]
fn open_recovers_multiple_commits() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("recover_multi.obj");
    let (a_id, b_id) = {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let b = p.alloc_page().expect("alloc");
        let mut pa = Page::zeroed();
        pa.as_bytes_mut()[10] = 0xAA;
        p.write_page(a, &pa).expect("write a");
        let _ = p.commit().expect("commit 1");
        let mut pa2 = Page::zeroed();
        pa2.as_bytes_mut()[10] = 0xCC;
        let mut pb = Page::zeroed();
        pb.as_bytes_mut()[20] = 0xBB;
        p.write_page(a, &pa2).expect("write a2");
        p.write_page(b, &pb).expect("write b");
        let _ = p.commit().expect("commit 2");
        (a.get(), b.get())
    };
    let mut p = open_file(&path, Config::default());
    let ra = p.read_page(id(a_id)).expect("read a");
    assert_eq!(ra.as_bytes()[10], 0xCC, "later commit wins");
    let rb = p.read_page(id(b_id)).expect("read b");
    assert_eq!(rb.as_bytes()[20], 0xBB);
}

/// Fault test. Allocate many fresh pages in ONE committed txn,
/// write distinctive content, commit (so the bodies + `page_count` ride
/// the SINGLE WAL group-commit — NOT a checkpoint), drop WITHOUT a
/// checkpoint to model a crash right after the commit, then reopen and
/// read every committed page. None may surface `UnexpectedEof` /
/// `Corruption`: the fresh pages live only in the WAL until
/// checkpoint, and recovery replays them into the view so `page_count`
/// never outruns what the WAL can produce. The main file is NOT
/// extended at alloc — recovery (or the next checkpoint) heals the
/// length.
#[test]
fn batch_extension_covers_all_committed_pages_after_crash() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("batch_ext.obj");
    let n_alloc = 37usize;
    let last = {
        let mut p = open_file(&path, Config::default());
        let mut ids = Vec::with_capacity(n_alloc);
        for k in 0..n_alloc {
            let a = p.alloc_page().expect("alloc");
            let mut page = Page::zeroed();
            page.as_bytes_mut()[0] = u8::try_from(k & 0xFF).expect("masked");
            p.write_page(a, &page).expect("write");
            ids.push(a);
        }
        let _ = p.commit().expect("commit");
        ids.last().copied().expect("at least one alloc")
    };
    let mut p = open_file(&path, Config::default());
    let pc = p.page_count();
    for raw in 1..pc {
        let read = p
            .read_page(id(raw))
            .unwrap_or_else(|e| panic!("page {raw} unreadable after crash: {e:?}"));
        let expected = u8::try_from((raw - 1) & 0xFF).expect("masked");
        assert_eq!(
            read.as_bytes()[0],
            expected,
            "page {raw} content lost across batched extension + crash"
        );
    }
    assert_eq!(
        pc,
        u64::try_from(n_alloc + 1).expect("fits"),
        "page_count must equal allocated pages plus page 0"
    );
    assert!(p.read_page(last).is_ok());
}

/// Fault test — the small-transaction case. Allocate a few pages,
/// write content, commit, drop WITHOUT a checkpoint (crash right after
/// commit), reopen. Every reachable page `[1, page_count)` reads clean
/// (no `UnexpectedEof`) from the recovered WAL view, and `page_count`
/// reflects exactly the committed allocations — proving recovery heals
/// the (un-extended) main-file length without leaking any phantom slot
/// into the recovered authority.
#[test]
fn partial_batch_commit_crash_reopen_reads_clean() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("partial_batch.obj");
    let m_alloc = 5usize;
    {
        let mut p = open_file(&path, Config::default());
        for k in 0..m_alloc {
            let a = p.alloc_page().expect("alloc");
            let mut page = Page::zeroed();
            page.as_bytes_mut()[0] = u8::try_from(0xA0 + k).expect("masked");
            p.write_page(a, &page).expect("write");
        }
        let _ = p.commit().expect("commit");
    }
    let mut p = open_file(&path, Config::default());
    let pc = p.page_count();
    assert_eq!(pc, u64::try_from(m_alloc + 1).expect("fits"));
    for raw in 1..pc {
        let read = p
            .read_page(id(raw))
            .unwrap_or_else(|e| panic!("reachable page {raw} unreadable: {e:?}"));
        let expected = u8::try_from(0xA0_u64 + (raw - 1)).expect("masked");
        assert_eq!(read.as_bytes()[0], expected, "page {raw} content lost");
    }
}

#[test]
fn uncommitted_writes_are_not_recovered() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("uncommitted.obj");
    let a_id = {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let mut zero = Page::zeroed();
        zero.as_bytes_mut()[0] = 0x00;
        p.write_page(a, &zero).expect("write zeros");
        let _ = p.commit().expect("commit");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x55;
        p.write_page(a, &page).expect("write");
        a.get()
    };
    let mut p = open_file(&path, Config::default());
    let read = p.read_page(id(a_id)).expect("read");
    assert_ne!(
        read.as_bytes()[0],
        0x55,
        "uncommitted writes MUST NOT survive a drop / reopen"
    );
}

#[test]
fn wal_sidecar_is_created_on_open() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal.obj");
    {
        let _p = open_file(&path, Config::default());
    }
    let wal_path = crate::pager::wal_path_for(&path);
    assert!(wal_path.exists(), "WAL sidecar must be created at open");
}

#[test]
fn close_removes_wal_sidecar() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("clean.obj");
    let a_id = {
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x33;
        p.write_page(a, &page).expect("write");
        let _ = p.commit().expect("commit");
        let aid = a.get();
        p.close().expect("close");
        aid
    };
    let wal_path = crate::pager::wal_path_for(&path);
    assert!(
        !wal_path.exists(),
        "close MUST remove the WAL sidecar (design.md guarantee)"
    );
    let mut p = open_file(&path, Config::default());
    let r = p.read_page(id(a_id)).expect("read");
    assert_eq!(r.as_bytes()[0], 0x33);
}

#[test]
fn checkpoint_is_idempotent() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("idem.obj");
    let mut p = open_file(&path, Config::default());
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0x44;
    p.write_page(a, &page).expect("write");
    let _ = p.commit().expect("commit");
    p.checkpoint().expect("checkpoint 1");
    p.checkpoint().expect("checkpoint 2");
    let r = p.read_page(a).expect("read");
    assert_eq!(r.as_bytes()[0], 0x44);
}

#[test]
fn auto_checkpoint_fires_at_threshold() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("auto_ckpt.obj");
    let cfg = Config::default()
        .with_checkpoint_threshold(3)
        .with_cache_frames(8)
        .expect("cache");
    let mut p = open_file(&path, cfg);
    let ids: Vec<PageId> = (0..5).map(|_| p.alloc_page().expect("alloc")).collect();
    for (i, &pid) in ids.iter().enumerate() {
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = u8::try_from(i).expect("masked");
        p.write_page(pid, &page).expect("write");
        let _ = p.commit().expect("commit");
    }
    for (i, &pid) in ids.iter().enumerate() {
        let r = p.read_page(pid).expect("read");
        assert_eq!(r.as_bytes()[0], u8::try_from(i).expect("masked"));
    }
}

#[test]
fn open_after_clean_close_has_no_wal_file() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("noreopen.obj");
    {
        let p = open_file(&path, Config::default());
        p.close().expect("close");
    }
    let wal_path = crate::pager::wal_path_for(&path);
    assert!(!wal_path.exists());
    let p = open_file(&path, Config::default());
    p.close().expect("close 2");
    assert!(!wal_path.exists(), "second close must again leave no WAL");
}

#[test]
fn salt_rotates_on_checkpoint() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("salt.obj");
    let mut p = open_file(&path, Config::default());
    let a = p.alloc_page().expect("alloc");
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = 0x77;
    p.write_page(a, &page).expect("write");
    let _ = p.commit().expect("commit");
    p.checkpoint().expect("checkpoint");
    drop(p);
    let mut p2 = open_file(&path, Config::default());
    let r = p2.read_page(a).expect("read");
    assert_eq!(r.as_bytes()[0], 0x77);
}

mod snapshot {
    use super::*;
    use crate::pager::PageHandle;
    use crate::platform::FileHandle;
    use std::sync::{Arc, Mutex};
    use std::thread;

    fn stamp(byte: u8) -> Page {
        let mut p = Page::zeroed();
        p.as_bytes_mut()[0] = byte;
        p
    }

    #[test]
    fn snapshot_sees_committed_view_at_pin_time() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit 1");

        let snap = p.reader_snapshot().expect("snap");

        p.write_page(a, &stamp(0x22)).expect("write 2");
        let _ = p.commit().expect("commit 2");

        let page = snap.read_page(&p, a).expect("snap read");
        assert_eq!(page.as_bytes()[0], 0x11, "snapshot must see frozen view");
    }

    #[test]
    fn fresh_snapshot_sees_latest_writer_commit() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit 1");
        p.write_page(a, &stamp(0x22)).expect("write 2");
        let _ = p.commit().expect("commit 2");

        let snap = p.reader_snapshot().expect("snap");
        let page = snap.read_page(&p, a).expect("snap read");
        assert_eq!(page.as_bytes()[0], 0x22, "fresh snapshot must see latest");
    }

    #[test]
    fn pending_writes_invisible_to_snapshot() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit");

        p.write_page(a, &stamp(0xFF)).expect("pending");

        let snap = p.reader_snapshot().expect("snap");
        let page = snap.read_page(&p, a).expect("snap read");
        assert_eq!(
            page.as_bytes()[0],
            0x11,
            "pending writes must NOT be visible to a snapshot",
        );
    }

    #[test]
    fn checkpoint_skipped_while_snapshot_pins_old_lsn() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit 1");

        let _snap = p.reader_snapshot().expect("snap");

        p.write_page(a, &stamp(0x22)).expect("write 2");
        let _ = p.commit().expect("commit 2");

        let frames_before = p.wal.as_ref().map_or(0, |s| s.wal.committed_frames());
        p.checkpoint().expect("checkpoint");
        let frames_after = p.wal.as_ref().map_or(0, |s| s.wal.committed_frames());
        assert_eq!(
            frames_before, frames_after,
            "checkpoint must defer while snapshot pins old LSN"
        );
    }

    #[test]
    fn dropping_snapshot_allows_checkpoint() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit");

        let snap = p.reader_snapshot().expect("snap");
        assert_eq!(p.live_snapshot_count(), 1);
        drop(snap);
        assert_eq!(p.live_snapshot_count(), 0);
        p.checkpoint()
            .expect("checkpoint must run cleanly after snapshot drop");
    }

    /// Reader thread body: loop `iters` times reading page `a` and
    /// asserting it equals the frozen byte the snapshot was pinned
    /// at (which is `0`).  Borrows `pager` and `snap` by reference so
    /// the thread closure that calls this owns the Arc/snapshot.
    fn reader_loop(
        r: usize,
        pager: &Arc<Mutex<Pager<FileHandle>>>,
        snap: &crate::pager::ReaderSnapshot<FileHandle>,
        a: PageId,
        iters: u32,
    ) {
        for i in 0..iters {
            let p = pager.lock().expect("lock");
            let page = snap.read_page(&p, a).expect("snap read");
            assert_eq!(
                page.as_bytes()[0],
                0,
                "reader {r} iter {i}: snapshot must see frozen byte 0",
            );
            drop(p);
            std::thread::yield_now();
        }
    }

    /// Writer thread body: commit `count` distinct versions, each
    /// stamping a new byte into page `a`.
    fn writer_loop(pager: &Arc<Mutex<Pager<FileHandle>>>, a: PageId, count: u32) {
        for v in 1u32..=count {
            let mut p = pager.lock().expect("lock");
            let byte = u8::try_from((v % 250) + 1).expect("byte fits");
            p.write_page(a, &stamp(byte)).expect("write");
            let _ = p.commit().expect("commit");
            drop(p);
            std::thread::yield_now();
        }
    }

    /// Multi-thread concurrency test.
    /// 8 reader threads each hold a snapshot while a writer commits
    /// 200 page updates concurrently.  Every snapshot read must see
    /// a page consistent with its pinned LSN.
    #[test]
    fn many_readers_one_writer_consistent_view() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut opened = Pager::<FileHandle>::open(&path, cfg).expect("open");
        opened.begin_txn();
        let pager = Arc::new(Mutex::new(opened));
        let a = {
            let mut p = pager.lock().expect("lock");
            let a = p.alloc_page().expect("alloc");
            p.write_page(a, &stamp(0)).expect("write");
            let _ = p.commit().expect("commit");
            a
        };
        let n_readers = 8usize;
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_readers);
            for r in 0..n_readers {
                let pager = Arc::clone(&pager);
                let snap = {
                    let mut p = pager.lock().expect("lock");
                    p.reader_snapshot().expect("snap")
                };
                handles.push(scope.spawn(move || reader_loop(r, &pager, &snap, a, 1000)));
            }
            let writer_pager = Arc::clone(&pager);
            let writer = scope.spawn(move || writer_loop(&writer_pager, a, 200));
            writer.join().expect("writer join");
            for h in handles {
                h.join().expect("reader join");
            }
        });
        let fresh = {
            let mut p = pager.lock().expect("lock");
            p.reader_snapshot().expect("fresh snap")
        };
        let p = pager.lock().expect("lock");
        let page = fresh.read_page(&p, a).expect("fresh read");
        assert_ne!(
            page.as_bytes()[0],
            0,
            "post-write fresh snapshot must NOT see the original byte 0",
        );
    }

    #[test]
    fn snapshot_id_is_monotonic() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let mut p = open_file(&path, Config::default());
        let s1 = p.reader_snapshot().expect("s1");
        let s2 = p.reader_snapshot().expect("s2");
        assert!(s2.id().get() > s1.id().get());
    }

    #[test]
    fn min_pinned_lsn_tracks_lowest_live_reader() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit 1");

        let s1 = p.reader_snapshot().expect("s1");
        let s1_lsn = s1.pinned_lsn();

        p.write_page(a, &stamp(0x22)).expect("write 2");
        let _ = p.commit().expect("commit 2");

        let s2 = p.reader_snapshot().expect("s2");
        assert!(s2.pinned_lsn() > s1.pinned_lsn());

        assert_eq!(p.min_pinned_lsn(), Some(s1_lsn));
        drop(s1);
        assert_eq!(p.min_pinned_lsn(), Some(s2.pinned_lsn()));
        drop(s2);
        assert_eq!(p.min_pinned_lsn(), None);
    }

    #[test]
    fn memory_pager_snapshot_works_with_empty_view() {
        let mut p = Pager::memory(Config::default()).expect("mem pager");
        let snap = p.reader_snapshot().expect("snap");
        assert_eq!(snap.pinned_lsn(), Lsn::ZERO, "memory pager has no WAL");
        assert_eq!(p.live_snapshot_count(), 1);
    }

    /// Regression: switching the committed view to `Arc<Page>`
    /// must NOT weaken snapshot isolation. A page first written and
    /// committed AFTER a snapshot is pinned must be invisible to that
    /// snapshot — the snapshot's frozen view is its OWN cloned map, so
    /// the writer's later `view.insert` of a fresh `Arc` cannot appear
    /// in it. This guards the isolation the deep clone provided before
    /// the Arc-share change. (Disable auto-checkpoint so the writer's
    /// post-pin commit stays in the WAL view and the only thing
    /// keeping the snapshot consistent is the per-snapshot map clone.)
    #[test]
    fn snapshot_does_not_observe_page_committed_after_pin() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);

        let a = p.alloc_page().expect("alloc a");
        p.write_page(a, &stamp(0x11)).expect("write a");
        let _ = p.commit().expect("commit a");

        let snap = p.reader_snapshot().expect("snap");

        let b = p.alloc_page().expect("alloc b");
        p.write_page(b, &stamp(0x22)).expect("write b");
        let _ = p.commit().expect("commit b");
        assert_ne!(a, b, "b must be a distinct page id");

        let seen_a = snap.read_page(&p, a).expect("snap read a");
        assert_eq!(
            seen_a.as_bytes()[0],
            0x11,
            "snapshot must still observe the page committed before its pin",
        );

        let frozen_has_b = snap.frozen_pages().any(|(id, _)| id == b);
        assert!(
            !frozen_has_b,
            "post-pin commit must NOT appear in the snapshot's frozen view",
        );
        let seen_b = snap.read_page(&p, b).expect("snap read b");
        assert_ne!(
            seen_b.as_bytes()[0],
            0x22,
            "snapshot must NOT observe the body committed after its pin",
        );
    }

    /// A frozen-view hit returns `PageHandle::Shared` (an
    /// `Arc::clone`, no 4 KiB body copy), and `into_page` on that arm
    /// materialises the exact frozen body.
    #[test]
    fn frozen_view_hit_is_shared_and_into_page_round_trips() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let mut p = open_file(&path, Config::default());
        let a = p.alloc_page().expect("alloc");
        p.write_page(a, &stamp(0x11)).expect("write");
        let _ = p.commit().expect("commit 1");

        let snap = p.reader_snapshot().expect("snap");
        p.write_page(a, &stamp(0x22)).expect("write 2");
        let _ = p.commit().expect("commit 2");

        let handle = snap.read_page(&p, a).expect("snap read");
        assert!(
            matches!(handle, PageHandle::Shared(_)),
            "frozen-view hit must return PageHandle::Shared (refcount bump, no body clone)",
        );
        assert_eq!(
            handle.as_bytes()[0],
            0x11,
            "shared arm must see frozen body"
        );
        let owned = handle.into_page();
        assert_eq!(
            owned.as_bytes()[0],
            0x11,
            "into_page on the Shared arm must round-trip the frozen body",
        );
    }

    /// A frozen-view MISS returns `PageHandle::Owned` via the
    /// checksum-verifying disk read path. First confirm a clean
    /// main-file page reads back as `Owned` with the right body; then
    /// corrupt that page on disk, reopen, and confirm a snapshot read
    /// surfaces `Error::Corruption` — i.e. the `Owned` arm still
    /// verifies integrity (`read_through` -> `page_trailer_valid`).
    #[test]
    fn frozen_view_miss_owned_arm_still_checksum_verifies() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap.obj");
        let a;
        {
            let mut p = open_file(&path, Config::default());
            a = p.alloc_page().expect("alloc");
            p.write_page(a, &stamp(0x11)).expect("write a");
            let _ = p.commit().expect("commit a");
            p.checkpoint().expect("checkpoint");
        }

        {
            let mut p = open_file(&path, Config::default());
            let snap = p.reader_snapshot().expect("snap");
            assert!(
                !snap.frozen_pages().any(|(id, _)| id == a),
                "fresh-open snapshot must not hold `a` in its frozen view",
            );
            let handle = snap.read_page(&p, a).expect("clean owned read");
            assert!(
                matches!(handle, PageHandle::Owned(_)),
                "frozen-view miss must return PageHandle::Owned via the disk path",
            );
            assert_eq!(
                handle.as_bytes()[0],
                0x11,
                "owned arm must see the main-file body",
            );
        }

        let file_offset = a.byte_offset(0) + 16;
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("reopen-rw");
            f.seek(SeekFrom::Start(file_offset)).expect("seek");
            let mut b = [0u8; 1];
            f.read_exact(&mut b).expect("read");
            b[0] ^= 0x01;
            f.seek(SeekFrom::Start(file_offset)).expect("seek-back");
            f.write_all(&b).expect("write");
            f.sync_all().expect("sync");
        }
        let mut p = open_file(&path, Config::default());
        let snap = p.reader_snapshot().expect("snap2");
        match snap.read_page(&p, a) {
            Err(Error::Corruption { page_id }) => assert_eq!(page_id, a.get()),
            other => panic!("Owned-arm disk miss must checksum-verify; got {other:?}"),
        }
    }

    /// MVCC: a snapshot taken BEFORE a growing commit must not
    /// observe the fresh pages — even though the writer advanced
    /// `page_count` and the fresh body never touched the main file. The
    /// snapshot read must NOT `UnexpectedEof` (the fresh slot is past the
    /// physical high-water) and must return the page's pre-existence
    /// state (a zeroed body), not the writer's post-pin content.
    #[test]
    fn snapshot_before_growing_commit_does_not_observe_fresh_pages() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap_before.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);
        let a = p.alloc_page().expect("alloc a");
        p.write_page(a, &stamp(0x11)).expect("write a");
        let _ = p.commit().expect("commit a");
        p.checkpoint().expect("checkpoint a");

        let snap = p.reader_snapshot().expect("snap");

        let b = p.alloc_page().expect("alloc b");
        p.write_page(b, &stamp(0x22)).expect("write b");
        let _ = p.commit().expect("commit b");
        assert_ne!(a, b);

        assert!(!snap.frozen_pages().any(|(idx, _)| idx == b));
        let seen_b = snap.read_page(&p, b).expect("snap read b must not EOF");
        assert_ne!(
            seen_b.as_bytes()[0],
            0x22,
            "snapshot-before must NOT observe the post-pin fresh page",
        );
        assert_eq!(
            seen_b.as_bytes()[0],
            0x00,
            "post-pin fresh page resolves to its pre-existence (zeroed) state",
        );
        let seen_a = snap.read_page(&p, a).expect("snap read a");
        assert_eq!(seen_a.as_bytes()[0], 0x11);
    }

    /// MVCC: a snapshot taken AFTER a growing commit reads the fresh
    /// page from its `frozen_view` — WITHOUT touching the main file,
    /// which is still too short to hold the page. This is the positive
    /// half of the isolation contract: the fresh body is visible to a
    /// post-commit reader purely via the WAL view.
    #[test]
    fn snapshot_after_growing_commit_reads_fresh_from_frozen_view() {
        let dir = TempDir::new().expect("tmp");
        let path = dir.path().join("snap_after.obj");
        let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
        let mut p = open_file(&path, cfg);
        let b = p.alloc_page().expect("alloc b");
        p.write_page(b, &stamp(0x77)).expect("write b");
        let _ = p.commit().expect("commit b");
        let physical = p.main_physical_page_count().expect("physical");
        assert!(
            b.get() >= physical,
            "fresh page must be beyond the physical high-water before checkpoint \
             (b={}, physical={physical})",
            b.get(),
        );

        let snap = p.reader_snapshot().expect("snap");
        assert!(
            snap.frozen_pages().any(|(idx, _)| idx == b),
            "snapshot-after must capture the fresh page in its frozen view",
        );
        let handle = snap.read_page(&p, b).expect("snap read b");
        assert!(
            matches!(handle, PageHandle::Shared(_)),
            "fresh page must be served from the frozen view (Shared), not the \
             main file (Owned) — the file does not hold it yet",
        );
        assert_eq!(handle.as_bytes()[0], 0x77, "frozen view returns the body");
    }
}

#[cfg(feature = "encryption")]
mod zeroize_key_material {
    use super::*;
    use crate::pager::{wrap_master_key, MasterKeyBytes};

    /// Compile-time guarantee: under the `encryption` feature the
    /// master-key storage type wipes its bytes on drop. If a future
    /// change swaps `MasterKeyBytes` back to a bare `[u8; 32]` (which
    /// is not `ZeroizeOnDrop`) this stops compiling.
    #[test]
    fn master_key_bytes_is_zeroize_on_drop() {
        fn assert_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}
        assert_zeroize_on_drop::<MasterKeyBytes>();
    }

    /// The stored key round-trips through the public builder and is
    /// readable internally as a plain `&[u8; 32]` (the `Zeroizing`
    /// wrapper is transparent to the open/derive call sites).
    #[test]
    fn key_round_trips_through_config_storage() {
        let raw = [0x5Au8; 32];
        let cfg = Config::default().with_encryption_key(Some(raw));
        let stored = cfg.master_key().expect("key present");
        assert_eq!(stored, &raw, "stored key must equal the supplied key");

        let cleared = cfg.with_encryption_key(None);
        assert!(cleared.master_key().is_none(), "key cleared");
    }

    /// `wrap_master_key` produces a wrapper that derefs back to the
    /// original bytes — sanity check the wrap/unwrap symmetry the
    /// crypto hot path relies on.
    #[test]
    fn wrap_master_key_preserves_bytes() {
        let raw = [0xC3u8; 32];
        let wrapped: MasterKeyBytes = wrap_master_key(raw);
        let view: &[u8; 32] = &wrapped;
        assert_eq!(view, &raw);
    }
}
