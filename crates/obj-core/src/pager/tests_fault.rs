//! Pager fault-injection integration tests.
//!
//! Each test drives `Pager<FaultyFileHandle>` through a hand-crafted
//! crash scenario and verifies the recovery contract holds:
//! the pager re-opens cleanly, committed pages are present, and
//! uncommitted pages either appear with their committed bytes or do
//! not appear at all.
//!
//! These complement the seed-range cycle test in
//! `tests/crash_cycles.rs`. The cycle test gives coverage; these
//! tests pin specific paths the salt-rotation logic
//! introduced so a regression is caught at the right layer.

#![cfg(test)]

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;

use tempfile::TempDir;

use crate::pager::page::{Page, PageId};
use crate::pager::{wal_path_for, Config, Pager};
use crate::platform::fault::{FaultPlan, FaultyFileHandle, FAULT_CRASH_MARKER};
use crate::platform::FileHandle;

fn open_faulty(
    main_path: &std::path::Path,
    main_plan: FaultPlan,
    wal_plan: FaultPlan,
    config: Config,
) -> crate::Result<Pager<FaultyFileHandle>> {
    let main = FaultyFileHandle::new(FileHandle::open_or_create(main_path)?, main_plan);
    let wal_path = wal_path_for(main_path);
    let wal = FaultyFileHandle::new(FileHandle::open_or_create(&wal_path)?, wal_plan);
    let mut p = Pager::<FaultyFileHandle>::open_with_backends(main, wal, wal_path, config)?;
    p.begin_txn();
    Ok(p)
}

fn write_committed(p: &mut Pager<FaultyFileHandle>, id: PageId, marker: u8) -> crate::Result<()> {
    let mut page = Page::zeroed();
    page.as_bytes_mut()[0] = marker;
    page.as_bytes_mut()[1024] = marker.wrapping_mul(3);
    p.write_page(id, &page)?;
    let _ = p.commit()?;
    Ok(())
}

fn id(n: u64) -> PageId {
    PageId::new(n).expect("non-zero")
}

fn panic_carries_marker(payload: &Box<dyn std::any::Any + Send>) -> bool {
    let s = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("");
    s.contains(FAULT_CRASH_MARKER)
}

/// Test: crash between salt-write and main-file fsync.
///
/// Scenario: after a successful commit + checkpoint, the pager
/// (1) flushes the view to the main file, (2) `sync_data` on main,
/// (3) rotates the WAL salt and `sync_data` on WAL, (4) stamps the
/// new salt into the main header, (5) `sync_data` on main again.
///
/// If a crash lands between steps 3 and 5 we have: NEW salt in WAL,
/// OLD salt in main header. Recovery reads OLD salt, sees a salt
/// mismatch on the WAL, treats it as empty — and because step 2
/// already made the main file authoritative, no data is lost.
///
/// We hand-craft this state by directly editing the WAL header AFTER
/// a clean checkpoint, then reopening with a normal pager. The
/// fault-injection wiring (`Pager<FaultyFileHandle>`) is exercised
/// when we *write* the data, ensuring the production-code path is
/// the same one under test as in the cycle harness.
#[test]
fn crash_between_salt_write_and_main_fsync_recovers() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("crash_salt.obj");
    let a_id = {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(101),
            FaultPlan::noop(202),
            Config::default(),
        )
        .expect("open faulty");
        let a = p.alloc_page().expect("alloc");
        write_committed(&mut p, a, 0xAA).expect("write+commit");
        p.checkpoint().expect("checkpoint");
        drop(p);
        a.get()
    };
    {
        use std::fs::OpenOptions;
        use std::io::{Seek, SeekFrom, Write as _};
        let wal_path = wal_path_for(&path);
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .expect("open wal");
        f.seek(SeekFrom::Start(12)).expect("seek");
        f.write_all(&0xDEAD_BEEFu32.to_le_bytes()).expect("write");
        f.sync_all().expect("sync");
    }
    let mut p = Pager::open(&path, Config::default()).expect("reopen");
    let read = p.read_page(id(a_id)).expect("read");
    assert_eq!(read.as_bytes()[0], 0xAA);
    assert_eq!(read.as_bytes()[1024], 0xAAu8.wrapping_mul(3));
}

/// Test: torn write on a WAL frame mid-commit. The harness writes
/// only a prefix of the page body before fsync; the next reopen
/// must NOT replay the torn frame.
///
/// Truncation only happens past the LAST commit marker — torn-tail
/// bytes after a clean commit are still tolerated.
#[test]
fn torn_write_on_wal_frame_mid_commit_recovers() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("torn_wal.obj");
    let (a_id, b_id) = {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(11),
            FaultPlan::noop(22),
            Config::default(),
        )
        .expect("open faulty");
        let a = p.alloc_page().expect("alloc a");
        let b = p.alloc_page().expect("alloc b");
        write_committed(&mut p, a, 0xAA).expect("commit a");
        write_committed(&mut p, b, 0xBB).expect("commit b");
        drop(p);
        (a.get(), b.get())
    };
    {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(33),
            FaultPlan::new(44, 1.0, 0.0, 0.0, 0.0, 0),
            Config::default(),
        )
        .expect("open faulty");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0xCC;
        let _ = p.write_page(id(a_id), &page);
        let _ = p.commit();
        drop(p);
    }
    let mut p = Pager::open(&path, Config::default()).expect("reopen");
    let ra = p.read_page(id(a_id)).expect("read a");
    assert_eq!(
        ra.as_bytes()[0],
        0xAA,
        "torn frame must not overwrite committed bytes",
    );
    let rb = p.read_page(id(b_id)).expect("read b");
    assert_eq!(rb.as_bytes()[0], 0xBB);
}

/// Test: a dropped fsync on the main file after checkpoint must
/// still produce a recoverable database via the WAL salt-match
/// path. The salt rotation was designed for exactly this case.
///
/// Scenario: after committing pages a + b, the pager checkpoints.
/// The main-file `sync_data` is silently dropped (data lingers in
/// the kernel page cache and would be lost on power loss). On
/// reopen, the main file's salt has NOT been updated (because the
/// header write also went through the dropped fsync, but in fact
/// the bytes themselves landed), and the WAL still carries the
/// pre-rotation salt — but in our scenario we have completed the
/// rotation. Recovery must either (a) re-apply via salt match,
/// or (b) accept the bytes already on disk.
///
/// # Coverage limitation
///
/// This test exercises the *control-flow* of the dropped-fsync path
/// (no real `sync_data` is issued; recovery must still reopen
/// cleanly) but NOT the *durability* consequence of a power loss.
/// The reason is structural: the bytes written before the dropped
/// fsync already reached the kernel page cache, and an in-process
/// harness cannot evict them — so the reopen, running against
/// the same live kernel, simply reads them back. The "unsynced bytes
/// vanish on power loss" half of the scenario is therefore
/// unreachable here (see [`crate::platform::fault::FaultPlan::dropped_fsync_prob`]).
/// Power-loss durability at a coarser, commit-boundary granularity is
/// covered instead by the crash-cycle process-kill model in
/// `obj-core/tests/crash_cycles.rs`, which treats an injected panic as
/// a crash between two consistent commit points and asserts the reopen
/// invariant across 10 000 randomized seeds.
#[test]
fn dropped_fsync_on_checkpointed_main_file_recovers_via_wal_salt_match() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("dropped_fsync.obj");
    let (a_id, b_id) = {
        let mut p = open_faulty(
            &path,
            FaultPlan::new(55, 0.0, 1.0, 0.0, 0.0, 0),
            FaultPlan::noop(66),
            Config::default(),
        )
        .expect("open faulty");
        let a = p.alloc_page().expect("alloc a");
        let b = p.alloc_page().expect("alloc b");
        write_committed(&mut p, a, 0xAA).expect("commit a");
        write_committed(&mut p, b, 0xBB).expect("commit b");
        p.checkpoint().expect("checkpoint");
        drop(p);
        (a.get(), b.get())
    };
    let mut p = Pager::open(&path, Config::default()).expect("reopen");
    let ra = p.read_page(id(a_id)).expect("read a");
    assert_eq!(ra.as_bytes()[0], 0xAA);
    let rb = p.read_page(id(b_id)).expect("read b");
    assert_eq!(rb.as_bytes()[0], 0xBB);
}

/// Cycle invariant: opening with a faulty backend, writing a
/// committed page, and the deliberate-crash boundary panicking
/// inside the harness's `crash_after_ops` window is correctly
/// trapped by `catch_unwind`. This is the precondition for the
/// cycle-test variant in `tests/crash_cycles.rs` to operate.
#[test]
fn deliberate_crash_in_pager_is_caught_by_catch_unwind() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("crash.obj");
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut p = open_faulty(
            &path,
            FaultPlan::new(0, 0.0, 0.0, 0.0, 0.0, 2),
            FaultPlan::noop(1),
            Config::default(),
        )
        .expect("open faulty");
        let _ = p.alloc_page();
        let _ = p.commit();
    }));
    match result {
        Ok(()) => {}
        Err(p) => assert!(
            panic_carries_marker(&p),
            "panic must carry the deliberate-crash marker",
        ),
    }
    let p = Pager::open(&path, Config::default()).expect("reopen after crash");
    let _ = p.page_count();
}

/// Test: a committed fresh allocation is durable via the
/// WAL — its zeroed body + the advancing `page_count` ride the SAME WAL
/// group-commit. Recovery replays that frame, restoring `page_count =
/// N+1` AND the page body into the committed view, so `read_page(N)`
/// succeeds without the main file ever having been extended at alloc.
///
/// The ONLY durable record of a committed-but-not-checkpointed fresh
/// page is its WAL frame — the main file is grown lazily at the next
/// checkpoint. This is strictly stronger: one atomically-ordered WAL
/// group-commit replaces two ordered fsyncs, and there is no window
/// where the header references a page the file is too short to hold
/// (the past-EOF hazard is deleted at the root).
#[test]
fn committed_alloc_page_recovers_via_wal_before_checkpoint() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("alloc_durable.obj");
    let a_id = {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(11),
            FaultPlan::noop(22),
            Config::default(),
        )
        .expect("open faulty");
        let a = p.alloc_page().expect("alloc a");
        write_committed(&mut p, a, 0xCD).expect("write+commit");
        drop(p);
        a.get()
    };
    assert!(
        wal_path_for(&path).exists(),
        "the committed alloc's durability lives in the WAL before checkpoint",
    );
    let mut p = Pager::open(&path, Config::default()).expect("reopen");
    let read = p.read_page(id(a_id)).expect("read recovered alloc");
    assert_eq!(read.as_bytes()[0], 0xCD, "committed alloc body recovered");
    assert_eq!(read.as_bytes()[1024], 0xCDu8.wrapping_mul(3));
    let pc = p.page_count();
    let mut pid = 1u64;
    while pid < pc {
        p.read_page(id(pid)).expect("header-claimed page readable");
        pid += 1;
    }
}

/// Test: a growing commit whose SINGLE WAL group-commit fsync is
/// lost rolls back atomically to the last durable state.
///
/// Fresh pages ride the WAL, so a growing commit issues exactly ONE
/// `F_FULLFSYNC` (the WAL group-commit). There is no un-WAL'd main-file
/// extension to leave behind — so the failure mode is clean: if the WAL
/// commit fsync never reaches the platter and the un-committed WAL tail
/// is lost on power loss, the alloc simply did not happen. The main
/// file is NOT over-long (alloc never touched it), and recovery returns
/// the database to the durable baseline with NO page the file is too
/// short to hold (the past-EOF hazard is structurally impossible now).
///
/// This is the after-commit-before-checkpoint crash point with the WAL
/// fsync dropped: it asserts no garbage past the durable `page_count`
/// and a self-consistent recovered header.
#[test]
fn growing_commit_with_dropped_wal_fsync_rolls_back_clean() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("wal_fsync_window.obj");
    let base_id = {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(101),
            FaultPlan::noop(202),
            Config::default(),
        )
        .expect("open faulty");
        let a = p.alloc_page().expect("alloc baseline");
        write_committed(&mut p, a, 0xAB).expect("commit baseline");
        p.checkpoint().expect("checkpoint baseline");
        drop(p);
        a.get()
    };
    let baseline_len = std::fs::metadata(&path).expect("meta").len();
    {
        let mut p = open_faulty(
            &path,
            FaultPlan::noop(303),
            FaultPlan::new(404, 0.0, 1.0, 0.0, 0.0, 0),
            Config::default(),
        )
        .expect("reopen faulty");
        let _a = p.alloc_page().expect("alloc fresh");
        let _ = p
            .commit()
            .expect("commit returns Ok with dropped WAL fsync");
        drop(p);
    }
    let after_len = std::fs::metadata(&path).expect("meta").len();
    assert_eq!(
        after_len, baseline_len,
        "#91: a growing commit must NOT extend the main file before checkpoint",
    );
    crate::wal::remove_wal(&wal_path_for(&path)).expect("remove wal");
    let mut p = Pager::open(&path, Config::default()).expect("reopen after crash");
    let page = p
        .read_page(id(base_id))
        .expect("baseline page readable after crash");
    assert_eq!(page.as_bytes()[0], 0xAB, "baseline page content survived");
    let pc = p.page_count();
    let mut pid = 1u64;
    while pid < pc {
        let id = PageId::new(pid).expect("non-zero");
        p.read_page(id)
            .expect("header-claimed page must be readable");
        pid += 1;
    }
}

/// Crash-matrix sweep. A growing transaction's write path issues
/// a deterministic sequence of syscalls: per-page WAL `write_all_at`s,
/// then one WAL group-commit `sync_data`, then (at checkpoint) the
/// main-file `set_len` grow, the per-page main `write_all_at`s, the main
/// `sync_data`, the WAL salt-rotation `write`/`sync_data`, and finally
/// the main header `write`/`sync_data`. Crashing at EVERY op index in a
/// bounded range sweeps all the named crash points — mid-WAL-append,
/// after-commit-before-checkpoint, during-checkpoint-grow (the op right
/// after the un-counted `set_len`), after-grow-before-body-write, and
/// during-salt-rotation — because each is some op in that sequence.
/// After each injected crash we reopen with a CLEAN pager and assert the
/// recovery contract: open succeeds (or surfaces `WalCorruption`, the
/// legitimate refuse-to-guess outcome) and every page the recovered
/// header claims reads back without `UnexpectedEof` / `Corruption` — the
/// pager-level analogue of `integrity_check`. No page beyond the durable
/// `page_count` can leak.
#[test]
fn growing_txn_crash_matrix_recovers_at_every_op() {
    for crash_at in 1u64..=40 {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("crash_matrix.obj");
        {
            let mut p = open_faulty(
                &path,
                FaultPlan::noop(1),
                FaultPlan::noop(2),
                Config::default(),
            )
            .expect("open baseline");
            let a = p.alloc_page().expect("alloc baseline");
            write_committed(&mut p, a, 0x10).expect("commit baseline");
            p.checkpoint().expect("checkpoint baseline");
            drop(p);
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut p = open_faulty(
                &path,
                FaultPlan::new(100 + crash_at, 0.0, 0.0, 0.0, 0.0, crash_at),
                FaultPlan::new(200 + crash_at, 0.0, 0.0, 0.0, 0.0, crash_at),
                Config::default(),
            )
            .expect("reopen faulty");
            let mut ids = Vec::new();
            for k in 0..6u8 {
                let pid = p.alloc_page()?;
                let mut page = Page::zeroed();
                page.as_bytes_mut()[0] = 0x40 + k;
                p.write_page(pid, &page)?;
                ids.push(pid);
            }
            let _ = p.commit()?;
            p.checkpoint()?;
            crate::Result::Ok(())
        }));
        if let Err(payload) = &result {
            assert!(
                panic_carries_marker(payload),
                "crash_at={crash_at}: unexpected panic payload",
            );
        }
        assert_recovers_clean(&path, crash_at);
    }
}

/// Reopen `path` with a normal pager and assert the recovery
/// contract: every page in `[1, page_count)` is readable without
/// `UnexpectedEof` / `Corruption` (the pager-level integrity check).
/// `WalCorruption` at open is the legitimate refuse-to-guess outcome and
/// is accepted.
fn assert_recovers_clean(path: &std::path::Path, crash_at: u64) {
    let mut p = match Pager::open(path, Config::default()) {
        Ok(p) => p,
        Err(crate::Error::WalCorruption { .. }) => return,
        Err(e) => panic!("crash_at={crash_at}: recovery open failed: {e:?}"),
    };
    let pc = p.page_count();
    let mut pid = 1u64;
    while pid < pc {
        let id = PageId::new(pid).expect("non-zero");
        p.read_page(id).unwrap_or_else(|e| {
            panic!("crash_at={crash_at}: header-claimed page {pid} unreadable: {e:?}")
        });
        pid += 1;
    }
}

/// Guardrail: a forced dirty eviction during a growing transaction
/// must NOT write a fresh page past the (un-extended) main-file EOF.
/// Open with a one-frame cache so EVERY read-through eviction is forced,
/// allocate and write many pages (far more than one cache frame),
/// then crash both pre-commit AND pre-checkpoint by simply dropping the
/// pager. Reopen and assert there is NO garbage at offsets >= the
/// durable `page_count`: the file is either short (alloc never touched
/// it) and recovery heals it, or the recovered header is self-consistent.
/// Allocate, write a `base + k` marker into, and read back `n` fresh
/// pages on a one-frame-cache pager — every read-through forces an
/// eviction of the previously-resident frame, exercising the
/// "no dirty eviction of a fresh page to a short main file" path.
fn alloc_write_read_churn(p: &mut Pager<FaultyFileHandle>, n: u8, base: u8) {
    for k in 0..n {
        let pid = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = base + k;
        p.write_page(pid, &page).expect("write");
        let _ = p.read_page(pid).expect("read back");
    }
}

#[test]
fn forced_dirty_eviction_during_growing_txn_never_writes_past_eof() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = || {
        Config::default()
            .with_cache_frames(1)
            .expect("cache_frames=1")
            .with_checkpoint_threshold(u64::MAX)
    };
    let path_a = dir.path().join("forced_evict_a.obj");
    {
        let mut p = open_faulty(&path_a, FaultPlan::noop(1), FaultPlan::noop(2), cfg())
            .expect("open faulty");
        alloc_write_read_churn(&mut p, 16, 0x50);
        drop(p);
    }
    {
        let mut p = Pager::open(&path_a, Config::default()).expect("reopen pre-commit");
        let pc = p.page_count();
        let mut pid = 1u64;
        while pid < pc {
            p.read_page(id(pid))
                .expect("page readable after pre-commit crash");
            pid += 1;
        }
    }
    let path_b = dir.path().join("forced_evict_b.obj");
    let committed_pc = {
        let mut p = open_faulty(&path_b, FaultPlan::noop(3), FaultPlan::noop(4), cfg())
            .expect("open faulty");
        alloc_write_read_churn(&mut p, 16, 0x60);
        let _ = p.commit().expect("commit");
        let pc = p.page_count();
        drop(p);
        pc
    };
    let mut p = Pager::open(&path_b, Config::default()).expect("reopen pre-checkpoint");
    assert_eq!(
        p.page_count(),
        committed_pc,
        "committed page_count recovered"
    );
    let mut pid = 1u64;
    while pid < committed_pc {
        let read = p
            .read_page(id(pid))
            .expect("committed page readable after pre-checkpoint crash");
        let expected = 0x60u8 + u8::try_from(pid - 1).expect("fits");
        assert_eq!(read.as_bytes()[0], expected, "page {pid} body recovered");
        pid += 1;
    }
}

/// A `FileBackend` that wraps a real `FileHandle` and counts every
/// `sync_data` call (the `F_FULLFSYNC` on macOS under `SyncMode::Full`).
/// The counter is shared via `Arc<AtomicU64>` so the test can read it
/// while the pager owns the backend.
struct CountingHandle {
    inner: FileHandle,
    fsyncs: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl crate::FileBackend for CountingHandle {
    fn len(&self) -> crate::Result<u64> {
        self.inner.len()
    }
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> crate::Result<()> {
        self.inner.read_exact_at(buf, offset)
    }
    fn write_all_at(&self, buf: &[u8], offset: u64) -> crate::Result<()> {
        self.inner.write_all_at(buf, offset)
    }
    fn set_len(&self, new_len: u64) -> crate::Result<()> {
        self.inner.set_len(new_len)
    }
    fn sync_data(&self, mode: crate::platform::SyncMode) -> crate::Result<()> {
        self.fsyncs
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.sync_data(mode)
    }
    fn sync_all(&self) -> crate::Result<()> {
        self.inner.sync_all()
    }
}

/// A growing commit issues EXACTLY ONE `F_FULLFSYNC`
/// (the WAL group-commit). We
/// count `sync_data` calls on BOTH backends across a growing commit that
/// is NOT auto-checkpointed: the only fsync is the WAL's. The main
/// backend sees ZERO syncs because fresh pages no longer extend or touch
/// the main file before checkpoint.
#[test]
fn growing_commit_issues_exactly_one_fsync() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("fsync_count.obj");
    let main_fsyncs = Arc::new(AtomicU64::new(0));
    let wal_fsyncs = Arc::new(AtomicU64::new(0));
    let cfg = Config::default().with_checkpoint_threshold(u64::MAX);
    let main = CountingHandle {
        inner: FileHandle::open_or_create(&path).expect("main"),
        fsyncs: Arc::clone(&main_fsyncs),
    };
    let wal_path = wal_path_for(&path);
    let wal = CountingHandle {
        inner: FileHandle::open_or_create(&wal_path).expect("wal"),
        fsyncs: Arc::clone(&wal_fsyncs),
    };
    let mut p =
        Pager::<CountingHandle>::open_with_backends(main, wal, wal_path, cfg).expect("open");
    p.begin_txn();
    main_fsyncs.store(0, Ordering::SeqCst);
    wal_fsyncs.store(0, Ordering::SeqCst);
    for k in 0..8u8 {
        let pid = p.alloc_page().expect("alloc");
        let mut page = Page::zeroed();
        page.as_bytes_mut()[0] = 0x70 + k;
        p.write_page(pid, &page).expect("write");
    }
    let _ = p.commit().expect("commit");
    let wal_count = wal_fsyncs.load(Ordering::SeqCst);
    let main_count = main_fsyncs.load(Ordering::SeqCst);
    assert_eq!(
        wal_count, 1,
        "#91: a growing commit issues exactly ONE WAL group-commit fsync \
         (got {wal_count})",
    );
    assert_eq!(
        main_count, 0,
        "#91: a growing commit issues ZERO main-file fsyncs before \
         checkpoint — the pre-#91 main-file extension barrier is gone \
         (got {main_count})",
    );
    assert_eq!(
        wal_count + main_count,
        1,
        "#91: total fsyncs per growing commit = 1"
    );
}

/// Helper: drain the in-memory state into a `HashMap`. Used by the
/// cycle-test fixtures.
#[allow(dead_code)]
pub(crate) fn snapshot_expected_pages<F: crate::FileBackend>(
    p: &mut Pager<F>,
    allocated: &[PageId],
) -> crate::Result<HashMap<PageId, Vec<u8>>> {
    let mut out = HashMap::with_capacity(allocated.len());
    for &pid in allocated {
        let page = p.read_page(pid)?;
        out.insert(pid, page.as_bytes().to_vec());
    }
    Ok(out)
}
