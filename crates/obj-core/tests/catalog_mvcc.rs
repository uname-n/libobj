//! Focused MVCC test for the catalog header-routing behavior.
//!
//! Verifies that a [`ReaderSnapshot`] taken before a writer creates a
//! new collection does NOT observe the new collection in its frozen
//! view, while a fresh snapshot taken after the writer commits DOES.
//!
//! The header update rides the WAL:
//! the snapshot captures `root_catalog` at pin time and a concurrent
//! writer's `set_root_catalog` cannot poison that captured value.
//!
//! It includes a multi-threaded variant (N reader threads + 1
//! writer thread via `std::thread::scope`) so we cover the
//! concurrent path, but the duration is short and the assertions are
//! deterministic.

#![forbid(unsafe_code)]

use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use obj_core::btree::BTree;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Catalog, CollectionDescriptor, TxnEnv, WriteTxn};
use tempfile::TempDir;

/// Open a fresh file-backed pager and lazily-initialise an empty
/// catalog inside it.  Returns the pager handle (already inside a
/// committed txn so the initial catalog root is durable on disk).
fn open_with_catalog(path: &std::path::Path) -> (Pager<FileHandle>, u64) {
    let mut pager = Pager::open(path, Config::default()).expect("pager");
    pager.begin_txn();
    let _ = Catalog::<FileHandle>::open_or_init(&mut pager).expect("init catalog");
    let _ = pager.commit().expect("commit init");
    pager.end_txn();
    let r0 = pager.root_catalog();
    assert_ne!(r0, 0, "catalog init must produce a non-zero root");
    (pager, r0)
}

/// Manually drive `Pager::begin_txn` / `commit` / `end_txn` so the
/// catalog's debug-assert sees a non-zero txn depth.  Modelled on
/// what `obj_core::txn::WriteTxn::begin/commit` do at the lower
/// abstraction level.
fn write_register_collection(pager: &mut Pager<FileHandle>, name: &str) -> u32 {
    pager.begin_txn();
    let mut catalog = Catalog::<FileHandle>::open_or_init(pager).expect("reopen catalog");
    let primary_root = BTree::<FileHandle>::empty(pager).expect("primary").root();
    let descriptor = CollectionDescriptor::new(0, primary_root.get(), 1);
    let assigned = catalog
        .insert(pager, name, descriptor)
        .expect("catalog.insert");
    let _ = pager.commit().expect("commit");
    pager.end_txn();
    assigned
}

/// `Catalog::lookup_via_snapshot` returns the descriptor as-of
/// the snapshot's pinned LSN, NOT the writer's live `Catalog.tree.root`.
///
/// Snapshot A is pinned BEFORE the writer registers `"beta"`. After
/// the writer commits, looking up `"beta"` via the snapshot's catalog
/// root returns `None` (the row did not exist at the snapshot's pin).
/// `"alpha"` (registered before snapshot A) is visible. A fresh
/// snapshot B taken AFTER the writer's commit observes both.
#[test]
fn lookup_via_snapshot_observes_only_collections_visible_at_pin_time() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc53.obj");
    let (mut pager, _r0) = open_with_catalog(&path);

    let _ = write_register_collection(&mut pager, "alpha");

    let snap_a = pager.reader_snapshot().expect("snap_a");

    let _ = write_register_collection(&mut pager, "beta");

    let beta_via_a = Catalog::<FileHandle>::lookup_via_snapshot(&pager, &snap_a, "beta")
        .expect("lookup beta via snap_a");
    assert!(
        beta_via_a.is_none(),
        "snapshot A pinned before beta's creation must NOT observe beta; \
         got {beta_via_a:?}",
    );
    let alpha_via_a = Catalog::<FileHandle>::lookup_via_snapshot(&pager, &snap_a, "alpha")
        .expect("lookup alpha via snap_a")
        .expect("alpha visible to snap_a");
    assert_eq!(alpha_via_a.collection_id, 1, "alpha gets the first id");

    let snap_b = pager.reader_snapshot().expect("snap_b");
    let beta_via_b = Catalog::<FileHandle>::lookup_via_snapshot(&pager, &snap_b, "beta")
        .expect("lookup beta via snap_b")
        .expect("beta visible to snap_b");
    assert_eq!(beta_via_b.collection_id, 2, "beta gets the second id");

    drop(snap_a);
    drop(snap_b);
}

#[test]
fn reader_snapshot_root_catalog_is_frozen_at_pin_time() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc.obj");
    let (mut pager, r0) = open_with_catalog(&path);

    let snap1 = pager.reader_snapshot().expect("snap1");
    assert_eq!(
        snap1.root_catalog(),
        r0,
        "snapshot at pin time must observe the initial catalog root",
    );

    let _ = write_register_collection(&mut pager, "users");
    let r1 = pager.root_catalog();
    assert_ne!(r0, r1, "writer must have advanced the catalog root");

    assert_eq!(
        snap1.root_catalog(),
        r0,
        "snapshot pinned before the writer's commit must NOT see the new root",
    );

    let snap2 = pager.reader_snapshot().expect("snap2");
    assert_eq!(
        snap2.root_catalog(),
        r1,
        "snapshot pinned after the commit must observe the new root",
    );

    drop(snap1);
    drop(snap2);
}

#[test]
fn fresh_reader_after_commit_sees_each_step() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_each.obj");
    let (mut pager, r0) = open_with_catalog(&path);

    let mut roots = vec![r0];
    for name in &["alpha", "beta", "gamma"] {
        let _ = write_register_collection(&mut pager, name);
        roots.push(pager.root_catalog());
    }

    let mut sorted = roots.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        roots.len(),
        "each commit must advance the catalog root via COW",
    );

    let snap = pager.reader_snapshot().expect("snap");
    assert_eq!(snap.root_catalog(), *roots.last().expect("non-empty"));
}

const CONCURRENT_READERS: usize = 4;
const CONCURRENT_WRITER_ITERS: u32 = 8;

/// Writer-thread body for the concurrent MVCC test.
fn concurrent_writer_loop(
    env: &Arc<TxnEnv<FileHandle>>,
    committed: &Arc<Mutex<Vec<u64>>>,
    barrier: &Arc<Barrier>,
) {
    barrier.wait();
    for i in 0..CONCURRENT_WRITER_ITERS {
        let name = format!("coll_{i}");
        let tx = WriteTxn::begin(env, Duration::from_secs(5)).expect("begin");
        {
            let mut pager = tx.lock_pager().expect("lock pager");
            let mut catalog = Catalog::<FileHandle>::open_or_init(&mut pager).expect("catalog");
            let primary = BTree::<FileHandle>::empty(&mut pager)
                .expect("primary")
                .root();
            let descriptor = CollectionDescriptor::new(0, primary.get(), 1);
            catalog
                .insert(&mut pager, &name, descriptor)
                .expect("insert");
        }
        tx.commit().expect("commit");
        let root = {
            let pager = env.pager().lock().expect("pager");
            pager.root_catalog()
        };
        committed.lock().expect("committed lock").push(root);
        thread::sleep(Duration::from_millis(2));
    }
}

/// Reader-thread body for the concurrent MVCC test.
fn concurrent_reader_loop(
    env: &Arc<TxnEnv<FileHandle>>,
    committed: &Arc<Mutex<Vec<u64>>>,
    barrier: &Arc<Barrier>,
) {
    barrier.wait();
    let deadline = std::time::Instant::now() + Duration::from_millis(60);
    let mut observations = Vec::<u64>::new();
    while std::time::Instant::now() < deadline {
        let snap = {
            let mut pager = env.pager().lock().expect("pager");
            pager.reader_snapshot().expect("snap")
        };
        observations.push(snap.root_catalog());
        thread::sleep(Duration::from_micros(500));
        drop(snap);
    }
    for obs in &observations {
        let g = committed.lock().expect("committed lock");
        assert!(
            g.contains(obs),
            "reader observed root {obs} that was never committed; \
             known-committed roots: {g:?}",
        );
    }
}

/// N reader threads + 1 writer thread.  Each reader takes a snapshot,
/// records its pinned `root_catalog`, then briefly delays.  The
/// writer creates a new collection in each iteration.  Every
/// reader's recorded value must equal one of the snapshot-time
/// committed roots (i.e. NO reader can observe a half-updated value
/// the writer was mid-staging).
#[test]
fn n_readers_one_writer_each_snapshot_sees_a_committed_root() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("mvcc_concurrent.obj");
    let (pager, _r0) = open_with_catalog(&path);
    let lock_file = Arc::new(FileHandle::open_or_create(&path).expect("lock file"));
    let env = Arc::new(TxnEnv::new(pager, Some(lock_file)));
    let barrier = Arc::new(Barrier::new(CONCURRENT_READERS + 1));
    let committed_roots = Arc::new(Mutex::new(Vec::<u64>::new()));
    {
        let mut g = committed_roots.lock().expect("seed lock");
        let pager = env.pager().lock().expect("pager");
        g.push(pager.root_catalog());
    }
    thread::scope(|scope| {
        let writer_env = Arc::clone(&env);
        let writer_committed = Arc::clone(&committed_roots);
        let writer_barrier = Arc::clone(&barrier);
        scope.spawn(move || {
            concurrent_writer_loop(&writer_env, &writer_committed, &writer_barrier);
        });
        for _ in 0..CONCURRENT_READERS {
            let env = Arc::clone(&env);
            let committed = Arc::clone(&committed_roots);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                concurrent_reader_loop(&env, &committed, &barrier);
            });
        }
    });
}
