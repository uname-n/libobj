//! Read-transaction snapshot isolation for range / count.
//!
//! `read_transaction` is documented as snapshot-isolated: every read
//! inside the closure observes the database as-of the snapshot's
//! pinned LSN, regardless of what a concurrent writer commits
//! afterwards. Before the fix, point `get` and `all` were
//! snapshot-pinned but `count` and `index_range` enumerated the LIVE
//! B-tree (`tree.range(&mut pager, ..)` against the pager's WAL
//! overlay), so a concurrent writer's POST-snapshot index entries and
//! primary rows leaked into a read txn's range/count.
//!
//! After the fix, `Collection::count_all`, `Collection::index_range`,
//! and `Collection::count_index_range` route through
//! `BTree::range_via_snapshot`, resolving every descent and leaf read
//! as-of the pinned snapshot. These tests assert the read txn's
//! range/count stay frozen at txn-begin while a fresh read txn opened
//! after the writer commits DOES observe the new entries.

#![forbid(unsafe_code)]

use std::ops::Bound;
use std::sync::mpsc;
use std::thread;

use obj::{Db, Document, IndexSpec};
use obj_core::codec::Dynamic;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// One indexed field (`placed_at`, `Standard`) so `index_range` /
/// `count_index_range` have an index B-tree to walk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    placed_at: u64,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::standard("placed_at", "placed_at").expect("standard")]
    }
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("placed_at", obj::DynamicSchema::U64),
        ])
    }
}

/// Hermetic file-backed `Db`. The file must be file-backed (not a
/// memory pager) for the MVCC surface to exist — a memory pager has
/// no WAL and the snapshot read falls through to the live cache.
fn fresh_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("txn-isolation.obj");
    let db = Db::open(&path).expect("open");
    (db, dir)
}

/// Insert `[lo, hi)` orders, one per committed write txn, so each row
/// is durable before the next.
fn seed(db: &Db, lo: u64, hi: u64) {
    for i in lo..hi {
        let _ = db
            .insert(Order {
                customer_id: i,
                placed_at: i,
            })
            .expect("seed insert");
    }
}

/// Count `placed_at ∈ [0, 1000)` index hits in a fresh read txn opened
/// AFTER the caller's writer commits, so it observes post-snapshot rows.
fn count_full_range(db: &Db) -> usize {
    db.read_transaction(|tx| {
        let coll = tx.collection::<Order>()?;
        let hits: Vec<(Vec<u8>, Order)> = coll
            .index_range("placed_at", 0u64..1000)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(hits.len())
    })
    .expect("fresh read")
}

/// Assert a read txn's snapshot is frozen at 10 pre-snapshot rows
/// (`placed_at ∈ [0, 10)`) even though the writer has since committed
/// rows in `[10, 30)`. Exercises all three snapshot-pinned read paths.
fn assert_frozen_at_ten(coll: &obj::Collection<'_, Order>) -> obj::Result<()> {
    let total = coll.count_all()?;
    assert_eq!(
        total, 10,
        "count_all leaked post-snapshot rows \
         (#12 — count walked the live B-tree, not the snapshot)",
    );

    let full: (Bound<Dynamic>, Bound<Dynamic>) = (Bound::Unbounded, Bound::Unbounded);
    let hits: Vec<(Vec<u8>, Order)> = coll
        .index_range("placed_at", full)?
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        hits.len(),
        10,
        "index_range leaked post-snapshot entries \
         (#12 — range descent walked the live B-tree)",
    );
    assert!(
        hits.iter().all(|(_k, o)| o.placed_at < 10),
        "index_range surfaced a post-snapshot placed_at",
    );

    let windowed = coll.count_index_range(
        "placed_at",
        (
            Bound::Included(Dynamic::U64(10)),
            Bound::Excluded(Dynamic::U64(30)),
        ),
    )?;
    assert_eq!(
        windowed, 0,
        "count_index_range observed post-snapshot entries in [10, 30) \
         (#12 — index range/count bypassed the snapshot)",
    );
    Ok(())
}

/// A reader that pins its snapshot BEFORE a concurrent writer commits
/// new indexed documents must NOT observe the post-snapshot rows in
/// `count_all`, `index_range`, or `count_index_range` — they stay
/// consistent with the pinned LSN. A fresh reader opened after the
/// writer commits DOES see them.
///
/// Ordering is deterministic via two channels (mirroring
/// `snapshot_catalog_mvcc.rs`): the writer waits for the reader to
/// confirm the snapshot is pinned, then commits; the reader waits for
/// the writer to confirm the commit before issuing its frozen reads.
#[test]
fn read_txn_range_and_count_ignore_post_snapshot_writes() {
    let (db, _dir) = fresh_db();
    seed(&db, 0, 10);

    let (snap_tx, snap_rx) = mpsc::channel::<()>();
    let (commit_tx, commit_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let db_r = &db;
        let db_w = &db;

        s.spawn(move || {
            db_r.read_transaction(|tx| {
                let coll = tx.collection::<Order>()?;
                snap_tx.send(()).expect("snap_tx");
                commit_rx.recv().expect("commit_rx");
                assert_frozen_at_ten(&coll)
            })
            .expect("read_transaction");
        });

        s.spawn(move || {
            snap_rx.recv().expect("snap_rx");
            seed(db_w, 10, 30);
            commit_tx.send(()).expect("commit_tx");
        });
    });

    let (total, windowed) = db
        .read_transaction(|tx| {
            let coll = tx.collection::<Order>()?;
            let total = coll.count_all()?;
            let windowed = coll.count_index_range(
                "placed_at",
                (
                    Bound::Included(Dynamic::U64(10)),
                    Bound::Excluded(Dynamic::U64(30)),
                ),
            )?;
            Ok((total, windowed))
        })
        .expect("fresh read");
    assert_eq!(total, 30, "fresh reader must see all 30 rows");
    assert_eq!(
        windowed, 20,
        "fresh reader must see the 20 rows in [10, 30)"
    );
}

/// Streaming `iter_range` must honour snapshot isolation across its
/// per-batch refills exactly like the eager `index_range`.
///
/// Regression for #9: `IterIndexRange::refill` walked the index B-tree
/// via the LIVE pager path (`BTree::open` + `tree.range`) regardless of
/// handle mode, so a Read-mode iterator descended the snapshot-pinned
/// index root through the live pager — observing a concurrent writer's
/// post-snapshot COW commits. With >256 in-range entries a second
/// `refill()` runs; if a writer commits an in-range insert between
/// `next()` calls, the live refill surfaces the new index entry while
/// the per-row `get`-back reads the snapshot primary tree (no such doc),
/// producing a spurious `Error::Corruption`. After the fix the Read-mode
/// refill walks `range_via_snapshot`, so the new entry stays invisible
/// and `iter_range` agrees with `index_range` on the same snapshot.
///
/// `seed(0, 300)` gives 300 matching entries (> the 256 batch size) so
/// the iterator MUST refill at least once after the writer commits.
#[test]
fn iter_range_streaming_refill_honors_snapshot() {
    let (db, _dir) = fresh_db();
    seed(&db, 0, 300);

    let (snap_tx, snap_rx) = mpsc::channel::<()>();
    let (commit_tx, commit_rx) = mpsc::channel::<()>();

    thread::scope(|s| {
        let db_r = &db;
        let db_w = &db;

        s.spawn(move || {
            db_r.read_transaction(|tx| {
                let coll = tx.collection::<Order>()?;
                snap_tx.send(()).expect("snap_tx");

                // First `next()` performs refill #1 (buffers 256 entries).
                let mut iter = coll.iter_range("placed_at", 0u64..1000)?;
                let first = iter.next().expect("iterator must yield a first row")?;

                // Let the writer commit an in-range insert, then drain
                // the rest — the drain triggers refill #2 AFTER the
                // commit, the window the bug manifested in.
                commit_rx.recv().expect("commit_rx");

                let mut streamed: Vec<(Vec<u8>, Order)> = vec![first];
                for step in iter {
                    // Pre-fix this `?` propagated Error::Corruption when
                    // refill #2 saw the writer's post-snapshot entry.
                    streamed.push(step?);
                }

                // The eager range over the IDENTICAL bounds on the SAME
                // snapshot handle must match the streamed result exactly.
                let eager: Vec<(Vec<u8>, Order)> = coll
                    .index_range("placed_at", 0u64..1000)?
                    .collect::<Result<Vec<_>, _>>()?;

                assert_eq!(
                    streamed.len(),
                    300,
                    "iter_range leaked/lost rows across refills under a \
                     concurrent writer (#9 — refill walked the live pager)",
                );
                assert!(
                    streamed.iter().all(|(_k, o)| o.placed_at < 300),
                    "iter_range surfaced a post-snapshot placed_at (#9)",
                );
                assert_eq!(
                    streamed, eager,
                    "streaming iter_range and eager index_range diverged \
                     on the same snapshot (#9)",
                );
                Ok(())
            })
            .expect("read_transaction must not surface Error::Corruption");
        });

        s.spawn(move || {
            snap_rx.recv().expect("snap_rx");
            // In-range insert (placed_at=290 sorts after the first 256
            // entries, so it lands in the reader's refill #2 window) with
            // a fresh id absent from the reader's pinned primary tree.
            let _ = db_w
                .insert(Order {
                    customer_id: 9999,
                    placed_at: 290,
                })
                .expect("writer insert");
            commit_tx.send(()).expect("commit_tx");
        });
    });

    // A fresh reader opened AFTER the writer commits sees all 301 rows.
    assert_eq!(
        count_full_range(&db),
        301,
        "fresh reader must see the post-snapshot insert"
    );
}
