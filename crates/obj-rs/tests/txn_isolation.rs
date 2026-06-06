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
