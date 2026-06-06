//! Portability example, exit-gate integration.
//!
//! Mirrors the verbatim snippet for the `Db::collection::<T>(name)`
//! accessor:
//!
//! ```text
//! db.attach("archive.obj", "archive")?;
//! let archived: Vec<Order> = db
//!     .collection::<Order>("archive.orders")
//!     .all()?
//!     .collect();
//! ```
//!
//! As with `design_md_queries::all_orders_iterates_every_doc`,
//! `Collection::all` ships as an owned `Vec<(Id, T)>` shape; the test
//! adapts the `.collect()` step to thread off the `Id` part exactly as
//! a user would.

#![forbid(unsafe_code)]

use obj::{Db, Document};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// The `Order` type — fields trimmed to what the portability example
/// references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Order {
    customer_id: u64,
    total_cents: u64,
}

impl Document for Order {
    const COLLECTION: &'static str = "orders";
    const VERSION: u32 = 1;
}

impl obj::Schema for Order {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("customer_id", obj::DynamicSchema::U64),
            ("total_cents", obj::DynamicSchema::U64),
        ])
    }
}

/// Exit-gate: the portability snippet compiles AND returns the
/// archived documents.
#[test]
fn design_md_attach_collection_reads_archived_orders() {
    let dir = TempDir::new().expect("tmp");
    let main_path = dir.path().join("main.obj");
    let archive_path = dir.path().join("archive.obj");

    {
        let archive_db = Db::open(&archive_path).expect("open archive");
        archive_db
            .insert(Order {
                customer_id: 1,
                total_cents: 100,
            })
            .expect("seed archive 1");
        archive_db
            .insert(Order {
                customer_id: 2,
                total_cents: 200,
            })
            .expect("seed archive 2");
    }

    let mut db = Db::open(&main_path).expect("open main");
    db.attach(&archive_path, "archive").expect("attach");

    let archived: Vec<Order> = db
        .collection::<Order>("archive.orders")
        .all()
        .expect("all on archive.orders")
        .into_iter()
        .map(|(_id, doc)| doc)
        .collect();
    assert_eq!(archived.len(), 2);
    let totals: Vec<u64> = archived.iter().map(|o| o.total_cents).collect();
    assert!(totals.contains(&100));
    assert!(totals.contains(&200));
}
