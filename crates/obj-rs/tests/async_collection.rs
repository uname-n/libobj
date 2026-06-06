//! `AsyncCollection` coverage tests.
//!
//! Exercises every public method on [`obj::asynchronous::AsyncCollection`]
//! — `get`, `all`, `count_all`, and `find_unique` — through the Tokio
//! multi-thread runtime, plus error paths for a missing document and a
//! missing collection.
//!
//! Writes are performed via [`obj::asynchronous::AsyncDb::transaction`] (the
//! intended write path for the async surface).  The `AsyncCollection` handle
//! obtained from [`obj::asynchronous::AsyncDb::collection`] provides the
//! read-only view exercised by every test below.

#![cfg(feature = "async")]

use obj::asynchronous::AsyncDb;
use obj::{Document, IndexSpec};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Test document type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Widget {
    name: String,
    weight_g: u64,
}

impl Document for Widget {
    const COLLECTION: &'static str = "widgets";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![IndexSpec::unique("by_name", "name").expect("unique spec")]
    }
}

impl obj::Schema for Widget {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("name", obj::DynamicSchema::String),
            ("weight_g", obj::DynamicSchema::U64),
        ])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Insert a batch of `Widget`s in a single write transaction.
async fn insert_widgets(db: &AsyncDb, widgets: Vec<Widget>) -> obj::Result<Vec<obj::Id>> {
    db.transaction(move |tx| {
        let coll = tx.collection::<Widget>()?;
        let mut ids = Vec::with_capacity(widgets.len());
        for w in widgets {
            ids.push(coll.insert(w)?);
        }
        Ok(ids)
    })
    .await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn collection_get_returns_inserted_document() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let ids = insert_widgets(
        &db,
        vec![Widget {
            name: "Sprocket".into(),
            weight_g: 42,
        }],
    )
    .await?;
    let id = ids[0];

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let got = coll.get(id).await?;
    assert_eq!(
        got,
        Some(Widget {
            name: "Sprocket".into(),
            weight_g: 42,
        })
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_get_returns_none_for_missing_id() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    // Insert one document to ensure the collection is created, then look
    // up an ID that was never inserted.
    let ids = insert_widgets(
        &db,
        vec![Widget {
            name: "Bolt".into(),
            weight_g: 5,
        }],
    )
    .await?;
    let existing_id = ids[0];

    // Manufacture an ID that cannot equal the one just allocated.
    let missing_raw = existing_id.get().saturating_add(999);
    let missing_id = obj::Id::try_new(missing_raw).expect("non-zero id");

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let got = coll.get(missing_id).await?;
    assert_eq!(got, None, "expected None for an ID that was never inserted");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_all_returns_every_inserted_document() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let batch: Vec<Widget> = (0..5u64)
        .map(|i| Widget {
            name: format!("part-{i}"),
            weight_g: i * 10,
        })
        .collect();
    insert_widgets(&db, batch.clone()).await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let rows = coll.all().await?;
    assert_eq!(rows.len(), 5, "all() must return all 5 inserted documents");

    // Verify every inserted name is present.
    let names: std::collections::HashSet<&str> =
        rows.iter().map(|(_id, w)| w.name.as_str()).collect();
    for i in 0..5u32 {
        assert!(
            names.contains(format!("part-{i}").as_str()),
            "part-{i} missing from all()"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_count_all_matches_insert_count() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    let batch: Vec<Widget> = (0..8u64)
        .map(|i| Widget {
            name: format!("item-{i}"),
            weight_g: i,
        })
        .collect();
    insert_widgets(&db, batch).await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let count = coll.count_all().await?;
    assert_eq!(count, 8, "count_all() must equal the number of inserts");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_count_all_is_zero_on_empty_collection() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    // Touch the collection via an empty transaction so it exists.
    db.transaction(|tx| {
        let _coll = tx.collection::<Widget>()?;
        Ok(())
    })
    .await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let count = coll.count_all().await?;
    assert_eq!(count, 0, "freshly-created collection must report count 0");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_find_unique_returns_document_by_index_key() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    insert_widgets(
        &db,
        vec![
            Widget {
                name: "Alpha".into(),
                weight_g: 1,
            },
            Widget {
                name: "Beta".into(),
                weight_g: 2,
            },
        ],
    )
    .await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let found = coll.find_unique("by_name", "Beta".to_owned()).await?;
    assert_eq!(
        found,
        Some(Widget {
            name: "Beta".into(),
            weight_g: 2,
        }),
        "find_unique must return the document whose 'name' index key matches"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_find_unique_returns_none_for_absent_key() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    insert_widgets(
        &db,
        vec![Widget {
            name: "Gamma".into(),
            weight_g: 7,
        }],
    )
    .await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let found = coll.find_unique("by_name", "Zeta".to_owned()).await?;
    assert_eq!(found, None, "find_unique must return None for an absent key");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_get_errors_on_nonexistent_collection() {
    // A collection that was never written does not exist in the catalog;
    // the lazy async handle surfaces CollectionNotFound on the first call.
    let db = AsyncDb::memory().await.expect("memory db");
    let coll = db.collection::<Widget>("no_such_collection");
    let id = obj::Id::try_new(1).expect("non-zero");
    let result = coll.get(id).await;
    assert!(
        result.is_err(),
        "get() on a non-existent collection must return Err"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn collection_clone_is_independently_usable() -> obj::Result<()> {
    let db = AsyncDb::memory().await?;
    insert_widgets(
        &db,
        vec![Widget {
            name: "Original".into(),
            weight_g: 99,
        }],
    )
    .await?;

    let coll = db.collection::<Widget>(Widget::COLLECTION);
    let coll2 = coll.clone();

    let count1 = coll.count_all().await?;
    let count2 = coll2.count_all().await?;
    assert_eq!(count1, count2, "cloned AsyncCollection must see the same data");
    Ok(())
}
