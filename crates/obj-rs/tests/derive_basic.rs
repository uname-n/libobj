//! `#[derive(obj::Document)]` integration tests.
//!
//! These tests verify the proc-macro emits a working
//! `obj_core::Document` implementation. We exercise the macro against
//! a live `Db` (insert + get) to catch any wiring mismatches between
//! the generated impl and the catalog reconciliation path.

use obj::{Db, Document, IndexKind};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Bare derive — no `#[obj(...)]` anywhere. Confirms the default
/// COLLECTION (type name) and VERSION (1) match what the
/// hand-written impls supplied.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct BareDoc {
    a: u32,
    b: String,
}

#[test]
fn bare_derive_constants_match_type_name() {
    assert_eq!(<BareDoc as Document>::COLLECTION, "BareDoc");
    assert_eq!(<BareDoc as Document>::VERSION, 1);
}

#[test]
fn bare_derive_round_trips_through_db() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("bare.obj");
    let db = Db::open(&path).expect("open");

    let id = db
        .insert(BareDoc {
            a: 42,
            b: "hello".to_owned(),
        })
        .expect("insert");

    let back: BareDoc = db.get::<BareDoc>(id).expect("get").expect("present");
    assert_eq!(
        back,
        BareDoc {
            a: 42,
            b: "hello".to_owned(),
        }
    );
}

#[test]
fn bare_derive_default_indexes_is_empty() {
    let specs = <BareDoc as Document>::indexes();
    assert!(specs.is_empty(), "no #[obj(index)] → empty indexes()");
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 3)]
struct CustomerV3 {
    name: String,
}

#[test]
fn version_override_sets_const() {
    assert_eq!(<CustomerV3 as Document>::VERSION, 3);
    assert_eq!(<CustomerV3 as Document>::COLLECTION, "CustomerV3");
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(collection = "people")]
struct Customer {
    name: String,
}

#[test]
fn collection_override_sets_const() {
    assert_eq!(<Customer as Document>::COLLECTION, "people");
    assert_eq!(<Customer as Document>::VERSION, 1);
}

#[test]
fn collection_override_round_trips_through_db() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("people.obj");
    let db = Db::open(&path).expect("open");

    let id = db
        .insert(Customer {
            name: "Ada".to_owned(),
        })
        .expect("insert");

    let back: Customer = db.get::<Customer>(id).expect("get").expect("present");
    assert_eq!(back.name, "Ada");
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 2)]
#[obj(collection = "two_attrs")]
struct TwoAttrs {
    x: u32,
}

#[test]
fn two_obj_attributes_compose() {
    assert_eq!(<TwoAttrs as Document>::VERSION, 2);
    assert_eq!(<TwoAttrs as Document>::COLLECTION, "two_attrs");
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 7, collection = "combo")]
struct Combo {
    x: u32,
}

#[test]
fn combined_obj_attribute_compose() {
    assert_eq!(<Combo as Document>::VERSION, 7);
    assert_eq!(<Combo as Document>::COLLECTION, "combo");
}

// allow: the `_id` / `_no` field-name suffixes mirror a realistic order record and
// are what the index-emission test asserts on; renaming them would weaken the fixture.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct Order {
    #[obj(index)]
    customer_id: u64,

    #[obj(index = unique)]
    order_no: String,

    #[obj(index = each)]
    tags: Vec<String>,

    total_cents: u64,
}

#[test]
fn field_index_attrs_emit_specs_in_declaration_order() {
    let specs = <Order as Document>::indexes();
    assert_eq!(specs.len(), 3, "three indexed fields");

    assert_eq!(specs[0].name, "customer_id");
    assert_eq!(specs[0].kind, IndexKind::Standard);
    assert_eq!(specs[0].key_paths, vec!["customer_id".to_owned()]);

    assert_eq!(specs[1].name, "order_no");
    assert_eq!(specs[1].kind, IndexKind::Unique);
    assert_eq!(specs[1].key_paths, vec!["order_no".to_owned()]);

    assert_eq!(specs[2].name, "tags");
    assert_eq!(specs[2].kind, IndexKind::Each);
    assert_eq!(specs[2].key_paths, vec!["tags".to_owned()]);
}

#[test]
fn field_index_attrs_drive_catalog_reconciliation() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("orders.obj");
    let db = Db::open(&path).expect("open");

    let id_a = db
        .insert(Order {
            customer_id: 7,
            order_no: "ORD-A".to_owned(),
            tags: vec!["red".to_owned(), "blue".to_owned()],
            total_cents: 100,
        })
        .expect("insert a");
    let _id_b = db
        .insert(Order {
            customer_id: 7,
            order_no: "ORD-B".to_owned(),
            tags: vec!["green".to_owned()],
            total_cents: 250,
        })
        .expect("insert b");

    let by_unique: Option<Order> = db
        .find_unique::<Order>("order_no", "ORD-A".to_owned())
        .expect("find_unique");
    assert_eq!(by_unique.expect("found A").customer_id, 7);

    let back: Option<Order> = db.get::<Order>(id_a).expect("get a");
    assert_eq!(back.expect("present").total_cents, 100);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(collection = "named_idx")]
struct NamedIdx {
    #[obj(index, name = "by_status")]
    status: u32,
}

#[test]
fn field_index_custom_name_overrides_default() {
    let specs = <NamedIdx as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "by_status");
    assert_eq!(specs[0].key_paths, vec!["status".to_owned()]);
    assert_eq!(specs[0].kind, IndexKind::Standard);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("customer_id", "placed_at")))]
struct OrderHistory {
    customer_id: u64,
    placed_at: u64,
    payload: String,
}

#[test]
fn composite_attr_emits_default_name() {
    let specs = <OrderHistory as Document>::indexes();
    assert_eq!(specs.len(), 1, "one composite, no field indexes");
    assert_eq!(specs[0].kind, IndexKind::Composite);
    assert_eq!(
        specs[0].key_paths,
        vec!["customer_id".to_owned(), "placed_at".to_owned()]
    );
    assert_eq!(specs[0].name, "customer_id__placed_at");
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("a", "b"), name = "by_a_b"))]
struct CompositeNamed {
    a: u32,
    b: u32,
}

#[test]
fn composite_attr_emits_custom_name() {
    let specs = <CompositeNamed as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "by_a_b");
    assert_eq!(specs[0].kind, IndexKind::Composite);
    assert_eq!(specs[0].key_paths, vec!["a".to_owned(), "b".to_owned()]);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("a", "b")))]
#[obj(index_composite(fields = ("b", "c"), name = "by_b_c"))]
struct TwoComposites {
    a: u32,
    b: u32,
    c: u32,
}

#[test]
fn two_composite_attrs_compose() {
    let specs = <TwoComposites as Document>::indexes();
    assert_eq!(specs.len(), 2, "two composites");
    assert_eq!(specs[0].name, "a__b");
    assert_eq!(specs[1].name, "by_b_c");
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("customer_id", "placed_at")))]
struct OrderMix {
    #[obj(index)]
    customer_id: u64,

    placed_at: u64,
    line_items: Vec<String>,
}

#[test]
fn field_and_composite_indexes_compose_field_first() {
    let specs = <OrderMix as Document>::indexes();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].kind, IndexKind::Standard);
    assert_eq!(specs[0].name, "customer_id");
    assert_eq!(specs[1].kind, IndexKind::Composite);
    assert_eq!(specs[1].name, "customer_id__placed_at");
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("customer_id", "placed_at"))]
struct OrderShortComposite {
    customer_id: u64,
    placed_at: u64,
    payload: String,
}

#[test]
fn short_composite_attr_emits_default_name() {
    let specs = <OrderShortComposite as Document>::indexes();
    assert_eq!(specs.len(), 1, "one composite, no field indexes");
    assert_eq!(specs[0].kind, IndexKind::Composite);
    assert_eq!(
        specs[0].key_paths,
        vec!["customer_id".to_owned(), "placed_at".to_owned()]
    );
    assert_eq!(specs[0].name, "customer_id__placed_at");
}

#[test]
fn short_composite_attr_round_trips_through_db() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("short_composite.obj");
    let db = Db::open(&path).expect("open");

    for i in 0..3u64 {
        let _ = db
            .insert(OrderShortComposite {
                customer_id: 7,
                placed_at: i,
                payload: format!("p{i}"),
            })
            .expect("insert");
    }

    let pairs: Vec<(Vec<u8>, OrderShortComposite)> = db
        .read_transaction(|tx| {
            tx.collection::<OrderShortComposite>()?
                .index_range("customer_id__placed_at", ..)?
                .collect()
        })
        .expect("range");
    assert_eq!(pairs.len(), 3);
    for window in pairs.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("a", "b"), name = "by_a_b")]
struct ShortCompositeNamed {
    a: u32,
    b: u32,
}

#[test]
fn short_composite_attr_accepts_custom_name() {
    let specs = <ShortCompositeNamed as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "by_a_b");
    assert_eq!(specs[0].kind, IndexKind::Composite);
    assert_eq!(specs[0].key_paths, vec!["a".to_owned(), "b".to_owned()]);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("a", "b"), name = "by_a_b")]
#[obj(index = ("b", "c"))]
struct TwoShortCompositesOneNamed {
    a: u32,
    b: u32,
    c: u32,
}

#[test]
fn short_composite_custom_name_is_per_attribute() {
    let specs = <TwoShortCompositesOneNamed as Document>::indexes();
    assert_eq!(specs.len(), 2);
    // The `name` binds only to the `index` in its own `#[obj(...)]`;
    // the second composite keeps the default joined name.
    assert_eq!(specs[0].name, "by_a_b");
    assert_eq!(specs[0].key_paths, vec!["a".to_owned(), "b".to_owned()]);
    assert_eq!(specs[1].name, "b__c");
    assert_eq!(specs[1].key_paths, vec!["b".to_owned(), "c".to_owned()]);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("a", "b"))]
#[obj(index = ("b", "c"))]
struct TwoShortComposites {
    a: u32,
    b: u32,
    c: u32,
}

#[test]
fn two_short_composite_attrs_compose() {
    let specs = <TwoShortComposites as Document>::indexes();
    assert_eq!(specs.len(), 2, "two composites");
    assert_eq!(specs[0].name, "a__b");
    assert_eq!(specs[0].key_paths, vec!["a".to_owned(), "b".to_owned()]);
    assert_eq!(specs[1].name, "b__c");
    assert_eq!(specs[1].key_paths, vec!["b".to_owned(), "c".to_owned()]);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("a", "b"))]
#[obj(index_composite(fields = ("b", "c"), name = "by_b_c"))]
struct MixedCompositeForms {
    a: u32,
    b: u32,
    c: u32,
}

#[test]
fn mixed_composite_forms_compose() {
    let specs = <MixedCompositeForms as Document>::indexes();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].name, "a__b");
    assert_eq!(specs[1].name, "by_b_c");
}

#[test]
fn composite_attr_drives_catalog_reconciliation_with_range_scan() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("history.obj");
    let db = Db::open(&path).expect("open");

    for i in 0..3u64 {
        let _ = db
            .insert(OrderHistory {
                customer_id: 7,
                placed_at: i,
                payload: format!("p{i}"),
            })
            .expect("insert");
    }

    let pairs: Vec<(Vec<u8>, OrderHistory)> = db
        .read_transaction(|tx| {
            tx.collection::<OrderHistory>()?
                .index_range("customer_id__placed_at", ..)?
                .collect()
        })
        .expect("range");
    assert_eq!(pairs.len(), 3);
    for window in pairs.windows(2) {
        assert!(window[0].0 <= window[1].0);
    }
}
