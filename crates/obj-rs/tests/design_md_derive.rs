//! Derive code-sample compile gate.
//!
//! Every `#[derive(obj::Document)]` block for documents and
//! collections + indexes must compile under the derive surface and
//! produce the described `Document` impl. This file mirrors each
//! sample 1:1, hand-rolling the few helper types the samples reference
//! (`Timestamp`, `OrderStatus`, `CustomerTier`, `Region`, `LineItem`,
//! `Address`).
//!
//! These tests serve as a gate: if a sample stops compiling, the
//! failure points at the exact divergence between the intended sample
//! and the actual derive surface.

use obj::{Document, DynamicSchema, EnumVariantSchema, IndexKind, Schema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Timestamp(u64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum OrderStatus {
    Pending,
    Shipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum CustomerTier {
    Standard,
    Gold,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Region {
    UsEast,
    EuWest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LineItem {
    sku: String,
    qty: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Address {
    street: String,
    city: String,
    zip: String,
}

impl Schema for Timestamp {
    fn schema() -> DynamicSchema {
        DynamicSchema::U64
    }
}

impl Schema for OrderStatus {
    fn schema() -> DynamicSchema {
        DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "Pending", DynamicSchema::Null),
            EnumVariantSchema::new(1, "Shipped", DynamicSchema::Null),
        ])
    }
}

impl Schema for CustomerTier {
    fn schema() -> DynamicSchema {
        DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "Standard", DynamicSchema::Null),
            EnumVariantSchema::new(1, "Gold", DynamicSchema::Null),
        ])
    }
}

impl Schema for Region {
    fn schema() -> DynamicSchema {
        DynamicSchema::enumeration([
            EnumVariantSchema::new(0, "UsEast", DynamicSchema::Null),
            EnumVariantSchema::new(1, "EuWest", DynamicSchema::Null),
        ])
    }
}

impl Schema for LineItem {
    fn schema() -> DynamicSchema {
        DynamicSchema::map([("sku", DynamicSchema::String), ("qty", DynamicSchema::U64)])
    }
}

impl Schema for Address {
    fn schema() -> DynamicSchema {
        DynamicSchema::map([
            ("street", DynamicSchema::String),
            ("city", DynamicSchema::String),
            ("zip", DynamicSchema::String),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct OrderS1 {
    customer_id: obj::Id,
    line_items: Vec<LineItem>,
    status: OrderStatus,
    placed_at: Timestamp,
    shipped_at: Option<Timestamp>,
}

#[test]
fn design_md_sample_1_order_compiles() {
    assert_eq!(<OrderS1 as Document>::COLLECTION, "OrderS1");
    assert_eq!(<OrderS1 as Document>::VERSION, 1);
    assert!(<OrderS1 as Document>::indexes().is_empty());
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 3)]
struct CustomerS2 {
    name: String,
    email: String,
    tier: CustomerTier,
    region: Region,
}

#[test]
fn design_md_sample_2_customer_version_3_compiles() {
    assert_eq!(<CustomerS2 as Document>::VERSION, 3);
    assert_eq!(<CustomerS2 as Document>::COLLECTION, "CustomerS2");
    assert!(<CustomerS2 as Document>::indexes().is_empty());
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct OrderS3 {
    #[obj(index)]
    customer_id: obj::Id,

    #[obj(index)]
    status: OrderStatus,

    #[obj(index)]
    placed_at: Timestamp,

    line_items: Vec<LineItem>,
}

#[test]
fn design_md_sample_3_order_with_three_field_indexes_compiles() {
    let specs = <OrderS3 as Document>::indexes();
    assert_eq!(specs.len(), 3);
    assert_eq!(specs[0].name, "customer_id");
    assert_eq!(specs[0].kind, IndexKind::Standard);
    assert_eq!(specs[1].name, "status");
    assert_eq!(specs[1].kind, IndexKind::Standard);
    assert_eq!(specs[2].name, "placed_at");
    assert_eq!(specs[2].kind, IndexKind::Standard);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct CustomerS4 {
    #[obj(index = unique)]
    email: String,

    #[obj(index)]
    tier: CustomerTier,

    name: String,
}

#[test]
fn design_md_sample_4_customer_with_unique_and_standard_compiles() {
    let specs = <CustomerS4 as Document>::indexes();
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].name, "email");
    assert_eq!(specs[0].kind, IndexKind::Unique);
    assert_eq!(specs[1].name, "tier");
    assert_eq!(specs[1].kind, IndexKind::Standard);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
struct OrderS5Each {
    #[obj(index = each)]
    line_items: Vec<LineItem>,
}

#[test]
fn design_md_sample_5_each_on_vec_compiles() {
    let specs = <OrderS5Each as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "line_items");
    assert_eq!(specs[0].kind, IndexKind::Each);
    assert_eq!(specs[0].key_paths, vec!["line_items".to_owned()]);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index = ("customer_id", "placed_at"))]
struct OrderS6Composite {
    customer_id: obj::Id,
    placed_at: Timestamp,
}

#[test]
fn design_md_sample_6_composite_index_compiles() {
    let specs = <OrderS6Composite as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "customer_id__placed_at");
    assert_eq!(specs[0].kind, IndexKind::Composite);
    assert_eq!(
        specs[0].key_paths,
        vec!["customer_id".to_owned(), "placed_at".to_owned()]
    );
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("customer_id", "placed_at")))]
struct OrderS6CompositeLong {
    customer_id: obj::Id,
    placed_at: Timestamp,
}

#[test]
fn design_md_sample_6_composite_long_form_still_compiles() {
    let specs = <OrderS6CompositeLong as Document>::indexes();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "customer_id__placed_at");
    assert_eq!(specs[0].kind, IndexKind::Composite);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, obj::Document)]
#[obj(version = 1)]
struct CustomerWithAddress {
    name: String,
    email: String,
    address: Address,
}

#[test]
fn design_md_customer_with_nested_address_compiles() {
    assert_eq!(
        <CustomerWithAddress as Document>::COLLECTION,
        "CustomerWithAddress"
    );
    assert_eq!(<CustomerWithAddress as Document>::VERSION, 1);
}
