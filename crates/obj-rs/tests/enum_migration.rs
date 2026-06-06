//! `DynamicSchema::Enum` end-to-end migration test.
//!
//! Postcard encodes enums as a varint `u32` discriminant + the
//! matched variant's payload (no length prefix). The `Enum` variant +
//! walker support + derive emission describe that shape.
//!
//! Three shapes covered here:
//!
//! 1. **Walker + hand-impl path.** A v1 `Order` carrying a
//!    `Status::Pending | Shipped { tracking } | Cancelled(reason)`
//!    enum is encoded via native postcard. We feed the bytes through
//!    `Dynamic::from_postcard_bytes` with a hand-built schema and
//!    confirm each variant shape decodes correctly.
//! 2. **Derive path.** A `#[derive(obj::Document)]` enum opts into
//!    `#[obj(schema)]`; the auto-emitted `Schema` impl matches the
//!    hand-built schema byte-for-byte.
//! 3. **Full migration.** A v1 `Order` (status: enum) is stored,
//!    then read back through a v2 `Order` whose `migrate` body
//!    matches on the `Dynamic::Enum` variant name and transforms
//!    the value.

#![forbid(unsafe_code)]

use obj::{Db, DynamicSchema, EnumVariantSchema, Schema};
use obj_core::codec::Dynamic;
use obj_core::{Document, Error, Id, Result};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

mod v1 {
    use super::{Deserialize, Document, Serialize};
    use obj_core::codec::{DynamicSchema, EnumVariantSchema, Schema};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum Status {
        Pending,
        Shipped { tracking: String },
        Cancelled(String),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Order {
        pub customer: String,
        pub status: Status,
    }

    impl Document for Order {
        const COLLECTION: &'static str = "enum_migration_orders";
        const VERSION: u32 = 1;
    }

    impl Schema for Status {
        fn schema() -> DynamicSchema {
            DynamicSchema::enumeration([
                EnumVariantSchema::new(0, "Pending", DynamicSchema::Null),
                EnumVariantSchema::new(
                    1,
                    "Shipped",
                    DynamicSchema::map([("tracking", DynamicSchema::String)]),
                ),
                EnumVariantSchema::new(2, "Cancelled", DynamicSchema::String),
            ])
        }
    }

    impl Schema for Order {
        fn schema() -> DynamicSchema {
            DynamicSchema::map([
                ("customer", DynamicSchema::String),
                ("status", <Status as Schema>::schema()),
            ])
        }
    }
}

/// Hand-built schema for `v1::Status` — exercised in the walker test
/// before the derive equivalent is verified to match.
fn handwritten_status_schema() -> DynamicSchema {
    DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Pending", DynamicSchema::Null),
        EnumVariantSchema::new(
            1,
            "Shipped",
            DynamicSchema::map([("tracking", DynamicSchema::String)]),
        ),
        EnumVariantSchema::new(2, "Cancelled", DynamicSchema::String),
    ])
}

/// Hand-built schema for the whole `v1::Order` struct, used by the
/// v2 migration path below.
fn handwritten_order_schema() -> DynamicSchema {
    DynamicSchema::map([
        ("customer", DynamicSchema::String),
        ("status", handwritten_status_schema()),
    ])
}

#[test]
fn walker_decodes_each_variant_shape() {
    let schema = handwritten_status_schema();
    for value in [
        v1::Status::Pending,
        v1::Status::Shipped {
            tracking: "TRK-001".to_owned(),
        },
        v1::Status::Cancelled("late".to_owned()),
    ] {
        let bytes = postcard::to_allocvec(&value).expect("encode");
        let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
        match (&value, dyn_view.enum_variant(), dyn_view.enum_payload()) {
            (v1::Status::Pending, Some("Pending"), Some(payload)) => {
                assert_eq!(payload, &Dynamic::Null);
            }
            (v1::Status::Shipped { tracking }, Some("Shipped"), Some(payload)) => {
                assert_eq!(
                    payload.get("tracking"),
                    Some(&Dynamic::String(tracking.clone())),
                );
            }
            (v1::Status::Cancelled(reason), Some("Cancelled"), Some(payload)) => {
                assert_eq!(payload, &Dynamic::String(reason.clone()));
            }
            other => panic!("unexpected decode: {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
#[obj(schema)]
enum DerivedStatus {
    Pending,
    Shipped { tracking: String },
    Cancelled(String),
}

#[test]
fn derive_emits_matching_enum_schema() {
    let derived = <DerivedStatus as Schema>::schema();
    let handwritten = handwritten_status_schema();
    assert_eq!(derived, handwritten);
}

#[test]
fn derive_handles_tuple_variant() {
    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq, obj::Document)]
    #[obj(schema)]
    enum WithTuple {
        Unit,
        Pair(u32, String),
    }
    let schema = <WithTuple as Schema>::schema();
    let expected = DynamicSchema::enumeration([
        EnumVariantSchema::new(0, "Unit", DynamicSchema::Null),
        EnumVariantSchema::new(
            1,
            "Pair",
            DynamicSchema::map([("0", DynamicSchema::U64), ("1", DynamicSchema::String)]),
        ),
    ]);
    assert_eq!(schema, expected);

    let bytes = postcard::to_allocvec(&WithTuple::Pair(7, "x".to_owned())).expect("encode");
    let dyn_view = Dynamic::from_postcard_bytes(&bytes, &schema).expect("walk");
    assert_eq!(dyn_view.enum_variant(), Some("Pair"));
    let payload = dyn_view.enum_payload().expect("payload");
    assert_eq!(payload.get("0"), Some(&Dynamic::U64(7)));
    assert_eq!(payload.get("1"), Some(&Dynamic::String("x".to_owned())));
}

mod v2 {
    use super::{handwritten_order_schema, Deserialize, Document, Error, Result, Serialize};
    use obj_core::codec::{Dynamic, DynamicSchema};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum Status {
        Pending,
        Shipped { tracking: String },
        Cancelled(String),
        Returned,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub struct Order {
        pub customer: String,
        pub status: Status,
        pub priority: u32,
    }

    impl Document for Order {
        const COLLECTION: &'static str = "enum_migration_orders";
        const VERSION: u32 = 2;

        fn historical_schemas() -> Vec<(u32, DynamicSchema)> {
            vec![(1, handwritten_order_schema())]
        }

        fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
            if from_version != 1 {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
            let customer = dynamic.get_str("customer")?.to_owned();
            let status_dyn = dynamic
                .get("status")
                .ok_or_else(|| Error::SchemaTypeMismatch {
                    expected: "Enum",
                    found: "absent",
                    path: "status".to_owned(),
                })?;
            let status = decode_v1_status(status_dyn)?;
            Ok(Order {
                customer,
                status,
                priority: 0,
            })
        }
    }

    fn decode_v1_status(value: &Dynamic) -> Result<Status> {
        let variant = value
            .enum_variant()
            .ok_or_else(|| Error::SchemaTypeMismatch {
                expected: "Enum",
                found: "non-enum",
                path: "status".to_owned(),
            })?;
        let payload = value
            .enum_payload()
            .ok_or_else(|| Error::SchemaTypeMismatch {
                expected: "Enum-payload",
                found: "absent",
                path: "status".to_owned(),
            })?;
        match variant {
            "Pending" => Ok(Status::Pending),
            "Shipped" => {
                let tracking = payload
                    .get("tracking")
                    .and_then(|d| match d {
                        Dynamic::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .ok_or_else(|| Error::SchemaTypeMismatch {
                        expected: "String",
                        found: "absent-or-wrong-type",
                        path: "status.Shipped.tracking".to_owned(),
                    })?;
                Ok(Status::Shipped { tracking })
            }
            "Cancelled" => match payload {
                Dynamic::String(reason) => Ok(Status::Cancelled(reason.clone())),
                _ => Err(Error::SchemaTypeMismatch {
                    expected: "String",
                    found: "non-string",
                    path: "status.Cancelled".to_owned(),
                }),
            },
            other => Err(Error::SchemaTypeMismatch {
                expected: "known variant",
                found: "unknown-variant",
                path: format!("status.{other}"),
            }),
        }
    }
}

#[test]
fn v1_to_v2_migration_through_enum_field() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("enum_orders.obj");

    let ids: Vec<Id> = {
        let db = Db::open(&path).expect("open v1");
        vec![
            db.insert(v1::Order {
                customer: "alice".to_owned(),
                status: v1::Status::Pending,
            })
            .expect("insert pending"),
            db.insert(v1::Order {
                customer: "bob".to_owned(),
                status: v1::Status::Shipped {
                    tracking: "TRK-42".to_owned(),
                },
            })
            .expect("insert shipped"),
            db.insert(v1::Order {
                customer: "carol".to_owned(),
                status: v1::Status::Cancelled("late".to_owned()),
            })
            .expect("insert cancelled"),
        ]
    };

    let db = Db::open(&path).expect("reopen v2");
    let got: Vec<v2::Order> = ids
        .iter()
        .map(|id| db.get::<v2::Order>(*id).expect("get").expect("present"))
        .collect();
    assert_eq!(got[0].customer, "alice");
    assert_eq!(got[0].status, v2::Status::Pending);
    assert_eq!(got[0].priority, 0, "v1 → v2 defaults priority");
    assert_eq!(got[1].customer, "bob");
    assert_eq!(
        got[1].status,
        v2::Status::Shipped {
            tracking: "TRK-42".to_owned(),
        },
    );
    assert_eq!(got[2].customer, "carol");
    assert_eq!(got[2].status, v2::Status::Cancelled("late".to_owned()));
}
