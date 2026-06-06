use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct OrderV1 {
    customer_id: u64,
}

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "orders")]
#[obj(history(v1 = OrderV1))]
struct Order {
    customer_id: u64,
    placed_at: u64,
}

fn main() {}
