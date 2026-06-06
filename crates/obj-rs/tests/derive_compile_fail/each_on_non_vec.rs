use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
struct EachOnNonVec {
    #[obj(index = each)]
    field: String,
}

fn main() {}
