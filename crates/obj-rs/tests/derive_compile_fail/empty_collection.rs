use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(collection = "")]
struct EmptyCollection {
    x: u32,
}

fn main() {}
