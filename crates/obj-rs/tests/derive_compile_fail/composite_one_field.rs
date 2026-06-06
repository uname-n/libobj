use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("a")))]
struct CompositeOneField {
    a: u32,
    b: u32,
}

fn main() {}
