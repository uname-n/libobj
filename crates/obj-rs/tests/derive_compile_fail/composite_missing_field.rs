use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(index_composite(fields = ("nonexistent", "a")))]
struct CompositeMissingField {
    a: u32,
    b: u32,
}

fn main() {}
