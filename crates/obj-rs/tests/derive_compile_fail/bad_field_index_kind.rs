use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
struct BadKind {
    #[obj(index = bogus)]
    field: u32,
}

fn main() {}
