use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = 2)]
#[obj(version = 3)]
struct DuplicateVersion {
    x: u32,
}

fn main() {}
