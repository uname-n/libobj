use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = -1)]
struct BadNegVersion {
    x: u32,
}

fn main() {}
