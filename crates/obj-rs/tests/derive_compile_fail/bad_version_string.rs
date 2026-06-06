use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = "two")]
struct BadStringVersion {
    x: u32,
}

fn main() {}
