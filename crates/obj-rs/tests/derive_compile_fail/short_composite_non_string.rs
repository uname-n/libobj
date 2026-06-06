use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(index = (1, 2))]
struct ShortCompositeNonString {
    a: u32,
    b: u32,
}

fn main() {}
