use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, obj::Document)]
struct Generic<T> {
    value: T,
}

fn main() {}
