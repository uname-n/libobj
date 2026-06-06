use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize)]
struct NoDefault;

impl obj::Schema for NoDefault {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::Null
    }
}

#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "auto_no_default", auto_migrate)]
struct Doc {
    id: u64,
    added: NoDefault,
}

fn main() {}
