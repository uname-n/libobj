use serde::{Deserialize, Serialize};

fn backfill(_old: &obj::Dynamic, _from: u32) -> obj::Result<String> {
    Ok("standard".to_owned())
}

// `default` and `default_with` on the SAME field is a compile error:
// they are two ways to supply the one absent-branch backfill.
#[derive(Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "default_conflict", auto_migrate)]
struct Doc {
    id: u64,
    #[obj(default = "standard".to_owned(), default_with = backfill)]
    tier: String,
}

fn main() {}
