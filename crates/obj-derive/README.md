# obj-derive

> The `#[derive(Document)]` procedural macro for `obj`.

Part of [`obj`](https://github.com/uname-n/libobj) — the embedded document
database. `obj-derive` provides the `#[derive(Document)]` macro that
generates a type's collection binding, schema, and secondary-index
descriptors.

> **Internal crate — not for direct use.** `obj-derive` is an UNSTABLE
> implementation detail with **no SemVer guarantee**. Depend on
> [`obj-rs`](../obj-rs) instead and use the re-exported derive
> (`obj::Document`).

---

## What it generates

`#[derive(Document)]` reads `#[obj(...)]` attributes and emits the
`Document` impl — the collection name, schema version, and index set:

```rust
#[derive(serde::Serialize, serde::Deserialize, obj::Document)]
#[obj(collection = "orders", version = 1)]
#[obj(index = ("region", "status"), name = "by_region_status")]
struct Order {
    #[obj(index = unique)]
    email: String,
    #[obj(index)]
    region: String,
    #[obj(index = each)]   // multi-value index; field must be a Vec<...>
    tags: Vec<String>,
    status: String,
}
```

| Attribute                                          | Level  | Effect                                              |
|----------------------------------------------------|--------|-----------------------------------------------------|
| `collection = "..."`                               | struct | Override the collection name (defaults to the type).|
| `version = N`                                      | struct | Set the schema version (defaults to `1`).           |
| `index = (...), name = "..."`                      | struct | Composite index over ≥ 2 fields (`name` optional). **Canonical form.** |
| `index_composite(fields = (...), name = "...")`    | struct | Same composite index, older long form — also accepted. |
| `index`                                            | field  | Standard secondary index on the field.              |
| `index = unique`                                   | field  | Unique secondary index on the field.                |
| `index = each`                                     | field  | Multi-value index over each element of a `Vec` field.|
| `name = "..."`                                     | field  | Override an index's default name (the field name).  |

A `history(v1 = OldType, ...)` struct attribute registers historical
schemas for lazy per-document migration.

The encoding is positional and schema-driven — no field names in the
bytes — so the generated codec round-trips byte-identically with the
Python `@obj.document` writer for the same logical schema.

---

## License

Dual-licensed under [MIT](https://github.com/uname-n/libobj/blob/main/LICENSE-MIT)
or [Apache 2.0](https://github.com/uname-n/libobj/blob/main/LICENSE-APACHE),
at your option.
