# obj-rs

> The embedded document database for Rust. Dependable. Portable. Zero-infrastructure.

Part of [`obj`](https://github.com/uname-n/libobj) — a self-contained,
serverless, single-file document database with a stable file format and
full ACID semantics. It is to document storage what SQLite is to
relational storage.

Crate name `obj-rs`; import as `obj`.

**Stability.** The on-disk format is versioned and stable: new databases are
written at `format_major = 1`, and readers still open older pre-1.0
(`format_major = 0`) files without a migration tool. The public Rust API is
still pre-1.0 (`0.5.0`) and is **not** yet frozen under SemVer — it may change
in a future `0.x` release before the 1.0 freeze; pin a specific tag (e.g.
`tag = "v0.5.0"`) to insulate yourself from breaking changes.

---

## Quickstart

```toml
# Cargo.toml
[dependencies]
obj-rs = { git = "https://github.com/uname-n/libobj", tag = "v0.5.0" }
serde = { version = "1", features = ["derive"] }
```

```rust
use obj::Db;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, obj::Document)]
struct Order {
    customer_id: u64,
    total_cents: u64,
}

fn main() -> obj::Result<()> {
    let dir = tempfile::tempdir()?;
    let db = Db::open(dir.path().join("app.obj"))?;
    let id = db.insert(Order { customer_id: 1, total_cents: 4_200 })?;
    let back: Order = db
        .get::<Order>(id)?
        .ok_or(obj::Error::InvalidArgument("just inserted"))?;
    assert_eq!(back.total_cents, 4_200);
    Ok(())
}
```

Collection name and schema version default to the type name and `1`;
override with `#[obj(collection = "...", version = N)]`. See the docs
for queries, indexes, transactions, and migrations.

---

## Features

All features are off by default — enable them on the git dependency in
`Cargo.toml`, e.g.
`obj-rs = { git = "https://github.com/uname-n/libobj", tag = "v0.5.0", features = ["serde"] }`.

| Feature       | What it does                                                                 |
|---------------|------------------------------------------------------------------------------|
| `serde`       | Derive `Serialize`/`Deserialize` on public types; re-export both traits.     |
| `tracing`     | Structured spans around open, transactions, queries, and checkpoints.        |
| `compression` | LZ4 per-page compression at the pager layer.                                 |
| `encryption`  | XChaCha20-Poly1305 per-page at-rest encryption.                              |

Build the API docs locally for per-feature details (see
[Documentation](#documentation)).

---

## Performance

The defaults favour durability and a small memory footprint over raw
throughput. Three levers, in rough order of impact:

- **Batch writes into one transaction.** Every committed write
  transaction costs one WAL durability sync under the default
  `SyncMode::Full`, so a single-document insert is dominated by that
  one sync. Inserting N documents in N transactions pays the sync N
  times; inserting them inside one `db.transaction(|tx| …)` closure
  pays it once. Prefer one transaction per unit of work, not one per
  row (splitting very large batches into chunks of a few thousand).
- **Size the cache for the working set.** The LRU page cache defaults
  to 64 frames (256 KiB) — fine for write-mostly or memory-constrained
  use, but read-heavy services over a large database should raise
  `Config::cache_size` so the hot pages stay resident (tens of MiB is a
  reasonable start).
- **Relax the sync mode only if durability allows.** `SyncMode::Full`
  (default) survives system-wide power loss. `SyncMode::Normal` is
  crash/panic-durable with faster commits but can lose the last few
  transactions on a sudden power loss. `SyncMode::Off` makes no
  durability call and is for tests/benchmarks/scratch data only.

```rust
use obj::{Config, Db, SyncMode};

let cfg = Config::default()
    .cache_size(64 * 1024 * 1024) // 64 MiB hot set
    .sync_mode(SyncMode::Normal);  // crash-durable, faster commits
let db = Db::open_with("app.obj", cfg)?;
```

See the `Config` docs (built locally, see [Documentation](#documentation))
for the full set of knobs and their durability tradeoffs.

---

## Documentation

- API docs are built locally with `cargo doc`. Run
  `cargo doc -p obj-rs --all-features --open` to build the full-feature
  docs (with worked examples on every public type); the output lands at
  `target/doc/obj/index.html`.
- [Project README](https://github.com/uname-n/libobj) — overview, other
  bindings, and the full documentation index.
- [Format spec](https://github.com/uname-n/libobj/blob/main/docs/format.md)

---

## License

Dual-licensed under [MIT](https://github.com/uname-n/libobj/blob/main/LICENSE-MIT)
or [Apache 2.0](https://github.com/uname-n/libobj/blob/main/LICENSE-APACHE),
at your option.
