# libobj

> An embedded document database. Dependable. Portable. Zero-infrastructure.

`libobj` is a self-contained, serverless, single-file document database with a
stable on-disk format and full ACID semantics. It is to document storage what
SQLite is to relational storage: no server, no setup — just a file.

The engine is written in Rust and shipped as a Rust crate and a C ABI.

## Quickstart (Rust)

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

    // `T` lives only behind the closure's `&mut T`, so annotate the
    // parameter — `|o: &mut Order|` — and inference fills in the rest
    // (no `db.update::<Order, _>(…)` turbofish needed).
    db.update(id, |o: &mut Order| o.total_cents = 5_000)?;

    let back: Order = db
        .get::<Order>(id)?
        .ok_or(obj::Error::InvalidArgument("just inserted"))?;
    assert_eq!(back.total_cents, 5_000);
    Ok(())
}
```

Collection name and schema version default to the type name and `1`; override
with `#[obj(collection = "...", version = N)]`. See the crate docs for queries,
indexes, transactions, and migrations.

## Schema migrations

Evolve a stored type by bumping its version; older records migrate to the
current shape lazily, on read. For the common case — **adding fields** — you
don't write the migration at all: add `#[obj(auto_migrate)]` and the derive
generates it. Pre-existing fields carry over and new fields backfill with
their `Default`, or with a per-field `#[obj(default = ...)]` override.

```rust
use serde::{Deserialize, Serialize};

// v2 of a type first stored at v1: `tier` is new. Existing records carry
// `name`/`email` over and backfill `tier` from `#[obj(default = ...)]`.
#[derive(Debug, Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "customers", auto_migrate)]
struct Customer {
    name: String,
    email: String,
    #[obj(default = "standard".to_owned())]
    tier: String,
}
```

When the backfill must read the old record — deriving the new field from an
existing one — point the field at a function with `#[obj(default_with = ...)]`
instead of a static expression. It fires on the same absent-field branch, is
handed the old record plus the stored version, and may fail:

```rust
use obj::Dynamic;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, obj::Document)]
#[obj(version = 2, collection = "customers", auto_migrate)]
struct Customer {
    name: String,
    email: String,
    #[obj(default_with = tier_from_email)] // derived, not a constant
    tier: String,
}

// fn(old: &Dynamic, from_version: u32) -> obj::Result<FieldTy>
fn tier_from_email(old: &Dynamic, _from: u32) -> obj::Result<String> {
    let email = old.get_str("email")?;
    Ok(if email.ends_with("@bigcorp.com") {
        "enterprise".to_owned()
    } else {
        "standard".to_owned()
    })
}
```

Because migration runs lazily on every read of an unmigrated record (the
result is not written back), keep a `default_with` function pure and cheap —
it's for values computable from the record, not for external I/O such as a
network or database lookup.

`auto_migrate` covers pure-additive changes only. Field removals, renames,
type changes, and version-dependent backfills need a hand-written
`Document::migrate` — including the recommended `From`-per-version pattern for
chained `v1 → v2 → v3` upgrades. See the `obj-rs` crate docs ("Schema
evolution") for both paths.

## Performance

The defaults favour durability and a small memory footprint over raw
throughput. Three levers, in rough order of impact:

- **Batch writes into one transaction.** Every committed write
  transaction costs one WAL durability sync under the default
  `SyncMode::Full`, so a single-document insert is dominated by that one
  sync. Inserting many documents inside one `db.transaction(|tx| …)`
  closure pays the sync once instead of once per row — dramatically
  faster per document. Split very large batches into chunks of a few
  thousand to stay under the WAL size limit.
- **Size the cache for the working set.** The LRU page cache defaults to
  64 frames (256 KiB). Read-heavy services over a large database should
  raise `Config::cache_size` (tens of MiB is a reasonable start) so the
  hot pages stay resident.
- **Relax the sync mode only if durability allows.** `SyncMode::Full`
  (default) survives system-wide power loss; `SyncMode::Normal` is
  crash/panic-durable with faster commits but can lose the last few
  transactions on sudden power loss; `SyncMode::Off` makes no durability
  call and is for tests/benchmarks/scratch data only.

```rust
use obj::{Config, Db, SyncMode};

let cfg = Config::default()
    .cache_size(64 * 1024 * 1024) // 64 MiB hot set
    .sync_mode(SyncMode::Normal);  // crash-durable, faster commits
let db = Db::open_with("app.obj", cfg)?;
```

See the `obj-rs` crate docs for the full set of `Config` knobs and their
durability tradeoffs.

## Workspace

| Crate                          | What it is                                                              |
|--------------------------------|-------------------------------------------------------------------------|
| [`obj-rs`](crates/obj-rs)      | The public Rust crate. Imported as `obj` (`use obj::Db`).               |
| [`obj-core`](crates/obj-core)  | Storage-engine internals: pager, WAL, B+tree, codec, catalog.           |
| [`obj-derive`](crates/obj-derive) | The `#[derive(obj::Document)]` proc-macro.                           |
| [`libobj`](crates/libobj)      | The C ABI (`cdylib` + `staticlib` + generated `include/libobj.h`).      |

`libobj` ships as a compiled artifact — the `cdylib`/`staticlib` plus
generated header.

## C ABI

C consumers link against `libobj` (the `cdylib`/`staticlib`) and include the
generated [`crates/libobj/include/libobj.h`](crates/libobj/include/libobj.h).
The header is regenerated from the Rust signatures by `cargo build -p libobj`
and validated against the committed copy by a drift test.

## Stability

The on-disk format is versioned and stable: new databases are written at
`format_major = 1`, and readers still open older pre-1.0 (`format_major = 0`)
files without a migration tool.

The public Rust API is still pre-1.0 (`0.5.0`) and is **not** yet frozen under
SemVer — it may change in a future `0.x` release before the 1.0 freeze. Pin a
specific tag (e.g. `tag = "v0.5.0"`) to insulate yourself from breaking changes.

## Coverage

Install [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) once:

```sh
cargo install cargo-llvm-cov --locked
```

Then measure coverage (not part of the mandatory safety checks):

```sh
cargo llvm-cov --workspace --all-features --summary-only --fail-under-lines 90
```

See [CLAUDE.md](CLAUDE.md) for the ratchet plan.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your
option.
