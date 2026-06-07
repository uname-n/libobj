# obj-core

> The storage engine internals behind `obj`.

Part of [`obj`](https://github.com/uname-n/libobj) — the embedded document
database. `obj-core` is the engine: pager, write-ahead log, B-tree,
codec, catalog, crypto, and integrity checks.

> **Internal crate — not for direct use.** `obj-core` is an UNSTABLE
> implementation detail with **no SemVer guarantee**; its API may change
> in any release. Depend on [`obj-rs`](../obj-rs), the stable public
> crate, instead.

---

## What's inside

| Module      | Responsibility                                                        |
|-------------|-----------------------------------------------------------------------|
| `pager`     | Fixed-size page cache over the file; optional LZ4 / XChaCha20-Poly1305.|
| `wal`       | Write-ahead log: frames, commit records, checkpoint folding.          |
| `btree`     | Byte-aware B+tree for primary and secondary keys.                     |
| `codec`     | Positional, schema-driven postcard encode / decode.                   |
| `catalog`   | Collection + index registry and the on-disk root map.                 |
| `index`     | Secondary index maintenance (standard, unique, multi-value, composite).|
| `txn`       | Reader-snapshot / writer-lock transaction state.                      |
| `crypto`    | At-rest page encryption primitives.                                   |
| `integrity` | The bidirectional consistency check behind `Db::integrity_check`.     |

The on-disk format these modules implement is specified in
[`docs/format.md`](https://github.com/uname-n/libobj/blob/main/docs/format.md)
and frozen at `format_major = 1`.

---

## License

Dual-licensed under [MIT](https://github.com/uname-n/libobj/blob/main/LICENSE-MIT)
or [Apache 2.0](https://github.com/uname-n/libobj/blob/main/LICENSE-APACHE),
at your option.
