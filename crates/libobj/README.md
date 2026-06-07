# libobj

> The C ABI for the `obj` embedded document database.

Part of [`obj`](https://github.com/uname-n/libobj) — the embedded document
database. `libobj` exposes the engine through a stable C ABI: a
`cdylib` + `staticlib` plus a cbindgen-generated `include/libobj.h`. The
C ABI is stable within a major version — an application linking `libobj`
need not be recompiled across `obj` patch / minor releases.

> **Artifact crate.** Consume `libobj` as a compiled library: build the
> `cdylib`/`staticlib` plus `include/libobj.h` from this repository and
> link your application against it.

---

## Build

```bash
cargo build --release -p libobj
# produces target/release/libobj.{dylib,so},
#          target/release/libobj.a,
#          and crates/libobj/include/libobj.h
```

Link against the library and `#include "libobj.h"`. See
[`examples/smoke.c`](./examples/smoke.c) for a complete, runnable C99
program and [`examples/build-smoke.sh`](./examples/build-smoke.sh) for a
compile-and-link invocation.

---

## Usage

```c
#include "libobj.h"

obj_db_t *db = NULL;
obj_open("app.obj", &db);

obj_write_txn_t *wtxn = NULL;
obj_txn_begin_write(db, &wtxn);
uint64_t id;
obj_doc_insert(wtxn, "orders", payload, payload_len, &id);
obj_txn_commit(wtxn);                 // one fsync for the batch

obj_read_txn_t *rtxn = NULL;
obj_txn_begin_read(db, &rtxn);
uint8_t *out = NULL; size_t out_len = 0;
obj_doc_get(rtxn, "orders", id, &out, &out_len);
obj_free_buffer(out, out_len);        // caller frees engine buffers
obj_txn_end_read(rtxn);

obj_close(db);
```

Payloads cross the boundary as raw bytes — `libobj` does not serialise
for you. Encode however you like and pass the bytes through.

---

## API surface

| Area          | Functions                                                            |
|---------------|----------------------------------------------------------------------|
| Lifecycle     | `obj_open`, `obj_open_with_config`, `obj_close`                       |
| Transactions  | `obj_txn_begin_write` / `_commit` / `_rollback`, `obj_txn_begin_read` / `_end_read` |
| Documents     | `obj_doc_insert` / `_get` / `_update` / `_delete` / `_upsert` (+ `_indexed` variants) |
| Iteration     | `obj_iter_all`, `obj_iter_index_range`, `obj_iter_next` / `_free`     |
| Lookups       | `obj_find_unique`, `obj_count_all`, `obj_count_index_range`           |
| Diagnostics   | `obj_stat`, `obj_integrity_check` (+ `obj_integrity_report_*`)        |
| Operations    | `obj_backup_to`                                                      |
| Errors        | `obj_strerror`, the `obj_error_t` enum (`OBJ_OK`, `OBJ_ERR_*`)        |
| Memory        | `obj_free_buffer` — frees every buffer the engine hands back         |

Every function returns an `obj_error_t`; check it against `OBJ_OK`.
Buffers returned through out-pointers are owned by the caller and must
be released with `obj_free_buffer`. The generated `libobj.h` is the
authoritative signature reference.

---

## License

Dual-licensed under [MIT](https://github.com/uname-n/libobj/blob/main/LICENSE-MIT)
or [Apache 2.0](https://github.com/uname-n/libobj/blob/main/LICENSE-APACHE),
at your option.
