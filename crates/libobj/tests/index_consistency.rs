//! C-ABI write path vs secondary-index consistency.
//!
//! There are TWO C write families:
//!
//! - the PRIMARY-ONLY family (`obj_doc_insert_raw` etc.) writes the
//!   primary record only — it does not maintain secondary indexes, so
//!   a doc written this way is invisible to `obj_find_unique` /
//!   `obj_iter_index_range`. [`c_plain_writes_stay_primary_only`]
//!   pins that.
//! - the INDEX-MAINTAINING family (`obj_doc_insert_indexed` /
//!   `obj_doc_update_indexed` / `obj_doc_delete_indexed`)
//!   maintains the named secondary indexes from caller-supplied
//!   field-encoded keys (built via `obj_index_key_encode`). A doc
//!   written this way IS discoverable through its `Unique` /
//!   `Standard` indexes. [`c_indexed_writes_maintain_secondary_indexes`]
//!   pins that, including the `Unique` collision, key-move on update,
//!   and index-entry removal on delete.
//!
//! The C ABI cannot DECLARE indexes (only the typed Rust
//! `Document::indexes()` path does), so each test first declares the
//! `customers` collection's indexes via a typed transaction, then
//! drives the C ABI.

// allow: this test crate exercises the unsafe C ABI directly, so every FFI call site is `unsafe`.
#![allow(unsafe_code)]

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use obj_engine::{Db, Document, IndexSpec};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use obj::{
    obj_close, obj_db_t, obj_doc_delete_indexed, obj_doc_get, obj_doc_insert_raw,
    obj_doc_insert_indexed, obj_doc_update_indexed, obj_find_unique, obj_free_buffer,
    obj_index_entry_t, obj_index_key_encode, obj_iter_free, obj_iter_index_range, obj_iter_next,
    obj_iter_t, obj_open, obj_read_txn_t, obj_txn_begin_read, obj_txn_begin_write, obj_txn_commit,
    obj_txn_end_read, obj_txn_rollback, obj_write_txn_t, ObjBound, OBJ_ERR_INVALID_ARG,
    OBJ_ERR_NOT_FOUND, OBJ_INDEX_VALUE_STRING, OBJ_INDEX_VALUE_U64, OBJ_OK,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Customer {
    email: String,
    age: u64,
}

impl obj_engine::Schema for Customer {
    fn schema() -> obj_engine::DynamicSchema {
        obj_engine::DynamicSchema::map([
            ("email", obj_engine::DynamicSchema::String),
            ("age", obj_engine::DynamicSchema::U64),
        ])
    }
}

impl Document for Customer {
    const COLLECTION: &'static str = "customers";
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        let mut specs = Vec::new();
        if let Ok(spec) = IndexSpec::unique("by_email", "email") {
            specs.push(spec);
        }
        if let Ok(spec) = IndexSpec::standard("by_age", "age") {
            specs.push(spec);
        }
        specs
    }
}

fn path_cstring(p: &Path) -> CString {
    CString::new(p.to_string_lossy().into_owned()).expect("non-NUL path")
}

fn open_db_c(path: &Path) -> *mut obj_db_t {
    let cs = path_cstring(path);
    let mut db: *mut obj_db_t = ptr::null_mut();
    let code = unsafe { obj_open(cs.as_ptr(), &raw mut db) };
    assert_eq!(code, OBJ_OK);
    db
}

/// Declare the `customers` collection + its `by_email` Unique and
/// `by_age` Standard indexes via the typed Rust API (the only path
/// that declares indexes). Leaves the collection EMPTY: opening a
/// write-side `Collection<Customer>` runs the index reconciler, which
/// stamps the `Active` descriptors; we commit without inserting.
fn declare_indexes(path: &Path) {
    let db = Db::open(path).expect("open");
    db.transaction(|tx| {
        let _coll = tx.collection::<Customer>()?;
        Ok(())
    })
    .expect("declare indexes");
}

/// Encode `value` of `kind` as an obj order-preserving field key via
/// the C `obj_index_key_encode`, using the documented size-query then
/// fill two-call idiom.
fn encode_field_key(kind: i32, value: &[u8]) -> Vec<u8> {
    let mut needed: usize = 0;
    let code = unsafe {
        obj_index_key_encode(
            kind,
            value.as_ptr(),
            value.len(),
            ptr::null_mut(),
            0,
            &raw mut needed,
        )
    };
    assert_eq!(code, OBJ_ERR_INVALID_ARG, "size query reports too-small");
    assert!(needed > 0);
    let mut buf = vec![0u8; needed];
    let mut written: usize = 0;
    let code = unsafe {
        obj_index_key_encode(
            kind,
            value.as_ptr(),
            value.len(),
            buf.as_mut_ptr(),
            buf.len(),
            &raw mut written,
        )
    };
    assert_eq!(code, OBJ_OK);
    assert_eq!(written, needed);
    buf
}

fn string_key(s: &str) -> Vec<u8> {
    encode_field_key(OBJ_INDEX_VALUE_STRING, s.as_bytes())
}

fn u64_key(n: u64) -> Vec<u8> {
    encode_field_key(OBJ_INDEX_VALUE_U64, &n.to_ne_bytes())
}

/// Plain `obj_doc_insert_raw` writes the primary record only — the doc is
/// fetchable by id but NOT discoverable through `by_email`. (The
/// primary-only family is the documented baseline.)
#[test]
fn c_plain_writes_stay_primary_only() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("idx.obj");
    declare_indexes(&path);

    let c_email = "grace@example.com";
    let c_payload: Vec<u8> = b"c-abi-raw-bytes-payload".to_vec();
    let c_id = {
        let db = open_db_c(&path);
        let txn = begin_write(db);
        let cs = CString::new("customers").expect("non-NUL");
        let mut id: u64 = 0;
        let code = unsafe {
            obj_doc_insert_raw(
                txn,
                cs.as_ptr(),
                c_payload.as_ptr(),
                c_payload.len(),
                &raw mut id,
            )
        };
        assert_eq!(code, OBJ_OK);
        assert_eq!(unsafe { obj_txn_commit(txn) }, OBJ_OK);
        unsafe { obj_close(db) };
        id
    };

    let db = open_db_c(&path);
    let rtxn = begin_read(db);
    let cs = CString::new("customers").expect("non-NUL");
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let code = unsafe { obj_doc_get(rtxn, cs.as_ptr(), c_id, &raw mut payload, &raw mut len) };
    assert_eq!(code, OBJ_OK, "C-written doc must be fetchable by id");
    let bytes = unsafe { std::slice::from_raw_parts(payload, len) }.to_vec();
    unsafe { obj_free_buffer(payload, len) };
    assert_eq!(bytes.as_slice(), c_payload.as_slice());

    let found = find_unique(rtxn, "customers", "by_email", &string_key(c_email));
    assert_eq!(
        found, None,
        "plain obj_doc_insert_raw must NOT appear in the typed secondary index \
         (primary-only family); use obj_doc_insert_indexed for index maintenance"
    );

    unsafe { obj_txn_end_read(rtxn) };
    unsafe { obj_close(db) };
}

/// The index-maintaining family makes a C-written doc discoverable
/// through both a `Unique` (`by_email`) and a `Standard` (`by_age`)
/// index, enforces the `Unique` constraint on a colliding insert,
/// moves the key on update, and removes the entries on delete.
#[test]
// allow: end-to-end C-API scenario test; splitting the round-trip would obscure what it documents.
#[allow(clippy::too_many_lines)]
fn c_indexed_writes_maintain_secondary_indexes() {
    let dir = TempDir::new().expect("tmp");
    let path = dir.path().join("idx.obj");
    declare_indexes(&path);

    let email = "ada@example.com";
    let age: u64 = 36;
    let payload: Vec<u8> = b"ada-payload".to_vec();

    let id = {
        let db = open_db_c(&path);
        let txn = begin_write(db);
        let id = insert_indexed(
            txn,
            "customers",
            &payload,
            &[("by_email", string_key(email)), ("by_age", u64_key(age))],
        )
        .expect("indexed insert");
        assert_eq!(unsafe { obj_txn_commit(txn) }, OBJ_OK);
        unsafe { obj_close(db) };
        id
    };

    {
        let db = open_db_c(&path);
        let rtxn = begin_read(db);
        assert_eq!(
            find_unique(rtxn, "customers", "by_email", &string_key(email)),
            Some(id),
            "indexed insert must be visible to obj_find_unique"
        );
        let range = index_range_ids(rtxn, "customers", "by_age", &u64_key(age));
        assert_eq!(
            range,
            vec![id],
            "indexed insert must be visible to obj_iter_index_range"
        );
        unsafe { obj_txn_end_read(rtxn) };
        unsafe { obj_close(db) };
    }

    {
        let db = open_db_c(&path);
        let txn = begin_write(db);
        let dup = b"dup-payload".to_vec();
        let code = insert_indexed_code(
            txn,
            "customers",
            &dup,
            &[("by_email", string_key(email)), ("by_age", u64_key(99))],
        );
        assert_eq!(
            code, OBJ_ERR_INVALID_ARG,
            "duplicate Unique key must trip the constraint"
        );
        unsafe { obj_txn_rollback(txn) };
        unsafe { obj_close(db) };
    }

    let new_email = "ada.lovelace@example.com";
    let new_age: u64 = 37;
    {
        let db = open_db_c(&path);
        let txn = begin_write(db);
        let new_payload = b"ada-v2".to_vec();
        let cs = CString::new("customers").expect("non-NUL");
        let remove = [("by_email", string_key(email)), ("by_age", u64_key(age))];
        let add = [
            ("by_email", string_key(new_email)),
            ("by_age", u64_key(new_age)),
        ];
        let code = update_indexed_code(txn, &cs, id, &new_payload, &remove, &add);
        assert_eq!(code, OBJ_OK, "indexed update");
        assert_eq!(unsafe { obj_txn_commit(txn) }, OBJ_OK);
        unsafe { obj_close(db) };
    }
    {
        let db = open_db_c(&path);
        let rtxn = begin_read(db);
        assert_eq!(
            find_unique(rtxn, "customers", "by_email", &string_key(email)),
            None,
            "old email key must be gone after update"
        );
        assert_eq!(
            find_unique(rtxn, "customers", "by_email", &string_key(new_email)),
            Some(id),
            "new email key must be found after update"
        );
        assert!(
            index_range_ids(rtxn, "customers", "by_age", &u64_key(age)).is_empty(),
            "old age key must be gone after update"
        );
        assert_eq!(
            index_range_ids(rtxn, "customers", "by_age", &u64_key(new_age)),
            vec![id],
            "new age key must be found after update"
        );
        unsafe { obj_txn_end_read(rtxn) };
        unsafe { obj_close(db) };
    }

    {
        let db = open_db_c(&path);
        let txn = begin_write(db);
        let cs = CString::new("customers").expect("non-NUL");
        let remove = [
            ("by_email", string_key(new_email)),
            ("by_age", u64_key(new_age)),
        ];
        let entries = build_entries(&remove);
        let (ptrs, _keep) = entry_array(&entries);
        let code =
            unsafe { obj_doc_delete_indexed(txn, cs.as_ptr(), id, ptrs.as_ptr(), ptrs.len()) };
        assert_eq!(code, OBJ_OK, "indexed delete");
        assert_eq!(unsafe { obj_txn_commit(txn) }, OBJ_OK);
        unsafe { obj_close(db) };
    }
    {
        let db = open_db_c(&path);
        let rtxn = begin_read(db);
        assert_eq!(
            find_unique(rtxn, "customers", "by_email", &string_key(new_email)),
            None,
            "deleted doc must be gone from the Unique index"
        );
        assert!(
            index_range_ids(rtxn, "customers", "by_age", &u64_key(new_age)).is_empty(),
            "deleted doc must be gone from the Standard index"
        );
        unsafe { obj_txn_end_read(rtxn) };
        unsafe { obj_close(db) };
    }
}

fn begin_read(db: *mut obj_db_t) -> *mut obj_read_txn_t {
    let mut txn: *mut obj_read_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_read(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

fn begin_write(db: *mut obj_db_t) -> *mut obj_write_txn_t {
    let mut txn: *mut obj_write_txn_t = ptr::null_mut();
    let code = unsafe { obj_txn_begin_write(db, &raw mut txn) };
    assert_eq!(code, OBJ_OK);
    txn
}

/// Owned backing storage for one C entry: the NUL-terminated index
/// name + the field-key bytes. Kept alive while the
/// `obj_index_entry_t` array borrows their pointers.
struct EntryStore {
    name: CString,
    key: Vec<u8>,
}

fn build_entries(entries: &[(&str, Vec<u8>)]) -> Vec<EntryStore> {
    entries
        .iter()
        .map(|(name, key)| EntryStore {
            name: CString::new(*name).expect("non-NUL index name"),
            key: key.clone(),
        })
        .collect()
}

/// Build the `obj_index_entry_t` array pointing into `stores`. The
/// returned array borrows `stores`, which the caller must keep alive
/// for the duration of the FFI call.
fn entry_array(stores: &[EntryStore]) -> (Vec<obj_index_entry_t>, ()) {
    let arr = stores
        .iter()
        .map(|s| obj_index_entry_t {
            index_name: s.name.as_ptr(),
            key: if s.key.is_empty() {
                ptr::null()
            } else {
                s.key.as_ptr()
            },
            key_len: s.key.len(),
        })
        .collect();
    (arr, ())
}

fn insert_indexed_code(
    txn: *mut obj_write_txn_t,
    collection: &str,
    payload: &[u8],
    entries: &[(&str, Vec<u8>)],
) -> i32 {
    let cs = CString::new(collection).expect("non-NUL");
    let stores = build_entries(entries);
    let (arr, _keep) = entry_array(&stores);
    let mut id: u64 = 0;
    unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            arr.as_ptr(),
            arr.len(),
            &raw mut id,
        )
    }
}

fn insert_indexed(
    txn: *mut obj_write_txn_t,
    collection: &str,
    payload: &[u8],
    entries: &[(&str, Vec<u8>)],
) -> Option<u64> {
    let cs = CString::new(collection).expect("non-NUL");
    let stores = build_entries(entries);
    let (arr, _keep) = entry_array(&stores);
    let mut id: u64 = 0;
    let code = unsafe {
        obj_doc_insert_indexed(
            txn,
            cs.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            arr.as_ptr(),
            arr.len(),
            &raw mut id,
        )
    };
    (code == OBJ_OK).then_some(id)
}

fn update_indexed_code(
    txn: *mut obj_write_txn_t,
    collection: &CString,
    id: u64,
    payload: &[u8],
    remove: &[(&str, Vec<u8>)],
    add: &[(&str, Vec<u8>)],
) -> i32 {
    let remove_stores = build_entries(remove);
    let add_stores = build_entries(add);
    let (remove_arr, _r) = entry_array(&remove_stores);
    let (add_arr, _a) = entry_array(&add_stores);
    unsafe {
        obj_doc_update_indexed(
            txn,
            collection.as_ptr(),
            id,
            payload.as_ptr(),
            payload.len(),
            remove_arr.as_ptr(),
            remove_arr.len(),
            add_arr.as_ptr(),
            add_arr.len(),
        )
    }
}

/// Run `obj_find_unique`; `Some(id)` on a hit, `None` on
/// `OBJ_ERR_NOT_FOUND`. Frees any returned payload.
fn find_unique(
    rtxn: *mut obj_read_txn_t,
    collection: &str,
    index: &str,
    key: &[u8],
) -> Option<u64> {
    let cs_collection = CString::new(collection).expect("non-NUL");
    let cs_index = CString::new(index).expect("non-NUL");
    let mut id: u64 = 0;
    let mut payload: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let code = unsafe {
        obj_find_unique(
            rtxn,
            cs_collection.as_ptr(),
            cs_index.as_ptr(),
            key.as_ptr(),
            key.len(),
            &raw mut id,
            &raw mut payload,
            &raw mut len,
        )
    };
    match code {
        OBJ_OK => {
            if !payload.is_null() {
                unsafe { obj_free_buffer(payload, len) };
            }
            Some(id)
        }
        OBJ_ERR_NOT_FOUND => None,
        other => panic!("obj_find_unique returned unexpected code {other}"),
    }
}

/// Collect the ids returned by an `obj_iter_index_range` over the
/// single-key `[key, key]` inclusive range on `index`.
fn index_range_ids(
    rtxn: *mut obj_read_txn_t,
    collection: &str,
    index: &str,
    key: &[u8],
) -> Vec<u64> {
    let cs_collection = CString::new(collection).expect("non-NUL");
    let cs_index = CString::new(index).expect("non-NUL");
    let mut iter: *mut obj_iter_t = ptr::null_mut();
    let code = unsafe {
        obj_iter_index_range(
            rtxn,
            cs_collection.as_ptr(),
            cs_index.as_ptr(),
            ObjBound { ptr: key.as_ptr(), len: key.len(), inclusive: true },
            ObjBound { ptr: key.as_ptr(), len: key.len(), inclusive: true },
            &raw mut iter,
        )
    };
    assert_eq!(code, OBJ_OK, "obj_iter_index_range open");
    let mut ids = Vec::new();
    loop {
        let mut id: u64 = 0;
        let mut payload: *mut u8 = ptr::null_mut();
        let mut len: usize = 0;
        let code = unsafe { obj_iter_next(iter, &raw mut id, &raw mut payload, &raw mut len) };
        match code {
            OBJ_OK => {
                if !payload.is_null() {
                    unsafe { obj_free_buffer(payload, len) };
                }
                ids.push(id);
            }
            OBJ_ERR_NOT_FOUND => break,
            other => panic!("obj_iter_next returned {other}"),
        }
    }
    unsafe { obj_iter_free(iter) };
    ids
}
