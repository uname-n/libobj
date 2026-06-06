//! End-to-end document round-trip.
//!
//! Wires the primitives directly:
//!
//! - [`obj_core::codec::encode`] / [`obj_core::codec::decode`]
//! - [`obj_core::codec::Dynamic`]
//! - [`obj_core::codec::Migrate`] (via the inherent
//!   [`Document::migrate`] override on each test type)
//! - [`obj_core::Id`] allocation
//! - [`obj_core::Catalog`] + [`obj_core::CollectionDescriptor`]
//!
//! Each test path:
//!   1. Open a fresh file-backed pager + catalog.
//!   2. Register a collection: allocate an empty primary B-tree,
//!      store its root + `type_version` in the catalog descriptor.
//!   3. For each document: allocate an `Id` via the catalog, encode
//!      via [`codec::encode`], insert into the primary tree keyed by
//!      `Id::to_be_bytes()`. The catalog row's `primary_root` shifts
//!      after every B-tree mutation (COW), so we re-fetch and
//!      re-store the descriptor.
//!   4. Commit, close, reopen.
//!   5. Re-attach to the catalog and the primary B-tree; verify
//!      every stored document decodes back to byte-equal Rust state.
//!
//! This test is NOT `#[ignore]`d — it runs in the default
//! `cargo test --workspace`.

#![forbid(unsafe_code)]

use std::path::Path;

use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use obj_core::btree::BTree;
use obj_core::catalog::{Catalog, CollectionDescriptor};
use obj_core::codec::{self, DocumentHeader, Dynamic, DOC_HEADER_SIZE};
use obj_core::pager::page::PageId;
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Document, Error, Id, Result};

/// A small struct with a nested struct. Demonstrates that the codec
/// + postcard correctly round-trip nested Rust types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct UserProfile {
    name: String,
    age: u32,
    address: Address,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Address {
    street: String,
    city: String,
}

impl Document for UserProfile {
    const COLLECTION: &'static str = "users";
    const VERSION: u32 = 1;
}

/// A doc with a `Vec<Id>` — exercises the `Id` serde integration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Thread {
    title: String,
    replies: Vec<Id>,
}

impl Document for Thread {
    const COLLECTION: &'static str = "threads";
    const VERSION: u32 = 1;
}

/// Open a fresh pager + catalog at `path`.
///
/// Catalog init mutates the file header via the WAL
/// (`stage_or_write_header`), which requires an open Pager txn.
/// Wrap the init call so a fresh-file open is durable on return.
fn open_or_create(path: &Path) -> Result<(Pager<FileHandle>, Catalog<FileHandle>)> {
    let mut pager = Pager::open(path, Config::default())?;
    pager.begin_txn();
    let init = Catalog::open_or_init(&mut pager);
    let catalog = match init {
        Ok(c) => {
            let r = pager.commit();
            pager.end_txn();
            r?;
            c
        }
        Err(e) => {
            pager.end_txn();
            return Err(e);
        }
    };
    Ok((pager, catalog))
}

/// Register a new collection with an empty primary B-tree.
fn register_collection(
    pager: &mut Pager<FileHandle>,
    catalog: &mut Catalog<FileHandle>,
    name: &str,
    type_version: u32,
) -> Result<u32> {
    let primary_root = BTree::<FileHandle>::empty(pager)?.root();
    let descriptor = CollectionDescriptor::new(0, primary_root.get(), type_version);
    catalog.insert(pager, name, descriptor)
}

/// Insert a document into a collection. Encodes via the codec, looks
/// up the current primary-root from the catalog, inserts into the
/// B-tree, and updates the catalog row with the (possibly shifted)
/// new primary-root.
fn insert_document<T: Document>(
    pager: &mut Pager<FileHandle>,
    catalog: &mut Catalog<FileHandle>,
    collection_name: &str,
    id: Id,
    doc: &T,
) -> Result<()> {
    let descriptor = catalog
        .get(pager, collection_name)?
        .ok_or(Error::InvalidArgument("collection not registered"))?;
    let bytes = codec::encode(doc, descriptor.collection_id)?;
    let primary_root_id =
        PageId::new(descriptor.primary_root).ok_or(Error::Corruption { page_id: 0 })?;
    let mut tree = BTree::<FileHandle>::open(pager, primary_root_id)?;
    tree.insert(pager, &id.to_be_bytes(), &bytes)?;
    let mut updated = descriptor;
    updated.primary_root = tree.root().get();
    catalog.update(pager, collection_name, &updated)?;
    Ok(())
}

/// Fetch + decode a document by id.
fn get_document<T: Document>(
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    collection_name: &str,
    id: Id,
) -> Result<Option<T>> {
    let Some(descriptor) = catalog.get(pager, collection_name)? else {
        return Ok(None);
    };
    let primary_root_id =
        PageId::new(descriptor.primary_root).ok_or(Error::Corruption { page_id: 0 })?;
    let tree = BTree::<FileHandle>::open(pager, primary_root_id)?;
    match tree.get(pager, &id.to_be_bytes())? {
        Some(bytes) => {
            let doc = codec::decode::<T>(&bytes, descriptor.collection_id)?;
            Ok(Some(doc))
        }
        None => Ok(None),
    }
}

/// Build the three test fixtures for the main round-trip test.
fn round_trip_fixtures() -> (UserProfile, UserProfile, Thread) {
    let alice = UserProfile {
        name: "Alice".to_owned(),
        age: 30,
        address: Address {
            street: "1 Main".to_owned(),
            city: "Anywhere".to_owned(),
        },
    };
    let bob = UserProfile {
        name: "Bob".to_owned(),
        age: 41,
        address: Address {
            street: "2 Side".to_owned(),
            city: "Elsewhere".to_owned(),
        },
    };
    let thread1 = Thread {
        title: "intro".to_owned(),
        replies: vec![Id::try_new(1).expect("nz"), Id::try_new(2).expect("nz")],
    };
    (alice, bob, thread1)
}

/// Write the three fixtures to a fresh database and return the
/// allocated ids.
fn write_round_trip(
    path: &Path,
    alice: &UserProfile,
    bob: &UserProfile,
    thread1: &Thread,
) -> (Id, Id, Id) {
    let (mut pager, mut catalog) = open_or_create(path).expect("open");
    pager.begin_txn();
    register_collection(
        &mut pager,
        &mut catalog,
        UserProfile::COLLECTION,
        UserProfile::VERSION,
    )
    .expect("register users");
    register_collection(
        &mut pager,
        &mut catalog,
        Thread::COLLECTION,
        Thread::VERSION,
    )
    .expect("register threads");
    let alice_id = catalog
        .next_id(&mut pager, UserProfile::COLLECTION)
        .expect("alice id");
    let bob_id = catalog
        .next_id(&mut pager, UserProfile::COLLECTION)
        .expect("bob id");
    let thread_id = catalog
        .next_id(&mut pager, Thread::COLLECTION)
        .expect("thread id");
    insert_document(
        &mut pager,
        &mut catalog,
        UserProfile::COLLECTION,
        alice_id,
        alice,
    )
    .expect("insert alice");
    insert_document(
        &mut pager,
        &mut catalog,
        UserProfile::COLLECTION,
        bob_id,
        bob,
    )
    .expect("insert bob");
    insert_document(
        &mut pager,
        &mut catalog,
        Thread::COLLECTION,
        thread_id,
        thread1,
    )
    .expect("insert thread");
    pager.commit().expect("commit");
    pager.end_txn();
    pager.close().expect("close");
    (alice_id, bob_id, thread_id)
}

#[test]
fn round_trip_two_collections_persists_across_reopen() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("rt.obj");
    let (alice, bob, thread1) = round_trip_fixtures();
    let (alice_id, bob_id, thread_id) = write_round_trip(&path, &alice, &bob, &thread1);

    let (mut pager, catalog) = open_or_create(&path).expect("reopen");
    let alice_back: UserProfile =
        get_document(&mut pager, &catalog, UserProfile::COLLECTION, alice_id)
            .expect("get alice")
            .expect("alice present");
    assert_eq!(alice_back, alice);
    let bob_back: UserProfile = get_document(&mut pager, &catalog, UserProfile::COLLECTION, bob_id)
        .expect("get bob")
        .expect("bob present");
    assert_eq!(bob_back, bob);
    let thread_back: Thread = get_document(&mut pager, &catalog, Thread::COLLECTION, thread_id)
        .expect("get thread")
        .expect("thread present");
    assert_eq!(thread_back, thread1);

    assert_eq!(alice_id.get(), 1, "alice is users#1");
    assert_eq!(bob_id.get(), 2, "bob is users#2");
    assert_eq!(thread_id.get(), 1, "thread is threads#1");
    pager.close().expect("close");
}

fn collection_id_of(
    pager: &mut Pager<FileHandle>,
    catalog: &Catalog<FileHandle>,
    name: &str,
) -> u32 {
    catalog
        .get(pager, name)
        .expect("catalog get")
        .expect("present")
        .collection_id
}

/// Set up the isolation test database with one user and one thread,
/// each at `Id(1)`. Returns the (users-collection-id,
/// threads-collection-id, shared-id) triple.
fn setup_isolation_db(path: &Path) -> (u32, u32, Id) {
    let (mut pager, mut catalog) = open_or_create(path).expect("open");
    pager.begin_txn();
    register_collection(
        &mut pager,
        &mut catalog,
        UserProfile::COLLECTION,
        UserProfile::VERSION,
    )
    .expect("users");
    register_collection(
        &mut pager,
        &mut catalog,
        Thread::COLLECTION,
        Thread::VERSION,
    )
    .expect("threads");
    let person_id = catalog
        .next_id(&mut pager, UserProfile::COLLECTION)
        .expect("user id");
    let topic_id = catalog
        .next_id(&mut pager, Thread::COLLECTION)
        .expect("thread id");
    assert_eq!(person_id, topic_id, "both collections issued Id(1)");
    let user = UserProfile {
        name: "u".to_owned(),
        age: 1,
        address: Address {
            street: "s".to_owned(),
            city: "c".to_owned(),
        },
    };
    let topic = Thread {
        title: "t".to_owned(),
        replies: Vec::new(),
    };
    insert_document(
        &mut pager,
        &mut catalog,
        UserProfile::COLLECTION,
        person_id,
        &user,
    )
    .expect("insert user");
    insert_document(
        &mut pager,
        &mut catalog,
        Thread::COLLECTION,
        topic_id,
        &topic,
    )
    .expect("insert thread");
    let users_cid = collection_id_of(&mut pager, &catalog, UserProfile::COLLECTION);
    let topics_cid = collection_id_of(&mut pager, &catalog, Thread::COLLECTION);
    pager.commit().expect("commit");
    pager.end_txn();
    pager.close().expect("close");
    (users_cid, topics_cid, topic_id)
}

#[test]
fn cross_collection_id_overlap_isolates() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("isolation.obj");
    let (user_collection_id, thread_collection_id, thread_id) = setup_isolation_db(&path);
    assert_ne!(
        user_collection_id, thread_collection_id,
        "collections must have distinct ids"
    );

    let (mut pager, catalog) = open_or_create(&path).expect("reopen");
    let thread_descriptor = catalog
        .get(&mut pager, Thread::COLLECTION)
        .expect("get threads")
        .expect("present");
    let thread_primary = PageId::new(thread_descriptor.primary_root).expect("non-zero");
    let thread_tree = BTree::<FileHandle>::open(&pager, thread_primary).expect("open thread");
    let raw_bytes = thread_tree
        .get(&mut pager, &thread_id.to_be_bytes())
        .expect("get raw")
        .expect("present");
    let err = codec::decode::<UserProfile>(&raw_bytes, user_collection_id).expect_err("mismatch");
    assert!(
        matches!(
            err,
            Error::CollectionIdMismatch { expected, found }
                if expected == user_collection_id && found == thread_collection_id
        ),
        "expected CollectionIdMismatch, got {err:?}",
    );
    pager.close().expect("close");
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct NotesV1 {
    body: String,
}

impl Document for NotesV1 {
    const COLLECTION: &'static str = "notes";
    const VERSION: u32 = 1;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct NotesV2 {
    body: String,
    tags: Vec<String>,
}

impl Document for NotesV2 {
    const COLLECTION: &'static str = "notes";
    const VERSION: u32 = 2;

    fn historical_schemas() -> Vec<(u32, obj_core::codec::DynamicSchema)> {
        use obj_core::codec::DynamicSchema;
        vec![(1, DynamicSchema::map([("body", DynamicSchema::String)]))]
    }

    fn migrate(dynamic: Dynamic, from_version: u32) -> Result<Self> {
        if from_version != 1 {
            return Err(Error::SchemaMigrationNotImplemented {
                collection: Self::COLLECTION,
                from_version,
                to_version: Self::VERSION,
            });
        }
        let body = match dynamic.get("body") {
            Some(Dynamic::String(s)) => s.clone(),
            _ => {
                return Err(Error::SchemaMigrationNotImplemented {
                    collection: Self::COLLECTION,
                    from_version,
                    to_version: Self::VERSION,
                });
            }
        };
        Ok(NotesV2 {
            body,
            tags: vec!["<migrated>".to_owned()],
        })
    }
}

/// Build a record buffer at `type_version = 1` for a `NotesV1`
/// payload. `encode<T>` always stamps `T::VERSION` so we hand-
/// assemble here.
fn assemble_v1_record(collection_id: u32, v1: &NotesV1) -> Vec<u8> {
    let payload = postcard::to_allocvec(v1).expect("postcard");
    let header = DocumentHeader {
        collection_id,
        type_version: NotesV1::VERSION,
        payload_len: u32::try_from(payload.len()).expect("fits u32"),
        payload_crc32c: obj_core::pager::checksum::crc32c(&payload),
    };
    let mut record = Vec::with_capacity(DOC_HEADER_SIZE + payload.len());
    header.write_to(&mut record);
    record.extend_from_slice(&payload);
    record
}

/// Write a v1 notes record to a fresh database. Returns (id,
/// `collection_id`) for the reader half of the test.
fn write_migration_database(path: &Path) -> (Id, u32) {
    let (mut pager, mut catalog) = open_or_create(path).expect("open");
    pager.begin_txn();
    let primary_root = BTree::<FileHandle>::empty(&mut pager)
        .expect("primary tree")
        .root();
    let descriptor = CollectionDescriptor::new(0, primary_root.get(), NotesV1::VERSION);
    let collection_id = catalog
        .insert(&mut pager, "notes", descriptor)
        .expect("register notes");
    let id = catalog.next_id(&mut pager, "notes").expect("id");
    let record = assemble_v1_record(
        collection_id,
        &NotesV1 {
            body: "first note".to_owned(),
        },
    );
    let descriptor = catalog
        .get(&mut pager, "notes")
        .expect("get")
        .expect("present");
    let primary_id = PageId::new(descriptor.primary_root).expect("non-zero");
    let mut tree = BTree::<FileHandle>::open(&pager, primary_id).expect("open primary");
    tree.insert(&mut pager, &id.to_be_bytes(), &record)
        .expect("insert v1");
    let mut updated = descriptor;
    updated.primary_root = tree.root().get();
    catalog
        .update(&mut pager, "notes", &updated)
        .expect("update");
    pager.commit().expect("commit");
    pager.end_txn();
    pager.close().expect("close");
    (id, collection_id)
}

#[test]
fn migration_shape_v1_to_v2_with_default_field() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("migration.obj");
    let (id, collection_id) = write_migration_database(&path);

    let (mut pager, catalog) = open_or_create(&path).expect("reopen");
    let descriptor = catalog
        .get(&mut pager, "notes")
        .expect("get notes")
        .expect("present");
    assert_eq!(descriptor.collection_id, collection_id);
    let primary_id = PageId::new(descriptor.primary_root).expect("non-zero");
    let tree = BTree::<FileHandle>::open(&pager, primary_id).expect("open primary");
    let bytes = tree
        .get(&mut pager, &id.to_be_bytes())
        .expect("get raw")
        .expect("present");
    let v2: NotesV2 = codec::decode(&bytes, descriptor.collection_id).expect("migrate v1 → v2");
    assert_eq!(
        v2,
        NotesV2 {
            body: "first note".to_owned(),
            tags: vec!["<migrated>".to_owned()],
        }
    );
    pager.close().expect("close");
}
