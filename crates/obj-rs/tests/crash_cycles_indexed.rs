//! Crash-injection orphan check for indexed inserts.
//!
//! Each cycle drives a randomised mix of `Db::transaction(insert /
//! update / delete)` against a hand-written [`Document`] whose
//! `T::indexes()` returns three specs covering [`IndexKind::Standard`],
//! [`IndexKind::Unique`], and [`IndexKind::Each`] (Composite is
//! deferred — see the module comment block on design choices). At
//! 25 % of cycles, a panic is injected AFTER a committed operation
//! to simulate a SIGKILL between two well-defined consistent points,
//! caught by `std::panic::catch_unwind`.
//!
//! After each cycle the test re-opens the database and walks both
//! directions:
//!
//! 1. **Forward:** every `(id, doc)` in the primary B-tree, projected
//!    through each `Active` [`IndexSpec`], must have its expected
//!    encoded-key + `Id` suffix entry in the matching index B-tree.
//!    For [`IndexKind::Each`] the indexed sequence may expand into
//!    multiple entries; every expected entry must be present.
//! 2. **Reverse:** every `(encoded_key + id_suffix, id_bytes)` entry
//!    in each index B-tree must point at a row in the primary tree
//!    whose decoded `Document` would produce the same `(encoded_key,
//!    id)` pair. A "deleted-but-not-cleaned-up" index entry — the
//!    canonical orphan — fails this check.
//!
//! Either direction failing is a categorical bug: orphaned index
//! entries in a "no silent corruption" database are not an acceptable
//! recovery outcome. The test surfaces a failing seed with a per-seed
//! operation log written to `target/crash_cycles_indexed/seed-<N>.log`.
//!
//! # Run
//!
//! ```text
//! cargo test --features fault-injection \
//!     --test crash_cycles_indexed -- --ignored --test-threads=1
//!
//! # Single seed, useful when CI surfaces a failing one:
//! OBJ_CRASH_CYCLES_INDEX_START=42 OBJ_CRASH_CYCLES_INDEX_END=43 \
//!     cargo test --features fault-injection \
//!     --test crash_cycles_indexed -- --ignored --test-threads=1
//! ```
//!
//! # Design choices
//!
//! - **Panic-only crashes**, no intra-syscall faults. The public
//!   `obj::Db` is hard-typed against `obj_core::platform::FileHandle`
//!   (see `Db`'s field layout in `crates/obj-rs/src/db.rs`), so the
//!   `FaultyFileHandle` indirection used by the `crash_cycles`
//!   test cannot be substituted without generalising `Db`. That
//!   generalisation is out of scope here — the intra-syscall
//!   variant already covers the WAL recovery contract end-to-end at
//!   the pager layer; here we verify that the **index maintenance**
//!   layer is atomic with the primary write at every
//!   commit boundary. Panic-only crashes at op boundaries are
//!   sufficient for that.
//!
//! - **The reopen-and-walk uses raw `obj_core::pager::Pager` +
//!   `obj_core::catalog::Catalog`**, not the `Db` API. The `Db`
//!   API's `Collection::all()` re-decodes documents through
//!   postcard, and `Collection::index_range` strips the per-entry
//!   id-suffix and de-duplicates on the `Each` kind — useful for
//!   user code, but the orphan check needs to see EVERY raw
//!   `(encoded_key + id_suffix, id_bytes)` entry to surface a
//!   dangling tombstone.
//!
//! - **Three indexes, one per scalar kind.** Composite is skipped
//!   for this first crash-cycle pass — its key-encoding surface
//!   doubles the number of code paths the workload must exercise.
//!   Add it as a
//!   follow-up once the three-kind variant is stable in CI.
//!
//! - **Mid-workload `Db` reopen.** The workload exercises Reopen at
//!   ~8 % frequency, covering the open → drop → open header-
//!   recovery path end-to-end through the public `Db` API.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use obj::{Db, Document, IndexKind, IndexSpec};
use obj_core::btree::BTree;
use obj_core::catalog::Catalog;
use obj_core::codec::{decode, Dynamic};
use obj_core::index::{encode_field, encode_index_key, MAX_EACH_ENTRIES};
use obj_core::pager::page::PageId;
use obj_core::pager::{Config as PagerConfig, Pager};
use obj_core::platform::FileHandle;
use obj_core::{Id, IndexDescriptor, IndexStatus};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

/// Raw `(full_key, id_bytes)` pair as it appears on disk inside an
/// index B-tree leaf. `full_key` includes the trailing per-entry
/// `Id` suffix for non-unique kinds; for `Unique` it is the user
/// key alone.
type IndexEntry = (Vec<u8>, Vec<u8>);

const DEFAULT_START: u64 = 0;
/// Default end of the seed range. CI's per-PR shard overrides via
/// `OBJ_CRASH_CYCLES_INDEX_END` (typically 250); the nightly job
/// covers the full `10_000` seeds.
const DEFAULT_END: u64 = 10_000;
/// Bound the recovery-check walk. A workload
/// of 30-100 ops over a fresh DB will not approach this number;
/// exceeding it returns `Err` rather than panicking.
const MAX_RECOVERY_CHECK_DOCS: usize = 10_000;
/// Max index-tree entries the reverse walk will visit before
/// declaring the workload pathological and returning an error.
/// `Each` indexes can multiply the entry count; `MAX_EACH_ENTRIES`
/// (= 16 384) is the upper bound per document on the `Each` side
/// alone. The test seeds tag counts to single digits, so this
/// budget is comfortably above the 30-100-op workload.
const MAX_RECOVERY_INDEX_ENTRIES: usize = 5 * MAX_EACH_ENTRIES;
/// Collection name shared across the test.
const COLLECTION: &str = "crash_indexed_docs";
const INDEX_BY_STATUS: &str = "by_status";
const INDEX_BY_EMAIL: &str = "by_email";
const INDEX_BY_TAG: &str = "by_tag";

/// The test document. Three indexes:
/// - `by_status` Standard — many-to-one on a 4-value enum.
/// - `by_email` Unique — collisions must surface as
///   [`obj::Error::UniqueConstraintViolation`] and roll the txn
///   back atomically with the primary write.
/// - `by_tag` Each — multi-value over a `Vec<String>`, zero or
///   more entries per doc.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestDoc {
    /// Unique-indexed: workload draws from a sparse pool so
    /// collisions are rare.
    email: String,
    /// Standard-indexed: drawn from a 4-value enum-shaped pool.
    status: String,
    /// Each-indexed: 0-3 tags per doc drawn from a small alphabet.
    tags: Vec<String>,
}

impl Document for TestDoc {
    const COLLECTION: &'static str = COLLECTION;
    const VERSION: u32 = 1;

    fn indexes() -> Vec<IndexSpec> {
        vec![
            IndexSpec::standard(INDEX_BY_STATUS, "status").expect("standard"),
            IndexSpec::unique(INDEX_BY_EMAIL, "email").expect("unique"),
            IndexSpec::each(INDEX_BY_TAG, "tags").expect("each"),
        ]
    }
}

impl obj::Schema for TestDoc {
    fn schema() -> obj::DynamicSchema {
        obj::DynamicSchema::map([
            ("email", obj::DynamicSchema::String),
            ("status", obj::DynamicSchema::String),
            ("tags", obj::DynamicSchema::seq(obj::DynamicSchema::String)),
        ])
    }
}

#[test]
#[ignore = "Indexed crash-cycles stress test (#62). Run via `cargo test --test crash_cycles_indexed -- --ignored`"]
fn crash_cycles_indexed_seed_range() {
    let (start, end) = read_seed_range();
    let mut failures: Vec<u64> = Vec::new();
    let mut total: u64 = 0;
    let mut crashed: u64 = 0;
    for seed in start..end {
        total += 1;
        match run_one_cycle(seed) {
            Ok(stats) => {
                if stats.crashed {
                    crashed += 1;
                }
            }
            Err(msg) => {
                eprintln!("OBJ_CRASH_CYCLES_INDEX_SEED={seed} FAILED: {msg}");
                failures.push(seed);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "crash_cycles_indexed: {} of {} seeds failed: first 10 = {:?}",
        failures.len(),
        total,
        &failures[..failures.len().min(10)]
    );
    eprintln!(
        "crash_cycles_indexed: {total} seeds passed (range {start}..{end}; \
         crash-injected = {crashed}, clean = {})",
        total - crashed
    );
}

/// `(start, end)` reads from the env. CI overrides these per shard.
fn read_seed_range() -> (u64, u64) {
    let start = std::env::var("OBJ_CRASH_CYCLES_INDEX_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_START);
    let end = std::env::var("OBJ_CRASH_CYCLES_INDEX_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_END);
    (start, end)
}

/// Result of a single cycle. `crashed` tracks whether the injected-
/// panic path actually fired (so the harness can report how many
/// cycles exercised fault-injection vs clean shutdown).
struct CycleStats {
    crashed: bool,
}

fn run_one_cycle(seed: u64) -> Result<CycleStats, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dir = TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
    let db_path = dir.path().join("crash-indexed.obj");
    let n_ops: u32 = rng.random_range(30u32..100);
    let panic_after_op: Option<u32> = if rng.random::<u32>() % 4 == 0 {
        Some(rng.random_range(5u32..n_ops))
    } else {
        None
    };
    let plan = build_op_plan(&mut rng, n_ops);
    let mut log: Vec<String> = Vec::with_capacity(n_ops as usize);
    let mut expected = ExpectedState::default();
    let crashed =
        run_workload_and_maybe_panic(&db_path, &plan, panic_after_op, &mut log, &mut expected)?;
    verify_indexes_bidirectional(&db_path, &expected).inspect_err(|_| {
        let _ = write_seed_log(seed, &log);
    })?;
    Ok(CycleStats { crashed })
}

/// Pre-roll the workload plan so the harness can re-run a failing
/// seed deterministically without re-driving the PRNG against the
/// db's commit flow. The plan is a flat `Vec<PlannedOp>`; the
/// workload loop walks it linearly.
fn build_op_plan(rng: &mut ChaCha8Rng, n_ops: u32) -> Vec<PlannedOp> {
    let mut plan = Vec::with_capacity(n_ops as usize);
    for _ in 0..n_ops {
        let pick = rng.random::<u32>() % 100;
        let op = match pick {
            0..=44 => PlannedOp::Insert(random_doc(rng)),
            45..=72 => PlannedOp::Update {
                pick: rng.random::<u32>(),
                new_doc: random_doc(rng),
            },
            73..=91 => PlannedOp::Delete {
                pick: rng.random::<u32>(),
            },
            _ => PlannedOp::Reopen,
        };
        plan.push(op);
    }
    plan
}

#[derive(Debug, Clone)]
enum PlannedOp {
    Insert(TestDoc),
    /// `pick` is reduced modulo the live-doc map's size at op time.
    Update {
        pick: u32,
        new_doc: TestDoc,
    },
    Delete {
        pick: u32,
    },
    /// Drop the live `Db` handle and reopen the file. The
    /// `ExpectedState` map is preserved — every previously-committed
    /// op must survive the reopen.
    Reopen,
}

/// Drives the workload through `Db::transaction` calls, optionally
/// panicking after the op at `panic_after_op` index. Returns `true`
/// if the panic fired (and thus the test exercised the fault path);
/// returns `false` on clean completion.
fn run_workload_and_maybe_panic(
    db_path: &Path,
    plan: &[PlannedOp],
    panic_after_op: Option<u32>,
    log: &mut Vec<String>,
    expected: &mut ExpectedState,
) -> Result<bool, String> {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        run_workload_inner(db_path, plan, panic_after_op, log, expected)
    }));
    match result {
        Ok(Ok(())) => Ok(false),
        Ok(Err(msg)) => Err(msg),
        Err(panic) => {
            let msg = panic_message(&panic);
            if msg.contains("__INJECTED_CRASH__") {
                Ok(true)
            } else {
                Err(format!("workload panicked: {msg}"))
            }
        }
    }
}

/// Runs the workload's `Db::transaction` loop. Panics with the
/// `__INJECTED_CRASH__` sentinel when `panic_after_op` matches the
/// current op index. The expected-state map is updated AFTER each
/// successful commit; a panicked commit boundary leaves it
/// unchanged so the recovery walk's expectations match what's
/// actually on disk.
fn run_workload_inner(
    db_path: &Path,
    plan: &[PlannedOp],
    panic_after_op: Option<u32>,
    log: &mut Vec<String>,
    expected: &mut ExpectedState,
) -> Result<(), String> {
    let mut db = open_db(db_path)?;
    for (idx, op) in plan.iter().enumerate() {
        match op {
            PlannedOp::Insert(doc) => apply_insert(&db, doc, expected, log, idx)?,
            PlannedOp::Update { pick, new_doc } => {
                apply_update(&db, *pick, new_doc, expected, log, idx)?;
            }
            PlannedOp::Delete { pick } => apply_delete(&db, *pick, expected, log, idx)?,
            PlannedOp::Reopen => {
                drop(db);
                db = open_db(db_path)?;
                log.push(format!("{idx}: reopen"));
            }
        }
        if Some(u32::try_from(idx).unwrap_or(u32::MAX)) == panic_after_op {
            log.push(format!("{idx}: __INJECTED_CRASH__"));
            panic!("__INJECTED_CRASH__ at step {idx}");
        }
    }
    drop(db);
    log.push("done".to_owned());
    Ok(())
}

fn open_db(path: &Path) -> Result<Db, String> {
    Db::open(path).map_err(|e| format!("open: {e}"))
}

/// Apply one Insert. Bumps the expected map iff the commit succeeded;
/// a `UniqueConstraintViolation` is an acceptable outcome (the
/// workload picks emails from a pool that DOES collide occasionally
/// to exercise the rollback path) and is silently absorbed.
fn apply_insert(
    db: &Db,
    doc: &TestDoc,
    expected: &mut ExpectedState,
    log: &mut Vec<String>,
    step: usize,
) -> Result<(), String> {
    let dup = expected.emails.contains(&doc.email);
    let outcome = db.insert::<TestDoc>(doc.clone());
    match (dup, &outcome) {
        (true, Err(obj::Error::UniqueConstraintViolation { .. })) => {
            log.push(format!("{step}: insert dup-rejected email={}", doc.email));
            Ok(())
        }
        (false, Ok(id)) => {
            expected.upsert(*id, doc.clone());
            log.push(format!(
                "{step}: insert id={} email={}",
                id.get(),
                doc.email
            ));
            Ok(())
        }
        (false, Err(obj::Error::UniqueConstraintViolation { .. })) => Err(format!(
            "{step}: unexpected unique-violation on email={}",
            doc.email
        )),
        (true, Ok(id)) => Err(format!(
            "{step}: expected unique-violation on email={} but got id={}",
            doc.email,
            id.get()
        )),
        (_, Err(other)) => Err(format!("{step}: insert error: {other}")),
    }
}

fn apply_update(
    db: &Db,
    pick: u32,
    new_doc: &TestDoc,
    expected: &mut ExpectedState,
    log: &mut Vec<String>,
    step: usize,
) -> Result<(), String> {
    let Some(id) = expected.pick_existing_id(pick) else {
        return Ok(());
    };
    let new_email_owner = expected.email_owner(&new_doc.email);
    if new_email_owner.is_some() && new_email_owner != Some(id) {
        log.push(format!(
            "{step}: update skip id={} (new email={} already owned)",
            id.get(),
            new_doc.email
        ));
        return Ok(());
    }
    let new_doc_clone = new_doc.clone();
    let outcome = db.update::<TestDoc, _>(id, move |d: &mut TestDoc| {
        *d = new_doc_clone;
    });
    match outcome {
        Ok(()) => {
            expected.upsert(id, new_doc.clone());
            log.push(format!(
                "{step}: update id={} -> email={} status={}",
                id.get(),
                new_doc.email,
                new_doc.status
            ));
            Ok(())
        }
        Err(obj::Error::DocumentNotFound { .. }) => Err(format!(
            "{step}: update id={} -> DocumentNotFound (map drift)",
            id.get()
        )),
        Err(obj::Error::UniqueConstraintViolation { .. }) => {
            log.push(format!("{step}: update dup-rejected id={}", id.get()));
            Ok(())
        }
        Err(e) => Err(format!("{step}: update id={}: {e}", id.get())),
    }
}

fn apply_delete(
    db: &Db,
    pick: u32,
    expected: &mut ExpectedState,
    log: &mut Vec<String>,
    step: usize,
) -> Result<(), String> {
    let Some(id) = expected.pick_existing_id(pick) else {
        return Ok(());
    };
    let removed = db
        .delete::<TestDoc>(id)
        .map_err(|e| format!("{step}: delete id={}: {e}", id.get()))?;
    if removed {
        expected.remove(id);
        log.push(format!("{step}: delete id={}", id.get()));
    } else {
        return Err(format!(
            "{step}: delete id={} returned false (map drift)",
            id.get()
        ));
    }
    Ok(())
}

/// Expected post-commit state for the recovery check. Keyed by `Id`
/// so the recovery walk can compare against the on-disk primary
/// tree row-by-row.
#[derive(Default)]
struct ExpectedState {
    docs: BTreeMap<u64, TestDoc>,
    /// Live emails — needed for the Unique pre-flight check that
    /// keeps the workload realistic.
    emails: BTreeSet<String>,
    /// Inverse map: email → id. Used by `email_owner`.
    email_owner: HashMap<String, Id>,
}

impl ExpectedState {
    fn upsert(&mut self, id: Id, doc: TestDoc) {
        if let Some(existing) = self.docs.get(&id.get()) {
            if existing.email != doc.email {
                self.emails.remove(&existing.email);
                self.email_owner.remove(&existing.email);
            }
        }
        self.emails.insert(doc.email.clone());
        self.email_owner.insert(doc.email.clone(), id);
        self.docs.insert(id.get(), doc);
    }

    fn remove(&mut self, id: Id) {
        if let Some(removed) = self.docs.remove(&id.get()) {
            self.emails.remove(&removed.email);
            self.email_owner.remove(&removed.email);
        }
    }

    fn pick_existing_id(&self, pick: u32) -> Option<Id> {
        if self.docs.is_empty() {
            return None;
        }
        let idx = (pick as usize) % self.docs.len();
        let raw = *self.docs.keys().nth(idx)?;
        Id::from_be_bytes(&raw.to_be_bytes())
    }

    fn email_owner(&self, email: &str) -> Option<Id> {
        self.email_owner.get(email).copied()
    }
}

/// Generate one random document. The email pool is sparse enough
/// that distinct-id collisions are rare; the tag alphabet is small
/// so the `Each` index sees genuine reuse (and thus exercises the
/// delete-side maintenance path through update transitions).
fn random_doc(rng: &mut ChaCha8Rng) -> TestDoc {
    let statuses = ["active", "archived", "pending", "suspended"];
    let tag_pool = ["red", "green", "blue", "yellow", "white", "black"];
    let email_pool_size: u32 = 64;
    let email_id = rng.random_range(0..email_pool_size);
    let n_tags = rng.random_range(0u32..4);
    let mut tags: Vec<String> = tag_pool.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    tags.shuffle(rng);
    let chosen_tags: Vec<String> = tags.into_iter().take(n_tags as usize).collect();
    TestDoc {
        email: format!("user{email_id}@example.com"),
        status: (*statuses
            .get(rng.random_range(0..statuses.len()))
            .unwrap_or(&"active"))
        .to_owned(),
        tags: chosen_tags,
    }
}

/// Walk BOTH directions over the recovered DB.
///
/// Re-opens the file via the public `Db::open` path (this is what
/// triggers the WAL recovery — the same code path a real crash
/// would take). After Db is open, we drop it and reattach a raw
/// `Pager + Catalog` to walk the index B-trees directly: the
/// `Db::Collection` API hides per-entry id-suffixes and de-dups on
/// `Each`, which would mask the orphan we are looking for.
fn verify_indexes_bidirectional(db_path: &Path, expected: &ExpectedState) -> Result<(), String> {
    {
        let db = Db::open(db_path).map_err(|e| format!("recovery open: {e}"))?;
        let _ = db.read_transaction(|tx| {
            let _ = tx.collection::<TestDoc>();
            Ok(())
        });
        drop(db);
    }
    let mut pager = Pager::<FileHandle>::open(db_path, PagerConfig::default())
        .map_err(|e| format!("raw pager open: {e}"))?;
    let catalog =
        Catalog::<FileHandle>::open_or_init(&mut pager).map_err(|e| format!("catalog: {e}"))?;
    let Some(descriptor) = catalog
        .get(&mut pager, COLLECTION)
        .map_err(|e| format!("catalog.get: {e}"))?
    else {
        if expected.docs.is_empty() {
            return Ok(());
        }
        return Err(format!(
            "recovery: collection {COLLECTION} missing but expected {} docs",
            expected.docs.len()
        ));
    };
    let active_indexes: Vec<IndexDescriptor> = descriptor
        .indexes
        .iter()
        .filter(|d| d.status == IndexStatus::Active)
        .cloned()
        .collect();
    let primary = walk_primary(
        &mut pager,
        descriptor.primary_root,
        descriptor.collection_id,
    )?;
    check_forward(&primary, &active_indexes, &mut pager)?;
    check_reverse(&primary, &active_indexes, &mut pager)?;
    check_expected_matches_primary(expected, &primary)?;
    Ok(())
}

/// Decode the primary B-tree into a `(Id, TestDoc)` map. Bounded by
/// `MAX_RECOVERY_CHECK_DOCS`.
fn walk_primary(
    pager: &mut Pager<FileHandle>,
    primary_root: u64,
    collection_id: u32,
) -> Result<BTreeMap<u64, TestDoc>, String> {
    let root =
        PageId::new(primary_root).ok_or_else(|| "recovery: primary root is zero".to_owned())?;
    let tree = BTree::<FileHandle>::open(pager, root).map_err(|e| format!("primary open: {e}"))?;
    let iter = tree
        .range(pager, ..)
        .map_err(|e| format!("primary range: {e}"))?;
    let mut out: BTreeMap<u64, TestDoc> = BTreeMap::new();
    let mut count = 0usize;
    for entry in iter {
        count += 1;
        if count > MAX_RECOVERY_CHECK_DOCS {
            return Err(format!(
                "recovery: primary tree exceeds MAX_RECOVERY_CHECK_DOCS={MAX_RECOVERY_CHECK_DOCS}"
            ));
        }
        let (key, value) = entry.map_err(|e| format!("primary iter: {e}"))?;
        let id = Id::from_be_bytes(&key).ok_or_else(|| "recovery: bad primary key".to_owned())?;
        let doc = decode::<TestDoc>(&value, collection_id).map_err(|e| format!("decode: {e}"))?;
        out.insert(id.get(), doc);
    }
    Ok(out)
}

/// Forward check: every `(id, doc)` in the primary tree projects to
/// a concrete set of `(encoded_key + id_suffix, id_bytes)` entries;
/// every such entry must be present in the matching index B-tree.
fn check_forward(
    primary: &BTreeMap<u64, TestDoc>,
    indexes: &[IndexDescriptor],
    pager: &mut Pager<FileHandle>,
) -> Result<(), String> {
    for descriptor in indexes {
        for (raw_id, doc) in primary {
            let id = Id::from_be_bytes(&raw_id.to_be_bytes())
                .ok_or_else(|| "recovery: bad primary id".to_owned())?;
            let entries = expected_entries_for(descriptor, doc, id)?;
            for (key_bytes, id_value) in entries {
                let got = open_index(pager, descriptor)?
                    .get(pager, &key_bytes)
                    .map_err(|e| format!("forward index get: {e}"))?;
                match got {
                    Some(bytes) if bytes == id_value => {}
                    Some(other) => {
                        return Err(format!(
                            "forward MISMATCH: index={} key={:?} expected_id_bytes={:?} got={:?}",
                            descriptor.name, key_bytes, id_value, other
                        ));
                    }
                    None => {
                        return Err(format!(
                            "forward MISSING: index={} key={:?} for id={}",
                            descriptor.name,
                            key_bytes,
                            id.get()
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Reverse check: every `(full_key, id_bytes)` entry in each index
/// B-tree must point at an id present in the primary tree, AND the
/// decoded primary doc must produce a key set that includes this
/// entry's user-key portion.
fn check_reverse(
    primary: &BTreeMap<u64, TestDoc>,
    indexes: &[IndexDescriptor],
    pager: &mut Pager<FileHandle>,
) -> Result<(), String> {
    for descriptor in indexes {
        let entries = collect_index_entries(pager, descriptor)?;
        for (full_key, value_bytes) in entries {
            let id = Id::from_be_bytes(&value_bytes).ok_or_else(|| {
                format!(
                    "reverse: index={} value not an Id (bytes={:?})",
                    descriptor.name, value_bytes
                )
            })?;
            let Some(doc) = primary.get(&id.get()) else {
                return Err(format!(
                    "reverse ORPHAN: index={} id={} not in primary tree (key={:?})",
                    descriptor.name,
                    id.get(),
                    full_key
                ));
            };
            let expected = expected_entries_for(descriptor, doc, id)?;
            if !expected.iter().any(|(k, _)| k == &full_key) {
                return Err(format!(
                    "reverse STALE: index={} id={} carries entry key={:?} that doc's current keys do not include",
                    descriptor.name,
                    id.get(),
                    full_key
                ));
            }
        }
    }
    Ok(())
}

/// Cross-check: the expected map should match the primary tree
/// AT THE LAST-COMMITTED OP. The crash boundary may have rolled
/// the disk state back to a slightly earlier commit (one not-yet-
/// committed insert lost), but it can never fabricate docs the
/// expected map never knew about. So `primary ⊆ expected`.
fn check_expected_matches_primary(
    expected: &ExpectedState,
    primary: &BTreeMap<u64, TestDoc>,
) -> Result<(), String> {
    for id in primary.keys() {
        if !expected.docs.contains_key(id) {
            return Err(format!(
                "primary contains id={id} the expected map never knew about (fabrication)"
            ));
        }
    }
    Ok(())
}

/// Produce the set of `(full_key, id_value)` pairs the doc would
/// own in `descriptor`'s index B-tree.
///
/// - `Standard` / `Each`: `full_key = encoded_user_key || id_be`,
///   `id_value = id_be`.
/// - `Unique`: `full_key = encoded_user_key`, `id_value = id_be`.
fn expected_entries_for(
    descriptor: &IndexDescriptor,
    doc: &TestDoc,
    id: Id,
) -> Result<Vec<IndexEntry>, String> {
    let id_bytes = id.get().to_be_bytes().to_vec();
    let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    match descriptor.kind {
        IndexKind::Standard => {
            let val = field_to_dynamic(&descriptor.key_paths[0], doc)?;
            let key = encode_index_key(&descriptor_to_spec(descriptor)?, &[val])
                .map_err(|e| format!("encode standard: {e}"))?
                .into_bytes();
            out.push((append_suffix(&key, &id_bytes), id_bytes.clone()));
        }
        IndexKind::Unique => {
            let val = field_to_dynamic(&descriptor.key_paths[0], doc)?;
            let key = encode_index_key(&descriptor_to_spec(descriptor)?, &[val])
                .map_err(|e| format!("encode unique: {e}"))?
                .into_bytes();
            out.push((key, id_bytes.clone()));
        }
        IndexKind::Each => {
            let path = descriptor.key_paths[0].as_str();
            let elements: &Vec<String> = match path {
                "tags" => &doc.tags,
                other => {
                    return Err(format!("unexpected Each path: {other}"));
                }
            };
            for el in elements {
                let key = encode_field(&Dynamic::String(el.clone()))
                    .map_err(|e| format!("encode each: {e}"))?
                    .into_bytes();
                out.push((append_suffix(&key, &id_bytes), id_bytes.clone()));
            }
        }
        IndexKind::Composite => {
            return Err("Composite indexes are out of scope for #62".to_owned());
        }
        other => {
            return Err(format!("unhandled IndexKind in test oracle: {other:?}"));
        }
    }
    Ok(out)
}

/// Reconstruct an `IndexSpec` from an on-disk descriptor.
fn descriptor_to_spec(d: &IndexDescriptor) -> Result<IndexSpec, String> {
    IndexSpec::from_parts(d.name.clone(), d.kind, d.key_paths.clone())
        .map_err(|e| format!("spec from_parts: {e}"))
}

/// Resolve a top-level field path on `TestDoc` to a `Dynamic`. The
/// production extractor uses serde reflection; here we know the
/// shape of `TestDoc` at compile time so a manual switch is clearer.
fn field_to_dynamic(path: &str, doc: &TestDoc) -> Result<Dynamic, String> {
    match path {
        "email" => Ok(Dynamic::String(doc.email.clone())),
        "status" => Ok(Dynamic::String(doc.status.clone())),
        "tags" => Ok(Dynamic::Seq(
            doc.tags
                .iter()
                .map(|s| Dynamic::String(s.clone()))
                .collect(),
        )),
        other => Err(format!("unexpected field path: {other}")),
    }
}

fn append_suffix(key: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(key.len() + suffix.len());
    out.extend_from_slice(key);
    out.extend_from_slice(suffix);
    out
}

/// Open a B-tree handle for the index's root page-id.
fn open_index(
    pager: &Pager<FileHandle>,
    descriptor: &IndexDescriptor,
) -> Result<BTree<FileHandle>, String> {
    let root = PageId::new(descriptor.root_page_id)
        .ok_or_else(|| format!("index {} root is zero", descriptor.name))?;
    BTree::<FileHandle>::open(pager, root).map_err(|e| format!("index open: {e}"))
}

/// Collect every `(full_key, value)` entry in the index's B-tree.
/// Bounded by `MAX_RECOVERY_INDEX_ENTRIES`.
fn collect_index_entries(
    pager: &mut Pager<FileHandle>,
    descriptor: &IndexDescriptor,
) -> Result<Vec<IndexEntry>, String> {
    let tree = open_index(pager, descriptor)?;
    let iter = tree
        .range(pager, ..)
        .map_err(|e| format!("index range: {e}"))?;
    let mut out: Vec<IndexEntry> = Vec::new();
    let mut count = 0usize;
    for entry in iter {
        count += 1;
        if count > MAX_RECOVERY_INDEX_ENTRIES {
            return Err(format!(
                "index {} exceeds MAX_RECOVERY_INDEX_ENTRIES={MAX_RECOVERY_INDEX_ENTRIES}",
                descriptor.name
            ));
        }
        let (k, v) = entry.map_err(|e| format!("index iter: {e}"))?;
        out.push((k, v));
    }
    Ok(out)
}

/// Convert a panic payload to its `Display` form.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    "<non-string panic payload>".to_string()
}

/// Write the per-seed operation log to disk for offline diagnosis.
fn write_seed_log(seed: u64, log: &[String]) -> std::io::Result<()> {
    let dir = PathBuf::from("target/crash_cycles_indexed");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("seed-{seed}.log"));
    std::fs::write(&path, log.join("\n"))?;
    eprintln!(
        "crash_cycles_indexed: wrote log for seed {seed} to {}",
        path.display()
    );
    Ok(())
}
