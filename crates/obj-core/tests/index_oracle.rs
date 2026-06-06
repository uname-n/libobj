//! Index-layer oracle.
//!
//! `btree_oracle.rs` drives the raw [`obj_core::btree::BTree`] against
//! a `BTreeMap` oracle, but the INDEX layer that sits on top of it —
//! the order-preserving key encoding, unique enforcement, multi-value
//! `Each`, composite keys, and range-bound widening — is only covered
//! by smaller targeted unit tests. For a format about to freeze, the
//! index encoding is itself a contract; this oracle exercises it under
//! many randomized operations.
//!
//! # What is the oracle, and what is under test?
//!
//! The SYSTEM UNDER TEST is the obj-core index encoding
//! ([`encode_index_key`] / [`extract_index_keys`]) wired into a real
//! per-index [`BTree`] using the SAME maintenance-path key composition
//! the `obj` crate uses (append the 8-byte big-endian `Id` for
//! non-unique kinds; store the id as the value for `Unique`), and the
//! SAME range-bound widening.
//!
//! The ORACLE is an independent in-memory reference model: a
//! `BTreeMap<Id, Doc>` of live documents plus a recomputed-from-
//! scratch expectation for every observable query. The reference
//! never reuses the SUT's index B-tree to answer a query — it
//! recomputes the answer directly from the document set, so a bug in
//! the encoding or maintenance path surfaces as a divergence.
//!
//! Per operation we compare:
//! - the full ordered `(user_key, id)` set of every live index entry;
//! - point lookups by an exact user value (`lookup`);
//! - bounded range scans with every combination of Included / Excluded
//!   / Unbounded endpoints, widened per kind;
//! - `Unique` collision detection (insert of a duplicate key must be
//!   refused, exactly as the maintenance path refuses it).
//!
//! # Scope
//!
//! Each of the four kinds runs [`OPS_PER_CYCLE`] randomized ops under a
//! single deterministic `ChaCha8` seed (or a seed band via
//! `OBJ_INDEX_ORACLE_START` / `_END`). The op count is kept modest so
//! the test runs under a default `cargo test`. A 1M-op variant in the
//! spirit of `btree_oracle` is deferred (see the note on
//! [`OPS_PER_CYCLE`]).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::ops::Bound;

use obj_core::btree::BTree;
use obj_core::codec::Dynamic;
use obj_core::index::key::ENCODED_ID_SUFFIX_LEN;
use obj_core::index::{encode_field, encode_index_key, EncodedIndexKey, IndexKind, IndexSpec};
use obj_core::pager::{Config, Pager};
use obj_core::platform::FileHandle;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Operations per kind per seed. Deliberately modest so the test runs
/// under a default `cargo test`. A 1M-op variant (matching
/// `btree_oracle`) is DEFERRED: the index layer's correctness budget
/// is dominated by the encoding edge cases (sign boundaries, empty
/// `Each`, composite ordering, suffix widening), which a few thousand
/// ops over a small key universe already saturate; the raw B-tree
/// underneath is independently covered by the 1M-op `btree_oracle`.
const OPS_PER_CYCLE: usize = 2_000;

const DEFAULT_START: u64 = 0;
const DEFAULT_END: u64 = 2;

/// Cadence of the expensive full-scan + range comparison. Every op
/// mutates both the SUT and the reference; we verify the full
/// observable surface every `COMPARE_EVERY` ops (and always on the
/// final op) to keep a default `cargo test` fast while still pinning
/// the invariant frequently across the churn. Unique-collision and
/// extract-vs-oracle cross-checks run on EVERY op regardless.
const COMPARE_EVERY: usize = 4;

/// Size of the user-value universe. Small so random draws collide
/// often — collisions are what exercise the non-unique `Id`-suffix
/// disambiguation and the `Unique` constraint path.
const VALUE_UNIVERSE: u64 = 32;

/// Maximum number of live documents. Keeps each per-op full-scan
/// comparison cheap and bounds the range-scan drains.
const MAX_LIVE_DOCS: usize = 64;

/// A synthetic document. Only the fields a given index kind reads are
/// meaningful; the reference model and the SUT both read the same
/// fields, so unused fields never affect a comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Doc {
    /// Scalar field used by `Standard` / `Unique` indexes.
    scalar: i64,
    /// Sequence field used by `Each` indexes (0..=3 elements).
    tags: Vec<i64>,
    /// First composite field.
    a: u64,
    /// Second composite field.
    b: i64,
}

impl Doc {
    /// The `Dynamic` for the kind's first key field (used by the
    /// reference model's direct re-encoding).
    fn scalar_dyn(&self) -> Dynamic {
        Dynamic::I64(self.scalar)
    }
}

/// The four index kinds under test, paired with the spec the SUT
/// drives and the reference key-derivation the oracle uses.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Standard,
    Unique,
    Each,
    Composite,
}

impl Kind {
    fn spec(self) -> IndexSpec {
        match self {
            Kind::Standard => IndexSpec::standard("ix", "scalar").expect("spec"),
            Kind::Unique => IndexSpec::unique("ix", "scalar").expect("spec"),
            Kind::Each => IndexSpec::each("ix", "tags").expect("spec"),
            Kind::Composite => IndexSpec::composite("ix", &["a", "b"]).expect("spec"),
        }
    }

    fn index_kind(self) -> IndexKind {
        match self {
            Kind::Standard => IndexKind::Standard,
            Kind::Unique => IndexKind::Unique,
            Kind::Each => IndexKind::Each,
            Kind::Composite => IndexKind::Composite,
        }
    }

    fn is_unique(self) -> bool {
        matches!(self, Kind::Unique)
    }
}

/// The set of order-preserving USER keys a document contributes to an
/// index of `kind`, computed directly from the document fields. This
/// is the oracle's independent re-derivation; it does NOT call
/// `extract_index_keys` (the SUT does), so the two are cross-checked.
fn reference_user_keys(kind: Kind, doc: &Doc) -> Vec<Vec<u8>> {
    match kind {
        Kind::Standard | Kind::Unique => {
            vec![encode_one(&doc.scalar_dyn())]
        }
        Kind::Each => doc
            .tags
            .iter()
            .map(|t| encode_one(&Dynamic::I64(*t)))
            .collect(),
        Kind::Composite => {
            let spec = kind.spec();
            let key = encode_index_key(&spec, &[Dynamic::U64(doc.a), Dynamic::I64(doc.b)])
                .expect("composite encode");
            vec![key.into_bytes()]
        }
    }
}

/// Encode a single scalar field via the public `encode_field`.
fn encode_one(d: &Dynamic) -> Vec<u8> {
    encode_field(d).expect("encode_field").into_bytes()
}

/// Compose the full B-tree key the maintenance path would write for a
/// non-unique entry: `user_key || id_be8`.
fn compose_nonunique(user_key: &[u8], id: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(user_key.len() + ENCODED_ID_SUFFIX_LEN);
    out.extend_from_slice(user_key);
    out.extend_from_slice(&id.to_be_bytes());
    out
}

const SUFFIX_HIGH: [u8; ENCODED_ID_SUFFIX_LEN] = [0xFF; ENCODED_ID_SUFFIX_LEN];

/// Translate a user-facing encoded range into the internal B-tree
/// range. Mirrors `obj::index_bound::widen_bounds_for_kind` — the
/// frozen widening table.
fn widen(
    start: Bound<Vec<u8>>,
    end: Bound<Vec<u8>>,
    kind: IndexKind,
) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
    if kind == IndexKind::Unique {
        return (start, end);
    }
    (widen_lower(start), widen_upper(end))
}

fn widen_lower(b: Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(v),
        Bound::Excluded(v) => Bound::Excluded(with_high(v)),
        Bound::Unbounded => Bound::Unbounded,
    }
}

fn widen_upper(b: Bound<Vec<u8>>) -> Bound<Vec<u8>> {
    match b {
        Bound::Included(v) => Bound::Included(with_high(v)),
        Bound::Excluded(v) => Bound::Excluded(v),
        Bound::Unbounded => Bound::Unbounded,
    }
}

fn with_high(mut v: Vec<u8>) -> Vec<u8> {
    v.extend_from_slice(&SUFFIX_HIGH);
    v
}

/// Wraps a real obj-core [`BTree`] keyed exactly as the obj crate's
/// index maintenance keys it. Insert / delete mirror
/// `apply_unique_diff` / `apply_nonunique_diff`.
struct IndexSut {
    kind: Kind,
    tree: BTree<FileHandle>,
}

impl IndexSut {
    fn new(pager: &mut Pager<FileHandle>, kind: Kind) -> Self {
        let tree = BTree::<FileHandle>::empty(pager).expect("empty index tree");
        Self { kind, tree }
    }

    /// Insert every key `doc` contributes for `id`. For `Unique`,
    /// returns `Err(())` on a collision against a DIFFERENT id —
    /// exactly as the maintenance path refuses it — without mutating
    /// the tree.
    fn insert_doc(
        &mut self,
        pager: &mut Pager<FileHandle>,
        id: u64,
        doc: &Doc,
    ) -> std::result::Result<(), ()> {
        let user_keys = self.extract(doc);
        if self.kind.is_unique() {
            if let Some(uk) = user_keys.first() {
                if let Some(existing) = self.tree.get(pager, uk).expect("unique probe") {
                    if existing.as_slice() != id.to_be_bytes() {
                        return Err(());
                    }
                }
                self.tree
                    .insert(pager, uk, &id.to_be_bytes())
                    .expect("unique insert");
            }
            return Ok(());
        }
        for uk in &user_keys {
            let full = compose_nonunique(uk, id);
            if self.tree.get(pager, &full).expect("probe").is_none() {
                self.tree
                    .insert(pager, &full, &id.to_be_bytes())
                    .expect("nonunique insert");
            }
        }
        Ok(())
    }

    /// Delete every key `doc` contributed for `id`.
    fn delete_doc(&mut self, pager: &mut Pager<FileHandle>, id: u64, doc: &Doc) {
        let user_keys = self.extract(doc);
        if self.kind.is_unique() {
            if let Some(uk) = user_keys.first() {
                let _ = self.tree.delete(pager, uk).expect("unique delete");
            }
            return;
        }
        let mut seen: std::collections::BTreeSet<Vec<u8>> = std::collections::BTreeSet::new();
        for uk in &user_keys {
            let full = compose_nonunique(uk, id);
            if seen.insert(full.clone()) {
                let _ = self.tree.delete(pager, &full).expect("nonunique delete");
            }
        }
    }

    /// Drive `extract_index_keys` (the SUT's extraction) so a bug in
    /// reflection / encoding is exercised, but cross-check it against
    /// the oracle's independent re-derivation.
    fn extract(&self, doc: &Doc) -> Vec<Vec<u8>> {
        let oracle_keys = reference_user_keys(self.kind, doc);
        let spec = self.kind.spec();
        let sut_keys = extract_via_doc(&spec, doc);
        assert_eq!(
            sut_keys, oracle_keys,
            "extract_index_keys diverged from the oracle re-derivation for {:?} on {doc:?}",
            self.kind,
        );
        oracle_keys
    }

    /// The full ordered `(user_key, id)` set of every live entry the
    /// SUT tree holds, recovered by scanning the whole tree and
    /// trimming the trailing id suffix (non-unique) or reading the id
    /// from the value (unique).
    fn observed_entries(&self, pager: &mut Pager<FileHandle>) -> Vec<(Vec<u8>, u64)> {
        let iter = self.tree.iter(pager).expect("iter");
        let mut out: Vec<(Vec<u8>, u64)> = Vec::new();
        for step in iter {
            let (full_key, value) = step.expect("iter step");
            let (user_key, id) = if self.kind.is_unique() {
                (full_key, id_from_value(&value))
            } else {
                let cut = full_key.len() - ENCODED_ID_SUFFIX_LEN;
                (full_key[..cut].to_vec(), id_from_value(&value))
            };
            out.push((user_key, id));
        }
        out
    }

    /// Observed `(user_key, id)` entries inside a user-facing range,
    /// after kind-aware bound widening.
    fn observed_range(
        &self,
        pager: &mut Pager<FileHandle>,
        start: Bound<Vec<u8>>,
        end: Bound<Vec<u8>>,
    ) -> Vec<(Vec<u8>, u64)> {
        let (ws, we) = widen(start, end, self.kind.index_kind());
        let iter = self.tree.range(pager, (ws, we)).expect("range");
        let mut out: Vec<(Vec<u8>, u64)> = Vec::new();
        for step in iter {
            let (full_key, value) = step.expect("range step");
            let user_key = if self.kind.is_unique() {
                full_key
            } else {
                let cut = full_key.len() - ENCODED_ID_SUFFIX_LEN;
                full_key[..cut].to_vec()
            };
            out.push((user_key, id_from_value(&value)));
        }
        out
    }
}

/// Read the 8-byte big-endian id stored as the B-tree value.
fn id_from_value(value: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&value[..8]);
    u64::from_be_bytes(buf)
}

/// Run `extract_index_keys` against a serde-shaped view of `Doc`.
/// `Doc` is not a `Document`, so we feed the public extractor a
/// small wrapper that serializes the fields the spec reads.
fn extract_via_doc(spec: &IndexSpec, doc: &Doc) -> Vec<Vec<u8>> {
    use obj_core::index::extract_index_keys;
    use serde::{Deserialize, Serialize};
    #[derive(Serialize, Deserialize)]
    struct Wire {
        scalar: i64,
        tags: Vec<i64>,
        a: u64,
        b: i64,
    }
    impl obj_core::Document for Wire {
        const COLLECTION: &'static str = "oracle";
        const VERSION: u32 = 1;
    }
    let wire = Wire {
        scalar: doc.scalar,
        tags: doc.tags.clone(),
        a: doc.a,
        b: doc.b,
    };
    extract_index_keys("oracle", spec, &wire)
        .expect("extract")
        .into_iter()
        .map(EncodedIndexKey::into_bytes)
        .collect()
}

/// The independent reference: live documents keyed by id. Every
/// observable query is recomputed from this map, never from the SUT.
struct Reference {
    kind: Kind,
    live: BTreeMap<u64, Doc>,
}

impl Reference {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            live: BTreeMap::new(),
        }
    }

    /// Would inserting `doc` under id collide with a DIFFERENT live id
    /// on a `Unique` index? Mirrors the maintenance-path constraint.
    fn unique_collision(&self, id: u64, doc: &Doc) -> bool {
        if !self.kind.is_unique() {
            return false;
        }
        let uk = reference_user_keys(self.kind, doc);
        let Some(new_key) = uk.first() else {
            return false;
        };
        self.live.iter().any(|(other_id, other_doc)| {
            *other_id != id
                && reference_user_keys(self.kind, other_doc)
                    .first()
                    .is_some_and(|k| k == new_key)
        })
    }

    /// Expected full ordered `(user_key, id)` set, recomputed from the
    /// live documents. For non-unique kinds, `Each` duplicate elements
    /// collapse to one entry (the maintenance path dedups); the sort
    /// key is `(user_key, id)`.
    fn expected_entries(&self) -> Vec<(Vec<u8>, u64)> {
        let mut entries: std::collections::BTreeSet<(Vec<u8>, u64)> =
            std::collections::BTreeSet::new();
        for (id, doc) in &self.live {
            for uk in reference_user_keys(self.kind, doc) {
                entries.insert((uk, *id));
            }
        }
        entries.into_iter().collect()
    }

    /// Expected entries whose USER key falls in `[lo, hi]` (semantic,
    /// recomputed). The bound values are user `Dynamic`s already
    /// encoded to bytes; we compare encoded bytes directly.
    fn expected_range(&self, start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> Vec<(Vec<u8>, u64)> {
        self.expected_entries()
            .into_iter()
            .filter(|(uk, _)| in_user_bounds(uk, start, end))
            .collect()
    }
}

/// `true` if `uk` (an encoded user key) is within the user-facing
/// bounds. Compares encoded bytes — the encoding is order-preserving,
/// so byte order equals semantic order within a type.
fn in_user_bounds(uk: &[u8], start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> bool {
    let lo_ok = match start {
        Bound::Included(s) => uk >= s.as_slice(),
        Bound::Excluded(s) => uk > s.as_slice(),
        Bound::Unbounded => true,
    };
    let hi_ok = match end {
        Bound::Included(e) => uk <= e.as_slice(),
        Bound::Excluded(e) => uk < e.as_slice(),
        Bound::Unbounded => true,
    };
    lo_ok && hi_ok
}

#[derive(Debug)]
enum Op {
    Insert { id: u64, doc: Doc },
    Delete { id: u64 },
    Update { id: u64, doc: Doc },
}

fn random_doc(rng: &mut ChaCha8Rng) -> Doc {
    let scalar = i64::try_from(rng.random_range(0..VALUE_UNIVERSE)).unwrap_or(0)
        - i64::try_from(VALUE_UNIVERSE / 2).unwrap_or(0);
    let tag_count = rng.random_range(0u32..=3);
    let tags = (0..tag_count)
        .map(|_| i64::try_from(rng.random_range(0..VALUE_UNIVERSE)).unwrap_or(0))
        .collect();
    let a = rng.random_range(0..VALUE_UNIVERSE);
    let b = i64::try_from(rng.random_range(0..VALUE_UNIVERSE)).unwrap_or(0);
    Doc { scalar, tags, a, b }
}

fn draw_op(rng: &mut ChaCha8Rng, reference: &Reference) -> Op {
    let live_ids: Vec<u64> = reference.live.keys().copied().collect();
    let pick = rng.random_range(0u32..100);
    let want_insert = reference.live.len() < MAX_LIVE_DOCS && (pick < 50 || live_ids.is_empty());
    if want_insert {
        let id = next_free_id(rng, reference);
        return Op::Insert {
            id,
            doc: random_doc(rng),
        };
    }
    let victim = live_ids[rng.random_range(0..live_ids.len())];
    if pick < 75 {
        Op::Delete { id: victim }
    } else {
        Op::Update {
            id: victim,
            doc: random_doc(rng),
        }
    }
}

/// Pick an id not currently live. Ids are drawn from a bounded window
/// so they collide with freed ids over time (exercising re-insert).
fn next_free_id(rng: &mut ChaCha8Rng, reference: &Reference) -> u64 {
    let mut tries = 0u32;
    loop {
        tries += 1;
        let candidate = rng.random_range(1..=(MAX_LIVE_DOCS as u64 * 2));
        if !reference.live.contains_key(&candidate) {
            return candidate;
        }
        if tries > 1_000 {
            return reference
                .live
                .keys()
                .last()
                .copied()
                .unwrap_or(1)
                .saturating_add(1);
        }
    }
}

#[test]
fn index_oracle_seed_range() {
    let (start, end) = read_seed_range();
    let mut failures: Vec<(u64, Kind)> = Vec::new();
    for seed in start..end {
        for kind in [Kind::Standard, Kind::Unique, Kind::Each, Kind::Composite] {
            if let Err(msg) = run_one_cycle(seed, kind) {
                eprintln!("OBJ_INDEX_ORACLE_SEED={seed} kind={kind:?} FAILED: {msg}");
                failures.push((seed, kind));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "index_oracle: {} (seed, kind) cycles failed: {:?}",
        failures.len(),
        &failures[..failures.len().min(10)],
    );
}

fn read_seed_range() -> (u64, u64) {
    let start = std::env::var("OBJ_INDEX_ORACLE_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_START);
    let end = std::env::var("OBJ_INDEX_ORACLE_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_END);
    (start, end)
}

fn run_one_cycle(seed: u64, kind: Kind) -> std::result::Result<(), String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ kind_salt(kind));
    let mut pager =
        Pager::<FileHandle>::memory(Config::default()).map_err(|e| format!("pager: {e}"))?;
    pager.begin_txn();
    let mut sut = IndexSut::new(&mut pager, kind);
    let mut reference = Reference::new(kind);
    for step in 0..OPS_PER_CYCLE {
        let op = draw_op(&mut rng, &reference);
        apply_op(&mut pager, &mut sut, &mut reference, op)
            .map_err(|m| format!("step {step}: {m}"))?;
        if step % COMPARE_EVERY == 0 || step + 1 == OPS_PER_CYCLE {
            compare_all(&mut rng, &mut pager, &sut, &reference)
                .map_err(|m| format!("step {step}: {m}"))?;
        }
    }
    pager.end_txn();
    Ok(())
}

/// Per-kind salt so the four kinds explore distinct op streams under
/// the same seed.
fn kind_salt(kind: Kind) -> u64 {
    match kind {
        Kind::Standard => 0x1111_1111_1111_1111,
        Kind::Unique => 0x2222_2222_2222_2222,
        Kind::Each => 0x3333_3333_3333_3333,
        Kind::Composite => 0x4444_4444_4444_4444,
    }
}

fn apply_op(
    pager: &mut Pager<FileHandle>,
    sut: &mut IndexSut,
    reference: &mut Reference,
    op: Op,
) -> std::result::Result<(), String> {
    match op {
        Op::Insert { id, doc } => {
            let expect_collision = reference.unique_collision(id, &doc);
            let got = sut.insert_doc(pager, id, &doc);
            if expect_collision {
                if got.is_ok() {
                    return Err(format!(
                        "expected unique collision on id={id} doc={doc:?}, SUT accepted it"
                    ));
                }
            } else {
                if got.is_err() {
                    return Err(format!(
                        "SUT refused a non-colliding insert id={id} doc={doc:?}"
                    ));
                }
                reference.live.insert(id, doc);
            }
        }
        Op::Delete { id } => {
            if let Some(doc) = reference.live.remove(&id) {
                sut.delete_doc(pager, id, &doc);
            }
        }
        Op::Update { id, doc } => {
            let Some(old) = reference.live.get(&id).cloned() else {
                return Ok(());
            };
            reference.live.remove(&id);
            let expect_collision = reference.unique_collision(id, &doc);
            if expect_collision {
                reference.live.insert(id, old);
                return Ok(());
            }
            sut.delete_doc(pager, id, &old);
            sut.insert_doc(pager, id, &doc)
                .map_err(|()| format!("SUT refused update insert id={id}"))?;
            reference.live.insert(id, doc);
        }
    }
    Ok(())
}

/// Compare every observable surface: full scan + a handful of random
/// bounded ranges + a point lookup.
fn compare_all(
    rng: &mut ChaCha8Rng,
    pager: &mut Pager<FileHandle>,
    sut: &IndexSut,
    reference: &Reference,
) -> std::result::Result<(), String> {
    let observed = sut.observed_entries(pager);
    let expected = reference.expected_entries();
    if observed != expected {
        return Err(format!(
            "full-scan mismatch: observed_len={} expected_len={}",
            observed.len(),
            expected.len()
        ));
    }
    for _ in 0..3 {
        let (start, end) = random_user_bounds(rng, sut.kind);
        let obs = sut.observed_range(pager, start.clone(), end.clone());
        let exp = reference.expected_range(&start, &end);
        if obs != exp {
            return Err(format!(
                "range mismatch: bounds=({start:?},{end:?}) obs_len={} exp_len={}",
                obs.len(),
                exp.len()
            ));
        }
    }
    Ok(())
}

/// Draw a random user-facing encoded bound pair for `kind`. The bound
/// value is an encoded user key of the kind's first field type so it
/// is comparable to the stored entries.
fn random_user_bounds(rng: &mut ChaCha8Rng, kind: Kind) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
    let s = random_bound(rng, kind);
    let e = random_bound(rng, kind);
    (s, e)
}

fn random_bound(rng: &mut ChaCha8Rng, kind: Kind) -> Bound<Vec<u8>> {
    match rng.random_range(0u32..3) {
        0 => Bound::Unbounded,
        1 => Bound::Included(random_user_key(rng, kind)),
        _ => Bound::Excluded(random_user_key(rng, kind)),
    }
}

/// A random encoded USER key in the same shape the kind stores, so a
/// bound is meaningfully comparable to stored entries.
fn random_user_key(rng: &mut ChaCha8Rng, kind: Kind) -> Vec<u8> {
    match kind {
        Kind::Standard | Kind::Unique | Kind::Each => {
            let v = i64::try_from(rng.random_range(0..VALUE_UNIVERSE)).unwrap_or(0)
                - i64::try_from(VALUE_UNIVERSE / 2).unwrap_or(0);
            encode_one(&Dynamic::I64(v))
        }
        Kind::Composite => {
            let spec = kind.spec();
            let a = rng.random_range(0..VALUE_UNIVERSE);
            let b = i64::try_from(rng.random_range(0..VALUE_UNIVERSE)).unwrap_or(0);
            encode_index_key(&spec, &[Dynamic::U64(a), Dynamic::I64(b)])
                .expect("composite bound")
                .into_bytes()
        }
    }
}
