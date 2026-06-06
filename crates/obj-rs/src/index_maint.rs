//! Index maintenance — wire `Collection<T>` writes through every
//! `Active` `IndexDescriptor`.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use obj_core::btree::BTree;
use obj_core::index::{extract_index_keys, EncodedIndexKey};
use obj_core::pager::page::PageId;
use obj_core::pager::Pager;
use obj_core::platform::FileHandle;
use obj_core::{
    CollectionDescriptor, Document, Error, Id, IndexDescriptor, IndexKind, IndexSpec, IndexStatus,
    Result,
};

/// Apply the per-doc index churn implied by a write.
///
/// - `old`: the document's pre-write state (None on insert).
/// - `new`: the document's post-write state (None on delete).
/// - `id`: the document's `Id`.
///
/// Calls `extract_index_keys` for every `Active` `IndexDescriptor`
/// on the collection, diffs old-vs-new key sets, and emits the
/// minimal B-tree mutation set. For `Unique` indexes the diff
/// includes a pre-insert existence check against the new key — a
/// hit returns [`Error::UniqueConstraintViolation`] before any
/// mutation lands.
///
/// Every index `root_page_id` advance lands IN PLACE on the
/// caller-supplied `descriptor` (which is the per-txn cache entry) —
/// this routine NO LONGER calls `Catalog::update`. The single
/// catalog flush is deferred to [`crate::WriteTxn::commit`] via
/// [`crate::WriteTxn::flush_descriptors`], so a 64-doc batch pays one
/// `Catalog::update` per touched collection instead of two per doc.
/// The `descriptor` the caller passes is the SOLE mid-txn source of
/// truth: the `Unique` pre-check (`apply_unique_diff` → `tree.get`)
/// and the index-tree open in [`maintain_index_from_keys`] descend
/// the roots on THIS descriptor, so they observe every prior eager
/// index write inside the uncommitted txn.
pub(crate) fn apply_doc_change<T: Document>(
    pager: &mut Pager<FileHandle>,
    descriptor: &mut CollectionDescriptor,
    old: Option<&T>,
    new: Option<&T>,
    id: Id,
) -> Result<()> {
    let active_indexes: Vec<usize> = descriptor
        .indexes
        .iter()
        .enumerate()
        .filter_map(|(i, d)| (d.status == IndexStatus::Active).then_some(i))
        .collect();
    for idx in active_indexes {
        maintain_one_index::<T>(pager, descriptor, idx, old, new, id)?;
    }
    Ok(())
}

/// Maintain a single `IndexDescriptor`'s B-tree under the
/// old → new transition. Updates `descriptor.indexes[idx].root_page_id`
/// in place on COW root advance.
fn maintain_one_index<T: Document>(
    pager: &mut Pager<FileHandle>,
    descriptor: &mut CollectionDescriptor,
    idx: usize,
    old: Option<&T>,
    new: Option<&T>,
    id: Id,
) -> Result<()> {
    let collection_name = T::COLLECTION;
    let spec = descriptor_to_spec(&descriptor.indexes[idx])?;
    let old_keys = match old {
        Some(doc) => extract_index_keys(collection_name, &spec, doc)?,
        None => Vec::new(),
    };
    let new_keys = match new {
        Some(doc) => extract_index_keys(collection_name, &spec, doc)?,
        None => Vec::new(),
    };
    maintain_index_from_keys(
        pager,
        descriptor,
        idx,
        collection_name,
        &spec,
        &old_keys,
        &new_keys,
        id,
    )
}

/// Maintain a single index B-tree from already-extracted (or
/// caller-supplied) field-encoded key sets — the kind-specific
/// **storage-key composition** shared by the typed write path
/// ([`maintain_one_index`]) and the raw-bytes write path
/// ([`crate::txn::WriteTxn::insert_raw_indexed`] et al.).
///
/// This is the NON-generic seam: it takes the OLD and NEW
/// field-encoded keys (an [`EncodedIndexKey`] is the
/// order-preserving encoding of one index field — what the C side
/// produces via `obj_index_key_encode` and what `extract_index_keys`
/// produces on the typed path) and applies the exact same per-kind
/// storage layout the engine has always used:
///
/// - `Unique`: the encoded key is the B-tree key as-is; the `Id`
///   (8 bytes BE) is the VALUE; a pre-insert existence check turns a
///   duplicate into [`Error::UniqueConstraintViolation`].
/// - `Standard` / `Each` / `Composite`: the 8-byte BE `Id` is
///   appended to each encoded key (so the B-tree key stays globally
///   unique) and stored with the `Id` as value too.
///
/// Updates `descriptor.indexes[idx].root_page_id` in place on a COW
/// root advance, exactly as [`maintain_one_index`] did inline before
/// the refactor — the on-disk key composition is unchanged.
// allow: these args preserve the exact inline call shape this seam was factored out of; bundling them into a struct would add churn on the hot per-doc index path without changing the on-disk key composition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn maintain_index_from_keys(
    pager: &mut Pager<FileHandle>,
    descriptor: &mut CollectionDescriptor,
    idx: usize,
    collection_name: &str,
    spec: &IndexSpec,
    old_keys: &[EncodedIndexKey],
    new_keys: &[EncodedIndexKey],
    id: Id,
) -> Result<()> {
    let id_suffix = id.get().to_be_bytes();
    let mut tree = open_index_tree(pager, &descriptor.indexes[idx])?;
    match spec.kind {
        IndexKind::Unique => {
            apply_unique_diff(
                pager,
                &mut tree,
                collection_name,
                spec,
                old_keys,
                new_keys,
                &id_suffix,
            )?;
        }
        IndexKind::Standard | IndexKind::Each | IndexKind::Composite => {
            apply_nonunique_diff(pager, &mut tree, old_keys, new_keys, &id_suffix)?;
        }
        _ => return Err(Error::InvalidArgument("unsupported index kind")),
    }
    let new_root = tree.root().get();
    if new_root != descriptor.indexes[idx].root_page_id {
        descriptor.indexes[idx].root_page_id = new_root;
    }
    Ok(())
}

/// Diff the OLD vs NEW key sets for a `Unique` index. Emits the
/// minimum set of `delete(old)` / `insert(new)` calls and runs the
/// pre-insert existence check that turns a duplicate into
/// [`Error::UniqueConstraintViolation`].
fn apply_unique_diff(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    collection: &str,
    spec: &IndexSpec,
    old_keys: &[EncodedIndexKey],
    new_keys: &[EncodedIndexKey],
    id_suffix: &[u8],
) -> Result<()> {
    debug_assert!(old_keys.len() <= 1, "unique old key set must be 0..=1");
    debug_assert!(new_keys.len() <= 1, "unique new key set must be 0..=1");
    let old = old_keys.first().map(EncodedIndexKey::as_bytes);
    let new = new_keys.first().map(EncodedIndexKey::as_bytes);
    if old == new {
        return Ok(());
    }
    if let Some(new_bytes) = new {
        if let Some(existing_value) = tree.get(pager, new_bytes)? {
            if existing_value.as_slice() != id_suffix {
                return Err(Error::UniqueConstraintViolation {
                    collection: collection.to_owned(),
                    index: spec.name.clone(),
                    key: new_bytes.to_vec(),
                });
            }
        }
    }
    if let Some(old_bytes) = old {
        let _ = tree.delete(pager, old_bytes)?;
    }
    if let Some(new_bytes) = new {
        tree.insert(pager, new_bytes, id_suffix)?;
    }
    Ok(())
}

/// Diff the OLD vs NEW key sets for a non-unique index (`Standard`
/// / `Each` / `Composite`). Each entry's B-tree key is the encoded
/// user key with the 8-byte big-endian `Id` suffix appended.
///
/// When `old_keys` is empty — the pure-insert hot path (`old=None`
/// on the typed write, an empty prior set on the raw write) — there
/// is nothing to delete and the whole NEW set is "to insert", so we
/// take a fast path that inserts each composed key directly WITHOUT
/// building the `old`/`new` `BTreeSet` diffs. The composed key is
/// identical (`user_key || id_suffix`) so on-disk composition is
/// byte-for-byte unchanged. For a single NEW key (the common
/// `Standard`/`Composite`/single-element `Each` shape) no allocation
/// happens at all. For an `Each` doc with multiple elements we still
/// de-dup — equal elements compose to an equal B-tree key, which the
/// engine rejects with `BTreeKeyExists` — but only pay the dedup set
/// when there is more than one key, exactly the case the prior
/// `BTreeSet` guarded.
fn apply_nonunique_diff(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    old_keys: &[EncodedIndexKey],
    new_keys: &[EncodedIndexKey],
    id_suffix: &[u8],
) -> Result<()> {
    if old_keys.is_empty() {
        return insert_new_keys(pager, tree, new_keys, id_suffix);
    }
    let old_set: BTreeSet<Vec<u8>> = old_keys
        .iter()
        .map(|k| append_id_suffix(k.as_bytes(), id_suffix))
        .collect();
    let new_set: BTreeSet<Vec<u8>> = new_keys
        .iter()
        .map(|k| append_id_suffix(k.as_bytes(), id_suffix))
        .collect();
    for to_delete in old_set.difference(&new_set) {
        let _ = tree.delete(pager, to_delete)?;
    }
    for to_insert in new_set.difference(&old_set) {
        tree.insert(pager, to_insert, id_suffix)?;
    }
    Ok(())
}

/// Pure-insert fast path for a non-unique index: insert each NEW
/// composed key (`user_key || id_suffix`) directly, no old/new diff.
///
/// A single key (the common `Standard`/`Composite`/single-element
/// `Each` shape) inserts with zero auxiliary allocation. With more
/// than one key — only an `Each` doc carrying several elements — we
/// de-dup through a `BTreeSet`, matching the prior diff path, since
/// two equal elements compose to the same B-tree key (which the
/// engine would otherwise reject with `BTreeKeyExists`).
fn insert_new_keys(
    pager: &mut Pager<FileHandle>,
    tree: &mut BTree<FileHandle>,
    new_keys: &[EncodedIndexKey],
    id_suffix: &[u8],
) -> Result<()> {
    if new_keys.len() <= 1 {
        for key in new_keys {
            let composed = append_id_suffix(key.as_bytes(), id_suffix);
            tree.insert(pager, &composed, id_suffix)?;
        }
        return Ok(());
    }
    let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
    for key in new_keys {
        let composed = append_id_suffix(key.as_bytes(), id_suffix);
        if seen.insert(composed.clone()) {
            tree.insert(pager, &composed, id_suffix)?;
        }
    }
    Ok(())
}

/// Concatenate `user_key || id_suffix` into a fresh `Vec<u8>`. The
/// suffix is the 8-byte big-endian `Id` so range scans can recover
/// the document id by trimming the trailing 8 bytes.
fn append_id_suffix(user_key: &[u8], id_suffix: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(user_key.len() + id_suffix.len());
    out.extend_from_slice(user_key);
    out.extend_from_slice(id_suffix);
    out
}

/// Open the B+tree handle for an `IndexDescriptor`'s root.
fn open_index_tree(
    pager: &Pager<FileHandle>,
    descriptor: &IndexDescriptor,
) -> Result<BTree<FileHandle>> {
    let root = PageId::new(descriptor.root_page_id)
        .ok_or(Error::InvalidArgument("index descriptor root is zero"))?;
    BTree::<FileHandle>::open(pager, root)
}

/// Reconstruct an [`IndexSpec`] from an on-disk
/// [`IndexDescriptor`]. Used by the maintenance routines because
/// extraction needs the spec shape (kind + `key_paths` + name).
///
/// `pub(crate)` so the raw-bytes write path in [`crate::txn`] can
/// rebuild the spec for the kind-specific composition seam
/// ([`maintain_index_from_keys`]) without duplicating the
/// from-parts reconstruction.
pub(crate) fn descriptor_to_spec(d: &IndexDescriptor) -> Result<IndexSpec> {
    IndexSpec::from_parts(d.name.clone(), d.kind, d.key_paths.clone())
}
