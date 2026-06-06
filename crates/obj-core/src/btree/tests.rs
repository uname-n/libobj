//! Unit tests for the B+tree read path.
//!
//! Tests hand-build pages by encoding [`DecodedNode`]s and writing
//! them via the pager, then exercise `BTree::get` against a fresh
//! tree (1-level) and a hand-built 2-level tree.

use crate::btree::node::{encode_node, DecodedNode, InternalEntry, LeafEntry, NodeKind};
use crate::btree::BTree;
use crate::pager::page::{Page, PageId};
use crate::pager::{Config, Pager};
use crate::platform::FileHandle;

fn config() -> Config {
    Config::default()
}

#[test]
fn empty_tree_get_returns_none() {
    let mut pager = Pager::<FileHandle>::memory(config()).expect("memory pager");
    let tree = BTree::<FileHandle>::empty(&mut pager).expect("empty tree");
    let result = tree.get(&mut pager, b"missing").expect("get");
    assert_eq!(result, None);
}

#[test]
fn one_level_lookup_hits_and_misses() {
    let mut pager = Pager::<FileHandle>::memory(config()).expect("memory pager");
    let root_id = pager.alloc_page().expect("alloc");
    let leaf = build_leaf(0, &[(b"alpha", b"A"), (b"bravo", b"B"), (b"charlie", b"C")]);
    let mut page = Page::zeroed();
    encode_node(&leaf, &mut page).expect("encode");
    pager.write_page(root_id, &page).expect("write");
    let tree = BTree::<FileHandle>::open(&pager, root_id).expect("open");
    assert_eq!(
        tree.get(&mut pager, b"alpha").expect("get"),
        Some(b"A".to_vec())
    );
    assert_eq!(
        tree.get(&mut pager, b"bravo").expect("get"),
        Some(b"B".to_vec())
    );
    assert_eq!(
        tree.get(&mut pager, b"charlie").expect("get"),
        Some(b"C".to_vec())
    );
    assert_eq!(tree.get(&mut pager, b"zulu").expect("get"), None);
    assert_eq!(tree.get(&mut pager, b"").expect("get"), None);
}

fn build_leaf(next_sibling: u64, entries: &[(&[u8], &[u8])]) -> DecodedNode {
    let mut leaf = DecodedNode {
        kind: NodeKind::Leaf,
        level: 0,
        next_sibling,
        children: Vec::new(),
        leaves: Vec::new(),
        internals: Vec::new(),
    };
    for (k, v) in entries {
        leaf.leaves.push(LeafEntry {
            key: k.to_vec(),
            value: v.to_vec(),
        });
    }
    leaf
}

#[test]
fn two_level_lookup_across_split_point() {
    let mut pager = Pager::<FileHandle>::memory(config()).expect("memory pager");
    let left_id = pager.alloc_page().expect("alloc left");
    let right_id = pager.alloc_page().expect("alloc right");
    let root_id = pager.alloc_page().expect("alloc root");
    let left = build_leaf(
        right_id.get(),
        &[(b"a", b"VA"), (b"f", b"VF"), (b"k", b"VK")],
    );
    let right = build_leaf(0, &[(b"n", b"VN"), (b"s", b"VS"), (b"z", b"VZ")]);
    let root = DecodedNode {
        kind: NodeKind::Internal,
        level: 1,
        next_sibling: 0,
        children: vec![left_id.get(), right_id.get()],
        leaves: Vec::new(),
        internals: vec![InternalEntry { key: b"m".to_vec() }],
    };

    for (node, pid) in [(&left, left_id), (&right, right_id), (&root, root_id)] {
        let mut page = Page::zeroed();
        encode_node(node, &mut page).expect("encode");
        pager.write_page(pid, &page).expect("write");
    }

    let tree = BTree::<FileHandle>::open(&pager, root_id).expect("open");
    for (k, want) in [
        (b"a".as_slice(), b"VA".as_slice()),
        (b"f", b"VF"),
        (b"k", b"VK"),
        (b"n", b"VN"),
        (b"s", b"VS"),
        (b"z", b"VZ"),
    ] {
        let got = tree.get(&mut pager, k).expect("get");
        assert_eq!(got.as_deref(), Some(want), "key {k:?}");
    }
    assert_eq!(tree.get(&mut pager, b"l").expect("get"), None);
    assert_eq!(tree.get(&mut pager, b"m").expect("get"), None);
    assert_eq!(tree.get(&mut pager, b"za").expect("get"), None);

    let _: PageId = tree.root();
}
