//! Macro hygiene gate for `#[derive(obj::Document)]`.
//!
//! The derive must use absolute paths (`::obj::Document`,
//! `::obj::IndexSpec`, `::obj::IndexKind`, `::std::vec::Vec`,
//! `::std::string::String`) so that a user crate that locally
//! shadows any of these names still compiles. This test sets up the
//! adversarial environment — a deeply-nested module path with local
//! types that match every name the derive references — and verifies
//! the emitted impl resolves to the real items.

// allow: the shadowing types below exist only to poison the namespace the derive
// resolves against; they are never constructed, so dead_code is expected here.
#![allow(dead_code)]

use obj::Document;

mod outer {
    pub mod middle {
        pub mod inner {
            use serde::{Deserialize, Serialize};

            // allow: a decoy `IndexSpec` shadowing the real name; never constructed.
            #[allow(dead_code)]
            pub struct IndexSpec;
            // allow: a decoy `IndexKind` shadowing the real name; never constructed.
            #[allow(dead_code)]
            pub struct IndexKind;
            // allow: a decoy `Document` shadowing the real name; never constructed.
            #[allow(dead_code)]
            pub struct Document;
            // allow: a decoy `Vec` shadowing the std name; never constructed.
            #[allow(dead_code)]
            pub struct Vec;
            // allow: a decoy `String` shadowing the std name; never constructed.
            #[allow(dead_code)]
            pub struct String;

            #[derive(Serialize, Deserialize, obj::Document)]
            #[obj(version = 2, collection = "shadowed")]
            #[obj(index_composite(fields = ("a", "b"), name = "by_a_b"))]
            pub struct Shadowed {
                #[obj(index = unique)]
                pub a: ::std::string::String,
                #[obj(index)]
                pub b: u32,
                #[obj(index = each)]
                pub c: ::std::vec::Vec<u32>,
            }
        }
    }
}

#[test]
fn derive_resolves_absolute_paths_under_local_shadows() {
    type S = outer::middle::inner::Shadowed;
    assert_eq!(<S as Document>::COLLECTION, "shadowed");
    assert_eq!(<S as Document>::VERSION, 2);

    let specs = <S as Document>::indexes();
    assert_eq!(specs.len(), 4);
    assert_eq!(specs[0].name, "a");
    assert_eq!(specs[1].name, "b");
    assert_eq!(specs[2].name, "c");
    assert_eq!(specs[3].name, "by_a_b");
}
