//! `obj-derive` — procedural macros for `obj`.
//!
//! # ⚠️ UNSTABLE — consume via `obj-rs`, not directly
//!
//! `obj-derive` is an implementation detail of `obj-rs` (re-exported as
//! `obj::Document`). It is published only so `obj-rs` can depend on it and
//! carries **no `SemVer` guarantee** as a standalone crate — depend on
//! `obj-rs` and write `#[derive(obj::Document)]`. Only `obj-rs`'s public
//! surface is the supported, `SemVer`-governed API; `obj-derive` is excluded
//! from the public-api stability gate.
//!
//! This crate provides `#[derive(obj::Document)]`, which emits the
//! `obj_core::Document` implementation for a user struct. The derive is intentionally
//! small — it fills in the trait's associated constants
//! (`COLLECTION`, `VERSION`) from optional `#[obj(...)]` attributes
//! and emits an `indexes()` override whenever any field carries an
//! `#[obj(index ...)]` attribute.
//!
//! # Supported attributes
//!
//! Struct-level (`#[obj(...)]` directly above the `struct` keyword):
//!
//! - `version = N` (integer ≥ 0) — sets `Document::VERSION`.
//! - `collection = "name"` (non-empty string literal) — sets
//!   `Document::COLLECTION`.
//!
//! Multiple `#[obj(...)]` attributes compose; the same scalar key
//! (`version`, `collection`) declared twice is a compile error.
//!
//! Struct-level composite (one or more occurrences compose, each
//! adding one `Composite` `IndexSpec`):
//!
//! - `index = ("a", "b")` — **canonical** composite form. Emits a
//!   `Composite` `IndexSpec` spanning the listed fields. The referenced
//!   fields must exist on the struct; fewer than two is a compile
//!   error. The default index name is the fields joined with `__`; an
//!   optional sibling `name = "..."` in the same `#[obj(...)]`
//!   overrides it, e.g. `#[obj(index = ("a", "b"), name = "by_a_b")]`.
//! - `index_composite(fields = ("a", "b"), name = "by_a_b")` — older
//!   long form, also accepted. Equivalent to the short form with the
//!   same downstream validation (≥ 2 fields, each declared on the
//!   struct); `name` likewise defaults to the fields joined with `__`.
//!   Prefer the short `index = (...)` form in new code.
//!
//! Field-level (`#[obj(...)]` on a struct field):
//!
//! - `index` — emit a `Standard` `IndexSpec` for this field.
//! - `index = unique` — emit a `Unique` `IndexSpec` for this field.
//! - `index = each` — emit an `Each` `IndexSpec` for this field. The
//!   field type must syntactically be `Vec<...>` — otherwise the
//!   derive errors at compile time.
//! - `name = "..."` — alongside any `index = ...`, overrides the
//!   default index name (which is the field name).
//!
//! Schema impl + enum opt-in:
//!
//! - `schema` — explicitly opt an **enum** into a derived
//!   `impl ::obj::Schema`. A bare `#[derive(obj::Document)]` on an
//!   enum is a hard error (an enum is never a `Document`); `schema`
//!   turns it into a `Schema`-only emission. The attribute is a
//!   no-op on structs, which always get a `Schema` impl.
//!
//! Auto-migration (pure-additive evolution):
//!
//! - `auto_migrate` (struct-level) — emit a `Document::migrate`
//!   override for the pure-additive case. The generated body reads
//!   every current field from the older record's `Dynamic::Map` by
//!   name: fields present in the old shape carry over (deserialised
//!   into the current field type), fields ABSENT from it (added in
//!   this version) backfill with `Default::default()`. This handles
//!   `(version bump) + (only new fields)` with no hand-written
//!   `migrate`. A type that needs a non-`Default` backfill, a field
//!   removal with side effects, or a type change must still
//!   hand-write the full `impl Document`. Declaring `auto_migrate`
//!   AND hand-writing `migrate` is a duplicate-method conflict —
//!   choose one.
//! - `default = <expr>` (field-level) — paired with `auto_migrate`,
//!   supplies a custom backfill expression for a field added in this
//!   version, used instead of `Default::default()` when the field is
//!   absent from the older shape.
//! - `default_with = <path>` (field-level) — paired with
//!   `auto_migrate`, points a newly-added field at a backfill function
//!   `fn(old: &Dynamic, from_version: u32) -> obj::Result<FieldTy>`.
//!   The function receives the old record (so it can derive the new
//!   value from any prior field) and the stored version, and may fail.
//!   It fires on the same absent branch as `default`. `default` and
//!   `default_with` on the same field is a compile error.
//!
//! The companion `impl ::obj::Schema` block's `schema()` body maps
//! each field to a `DynamicSchema` variant. Scalar primitives
//! (bool, u\*, i\*, f\*, String) map directly; `Vec<T>` maps to
//! `DynamicSchema::seq(<lowered T>)`; `Option<T>` maps to the
//! two-variant enum `[None = Null, Some = <lowered T>]` (recursing on
//! `T`, so `Option<scalar>` compiles without a scalar `Schema` impl —
//! byte-identical to the `Option<T>: Schema` blanket); anything else
//! delegates via `<T as ::obj::Schema>::schema()`, which fails to
//! compile if `T` lacks a `Schema` impl.
//!
//! # Serde requirements
//!
//! The derive does **NOT** emit `serde::Serialize` or
//! `serde::Deserialize` for you. Users still write
//! `#[derive(serde::Serialize, serde::Deserialize)]` on the struct
//! alongside `#[derive(obj::Document)]`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
// Rule 7 — the derive's shipping code path returns `syn::Error` for every failure
// rather than panicking. Gated on `not(test)` so unit and trybuild tests keep
// using unwrap/expect/panic freely. (`unsafe` is forbidden crate-wide.)
#![cfg_attr(not(test), deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_in_result,
    clippy::get_unwrap,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
))]

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, Attribute, Data, DataStruct, DeriveInput, Field, Fields, LitInt, LitStr,
    Type, TypePath,
};

/// Derive macro for `obj::Document`.
///
/// Emits `impl ::obj::Document for <Ident> { ... }` with sensible
/// defaults:
///
/// - `COLLECTION` defaults to the unqualified type name as a string;
///   `#[obj(collection = "explicit_name")]` overrides.
/// - `VERSION` defaults to `1`; `#[obj(version = N)]` overrides.
/// - `indexes()` is omitted (the trait default `Vec::new()` is used)
///   when the struct carries no index-related attributes; otherwise
///   the derive emits a `Vec<::obj::IndexSpec>` in field-declaration
///   order.
///
/// All emitted paths are absolute (`::obj::Document`,
/// `::obj::IndexSpec`) so the derive is hygienic against local items
/// that shadow these names.
#[proc_macro_derive(Document, attributes(obj))]
pub fn derive_document(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match emit_impl(&input) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Build the `impl ::obj::Document` block for `input`, plus the
/// companion `impl ::obj::Schema` for structs.
///
/// Every `#[derive(Document)]` struct gets a `Schema` impl
/// unconditionally: the obj insert path calls
/// `<T as ::obj::Schema>::schema()` for arbitrary `T: Document` to
/// persist the current-version schema on disk. (One consequence: a
/// struct whose fields are not built-in scalars or `Vec<T>` requires
/// each such field type to itself implement `Schema`.)
///
/// Enums are different: an enum is never a `Document`, so it gets a
/// `Schema`-ONLY emission, and only when the user opts in via
/// `#[obj(schema)]`. A bare `#[derive(Document)]` on an enum remains
/// a hard error — that gate is what `attrs.emit_schema` still guards.
fn emit_impl(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    reject_generics(input)?;
    let attrs = parse_struct_attrs(input)?;
    if matches!(input.data, Data::Enum(_)) {
        if !attrs.emit_schema {
            return Err(syn::Error::new(
                input.span(),
                "#[derive(obj::Document)] on an enum requires `#[obj(schema)]`; \
                 an enum is never a Document itself",
            ));
        }
        return emit_schema_impl(input);
    }
    emit_struct_impl(input, &attrs)
}

/// Reject any input carrying generic parameters (type, lifetime, or
/// const) before emission.
///
/// Every emitted `impl` block (`impl ::obj::Document for #ident`,
/// `impl ::obj::Schema for #ident`) uses the bare `input.ident` with
/// no `impl_generics` / `ty_generics` / `where_clause`. A generic
/// input (`struct Doc<T> { .. }`) would otherwise expand to
/// `impl Document for Doc` — missing `<T>` — and fail downstream with
/// a confusing "wrong number of type arguments" error. Generic
/// documents are not a supported feature, so we fail fast here with a
/// targeted diagnostic pointing at the generics.
fn reject_generics(input: &DeriveInput) -> syn::Result<()> {
    if input.generics.params.is_empty() {
        return Ok(());
    }
    Err(syn::Error::new(
        input.generics.span(),
        "#[derive(obj::Document)] does not support generic types",
    ))
}

/// Build the `impl ::obj::Document` block for a struct + the
/// companion `impl ::obj::Schema`, which is emitted unconditionally
/// for every struct (see [`emit_impl`] for why the gate was removed).
fn emit_struct_impl(
    input: &DeriveInput,
    attrs: &StructAttrs,
) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    let collection = attrs
        .collection
        .clone()
        .unwrap_or_else(|| ident.to_string());
    let version: u32 = attrs.version.unwrap_or(1);
    let mut index_specs = collect_field_indexes(input)?;
    let composite_specs = validate_and_lift_composites(input, &attrs.composites)?;
    index_specs.extend(composite_specs);
    let indexes_body = emit_indexes_body(&index_specs);
    let migrate_body = if attrs.auto_migrate {
        emit_auto_migrate_body(input)?
    } else {
        proc_macro2::TokenStream::new()
    };
    let schema_impl = emit_schema_impl(input)?;
    let out = quote! {
        #[automatically_derived]
        impl ::obj::Document for #ident {
            const COLLECTION: &'static str = #collection;
            const VERSION: u32 = #version;
            #indexes_body
            #migrate_body
        }
        #schema_impl
    };
    Ok(out)
}

/// Emit an `impl ::obj::Schema for <Ident>` block whose `schema()`
/// returns the `DynamicSchema::Map(...)` corresponding to the
/// struct's declared fields.
///
/// The mapping from Rust field type to `DynamicSchema` is the
/// syntactic table documented in `obj_core::codec::schema`:
/// scalar primitives map directly; `Vec<T>` maps to
/// `DynamicSchema::seq(<lowered T>)`; `Option<T>` maps to the
/// two-variant enum `[None = Null, Some = <lowered T>]`; anything
/// else is treated as a `Schema`-implementing path and delegates via
/// `<T as ::obj::Schema>::schema()`.
fn emit_schema_impl(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let ident = &input.ident;
    let body = match &input.data {
        Data::Struct(_) => emit_schema_body_struct(input)?,
        Data::Enum(data) => emit_schema_body_enum(data)?,
        Data::Union(_) => {
            return Err(syn::Error::new(
                input.span(),
                "#[derive(obj::Document)] does not support unions",
            ));
        }
    };
    Ok(quote! {
        #[automatically_derived]
        impl ::obj::Schema for #ident {
            fn schema() -> ::obj::DynamicSchema {
                #body
            }
        }
    })
}

/// Build the `Schema::schema()` body for a struct: a
/// `DynamicSchema::Map` over each named field's syntactic type.
fn emit_schema_body_struct(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let fields = named_fields(input)?;
    let entries = fields
        .iter()
        .map(|f| {
            let name = named_field_name(f)?;
            let ty_schema = field_type_to_schema(&f.ty);
            Ok(quote! { (::std::string::String::from(#name), #ty_schema) })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        ::obj::DynamicSchema::Map(::std::vec![ #( #entries ),* ])
    })
}

/// Emit the body of a derive-generated `Document::migrate` for the
/// **pure-additive** evolution case (`#[obj(auto_migrate)]`).
///
/// The generated method reads each current field out of the older
/// record's [`Dynamic`](obj_core::codec::Dynamic) map by name:
///
/// - **Present** — the field existed in the older shape; its decoded
///   sub-value is deserialised into the current field type via
///   `Dynamic::deserialize` and propagated with `?` on a type / shape
///   mismatch.
/// - **Absent** — the field was ADDED in this version; it backfills
///   with the per-field `#[obj(default_with = <path>)]` function
///   (`#path(&dynamic, _from_version)?`) if one was supplied, else the
///   per-field `#[obj(default = <expr>)]` expression, else
///   `::core::default::Default::default()`.
///
/// `_from_version` is forwarded to any `default_with` function but is
/// otherwise unused: the additive case treats every older version
/// identically (read what is there, default the rest). Types needing a
/// field removal with side effects or a type change must hand-write the
/// full `impl Document` and override `migrate` themselves.
fn emit_auto_migrate_body(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let fields = named_fields(input)?;
    let inits = fields
        .iter()
        .map(emit_auto_migrate_field)
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        fn migrate(
            dynamic: ::obj::Dynamic,
            _from_version: u32,
        ) -> ::obj::Result<Self> {
            ::std::result::Result::Ok(Self {
                #( #inits ),*
            })
        }
    })
}

/// Emit one `field: <expr>` initialiser for the auto-migrate `Self {
/// ... }` literal. Reads the field from the `Dynamic::Map` by name,
/// deserialising the present sub-value or falling back to the field's
/// backfill when absent. The absent-branch backfill is, in priority
/// order: a `#[obj(default_with = <path>)]` function applied to the old
/// record (`#path(&dynamic, _from_version)?`), a `#[obj(default =
/// <expr>)]` static expression, or `Default::default()`.
fn emit_auto_migrate_field(field: &Field) -> syn::Result<proc_macro2::TokenStream> {
    let ident = field
        .ident
        .as_ref()
        .ok_or_else(|| syn::Error::new(field.span(), "expected named field"))?;
    let name = ident.to_string();
    let ty = &field.ty;
    let backfill = match field_backfill(field)? {
        Some(Backfill::Expr(ts)) => ts,
        Some(Backfill::With(path)) => quote! { #path(&dynamic, _from_version)? },
        None => quote! { ::core::default::Default::default() },
    };
    Ok(quote! {
        #ident: match ::obj::Dynamic::get(&dynamic, #name) {
            ::std::option::Option::Some(__obj_v) => {
                ::obj::Dynamic::deserialize::<#ty>(__obj_v)?
            }
            ::std::option::Option::None => #backfill,
        }
    })
}

/// A per-field backfill source for the absent (newly-added-field)
/// branch of an `auto_migrate`-generated `migrate`.
enum Backfill {
    /// `#[obj(default = <expr>)]` — a static expression with no access
    /// to the migrating record; emitted verbatim.
    Expr(proc_macro2::TokenStream),
    /// `#[obj(default_with = <path>)]` — a function
    /// `fn(old: &Dynamic, from_version: u32) -> obj::Result<FieldTy>`
    /// that receives the old record and the stored version, so the
    /// backfill can read any prior field and may fail. Emitted as
    /// `#path(&dynamic, _from_version)?`.
    With(syn::Path),
}

/// Parse a field-level `#[obj(default = <expr>)]` /
/// `#[obj(default_with = <path>)]` backfill override.
///
/// Returns the parsed [`Backfill`], or `None` when the field carries
/// neither key (the caller then uses `Default::default()`). `default`
/// supplies a static expression; `default_with` supplies a function
/// applied to the old record. Either key declared twice, or `default`
/// and `default_with` declared on the **same field**, is a compile
/// error.
fn field_backfill(field: &Field) -> syn::Result<Option<Backfill>> {
    let mut backfill: Option<Backfill> = None;
    for attr in &field.attrs {
        if !attr.path().is_ident("obj") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("default") {
                if backfill.is_some() {
                    return Err(meta.error(
                        "`default` / `default_with` declared twice on the same field",
                    ));
                }
                let parsed: syn::Expr = meta.value()?.parse()?;
                backfill = Some(Backfill::Expr(quote! { #parsed }));
                return Ok(());
            }
            if meta.path.is_ident("default_with") {
                if backfill.is_some() {
                    return Err(meta.error(
                        "`default` / `default_with` declared twice on the same field",
                    ));
                }
                let path: syn::Path = meta.value()?.parse()?;
                backfill = Some(Backfill::With(path));
                return Ok(());
            }
            skip_unrelated_field_meta(&meta)
        })?;
    }
    Ok(backfill)
}

/// Consume (and ignore) the payload of a field-`#[obj(...)]` key that
/// `field_default_expr` does not handle, so the nested-meta parser can
/// move past it without erroring. Index keys carry either nothing,
/// `= <ident|tuple>`, or `= "name"`; we eat whatever follows an `=`.
fn skip_unrelated_field_meta(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<()> {
    if meta.input.peek(syn::Token![=]) {
        let value = meta.value()?;
        let _: proc_macro2::TokenStream = value.parse()?;
    }
    Ok(())
}

/// Pull the string name out of a named struct/variant field, returning
/// a `syn::Error` (never a panic) if the field has no `ident`.
///
/// `syn` only constructs `Field` values with `ident == None` inside
/// `Fields::Unnamed`. Every call site here is already guarded by a
/// `Fields::Named(_)` pattern, so the `None` branch is structurally
/// unreachable, but a panicking unwrap is avoided regardless. A
/// surfaced `syn::Error` is the safe fallback if a future refactor
/// breaks the invariant.
fn named_field_name(field: &Field) -> syn::Result<String> {
    field
        .ident
        .as_ref()
        .map(ToString::to_string)
        .ok_or_else(|| syn::Error::new(field.span(), "expected named field"))
}

/// Build the `Schema::schema()` body for an enum: a
/// `DynamicSchema::Enum` over each variant in declaration order
/// (postcard assigns discriminants by declaration order; the derive
/// matches that). Unit variants get `Null` payloads; newtype
/// variants get the inner type's schema; tuple variants get a
/// synthetic `Map` keyed by `"0"`, `"1"`, …; struct variants get a
/// `Map` keyed by the field names.
fn emit_schema_body_enum(data: &syn::DataEnum) -> syn::Result<proc_macro2::TokenStream> {
    let entries = data
        .variants
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let discriminant = u32::try_from(idx).unwrap_or(u32::MAX);
            let name = v.ident.to_string();
            let payload = variant_payload_schema(&v.fields)?;
            Ok(quote! {
                ::obj::EnumVariantSchema::new(
                    #discriminant,
                    #name,
                    #payload,
                )
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        ::obj::DynamicSchema::Enum(::std::vec![ #( #entries ),* ])
    })
}

/// Map an enum variant's `Fields` shape to the token stream that
/// constructs its payload [`DynamicSchema`] at runtime.
fn variant_payload_schema(fields: &Fields) -> syn::Result<proc_macro2::TokenStream> {
    match fields {
        Fields::Unit => Ok(quote! { ::obj::DynamicSchema::Null }),
        Fields::Unnamed(unnamed) => {
            let count = unnamed.unnamed.len();
            if count == 1 {
                let ty = &unnamed.unnamed[0].ty;
                Ok(field_type_to_schema(ty))
            } else {
                let entries = unnamed.unnamed.iter().enumerate().map(|(i, f)| {
                    let key = i.to_string();
                    let ty_schema = field_type_to_schema(&f.ty);
                    quote! { (::std::string::String::from(#key), #ty_schema) }
                });
                Ok(quote! {
                    ::obj::DynamicSchema::Map(::std::vec![ #( #entries ),* ])
                })
            }
        }
        Fields::Named(named) => {
            let entries = named
                .named
                .iter()
                .map(|f| {
                    let name = named_field_name(f)?;
                    let ty_schema = field_type_to_schema(&f.ty);
                    Ok(quote! { (::std::string::String::from(#name), #ty_schema) })
                })
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(quote! {
                ::obj::DynamicSchema::Map(::std::vec![ #( #entries ),* ])
            })
        }
    }
}

/// Map a struct field's syntactic Rust type to a token-stream that
/// constructs a [`DynamicSchema`] value at runtime.
fn field_type_to_schema(ty: &Type) -> proc_macro2::TokenStream {
    if let Some(name) = scalar_schema_for(ty) {
        let ident = quote::format_ident!("{name}");
        return quote! { ::obj::DynamicSchema::#ident };
    }
    if let Some(inner) = vec_inner_type(ty) {
        let inner_schema = field_type_to_schema(inner);
        return quote! { ::obj::DynamicSchema::seq(#inner_schema) };
    }
    if let Some(inner) = option_inner_type(ty) {
        let inner_schema = field_type_to_schema(inner);
        return quote! {
            ::obj::DynamicSchema::enumeration([
                ::obj::EnumVariantSchema::new(0, "None", ::obj::DynamicSchema::Null),
                ::obj::EnumVariantSchema::new(1, "Some", #inner_schema),
            ])
        };
    }
    quote! { <#ty as ::obj::Schema>::schema() }
}

/// Return the [`DynamicSchema`] variant name for `ty` if `ty` is one
/// of the built-in scalars; `None` otherwise. The result is used by
/// [`field_type_to_schema`] to construct the leaf token stream.
fn scalar_schema_for(ty: &Type) -> Option<&'static str> {
    let Type::Path(TypePath { qself: None, path }) = ty else {
        return None;
    };
    let segment = path.segments.last()?;
    if !segment.arguments.is_none() {
        return None;
    }
    let s = segment.ident.to_string();
    match s.as_str() {
        "bool" => Some("Bool"),
        "u8" | "u16" | "u32" | "u64" | "usize" => Some("U64"),
        "i8" | "i16" | "i32" | "i64" | "isize" => Some("I64"),
        "f32" | "f64" => Some("F64"),
        "String" => Some("String"),
        _ => None,
    }
}

/// If `ty` is `Vec<T>`, return `&T`; otherwise `None`.
fn vec_inner_type(ty: &Type) -> Option<&Type> {
    single_generic_arg(ty, "Vec")
}

/// If `ty` is `Option<T>`, return `&T`; otherwise `None`.
///
/// Used by [`field_type_to_schema`] to lower an `Option<T>` field into
/// the two-variant enum schema syntactically (so `Option<scalar>`
/// compiles without `scalar: Schema`). Matches the last path segment's
/// ident so `Option`, `::core::option::Option`, `std::option::Option`,
/// etc. all resolve.
fn option_inner_type(ty: &Type) -> Option<&Type> {
    single_generic_arg(ty, "Option")
}

/// If `ty` is a path whose last segment is `wrapper<T>` (a single
/// angle-bracketed type argument), return `&T`; otherwise `None`.
/// Shared by [`vec_inner_type`] and [`option_inner_type`].
fn single_generic_arg<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    let Type::Path(TypePath { qself: None, path }) = ty else {
        return None;
    };
    let seg = path.segments.last()?;
    if seg.ident != wrapper {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

/// Validate every composite declaration against the struct's named
/// fields and lift each one into an `IndexSpecEmit`. Errors on:
///
/// - composite with fewer than 2 fields,
/// - a referenced field name that is not declared on the struct.
fn validate_and_lift_composites(
    input: &DeriveInput,
    composites: &[CompositeAttr],
) -> syn::Result<Vec<IndexSpecEmit>> {
    if composites.is_empty() {
        return Ok(Vec::new());
    }
    let fields = named_fields(input)?;
    let known: std::collections::HashSet<String> = fields
        .iter()
        .filter_map(|f| f.ident.as_ref().map(ToString::to_string))
        .collect();
    let mut out: Vec<IndexSpecEmit> = Vec::with_capacity(composites.len());
    for c in composites {
        if c.fields.len() < 2 {
            return Err(syn::Error::new(c.span, "composite needs ≥ 2 fields"));
        }
        for field in &c.fields {
            if !known.contains(field) {
                return Err(syn::Error::new(
                    c.span,
                    format!("field '{field}' not declared on struct"),
                ));
            }
        }
        let index_name = c.custom_name.clone().unwrap_or_else(|| c.fields.join("__"));
        out.push(IndexSpecEmit {
            kind: IndexKind::Composite(c.fields.clone()),
            field_name: String::new(),
            index_name,
        });
    }
    Ok(out)
}

/// Emit either an `indexes()` override or an empty token stream
/// (which leaves the trait default in place).
fn emit_indexes_body(specs: &[IndexSpecEmit]) -> proc_macro2::TokenStream {
    if specs.is_empty() {
        return proc_macro2::TokenStream::new();
    }
    let entries = specs.iter().map(IndexSpecEmit::emit);
    quote! {
        fn indexes() -> ::std::vec::Vec<::obj::IndexSpec> {
            let mut out: ::std::vec::Vec<::obj::IndexSpec> = ::std::vec::Vec::new();
            #(
                if let ::std::result::Result::Ok(spec) = #entries {
                    out.push(spec);
                }
            )*
            out
        }
    }
}

/// One parsed `#[obj(index_composite(...))]` declaration.
#[derive(Debug)]
struct CompositeAttr {
    /// User-provided field names. Each MUST exist on the struct.
    fields: Vec<String>,
    /// Optional `name = "..."` override; default is the fields joined
    /// with `__`.
    custom_name: Option<String>,
    /// Span used for "field 'x' not declared on struct" diagnostics.
    span: proc_macro2::Span,
}

/// Parsed struct-level attributes.
#[derive(Default, Debug)]
struct StructAttrs {
    /// `#[obj(version = N)]` override.
    version: Option<u32>,
    /// `#[obj(collection = "name")]` override.
    collection: Option<String>,
    /// Zero or more `#[obj(index_composite(...))]` declarations,
    /// preserved in declaration order so the emitted `indexes()` is
    /// deterministic.
    composites: Vec<CompositeAttr>,
    /// `true` iff the user opted in via `#[obj(schema)]`.
    ///
    /// Structs emit `impl ::obj::Schema` UNCONDITIONALLY, so this
    /// flag does not gate struct emission. It survives to gate the
    /// enum path: a bare `#[derive(Document)]` on an enum is a hard
    /// error, and only `#[obj(schema)]` turns it into a `Schema`-only
    /// emission. It also lets `#[obj(schema)]` detect a redundant
    /// second opt-in (`schema` declared twice).
    emit_schema: bool,
    /// `true` iff the user opted in via `#[obj(auto_migrate)]`.
    ///
    /// When set, the derive emits a `Document::migrate` override that
    /// handles the **pure-additive** evolution case: every current
    /// field is read from the older record's `Dynamic::Map` by name;
    /// fields present in the old shape carry over, fields absent from
    /// it (added in this version) fall back to `Default::default()` (or
    /// a per-field `#[obj(default = <expr>)]` /
    /// `#[obj(default_with = <path>)]` backfill). Struct-only; declared
    /// twice is a compile error.
    auto_migrate: bool,
}

/// Walk every `#[obj(...)]` on the struct and merge them into a
/// single `StructAttrs`. Duplicates (within one `#[obj(...)]` OR
/// across two) error.
fn parse_struct_attrs(input: &DeriveInput) -> syn::Result<StructAttrs> {
    let mut acc = StructAttrs::default();
    for attr in &input.attrs {
        if !attr.path().is_ident("obj") {
            continue;
        }
        parse_one_struct_attr(attr, &mut acc)?;
    }
    Ok(acc)
}

/// Parse a single `#[obj(...)]` attribute into `acc`. Duplicate
/// scalar keys (within this attribute OR already present in `acc`)
/// error; `index_composite(...)` is non-scalar and appends new
/// entries. A short-form `index = (...)` and its optional sibling
/// `name = "..."` are accumulated locally and merged once the whole
/// attribute has parsed (see [`apply_struct_index_name`]). The removed
/// `history(...)` key is rejected with a migration-pointing
/// diagnostic.
fn parse_one_struct_attr(attr: &Attribute, acc: &mut StructAttrs) -> syn::Result<()> {
    let mut short: Option<CompositeAttr> = None;
    let mut short_name: Option<LitStr> = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("version") {
            return parse_struct_version(&meta, acc);
        }
        if meta.path.is_ident("collection") {
            return parse_struct_collection(&meta, acc);
        }
        if meta.path.is_ident("index_composite") {
            let composite = parse_index_composite(&meta)?;
            acc.composites.push(composite);
            return Ok(());
        }
        if meta.path.is_ident("index") {
            if short.is_some() {
                return Err(meta.error("`index` declared twice in one #[obj(...)]"));
            }
            short = Some(parse_struct_index_short(&meta)?);
            return Ok(());
        }
        if meta.path.is_ident("name") {
            if short_name.is_some() {
                return Err(meta.error("`name` declared twice"));
            }
            short_name = Some(meta.value()?.parse()?);
            return Ok(());
        }
        if meta.path.is_ident("history") {
            return Err(meta.error(
                "#[obj(history(...))] is no longer supported: schemas are now persisted on disk; \
                 bump #[obj(version = N)] and add a Document::migrate impl instead",
            ));
        }
        if meta.path.is_ident("schema") {
            if acc.emit_schema {
                return Err(meta.error("`schema` declared twice"));
            }
            acc.emit_schema = true;
            return Ok(());
        }
        if meta.path.is_ident("auto_migrate") {
            if acc.auto_migrate {
                return Err(meta.error("`auto_migrate` declared twice"));
            }
            acc.auto_migrate = true;
            return Ok(());
        }
        Err(meta.error(
            "unknown obj attribute (expected `version`, `collection`, `index`, `index_composite`, `schema`, or `auto_migrate`)",
        ))
    })?;
    apply_struct_index_name(acc, short, short_name)
}

/// Merge an optional struct-level `name = "..."` into the short-form
/// `index = (...)` parsed from the same `#[obj(...)]` attribute, then
/// push the resulting [`CompositeAttr`]. An empty name is rejected
/// (mirroring `index_composite(name = "...")`), and a `name` with no
/// accompanying short `index` in the same attribute is an error —
/// `name` at struct level only ever qualifies a short composite index.
fn apply_struct_index_name(
    acc: &mut StructAttrs,
    short: Option<CompositeAttr>,
    short_name: Option<LitStr>,
) -> syn::Result<()> {
    match (short, short_name) {
        (Some(mut composite), name) => {
            if let Some(lit) = name {
                let s = lit.value();
                if s.is_empty() {
                    return Err(syn::Error::new(
                        lit.span(),
                        "composite index name must not be empty",
                    ));
                }
                composite.custom_name = Some(s);
            }
            acc.composites.push(composite);
            Ok(())
        }
        (None, Some(lit)) => Err(syn::Error::new(
            lit.span(),
            "struct-level `name = \"...\"` requires an `index = (...)` in the same #[obj(...)]",
        )),
        (None, None) => Ok(()),
    }
}

/// Parse the short composite-index form `#[obj(index = ("a", "b"))]`
/// at struct level. The only valid RHS is a parenthesised tuple of
/// string literals — `unique` / `each` / a bare path are field-level
/// shapes and yield a struct-level diagnostic that points back at
/// `index_composite` / field-level placement.
///
/// The returned [`CompositeAttr`] carries `custom_name: None`; an
/// optional sibling `name = "..."` in the same `#[obj(...)]` is merged
/// in afterwards by [`apply_struct_index_name`]. It is then validated
/// downstream by [`validate_and_lift_composites`], which already
/// enforces the `≥ 2 fields` and "field declared on struct"
/// invariants — both the long and short forms share the same
/// downstream gate.
fn parse_struct_index_short(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<CompositeAttr> {
    let span = meta.path.span();
    let kind = parse_index_kind(meta)?;
    match kind {
        IndexKind::Composite(fields) => Ok(CompositeAttr {
            fields,
            custom_name: None,
            span,
        }),
        _ => Err(syn::Error::new(
            span,
            "struct-level `index = ...` only accepts a tuple of field-name string literals \
             (e.g. `index = (\"a\", \"b\")`); place `index`, `index = unique`, or `index = each` \
             on a field instead",
        )),
    }
}

/// Parse `version = N`.
fn parse_struct_version(
    meta: &syn::meta::ParseNestedMeta<'_>,
    acc: &mut StructAttrs,
) -> syn::Result<()> {
    if acc.version.is_some() {
        return Err(meta.error("`version` declared twice"));
    }
    let value = meta.value()?;
    let lit: LitInt = value.parse()?;
    let n: u32 = lit
        .base10_parse()
        .map_err(|_| syn::Error::new(lit.span(), "expected unsigned integer for `version`"))?;
    acc.version = Some(n);
    Ok(())
}

/// Parse `collection = "name"`.
fn parse_struct_collection(
    meta: &syn::meta::ParseNestedMeta<'_>,
    acc: &mut StructAttrs,
) -> syn::Result<()> {
    if acc.collection.is_some() {
        return Err(meta.error("`collection` declared twice"));
    }
    let value = meta.value()?;
    let lit: LitStr = value.parse()?;
    let s = lit.value();
    if s.is_empty() {
        return Err(syn::Error::new(
            lit.span(),
            "collection name must not be empty",
        ));
    }
    acc.collection = Some(s);
    Ok(())
}

/// Parse `index_composite(fields = ("a", "b"), name = "by_a_b")`.
///
/// `fields` is required. `name` is optional and defaults to the
/// fields joined with `__`. Field-existence validation runs after
/// the struct's named fields are known — see
/// `validate_and_emit_composites`.
fn parse_index_composite(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<CompositeAttr> {
    let span = meta.path.span();
    let mut fields: Option<Vec<String>> = None;
    let mut custom_name: Option<String> = None;
    meta.parse_nested_meta(|inner| {
        if inner.path.is_ident("fields") {
            if fields.is_some() {
                return Err(inner.error("`fields` declared twice"));
            }
            fields = Some(parse_composite_fields(&inner)?);
            return Ok(());
        }
        if inner.path.is_ident("name") {
            if custom_name.is_some() {
                return Err(inner.error("`name` declared twice"));
            }
            let value = inner.value()?;
            let lit: LitStr = value.parse()?;
            let s = lit.value();
            if s.is_empty() {
                return Err(syn::Error::new(
                    lit.span(),
                    "composite index name must not be empty",
                ));
            }
            custom_name = Some(s);
            return Ok(());
        }
        Err(inner.error("expected `fields = (...)` or `name = \"...\"`"))
    })?;
    let fields = fields.ok_or_else(|| {
        syn::Error::new(
            span,
            "index_composite requires `fields = (\"a\", \"b\", ...)`",
        )
    })?;
    Ok(CompositeAttr {
        fields,
        custom_name,
        span,
    })
}

/// Parse the `fields = ("a", "b", ...)` parenthesised tuple of
/// string literals. Returns the literal values verbatim.
///
/// Delegates to [`parse_composite_paren_paths`] so the long-form
/// (`index_composite(fields = (...))`) and short-form
/// (`index = (...)`) syntaxes go through one shared parser.
fn parse_composite_fields(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<Vec<String>> {
    let value = meta.value()?;
    parse_composite_paren_paths(value)
}

/// Index-kind discriminator parsed from `#[obj(index = ...)]` or
/// `#[obj(index_composite(...))]`.
#[derive(Debug, Clone)]
enum IndexKind {
    Standard,
    Unique,
    Each,
    /// Composite over the listed field paths (always ≥ 2).
    Composite(Vec<String>),
}

/// One index emitted by the derive — carries the kind discriminator
/// and the (key path, index name) pair to render.
#[derive(Debug)]
struct IndexSpecEmit {
    kind: IndexKind,
    /// The single struct field this index reads from (Standard /
    /// Unique / Each). Unused for `Composite` — paths live inside
    /// `IndexKind::Composite(...)`.
    field_name: String,
    /// User override via `#[obj(index, name = "...")]` or
    /// `index_composite(name = "...")`, or the default name if none
    /// was provided.
    index_name: String,
}

impl IndexSpecEmit {
    /// Emit the constructor call for this spec.
    ///
    /// We route through the kind-specific `IndexSpec` constructors
    /// (`IndexSpec::standard` / `::unique` / `::each` / `::composite`)
    /// rather than a struct literal: `IndexSpec` is `#[non_exhaustive]`
    /// and so cannot be struct-literal-constructed from a downstream
    /// user crate. The constructors return `Result`, but the derive
    /// has already validated their inputs at proc-macro time (empty
    /// struct field names are syntactically impossible, empty
    /// `name = "..."` is rejected at parse time, and composites are
    /// checked for ≥ 2 fields). The emitted code therefore handles the
    /// (statically-unreachable) error arm by skipping rather than
    /// panicking — keeping the generated `indexes()` panic-free.
    fn emit(&self) -> proc_macro2::TokenStream {
        let name = &self.index_name;
        match &self.kind {
            IndexKind::Standard => self.emit_scalar(name, &quote! { standard }),
            IndexKind::Unique => self.emit_scalar(name, &quote! { unique }),
            IndexKind::Each => self.emit_scalar(name, &quote! { each }),
            IndexKind::Composite(paths) => Self::emit_composite(name, paths),
        }
    }

    fn emit_scalar(&self, name: &str, ctor: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let path = &self.field_name;
        quote! {
            ::obj::IndexSpec::#ctor(
                ::std::string::String::from(#name),
                ::std::string::String::from(#path),
            )
        }
    }

    fn emit_composite(name: &str, paths: &[String]) -> proc_macro2::TokenStream {
        let path_tokens = paths.iter().map(|p| quote! { #p });
        quote! {
            ::obj::IndexSpec::composite(
                ::std::string::String::from(#name),
                &[ #( #path_tokens ),* ],
            )
        }
    }
}

/// Iterate the struct's fields and collect every field-level
/// `#[obj(index ...)]` declaration in declaration order.
fn collect_field_indexes(input: &DeriveInput) -> syn::Result<Vec<IndexSpecEmit>> {
    let fields = named_fields(input)?;
    let mut out: Vec<IndexSpecEmit> = Vec::new();
    for field in fields {
        for spec in parse_field_attrs(field)? {
            out.push(spec);
        }
    }
    Ok(out)
}

/// Extract `&FieldsNamed` from the `DeriveInput`. The derive is
/// defined only for braced structs; anything else is a compile
/// error at the struct's span.
fn named_fields(
    input: &DeriveInput,
) -> syn::Result<&syn::punctuated::Punctuated<Field, syn::Token![,]>> {
    match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(named),
            ..
        }) => Ok(&named.named),
        _ => Err(syn::Error::new(
            input.span(),
            "#[derive(obj::Document)] only supports structs with named fields",
        )),
    }
}

/// Parse all `#[obj(...)]` attributes on a single field. Returns the
/// list of `IndexSpecEmit`s contributed by this field (typically 0 or
/// 1, but multiple `#[obj(index ...)]` attributes compose).
fn parse_field_attrs(field: &Field) -> syn::Result<Vec<IndexSpecEmit>> {
    let mut specs: Vec<IndexSpecEmit> = Vec::new();
    let field_name = field
        .ident
        .as_ref()
        .ok_or_else(|| syn::Error::new(field.span(), "expected named field"))?
        .to_string();
    for attr in &field.attrs {
        if !attr.path().is_ident("obj") {
            continue;
        }
        parse_one_field_attr(attr, field, &field_name, &mut specs)?;
    }
    Ok(specs)
}

/// Parse a single `#[obj(...)]` field attribute, contributing any
/// `IndexSpecEmit` it declares into `specs`.
fn parse_one_field_attr(
    attr: &Attribute,
    field: &Field,
    field_name: &str,
    specs: &mut Vec<IndexSpecEmit>,
) -> syn::Result<()> {
    let mut kind: Option<IndexKind> = None;
    let mut custom_name: Option<String> = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("index") {
            if kind.is_some() {
                return Err(meta.error("`index` declared twice on the same field"));
            }
            let parsed = parse_index_kind(&meta)?;
            if matches!(parsed, IndexKind::Composite(_)) {
                return Err(syn::Error::new(
                    meta.path.span(),
                    "tuple-form `index = (\"a\", \"b\")` is struct-level only; \
                     place it directly above the struct, not on a field",
                ));
            }
            kind = Some(parsed);
            return Ok(());
        }
        if meta.path.is_ident("name") {
            if custom_name.is_some() {
                return Err(meta.error("`name` declared twice on the same field"));
            }
            let value = meta.value()?;
            let lit: LitStr = value.parse()?;
            let s = lit.value();
            if s.is_empty() {
                return Err(syn::Error::new(lit.span(), "index name must not be empty"));
            }
            custom_name = Some(s);
            return Ok(());
        }
        if meta.path.is_ident("default") || meta.path.is_ident("default_with") {
            return skip_unrelated_field_meta(&meta);
        }
        Err(meta.error(
            "unknown obj field attribute (expected `index`, `name`, `default`, or `default_with`)",
        ))
    })?;
    finalize_field_index(field, field_name, kind, custom_name, specs)
}

/// Decode the right-hand side of `index = ...` into an `IndexKind`.
///
/// Three syntactic shapes are accepted:
///
/// - `#[obj(index)]` (no `= ...`) → [`IndexKind::Standard`].
/// - `#[obj(index = unique)]` / `#[obj(index = each)]` → the keyword
///   variants. Field-level only; the struct-level caller rejects
///   them with a more specific diagnostic.
/// - `#[obj(index = ("a", "b", ...))]` → [`IndexKind::Composite`]
///   over the listed field-name string literals. Struct-level only;
///   the field-level caller rejects it for the same reason.
///
/// Single-element tuples (`("a",)`) are accepted and degenerate to
/// `IndexKind::Composite(vec!["a".into()])`. The struct-level caller
/// then runs the same field-existence + ≥-2 validation it runs for
/// the long form, so a one-element short tuple produces the existing
/// "composite needs ≥ 2 fields" diagnostic without bespoke handling.
fn parse_index_kind(meta: &syn::meta::ParseNestedMeta<'_>) -> syn::Result<IndexKind> {
    if !meta.input.peek(syn::Token![=]) {
        return Ok(IndexKind::Standard);
    }
    let value = meta.value()?;
    if value.peek(syn::token::Paren) {
        let paths = parse_composite_paren_paths(value)?;
        return Ok(IndexKind::Composite(paths));
    }
    let id: syn::Ident = value.parse().map_err(|_| {
        syn::Error::new(
            value.span(),
            "expected one of: unique, each, or a tuple of field-name string literals like (\"a\", \"b\")",
        )
    })?;
    if id == "unique" {
        return Ok(IndexKind::Unique);
    }
    if id == "each" {
        return Ok(IndexKind::Each);
    }
    Err(syn::Error::new(
        id.span(),
        "expected one of: unique, each (or omit `= ...` for a standard index)",
    ))
}

/// Parse a parenthesised tuple of string literals into a `Vec<String>`.
///
/// Shared by the short composite form `index = ("a", "b")`; the long
/// form `index_composite(fields = ("a", "b"))` runs through
/// [`parse_composite_fields`] which wraps an outer
/// `meta.value()` call before delegating here.
///
/// Non-`LitStr` entries (`(1, 2)`, `(foo, bar)`, …) produce a
/// `syn::Error` pointing at the offending token with the message
/// `expected a tuple of field-name string literals, e.g. ("a", "b")`,
/// rather than the bare `expected string literal` diagnostic that
/// `LitStr::parse` would otherwise emit.
fn parse_composite_paren_paths(value: syn::parse::ParseStream<'_>) -> syn::Result<Vec<String>> {
    const MAX_FIELDS: usize = 64;
    let content;
    syn::parenthesized!(content in value);
    if content.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "expected a tuple of field-name string literals, e.g. (\"a\", \"b\")",
        ));
    }
    let mut out: Vec<String> = Vec::new();
    while !content.is_empty() {
        if out.len() >= MAX_FIELDS {
            return Err(syn::Error::new(
                content.span(),
                "too many composite-index fields (limit 64)",
            ));
        }
        let lit: LitStr = content.parse().map_err(|e| {
            syn::Error::new(
                e.span(),
                "expected a tuple of field-name string literals, e.g. (\"a\", \"b\")",
            )
        })?;
        let s = lit.value();
        if s.is_empty() {
            return Err(syn::Error::new(
                lit.span(),
                "composite field name must not be empty",
            ));
        }
        out.push(s);
        if content.is_empty() {
            break;
        }
        content.parse::<syn::Token![,]>()?;
    }
    Ok(out)
}

/// Combine the parsed `kind` + `custom_name` into an `IndexSpecEmit`.
/// Enforces the `each` ⇒ `Vec<_>` invariant.
fn finalize_field_index(
    field: &Field,
    field_name: &str,
    kind: Option<IndexKind>,
    custom_name: Option<String>,
    specs: &mut Vec<IndexSpecEmit>,
) -> syn::Result<()> {
    let Some(kind) = kind else {
        if custom_name.is_some() {
            return Err(syn::Error::new(
                field.span(),
                "`#[obj(name = \"...\")]` requires an `index` declaration on the same field",
            ));
        }
        return Ok(());
    };
    if matches!(kind, IndexKind::Each) && !type_is_vec(&field.ty) {
        return Err(syn::Error::new(
            field.ty.span(),
            "#[obj(index = each)] requires Vec<T>",
        ));
    }
    specs.push(IndexSpecEmit {
        kind,
        field_name: field_name.to_owned(),
        index_name: custom_name.unwrap_or_else(|| field_name.to_owned()),
    });
    Ok(())
}

/// Cheap syntactic check: is `ty` a `Vec<...>`?
///
/// We accept any path whose last segment ident is `Vec`. That covers
/// `Vec`, `::std::vec::Vec`, `alloc::vec::Vec` etc. Anything else
/// (including `Option<Vec<T>>` or a typedef) is rejected — the user
/// can use `#[obj(index)]` instead.
fn type_is_vec(ty: &Type) -> bool {
    let Type::Path(TypePath { qself: None, path }) = ty else {
        return false;
    };
    match path.segments.last() {
        Some(seg) => seg.ident == "Vec",
        None => false,
    }
}

#[cfg(test)]
mod tests {
    //! Internal proc-macro tests. Tests live next to the derive's
    //! helpers so they can exercise the emit pipeline without
    //! depending on the `obj` crate (which would be a cycle, since
    //! `obj` depends on `obj-derive`).

    use super::*;
    use syn::parse_str;

    /// Expanded code for a typical struct must stay under ~200 lines.
    ///
    /// "Typical struct": 5 fields + 3 indexes.
    /// We pick a shape representative of an everyday user document:
    /// two scalar fields, one unique index, one each-index over a
    /// `Vec`, one composite spanning two of the scalars.
    #[test]
    fn typical_struct_expansion_is_under_200_lines() {
        let input: DeriveInput = parse_str(
            r#"
            #[obj(version = 2, collection = "orders")]
            #[obj(index_composite(fields = ("customer_id", "placed_at")))]
            struct Order {
                #[obj(index)]
                customer_id: u64,
                #[obj(index = unique)]
                order_no: String,
                #[obj(index = each)]
                tags: Vec<String>,
                placed_at: u64,
                total_cents: u64,
            }
            "#,
        )
        .expect("parse typical struct");
        let ts = emit_impl(&input).expect("emit");
        let expanded = ts.to_string();
        let line_count = expanded.lines().count();
        let approx_lines = expanded.matches(';').count()
            + expanded.matches('{').count()
            + expanded.matches('}').count();
        assert!(
            approx_lines <= 200,
            "expanded `#[derive(Document)]` exceeds 200-line budget: \
             approx_lines = {approx_lines}; line_count = {line_count}; \
             expansion = {expanded}",
        );
    }

    /// Normalise a token stream to a whitespace-free string so the
    /// `Option<T>` lowering can be asserted structurally without
    /// depending on `proc_macro2`'s inter-token spacing.
    fn squash(ty: &str) -> String {
        let parsed: Type = parse_str(ty).expect("parse type");
        field_type_to_schema(&parsed)
            .to_string()
            .split_whitespace()
            .collect()
    }

    /// `Option<scalar>` lowers to the two-variant enum syntactically:
    /// the inner scalar (`u64`) does NOT implement `Schema`, yet the
    /// field must compile. The emitted
    /// structure mirrors the `Option<T>: Schema` blanket exactly:
    /// discriminant 0 = `None` (Null payload), 1 = `Some` (lowered
    /// inner).
    #[test]
    fn option_scalar_lowers_to_two_variant_enum() {
        let expected_some_u64 = squash("Option < u64 >");
        assert!(
            expected_some_u64.contains("::obj::DynamicSchema::enumeration"),
            "Option<u64> must lower to an enumeration: {expected_some_u64}",
        );
        assert!(
            expected_some_u64.contains(
                r#"::obj::EnumVariantSchema::new(0u32,"None",::obj::DynamicSchema::Null)"#
            ) || expected_some_u64
                .contains(r#"::obj::EnumVariantSchema::new(0,"None",::obj::DynamicSchema::Null)"#),
            "Option<u64> None arm wrong: {expected_some_u64}",
        );
        assert!(
            expected_some_u64.contains(r#""Some",::obj::DynamicSchema::U64)"#),
            "Option<u64> Some payload must be U64 (not a Schema delegation): {expected_some_u64}",
        );
        assert!(
            !expected_some_u64.contains("u64asobj::Schema")
                && !expected_some_u64.contains("u64as::obj::Schema"),
            "Option<u64> must NOT delegate to <u64 as Schema>: {expected_some_u64}",
        );
    }

    /// `Option<String>` Some-payload lowers to the `String` scalar.
    #[test]
    fn option_string_lowers_some_payload_to_string() {
        let s = squash("Option < String >");
        assert!(
            s.contains(r#""Some",::obj::DynamicSchema::String)"#),
            "Option<String> Some payload must be String: {s}",
        );
    }

    /// `Option<NestedStruct>` still routes its Some payload through the
    /// `<T as Schema>::schema()` fallthrough — byte-identical to what
    /// the `Option<T>: Schema` blanket produces (no regression for
    /// nested-struct Option fields).
    #[test]
    fn option_nested_struct_some_payload_delegates_to_schema() {
        let s = squash("Option < Nested >");
        assert!(
            s.contains("::obj::DynamicSchema::enumeration"),
            "Option<Nested> still lowers to an enumeration: {s}",
        );
        assert!(
            s.contains(r#""Some",<Nestedas::obj::Schema>::schema())"#),
            "Option<Nested> Some payload must delegate to <Nested as Schema>: {s}",
        );
    }

    /// Nestings thread through the recursion: `Option<Vec<u64>>`,
    /// `Vec<Option<u64>>`, and `Option<Option<u64>>` all lower without
    /// touching the scalar-`Schema` fallthrough.
    #[test]
    fn option_nestings_recurse() {
        let opt_vec = squash("Option < Vec < u64 > >");
        assert!(
            opt_vec.contains(r#""Some",::obj::DynamicSchema::seq(::obj::DynamicSchema::U64))"#),
            "Option<Vec<u64>> Some payload must be seq(U64): {opt_vec}",
        );
        let vec_opt = squash("Vec < Option < u64 > >");
        assert!(
            vec_opt.contains("::obj::DynamicSchema::seq(::obj::DynamicSchema::enumeration"),
            "Vec<Option<u64>> must be seq of the Option enum: {vec_opt}",
        );
        let opt_opt = squash("Option < Option < u64 > >");
        assert_eq!(
            opt_opt.matches("::obj::DynamicSchema::enumeration").count(),
            2,
            "Option<Option<u64>> must nest two enumerations: {opt_opt}",
        );
        assert!(
            opt_opt.contains(r#""Some",::obj::DynamicSchema::U64)"#),
            "Option<Option<u64>> innermost Some must be U64: {opt_opt}",
        );
    }

    /// Bare-derive shape: emitted impl carries only COLLECTION +
    /// VERSION + the `// auto-generated` marker. Confirms the
    /// "minimal output" promise.
    #[test]
    fn bare_derive_expansion_is_small() {
        let input: DeriveInput = parse_str("struct Bare { x: u32 }").expect("parse");
        let ts = emit_impl(&input).expect("emit");
        let expanded = ts.to_string();
        assert!(
            !expanded.contains("fn indexes"),
            "bare derive must NOT emit an indexes() override (expanded: {expanded})",
        );
        assert!(expanded.contains("COLLECTION"));
        assert!(expanded.contains("VERSION"));
    }
}
