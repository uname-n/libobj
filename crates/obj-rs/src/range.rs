//! Ergonomic range bounds for the index range query APIs.
//!
//! Point lookups (`find_unique` / `lookup`) accept any
//! `impl Into<`[`Dynamic`]`>`, so a scalar key is written bare:
//! `find_unique("order_no", "A-100")`. The range APIs
//! ([`crate::Query::index_range`], [`crate::Collection::index_range`],
//! [`crate::Collection::iter_range`],
//! [`crate::Collection::count_index_range`],
//! [`crate::Collection::count_distinct_ids_in_range`], and their async
//! mirror) used to force every bound through an explicit
//! `Dynamic::U64(..)` wrapper because they took `R: RangeBounds<Dynamic>`.
//!
//! [`DynamicRange`] removes that asymmetry: any standard range whose
//! endpoints are `impl Into<`[`Dynamic`]`>` is accepted, so
//! `.index_range("placed_at", 40u64..60)` and
//! `.index_range("email", "a".."z")` work directly. Bare-[`Dynamic`]
//! ranges (`lo..hi` where `lo`/`hi` are [`Dynamic`]) keep compiling
//! because [`Dynamic`] satisfies `Into<Dynamic>` via the reflexive
//! `From` impl.
//!
//! The trait is implemented for each concrete std range type rather
//! than blanket-over-`RangeBounds<D>` so that the type-less
//! [`RangeFull`] (`..`) still resolves — a blanket impl would leave the
//! element type `D` uninferrable for `..`.

use std::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use obj_core::codec::Dynamic;

/// Private sealing supertrait: keeps [`DynamicRange`] closed to the impls
/// in this module so new trait methods or impls can be added without a
/// breaking change. Downstream crates cannot name or implement it.
mod sealed {
    pub trait Sealed {}
}

/// A range of index-key values, accepted by the range query APIs.
///
/// Implemented for every standard range type
/// ([`Range`], [`RangeInclusive`], [`RangeFrom`], [`RangeTo`],
/// [`RangeToInclusive`], [`RangeFull`]) whose endpoints are
/// `impl Into<`[`Dynamic`]`>`, plus the `(Bound, Bound)` tuple form. The
/// endpoints are converted to owned [`Dynamic`] values up front; the
/// caller-facing API then runs them through the order-preserving field
/// encoder.
///
/// This trait is **sealed**: it is not implementable by downstream crates.
/// For any range the std sugar doesn't cover, use the general-purpose
/// `(Bound<D>, Bound<D>)` tuple form, which can express any combination of
/// included, excluded, and unbounded endpoints.
pub trait DynamicRange: sealed::Sealed {
    /// Lower the range into an owned `(start, end)` pair of
    /// [`Dynamic`] bounds.
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>);
}

/// Map an owned `Bound<D>` into a `Bound<Dynamic>` by running the
/// endpoint (if any) through its `Into<Dynamic>` conversion.
fn map_bound<D: Into<Dynamic>>(b: Bound<D>) -> Bound<Dynamic> {
    match b {
        Bound::Included(v) => Bound::Included(v.into()),
        Bound::Excluded(v) => Bound::Excluded(v.into()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for Range<D> {}
impl<D: Into<Dynamic>> DynamicRange for Range<D> {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (
            Bound::Included(self.start.into()),
            Bound::Excluded(self.end.into()),
        )
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for RangeInclusive<D> {}
impl<D: Into<Dynamic>> DynamicRange for RangeInclusive<D> {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        let (start, end) = self.into_inner();
        (Bound::Included(start.into()), Bound::Included(end.into()))
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for RangeFrom<D> {}
impl<D: Into<Dynamic>> DynamicRange for RangeFrom<D> {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (Bound::Included(self.start.into()), Bound::Unbounded)
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for RangeTo<D> {}
impl<D: Into<Dynamic>> DynamicRange for RangeTo<D> {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (Bound::Unbounded, Bound::Excluded(self.end.into()))
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for RangeToInclusive<D> {}
impl<D: Into<Dynamic>> DynamicRange for RangeToInclusive<D> {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (Bound::Unbounded, Bound::Included(self.end.into()))
    }
}

impl sealed::Sealed for RangeFull {}
impl DynamicRange for RangeFull {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (Bound::Unbounded, Bound::Unbounded)
    }
}

impl<D: Into<Dynamic>> sealed::Sealed for (Bound<D>, Bound<D>) {}
impl<D: Into<Dynamic>> DynamicRange for (Bound<D>, Bound<D>) {
    fn into_dynamic_bounds(self) -> (Bound<Dynamic>, Bound<Dynamic>) {
        (map_bound(self.0), map_bound(self.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_range_lowers_to_dynamic() {
        let (s, e) = (40u64..60).into_dynamic_bounds();
        assert_eq!(s, Bound::Included(Dynamic::U64(40)));
        assert_eq!(e, Bound::Excluded(Dynamic::U64(60)));
    }

    #[test]
    fn inclusive_scalar_range() {
        let (s, e) = (1i32..=3).into_dynamic_bounds();
        assert_eq!(s, Bound::Included(Dynamic::I64(1)));
        assert_eq!(e, Bound::Included(Dynamic::I64(3)));
    }

    #[test]
    fn str_range_lowers_to_dynamic_string() {
        let (s, e) = ("a".."z").into_dynamic_bounds();
        assert_eq!(s, Bound::Included(Dynamic::String("a".to_owned())));
        assert_eq!(e, Bound::Excluded(Dynamic::String("z".to_owned())));
    }

    #[test]
    fn open_ended_ranges() {
        let (s, e) = (10u64..).into_dynamic_bounds();
        assert_eq!(s, Bound::Included(Dynamic::U64(10)));
        assert_eq!(e, Bound::Unbounded);

        let (s, e) = (..10u64).into_dynamic_bounds();
        assert_eq!(s, Bound::Unbounded);
        assert_eq!(e, Bound::Excluded(Dynamic::U64(10)));

        let (s, e) = (..=10u64).into_dynamic_bounds();
        assert_eq!(s, Bound::Unbounded);
        assert_eq!(e, Bound::Included(Dynamic::U64(10)));
    }

    #[test]
    fn range_full_is_doubly_unbounded() {
        let (s, e) = DynamicRange::into_dynamic_bounds(..);
        assert_eq!(s, Bound::Unbounded);
        assert_eq!(e, Bound::Unbounded);
    }

    #[test]
    fn dynamic_typed_range_still_works() {
        let lo = Dynamic::U64(5);
        let hi = Dynamic::U64(9);
        let (s, e) = (lo..hi).into_dynamic_bounds();
        assert_eq!(s, Bound::Included(Dynamic::U64(5)));
        assert_eq!(e, Bound::Excluded(Dynamic::U64(9)));
    }

    #[test]
    fn bound_tuple_form() {
        let r = (Bound::Excluded(3u64), Bound::Included(7u64));
        let (s, e) = r.into_dynamic_bounds();
        assert_eq!(s, Bound::Excluded(Dynamic::U64(3)));
        assert_eq!(e, Bound::Included(Dynamic::U64(7)));
    }
}
