// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits and utilities to compute and access array statistics.

use arrow_buffer::BooleanBufferBuilder;
use arrow_buffer::MutableBuffer;
use arrow_buffer::bit_iterator::BitIterator;
use enum_iterator::last;
pub use expr::all_nan;
pub use expr::all_non_nan;
pub use expr::all_non_null;
pub use expr::all_null;
pub use expr::bound;
pub use expr::min_max;
pub use expr::nan_count;
pub use expr::null_count;
pub use expr::stat;
pub use expr::sum;
pub use stats_set::*;

mod array;
pub mod bind;
pub mod expr;
pub mod flatbuffers;
pub mod rewrite;
pub mod session;
mod stats_set;

pub use array::*;
pub use session::*;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::expr::stats::Stat;

/// Return whether adapting exact extrema from `source` to `target` is value-preserving.
///
/// Accepts three shapes, and nothing else:
///
/// - an identical dtype;
/// - the identical dtype made nullable, which only ever admits more values;
/// - widening between primitive integers of the same signedness, where nullability does not
///   strengthen.
///
/// Everything else is refused even when a scalar cast happens to succeed, because the scalar and
/// array conversion paths need not agree. Decimal-to-integer above 2^53 rounds through an f64
/// scalar path while the scan stays exact; a temporal unit change rewraps the stored integer while
/// the scan rescales it; decimal scale changes can overflow, invert bounds, or produce infinities
/// where the array cast yields finite values.
///
/// # Why nullability has a direction
///
/// An earlier version compared with `eq_ignore_nullability`, which is both recursive and
/// *directionless*. That accepted a **nullable source against a non-nullable target** — and those
/// are not interchangeable: casting the array fails on the first null, while the extrema scalars are
/// by construction non-null and cast cleanly. Publishing bounds derived that way describes a scan
/// that would have errored, so the direction is now checked rather than ignored.
pub fn extrema_cast_is_value_preserving(source: &DType, target: &DType) -> bool {
    // Identical, or identical-but-nullable. Compared by full equality so a nullability difference
    // nested inside a struct or list cannot slip through the way it did under
    // `eq_ignore_nullability`.
    if source == target || &source.as_nullable() == target {
        return true;
    }

    let (DType::Primitive(source_ptype, source_null), DType::Primitive(target_ptype, target_null)) =
        (source, target)
    else {
        return false;
    };

    // Nullability must not strengthen here either: a nullable source cast to a non-nullable target
    // fails on nulls at scan time whatever the widths are.
    if source_null.is_nullable() && !target_null.is_nullable() {
        return false;
    }

    let same_integer_family = (source_ptype.is_signed_int() && target_ptype.is_signed_int())
        || (source_ptype.is_unsigned_int() && target_ptype.is_unsigned_int());
    same_integer_family && source_ptype.byte_width() < target_ptype.byte_width()
}

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
/// Sum is included for boolean arrays.
pub const PRUNING_STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::NaNCount,
];

pub fn as_stat_bitset_bytes(stats: &[Stat]) -> Vec<u8> {
    let max_stat = u8::from(last::<Stat>().vortex_expect("last stat")) as usize + 1;
    // TODO(ngates): use vortex-buffer::BitBuffer
    let mut stat_bitset = BooleanBufferBuilder::new_from_buffer(
        MutableBuffer::from_len_zeroed(max_stat.div_ceil(8)),
        max_stat,
    );
    for stat in stats {
        stat_bitset.set_bit(u8::from(*stat) as usize, true);
    }

    stat_bitset
        .finish()
        .into_inner()
        .into_vec()
        .unwrap_or_else(|b| b.to_vec())
}

pub fn stats_from_bitset_bytes(bytes: &[u8]) -> Vec<Stat> {
    BitIterator::new(bytes, 0, bytes.len() * 8)
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        // Filter out indices failing conversion, these are stats written by newer version of library
        .filter_map(|i| {
            let Ok(stat) = u8::try_from(i) else {
                tracing::debug!("invalid stat encountered: {i}");
                return None;
            };
            Stat::try_from(stat).ok()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn primitive(ptype: PType, nullability: Nullability) -> DType {
        DType::Primitive(ptype, nullability)
    }

    #[test]
    fn classifies_extrema_safe_casts() {
        assert!(extrema_cast_is_value_preserving(
            &primitive(PType::I32, Nullability::NonNullable),
            &primitive(PType::I32, Nullability::Nullable),
        ));
        assert!(extrema_cast_is_value_preserving(
            &primitive(PType::I32, Nullability::NonNullable),
            &primitive(PType::I64, Nullability::NonNullable),
        ));
        assert!(extrema_cast_is_value_preserving(
            &primitive(PType::U8, Nullability::NonNullable),
            &primitive(PType::U64, Nullability::NonNullable),
        ));

        assert!(!extrema_cast_is_value_preserving(
            &primitive(PType::I64, Nullability::NonNullable),
            &primitive(PType::I32, Nullability::NonNullable),
        ));
        assert!(!extrema_cast_is_value_preserving(
            &primitive(PType::U64, Nullability::NonNullable),
            &primitive(PType::I64, Nullability::NonNullable),
        ));
        assert!(!extrema_cast_is_value_preserving(
            &primitive(PType::I64, Nullability::NonNullable),
            &primitive(PType::F64, Nullability::NonNullable),
        ));
        assert!(!extrema_cast_is_value_preserving(
            &DType::Decimal(DecimalDType::new(38, 0), Nullability::NonNullable),
            &primitive(PType::I64, Nullability::NonNullable),
        ));
    }

    /// Nullability has a direction: a nullable source must never be adapted to a non-nullable
    /// target.
    ///
    /// The array cast fails on the first null while the extrema scalars, being non-null by
    /// construction, cast cleanly — so accepting this publishes bounds describing a scan that would
    /// have errored. The previous `eq_ignore_nullability` comparison accepted it in both directions.
    #[test]
    fn a_nullable_source_is_never_adapted_to_a_non_nullable_target() {
        let nullable = primitive(PType::I32, Nullability::Nullable);
        let non_nullable = primitive(PType::I32, Nullability::NonNullable);

        assert!(
            !extrema_cast_is_value_preserving(&nullable, &non_nullable),
            "nullable -> non-nullable must be refused"
        );
        assert!(
            extrema_cast_is_value_preserving(&non_nullable, &nullable),
            "non-nullable -> nullable only ever admits more values"
        );

        // The same direction rule applies to the integer-widening arm.
        assert!(!extrema_cast_is_value_preserving(
            &primitive(PType::I32, Nullability::Nullable),
            &primitive(PType::I64, Nullability::NonNullable)
        ));
        assert!(extrema_cast_is_value_preserving(
            &primitive(PType::I32, Nullability::NonNullable),
            &primitive(PType::I64, Nullability::Nullable)
        ));
    }
}
