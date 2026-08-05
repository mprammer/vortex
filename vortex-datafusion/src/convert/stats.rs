// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::ColumnStatistics;
use datafusion_common::stats::Precision;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

/// Convert a stats set for an array with the given dtype.
pub(crate) fn stats_set_to_df(
    stats_set: &StatsSet,
    dtype: &DType,
    row_count: &VortexPrecision<u64>,
) -> VortexResult<ColumnStatistics> {
    let compatible_extrema = stats_set.is_nan_free(dtype);

    // Update the total size in bytes.
    let column_size = u64_stat_as_usize(stats_set, Stat::UncompressedSizeInBytes);

    let (min, max) = if compatible_extrema {
        (
            scalar_stat_to_df(stats_set, Stat::Min, dtype),
            scalar_stat_to_df(stats_set, Stat::Max, dtype),
        )
    } else {
        (VortexPrecision::Absent, VortexPrecision::Absent)
    };
    // Vortex accumulates floats sequentially while Arrow uses a lane reduction. Even without NaNs,
    // IEEE addition is not associative, so a Vortex footer sum is not an exact DataFusion sum.
    let sum = if dtype.may_contain_nan() {
        VortexPrecision::Absent
    } else {
        scalar_stat_to_df(stats_set, Stat::Sum, dtype)
    };

    let null_count = u64_stat_as_usize(stats_set, Stat::NullCount);
    let row_count = u64_precision_as_usize(row_count);

    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: min.to_df(),
        max_value: max.to_df(),
        sum_value: sum.to_df(),
        distinct_count: is_constant_to_distinct_count(
            bool_stat(stats_set, Stat::IsConstant),
            &null_count,
            &row_count,
        ),
        byte_size: column_size.to_df(),
    })
}

/// A constant column has one distinct value only if it is non-empty and has no nulls.
///
/// `COUNT(DISTINCT x)` excludes nulls, while Vortex's `IsConstant` is satisfied by an all-null
/// column just as much as by a column of one repeated value. Reporting `Exact(1)` for the former
/// lets DataFusion fold `COUNT(DISTINCT x)` to 1 where the answer is 0.
///
/// An empty column also has zero distinct values. Although Vortex does not normally record
/// `IsConstant` for an empty array, footer statistics are treated as untrusted input here.
pub(crate) fn is_constant_to_distinct_count(
    is_constant: VortexPrecision<bool>,
    null_count: &VortexPrecision<usize>,
    row_count: &VortexPrecision<usize>,
) -> Precision<usize> {
    match (is_constant.as_exact(), null_count, row_count) {
        (Some(true), VortexPrecision::Exact(0), VortexPrecision::Exact(rows)) if *rows > 0 => {
            Precision::Exact(1)
        }
        _ => Precision::Absent,
    }
}

/// Reads a boolean statistic without the panic path in [`StatsSet::get_as`].
///
/// A malformed scalar in a file footer must not abort planning.
pub(crate) fn bool_stat(stats_set: &StatsSet, stat: Stat) -> VortexPrecision<bool> {
    stats_set.get(stat).and_then(|value| {
        Scalar::try_new(DType::Bool(Nullability::NonNullable), Some(value))
            .inspect_err(|error| {
                tracing::warn!(
                    %stat,
                    dtype = %DType::Bool(Nullability::NonNullable),
                    %error,
                    "ignoring malformed Vortex statistic"
                );
            })
            .ok()?
            .as_bool_opt()?
            .value()
    })
}

/// Reads a `u64`-valued statistic as a `usize`, yielding `Absent` rather than panicking.
///
/// `StatsSet::get_as` panics when the stored value has the wrong type or does not fit the requested
/// type. A malformed value, or a legitimate value above `u32::MAX` on a 32-bit target, must instead
/// cost only the optimisation.
pub(crate) fn u64_stat_as_usize(stats_set: &StatsSet, stat: Stat) -> VortexPrecision<usize> {
    let raw = stats_set.get(stat).and_then(|value| {
        Scalar::try_new(PType::U64.into(), Some(value))
            .inspect_err(|error| {
                tracing::warn!(
                    %stat,
                    dtype = %DType::from(PType::U64),
                    %error,
                    "ignoring malformed Vortex statistic"
                );
            })
            .ok()?
            .as_primitive_opt()?
            .as_opt::<u64>()?
    });
    u64_precision_as_usize(&raw)
}

/// Converts a `u64` precision to `usize`, withholding values that do not fit.
pub(crate) fn u64_precision_as_usize(value: &VortexPrecision<u64>) -> VortexPrecision<usize> {
    match value {
        VortexPrecision::Exact(v) => {
            usize::try_from(*v).map_or(VortexPrecision::Absent, VortexPrecision::Exact)
        }
        VortexPrecision::Inexact(v) => {
            usize::try_from(*v).map_or(VortexPrecision::Absent, VortexPrecision::Inexact)
        }
        VortexPrecision::Absent => VortexPrecision::Absent,
    }
}

/// Converts a scalar-valued statistic to DataFusion's representation, or reports none.
///
/// This yields nothing rather than failing a query when:
///
/// - the statistic has no dtype for this column (`Stat::Min` over `DType::Null`);
/// - the recorded value is not valid for that dtype;
/// - DataFusion has no representation for the resulting scalar.
///
/// Footer statistics are metadata, so any disagreement costs the optimisation rather than the
/// query.
fn scalar_stat_to_df(
    stats_set: &StatsSet,
    stat: Stat,
    dtype: &DType,
) -> VortexPrecision<datafusion_common::ScalarValue> {
    stats_set.get(stat).and_then(|value| {
        let stat_dtype = stat.dtype(dtype)?;
        let scalar = Scalar::try_new(stat_dtype.clone(), Some(value))
            .inspect_err(|error| {
                tracing::warn!(
                    %stat,
                    dtype = %stat_dtype,
                    %error,
                    "ignoring malformed Vortex statistic"
                );
            })
            .ok()?;
        scalar
            .try_to_df()
            .inspect_err(|error| {
                tracing::warn!(
                    %stat,
                    dtype = %stat_dtype,
                    %error,
                    "ignoring Vortex statistic that DataFusion cannot represent"
                );
            })
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use vortex::expr::stats::Precision as VortexPrecision;
    use vortex::scalar::ScalarValue;

    use super::*;

    /// A constant column has one distinct value only when it is provably non-empty and null-free.
    ///
    /// This test previously asserted `Exact(1)` for a constant column with *no recorded null
    /// count*, which is unsound: an all-null column satisfies `IsConstant`, and `COUNT(DISTINCT x)`
    /// excludes nulls, so the true answer there is 0. DataFusion folds the aggregate from this
    /// value, so the old behaviour could return 1 for a query whose answer is 0.
    #[test]
    fn one_distinct_value_requires_a_non_empty_null_free_column() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let one_row = VortexPrecision::exact(1u64);

        // Not constant: nothing to say.
        let mut not_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(false));
        not_constant.set(
            Stat::NullCount,
            VortexPrecision::exact(ScalarValue::from(0u64)),
        );
        assert_eq!(
            stats_set_to_df(&not_constant, &dtype, &one_row)?.distinct_count,
            Precision::Absent
        );

        // Constant and provably null-free: exactly one distinct value.
        let mut constant_no_nulls = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
        constant_no_nulls.set(
            Stat::NullCount,
            VortexPrecision::exact(ScalarValue::from(0u64)),
        );
        assert_eq!(
            stats_set_to_df(&constant_no_nulls, &dtype, &one_row)?.distinct_count,
            Precision::Exact(1)
        );

        // Empty: even contradictory footer metadata cannot prove one distinct value.
        assert_eq!(
            stats_set_to_df(&constant_no_nulls, &dtype, &VortexPrecision::exact(0u64),)?
                .distinct_count,
            Precision::Absent
        );

        // Unknown row count: non-emptiness is not proven.
        assert_eq!(
            stats_set_to_df(&constant_no_nulls, &dtype, &VortexPrecision::Absent)?.distinct_count,
            Precision::Absent
        );

        // Constant with nulls present: could be all-null, so nothing may be claimed.
        let mut constant_with_nulls = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
        constant_with_nulls.set(
            Stat::NullCount,
            VortexPrecision::exact(ScalarValue::from(3u64)),
        );
        assert_eq!(
            stats_set_to_df(&constant_with_nulls, &dtype, &one_row)?.distinct_count,
            Precision::Absent
        );

        // Constant with an unknown null count: same, and this is the case the old test got wrong.
        let unknown_nulls = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
        assert_eq!(
            stats_set_to_df(&unknown_nulls, &dtype, &one_row)?.distinct_count,
            Precision::Absent
        );

        Ok(())
    }

    /// A `u64` statistic too large for `usize` is absent, never a saturated exact value.
    #[test]
    fn an_oversized_u64_statistic_is_absent_rather_than_saturated() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut huge = StatsSet::of(
            Stat::NullCount,
            VortexPrecision::exact(ScalarValue::from(u64::MAX)),
        );
        huge.set(Stat::IsConstant, VortexPrecision::exact(false));

        let out = stats_set_to_df(&huge, &dtype, &VortexPrecision::exact(1u64))?;
        if usize::try_from(u64::MAX).is_err() {
            assert_eq!(out.null_count, Precision::Absent);
        } else {
            assert_eq!(out.null_count, Precision::Exact(usize::MAX));
        }
        Ok(())
    }

    /// Wrongly typed footer values are absent rather than panic paths.
    #[test]
    fn malformed_fixed_type_statistics_are_absent() {
        let malformed_bool = StatsSet::of(
            Stat::IsConstant,
            VortexPrecision::exact(ScalarValue::from(1u64)),
        );
        assert_eq!(
            bool_stat(&malformed_bool, Stat::IsConstant),
            VortexPrecision::Absent
        );

        let malformed_u64 = StatsSet::of(
            Stat::NullCount,
            VortexPrecision::exact(ScalarValue::from(true)),
        );
        assert_eq!(
            u64_stat_as_usize(&malformed_u64, Stat::NullCount),
            VortexPrecision::Absent
        );
    }

    /// Float bounds are absent unless an exact zero NaN count makes their semantics compatible.
    #[test]
    fn float_bounds_require_proof_that_no_nans_were_skipped() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut stats = StatsSet::of(Stat::Max, VortexPrecision::exact(ScalarValue::from(1.0f64)));
        let row_count = VortexPrecision::exact(1u64);

        assert_eq!(
            stats_set_to_df(&stats, &dtype, &row_count)?.max_value,
            Precision::Absent
        );

        stats.set(
            Stat::NaNCount,
            VortexPrecision::exact(ScalarValue::from(0u64)),
        );
        assert_eq!(
            stats_set_to_df(&stats, &dtype, &row_count)?.max_value,
            Precision::Exact(datafusion_common::ScalarValue::Float64(Some(1.0)))
        );
        Ok(())
    }
}
