// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::stats::bind::StatBinder;
use vortex_array::stats::bind::bind_stats;
use vortex_array::stats::extrema_cast_is_value_preserving;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FileStatistics;

pub(crate) fn can_prune_file_stats(
    expr: &BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    struct_fields: &StructFields,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(pruning_expr) = expr.falsify(session)? else {
        return Ok(false);
    };

    let binder = FileStatsBinder {
        file_stats,
        struct_fields,
    };
    let pruning_expr = bind_stats(pruning_expr, &binder)?;

    if let Some(result) = pruning_expr.as_opt::<Literal>() {
        return Ok(result.as_bool().value() == Some(true));
    }

    let pruning = NullArray::new(1).into_array().apply_bound(&pruning_expr)?;
    let row_count_replacement = ConstantArray::new(row_count, pruning.len()).into_array();
    let pruning = substitute_row_count(pruning, &row_count_replacement)?;

    let mut ctx = session.create_execution_ctx();
    let result = pruning
        .execute::<Canonical>(&mut ctx)?
        .into_bool()
        .into_array()
        .execute_scalar(0, &mut ctx)?;

    Ok(result.as_bool().value() == Some(true))
}

struct FileStatsBinder<'a> {
    file_stats: &'a FileStatistics,
    struct_fields: &'a StructFields,
}

impl StatBinder for FileStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        let Some(field_path) = direct_field_path(input, matches!(stat, Stat::Min | Stat::Max))
        else {
            return Ok(None);
        };
        Ok(self.stat_ref(&field_path, stat))
    }
}

impl FileStatsBinder<'_> {
    fn stat_ref(&self, field_path: &FieldPath, stat: Stat) -> Option<BoundExpression> {
        // FileStats currently only holds top-level field statistics.
        let [field] = field_path.parts() else {
            return None;
        };
        let field_name = field.as_name()?;
        let mut matches = self
            .struct_fields
            .names()
            .iter()
            .zip(self.struct_fields.fields())
            .enumerate()
            .filter(|(_, (name, _))| name.as_ref() == field_name);
        let (field_idx, (_, field_dtype)) = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        let field_stats = self.file_stats.stats_sets().get(field_idx)?;
        if matches!(stat, Stat::Min | Stat::Max | Stat::Sum)
            && !field_stats.is_nan_free(&field_dtype)
        {
            return None;
        }

        let stat_value = field_stats.get(stat).as_exact()?;
        let stat_dtype = stat.dtype(&field_dtype)?;
        let stat_scalar = Scalar::try_new(stat_dtype.clone(), Some(stat_value))
            .inspect_err(|error| {
                tracing::warn!(
                    %stat,
                    dtype = %stat_dtype,
                    %error,
                    "ignoring malformed Vortex statistic"
                );
            })
            .ok()?;

        Some(lit(stat_scalar))
    }
}

fn direct_field_path(
    expr: &BoundExpression,
    require_extrema_safe_casts: bool,
) -> Option<FieldPath> {
    if expr.is_root() {
        return Some(FieldPath::root());
    }

    if expr.is::<Cast>() {
        if require_extrema_safe_casts
            && !extrema_cast_is_value_preserving(expr.child(0).dtype(), expr.dtype())
        {
            return None;
        }
        return direct_field_path(expr.child(0), require_extrema_safe_casts);
    }

    let field_name = expr.as_opt::<GetItem>()?;
    direct_field_path(expr.child(0), require_extrema_safe_casts)
        .map(|path| path.push(field_name.clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::cast;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_array::extension::datetime::TimestampOptions;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::stats::StatsSet;

    use super::*;

    #[test]
    fn duplicate_field_names_disable_public_file_pruning() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let stats = StatsSet::of(Stat::Max, Precision::exact(ScalarValue::from(7i64)));
        let file_stats = FileStatistics::new(
            Arc::from([stats.clone(), stats]),
            Arc::from([dtype.clone(), dtype.clone()]),
        );
        let struct_fields = StructFields::from_iter([("x", dtype.clone()), ("x", dtype)]);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullability::NonNullable);
        let expr = gt(get_item("x", root()), lit(8i64)).bind(&struct_dtype)?;
        let session = vortex_array::array_session();

        assert!(!can_prune_file_stats(
            &expr,
            2,
            &file_stats,
            &struct_fields,
            &session,
        )?);
        Ok(())
    }

    #[test]
    fn public_float_pruning_requires_an_exact_zero_nan_count() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let struct_fields = StructFields::from_iter([("x", dtype.clone())]);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullability::NonNullable);
        let expr = gt(get_item("x", root()), lit(2.0f64)).bind(&struct_dtype)?;
        let session = vortex_array::array_session();
        let mut stats = StatsSet::of(Stat::Max, Precision::exact(ScalarValue::from(1.0f64)));
        stats.set(Stat::NaNCount, Precision::exact(ScalarValue::from(1u64)));
        let file_stats =
            FileStatistics::new(Arc::from([stats.clone()]), Arc::from([dtype.clone()]));
        assert!(!can_prune_file_stats(
            &expr,
            1,
            &file_stats,
            &struct_fields,
            &session,
        )?);

        stats.set(Stat::NaNCount, Precision::exact(ScalarValue::from(0u64)));
        let file_stats = FileStatistics::new(Arc::from([stats]), Arc::from([dtype]));
        assert!(can_prune_file_stats(
            &expr,
            1,
            &file_stats,
            &struct_fields,
            &session,
        )?);
        Ok(())
    }

    #[test]
    fn temporal_unit_casts_do_not_prune_from_rewrapped_extrema() -> VortexResult<()> {
        let source_dtype = DType::Extension(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        );
        let target_dtype =
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased());
        let struct_fields = StructFields::from_iter([("x", source_dtype.clone())]);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullability::NonNullable);
        let target_literal = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: TimeUnit::Seconds,
                tz: None,
            },
            Scalar::from(100i64),
        );
        let expr = lt(
            cast(get_item("x", root()), target_dtype),
            lit(target_literal),
        )
        .bind(&struct_dtype)?;
        let file_stats = FileStatistics::new(
            Arc::from([StatsSet::of(
                Stat::Min,
                Precision::exact(ScalarValue::from(1_500i64)),
            )]),
            Arc::from([source_dtype]),
        );
        let session = vortex_array::array_session();

        // The array cast rescales 1,500ms to 1s, so the row matches `< 100s`. Rewrapping the
        // footer integer as 1,500s would instead (and incorrectly) prove that no row matches.
        assert!(!can_prune_file_stats(
            &expr,
            1,
            &file_stats,
            &struct_fields,
            &session,
        )?);
        Ok(())
    }

    #[test]
    fn decimal_to_integer_casts_do_not_prune_from_rounded_extrema() -> VortexResult<()> {
        let source_dtype = DType::Decimal(DecimalDType::new(38, 0), Nullability::NonNullable);
        let target_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let struct_fields = StructFields::from_iter([("x", source_dtype.clone())]);
        let struct_dtype = DType::Struct(struct_fields.clone(), Nullability::NonNullable);
        let expr = gt(
            cast(get_item("x", root()), target_dtype),
            lit(9_007_199_254_740_992i64),
        )
        .bind(&struct_dtype)?;
        let file_stats = FileStatistics::new(
            Arc::from([StatsSet::of(
                Stat::Max,
                Precision::exact(ScalarValue::Decimal(DecimalValue::I128(
                    9_007_199_254_740_993,
                ))),
            )]),
            Arc::from([source_dtype]),
        );
        let session = vortex_array::array_session();

        // The array cast preserves this i64-representable decimal, while the old scalar path
        // rounded it through f64 to the literal and incorrectly proved `x > literal` impossible.
        assert!(!can_prune_file_stats(
            &expr,
            1,
            &file_stats,
            &struct_fields,
            &session,
        )?);
        Ok(())
    }
}
