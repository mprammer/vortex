// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Use [`VortexDataSource`] to adapt an existing Vortex [`DataSourceRef`] into
//! a DataFusion [`DataSource`] without going through file discovery.
//!
//! [`VortexDataSource`] is responsible for:
//!
//! - exposing an Arrow schema and output statistics to DataFusion,
//! - translating DataFusion projection, filter, and limit pushdown into a
//!   Vortex [`ScanRequest`],
//! - executing the Vortex scan and converting the results into Arrow
//!   `RecordBatch` values.
//!
//! # Example: Create a `DataSourceExec`
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use arrow_schema::Schema;
//! use datafusion_datasource::source::DataSourceExec;
//! use vortex::VortexSessionDefault;
//! use vortex::scan::DataSourceRef;
//! use vortex::session::VortexSession;
//! use vortex_datafusion::v2::VortexDataSource;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let data_source: DataSourceRef = todo!();
//! let data_source = VortexDataSource::builder(data_source, VortexSession::default())
//!     .with_arrow_schema(Arc::new(Schema::empty()))
//!     .build()
//!     .await?;
//!
//! let exec = DataSourceExec::from_data_source(data_source);
//! # let _ = exec;
//! # Ok(())
//! # }
//! ```
//!
//! # Execution Flow
//!
//! ```text
//!             ▲
//!             │  RecordBatch stream
//!             │
//! ┌───────────────────────┐
//! │     DataSourceExec    │
//! └───────────────────────┘
//!             ▲
//!             │  DataFusion pushdown
//!             │  (projection/filter/limit)
//! ┌───────────────────────┐
//! │   VortexDataSource    │
//! └───────────────────────┘
//!             ▲
//!             │  final ScanRequest
//! ┌───────────────────────┐
//! │    DataSourceRef      │
//! └───────────────────────┘
//! ```
//!
//! Compared with [`crate::VortexSource`], this path starts from an existing
//! Vortex source rather than from DataFusion-managed file discovery.
//!
//! [`DataSource`]: datafusion_datasource::source::DataSource
//! [`DataSourceRef`]: vortex::scan::DataSourceRef
//! [`ScanRequest`]: vortex::scan::ScanRequest

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::arrow::array::AsArray;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_datasource::source::DataSource;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_expr::type_coercion::functions::fields_with_udf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use vortex::array::VortexSessionExecute;
use vortex::array::stats::extrema_cast_is_value_preserving;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::Expression;
use vortex::expr::and as vx_and;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use crate::PrecisionExt;
use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::stats::stats_set_to_df;
use crate::convert::stats::u64_precision_as_usize;

/// Builder for [`VortexDataSource`].
///
/// Use the builder to declare how an existing Vortex
/// [`DataSourceRef`] should appear to DataFusion.
/// In particular, it lets you choose:
///
/// - the Arrow schema DataFusion should see,
/// - an initial top-level projection if the embedding system already knows
///   which columns are needed.
///
/// The resulting [`VortexDataSource`] is ready to plug into
/// [`DataSourceExec`] or other DataFusion physical planning code.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use vortex::VortexSessionDefault;
/// use vortex::scan::DataSourceRef;
/// use vortex::session::VortexSession;
/// use vortex_datafusion::v2::VortexDataSource;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let data_source: DataSourceRef = todo!();
/// let data_source = VortexDataSource::builder(data_source, VortexSession::default())
///     .with_arrow_schema(Arc::new(Schema::empty()))
///     .with_projection(vec![0])
///     .build()
///     .await?;
/// # let _ = data_source;
/// # Ok(())
/// # }
/// ```
///
/// [`DataSourceRef`]: vortex::scan::DataSourceRef
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
pub struct VortexDataSourceBuilder {
    data_source: DataSourceRef,
    session: VortexSession,

    arrow_schema: Option<SchemaRef>,
    projection: Option<Vec<usize>>,
}

impl VortexDataSourceBuilder {
    /// Sets the Arrow schema exposed to DataFusion.
    ///
    /// If not specified, the builder derives an Arrow schema from the Vortex
    /// dtype.
    ///
    /// Note that this schema is not validated against the Vortex DType so any errors will be
    /// deferred until read time.
    pub fn with_arrow_schema(mut self, arrow_schema: SchemaRef) -> Self {
        self.arrow_schema = Some(arrow_schema);
        self
    }

    /// Configures an initial top-level projection.
    ///
    /// This is useful when the embedding system already knows which columns are
    /// needed before DataFusion applies its own optimizer pushdown.
    pub fn with_projection(mut self, indices: Vec<usize>) -> Self {
        self.projection = Some(indices);
        self
    }

    /// Like [`Self::with_projection`], but accepts an optional projection.
    pub fn with_some_projection(mut self, indices: Option<Vec<usize>>) -> Self {
        self.projection = indices;
        self
    }

    /// Builds the [`VortexDataSource`].
    ///
    /// The builder eagerly resolves statistics for the initial projection
    /// columns because DataFusion expects the `DataSource` to report output
    /// statistics before execution begins.
    pub async fn build(self) -> VortexResult<VortexDataSource> {
        // The projection expression
        let mut projection = root();

        // Resolve the Arrow schema
        let mut arrow_schema = match self.arrow_schema {
            Some(schema) => schema,
            None => Arc::new(
                self.session
                    .arrow()
                    .to_arrow_schema(self.data_source.dtype())?,
            ),
        };

        // Apply any selection and create a projection expression.
        if let Some(indices) = self.projection {
            let fields = indices.iter().map(|&i| {
                let name = arrow_schema.field(i).name().clone();
                let expr = get_item(name.as_str(), root());
                (name, expr)
            });

            // Update the projection expression
            projection = pack(fields, Nullability::NonNullable);

            // Update the arrow schema
            arrow_schema = Arc::new(Schema::new(
                indices
                    .iter()
                    .map(|&i| arrow_schema.field(i).clone())
                    .collect::<Vec<_>>(),
            ));
        }

        let DType::Struct(fields, ..) = projection.return_dtype(self.data_source.dtype())? else {
            vortex_bail!("Projection does not evaluate to a struct");
        };

        // Whether the Arrow schema and the Vortex fields describe the same columns in the same
        // order — checked, rather than assumed or blanket-refused.
        //
        // `with_arrow_schema` is documented as unvalidated against the Vortex dtype, and
        // `vortex-arrow`'s struct executor zips the two *positionally*. A supplied schema of
        // `[b, a]` over a Vortex dtype of `[a, b]` therefore emits Vortex field `a` under Arrow
        // name `b`: data identity is positional while statistics resolve by name, and nothing
        // downstream reconciles that.
        //
        // Refusing whenever a schema was supplied would be sound and useless: `VortexTable::scan`
        // always supplies one, so it would disable footer statistics on the only public path that
        // has them. The supplied schema is virtually always *derived* from this same dtype, in
        // which case both identities agree, so the agreement is verified instead.
        let schema_matches_dtype = schema_names_match(&arrow_schema, &fields);
        let Some(source_fields) = self.data_source.dtype().as_struct_fields_opt() else {
            vortex_bail!("Statistics require a struct data source");
        };
        let field_indices =
            statistics_field_indices(&arrow_schema, &fields, source_fields, &self.session);

        // We now compute initial statistics.
        //
        // Skipped entirely when they could not be published: they are optional metadata, and
        // resolving them is fallible I/O that would otherwise be able to fail a scan for a result
        // about to be discarded.
        let field_paths: Vec<_> = field_indices
            .iter()
            .map(|&idx| FieldPath::from_name(fields.names()[idx].clone()))
            .collect();
        let row_count = self.data_source.row_count();
        let fetched_statistics = try_join_all(
            field_paths
                .iter()
                .map(|path| self.data_source.field_statistics(path)),
        )
        .await?;

        // Start unknown and install only statistics fetched for an unambiguous source field. A
        // duplicate name cannot be resolved by `FieldPath::from_name`, so installing either match
        // positionally would assign one source field's exact value to both output columns.
        let mut statistics = vec![ColumnStatistics::new_unknown(); arrow_schema.fields().len()];
        for (&idx, stats) in field_indices.iter().zip(&fetched_statistics) {
            if let Some(dtype) = fields.field_by_index(idx) {
                statistics[idx] = stats_set_to_df(stats, &dtype, &row_count)?;
            }
        }

        // The scan selects a pushed-down column by name, so retain the statistics and source Arrow
        // type under that identity. Duplicate names are ambiguous and removed.
        //
        // A caller-supplied Arrow schema is unvalidated and is paired with Vortex fields
        // positionally during conversion. When its names do not agree with the projected Vortex
        // fields, `field_indices` is empty and no name-based statistics are published. A matching
        // supplied schema remains usable; `VortexTable::scan` always supplies one.
        let mut statistics_by_name: HashMap<String, NamedColumnStatistics> = HashMap::new();
        let mut ambiguous: HashSet<String> = HashSet::new();
        for &idx in &field_indices {
            let name = fields.names()[idx].to_string();
            if statistics_by_name
                .insert(
                    name.clone(),
                    NamedColumnStatistics {
                        data_type: arrow_schema.field(idx).data_type().clone(),
                        statistics: statistics[idx].clone(),
                    },
                )
                .is_some()
            {
                ambiguous.insert(name);
            }
        }
        for name in &ambiguous {
            statistics_by_name.remove(name);
        }

        // Type-checked here as well as on swap: a source that is never swapped still publishes
        // these, and the Arrow mapping may have chosen a view or wider decimal type than the scalar
        // conversion produced.
        let initial_leftover_statistics: Vec<ColumnStatistics> = if !schema_matches_dtype {
            vec![ColumnStatistics::new_unknown(); arrow_schema.fields().len()]
        } else {
            statistics
                .iter()
                .zip(arrow_schema.fields())
                .map(|(stat, field)| retain_type_compatible(stat, field.data_type()))
                .collect()
        };

        Ok(VortexDataSource {
            data_source: self.data_source,
            session: self.session,
            initial_schema: Arc::clone(&arrow_schema),
            initial_projection: projection.clone(),
            initial_statistics_by_name: Arc::new(statistics_by_name),
            projection_pushdown_phase: ProjectionPushdownPhase::Initial,
            projected_projection: projection.clone(),
            projected_schema: Arc::clone(&arrow_schema),
            leftover_projection: None,
            leftover_schema: arrow_schema,
            leftover_statistics: initial_leftover_statistics,
            filter: None,
            limit: None,
            ordered: false,
            num_partitions: get_available_parallelism().unwrap_or(1),
        })
    }
}

/// Whether the Arrow output names agree positionally with the projected Vortex fields.
fn schema_names_match(schema: &Schema, fields: &StructFields) -> bool {
    schema.fields().len() == fields.nfields()
        && schema
            .fields()
            .iter()
            .zip(fields.names().iter())
            .all(|(arrow_field, vortex_name)| arrow_field.name() == vortex_name.as_ref())
}

/// Return the source-field positions whose footer statistics can be installed safely.
///
/// Statistics are resolved by name but installed by position. The two schemas must therefore
/// agree in order, and a name that occurs more than once in the source must remain unknown because
/// a [`FieldPath`] cannot identify which occurrence supplied the value.
fn statistics_field_indices(
    schema: &Schema,
    fields: &StructFields,
    source_fields: &StructFields,
    session: &VortexSession,
) -> Vec<usize> {
    if !schema_names_match(schema, fields) {
        return Vec::new();
    }

    let mut name_counts = HashMap::with_capacity(source_fields.nfields());
    for name in source_fields.names().iter() {
        *name_counts.entry(name.as_ref()).or_insert(0usize) += 1;
    }

    fields
        .names()
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            if name_counts.get(name.as_ref()).copied() != Some(1) {
                return None;
            }
            let source_dtype = fields.field_by_index(idx)?;
            let target_dtype = session.arrow().from_arrow_field(schema.field(idx)).ok()?;
            extrema_cast_is_value_preserving(&source_dtype, &target_dtype).then_some(idx)
        })
        .collect()
}

impl VortexDataSource {
    /// Create a builder for a [`VortexDataSource`].
    pub fn builder(data_source: DataSourceRef, session: VortexSession) -> VortexDataSourceBuilder {
        VortexDataSourceBuilder {
            data_source,
            session,
            arrow_schema: None,
            projection: None,
        }
    }
}

#[derive(Clone)]
struct NamedColumnStatistics {
    data_type: DataType,
    statistics: ColumnStatistics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectionPushdownPhase {
    Initial,
    Swapped,
}

/// DataFusion [`DataSource`] backed by a Vortex [`DataSourceRef`].
///
/// `VortexDataSource` is the core execution adapter for the `v2` integration.
/// It presents DataFusion with a scanable Arrow data source while preserving the
/// underlying Vortex source until execution time.
///
/// During planning, it reports the current output schema and column statistics.
/// During execution, it builds the final Vortex [`ScanRequest`] from the
/// current projection, pushed filters, ordering hints, and row limit.
///
/// This integration intentionally reports a single DataFusion output partition.
/// Vortex then handles split-level concurrency internally by polling multiple
/// split streams concurrently.
///
/// Use [`crate::VortexSource`] instead when DataFusion should discover and plan
/// `.vortex` files on its own.
#[derive(Clone)]
pub struct VortexDataSource {
    /// The Vortex data source.
    data_source: DataSourceRef,
    /// Vortex session handle.
    session: VortexSession,

    // --- Phase 1: Initial (from the builder, before any optimizer pushdown) ---
    /// The Arrow schema of the data source before any DataFusion projection pushdown.
    initial_schema: SchemaRef,
    /// The initial Vortex projection expression (e.g. column selection from the builder).
    initial_projection: Expression,
    /// Statistics and source types keyed by the Vortex field name used for scan resolution.
    initial_statistics_by_name: Arc<HashMap<String, NamedColumnStatistics>>,
    /// Whether DataFusion has already swapped a projection through this source.
    projection_pushdown_phase: ProjectionPushdownPhase,

    // --- Phase 2: Projected (pushed into the Vortex scan) ---
    /// The Vortex projection expression sent in the [`ScanRequest`].
    /// Composed with `initial_projection` so it operates on the original source columns.
    projected_projection: Expression,
    /// The Arrow schema of the Vortex scan output (before any leftover projection).
    projected_schema: SchemaRef,

    // --- Phase 3: Leftover (applied by DataFusion after the scan) ---
    /// DataFusion projection expressions that could not be pushed into the Vortex scan.
    /// Applied after converting arrays to record batches in [`DataSource::open`].
    /// Set after the first projection swap, including when the expression set is empty; this option
    /// controls projector construction and is not the projection-pushdown phase marker.
    leftover_projection: Option<ProjectionExprs>,
    /// The Arrow schema after applying the leftover projection.
    /// This is the output schema seen by DataFusion.
    leftover_schema: SchemaRef,
    /// Column statistics matching `leftover_schema`.
    leftover_statistics: Vec<ColumnStatistics>,

    /// An optional filter expression.
    /// Populated by [`DataSource::try_pushdown_filters`] when DataFusion pushes filters down.
    filter: Option<Expression>,
    /// An optional row limit populated by [`DataSource::with_fetch`].
    limit: Option<usize>,
    /// Whether to preserve the order of the output rows.
    ordered: bool,

    /// The requested partition count from DataFusion, populated by [`DataSource::repartitioned`].
    /// We use this as a hint for how many splits to execute concurrently in `open()`, but we
    /// always declare to DataFusion that we only have a single partition so that we can
    /// internally manage concurrency and fix the problem of partition skew.
    num_partitions: usize,
}

impl fmt::Debug for VortexDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanSource")
            .field("schema", &self.leftover_schema)
            .field("projection", &format!("{}", &self.projected_projection))
            .field("filter", &self.filter.as_ref().map(|e| format!("{}", e)))
            .field("limit", &self.limit)
            .finish()
    }
}

impl DataSource for VortexDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // VortexScanSource always uses a single partition since Vortex handles parallelism
        // and concurrency internally.
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "VortexScanSource: expected partition 0, got {partition}"
            )));
        }

        // Build the scan request with pushed-down projection, filter, and limit.
        // The projection is included so the scan can prune columns at the I/O level.
        let scan_request = ScanRequest {
            projection: self.projected_projection.clone(),
            filter: self.filter.clone(),
            limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ordered: self.ordered,
            ..Default::default()
        };

        let data_source = Arc::clone(&self.data_source);
        let projected_schema = Arc::clone(&self.projected_schema);
        let projected_target_field = Arc::new(Field::new_struct(
            "",
            projected_schema.fields().clone(),
            false,
        ));
        let session = self.session.clone();
        let num_partitions = self.num_partitions;

        // Pre-build the leftover projector (if any) so we can apply it after batch conversion.
        let leftover_projector = self
            .leftover_projection
            .as_ref()
            .map(|proj| proj.make_projector(&self.projected_schema))
            .transpose()?;

        // Defer the async DataSource::scan() call to the first poll of the stream.
        let stream = futures::stream::once(async move {
            let scan = data_source
                .scan(scan_request)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Each split.execute() returns a lazy stream whose early polls do preparation
            // work (expression resolution, layout traversal, first I/O spawns). We use
            // try_flatten_unordered to poll multiple split streams concurrently so that
            // the next split is already warm when the current one finishes.
            let scan_streams = scan.partitions().map(|split_result| {
                let split = split_result?;
                split.execute()
            });

            let handle = session.handle();
            let stream = scan_streams
                .try_flatten_unordered(Some(num_partitions * 2))
                .map(move |result| {
                    let session = session.clone();
                    let target_field = Arc::clone(&projected_target_field);
                    handle.spawn_cpu(move || {
                        let mut ctx = session.create_execution_ctx();
                        result.and_then(|chunk| {
                            let arrow = session.arrow().execute_arrow(
                                chunk,
                                Some(target_field.as_ref()),
                                &mut ctx,
                            )?;
                            Ok(RecordBatch::from(arrow.as_struct().clone()))
                        })
                    })
                })
                .buffered(num_partitions)
                .map(|result| result.map_err(|e| DataFusionError::External(Box::new(e))));

            // Apply leftover projection (expressions that couldn't be pushed into Vortex).
            let stream = if let Some(projector) = leftover_projector {
                stream
                    .map(move |batch_result| {
                        batch_result.and_then(|batch| projector.project_batch(&batch))
                    })
                    .boxed()
            } else {
                stream.boxed()
            };

            Ok::<_, DataFusionError>(stream)
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.leftover_schema),
            stream,
        )))
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "VortexScanSource: projection={}",
            self.projected_projection
        )?;
        if let Some(filter) = &self.filter {
            write!(f, ", filter={filter}")?;
        }
        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        // Vortex handles parallelism internally — always use a single partition.
        let mut this = self.clone();
        this.num_partitions = target_partitions;
        this.ordered |= output_ordering.is_some();
        Ok(Some(Arc::new(this)))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.leftover_schema))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        // FIXME(ngates): this should be adjusted based on filters. See DuckDB for heuristics,
        //  and in the future, store the selectivity stats in the session.
        let num_rows = estimate_to_df_precision(&self.data_source.row_count());

        // FIXME(ngates): byte size should be adjusted for the initial projection...
        let total_byte_size = estimate_to_df_precision(&self.data_source.byte_size());

        // Column statistics must match the output schema (leftover_schema), which may differ
        // from the initial schema after try_swapping_with_projection adds computed columns.
        let column_statistics = self
            .leftover_statistics
            .iter()
            .map(|c| sanitise_against_row_count(c, &num_rows))
            .collect::<Vec<_>>();

        let statistics = Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        };

        // A pushed-down filter or fetch limit makes every one of these an upper bound rather than a
        // fact, so none of them may stay `Exact`.
        //
        // This was harmless while column statistics were all unknown, and stops being harmless the
        // moment a Vortex file answers `field_statistics` from its footer. The filter is applied
        // *inside* this source, so there is no `FilterExec` above it for DataFusion to downgrade
        // statistics at — it takes what is reported here at face value. Reporting a whole-file
        // `Exact` minimum under `WHERE ...` is enough for the aggregate-statistics rule to rewrite
        // `SELECT MIN(x) ... WHERE y` into a literal computed from rows the query excludes, which is
        // a wrong answer rather than a slow one.
        //
        // Min/max of a filtered subset still lie within the whole-file bounds, so the values remain
        // valid *bounds*; `Inexact` is precisely that claim.
        Ok(Arc::new(if self.filter.is_some() || self.limit.is_some() {
            statistics.to_inexact()
        } else {
            statistics
        }))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let mut this = self.clone();
        this.limit = limit;
        Some(Arc::new(this))
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }

    // Note that we're explicitly "swapping" the projection. That means everything we do must
    // be computed over the original input schema, rather than the projected output schema.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        // A projection handed to this method is expressed against the current output schema. This
        // implementation can compose only with the builder's initial projection, so accepting a
        // second swap would resolve its indices and output types against the wrong schema. Leave
        // the later ProjectionExec above this source instead.
        if self.projection_pushdown_phase == ProjectionPushdownPhase::Swapped {
            return Ok(None);
        }

        tracing::debug!(
            "VortexScanSource: trying to swap with projection: {}",
            projection
        );

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.initial_schema.as_ref();
        let projected_schema = projection.project_schema(input_schema)?;

        // Use the shared ExpressionConvertor to split the projection into a Vortex
        // scan_projection and a leftover DataFusion projection for expressions that
        // can't be pushed down (e.g., unsupported scalar functions, decimal binary).
        let ProcessedProjection {
            scan_projection,
            leftover_projection,
        } = convertor.split_projection(projection.clone(), input_schema, &projected_schema)?;

        // Compose with the initial projection so the scan operates on the original
        // source columns, not the initial projection's output columns.
        let scan_projection = replace(scan_projection, &root(), self.initial_projection.clone());

        // Compute the scan output schema from the Vortex expression's return dtype.
        let scan_dtype = scan_projection
            .return_dtype(self.data_source.dtype())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let scan_output_schema = Arc::new(
            self.session
                .arrow()
                .to_arrow_schema(&scan_dtype)
                .map_err(|e| DataFusionError::External(Box::new(e)))?,
        );

        // Remap the leftover column references to match the scan output schema.
        let leftover_projection = leftover_projection
            .try_map_exprs(|expr| reassign_expr_columns(expr, &scan_output_schema))?;

        let final_schema = Arc::new(projected_schema);

        let mut this = self.clone();
        this.projection_pushdown_phase = ProjectionPushdownPhase::Swapped;
        this.projected_projection = scan_projection;
        this.projected_schema = Arc::clone(&scan_output_schema);
        this.leftover_projection = Some(leftover_projection);
        this.leftover_schema = Arc::clone(&final_schema);
        this.leftover_statistics = project_statistics(
            &self.initial_statistics_by_name,
            projection,
            final_schema.as_ref(),
        );

        Ok(Some(Arc::new(this)))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![],
            ));
        }

        // DataFusion supplies filters in terms of this source's current output schema. After a
        // projection swap, this implementation would still resolve them against `initial_schema`,
        // potentially filtering on a different source field. Keep the FilterExec above the source.
        if self.projection_pushdown_phase == ProjectionPushdownPhase::Swapped {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.initial_schema.as_ref();

        // Classify each filter: pushable filters are passed into the ScanRequest in open(),
        // so we can safely claim PushedDown::Yes for them.
        let pushdown_results: Vec<PushedDown> = filters
            .iter()
            .map(|expr| {
                if convertor.can_be_pushed_down(expr, input_schema) {
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect();

        // If nothing can be pushed down, return early.
        if pushdown_results.iter().all(|p| matches!(p, PushedDown::No)) {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                pushdown_results,
            ));
        }

        // Collect the pushable filter expressions.
        let pushable: Vec<Arc<dyn PhysicalExpr>> = filters
            .iter()
            .zip(pushdown_results.iter())
            .filter_map(|(expr, pushed)| match pushed {
                PushedDown::Yes => Some(Arc::clone(expr)),
                PushedDown::No => None,
            })
            .collect();

        // Convert to Vortex conjunction.
        let vortex_pred = make_vortex_predicate(&convertor, &pushable)?;

        // Combine with existing filter.
        let new_filter = match (&self.filter, vortex_pred) {
            (Some(existing), Some(new_pred)) => Some(vx_and(existing.clone(), new_pred)),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new_pred)) => Some(new_pred),
            (None, None) => None,
        };

        let mut this = self.clone();
        this.filter = new_filter;
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(pushdown_results)
                .with_updated_node(Arc::new(this) as _),
        )
    }
}

/// Converts a `u64` estimate to DataFusion's `usize`-typed precision.
///
/// A value that does not fit becomes `Absent`, not a saturated exact value. In particular,
/// DataFusion can fold `COUNT(*)` from an exact row count, so saturation would change a query result
/// on 32-bit targets.
fn estimate_to_df_precision(est: &Precision<u64>) -> DFPrecision<usize> {
    u64_precision_as_usize(est).to_df()
}

/// Drops a null count that cannot be true for this many rows.
///
/// The source reports these values independently. DataFusion subtracts an exact null count from an
/// exact row count to evaluate `COUNT(column)`, so an inconsistent pair must not reach it.
fn sanitise_against_row_count(
    column: &ColumnStatistics,
    num_rows: &DFPrecision<usize>,
) -> ColumnStatistics {
    let inconsistent = matches!(
        (&column.null_count, num_rows),
        (DFPrecision::Exact(nulls), DFPrecision::Exact(rows)) if nulls > rows
    );
    if !inconsistent {
        return column.clone();
    }
    let mut out = column.clone();
    out.null_count = DFPrecision::Absent;
    out
}

/// Carries the source's column statistics through a projection, where that is provably exact.
///
/// # Why by name, and why so many refusals
///
/// The scan resolves a pushed-down column by **name** — `convert::exprs` lowers a `Column` to
/// `get_item(col.name(), root())` — so the statistic follows that name. The source Arrow type is
/// retained with it because DataFusion derives the projection's output type from the column index;
/// a stale name/index pair can therefore select one source field while declaring another's type.
///
/// Everything this cannot prove yields `new_unknown()`, because the asymmetry is severe. An absent
/// statistic costs an optimisation; a wrong one is used to prune partitions and to fold an aggregate
/// into a literal, and produces a wrong answer. The refusals are:
///
/// - a non-`Column` output, whose bounds do not follow from its input's;
/// - a name with no statistics, or one that was ambiguous in the source dtype;
/// - a **repeated output alias**, because alias collisions let the pushed-down projection deliver
///   the same column twice while these entries still describe two different ones;
/// - a source field whose Arrow type disagrees with the output type;
/// - a scalar statistic whose type disagrees with that type (or, for `Sum`, with DataFusion's
///   widened result type), since separate conversion paths may produce, for example, a `Utf8`
///   bound for a `Utf8View` field.
fn project_statistics(
    by_name: &HashMap<String, NamedColumnStatistics>,
    projection: &ProjectionExprs,
    output_schema: &Schema,
) -> Vec<ColumnStatistics> {
    let output_len = output_schema.fields().len();
    let mut out = vec![ColumnStatistics::new_unknown(); output_len];

    // Repeated aliases are refused wholesale rather than per-column: a collision changes which
    // column the scan delivers for an output, and that is not localised to the colliding pair.
    let mut aliases = HashSet::new();
    if !projection.iter().all(|p| aliases.insert(p.alias.as_str())) {
        return out;
    }

    for (idx, proj) in projection.iter().enumerate().take(output_len) {
        let Some(column) = (proj.expr.as_ref() as &dyn Any).downcast_ref::<Column>() else {
            continue;
        };
        let Some(named_stats) = by_name.get(column.name()) else {
            continue;
        };
        let output_type = output_schema.field(idx).data_type();
        if named_stats.data_type != *output_type {
            continue;
        }
        out[idx] = retain_type_compatible(&named_stats.statistics, output_type);
    }
    out
}

/// Drops scalar statistics whose types disagree with DataFusion's interpretation of the column.
///
/// The Arrow schema and scalar conversion are produced by different code paths. For example, a
/// Vortex string column surfaces as `Utf8View` while its scalar bounds convert to `Utf8`.
/// `Sum` is checked against DataFusion's widened result type, excluding unsupported inputs such as
/// booleans and decimal values whose Vortex storage width differs from the width DataFusion keeps
/// while widening precision.
fn retain_type_compatible(stats: &ColumnStatistics, data_type: &DataType) -> ColumnStatistics {
    let compatible =
        |p: &DFPrecision<ScalarValue>, expected: &DataType| -> DFPrecision<ScalarValue> {
            match p {
                DFPrecision::Exact(v) | DFPrecision::Inexact(v) if v.data_type() != *expected => {
                    DFPrecision::Absent
                }
                other => other.clone(),
            }
        };
    let mut out = stats.clone();
    out.min_value = compatible(&stats.min_value, data_type);
    out.max_value = compatible(&stats.max_value, data_type);
    out.sum_value = match datafusion_sum_type(data_type) {
        Ok(Some(sum_type)) => compatible(&stats.sum_value, &sum_type),
        Ok(None) => DFPrecision::Absent,
        Err(error) => {
            tracing::warn!(
                input_type = ?data_type,
                %error,
                "failed to resolve DataFusion SUM type; withholding the Vortex sum statistic"
            );
            DFPrecision::Absent
        }
    };
    out
}

/// DataFusion's result dtype for `SUM` after its implicit numeric coercions.
///
/// Expected non-numeric inputs return `Ok(None)`. Once an input is in SUM's supported domain,
/// coercion and return-field failures indicate a DataFusion/API mismatch and are returned with
/// context rather than being silently treated as an unsupported type.
fn datafusion_sum_type(data_type: &DataType) -> DFResult<Option<DataType>> {
    datafusion_sum_type_with(data_type, |data_type| {
        let sum = sum_udaf();
        let input = Arc::new(Field::new("sum_arg", data_type.clone(), true));
        let coerced = fields_with_udf(&[input], sum.as_ref()).map_err(|error| {
            DataFusionError::Context(
                format!("failed to coerce DataFusion SUM input {data_type}"),
                Box::new(error),
            )
        })?;
        sum.return_field(&coerced)
            .map(|field| field.data_type().clone())
            .map_err(|error| {
                DataFusionError::Context(
                    format!("failed to derive DataFusion SUM return field for {data_type}"),
                    Box::new(error),
                )
            })
    })
}

fn datafusion_sum_type_with(
    data_type: &DataType,
    resolve: impl FnOnce(&DataType) -> DFResult<DataType>,
) -> DFResult<Option<DataType>> {
    if !datafusion_sum_supports(data_type) {
        return Ok(None);
    }

    resolve(data_type).map(Some).map_err(|error| {
        DataFusionError::Context(
            format!("failed to resolve supported DataFusion SUM input {data_type}"),
            Box::new(error),
        )
    })
}

fn datafusion_sum_supports(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
            | DataType::Duration(_)
    ) || matches!(data_type, DataType::Dictionary(_, value) if datafusion_sum_supports(value))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use datafusion::prelude::SessionContext;
    use datafusion_common::arrow::array::Float64Array;
    use datafusion_common::arrow::array::Int64Array;
    use datafusion_common::arrow::datatypes::i256;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_expr::projection::ProjectionExpr;
    use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::stats::StatsSet;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::PType;
    use vortex::expr::stats::Stat;
    use vortex::extension::datetime::TimeUnit;
    use vortex::extension::datetime::Timestamp;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::scalar::Scalar as VortexScalar;
    use vortex::scalar::ScalarValue as VortexScalarValue;

    use super::*;
    use crate::v2::VortexTable;

    struct StatisticsDataSource {
        dtype: DType,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl vortex::scan::DataSource for StatisticsDataSource {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        async fn scan(
            &self,
            _scan_request: ScanRequest,
        ) -> VortexResult<vortex::scan::DataSourceScanRef> {
            vortex_bail!("test source is not scanned")
        }

        async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(StatsSet::of(
                Stat::Max,
                Precision::exact(VortexScalarValue::from(9i64)),
            ))
        }
    }

    struct CountingDataSource {
        inner: DataSourceRef,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl vortex::scan::DataSource for CountingDataSource {
        fn dtype(&self) -> &DType {
            self.inner.dtype()
        }

        fn row_count(&self) -> Precision<u64> {
            self.inner.row_count()
        }

        fn byte_size(&self) -> Precision<u64> {
            self.inner.byte_size()
        }

        async fn scan(
            &self,
            scan_request: ScanRequest,
        ) -> VortexResult<vortex::scan::DataSourceScanRef> {
            self.inner.scan(scan_request).await
        }

        async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.field_statistics(field_path).await
        }
    }

    fn counting_source(inner: DataSourceRef) -> (DataSourceRef, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(CountingDataSource {
                inner,
                calls: Arc::clone(&calls),
            }),
            calls,
        )
    }

    #[tokio::test]
    async fn vortex_file_footer_extrema_fold_through_vortex_table() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let array = StructArray::from_fields(&[(
            "x",
            PrimitiveArray::from_iter([7i64, 1, 4]).into_array(),
        )])?
        .into_array();
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, array.to_array_stream())
            .await?;

        let file = session.open_options().open_buffer(bytes)?;
        let file_source = file.data_source()?;
        let footer_stats = file_source
            .field_statistics(&FieldPath::from_name("x"))
            .await?;
        assert_eq!(
            footer_stats.get(Stat::Min),
            Precision::exact(VortexScalarValue::from(1i64))
        );
        assert_eq!(
            footer_stats.get(Stat::Max),
            Precision::exact(VortexScalarValue::from(7i64))
        );

        let arrow_schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
        let (matching_source, matching_calls) = counting_source(Arc::clone(&file_source));
        let ctx = SessionContext::new();
        ctx.register_table(
            "matching",
            Arc::new(VortexTable::new(
                matching_source,
                session.clone(),
                Arc::clone(&arrow_schema),
            )),
        )?;
        let matching_plan = ctx
            .sql("SELECT MIN(x), MAX(x) FROM matching")
            .await?
            .create_physical_plan()
            .await?;

        assert_eq!(matching_calls.load(Ordering::Relaxed), 1);
        let projection = matching_plan
            .downcast_ref::<ProjectionExec>()
            .expect("exact footer extrema should replace the aggregate");
        assert!(projection.input().is::<PlaceholderRowExec>());
        let literals = projection
            .expr()
            .iter()
            .map(|expr| {
                expr.expr
                    .downcast_ref::<Literal>()
                    .expect("folded aggregate expression should be a literal")
                    .value()
                    .clone()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            literals,
            vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(7))]
        );

        let (mismatched_source, mismatched_calls) = counting_source(file_source);
        let mismatched_schema =
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let mismatch_ctx = SessionContext::new();
        mismatch_ctx.register_table(
            "mismatched",
            Arc::new(VortexTable::new(
                mismatched_source,
                session,
                mismatched_schema,
            )),
        )?;
        let mismatched_plan = mismatch_ctx
            .sql("SELECT MIN(x) FROM mismatched")
            .await?
            .create_physical_plan()
            .await?;

        assert_eq!(mismatched_calls.load(Ordering::Relaxed), 0);
        assert!(
            !mismatched_plan
                .downcast_ref::<ProjectionExec>()
                .is_some_and(|projection| projection.input().is::<PlaceholderRowExec>())
        );
        Ok(())
    }

    #[tokio::test]
    async fn vortex_table_scans_instead_of_folding_a_floating_sum() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let array = StructArray::from_fields(&[(
            "x",
            PrimitiveArray::from_iter([f64::MAX, f64::MAX, -f64::MAX]).into_array(),
        )])?
        .into_array();
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, array.to_array_stream())
            .await?;

        let file = session.open_options().open_buffer(bytes)?;
        let raw_sum = file
            .file_stats()
            .and_then(|statistics| statistics.stats_sets().first())
            .and_then(|stats| stats.get(Stat::Sum).as_exact())
            .ok_or_else(|| anyhow::anyhow!("written file did not contain a SUM statistic"))?;
        let raw_sum = VortexScalar::try_new(PType::F64.into(), Some(raw_sum))?;
        assert_eq!(f64::try_from(&raw_sum)?, f64::INFINITY);

        let file_source = file.data_source()?;
        assert!(
            file_source
                .field_statistics(&FieldPath::from_name("x"))
                .await?
                .get(Stat::Sum)
                .is_absent(),
            "a sequential floating footer sum must not be published to DataFusion"
        );

        let arrow_schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
        let ctx = SessionContext::new();
        ctx.register_table(
            "floats",
            Arc::new(VortexTable::new(file_source, session, arrow_schema)),
        )?;

        let plan = ctx
            .sql("SELECT SUM(x) FROM floats")
            .await?
            .create_physical_plan()
            .await?;
        assert!(
            !plan
                .downcast_ref::<ProjectionExec>()
                .is_some_and(|projection| projection.input().is::<PlaceholderRowExec>()),
            "the non-associative floating sum must be computed by scanning"
        );

        let batches = ctx
            .sql("SELECT SUM(x) FROM floats")
            .await?
            .collect()
            .await?;
        let result = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow::anyhow!("SUM(x) did not produce Float64"))?;
        assert_eq!(result.value(0), f64::MAX);
        Ok(())
    }

    #[tokio::test]
    async fn vortex_table_projection_preserves_only_column_extrema() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let array = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([7i64, 1, 4]).into_array()),
            ("b", PrimitiveArray::from_iter([20i64, 30, 10]).into_array()),
        ])?
        .into_array();
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, array.to_array_stream())
            .await?;

        let file = session.open_options().open_buffer(bytes)?;
        let arrow_schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
        let ctx = SessionContext::new();
        ctx.register_table(
            "projected",
            Arc::new(VortexTable::new(file.data_source()?, session, arrow_schema)),
        )?;

        let extrema = ctx
            .sql(
                "SELECT MIN(b_alias), MAX(a_alias) \
                 FROM (SELECT b AS b_alias, a AS a_alias FROM projected)",
            )
            .await?;
        let extrema_plan = extrema.create_physical_plan().await?;
        let projection = extrema_plan
            .downcast_ref::<ProjectionExec>()
            .ok_or_else(|| anyhow::anyhow!("projected extrema were not folded to literals"))?;
        if !projection.input().is::<PlaceholderRowExec>() {
            anyhow::bail!("projected extrema did not use footer statistics");
        }
        let literals = projection
            .expr()
            .iter()
            .map(|expr| {
                expr.expr
                    .downcast_ref::<Literal>()
                    .ok_or_else(|| anyhow::anyhow!("folded extrema were not literals"))
                    .map(|literal| literal.value().clone())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        assert_eq!(
            literals,
            vec![ScalarValue::Int64(Some(10)), ScalarValue::Int64(Some(7))]
        );

        let computed = ctx
            .sql(
                "SELECT MIN(computed) \
                 FROM (SELECT a + b AS computed FROM projected)",
            )
            .await?;
        let computed_plan = computed.create_physical_plan().await?;
        assert!(
            !computed_plan
                .downcast_ref::<ProjectionExec>()
                .is_some_and(|projection| projection.input().is::<PlaceholderRowExec>()),
            "a computed projection must not inherit either input column's extrema"
        );
        let batches = computed.collect().await?;
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("MIN(computed) did not produce Int64"))?;
        assert_eq!(values.value(0), 14);
        Ok(())
    }

    async fn unique_adapter() -> VortexResult<VortexDataSource> {
        let dtype = DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        );
        let source: DataSourceRef = Arc::new(StatisticsDataSource {
            dtype,
            calls: AtomicUsize::new(0),
        });
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        VortexDataSource::builder(source, VortexSession::default())
            .with_arrow_schema(arrow_schema)
            .build()
            .await
    }

    fn column_projection(name: &str, index: usize) -> ProjectionExprs {
        ProjectionExprs::new(vec![ProjectionExpr {
            expr: Arc::new(Column::new(name, index)),
            alias: name.to_string(),
        }])
    }

    #[test]
    fn initial_statistics_require_unambiguous_matching_names() {
        let fields = StructFields::from_iter([
            ("x", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ("x", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
        ]);
        let matching_schema = Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
        ]);

        assert_eq!(
            statistics_field_indices(
                &matching_schema,
                &fields,
                &fields,
                &VortexSession::default(),
            ),
            vec![2]
        );

        let reordered_schema = Schema::new(vec![
            Field::new("y", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
        ]);
        assert!(
            statistics_field_indices(
                &reordered_schema,
                &fields,
                &fields,
                &VortexSession::default(),
            )
            .is_empty()
        );

        let projected_fields = StructFields::from_iter([
            ("x", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
        ]);
        let projected_schema = Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
        ]);
        assert_eq!(
            statistics_field_indices(
                &projected_schema,
                &projected_fields,
                &fields,
                &VortexSession::default(),
            ),
            vec![1]
        );
    }

    #[test]
    fn initial_statistics_reject_temporal_and_decimal_schema_casts() {
        let fields = StructFields::from_iter([
            (
                "time",
                DType::Extension(
                    Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
                ),
            ),
            (
                "decimal",
                DType::Decimal(DecimalDType::new(38, 0), Nullability::NonNullable),
            ),
        ]);
        let schema = Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
                false,
            ),
            Field::new("decimal", DataType::Decimal128(38, 2), false),
        ]);

        assert!(
            statistics_field_indices(&schema, &fields, &fields, &VortexSession::default())
                .is_empty()
        );
    }

    #[tokio::test]
    async fn duplicate_source_names_stay_unknown_after_initial_projection() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([
                ("x", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("x", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        );
        let source = Arc::new(StatisticsDataSource {
            dtype,
            calls: AtomicUsize::new(0),
        });
        let source_ref = Arc::clone(&source);
        let source_ref: DataSourceRef = source_ref;
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
        ]));

        let adapter = VortexDataSource::builder(source_ref, VortexSession::default())
            .with_arrow_schema(arrow_schema)
            .with_projection(vec![0, 2])
            .build()
            .await?;

        assert_eq!(source.calls.load(Ordering::Relaxed), 1);
        assert_eq!(adapter.leftover_statistics.len(), 2);
        assert_eq!(
            adapter.leftover_statistics[0],
            ColumnStatistics::new_unknown()
        );
        assert_eq!(
            adapter.leftover_statistics[1].max_value,
            DFPrecision::Exact(ScalarValue::Int64(Some(9)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_second_projection_swap_is_declined() -> Result<(), Box<dyn std::error::Error>> {
        let adapter = unique_adapter().await?;
        let first = adapter
            .try_swapping_with_projection(&column_projection("b", 1))?
            .ok_or_else(|| std::io::Error::other("first projection was not swapped"))?;
        let mut first = first
            .downcast_ref::<VortexDataSource>()
            .ok_or_else(|| std::io::Error::other("expected VortexDataSource"))?
            .clone();
        assert_eq!(
            first.projection_pushdown_phase,
            ProjectionPushdownPhase::Swapped
        );

        // The safety boundary is the explicit phase, not the representation of residual work.
        first.leftover_projection = None;

        assert!(
            first
                .try_swapping_with_projection(&column_projection("b", 0))?
                .is_none()
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_filter_after_projection_swap_is_declined() -> Result<(), Box<dyn std::error::Error>>
    {
        let adapter = unique_adapter().await?;
        let first = adapter
            .try_swapping_with_projection(&column_projection("b", 1))?
            .ok_or_else(|| std::io::Error::other("first projection was not swapped"))?;
        let mut first = first
            .downcast_ref::<VortexDataSource>()
            .ok_or_else(|| std::io::Error::other("expected VortexDataSource"))?
            .clone();
        assert_eq!(
            first.projection_pushdown_phase,
            ProjectionPushdownPhase::Swapped
        );

        // The safety boundary is the explicit phase, not the representation of residual work.
        first.leftover_projection = None;
        let filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int64(Some(9)))),
        ));

        let result = first.try_pushdown_filters(vec![filter], &ConfigOptions::default())?;
        assert!(matches!(result.filters.as_slice(), [PushedDown::No]));
        assert!(result.updated_node.is_none());
        Ok(())
    }

    #[test]
    fn sum_statistics_follow_datafusion_result_types() {
        let mut stats = ColumnStatistics::new_unknown();
        stats.sum_value = DFPrecision::Exact(ScalarValue::Int64(Some(3)));
        assert_eq!(
            retain_type_compatible(&stats, &DataType::Int32).sum_value,
            stats.sum_value
        );

        stats.sum_value = DFPrecision::Exact(ScalarValue::UInt64(Some(3)));
        assert_eq!(
            retain_type_compatible(&stats, &DataType::Boolean).sum_value,
            DFPrecision::Absent
        );

        // Vortex represents precision 18 in Decimal64, while DataFusion keeps a Decimal128 input
        // in Decimal128 when widening its precision for SUM.
        stats.sum_value = DFPrecision::Exact(ScalarValue::Decimal64(Some(3), 18, 2));
        assert_eq!(
            retain_type_compatible(&stats, &DataType::Decimal128(8, 2)).sum_value,
            DFPrecision::Absent
        );

        stats.sum_value = DFPrecision::Exact(ScalarValue::Decimal128(Some(3), 30, 2));
        assert_eq!(
            retain_type_compatible(&stats, &DataType::Decimal128(20, 2)).sum_value,
            stats.sum_value
        );

        // Vortex crosses into Decimal256 here, whereas DataFusion caps this Decimal128 result at
        // precision 38 without changing its storage width.
        stats.sum_value =
            DFPrecision::Exact(ScalarValue::Decimal256(Some(i256::from_i128(3)), 40, 2));
        assert_eq!(
            retain_type_compatible(&stats, &DataType::Decimal128(30, 2)).sum_value,
            DFPrecision::Absent
        );
    }

    #[test]
    fn datafusion_sum_type_distinguishes_unsupported_inputs_from_failures() -> anyhow::Result<()> {
        let unsupported = datafusion_sum_type_with(&DataType::Boolean, |_| {
            Err(DataFusionError::Internal(
                "the resolver ran for an unsupported SUM input".to_string(),
            ))
        })?;
        assert_eq!(unsupported, None);

        let failure = datafusion_sum_type_with(&DataType::Int32, |_| {
            Err(DataFusionError::Plan(
                "synthetic SUM coercion failure".to_string(),
            ))
        });
        let Err(failure) = failure else {
            anyhow::bail!("a supported SUM input's coercion failure was swallowed")
        };
        let message = failure.to_string();
        assert!(message.contains("supported DataFusion SUM input Int32"));
        assert!(message.contains("synthetic SUM coercion failure"));
        Ok(())
    }
}

#[cfg(test)]
mod statistics_tests {
    use arrow_schema::Field;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_expr::projection::ProjectionExpr;
    use datafusion_physical_expr::projection::ProjectionExprs;
    use vortex::dtype::PType;

    use super::*;

    fn stats_with_max(v: i64) -> ColumnStatistics {
        ColumnStatistics::new_unknown()
            .with_max_value(DFPrecision::Exact(ScalarValue::Int64(Some(v))))
    }

    fn named_statistics(
        data_type: DataType,
        statistics: ColumnStatistics,
    ) -> NamedColumnStatistics {
        NamedColumnStatistics {
            data_type,
            statistics,
        }
    }

    fn by_name(pairs: &[(&str, i64)]) -> HashMap<String, NamedColumnStatistics> {
        pairs
            .iter()
            .map(|(n, v)| {
                (
                    (*n).to_string(),
                    named_statistics(DataType::Int64, stats_with_max(*v)),
                )
            })
            .collect()
    }

    fn schema_of(fields: &[(&str, DataType)]) -> Schema {
        Schema::new(
            fields
                .iter()
                .map(|(n, t)| Field::new(*n, t.clone(), true))
                .collect::<Vec<_>>(),
        )
    }

    fn int_schema(names: &[&str]) -> Schema {
        schema_of(
            &names
                .iter()
                .map(|n| (*n, DataType::Int64))
                .collect::<Vec<_>>(),
        )
    }

    fn col(name: &str, idx: usize) -> ProjectionExpr {
        ProjectionExpr {
            expr: Arc::new(Column::new(name, idx)),
            alias: name.to_string(),
        }
    }

    /// A reordering projection carries each column's statistics to its new position.
    #[test]
    fn a_reordering_projection_moves_statistics_with_the_columns() {
        let stats = by_name(&[("a", 10), ("b", 20), ("c", 30)]);
        let out_schema = int_schema(&["c", "a"]);
        let projection = ProjectionExprs::new(vec![col("c", 2), col("a", 0)]);

        let out = project_statistics(&stats, &projection, &out_schema);

        assert_eq!(out[0].max_value, stats["c"].statistics.max_value);
        assert_eq!(out[1].max_value, stats["a"].statistics.max_value);
    }

    /// Resolution follows the column NAME, not its index.
    ///
    /// The scan lowers a column to `get_item(name, root())`. When the named field and output field
    /// have the same type, a stale index must not change which statistics follow the scanned data.
    #[test]
    fn resolution_follows_the_name_not_the_index() {
        let stats = by_name(&[("a", 10), ("b", 20)]);
        let out_schema = int_schema(&["b"]);
        // Names "b" but carries a stale index of 0.
        let projection = ProjectionExprs::new(vec![col("b", 0)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(
            out[0].max_value, stats["b"].statistics.max_value,
            "the name must decide, because the scan resolves by name"
        );
    }

    /// A computed output must report unknown rather than inherit its input's bounds.
    #[test]
    fn a_computed_column_is_not_given_its_inputs_statistics() {
        let stats = by_name(&[("a", 10)]);
        let out_schema = int_schema(&["a", "lit"]);
        let projection = ProjectionExprs::new(vec![
            col("a", 0),
            ProjectionExpr {
                expr: Arc::new(Literal::new(ScalarValue::Int64(Some(7)))),
                alias: "lit".to_string(),
            },
        ]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(out[0].max_value, stats["a"].statistics.max_value);
        assert_eq!(out[1].max_value, DFPrecision::Absent);
    }

    /// Repeated output aliases suppress statistics entirely.
    ///
    /// A collision lets the pushed-down projection deliver one column twice while these entries
    /// still describe two different ones, so nothing here can be trusted positionally.
    #[test]
    fn a_repeated_output_alias_suppresses_everything() {
        let stats = by_name(&[("a", 10), ("b", 20)]);
        let out_schema = int_schema(&["dup", "dup"]);
        let projection = ProjectionExprs::new(vec![
            ProjectionExpr {
                expr: Arc::new(Column::new("a", 0)),
                alias: "dup".to_string(),
            },
            ProjectionExpr {
                expr: Arc::new(Column::new("b", 1)),
                alias: "dup".to_string(),
            },
        ]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(out[0].max_value, DFPrecision::Absent);
        assert_eq!(out[1].max_value, DFPrecision::Absent);
    }

    /// A bound whose scalar type disagrees with the output column is dropped.
    ///
    /// The Arrow mapping and the scalar conversion are produced by different code, so a column can
    /// surface as `Utf8View` while its bounds convert to `Utf8`.
    #[test]
    fn a_type_mismatched_bound_is_dropped() {
        let mut stats = HashMap::new();
        stats.insert(
            "s".to_string(),
            named_statistics(
                DataType::Utf8View,
                ColumnStatistics::new_unknown()
                    .with_max_value(DFPrecision::Exact(ScalarValue::Utf8(Some("z".into())))),
            ),
        );
        let out_schema = schema_of(&[("s", DataType::Utf8View)]);
        let projection = ProjectionExprs::new(vec![col("s", 0)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(
            out[0].max_value,
            DFPrecision::Absent,
            "a Utf8 bound must not be published for a Utf8View column"
        );
    }

    /// A name with no statistics yields unknown.
    #[test]
    fn an_unknown_name_is_unknown_rather_than_a_panic() {
        let stats = by_name(&[("a", 10)]);
        let out_schema = int_schema(&["missing"]);
        let projection = ProjectionExprs::new(vec![col("missing", 7)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(out[0].max_value, DFPrecision::Absent);
    }

    /// A schema whose names match the Vortex fields in order is usable.
    ///
    /// This is the case that matters: `VortexTable::scan` *always* supplies an Arrow schema, so a
    /// rule refusing whenever one was supplied would be sound and useless — it would disable footer
    /// statistics on the only public path that has them. The supplied schema is virtually always
    /// derived from the same dtype, and agreement is what makes name and position interchangeable.
    #[test]
    fn a_matching_supplied_schema_is_usable() {
        assert!(schema_agrees(&["a", "b", "c"], &["a", "b", "c"]));
    }

    /// Reordered, renamed, or differently sized schemas are refused.
    ///
    /// The struct executor zips Vortex to Arrow fields positionally, so `[b, a]` over `[a, b]`
    /// emits Vortex `a` under Arrow name `b` — data identity positional, statistics by name.
    #[test]
    fn a_reordered_or_renamed_supplied_schema_is_refused() {
        assert!(!schema_agrees(&["b", "a"], &["a", "b"]), "reordered");
        assert!(!schema_agrees(&["x", "b"], &["a", "b"]), "renamed");
        assert!(!schema_agrees(&["a"], &["a", "b"]), "wrong arity");
    }

    /// Exercise the same agreement helper used by `build`.
    fn schema_agrees(arrow_names: &[&str], vortex_names: &[&str]) -> bool {
        let schema = int_schema(arrow_names);
        let fields = StructFields::from_iter(vortex_names.iter().map(|name| {
            (
                *name,
                DType::Primitive(PType::I64, Nullability::NonNullable),
            )
        }));
        schema_names_match(&schema, &fields)
    }

    /// A widened Sum survives when the named source field and output field have the same type.
    #[test]
    fn a_widened_sum_is_not_discarded() {
        let mut stats = HashMap::new();
        stats.insert(
            "n".to_string(),
            named_statistics(
                DataType::Int32,
                ColumnStatistics::new_unknown()
                    .with_sum_value(DFPrecision::Exact(ScalarValue::Int64(Some(99)))),
            ),
        );
        // The column is Int32; its sum is Int64.
        let out_schema = schema_of(&[("n", DataType::Int32)]);
        let projection = ProjectionExprs::new(vec![col("n", 0)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(
            out[0].sum_value,
            DFPrecision::Exact(ScalarValue::Int64(Some(99))),
            "a widened sum must be kept"
        );
    }

    /// A stale index cannot carry a named field's widened sum into an output of another type.
    #[test]
    fn a_source_output_type_mismatch_drops_all_statistics() {
        let mut stats = HashMap::new();
        stats.insert(
            "n".to_string(),
            named_statistics(
                DataType::Int32,
                ColumnStatistics::new_unknown()
                    .with_sum_value(DFPrecision::Exact(ScalarValue::Int64(Some(99)))),
            ),
        );
        // The name selects the Int32 field `n`, but the stale index made DataFusion declare an
        // Int64 output field. The Int64 sum scalar alone cannot prove it belongs to that output.
        let out_schema = schema_of(&[("n", DataType::Int64)]);
        let projection = ProjectionExprs::new(vec![col("n", 1)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(out[0], ColumnStatistics::new_unknown());
    }

    /// More projection expressions than output fields must not overrun.
    #[test]
    fn a_longer_projection_than_output_does_not_overrun() {
        let stats = by_name(&[("a", 10), ("b", 20)]);
        let out_schema = int_schema(&["a"]);
        let projection = ProjectionExprs::new(vec![col("a", 0), col("b", 1)]);

        let out = project_statistics(&stats, &projection, &out_schema);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].max_value, stats["a"].statistics.max_value);
    }
}
