// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the [`VortexFile`] struct, which represents a Vortex file on disk or in memory.
//!
//! The `VortexFile` provides methods for accessing file metadata, creating segment sources for reading
//! data from the file, and initiating scans to read the file's contents into memory as Vortex arrays.

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::layout::LayoutReaderDataSource;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::segments::SegmentSource;
use vortex_scan::DataSource;
use vortex_scan::DataSourceRef;
use vortex_scan::DataSourceScanRef;
use vortex_scan::PartitionRef;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::FileStatistics;
use crate::footer::Footer;
use crate::pruning::can_prune_file_stats;
use crate::v2::FileStatsLayoutReader;

/// Represents a Vortex file, providing access to its metadata and content.
///
/// A `VortexFile` is created by opening a Vortex file using [`VortexOpenOptions`](crate::VortexOpenOptions).
/// It provides methods for accessing file metadata (such as row count, data type, and statistics)
/// and for initiating scans to read the file's contents.
#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file, containing metadata and layout information.
    footer: Footer,
    /// The segment source used to read segments from this file.
    segment_source: Arc<dyn SegmentSource>,
    /// The Vortex session used to open this file.
    session: VortexSession,
    /// User-defined metadata values resolved for this file open.
    metadata: Arc<HashMap<String, ByteBuffer>>,
    /// None id LayoutReader caching is turned off
    layout_reader_cache: Option<OnceLock<Arc<dyn LayoutReader>>>,
}

fn layout_reader(
    segment_source: Arc<dyn SegmentSource>,
    footer: &Footer,
    session: &VortexSession,
) -> VortexResult<Arc<dyn LayoutReader>> {
    let root_reader = footer
        .layout()
        // TODO(ngates): we may want to allow the user pass in a name here?
        .new_reader("".into(), segment_source, session, &Default::default())?;

    Ok(if let Some(stats) = footer.statistics().cloned() {
        Arc::new(FileStatsLayoutReader::new(
            root_reader,
            stats,
            session.clone(),
        ))
    } else {
        root_reader
    })
}

impl VortexFile {
    /// Creates a new `VortexFile` from the given footer, segment source, and session.
    pub fn new(
        footer: Footer,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        Self {
            footer,
            segment_source,
            session,
            metadata: Arc::new(HashMap::new()),
            layout_reader_cache: None,
        }
    }

    pub(crate) fn with_metadata(mut self, metadata: Arc<HashMap<String, ByteBuffer>>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Enable layout reader caching.
    ///
    /// Repeated calls to [`layout_reader`](Self::layout_reader), [`scan`](Self::scan), and
    /// [`data_source`](Self::data_source) will share the same reader tree.
    pub fn with_caching(self) -> Self {
        Self {
            footer: self.footer,
            segment_source: self.segment_source,
            session: self.session,
            metadata: self.metadata,
            layout_reader_cache: Some(OnceLock::new()),
        }
    }

    /// Returns a reference to the file's footer, which contains metadata and layout information.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    /// Returns the data type of the file's contents.
    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    /// Returns the file's statistics, if available.
    ///
    /// Statistics can be used for query optimization and data exploration.
    pub fn file_stats(&self) -> Option<&FileStatistics> {
        self.footer.statistics()
    }

    /// Returns the user-defined metadata segments loaded for this file.
    ///
    /// Metadata is only loaded when requested during open. Iteration order is unspecified.
    pub fn metadata_segments(&self) -> impl Iterator<Item = (&str, &ByteBuffer)> {
        self.metadata
            .iter()
            .map(|(key, metadata)| (key.as_str(), metadata))
    }

    /// Returns the loaded user-defined metadata segment for the given key.
    ///
    /// Returns `None` when the key is absent or metadata was not loaded.
    pub fn metadata_segment(&self, key: &str) -> Option<&ByteBuffer> {
        self.metadata.get(key)
    }

    /// Create a new segment source for reading from the file.
    ///
    /// This may spawn a background I/O driver that will exit when the returned segment source
    /// is dropped.
    pub fn segment_source(&self) -> Arc<dyn SegmentSource> {
        Arc::clone(&self.segment_source)
    }

    /// Replace the segment source used by this file.
    ///
    /// Any cached layout reader is cleared so that subsequent scans construct readers over the
    /// replacement source.
    pub fn with_segment_source(mut self, segment_source: Arc<dyn SegmentSource>) -> Self {
        self.segment_source = segment_source;
        if self.layout_reader_cache.is_some() {
            self.layout_reader_cache = Some(OnceLock::new());
        }
        self
    }

    /// Returns a reference to the Vortex session used to open this file.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Create a new layout reader for the file.
    ///
    /// Wraps the root layout in a [`FileStatsLayoutReader`] if file stats are available.
    pub fn layout_reader(&self) -> VortexResult<Arc<dyn LayoutReader>> {
        match &self.layout_reader_cache {
            None => layout_reader(
                Arc::clone(&self.segment_source),
                &self.footer,
                &self.session,
            ),
            Some(reader) => {
                // get_or_try_init is unstable
                if let Some(val) = reader.get() {
                    Ok(Arc::clone(val))
                } else {
                    let inner = layout_reader(
                        Arc::clone(&self.segment_source),
                        &self.footer,
                        &self.session,
                    )?;
                    Ok(if let Err(val) = reader.set(Arc::clone(&inner)) {
                        val
                    } else {
                        inner
                    })
                }
            }
        }
    }

    /// Create a [`DataSource`](vortex_scan::DataSource) from this file for scanning.
    ///
    /// Wraps the file's layout reader with [`FileStatsLayoutReader`] (when file-level
    /// statistics are available) and [`LayoutReaderDataSource`].
    ///
    /// The result answers [`DataSource::field_statistics`] from the footer.
    /// [`LayoutReaderDataSource`] cannot: a layout reader alone has no route to the file's statistics
    /// segment, so it returns an empty [`StatsSet`] for every field. That is why a query answerable
    /// purely from metadata — `SELECT MIN(c), MAX(c)`, say — otherwise reads the column in full even
    /// though the bounds were in the footer the whole time.
    pub fn data_source(&self) -> VortexResult<DataSourceRef> {
        let reader = self.layout_reader()?;

        Ok(Arc::new(FileDataSource {
            inner: LayoutReaderDataSource::new(reader, self.session.clone()),
            statistics: self.footer.statistics().cloned(),
            dtype: self.footer.dtype().clone(),
        }))
    }

    /// Initiate a scan of the file, returning a builder for projection, filtering, selection, and
    /// execution options.
    pub fn scan(&self) -> VortexResult<ScanBuilder<ArrayRef>> {
        Ok(ScanBuilder::new(
            self.session.clone(),
            self.layout_reader()?,
        ))
    }

    /// Returns `true` if file-level statistics prove the expression cannot
    /// match any rows in this file.
    ///
    /// Row-count-aware pruning predicates are evaluated with the file's total
    /// row count as their scope.
    pub fn can_prune(&self, filter: &Expression) -> VortexResult<bool> {
        let Some((stats, fields)) = self
            .footer
            .statistics()
            .zip(self.footer.dtype().as_struct_fields_opt())
        else {
            return Ok(false);
        };

        can_prune_file_stats(
            &filter.bind(self.footer.dtype())?,
            self.footer.row_count(),
            stats,
            fields,
            &self.session,
        )
    }

    /// Return the file's natural row splits as root-coordinate ranges.
    ///
    /// These are the ranges that [`SplitBy::Layout`] would use for an all-fields scan.
    pub fn splits(&self) -> VortexResult<Vec<Range<u64>>> {
        let reader = self.layout_reader()?;
        Ok(SplitBy::Layout
            .splits(reader.as_ref(), &(0..reader.row_count()), &[FieldMask::All])?
            .into_iter()
            .tuple_windows()
            .map(|(start, end)| start..end)
            .collect())
    }
}

/// A [`DataSource`] over a Vortex file that can answer per-field statistics from the footer.
///
/// Scanning is delegated to [`LayoutReaderDataSource`]; this wrapper adds
/// [`field_statistics`](DataSource::field_statistics) because the footer, not the layout reader,
/// owns the file-level [`StatsSet`] values.
struct FileDataSource {
    inner: LayoutReaderDataSource,
    /// The file's per-field statistics, absent for a file written without them.
    statistics: Option<FileStatistics>,
    /// The file's (unprojected) dtype, used to resolve a field name to a statistics index.
    dtype: DType,
}

#[async_trait::async_trait]
impl DataSource for FileDataSource {
    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.inner.row_count()
    }

    fn byte_size(&self) -> Precision<u64> {
        self.inner.byte_size()
    }

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.inner.serialize()
    }

    fn deserialize_partition(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        self.inner.deserialize_partition(data, session)
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        self.inner.scan(scan_request).await
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        let Some(statistics) = &self.statistics else {
            return Ok(StatsSet::default());
        };

        // File statistics are recorded per top-level field, so only a single-component path can be
        // answered. A nested path is not wrong to ask about, there is simply nothing recorded for it.
        let [field] = field_path.parts() else {
            return Ok(StatsSet::default());
        };
        let Some(name) = field.as_name() else {
            return Ok(StatsSet::default());
        };
        let Some(fields) = self.dtype.as_struct_fields_opt() else {
            return Ok(StatsSet::default());
        };
        // Resolved by name, so an ambiguous name must not be resolved at all.
        //
        // Duplicate field names are legal in a struct dtype, and `position` would silently hand back
        // the first match — publishing one field's bounds for another. When the two happen to share
        // a dtype nothing downstream can detect the substitution, and these bounds are exact enough
        // for DataFusion to rewrite an aggregate into a literal on. An absent statistic costs an
        // optimisation; a confidently wrong one costs a correct answer.
        let mut matches = fields
            .names()
            .iter()
            .zip(fields.fields())
            .enumerate()
            .filter(|(_, (field_name, _))| field_name.as_ref() == name);
        let Some((idx, (_, field_dtype))) = matches.next() else {
            return Ok(StatsSet::default());
        };
        if matches.next().is_some() {
            return Ok(StatsSet::default());
        }
        let Some(stats) = statistics.stats_sets().get(idx) else {
            return Ok(StatsSet::default());
        };

        // The dtype is taken from the FILE's own field list, not from the statistics sidecar.
        //
        // `FileStatistics::get` returns a dtype alongside each set, but public construction allows
        // that to disagree with the file dtype this lookup resolved against. Deciding "is this a
        // float?" from the sidecar would let a mismatched one steer the gate below and publish exact
        // extrema for a float column. The authoritative answer is the field we actually matched.
        let mut stats = stats.clone();

        // Extrema whose NaN semantics differ from DataFusion's are withheld unless the column is
        // provably NaN-free.
        //
        // Vortex computes MIN/MAX and SUM skipping NaNs, while DataFusion orders NaN as an ordinary
        // value under Arrow's total order — where negative NaN sorts below every finite value and
        // positive NaN above them. Over `[1.0, NaN]`, Vortex records a maximum of 1.0, so the only
        // safe exact extrema are those accompanied by proof that the column contains no NaNs.
        //
        // Lists, structs, and extension storage are inspected recursively. `NaNCount` is not defined
        // for NaN-bearing composite logical dtypes, so they remain unproven and their extrema are
        // withheld rather than letting a floating leaf through unexamined.
        if !stats.is_nan_free(&field_dtype) {
            stats.clear(Stat::Min);
            stats.clear(Stat::Max);
        }

        // A zero NaN count is not enough to make a floating SUM exact. Vortex adds sequentially,
        // whereas Arrow/DataFusion use a differently parenthesised lane reduction. IEEE addition is
        // non-associative even for finite values, so no floating footer sum is published.
        if field_dtype.may_contain_nan() {
            stats.clear(Stat::Sum);
        }

        Ok(stats)
    }
}
