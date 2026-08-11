// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared zone-map fixtures for the pruning-rule tests; each rule module owns its own cases.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::BoundExpression;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;
use vortex_layout::segments::SegmentSink;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;

use crate::aggregate_fn::GeometryAabb;
use crate::aggregate_fn::LegacyGeometryAabb;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::test_harness::spatial_session;

/// A single-column zone map holding one native-box AABB stat row (`[xmin, ymin, xmax, ymax]`)
/// per zone, with default (unreferenced) metadata to match the aggregate's return dtype.
pub(super) fn aabb_zone_map(point_dtype: &DType, boxes: &[[f64; 4]]) -> VortexResult<ZoneMap> {
    let aabb_fn = GeometryAabb.bind(EmptyOptions);
    aabb_zone_map_with_aggregate(point_dtype, boxes, aabb_fn)
}

/// A zone map using the empty-metadata, NaN-skipping AABB semantics from older files.
pub(super) fn legacy_aabb_zone_map(
    point_dtype: &DType,
    boxes: &[[f64; 4]],
) -> VortexResult<ZoneMap> {
    let aabb_fn = LegacyGeometryAabb.bind(EmptyOptions);
    aabb_zone_map_with_aggregate(point_dtype, boxes, aabb_fn)
}

fn aabb_zone_map_with_aggregate(
    point_dtype: &DType,
    boxes: &[[f64; 4]],
    aabb_fn: vortex_array::aggregate_fn::AggregateFnRef,
) -> VortexResult<ZoneMap> {
    let col = |i: usize| PrimitiveArray::from_iter(boxes.iter().map(move |b| b[i])).into_array();
    let storage = StructArray::try_new(
        ["xmin", "ymin", "xmax", "ymax"].into(),
        vec![col(0), col(1), col(2), col(3)],
        boxes.len(),
        Validity::AllValid,
    )?
    .into_array();
    let box_dtype =
        ExtDType::<Rect>::try_new(SpatialMetadata::default(), storage.dtype().clone())?.erased();
    let aabbs = ExtensionArray::try_new(box_dtype, storage)?.into_array();
    let zone_array = StructArray::from_fields(&[(aabb_fn.to_string().as_str(), aabbs)])?;
    ZoneMap::try_new(
        point_dtype.clone(),
        zone_array,
        Arc::new([aabb_fn]),
        1,
        boxes.len() as u64,
    )
}

/// Write a one-zone layout through the production zoned strategy, reopen it, and ask the zoned
/// reader for the row mask proved by `predicate`.
pub(super) fn zoned_pruning_mask(
    column: &ArrayRef,
    predicate: &BoundExpression,
) -> VortexResult<Mask> {
    let column = column.clone();
    let predicate = predicate.clone();
    block_on(|handle| async move {
        let row_count = u64::try_from(column.len())?;
        let block_size = NonZeroUsize::new(column.len())
            .ok_or_else(|| vortex_err!("zoned pruning test requires a non-empty column"))?;
        let session = spatial_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        let segments = Arc::new(TestSegments::default());
        let (pointer, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size,
                ..Default::default()
            },
        );
        let segment_sink: Arc<dyn SegmentSink> = Arc::<TestSegments>::clone(&segments);
        let layout = strategy
            .write_stream(
                ArrayContext::empty().into(),
                segment_sink,
                column.to_array_stream().sequenced(pointer),
                eof,
                &session,
            )
            .await?;
        let segment_source: Arc<dyn SegmentSource> = segments;
        let reader = layout.new_reader(
            "nan-coordinate-zone".into(),
            segment_source,
            &session,
            &Default::default(),
        )?;
        reader
            .pruning_evaluation(&(0..row_count), &predicate, Mask::new_true(column.len()))?
            .await
    })
}

/// A two-zone zone map with no stat columns at all, as written before the AABB stat existed.
pub(super) fn empty_zone_map(point_dtype: &DType) -> VortexResult<ZoneMap> {
    ZoneMap::try_new(
        point_dtype.clone(),
        StructArray::try_new(FieldNames::empty(), vec![], 2, Validity::NonNullable)?,
        Arc::new([]),
        1,
        2,
    )
}
