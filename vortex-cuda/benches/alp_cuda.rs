// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for ALP decompression.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

use std::f64;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::array_session;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALPArray;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::Exponents;
use vortex::encodings::alp::alp_encode;
use vortex::error::VortexExpect;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Patch frequencies to benchmark (as fractions).
const PATCH_FREQUENCIES: &[(f64, &str)] = &[(0.0, "0%"), (0.01, "1%"), (0.10, "10%")];

/// Exponents pinned for the benchmark rather than discovered by `find_best_exponents`.
///
/// A benchmark whose independent variable is the patch rate must not let the encoding
/// change underneath that variable. Left to the search, f32 picks `e: 8`, and
/// `255 * 10^8` overflows the 24-bit f32 mantissa, so 43 of the 256 base values fail to
/// round-trip and the nominal "0% patches" arm actually patches 16.8%.
///
/// `e - f == 2` mirrors two-decimal-place data (the common real-world case). Base values
/// are integers in `0..256`, so `v * 10^2 = 25_500` at most — exactly representable in
/// both f32 and f64 — while PI never round-trips. The patch set is therefore exactly the
/// set of scattered PI positions, by construction.
const BENCH_EXPONENTS: Exponents = Exponents { e: 2, f: 0 };

/// Finalizer of SplitMix64 — a cheap, well-distributed integer hash.
///
/// Used to scatter patch positions deterministically without allocating a position set.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Create an ALP-encoded array of `len` floats with the requested patch frequency.
///
/// Base values are integers in `0..256`, which round-trip exactly under
/// [`BENCH_EXPONENTS`]. When `patch_frequency > 0`, PI is scattered through the array; PI
/// never round-trips, so the patch set is exactly the set of PI positions and the realized
/// patch rate matches the label to within ~0.03%.
///
/// Two things are deliberate here, and both were needed to make the patch rate mean what
/// it says:
///
/// - **Exponents are pinned** rather than searched — see [`BENCH_EXPONENTS`].
/// - **Patch positions are hashed**, not placed at `i % interval == 0`. A fixed interval is
///   periodic, and `ALPFloat::find_best_exponents` samples 32 elements strided by
///   `len / 32`. When the interval divides that stride — as it does at `len = 100M` for
///   intervals of 100 and 10 — *every* sampled element is the outlier, so a search-chosen
///   encoding fits PI and the base values patch instead. Hashing removes the periodicity,
///   which keeps the array honest if the exponents are ever unpinned, and better reflects
///   real data, where exceptions are not evenly spaced. See vortex#919.
fn make_alp_array<T>(len: usize, patch_frequency: f64) -> ALPArray
where
    T: ALPFloat + NativePType,
{
    // Fraction of the u64 space below which an index is a patch position. Expected count is
    // `len * patch_frequency`; at these sizes the realized rate matches to within ~0.01%.
    let patch_threshold = if patch_frequency > 0.0 {
        (patch_frequency * (u64::MAX as f64)) as u64
    } else {
        0
    };
    let outlier = T::from(f64::consts::PI).unwrap();

    let values: Buffer<T> = (0..len)
        .map(|i| {
            if patch_threshold > 0 && splitmix64(i as u64) < patch_threshold {
                outlier
            } else {
                T::from((i % 256) as u32).unwrap()
            }
        })
        .collect();

    let primitive_array = PrimitiveArray::new(values, NonNullable);
    let encoded = alp_encode(
        primitive_array.as_view(),
        Some(BENCH_EXPONENTS),
        &mut array_session().create_execution_ctx(),
    )
    .vortex_expect("failed to ALP-encode array");

    if patch_frequency > 0.0 {
        assert!(
            encoded.patches().is_some(),
            "expected patches for patch_frequency={patch_frequency}",
        );
    }

    encoded
}

fn benchmark_alp_decode_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: ALPFloat + NativePType + DeviceRepr,
{
    let mut group = c.benchmark_group("cuda");

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        for &(patch_freq, patch_label) in PATCH_FREQUENCIES {
            let array = make_alp_array::<T>(len, patch_freq);

            group.bench_with_input(
                BenchmarkId::new(format!("cuda/alp_{}/{}", type_name, patch_label), len_str),
                &array,
                |b, array| {
                    b.iter_custom(|iters| {
                        let timed = TimedLaunchStrategy::default();
                        let timer = timed.timer();

                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                                .vortex_expect("failed to create execution context")
                                .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                                .with_launch_strategy(Arc::new(timed));

                        for _ in 0..iters {
                            block_on(array.clone().into_array().execute_cuda(&mut cuda_ctx))
                                .unwrap();
                        }

                        Duration::from_nanos(timer.load(Ordering::Relaxed))
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_alp_decode(c: &mut Criterion) {
    benchmark_alp_decode_typed::<f32>(c, "f32");
    benchmark_alp_decode_typed::<f64>(c, "f64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_alp_decode
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
