// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unified-comparison harness: runs the Vortex FSST LIKE matcher (Teddy prefilter
//! + DFA verify) over a real newline-separated text column, with a shared pattern
//! set, printing match counts + median match time + throughput. Mirrors the TUM
//! `measure_clickbench` harness so the two can be compared on identical data.
//!
//! Env: CB_URLS_FILE=<path> [CB_ITERS=7] [CB_PATTERNS=%google%,%yandex%,...]

#![expect(clippy::unwrap_used)]

use std::time::Instant;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_buffer::BitBuffer;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

fn main() {
    let path = std::env::var("CB_URLS_FILE").expect("set CB_URLS_FILE");
    let iters: usize = std::env::var("CB_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let patterns: Vec<String> = std::env::var("CB_PATTERNS")
        .ok()
        .map(|s| s.split(',').map(|x| x.to_string()).collect())
        .unwrap_or_else(|| {
            ["%google%", "%yandex%", "%.ru%", "%http%"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        });

    let data = std::fs::read_to_string(&path).unwrap();
    let urls: Vec<&str> = data.lines().collect();
    let n = urls.len();
    let uncompressed: usize = urls.iter().map(|s| s.len()).sum();
    eprintln!("rows={n} uncompressed_bytes={uncompressed}");

    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    let session = VortexSession::empty().with::<ArraySession>();
    let fsst = fsst_compress(
        varbin,
        len,
        &dtype,
        &compressor,
        &mut session.create_execution_ctx(),
    );
    eprintln!("fsst built; compressed≈{} bytes", fsst.nbytes());

    let arr = fsst.into_array();
    let mb = uncompressed as f64 / 1e6;

    println!("pattern,algorithm,count,match_ms_median,match_MBps,rows_per_s");
    for pat in &patterns {
        let pattern = ConstantArray::new(pat.as_str(), n).into_array();

        // correctness cross-check: count matches once
        let mut ctx = session.create_execution_ctx();
        let count = Like
            .try_new_array(n, LikeOptions::default(), [arr.clone(), pattern.clone()])
            .unwrap()
            .into_array()
            .execute::<BitBuffer>(&mut ctx)
            .unwrap()
            .true_count();

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut ctx = session.create_execution_ctx();
            let like = Like
                .try_new_array(n, LikeOptions::default(), [arr.clone(), pattern.clone()])
                .unwrap()
                .into_array();
            let t = Instant::now();
            let r = like.execute::<BitBuffer>(&mut ctx).unwrap();
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(&r);
            times.push(dt);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mt = times[times.len() / 2];
        println!(
            "{pat},Vortex-Teddy,{count},{mt:.3},{:.1},{:.0}",
            mb / (mt / 1000.0),
            n as f64 / (mt / 1000.0)
        );
    }
}
