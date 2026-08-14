// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::expect_used)]
#![expect(clippy::panic)]
#![expect(clippy::tests_outside_test_module)]
#![expect(clippy::unwrap_used)]

//! Differential and boundary tests for fused OnPair output positioning.
//!
//! `onpair_shmem_4tpt_split8read` is the shipped oracle. The look-back kernel
//! receives the same codes and dictionaries, but derives output positions from
//! per-block descriptors rather than a `chunk_offsets` sidecar.
//!
//! The test is gated on `nvcc` availability via `#[vortex_cuda_macros::test]`;
//! when CUDA is unavailable it expands to `#[test] #[ignore]`.

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use futures::executor::block_on;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaSession;

const SEED: u64 = 0x5eed_fa57_0a11_9a1f;
const TOKEN_PAD: usize = 16;
const BATCH_TOKENS: usize = 128;
const WARPS_PER_BLOCK: u32 = 16;
const WARP_BUF_BYTES: u32 = 2080;
// Keep these test constants equal to LB_TICKET_SLOTS and LB_EPOCH_MASK + 1 in
// onpair_shmem_4tpt_split8read_lookback.cu.
const TICKET_SLOTS: usize = 16_384;
const EPOCH_TAG_PERIOD: u32 = 1 << 30;
const OUTPUT_GUARD: usize = 32;
const OUTPUT_SENTINEL: u8 = 0xa5;
const CONTEXT_BYTES: usize = 16;

// Code N has length N, including an explicit zero-length entry and every length
// accepted by the kernel. Lengths 9..=16 take the long-token fallback.
const DICT_LENS: [u8; 17] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

struct Dictionary {
    padded: Vec<u8>,
    split8: Vec<u8>,
    lens: Vec<u8>,
}

struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64: compact, deterministic, and independent of rand crate versions.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

fn build_dictionary(seed: u64) -> Dictionary {
    let mut rng = SeededRng::new(seed);
    let mut padded = vec![0u8; DICT_LENS.len() * TOKEN_PAD + TOKEN_PAD];
    for (code, &len) in DICT_LENS.iter().enumerate() {
        for byte in 0..usize::from(len) {
            let random = rng.next_u64() as u8;
            let value = random
                .wrapping_add((code as u8).wrapping_mul(29))
                .wrapping_add((byte as u8).wrapping_mul(17))
                .wrapping_add(1);
            // No decoded byte equals the sentinel, so an omitted output store
            // cannot accidentally compare equal to the initialized buffer.
            padded[code * TOKEN_PAD + byte] = if value == OUTPUT_SENTINEL {
                OUTPUT_SENTINEL - 1
            } else {
                value
            };
        }
    }

    let mut split8 = vec![0u8; DICT_LENS.len() * 8 + TOKEN_PAD];
    for (code, &len) in DICT_LENS.iter().enumerate() {
        let copied = usize::from(len).min(8);
        split8[code * 8..code * 8 + copied]
            .copy_from_slice(&padded[code * TOKEN_PAD..code * TOKEN_PAD + copied]);
    }

    Dictionary {
        padded,
        split8,
        lens: DICT_LENS.to_vec(),
    }
}

fn decoded_len(codes: &[u16], lens: &[u8]) -> usize {
    codes
        .iter()
        .map(|&code| usize::from(lens[usize::from(code)]))
        .sum()
}

fn chunk_offsets(codes: &[u16], lens: &[u8]) -> Vec<u64> {
    let batches = codes.len().div_ceil(BATCH_TOKENS);
    let mut offsets = Vec::with_capacity(batches + 1);
    offsets.push(0);
    let mut total = 0u64;
    for batch in codes.chunks(BATCH_TOKENS) {
        total += batch
            .iter()
            .map(|&code| u64::from(lens[usize::from(code)]))
            .sum::<u64>();
        offsets.push(total);
    }
    offsets
}

fn fill_mixed_batch(batch: &mut [u16], rng: &mut SeededRng) {
    for (token, code) in batch.iter_mut().enumerate() {
        let lane = token & 31;
        let round = token >> 5;
        let len = if (lane + round) % 4 == 0 {
            // Every 32-lane group sends some, but not all, lanes down the >8 B path.
            9 + ((lane + round + rng.index(8)) & 7)
        } else {
            rng.index(9)
        };
        *code = len as u16;
    }
}

fn single_mixed_batch(seed: u64) -> Vec<u16> {
    let mut codes = vec![0u16; BATCH_TOKENS];
    fill_mixed_batch(&mut codes, &mut SeededRng::new(seed));
    codes
}

fn boundary_stress_codes(seed: u64) -> Vec<u16> {
    // 4,101 full batches plus a one-token tail = 4,102 batches and 257 blocks.
    // This is deliberately not a multiple of 16 warps/block and crosses eight
    // 32-descriptor look-back windows.
    const FULL_BATCHES: usize = 16 * 256 + 5;
    let mut rng = SeededRng::new(seed);
    let mut codes = vec![0u16; FULL_BATCHES * BATCH_TOKENS + 1];

    // Alternating empty and one-nonempty-token batches advance the next output
    // base by exactly one byte. Their batch bases therefore cover all residues
    // modulo 16 while also exercising zero-total and one-byte batch aggregates.
    for residue in 0..16 {
        let batch = residue * 2 + 1;
        let token = rng.index(BATCH_TOKENS);
        codes[batch * BATCH_TOKENS + token] = 1;
    }

    for batch in 32..FULL_BATCHES {
        let start = batch * BATCH_TOKENS;
        fill_mixed_batch(&mut codes[start..start + BATCH_TOKENS], &mut rng);
    }

    // The final partial batch contains exactly one long token.
    codes[FULL_BATCHES * BATCH_TOKENS] = 16;
    codes
}

fn output_buffer(decoded_bytes: usize) -> Vec<u8> {
    vec![OUTPUT_SENTINEL; decoded_bytes + OUTPUT_GUARD]
}

fn launch_shipped(codes: &[u16], dict: &Dictionary) -> Vec<u8> {
    let decoded_bytes = decoded_len(codes, &dict.lens);
    if codes.is_empty() {
        return output_buffer(decoded_bytes);
    }

    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .expect("create CUDA execution context for shipped oracle");
    let offsets = chunk_offsets(codes, &dict.lens);
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.to_vec()).unwrap()).unwrap();
    let offsets_dev: BufferHandle = block_on(ctx.copy_to_device(offsets).unwrap()).unwrap();
    let split8_dev: BufferHandle =
        block_on(ctx.copy_to_device(dict.split8.clone()).unwrap()).unwrap();
    let padded_dev: BufferHandle =
        block_on(ctx.copy_to_device(dict.padded.clone()).unwrap()).unwrap();
    let lens_dev: BufferHandle = block_on(ctx.copy_to_device(dict.lens.clone()).unwrap()).unwrap();
    let output_dev: BufferHandle =
        block_on(ctx.copy_to_device(output_buffer(decoded_bytes)).unwrap()).unwrap();

    let batches = codes.len().div_ceil(BATCH_TOKENS);
    let cfg = LaunchConfig {
        grid_dim: (
            u32::try_from(batches.div_ceil(WARPS_PER_BLOCK as usize)).unwrap(),
            1,
            1,
        ),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let function = ctx
        .load_function("onpair_shmem_4tpt_split8read", &[])
        .expect("load shipped OnPair PTX");
    let codes_view = codes_dev.cuda_view::<u16>().unwrap();
    let offsets_view = offsets_dev.cuda_view::<u64>().unwrap();
    let split8_view = split8_dev.cuda_view::<u8>().unwrap();
    let padded_view = padded_dev.cuda_view::<u8>().unwrap();
    let lens_view = lens_dev.cuda_view::<u8>().unwrap();
    let output_view = output_dev.cuda_view::<u8>().unwrap();
    let total_tokens = codes.len() as u64;

    ctx.launch_kernel_config(&function, cfg, codes.len(), |args| {
        args.arg(&codes_view)
            .arg(&offsets_view)
            .arg(&split8_view)
            .arg(&padded_view)
            .arg(&lens_view)
            .arg(&output_view)
            .arg(&total_tokens);
    })
    .expect("launch shipped OnPair oracle");

    output_dev
        .as_device()
        .copy_to_host_sync(Alignment::of::<u8>())
        .expect("copy shipped output to host")
        .as_ref()
        .to_vec()
}

fn launch_lookback_sequence(codes: &[u16], dict: &Dictionary, epochs: &[u32]) -> Vec<Vec<u8>> {
    let decoded_bytes = decoded_len(codes, &dict.lens);
    if codes.is_empty() {
        return epochs
            .iter()
            .map(|_| output_buffer(decoded_bytes))
            .collect();
    }

    let mut ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .expect("create CUDA execution context for look-back kernel");
    let codes_dev: BufferHandle = block_on(ctx.copy_to_device(codes.to_vec()).unwrap()).unwrap();
    let split8_dev: BufferHandle =
        block_on(ctx.copy_to_device(dict.split8.clone()).unwrap()).unwrap();
    let padded_dev: BufferHandle =
        block_on(ctx.copy_to_device(dict.padded.clone()).unwrap()).unwrap();
    let lens_dev: BufferHandle = block_on(ctx.copy_to_device(dict.lens.clone()).unwrap()).unwrap();

    let batches = codes.len().div_ceil(BATCH_TOKENS);
    let blocks = batches.div_ceil(WARPS_PER_BLOCK as usize);
    let ticket_dev: BufferHandle =
        block_on(ctx.copy_to_device(vec![0u32; TICKET_SLOTS]).unwrap()).unwrap();
    let aggregate_dev: BufferHandle =
        block_on(ctx.copy_to_device(vec![0u64; blocks]).unwrap()).unwrap();
    let inclusive_dev: BufferHandle =
        block_on(ctx.copy_to_device(vec![0u64; blocks]).unwrap()).unwrap();
    let flags_dev: BufferHandle =
        block_on(ctx.copy_to_device(vec![0u32; blocks]).unwrap()).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (u32::try_from(blocks).unwrap(), 1, 1),
        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
        shared_mem_bytes: WARPS_PER_BLOCK * WARP_BUF_BYTES,
    };
    let function = ctx
        .load_function("onpair_shmem_4tpt_split8read_lookback", &[])
        .expect("load look-back OnPair PTX");
    let codes_view = codes_dev.cuda_view::<u16>().unwrap();
    let split8_view = split8_dev.cuda_view::<u8>().unwrap();
    let padded_view = padded_dev.cuda_view::<u8>().unwrap();
    let lens_view = lens_dev.cuda_view::<u8>().unwrap();
    let ticket_view = ticket_dev.cuda_view::<u32>().unwrap();
    let aggregate_view = aggregate_dev.cuda_view::<u64>().unwrap();
    let inclusive_view = inclusive_dev.cuda_view::<u64>().unwrap();
    let flags_view = flags_dev.cuda_view::<u32>().unwrap();
    let total_tokens = codes.len() as u64;

    epochs
        .iter()
        .map(|&epoch| {
            // A fresh sentinel-filled output per launch prevents an earlier correct
            // launch from hiding a later launch that omitted stores.
            let output_dev: BufferHandle =
                block_on(ctx.copy_to_device(output_buffer(decoded_bytes)).unwrap()).unwrap();
            let output_view = output_dev.cuda_view::<u8>().unwrap();
            ctx.launch_kernel_config(&function, cfg, codes.len(), |args| {
                args.arg(&codes_view)
                    .arg(&split8_view)
                    .arg(&padded_view)
                    .arg(&lens_view)
                    .arg(&output_view)
                    .arg(&total_tokens)
                    .arg(&ticket_view)
                    .arg(&aggregate_view)
                    .arg(&inclusive_view)
                    .arg(&flags_view)
                    .arg(&epoch);
            })
            .unwrap_or_else(|error| panic!("launch look-back OnPair at epoch {epoch}: {error}"));

            output_dev
                .as_device()
                .copy_to_host_sync(Alignment::of::<u8>())
                .unwrap_or_else(|error| {
                    panic!("copy look-back output at epoch {epoch} to host: {error}")
                })
                .as_ref()
                .to_vec()
        })
        .collect()
}

fn hex_context(bytes: &[u8], offset: usize) -> String {
    let start = offset.saturating_sub(CONTEXT_BYTES);
    let end = (offset + CONTEXT_BYTES + 1).min(bytes.len());
    let encoded = bytes[start..end]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("[{start}..{end}]: {encoded}")
}

fn assert_guard_untouched(case: &str, implementation: &str, output: &[u8], decoded_bytes: usize) {
    if let Some(relative) = output[decoded_bytes..]
        .iter()
        .position(|&byte| byte != OUTPUT_SENTINEL)
    {
        let offset = decoded_bytes + relative;
        panic!(
            "{case}: {implementation} modified output guard at byte offset {offset}; context {}",
            hex_context(output, offset)
        );
    }
}

fn assert_byte_exact(case: &str, epoch: u32, oracle: &[u8], lookback: &[u8], decoded_bytes: usize) {
    assert_eq!(
        oracle.len(),
        lookback.len(),
        "{case}: output allocation lengths differ at epoch {epoch}"
    );

    if let Some(offset) = oracle[..decoded_bytes]
        .iter()
        .zip(&lookback[..decoded_bytes])
        .position(|(expected, actual)| expected != actual)
    {
        panic!(
            "{case}: first mismatching byte offset {offset} at epoch {epoch}, seed {SEED:#018x}; \
             shipped=0x{:02x}, lookback=0x{:02x}; shipped context {}; lookback context {}",
            oracle[offset],
            lookback[offset],
            hex_context(&oracle[..decoded_bytes], offset),
            hex_context(&lookback[..decoded_bytes], offset),
        );
    }

    assert_guard_untouched(case, "shipped oracle", oracle, decoded_bytes);
    assert_guard_untouched(case, "look-back kernel", lookback, decoded_bytes);
}

fn run_differential_case(case: &str, codes: &[u16], dict: &Dictionary, epochs: &[u32]) {
    let decoded_bytes = decoded_len(codes, &dict.lens);
    let oracle = launch_shipped(codes, dict);
    for (&epoch, lookback) in epochs
        .iter()
        .zip(launch_lookback_sequence(codes, dict, epochs))
    {
        assert_byte_exact(case, epoch, &oracle, &lookback, decoded_bytes);
    }
}

#[vortex_cuda_macros::test]
fn lookback_matches_shipped_for_boundary_batches() {
    let dict = build_dictionary(SEED);
    let stress = boundary_stress_codes(SEED ^ 0x626f_756e_6461_7279);
    let stress_offsets = chunk_offsets(&stress, &dict.lens);
    let mut residues = [false; 16];
    for &offset in stress_offsets.iter().take(32) {
        residues[offset as usize & 15] = true;
    }
    assert!(residues.into_iter().all(|covered| covered));
    assert_ne!(
        stress.len().div_ceil(BATCH_TOKENS) % WARPS_PER_BLOCK as usize,
        0
    );

    let cases = [
        ("empty input", Vec::new()),
        ("one short token", vec![1]),
        ("one long token", vec![16]),
        (
            "single mixed batch",
            single_mixed_batch(SEED ^ 0x7369_6e67_6c65),
        ),
    ];
    for (case, codes) in cases {
        run_differential_case(case, &codes, &dict, &[1]);
    }

    // Repeated launches make stale descriptors visible and give an ordering race
    // many independent opportunities to perturb the byte-exact result.
    run_differential_case(
        "inter-block boundary stress",
        &stress,
        &dict,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    );
}

#[vortex_cuda_macros::test]
fn lookback_matches_shipped_across_counter_wrap_boundaries() {
    let dict = build_dictionary(SEED);
    let codes = boundary_stress_codes(SEED ^ 0x7772_6170);

    // The ticket selector is epoch & (16,384 - 1). These consecutive launches
    // cross from its last slot to its first, without reusing either live slot.
    run_differential_case(
        "ticket-ring selector wrap",
        &codes,
        &dict,
        &[TICKET_SLOTS as u32 - 1, TICKET_SLOTS as u32],
    );

    // Descriptor flags carry only 30 epoch bits. Passing the exact 2^30 boundary
    // catches a reader that compares the stored tag with the unmasked host epoch.
    run_differential_case(
        "30-bit descriptor epoch wrap",
        &codes,
        &dict,
        &[EPOCH_TAG_PERIOD - 1, EPOCH_TAG_PERIOD],
    );
}
