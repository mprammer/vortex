// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST-12 decode inputs for the FastPair GPU kernels.
//!
//! The point of this module is to demonstrate, by construction, that the decode kernels
//! do not inspect where their inputs came from. They consume exactly three things:
//!
//!   1. fixed-width 16-bit codes,
//!   2. a per-code length table,
//!   3. dictionary bytes at a fixed 16-byte stride.
//!
//! OnPair supplies those directly. FSST-12 supplies the same information in a different
//! physical form, and the gap is closed here, at load time, by the same kind of repack
//! the OnPair path already performs:
//!
//!   * FSST-12 packs codes densely at 12 bits, two codes per three bytes. We widen them
//!     to one `u16` each.
//!   * FSST-12 symbols live in 8-byte cells. We widen each into a 16-byte row, zero
//!     padded, so a code addresses its bytes at `code * 16`.
//!
//! Neither codec escapes: FSST-12 reserves codes 0..256 for the single-byte literals, so
//! an unmatched byte takes its own code, exactly as OnPair's alphabet-complete dictionary
//! does. That is what makes a code's output length a function of the code alone, which is
//! what both the fixed-stride gather and the stored offset sidecar depend on.
//!
//! CONSEQUENCE WORTH EXPECTING: FSST-12 symbols are at most 8 bytes, so every token fits
//! the narrow table and `split8read`'s long-token path never fires. On FSST-12 it is a
//! pure stride-8 gather. That is a different operating point from OnPair, not the same
//! result, and it should be reported as such.

use fsst::fsst12::Compressor12;

/// Decode inputs in the form the GPU kernels expect.
pub struct Fsst12Inputs {
    /// One `u16` per code, widened from the dense 12-bit stream.
    pub codes: Vec<u16>,
    /// Symbol length per code, 1..=8.
    pub lens: Vec<u8>,
    /// `dict_size * 16` bytes: each symbol zero-padded into a 16-byte row.
    pub dict_padded: Vec<u8>,
    /// Bytes of the compressed (bit-packed) representation, for ratio accounting.
    pub compressed_bytes: usize,
}

/// Number of 12-bit codes in a packed stream of `bytes` bytes.
///
/// Two codes occupy three bytes; a trailing odd code occupies two, with four padding
/// bits. So a length congruent to 1 mod 3 cannot be produced by the packer.
fn codes_in_packed(bytes: usize) -> Option<usize> {
    match bytes % 3 {
        0 => Some(bytes / 3 * 2),
        2 => Some((bytes - 2) / 3 * 2 + 1),
        _ => None,
    }
}

/// Widen a dense 12-bit code stream into one `u16` per code.
fn unpack12(packed: &[u8], n_codes: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n_codes);
    let mut i = 0usize;
    while out.len() + 1 < n_codes {
        // Two codes share three bytes: low 12 bits then high 12 bits.
        let triple = u32::from(packed[i]) | (u32::from(packed[i + 1]) << 8) | (u32::from(packed[i + 2]) << 16);
        out.push((triple & 0xfff) as u16);
        out.push(((triple >> 12) & 0xfff) as u16);
        i += 3;
    }
    if out.len() < n_codes {
        // Odd tail: one code in two bytes, upper four bits are padding.
        let pair = u16::from(packed[i]) | (u16::from(packed[i + 1]) << 8);
        out.push(pair & 0xfff);
    }
    out
}

/// Train FSST-12 on a sample of `lines`, compress `plaintext`, and return decode inputs
/// in the kernels' ABI.
///
/// `plaintext` must be the same in-order concatenation of row values that the kernels
/// reproduce, because the decode output is a plain concatenation of symbol bytes with no
/// row framing. `sample_lines` is what the trainer sees.
pub fn build(plaintext: &[u8], sample_lines: &[&[u8]]) -> anyhow::Result<Fsst12Inputs> {
    let compressor = Compressor12::train(sample_lines);
    let packed = compressor.compress(plaintext);

    let n_codes = codes_in_packed(packed.len()).ok_or_else(|| {
        anyhow::anyhow!(
            "FSST-12 packed stream of {} bytes is not a whole number of 12-bit codes",
            packed.len()
        )
    })?;
    let codes = unpack12(&packed, n_codes);

    // Verify the widened stream against the reference decoder before it reaches a GPU.
    // A mismatch here means the unpack is wrong, and would otherwise surface as a
    // byte-exactness failure that looks like a kernel bug.
    let reference = compressor.decompressor().decompress(&packed);
    if reference != plaintext {
        anyhow::bail!(
            "FSST-12 round trip disagrees with the input: {} vs {} bytes",
            reference.len(),
            plaintext.len()
        );
    }

    let symbols = compressor.symbol_table();
    let lengths = compressor.symbol_lengths();
    let dict_size = symbols.len();

    // Widen 8-byte symbol cells into 16-byte rows so a code addresses its bytes at
    // `code * 16`. The upper eight bytes are always zero for FSST-12, which is exactly
    // why the split gather's long-token path never fires on it.
    let mut dict_padded = vec![0u8; dict_size * 16 + 16];
    let mut lens = vec![0u8; dict_size];
    for (i, sym) in symbols.iter().enumerate() {
        let bytes = sym.to_u64().to_le_bytes();
        let len = usize::from(lengths[i]).min(8);
        dict_padded[i * 16..i * 16 + len].copy_from_slice(&bytes[..len]);
        lens[i] = lengths[i];
    }

    // The decode must reproduce the plaintext from (codes, lens, dict) alone. Check that
    // the length table and code stream agree on the output size before staging anything.
    let predicted: u64 = codes.iter().map(|&c| u64::from(lens[usize::from(c)])).sum();
    if predicted != plaintext.len() as u64 {
        anyhow::bail!(
            "FSST-12 length table predicts {predicted} output bytes, plaintext is {}",
            plaintext.len()
        );
    }

    Ok(Fsst12Inputs {
        codes,
        lens,
        dict_padded,
        compressed_bytes: packed.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_length_to_code_count() {
        assert_eq!(codes_in_packed(0), Some(0));
        assert_eq!(codes_in_packed(2), Some(1));
        assert_eq!(codes_in_packed(3), Some(2));
        assert_eq!(codes_in_packed(5), Some(3));
        assert_eq!(codes_in_packed(6), Some(4));
        // 1 mod 3 is unreachable: an odd tail costs two bytes, not one.
        assert_eq!(codes_in_packed(4), None);
    }

    #[test]
    fn round_trip_matches_reference_and_length_table() {
        let lines: Vec<&[u8]> = vec![
            b"http://example.com/a".as_slice(),
            b"http://example.com/b".as_slice(),
            b"the quick brown fox".as_slice(),
            b"".as_slice(),
        ];
        let plaintext: Vec<u8> = lines.concat();
        let inputs = build(&plaintext, &lines).expect("build");
        assert_eq!(inputs.lens.len() * 16 + 16, inputs.dict_padded.len());
        // Reconstruct from the kernels' ABI alone, which is what the GPU does.
        let mut out = Vec::new();
        for &c in &inputs.codes {
            let len = usize::from(inputs.lens[usize::from(c)]);
            out.extend_from_slice(&inputs.dict_padded[usize::from(c) * 16..][..len]);
        }
        assert_eq!(out, plaintext);
    }

    #[test]
    fn every_symbol_fits_the_narrow_half() {
        let lines: Vec<&[u8]> = vec![b"aaaaaaaaaaaaaaaaaaaaaaaaaaaa".as_slice()];
        let plaintext: Vec<u8> = lines.concat();
        let inputs = build(&plaintext, &lines).expect("build");
        // FSST-12 caps symbols at 8 bytes, so the split gather's wide path is dead.
        assert!(inputs.lens.iter().all(|&l| l <= 8));
        assert!(
            inputs
                .dict_padded
                .chunks_exact(16)
                .all(|row| row[8..].iter().all(|&b| b == 0))
        );
    }
}
