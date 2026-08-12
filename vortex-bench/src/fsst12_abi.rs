//! Normalize FSST-12 into the FastPair decode ABI.
//!
//! The decode kernels consume three buffers and never inspect what produced them:
//!
//!   1. `codes`       -- fixed-width `u16`, one per token
//!   2. `lens[code]`  -- that code's true token length
//!   3. `dict`        -- token bytes at a fixed 16-byte stride, addressed by `code * 16`
//!
//! OnPair reaches that shape through a host repack of its offset-table dictionary.
//! FSST-12 reaches the same shape through the repack implemented here. Neither the
//! kernels nor the sidecar construction below them observe the difference, which is the
//! whole content of the paper's shared-decode-ABI claim.
//!
//! FSST-12 differs from OnPair's stored form in exactly two ways, both resolved here:
//!
//!   * codes are packed densely, two 12-bit codes per three bytes, rather than as `u16`;
//!   * symbols are `u64`s of at most eight bytes, rather than 16-byte-max token bytes.
//!
//! Correctness rests on FSST-12 being escape-free. Its first 256 codes are the identity
//! single-byte symbols, so every input byte has a code and nothing in the stream is a
//! literal -- which is what makes a code's index sufficient to know what to read, and
//! therefore what makes the output-offset sidecar computable at write time.
//!
//! This module is deliberately free of any `cuda` gating: it is pure host-side data
//! movement, and it is unit-tested on machines that have no GPU.

use fsst12::fsst12::Compressor12;

/// Byte stride of the padded decode table, matching the kernels' `dict + code * 16`.
pub const DICT_STRIDE: usize = 16;

/// 12-bit code mask and the shift separating the two codes packed in a 3-byte triple.
const CODE12_MASK: u32 = 0x0FFF;
const CODE12_SHIFT: u32 = 12;

/// Number of codes the 12-bit space can address.
const FSST12_CODE_SPACE: usize = 1 << 12;

/// An FSST-12 column expressed in the decode ABI.
#[derive(Debug, Clone)]
pub struct Fsst12Abi {
    /// One `u16` per token; values are 12-bit, so the high nibble is always zero.
    pub codes: Vec<u16>,
    /// `lens[code]`, in `1..=8`. Sized to the full code space, not the trained table.
    pub lens: Vec<u8>,
    /// `dict[code * 16 .. code * 16 + 8]` is the token; the upper half is zero and is
    /// never read, because no FSST-12 symbol exceeds eight bytes. Carries the same
    /// trailing 16-byte pad the OnPair path leaves, so a wide load on the final entry
    /// has slack.
    pub dict_padded: Vec<u8>,
    /// Sum of `lens[c]` over the code stream.
    pub decoded_bytes: usize,
    /// Size of the trained symbol table, including the 256 reserved singletons.
    pub table_entries: usize,
}

/// Errors that mean the payload is not a valid FSST-12 stream, rather than that decoding
/// went wrong. Every one of these is unreachable for output of `Compressor12::compress`.
#[derive(Debug, thiserror::Error)]
pub enum Fsst12AbiError {
    #[error("invalid FSST-12 packed length {0} (expected 0 or 2 mod 3)")]
    PackedLength(usize),
    #[error("symbol table has {symbols} symbols but {lengths} lengths")]
    TableMismatch { symbols: usize, lengths: usize },
    #[error("FSST-12 table size {0} outside [256, 4096]")]
    TableSize(usize),
    #[error("code {code} has length {len}, outside 1..=8")]
    SymbolLength { code: usize, len: u8 },
}

/// Unpack the dense 12-bit code stream.
///
/// Three bytes carry two codes, low code first; a two-byte remainder carries one trailing
/// odd code. Any other remainder is not a valid payload. This mirrors the codec's own
/// `decompress_into`, including the tail rule -- if the two ever disagree, the byte-exact
/// check in `tests` below fails rather than a kernel silently emitting wrong bytes.
pub fn unpack_codes(compressed: &[u8]) -> Result<Vec<u16>, Fsst12AbiError> {
    if compressed.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(compressed.len() % 3, 0 | 2) {
        return Err(Fsst12AbiError::PackedLength(compressed.len()));
    }

    let n_triples = compressed.len() / 3;
    let has_odd = compressed.len() % 3 == 2;
    let mut codes = Vec::with_capacity(n_triples * 2 + usize::from(has_odd));

    for triple in compressed[..n_triples * 3].chunks_exact(3) {
        let raw = (triple[0] as u32) | ((triple[1] as u32) << 8) | ((triple[2] as u32) << 16);
        codes.push((raw & CODE12_MASK) as u16);
        codes.push(((raw >> CODE12_SHIFT) & CODE12_MASK) as u16);
    }
    if has_odd {
        let tail = &compressed[n_triples * 3..];
        let raw = (tail[0] as u32) | ((tail[1] as u32) << 8);
        codes.push((raw & CODE12_MASK) as u16);
    }
    Ok(codes)
}

/// Widen an FSST-12 symbol table into the padded, code-addressed decode table.
///
/// Both tables are sized to the full 4,096-code space rather than to the trained table,
/// so a code outside the trained range reads zeroes at length zero instead of running off
/// the end -- the kernels index without a bounds check.
pub fn widen_table(
    symbols: &[fsst12::Symbol],
    lengths: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), Fsst12AbiError> {
    if symbols.len() != lengths.len() {
        return Err(Fsst12AbiError::TableMismatch {
            symbols: symbols.len(),
            lengths: lengths.len(),
        });
    }
    if symbols.len() < 256 || symbols.len() > FSST12_CODE_SPACE {
        return Err(Fsst12AbiError::TableSize(symbols.len()));
    }

    let mut dict = vec![0u8; FSST12_CODE_SPACE * DICT_STRIDE + DICT_STRIDE];
    let mut lens = vec![0u8; FSST12_CODE_SPACE];

    for (code, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
        if len == 0 || len > 8 {
            return Err(Fsst12AbiError::SymbolLength { code, len });
        }
        let bytes = sym.to_u64().to_le_bytes();
        dict[code * DICT_STRIDE..code * DICT_STRIDE + 8].copy_from_slice(&bytes);
        lens[code] = len;
    }
    Ok((dict, lens))
}

/// Normalize one compressed FSST-12 buffer into the decode ABI.
pub fn normalize(
    compressor: &Compressor12,
    compressed: &[u8],
) -> Result<Fsst12Abi, Fsst12AbiError> {
    let codes = unpack_codes(compressed)?;
    let (dict_padded, lens) = widen_table(compressor.symbol_table(), compressor.symbol_lengths())?;
    let decoded_bytes = codes.iter().map(|&c| lens[c as usize] as usize).sum();
    Ok(Fsst12Abi {
        codes,
        lens,
        dict_padded,
        decoded_bytes,
        table_entries: compressor.symbol_table().len(),
    })
}

/// An FSST-12 column compressed row by row, in the decode ABI.
///
/// The whole-buffer form of [`normalize`] produces a denser code stream but destroys row
/// boundaries: the 12-bit packing carries no row delimiter, so there is no way to recover
/// where row `i`'s codes begin. OnPair's stored form keeps per-row code offsets, and the
/// bench's early-materialization and row-decode paths read them, so FSST-12 has to keep
/// them too or it is not comparable on layout -- only on rate.
///
/// The cost of keeping them is real and must be disclosed: each row's code stream is
/// independently 12-bit packed, so a row with an odd code count wastes up to 1.5 bytes.
/// On a column of many short rows that is a measurable ratio penalty against the
/// whole-buffer form. It is also what a random-access string codec actually stores, which
/// is the operating point FSST is designed for.
#[derive(Debug, Clone)]
pub struct Fsst12RowsAbi {
    /// Concatenated per-row codes, in row order.
    pub abi: Fsst12Abi,
    /// `row_code_offsets[i]..row_code_offsets[i+1]` indexes `abi.codes` for row `i`.
    /// Length is `rows + 1`.
    pub row_code_offsets: Vec<u64>,
    /// Sum of the per-row compressed lengths, i.e. what this form actually stores.
    pub compressed_bytes: usize,
}

/// Compress and normalize row by row, preserving row boundaries in code space.
pub fn normalize_rows(
    compressor: &Compressor12,
    rows: &[&[u8]],
) -> Result<Fsst12RowsAbi, Fsst12AbiError> {
    let per_row = compressor.compress_bulk(rows);
    let (dict_padded, lens) = widen_table(compressor.symbol_table(), compressor.symbol_lengths())?;

    let mut codes: Vec<u16> = Vec::new();
    let mut row_code_offsets: Vec<u64> = Vec::with_capacity(rows.len() + 1);
    let mut compressed_bytes = 0usize;
    row_code_offsets.push(0);
    for packed in &per_row {
        compressed_bytes += packed.len();
        codes.extend_from_slice(&unpack_codes(packed)?);
        row_code_offsets.push(codes.len() as u64);
    }

    let decoded_bytes = codes.iter().map(|&c| lens[c as usize] as usize).sum();
    Ok(Fsst12RowsAbi {
        abi: Fsst12Abi {
            codes,
            lens,
            dict_padded,
            decoded_bytes,
            table_entries: compressor.symbol_table().len(),
        },
        row_code_offsets,
        compressed_bytes,
    })
}

/// Synthesize OnPair's packed dictionary directory, `(offset << 16) | length`, over a
/// contiguous dictionary-bytes buffer.
///
/// The compact-layout kernels read this instead of the padded table, so FSST-12 needs an
/// equivalent to exercise them. Returns the directory and the contiguous bytes it indexes.
pub fn compact_dict(
    symbols: &[fsst12::Symbol],
    lengths: &[u8],
) -> Result<(Vec<u64>, Vec<u8>), Fsst12AbiError> {
    if symbols.len() != lengths.len() {
        return Err(Fsst12AbiError::TableMismatch {
            symbols: symbols.len(),
            lengths: lengths.len(),
        });
    }
    let mut table = Vec::with_capacity(FSST12_CODE_SPACE);
    let mut bytes: Vec<u8> = Vec::with_capacity(symbols.len() * 8);
    for (code, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
        if len == 0 || len > 8 {
            return Err(Fsst12AbiError::SymbolLength { code, len });
        }
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&sym.to_u64().to_le_bytes()[..len as usize]);
        table.push((off << 16) | len as u64);
    }
    // Untrained codes: zero length at the end of the buffer, so a stray read yields
    // nothing rather than another entry's bytes.
    let end = bytes.len() as u64;
    table.resize(FSST12_CODE_SPACE, end << 16);
    Ok((table, bytes))
}

/// Decode through the ABI exactly as a kernel lane does: copy a fixed sixteen bytes from
/// `dict + code * 16`, then advance the cursor by the true length.
///
/// Written to mirror the kernel rather than to be the fastest correct decoder. A
/// length-exact copy would hide a wrong padded upper half; the over-copy exposes it.
pub fn decode_via_abi(abi: &Fsst12Abi) -> Vec<u8> {
    let mut out = vec![0u8; abi.decoded_bytes + DICT_STRIDE];
    let mut cursor = 0usize;
    for &c in &abi.codes {
        let src = c as usize * DICT_STRIDE;
        out[cursor..cursor + DICT_STRIDE].copy_from_slice(&abi.dict_padded[src..src + DICT_STRIDE]);
        cursor += abi.lens[c as usize] as usize;
    }
    out.truncate(cursor);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Vec<u8>> {
        // Mixed lengths, repeated phrases to give the trainer pairs to merge, and every
        // byte value so the reserved single-byte codes are exercised.
        let mut rows: Vec<Vec<u8>> = Vec::new();
        for i in 0..2000u32 {
            rows.push(format!("http://example.com/path/{i}?q=value&lang=en").into_bytes());
            rows.push(format!("the quick brown fox jumps over the lazy dog {i}").into_bytes());
            rows.push(vec![(i % 256) as u8; 1 + (i % 9) as usize]);
        }
        rows.push((0u8..=255).collect());
        rows
    }

    fn train_and_compress(rows: &[Vec<u8>]) -> (Compressor12, Vec<u8>, Vec<u8>) {
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let flat: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let compressed = compressor.compress(&flat);
        (compressor, flat, compressed)
    }

    /// The load-bearing test: the ABI decode must equal both the codec's own
    /// decompressor AND the original bytes. Checking only against the decompressor
    /// would let a misreading shared by both implementations pass.
    #[test]
    fn abi_decode_is_byte_exact() {
        let rows = corpus();
        let (compressor, flat, compressed) = train_and_compress(&rows);

        let abi = normalize(&compressor, &compressed).expect("normalize");
        let via_abi = decode_via_abi(&abi);
        let via_codec = compressor.decompressor().decompress(&compressed);

        assert_eq!(via_abi.len(), flat.len(), "ABI decode length");
        assert_eq!(via_abi, via_codec, "ABI decode vs codec decompressor");
        assert_eq!(via_abi, flat, "ABI decode vs original bytes");
        assert_eq!(abi.decoded_bytes, flat.len(), "predicted decoded length");
    }

    /// Every FSST-12 symbol fits the narrow half, so the split dictionary's long-token
    /// fallback is unreachable for this codec. Asserted rather than assumed, because the
    /// paper says so in prose and a future codec change would otherwise silently break it.
    #[test]
    fn no_symbol_exceeds_the_narrow_half() {
        let rows = corpus();
        let (compressor, _, _) = train_and_compress(&rows);
        assert!(
            compressor.symbol_lengths().iter().all(|&l| (1..=8).contains(&l)),
            "FSST-12 symbols must all fit the 8-byte narrow table"
        );
    }

    /// Per-row form: concatenated decode must still equal the concatenated input, and the
    /// row offsets must actually delimit rows -- decoding one row's slice in isolation has
    /// to reproduce that row.
    #[test]
    fn row_form_is_byte_exact_and_row_addressable() {
        let rows = corpus();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let flat: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();

        let rowsabi = normalize_rows(&compressor, &refs).expect("normalize_rows");
        assert_eq!(rowsabi.row_code_offsets.len(), refs.len() + 1);
        assert_eq!(decode_via_abi(&rowsabi.abi), flat, "concatenated decode");

        // Spot-check row addressability at both ends and in the middle: a row's code
        // slice must decode to exactly that row.
        for &i in &[0usize, 1, refs.len() / 2, refs.len() - 1] {
            let lo = rowsabi.row_code_offsets[i] as usize;
            let hi = rowsabi.row_code_offsets[i + 1] as usize;
            let slice = Fsst12Abi {
                codes: rowsabi.abi.codes[lo..hi].to_vec(),
                lens: rowsabi.abi.lens.clone(),
                dict_padded: rowsabi.abi.dict_padded.clone(),
                decoded_bytes: rowsabi.abi.codes[lo..hi]
                    .iter()
                    .map(|&c| rowsabi.abi.lens[c as usize] as usize)
                    .sum(),
                table_entries: rowsabi.abi.table_entries,
            };
            assert_eq!(decode_via_abi(&slice), refs[i], "row {i} decodes in isolation");
        }
    }

    /// The compact directory must address the same bytes the padded table holds.
    #[test]
    fn compact_dict_agrees_with_padded_table() {
        let rows = corpus();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let (table, bytes) =
            compact_dict(compressor.symbol_table(), compressor.symbol_lengths()).expect("compact");
        let (padded, lens) =
            widen_table(compressor.symbol_table(), compressor.symbol_lengths()).expect("widen");

        for code in 0..compressor.symbol_table().len() {
            let e = table[code];
            let off = (e >> 16) as usize;
            let len = (e & 0xffff) as usize;
            assert_eq!(len, lens[code] as usize, "code {code} length");
            assert_eq!(
                &bytes[off..off + len],
                &padded[code * DICT_STRIDE..code * DICT_STRIDE + len],
                "code {code} bytes"
            );
        }
    }

    #[test]
    fn rejects_invalid_packed_length() {
        // 1 mod 3 cannot be produced by a valid encode: codes come in 3-byte pairs with
        // an optional 2-byte odd tail.
        assert!(matches!(
            unpack_codes(&[0u8; 4]),
            Err(Fsst12AbiError::PackedLength(4))
        ));
        assert!(unpack_codes(&[]).expect("empty is valid").is_empty());
    }

    /// Tables are sized to the code space, not the trained table, so an untrained code
    /// is in bounds and contributes nothing.
    #[test]
    fn tables_cover_the_whole_code_space() {
        let rows = corpus();
        let (compressor, _, _) = train_and_compress(&rows);
        let abi = normalize(&compressor, &compressor.compress(b"abc")).expect("normalize");
        assert_eq!(abi.lens.len(), FSST12_CODE_SPACE);
        assert!(abi.dict_padded.len() >= FSST12_CODE_SPACE * DICT_STRIDE);
        assert_eq!(abi.lens[FSST12_CODE_SPACE - 1], 0, "untrained code has length 0");
    }
}
