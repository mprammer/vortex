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
use fsst12::fsst12::FSST12_MAX_SYMBOLS;
use fsst12::fsst12::FSST12_RESERVED_CODES;

/// Byte stride of the padded decode table, matching the kernels' `dict + code * 16`.
pub const DICT_STRIDE: usize = 16;

/// 12-bit code mask and the shift separating the two codes packed in a 3-byte triple.
const CODE12_MASK: u32 = 0x0FFF;
const CODE12_SHIFT: u32 = 12;

/// Number of codes the 12-bit space can address.
const FSST12_CODE_SPACE: usize = 1 << 12;

/// The reserved identity single-byte codes.
const FSST12_RESERVED: usize = 256;

// Local ABI constants must not drift from the canonical codec and kernel definitions.
const _: () = assert!(FSST12_CODE_SPACE == FSST12_MAX_SYMBOLS);
const _: () = assert!(FSST12_RESERVED == FSST12_RESERVED_CODES);
const _: () = assert!(DICT_STRIDE == vortex_onpair::MAX_TOKEN_SIZE);

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
    #[error(
        "reserved code {code} is not the identity single-byte symbol (len {len}, value {value:#x})"
    )]
    ReservedCode { code: usize, len: u8, value: u64 },
    #[error(
        "code {code} at stream position {pos} is outside the trained table of {entries} entries"
    )]
    UntrainedCode {
        code: u16,
        pos: usize,
        entries: usize,
    },
}

/// Validate an FSST-12 symbol table against the invariants the reference decompressor
/// asserts in its constructor.
///
/// Both table conversions below go through this, so neither can accept a table the codec
/// itself would reject. Without the reserved-code check a table that merely happens to be
/// the right size is accepted as FSST-12, and every downstream decode is then wrong in a
/// way no byte-exactness test on *our* pipeline would catch -- both sides would share the
/// misreading.
fn validate_table(symbols: &[fsst12::Symbol], lengths: &[u8]) -> Result<(), Fsst12AbiError> {
    if symbols.len() != lengths.len() {
        return Err(Fsst12AbiError::TableMismatch {
            symbols: symbols.len(),
            lengths: lengths.len(),
        });
    }
    if symbols.len() < FSST12_RESERVED || symbols.len() > FSST12_CODE_SPACE {
        return Err(Fsst12AbiError::TableSize(symbols.len()));
    }
    for (code, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
        if len == 0 || len > 8 {
            return Err(Fsst12AbiError::SymbolLength { code, len });
        }
        if code < FSST12_RESERVED && (len != 1 || sym.to_u64() != code as u64) {
            return Err(Fsst12AbiError::ReservedCode {
                code,
                len,
                value: sym.to_u64(),
            });
        }
    }
    Ok(())
}

/// Check that every code in a stream addresses a trained entry.
///
/// Untrained slots widen to length zero, so an out-of-range code would otherwise vanish
/// silently and understate `decoded_bytes` rather than fail. `Compressor12::compress` never
/// emits one, so this guards against a mis-unpacked stream, not against the codec.
fn validate_codes(codes: &[u16], entries: usize) -> Result<(), Fsst12AbiError> {
    for (pos, &code) in codes.iter().enumerate() {
        if code as usize >= entries {
            return Err(Fsst12AbiError::UntrainedCode { code, pos, entries });
        }
    }
    Ok(())
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
    validate_table(symbols, lengths)?;

    let mut dict = vec![0u8; FSST12_CODE_SPACE * DICT_STRIDE + DICT_STRIDE];
    let mut lens = vec![0u8; FSST12_CODE_SPACE];

    for (code, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
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
    let entries = compressor.symbol_table().len();
    validate_codes(&codes, entries)?;
    let decoded_bytes = codes.iter().map(|&c| lens[c as usize] as usize).sum();
    Ok(Fsst12Abi {
        codes,
        lens,
        dict_padded,
        decoded_bytes,
        table_entries: entries,
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
/// independently 12-bit packed, so a row with an odd code count rounds its final code up
/// from 1.5 to 2 bytes -- half a byte of waste, not the 1.5 an earlier revision of this
/// comment claimed. On a column of many short rows even half a byte per row is a
/// measurable ratio penalty against the whole-buffer form. It is also what a random-access
/// string codec actually stores, which is the operating point FSST is designed for.
#[derive(Debug, Clone)]
pub struct Fsst12RowsAbi {
    /// Concatenated per-row codes, in row order.
    pub abi: Fsst12Abi,
    /// `row_code_offsets[i]..row_code_offsets[i+1]` indexes `abi.codes` for row `i`.
    /// Length is `rows + 1`.
    pub row_code_offsets: Vec<u64>,
    /// Stored footprint, by component. See [`Fsst12StoredSize`] for why this is not a
    /// single number.
    pub stored: Fsst12StoredSize,
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
    let mut packed_codes = 0usize;
    row_code_offsets.push(0);
    for packed in &per_row {
        packed_codes += packed.len();
        codes.extend_from_slice(&unpack_codes(packed)?);
        row_code_offsets.push(codes.len() as u64);
    }

    let entries = compressor.symbol_table().len();
    validate_codes(&codes, entries)?;
    let decoded_bytes = codes.iter().map(|&c| lens[c as usize] as usize).sum();
    Ok(Fsst12RowsAbi {
        abi: Fsst12Abi {
            codes,
            lens,
            dict_padded,
            decoded_bytes,
            table_entries: entries,
        },
        stored: Fsst12StoredSize {
            packed_codes,
            row_offsets: row_code_offsets.len() * size_of::<u64>(),
            table: entries * (size_of::<u64>() + size_of::<u8>()),
            codes_btrblocks: 0,
        },
        row_code_offsets,
    })
}

/// Synthesize OnPair's packed dictionary directory, `(offset << 16) | length`, over a
/// contiguous dictionary-bytes buffer.
///
/// The compact-layout kernels read this instead of the padded table, so FSST-12 needs an
/// equivalent to exercise them. Returns the directory, the bytes it indexes, and the
/// logical byte length (excluding the trailing read pad).
///
/// The returned buffer carries `DICT_STRIDE` initialized trailing bytes because the compact
/// kernels issue FIXED-WIDTH reads at a directory offset, not length-exact ones: a
/// `uint4` load for the last trained symbol, or for an untrained entry pointing at the
/// logical end, would otherwise run past the allocation. The logical length is returned
/// separately so callers report dictionary footprint without the pad.
pub fn compact_dict(
    symbols: &[fsst12::Symbol],
    lengths: &[u8],
) -> Result<(Vec<u64>, Vec<u8>, usize), Fsst12AbiError> {
    validate_table(symbols, lengths)?;

    let mut table = Vec::with_capacity(FSST12_CODE_SPACE);
    let mut bytes: Vec<u8> = Vec::with_capacity(symbols.len() * 8 + DICT_STRIDE);
    for (sym, &len) in symbols.iter().zip(lengths.iter()) {
        let off = bytes.len() as u64;
        bytes.extend_from_slice(&sym.to_u64().to_le_bytes()[..len as usize]);
        // The directory packs the offset into the high 48 bits, so a dictionary larger
        // than 2^48 bytes would alias. Unreachable at 4096 x 8 B, asserted rather than
        // assumed because the shift is silent on overflow.
        debug_assert!(off < (1u64 << 48));
        table.push((off << 16) | len as u64);
    }
    let logical_len = bytes.len();
    // Untrained codes: zero length at the logical end, so a stray fixed-width read lands
    // in the pad rather than in another entry's bytes.
    table.resize(FSST12_CODE_SPACE, (logical_len as u64) << 16);
    bytes.extend(std::iter::repeat_n(0u8, DICT_STRIDE));
    Ok((table, bytes, logical_len))
}

/// What a serialized, row-addressable FSST-12 column actually stores.
///
/// Kept as separate components, and deliberately NOT collapsed into one `compressed_bytes`
/// field, because the obvious single number is the one that is wrong: the per-row 12-bit
/// payload alone omits both the row offsets that make the column addressable and the
/// dictionary needed to decode it. A ratio built from the payload alone is optimistic, and
/// it is not commensurable with OnPair's `in_memory_bytes`, which counts codes, offsets,
/// and dictionary together. Any FSST-12-vs-OnPair ratio must compare [`Self::total`]
/// against that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fsst12StoredSize {
    /// Per-row 12-bit packed code payload, summed over rows.
    pub packed_codes: usize,
    /// Row-boundary vector as STORED, i.e. BtrBlocks-compressed, which is how OnPair's
    /// `codes_offsets` are counted in its `in_memory_bytes`. `normalize_rows` fills this with
    /// the raw `u64` size as a placeholder that the caller MUST overwrite with the compressed
    /// size -- charging raw offsets against OnPair's compressed ones produced ratios below 1.0
    /// on short-row columns, which no compressor can do.
    pub row_offsets: usize,
    /// Dictionary as stored: 8 B symbol plus 1 B length per trained entry. Excludes the
    /// GPU-side padding and the widened decode table, which are load-time artifacts.
    pub table: usize,
    /// The code stream measured the way OnPair's is: as an integer array handed to
    /// BtrBlocks, rather than as FSST-12's own fixed-width 12-bit packing. Filled by the
    /// caller, which owns the compressor; zero means not measured.
    ///
    /// Both exist because they answer different questions and the difference is large.
    /// FSST-12's native packing is FIXED at 12 bits per code, while BtrBlocks bitpacks an
    /// integer code array down to about log2(cardinality) bits -- so on a column with two
    /// distinct values OnPair pays ~2 bits per code and native FSST-12 pays 12. Comparing
    /// native FSST-12 against OnPair-inside-Vortex therefore conflates the codec with its
    /// container. [`Fsst12StoredSize::total`] is the native measure; [`Self::total_container_matched`]
    /// is the one that isolates the codec.
    pub codes_btrblocks: usize,
}

impl Fsst12StoredSize {
    /// Native total: FSST-12 stored as its reference implementation stores it, with codes in
    /// dense 12-bit packing. This is the honest figure to compare against published FSST
    /// numbers, and the pessimistic one to compare against OnPair-in-Vortex.
    pub fn total(&self) -> usize {
        self.packed_codes + self.row_offsets + self.table
    }

    /// Container-matched total: the code stream measured by the same instrument as OnPair's,
    /// so the comparison isolates the codec rather than the storage format. Falls back to the
    /// native total when `codes_btrblocks` was not measured.
    pub fn total_container_matched(&self) -> usize {
        if self.codes_btrblocks == 0 {
            return self.total();
        }
        self.codes_btrblocks + self.row_offsets + self.table
    }
}

/// Decode through the ABI as a kernel lane does: read a fixed sixteen bytes at
/// `dict + code * 16`, then advance the cursor by the true length.
///
/// It mirrors the kernel's fixed-width READ, which is what makes the padded table's
/// allocation and stride load-bearing. It does NOT prove the padded upper half is clean:
/// the next token's write overwrites the excess and the final excess is truncated, so a
/// dirty upper half is invisible here. An earlier revision of this comment claimed
/// otherwise. `padded_upper_half_is_clean` asserts that property directly instead.
///
/// Test oracle only -- not a decoder anything should depend on.
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

    /// Extract one row's code slice as a standalone ABI, for isolated decode.
    fn slice_rows(r: &Fsst12RowsAbi, lo: usize, hi: usize) -> Fsst12Abi {
        Fsst12Abi {
            codes: r.abi.codes[lo..hi].to_vec(),
            lens: r.abi.lens.clone(),
            dict_padded: r.abi.dict_padded.clone(),
            decoded_bytes: r.abi.codes[lo..hi]
                .iter()
                .map(|&c| r.abi.lens[c as usize] as usize)
                .sum(),
            table_entries: r.abi.table_entries,
        }
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
        let (compressor, ..) = train_and_compress(&rows);
        assert!(
            compressor
                .symbol_lengths()
                .iter()
                .all(|&l| (1..=8).contains(&l)),
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

        // EVERY offset, not a sample: a desynchronized offset vector can be correct at
        // the ends and wrong in between.
        let small: Vec<&[u8]> = refs.iter().copied().take(400).collect();
        let small_abi = normalize_rows(&compressor, &small).expect("normalize_rows small");
        for (i, row) in small.iter().enumerate() {
            let lo = small_abi.row_code_offsets[i] as usize;
            let hi = small_abi.row_code_offsets[i + 1] as usize;
            let sliced = slice_rows(&small_abi, lo, hi);
            assert_eq!(
                decode_via_abi(&sliced),
                *row,
                "row {i} decodes in isolation"
            );
        }

        // Accounting: components must be individually nonzero and sum to total.
        let st = rowsabi.stored;
        assert!(st.packed_codes > 0 && st.row_offsets > 0 && st.table > 0);
        assert_eq!(st.total(), st.packed_codes + st.row_offsets + st.table);
    }

    /// The compact directory must address the same bytes the padded table holds.
    #[test]
    fn compact_dict_agrees_with_padded_table() {
        let rows = corpus();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let (table, bytes, logical) =
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

        // Fixed-width reads must stay in bounds for the LAST trained code and for an
        // untrained entry pointing at the logical end -- the case true-length slicing
        // cannot detect.
        let last = compressor.symbol_table().len() - 1;
        let last_off = (table[last] >> 16) as usize;
        assert!(
            last_off + DICT_STRIDE <= bytes.len(),
            "last trained fixed-width read"
        );
        let untrained_off = (table[FSST12_CODE_SPACE - 1] >> 16) as usize;
        assert_eq!(
            untrained_off, logical,
            "untrained entry points at the logical end"
        );
        assert!(
            untrained_off + DICT_STRIDE <= bytes.len(),
            "untrained fixed-width read in bounds"
        );
        assert_eq!(
            table[FSST12_CODE_SPACE - 1] & 0xffff,
            0,
            "untrained length is zero"
        );
        assert_eq!(
            table.len(),
            FSST12_CODE_SPACE,
            "directory covers the code space"
        );
    }

    /// The padded table's unused upper half must actually be zero. decode_via_abi cannot
    /// show this -- the next token overwrites the excess -- so assert it at the source.
    #[test]
    fn padded_upper_half_is_clean() {
        let rows = corpus();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let (padded, lens) =
            widen_table(compressor.symbol_table(), compressor.symbol_lengths()).expect("widen");
        for code in 0..FSST12_CODE_SPACE {
            let len = lens[code] as usize;
            let cell = &padded[code * DICT_STRIDE..(code + 1) * DICT_STRIDE];
            assert!(
                cell[len..].iter().all(|&b| b == 0),
                "code {code} has dirty bytes past its length {len}"
            );
        }
        assert!(
            padded.len() >= FSST12_CODE_SPACE * DICT_STRIDE + DICT_STRIDE,
            "trailing pad"
        );
    }

    /// Hand-built packing vectors, independent of the codec, so a shared misreading between
    /// our unpacker and the reference cannot hide.
    #[test]
    fn unpack_hand_built_vectors() {
        // One 3-byte triple: low code 0x123, high code 0x456 -> bytes 23 61 45.
        assert_eq!(
            unpack_codes(&[0x23, 0x61, 0x45]).unwrap(),
            vec![0x123, 0x456]
        );
        // 2-byte odd tail: only the low 12 bits are a code; the high nibble is ignored.
        assert_eq!(unpack_codes(&[0x23, 0xf1]).unwrap(), vec![0x123]);
        // 5 bytes = one triple plus a 2-byte odd tail.
        assert_eq!(
            unpack_codes(&[0x23, 0x61, 0x45, 0x89, 0x07]).unwrap(),
            vec![0x123, 0x456, 0x789]
        );
        // Lengths 1 and 4 are 1 mod 3 and invalid.
        assert!(unpack_codes(&[0x00]).is_err());
        assert!(unpack_codes(&[0u8; 4]).is_err());
    }

    /// Degenerate row shapes: no rows at all, and empty rows interleaved with real ones.
    #[test]
    fn row_form_handles_degenerate_rows() {
        let base = corpus();
        let refs: Vec<&[u8]> = base.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);

        let zero = normalize_rows(&compressor, &[]).expect("zero rows");
        assert_eq!(zero.row_code_offsets, vec![0]);
        assert!(zero.abi.codes.is_empty());
        assert_eq!(zero.abi.decoded_bytes, 0);

        let mixed: Vec<&[u8]> = vec![b"", b"", b"abc", b"", b"defgh", b""];
        let got = normalize_rows(&compressor, &mixed).expect("mixed rows");
        assert_eq!(got.row_code_offsets.len(), mixed.len() + 1);
        for (i, row) in mixed.iter().enumerate() {
            let lo = got.row_code_offsets[i] as usize;
            let hi = got.row_code_offsets[i + 1] as usize;
            let sliced = slice_rows(&got, lo, hi);
            assert_eq!(decode_via_abi(&sliced), *row, "row {i}");
        }
    }

    /// Untrained codes must fail loudly rather than decode to nothing.
    #[test]
    fn rejects_untrained_code() {
        let base = corpus();
        let refs: Vec<&[u8]> = base.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let entries = compressor.symbol_table().len();
        assert!(
            entries < FSST12_CODE_SPACE,
            "corpus should not fill the table"
        );
        // Embed an untrained code in the middle of an otherwise valid stream.
        let codes = vec![b'a' as u16, entries as u16, b'b' as u16];
        assert!(matches!(
            validate_codes(&codes, entries),
            Err(Fsst12AbiError::UntrainedCode { pos: 1, .. })
        ));
    }

    /// A table that is the right size but whose reserved codes are not the identity
    /// singletons is not FSST-12, and must be rejected by both conversions.
    #[test]
    fn rejects_non_identity_reserved_codes() {
        let base = corpus();
        let refs: Vec<&[u8]> = base.iter().map(|r| r.as_slice()).collect();
        let compressor = Compressor12::train(&refs);
        let mut symbols = compressor.symbol_table().to_vec();
        let mut lengths = compressor.symbol_lengths().to_vec();
        symbols[7] = fsst12::Symbol::from_slice(b"XXXXXXXX");
        lengths[7] = 8;
        assert!(matches!(
            widen_table(&symbols, &lengths),
            Err(Fsst12AbiError::ReservedCode { code: 7, .. })
        ));
        assert!(matches!(
            compact_dict(&symbols, &lengths),
            Err(Fsst12AbiError::ReservedCode { code: 7, .. })
        ));
        // Oversized tables must be rejected, not silently truncated.
        let over = vec![fsst12::Symbol::ZERO; FSST12_CODE_SPACE + 1];
        let over_len = vec![1u8; FSST12_CODE_SPACE + 1];
        assert!(matches!(
            compact_dict(&over, &over_len),
            Err(Fsst12AbiError::TableSize(_))
        ));
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
        let (compressor, ..) = train_and_compress(&rows);
        let abi = normalize(&compressor, &compressor.compress(b"abc")).expect("normalize");
        assert_eq!(abi.lens.len(), FSST12_CODE_SPACE);
        assert!(abi.dict_padded.len() >= FSST12_CODE_SPACE * DICT_STRIDE);
        assert_eq!(
            abi.lens[FSST12_CODE_SPACE - 1],
            0,
            "untrained code has length 0"
        );
    }
}
