# OnPair GPU decompression design

## Representation facts

The native Vortex OnPair dictionary offset type is 32-bit, not 64-bit:
`OnPairDictionaryStorage::offsets` is `Buffer<u32>` and implements
`DictionaryStorage<u32>` in `encodings/onpair/src/array.rs`.

The optimized CUDA decoder does not read that offset directory for every token.
CPU staging flattens each dictionary entry into cache-oriented planes:

- `dict_s8_lo`: bytes 0 through 7, zero-padded, 8 bytes per entry.
- `dict_s8_hi`: bytes 8 through 15, zero-padded, 8 bytes per entry.
- `packed_lens`: two four-bit encoded lengths per byte.

The code stream remains native `u16`, or 2 bytes per token. It is a large
streaming input and is not expected to remain resident in L1. The three
dictionary tables are the reusable, cache-sized working set.

## Packed length encoding

OnPair dictionary entries have lengths in `[1, 16]`. A raw nibble cannot
represent 16, so the host stores `length - 1`:

```text
even code: packed_lens[code / 2] bits 3:0
odd code:  packed_lens[code / 2] bits 7:4
decoded:   nibble + 1
```

The device helper is:

```cpp
__device__ inline uint32_t unpack_length(
    const uint8_t *__restrict packed_lens, uint32_t code) {
    const uint32_t packed = (uint32_t)packed_lens[code >> 1u];
    const uint32_t shift = (code & 1u) << 2u;
    return ((packed >> shift) & 0xfu) + 1u;
}
```

This handles the full range without a sentinel or side table. Tokens past the
end of the stream still receive length zero in registers and do not read the
packed table.

## Kernel ABI and loads

`onpair_decompress` receives:

```cpp
const uint16_t *codes;
const uint64_t *chunk_offsets;
const uint8_t *dict_s8_lo;
const uint8_t *dict_s8_hi;
const uint8_t *packed_lens;
uint8_t *output_bytes;
uint64_t total_tokens;
```

Every in-range token performs one 8-byte low-plane load and one packed-length
byte load. Only entries longer than eight bytes enter the existing dense
per-warp request queue and perform an 8-byte high-plane load. The scan, shared
staging, aligned output drain, and 128-token-per-warp assignment are unchanged.

## Working-set sizes

For `N` dictionary entries:

| table | previous split decoder | packed split decoder |
|---|---:|---:|
| low bytes | `8N` | `8N` |
| high-byte source | `16N` padded dictionary | `8N` high plane |
| lengths | `N` | `ceil(N / 2)` |
| total GPU dictionary staging | `25N` | `16.5N` |

At 4096 entries this is 102,400 bytes previously versus 67,584 bytes now:
32 KiB low + 32 KiB high + 2 KiB lengths, a 34.0% reduction. The length table
itself is exactly halved, and the high-byte source is halved.

The benchmark metadata records the actual per-cell values as `code_bytes`,
`dict_s8_lo_bytes`, `dict_s8_hi_bytes`, and `packed_lens_bytes`.

## Correctness constraints

- Host staging rejects dictionary lengths outside `[1, 16]`.
- Low and high planes are 8-byte-strided and loaded as aligned `uint2`.
- High bytes are read only when decoded length exceeds eight.
- The high request stores `high_length - 1`, which fits in three bits for
  high lengths `[1, 8]`.
- GPU output is copied back and compared byte-for-byte with CPU decode for
  every benchmark process.

## Source files

- `onpair_decompress.cu`: packed-length, low/high-plane candidate.
- `onpair_decompress_u8_lens.cu`: preserved previous split candidate.
- `onpair_old_2.cu`: preserved legacy comparison kernel.
