// Standalone nvCOMP hardware-decompression-engine baseline (two Deflate presets + LZ4 + Snappy).
// Compresses a raw byte file with nvCOMP, then times decompression on the
// dedicated hardware Decompression Engine (backend=HARDWARE). Reports decode
// GiB/s over the *uncompressed* bytes (directly comparable to OnPair decode).
//
// GOOD-BASELINE DEFAULTS (do not weaken):
//   - Deflate compress algorithm = 5 (max ratio). The SDK default (algo=1) is
//     "low compression ratio" and understates Deflate badly — see below.
//   - chunk = 256 KiB: near-optimal DE throughput. Ratio is NOT chunk-sensitive
//     (Deflate caps its back-reference window at 32 KiB), so the compression
//     *level* is the lever, not the chunk size. We nevertheless SWEEP the chunk
//     size {32,64,128,256,512} KiB and report DE's BEST decode across all
//     (chunk x codec) configs, so the published DE multiple uses DE's true best.
//   - LZ4 data_type = CHAR (single-pass, no level knob).
//   - nvCOMP Zstd has NO hardware-engine path (DE returns status 10); for a Zstd
//     CUDA-backend baseline use the onpair-chunk-bench at level 3 (not -10).
//
// Build (CUDA >= 12.8, nvcomp SDK under target/.../nvcomp-sdk):
//   SDK=$(find target -path '*nvcomp-sdk' -type d | head -1)
//   nvcc -O3 -arch=native nvcomp_hw_bench.cu -o nvbench \
//     -I"$SDK/include" -L"$SDK/lib" -lnvcomp -lcudart
//   LD_LIBRARY_PATH="$SDK/lib" ./nvbench <file> [legacy_chunk_bytes] [deflate_algo]
// Input <file> = raw concatenated column bytes (dump with pyarrow).
//
// OUTPUT: a single JSON object on stdout (human-readable per-config lines go to
// stderr). The object preserves the historical top-level fields — raw_bytes,
// chunk_bytes (=262144, the 256 KiB cell), the four per-codec objects, and
// best_codec/best_decode_gib_s/best_ratio — for backward compatibility with the
// figure pipeline (common.py de_map()). best_* is now the max over the WHOLE
// chunk sweep. ADDED: a `chunk_sweep` array covering all five chunk sizes, and a
// per-codec `decode_ns_iters` array of the raw per-iteration decode times in
// integer nanoseconds (GOLD provenance, mirroring the OnPair bench).
#include <cstdio>
#include <cstdlib>

// PAYLOAD BASIS. `N` is the size of the input FILE, which is what gets compressed. When the file
// carries row structure (a u32 length prefix per string, so the DE is measured on a stream rows can
// be recovered from, as Zstd and Parquet are), the file is LARGER than the string payload. The
// ratio and the decode rate must stay on the payload -- the useful bytes -- or prefix bytes count
// as output and both numbers inflate. NVCOMP_PAYLOAD_BYTES carries the payload size; unset means
// the file IS the payload, which is the pre-existing flat-concatenation behaviour.
static size_t g_payload_bytes = 0;

#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define NK(x) do{ nvcompStatus_t s=(x); if(s!=nvcompSuccess){ fprintf(stderr,"nvcomp %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

// The legacy top-level cell + the entry duplicated inside chunk_sweep.
static const size_t LEGACY_CHUNK = 262144; // 256 KiB

// One codec's result at one chunk size. `valid` is false when the HW engine
// rejected this (chunk,codec) config or the byte round-trip failed; in that case
// the scalar fields are 0 and decode_ns_iters is empty.
struct CodecResult {
    double ratio = 0.0;          // payload_bytes / compressed_bytes
    // RECORDED, NOT DERIVED. ratio and the rates are doubles rounded on output, so a consumer
    // cannot recover the bytes from them or check the basis it was given. Carrying the two integers
    // lets any reducer recompute every basis exactly and assert ratio against them.
    unsigned long long compressed_bytes = 0;
    unsigned long long basis_bytes = 0;   // the numerator actually used (payload, or file if flat)
    double compress_gib_s = 0.0; // encode throughput over uncompressed bytes
    double decode_gib_s = 0.0;   // min-reduced decode throughput over uncompressed bytes
    bool valid = false;          // HW-supported AND byte-exact
    bool supported = false;      // HW API accepted this codec/chunk configuration
    bool validation_failed = false; // status/size/byte validation detected corruption
    std::vector<unsigned long long> decode_ns_iters; // GOLD: raw per-iter decode times, integer ns
};

// Run compress + (warmup + 100 timed) HW decode for one codec at one chunk size.
// Mirrors the original timing/batching exactly (single decompAsync over ALL chunks
// per iteration, so the engine stays saturated). On HW rejection of this chunk
// size, sets out.valid=false and returns WITHOUT aborting the process. All device
// allocations and the stream/events are freed before return so the chunk sweep
// (20 invocations) does not leak.
template<class CompOpts, class DecompOpts>
void run(const char* name, std::vector<unsigned char>& host, size_t chunk,
         CompOpts copts, DecompOpts dopts, CodecResult& out,
         nvcompStatus_t(*compTemp)(size_t,size_t,CompOpts,size_t*,size_t),
         nvcompStatus_t(*compMax)(size_t,CompOpts,size_t*),
         nvcompStatus_t(*compAsync)(const void* const*,const size_t*,size_t,size_t,void*,size_t,void* const*,size_t*,CompOpts,nvcompStatus_t*,cudaStream_t),
         nvcompStatus_t(*decompTemp)(size_t,size_t,DecompOpts,size_t*,size_t),
         nvcompStatus_t(*decompAsync)(const void* const*,const size_t*,const size_t*,size_t*,size_t,void* const,size_t,void* const*,DecompOpts,nvcompStatus_t*,cudaStream_t))
{
    out = CodecResult{};
    size_t N = host.size();
    size_t num = (N + chunk - 1) / chunk;
    const size_t PB = g_payload_bytes ? g_payload_bytes : N;  // ratio/rate basis
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    // ---- upload uncompressed, build chunk ptr/size arrays ----
    unsigned char* d_in; CK(cudaMalloc(&d_in, N)); CK(cudaMemcpy(d_in, host.data(), N, cudaMemcpyHostToDevice));
    std::vector<void*> h_inptr(num); std::vector<size_t> h_insz(num);
    for(size_t i=0;i<num;i++){ h_inptr[i]=d_in+i*chunk; h_insz[i]=(i+1<num)?chunk:(N-i*chunk); }
    void** d_inptr; size_t* d_insz;
    CK(cudaMalloc(&d_inptr,num*sizeof(void*))); CK(cudaMemcpy(d_inptr,h_inptr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_insz,num*sizeof(size_t))); CK(cudaMemcpy(d_insz,h_insz.data(),num*sizeof(size_t),cudaMemcpyHostToDevice));

    // ---- compress ----
    size_t ctemp=0; NK(compTemp(num,chunk,copts,&ctemp,N));
    size_t maxout=0; NK(compMax(chunk,copts,&maxout));
    void* d_ctemp=nullptr; if(ctemp) CK(cudaMalloc(&d_ctemp,ctemp));
    unsigned char* d_cbuf; CK(cudaMalloc(&d_cbuf,num*maxout));
    std::vector<void*> h_cptr(num); for(size_t i=0;i<num;i++) h_cptr[i]=d_cbuf+i*maxout;
    void** d_cptr; CK(cudaMalloc(&d_cptr,num*sizeof(void*))); CK(cudaMemcpy(d_cptr,h_cptr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    size_t* d_csz; CK(cudaMalloc(&d_csz,num*sizeof(size_t)));
    nvcompStatus_t* d_st; CK(cudaMalloc(&d_st,num*sizeof(nvcompStatus_t)));
    for(int w=0;w<2;w++) NK(compAsync(d_inptr,d_insz,chunk,num,d_ctemp,ctemp,d_cptr,d_csz,copts,d_st,stream));
    CK(cudaStreamSynchronize(stream));
    // time compression (encode throughput over uncompressed bytes)
    cudaEvent_t ca,cb; CK(cudaEventCreate(&ca)); CK(cudaEventCreate(&cb));
    int citers=20; CK(cudaEventRecord(ca,stream));
    for(int i=0;i<citers;i++) NK(compAsync(d_inptr,d_insz,chunk,num,d_ctemp,ctemp,d_cptr,d_csz,copts,d_st,stream));
    CK(cudaEventRecord(cb,stream)); CK(cudaEventSynchronize(cb));
    float cms=0; CK(cudaEventElapsedTime(&cms,ca,cb)); cms/=citers;
    double enc_gibs=(double)PB/(cms/1e3)/(1024.0*1024*1024);
    std::vector<size_t> h_csz(num); CK(cudaMemcpy(h_csz.data(),d_csz,num*sizeof(size_t),cudaMemcpyDeviceToHost));
    std::vector<nvcompStatus_t> h_status(num);
    CK(cudaMemcpy(h_status.data(),d_st,num*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));
    size_t ctot=0;
    bool compression_ok=true;
    for(size_t i=0;i<num;i++){
        if(h_status[i]!=nvcompSuccess || h_csz[i]==0 || h_csz[i]>maxout){
            fprintf(stderr,"%-13s chunk=%zu compression result[%zu] status=%d size=%zu max=%zu\n",
                    name,chunk,i,(int)h_status[i],h_csz[i],maxout);
            compression_ok=false;
        }
        ctot+=h_csz[i];
    }
    if(!compression_ok || ctot==0){
        out.validation_failed=true;
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
        if(d_ctemp) cudaFree(d_ctemp);
        cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
        cudaStreamDestroy(stream);
        return;
    }

    // ---- decompress on HARDWARE engine ----
    // The HW path can reject a (chunk,codec) config here (e.g. Zstd has no HW
    // path: status 10). Treat that as "unsupported": mark invalid, free, return —
    // do NOT abort the whole sweep.
    size_t dtemp=0; nvcompStatus_t ds=decompTemp(num,chunk,dopts,&dtemp,N);
    if(ds!=nvcompSuccess){
        fprintf(stderr,"%-13s chunk=%zu HW decompress GetTempSize status=%d (UNSUPPORTED)\n",name,chunk,(int)ds);
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
        if(d_ctemp) cudaFree(d_ctemp);
        cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
        cudaStreamDestroy(stream);
        out.valid=false; return;
    }
    void* d_dtemp=nullptr; if(dtemp) CK(cudaMalloc(&d_dtemp,dtemp));
    unsigned char* d_out; CK(cudaMalloc(&d_out,N));
    std::vector<void*> h_optr(num); for(size_t i=0;i<num;i++) h_optr[i]=d_out+i*chunk;
    void** d_optr; CK(cudaMalloc(&d_optr,num*sizeof(void*))); CK(cudaMemcpy(d_optr,h_optr.data(),num*sizeof(void*),cudaMemcpyHostToDevice));
    size_t* d_obufsz; CK(cudaMalloc(&d_obufsz,num*sizeof(size_t))); CK(cudaMemcpy(d_obufsz,d_insz,num*sizeof(size_t),cudaMemcpyDeviceToDevice));
    size_t* d_actual; CK(cudaMalloc(&d_actual,num*sizeof(size_t)));

    // First warmup decode is a soft check: if the HW engine rejects this config
    // only at the async call (not at GetTempSize), bail to invalid rather than
    // abort. Remaining warmups + the timed loop use NK (a hard error there is a
    // real bug, not a config-rejection).
    nvcompStatus_t dw = decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream);
    if(dw!=nvcompSuccess){
        fprintf(stderr,"%-13s chunk=%zu HW decompress async status=%d (UNSUPPORTED)\n",name,chunk,(int)dw);
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
        if(d_ctemp) cudaFree(d_ctemp);
        cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
        if(d_dtemp) cudaFree(d_dtemp);
        cudaFree(d_out); cudaFree(d_optr); cudaFree(d_obufsz); cudaFree(d_actual);
        cudaStreamDestroy(stream);
        out.valid=false; return;
    }
    out.supported=true;
    for(int w=1;w<3;w++){ NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream)); }
    CK(cudaStreamSynchronize(stream));

    // Validate every chunk's API status and decoded size, not just the aggregate bytes.
    std::vector<size_t> h_actual(num);
    CK(cudaMemcpy(h_status.data(),d_st,num*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_actual.data(),d_actual,num*sizeof(size_t),cudaMemcpyDeviceToHost));
    bool metadata_ok=true;
    for(size_t i=0;i<num;i++){
        if(h_status[i]!=nvcompSuccess || h_actual[i]!=h_insz[i]){
            fprintf(stderr,"%-13s chunk=%zu decompress result[%zu] status=%d actual=%zu expected=%zu\n",
                    name,chunk,i,(int)h_status[i],h_actual[i],h_insz[i]);
            metadata_ok=false;
        }
    }
    std::vector<unsigned char> back(N); CK(cudaMemcpy(back.data(),d_out,N,cudaMemcpyDeviceToHost));
    bool ok = metadata_ok && memcmp(back.data(),host.data(),N)==0;
    if(!ok){
        fprintf(stderr,"%-13s chunk=%zu warmup validation FAILED\n",name,chunk);
        out.validation_failed=true;
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
        if(d_ctemp) cudaFree(d_ctemp);
        cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
        if(d_dtemp) cudaFree(d_dtemp);
        cudaFree(d_out); cudaFree(d_optr); cudaFree(d_obufsz); cudaFree(d_actual);
        cudaStreamDestroy(stream);
        return;
    }

    // MIN single-pass time over the iterations (matches FastPair's reduction): each
    // decode is timed in isolation and we keep the fastest, not the mean. GOLD: in
    // addition to the min, store every per-iteration time as integer nanoseconds in
    // out.decode_ns_iters — recorded AFTER cudaEventElapsedTime reads the bracket,
    // so the timed region (cudaEventRecord(a)..cudaEventRecord(b)) is unchanged.
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int iters=100; float ms=1e30f;
    out.decode_ns_iters.reserve((size_t)iters);
    for(int i=0;i<iters;i++){
        CK(cudaEventRecord(a,stream));
        NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream));
        CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b));
        float it=0; CK(cudaEventElapsedTime(&it,a,b));
        // it >= 0, so +0.5 then truncate == round-to-nearest (no <cmath> needed).
        out.decode_ns_iters.push_back((unsigned long long)((double)it*1e6+0.5));
        if(it<ms) ms=it;
    }
    CK(cudaMemcpy(h_status.data(),d_st,num*sizeof(nvcompStatus_t),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_actual.data(),d_actual,num*sizeof(size_t),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(back.data(),d_out,N,cudaMemcpyDeviceToHost));
    bool timed_ok = memcmp(back.data(),host.data(),N)==0;
    for(size_t i=0;i<num;i++)
        timed_ok = timed_ok && h_status[i]==nvcompSuccess && h_actual[i]==h_insz[i];
    if(!timed_ok){
        fprintf(stderr,"%-13s chunk=%zu timed-pass validation FAILED\n",name,chunk);
        out.decode_ns_iters.clear();
        out.validation_failed=true;
        ok=false;
    }
    double gibs = (double)PB/(ms/1e3)/ (1024.0*1024*1024);
    double gbs  = (double)PB/(ms/1e3)/ 1e9;
    fprintf(stderr,"%-13s chunk=%6zu  ratio=%.2fx  compress=%6.1f GiB/s  decode=%6.1f GiB/s (%.0f GB/s)  valid=%s\n",
           name, chunk, (double)PB/ctot, enc_gibs, gibs, gbs, ok?"YES":"NO");

    out.ratio = ok ? (double)PB/ctot : 0.0;
    out.compressed_bytes = (unsigned long long)ctot;
    out.basis_bytes = (unsigned long long)PB;
    out.compress_gib_s = ok ? enc_gibs : 0.0;
    out.decode_gib_s = ok ? gibs : 0.0;
    out.valid = ok;

    // ---- free everything (the sweep calls run() 20x; leaking would OOM) ----
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaEventDestroy(ca); cudaEventDestroy(cb);
    cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
    if(d_ctemp) cudaFree(d_ctemp);
    cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
    if(d_dtemp) cudaFree(d_dtemp);
    cudaFree(d_out); cudaFree(d_optr); cudaFree(d_obufsz); cudaFree(d_actual);
    cudaStreamDestroy(stream);
}

// Run all configured codecs at one chunk size, filling `results` keyed by the
// codec display name (parallel arrays `names`/`results`).
static void run_all_codecs(std::vector<unsigned char>& host, size_t chunk,
                           const int* deflate_algos, const char* const* deflate_names,
                           std::vector<const char*>& names,
                           std::vector<CodecResult>& results)
{
    names.clear();
    results.clear();

    nvcompBatchedDeflateDecompressOpts_t ddo = nvcompBatchedDeflateDecompressDefaultOpts;
    ddo.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    for(int p=0;p<2;p++){
        if(deflate_algos[p]<0) continue;
        nvcompBatchedDeflateCompressOpts_t dco = nvcompBatchedDeflateCompressDefaultOpts;
        dco.algorithm = deflate_algos[p];
        CodecResult r;
        run<nvcompBatchedDeflateCompressOpts_t,nvcompBatchedDeflateDecompressOpts_t>(
            deflate_names[p], host, chunk, dco, ddo, r,
            nvcompBatchedDeflateCompressGetTempSizeAsync, nvcompBatchedDeflateCompressGetMaxOutputChunkSize,
            nvcompBatchedDeflateCompressAsync, nvcompBatchedDeflateDecompressGetTempSizeAsync,
            nvcompBatchedDeflateDecompressAsync);
        names.push_back(deflate_names[p]);
        results.push_back(std::move(r));
    }

    nvcompBatchedLZ4DecompressOpts_t ldo = nvcompBatchedLZ4DecompressDefaultOpts;
    ldo.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    {
        CodecResult r;
        run<nvcompBatchedLZ4CompressOpts_t,nvcompBatchedLZ4DecompressOpts_t>(
            "LZ4", host, chunk, nvcompBatchedLZ4CompressDefaultOpts, ldo, r,
            nvcompBatchedLZ4CompressGetTempSizeAsync, nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
            nvcompBatchedLZ4CompressAsync, nvcompBatchedLZ4DecompressGetTempSizeAsync,
            nvcompBatchedLZ4DecompressAsync);
        names.push_back("LZ4");
        results.push_back(std::move(r));
    }

    // Snappy: the remaining HW-engine codec family. Same single-pass, no-level
    // shape as LZ4; `run` marks the config invalid if the engine rejects it.
    nvcompBatchedSnappyDecompressOpts_t sdo = nvcompBatchedSnappyDecompressDefaultOpts;
    sdo.backend = NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
    {
        CodecResult r;
        run<nvcompBatchedSnappyCompressOpts_t,nvcompBatchedSnappyDecompressOpts_t>(
            "Snappy", host, chunk, nvcompBatchedSnappyCompressDefaultOpts, sdo, r,
            nvcompBatchedSnappyCompressGetTempSizeAsync, nvcompBatchedSnappyCompressGetMaxOutputChunkSize,
            nvcompBatchedSnappyCompressAsync, nvcompBatchedSnappyDecompressGetTempSizeAsync,
            nvcompBatchedSnappyDecompressAsync);
        names.push_back("Snappy");
        results.push_back(std::move(r));
    }
}

// Print one codec object's body (the {ratio, compress_gib_s, decode_gib_s,
// valid, decode_ns_iters} fields). `indent` is the leading whitespace for the
// "ratio" line; the object braces are printed by the caller. `trailing_comma`
// controls the comma after the closing brace (set false for the last codec).
static void print_codec_obj(const char* indent, const char* name, const CodecResult& r,
                            bool trailing_comma)
{
    printf("%s\"%s\": {\n", indent, name);
    printf("%s  \"ratio\": %.2f,\n", indent, r.ratio);
    printf("%s  \"compressed_bytes\": %llu,\n", indent, r.compressed_bytes);
    printf("%s  \"basis_bytes\": %llu,\n", indent, r.basis_bytes);
    printf("%s  \"compress_gib_s\": %.1f,\n", indent, r.compress_gib_s);
    printf("%s  \"decode_gib_s\": %.1f,\n", indent, r.decode_gib_s);
    printf("%s  \"supported\": %s,\n", indent, r.supported?"true":"false");
    printf("%s  \"validation_failed\": %s,\n", indent, r.validation_failed?"true":"false");
    printf("%s  \"valid\": %s,\n", indent, r.valid?"true":"false");
    printf("%s  \"decode_ns_iters\": [", indent);
    for(size_t i=0;i<r.decode_ns_iters.size();i++)
        printf("%s%llu", i?",":"", (unsigned long long)r.decode_ns_iters[i]);
    printf("]\n");
    printf("%s}%s\n", indent, trailing_comma?",":"");
}

int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"/tmp/l_comment.bin";
    if(argc>2 && (size_t)atol(argv[2])!=LEGACY_CHUNK){
        fprintf(stderr,"chunk override is not supported: this producer always emits the fixed five-size sweep\n");
        return 1;
    }
    FILE* f=fopen(path,"rb"); if(!f){ perror("open"); return 1; }
    if(fseek(f,0,SEEK_END)!=0){ perror("seek"); fclose(f); return 1; }
    long sz=ftell(f);
    if(sz<=0 || fseek(f,0,SEEK_SET)!=0){ fprintf(stderr,"input must be a non-empty regular file\n"); fclose(f); return 1; }
    std::vector<unsigned char> host((size_t)sz);
    if(fread(host.data(),1,(size_t)sz,f)!=(size_t)sz){ fprintf(stderr,"short read from %s\n",path); fclose(f); return 1; }
    fclose(f);
    if(const char* e = getenv("NVCOMP_PAYLOAD_BYTES")){
        long v = atol(e);
        if(v <= 0 || v > sz){ fprintf(stderr,"NVCOMP_PAYLOAD_BYTES=%s invalid for a %ld-byte file\n", e, sz); return 1; }
        g_payload_bytes = (size_t)v;
    }
    fprintf(stderr,"input: %s  %.1f MiB (payload %.1f MiB)\n", path, sz/1048576.0,
            (g_payload_bytes?g_payload_bytes:(size_t)sz)/1048576.0);
    CK(cudaSetDevice(0));

    // Two presets per HW codec: "hi" = max compression ratio, "fast" = best
    // (de)compression throughput. Deflate exposes a level (algo 0..5): algo=5 is
    // max ratio, algo=0 is "entropy-only, symmetric comp/decomp performance".
    // LZ4 has no level (single-pass) — it is inherently the throughput option.
    int deflate_algos[2] = {5, 0};            // hi, fast
    const char* deflate_names[2] = {"DEFLATE-hi","DEFLATE-fast"};
    if(argc>3){ deflate_algos[0]=atoi(argv[3]); deflate_names[0]="DEFLATE-custom"; deflate_algos[1]=-1; }

    // ---- chunk sweep: report DE's BEST decode over all (chunk x codec) configs ----
    const size_t sweep_chunks[5] = {32*1024, 64*1024, 128*1024, 256*1024, 512*1024};
    const int n_sweep = 5;

    // Per-chunk results, in sweep order. legacy_idx marks the 256 KiB cell that is
    // duplicated at the top level for exact backward-compat.
    std::vector<std::vector<const char*>> sweep_names(n_sweep);
    std::vector<std::vector<CodecResult>> sweep_results(n_sweep);
    int legacy_idx = -1;
    for(int ci=0; ci<n_sweep; ci++){
        size_t chunk = sweep_chunks[ci];
        if(chunk==LEGACY_CHUNK) legacy_idx = ci;
        fprintf(stderr,"\n== chunk %zu KiB (%zu chunks) ==\n", chunk/1024, (size_t)((sz+chunk-1)/chunk));
        run_all_codecs(host, chunk, deflate_algos, deflate_names,
                       sweep_names[ci], sweep_results[ci]);
    }

    // best over the WHOLE sweep (valid configs only)
    const char* best_codec = "";
    double best_decode = 0.0, best_ratio = 0.0;
    size_t best_chunk = 0;
    bool any_validation_failed = false;
    for(int ci=0; ci<n_sweep; ci++){
        for(size_t k=0;k<sweep_results[ci].size();k++){
            const CodecResult& r = sweep_results[ci][k];
            any_validation_failed = any_validation_failed || r.validation_failed;
            if(r.valid && r.decode_gib_s > best_decode){
                best_decode = r.decode_gib_s;
                best_ratio = r.ratio;
                best_codec = sweep_names[ci][k];
                best_chunk = sweep_chunks[ci];
            }
        }
    }

    // ---- emit JSON ----
    // Top level preserves the legacy contract: raw_bytes, chunk_bytes (=256 KiB),
    // the four per-codec objects (the 256 KiB cell), and best_* (now the sweep max).
    if(legacy_idx<0){ fprintf(stderr,"FATAL: 256 KiB cell missing from sweep\n"); return 1; }
    const std::vector<const char*>& lnames = sweep_names[legacy_idx];
    const std::vector<CodecResult>& lres = sweep_results[legacy_idx];

    printf("{\n");
    // EVERY BASIS IS RECORDED; NONE IS INFERRED. raw_bytes stays the FILE size, because
    // de_stage.py validates it against the dumped byte count and figures/suite.py recomputes rate
    // from it -- changing its meaning in place would have made those consumers silently wrong
    // rather than loudly broken. The new fields are additive and say what the numbers are on:
    //   payload_bytes  the ratio's and rate's numerator -- the useful string bytes
    //   file_bytes     what was actually compressed, payload plus framing
    //   framing        how rows are delimited in the compressed stream, or "flat" for none
    // A consumer can now recompute any basis and check it against `ratio` instead of trusting a
    // convention it cannot see. An old consumer keeps reading raw_bytes and keeps working.
    printf("  \"raw_bytes\": %ld,\n", sz);
    printf("  \"file_bytes\": %ld,\n", sz);
    printf("  \"payload_bytes\": %zu,\n", g_payload_bytes ? g_payload_bytes : (size_t)sz);
    printf("  \"framing\": \"%s\",\n", g_payload_bytes ? "u32-length" : "flat");
    printf("  \"chunk_bytes\": %zu,\n", (size_t)LEGACY_CHUNK);
    printf("  \"codecs\": {\n");
    for(size_t k=0;k<lres.size();k++)
        print_codec_obj("    ", lnames[k], lres[k], k+1<lres.size());
    printf("  },\n");

    // chunk_sweep: every chunk size, with the same per-codec objects.
    printf("  \"chunk_sweep\": [\n");
    for(int ci=0; ci<n_sweep; ci++){
        printf("    {\n");
        printf("      \"chunk_bytes\": %zu,\n", sweep_chunks[ci]);
        printf("      \"codecs\": {\n");
        const std::vector<const char*>& nm = sweep_names[ci];
        const std::vector<CodecResult>& rs = sweep_results[ci];
        for(size_t k=0;k<rs.size();k++)
            print_codec_obj("        ", nm[k], rs[k], k+1<rs.size());
        printf("      }\n");
        printf("    }%s\n", ci+1<n_sweep?",":"");
    }
    printf("  ],\n");

    printf("  \"best_codec\": \"%s\",\n", best_codec);
    printf("  \"best_chunk_bytes\": %zu,\n", best_chunk);
    printf("  \"best_decode_gib_s\": %.1f,\n", best_decode);
    printf("  \"best_ratio\": %.2f\n", best_ratio);
    printf("}\n");
    if(any_validation_failed){
        fprintf(stderr,"FATAL: at least one accepted configuration failed validation\n");
        return 4;
    }
    if(best_chunk==0){
        fprintf(stderr,"FATAL: no supported, byte-exact hardware configuration\n");
        return 5;
    }
    return 0;
}
