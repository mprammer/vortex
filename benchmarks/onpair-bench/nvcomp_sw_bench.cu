// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Standalone nvCOMP SOFTWARE (SM/CUDA) codec baseline: gANS + Bitcomp.
// The §7 companion to nvcomp_hw_bench.cu (the fixed-function Decompression Engine).
// These are nvCOMP's GPU-optimized SM-based codecs (NOT DE codecs): the paper's
// "nvCOMP's faster software codecs, gANS and Bitcomp" comparison. This bench
// measures their DECODE throughput + compression ratio on the SAME uncompressed
// column bytes OnPair decodes, so the §7 "FastPair decodes faster than both at a
// higher compression ratio" sentence is backed (or revised) by real data.
//
// SAME MEASUREMENT PROTOCOL AS THE DE BENCH (do not diverge — comparability):
//   identical uncompressed input bytes; batched compress; 3 warmups; MIN over 100
//   CUDA-event-timed single-pass decodes; memcmp byte-exact validation; decode
//   GiB/s over the *uncompressed* bytes; chunk sweep {32,64,128,256,512} KiB with
//   best-over-sweep reported; per-codec raw decode_ns_iters (GOLD provenance).
//   The run<> template + JSON emitter are copied verbatim from nvcomp_hw_bench.cu
//   (proven); only the codecs and the DECOMPRESS backend differ.
//
// BACKEND: these codecs have NO DE path, so decompress runs on the CUDA/SM backend.
//   We set DecompressOpts .backend = NVCOMP_DECOMPRESS_BACKEND_CUDA EXPLICITLY (not DEFAULT,
//   which prioritizes the DE for compatible formats) so the SM baseline can't drift to
//   hardware in a future SDK. We never set NVCOMP_DECOMPRESS_BACKEND_HARDWARE.
//
// API (opts TYPES verified against nvCOMP 5.1.0.21 C API — the repo's pinned SDK — via
// the gpt-5.6-sol meta-review of the gauntlet findings; a few FIELD-level items remain
// for on-box compile):
//   - nvcomp/ans.h:     compress opts = nvcompBatchedANSCompressOpts_t (NOT the bare
//                       ...Opts_t); its data-type field is `type` (NOT `algorithm`), and
//                       the byte DEFAULT is NVCOMP_TYPE_CHAR — so we leave it at default
//                       rather than risk a wrong field-name. `nvcompBatchedANSCompressDefaultOpts`
//                       + the {Compress,Decompress}{GetTempSizeAsync,...,Async} family.
//   - nvcomp/bitcomp.h: compress opts = nvcompBatchedBitcompCompressOpts_t (NOT
//                       ...FormatOpts_t); `.algorithm` = 0 (default) / 1 (sparse); byte
//                       default NVCOMP_TYPE_UCHAR. NOTE: this is the LOW-LEVEL batched
//                       decompress, which nvCOMP documents as NOT fully async (it syncs to
//                       inspect the data). A faster HLIF BitcompManager path exists — so the
//                       codec names below are "Bitcomp-*" and this bench must NOT be read as
//                       the fastest-possible Bitcomp; it is the low-level C API baseline.
//   - Decompress backend set to NVCOMP_DECOMPRESS_BACKEND_CUDA explicitly (below): DEFAULT
//     prioritizes the DE for compatible formats — ANS/Bitcomp resolve to CUDA on the B300
//     today, but the explicit CUDA pin stops the SM baseline drifting in a future SDK.
//   ON-BOX VERIFY: exact opts data-type field names; ANS/Bitcomp alignment via
//     *GetRequiredAlignments (round the compressed-chunk stride if a codec errors).
//   - Bitcomp targets numeric/sparse data — expect POOR ratio on text columns. That is
//     a valid measurement, not a bug: it directly informs the §7 claim.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include <nvcomp/ans.h>
#include <nvcomp/bitcomp.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define NK(x) do{ nvcompStatus_t s=(x); if(s!=nvcompSuccess){ fprintf(stderr,"nvcomp %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

static const size_t LEGACY_CHUNK = 262144; // 256 KiB, the duplicated top-level cell

// One codec's result at one chunk size. `valid` is false when the codec rejected
// this config or the byte round-trip failed; scalars are 0 and iters empty then.
struct CodecResult {
    double ratio = 0.0;          // raw_bytes / compressed_bytes
    double compress_gib_s = 0.0; // encode throughput over uncompressed bytes
    double decode_gib_s = 0.0;   // min-reduced decode throughput over uncompressed bytes
    bool valid = false;          // supported AND byte-exact
    std::vector<unsigned long long> decode_ns_iters; // GOLD: raw per-iter decode times, integer ns
};

// Generic codec runner — VERBATIM from nvcomp_hw_bench.cu (proven). Compress +
// (2 warmups + 100 timed) decode for one codec at one chunk size; MIN over iters.
// On codec rejection (GetTempSize or the first async), marks invalid and returns
// WITHOUT aborting the sweep. All device allocations + stream/events freed before
// return so the sweep (10 invocations) does not leak.
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
    cudaEvent_t ca,cb; CK(cudaEventCreate(&ca)); CK(cudaEventCreate(&cb));
    int citers=20; CK(cudaEventRecord(ca,stream));
    for(int i=0;i<citers;i++) NK(compAsync(d_inptr,d_insz,chunk,num,d_ctemp,ctemp,d_cptr,d_csz,copts,d_st,stream));
    CK(cudaEventRecord(cb,stream)); CK(cudaEventSynchronize(cb));
    float cms=0; CK(cudaEventElapsedTime(&cms,ca,cb)); cms/=citers;
    double enc_gibs=(double)N/(cms/1e3)/(1024.0*1024*1024);
    std::vector<size_t> h_csz(num); CK(cudaMemcpy(h_csz.data(),d_csz,num*sizeof(size_t),cudaMemcpyDeviceToHost));
    size_t ctot=0; for(size_t i=0;i<num;i++) ctot+=h_csz[i];

    // ---- decompress on the CUDA/SM backend (dopts.backend set to CUDA by the caller) ----
    size_t dtemp=0; nvcompStatus_t ds=decompTemp(num,chunk,dopts,&dtemp,N);
    if(ds!=nvcompSuccess){
        fprintf(stderr,"%-13s chunk=%zu decompress GetTempSize status=%d (UNSUPPORTED)\n",name,chunk,(int)ds);
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
    size_t* d_actual; CK(cudaMalloc(&d_actual,num*sizeof(size_t))); // Bitcomp/ANS require non-null

    nvcompStatus_t dw = decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream);
    if(dw!=nvcompSuccess){
        fprintf(stderr,"%-13s chunk=%zu decompress async status=%d (UNSUPPORTED)\n",name,chunk,(int)dw);
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
        if(d_ctemp) cudaFree(d_ctemp);
        cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
        if(d_dtemp) cudaFree(d_dtemp);
        cudaFree(d_out); cudaFree(d_optr); cudaFree(d_obufsz); cudaFree(d_actual);
        cudaStreamDestroy(stream);
        out.valid=false; return;
    }
    for(int w=1;w<3;w++){ NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream)); }
    CK(cudaStreamSynchronize(stream));

    std::vector<unsigned char> back(N); CK(cudaMemcpy(back.data(),d_out,N,cudaMemcpyDeviceToHost));
    bool ok = memcmp(back.data(),host.data(),N)==0;

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    int iters=100; float ms=1e30f;
    out.decode_ns_iters.reserve((size_t)iters);
    for(int i=0;i<iters;i++){
        CK(cudaEventRecord(a,stream));
        NK(decompAsync(d_cptr,d_csz,d_obufsz,d_actual,num,d_dtemp,dtemp,d_optr,dopts,d_st,stream));
        CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b));
        float it=0; CK(cudaEventElapsedTime(&it,a,b));
        out.decode_ns_iters.push_back((unsigned long long)((double)it*1e6+0.5));
        if(it<ms) ms=it;
    }
    double gibs = (double)N/(ms/1e3)/ (1024.0*1024*1024);
    double gbs  = (double)N/(ms/1e3)/ 1e9;
    fprintf(stderr,"%-13s chunk=%6zu  ratio=%.2fx  compress=%6.1f GiB/s  decode=%6.1f GiB/s (%.0f GB/s)  valid=%s\n",
           name, chunk, (double)N/ctot, enc_gibs, gibs, gbs, ok?"YES":"NO");

    // Publish rates/samples ONLY on a validated round-trip (CodecResult's invalid
    // contract): a failed cell keeps default-zero scalars + an empty iters vector, so it
    // can never surface as best_*. (Tightens the copied-from-hw-bench behavior.)
    out.valid = ok;
    if(ok){
        out.ratio = (double)N/ctot;
        out.compress_gib_s = enc_gibs;
        out.decode_gib_s = gibs;
    } else {
        out.decode_ns_iters.clear();
    }

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaEventDestroy(ca); cudaEventDestroy(cb);
    cudaFree(d_in); cudaFree(d_inptr); cudaFree(d_insz);
    if(d_ctemp) cudaFree(d_ctemp);
    cudaFree(d_cbuf); cudaFree(d_cptr); cudaFree(d_csz); cudaFree(d_st);
    if(d_dtemp) cudaFree(d_dtemp);
    cudaFree(d_out); cudaFree(d_optr); cudaFree(d_obufsz); cudaFree(d_actual);
    cudaStreamDestroy(stream);
}

// Run the two SW codecs (gANS, Bitcomp) at one chunk size. gANS has no level
// knob for our purposes; Bitcomp sweeps its two algorithms (0=default, 1=sparse).
static void run_all_codecs(std::vector<unsigned char>& host, size_t chunk,
                           std::vector<const char*>& names,
                           std::vector<CodecResult>& results)
{
    names.clear();
    results.clear();

    // --- gANS (entropy coder; lossless on arbitrary bytes) ---
    {
        nvcompBatchedANSCompressOpts_t aco = nvcompBatchedANSCompressDefaultOpts;
        // default data type is NVCOMP_TYPE_CHAR (byte data) — correct for strings; left
        // unset to avoid a wrong-field-name compile risk (the field is `type`, on-box-verify).
        nvcompBatchedANSDecompressOpts_t ado = nvcompBatchedANSDecompressDefaultOpts;
        ado.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;  // force SM path, not DEFAULT->DE
        CodecResult r;
        run<nvcompBatchedANSCompressOpts_t,nvcompBatchedANSDecompressOpts_t>(
            "gANS", host, chunk, aco, ado, r,
            nvcompBatchedANSCompressGetTempSizeAsync, nvcompBatchedANSCompressGetMaxOutputChunkSize,
            nvcompBatchedANSCompressAsync, nvcompBatchedANSDecompressGetTempSizeAsync,
            nvcompBatchedANSDecompressAsync);
        names.push_back("gANS");
        results.push_back(std::move(r));
    }

    // --- Bitcomp, both algorithms (0=default best-ratio, 1=sparse/faster) ---
    // Bitcomp targets numeric/sparse data; on text expect poor ratio (a valid
    // measurement for the §7 claim, not a failure).
    {
        const int bc_algos[2] = {0, 1};
        const char* bc_names[2] = {"Bitcomp-default", "Bitcomp-sparse"};
        for(int p=0;p<2;p++){
            nvcompBatchedBitcompCompressOpts_t bco = nvcompBatchedBitcompCompressDefaultOpts;
            bco.algorithm = bc_algos[p];  // 0=default, 1=sparse
            // default data type is NVCOMP_TYPE_UCHAR (byte data); left unset (on-box-verify).
            nvcompBatchedBitcompDecompressOpts_t bdo = nvcompBatchedBitcompDecompressDefaultOpts;
            bdo.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;  // force SM path, not DEFAULT->DE
            CodecResult r;
            run<nvcompBatchedBitcompCompressOpts_t,nvcompBatchedBitcompDecompressOpts_t>(
                bc_names[p], host, chunk, bco, bdo, r,
                nvcompBatchedBitcompCompressGetTempSizeAsync, nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
                nvcompBatchedBitcompCompressAsync, nvcompBatchedBitcompDecompressGetTempSizeAsync,
                nvcompBatchedBitcompDecompressAsync);
            names.push_back(bc_names[p]);
            results.push_back(std::move(r));
        }
    }
}

// JSON codec-object emitter — VERBATIM from nvcomp_hw_bench.cu.
static void print_codec_obj(const char* indent, const char* name, const CodecResult& r,
                            bool trailing_comma)
{
    printf("%s\"%s\": {\n", indent, name);
    printf("%s  \"ratio\": %.2f,\n", indent, r.ratio);
    printf("%s  \"compress_gib_s\": %.1f,\n", indent, r.compress_gib_s);
    printf("%s  \"decode_gib_s\": %.1f,\n", indent, r.decode_gib_s);
    printf("%s  \"valid\": %s,\n", indent, r.valid?"true":"false");
    printf("%s  \"decode_ns_iters\": [", indent);
    for(size_t i=0;i<r.decode_ns_iters.size();i++)
        printf("%s%llu", i?",":"", (unsigned long long)r.decode_ns_iters[i]);
    printf("]\n");
    printf("%s}%s\n", indent, trailing_comma?",":"");
}

int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"/tmp/l_comment.bin";
    FILE* f=fopen(path,"rb"); if(!f){ perror("open"); return 1; }
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<unsigned char> host(sz); fread(host.data(),1,sz,f); fclose(f);
    fprintf(stderr,"input: %s  %.1f MiB\n", path, sz/1048576.0);
    CK(cudaSetDevice(0));

    const size_t sweep_chunks[5] = {32*1024, 64*1024, 128*1024, 256*1024, 512*1024};
    const int n_sweep = 5;
    std::vector<std::vector<const char*>> sweep_names(n_sweep);
    std::vector<std::vector<CodecResult>> sweep_results(n_sweep);
    int legacy_idx = -1;
    for(int ci=0; ci<n_sweep; ci++){
        size_t chunk = sweep_chunks[ci];
        if(chunk==LEGACY_CHUNK) legacy_idx = ci;
        fprintf(stderr,"\n== chunk %zu KiB (%zu chunks) ==\n", chunk/1024, (size_t)((sz+chunk-1)/chunk));
        run_all_codecs(host, chunk, sweep_names[ci], sweep_results[ci]);
    }

    const char* best_codec = "";
    double best_decode = 0.0, best_ratio = 0.0;
    for(int ci=0; ci<n_sweep; ci++){
        for(size_t k=0;k<sweep_results[ci].size();k++){
            const CodecResult& r = sweep_results[ci][k];
            if(r.valid && r.decode_gib_s > best_decode){
                best_decode = r.decode_gib_s; best_ratio = r.ratio; best_codec = sweep_names[ci][k];
            }
        }
    }

    if(legacy_idx<0){ fprintf(stderr,"FATAL: 256 KiB cell missing from sweep\n"); return 1; }
    const std::vector<const char*>& lnames = sweep_names[legacy_idx];
    const std::vector<CodecResult>& lres = sweep_results[legacy_idx];

    printf("{\n");
    printf("  \"raw_bytes\": %ld,\n", sz);
    printf("  \"chunk_bytes\": %zu,\n", (size_t)LEGACY_CHUNK);
    printf("  \"codecs\": {\n");
    for(size_t k=0;k<lres.size();k++)
        print_codec_obj("    ", lnames[k], lres[k], k+1<lres.size());
    printf("  },\n");
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
    printf("  \"best_decode_gib_s\": %.1f,\n", best_decode);
    printf("  \"best_ratio\": %.2f\n", best_ratio);
    printf("}\n");
    return 0;
}
