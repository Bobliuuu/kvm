#pragma once
// GPT operator library for the KVM megakernel framework: header-only block-cooperative device functions
//   __device__ void kvm_op_<name>(const KvmCtx& ctx, const int32_t* args)
// one per entry of kvm_ops_gpt.def, plus kvm_gpt_dispatch(). Generic and shape-parametrized: no model-specific code.
//
// Requirements on the includer
//  * KvmCtx must be declared before this header and provide:  const KvmBuf* bufs;  int nbufs;  and
//    __device__ void fail(int code, int a = 0, int b = 0) const  (first error wins, thread safe, callable from any thread).
//    kvm_exec.cuh's KvmCtx is exactly that; tests/gpu_ops/kvm_ctx_shim.cuh is the standalone equivalent. Nothing else of the
//    executor is used.
//  * Included from kvm_exec.cuh (KVM_OP defined) the entry points are KVM_OP(name) { ... } for every X line of
//    kvm_ops_gpt.def; standalone they are kvm_op_<name> plus kvm_gpt_dispatch().
//  * Shared memory: ops use DYNAMIC shared memory from offset 0 (extern __shared__), up to KVM_GPT_SMEM_BYTES (matmul 48 KiB,
//    attention 6 tiles of 32 x (hd+4) floats: 12..52 KiB, 101 KiB at hd=128). The kernel launch must pass that many dynamic bytes (and call
//    cudaFuncSetAttribute(MaxDynamicSharedMemorySize) when above 48 KiB). The executor's own static scratch is not used.
//  * Every op must be run by a block of exactly KVM_GPT_THREADS (256) threads, all threads entering together; ops use
//    __syncthreads() freely and take dynamic shared memory from offset 0 (kvm_gpt_smem_bytes()). Between two ops of one block
//    the executor must __syncthreads() (the ops leave shared memory dirty).
//  * Memory ordering: an op only reads what its DAG predecessors published and writes plain global stores; the executor's
//    release/acquire (device fence + version publish) around a task provides all cross-task visibility. No op reads data
//    written by a concurrently running task, and no two concurrent tasks write the same address (backward ops are split so each
//    output element has a single writer; accumulation order is fixed, so results are bitwise reproducible run to run).
//
// Numerics (fp32 storage everywhere). Tolerances are relative to the max abs of the reference output unless stated:
//   matmul_tile 1xTF32: 4e-3 (10-bit mantissa inputs, fp32 accumulate); precise (3xTF32): 2e-6 * sqrt(K)
//   norms / gelu / rope / residual / embeddings / adamw / ce: 1e-5 (fp32 reductions, 32-lane tree; tanhf/expf full precision)
//   attention fwd/bwd: 2e-5 (3xTF32 tensor-core products with fp32 accumulate and online softmax; measured 1e-6..1.2e-5 for T <= 2048; dK/dV grow with T, the tensor core truncating its accumulator once per mma)   decode: 2e-5   sample: exact index vs a double reference (up to
//   measure-zero CDF boundary cases)
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>
#include "kvm_task_abi.h"
#include "kvm_hopper.cuh"   // sm_90a TMA + wgmma tiles (no-op on every other target)

#define KVM_GPT_THREADS 256
#define KVM_GPT_DEVI __device__ __forceinline__
// Phase probe for op-body profiling beds: expands to nothing in the executor (a bed defines it before including this header to timestamp phases of a task body).
#ifndef KVM_GPT_PROBE
#define KVM_GPT_PROBE(n)
#endif
// Rows one causal attention task covers: 128 queries (attn_fwd, attn_bwd_dq: rows [q0, q0+128)) or 128 keys (attn_bwd_dkv:
// [k0, k0+128)), 16 per warp. Callers must step q0 / k0 by exactly this value: tasks that overlap write the same output rows from
// differently grouped sums (equal to rounding, not bitwise), which is a write race and breaks the reproducibility contract.
// Mirrored by ATTN_BLOCK in frontend/kvmfe/opset.py. The bidirectional attn_full_* ops keep their own 32-row blocks.
#define KVM_GPT_ATTN_ROWS 128
static_assert(KVM_GPT_THREADS / 32 * 16 == KVM_GPT_ATTN_ROWS, "attention: one m16 row tile per warp");


#ifndef KVM_GPT_SMEM_CVT
#define KVM_GPT_SMEM_CVT 1   // 1: matmul operands are rounded to tf32 once per CTA in shared memory (tile_cvt); 0: per fragment, per warp
#endif
#ifndef KVM_GPT_TILE64_NST
#define KVM_GPT_TILE64_NST 3    // pipeline stages of the 64x64 tile (a 4th stage measured no gain at the panel's shapes)
#endif
// TileStream (per-thread copy pointers precomputed once) against per-chunk address arithmetic: it starts a tile sooner (first-stage wait, a short K's cost) and streams slower per K slab (the long K's cost).
// Measured, 64x64 tile: chain 9.6 against 13.6 us at K=72, 13.7 against 14.5 at K=96, 61 against 51 at K=768, 370 against 249 at K=4096; so K <= KMAX streams.
#ifndef KVM_GPT_TILE64_STREAM_KMAX
#define KVM_GPT_TILE64_STREAM_KMAX 127
#endif
#ifndef KVM_GPT_BIG_PIPE2
#define KVM_GPT_BIG_PIPE2 1   // 1: the truncated big tile's k-step pipeline (see mm_tile_big)
#endif
#ifndef KVM_GPT_BIG_PIPE_RND
#define KVM_GPT_BIG_PIPE_RND 1   // the rounded big tile: 1: the pipeline, each thread rounding the chunks it copied once they land, before the barrier that publishes them (0.73-0.84x cuBLAS at the 162M shapes; rounding each fragment as it is loaded measured 0.66-0.74x, every warp that reads a value rounding it again); 0: the plain loop with rounding in shared memory (0.53-0.60x)
#endif
#ifndef KVM_GPT_RZ_GAIN
#define KVM_GPT_RZ_GAIN 1.000704f   // output scale of the truncated-operand GEMM (flag 256): the tensor core drops the low 13 bits of each operand, so a product shrinks by about 2 * 2^-11 * E[1/mantissa] = 7.04e-4 for log-uniform mantissas. Scaling the tile's output removes that mean, the systematic part of the truncation error (the numerics gates of the panel's rows pass with it, and fail without it)
#endif
#ifndef KVM_GPT_BIG_STREAM_KMAX
#define KVM_GPT_BIG_STREAM_KMAX 0x40000000   // big tiles: the same switch (0: never, default: always)
#endif
#ifndef KVM_GPT_BIG_BK
#define KVM_GPT_BIG_BK 16    // k per pipeline stage of the big tiles (16 or 32). A100, one block per SM, the 256x128 tile at the 162M row's shapes
                             // (Nov 10, tests/gpu_ops/bench_tile.cu): BK 16 x 4 stages 0.69-0.72x cuBLAS NN and 0.55x TN at 96 KiB, against
                             // BK 32 x 3's 0.73x and 0.56x at 144 KiB; BK 32 x 2 (96 KiB) 0.41-0.45x and 0.31x. A 144 KiB launch leaves every
                             // op 28 KB of L1 (the same executor launched at 90 KB ran gpt-tiny x1.12 on an A100). The 0.41-0.55x measured
                             // earlier for BK 16 was read at two blocks per SM, where 128 registers make the 256x128 tile spill.
#endif
#ifndef KVM_GPT_BIG_NST
#define KVM_GPT_BIG_NST 4    // pipeline stages of the big tiles (4 stages of (BM+BN) x 16 floats: 128x128 64 KiB, 256x128 96 KiB, one block
                             // per SM). Small-memory parts (99 KiB opt-in, sm_86/sm_89/sm_120) build -DKVM_GPT_BIG_NST=3: 96 KiB plus the
                             // kernel's static shared memory is over their limit.
#endif
// Dynamic shared memory an op needs for a launch of the op set: 104 KiB covers hd=128 attention backward/forward (6 tiles of
// 32 x 132 floats = 101 KiB); 56 KiB covers every other op (matmul 48 KiB, attention up to hd=64 at 52 KiB), so two blocks per SM still
// fit in the 164 KiB of an A100. The executor must opt in with cudaFuncSetAttribute(MaxDynamicSharedMemorySize) (it does, from
// KVM_GPT_SMEM_BYTES).
// KVM_GPT_MMA_SMEM: the GEMM tiles' stage buffers, 96 KiB (4 stages of 128 x 256 x 32 bf16; fits the 99 KiB opt-in limit of sm_86/sm_89/
// sm_120). -DKVM_GPT_MMA_SMEM=147456 (A100/H100/B200 only) also compiles the 256 x 128 TF32 tile with 3 stages of BK = 32, 0.03-0.05x of
// cuBLAS faster in isolation, but the executor then launches with 144 KiB of shared memory, the L1 carveout shrinks for every other op, and
// the gpt_162m step got slower (f32 249 vs 229 ms, bf16s 270 vs 257 ms; results/2026-11-09-a100-matmul-parity). mm_tf32_tile picks by
// the launch's %dynamic_smem_size, so either build stays correct.
// #13's tiles: the big tiles' stages (KVM_GPT_BIG_BK x KVM_GPT_BIG_NST) enter the executor's launch through kvm_gpt_smem_bytes. At the
// default BK 16 x 4 the 256 x 128 tile needs 96 KiB, the same as KVM_GPT_MMA_SMEM's tiles, so a default build launches with 96 KiB;
// BK 32 x 3 (-DKVM_GPT_BIG_BK=32 -DKVM_GPT_BIG_NST=3) would set it to 144 KiB, at the L1 cost above.
#ifndef KVM_GPT_MMA_SMEM
#define KVM_GPT_MMA_SMEM (96 * 1024)   // matmul_mma_bf16 (kvm_mma_bf16.cuh): 4 stages of 128 x 256 x 32 bf16; fits the 99 KiB opt-in limit of sm_86/sm_120
#endif
__host__ __device__ constexpr int kvm_gpt_smem_bytes(int max_hd) {
    const int tiles = 1536 * KVM_GPT_BIG_BK * KVM_GPT_BIG_NST > KVM_GPT_MMA_SMEM ? 1536 * KVM_GPT_BIG_BK * KVM_GPT_BIG_NST : KVM_GPT_MMA_SMEM;   // (256+128)*4 bytes per k of the 256x128 tile, per stage
    return max_hd > 64 && tiles < 104 * 1024 ? 104 * 1024 : tiles;
}
#ifndef KVM_GPT_MAX_HD
#define KVM_GPT_MAX_HD 64   // largest attention head size the launch must support (16/32/64 -> 56 KiB, 128 -> 104 KiB)
#endif
#define KVM_GPT_SMEM_BYTES kvm_gpt_smem_bytes(KVM_GPT_MAX_HD)

extern __shared__ __align__(128) unsigned char kvm_gpt_smem[];

// Device error codes reported through ctx.fail (distinct from the executor's own KVM_ERR_* values, which are small).
#define KVM_GPT_ERR_BASE 1000
enum KvmGptErr {
    KVM_GPT_E_BUF = KVM_GPT_ERR_BASE + 1,     // buffer index out of the table
    KVM_GPT_E_DTYPE = KVM_GPT_ERR_BASE + 2,   // null pointer or wrong dtype (f32 = 0, i32 = 1)
    KVM_GPT_E_RANGE = KVM_GPT_ERR_BASE + 3,   // range / shape arguments outside the buffer
    KVM_GPT_E_SHAPE = KVM_GPT_ERR_BASE + 4,   // unsupported shape (head size), bad flags or misaligned buffer
    KVM_GPT_E_TOKEN = KVM_GPT_ERR_BASE + 5,   // token id / target outside [0, V)
    KVM_GPT_E_KVLEN = KVM_GPT_ERR_BASE + 6    // KV cache length overflow
};
enum { KVM_GPT_F32 = 0, KVM_GPT_I32 = 1, KVM_GPT_BF16 = 2 };   // KvmBuf.dtype; bf16 = raw 16-bit storage

namespace kvmgpt {

// ---- argument validation: all checks are block-uniform, so on failure every thread returns before any barrier -------------
struct Chk {
    const KvmCtx& c;
    bool bad;
    KVM_GPT_DEVI explicit Chk(const KvmCtx& cc) : c(cc), bad(false) {}
    KVM_GPT_DEVI void fail(int code) {   // any thread may report (token ids fail per thread); first error wins in ctx.fail
        if (!bad) c.fail(code);
        bad = true;
    }
    KVM_GPT_DEVI void need(bool ok, int code = KVM_GPT_E_RANGE) { if (!ok) fail(code); }
    KVM_GPT_DEVI const KvmBuf* get(int idx, int dtype, bool opt) {
        if (idx < 0 && opt) return nullptr;
        if (idx < 0 || idx >= c.nbufs) { fail(KVM_GPT_E_BUF); return nullptr; }
        const KvmBuf* b = c.bufs + idx;
        if (!b->p || (dtype >= 0 && b->dtype != dtype)) { fail(KVM_GPT_E_DTYPE); return nullptr; }
        return b;
    }
    // [rows x cols] matrix view with leading dimension *ld. Absent (opt) -> nullptr.
    KVM_GPT_DEVI float* mat(int idx, long rows, long cols, int* ld = nullptr, bool opt = false) {
        const KvmBuf* b = get(idx, KVM_GPT_F32, opt);
        if (!b) return nullptr;
        if (rows > b->rows || cols > b->cols || b->ld < b->cols) { fail(KVM_GPT_E_RANGE); return nullptr; }
        if (ld) *ld = b->ld;
        return (float*)b->p;
    }
    // [rows x cols] matrix view of a buffer that is fp32 (dtype 0) or raw bf16 (dtype 2); *bf tells which. Absent (opt) -> nullptr.
    KVM_GPT_DEVI void* mat_s(int idx, long rows, long cols, int* ld, bool* bf, bool opt = false) {
        if (idx < 0 && opt) return nullptr;
        if (idx < 0 || idx >= c.nbufs) { fail(KVM_GPT_E_BUF); return nullptr; }
        const KvmBuf* b = c.bufs + idx;
        if (!b->p || (b->dtype != KVM_GPT_F32 && b->dtype != KVM_GPT_BF16)) { fail(KVM_GPT_E_DTYPE); return nullptr; }
        if (rows > b->rows || cols > b->cols || b->ld < b->cols) { fail(KVM_GPT_E_RANGE); return nullptr; }
        if (ld) *ld = b->ld;
        *bf = b->dtype == KVM_GPT_BF16;
        return b->p;
    }
    // contiguous vector of n elements of a buffer that is fp32 or raw bf16 (1 row, or ld == cols); *bf tells which.
    KVM_GPT_DEVI void* vec_s(int idx, long n, bool* bf) {
        if (idx < 0 || idx >= c.nbufs) { fail(KVM_GPT_E_BUF); return nullptr; }
        const KvmBuf* b = c.bufs + idx;
        if (!b->p || (b->dtype != KVM_GPT_F32 && b->dtype != KVM_GPT_BF16)) { fail(KVM_GPT_E_DTYPE); return nullptr; }
        if ((long)b->rows * b->cols < n || (b->rows > 1 && b->ld != b->cols)) { fail(KVM_GPT_E_RANGE); return nullptr; }
        *bf = b->dtype == KVM_GPT_BF16;
        return b->p;
    }
    // contiguous vector of n elements (a 1-row buffer, or ld == cols)
    template <typename T> KVM_GPT_DEVI T* vec(int idx, long n, int dtype, bool opt = false) {
        const KvmBuf* b = get(idx, dtype, opt);
        if (!b) return nullptr;
        if ((long)b->rows * b->cols < n || (b->rows > 1 && b->ld != b->cols)) { fail(KVM_GPT_E_RANGE); return nullptr; }
        return (T*)b->p;
    }
    KVM_GPT_DEVI float* fvec(int idx, long n, bool opt = false) { return vec<float>(idx, n, KVM_GPT_F32, opt); }
    KVM_GPT_DEVI int32_t* ivec(int idx, long n, bool opt = false) { return vec<int32_t>(idx, n, KVM_GPT_I32, opt); }
};
KVM_GPT_DEVI float af(const int32_t* a, int i) { return __int_as_float(a[i]); }
KVM_GPT_DEVI bool aligned16(const void* p, int ld) { return (((uintptr_t)p & 15) == 0) && ((ld & 3) == 0); }

// Storage-dtype helpers: element i of a buffer that is either fp32 or raw bf16 (KvmBuf.dtype 2). bf16 -> fp32 is exact; fp32 -> bf16 is
// round to nearest even, once per store. Ops that accept bf16 storage read and write through these (the fp32 fast paths stay as they were).
KVM_GPT_DEVI float ld_s(const void* p, size_t i, bool bf) { return bf ? __uint_as_float((uint32_t)((const uint16_t*)p)[i] << 16) : ((const float*)p)[i]; }
KVM_GPT_DEVI uint16_t bf16_bits(float x) {   // round to nearest even, NaN stays NaN
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
KVM_GPT_DEVI void st_s(void* p, size_t i, float x, bool bf) { if (bf) ((uint16_t*)p)[i] = bf16_bits(x); else ((float*)p)[i] = x; }
KVM_GPT_DEVI float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }            // the two bf16 halves of a 32-bit word, widened (exact)
KVM_GPT_DEVI float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
KVM_GPT_DEVI uint2 bf16x4(float a, float b, float c, float d) { return make_uint2(bf16_bits(a) | ((uint32_t)bf16_bits(b) << 16), bf16_bits(c) | ((uint32_t)bf16_bits(d) << 16)); }
// Four consecutive elements starting at element i of an fp32 (16-byte accesses) or bf16 (8-byte accesses) buffer. vec4_ok: the base pointer and
// the leading dimension allow them for every row.
KVM_GPT_DEVI float4 ld4(const void* p, size_t i, bool bf) {
    if (bf) { const uint2 u = *reinterpret_cast<const uint2*>((const uint16_t*)p + i); return make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y)); }
    return *reinterpret_cast<const float4*>((const float*)p + i);
}
KVM_GPT_DEVI void st4(void* p, size_t i, float4 v, bool bf) {
    if (bf) *reinterpret_cast<uint2*>((uint16_t*)p + i) = bf16x4(v.x, v.y, v.z, v.w);
    else *reinterpret_cast<float4*>((float*)p + i) = v;
}
KVM_GPT_DEVI bool vec4_ok(const void* p, int ld, bool bf) { return (((uintptr_t)p & (bf ? 7 : 15)) == 0) && ((ld & 3) == 0); }

// ---- reductions -------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI float wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// Sum over the warp's group of d4 lanes that share a row when one warp works on 32/d4 rows at once (rpw > 1, d4 a power of two), or over the whole warp (rpw == 1).
KVM_GPT_DEVI float gsum(float v, int d4, int rpw) {
    for (int o = rpw > 1 ? d4 / 2 : 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
KVM_GPT_DEVI float wmax(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// exp(x) via ex2.approx (about 2^-22 relative error plus the rounding of x*log2e, < 2e-6 for |x| < 30): the softmax passes are
// bandwidth bound but expf's full-precision path cost enough ALU to leave ~40% of HBM bandwidth unused at one block per SM.
KVM_GPT_DEVI float fexp(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}
// Fixed-order block sum in double (bitwise deterministic): tree over buf[0..255]. Returns the total to every thread.
KVM_GPT_DEVI double bsum_d(double v, double* buf) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = KVM_GPT_THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    double r = buf[0];
    __syncthreads();
    return r;
}

// ---- embeddings -------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void embedding_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[4], r1 = a[5], T = a[6], d = a[7], ldw, ldp = 0, ldo;
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && d > 0);
    if (k.bad) return;
    const int32_t* ids = k.ivec(a[0], r1);
    const float* wte = k.mat(a[1], 0, d, &ldw);
    const float* wpe = k.mat(a[2], T, d, &ldp, true);
    bool ob;
    void* out = k.mat_s(a[3], r1, d, &ldo, &ob);
    if (k.bad) return;
    const int V = ctx.bufs[a[1]].rows, n = (r1 - r0) * d;
    constexpr int U = 8;   // a thread's next U elements in flight at once: ids of the U rows, then the U table entries, instead of two dependent round trips per element
    if (!ob && (d & 3) == 0 && aligned16(wte, ldw) && (!wpe || aligned16(wpe, ldp)) && aligned16(out, ldo)) {   // whole columns in float4s: a pass of two dependent round trips covers U float4 chunks per thread (a task of 2,048 chunks in one pass)
        const int d4 = d >> 2, n4 = (r1 - r0) * d4;
        for (int e = threadIdx.x; e < n4; e += KVM_GPT_THREADS * U) {
            int rr[U], cc[U], id[U]; float4 wv[U], pv[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int q = e + u * KVM_GPT_THREADS;
                rr[u] = r0 + q / d4; cc[u] = 4 * (q % d4);
                id[u] = q < n4 ? ids[rr[u]] : 0;
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int q = e + u * KVM_GPT_THREADS;
                const bool ok = q < n4 && (unsigned)id[u] < (unsigned)V;
                wv[u] = ok ? *reinterpret_cast<const float4*>(wte + (size_t)id[u] * ldw + cc[u]) : make_float4(0.f, 0.f, 0.f, 0.f);
                pv[u] = (q < n4 && wpe) ? *reinterpret_cast<const float4*>(wpe + (size_t)(rr[u] % T) * ldp + cc[u]) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int q = e + u * KVM_GPT_THREADS;
                if (q >= n4) continue;
                if ((unsigned)id[u] >= (unsigned)V) k.fail(KVM_GPT_E_TOKEN);
                *reinterpret_cast<float4*>((float*)out + (size_t)rr[u] * ldo + cc[u]) = make_float4(wv[u].x + pv[u].x, wv[u].y + pv[u].y, wv[u].z + pv[u].z, wv[u].w + pv[u].w);
            }
        }
        return;
    }
    for (int e = threadIdx.x; e < n; e += KVM_GPT_THREADS * U) {
        int rr[U], cc[U], id[U]; float wv[U], pv[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int q = e + u * KVM_GPT_THREADS;
            rr[u] = r0 + q / d; cc[u] = q % d;
            id[u] = q < n ? ids[rr[u]] : 0;
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const bool ok = e + u * KVM_GPT_THREADS < n && (unsigned)id[u] < (unsigned)V;
            wv[u] = ok ? wte[(size_t)id[u] * ldw + cc[u]] : 0.f;
            pv[u] = (e + u * KVM_GPT_THREADS < n && wpe) ? wpe[(size_t)(rr[u] % T) * ldp + cc[u]] : 0.f;
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (e + u * KVM_GPT_THREADS >= n) continue;
            if ((unsigned)id[u] >= (unsigned)V) k.fail(KVM_GPT_E_TOKEN);
            st_s(out, (size_t)rr[u] * ldo + cc[u], wv[u] + pv[u], ob);
        }
    }
}

// Warp w owns keys v0+w, v0+w+8, ...; scans the id array 32 at a time (ballot gives matches in ascending row order), so each
// table row is summed over rows in ascending order by one warp: deterministic. Positional tables (ids absent) enumerate
// r = key + i*mod directly. Cost O(N*keys/32) for token tables; give each task a modest key range.
KVM_GPT_DEVI void embedding_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], mod = a[4], v0 = a[5], v1 = a[6], d = a[7], acc = a[8], lddo, lddt;
    k.need(N >= 0 && v0 >= 0 && v0 <= v1 && d > 0);
    if (k.bad) return;
    const int32_t* ids = k.ivec(a[0], N, true);
    k.need(ids || mod > 0);
    const float* dout = k.mat(a[1], N, d, &lddo);
    float* dt = k.mat(a[2], v1, d, &lddt);
    if (k.bad) return;
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (ids && !acc && N <= 128 && d <= 128) {   // a small token stream: every lane keeps its ids of the (up to 4) 32-token windows in registers, and each key's row is accumulated in registers
        const int nw = (N + 31) / 32, nc = (d + 31) / 32;
        int idv[4];
#pragma unroll
        for (int b = 0; b < 4; b++) idv[b] = (b < nw && b * 32 + lane < N) ? ids[b * 32 + lane] : -1;
        for (int key = v0 + w; key < v1; key += KVM_GPT_THREADS / 32) {
            float rw[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int b = 0; b < 4; b++) {
                if (b >= nw) continue;
                unsigned m = __ballot_sync(0xffffffffu, idv[b] == key);
                while (m) {
                    const int j = __ffs(m) - 1; m &= m - 1;
                    const float* src = dout + (size_t)(b * 32 + j) * lddo;
#pragma unroll
                    for (int i = 0; i < 4; i++) if (i < nc && lane + i * 32 < d) rw[i] += src[lane + i * 32];
                }   // registers only and a warp-uniform mask: no barrier, so the rows' loads overlap
            }
            float* row = dt + (size_t)key * lddt;
#pragma unroll
            for (int i = 0; i < 4; i++) if (i < nc && lane + i * 32 < d) row[lane + i * 32] = rw[i];
        }
        return;
    }
    for (int key = v0 + w; key < v1; key += KVM_GPT_THREADS / 32) {
        float* row = dt + (size_t)key * lddt;
        if (!acc) for (int c = lane; c < d; c += 32) row[c] = 0.f;
        __syncwarp();
        if (ids) {
            for (int base = 0; base < N; base += 32) {
                int r = base + lane;
                unsigned m = __ballot_sync(0xffffffffu, r < N && ids[r] == key);
                while (m) {
                    int j = __ffs(m) - 1; m &= m - 1;
                    const float* src = dout + (size_t)(base + j) * lddo;
                    for (int c = lane; c < d; c += 32) row[c] += src[c];
                    __syncwarp();
                }
            }
        } else {
            for (int r = key; r < N; r += mod) {
                const float* src = dout + (size_t)r * lddo;
                for (int c = lane; c < d; c += 32) row[c] += src[c];
            }
        }
    }
}

// ---- tiled matmul (TF32 tensor cores) ---------------------------------------------------------------------------------
KVM_GPT_DEVI float gelu_f(float x) {
    const float k0 = 0.7978845608028654f;
    return 0.5f * x * (1.f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}
KVM_GPT_DEVI float gelu_grad(float x) {
    const float k0 = 0.7978845608028654f;
    float t = tanhf(k0 * (x + 0.044715f * x * x * x));
    return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * k0 * (1.f + 3.f * 0.044715f * x * x);
}

// R x C float tile held in registers as float4s (element (row, col) of a row-major global matrix), moved to shared memory.
// Out-of-range elements read as 0. `vec` = 16-byte loads are legal (pointer and ld aligned).
template <int R, int C>
struct TileReg {
    static constexpr int NV = (R * C / 4 + KVM_GPT_THREADS - 1) / KVM_GPT_THREADS;
    float4 v[NV];
    KVM_GPT_DEVI void load(const float* src, int ld, int r0, int c0, int rmax, int cmax, bool vec) {
#pragma unroll
        for (int s = 0; s < NV; s++) {
            int q = threadIdx.x + s * KVM_GPT_THREADS;
            float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < R * C / 4) {
                int row = q / (C / 4), gc = c0 + (q % (C / 4)) * 4, gr = r0 + row;
                if (gr < rmax) {
                    const float* p = src + (size_t)gr * ld + gc;
                    if (vec && gc + 4 <= cmax) t = *reinterpret_cast<const float4*>(p);
                    else {
                        if (gc < cmax) t.x = p[0];
                        if (gc + 1 < cmax) t.y = p[1];
                        if (gc + 2 < cmax) t.z = p[2];
                        if (gc + 3 < cmax) t.w = p[3];
                    }
                }
            }
            v[s] = t;
        }
    }
    KVM_GPT_DEVI void store(float* dst, int sld) const {
#pragma unroll
        for (int s = 0; s < NV; s++) {
            int q = threadIdx.x + s * KVM_GPT_THREADS;
            if (q < R * C / 4) *reinterpret_cast<float4*>(dst + (q / (C / 4)) * sld + (q % (C / 4)) * 4) = v[s];
        }
    }
};

// ---- cp.async / ldmatrix / mma.sync primitives -------------------------------------------------------------------------
KVM_GPT_DEVI unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
KVM_GPT_DEVI void cp_async16(float* dst, const float* src, int src_bytes = 16) {   // src_bytes < 16 zero-fills the rest (0 reads nothing)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}
KVM_GPT_DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> KVM_GPT_DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// fp32 -> tf32 with round-to-nearest (ties away) in two integer ops (the same value as cvt.rna.tf32.f32, which runs at 1/4 of the
// integer rate and made the tensor-core loops conversion bound). A carry out of the mantissa bumps the exponent, as it should.
KVM_GPT_DEVI unsigned f2tf32(float x) {
    const unsigned u = __float_as_uint(x);
    return (((u & 0x7fffffffu) > 0x7f800000u) ? u : u + 0x1000u) & 0xffffe000u;   // NaN keeps its bits (the carry would turn 0x7fffffff into -0)
}
// Operand for a single (1x) TF32 mma: the tensor core ignores the low 13 bits, so the mask is free and adding the rounding bias is
// round-to-nearest in ONE op. (Round-toward-zero would save that op for ~8% speed but triples the error: measured 7.6e-4 vs 2.7e-4.)
// (A 2-op max(u + 0x1000, u) was tried and is WRONG: the canonical NaN 0x7fffffff carries into the sign bit and the unsigned max keeps the sum.)
// (CUTLASS's form, a float compare |x| < inf and a predicated IMAD.IADD, measured 10-30% slower in the rounded big tile than this bit test.)
KVM_GPT_DEVI unsigned f2tf32_op(float x) { const unsigned u = __float_as_uint(x); return ((u & 0x7fffffffu) > 0x7f800000u) ? u : u + 0x1000u; }
KVM_GPT_DEVI void ldmatrix_x4(unsigned (&r)[4], const float* p) {   // four 8x8 b16 matrices == four 8x4 tf32 matrices
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
KVM_GPT_DEVI void mma_tf32(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

KVM_GPT_DEVI unsigned pack_bf16(float lo, float hi) {   // two fp32 -> bf16x2 (round to nearest even), `lo` in the low half
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
KVM_GPT_DEVI void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {   // m16n8k16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Shared-memory tiles of the matmul are 2048 floats, unpadded, with an XOR swizzle that keeps 16-byte chunks contiguous (so
// cp.async and ldmatrix work) and makes every fragment access bank-conflict free:
//  KC: rows x BK floats (BK = 32 or 16), k contiguous (A as [m][k], B as [n][k]);  the 16-byte chunk index is XORed with a row function.
//  MC: BK rows x (64 or 128) floats, m or n contiguous (A as [k][m], B as [k][n]);  column bits 3..4 are XORed with row&3.
template <int BK>   // BK = 32: chunk ^= row&7;  BK = 16: chunk ^= (row>>1)&3  (eight consecutive rows hit eight distinct 16-byte bank groups)
KVM_GPT_DEVI int sw_kc(int r, int c) { return r * BK + ((((c >> 2) ^ (BK == 32 ? r : (r >> 1))) & (BK / 4 - 1)) << 2) + (c & 3); }
template <int W>
KVM_GPT_DEVI int sw_mc(int r, int c) { return r * W + (c ^ ((r & 3) << 3)); }

// Copies one ROWS x COLS tile (KC: k along the columns; MC: m/n along the columns) whose origin is (gr0, gc0) in the row-major global matrix [rmax x cmax] into shared
// memory (ROWS*COLS/4 chunks of 4 floats, spread over the 256 threads). Aligned in-bounds chunks use cp.async (out-of-range rows zero-fill in hardware);
// ragged chunks (cmax or ld not a multiple of 4, unaligned base) are written with plain stores, published by the same barrier.
template <bool KC, int ROWS, int COLS>
KVM_GPT_DEVI void tile_async(float* dst, const float* src, int ld, int gr0, int gc0, int rmax, int cmax, bool vec) {
    constexpr int CPR = COLS / 4;   // chunks per row
#pragma unroll
    for (int s = 0; s < ROWS * CPR / KVM_GPT_THREADS; s++) {
        int q = threadIdx.x + s * KVM_GPT_THREADS, row = q / CPR, c4 = (q % CPR) * 4, gr = gr0 + row, gc = gc0 + c4;
        float* d = dst + (KC ? sw_kc<COLS>(row, c4) : sw_mc<COLS>(row, c4));
        if (vec && gc + 4 <= cmax) cp_async16(d, gr < rmax ? src + (size_t)gr * ld + gc : src, gr < rmax ? 16 : 0);
        else {
            const float* p = src + (size_t)gr * ld + gc;
            bool rok = gr < rmax;
            d[0] = rok && gc < cmax ? p[0] : 0.f;
            d[1] = rok && gc + 1 < cmax ? p[1] : 0.f;
            d[2] = rok && gc + 2 < cmax ? p[2] : 0.f;
            d[3] = rok && gc + 3 < cmax ? p[3] : 0.f;
        }
    }
}

// Rounds, in place, the chunks of a tile that THIS thread copied with tile_async (same chunk-to-thread mapping) to tf32 operands. Each
// element is converted once per CTA instead of once per fragment use (a warp tile re-reads A across WN warps and B across WM warps, and
// the conversion ALU ops, not the tensor pipe, bound the 128x128 tile). The thread reads only its own cp.async/plain writes, which
// cp_wait makes visible to it; other warps see the result after the next __syncthreads. Same rounding as f2tf32_op, so results are
// bitwise those of converting at fragment-load time.
KVM_GPT_DEVI unsigned tf32_rnd(unsigned u) { return ((u << 1) > 0xff000000u) ? u : u + 0x1000u; }
template <bool KC, int ROWS, int COLS>
KVM_GPT_DEVI void tile_cvt(float* dst) {
    constexpr int CPR = COLS / 4;
#pragma unroll
    for (int s = 0; s < ROWS * CPR / KVM_GPT_THREADS; s++) {
        const int q = threadIdx.x + s * KVM_GPT_THREADS, row = q / CPR, c4 = (q % CPR) * 4;
        uint4* d = reinterpret_cast<uint4*>(dst + (KC ? sw_kc<COLS>(row, c4) : sw_mc<COLS>(row, c4)));
        uint4 v = *d;
        v.x = tf32_rnd(v.x); v.y = tf32_rnd(v.y); v.z = tf32_rnd(v.z); v.w = tf32_rnd(v.w);
        *d = v;
    }
}

// Fast per-thread stream of one operand tile (ROWS x COLS, KC or MC layout as in tile_async) for the common case where every chunk is
// in range and 16-byte aligned (checked once per task by the caller): the global pointer, row stride and shared offset of the thread's
// first chunk are computed once and each stage is `NC` cp.async with a pointer bump, instead of re-deriving row, column, bounds and swizzle per
// chunk per k step. Chunk s of a thread is RSTEP rows below chunk 0; RSTEP is a multiple of 8, so it leaves the swizzle (row&7, (row>>1)&3,
// row&3) unchanged and the shared offset is chunk 0's plus s*RSTEP*COLS.
template <bool KC, int ROWS, int COLS>
struct TileStream {
    static constexpr int CPR = COLS / 4, NC = ROWS * CPR / KVM_GPT_THREADS, RSTEP = KVM_GPT_THREADS / CPR, BKV = KC ? COLS : ROWS;
    static_assert(ROWS * CPR % KVM_GPT_THREADS == 0 && RSTEP % (KC ? 8 : 4) == 0, "tile must give whole, swizzle-neutral chunk strides");
    const float* p; const float* src0; size_t rstride, adv; int s0, row0, c40; int bytes[NC];
    // limit: size of the dimension that is not K (M for A, N for B): rows of a KC tile, or columns of an MC tile, beyond it zero-fill in hardware (edge tiles keep this path).
    KVM_GPT_DEVI void init(const float* src, int ld, int gr0, int gc0, int kstep, int limit) {   // kstep: k elements per stage
        row0 = threadIdx.x / CPR; c40 = (threadIdx.x % CPR) * 4;
        src0 = src;
        p = src + (size_t)(gr0 + row0) * ld + gc0 + c40;
        rstride = (size_t)RSTEP * ld; adv = KC ? (size_t)kstep : (size_t)kstep * ld;
        s0 = KC ? sw_kc<COLS>(row0, c40) : sw_mc<COLS>(row0, c40);
#pragma unroll
        for (int s = 0; s < NC; s++) bytes[s] = (KC ? gr0 + row0 + s * RSTEP < limit : gc0 + c40 + 4 <= limit) ? 16 : 0;
    }
    KVM_GPT_DEVI void issue(float* dst, int kv) {   // kv: valid k elements of this stage (BKV, except the last, partial slab of K)
        if (kv >= BKV) {
#pragma unroll
            for (int s = 0; s < NC; s++) cp_async16(dst + s0 + s * RSTEP * COLS, bytes[s] ? p + s * rstride : src0, bytes[s]);
        } else {
#pragma unroll
            for (int s = 0; s < NC; s++) {
                const int by = (KC ? c40 + 4 <= kv : row0 + s * RSTEP < kv) ? bytes[s] : 0;
                cp_async16(dst + s0 + s * RSTEP * COLS, by ? p + s * rstride : src0, by);
            }
        }
        p += adv;
    }
    KVM_GPT_DEVI void issue_part(float* dst, int part, int nparts) {   // one share of a stage's copies (the caller has checked that every chunk is whole and in range); the last share advances the stream
#pragma unroll
        for (int s = part * NC / nparts; s < (part + 1) * NC / nparts; s++) cp_async16(dst + s0 + s * RSTEP * COLS, p + s * rstride, 16);
        if (part + 1 == nparts) p += adv;
    }
    KVM_GPT_DEVI void cvt(float* dst) const {   // rounds this thread's chunks to tf32 in place (see tile_cvt)
#pragma unroll
        for (int s = 0; s < NC; s++) {
            uint4* d = reinterpret_cast<uint4*>(dst + s0 + s * RSTEP * COLS);
            uint4 v = *d;
            v.x = tf32_rnd(v.x); v.y = tf32_rnd(v.y); v.z = tf32_rnd(v.z); v.w = tf32_rnd(v.w);
            *d = v;
        }
    }
};

// bf16-storage variant of tile_async: the global source holds raw bf16 (uint16) elements, widened to fp32 on the way into shared memory
// (plain loads and stores; exact, a bf16 value is the top half of an fp32). Same tile geometry and swizzle as tile_async.
template <bool KC, int ROWS, int COLS>
KVM_GPT_DEVI void tile_bf(float* dst, const uint16_t* src, int ld, int gr0, int gc0, int rmax, int cmax) {
    constexpr int CPR = COLS / 4;
#pragma unroll
    for (int s = 0; s < ROWS * CPR / KVM_GPT_THREADS; s++) {
        int q = threadIdx.x + s * KVM_GPT_THREADS, row = q / CPR, c4 = (q % CPR) * 4, gr = gr0 + row, gc = gc0 + c4;
        float* d = dst + (KC ? sw_kc<COLS>(row, c4) : sw_mc<COLS>(row, c4));
        const uint16_t* p = src + (size_t)gr * ld + gc;
        const bool rok = gr < rmax;
#pragma unroll
        for (int i = 0; i < 4; i++) d[i] = rok && gc + i < cmax ? __uint_as_float((uint32_t)p[i] << 16) : 0.f;
    }
}

// One K slab of a TF32 tile (mma m16n8k8): this warp's NI m16 x NJ n8 fragments at rows mbase.. and columns nbase.., KSTEPS k8 steps from kb. The slab's operand tiles are in shared memory at As and Bs, in the layouts tile_async writes:
// K-contiguous (A as [m][k], B as [n][k]; sw_kc<BK>, fed by ldmatrix) or not (A as [k][m], B as [k][n]; sw_mc<ATW / BTW>, 32-bit loads). It reads only shared memory and writes only the accumulators, so a tile whose operands are already
// in shared memory can run through it without the stage loader. PREC: 3xTF32 (a_lo*b_hi + a_hi*b_lo + a_hi*b_hi); RZ: the operands stay as they are for the tensor core's truncation.
template <bool TA, bool TB, bool PREC, bool RZ, int BK, int ATW, int BTW, int NI, int NJ>
KVM_GPT_DEVI void slab_load(const float* As, const float* Bs, unsigned (&ah)[NI][4], unsigned (&al)[NI][4], unsigned (&bh)[NJ][2], unsigned (&bl)[NJ][2], int mbase, int nbase, int kk) {   // one k8 step's fragments into registers
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int mb = mbase + i * 16;
        if (!TA) ldmatrix_x4(ah[i], As + sw_kc<BK>(mb + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
        else {   // A stored [k][m]: a0=(m g, k t) a1=(m g+8, k t) a2=(m g, k t+4) a3=(m g+8, k t+4)
            ah[i][0] = __float_as_uint(As[sw_mc<ATW>(kk + t, mb + g)]);     ah[i][1] = __float_as_uint(As[sw_mc<ATW>(kk + t, mb + g + 8)]);
            ah[i][2] = __float_as_uint(As[sw_mc<ATW>(kk + t + 4, mb + g)]); ah[i][3] = __float_as_uint(As[sw_mc<ATW>(kk + t + 4, mb + g + 8)]);
        }
#pragma unroll
        if constexpr (PREC || !KVM_GPT_SMEM_CVT) for (int e = 0; e < 4; e++) {
            float x = __uint_as_float(ah[i][e]);
            if (PREC) { unsigned h = f2tf32(x); al[i][e] = __float_as_uint(x - __uint_as_float(h)); ah[i][e] = h; }
            else if constexpr (!RZ) ah[i][e] = f2tf32_op(x);
        }
    }
#pragma unroll
    for (int jp = 0; jp < NJ / 2; jp++) {
        const int nb = nbase + jp * 16;
        if (TB) {
            unsigned r[4];
            ldmatrix_x4(r, Bs + sw_kc<BK>(nb + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
            bh[2 * jp][0] = r[0]; bh[2 * jp][1] = r[1]; bh[2 * jp + 1][0] = r[2]; bh[2 * jp + 1][1] = r[3];
        } else {   // B stored [k][n]: b0=(k t, n g) b1=(k t+4, n g)
            bh[2 * jp][0] = __float_as_uint(Bs[sw_mc<BTW>(kk + t, nb + g)]);          bh[2 * jp][1] = __float_as_uint(Bs[sw_mc<BTW>(kk + t + 4, nb + g)]);
            bh[2 * jp + 1][0] = __float_as_uint(Bs[sw_mc<BTW>(kk + t, nb + 8 + g)]);  bh[2 * jp + 1][1] = __float_as_uint(Bs[sw_mc<BTW>(kk + t + 4, nb + 8 + g)]);
        }
#pragma unroll
        if constexpr (PREC || !KVM_GPT_SMEM_CVT) for (int jj = 0; jj < 2; jj++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
            float x = __uint_as_float(bh[2 * jp + jj][e]);
            if (PREC) { unsigned h = f2tf32(x); bl[2 * jp + jj][e] = __float_as_uint(x - __uint_as_float(h)); bh[2 * jp + jj][e] = h; }
            else if constexpr (!RZ) bh[2 * jp + jj][e] = f2tf32_op(x);
        }
    }
}
template <bool PREC, int NI, int NJ>
KVM_GPT_DEVI void slab_mma(float (*acc)[NJ][4], const unsigned (&ah)[NI][4], const unsigned (&al)[NI][4], const unsigned (&bh)[NJ][2], const unsigned (&bl)[NJ][2]) {   // the MMAs of one k8 step
#pragma unroll
    for (int i = 0; i < NI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++) {
            if (PREC) { mma_tf32(acc[i][j], al[i], bh[j][0], bh[j][1]); mma_tf32(acc[i][j], ah[i], bl[j][0], bl[j][1]); }
            mma_tf32(acc[i][j], ah[i], bh[j][0], bh[j][1]);
        }
}
// One slab of k (KSTEPS k8 steps) for the 64x64 tile and the big tile's plain loop, loads and MMAs in one body (slab_load and slab_mma split the
// same work for the pipelined big tile). Inside the executor's union kernel the 64x64 tile's time per task moves 12-30% between builds that do
// not touch it (its k-loop's spill reloads follow the whole kernel's register allocation), so compare its speed only within one build.
template <bool TA, bool TB, bool PREC, bool RZ, int BK, int ATW, int BTW, int NI, int NJ, int KSTEPS>
KVM_GPT_DEVI void tile_slab(const float* As, const float* Bs, float (*acc)[NJ][4], int mbase, int nbase, int kb) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int kq = 0; kq < KSTEPS; kq++) {
        const int kk = kb + kq * 8;
        unsigned ah[NI][4], al[NI][4], bh[NJ][2], bl[NJ][2];
#pragma unroll
        for (int i = 0; i < NI; i++) {
            const int mb = mbase + i * 16;
            if (!TA) ldmatrix_x4(ah[i], As + sw_kc<BK>(mb + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
            else {   // A stored [k][m]: a0=(m g, k t) a1=(m g+8, k t) a2=(m g, k t+4) a3=(m g+8, k t+4)
                ah[i][0] = __float_as_uint(As[sw_mc<ATW>(kk + t, mb + g)]);     ah[i][1] = __float_as_uint(As[sw_mc<ATW>(kk + t, mb + g + 8)]);
                ah[i][2] = __float_as_uint(As[sw_mc<ATW>(kk + t + 4, mb + g)]); ah[i][3] = __float_as_uint(As[sw_mc<ATW>(kk + t + 4, mb + g + 8)]);
            }
#pragma unroll
            if constexpr (PREC || !KVM_GPT_SMEM_CVT) for (int e = 0; e < 4; e++) {
                float x = __uint_as_float(ah[i][e]);
                if (PREC) { unsigned h = f2tf32(x); al[i][e] = __float_as_uint(x - __uint_as_float(h)); ah[i][e] = h; }
                else if constexpr (!RZ) ah[i][e] = f2tf32_op(x);
            }
        }
#pragma unroll
        for (int jp = 0; jp < NJ / 2; jp++) {
            const int nb = nbase + jp * 16;
            if (TB) {
                unsigned r[4];
                ldmatrix_x4(r, Bs + sw_kc<BK>(nb + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
                bh[2 * jp][0] = r[0]; bh[2 * jp][1] = r[1]; bh[2 * jp + 1][0] = r[2]; bh[2 * jp + 1][1] = r[3];
            } else {   // B stored [k][n]: b0=(k t, n g) b1=(k t+4, n g)
                bh[2 * jp][0] = __float_as_uint(Bs[sw_mc<BTW>(kk + t, nb + g)]);          bh[2 * jp][1] = __float_as_uint(Bs[sw_mc<BTW>(kk + t + 4, nb + g)]);
                bh[2 * jp + 1][0] = __float_as_uint(Bs[sw_mc<BTW>(kk + t, nb + 8 + g)]);  bh[2 * jp + 1][1] = __float_as_uint(Bs[sw_mc<BTW>(kk + t + 4, nb + 8 + g)]);
            }
#pragma unroll
            if constexpr (PREC || !KVM_GPT_SMEM_CVT) for (int jj = 0; jj < 2; jj++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                float x = __uint_as_float(bh[2 * jp + jj][e]);
                if (PREC) { unsigned h = f2tf32(x); bl[2 * jp + jj][e] = __float_as_uint(x - __uint_as_float(h)); bh[2 * jp + jj][e] = h; }
                else if constexpr (!RZ) bh[2 * jp + jj][e] = f2tf32_op(x);
            }
        }
#pragma unroll
        for (int i = 0; i < NI; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                if (PREC) { mma_tf32(acc[i][j], al[i], bh[j][0], bh[j][1]); mma_tf32(acc[i][j], ah[i], bl[j][0], bl[j][1]); }
                mma_tf32(acc[i][j], ah[i], bh[j][0], bh[j][1]);
            }
    }
}

enum { ACT_NONE = 0, ACT_GELU = 1, ACT_DGELU = 2 };   // epilogue activation of a GEMM tile: gelu(v) after pre[m,n] = v; or (flag 512, backward of a GELU layer) v *= gelu_grad(pre[m,n]) with pre an input

// Epilogue of an output tile whose accumulated partials sit in shared memory: NPAR partial tiles of TM rows (row stride CLD floats, tile p at sm + p*TM*CLD), element = alpha * (sum of the partials), +bias[n],
// pre[m,n] = v (before gelu), gelu (or the dgelu form above), +res[m,n], (+out[m,n]), store. Four consecutive columns per thread (each warp writes whole rows, 512-byte strokes of 16-byte accesses when every operand allows them).
template <int TM, int TN, int NPAR, int CLD>
KVM_GPT_DEVI void tile_wb(const float* sm, int m0, int n0, int M, int N, float* out, int ldo, const float* bias, const float* res, int ldr, float* pre, int ldp, float alpha, int act, bool accum) {
    const bool vec = ((n0 & 3) == 0) && aligned16(out, ldo) && (!bias || aligned16(bias, 0)) && (!res || aligned16(res, ldr)) && (!pre || aligned16(pre, ldp));
    for (int q = threadIdx.x; q < TM * TN / 4; q += KVM_GPT_THREADS) {
        const int r = q / (TN / 4), c = (q % (TN / 4)) * 4, gm = m0 + r, gn = n0 + c;
        if (gm >= M || gn >= N) continue;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int p = 0; p < NPAR; p++) { const float4 t = *reinterpret_cast<const float4*>(sm + p * TM * CLD + r * CLD + c); v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w; }
        v[0] *= alpha; v[1] *= alpha; v[2] *= alpha; v[3] *= alpha;
        if (vec && gn + 4 <= N) {
            if (bias) { const float4 b4 = *reinterpret_cast<const float4*>(bias + gn); v[0] += b4.x; v[1] += b4.y; v[2] += b4.z; v[3] += b4.w; }
            if (act == ACT_DGELU) { const float4 p4 = *reinterpret_cast<const float4*>(pre + (size_t)gm * ldp + gn); v[0] *= gelu_grad(p4.x); v[1] *= gelu_grad(p4.y); v[2] *= gelu_grad(p4.z); v[3] *= gelu_grad(p4.w); }
            else {
                if (pre) *reinterpret_cast<float4*>(pre + (size_t)gm * ldp + gn) = make_float4(v[0], v[1], v[2], v[3]);
                if (act == ACT_GELU) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); v[2] = gelu_f(v[2]); v[3] = gelu_f(v[3]); }
            }
            if (res) { const float4 r4 = *reinterpret_cast<const float4*>(res + (size_t)gm * ldr + gn); v[0] += r4.x; v[1] += r4.y; v[2] += r4.z; v[3] += r4.w; }
            float4* o = reinterpret_cast<float4*>(out + (size_t)gm * ldo + gn);
            if (accum) { const float4 o4 = *o; v[0] += o4.x; v[1] += o4.y; v[2] += o4.z; v[3] += o4.w; }
            *o = make_float4(v[0], v[1], v[2], v[3]);
        } else {
            for (int i = 0; i < 4 && gn + i < N; i++) {
                float x = v[i];
                if (bias) x += bias[gn + i];
                if (act == ACT_DGELU) x *= gelu_grad(pre[(size_t)gm * ldp + gn + i]);
                else {
                    if (pre) pre[(size_t)gm * ldp + gn + i] = x;
                    if (act == ACT_GELU) x = gelu_f(x);
                }
                if (res) x += res[(size_t)gm * ldr + gn + i];
                float* o = out + (size_t)gm * ldo + gn + i;
                *o = accum ? *o + x : x;
            }
        }
    }
}

// 64x64 output tile, BK=32, 3-stage cp.async pipeline (3 x 16 KiB = 48 KiB, the epilogue reuses it), 8 warps, mma.sync m16n8k8
// TF32. Warp (wm, wn, kh): 32x32 sub-tile at (wm*32, wn*32) = 2 m16 x 4 n8 fragments, reducing k8-steps {2kh, 2kh+1} of each BK=32
// slab; the two k-halves are summed in the epilogue. Shared tiles keep the global orientation (coalesced loads for all four
// transpose combinations). K-contiguous operands (A as [m][k], B as [n][k]) are fed by ldmatrix, the others by 32-bit loads.
// PREC: 3xTF32 split (a_lo*b_hi + a_hi*b_lo + a_hi*b_hi) restores ~fp32 accuracy.
template <bool TA, bool TB, bool PREC, bool BF = false, bool RZ = false>   // BF: operands rounded to bf16, m16n8k16 bf16 mma (one k16 step per slab per warp half); RZ: big tiles only
KVM_GPT_DEVI void mm_tile(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                          int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, int act, bool accum,
                          int sd = 0) {   // sd (BF only): storage-dtype bits, 1 A, 2 B, 4 out, 8 res, 16 pre are raw bf16 instead of fp32
    constexpr int BK = 32, TM = 64, TN = 64, TILE = 2048, STAGE = 2 * TILE, NST = KVM_GPT_TILE64_NST, CLD = TN + 8;
    float* sm = reinterpret_cast<float*>(kvm_gpt_smem);   // [stage][A | B]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wmi = (warp >> 1) & 1, wni = warp & 1, kh = warp >> 2;
    const int g = lane >> 2, t = lane & 3;
    const bool va = aligned16(A, lda) && (!TA || (m0 & 3) == 0), vb = aligned16(B, ldb) && (TB || (n0 & 3) == 0);   // tile origin column too
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    const int nk = (K + BK - 1) / BK;
    using SA = TileStream<!TA, TA ? BK : TM, TA ? TM : BK>; using SB = TileStream<TB, TB ? TN : BK, TB ? BK : TN>;
    SA sa; SB sb;
    const bool stream = K <= KVM_GPT_TILE64_STREAM_KMAX;
    const bool fa = stream && va && !(BF && (sd & 1)) && (TA ? (M & 3) == 0 : (K & 3) == 0), fb = stream && vb && !(BF && (sd & 2)) && (TB ? (K & 3) == 0 : (N & 3) == 0);   // the stream path (edge tiles included: outside chunks zero-fill)
    if (fa) { if (TA) sa.init(A, lda, 0, m0, BK, M); else sa.init(A, lda, m0, 0, BK, M); }
    if (fb) { if (TB) sb.init(B, ldb, n0, 0, BK, N); else sb.init(B, ldb, 0, n0, BK, N); }
    auto issue = [&](int it) {
        float* As = sm + (it % NST) * STAGE; float* Bs = As + TILE;
        const int kk = it * BK, kv = min(BK, K - kk);
        if (fa) sa.issue(As, kv);
        else if (BF && (sd & 1)) { const uint16_t* Ah = (const uint16_t*)A;
            if (TA) tile_bf<false, 32, 64>(As, Ah, lda, kk, m0, K, M); else tile_bf<true, 64, 32>(As, Ah, lda, m0, kk, M, K); }
        else if (TA) tile_async<false, 32, 64>(As, A, lda, kk, m0, K, M, va); else tile_async<true, 64, 32>(As, A, lda, m0, kk, M, K, va);
        if (fb) sb.issue(Bs, kv);
        else if (BF && (sd & 2)) { const uint16_t* Bh = (const uint16_t*)B;
            if (TB) tile_bf<true, 64, 32>(Bs, Bh, ldb, n0, kk, N, K); else tile_bf<false, 32, 64>(Bs, Bh, ldb, kk, n0, K, N); }
        else if (TB) tile_async<true, 64, 32>(Bs, B, ldb, n0, kk, N, K, vb); else tile_async<false, 32, 64>(Bs, B, ldb, kk, n0, K, N, vb);
    };
#pragma unroll
    for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }   // always commit, empty groups included, so wait_group counts stay uniform
    for (int it = 0; it < nk; it++) {
        cp_wait<NST - 2>();
        if constexpr (!PREC && !BF && !RZ && KVM_GPT_SMEM_CVT) {   // round the operands once per CTA, in shared memory (see tile_cvt)
            float* Aw = sm + (it % NST) * STAGE; float* Bw = Aw + TILE;
            if (fa) sa.cvt(Aw); else if (TA) tile_cvt<false, 32, 64>(Aw); else tile_cvt<true, 64, 32>(Aw);
            if (fb) sb.cvt(Bw); else if (TB) tile_cvt<true, 64, 32>(Bw); else tile_cvt<false, 32, 64>(Bw);
        }
        __syncthreads();   // tile `it` landed everywhere; everyone finished the slab that used the stage refilled next
        if (it == 0) KVM_GPT_PROBE(1);
        if (it + NST - 1 < nk) issue(it + NST - 1);
        cp_commit();
        const float* As = sm + (it % NST) * STAGE; const float* Bs = As + TILE;
        if constexpr (BF) {
            const int kk = kh * 16;
            unsigned af_[2][4], bf_[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int mb = wmi * 32 + i * 16;
                if (!TA) {   // A [m][k]: (row, k..k+1) is an adjacent float2
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float2 v = *reinterpret_cast<const float2*>(As + sw_kc<32>(mb + g + 8 * (q & 1), kk + 2 * t + 8 * (q >> 1)));
                        af_[i][q] = pack_bf16(v.x, v.y);
                    }
                } else {
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const int kr = kk + 2 * t + 8 * (q >> 1), mc = mb + g + 8 * (q & 1);
                        af_[i][q] = pack_bf16(As[sw_mc<64>(kr, mc)], As[sw_mc<64>(kr + 1, mc)]);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int nc = wni * 32 + j * 8 + g;
#pragma unroll
                for (int q = 0; q < 2; q++) {
                    const int kr = kk + 2 * t + 8 * q;
                    if (TB) { const float2 v = *reinterpret_cast<const float2*>(Bs + sw_kc<32>(nc, kr)); bf_[j][q] = pack_bf16(v.x, v.y); }
                    else bf_[j][q] = pack_bf16(Bs[sw_mc<64>(kr, nc)], Bs[sw_mc<64>(kr + 1, nc)]);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af_[i], bf_[j][0], bf_[j][1]);
        } else tile_slab<TA, TB, PREC, RZ, 32, TM, TN, 2, 4, 2>(As, Bs, acc, wmi * 32, wni * 32, kh * 16);
    }
    KVM_GPT_PROBE(2);
    cp_wait<0>();
    __syncthreads();   // all warps done reading the last stage; reuse shared memory for the two k-half partial tiles
    float* Cs = sm + kh * TM * CLD;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {   // c0,c1 = (row g, cols 2t,2t+1); c2,c3 = (row g+8, ...)
        const int r = wmi * 32 + i * 16 + g, c = wni * 32 + j * 8 + 2 * t;
        *reinterpret_cast<float2*>(Cs + r * CLD + c) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(Cs + (r + 8) * CLD + c) = make_float2(acc[i][j][2], acc[i][j][3]);
    }
    __syncthreads();
    if (BF && (sd & 28)) {   // bf16-stored out / res / pre: scalar epilogue, math in fp32, one rounding per stored value
        const bool ob = sd & 4, rb = sd & 8, pb = sd & 16;
        for (int q = threadIdx.x; q < TM * TN; q += KVM_GPT_THREADS) {
            const int r = q / TN, c = q % TN, gm = m0 + r, gn = n0 + c;
            if (gm >= M || gn >= N) continue;
            float x = (sm[r * CLD + c] + sm[TM * CLD + r * CLD + c]) * alpha;
            if (bias) x += bias[gn];
            if (pre) st_s(pre, (size_t)gm * ldp + gn, x, pb);
            if (act == ACT_GELU) x = gelu_f(x);
            if (res) x += ld_s(res, (size_t)gm * ldr + gn, rb);
            const size_t oi = (size_t)gm * ldo + gn;
            if (accum) x += ld_s(out, oi, ob);
            st_s(out, oi, x, ob);
        }
        return;
    }
    tile_wb<TM, TN, 2, CLD>(sm, m0, n0, M, N, out, ldo, bias, res, ldr, pre, ldp, alpha, act, accum);
}

// 128x128 output tile (flag 32): the same math as mm_tile with a quarter of the shared-to-L2 traffic per FLOP, which is what bounds
// the 64x64 tile. BK=16, 3-stage cp.async pipeline (3 x 16 KiB = 48 KiB), 8 warps as 2 (m) x 4 (n), each owning a 64x32 sub-tile
// = 4 m16 x 4 n8 fragments over the whole K (no split-K); the epilogue runs straight from the accumulator registers.
template <bool TA, bool TB, bool PREC, int WM, int WN, int WTN, bool BF = false, bool RZ = false>   // RZ: operands truncated by the tensor core, not rounded to tf32 (matmul_tile flag 256). BF: bf16 operands (see below). WM x WN warps, each owning a 64 x WTN sub-tile; tile = 64*WM x WTN*WN
KVM_GPT_DEVI void mm_tile_big(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                              int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, int act, bool accum) {
    constexpr int BK = KVM_GPT_BIG_BK, NST = KVM_GPT_BIG_NST, BM = 64 * WM, BN = WTN * WN, NJ = WTN / 8, ATILE = BM * BK, TILE = ATILE, STAGE = (BM + BN) * BK;
    static_assert(WM * WN == 8, "8 warps");
    float* sm = reinterpret_cast<float*>(kvm_gpt_smem);   // [stage][A | B]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wmi = warp / WN, wni = warp % WN, g = lane >> 2, t = lane & 3;
    const bool va = aligned16(A, lda) && (!TA || (m0 & 3) == 0), vb = aligned16(B, ldb) && (TB || (n0 & 3) == 0);   // tile origin column too
    float acc[4][NJ][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    const int nk = (K + BK - 1) / BK;
    using SA = TileStream<!TA, TA ? BK : BM, TA ? BM : BK>; using SB = TileStream<TB, TB ? BN : BK, TB ? BK : BN>;
    SA sa; SB sb;
    const bool stream = K <= KVM_GPT_BIG_STREAM_KMAX;
    const bool fa = stream && va && (TA ? (M & 3) == 0 : (K & 3) == 0), fb = stream && vb && (TB ? (K & 3) == 0 : (N & 3) == 0);   // whole chunks in range along k and along the operand's other dimension: TileStream path (edge tiles included: their outside chunks zero-fill)
    if (fa) { if (TA) sa.init(A, lda, 0, m0, BK, M); else sa.init(A, lda, m0, 0, BK, M); }
    if (fb) { if (TB) sb.init(B, ldb, n0, 0, BK, N); else sb.init(B, ldb, 0, n0, BK, N); }
    auto issue = [&](int it) {
        float* As = sm + (it % NST) * STAGE; float* Bs = As + ATILE;
        const int kk = it * BK, kv = min(BK, K - kk);
        if (fa) sa.issue(As, kv);
        else if (TA) tile_async<false, BK, BM>(As, A, lda, kk, m0, K, M, va); else tile_async<true, BM, BK>(As, A, lda, m0, kk, M, K, va);
        if (fb) sb.issue(Bs, kv);
        else if (TB) tile_async<true, BN, BK>(Bs, B, ldb, n0, kk, N, K, vb); else tile_async<false, BK, BN>(Bs, B, ldb, kk, n0, K, N, vb);
    };
    if constexpr (KVM_GPT_BIG_PIPE2 && !PREC && !BF && (RZ || KVM_GPT_BIG_PIPE_RND)) {   // the k-step pipeline: a step's fragments are in registers before its MMAs start and the next step's load overlaps them; the next stage's copies are issued after the first step's MMAs; the wait and barrier for the next stage come before the last step
        constexpr int K8 = BK / 8;
        unsigned rA[2][4][4], rAl[2][4][4], rB[2][NJ][2], rBl[2][NJ][2];
        auto load_step = [&](int st, int kq, int buf) {
            const float* As = sm + (st % NST) * STAGE; const float* Bs = As + ATILE;
            slab_load<TA, TB, false, RZ, BK, BM, BN, 4, NJ>(As, Bs, rA[buf], rAl[buf], rB[buf], rBl[buf], wmi * 64, wni * WTN, kq * 8);
        };
        auto round_stage = [&](int st) {   // rounded tiles: this thread's chunks of stage st, which cp_wait made visible to it, rounded in place; the barrier after it publishes them
            if constexpr (!RZ) {
                float* Aw = sm + (st % NST) * STAGE; float* Bw = Aw + ATILE;
                if (fa) sa.cvt(Aw); else if (TA) tile_cvt<false, BK, BM>(Aw); else tile_cvt<true, BM, BK>(Aw);
                if (fb) sb.cvt(Bw); else if (TB) tile_cvt<true, BN, BK>(Bw); else tile_cvt<false, BK, BN>(Bw);
            }
        };
#pragma unroll
        for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }
        cp_wait<NST - 2>();
        round_stage(0);
        __syncthreads();
        KVM_GPT_PROBE(1);
        load_step(0, 0, 0);
        // Whole, in-range chunks along both operands (the common tile: checked once per task) take the loop with the next stage's copies spread over the k-steps, each step issuing its share between its MMAs: issued in one burst after the first step, the 8 warps' 12 copies each leave the tensor units idle (measured: 0.77 → 0.94x of cuBLAS at 4096x3072x768). Spread copies need NST >= 3, so that the group committed before the wait is not the one the wait must drain.
        const bool spread = NST >= 3 && fa && fb && (K % BK) == 0 && m0 + BM <= M && n0 + BN <= N;
        auto k_loop = [&](auto spread_tag) {
            for (int it = 0; it < nk; it++) {
#pragma unroll
                for (int kq = 0; kq < K8; kq++) {
                    if (kq + 1 < K8) load_step(it, kq + 1, (kq + 1) % 2);
                    else if (it + 1 < nk) load_step(it + 1, 0, (kq + 1) % 2);
                    slab_mma<false, 4, NJ>(acc, rA[kq % 2], rAl[kq % 2], rB[kq % 2], rBl[kq % 2]);
                    if constexpr (decltype(spread_tag)::value) {
                        if (kq + 1 < K8 && it + NST - 1 < nk) {
                            float* As = sm + ((it + NST - 1) % NST) * STAGE;
                            sa.issue_part(As, kq, K8 - 1); sb.issue_part(As + ATILE, kq, K8 - 1);
                        }
                        if (kq + 2 == K8) { cp_commit(); cp_wait<NST - 2>(); if (it + 1 < nk) round_stage(it + 1); __syncthreads(); }
                    } else {
                        if (kq == 0) { if (it + NST - 1 < nk) issue(it + NST - 1); cp_commit(); }
                        if (kq + 2 == K8) { cp_wait<NST - 2>(); if (it + 1 < nk) round_stage(it + 1); __syncthreads(); }
                    }
                }
            }
        };
        if (spread) k_loop(std::true_type{}); else k_loop(std::false_type{});
    } else {
#pragma unroll
    for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }
    for (int it = 0; it < nk; it++) {
        cp_wait<NST - 2>();
        const float* As = sm + (it % NST) * STAGE; const float* Bs = As + ATILE;
        if constexpr (!PREC && !BF && KVM_GPT_SMEM_CVT) {   // round the operands once per CTA, in shared memory (see tile_cvt)
            float* Aw = sm + (it % NST) * STAGE; float* Bw = Aw + ATILE;
            if constexpr (!RZ) { if (fa) sa.cvt(Aw); else if (TA) tile_cvt<false, BK, BM>(Aw); else tile_cvt<true, BM, BK>(Aw); }
            if constexpr (!RZ) { if (fb) sb.cvt(Bw); else if (TB) tile_cvt<true, BN, BK>(Bw); else tile_cvt<false, BK, BN>(Bw); }
        }
        __syncthreads();
        if (it == 0) KVM_GPT_PROBE(1);
        if (it + NST - 1 < nk) issue(it + NST - 1);
        cp_commit();
        if constexpr (BF) {
            // bf16 operands, m16n8k16: fragments are read from the fp32 tiles exactly like the tf32 path (ldmatrix / scalar loads), then
            // packed to bf16x2 (round to nearest even). One k16 step = two k8 halves; mma pair (k, k+4) of a half becomes one bf16x2, i.e.
            // the k index of every k16 step is permuted the same way in A and B, which a sum over k does not care about. The tensor work
            // per k is half the tf32 path's and the shared-memory traffic is the same, so it runs at up to twice the tf32 rate.
            static_assert(BK % 16 == 0, "bf16 steps are k16");
#pragma unroll
            for (int kb = 0; kb < BK / 16; kb++) {
                unsigned af[4][4], bfr[NJ][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int kk = kb * 16 + h * 8;
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const int mb = wmi * 64 + i * 16;
                        unsigned r[4];
                        if (!TA) ldmatrix_x4(r, As + sw_kc<BK>(mb + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
                        else {
                            r[0] = __float_as_uint(As[sw_mc<BM>(kk + t, mb + g)]);     r[1] = __float_as_uint(As[sw_mc<BM>(kk + t, mb + g + 8)]);
                            r[2] = __float_as_uint(As[sw_mc<BM>(kk + t + 4, mb + g)]); r[3] = __float_as_uint(As[sw_mc<BM>(kk + t + 4, mb + g + 8)]);
                        }
                        af[i][2 * h] = pack_bf16(__uint_as_float(r[0]), __uint_as_float(r[2]));
                        af[i][2 * h + 1] = pack_bf16(__uint_as_float(r[1]), __uint_as_float(r[3]));
                    }
#pragma unroll
                    for (int jp = 0; jp < NJ / 2; jp++) {
                        const int nb = wni * WTN + jp * 16;
                        unsigned r[4];
                        if (TB) ldmatrix_x4(r, Bs + sw_kc<BK>(nb + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
                        else {
                            r[0] = __float_as_uint(Bs[sw_mc<BN>(kk + t, nb + g)]);     r[1] = __float_as_uint(Bs[sw_mc<BN>(kk + t + 4, nb + g)]);
                            r[2] = __float_as_uint(Bs[sw_mc<BN>(kk + t, nb + 8 + g)]); r[3] = __float_as_uint(Bs[sw_mc<BN>(kk + t + 4, nb + 8 + g)]);
                        }
                        bfr[2 * jp][h] = pack_bf16(__uint_as_float(r[0]), __uint_as_float(r[1]));
                        bfr[2 * jp + 1][h] = pack_bf16(__uint_as_float(r[2]), __uint_as_float(r[3]));
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < NJ; j++) mma_bf16(acc[i][j], af[i], bfr[j][0], bfr[j][1]);
            }
        } else tile_slab<TA, TB, PREC, RZ, BK, BM, BN, 4, NJ, BK / 8>(As, Bs, acc, wmi * 64, wni * WTN, 0);
    }
    }
    KVM_GPT_PROBE(2);
    cp_wait<0>();
    constexpr int CLD = BN + 8;   // row stride of the accumulator tile in shared memory (padded against bank conflicts)
    if constexpr (BM * CLD * 4 <= kvm_gpt_smem_bytes(64)) {   // accumulators through shared memory (the register epilogue below costs one 8-byte store per accumulator pair, in instructions the block's 8 warps share)
        __syncthreads();   // all warps done reading the last stage; shared memory now takes the accumulators
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {   // c0,c1 = (row g, cols 2t,2t+1); c2,c3 = (row g+8, ...)
                const int r = wmi * 64 + i * 16 + g, c = wni * WTN + j * 8 + 2 * t;
                *reinterpret_cast<float2*>(sm + r * CLD + c) = make_float2(acc[i][j][0], acc[i][j][1]);
                *reinterpret_cast<float2*>(sm + (r + 8) * CLD + c) = make_float2(acc[i][j][2], acc[i][j][3]);
            }
        __syncthreads();
        tile_wb<BM, BN, 1, CLD>(sm, m0, n0, M, N, out, ldo, bias, res, ldr, pre, ldp, alpha, act, accum);
        return;
    }
    // Epilogue from registers: thread holds (row g | g+8, columns 2t, 2t+1) of every fragment. Pairs of columns go out as float2
    // when every operand is 8-byte aligned.
    const bool pair = ((n0 & 1) == 0) && (((uintptr_t)out | (uintptr_t)(ldo * 4)) & 7) == 0 && (!bias || (((uintptr_t)bias) & 7) == 0) &&
                      (!res || (((uintptr_t)res | (uintptr_t)(ldr * 4)) & 7) == 0) && (!pre || (((uintptr_t)pre | (uintptr_t)(ldp * 4)) & 7) == 0);
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
        const int gm = m0 + wmi * 64 + i * 16 + g + hf * 8, gn = n0 + wni * WTN + j * 8 + 2 * t;
        if (gm >= M || gn >= N) continue;
        float v[2] = {acc[i][j][2 * hf] * alpha, acc[i][j][2 * hf + 1] * alpha};
        if (pair && gn + 1 < N) {
            if (bias) { const float2 b2 = *reinterpret_cast<const float2*>(bias + gn); v[0] += b2.x; v[1] += b2.y; }
            if (act == ACT_DGELU) { const float2 p2 = *reinterpret_cast<const float2*>(pre + (size_t)gm * ldp + gn); v[0] *= gelu_grad(p2.x); v[1] *= gelu_grad(p2.y); }
            else {
                if (pre) *reinterpret_cast<float2*>(pre + (size_t)gm * ldp + gn) = make_float2(v[0], v[1]);
                if (act == ACT_GELU) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); }
            }
            if (res) { const float2 r2 = *reinterpret_cast<const float2*>(res + (size_t)gm * ldr + gn); v[0] += r2.x; v[1] += r2.y; }
            float2* o = reinterpret_cast<float2*>(out + (size_t)gm * ldo + gn);
            if (accum) { const float2 o2 = *o; v[0] += o2.x; v[1] += o2.y; }
            *o = make_float2(v[0], v[1]);
        } else {
            for (int e = 0; e < 2 && gn + e < N; e++) {
                float x = v[e];
                if (bias) x += bias[gn + e];
                if (act == ACT_DGELU) x *= gelu_grad(pre[(size_t)gm * ldp + gn + e]);
                else {
                    if (pre) pre[(size_t)gm * ldp + gn + e] = x;
                    if (act == ACT_GELU) x = gelu_f(x);
                }
                if (res) x += res[(size_t)gm * ldr + gn + e];
                float* o = out + (size_t)gm * ldo + gn + e;
                *o = accum ? *o + x : x;
            }
        }
    }
}

// Out-of-line copy of each big-tile variant for the 256x128 task's two passes (see matmul_tile). Inlined into a pass LOOP the tile spilled 28 KB (a third
// slower), as a function it keeps its own register allocation.
template <bool TA, bool TB, bool PREC, int WM, int WN, int WTN, bool BF, bool RZ>
__device__ __noinline__ void mm_tile_big_ni(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                                            int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, int act, bool accum) {
    mm_tile_big<TA, TB, PREC, WM, WN, WTN, BF, RZ>(A, lda, B, ldb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, act, accum);
}

KVM_GPT_DEVI void matmul_tile(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8, prec = fl & 16, big = fl & 32, huge = fl & 64, bf = fl & 128, trunc = fl & 256, dg = fl & 512, exact = fl & 1024;
    const bool rz = trunc || exact;   // the tile hands its operands to the tensor core as they are: truncated (256, output scaled to remove the bias) or already TF32 values (1024, from round_tf32)
    k.need(!(big && huge) && !(bf && prec) && (!rz || (!prec && !bf)) && !(trunc && exact), KVM_GPT_E_SHAPE);   // 256 and 1024 only for the plain tf32 path, and not together
    k.need(!dg || (a[5] != -1 && a[3] == -1 && a[4] == -1 && !accum && !gelu && !bf), KVM_GPT_E_SHAPE);   // 512 (dgelu): pre is its input, and it replaces the rest of the epilogue
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~2047) == 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    int lda, ldb, ldo, ldr = 0, ldp = 0;
    const float* A = k.mat(a[0], ta ? K : M, ta ? M : K, &lda);
    const float* B = k.mat(a[1], tb ? N : K, tb ? K : N, &ldb);
    float* out = k.mat(a[2], M, N, &ldo);
    const float* bias = k.fvec(a[3], N, true);
    const float* res = k.mat(a[4], M, N, &ldr, true);
    float* pre = k.mat(a[5], M, N, &ldp, true);
    if (k.bad) return;
    KVM_GPT_PROBE(0);
    float alpha = af(a, 12);
#ifdef KVM_HOPPER_DEVICE
    if (!prec && !bf && !rz && !dg) {   // plain 1xTF32 on sm_90a: TMA + wgmma (kvm_hopper.cuh); declines (alignment, shared memory, no host init) fall through. bf16 operands (128), truncated or exact operands (256, 1024) and the dgelu epilogue (512) stay on the tiles below
        const bool done = big ? kvmhop::mm_f32<128, 128>(A, lda, ta, B, ldb, tb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, gelu, accum, kvm_gpt_smem)
                        : huge ? kvmhop::mm_f32<256, 128>(A, lda, ta, B, ldb, tb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, gelu, accum, kvm_gpt_smem)
                               : kvmhop::mm_f32<64, 64>(A, lda, ta, B, ldb, tb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, gelu, accum, kvm_gpt_smem);
        if (threadIdx.x == 0) atomicAdd(&kvm_hop_count[done ? 0 : 2], 1ull);
        if (done) return;
    }
#endif
    if (trunc) alpha *= KVM_GPT_RZ_GAIN;
    const int act = dg ? ACT_DGELU : gelu ? ACT_GELU : ACT_NONE;
// Tile variants are template arguments: transposes, 3xTF32 (PREC), bf16 operands (BF, flag 128), truncated operands (RZ, flag 256). bf16 has
// no 3xTF32 form and RZ only exists for the big tiles (validated above).
#define KVM_GPT_ARGS A, lda, B, ldb, out, ldo, bias, res, ldr, pre, ldp, mm0, n0, M, N, K, alpha, act, accum
#define KVM_GPT_ONE_BIG(F, TA, TB, P, BF, RZ) F<TA, TB, P, 2, 4, 32, BF, RZ>(KVM_GPT_ARGS)   // 8 warps as 2 (m) x 4 (n), 64 x 32 each
#define KVM_GPT_ONE_HUGE(F, TA, TB, P, BF, RZ) F<TA, TB, P, 4, 2, 64, BF, RZ>(KVM_GPT_ARGS)   // 8 warps as 4 (m) x 2 (n), 64 x 64 each
#define KVM_GPT_ONE_64(F, TA, TB, P, BF, RZ) F<TA, TB, P, BF, RZ>(KVM_GPT_ARGS)
#define KVM_GPT_MM_T(ONE, F, P, BF, RZ)                                                                                         \
    if (!ta && !tb) ONE(F, false, false, P, BF, RZ); else if (ta && !tb) ONE(F, true, false, P, BF, RZ);                       \
    else if (!ta && tb) ONE(F, false, true, P, BF, RZ); else ONE(F, true, true, P, BF, RZ)
#define KVM_GPT_MM_BIG(F)                                                                                                        \
    if (bf) { KVM_GPT_MM_T(KVM_GPT_ONE_BIG, F, false, true, false); }                                                          \
    else if (rz) { KVM_GPT_MM_T(KVM_GPT_ONE_BIG, F, false, false, true); }                                                     \
    else if (prec) { KVM_GPT_MM_T(KVM_GPT_ONE_BIG, F, true, false, false); }                                                   \
    else { KVM_GPT_MM_T(KVM_GPT_ONE_BIG, F, false, false, false); }
    if (big) {
        const int mm0 = m0;
        KVM_GPT_MM_BIG(mm_tile_big)
    } else if (huge && !bf && !prec) {
        // 256x128 task (flag 64) as one tile of 4x2 warps, 64x64 per warp: 128 accumulators, so it needs the one-block-per-SM register budget (__launch_bounds__(256, 1)), which the executor uses.
        // A quarter of the shared-memory traffic per MMA of the 128x128 tile and half its operand traffic per FLOP (0.83x cuBLAS at 4096^3 against 0.76x, A100, truncated operands). Out of line: the union kernel's code size.
        const int mm0 = m0;
        if (rz) { KVM_GPT_MM_T(KVM_GPT_ONE_HUGE, mm_tile_big_ni, false, false, true); }
        else { KVM_GPT_MM_T(KVM_GPT_ONE_HUGE, mm_tile_big_ni, false, false, false); }
    } else if (huge) {
        // 256x128 task with bf16 operands or 3xTF32: two 128x128 passes, rows m0 and m0+128 (their warp tiles do not fit 128 accumulators plus the 3xTF32 lows).
        for (int pass = 0; pass < 2; pass++) {
            const int mm0 = m0 + 128 * pass;
            if (mm0 >= M) break;
            if (pass) __syncthreads();   // the second pass reuses the shared-memory ring
            KVM_GPT_MM_BIG(mm_tile_big_ni)
        }
    } else {
        const int mm0 = m0;
        if (bf) { KVM_GPT_MM_T(KVM_GPT_ONE_64, mm_tile, false, true, false); }
        else if (prec) { KVM_GPT_MM_T(KVM_GPT_ONE_64, mm_tile, true, false, false); }
        else if (rz) { KVM_GPT_MM_T(KVM_GPT_ONE_64, mm_tile, false, false, true); }
        else { KVM_GPT_MM_T(KVM_GPT_ONE_64, mm_tile, false, false, false); }
    }
#undef KVM_GPT_MM_BIG
#undef KVM_GPT_MM_T
#undef KVM_GPT_ONE_64
#undef KVM_GPT_ONE_BIG
#undef KVM_GPT_ONE_HUGE
#undef KVM_GPT_ARGS
}

// ---- Staged GEMM epilogue (#16), shared: the bf16 tile (kvm_mma_bf16.cuh) and the #16 tile set below use it ----------------------
// Staged epilogue of the tensor-core GEMM tiles (mm_tile_big, mma_bf16_tile). The accumulators of one warp row (64 output rows) at a
// time go to shared memory as fp32 (row stride BN + 8 floats: the float2 stores of a half warp hit 32 distinct banks), then all 256
// threads walk the 64 x BN slab in a ROLLED loop of 4-element pieces: v = acc * alpha (+ bias) -> pre (+ gelu) (+ res) (+ out if
// accumulate) -> out, the same operations in the same order as the register epilogue it replaces. Why: the register epilogue was fully
// unrolled over 4 x NJ x 2 fragment rows with every optional operand tested per element, about 180 KB of straight-line SASS run once per
// tile; it missed the instruction cache on almost every instruction and took 42% of a 4096^3 bf16 GEMM (ncu: stall_no_inst 36% of all
// warp samples, results/2026-11-09-a100-matmul-parity). The rolled loop is a few hundred bytes and its global accesses are 16-byte
// (fp32) / 8-byte (bf16) per thread, coalesced along rows. Callers must have drained their cp.async pipeline (cp_wait<0>()): the slab
// reuses the stage buffers. Storage: out / res / pre are fp32 or bf16 (ob / rb / pb), bias fp32.
// PM / PN: the tile's fragments use the permuted row / column ownership of mm_tf32_tile (MN-major operands, see there): fragment row
// (i, g, half) is warp row 8g + 2(i ^ (g&1)) + half instead of i*16 + g + 8*half; fragment column (j, 2t+e) is warp column (2t+e)*NJ + j.
template <int WM, int WN, int WTN, bool PM = false, bool PN = false>
KVM_GPT_DEVI void mm_epilogue_staged(const float (&acc)[4][WTN / 8][4], void* out, int ldo, bool ob, const float* bias, const void* res, int ldr,
                                     bool rb, void* pre, int ldp, bool pb, int m0, int n0, int M, int N, float alpha, bool gelu, bool accum) {
    constexpr int BN = WN * WTN, LDS = BN + 8, NJ = WTN / 8, PIECES = 64 * BN / 4;
    static_assert(WM * WN * 32 == KVM_GPT_THREADS, "8 warps");
    static_assert(64 * LDS * 4 <= KVM_GPT_MMA_SMEM, "the staged epilogue slab must fit the dynamic shared memory");
    float* st = reinterpret_cast<float*>(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wmi = warp / WN, wni = warp % WN, g = lane >> 2, t = lane & 3;
    // 4-wide vector accesses are legal for a piece when its first column is a multiple of 4 and each operand's base and ld allow it
    const bool vo = (n0 & 3) == 0 && vec4_ok(out, ldo, ob), vr = !res || vec4_ok(res, ldr, rb), vp = !pre || vec4_ok(pre, ldp, pb),
               vbias = !bias || ((uintptr_t)bias & 15) == 0, vall = vo && vr && vp && vbias;
    __syncthreads();   // every warp is done reading the last pipeline stage
    for (int p = 0; p < WM; p++) {
        if (wmi == p) {
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        float* row = st + (PM ? 8 * g + 2 * (i ^ (g & 1)) + hf : i * 16 + g + hf * 8) * LDS + wni * WTN;
                        if (PN) { row[(2 * t) * NJ + j] = acc[i][j][2 * hf]; row[(2 * t + 1) * NJ + j] = acc[i][j][2 * hf + 1]; }
                        else *reinterpret_cast<float2*>(row + j * 8 + 2 * t) = make_float2(acc[i][j][2 * hf], acc[i][j][2 * hf + 1]);
                    }
        }
        __syncthreads();
        const int r0 = m0 + p * 64;
#pragma unroll 1
        for (int q = threadIdx.x; q < PIECES; q += KVM_GPT_THREADS) {
            const int r = q / (BN / 4), c = (q % (BN / 4)) * 4, gm = r0 + r, gn = n0 + c;
            if (gm >= M || gn >= N) continue;
            const float4 a4 = *reinterpret_cast<const float4*>(st + r * LDS + c);
            float v[4] = {a4.x * alpha, a4.y * alpha, a4.z * alpha, a4.w * alpha};
            const size_t oi = (size_t)gm * ldo + gn, ri = (size_t)gm * ldr + gn, pi = (size_t)gm * ldp + gn;
            if (vall && gn + 3 < N) {
                if (bias) { const float4 b = *reinterpret_cast<const float4*>(bias + gn); v[0] += b.x; v[1] += b.y; v[2] += b.z; v[3] += b.w; }
                if (pre) st4(pre, pi, make_float4(v[0], v[1], v[2], v[3]), pb);
                if (gelu) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); v[2] = gelu_f(v[2]); v[3] = gelu_f(v[3]); }
                if (res) { const float4 x = ld4(res, ri, rb); v[0] += x.x; v[1] += x.y; v[2] += x.z; v[3] += x.w; }
                if (accum) { const float4 x = ld4(out, oi, ob); v[0] += x.x; v[1] += x.y; v[2] += x.z; v[3] += x.w; }
                st4(out, oi, make_float4(v[0], v[1], v[2], v[3]), ob);
            } else {
#pragma unroll 1
                for (int e = 0; e < 4 && gn + e < N; e++) {
                    float x = v[e];
                    if (bias) x += bias[gn + e];
                    if (pre) st_s(pre, pi + e, x, pb);
                    if (gelu) x = gelu_f(x);
                    if (res) x += ld_s(res, ri + e, rb);
                    if (accum) x += ld_s(out, oi + e, ob);
                    st_s(out, oi + e, x, ob);
                }
            }
        }
        __syncthreads();   // the slab is overwritten by the next warp row
    }
}

// ---- The #16 tile set, beside the tiles matmul_tile dispatches to ---------------------------------------------------------------
// Jerry's fast 1xTF32 big tiles (mm_tf32_tile) and their dispatch, as he wrote them (Bobliuuu/kvm e7186a7; the staged epilogue above),
// so tests/gpu_ops/bench_mm.cu can time both tile sets in one binary on the same task words. Templates and an inline dispatch:
// none of it is compiled into a binary that does not call it (the executor does not). The A/B decides which set stays.
#define KVM_GPT_TF32_TILE 1   // kvmgpt::j16::mm_tf32_tile exists (benchmarks that also build against older trees test this)
namespace j16 {
// Operand rounding of this set (the same bits as kvmgpt::f2tf32_op for every input):
// Operand for a single (1x) TF32 mma: the tensor core ignores the low 13 bits, so adding the rounding bias is round-to-nearest (ties away,
// the value cvt.rna.tf32 gives) in one add. Finite values only: Inf and NaN keep their bits (the core reads Inf either way, and a NaN with a
// full mantissa would carry into the sign). Two instructions (FSETP on |x| + a predicated IADD; CUTLASS's round_half_ulp_truncate),
// one fewer than the integer-compare form it replaces, and bitwise the operands cuBLAS TF32 feeds its tensor cores (results/
// 2026-11-09-a100-matmul-parity: 0 difference on 4096^3). (Round-toward-zero would save the add but triples the error: 7.6e-4 vs 2.7e-4.)
KVM_GPT_DEVI unsigned f2tf32_op(float x) { const unsigned u = __float_as_uint(x); return fabsf(x) < __int_as_float(0x7f800000) ? u + 0x1000u : u; }
KVM_GPT_DEVI unsigned dyn_smem_bytes() { unsigned r; asm("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(r)); return r; }   // of this launch


// ---- mm_tf32_tile: the fast path of the 128x128 (flag 32) and 256x128 (flag 64) 1xTF32 tiles ------------------------------------
// Same math as mm_tile_big (operands rounded to nearest by f2tf32_op, fp32 accumulation in increasing k: bitwise the result of cuBLAS
// TF32 on the same shapes), a different schedule: NST cp.async stages of BK floats of k, the fragments of the next k8 step loaded while
// the current one feeds the tensor cores, one barrier per stage, and the staged epilogue. K-major operands (A as [m][k], B as [n][k]) use
// ldmatrix. MN-major operands (A as [k][m], B as [k][n]) cannot (ldmatrix moves 16-bit elements), and the loads must land in the
// registers the mma reads (a register shuffle per value cost 20% of the loop): B uses one LDS.32 per value in the standard fragment
// layout; A uses one LDS.64 per (a0, a1) / (a2, a3) pair, which needs the two rows of a pair adjacent in m, so fragment i of thread g
// owns warp rows 8g + 2(i ^ (g&1)) + {0, 1} (the parity flip spreads a half warp over both 8-byte halves of each 16-byte bank group).
// A sum over k does not care which output element a register pair belongs to; only the epilogue does (PM of mm_epilogue_staged).
// Measured on A100-SXM4-40GB, 4096^3, same process as cuBLAS TF32 (results/2026-11-09-a100-matmul-parity): 256 x 128 BK 32 x 3 stages
// (144 KiB) 0.76-0.80x cuBLAS, BK 16 x 4 stages (96 KiB) 0.69-0.78x; 128 x 128 (warp tile 64 x 32) 0.63-0.65x.
// Preconditions (else matmul_tile falls back to mm_tile_big): A, B 16-byte aligned with lda, ldb, K multiples of 4, and the MN-major
// extents multiples of 4 (a 16-byte chunk is either wholly inside the matrix or wholly zero-filled).
// MN-major stage rows: 16-byte chunk c of k row k is stored at chunk c ^ sw_f<S>(k). S = 1 (B, read by LDS.32 in the standard column
// mapping: the 4 k rows of a warp's load land on chunk pairs 2j | 2j+1 XOR {0, 2, 4, 6}), S = 2 (A, read by LDS.64 in the permuted row
// mapping, see mm_tf32_tile: the 16 lanes of a half warp cover 4 chunks x 4 k rows x 2 halves). Both are bank-conflict free.
template <int S> KVM_GPT_DEVI int sw_f(int k) { return S == 1 ? (k & 3) << 1 : (k & 3); }
// One operand stage tile: KMAJ: ROWS x 16 floats ([mn][k]); else 16 x ROWS floats ([k][mn]). 256 threads, NS = ROWS*4/256 16-byte chunks
// each. A thread's chunks are RS rows apart (same swizzle phase), so its state is one running global pointer (advanced one stage per
// issue: stages are issued in order), one shared offset and a validity mask; the rest is immediates. Keeping this small matters: the
// warp tile holds 128 accumulators, and a larger loader state made the compiler rebuild every address from threadIdx in the k loop.
template <bool KMAJ, int ROWS, int S, int BK = 16>
struct Tf32Loader {
    static constexpr int NS = ROWS * BK / 4 / KVM_GPT_THREADS, CPR = KMAJ ? BK / 4 : ROWS / 4, RS = KVM_GPT_THREADS / CPR;
    const float* p;   // this thread's chunk 0 of the next stage to issue (src when chunk 0 is outside the mn extent)
    int off;          // shared offset of chunk 0 (floats)
    int kq;           // k of chunk 0 within a stage
    int mask;         // bit s: chunk s inside the mn extent
    KVM_GPT_DEVI void init(const float* src, int ld, int mn0, int MN) {
        const int row = threadIdx.x / CPR, c = threadIdx.x % CPR;
        mask = 0;
        if (KMAJ) {
            off = sw_kc<BK>(row, c * 4); kq = c * 4;
#pragma unroll
            for (int s = 0; s < NS; s++) mask |= (mn0 + row + RS * s < MN) << s;
            p = (mask & 1) ? src + (size_t)(mn0 + row) * ld + kq : src;
        } else {
            off = row * ROWS + ((c ^ sw_f<S>(row)) << 2); kq = row;
            mask = mn0 + c * 4 < MN ? (1 << NS) - 1 : 0;
            p = mask ? src + (size_t)row * ld + mn0 + c * 4 : src;
        }
    }
    // stage at k0 into `tile`; ld and K are the caller's (shared by both operands' loaders to save registers)
    KVM_GPT_DEVI void issue(float* tile, int k0, int ld, int K) {
#pragma unroll
        for (int s = 0; s < NS; s++) {
            const bool ok = ((mask >> s) & 1) && (KMAJ ? k0 + kq : k0 + kq + RS * s) < K;
            cp_async16(tile + off + s * RS * (KMAJ ? BK : ROWS), ok ? p + s * RS * ld : p, ok ? 16 : 0);
        }
        if (mask) p += KMAJ ? BK : BK * ld;
    }
};
template <bool PREC> KVM_GPT_DEVI void tf32_split(unsigned& h, unsigned& l) {
    const float x = __uint_as_float(h);
    if (PREC) { const unsigned hi = f2tf32(x); l = __float_as_uint(x - __uint_as_float(hi)); h = hi; }
    else h = f2tf32_op(x);
}
// The epilogue operands (out, bias, res, pre, their leading dimensions, alpha, flags) are read from the task words only after the k loop:
// passed in, they stayed live through it, and the 256 x 128 loop (128 accumulators + 64 fragment registers) spilled and ran at 0.5x
// instead of 0.8x cuBLAS. matmul_tile has validated them already.
// Out of line (__noinline__): inlined into matmul_tile, or into the executor's op switch, the loop shared the register budget with every
// value live in the caller and spilled in the k loop (14 local loads per stage, 0.37-0.5x cuBLAS); as its own function it allocates like a
// standalone kernel. The call costs a few hundred cycles once per tile.
template <bool TA, bool TB, bool PREC, int WM, int WN, int WTN, int BK, int NST>
__device__ __noinline__ void mm_tf32_tile(const float* A, int lda, const float* B, int ldb, int m0, int n0, int M, int N, int K, const KvmCtx& ctx,
                               const int32_t* a) {
    constexpr int BM = 64 * WM, BN = WTN * WN, NJ = WTN / 8, ATILE = BM * BK, STAGE = (BM + BN) * BK, KS = BK / 8;   // KS k8 steps per stage
    constexpr int SA = 2, SB = 1;        // swizzle modes of the MN-major A / B stage tiles (sw_f)
    static_assert(WM * WN == 8 && (NJ == 4 || NJ == 8), "8 warps, warp tile 64 x 32|64");
    static_assert(NST * STAGE * 4 <= KVM_GPT_MMA_SMEM, "mm_tf32_tile stages must fit the dynamic shared memory");
    float* sm = reinterpret_cast<float*>(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wmi = warp / WN, wni = warp % WN, g = lane >> 2, t = lane & 3;
    float acc[4][NJ][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    const int nk = (K + BK - 1) / BK;
    Tf32Loader<!TA, BM, SA, BK> la;
    Tf32Loader<TB, BN, SB, BK> lb;
    la.init(A, lda, m0, M);
    lb.init(B, ldb, n0, N);
    int wst = 0;   // stage the next issue writes (stages are issued in order, it % NST without the division)
    auto issue = [&](int it) { float* st = sm + wst * STAGE; la.issue(st, it * BK, lda, K); lb.issue(st + ATILE, it * BK, ldb, K); wst = wst + 1 == NST ? 0 : wst + 1; };
    unsigned ah[2][4][4], bh[2][NJ][2], al[PREC ? 2 : 1][4][4], bl[PREC ? 2 : 1][NJ][2];
    auto load_frags = [&](int buf, int rs, int kk) {   // rs: stage index to read
        const float* As = sm + rs * STAGE; const float* Bs = As + ATILE;
        if (!TA) {
#pragma unroll
            for (int i = 0; i < 4; i++) ldmatrix_x4(ah[buf][i], As + sw_kc<BK>(wmi * 64 + i * 16 + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
        } else {
#pragma unroll
            for (int hk = 0; hk < 2; hk++) {   // k rows kk+t (a0, a1) and kk+t+4 (a2, a3); fragment i rows = warp rows 8g + 2(i ^ (g&1)) + {0, 1}
                const int k = kk + t + 4 * hk;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int m = wmi * 64 + 8 * g + 2 * (i ^ (g & 1));
                    const float2 x = *reinterpret_cast<const float2*>(As + k * BM + ((((m >> 2) ^ sw_f<SA>(k))) << 2) + (m & 3));
                    ah[buf][i][2 * hk] = __float_as_uint(x.x); ah[buf][i][2 * hk + 1] = __float_as_uint(x.y);
                }
            }
        }
        if (TB) {
#pragma unroll
            for (int jp = 0; jp < NJ / 2; jp++) {
                unsigned r[4];
                ldmatrix_x4(r, Bs + sw_kc<BK>(wni * WTN + jp * 16 + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
                bh[buf][2 * jp][0] = r[0]; bh[buf][2 * jp][1] = r[1]; bh[buf][2 * jp + 1][0] = r[2]; bh[buf][2 * jp + 1][1] = r[3];
            }
        } else {
#pragma unroll
            for (int j = 0; j < NJ; j++) {   // standard column mapping: n = warp column 8j + g; k rows kk+t (b0) and kk+t+4 (b1)
                const int n = wni * WTN + 8 * j + g, k = kk + t;
                const float* q = Bs + k * BN + ((((n >> 2) ^ sw_f<SB>(k))) << 2) + (n & 3);
                bh[buf][j][0] = __float_as_uint(q[0]); bh[buf][j][1] = __float_as_uint(q[4 * BN]);
            }
        }
    };
    auto compute = [&](int buf) {
        constexpr int LB = PREC ? 1 : 0;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) tf32_split<PREC>(ah[buf][i][e], al[LB * buf][i][e]);
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) tf32_split<PREC>(bh[buf][j][e], bl[LB * buf][j][e]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                if (PREC) { mma_tf32(acc[i][j], al[LB * buf][i], bh[buf][j][0], bh[buf][j][1]); mma_tf32(acc[i][j], ah[buf][i], bl[LB * buf][j][0], bl[LB * buf][j][1]); }
                mma_tf32(acc[i][j], ah[buf][i], bh[buf][j][0], bh[buf][j][1]);
            }
    };
#pragma unroll
    for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }
    cp_wait<NST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
    int rst = 0;   // stage of slab it (it % NST)
    for (int it = 0; it < nk; it++) {
        const int rnx = rst + 1 == NST ? 0 : rst + 1;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {   // k8 step ks: fragments of step ks+1 load while step ks computes
            const int b = ks & 1;
            if (ks + 1 < KS) { load_frags(b ^ 1, rst, (ks + 1) * 8); compute(b); continue; }
            cp_wait<NST - 3>();     // last k8 step of the stage: stage it+1 landed (this thread's copies) ...
            __syncthreads();        // ... for every thread, and nobody still reads the stage refilled below
            if (it + NST - 1 < nk) issue(it + NST - 1);
            cp_commit();
            if (it + 1 < nk) load_frags(b ^ 1, rnx, 0);
            compute(b);
        }
        rst = rnx;
    }
    cp_wait<0>();
    __syncthreads();   // also keeps the operand reads below after the loop
    Chk k(ctx);
    int ldo, ldr = 0, ldp = 0;
    float* out = k.mat(a[2], M, N, &ldo);
    const float* bias = k.fvec(a[3], N, true);
    const float* res = k.mat(a[4], M, N, &ldr, true);
    float* pre = k.mat(a[5], M, N, &ldp, true);
    const int fl = a[11];
    mm_epilogue_staged<WM, WN, WTN, TA, false>(acc, out, ldo, false, bias, res, ldr, false, pre, ldp, false, m0, n0, M, N, af(a, 12), fl & 8, fl & 4);
}


// Its dispatch: the cases #16's matmul_tile sends to mm_tf32_tile (big or huge tile, plain 1xTF32: none of flags 16, 128, 256, 512, 1024;
// operands in whole aligned 16-byte chunks); every other task goes to kvmgpt::matmul_tile. The 256 x 128 tile takes BK 32 x 3 stages
// when the launch has 144 KiB of shared memory and the build compiled it (KVM_GPT_MMA_SMEM >= 147456), else BK 16 x 4 stages (96 KiB).
KVM_GPT_DEVI void matmul_tile(const KvmCtx& ctx, const int32_t* a) {
    const int fl = a[11], m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10];
    const bool ta = fl & 1, tb = fl & 2, big = fl & 32, huge = fl & 64, plain = (fl & (16 | 128 | 256 | 512 | 1024)) == 0;
    if (!(big || huge) || !plain) { kvmgpt::matmul_tile(ctx, a); return; }
    Chk k(ctx);
    int lda = 0, ldb = 0;
    const float* A = k.mat(a[0], ta ? K : M, ta ? M : K, &lda);
    const float* B = k.mat(a[1], tb ? N : K, tb ? K : N, &ldb);
    const bool fast = !k.bad && aligned16(A, lda) && aligned16(B, ldb) && (K & 3) == 0 && (m0 & 3) == 0 && (n0 & 3) == 0 &&
                      (!ta || (M & 3) == 0) && (tb || (N & 3) == 0);
    if (!fast) { kvmgpt::matmul_tile(ctx, a); return; }   // also every argument error: kvmgpt::matmul_tile reports it
    const bool deep = dyn_smem_bytes() >= 3 * (256 + 128) * 32 * 4;
    (void)deep;
#define KVM_GPT_MMF(TA, TB, ...) mm_tf32_tile<TA, TB, false, __VA_ARGS__>(A, lda, B, ldb, m0, n0, M, N, K, ctx, a)
#define KVM_GPT_MMF_T(...)                                                                                                   \
    if (!ta && !tb) KVM_GPT_MMF(false, false, __VA_ARGS__); else if (ta && !tb) KVM_GPT_MMF(true, false, __VA_ARGS__);      \
    else if (!ta && tb) KVM_GPT_MMF(false, true, __VA_ARGS__); else KVM_GPT_MMF(true, true, __VA_ARGS__)
    // each variant is compiled only where KVM_GPT_MMA_SMEM holds its stages (its static_assert); a build too small for one hands the task to ours
    if (big) {
#if KVM_GPT_MMA_SMEM >= 3 * (128 + 128) * 32 * 4
        KVM_GPT_MMF_T(2, 4, 32, 32, 3);                                  // 128 x 128, BK 32 x 3 stages (96 KiB)
#else
        kvmgpt::matmul_tile(ctx, a);
#endif
    }
#if KVM_GPT_MMA_SMEM >= 3 * (256 + 128) * 32 * 4                         // only builds whose launch can have it compile the deep tile
    else if (deep) { KVM_GPT_MMF_T(4, 2, 64, 32, 3); }                   // 256 x 128, BK 32 x 3 stages (144 KiB)
#endif
    else {
#if KVM_GPT_MMA_SMEM >= 4 * (256 + 128) * 16 * 4
        KVM_GPT_MMF_T(4, 2, 64, 16, 4);                                  // 256 x 128, BK 16 x 4 stages (96 KiB)
#else
        kvmgpt::matmul_tile(ctx, a);
#endif
    }
#undef KVM_GPT_MMF_T
#undef KVM_GPT_MMF
}
}   // namespace j16

// ---- normalization: one warp per row ----------------------------------------------------------------------------------
// Fast path: every lane keeps up to KVM_GPT_NORM_NV float4 of its row in registers (rows up to 128*NV = 1024 columns), so global
// memory is read once and written once. Needs d%4==0 and 16-byte aligned rows/params; otherwise the scalar path runs.
#define KVM_GPT_NORM_NV 8
#ifndef KVM_GPT_ROPE_U
#define KVM_GPT_ROPE_U 4   // rope: heads of a row in flight per pass
#endif
#ifndef KVM_GPT_NORM_U
#define KVM_GPT_NORM_U 4   // narrow rows: row groups of a warp in flight per pass
#endif
KVM_GPT_DEVI bool norm_rows_vec(int d, const void* x, int ldx, bool xb, const float* p0, const void* y, int ldy, bool yb, const float* p1) {
    return (d & 3) == 0 && (d >> 2) <= 32 * KVM_GPT_NORM_NV && vec4_ok(x, ldx, xb) && vec4_ok(y, ldy, yb) && aligned16(p0, 0) && aligned16(p1, 0);
}
template <bool RMS>
KVM_GPT_DEVI void norm_fwd(const KvmCtx& ctx, const int32_t* a, int xi, int gi, int bi, int oi, int mi, int ri, int base) {
    Chk k(ctx);
    int r0 = a[base], r1 = a[base + 1], d = a[base + 2], ldx, ldo;
    float eps = af(a, base + 3);
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    bool xb, ob;
    const void* xs = k.mat_s(a[xi], r1, d, &ldx, &xb);
    const float* g = k.fvec(a[gi], d);
    const float* be = RMS ? nullptr : k.fvec(a[bi], d);
    void* os = k.mat_s(a[oi], r1, d, &ldo, &ob);
    float* mean = RMS ? nullptr : k.fvec(a[mi], r1);
    float* rstd = k.fvec(a[ri], r1);
    if (k.bad) return;
    int lane = threadIdx.x & 31;
    if (norm_rows_vec(d, xs, ldx, xb, g, os, ldo, ob, RMS ? g : be)) {   // one pass over global memory: the row lives in registers
        const int d4 = d >> 2, rpw = ((d4 & (d4 - 1)) == 0 && d4 <= 16) ? 32 / d4 : 1, grp = rpw > 1 ? lane / d4 : 0, lc = rpw > 1 ? lane % d4 : lane;   // narrow rows: rpw rows per warp at once, d4 lanes on each
        if (rpw > 1) {   // one float4 of a row per lane: KVM_GPT_NORM_U row groups in flight, so their loads overlap instead of paying one memory round trip each; the parameters are read once
            constexpr int U = KVM_GPT_NORM_U, STRIDE = KVM_GPT_THREADS / 32;
            const float4 gg = reinterpret_cast<const float4*>(g)[lc], bb = RMS ? make_float4(0.f, 0.f, 0.f, 0.f) : reinterpret_cast<const float4*>(be)[lc];
            for (int rb = r0 + (threadIdx.x >> 5) * rpw; rb < r1; rb += U * STRIDE * rpw) {
                float4 xv[U];
                int r[U];
#pragma unroll
                for (int u = 0; u < U; u++) {
                    if (rb + u * STRIDE * rpw >= r1) continue;   // warp-uniform: the task's rows end before this group (a 32-row task runs one group, not U)
                    r[u] = rb + u * STRIDE * rpw + grp;
                    xv[u] = r[u] < r1 ? ld4(xs, (size_t)r[u] * ldx + 4 * lc, xb) : make_float4(0.f, 0.f, 0.f, 0.f);
                }
#pragma unroll
                for (int u = 0; u < U; u++) {
                    if (rb + u * STRIDE * rpw >= r1) continue;
                    const float mu = RMS ? 0.f : gsum((xv[u].x + xv[u].y) + (xv[u].z + xv[u].w), d4, rpw) / d;
                    const float t0 = xv[u].x - mu, t1 = xv[u].y - mu, t2 = xv[u].z - mu, t3 = xv[u].w - mu;
                    const float rs = 1.f / sqrtf(gsum((t0 * t0 + t1 * t1) + (t2 * t2 + t3 * t3), d4, rpw) / d + eps);
                    if (r[u] < r1) {
                        float4 o;
                        o.x = t0 * rs * gg.x + bb.x; o.y = t1 * rs * gg.y + bb.y;
                        o.z = t2 * rs * gg.z + bb.z; o.w = t3 * rs * gg.w + bb.w;
                        st4(os, (size_t)r[u] * ldo + 4 * lc, o, ob);
                        if (lc == 0) { rstd[r[u]] = rs; if (!RMS) mean[r[u]] = mu; }
                    }
                }
            }
            return;
        }
        for (int rb = r0 + (threadIdx.x >> 5) * rpw; rb < r1; rb += KVM_GPT_THREADS / 32 * rpw) {
            const int r = rb + grp;
            const bool ok = r < r1;   // a task may end inside a warp's group of rows
            const size_t xo = (size_t)r * ldx;
            float4 xv[KVM_GPT_NORM_NV];
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lc + 32 * j;
                xv[j] = (ok && c < d4) ? ld4(xs, xo + 4 * c, xb) : make_float4(0.f, 0.f, 0.f, 0.f);
                s += (xv[j].x + xv[j].y) + (xv[j].z + xv[j].w);
            }
            float mu = RMS ? 0.f : gsum(s, d4, rpw) / d, v = 0.f;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                if (lc + 32 * j < d4) {
                    float t0 = xv[j].x - mu, t1 = xv[j].y - mu, t2 = xv[j].z - mu, t3 = xv[j].w - mu;
                    v += (t0 * t0 + t1 * t1) + (t2 * t2 + t3 * t3);
                }
            }
            float rs = 1.f / sqrtf(gsum(v, d4, rpw) / d + eps);
            const size_t oo = (size_t)r * ldo;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lc + 32 * j;
                if (ok && c < d4) {
                    float4 gg = reinterpret_cast<const float4*>(g)[c], bb = RMS ? make_float4(0.f, 0.f, 0.f, 0.f) : reinterpret_cast<const float4*>(be)[c], o;
                    o.x = (xv[j].x - mu) * rs * gg.x + bb.x; o.y = (xv[j].y - mu) * rs * gg.y + bb.y;
                    o.z = (xv[j].z - mu) * rs * gg.z + bb.z; o.w = (xv[j].w - mu) * rs * gg.w + bb.w;
                    st4(os, oo + 4 * c, o, ob);
                }
            }
            if (ok && lc == 0) { rstd[r] = rs; if (!RMS) mean[r] = mu; }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        const size_t xo = (size_t)r * ldx, oo = (size_t)r * ldo;
        float mu = 0.f;
        if (!RMS) {
            float s = 0.f;
            for (int c = lane; c < d; c += 32) s += ld_s(xs, xo + c, xb);
            mu = wsum(s) / d;
        }
        float v = 0.f;
        for (int c = lane; c < d; c += 32) { float t = ld_s(xs, xo + c, xb) - mu; v += t * t; }
        float rs = 1.f / sqrtf(wsum(v) / d + eps);
        for (int c = lane; c < d; c += 32) st_s(os, oo + c, (ld_s(xs, xo + c, xb) - mu) * rs * g[c] + (RMS ? 0.f : be[c]), ob);
        if (lane == 0) { rstd[r] = rs; if (!RMS) mean[r] = mu; }
    }
}
KVM_GPT_DEVI void layernorm_fwd(const KvmCtx& c, const int32_t* a) { norm_fwd<false>(c, a, 0, 1, 2, 3, 4, 5, 6); }
KVM_GPT_DEVI void rmsnorm_fwd(const KvmCtx& c, const int32_t* a) { norm_fwd<true>(c, a, 0, 1, -1, 2, -1, 3, 4); }

// dx = rstd * (g - mean(g) - xhat*mean(g*xhat)), g = dy*gamma  (LayerNorm); RMSNorm drops the mean(g) term, xhat = x*rstd.
template <bool RMS>
KVM_GPT_DEVI void norm_bwd_dx(const KvmCtx& ctx, const int32_t* a, int mi, int ri, int dxi, int base) {
    Chk k(ctx);
    int r0 = a[base], r1 = a[base + 1], d = a[base + 2], acc = a[base + 3], ldx, ldy, ldd;
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    bool xb, yb, db;
    const void* xs = k.mat_s(a[0], r1, d, &ldx, &xb);
    const void* ys = k.mat_s(a[1], r1, d, &ldy, &yb);
    const float* g = k.fvec(a[2], d);
    const float* mean = RMS ? nullptr : k.fvec(a[mi], r1);
    const float* rstd = k.fvec(a[ri], r1);
    void* ds = k.mat_s(a[dxi], r1, d, &ldd, &db);
    if (k.bad) return;
    int lane = threadIdx.x & 31;
    if (norm_rows_vec(d, xs, ldx, xb, g, ds, ldd, db, g) && vec4_ok(ys, ldy, yb)) {
        const int d4 = d >> 2, rpw = ((d4 & (d4 - 1)) == 0 && d4 <= 16) ? 32 / d4 : 1, grp = rpw > 1 ? lane / d4 : 0, lc = rpw > 1 ? lane % d4 : lane;   // as norm_fwd: rpw rows per warp at once
        if (rpw > 1) {   // as norm_fwd: KVM_GPT_NORM_U row groups in flight, every load of a group issued before any group's math
            constexpr int U = KVM_GPT_NORM_U, STRIDE = KVM_GPT_THREADS / 32;
            const float4 gg = reinterpret_cast<const float4*>(g)[lc];
            for (int rb = r0 + (threadIdx.x >> 5) * rpw; rb < r1; rb += U * STRIDE * rpw) {
                float4 xx[U], dd[U], old[U];
                float mu[U], rs[U];
                int r[U];
#pragma unroll
                for (int u = 0; u < U; u++) {
                    if (rb + u * STRIDE * rpw >= r1) continue;   // warp-uniform, as norm_fwd
                    r[u] = rb + u * STRIDE * rpw + grp;
                    const bool ok = r[u] < r1;
                    mu[u] = (RMS || !ok) ? 0.f : mean[r[u]]; rs[u] = ok ? rstd[r[u]] : 0.f;
                    xx[u] = ok ? ld4(xs, (size_t)r[u] * ldx + 4 * lc, xb) : make_float4(0.f, 0.f, 0.f, 0.f);
                    dd[u] = ok ? ld4(ys, (size_t)r[u] * ldy + 4 * lc, yb) : make_float4(0.f, 0.f, 0.f, 0.f);
                    old[u] = (ok && acc) ? ld4(ds, (size_t)r[u] * ldd + 4 * lc, db) : make_float4(0.f, 0.f, 0.f, 0.f);
                }
#pragma unroll
                for (int u = 0; u < U; u++) {
                    if (rb + u * STRIDE * rpw >= r1) continue;
                    const float4 gc = make_float4(dd[u].x * gg.x, dd[u].y * gg.y, dd[u].z * gg.z, dd[u].w * gg.w);
                    const float4 xh = make_float4((xx[u].x - mu[u]) * rs[u], (xx[u].y - mu[u]) * rs[u], (xx[u].z - mu[u]) * rs[u], (xx[u].w - mu[u]) * rs[u]);
                    const float s1 = RMS ? 0.f : gsum((gc.x + gc.y) + (gc.z + gc.w), d4, rpw) / d;
                    const float s2 = gsum((gc.x * xh.x + gc.y * xh.y) + (gc.z * xh.z + gc.w * xh.w), d4, rpw) / d;
                    if (r[u] < r1) {
                        float4 v;
                        v.x = rs[u] * (gc.x - s1 - xh.x * s2); v.y = rs[u] * (gc.y - s1 - xh.y * s2);
                        v.z = rs[u] * (gc.z - s1 - xh.z * s2); v.w = rs[u] * (gc.w - s1 - xh.w * s2);
                        if (acc) { v.x += old[u].x; v.y += old[u].y; v.z += old[u].z; v.w += old[u].w; }
                        st4(ds, (size_t)r[u] * ldd + 4 * lc, v, db);
                    }
                }
            }
            return;
        }
        for (int rb = r0 + (threadIdx.x >> 5) * rpw; rb < r1; rb += KVM_GPT_THREADS / 32 * rpw) {
            const int r = rb + grp;
            const bool ok = r < r1;
            const size_t xo = (size_t)r * ldx, yo = (size_t)r * ldy;
            float mu = (RMS || !ok) ? 0.f : mean[r], rs = ok ? rstd[r] : 0.f, s1 = 0.f, s2 = 0.f;
            float4 xh[KVM_GPT_NORM_NV], gc[KVM_GPT_NORM_NV];   // normalized input and dy*gamma, kept for the second pass
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lc + 32 * j;
                if (ok && c < d4) {
                    float4 xx = ld4(xs, xo + 4 * c, xb), dd = ld4(ys, yo + 4 * c, yb), gg = reinterpret_cast<const float4*>(g)[c];
                    gc[j] = make_float4(dd.x * gg.x, dd.y * gg.y, dd.z * gg.z, dd.w * gg.w);
                    xh[j] = make_float4((xx.x - mu) * rs, (xx.y - mu) * rs, (xx.z - mu) * rs, (xx.w - mu) * rs);
                    s1 += (gc[j].x + gc[j].y) + (gc[j].z + gc[j].w);
                    s2 += (gc[j].x * xh[j].x + gc[j].y * xh[j].y) + (gc[j].z * xh[j].z + gc[j].w * xh[j].w);
                }
            }
            s1 = RMS ? 0.f : gsum(s1, d4, rpw) / d;
            s2 = gsum(s2, d4, rpw) / d;
            const size_t oo = (size_t)r * ldd;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lc + 32 * j;
                if (ok && c < d4) {
                    float4 v;
                    v.x = rs * (gc[j].x - s1 - xh[j].x * s2); v.y = rs * (gc[j].y - s1 - xh[j].y * s2);
                    v.z = rs * (gc[j].z - s1 - xh[j].z * s2); v.w = rs * (gc[j].w - s1 - xh[j].w * s2);
                    if (acc) { float4 old = ld4(ds, oo + 4 * c, db); v.x += old.x; v.y += old.y; v.z += old.z; v.w += old.w; }
                    st4(ds, oo + 4 * c, v, db);
                }
            }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        const size_t xo = (size_t)r * ldx, yo = (size_t)r * ldy, oo = (size_t)r * ldd;
        float mu = RMS ? 0.f : mean[r], rs = rstd[r], s1 = 0.f, s2 = 0.f;
        for (int c = lane; c < d; c += 32) {
            float gc = ld_s(ys, yo + c, yb) * g[c], xh = (ld_s(xs, xo + c, xb) - mu) * rs;
            s1 += gc; s2 += gc * xh;
        }
        s1 = RMS ? 0.f : wsum(s1) / d;
        s2 = wsum(s2) / d;
        for (int c = lane; c < d; c += 32) {
            float gc = ld_s(ys, yo + c, yb) * g[c], xh = (ld_s(xs, xo + c, xb) - mu) * rs, v = rs * (gc - s1 - xh * s2);
            st_s(ds, oo + c, acc ? ld_s(ds, oo + c, db) + v : v, db);
        }
    }
}
KVM_GPT_DEVI void layernorm_bwd_dx(const KvmCtx& c, const int32_t* a) { norm_bwd_dx<false>(c, a, 3, 4, 5, 6); }
KVM_GPT_DEVI void rmsnorm_bwd_dx(const KvmCtx& c, const int32_t* a) { norm_bwd_dx<true>(c, a, -1, 3, 4, 5); }

// Column reductions: out1[c] (+)= sum_r f1(r,c), out2[c] (+)= sum_r f2(r,c) for c in [c0,c1), r in [0,R). The block is split into
// NC columns (8..64, the smallest that covers the range) x RG row groups (NC*RG = 256): group g sums rows g, g+RG, ... in order, and the RG partials are added in group order,
// so the result is bitwise reproducible. `fn(r, c, &v1, &v2)` produces both terms of one element.
template <class Fn>
KVM_GPT_DEVI void col_reduce(int R, int c0, int c1, float* out1, float* out2, bool acc, Fn fn) {
    const int ncols = c1 - c0, NC = ncols <= 8 ? 8 : ncols <= 16 ? 16 : ncols <= 32 ? 32 : 64, RG = KVM_GPT_THREADS / NC;
    float* part = reinterpret_cast<float*>(kvm_gpt_smem);   // [2][RG][NC] (at most 2 x 256 floats)
    const int cl = threadIdx.x % NC, rg = threadIdx.x / NC;
    for (int cb = c0; cb < c1; cb += NC) {
        const int c = cb + cl;
        float s1 = 0.f, s2 = 0.f;
        if (c < c1) {
#pragma unroll 8
            for (int r = rg; r < R; r += RG) { float v1, v2; fn(r, c, v1, v2); s1 += v1; s2 += v2; }
        }
        part[rg * NC + cl] = s1; part[(RG + rg) * NC + cl] = s2;
        __syncthreads();
        if (threadIdx.x < NC && cb + (int)threadIdx.x < c1) {
            float t1 = 0.f, t2 = 0.f;
            for (int g = 0; g < RG; g++) { t1 += part[g * NC + threadIdx.x]; t2 += part[(RG + g) * NC + threadIdx.x]; }
            const int cc = cb + threadIdx.x;
            out1[cc] = acc ? out1[cc] + t1 : t1;
            if (out2) out2[cc] = acc ? out2[cc] + t2 : t2;
        }
        __syncthreads();
    }
}

// dgamma[c] = sum_r dy*xhat, dbeta[c] = sum_r dy (fixed summation order, see col_reduce)
KVM_GPT_DEVI void norm_bwd_params(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int R = a[6], d = a[7], c0 = a[8], c1 = a[9], acc = a[10], ldx, ldy;
    k.need(R >= 0 && c0 >= 0 && c0 <= c1 && c1 <= d);
    if (k.bad) return;
    bool xb, yb;
    const void* x = k.mat_s(a[0], R, d, &ldx, &xb);
    const void* dy = k.mat_s(a[1], R, d, &ldy, &yb);
    const float* mean = k.fvec(a[2], R, true);
    const float* rstd = k.fvec(a[3], R);
    float* dg = k.fvec(a[4], d);
    float* db = k.fvec(a[5], d, true);
    if (k.bad || c0 == c1) return;
    col_reduce(R, c0, c1, dg, db, acc, [&](int r, int c, float& v1, float& v2) {
        float g = ld_s(dy, (size_t)r * ldy + c, yb);
        v1 = g * ((ld_s(x, (size_t)r * ldx + c, xb) - (mean ? mean[r] : 0.f)) * rstd[r]);
        v2 = g;
    });
}

KVM_GPT_DEVI void colsum(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int R = a[2], c0 = a[3], c1 = a[4], acc = a[5], ld;
    k.need(R >= 0 && c0 >= 0 && c0 <= c1);
    if (k.bad) return;
    bool xb;
    const void* x = k.mat_s(a[0], R, c1, &ld, &xb);
    float* o = k.fvec(a[1], c1);
    if (k.bad || c0 == c1) return;
    col_reduce(R, c0, c1, o, nullptr, acc, [&](int r, int c, float& v1, float& v2) { v1 = ld_s(x, (size_t)r * ld + c, xb); v2 = 0.f; });
}

// ---- elementwise ------------------------------------------------------------------------------------------------------
// out[i] = fn(in[0][i], ..., in[NI-1][i], out[i] (read only if RMW) for i in [e0,e1). Memory bound, and the executor runs ONE block per SM when the
// union's registers are high, so each thread keeps U float4 loads of every stream in flight before computing. 16-byte path when
// every pointer and e0 are 16-byte aligned; scalar loop for the ragged head/tail and for unaligned buffers.
template <int NI, bool RMW, int U = 4, class Fn>
KVM_GPT_DEVI void ew_run(const float* const (&in)[NI], float* out, int e0, int e1, Fn fn) {
    bool vec = ((e0 | (uintptr_t)out) & 15) == 0;
#pragma unroll
    for (int j = 0; j < NI; j++) vec = vec && (((uintptr_t)in[j] & 15) == 0);
    int i = e0;
    if (vec) {
        const int n4 = (e1 - e0) >> 2, step = KVM_GPT_THREADS * U;
        for (int q = threadIdx.x; q < n4; q += step) {
            float4 v[U][NI + 1];
#pragma unroll
            for (int u = 0; u < U; u++) {
                int qq = q + u * KVM_GPT_THREADS;
                if (qq < n4) {
#pragma unroll
                    for (int j = 0; j < NI; j++) v[u][j] = *reinterpret_cast<const float4*>(in[j] + e0 + 4 * qq);
                    if (RMW) v[u][NI] = *reinterpret_cast<const float4*>(out + e0 + 4 * qq);
                }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                int qq = q + u * KVM_GPT_THREADS;
                if (qq < n4) {
                    float4 r;
                    r.x = fn(v[u], 0); r.y = fn(v[u], 1); r.z = fn(v[u], 2); r.w = fn(v[u], 3);
                    *reinterpret_cast<float4*>(out + e0 + 4 * qq) = r;
                }
            }
        }
        i = e0 + n4 * 4;
    }
    for (int t = i + threadIdx.x; t < e1; t += KVM_GPT_THREADS) {
        float4 v[NI + 1];
#pragma unroll
        for (int j = 0; j < NI; j++) v[j] = make_float4(in[j][t], 0.f, 0.f, 0.f);
        v[NI] = make_float4(RMW ? out[t] : 0.f, 0.f, 0.f, 0.f);
        out[t] = fn(v, 0);
    }
}
// ew_run over fp32-or-bf16 streams (scalar, no vector path): used when any stream is bf16 storage. Same fn(v, 0) contract with lane 0 only.
template <int NI, bool RMW, class Fn>
KVM_GPT_DEVI void ew_run_s(const void* const (&in)[NI], const bool (&ib)[NI], void* out, bool ob, int e0, int e1, Fn fn) {
    bool vec = ((e0 & 3) == 0) && vec4_ok(out, 0, ob);   // e0 multiple of 4 keeps every 4-element group 8/16-byte aligned for aligned bases
#pragma unroll
    for (int j = 0; j < NI; j++) vec = vec && vec4_ok(in[j], 0, ib[j]);
    int i = e0;
    if (vec) {
        const int n4 = (e1 - e0) >> 2;
        for (int q = threadIdx.x; q < n4; q += KVM_GPT_THREADS) {
            const size_t t = (size_t)e0 + 4 * q;
            float4 v[NI + 1];
#pragma unroll
            for (int j = 0; j < NI; j++) v[j] = ld4(in[j], t, ib[j]);
            if (RMW) v[NI] = ld4(out, t, ob);
            float4 r;
            r.x = fn(v, 0); r.y = fn(v, 1); r.z = fn(v, 2); r.w = fn(v, 3);
            st4(out, t, r, ob);
        }
        i = e0 + n4 * 4;
    }
    for (int t = i + threadIdx.x; t < e1; t += KVM_GPT_THREADS) {
        float4 v[NI + 1];
#pragma unroll
        for (int j = 0; j < NI; j++) v[j] = make_float4(ld_s(in[j], t, ib[j]), 0.f, 0.f, 0.f);
        v[NI] = make_float4(RMW ? ld_s(out, t, ob) : 0.f, 0.f, 0.f, 0.f);
        st_s(out, t, fn(v, 0), ob);
    }
}
KVM_GPT_DEVI float lane4(const float4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }

KVM_GPT_DEVI void gelu_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[2], e1 = a[3];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    const float* in[1] = {x};
    ew_run<1, false>(in, o, e0, e1, [](const float4* v, int c) { return gelu_f(lane4(v[0], c)); });
}
KVM_GPT_DEVI void gelu_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4], acc = a[5];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    bool xb, yb, db;
    const void* xs = k.vec_s(a[0], e1, &xb);
    const void* ys = k.vec_s(a[1], e1, &yb);
    void* ds = k.vec_s(a[2], e1, &db);
    if (k.bad) return;
    if (xb || yb || db) {
        const void* ins[2] = {xs, ys};
        const bool ib[2] = {xb, yb};
        if (acc) ew_run_s<2, true>(ins, ib, ds, db, e0, e1, [](const float4* v, int c) { return lane4(v[2], c) + lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
        else ew_run_s<2, false>(ins, ib, ds, db, e0, e1, [](const float4* v, int c) { return lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
        return;
    }
    const float* in[2] = {(const float*)xs, (const float*)ys};
    float* dx = (float*)ds;
    if (acc) ew_run<2, true>(in, dx, e0, e1, [](const float4* v, int c) { return lane4(v[2], c) + lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
    else ew_run<2, false>(in, dx, e0, e1, [](const float4* v, int c) { return lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
}
KVM_GPT_DEVI void residual_add(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    bool xb, yb, ob;
    const void* xs = k.vec_s(a[0], e1, &xb);
    const void* ys = k.vec_s(a[1], e1, &yb);
    void* os = k.vec_s(a[2], e1, &ob);
    if (k.bad) return;
    if (xb || yb || ob) {
        const void* ins[2] = {xs, ys};
        const bool ib[2] = {xb, yb};
        ew_run_s<2, false>(ins, ib, os, ob, e0, e1, [](const float4* v, int c) { return lane4(v[0], c) + lane4(v[1], c); });
        return;
    }
    const float* in[2] = {(const float*)xs, (const float*)ys};
    ew_run<2, false>(in, (float*)os, e0, e1, [](const float4* v, int c) { return lane4(v[0], c) + lane4(v[1], c); });
}
// SGD update over a flat element range [e0, e1) of one weight buffer (fp32): w[i] = w[i] - lr * g[i]. A weight bigger than one SM can stream quickly is split into many of these by the
// frontend (Tiling.ew_elems); the vectorized elementwise runner keeps the block's loads in flight, unlike the core `sgd` op's one-thread-one-element scalar loop.
KVM_GPT_DEVI void sgd_range(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const float lr = af(a, 2);
    const int e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    bool wb, gb;
    void* ws = k.vec_s(a[0], e1, &wb);
    const void* gs = k.vec_s(a[1], e1, &gb);
    if (k.bad) return;
    k.need(!wb && !gb, KVM_GPT_E_DTYPE);
    if (k.bad) return;
    const float* in[2] = {(const float*)ws, (const float*)gs};
    ew_run<2, false>(in, (float*)ws, e0, e1, [=](const float4* v, int c) { return lane4(v[0], c) - lr * lane4(v[1], c); });
}
// The operand copy a pre-rounded GEMM reads (matmul_tile flag 1024): out[e] = x[e] as a TF32 value, e in [e0, e1). The value is f2tf32's (to
// nearest, ties away, the low 13 bits cleared; NaN stays NaN): the operand the tiles' own rounding hands the tensor core, so a GEMM on this copy
// with flag 1024 computes, bit for bit, the products of a GEMM on x that rounds its operands, without rounding each element once per tile.
KVM_GPT_DEVI void round_tf32(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const int e0 = a[2], e1 = a[3];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    const float* in[1] = {x};
    ew_run<1, false>(in, o, e0, e1, [](const float4* v, int c) { return __uint_as_float(f2tf32(lane4(v[0], c))); });
}
KVM_GPT_DEVI void zero_buf(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[1], e1 = a[2];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    int32_t* x = k.vec<int32_t>(a[0], e1, -1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) x[i] = 0;
}
KVM_GPT_DEVI void counter_inc(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int i0 = a[1], i1 = a[2];
    k.need(i0 >= 0 && i0 <= i1);
    if (k.bad) return;
    int32_t* x = k.ivec(a[0], i1);
    if (k.bad) return;
    for (int i = i0 + threadIdx.x; i < i1; i += KVM_GPT_THREADS) x[i] += a[3];
}

template <int SIGN>
KVM_GPT_DEVI void rope(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], T = a[5], H = a[6], hd = a[7], ldx, ldo;
    float theta = af(a, 8);
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && H > 0 && hd > 0 && (hd & 1) == 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    bool xb, ob;
    const void* x = k.mat_s(a[0], r1, (long)H * hd, &ldx, &xb);
    void* o = k.mat_s(a[1], r1, (long)H * hd, &ldo, &ob);
    const int32_t* pos = k.ivec(a[2], 1, true);
    if (k.bad) return;
    int p0 = pos ? pos[0] : 0, half = hd / 2;
    const int q4 = half / 4;
    if ((half & 3) == 0 && KVM_GPT_THREADS % q4 == 0 && vec4_ok(x, ldx, xb) && vec4_ok(o, ldo, ob)) {
        // A thread owns four consecutive frequencies (one float4 of each half-row) and a row slot: a row's four angles are computed once for all its
        // heads, and the loads of KVM_GPT_ROPE_U heads are issued before their math, so their memory round trips overlap (one dependent load pair per
        // element per thread left the 162M row's rope at about 7x its memory time). The values are the general path's, bit for bit.
        constexpr int U = KVM_GPT_ROPE_U;
        const int tq = threadIdx.x % q4, rp = KVM_GPT_THREADS / q4;
        float pw[4];
#pragma unroll
        for (int j = 0; j < 4; j++) pw[j] = powf(theta, -(float)(2 * (4 * tq + j)) / (float)hd);
        for (int r = r0 + threadIdx.x / q4; r < r1; r += rp) {
            const float pr = (float)(r % T + p0);
            float c[4], sn[4];
#pragma unroll
            for (int j = 0; j < 4; j++) { sincosf(pr * pw[j], &sn[j], &c[j]); sn[j] *= (float)SIGN; }
            const size_t xo = (size_t)r * ldx + 4 * tq, oo = (size_t)r * ldo + 4 * tq;
            for (int h0 = 0; h0 < H; h0 += U) {
                float4 x1[U], x2[U];
#pragma unroll
                for (int u = 0; u < U; u++)
                    if (h0 + u < H) { x1[u] = ld4(x, xo + (size_t)(h0 + u) * hd, xb); x2[u] = ld4(x, xo + (size_t)(h0 + u) * hd + half, xb); }
#pragma unroll
                for (int u = 0; u < U; u++) {
                    if (h0 + u >= H) break;
                    const float4 a1 = x1[u], a2 = x2[u];
                    const float4 y1 = make_float4(a1.x * c[0] - a2.x * sn[0], a1.y * c[1] - a2.y * sn[1], a1.z * c[2] - a2.z * sn[2], a1.w * c[3] - a2.w * sn[3]);
                    const float4 y2 = make_float4(a1.x * sn[0] + a2.x * c[0], a1.y * sn[1] + a2.y * c[1], a1.z * sn[2] + a2.z * c[2], a1.w * sn[3] + a2.w * c[3]);
                    st4(o, oo + (size_t)(h0 + u) * hd, y1, ob);
                    st4(o, oo + (size_t)(h0 + u) * hd + half, y2, ob);
                }
            }
        }
        return;
    }
    const bool fi = (KVM_GPT_THREADS % half) == 0;   // half divides the thread count: a thread keeps one frequency index, so its power of theta is computed once, not per element
    const float pwt = fi ? powf(theta, -(float)(2 * (threadIdx.x % half)) / (float)hd) : 0.f;
    for (int e = threadIdx.x; e < (r1 - r0) * H * half; e += KVM_GPT_THREADS) {
        int i = e % half, h = (e / half) % H, r = r0 + e / (half * H);
        float ang = (float)(r % T + p0) * (fi ? pwt : powf(theta, -(float)(2 * i) / (float)hd)), s, c;
        sincosf(ang, &s, &c);
        s *= (float)SIGN;
        const size_t xo = (size_t)r * ldx + h * hd, oo = (size_t)r * ldo + h * hd;
        float x1 = ld_s(x, xo + i, xb), x2 = ld_s(x, xo + i + half, xb);
        st_s(o, oo + i, x1 * c - x2 * s, ob);
        st_s(o, oo + i + half, x1 * s + x2 * c, ob);
    }
}
KVM_GPT_DEVI void rope_fwd(const KvmCtx& c, const int32_t* a) { rope<1>(c, a); }
KVM_GPT_DEVI void rope_bwd(const KvmCtx& c, const int32_t* a) { rope<-1>(c, a); }

// ---- attention (fp32 SIMT flash-style) --------------------------------------------------------------------------------
// Tiles are 32 query rows x 32 keys. Thread (ri = tid/8, cj = tid%8): scores S[ri][cj+8*jj] (jj<4; the 8 lanes of a row are
// adjacent so row reductions are 3 shuffles and K-row reads are bank-conflict-free); output/gradient chunk
// [cj*HD/8, (cj+1)*HD/8) of row ri. Shared rows are padded by 4 floats.
template <int HD> struct AT {
    static_assert(HD % 8 == 0 && HD % 4 == 0, "hd");
    static constexpr int LD = HD + 4, C = HD / 8, TILE = 32 * LD, PS = 36;
    using Ld = TileReg<32, HD>;
};
template <int HD>
KVM_GPT_DEVI void dot32(const float* A, const float* Bm, int ri, int cj, float s[4]) {
    using T = AT<HD>;
    const float4* a = reinterpret_cast<const float4*>(A + ri * T::LD);
    const float4* b[4];
#pragma unroll
    for (int jj = 0; jj < 4; jj++) { b[jj] = reinterpret_cast<const float4*>(Bm + (cj + 8 * jj) * T::LD); s[jj] = 0.f; }
#pragma unroll
    for (int d = 0; d < HD / 4; d++) {
        float4 av = a[d];
#pragma unroll
        for (int jj = 0; jj < 4; jj++) { float4 bv = b[jj][d]; s[jj] += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w; }
    }
}
// acc[c] += sum_j P(ri,j) * Vt[j][chunk]; TR: P stored [j][ri] (key-owner view, for dK/dV).
template <int HD, bool TR>
KVM_GPT_DEVI void acc32(float (&acc)[AT<HD>::C], const float* P, const float* Vt, int ri, int cj) {
    using T = AT<HD>;
#pragma unroll 8
    for (int j = 0; j < 32; j++) {
        float p = TR ? P[j * T::PS + ri] : P[ri * T::PS + j];
        const float* v = Vt + j * T::LD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) acc[c] += p * v[c];
    }
}
template <int HD>
KVM_GPT_DEVI void tile_to_smem(const float* base, int ld, int row0, int rmax, float* dst) {
    typename AT<HD>::Ld t;
    t.load(base, ld, row0, 0, rmax, HD, true);
    t.store(dst, AT<HD>::LD);
}

struct AttnArgs {
    const float *q, *k, *v, *dout, *lse, *delta;
    float *o, *lsew, *dq, *dk, *dv;
    int ldq, ldk, ldv, ldo, lddq, lddk, lddv, b, h, i0, T, H, Hkv;
    float scale;
    unsigned sd;   // storage dtype: bit set = that operand is raw bf16 (KVM_ATT_*); lse / delta are always fp32
};
// bf16-storage bits of AttnArgs::sd. A bf16 operand is widened (exactly) into the same fp32 shared tiles / register fragments the fp32 path
// uses, so the arithmetic is identical; a bf16 output is rounded once (RNE) when stored.
enum { KVM_ATT_Q = 1, KVM_ATT_K = 2, KVM_ATT_V = 4, KVM_ATT_DO = 8, KVM_ATT_O = 16, KVM_ATT_DQ = 32, KVM_ATT_DK = 64, KVM_ATT_DV = 128 };
// element offset into a buffer that is fp32 or raw bf16 (pointer arithmetic in units of the storage element)
KVM_GPT_DEVI const float* eoff(const float* p, size_t off, bool bf) { return bf ? reinterpret_cast<const float*>(reinterpret_cast<const uint16_t*>(p) + off) : p + off; }
KVM_GPT_DEVI float* eoff(float* p, size_t off, bool bf) { return bf ? reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(p) + off) : p + off; }
// two adjacent outputs (columns c, c+1 of one row) to fp32 or bf16 storage
KVM_GPT_DEVI void st_pair(float* p, size_t i, float x, float y, bool bf) { st_s(p, i, x, bf); st_s(p, i + 1, y, bf); }

// ---- tensor-core attention (flash-style) --------------------------------------------------------------------------------
// Products run as 3xTF32 mma.sync m16n8k8 (hi/lo split: x ~= hi + lo with both exactly representable; hi*hi + lo*hi + hi*lo
// reproduces an fp32 product to ~2^-21), so attention keeps its fp32-class accuracy (the 2e-5 contract) on the tensor cores.
// One task covers KVM_GPT_ATTN_ROWS = 128 consecutive rows (queries for attn_fwd / attn_bwd_dq, keys for attn_bwd_dkv): warp w owns
// 16 of them (an m16 tile), so there is no cross-warp reduction and no partial-result merge; the 8 warps share each K/V (or Q/dO)
// tile of 32 rows through a 3-stage cp.async pipeline. The warp's own operand (Q and dO, or K and V) lives in registers as raw fp32
// mma fragments and is split per k-step. Softmax runs in the log2 domain: scores are scaled by scale*log2(e) once, p = ex2(s - m),
// so a masked score of -inf gives p = 0 without a branch. Causal blocks above the diagonal are skipped per warp (warp-uniform).
KVM_GPT_DEVI void split_tf32(float x, unsigned& hi, unsigned& lo) {
    hi = __float_as_uint(x) & 0xffffe000u;                  // truncated hi: one LOP; lo = x - hi is then exact
    lo = __float_as_uint(x - __uint_as_float(hi));          // raw fp32: the tensor core reads its top 10 mantissa bits
}
#define KVM_GPT_L2E 1.4426950408889634f
#define KVM_GPT_LN2 0.6931471805599453f
KVM_GPT_DEVI float ex2f(float x) {   // 2^x, relative error 2^-22; ex2(-inf) = +0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
template <int HD> struct AM {
    static_assert(HD % 8 == 0, "hd");
    static constexpr int LD = HD + 4, NT = HD / 8, TILE = 32 * LD, STAGE = 2 * TILE, NST = 3, NF = NT < 4 ? NT : 4;
    static constexpr int SMEM = NST * STAGE * 4;   // bytes: 3 stages of two 32 x (HD+4) tiles = 51 KiB at hd=64, 99 KiB at hd=128
};
// c[n0 + i] += A * B[i], i < N, three products each; product-major so the N independent accumulator chains interleave.
template <int N, int NA>
KVM_GPT_DEVI void mma3n(float (&c)[NA][4], int n0, const unsigned (&ah)[4], const unsigned (&al)[4], const unsigned (&bh)[N][2], const unsigned (&bl)[N][2]) {
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(c[n0 + i], ah, bl[i][0], bl[i][1]);   // small terms first
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(c[n0 + i], al, bh[i][0], bh[i][1]);
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(c[n0 + i], ah, bh[i][0], bh[i][1]);
}
// Same products for a long-lived accumulator: the tensor core truncates its fp32 accumulator on every instruction, so three adds per
// k-step into c over ~1000 steps cost ~1e-5 relative (measured: dK/dV 2.1e-5 at T=1024). Here c gets one add per step (hi*hi) and the two small
// products go to a zeroed t, folded into c once per tile with an ordinary (round-to-nearest) add: ~2.6x lower error for ~3% more time.
template <int N, int NA>
KVM_GPT_DEVI void mma3s(float (&c)[NA][4], float (&t)[N][4], int n0, const unsigned (&ah)[4], const unsigned (&al)[4], const unsigned (&bh)[N][2], const unsigned (&bl)[N][2]) {
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(t[i], ah, bl[i][0], bl[i][1]);
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(t[i], al, bh[i][0], bh[i][1]);
#pragma unroll
    for (int i = 0; i < N; i++) mma_tf32(c[n0 + i], ah, bh[i][0], bh[i][1]);
}
template <int N, int NA>
KVM_GPT_DEVI void fold(float (&c)[NA][4], float (&t)[N][4], int n0) {
#pragma unroll
    for (int i = 0; i < N; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) { c[n0 + i][e] += t[i][e]; t[i][e] = 0.f; }
}
KVM_GPT_DEVI void split4(const float (&x)[4], unsigned (&h)[4], unsigned (&l)[4]) {
#pragma unroll
    for (int e = 0; e < 4; e++) split_tf32(x[e], h[e], l[e]);
}
// B fragments of X * Tl^T (the n index runs over rows of the shared tile Tl): n-tiles n0..n0+N-1 of 8 rows, k-step ks.
template <int LD, int N>
KVM_GPT_DEVI void frag_nt(unsigned (&bh)[N][2], unsigned (&bl)[N][2], const float* Tl, int n0, int ks, int g, int t) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        split_tf32(Tl[((n0 + i) * 8 + g) * LD + ks * 8 + t], bh[i][0], bl[i][0]);
        split_tf32(Tl[((n0 + i) * 8 + g) * LD + ks * 8 + t + 4], bh[i][1], bl[i][1]);
    }
}
// B fragments of P * Tl (k runs over the rows of the shared tile, permuted: slot t = row 2t, slot t+4 = row 2t+1 of the k8 step `kt`,
// which is the order the m16n8 accumulator layout hands P over in); n-tiles n0..n0+N-1 of 8 columns.
template <int LD, int N>
KVM_GPT_DEVI void frag_kn(unsigned (&bh)[N][2], unsigned (&bl)[N][2], const float* Tl, int kt, int n0, int g, int t) {
    const float* r0 = Tl + (kt * 8 + 2 * t) * LD + g;
    const float* r1 = r0 + LD;
#pragma unroll
    for (int i = 0; i < N; i++) {
        split_tf32(r0[(n0 + i) * 8], bh[i][0], bl[i][0]);
        split_tf32(r1[(n0 + i) * 8], bh[i][1], bl[i][1]);
    }
}
// A fragment (16 rows x k8) of a probability-like tile from the four accumulator registers of n-tile `kt` (rows g, g+8; keys 2t, 2t+1).
KVM_GPT_DEVI void acc_to_a(const float (&c)[4], unsigned (&h)[4], unsigned (&l)[4]) {
    split_tf32(c[0], h[0], l[0]); split_tf32(c[2], h[1], l[1]); split_tf32(c[1], h[2], l[2]); split_tf32(c[3], h[3], l[3]);
}
// Raw fp32 A fragments of the warp's 16 rows of a [rows x HD] head slice: row g / g+8, column ks*8 + t / t+4; rows >= rmax are zero.
template <int HD>
KVM_GPT_DEVI void load_a_raw(float (&a)[HD / 8][4], const float* base, int ld, int row, int rmax, int t, bool bf = false) {
    const bool v0 = row < rmax, v1 = row + 8 < rmax;
    const size_t i0 = (size_t)row * ld + t, i1 = i0 + (size_t)8 * ld;   // base is already offset in storage elements (eoff)
#pragma unroll
    for (int ks = 0; ks < HD / 8; ks++) {
        a[ks][0] = v0 ? ld_s(base, i0 + ks * 8, bf) : 0.f; a[ks][1] = v1 ? ld_s(base, i1 + ks * 8, bf) : 0.f;
        a[ks][2] = v0 ? ld_s(base, i0 + ks * 8 + 4, bf) : 0.f; a[ks][3] = v1 ? ld_s(base, i1 + ks * 8 + 4, bf) : 0.f;
    }
}
// cp.async copy of a 32 x HD tile (rows row0.. of a row-major matrix, leading dimension ldg) to shared rows of stride HD+4;
// rows >= rmax are zero-filled. The padding makes every mma fragment load bank-conflict free.
template <int HD>
KVM_GPT_DEVI void tile32_async(float* dst, const float* src, int ldg, int row0, int rmax, bool bf = false) {
    constexpr int CPR = HD / 4;
    if (bf) {   // raw bf16 source: 8-byte loads widened (exactly) into the same fp32 tile; plain stores, visible after the consumer's __syncthreads
        for (int q = threadIdx.x; q < 32 * CPR; q += KVM_GPT_THREADS) {
            int r = q / CPR, c = (q % CPR) * 4, gr = row0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < rmax) {
                const uint2 u = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(src) + (size_t)gr * ldg + c);
                v = make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
            }
            *reinterpret_cast<float4*>(dst + r * (HD + 4) + c) = v;
        }
        return;
    }
    for (int q = threadIdx.x; q < 32 * CPR; q += KVM_GPT_THREADS) {
        int r = q / CPR, c = (q % CPR) * 4, gr = row0 + r;
        cp_async16(dst + r * (HD + 4) + c, gr < rmax ? src + (size_t)gr * ldg + c : src, gr < rmax ? 16 : 0);
    }
}
// p = exp(scale*s - lse) (c2 = scale*log2(e), lse2 = lse*log2(e)) where the mask holds, else 0.
KVM_GPT_DEVI float p_of(float s, float c2, float lse2, bool ok) { return ok ? ex2f(fmaf(s, c2, -lse2)) : 0.f; }

// Forward attention for 128 query rows. Per 32-key tile a warp computes S (16 x 32) = Q K^T in four independent n-tile chains,
// updates its online (max, sum) per row and rescales O (16 x HD, in registers), then O += P V with P taken straight from the S
// accumulator registers.
template <int HD>
KVM_GPT_DEVI void attn_fwd_t(const AttnArgs& A) {
    using M = AM<HD>;
    constexpr int LD = M::LD, NT = M::NT, NF = M::NF;
    float* KV = reinterpret_cast<float*>(kvm_gpt_smem);   // [3 stages][K | V]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const bool kbf = A.sd & KVM_ATT_K, vbf = A.sd & KVM_ATT_V;
    const float* kb_ = eoff(A.k, (size_t)A.b * Tn * A.ldk + kvh * HD, kbf);
    const float* vb_ = eoff(A.v, (size_t)A.b * Tn * A.ldv + kvh * HD, vbf);
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + 31) / 32;   // key tiles up to and including the block's last diagonal one
    auto issue = [&](int it) {
        if (it < nk) {
            float* st = KV + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, kb_, A.ldk, it * 32, Tn, kbf);
            tile32_async<HD>(st + M::TILE, vb_, A.ldv, it * 32, Tn, vbf);
        }
        cp_commit();   // empty groups too, so the wait counts stay uniform
    };
    issue(0); issue(1);
    float aq[NT][4];
    load_a_raw<HD>(aq, eoff(A.q, (size_t)A.b * Tn * A.ldq + A.h * HD, A.sd & KVM_ATT_Q), A.ldq, rowA, Tn, t, A.sd & KVM_ATT_Q);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, o[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[n][e] = 0.f;
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nk; it++) {
        cp_wait<1>();
        __syncthreads();   // tile `it` landed; every warp finished tile it-1, so the stage the next issue fills is free
        issue(it + 2);
        const int kbase = it * 32;
        if (kbase > wrow + 15 || wrow >= Tn) continue;   // all of this tile's keys lie above all of the warp's rows (warp-uniform)
        const float* Ks = KV + (it % M::NST) * M::STAGE; const float* Vs = Ks + M::TILE;
        float s[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[kt][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NT; ks++) {
            unsigned ah[4], al[4], bh[4][2], bl[4][2];
            split4(aq[ks], ah, al);
            frag_nt<LD, 4>(bh, bl, Ks, 0, ks, g, t);
            mma3n<4>(s, 0, ah, al, bh, bl);
        }
        float r0m = -INFINITY, r1m = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < 4; kt++) {
            const int key = kbase + kt * 8 + 2 * t;
            s[kt][0] = key <= rowA ? s[kt][0] * c2 : -INFINITY;     s[kt][1] = key + 1 <= rowA ? s[kt][1] * c2 : -INFINITY;
            s[kt][2] = key <= rowB ? s[kt][2] * c2 : -INFINITY;     s[kt][3] = key + 1 <= rowB ? s[kt][3] * c2 : -INFINITY;
            r0m = fmaxf(r0m, fmaxf(s[kt][0], s[kt][1])); r1m = fmaxf(r1m, fmaxf(s[kt][2], s[kt][3]));
        }
        r0m = fmaxf(r0m, __shfl_xor_sync(0xffffffffu, r0m, 1)); r0m = fmaxf(r0m, __shfl_xor_sync(0xffffffffu, r0m, 2));
        r1m = fmaxf(r1m, __shfl_xor_sync(0xffffffffu, r1m, 1)); r1m = fmaxf(r1m, __shfl_xor_sync(0xffffffffu, r1m, 2));
        const float mn0 = fmaxf(m0, r0m), mn1 = fmaxf(m1, r1m);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;   // a fully masked row stays at p = 0, no NaN
        const float al0 = ex2f(m0 - ms0), al1 = ex2f(m1 - ms1);
        m0 = mn0; m1 = mn1;
        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int kt = 0; kt < 4; kt++) {
            s[kt][0] = ex2f(s[kt][0] - ms0); s[kt][1] = ex2f(s[kt][1] - ms0);
            s[kt][2] = ex2f(s[kt][2] - ms1); s[kt][3] = ex2f(s[kt][3] - ms1);
            ps0 += s[kt][0] + s[kt][1]; ps1 += s[kt][2] + s[kt][3];
        }
        l0 = l0 * al0 + ps0; l1 = l1 * al1 + ps1;
#pragma unroll
        for (int n = 0; n < NT; n++) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }
#pragma unroll
        for (int n0 = 0; n0 < NT; n0 += NF) {
            float tf[NF][4] = {};
#pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                unsigned ph[4], pl[4], bh[NF][2], bl[NF][2];
                acc_to_a(s[kt], ph, pl);
                frag_kn<LD, NF>(bh, bl, Vs, kt, n0, g, t);
                mma3s<NF>(o, tf, n0, ph, pl, bh, bl);
            }
            fold<NF>(o, tf, n0);
        }
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const bool obf = A.sd & KVM_ATT_O;
    const size_t oa = ((size_t)A.b * Tn + rowA) * A.ldo + A.h * HD + 2 * t, ob = oa + (size_t)8 * A.ldo;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (rowA < Tn) st_pair(A.o, oa + n * 8, o[n][0] * inv0, o[n][1] * inv0, obf);
        if (rowB < Tn) st_pair(A.o, ob + n * 8, o[n][2] * inv1, o[n][3] * inv1, obf);
    }
    if (t == 0) {
        float* ls = A.lsew + ((size_t)A.b * A.H + A.h) * Tn;
        if (rowA < Tn) ls[rowA] = l0 > 0.f ? (m0 + log2f(l0)) * KVM_GPT_LN2 : 0.f;
        if (rowB < Tn) ls[rowB] = l1 > 0.f ? (m1 + log2f(l1)) * KVM_GPT_LN2 : 0.f;
    }
}

// dQ = scale * dS K,  dS = P (dP - delta),  P = exp(scale*QK^T - lse),  dP = dO V^T. Recomputes P from lse (no stored P). Same
// warp ownership and pipeline as attn_fwd_t: Q and dO fragments in registers, K/V tiles streamed, dQ accumulated in registers.
template <int HD>
KVM_GPT_DEVI void attn_dq_t(const AttnArgs& A) {
    using M = AM<HD>;
    constexpr int LD = M::LD, NT = M::NT, NF = M::NF;
    float* KV = reinterpret_cast<float*>(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const bool kbf = A.sd & KVM_ATT_K, vbf = A.sd & KVM_ATT_V;
    const float* kb_ = eoff(A.k, (size_t)A.b * Tn * A.ldk + kvh * HD, kbf);
    const float* vb_ = eoff(A.v, (size_t)A.b * Tn * A.ldv + kvh * HD, vbf);
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + 31) / 32;
    auto issue = [&](int it) {
        if (it < nk) {
            float* st = KV + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, kb_, A.ldk, it * 32, Tn, kbf);
            tile32_async<HD>(st + M::TILE, vb_, A.ldv, it * 32, Tn, vbf);
        }
        cp_commit();
    };
    issue(0); issue(1);
    float aq[NT][4], ad[NT][4];
    load_a_raw<HD>(aq, eoff(A.q, (size_t)A.b * Tn * A.ldq + A.h * HD, A.sd & KVM_ATT_Q), A.ldq, rowA, Tn, t, A.sd & KVM_ATT_Q);
    load_a_raw<HD>(ad, eoff(A.dout, (size_t)A.b * Tn * A.ldo + A.h * HD, A.sd & KVM_ATT_DO), A.ldo, rowA, Tn, t, A.sd & KVM_ATT_DO);
    const size_t hrow = ((size_t)A.b * A.H + A.h) * Tn;
    const float lr0 = rowA < Tn ? A.lse[hrow + rowA] * KVM_GPT_L2E : 0.f, lr1 = rowB < Tn ? A.lse[hrow + rowB] * KVM_GPT_L2E : 0.f;
    const float dr0 = rowA < Tn ? A.delta[hrow + rowA] : 0.f, dr1 = rowB < Tn ? A.delta[hrow + rowB] : 0.f;
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = 0.f;
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nk; it++) {
        cp_wait<1>();
        __syncthreads();
        issue(it + 2);
        const int kbase = it * 32;
        if (kbase > wrow + 15 || wrow >= Tn) continue;
        const float* Ks = KV + (it % M::NST) * M::STAGE; const float* Vs = Ks + M::TILE;
        float s[4][4], dp[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) { s[kt][e] = 0.f; dp[kt][e] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < NT; ks++) {
            unsigned ah[4], al[4], bh[4][2], bl[4][2];
            split4(aq[ks], ah, al);
            frag_nt<LD, 4>(bh, bl, Ks, 0, ks, g, t);
            mma3n<4>(s, 0, ah, al, bh, bl);
            split4(ad[ks], ah, al);
            frag_nt<LD, 4>(bh, bl, Vs, 0, ks, g, t);
            mma3n<4>(dp, 0, ah, al, bh, bl);
        }
#pragma unroll
        for (int kt = 0; kt < 4; kt++) {
            const int key = kbase + kt * 8 + 2 * t;
            s[kt][0] = p_of(s[kt][0], c2, lr0, key <= rowA) * (dp[kt][0] - dr0);
            s[kt][1] = p_of(s[kt][1], c2, lr0, key + 1 <= rowA) * (dp[kt][1] - dr0);
            s[kt][2] = p_of(s[kt][2], c2, lr1, key <= rowB) * (dp[kt][2] - dr1);
            s[kt][3] = p_of(s[kt][3], c2, lr1, key + 1 <= rowB) * (dp[kt][3] - dr1);
        }
#pragma unroll
        for (int n0 = 0; n0 < NT; n0 += NF) {
            float tf[NF][4] = {};
#pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                unsigned dh[4], dl[4], bh[NF][2], bl[NF][2];
                acc_to_a(s[kt], dh, dl);
                frag_kn<LD, NF>(bh, bl, Ks, kt, n0, g, t);
                mma3s<NF>(acc, tf, n0, dh, dl, bh, bl);
            }
            fold<NF>(acc, tf, n0);
        }
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    const bool qbf = A.sd & KVM_ATT_DQ;
    const size_t oa = ((size_t)A.b * Tn + rowA) * A.lddq + A.h * HD + 2 * t, ob = oa + (size_t)8 * A.lddq;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (rowA < Tn) st_pair(A.dq, oa + n * 8, acc[n][0] * A.scale, acc[n][1] * A.scale, qbf);
        if (rowB < Tn) st_pair(A.dq, ob + n * 8, acc[n][2] * A.scale, acc[n][3] * A.scale, qbf);
    }
}

// dV = sum P^T dO, dK = scale * sum dS^T Q over the GQA group's heads and all queries >= the key block, fixed loop order.
// The warp's 16 keys stay resident as K and V fragments; Q and dO tiles of (head, 32 queries) stream through the pipeline. The
// products are computed transposed (S^T = K Q^T: keys are the mma rows), so P^T and dS^T come out of the accumulators already in
// the A-operand layout of dV += P^T dO and dK += dS^T Q (query order permuted inside a k8 step, as in attn_fwd_t).
template <int HD>
KVM_GPT_DEVI void attn_dkv_t(const AttnArgs& A) {
    using M = AM<HD>;
    constexpr int LD = M::LD, NT = M::NT, NF = M::NF;
    float* QO = reinterpret_cast<float*>(kvm_gpt_smem);   // [3 stages][Q | dO]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0, wkey = k0 + warp * 16, keyA = wkey + g, keyB = keyA + 8;
    const int nq = (Tn - k0 + 31) / 32, nit = G * nq;
    auto issue = [&](int it) {
        if (it < nit) {
            const int h = kvh * G + it / nq, q0 = k0 + (it % nq) * 32;
            float* st = QO + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, eoff(A.q, (size_t)A.b * Tn * A.ldq + h * HD, A.sd & KVM_ATT_Q), A.ldq, q0, Tn, A.sd & KVM_ATT_Q);
            tile32_async<HD>(st + M::TILE, eoff(A.dout, (size_t)A.b * Tn * A.ldo + h * HD, A.sd & KVM_ATT_DO), A.ldo, q0, Tn, A.sd & KVM_ATT_DO);
        }
        cp_commit();
    };
    issue(0); issue(1);
    float ak[NT][4], av[NT][4];
    load_a_raw<HD>(ak, eoff(A.k, (size_t)A.b * Tn * A.ldk + kvh * HD, A.sd & KVM_ATT_K), A.ldk, keyA, Tn, t, A.sd & KVM_ATT_K);
    load_a_raw<HD>(av, eoff(A.v, (size_t)A.b * Tn * A.ldv + kvh * HD, A.sd & KVM_ATT_V), A.ldv, keyA, Tn, t, A.sd & KVM_ATT_V);
    float dk[NT][4], dv[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) { dk[n][e] = 0.f; dv[n][e] = 0.f; }
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nit; it++) {
        cp_wait<1>();
        __syncthreads();
        issue(it + 2);
        const int h = kvh * G + it / nq, q0 = k0 + (it % nq) * 32;
        if (q0 + 31 < wkey || wkey >= Tn) continue;   // every query of this tile precedes every key of this warp (warp-uniform)
        const float* Qs = QO + (it % M::NST) * M::STAGE; const float* Os = Qs + M::TILE;
        const size_t hrow = ((size_t)A.b * A.H + h) * Tn;
        float lq[4][2], dl_[4][2];   // lse (log2 units) and delta of the thread's accumulator columns: query q0 + kt*8 + 2t + {0,1}
#pragma unroll
        for (int kt = 0; kt < 4; kt++)
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int q = q0 + kt * 8 + 2 * t + j;
                lq[kt][j] = q < Tn ? A.lse[hrow + q] * KVM_GPT_L2E : 0.f;
                dl_[kt][j] = q < Tn ? A.delta[hrow + q] : 0.f;
            }
        float s[4][4], dp[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) { s[kt][e] = 0.f; dp[kt][e] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < NT; ks++) {
            unsigned ah[4], al[4], bh[4][2], bl[4][2];
            split4(ak[ks], ah, al);
            frag_nt<LD, 4>(bh, bl, Qs, 0, ks, g, t);
            mma3n<4>(s, 0, ah, al, bh, bl);
            split4(av[ks], ah, al);
            frag_nt<LD, 4>(bh, bl, Os, 0, ks, g, t);
            mma3n<4>(dp, 0, ah, al, bh, bl);
        }
        float p[4][4];
#pragma unroll
        for (int kt = 0; kt < 4; kt++) {
            const int qa = q0 + kt * 8 + 2 * t, qb = qa + 1;
            p[kt][0] = p_of(s[kt][0], c2, lq[kt][0], qa < Tn && keyA <= qa);   p[kt][1] = p_of(s[kt][1], c2, lq[kt][1], qb < Tn && keyA <= qb);
            p[kt][2] = p_of(s[kt][2], c2, lq[kt][0], qa < Tn && keyB <= qa);   p[kt][3] = p_of(s[kt][3], c2, lq[kt][1], qb < Tn && keyB <= qb);
            s[kt][0] = p[kt][0] * (dp[kt][0] - dl_[kt][0]); s[kt][1] = p[kt][1] * (dp[kt][1] - dl_[kt][1]);
            s[kt][2] = p[kt][2] * (dp[kt][2] - dl_[kt][0]); s[kt][3] = p[kt][3] * (dp[kt][3] - dl_[kt][1]);
        }
#pragma unroll
        for (int n0 = 0; n0 < NT; n0 += NF) {   // dV then dK of the n-tile chunk, one zeroed small-term accumulator at a time (registers)
            float tf[NF][4] = {};
#pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                unsigned ph[4], pl[4], bh[NF][2], bl[NF][2];
                acc_to_a(p[kt], ph, pl);
                frag_kn<LD, NF>(bh, bl, Os, kt, n0, g, t);
                mma3s<NF>(dv, tf, n0, ph, pl, bh, bl);
            }
            fold<NF>(dv, tf, n0);
#pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                unsigned dh[4], dl[4], bh[NF][2], bl[NF][2];
                acc_to_a(s[kt], dh, dl);
                frag_kn<LD, NF>(bh, bl, Qs, kt, n0, g, t);
                mma3s<NF>(dk, tf, n0, dh, dl, bh, bl);
            }
            fold<NF>(dk, tf, n0);
        }
    }
    cp_wait<0>();
    if (wkey >= Tn) return;
    const bool kbf = A.sd & KVM_ATT_DK, vbf = A.sd & KVM_ATT_DV;
    const size_t da = ((size_t)A.b * Tn + keyA) * A.lddk + kvh * HD + 2 * t, db = da + (size_t)8 * A.lddk;
    const size_t va = ((size_t)A.b * Tn + keyA) * A.lddv + kvh * HD + 2 * t, vb = va + (size_t)8 * A.lddv;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (keyA < Tn) { st_pair(A.dk, da + n * 8, dk[n][0] * A.scale, dk[n][1] * A.scale, kbf); st_pair(A.dv, va + n * 8, dv[n][0], dv[n][1], vbf); }
        if (keyB < Tn) { st_pair(A.dk, db + n * 8, dk[n][2] * A.scale, dk[n][3] * A.scale, kbf); st_pair(A.dv, vb + n * 8, dv[n][2], dv[n][3], vbf); }
    }
}

// Head sizes above KVM_GPT_MAX_HD are not compiled at all (they would fix the persistent kernel's registers at the hd=128 level and
// need a 104 KiB launch); a task asking for one fails with KVM_GPT_E_SHAPE.
template <int HD> KVM_GPT_DEVI void attn_fwd_g(const AttnArgs& A) { if constexpr (HD <= KVM_GPT_MAX_HD) attn_fwd_t<HD>(A); }
template <int HD> KVM_GPT_DEVI void attn_dq_g(const AttnArgs& A) { if constexpr (HD <= KVM_GPT_MAX_HD) attn_dq_t<HD>(A); }
template <int HD> KVM_GPT_DEVI void attn_dkv_g(const AttnArgs& A) { if constexpr (HD <= KVM_GPT_MAX_HD) attn_dkv_t<HD>(A); }
#define KVM_GPT_HD_SWITCH(hd, FN, ARGS)                                                  \
    switch (hd) {                                                                        \
        case 16: FN<16>(ARGS); break;  case 32: FN<32>(ARGS); break;                     \
        case 64: FN<64>(ARGS); break;  case 128: FN<128>(ARGS); break;                   \
        default: break;                                                                  \
    }
KVM_GPT_DEVI bool hd_ok(int hd) { return (hd == 16 || hd == 32 || hd == 64 || hd == 128) && hd <= KVM_GPT_MAX_HD; }

// Common validation: geometry b,h,i0,T,H,Hkv,hd. Kind: 0 fwd/dq (i0 = query block, h = head), 1 dkv (h = kv head).
KVM_GPT_DEVI bool attn_geom(Chk& k, int b, int h, int i0, int T, int H, int Hkv, int hd, bool kv) {
    k.need(T > 0 && H > 0 && Hkv > 0 && H % Hkv == 0 && b >= 0 && i0 >= 0 && i0 < T && h >= 0 && h < (kv ? Hkv : H));
    k.need(hd_ok(hd), KVM_GPT_E_SHAPE);
    return !k.bad;
}

// matrix view of an attention operand that may be fp32 or raw bf16 storage; sets `bit` in A.sd for bf16. Alignment: 16 bytes for fp32
// (cp.async), 8 bytes for bf16 (4-element loads); ld a multiple of 4 elements either way.
KVM_GPT_DEVI float* attn_mat(Chk& k, AttnArgs& A, int idx, long rows, long cols, int* ld, unsigned bit) {
    bool bf = false;
    void* p = k.mat_s(idx, rows, cols, ld, &bf);
    if (k.bad) return nullptr;
    if (bf) A.sd |= bit;
    k.need(((uintptr_t)p & (bf ? 7 : 15)) == 0 && (*ld & 3) == 0, KVM_GPT_E_SHAPE);
    return (float*)p;
}

KVM_GPT_DEVI void attn_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[5]; A.h = a[6]; A.i0 = a[7]; A.T = a[8]; A.H = a[9]; A.Hkv = a[10]; int hd = a[11]; A.scale = af(a, 12);
    if (!attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T;
    A.q = attn_mat(k, A, a[0], R, (long)A.H * hd, &A.ldq, KVM_ATT_Q); A.k = attn_mat(k, A, a[1], R, (long)A.Hkv * hd, &A.ldk, KVM_ATT_K);
    A.v = attn_mat(k, A, a[2], R, (long)A.Hkv * hd, &A.ldv, KVM_ATT_V);
    bool obf = false;
    A.o = (float*)k.mat_s(a[3], R, (long)A.H * hd, &A.ldo, &obf);
    if (obf) A.sd |= KVM_ATT_O;
    A.lsew = k.fvec(a[4], (long)(A.b + 1) * A.H * A.T);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_fwd_g, A)
}
KVM_GPT_DEVI void attn_bwd_delta(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], T = a[5], H = a[6], hd = a[7], ldo, ldd;
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && H > 0 && hd > 0);
    if (k.bad) return;
    bool ob = false, db = false;   // o and do: fp32 or raw bf16 storage
    const void* o = k.mat_s(a[0], r1, (long)H * hd, &ldo, &ob);
    const void* d = k.mat_s(a[1], r1, (long)H * hd, &ldd, &db);
    float* del = k.fvec(a[2], (long)((r1 + T - 1) / T) * H * T);
    if (k.bad) return;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = (r1 - r0) * H;
    if (hd <= 64) {   // up to two scalars per lane: U (row, head) pairs in flight per warp (the loop below is one dependent round trip per pair)
        constexpr int U = 4, NV = 2;
        for (int ee = w; ee < n; ee += KVM_GPT_THREADS / 32 * U) {
            float ov[U][NV], dv[U][NV];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int e = ee + u * (KVM_GPT_THREADS / 32), r = r0 + e / H, h = e % H;
#pragma unroll
                for (int v = 0; v < NV; v++) {
                    const int c = lane + v * 32;
                    const bool in = e < n && c < hd;
                    ov[u][v] = in ? ld_s(o, (size_t)r * ldo + h * hd + c, ob) : 0.f;   // o and do in fp32 or bf16 storage (ob, db: uniform per task)
                    dv[u][v] = in ? ld_s(d, (size_t)r * ldd + h * hd + c, db) : 0.f;
                }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int e = ee + u * (KVM_GPT_THREADS / 32);
                if (e >= n) continue;
                const int r = r0 + e / H, h = e % H;
                float s = 0.f;
#pragma unroll
                for (int v = 0; v < NV; v++) if (lane + v * 32 < hd) s += ov[u][v] * dv[u][v];
                s = wsum(s);
                if (lane == 0) del[((size_t)(r / T) * H + h) * T + r % T] = s;
            }
        }
        return;
    }
    for (int e = w; e < n; e += KVM_GPT_THREADS / 32) {
        int r = r0 + e / H, h = e % H;
        float s = 0.f;
        for (int c = lane; c < hd; c += 32) s += ld_s(o, (size_t)r * ldo + h * hd + c, ob) * ld_s(d, (size_t)r * ldd + h * hd + c, db);
        s = wsum(s);
        if (lane == 0) del[((size_t)(r / T) * H + h) * T + r % T] = s;
    }
}
KVM_GPT_DEVI void attn_bwd_dq(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[7]; A.h = a[8]; A.i0 = a[9]; A.T = a[10]; A.H = a[11]; A.Hkv = a[12]; int hd = a[13]; A.scale = af(a, 14);
    if (!attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = attn_mat(k, A, a[0], R, (long)A.H * hd, &A.ldq, KVM_ATT_Q); A.k = attn_mat(k, A, a[1], R, (long)A.Hkv * hd, &A.ldk, KVM_ATT_K);
    A.v = attn_mat(k, A, a[2], R, (long)A.Hkv * hd, &A.ldv, KVM_ATT_V); A.dout = attn_mat(k, A, a[3], R, (long)A.H * hd, &A.ldo, KVM_ATT_DO);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    bool qbf = false;
    A.dq = (float*)k.mat_s(a[6], R, (long)A.H * hd, &A.lddq, &qbf);
    if (qbf) A.sd |= KVM_ATT_DQ;
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_dq_g, A)
}
KVM_GPT_DEVI void attn_bwd_dkv(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[8]; A.h = a[9]; A.i0 = a[10]; A.T = a[11]; A.H = a[12]; int hd = a[13]; A.scale = af(a, 14);
    k.need(hd > 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const KvmBuf* kb = k.get(a[1], -1, false);   // Hkv is not an argument (15-word task limit): Hkv = k.cols / hd; k is fp32 or bf16
    if (k.bad) return;
    k.need(kb->dtype == KVM_GPT_F32 || kb->dtype == KVM_GPT_BF16, KVM_GPT_E_DTYPE);
    if (k.bad) return;
    A.Hkv = kb->cols / hd;
    k.need(A.Hkv > 0 && kb->cols % hd == 0, KVM_GPT_E_SHAPE);
    if (k.bad || !attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, true)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = attn_mat(k, A, a[0], R, (long)A.H * hd, &A.ldq, KVM_ATT_Q); A.k = attn_mat(k, A, a[1], R, (long)A.Hkv * hd, &A.ldk, KVM_ATT_K);
    A.v = attn_mat(k, A, a[2], R, (long)A.Hkv * hd, &A.ldv, KVM_ATT_V); A.dout = attn_mat(k, A, a[3], R, (long)A.H * hd, &A.ldo, KVM_ATT_DO);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    bool kbf = false, vbf = false;
    A.dk = (float*)k.mat_s(a[6], R, (long)A.Hkv * hd, &A.lddk, &kbf); A.dv = (float*)k.mat_s(a[7], R, (long)A.Hkv * hd, &A.lddv, &vbf);
    if (kbf) A.sd |= KVM_ATT_DK;
    if (vbf) A.sd |= KVM_ATT_DV;
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_dkv_g, A)
}

// ---- serving: KV cache and single-query attention ----------------------------------------------------------------------
KVM_GPT_DEVI void kv_append(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int b0 = a[5], b1 = a[6], nn = a[7], Hkv = a[8], hd = a[9], Tmax = a[10], ldk, ldv;
    k.need(b0 >= 0 && b0 <= b1 && nn > 0 && Hkv > 0 && hd > 0 && Tmax > 0);
    if (k.bad) return;
    const float* kn = k.mat(a[0], (long)b1 * nn, (long)Hkv * hd, &ldk);
    const float* vn = k.mat(a[1], (long)b1 * nn, (long)Hkv * hd, &ldv);
    float* kc = k.mat(a[2], (long)b1 * Hkv * Tmax, hd);
    float* vc = k.mat(a[3], (long)b1 * Hkv * Tmax, hd);
    const int32_t* lens = k.ivec(a[4], b1);
    if (k.bad) return;
    const int ldc = ctx.bufs[a[2]].ld;
    if (ctx.bufs[a[3]].ld != ldc) k.fail(KVM_GPT_E_SHAPE);
    for (int b = b0; b < b1 && !k.bad; b++) if (lens[b] < 0 || lens[b] + nn > Tmax) k.fail(KVM_GPT_E_KVLEN);
    if (k.bad) return;
    for (int e = threadIdx.x; e < (b1 - b0) * nn * Hkv * hd; e += KVM_GPT_THREADS) {
        int c = e % hd, h = (e / hd) % Hkv, t = (e / (hd * Hkv)) % nn, b = b0 + e / (hd * Hkv * nn);
        size_t dst = ((size_t)(b * Hkv + h) * Tmax + lens[b] + t) * ldc + c, src = ((size_t)b * nn + t);
        kc[dst] = kn[src * ldk + h * hd + c];
        vc[dst] = vn[src * ldv + h * hd + c];
    }
}

// One (sequence, head) per task. Warp w handles keys w, w+8, ... with an online softmax (lane-strided over hd); the 8 partial
// (m, l, acc) states are merged in fixed order through shared memory. hd <= 128.
KVM_GPT_DEVI void attn_decode(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int b = a[5], h = a[6], H = a[7], Hkv = a[8], hd = a[9], Tmax = a[10], ldq, ldo;
    float scale = af(a, 11);
    k.need(b >= 0 && h >= 0 && h < H && Hkv > 0 && H % Hkv == 0 && Tmax > 0);
    k.need(hd > 0 && hd <= 128, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const float* q = k.mat(a[0], b + 1, (long)H * hd, &ldq);
    const float* kc = k.mat(a[1], (long)(b + 1) * Hkv * Tmax, hd);
    const float* vc = k.mat(a[2], (long)(b + 1) * Hkv * Tmax, hd);
    const int32_t* lens = k.ivec(a[3], b + 1);
    float* o = k.mat(a[4], b + 1, (long)H * hd, &ldo);
    if (k.bad) return;
    const int ldc = ctx.bufs[a[1]].ld;
    if (ctx.bufs[a[2]].ld != ldc) k.fail(KVM_GPT_E_SHAPE);
    const int n = lens[b] + 1;
    k.need(lens[b] >= 0 && n <= Tmax, KVM_GPT_E_KVLEN);
    if (k.bad) return;
    float* sm_m = reinterpret_cast<float*>(kvm_gpt_smem);   // [8] m, [8] l, [8][128] acc
    float* sm_l = sm_m + 8;
    float* sm_a = sm_l + 8;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, kvh = h / (H / Hkv);
    const size_t cbase = (size_t)(b * Hkv + kvh) * Tmax;
    float qv[4], acc[4] = {0.f, 0.f, 0.f, 0.f}, m = -INFINITY, l = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { int c = lane + 32 * i; qv[i] = c < hd ? q[(size_t)b * ldq + h * hd + c] : 0.f; }
    for (int j = w; j < n; j += 8) {
        const float* kr = kc + (cbase + j) * ldc;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) { int c = lane + 32 * i; if (c < hd) s += qv[i] * kr[c]; }
        s = wsum(s) * scale;
        float mn = fmaxf(m, s), al = expf(m - mn), p = expf(s - mn);   // m=-inf -> al=0
        l = l * al + p;
        const float* vr = vc + (cbase + j) * ldc;
#pragma unroll
        for (int i = 0; i < 4; i++) { int c = lane + 32 * i; acc[i] = acc[i] * al + (c < hd ? p * vr[c] : 0.f); }
        m = mn;
    }
    if (lane == 0) { sm_m[w] = m; sm_l[w] = l; }
#pragma unroll
    for (int i = 0; i < 4; i++) sm_a[w * 128 + lane + 32 * i] = acc[i];
    __syncthreads();
    if ((int)threadIdx.x < hd) {
        float M = -INFINITY;
        for (int i = 0; i < 8; i++) M = fmaxf(M, sm_m[i]);
        float L = 0.f, S = 0.f;
        for (int i = 0; i < 8; i++) {
            float e = sm_m[i] == -INFINITY ? 0.f : expf(sm_m[i] - M);
            L += e * sm_l[i]; S += e * sm_a[i * 128 + threadIdx.x];
        }
        o[(size_t)b * ldo + h * hd + threadIdx.x] = S / L;
    }
}

// ---- loss -------------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void softmax_ce_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[4], r1 = a[5], V = a[6], ign = a[7], ld;
    k.need(r0 >= 0 && r0 <= r1 && V > 0);
    if (k.bad) return;
    bool lb;
    const void* lgs = k.mat_s(a[0], r1, V, &ld, &lb);   // logits: fp32 or bf16 storage
    const int32_t* tg = k.ivec(a[1], r1);
    float* rl = k.fvec(a[2], r1);
    float* ls = k.fvec(a[3], r1);
    if (k.bad) return;
    const float* lg = (const float*)lgs;
    int lane = threadIdx.x & 31;
    if (!lb && V <= 128 && aligned16(lg, ld) && (V & 3) == 0) {   // narrow vocabulary, one float4 per lane: U rows in flight per warp (the loop below is one dependent round trip per row, then a second for the target's logit)
        constexpr int U = 4, STEP = KVM_GPT_THREADS / 32;
        const int c0 = lane * 4;
        for (int rb = r0 + (threadIdx.x >> 5); rb < r1; rb += STEP * U) {
            int tt[U]; float4 xv[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int r = rb + u * STEP;
                tt[u] = r < r1 ? tg[r] : ign;
                xv[u] = (r < r1 && c0 < V) ? *reinterpret_cast<const float4*>(lg + (size_t)r * ld + c0) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int r = rb + u * STEP, t = tt[u];
                if (r >= r1) continue;
                if (t == ign) { if (lane == 0) { rl[r] = 0.f; ls[r] = 0.f; } continue; }
                if ((unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); continue; }
                float mx = -INFINITY, sm_ = 0.f;
                auto acc1 = [&](float x) { if (x > mx) { sm_ = mx == -INFINITY ? 0.f : sm_ * fexp(mx - x); mx = x; } if (x != -INFINITY) sm_ += fexp(x - mx); };
                acc1(xv[u].x); acc1(xv[u].y); acc1(xv[u].z); acc1(xv[u].w);
                const float M = wmax(mx), s = wsum(mx == -INFINITY ? 0.f : sm_ * expf(mx - M));
                const float lt = __shfl_sync(0xffffffffu, lane4(xv[u], t & 3), t >> 2);   // the target's logit, from the lane that holds it (no third global read)
                if (lane == 0) { const float l = M + logf(s); ls[r] = l; rl[r] = l - lt; }
            }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        int t = tg[r];
        if (t == ign) { if (lane == 0) { rl[r] = 0.f; ls[r] = 0.f; } continue; }
        if ((unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); continue; }
        const float* row = lg + (size_t)r * ld;
        const uint16_t* rowb = (const uint16_t*)lgs + (size_t)r * ld;
        // single pass: per-lane running (max, sum exp(x-max)), merged across the warp afterwards
        float mx = -INFINITY, sm_ = 0.f;
        auto acc1 = [&](float x) { if (x > mx) { sm_ = mx == -INFINITY ? 0.f : sm_ * fexp(mx - x); mx = x; } if (x != -INFINITY) sm_ += fexp(x - mx); };   // -inf logits contribute 0
        if (lb) {
            if ((((uintptr_t)rowb & 7) == 0) && (V & 3) == 0 && (ld & 3) == 0) {   // 4 bf16 per 8-byte load, 4 loads in flight per lane
                int c = lane * 4;
                for (; c + 384 < V; c += 512) {
                    uint2 u0 = *reinterpret_cast<const uint2*>(rowb + c), u1 = *reinterpret_cast<const uint2*>(rowb + c + 128);
                    uint2 u2 = *reinterpret_cast<const uint2*>(rowb + c + 256), u3 = *reinterpret_cast<const uint2*>(rowb + c + 384);
                    acc1(bf_lo(u0.x)); acc1(bf_hi(u0.x)); acc1(bf_lo(u0.y)); acc1(bf_hi(u0.y)); acc1(bf_lo(u1.x)); acc1(bf_hi(u1.x)); acc1(bf_lo(u1.y)); acc1(bf_hi(u1.y));
                    acc1(bf_lo(u2.x)); acc1(bf_hi(u2.x)); acc1(bf_lo(u2.y)); acc1(bf_hi(u2.y)); acc1(bf_lo(u3.x)); acc1(bf_hi(u3.x)); acc1(bf_lo(u3.y)); acc1(bf_hi(u3.y));
                }
                for (; c < V; c += 128) { uint2 u = *reinterpret_cast<const uint2*>(rowb + c); acc1(bf_lo(u.x)); acc1(bf_hi(u.x)); acc1(bf_lo(u.y)); acc1(bf_hi(u.y)); }
            } else {
                for (int c = lane; c < V; c += 32) acc1(ld_s(rowb, c, true));
            }
        } else if (aligned16(row, ld) && (V & 3) == 0) {
            int c = lane * 4;
            for (; c + 384 < V; c += 512) {   // 4 independent 16-byte loads in flight per lane (latency hiding at low occupancy)
                float4 x0 = *reinterpret_cast<const float4*>(row + c), x1 = *reinterpret_cast<const float4*>(row + c + 128);
                float4 x2 = *reinterpret_cast<const float4*>(row + c + 256), x3 = *reinterpret_cast<const float4*>(row + c + 384);
                acc1(x0.x); acc1(x0.y); acc1(x0.z); acc1(x0.w); acc1(x1.x); acc1(x1.y); acc1(x1.z); acc1(x1.w);
                acc1(x2.x); acc1(x2.y); acc1(x2.z); acc1(x2.w); acc1(x3.x); acc1(x3.y); acc1(x3.z); acc1(x3.w);
            }
            for (; c < V; c += 128) { float4 x = *reinterpret_cast<const float4*>(row + c); acc1(x.x); acc1(x.y); acc1(x.z); acc1(x.w); }
        } else {
            for (int c = lane; c < V; c += 32) acc1(row[c]);
        }
        float M = wmax(mx);
        float s = wsum(mx == -INFINITY ? 0.f : sm_ * expf(mx - M));
        if (lane == 0) { float l = M + logf(s); ls[r] = l; rl[r] = l - ld_s(lgs, (size_t)r * ld + t, lb); }
    }
}
KVM_GPT_DEVI void softmax_ce_reduce(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], ign = a[4];
    k.need(N >= 0);
    if (k.bad) return;
    const float* rl = k.fvec(a[0], N);
    const int32_t* tg = k.ivec(a[1], N);
    float* loss = k.fvec(a[2], 2);
    if (k.bad) return;
    double* buf = reinterpret_cast<double*>(kvm_gpt_smem);
    double s = 0, c = 0;
    for (int r = threadIdx.x; r < N; r += KVM_GPT_THREADS) if (tg[r] != ign) { s += rl[r]; c += 1; }
    s = bsum_d(s, buf);
    c = bsum_d(c, buf);
    if (threadIdx.x == 0) { loss[0] = c > 0 ? (float)(s / c) : 0.f; loss[1] = c > 0 ? (float)(1.0 / c) : 0.f; }
}
// inv_count >= 0: the reciprocal count is known (softmax_ce_fused); the loss buffer is not read.
KVM_GPT_DEVI void softmax_ce_bwd(const KvmCtx& ctx, const int32_t* a, float inv_count = -1.f) {
    Chk k(ctx);
    int r0 = a[5], r1 = a[6], V = a[7], ign = a[8], ld, ldd;
    float sc = af(a, 9);
    k.need(r0 >= 0 && r0 <= r1 && V > 0);
    if (k.bad) return;
    bool lb, db;
    const void* lgs = k.mat_s(a[0], r1, V, &ld, &lb);   // logits and their gradient: fp32 or bf16 storage
    const int32_t* tg = k.ivec(a[1], r1);
    const float* ls = k.fvec(a[2], r1);
    const float* loss = inv_count < 0.f ? k.fvec(a[3], 2) : nullptr;
    void* dls = k.mat_s(a[4], r1, V, &ldd, &db);
    if (k.bad) return;
    const float* lg = (const float*)lgs;
    float* dl = (float*)dls;
    const float f = (inv_count < 0.f ? loss[1] : inv_count) * sc;
    if (!lb && !db && V <= 128 && (V & 3) == 0 && aligned16(lg, ld) && aligned16(dl, ldd)) {   // narrow vocabulary, one float4 per lane: U rows in flight per warp, as softmax_ce_fwd
        constexpr int U = 4, STEP = KVM_GPT_THREADS / 32;
        const int lane = threadIdx.x & 31, c0 = lane * 4;
        for (int rb = r0 + (threadIdx.x >> 5); rb < r1; rb += STEP * U) {
            int tt[U]; float lrs[U]; float4 xv[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int r = rb + u * STEP;
                tt[u] = r < r1 ? tg[r] : ign;
                lrs[u] = r < r1 ? ls[r] : 0.f;
                xv[u] = (r < r1 && c0 < V) ? *reinterpret_cast<const float4*>(lg + (size_t)r * ld + c0) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int r = rb + u * STEP, t = tt[u];
                if (r >= r1) continue;
                bool skip = t == ign;
                if (!skip && (unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); skip = true; }
                if (c0 >= V) continue;
                const float lr = lrs[u];
                float4 y;
                y.x = skip ? 0.f : (fexp(xv[u].x - lr) - (c0 == t ? 1.f : 0.f)) * f;
                y.y = skip ? 0.f : (fexp(xv[u].y - lr) - (c0 + 1 == t ? 1.f : 0.f)) * f;
                y.z = skip ? 0.f : (fexp(xv[u].z - lr) - (c0 + 2 == t ? 1.f : 0.f)) * f;
                y.w = skip ? 0.f : (fexp(xv[u].w - lr) - (c0 + 3 == t ? 1.f : 0.f)) * f;
                *reinterpret_cast<float4*>(dl + (size_t)r * ldd + c0) = y;
            }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        int t = tg[r], lane = threadIdx.x & 31;
        const float* row = lg + (size_t)r * ld;
        float* o = dl + (size_t)r * ldd;
        bool skip = t == ign;
        if (!skip && (unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); skip = true; }
        const float lr = ls[r];
        if (lb || db) {   // bf16 storage on either side: 4 elements per lane and step, 8-byte (bf16) or 16-byte (fp32) accesses; scalar when ragged
            const uint16_t* rb = (const uint16_t*)lgs + (size_t)r * ld;
            uint16_t* ob = (uint16_t*)dls + (size_t)r * ldd;
            const bool vec = (V & 3) == 0 && (ld & 3) == 0 && (ldd & 3) == 0 && (lb ? ((uintptr_t)rb & 7) == 0 : ((uintptr_t)row & 15) == 0) &&
                             (db ? ((uintptr_t)ob & 7) == 0 : ((uintptr_t)o & 15) == 0);
            if (vec) {
                for (int c = lane * 4; c < V; c += 128) {
                    float x[4];
                    if (lb) { uint2 u = *reinterpret_cast<const uint2*>(rb + c); x[0] = bf_lo(u.x); x[1] = bf_hi(u.x); x[2] = bf_lo(u.y); x[3] = bf_hi(u.y); }
                    else { float4 v = *reinterpret_cast<const float4*>(row + c); x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w; }
                    float y[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) y[j] = skip ? 0.f : (fexp(x[j] - lr) - (c + j == t ? 1.f : 0.f)) * f;
                    if (db) *reinterpret_cast<uint2*>(ob + c) = bf16x4(y[0], y[1], y[2], y[3]);
                    else *reinterpret_cast<float4*>(o + c) = make_float4(y[0], y[1], y[2], y[3]);
                }
            } else {
                for (int c = lane; c < V; c += 32) st_s(dls, (size_t)r * ldd + c, skip ? 0.f : (expf(ld_s(lgs, (size_t)r * ld + c, lb) - lr) - (c == t ? 1.f : 0.f)) * f, db);
            }
        } else if (aligned16(row, ld) && aligned16(o, ldd) && (V & 3) == 0) {
            auto grad4 = [&](float4 x, int c) {
                float4 y;
                y.x = skip ? 0.f : (fexp(x.x - lr) - (c == t ? 1.f : 0.f)) * f;
                y.y = skip ? 0.f : (fexp(x.y - lr) - (c + 1 == t ? 1.f : 0.f)) * f;
                y.z = skip ? 0.f : (fexp(x.z - lr) - (c + 2 == t ? 1.f : 0.f)) * f;
                y.w = skip ? 0.f : (fexp(x.w - lr) - (c + 3 == t ? 1.f : 0.f)) * f;
                *reinterpret_cast<float4*>(o + c) = y;
            };
            int c = lane * 4;
            for (; c + 384 < V; c += 512) {   // 4 independent 16-byte loads in flight per lane
                float4 x0 = *reinterpret_cast<const float4*>(row + c), x1 = *reinterpret_cast<const float4*>(row + c + 128);
                float4 x2 = *reinterpret_cast<const float4*>(row + c + 256), x3 = *reinterpret_cast<const float4*>(row + c + 384);
                grad4(x0, c); grad4(x1, c + 128); grad4(x2, c + 256); grad4(x3, c + 384);
            }
            for (; c < V; c += 128) grad4(*reinterpret_cast<const float4*>(row + c), c);
        } else {
            for (int c = lane; c < V; c += 32) o[c] = skip ? 0.f : (expf(row[c] - lr) - (c == t ? 1.f : 0.f)) * f;
        }
    }
}
// The forward and backward of softmax cross-entropy for rows [r0, r1) in one task: the same arithmetic as softmax_ce_fwd then softmax_ce_bwd, in one block with no
// dependency hop between them. lse is written by the forward and read by the backward through the same global memory (the barrier orders them inside the block).
// No ignored targets: loss[1] of softmax_ce_reduce is 1/N, known here, so the reduce leaves the critical path.
KVM_GPT_DEVI void softmax_ce_fused(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const int r0 = a[5], r1 = a[6], V = a[7], ign = a[8];
    k.need(ign == -1 && a[9] > 0 && r0 >= 0 && r0 <= r1 && V > 0);     // the frontend fuses only when no target is ignored
    if (k.bad) return;
    int ld, ldd;
    bool lb, db;
    const void* lgs = k.mat_s(a[0], r1, V, &ld, &lb);
    const int32_t* tg = k.ivec(a[1], r1);
    float* rl = k.fvec(a[2], r1);
    float* ls = k.fvec(a[3], r1);
    void* dls = k.mat_s(a[4], r1, V, &ldd, &db);
    if (k.bad) return;
    const float* lg = (const float*)lgs;
    float* dl = (float*)dls;
    if (lb || db || V > 128 || (V & 3) != 0 || !aligned16(lg, ld) || !aligned16(dl, ldd)) {   // bf16 storage or a wide vocabulary: the two phases, one after the other
        const int32_t fw[8] = {a[0], a[1], a[2], a[3], a[5], a[6], a[7], a[8]};    // logits, tgt, rowloss, lse, r0, r1, V, ignore
        softmax_ce_fwd(ctx, fw);
        __syncthreads();
        const int32_t bw[10] = {a[0], a[1], a[3], -1, a[4], a[5], a[6], a[7], a[8], a[10]};    // logits, tgt, lse, (no loss buffer), dlogits, r0, r1, V, ignore, scale
        softmax_ce_bwd(ctx, bw, (float)(1.0 / (double)a[9]));
        return;
    }
    // Narrow vocabulary, one float4 per lane, U rows in flight per warp: softmax_ce_fwd's and softmax_ce_bwd's narrow paths, the same arithmetic in one pass over each row
    // (the row's logits and target are read once; its log-sum-exp l is known in the warp and is the value the backward would read back from lse).
    const float f = (float)(1.0 / (double)a[9]) * af(a, 10);
    constexpr int U = 4, STEP = KVM_GPT_THREADS / 32;
    const int lane = threadIdx.x & 31, c0 = lane * 4;
    for (int rb = r0 + (threadIdx.x >> 5); rb < r1; rb += STEP * U) {
        int tt[U]; float4 xv[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int r = rb + u * STEP;
            tt[u] = r < r1 ? tg[r] : ign;
            xv[u] = (r < r1 && c0 < V) ? *reinterpret_cast<const float4*>(lg + (size_t)r * ld + c0) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int r = rb + u * STEP, t = tt[u];
            if (r >= r1) continue;
            if (t == ign) {       // an ignored row (not generated by the frontend): what the two phases write
                if (lane == 0) { rl[r] = 0.f; ls[r] = 0.f; }
                if (c0 < V) *reinterpret_cast<float4*>(dl + (size_t)r * ldd + c0) = make_float4(0.f, 0.f, 0.f, 0.f);
                continue;
            }
            if ((unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); continue; }
            float mx = -INFINITY, sm_ = 0.f;
            auto acc1 = [&](float x) { if (x > mx) { sm_ = mx == -INFINITY ? 0.f : sm_ * fexp(mx - x); mx = x; } if (x != -INFINITY) sm_ += fexp(x - mx); };
            acc1(xv[u].x); acc1(xv[u].y); acc1(xv[u].z); acc1(xv[u].w);
            const float M = wmax(mx), s = wsum(mx == -INFINITY ? 0.f : sm_ * expf(mx - M));
            const float lt = __shfl_sync(0xffffffffu, lane4(xv[u], t & 3), t >> 2);   // the target's logit, from the lane that holds it
            const float l = M + logf(s);
            if (lane == 0) { ls[r] = l; rl[r] = l - lt; }
            if (c0 >= V) continue;
            float4 y;
            y.x = (fexp(xv[u].x - l) - (c0 == t ? 1.f : 0.f)) * f;
            y.y = (fexp(xv[u].y - l) - (c0 + 1 == t ? 1.f : 0.f)) * f;
            y.z = (fexp(xv[u].z - l) - (c0 + 2 == t ? 1.f : 0.f)) * f;
            y.w = (fexp(xv[u].w - l) - (c0 + 3 == t ? 1.f : 0.f)) * f;
            *reinterpret_cast<float4*>(dl + (size_t)r * ldd + c0) = y;
        }
    }
}


// ---- optimizer --------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void adamw(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[6], e1 = a[7];
    float lr = af(a, 8), b1 = af(a, 9), b2 = af(a, 10), eps = af(a, 11), wd = af(a, 12);
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    float* p = k.fvec(a[0], e1);
    const float* g = k.fvec(a[1], e1);
    float* m = k.fvec(a[2], e1);
    float* v = k.fvec(a[3], e1);
    const float* gs = k.fvec(a[4], 1, true);
    const int32_t* st = k.ivec(a[5], 1);
    if (k.bad) return;
    const int t = st[0];
    k.need(t >= 1);
    if (k.bad) return;
    const float gsc = gs ? gs[0] : 1.f;
    const float bc1 = 1.f - (float)pow((double)b1, (double)t), bc2 = 1.f - (float)pow((double)b2, (double)t);
    auto upd = [&](float& pi, float gi_, float& mi_, float& vi_) {
        float gi = gi_ * gsc, mi = b1 * mi_ + (1.f - b1) * gi, vi = b2 * vi_ + (1.f - b2) * gi * gi;
        mi_ = mi; vi_ = vi;
        pi = pi - lr * wd * pi - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    };
    if (((e0 | e1) & 3) == 0 && aligned16(p, 0) && aligned16(g, 0) && aligned16(m, 0) && aligned16(v, 0)) {   // 16-byte path
        // U float4 of each of the four streams in flight per thread (one block per SM cannot hide DRAM latency otherwise)
        constexpr int U = 4, STEP = 4 * KVM_GPT_THREADS * U;
        for (int i0 = e0 + 4 * threadIdx.x; i0 < e1; i0 += STEP) {
            float4 pp[U], gg[U], mm[U], vv[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                int i = i0 + 4 * KVM_GPT_THREADS * u;
                if (i < e1) {
                    pp[u] = *reinterpret_cast<float4*>(p + i); gg[u] = *reinterpret_cast<const float4*>(g + i);
                    mm[u] = *reinterpret_cast<float4*>(m + i); vv[u] = *reinterpret_cast<float4*>(v + i);
                }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                int i = i0 + 4 * KVM_GPT_THREADS * u;
                if (i < e1) {
                    upd(pp[u].x, gg[u].x, mm[u].x, vv[u].x); upd(pp[u].y, gg[u].y, mm[u].y, vv[u].y);
                    upd(pp[u].z, gg[u].z, mm[u].z, vv[u].z); upd(pp[u].w, gg[u].w, mm[u].w, vv[u].w);
                    *reinterpret_cast<float4*>(p + i) = pp[u]; *reinterpret_cast<float4*>(m + i) = mm[u]; *reinterpret_cast<float4*>(v + i) = vv[u];
                }
            }
        }
    } else {
        for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) upd(p[i], g[i], m[i], v[i]);
    }
}
KVM_GPT_DEVI void gradnorm_partial(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int slot = a[2], e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1 && slot >= 0);
    if (k.bad) return;
    const float* g = k.fvec(a[0], e1);
    float* part = k.fvec(a[1], slot + 1);
    if (k.bad) return;
    double s = 0;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) s += (double)g[i] * g[i];
    s = bsum_d(s, reinterpret_cast<double*>(kvm_gpt_smem));
    if (threadIdx.x == 0) part[slot] = (float)s;
}
KVM_GPT_DEVI void gradnorm_finish(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int n = a[1];
    float mx = af(a, 3);
    k.need(n >= 0);
    if (k.bad) return;
    const float* part = k.fvec(a[0], n);
    float* out = k.fvec(a[2], 2);
    if (k.bad) return;
    if (threadIdx.x == 0) {
        double s = 0;
        for (int i = 0; i < n; i++) s += part[i];
        float nrm = (float)sqrt(s);
        out[0] = mx > 0.f ? fminf(1.f, mx / (nrm + 1e-6f)) : 1.f;   // clip coefficient first: out is directly an adamw gscale
        out[1] = nrm;
    }
}

// ---- sampling ---------------------------------------------------------------------------------------------------------
// Philox-4x32-10 (Salmon et al. 2011), counter = (row, ctr, 0, 0), key = (seed, 0x9E3779B9).
KVM_GPT_DEVI uint4 philox4x32_10(uint4 c, uint2 key) {
    for (int i = 0; i < 10; i++) {
        unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x, hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ key.x, lo1, hi0 ^ c.w ^ key.y, lo0);
        key.x += 0x9E3779B9u; key.y += 0xBB67AE85u;
    }
    return c;
}
KVM_GPT_DEVI float philox_uniform(int seed, int ctr, int row) {
    uint4 r = philox4x32_10(make_uint4((unsigned)row, (unsigned)ctr, 0u, 0u), make_uint2((unsigned)seed, 0x9E3779B9u));
    return (float)(r.x >> 8) * (1.0f / 16777216.0f);   // [0,1), 24 bits
}
KVM_GPT_DEVI unsigned order_key(float x) { unsigned b = __float_as_uint(x); return b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u); }

// argmax with lowest-index tie-break; NaN never wins. temp<=0 path of `sample`.
KVM_GPT_DEVI int row_argmax(const float* row, int V, float* sv, int* si) {
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int c = threadIdx.x; c < V; c += KVM_GPT_THREADS) {
        float x = row[c];
        if (x > bv || (x == bv && c < bi)) { bv = x; bi = c; }
    }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = KVM_GPT_THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) {
            float v2 = sv[threadIdx.x + s]; int i2 = si[threadIdx.x + s];
            if (v2 > sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x])) { sv[threadIdx.x] = v2; si[threadIdx.x] = i2; }
        }
        __syncthreads();
    }
    int r = si[0] == 0x7fffffff ? 0 : si[0];   // all -inf / NaN row -> token 0
    __syncthreads();
    return r;
}

KVM_GPT_DEVI void sample(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], V = a[5], topk = a[7], seed = a[8], ld;
    float temp = af(a, 6);
    k.need(r0 >= 0 && r0 <= r1 && V > 0);
    if (k.bad) return;
    const float* lg = k.mat(a[0], r1, V, &ld);
    int32_t* out = k.ivec(a[1], r1);
    const int32_t* ctr = k.ivec(a[2], 1, true);
    if (k.bad) return;
    // shared: double csum[256], double pre[256], float sv[256], int si[256], int cnt, unsigned thr
    double* csum = reinterpret_cast<double*>(kvm_gpt_smem);
    double* pre = csum + KVM_GPT_THREADS;
    float* sv = reinterpret_cast<float*>(pre + KVM_GPT_THREADS);
    int* si = reinterpret_cast<int*>(sv + KVM_GPT_THREADS);
    int* cnt = si + KVM_GPT_THREADS;
    int* pick = cnt + 1;
    const int cv = ctr ? ctr[0] : 0, tid = threadIdx.x, ch = (V + KVM_GPT_THREADS - 1) / KVM_GPT_THREADS;
    for (int r = r0; r < r1; r++) {
        const float* row = lg + (size_t)r * ld;
        if (temp <= 0.f) {
            int j = row_argmax(row, V, sv, si);
            if (tid == 0) out[r] = j;
            continue;
        }
        // max logit and (optionally) the top-k threshold in order-preserving integer key space
        float mx = -INFINITY;
        for (int c = tid; c < V; c += KVM_GPT_THREADS) if (row[c] == row[c]) mx = fmaxf(mx, row[c]);
        sv[tid] = mx;
        __syncthreads();
        for (int s = KVM_GPT_THREADS / 2; s; s >>= 1) { if (tid < s) sv[tid] = fmaxf(sv[tid], sv[tid + s]); __syncthreads(); }
        mx = sv[0];
        __syncthreads();
        unsigned thr = 0;
        if (topk > 0 && topk < V) {
            for (int bit = 31; bit >= 0; bit--) {
                unsigned cand = thr | (1u << bit);
                if (tid == 0) *cnt = 0;
                __syncthreads();
                int c = 0;
                for (int j = tid; j < V; j += KVM_GPT_THREADS) c += order_key(row[j]) >= cand;
                atomicAdd(cnt, c);
                __syncthreads();
                if (*cnt >= topk) thr = cand;
                __syncthreads();
            }
        }
        // inverse CDF in index order: thread t owns the contiguous chunk [t*ch, (t+1)*ch)
        double cs = 0;
        for (int j = tid * ch; j < min(V, (tid + 1) * ch); j++) {
            float x = row[j];
            if (x == x && order_key(x) >= thr) cs += exp((double)(x - mx) / (double)temp);
        }
        csum[tid] = cs;
        __syncthreads();
        if (tid == 0) { double run = 0; for (int t = 0; t < KVM_GPT_THREADS; t++) { run += csum[t]; pre[t] = run; } *pick = -1; }
        __syncthreads();
        const double Z = pre[KVM_GPT_THREADS - 1], target = (double)philox_uniform(seed, cv, r) * Z;
        const double lo = tid ? pre[tid - 1] : 0.0;
        // the owner is the first chunk with pre > target; if rounding leaves none, the last chunk with weight
        bool own = csum[tid] > 0.0 && pre[tid] > target && lo <= target;
        bool lastw = csum[tid] > 0.0 && Z <= target && pre[tid] == Z;
        if (own || lastw) {
            double run = lo; int last = -1, found = -1;
            for (int j = tid * ch; j < min(V, (tid + 1) * ch) && found < 0; j++) {
                float x = row[j];
                if (x == x && order_key(x) >= thr) { run += exp((double)(x - mx) / (double)temp); last = j; if (run > target) found = j; }
            }
            int res = found >= 0 ? found : last;
            if (res >= 0) atomicCAS(pick, -1, res);
        }
        __syncthreads();
        if (tid == 0) out[r] = *pick >= 0 ? *pick : 0;
        __syncthreads();
    }
}
KVM_GPT_DEVI void argmax(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[2], r1 = a[3], V = a[4], ld;
    k.need(r0 >= 0 && r0 <= r1 && V > 0);
    if (k.bad) return;
    const float* lg = k.mat(a[0], r1, V, &ld);
    int32_t* out = k.ivec(a[1], r1);
    if (k.bad) return;
    float* sv = reinterpret_cast<float*>(kvm_gpt_smem);
    int* si = reinterpret_cast<int*>(sv + KVM_GPT_THREADS);
    for (int r = r0; r < r1; r++) { int j = row_argmax(lg + (size_t)r * ld, V, sv, si); if (threadIdx.x == 0) out[r] = j; }
}

// ---- batch plumbing ---------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void select_window(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int i0 = a[3], i1 = a[4], stride = a[5], off = a[6];
    k.need(i0 >= 0 && i0 <= i1);
    if (k.bad) return;
    const KvmBuf* sb = k.get(a[0], -1, false);
    const KvmBuf* db = k.get(a[2], -1, false);
    const int32_t* step = k.ivec(a[1], 1);
    if (k.bad) return;
    k.need(sb->dtype == db->dtype && (sb->rows == 1 || sb->ld == sb->cols) && (db->rows == 1 || db->ld == db->cols), KVM_GPT_E_DTYPE);
    if (k.bad) return;
    const long base = (long)step[0] * stride + off;
    k.need(base >= 0 && base + i1 <= (long)sb->rows * sb->cols && i1 <= (long)db->rows * db->cols);
    if (k.bad) return;
    const int32_t* src = (const int32_t*)sb->p + base;
    int32_t* dst = (int32_t*)db->p;
    for (int i = i0 + threadIdx.x; i < i1; i += KVM_GPT_THREADS) dst[i] = src[i];
}
KVM_GPT_DEVI void gather_rows(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], d = a[5], lds, ldd;
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    const float* src = k.mat(a[0], 0, d, &lds);
    const int32_t* idx = k.ivec(a[1], r1);
    float* dst = k.mat(a[2], r1, d, &ldd);
    if (k.bad) return;
    const int R = ctx.bufs[a[0]].rows;
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += KVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d, id = idx[r];
        float v = 0.f;
        if ((unsigned)id < (unsigned)R) v = src[(size_t)id * lds + c]; else k.fail(KVM_GPT_E_TOKEN);
        dst[(size_t)r * ldd + c] = v;
    }
}

}  // namespace kvmgpt

// ---- per-op entry points (same X-macro convention as the executor) -------------------------------------------------------
#ifdef KVM_OP   // inside kvm_exec.cuh: the executor's dispatcher calls op_<name>
#define X(name, id, sig) KVM_OP(name) { kvmgpt::name(c, a); }
#include "kvm_ops_gpt.def"
#undef X
#else           // standalone (tests): kvm_op_<name> and a switch dispatcher
#define X(name, id, sig) KVM_GPT_DEVI void kvm_op_##name(const KvmCtx& ctx, const int32_t* args) { kvmgpt::name(ctx, args); }
#include "kvm_ops_gpt.def"
#undef X
// Returns false for an op id outside the GPT library (ids 32..63).
KVM_GPT_DEVI bool kvm_gpt_dispatch(const KvmCtx& ctx, int op, const int32_t* args) {
    switch (op) {
#define X(name, id, sig) case id: kvm_op_##name(ctx, args); return true;
#include "kvm_ops_gpt.def"
#undef X
        default: return false;
    }
}
#endif
